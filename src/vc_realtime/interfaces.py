"""
vc_realtime.interfaces — Python Protocols that lock the v1.0 ↔ v2.0 swap boundary.

These structural interfaces (PEP 544 Protocols) define the contracts every
runtime module must satisfy. The v1.0 Python implementations in this package
(``Encoder``, ``KNNRetrieval``, ``Decoder``, ``SpeakerEncoder``,
``BiCodecSpeakerEncoder``, ``VocosV2``, ``SherpaOnnxSileroVAD``,
``StreamingInfer``) are *statically* checked against these Protocols by mypy
in CI. The future v2.0 Rust + C rewrite will provide PyO3 classes that
satisfy the same Protocols, so the streaming shell and benchmark harness can
swap implementations without touching call sites.

Design rules
------------
* All Protocols are ``@runtime_checkable`` so we can use ``isinstance`` for
  defensive checks at glue points (e.g. before handing a Rust object to code
  that expects a Python ``Vocoder``).
* Tensor types are ``numpy.ndarray`` (v1.0 path). When the v2.0 Rust path
  arrives we will widen these to a ``npt.NDArray[np.float32]`` / torch tensor
  union via ``@overload``, but the *shape semantics* (documented in the
  docstrings) must stay identical.
* Audio sample-rate / frame-rate contracts are documented inline; v2.0 must
  honor the same numerical rates (24 kHz, 50 Hz content, 100 Hz Vocos) so that
  downstream SOLA crossfade and ring-buffer math is unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Audio I/O primitives
# ---------------------------------------------------------------------------

# A 1-D float32 array in [-1, 1]; sample rate is implied by the caller.
MonoAudio = np.ndarray  # shape: [T_samples]

# A 2-D mel-spectrogram [B, n_mels, T_frames] used as ONNX input.
MelSpec = np.ndarray

# A 3-D content feature tensor [B, C=768, T_frames] from the content encoder.
ContentFeat = np.ndarray

# A 1-D f0 track [T_frames] in Hz, or [B, T_frames] when batched.
F0 = np.ndarray

# A 1-D energy track [T_frames], or [B, T_frames] when batched.
Energy = np.ndarray


# ---------------------------------------------------------------------------
# Content + pitch encoder
# ---------------------------------------------------------------------------


@runtime_checkable
class ContentEncoder(Protocol):
    """ONNX wrapper that turns a mel-spectrogram into content + F0 + energy.

    v1.0 implementation: :class:`vc_realtime.encoder.Encoder` (TinyVC
    ConvNeXt-v2 6-layer, INT8). The exported graph returns 3 outputs in order
    (content, f0, energy); v2.0 must preserve that order.

    Contracts
    ---------
    * Input mel: ``[B, n_mels=128, T_frames]`` float32, log-mel normalized to
      roughly [-4, +4] (see :func:`vc_realtime.encoder.make_mel_spec`).
    * Output content: ``[B, 768, T_frames]`` float32.
    * Output f0:       ``[B, T_frames]`` float32 in Hz (0 = unvoiced).
    * Output energy:   ``[B, T_frames]`` float32.
    * Frame rate:      50 Hz (hop=480 @ 24 kHz). v2.0 must match.
    """

    def encode(self, mel_spec: MelSpec) -> tuple[ContentFeat, F0, Energy]:
        """Return (content_feat, f0, energy) for the given mel-spectrogram."""
        ...


@runtime_checkable
class PitchExtractor(Protocol):
    """Optional F0 re-estimator; in v1.0 the encoder already returns F0, so
    the default implementation simply forwards the encoder's F0 unless the
    RMVPE fallback (P2-4) detects octave errors."""

    def extract_or_fallback(self, mel_spec: MelSpec, encoder_f0: F0) -> F0:
        """Return the chosen F0 track (Hz), shape ``[B, T_frames]``."""
        ...


# ---------------------------------------------------------------------------
# Speaker conditioning (kNN retrieval OR learned embedding encoder)
# ---------------------------------------------------------------------------


@runtime_checkable
class SpeakerEncoder(Protocol):
    """Encodes a 30 s reference wav into a fixed-dimensional speaker vector,
    and exposes an O(1) voice-id lookup at runtime.

    v1.0 has two implementations satisfying this Protocol:
      * :class:`vc_realtime.speaker_encoder.SpeakerEncoder` (OpenVoice v2, 256-d)
      * :class:`vc_realtime.speaker_encoder_v3.BiCodecSpeakerEncoder` (Spark
        BiCodec, 512-d + 48-byte FSQ code)

    Contracts
    ---------
    * ``encode_reference(mel)``: input mel shape and sample-rate are
      implementation-specific (see each class docstring), output is a 1-D
      ``[D]`` float32 embedding (D ∈ {256, 512}).
    * ``load_voice_registry(pattern)``: side-effecting; populates an internal
      ``voice_id -> embedding`` map. Must be idempotent across re-calls.
    * ``get_speaker_embedding(voice_id)``: O(1) lookup; raises ``KeyError``
      if ``voice_id`` was not loaded.
    """

    def encode_reference(self, ref_mel_spec: MelSpec) -> np.ndarray:
        """Return a 1-D float32 speaker embedding for the given mel-spec."""
        ...

    def load_voice_registry(self, pattern: str) -> None:
        """Pre-load all reference voice embeddings from ``pattern`` (glob)."""
        ...

    def get_speaker_embedding(self, voice_id: int) -> np.ndarray:
        """Return the pre-loaded 1-D float32 embedding for ``voice_id``."""
        ...


@runtime_checkable
class SpeakerConditioner(Protocol):
    """Either kNN retrieval OR a learned flow that injects the target speaker
    identity into the content features.

    v1.0 implementations:
      * :class:`vc_realtime.knn_retrieval.KNNRetrieval` (per-frame top-k cosine
        replacement, 768-d feature library)
      * :class:`vc_realtime.flow.SpeakerFlow` (OpenVoice v2 ResidualCoupling
        flow, src→tgt disentanglement in 192-d projected space)

    The streaming shell only requires ``replace(content_feat)``; both v1.0
    implementations satisfy that (kNN directly, flow via the same signature
    by treating ``se_src``/``se_tgt`` as pre-bound state).
    """

    def select_voice(self, voice_id: int) -> None:
        """Hot-swap the active target voice. O(1)."""
        ...

    def replace(self, source_feat: ContentFeat) -> ContentFeat:
        """Return target-voiced content features with identical shape."""
        ...


# ---------------------------------------------------------------------------
# Vocoder
# ---------------------------------------------------------------------------


@runtime_checkable
class Vocoder(Protocol):
    """Waveform synthesizer that turns predicted acoustic features into audio.

    v1.0 implementations:
      * :class:`vc_realtime.decoder.Decoder` (TinyVC DDSP, 24 kHz, 5 MB INT8)
      * :class:`vc_realtime.vocoder_v2.VocosV2` (F5-TTS Vocos, 24 kHz, 7 MB INT8)

    Contracts
    ---------
    * Input: predicted acoustic features, shape and frame rate are
      implementation-specific (DDSP accepts ``[B, 768, T_50hz]``; Vocos
      accepts ``[B, 100, T_100hz]`` after upsample).
    * Output: ``[B, 1, T_samples]`` float32 in [-1, 1] at 24 kHz.
    """

    def decode(self, *args, **kwargs) -> np.ndarray:
        """Return a 1-channel float32 waveform at 24 kHz, shape ``[B, 1, T]``."""
        ...


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


@runtime_checkable
class VAD(Protocol):
    """Voice-activity detector; consumes 30-ms int16 PCM frames.

    v1.0 default: :class:`vc_realtime.vad.SherpaOnnxSileroVAD` (silero-vad,
    ~22 MB ONNX, +30% accuracy over WebRTC on noisy speech).
    v1.0 fallback: :class:`vc_realtime.vad.WebRtcVAD` (100 KB, BSD-3).

    v2.0 Rust: the ``sherpa-onnx`` Rust crate provides the same C++ core via
    a safe FFI; the Python Protocol here is unchanged.

    Contracts
    ---------
    * Input: 30-ms mono PCM16 bytes at 16 kHz (the VAD may downsample
      internally; ``sample_rate`` is passed in for safety).
    * Output: ``True`` iff the frame is judged to contain speech.
    """

    def is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        """Classify a 30-ms PCM16 frame as speech (True) or silence (False)."""
        ...

    def reset(self) -> None:
        """Flush any internal state (required at chunk boundaries)."""
        ...


# ---------------------------------------------------------------------------
# Streaming inference shell
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamingInfer(Protocol):
    """Glue object that wires VAD → Encoder → SpeakerConditioner → Vocoder
    → SOLA crossfade inside a PyAudio callback loop.

    v1.0 implementation: :class:`vc_realtime.streaming.StreamingInfer`.
    v2.0 Rust: a PyO3 class exposing the same callback surface so the host
    Python (PyAudio / sounddevice) does not change.

    Contracts
    ---------
    * ``audio_callback(in_data, frame_count, time_info, status) -> (bytes, int)``
      matches the PyAudio callback signature (PyAudio ``paContinue``).
    * ``start()`` opens input + output streams and blocks until ``stop()``.
    * ``stop()`` is idempotent and safe to call from any thread.
    """

    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio-compatible callback; returns ``(out_bytes, paContinue)``."""
        ...

    def start(self) -> None:
        """Open audio I/O and block until interrupted or ``stop()``."""
        ...

    def stop(self) -> None:
        """Close audio I/O. Idempotent."""
        ...


__all__ = [
    "ContentEncoder",
    "PitchExtractor",
    "SpeakerEncoder",
    "SpeakerConditioner",
    "Vocoder",
    "VAD",
    "StreamingInfer",
    "MonoAudio",
    "MelSpec",
    "ContentFeat",
    "F0",
    "Energy",
]
