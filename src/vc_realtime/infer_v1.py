"""
vc_realtime.infer_v1 — v1 baseline end-to-end inference (TinyVC + kNN + DDSP).

Architecture
------------
This module wires up the canonical TinyVC v1 baseline pipeline directly from
the upstream PyTorch implementation (Apache-2.0, https://github.com/uthree/tinyvc).

Pipeline::

    source_audio (24 kHz mono, float32 in [-1, 1])
        |
        |  autopad_waveform  (multiple of 480)
        v
    spectrogram (1920-pt STFT, hop=480, Hann window)            -> [B, 961, T]
        |
        v
    Encoder (SSLFeatureEstimator ConvNeXt-v2 + PitchEstimator)
        |   4.704 M params total (4.667 M SSL + 36.9 K F0)
        +-- content  [B, 768, T]   (distilled from WavLM-Base-Plus)
        +-- f0       [B, 1, T]     (top-4 softmax over 512 log-freq classes)
        |
        v
    estimate_energy(wf, frame_size=64)  (max-pool abs, interpolated to L) -> [B, 1, L]
        |
        v
    match_features(content, voice_index, k=4, alpha=0, metrics='cos')
        |   kNN-VC per-frame top-4 cosine similarity weighted average
        v
    content_replaced  [B, 768, T]
        |
        v
    Decoder (SourceNet [4.43 M] + FilterNet [4.23 M] = 4.661 M params)
        |   SourceNet -> 14-harmonic amps + FFT noise kernel
        |   DSP       -> harmonic synthesis + filtered noise  -> [B, 16, L]
        |   FilterNet -> UNet with FiLM conditioning           -> [B, 1, L]
        v
    output_audio (24 kHz mono, float32 in [-1, 1])

Total pretrained footprint: 9.365 M params = 36.4 MB FP32 (encoder 18 MB +
decoder 18 MB on disk). The 5-voice kNN index (`voices.pt`) holds
5 × [1, 768, 1500] FP32 tensors = 23 MB (30 s × 50 Hz reference each).

This module is self-contained — it does NOT depend on `vc_realtime.encoder`
or `vc_realtime.decoder` (those are ONNXRuntime wrappers targeted at the v2
INT8-export path). For v1 we drive the native PyTorch modules directly so we
can verify correctness end-to-end before any export/quantization.

CLI::

    python3 -m vc_realtime.infer_v1 --source input.wav --voice-id 0 --output out.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

# Optional, but recommended for resampling when input sr != 24 kHz
try:
    import librosa  # noqa: F401

    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover — librosa is in requirements.txt
    _HAS_LIBROSA = False

import soundfile as sf

# --- Stub optional TinyVC deps we don't exercise on the v1 default path -----
# `module/utils/__init__.py` imports `f0_estimation`, which in turn imports
# `torchfcpe` and `pyworld`. Those are only needed for `f0_estimation='fcpe'`
# / 'dio' / 'harvest' (used by StreamInfer). The v1 default uses the
# PitchEstimator built into the encoder, so we stub the missing modules to
# keep `from module.tinyvc import ...` working without heavy optional deps.
for _mod_name, _attrs in (
    ("torchfcpe", {"spawn_bundled_infer_model": lambda *a, **kw: None}),
    (
        "pyworld",
        {
            "dio": lambda *a, **kw: None,
            "stonemask": lambda *a, **kw: None,
            "harvest": lambda *a, **kw: None,
        },
    ),
):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        for _k, _v in _attrs.items():
            setattr(_stub, _k, _v)
        sys.modules[_mod_name] = _stub

# --- Make the upstream TinyVC Python source importable ----------------------
# The repository is cloned at ../repos/tinyvc (read-only).
TINYVC_ROOT = Path(os.environ.get("TINYVC_ROOT", "../repos/tinyvc"))
if str(TINYVC_ROOT) not in sys.path:
    sys.path.insert(0, str(TINYVC_ROOT))

# TinyVC module imports (Encoder, Decoder, match_features + util fn's).
# These imports match the *real* upstream API verified at:
#   module/tinyvc/__init__.py    -> Encoder, Decoder, match_features
#   module/tinyvc/encoder.py     -> Encoder.infer(spec) -> (ssl, f0)
#   module/tinyvc/decoder.py     -> Decoder.infer(content, f0, energy) -> wav
#   module/tinyvc/feature_retrieval.py -> match_features(src, ref, k=4, ...)
#   module/utils/spectrogram.py      -> spectrogram(wave, n_fft, hop_size)
#   module/utils/energy_estimation.py-> estimate_energy(wave, frame_size=64)
#   module/utils/auto_padding.py     -> autopad_waveform(wf, frame_size=480)
#   module/utils/pitch_shift.py      -> shift_frequency(f0, shift_semitones)
from module.tinyvc import Decoder as _TinyVCDecoder  # noqa: E402
from module.tinyvc import Encoder as _TinyVCEncoder  # noqa: E402
from module.tinyvc import match_features as _tinyvc_match_features  # noqa: E402
from module.utils.auto_padding import autopad_waveform as _tinyvc_autopad  # noqa: E402
from module.utils.energy_estimation import estimate_energy as _tinyvc_estimate_energy  # noqa: E402
from module.utils.pitch_shift import shift_frequency as _tinyvc_shift_f0  # noqa: E402
from module.utils.spectrogram import spectrogram as _tinyvc_spectrogram  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — sourced from TinyVC's defaults (verified in encoder.py L101,
# decoder.py L236-248, and `python3 -c "import module.tinyvc..."` smoke test).
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000  # TinyVC target sample rate (24 kHz mono)
N_FFT = 1920  # 80 ms analysis window
HOP_SIZE = 480  # 20 ms hop = 50 Hz frame rate
FRAME_SIZE = 480  # DDSP synthesis frame (= hop)
NUM_HARMONICS = 14  # SourceNet harmonics (14 + 1 noise channel = 15)
CONTENT_DIM = 768  # distilled WavLM-Base-Plus feature dim
F0_CLASSES = 512  # PitchEstimator output classes (48 per octave)
DEFAULT_TOP_K = 4  # kNN-VC top-k (matches upstream default)
DEFAULT_ALPHA = 0.0  # kNN-VC blend factor (0 = full replace)
DEFAULT_NORM_DB = -3.0  # peak-normalize input to -3 dBFS


class V1Infer:
    """End-to-end v1 baseline inference wrapper.

    Loads the TinyVC encoder + decoder PyTorch weights and a multi-voice kNN
    index from disk, then exposes :meth:`encode` / :meth:`knn_replace` /
    :meth:`decode` / :meth:`process_audio`.

    Parameters
    ----------
    models_dir : str | Path
        Directory containing ``encoder.pt``, ``decoder.pt`` and either
        ``voices_v1.safetensors`` (preferred) or legacy ``voices.pt``.
    device : str
        Torch device string. Defaults to ``"cpu"`` per project constraint.
    top_k : int
        kNN-VC top-k (default 4, matching upstream TinyVC).
    alpha : float
        kNN-VC blend factor (0 = full target replace, 1 = identity).

    Notes
    -----
    The voice index is loaded via :meth:`_load_voices`, which prefers the
    safe ``voices_v1.safetensors`` file (no arbitrary-code execution on
    load) over the legacy ``voices.pt`` pickle. To migrate an existing
    ``voices.pt``::

        python3 scripts/migrate_voices_to_safetensors.py
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        device: str = "cpu",
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
    ):
        self.device = torch.device(device)
        self.top_k = top_k
        self.alpha = alpha

        models_dir = Path(models_dir)
        enc_path = models_dir / "encoder.pt"
        dec_path = models_dir / "decoder.pt"

        if not enc_path.exists():
            raise FileNotFoundError(
                f"encoder.pt not found at {enc_path}. Download from HF "
                f"'uthree/tinyvc': "
                f"hf_hub_download('uthree/tinyvc', 'models/encoder.pt')"
            )
        if not dec_path.exists():
            raise FileNotFoundError(
                f"decoder.pt not found at {dec_path}. Download from HF "
                f"'uthree/tinyvc': "
                f"hf_hub_download('uthree/tinyvc', 'models/decoder.pt')"
            )

        # Load TinyVC encoder + decoder
        self.encoder = _TinyVCEncoder().to(self.device).eval()
        self.encoder.load_state_dict(torch.load(str(enc_path), map_location=self.device))
        self.decoder = _TinyVCDecoder().to(self.device).eval()
        self.decoder.load_state_dict(torch.load(str(dec_path), map_location=self.device))

        # Load the multi-voice kNN index. Prefers the safe
        # ``voices_v1.safetensors`` file (no arbitrary-code execution on
        # load); falls back to legacy ``voices.pt`` with a DeprecationWarning.
        self.voices: dict[str, torch.Tensor] = self._load_voices(str(models_dir))
        # Backwards-compat alias used by some tests / external callers.
        self.tinyvc_voices = self.voices
        # Sort by voice id for predictable indexing
        self._voice_keys = sorted(
            self.voices.keys(),
            key=lambda k: int(k.rsplit("_", 1)[-1]) if k.startswith("voice_") else 0,
        )
        self.n_voices = len(self._voice_keys)

        # Warm up ONNX/PyTorch sessions with 1s of silence per voice. This
        # forces the JIT/autotune paths in torch's conv/matmul kernels to
        # settle before the first "real" inference call (avoids the ~100ms
        # first-call latency penalty on cold-start benchmarks / realtime).
        self._warmup()

    # ------------------------------------------------------------------
    # Voice index loading (safetensors first, .pt legacy fallback)
    # ------------------------------------------------------------------
    def _load_voices(self, models_dir: str) -> dict[str, torch.Tensor]:
        """Load the multi-voice kNN index.

        Tries ``voices_v1.safetensors`` first (safe — safetensors does NOT
        execute arbitrary code on load, unlike ``torch.load`` on a pickle).
        Falls back to legacy ``voices.pt`` with a DeprecationWarning if the
        safetensors file is missing.

        Raises
        ------
        FileNotFoundError
            If neither file is present in ``models_dir``.
        """
        safetensors_path = Path(f"{models_dir}/voices_v1.safetensors")
        pt_path = Path(f"{models_dir}/voices.pt")

        if safetensors_path.exists():
            # Safe: safetensors doesn't execute arbitrary code on load.
            from safetensors.torch import load_file

            voices_state = load_file(str(safetensors_path))
            if not isinstance(voices_state, dict):
                raise TypeError(
                    f"voices_v1.safetensors at {safetensors_path} did not "
                    f"contain a dict[str -> Tensor] mapping."
                )
            # Convert FP16 storage back to FP32 for inference (kNN-VC cosine
            # retrieval is FP32 in upstream TinyVC).
            voices = {
                k: v.to(self.device).to(torch.float32) for k, v in voices_state.items()
            }
            print(f"[v1] loaded voices from safetensors "
                  f"({len(voices)} voices) at {safetensors_path.name}")
        elif pt_path.exists():
            # Legacy: torch.load with weights_only=False (security risk).
            # Migrate via `scripts/migrate_voices_to_safetensors.py`.
            import warnings

            warnings.warn(
                "voices.pt is deprecated — running "
                "scripts/migrate_voices_to_safetensors.py to convert it to "
                "voices_v1.safetensors will remove this security risk.",
                DeprecationWarning,
                stacklevel=2,
            )
            voices_state = torch.load(
                str(pt_path), map_location=self.device, weights_only=False
            )
            if not isinstance(voices_state, dict):
                # Backwards-compat: a single stacked tensor [N, 1, 768, T]
                voices_state = {
                    f"voice_{i}": voices_state[i : i + 1]
                    for i in range(voices_state.shape[0])
                }
            voices = {
                k: v.to(self.device).to(torch.float32) for k, v in voices_state.items()
            }
            print(f"[v1] loaded voices from .pt (legacy, "
                  f"{len(voices)} voices) at {pt_path.name}")
        else:
            raise FileNotFoundError(
                f"No voice index found at {models_dir}/ — expected "
                f"voices_v1.safetensors (preferred) or voices.pt (legacy). "
                f"Build it with: python3 scripts/build_voices_index.py"
            )
        return voices

    # ------------------------------------------------------------------
    # Warmup — run 1s of silence per voice to prime PyTorch kernels
    # ------------------------------------------------------------------
    def _warmup(self) -> None:
        """Warm up ONNX/PyTorch sessions with 1s of silence per voice."""
        sr = SAMPLE_RATE
        silence = np.zeros(sr, dtype=np.float32)
        for vid in range(min(5, self.n_voices)):
            try:
                _ = self.process_audio(silence, sr, voice_id=vid)
            except Exception as e:  # noqa: BLE001 — best-effort warmup
                print(f"[warmup] voice {vid} failed: {e}")

    # ------------------------------------------------------------------
    # Stage 1: encode source wav -> (content, f0, energy)
    # ------------------------------------------------------------------
    def encode(self, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the TinyVC content + F0 + energy encoder.

        Parameters
        ----------
        wav : np.ndarray, shape [L_samples], float32 in [-1, 1] (or any peak)

        Returns
        -------
        content : np.ndarray, shape [1, 768, T_frames]  float32
        f0      : np.ndarray, shape [1, 1, T_frames]    float32 (Hz, 0 = unvoiced)
        energy  : np.ndarray, shape [1, 1, L_padded]    float32 (sample-rate)
        """
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)  # stereo → mono
        # Generator.encode convention: [Batch, Length] (2D), as torchaudio.load
        wf = torch.from_numpy(wav).unsqueeze(0).to(self.device)  # [1, L]
        wf = _tinyvc_autopad(wf)  # pad to multiple of FRAME_SIZE
        spec = _tinyvc_spectrogram(wf, self.encoder.n_fft, self.encoder.hop_size)
        energy = _tinyvc_estimate_energy(wf)
        with torch.inference_mode():
            content, f0 = self.encoder.infer(spec)
        return (content.cpu().numpy(), f0.cpu().numpy(), energy.cpu().numpy())

    # ------------------------------------------------------------------
    # Stage 2: kNN-VC feature replacement
    # ------------------------------------------------------------------
    def knn_replace(
        self,
        content: np.ndarray,
        voice_id: int,
        top_k: int | None = None,
        alpha: float | None = None,
    ) -> np.ndarray:
        """Per-frame top-k cosine-similarity replacement with target voice.

        Wraps upstream `module.tinyvc.match_features`. Defaults to top-4
        cosine (matches upstream TinyVC `infer.py`).

        Parameters
        ----------
        content  : np.ndarray [1, 768, T_frames]
        voice_id : int  (0 .. n_voices-1)
        top_k    : optional override (default = self.top_k = 4)
        alpha    : optional override (default = self.alpha = 0.0)

        Returns
        -------
        np.ndarray [1, 768, T_frames]
        """
        if not (0 <= voice_id < self.n_voices):
            raise ValueError(f"voice_id {voice_id} out of range [0, {self.n_voices})")
        k = self.top_k if top_k is None else top_k
        a = self.alpha if alpha is None else alpha
        tgt = self.voices[self._voice_keys[voice_id]]  # [1, 768, T_ref]
        z = torch.from_numpy(content).to(self.device).to(torch.float32)
        with torch.inference_mode():
            z_replaced = _tinyvc_match_features(z, tgt, k=k, alpha=a, metrics="cos")
        return z_replaced.cpu().numpy()

    # ------------------------------------------------------------------
    # Stage 3: DDSP decoder
    # ------------------------------------------------------------------
    def decode(self, content: np.ndarray, f0: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """Run the TinyVC DDSP decoder (SourceNet + FilterNet).

        Parameters
        ----------
        content : np.ndarray [1, 768, T_frames]
        f0      : np.ndarray [1, 1,   T_frames]   (Hz, 0 = unvoiced)
        energy  : np.ndarray [1, 1,   L_samples]   (sample-rate)

        Returns
        -------
        np.ndarray [L_samples]  float32 in [-1, 1]
        """
        z = torch.from_numpy(content).to(self.device).to(torch.float32)
        f0_t = torch.from_numpy(f0).to(self.device).to(torch.float32)
        e_t = torch.from_numpy(energy).to(self.device).to(torch.float32)
        with torch.inference_mode():
            out = self.decoder.infer(z, f0_t, e_t)
        # decoder.infer returns [B, 1, L_samples] -> squeeze to [L_samples]
        return out.cpu().numpy().squeeze()

    # ------------------------------------------------------------------
    # Internal: encode → (optional pitch shift) → kNN replace → decode
    # ------------------------------------------------------------------
    def _run_pipeline(
        self,
        wav: np.ndarray,
        voice_id: int = 0,
        pitch_shift_semitones: float = 0.0,
    ) -> np.ndarray:
        """Run the core VC pipeline on already-preprocessed wav.

        Expects ``wav`` to be 24 kHz mono float32, peak-normalized (caller's
        responsibility). Shared by :meth:`process_audio` (single shot) and
        :meth:`process_audio_chunked` (chunked, so PyTorch does not cache
        large intermediate tensors for long inputs).

        Edge cases handled
        ------------------
        * Empty input (``wav.size == 0``) → empty float32 output (no-op).
        * Too-short input (``< n_fft`` samples = 80 ms) → zero-padded to
          ``n_fft`` before encoding (the encoder STFT requires at least
          ``n_fft`` samples for its centered reflect-padding), then the
          output is trimmed back to the original input length so the
          caller sees a length-preserving transform.
        """
        if wav.size == 0:
            # Empty input → empty output (length-preserving no-op).
            return np.zeros(0, dtype=np.float32)

        original_len = int(wav.size)
        # The encoder's centered STFT uses reflect-padding of
        # ``n_fft // 2`` on each side; reflect-pad requires the input
        # length to exceed the pad size. Pad with zeros to ``n_fft`` so
        # the STFT can run on inputs shorter than 80 ms (1 sample,
        # 100 samples, etc.).
        n_fft = int(getattr(self.encoder, "n_fft", N_FFT))
        if original_len < n_fft:
            wav = np.pad(wav, (0, n_fft - original_len), mode="constant")

        content, f0, energy = self.encode(wav)

        # Optional pitch shift (default 0 semitones — identity)
        if abs(pitch_shift_semitones) > 1e-9:
            f0_t = torch.from_numpy(f0).to(self.device).to(torch.float32)
            f0_t = _tinyvc_shift_f0(f0_t, pitch_shift_semitones)
            f0 = f0_t.cpu().numpy()

        content_replaced = self.knn_replace(content, voice_id)
        out = self.decode(content_replaced, f0, energy)

        # Length-preserving trim/pad so the caller sees the same number of
        # samples they passed in (post-resample). Without this, a 1-sample
        # input would be padded to ``n_fft`` and the caller would receive
        # ``n_fft`` samples back, surprising the test harness / streaming
        # shell that expects length-preserving transforms.
        if out.size > original_len:
            out = out[:original_len]
        elif out.size < original_len:
            out = np.pad(out, (0, original_len - out.size), mode="constant")
        return out

    @staticmethod
    def _preprocess(wav: np.ndarray, sr: int) -> np.ndarray:
        """Resample → mono mixdown → sanitize → peak-normalize to -3 dBFS.

        Returns 24 kHz mono float32 with peak = 10**(-3/20) ≈ 0.7079.

        Edge cases
        ----------
        * Empty input → empty float32 output (length-preserving no-op; the
          downstream :meth:`_run_pipeline` short-circuits on empty input).
        * NaN/Inf samples → replaced with 0 and a ``RuntimeWarning`` is
          emitted. Without sanitization, NaN/Inf propagate through the
          encoder/decoder and produce NaN output (cosine kNN is undefined
          on NaN features; DDSP oscillator phase is undefined on Inf).
        * All-zero input → peak is floored at ``1e-8`` so the divide-by-zero
          doesn't produce NaN; the result is still all-zero (multiplied by
          ``0.7079 / 1e-8 ≈ 7e7`` then 0). The 1e-8 floor also keeps
          near-silent input (e.g. ``1e-10``) from being amplified to
          numerical garbage.
        """
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)  # stereo → mono
        if wav.size == 0:
            return wav  # empty → empty (no further processing possible)
        # Sanitize NaN/Inf before any math (prevents NaN propagation).
        if not np.all(np.isfinite(wav)):
            import warnings

            n_bad = int(np.sum(~np.isfinite(wav)))
            warnings.warn(
                f"Input audio contains {n_bad} non-finite (NaN/Inf) sample(s); "
                f"replacing with 0 before processing.",
                RuntimeWarning,
                stacklevel=2,
            )
            wav = np.where(np.isfinite(wav), wav, np.float32(0.0))
        if sr != SAMPLE_RATE:
            if not _HAS_LIBROSA:
                raise RuntimeError(
                    f"Input sr={sr} != {SAMPLE_RATE}; librosa required to "
                    f"resample. Install librosa or feed 24 kHz audio."
                )
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        # Peak-normalize to -3 dBFS (matches TinyVC's infer.py convention).
        # ``max(peak, 1e-8)`` floors the divisor to avoid divide-by-zero on
        # silent input (the previous ``peak + 1e-8`` was incorrect: for a
        # near-zero peak (e.g. 1e-10) it amplified by 0.7079/1e-8 = 7e7,
        # producing numerical garbage; for a large peak it had no effect).
        peak = float(np.max(np.abs(wav)))
        peak = max(peak, 1e-8)
        wav = wav * (10 ** (DEFAULT_NORM_DB / 20.0) / peak)
        return wav

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------
    def process_audio(
        self, wav: np.ndarray, sr: int, voice_id: int = 0, pitch_shift_semitones: float = 0.0
    ) -> np.ndarray:
        """End-to-end VC: source wav -> converted wav (24 kHz mono).

        Steps
        -----
        1. (optional) resample to 24 kHz
        2. mono mixdown (if stereo)
        3. peak-normalize to -3 dBFS (TinyVC convention)
        4. encode -> (content, f0, energy)
        5. (optional) pitch shift f0 by N semitones
        6. kNN-VC replace -> content_replaced
        7. decode(content_replaced, f0, energy) -> wav

        For long inputs (60s+), prefer :meth:`process_audio_chunked` which
        processes the input in fixed-length chunks to avoid PyTorch caching
        large intermediate tensors (60s single-shot ≈ 182 MB RSS growth;
        5s chunks ≈ <50 MB).
        """
        wav = self._preprocess(wav, sr)
        return self._run_pipeline(wav, voice_id, pitch_shift_semitones)

    def process_audio_chunked(
        self,
        wav: np.ndarray,
        sr: int,
        voice_id: int = 0,
        chunk_sec: float = 5.0,
    ) -> np.ndarray:
        """End-to-end VC with chunked processing for long inputs.

        The full wav is preprocessed (resampled + peak-normalized) ONCE so
        relative loudness is preserved across chunk boundaries, then split
        into ``chunk_sec``-second pieces. Each chunk is run independently
        through the encode → kNN replace → decode pipeline, and the outputs
        are concatenated.

        This avoids PyTorch caching large intermediate tensors for long
        audio (60s single-shot ≈ 182 MB RSS growth → 5s chunks ≈ <50 MB).

        Parameters
        ----------
        wav : np.ndarray
            Source audio (any sr, mono or stereo, float32).
        sr : int
            Source sample rate. Will be resampled to 24 kHz if different.
        voice_id : int
            Target voice index (0 .. n_voices-1).
        chunk_sec : float
            Chunk length in seconds. Default 5.0 (12 chunks for 60s input).

        Returns
        -------
        np.ndarray, float32 — concatenated output (24 kHz mono).
        """
        # Preprocess (resample + peak-normalize) ONCE on the full wav so
        # relative loudness is preserved across chunk boundaries.
        wav = self._preprocess(wav, sr)

        # Process in chunks to bound PyTorch intermediate tensor size.
        chunk_size = int(SAMPLE_RATE * chunk_sec)
        n_chunks = (len(wav) + chunk_size - 1) // chunk_size

        outputs: list[np.ndarray] = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(wav))
            chunk = wav[start:end]
            out = self._run_pipeline(chunk, voice_id)
            outputs.append(out)
        return np.concatenate(outputs)

    def process_audio_batched(
        self,
        wav: np.ndarray,
        sr: int,
        voice_id: int = 0,
        batch_sec: float = 5.0,
    ) -> np.ndarray:
        """Process audio in batches for offline (non-streaming) use.

        Splits the input into ``batch_sec``-second chunks, pads the last
        chunk with zeros to a full ``chunk_size`` length, then runs each
        chunk through :meth:`_run_pipeline`. This is the offline
        counterpart to :meth:`process_audio_chunked` — both bound peak
        PyTorch intermediate tensor size, but ``batched`` produces chunks
        of identical length (convenient for stacking into a [N, T] batch
        in a future optimisation).

        Note
        ----
        True batched inference (stacking N chunks into one encoder forward
        pass along the batch dim) would require modifying the upstream
        TinyVC encoder to accept ``[N, C, T]`` input. The current
        implementation processes chunks sequentially but pre-allocates the
        full padded buffer once (instead of a Python list of slices),
        which avoids per-chunk numpy allocation overhead.

        Parameters
        ----------
        wav : np.ndarray
            Source audio (any sr, mono or stereo, float32).
        sr : int
            Source sample rate. Will be resampled to 24 kHz if different.
        voice_id : int
            Target voice index (0 .. n_voices-1).
        batch_sec : float
            Chunk length in seconds. Default 5.0 (matches
            :meth:`process_audio_chunked`).

        Returns
        -------
        np.ndarray, float32 — concatenated output trimmed to ``len(wav)``.
        """
        # Preprocess (resample + peak-normalize) ONCE on the full wav so
        # relative loudness is preserved across chunk boundaries (same
        # invariant as process_audio_chunked).
        wav = self._preprocess(wav, sr)

        chunk_size = int(SAMPLE_RATE * batch_sec)
        n_chunks = (len(wav) + chunk_size - 1) // chunk_size

        # Pad the tail with zeros so every chunk has the same length
        # (chunk_size). For batched processing this lets us reshape into
        # [N, chunk_size] without per-chunk length checks.
        padded = np.zeros(n_chunks * chunk_size, dtype=np.float32)
        padded[: len(wav)] = wav

        # Reshape to [N, chunk_size] — N forward passes (sequential for
        # now; a future version can stack along the batch dim).
        batched = padded.reshape(n_chunks, chunk_size)

        outputs: list[np.ndarray] = []
        for i in range(n_chunks):
            chunk = batched[i]
            # chunk is already 24 kHz mono peak-normalized — go straight
            # to the core pipeline (no double preprocessing).
            out = self._run_pipeline(chunk, voice_id)
            outputs.append(out)

        # Trim the padding-induced tail off the concatenated output so
        # len(out) == len(input).
        return np.concatenate(outputs)[: len(wav)]

    # ------------------------------------------------------------------
    # Hot-swap API (for streaming.py / realtime_infer.py)
    # ------------------------------------------------------------------
    def select_voice(self, voice_id: int) -> None:
        """Validate that voice_id is in range. (kNN-VC is O(1) on the
        retrieve path, so this is essentially a no-op assertion.)"""
        if not (0 <= voice_id < self.n_voices):
            raise ValueError(f"voice_id {voice_id} out of range [0, {self.n_voices})")


# ---------------------------------------------------------------------------
# Module-level singleton for benchmark.py / realtime_infer.py compatibility.
# ---------------------------------------------------------------------------
_default_infer: V1Infer | None = None


def process_audio(wav: np.ndarray, sr: int, voice_id: int = 0) -> np.ndarray:
    """Module-level shortcut using a cached singleton V1Infer.

    Used by `scripts/benchmark.py` and `scripts/realtime_infer.py` so they
    can call ``vc_realtime.infer_v1.process_audio(wav, sr, voice_id)``
    without instantiating the heavy encoder/decoder per call.
    """
    global _default_infer
    if _default_infer is None:
        _default_infer = V1Infer()
    return _default_infer.process_audio(wav, sr, voice_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vc_realtime.infer_v1",
        description="v1 baseline VC: TinyVC encoder + kNN-VC + DDSP decoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", required=True, help="source wav path (any sr, mono or stereo).")
    p.add_argument(
        "--voice-id", type=int, default=0, help="target voice index (0..4 for 5 voices)."
    )
    p.add_argument("--output", required=True, help="output wav path (24 kHz mono PCM16).")
    p.add_argument(
        "--models-dir", default="models", help="directory with encoder.pt, decoder.pt, voices.pt"
    )
    p.add_argument("--device", default="cpu", help="torch device (default: cpu).")
    p.add_argument(
        "--pitch-shift", type=float, default=0.0, help="pitch shift in semitones (default: 0)."
    )
    p.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help=f"kNN-VC top-k (default: {DEFAULT_TOP_K})."
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"kNN-VC blend factor 0..1 (default: {DEFAULT_ALPHA}).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    wav, sr = sf.read(args.source, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    infer = V1Infer(
        models_dir=args.models_dir, device=args.device, top_k=args.top_k, alpha=args.alpha
    )
    out = infer.process_audio(
        wav, sr, voice_id=args.voice_id, pitch_shift_semitones=args.pitch_shift
    )

    # PCM16 output, clipped to [-1, 1]
    out_clip = np.clip(out, -1.0, 1.0)
    sf.write(args.output, (out_clip * 32767).astype(np.int16), SAMPLE_RATE)

    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    dur = len(out) / SAMPLE_RATE
    print(f"[infer_v1] source sr={sr}, voice_id={args.voice_id}, pitch_shift={args.pitch_shift}")
    print(f"[infer_v1] wrote {args.output}: {SAMPLE_RATE}Hz mono, {dur:.3f}s, output RMS={rms:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Local alias so `from vc_realtime.infer_v1 import autopad_waveform` etc.
# works in case glue code imports it.
autopad_waveform = _tinyvc_autopad
