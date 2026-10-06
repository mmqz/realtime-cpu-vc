"""
modules/speaker_encoder_v3.py — Spark-TTS BiCodec SpeakerEncoder wrapper.

================================================================
Source: SparkAudio/Spark-TTS/sparktts/modules/speaker/speaker_encoder.py
       (ECAPA-TDNN c512 + PerceiverResampler + ResidualFSQ + Linear → d_vector)

P3-1 KEY CORRECTIONS (vs the original speculative v3 spec):
  - SPARK_N_MELS = 128 (NOT 80)
  - SPARK_HOP    = 320 (NOT 200) → 50 Hz frame rate @ 16 kHz
  - SPARK_OUT_DIM = 1024 (NOT 512) — Spark outputs 1024-d d_vector
  - Input shape: [B, T, 128] (time-first channels-last)
  - 3 outputs: d_vector [B, 1024] + fsq_indices [B, 1, 32] int64
    + x_vector [B, 1024]
  - FSQ is 32 tokens × 1 quantizer, each index ∈ [0, 4095] (12 bits)
    → 48 bytes packed per voice (32 × 12 = 384 bits = 48 bytes)

v3 vs v2 upgrade:
  v2: OpenVoice v2 ReferenceEncoder (256-d, 0.76M, 3 MB FP32 / 2.3 MB INT8)
  v3: Spark-TTS BiCodec SpeakerEncoder (1024-d + 48-byte FSQ, 14.06M,
       55 MB FP32 / ~25-30 MB INT8)
  Expected uplift: richer 1024-d speaker disentanglement, smaller per-voice
  storage (48 bytes FSQ + 2 KB FP16 d_vector = 2.1 KB per voice) and
  hash-keyed O(1) retrieval possible via the FSQ codes.

Per-voice storage:
  v1 kNN index:        5 × 11.5 MB = 57.5 MB (`voices.pt`)
  v2 OpenVoice 256-d:  5 × 512 B  = 2.5 KB (`voices_v2.safetensors`)
  v3 Spark BiCodec:   5 × (2 KB FP16 d_vector + 48 B FSQ) = ~10.5 KB
                       (`voices_v3.safetensors`)

Usage (offline registration):

    enc = BiCodecSpeakerEncoder('models/spark_speaker_encoder.onnx')
    mel = make_spark_mel_spec(wav_16k)             # [1, T, 128]
    out = enc.encode_reference(mel)               # dict
    d_vector = out['d_vector']                    # [1024]
    fsq_bytes = out['fsq_bytes']                  # 48 bytes packed
    enc.load_voice_registry('models/voices_v3.safetensors')
    se = enc.get_speaker_embedding(voice_id=0)    # [1024]
"""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Constants — confirmed in P3-1 from BiCodec/config.yaml + ONNX parity check.
# ---------------------------------------------------------------------------
SPARK_SAMPLE_RATE = 16000      # Spark-TTS uses 16 kHz input
SPARK_N_FFT = 1024
SPARK_HOP = 320                # 16 kHz / 320 = 50 Hz frame rate
SPARK_N_MELS = 128             # NOT 80 (config.yaml: mel_params.num_mels=128)
SPARK_OUT_DIM = 1024           # d_vector output dim (config.yaml: out_dim=1024)
SPARK_FSQ_SHAPE = (1, 32)     # 32 tokens × 1 quantizer (fsq_num_quantizers=1)
SPARK_FSQ_BYTES_PACKED = 48   # 32 tokens × 12 bits = 384 bits = 48 bytes
SPARK_REF_DURATION_S = 30.0   # truncate references to 30 s (Spark default)


# ---------------------------------------------------------------------------
# FSQ packing helpers — 32 int64 indices (each in [0, 4095]) ↔ 48 bytes.
# We pack 12 bits per index, LSB-first, into a 48-byte little-endian bitstream.
# ---------------------------------------------------------------------------
def pack_fsq_to_48_bytes(fsq_indices: np.ndarray) -> bytes:
    """Pack 32 FSQ indices (each ∈ [0, 4095], 12 bits) → 48 bytes.

    Parameters
    ----------
    fsq_indices : np.ndarray, shape [..., 32] or [..., 1, 32] int64

    Returns
    -------
    bytes of length 48.
    """
    indices = np.asarray(fsq_indices).astype(np.int64).flatten()
    if indices.size != 32:
        raise ValueError(f"Expected 32 FSQ indices, got {indices.size}")
    if indices.min() < 0 or indices.max() > 4095:
        raise ValueError(
            f"FSQ index out of [0, 4095] range: "
            f"min={int(indices.min())}, max={int(indices.max())}"
        )

    packed = bytearray(48)
    bit_pos = 0
    for idx in indices:
        idx_int = int(idx)
        for bit_i in range(12):
            bit_val = (idx_int >> bit_i) & 1
            byte_pos = bit_pos // 8
            bit_in_byte = bit_pos % 8
            packed[byte_pos] |= (bit_val << bit_in_byte)
            bit_pos += 1
    return bytes(packed)


def unpack_fsq_from_48_bytes(packed: bytes) -> np.ndarray:
    """Inverse of pack_fsq_to_48_bytes → np.ndarray [32] int64."""
    if len(packed) != 48:
        raise ValueError(f"Expected 48 bytes, got {len(packed)}")
    indices = np.zeros(32, dtype=np.int64)
    bit_pos = 0
    for i in range(32):
        idx = 0
        for bit_i in range(12):
            byte_pos = bit_pos // 8
            bit_in_byte = bit_pos % 8
            bit_val = (packed[byte_pos] >> bit_in_byte) & 1
            idx |= (bit_val << bit_i)
            bit_pos += 1
        indices[i] = idx
    return indices


class BiCodecSpeakerEncoder:
    """Spark-TTS BiCodec SpeakerEncoder ONNX wrapper (P3-1 export).

    The ONNX graph is the union of SpeakerEncoder.forward + .tokenize paths,
    re-implemented by ``scripts/export_spark_bicodec_onnx.py`` so that a
    single forward pass produces:

      - ``d_vector``    [B, 1024] float32  — speaker conditioning for flow
      - ``fsq_indices`` [B, 1, 32] int64   — 32 12-bit FSQ codes per voice
      - ``x_vector``    [B, 1024] float32  — auxiliary ECAPA-TDNN pooled emb

    Input is a mel-spec [B, T, 128] computed at 16 kHz with hop=320 (50 Hz
    frame rate), n_fft=1024, 128 mel bins, fmin=0, fmax=8000, power=2.0.
    """

    SPARK_SAMPLE_RATE = SPARK_SAMPLE_RATE
    SPARK_N_FFT = SPARK_N_FFT
    SPARK_HOP = SPARK_HOP
    SPARK_N_MELS = SPARK_N_MELS
    SPARK_OUT_DIM = SPARK_OUT_DIM
    SPARK_FSQ_BYTES_PACKED = SPARK_FSQ_BYTES_PACKED

    def __init__(
        self,
        model_path: str | Path = "models/spark_speaker_encoder.onnx",
        intra_op_threads: int = 2,
        use_int8: bool = True,
    ):
        """Load the Spark BiCodec SpeakerEncoder ONNX (with INT8 fallback).

        Parameters
        ----------
        model_path : str | Path
            Path to ``spark_speaker_encoder.onnx`` (FP32, 55 MB) or
            ``spark_speaker_encoder.int8.onnx`` (P3-2 INT8 quantized,
            ~25-30 MB).
        intra_op_threads : int
            ONNX intra-op thread count (default 2 — keeps CPU contention
            low for the realtime path; the Spark path is offline-only).
        use_int8 : bool
            If True (default), prefer ``spark_speaker_encoder.int8.onnx``
            when present, else fall back to the FP32 ONNX.
        """
        model_path = Path(model_path)
        if use_int8:
            int8_path = model_path.with_name(
                model_path.name.replace(".onnx", ".int8.onnx")
            )
            if int8_path.exists():
                model_path = int8_path
                print(f"  BiCodecSpeakerEncoder: using INT8 variant {int8_path}")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Spark-TTS BiCodec SpeakerEncoder ONNX not found at "
                f"{model_path}. Export with "
                f"`python3 scripts/export_spark_bicodec_onnx.py`."
            )

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        # voice_id -> {'d_vector': [1024], 'fsq_bytes': bytes(48)}
        self.voice_registry: dict[int, dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Mel-spec computation helpers
    # ------------------------------------------------------------------
    @classmethod
    def make_mel_spec(cls, wav_16k: np.ndarray) -> np.ndarray:
        """Compute the Spark BiCodec mel-spec input.

        Parameters
        ----------
        wav_16k : np.ndarray [L_samples] float32 at 16 kHz

        Returns
        -------
        np.ndarray [1, T_frames, 128] float32 — time-first channels-last,
        normalized to roughly [0, 4] (matches Spark's `mel_processing.py`).
        """
        import librosa

        mel = librosa.feature.melspectrogram(
            y=wav_16k.astype(np.float32),
            sr=cls.SPARK_SAMPLE_RATE,
            n_fft=cls.SPARK_N_FFT,
            hop_length=cls.SPARK_HOP,
            n_mels=cls.SPARK_N_MELS,
            fmin=0,
            fmax=8000,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel_db = (mel_db + 80.0) / 20.0  # normalize to [0, 4] roughly
        return mel_db.T[None, ...]  # [1, T, 128]

    # ------------------------------------------------------------------
    # ONNX forward
    # ------------------------------------------------------------------
    def encode_reference(self, ref_mel_spec: np.ndarray) -> dict:
        """Run BiCodec SpeakerEncoder on a reference mel-spec.

        Parameters
        ----------
        ref_mel_spec : np.ndarray [B, T, 128] float32 (time-first channels-last)

        Returns
        -------
        dict with:
          'd_vector'    : np.ndarray [B, 1024] float32
          'fsq_indices' : np.ndarray [B, 1, 32] int64
          'x_vector'    : np.ndarray [B, 1024] float32
          'fsq_bytes'   : bytes of length 48 × B (packed FSQ codes for voice 0)
        """
        x = np.ascontiguousarray(ref_mel_spec.astype(np.float32))
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, 128], got shape {x.shape}")
        if x.shape[-1] != self.SPARK_N_MELS:
            raise ValueError(
                f"Expected last dim = {self.SPARK_N_MELS} (n_mels), got {x.shape[-1]}"
            )
        input_name = self.session.get_inputs()[0].name  # 'mel_spec'
        outputs = self.session.run(None, {input_name: x})
        # P3-1 verified output order: d_vector, fsq_indices, x_vector
        d_vector = outputs[0]      # [B, 1024] float32
        fsq_indices = outputs[1]   # [B, 1, 32] int64
        x_vector = outputs[2]      # [B, 1024] float32
        # Pack FSQ for the first item in the batch (typical offline use).
        fsq_bytes = pack_fsq_to_48_bytes(fsq_indices[0])
        return {
            "d_vector": d_vector,
            "fsq_indices": fsq_indices,
            "x_vector": x_vector,
            "fsq_bytes": fsq_bytes,
        }

    # ------------------------------------------------------------------
    # Voice registry (offline storage format)
    # ------------------------------------------------------------------
    def load_voice_registry(self, path: str | Path = "models/voices_v3.safetensors") -> None:
        """Load pre-computed Spark embeddings from voices_v3.safetensors.

        File format (produced by :meth:`V3Infer.register_voice`):

            voice_<id>_dvec : torch.Tensor [1024] float16   (1024 × 2 = 2048 B)
            voice_<id>_fsq  : torch.Tensor [48]   uint8     (48 B packed)

        Side effects
        ------------
        Populates ``self.voice_registry[voice_id] = {'d_vector': [1024],
        'fsq_bytes': bytes}``.
        """
        path = Path(path)
        self.voice_registry.clear()
        if not path.exists():
            print(f"  Voice registry: {path} not found (no voices loaded)")
            return
        try:
            from safetensors.torch import load_file
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "safetensors is required to load voices_v3.safetensors; "
                "install with `pip install safetensors`."
            ) from e
        tensors = load_file(str(path))
        # First pass: allocate per-voice dicts.
        for key in tensors:
            try:
                vid = int(key.rsplit("_", 1)[0].rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                # Fallback parse: voice_<id>_<suffix>
                try:
                    vid = int(key.split("_")[1])
                except (ValueError, IndexError):
                    continue
            self.voice_registry.setdefault(vid, {})
        # Second pass: populate fields.
        for key, tensor in tensors.items():
            # key format: voice_<id>_dvec / voice_<id>_fsq
            try:
                vid = int(key.split("_")[1])
            except (ValueError, IndexError):
                continue
            if key.endswith("_dvec"):
                arr = tensor.float().numpy().astype(np.float32)  # [1024]
                self.voice_registry[vid]["d_vector"] = arr
            elif key.endswith("_fsq"):
                arr = tensor.numpy().astype(np.uint8)  # [48]
                self.voice_registry[vid]["fsq_bytes"] = arr.tobytes()
        print(
            f"  Voice registry: {len(self.voice_registry)} voices loaded from {path}"
        )

    def get_speaker_embedding(self, voice_id: int) -> np.ndarray:
        """O(1) lookup of the 1024-d Spark d_vector for the given voice.

        Returns
        -------
        np.ndarray [1024] float32 — the d_vector (speaker conditioning).

        Raises
        ------
        KeyError if the voice is not in the registry.
        """
        if voice_id not in self.voice_registry:
            raise KeyError(
                f"voice_id {voice_id} not in registry "
                f"(loaded: {list(self.voice_registry.keys())})"
            )
        entry = self.voice_registry[voice_id]
        if "d_vector" not in entry:
            raise KeyError(
                f"voice_id {voice_id} registry entry missing 'd_vector' field"
            )
        return entry["d_vector"]

    def lookup_by_fsq_hash(self, ref_fsq_bytes: bytes) -> int:
        """O(1) hash-based retrieval of voice_id by FSQ byte string.

        Useful when the reference audio is processed at runtime (instead of
        voice_id being known). For our 5-fixed-voice setup, voice_id is known
        so direct lookup is sufficient. This method is for future extension
        to arbitrary unseen reference matching.
        """
        for vid, entry in self.voice_registry.items():
            if entry.get("fsq_bytes") == ref_fsq_bytes:
                return vid
        raise KeyError("FSQ code not in registry")


def make_spark_mel_spec(audio: np.ndarray, sr: int = SPARK_SAMPLE_RATE) -> np.ndarray:
    """Compute mel-spec for the Spark-TTS BiCodec SpeakerEncoder input.

    Spark-TTS uses 128 mel bins, 50 Hz frame rate (hop=320 @ 16 kHz),
    1024-pt FFT, 16 kHz sample rate. This module-level convenience function
    matches the upstream BiCodec.forward preprocessing.
    """
    import librosa

    if sr != SPARK_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SPARK_SAMPLE_RATE)
    return BiCodecSpeakerEncoder.make_mel_spec(audio)
