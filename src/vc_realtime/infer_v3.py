"""
vc_realtime.infer_v3 — v3 hybrid inference (TinyVC + kNN + DDSP + Spark
BiCodec SpeakerEncoder for offline 1024-d + 48-byte FSQ speaker embedding).

Architecture (verified shapes)
-----------------------------
The v3 hybrid design is *option A* — same as v2 — keeping the v1 baseline
runtime path (TinyVC encoder + kNN-VC retrieval + DDSP decoder) intact and
adding a *parallel* offline-only path that uses the Spark BiCodec
SpeakerEncoder ONNX (P3-1 export) to produce a richer 1024-d speaker
embedding (vs v2's 256-d OpenVoice RefEncoder) + 48-byte FSQ codes for
hash-keyed O(1) retrieval.

Architectural mismatches (carried over from v2, NOT fixed):

  TinyVC encoder output (verified):  content [1, 768, T] @ 50 Hz (hop=480 @ 24 kHz)
  OpenVoice flow input (verified):   content [B, 192, T] (needs 192-d, NOT 768-d!)
  Vocos input (verified):            mel     [B, 100, T_mel] @ 93.75 Hz
                                      (needs 100-mel spectrogram, NOT content features!)

Bridging the mismatches would require training projection layers
(``Linear(768→192)`` and ``Linear(192→100) + TimeUpsample(50→93.75 Hz)``),
which is out of scope for the v3 Python phase (no GPU + no parallel data here).
v3.5 / v4 future work will train the projection layers.

v3 vs v2 differences (verified):
  - Spark BiCodec outputs **1024-d** d_vector (vs OpenVoice 256-d)
  - Spark BiCodec outputs **FSQ codes** (32 int64 in [0, 4095], 48 bytes
    packed) — vs OpenVoice's float-only 256-d → enables hash-keyed O(1)
    voice retrieval (no float-distance search needed at registration time)
  - Spark ONNX: 14.06 M params, 55 MB FP32 / ~25-30 MB INT8
    (vs OpenVoice 0.76 M, 3 MB / 2.3 MB)
  - Per-voice storage: 1024-d FP16 + 48 bytes FSQ = 2096 bytes
    (v3 total for 5 voices: ~10.5 KB; vs v2's 2.9 KB — slightly larger
    but richer speaker representation, and 2200× smaller than v1's
    23 MB kNN index)

Runtime path: identical to v1/v2 (TinyVC encoder + kNN + DDSP).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Optional, used to resample to 16 kHz for the Spark BiCodec input.
try:
    import librosa  # noqa: F401

    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover — librosa is in requirements.txt
    _HAS_LIBROSA = False

import onnxruntime as ort  # noqa: E402

# Inherit the v1 baseline — runtime path is identical to v1 / v2.
from vc_realtime.infer_v1 import (  # noqa: E402
    SAMPLE_RATE,  # noqa: E402
    V1Infer,  # noqa: E402
)

# Re-use the corrected P3-1 constants and FSQ pack/unpack helpers.
from vc_realtime.speaker_encoder_v3 import (  # noqa: E402
    SPARK_HOP,  # noqa: E402
    SPARK_N_FFT,  # noqa: E402
    SPARK_N_MELS,  # noqa: E402
    SPARK_OUT_DIM,  # noqa: E402
    SPARK_REF_DURATION_S,  # noqa: E402
    SPARK_SAMPLE_RATE,  # noqa: E402
    pack_fsq_to_48_bytes,  # noqa: E402
    )

DEFAULT_NORM_DB = -3.0  # peak-normalize to -3 dBFS (matches v1 / v2 convention)


class V3Infer(V1Infer):
    """v3 hybrid inference: v1 baseline (TinyVC + kNN + DDSP) runtime path
    + Spark BiCodec SpeakerEncoder ONNX for offline 1024-d + 48-byte FSQ
    speaker embedding storage.

    Runtime path (identical to v1 / v2):
      source_audio (24 kHz mono)
          → TinyVC SSLFeatureEstimator → content [1, 768, T] + f0 + energy
          → kNN-VC top-4 cosine retrieval → content_replaced [1, 768, T]
          → TinyVC DDSP decoder → output_audio (24 kHz mono)

    Offline path (NEW in v3, used by :meth:`register_voice`):
      reference_audio (any sr, mono)
          → resample to 16 kHz + truncate to 30 s + peak-normalize to -3 dBFS
          → librosa mel-spec (n_fft=1024, hop=320, n_mels=128, fmax=8000) → [128, T]
          → transpose+batch to [1, T, 128] (time-first, as the ONNX expects)
          → Spark BiCodec SpeakerEncoder ONNX → 3 outputs:
              d_vector    [1, 1024]  float32  — speaker conditioning (FP16 stored)
              fsq_indices [1, 1, 32] int64   — 32 × 12-bit FSQ codes (48 bytes packed)
              x_vector    [1, 1024]  float32  — auxiliary ECAPA-TDNN pooled emb
          → save FP16 d_vector + 48-byte FSQ to models/voices_v3.safetensors
  """

    SPARK_ENCODER_FP32 = "spark_speaker_encoder.onnx"
    SPARK_ENCODER_INT8 = "spark_speaker_encoder.int8.onnx"
    VOICES_V3_FILE = "voices_v3.safetensors"

    def __init__(
        self,
        models_dir: str | Path = "models",
        device: str = "cpu",
        top_k: int = 4,
        alpha: float = 0.0,
        use_int8: bool = False,
        load_spark: bool = True,
    ):
        """Create a v3 hybrid inference wrapper.

        Parameters
        ----------
        models_dir : str | Path
            Directory containing encoder.pt, decoder.pt, voices.pt
            (for the runtime TinyVC path) AND spark_speaker_encoder.onnx
            (for the offline Spark path).
        device, top_k, alpha : inherited from V1Infer.
        use_int8 : bool
            If True, prefer ``spark_speaker_encoder.int8.onnx`` (P3-2 INT8
            quantized). Falls back to FP32 if the INT8 file is missing.
            Default: False (the FP32 path was used in P3-1 export parity
            tests; INT8 is opt-in to keep determinism identical to P3-1).
        load_spark : bool
            If False, skip loading the Spark ONNX. Useful for runtime-only
            invocations where registration is not needed (saves ~55 MB RSS
            for FP32, ~25-30 MB for INT8).
        """
        # Initialize the v1 baseline (loads TinyVC encoder/decoder/voices.pt).
        super().__init__(models_dir=models_dir, device=device, top_k=top_k, alpha=alpha)

        models_dir = Path(models_dir)
        self._models_dir = models_dir
        self._use_int8 = use_int8

        # Lazily load Spark BiCodec SpeakerEncoder ONNX.
        self._spark_session: ort.InferenceSession | None = None
        if load_spark:
            self._spark_session = self._load_spark_session(models_dir, use_int8)

        # Load any pre-computed Spark 1024-d + 48-byte FSQ embeddings from
        # voices_v3.safetensors (created by register_voice). Empty dict if
        # the file does not exist (first run before any voice registered).
        self.spark_embeddings: dict[int, dict[str, object]] = self._load_spark_embeddings(
            models_dir
        )

    # ------------------------------------------------------------------
    # Spark ONNX session management
    # ------------------------------------------------------------------
    @classmethod
    def _load_spark_session(
        cls, models_dir: Path, use_int8: bool
    ) -> ort.InferenceSession:
        """Load the Spark BiCodec SpeakerEncoder ONNX (with INT8 fallback)."""
        int8_path = models_dir / cls.SPARK_ENCODER_INT8
        fp32_path = models_dir / cls.SPARK_ENCODER_FP32
        if use_int8 and int8_path.exists():
            path = int8_path
        elif fp32_path.exists():
            path = fp32_path
        elif int8_path.exists():
            # FP32 not requested but FP32 missing — use INT8 as fallback.
            path = int8_path
        else:
            raise FileNotFoundError(
                f"Spark BiCodec SpeakerEncoder ONNX not found in {models_dir} "
                f"(looked for {cls.SPARK_ENCODER_FP32} and "
                f"{cls.SPARK_ENCODER_INT8}). Run "
                f"scripts/export_spark_bicodec_onnx.py to produce it."
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def _load_spark_embeddings(cls, models_dir: Path) -> dict[int, dict[str, object]]:
        """Load pre-computed Spark 1024-d d_vectors + 48-byte FSQ codes.

        File format: safetensors with two tensors per voice, keys
        ``voice_<id>_dvec`` ([1024] FP16) and ``voice_<id>_fsq`` ([48] uint8).

        Returns an empty dict if the file does not exist (first run).
        """
        path = models_dir / cls.VOICES_V3_FILE
        if not path.exists():
            return {}
        try:
            from safetensors.torch import load_file
        except ImportError as e:  # pragma: no cover — safetensors in requirements
            raise RuntimeError(
                "safetensors is required to load voices_v3.safetensors; "
                "install with `pip install safetensors`."
            ) from e
        out: dict[int, dict[str, object]] = {}
        for key, tensor in load_file(str(path)).items():
            # key format: voice_<id>_dvec / voice_<id>_fsq
            try:
                vid = int(key.split("_")[1])
            except (ValueError, IndexError):
                continue
            out.setdefault(vid, {})
            if key.endswith("_dvec"):
                import torch as _torch
                if hasattr(tensor, "float") and isinstance(tensor, _torch.Tensor):
                    arr = tensor.float().numpy().astype(np.float32)
                else:
                    arr = np.asarray(tensor).astype(np.float32)
                out[vid]["d_vector"] = arr
            elif key.endswith("_fsq"):
                import torch as _torch
                if isinstance(tensor, _torch.Tensor):
                    arr = tensor.numpy().astype(np.uint8)
                else:
                    arr = np.asarray(tensor).astype(np.uint8)
                out[vid]["fsq_bytes"] = arr.tobytes()
        return out

    # ------------------------------------------------------------------
    # Spark BiCodec: mel-spec computation + ONNX run
    # ------------------------------------------------------------------
    @staticmethod
    def compute_spark_mel_spec(
        wav_16k: np.ndarray,
        n_fft: int = SPARK_N_FFT,
        hop: int = SPARK_HOP,
        n_mels: int = SPARK_N_MELS,
    ) -> np.ndarray:
        """Compute the mel-spectrogram the Spark BiCodec ONNX expects.

        Parameters
        ----------
        wav_16k : np.ndarray [L_samples] at 16 kHz, float32 in [-1, 1]

        Returns
        -------
        np.ndarray [1, T, 128] float32 — ready to feed to the ONNX session.
        """
        if not _HAS_LIBROSA:
            raise RuntimeError(
                "librosa is required to compute the Spark mel spec; "
                "install librosa or pre-compute the spec offline."
            )
        wav = np.asarray(wav_16k, dtype=np.float32)
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=SPARK_SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            fmin=0,
            fmax=8000,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel_db = (mel_db + 80.0) / 20.0  # normalize to [0, 4] roughly
        # ONNX expects time-first: [B, T, n_mels]
        return mel_db.T[None, ...]

    def encode_spark_embedding(self, wav_16k: np.ndarray) -> tuple[np.ndarray, bytes, np.ndarray]:
        """Run the Spark BiCodec SpeakerEncoder ONNX on a 16 kHz wav.

        Used OFFLINE during voice registration (not at runtime).

        Parameters
        ----------
        wav_16k : np.ndarray [L_samples] float32 at 16 kHz.

        Returns
        -------
        (d_vector [1024] float32,
         fsq_bytes (48 bytes),
         x_vector [1024] float32)
        """
        if self._spark_session is None:
            raise RuntimeError(
                "Spark ONNX not loaded (load_spark=False). Re-instantiate "
                "V3Infer with load_spark=True to register voices."
            )
        spec_in = self.compute_spark_mel_spec(wav_16k)
        input_name = self._spark_session.get_inputs()[0].name  # 'mel_spec'
        outputs = self._spark_session.run(None, {input_name: spec_in})
        # P3-1 verified output order: d_vector, fsq_indices, x_vector
        d_vector = outputs[0]      # [B, 1024] float32
        fsq_indices = outputs[1]   # [B, 1, 32] int64
        x_vector = outputs[2]      # [B, 1024] float32
        d_vec = d_vector[0] if d_vector.ndim == 2 and d_vector.shape[0] == 1 else d_vector
        x_vec = x_vector[0] if x_vector.ndim == 2 and x_vector.shape[0] == 1 else x_vector
        fsq_bytes = pack_fsq_to_48_bytes(fsq_indices[0])
        return (
            np.ascontiguousarray(d_vec, dtype=np.float32),
            fsq_bytes,
            np.ascontiguousarray(x_vec, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Voice registration (offline)
    # ------------------------------------------------------------------
    def register_voice(
        self, wav_path: str | Path, voice_id: int, persist: bool = True
    ) -> np.ndarray:
        """OFFLINE: encode a 30 s voice reference wav to a 1024-d Spark d_vector
        + 48-byte FSQ code and persist them to ``models/voices_v3.safetensors``.

        Steps:
          1. Load wav (any sr, mono or stereo).
          2. Resample to 16 kHz (Spark native rate).
          3. Truncate to 30 s (Spark default reference length).
          4. Peak-normalize to -3 dBFS (matches v1 / v2 convention).
          5. Compute mel spec (n_fft=1024, hop=320, n_mels=128, fmax=8000)
             → [1, T, 128].
          6. Run Spark BiCodec SpeakerEncoder ONNX → 3 outputs:
             - d_vector    [1, 1024]  float32
             - fsq_indices [1, 1, 32] int64   (48 bytes packed)
             - x_vector    [1, 1024]  float32  (auxiliary, discarded)
          7. Save FP16 d_vector + 48-byte FSQ to models/voices_v3.safetensors
             (keys: voice_<id>_dvec, voice_<id>_fsq).

        Parameters
        ----------
        wav_path : str | Path — reference wav (any sr, mono or stereo).
        voice_id : int — index 0..N-1 (must match the v1 kNN voice_id).
        persist : bool — if True (default), write the embeddings to disk.

        Returns
        -------
        np.ndarray [1024] float32 — the d_vector (also cached in self.spark_embeddings).
        """
        if not _HAS_LIBROSA:
            raise RuntimeError("librosa required for register_voice (resampling).")
        wav, sr = sf.read(str(wav_path), always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)  # stereo → mono
        wav = wav.astype(np.float32)
        if sr != SPARK_SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SPARK_SAMPLE_RATE)
        # Truncate to 30 s (Spark default reference length)
        max_samples = int(SPARK_SAMPLE_RATE * SPARK_REF_DURATION_S)
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        # Peak-normalize to -3 dBFS
        peak = float(np.max(np.abs(wav))) + 1e-8
        wav = wav * (10 ** (DEFAULT_NORM_DB / 20.0) / peak)
        # Encode
        d_vector, fsq_bytes, x_vector = self.encode_spark_embedding(wav)
        # Cache in-memory
        self.spark_embeddings[voice_id] = {
            "d_vector": d_vector,
            "fsq_bytes": fsq_bytes,
            "x_vector": x_vector,
        }
        # Persist
        if persist:
            self._save_embeddings_to_safetensors()
            print(
                f"[infer_v3] registered voice {voice_id} from {wav_path}: "
                f"1024-d d_vector (norm={float(np.linalg.norm(d_vector)):.3f}) "
                f"+ 48-byte FSQ saved to "
                f"{self._models_dir / self.VOICES_V3_FILE}"
            )
        return d_vector

    def _save_embeddings_to_safetensors(self) -> None:
        """Persist self.spark_embeddings to models/voices_v3.safetensors.

        Per-voice tensors:
          - ``voice_<id>_dvec`` : torch.Tensor [1024] float16  (2 KB)
          - ``voice_<id>_fsq``  : torch.Tensor [48]   uint8    (48 B)
        """
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "safetensors + torch required to save voices_v3.safetensors."
            ) from e
        self._models_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._models_dir / self.VOICES_V3_FILE
        tensors: dict[str, torch.Tensor] = {}
        for vid, emb in sorted(self.spark_embeddings.items()):
            d_vec = np.asarray(emb["d_vector"], dtype=np.float32)
            tensors[f"voice_{vid}_dvec"] = torch.from_numpy(d_vec).to(torch.float16)
            fsq = emb["fsq_bytes"]
            if isinstance(fsq, (bytes, bytearray)):
                # np.array copies to a writable buffer so torch.from_numpy
                # does not emit a "buffer not writable" UserWarning.
                fsq_arr = np.array(
                    np.frombuffer(fsq, dtype=np.uint8), dtype=np.uint8
                )
            else:
                fsq_arr = np.asarray(fsq, dtype=np.uint8)
            tensors[f"voice_{vid}_fsq"] = torch.from_numpy(fsq_arr)
        if not tensors:
            if out_path.exists():
                out_path.unlink()
            return
        save_file(tensors, str(out_path))

    def get_spark_embedding(self, voice_id: int) -> np.ndarray | None:
        """Lookup the Spark 1024-d d_vector for ``voice_id``, or None.

        v3 does NOT actually use the Spark d_vector at runtime (the v1 kNN
        path is still the active converter). It will be wired up in v3.5 /
        v4 when the projection layers are trained to bridge the
        768→192-dim gap to the OpenVoice flow.
        """
        entry = self.spark_embeddings.get(voice_id)
        return None if entry is None else entry.get("d_vector")

    def get_spark_fsq_bytes(self, voice_id: int) -> bytes | None:
        """Lookup the packed 48-byte FSQ for ``voice_id``, or None."""
        entry = self.spark_embeddings.get(voice_id)
        return None if entry is None else entry.get("fsq_bytes")


# ---------------------------------------------------------------------------
# Module-level singleton for benchmark.py / realtime_infer.py compatibility.
# ---------------------------------------------------------------------------
_default_infer: V3Infer | None = None


def process_audio(wav: np.ndarray, sr: int, voice_id: int = 0) -> np.ndarray:
    """Module-level shortcut using a cached singleton V3Infer.

    Used by ``scripts/benchmark.py`` and ``scripts/realtime_infer.py`` so
    they can call ``vc_realtime.infer_v3.process_audio(wav, sr, voice_id)``
    without instantiating the heavy encoder/decoder per call.

    Note: the singleton is created with ``load_spark=False`` because the
    runtime path does NOT need the Spark BiCodec ONNX — that ONNX is only
    used during offline ``register_voice`` calls. This saves ~55 MB RSS
    (FP32) or ~25-30 MB (INT8) and ~1.5 s startup time.
    """
    global _default_infer
    if _default_infer is None:
        _default_infer = V3Infer(load_spark=False)
    return _default_infer.process_audio(wav, sr, voice_id=voice_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vc_realtime.infer_v3",
        description=(
            "v3 hybrid VC: TinyVC encoder + kNN-VC + DDSP (runtime, "
            "identical to v1 / v2) + Spark BiCodec SpeakerEncoder (offline "
            "1024-d d_vector + 48-byte FSQ speaker embedding storage)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        required=True,
        help="Source wav path (any sr, mono or stereo). For --register-voice, "
        "this is the reference voice wav.",
    )
    p.add_argument(
        "--voice-id", type=int, default=0,
        help="Target voice index (0..4 for 5 voices).",
    )
    p.add_argument(
        "--output", required=False, default=None,
        help="Output wav path (24 kHz mono PCM16). Required for runtime mode, "
        "ignored for --register-voice.",
    )
    p.add_argument(
        "--models-dir", default="models",
        help="Directory with encoder.pt, decoder.pt, voices.pt, "
        "spark_speaker_encoder.onnx.",
    )
    p.add_argument("--device", default="cpu", help="torch device (default: cpu).")
    p.add_argument(
        "--pitch-shift", type=float, default=0.0,
        help="Pitch shift in semitones (default: 0).",
    )
    p.add_argument(
        "--top-k", type=int, default=4, help="kNN-VC top-k (default: 4).",
    )
    p.add_argument(
        "--alpha", type=float, default=0.0,
        help="kNN-VC blend factor 0..1 (default: 0.0 = full replace).",
    )
    p.add_argument(
        "--register-voice", action="store_true",
        help="OFFLINE mode: register a voice reference wav → 1024-d Spark "
        "d_vector + 48-byte FSQ saved to models/voices_v3.safetensors. "
        "No inference run.",
    )
    p.add_argument(
        "--use-int8", action="store_true",
        help="Prefer the INT8-quantized Spark ONNX if available (P3-2).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.register_voice:
        # OFFLINE: encode the reference wav → 1024-d + 48-byte FSQ → save.
        infer = V3Infer(
            models_dir=args.models_dir,
            device=args.device,
            top_k=args.top_k,
            alpha=args.alpha,
            use_int8=args.use_int8,
            load_spark=True,
        )
        infer.register_voice(args.source, args.voice_id)
        return 0

    if args.output is None:
        print("ERROR: --output is required for runtime inference mode.",
              file=sys.stderr)
        return 2

    wav, sr = sf.read(args.source, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    # Runtime: do NOT load the Spark BiCodec ONNX (saves RSS).
    infer = V3Infer(
        models_dir=args.models_dir,
        device=args.device,
        top_k=args.top_k,
        alpha=args.alpha,
        use_int8=args.use_int8,
        load_spark=False,
    )
    out = infer.process_audio(
        wav, sr, voice_id=args.voice_id,
        pitch_shift_semitones=args.pitch_shift,
    )

    out_clip = np.clip(out, -1.0, 1.0)
    sf.write(args.output, (out_clip * 32767).astype(np.int16), SAMPLE_RATE)

    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    dur = len(out) / SAMPLE_RATE
    print(f"[infer_v3] source sr={sr}, voice_id={args.voice_id}, "
          f"pitch_shift={args.pitch_shift}")
    print(f"[infer_v3] wrote {args.output}: {SAMPLE_RATE}Hz mono, "
          f"{dur:.3f}s, output RMS={rms:.4f}")
    print("[infer_v3] (runtime path identical to v1 / v2; Spark BiCodec "
          "SpeakerEncoder loaded only at registration time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
