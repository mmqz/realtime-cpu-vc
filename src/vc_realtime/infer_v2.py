"""
from typing import Any
vc_realtime.infer_v2 — v2 hybrid inference (Option A: minimal integration).

Architecture (verified shapes)
-----------------------------
The original v3 hybrid design assumed OpenVoice flow + Vocos can directly take
TinyVC content features. **This is incorrect** — there are TWO architectural
dimension mismatches that were uncovered by loading the actual ONNX graphs
exported in P2-1/P2-2:

  TinyVC encoder output (verified):  content [1, 768, T] @ 50 Hz (hop=480 @ 24 kHz)
  OpenVoice flow input (verified):  content [B, 192, T] (needs 192-d, NOT 768-d!)
  Vocos input (verified):           mel     [B, 100, T_mel] @ 93.75 Hz
                                    (needs 100-mel spectrogram, NOT content features!)

Bridging the mismatches would require training projection layers
(``Linear(768→192)`` and ``Linear(192→100) + TimeUpsample(50→93.75 Hz)``),
which is out of scope for the v2 Python phase (no GPU + no parallel data here).

Option A (this module): a *minimal but real* integration that produces
working code today:

  - The runtime inference path is kept identical to v1 (TinyVC encoder + kNN-VC
    retrieval + DDSP decoder — proven to RTF=0.1086 with 9/9 tests passing).
  - The OpenVoice v2 ``ReferenceEncoder`` ONNX (P2-1) is wired up for
    OFFLINE voice registration only: a 30 s reference wav → 256-d speaker
    embedding stored as FP16 in ``models/voices_v2.safetensors``. This is the
    *real* OpenVoice v2 model running in production mode, not a stub.
  - Storage drops from 5 × 11.5 MB kNN index (v1) → 5 × 512 B FP16 embeddings
    (v2) — a ~23 000× reduction in *speaker embedding* footprint. (The v1
    kNN index is still loaded for the runtime retrieval path; the v2
    embedding is a *parallel* representation that will be used at v2.5.)
  - OpenVoice flow ONNX (P2-1) and Vocos ONNX (P2-2) are EXPORTED and
    self-tested by their respective subagents, but they are NOT invoked at
    runtime in v2 — they await the v2.5 projection-layer training.

OpenVoice RefEncoder input — CORRECT shape (P2-1 worklog)
---------------------------------------------------------
The original spec (`vc_realtime/speaker_encoder.py`) says the OpenVoice
ReferenceEncoder takes an 80-mel spectrogram. **This is wrong.** The actual
ONNX graph (verified in P2-1 and re-verified below) takes a LINEAR spectrogram
of shape ``[B, T, 513]`` (time-first, NOT mel-first) computed at 22.05 kHz
with ``n_fft=1024, hop=256, win=1024``. This matches what
``spectrogram_torch(...)`` produces after `.transpose(1, 2)` in OpenVoice's
`api.py` line 131. This module uses the real shape.

CLI usage
---------
Offline voice registration (one-time per voice)::

    python3 -m vc_realtime.infer_v2 --register-voice \
        --source data/voices/voice_0.wav --voice-id 0

Runtime inference (same CLI as v1)::

    python3 -m vc_realtime.infer_v2 --source data/source/source_001.wav \
        --voice-id 0 --output /tmp/out_v2.wav

Benchmarking::

    python3 scripts/benchmark.py --infer-script vc_realtime.infer_v2 \
        --source data/source/source_001.wav --voice-id 0 \
        --ref data/voices/voice_0.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Optional, used to resample to 22.05 kHz for OpenVoice input.
try:
    import librosa  # noqa: F401

    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover — librosa is in requirements.txt
    _HAS_LIBROSA = False

# Import the v1 baseline — we inherit the TinyVC + kNN + DDSP runtime path.
# This is deliberate: v2's runtime path is *identical* to v1; the OpenVoice
# ReferenceEncoder is offline-only. Inheriting keeps the runtime behavior
# provably identical to the v1 baseline that already passes 9/9 tests.
import onnxruntime as ort  # noqa: E402

from vc_realtime.infer_v1 import (
    SAMPLE_RATE,  # noqa: E402
    V1Infer,  # noqa: E402
)

# ---------------------------------------------------------------------------
# OpenVoice v2 ReferenceEncoder constants (verified P2-1)
# ---------------------------------------------------------------------------
# The ONNX expects a LINEAR spectrogram (not mel), shape [B, T, 513],
# at 22.05 kHz with n_fft=1024, hop_length=256, win_length=1024.
OPENVOICE_SAMPLE_RATE = 22050
OPENVOICE_N_FFT = 1024
OPENVOICE_HOP = 256
OPENVOICE_WIN = 1024
OPENVOICE_SPEC_DIM = 513  # n_fft // 2 + 1
OPENVOICE_EMB_DIM = 256

# Offline registration: 30 s reference is the OpenVoice default. Truncating
# (rather than padding) keeps the GRU state stationary in time so that two
# wavs of slightly different lengths still produce comparable embeddings.
OPENVOICE_REF_DURATION_S = 30.0

DEFAULT_NORM_DB = -3.0  # peak-normalize to -3 dBFS (matches v1)


class V2Infer(V1Infer):
    """v2 hybrid inference: v1 baseline (TinyVC + kNN + DDSP) runtime path
    + OpenVoice v2 ReferenceEncoder ONNX for offline 256-d speaker embedding.

    Runtime path (identical to v1):
      source_audio (24 kHz mono)
          → TinyVC SSLFeatureEstimator → content [1, 768, T] + f0 + energy
          → kNN-VC top-4 cosine retrieval → content_replaced [1, 768, T]
          → TinyVC DDSP decoder → output_audio (24 kHz mono)

    Offline path (NEW in v2, used by :meth:`register_voice`):
      reference_audio (any sr, mono)
          → resample to 22.05 kHz + truncate to 30 s + peak-normalize to -3 dBFS
          → librosa.linear-spec (n_fft=1024, hop=256, win=1024) → [513, T]
          → transpose+batch to [1, T, 513] (time-first, as the ONNX expects)
          → OpenVoice ReferenceEncoder ONNX → 256-d speaker embedding
          → save FP16 to models/voices_v2.safetensors (key: "voice_<id>")
    """

    # Default ONNX model filename for the OpenVoice ReferenceEncoder.
    # INT8 path is optional (P2-3 quantization output); we fall back to FP32.
    REF_ENCODER_FP32 = "openvoice_ref_encoder.onnx"
    REF_ENCODER_INT8 = "openvoice_ref_encoder.int8.onnx"
    VOICES_V2_FILE = "voices_v2.safetensors"

    def __init__(
        self,
        models_dir: str | Path = "models",
        device: str = "cpu",
        top_k: int = 4,
        alpha: float = 0.0,
        use_int8: bool = False,
        load_ref_encoder: bool = True,
    ):
        """Create a v2 hybrid inference wrapper.

        Parameters
        ----------
        models_dir : str | Path
            Directory containing encoder.pt, decoder.pt, voices.pt
            (for the runtime TinyVC path) AND openvoice_ref_encoder.onnx
            (for the offline RefEncoder path).
        device, top_k, alpha : inherited from V1Infer.
        use_int8 : bool
            If True, prefer ``openvoice_ref_encoder.int8.onnx`` (P2-3
            INT8 quantized). Falls back to FP32 if the INT8 file is missing.
        load_ref_encoder : bool
            If False, skip loading the OpenVoice RefEncoder ONNX. Useful
            for runtime-only invocations where registration is not needed
            (saves ~3 MB RSS).
        """
        # Initialize the v1 baseline (loads TinyVC encoder/decoder/voices.pt).
        super().__init__(models_dir=models_dir, device=device, top_k=top_k, alpha=alpha)

        models_dir = Path(models_dir)
        self._models_dir = models_dir
        self._use_int8 = use_int8

        # Lazily load OpenVoice RefEncoder ONNX. Skip if the caller only wants
        # the runtime path (saves memory).
        self._ref_encoder: ort.InferenceSession | None = None
        if load_ref_encoder:
            self._ref_encoder = self._load_ref_encoder_session(models_dir, use_int8)

        # Load any pre-computed OpenVoice 256-d speaker embeddings from
        # voices_v2.safetensors (created by register_voice or by the
        # bulk-registration loop in tests). Empty dict if the file does not
        # exist (e.g., first run before any voice has been registered).
        self.ov_embeddings: dict[int, np.ndarray] = self._load_openvoice_embeddings(models_dir)

    # ------------------------------------------------------------------
    # OpenVoice RefEncoder ONNX session management
    # ------------------------------------------------------------------
    @classmethod
    def _load_ref_encoder_session(
        cls, models_dir: Path, use_int8: bool
    ) -> ort.InferenceSession:
        """Load the OpenVoice ReferenceEncoder ONNX (with INT8 fallback)."""
        int8_path = models_dir / cls.REF_ENCODER_INT8
        fp32_path = models_dir / cls.REF_ENCODER_FP32
        if use_int8 and int8_path.exists():
            path = int8_path
        elif fp32_path.exists():
            path = fp32_path
        elif int8_path.exists():
            # FP32 not requested but FP32 missing — use INT8 as fallback.
            path = int8_path
        else:
            raise FileNotFoundError(
                f"OpenVoice RefEncoder ONNX not found in {models_dir} "
                f"(looked for {cls.REF_ENCODER_FP32} and {cls.REF_ENCODER_INT8}). "
                f"Run scripts/export_openvoice_onnx.py to produce it."
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def _load_openvoice_embeddings(cls, models_dir: Path) -> dict[int, np.ndarray]:
        """Load pre-computed OpenVoice 256-d speaker embeddings.

        File format: safetensors with one tensor per voice, key "voice_<id>"
        shape [256]. Stored as FP16 to halve storage (256 × 2 = 512 B per voice).

        Returns an empty dict if the file does not exist (first run).
        """
        path = models_dir / cls.VOICES_V2_FILE
        if not path.exists():
            return {}
        try:
            from safetensors.torch import load_file
        except ImportError as e:  # pragma: no cover — safetensors is in requirements
            raise RuntimeError(
                "safetensors is required to load voices_v2.safetensors; "
                "install with `pip install safetensors`."
            ) from e
        out: dict[int, np.ndarray] = {}
        for key, tensor in load_file(str(path)).items():
            # key format: "voice_<id>"
            try:
                vid = int(key.rsplit("_", 1)[-1])
            except ValueError:
                continue
            # safetensors returns a torch.Tensor — convert to float32 numpy.
            # The file stores FP16; .to(float32).numpy() upcasts.
            try:
                import torch as _torch  # local import to avoid module-level dep
                if hasattr(tensor, "to") and isinstance(tensor, _torch.Tensor):
                    arr = tensor.to(_torch.float32).numpy()
                else:
                    arr = np.asarray(tensor)
            except ImportError:  # pragma: no cover — torch is a hard dep
                arr = np.asarray(tensor)
            out[vid] = arr.astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # OpenVoice RefEncoder: linear-spec computation + ONNX run
    # ------------------------------------------------------------------
    @staticmethod
    def compute_openvoice_linear_spec(
        wav_22k: np.ndarray,
        n_fft: int = OPENVOICE_N_FFT,
        hop: int = OPENVOICE_HOP,
        win: int = OPENVOICE_WIN,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Compute the LINEAR spectrogram the OpenVoice RefEncoder expects.

        The OpenVoice v2 ReferenceEncoder takes the magnitude linear-spec
        (NOT mel), shape [B, T, 513] (time-first), dtype float32.

        Parameters
        ----------
        wav_22k : np.ndarray [L] at 22.05 kHz, float32 in [-1, 1] (or any peak)

        Returns
        -------
        np.ndarray [1, T, 513] float32 — ready to feed to the ONNX session.
        """
        if not _HAS_LIBROSA:
            raise RuntimeError(
                "librosa is required to compute the OpenVoice linear spec; "
                "install librosa or pre-compute the spec offline."
            )
        wav = np.asarray(wav_22k, dtype=np.float32)
        linear_spec = librosa.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win)
        mag = np.abs(linear_spec).astype(np.float32)  # [513, T]
        # ONNX expects time-first: [B, T, 513]
        return mag.T[None, ...]

    def encode_openvoice_embedding(self, wav_22k: np.ndarray) -> np.ndarray[Any, np.dtype[Any]]:
        """Run the OpenVoice ReferenceEncoder ONNX on a 22.05 kHz wav.

        Used OFFLINE during voice registration (not at runtime).

        Parameters
        ----------
        wav_22k : np.ndarray [L_samples] float32 at 22.05 kHz.

        Returns
        -------
        np.ndarray [256] float32 — the 256-d speaker embedding.
        """
        if self._ref_encoder is None:
            raise RuntimeError(
                "RefEncoder ONNX not loaded (load_ref_encoder=False). "
                "Re-instantiate V2Infer with load_ref_encoder=True to register voices."
            )
        spec_in = self.compute_openvoice_linear_spec(wav_22k)
        input_name = self._ref_encoder.get_inputs()[0].name  # "linear_spec"
        out = self._ref_encoder.run(None, {input_name: spec_in})
        # Output name: "speaker_embedding", shape [B, 256]
        emb = out[0]
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb[0]
        return np.ascontiguousarray(emb, dtype=np.float32)

    # ------------------------------------------------------------------
    # Voice registration (offline)
    # ------------------------------------------------------------------
    def register_voice(
        self, wav_path: str | Path, voice_id: int, persist: bool = True
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """OFFLINE: encode a 30 s voice reference wav to a 256-d OpenVoice
        embedding and persist it to ``models/voices_v2.safetensors``.

        Steps:
          1. Load wav (any sr, mono or stereo).
          2. Resample to 22.05 kHz (OpenVoice v2 native rate).
          3. Truncate to 30 s (OpenVoice's default reference length).
          4. Peak-normalize to -3 dBFS (matches v1 convention).
          5. Compute linear spec (n_fft=1024, hop=256, win=1024) → [1, T, 513].
          6. Run OpenVoice ReferenceEncoder ONNX → [256].
          7. Save FP16 to models/voices_v2.safetensors (key: "voice_<id>").

        Parameters
        ----------
        wav_path : str | Path — reference wav (any sr, mono or stereo).
        voice_id : int — index 0..N-1 (must match the v1 kNN voice_id).
        persist : bool — if True (default), write the embedding to disk.

        Returns
        -------
        np.ndarray [256] float32 — the embedding (also cached in self.ov_embeddings).
        """
        if not _HAS_LIBROSA:
            raise RuntimeError("librosa required for register_voice (resampling).")
        wav, sr = sf.read(str(wav_path), always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)  # stereo → mono
        wav = wav.astype(np.float32)
        if sr != OPENVOICE_SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=OPENVOICE_SAMPLE_RATE)
        # Truncate to 30 s (OpenVoice default reference length)
        max_samples = int(OPENVOICE_SAMPLE_RATE * OPENVOICE_REF_DURATION_S)
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        # Peak-normalize to -3 dBFS
        peak = float(np.max(np.abs(wav))) + 1e-8
        wav = wav * (10 ** (DEFAULT_NORM_DB / 20.0) / peak)
        # Encode
        emb = self.encode_openvoice_embedding(wav)
        # Cache in-memory
        self.ov_embeddings[voice_id] = emb
        # Persist
        if persist:
            self._save_embeddings_to_safetensors()
            print(
                f"[infer_v2] registered voice {voice_id} from {wav_path}: "
                f"256-d embedding (norm={float(np.linalg.norm(emb)):.3f}) "
                f"saved to {self._models_dir / self.VOICES_V2_FILE}"
            )
        return emb

    def _save_embeddings_to_safetensors(self) -> None:
        """Persist self.ov_embeddings to models/voices_v2.safetensors as FP16."""
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "safetensors + torch required to save voices_v2.safetensors."
            ) from e
        self._models_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._models_dir / self.VOICES_V2_FILE
        tensors = {
            f"voice_{vid}": torch.from_numpy(emb).to(torch.float16)
            for vid, emb in sorted(self.ov_embeddings.items())
        }
        if not tensors:
            # If the file exists and the registry is empty, remove it.
            if out_path.exists():
                out_path.unlink()
            return
        save_file(tensors, str(out_path))

    def get_openvoice_embedding(self, voice_id: int) -> np.ndarray | None:
        """Lookup the OpenVoice 256-d embedding for `voice_id`, or None.

        This is a runtime convenience accessor; v2 does NOT actually use the
        OpenVoice embedding at runtime (the v1 kNN path is still the active
        converter). It will be wired up in v2.5 when the projection layers
        are trained.
        """
        return self.ov_embeddings.get(voice_id)


# ---------------------------------------------------------------------------
# Module-level singleton for benchmark.py / realtime_infer.py compatibility.
# ---------------------------------------------------------------------------
_default_infer: V2Infer | None = None


def process_audio(wav: np.ndarray, sr: int, voice_id: int = 0) -> np.ndarray[Any, np.dtype[Any]]:
    """Module-level shortcut using a cached singleton V2Infer.

    Used by `scripts/benchmark.py` and `scripts/realtime_infer.py` so they
    can call ``vc_realtime.infer_v2.process_audio(wav, sr, voice_id)``
    without instantiating the heavy encoder/decoder per call.

    Note: the singleton is created with `load_ref_encoder=False` because
    the runtime path does NOT need the OpenVoice RefEncoder — that ONNX
    is only used during offline `register_voice` calls. This saves ~3 MB
    RSS and ~150 ms startup time.
    """
    global _default_infer
    if _default_infer is None:
        _default_infer = V2Infer(load_ref_encoder=False)
    return _default_infer.process_audio(wav, sr, voice_id=voice_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vc_realtime.infer_v2",
        description=(
            "v2 hybrid VC: TinyVC encoder + kNN-VC + DDSP (runtime, "
            "identical to v1) + OpenVoice ReferenceEncoder (offline 256-d "
            "speaker embedding storage)."
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
        "openvoice_ref_encoder.onnx.",
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
        help="OFFLINE mode: register a voice reference wav → 256-d OpenVoice "
        "embedding saved to models/voices_v2.safetensors. No inference run.",
    )
    p.add_argument(
        "--use-int8", action="store_true",
        help="Prefer the INT8-quantized OpenVoice RefEncoder ONNX if available.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.register_voice:
        # OFFLINE: encode the reference wav → 256-d embedding → save to disk.
        infer = V2Infer(
            models_dir=args.models_dir,
            device=args.device,
            top_k=args.top_k,
            alpha=args.alpha,
            use_int8=args.use_int8,
            load_ref_encoder=True,
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

    # Runtime: do NOT load the OpenVoice RefEncoder ONNX (saves RSS).
    infer = V2Infer(
        models_dir=args.models_dir,
        device=args.device,
        top_k=args.top_k,
        alpha=args.alpha,
        use_int8=args.use_int8,
        load_ref_encoder=False,
    )
    out = infer.process_audio(
        wav, sr, voice_id=args.voice_id,
        pitch_shift_semitones=args.pitch_shift,
    )

    out_clip = np.clip(out, -1.0, 1.0)
    sf.write(args.output, (out_clip * 32767).astype(np.int16), SAMPLE_RATE)

    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    dur = len(out) / SAMPLE_RATE
    print(f"[infer_v2] source sr={sr}, voice_id={args.voice_id}, "
          f"pitch_shift={args.pitch_shift}")
    print(f"[infer_v2] wrote {args.output}: {SAMPLE_RATE}Hz mono, "
          f"{dur:.3f}s, output RMS={rms:.4f}")
    print("[infer_v2] (runtime path identical to v1; OpenVoice RefEncoder "
          "loaded only at registration time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
