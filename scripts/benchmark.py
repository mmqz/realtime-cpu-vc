#!/usr/bin/env python3
"""Benchmark harness for the real-time CPU voice-conversion system.

Measures three metrics (no stubs):

  1. **RTF** (Real-Time Factor) = processing_time / audio_duration
     Actually times `infer_fn(wav, sr, voice_id)` using `time.perf_counter()`.

  2. **RSS** (Resident Set Size in MB) via `psutil.Process().memory_info().rss`.

  3. **Speaker similarity** = cosine(ref_embedding, out_embedding) where the
     192-d embedding is produced by `sherpa_onnx.SpeakerEmbeddingExtractor`
     using the 3D-Speaker CAMPPlus ONNX model
     (3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx, 192-d).

Usage:

    python3 scripts/benchmark.py \\
        --infer-script vc_realtime.infer_v1 \\
        --source data/source/source_001.wav \\
        --voice-id 0 \\
        --ref data/voices/voice_0.wav \\
        --output /tmp/benchmark_out.wav
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import psutil
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants — CAMPPlus 192-d speaker embedding model
# ---------------------------------------------------------------------------
MODEL_DIR: Path = Path("models")
CAMPPLUS_ONNX_NAME: str = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
CAMPPLUS_HF_REPO: str = "bitsydarel/campplus-onnx"
CAMPPLUS_HF_FILENAME: str = CAMPPLUS_ONNX_NAME
CAMPPLUS_SR: int = 16_000  # model trained at 16 kHz
CAMPPLUS_DIM: int = 192


# ---------------------------------------------------------------------------
# Metric 1: RTF
# ---------------------------------------------------------------------------
def measure_rtf(infer_fn: Callable, source_wav_path: str,
                duration_s: float = 1.0) -> Tuple[float, np.ndarray]:
    """Real-Time Factor: processing_time_s / audio_duration_s.

    Loads `source_wav_path`, truncates to `duration_s`, times a single
    invocation of `infer_fn(wav, sr, voice_id=0)`.

    Returns (rtf, out_wav) where out_wav is float32 [-1, 1].
    """
    wav, sr = sf.read(source_wav_path, always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    # Convert to float32 [-1, 1] regardless of input subtype
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32767.0
    elif wav.dtype == np.int32:
        wav = wav.astype(np.float32) / 2147483647.0
    elif wav.dtype == np.float64:
        wav = wav.astype(np.float32)
    target_samples = int(sr * duration_s)
    if len(wav) < target_samples:
        # pad with zeros if source is shorter than requested duration
        pad = np.zeros(target_samples - len(wav), dtype=wav.dtype)
        wav = np.concatenate([wav, pad])
    else:
        wav = wav[:target_samples]
    t0 = time.perf_counter()
    out = infer_fn(wav, sr)
    t1 = time.perf_counter()
    proc_s = t1 - t0
    rtf = proc_s / duration_s
    # Coerce out to float32 numpy array (defensive: some infer_fns return tuple)
    if isinstance(out, tuple):
        out = out[0]
    out = np.asarray(out, dtype=np.float32)
    if out.ndim > 1:
        out = out[:, 0]
    return rtf, out


# ---------------------------------------------------------------------------
# Metric 2: RSS
# ---------------------------------------------------------------------------
def measure_rss() -> float:
    """Peak RSS in MB via psutil. Returns current process RSS."""
    return psutil.Process().memory_info().rss / 1024.0 / 1024.0


# ---------------------------------------------------------------------------
# Metric 3: Speaker similarity
# ---------------------------------------------------------------------------
def _ensure_campplus_model() -> Path:
    """Download CAMPPlus ONNX to `models/` if missing. Returns local path."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local = MODEL_DIR / CAMPPLUS_ONNX_NAME
    if local.exists() and local.stat().st_size > 1_000_000:
        return local
    print(f"[benchmark] downloading CAMPPlus model from HF "
          f"({CAMPPLUS_HF_REPO}/{CAMPPLUS_HF_FILENAME})...", file=sys.stderr)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=CAMPPLUS_HF_REPO,
        filename=CAMPPLUS_HF_FILENAME,
        local_dir=str(MODEL_DIR),
    )
    if not local.exists():
        raise RuntimeError(f"Model download failed: {local}")
    print(f"[benchmark] model cached at {local} "
          f"({local.stat().st_size // 1024 // 1024} MB)", file=sys.stderr)
    return local


_EXTRACTOR_SINGLETON = None  # type: Optional["sherpa_onnx.SpeakerEmbeddingExtractor"]


def _get_extractor():
    """Lazily build the sherpa_onnx SpeakerEmbeddingExtractor (single instance)."""
    global _EXTRACTOR_SINGLETON
    if _EXTRACTOR_SINGLETON is not None:
        return _EXTRACTOR_SINGLETON
    import sherpa_onnx
    model_path = _ensure_campplus_model()
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path),
        num_threads=1,
        debug=False,
        provider="cpu",
    )
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    if not cfg.validate():
        raise RuntimeError("SpeakerEmbeddingExtractorConfig.validate() failed")
    if ext.dim != CAMPPLUS_DIM:
        raise RuntimeError(
            f"Expected CAMPPlus dim={CAMPPLUS_DIM}, got {ext.dim}")
    _EXTRACTOR_SINGLETON = ext
    return ext


def _extract_embedding(wav_path: str) -> np.ndarray:
    """Compute 192-d CAMPPlus embedding for `wav_path` (any sample rate)."""
    ext = _get_extractor()
    wav, sr = sf.read(wav_path, always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    # CAMPPlus trained at 16 kHz — sherpa_onnx resamples internally
    # when accept_waveform sample_rate differs. We still pre-convert to float32
    # [-1, 1] as required by accept_waveform's contract.
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32767.0
    elif wav.dtype == np.int32:
        wav = wav.astype(np.float32) / 2147483647.0
    elif wav.dtype == np.float64:
        wav = wav.astype(np.float32)
    # CAMPPlus expects at least ~3 s of audio; pad short clips with silence.
    min_samples = int(CAMPPLUS_SR * 0.5)
    if len(wav) < min_samples:
        wav = np.concatenate(
            [wav, np.zeros(min_samples - len(wav), dtype=np.float32)])
    stream = ext.create_stream()
    stream.accept_waveform(sr, wav)
    stream.input_finished()
    if not ext.is_ready(stream):
        # Some models need a final flush — accept_waveform + input_finished
        # should normally suffice, but loop just in case.
        while not ext.is_ready(stream):
            time.sleep(0.001)
    emb = ext.compute(stream)
    return np.asarray(emb, dtype=np.float32)


def measure_speaker_similarity(ref_wav_path: str,
                               out_wav_path: str) -> float:
    """Cosine similarity between ref and out embeddings (CAMPPlus 192-d).

    Returns a float in [-1, 1]; higher is more similar (1 = identical speaker).
    """
    ref = _extract_embedding(ref_wav_path)
    out = _extract_embedding(out_wav_path)
    # Cosine similarity
    ref_n = float(np.linalg.norm(ref))
    out_n = float(np.linalg.norm(out))
    if ref_n < 1e-8 or out_n < 1e-8:
        return 0.0
    return float(np.dot(ref, out) / (ref_n * out_n))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_infer_fn(infer_script: str) -> Callable:
    """Resolve a 'module' or 'module.symbol' string into a callable.

    Convention: module exposes `process_audio(wav, sr, voice_id=0) -> wav`.
    If `infer_script` contains a dot AND the prefix resolves as an attribute,
    we return that attribute; otherwise we fall back to `process_audio`.
    """
    parts = infer_script.split(".")
    # Try: import the whole string as a module, get process_audio
    try:
        mod = importlib.import_module(infer_script)
        if hasattr(mod, "process_audio"):
            return mod.process_audio
    except ImportError:
        pass
    # Try: split at last dot — module + attribute name
    if len(parts) >= 2:
        mod_name, attr = ".".join(parts[:-1]), parts[-1]
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, attr):
                fn = getattr(mod, attr)
                if callable(fn):
                    return fn
        except ImportError:
            pass
    raise ImportError(
        f"Could not resolve infer_fn from --infer-script={infer_script!r}. "
        f"Expected a Python module exposing process_audio(wav, sr, voice_id=0)."
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Benchmark voice conversion: RTF, RSS, speaker similarity.",
    )
    p.add_argument("--infer-script", required=False, default=None,
                   help="Python module exposing process_audio(wav, sr, voice_id=0). "
                        "E.g. 'vc_realtime.infer_v1'.")
    p.add_argument("--source", required=False, default="data/source/source_001.wav",
                   help="Source wav path (24 kHz mono PCM16 expected).")
    p.add_argument("--voice-id", type=int, default=0,
                   help="Voice id to pass to process_audio.")
    p.add_argument("--ref", default="data/voices/voice_0.wav",
                   help="Reference voice wav for speaker similarity.")
    p.add_argument("--output", default=None,
                   help="Write converted output wav (default /tmp/benchmark_out.wav).")
    p.add_argument("--duration", type=float, default=1.0,
                   help="Audio duration in seconds to benchmark (default 1.0).")
    args = p.parse_args()

    if args.infer_script is None:
        # Smoke test mode — verify functions are callable, print sample JSON.
        print(json.dumps({
            "smoke_test": True,
            "measure_rtf": callable(measure_rtf),
            "measure_rss": callable(measure_rss),
            "measure_speaker_similarity": callable(measure_speaker_similarity),
            "note": "Pass --infer-script module.name to run a real benchmark.",
        }, indent=2))
        return 0

    infer_fn = _load_infer_fn(args.infer_script)

    # Wrap so we can pass voice_id through
    def _wrapped(wav, sr, voice_id=args.voice_id):
        return infer_fn(wav, sr, voice_id=voice_id)

    # Pre-allocate any lazily-initialized models by running once on a small slice
    # so RTF measurement reflects steady-state inference, not first-load cost.
    # (We do NOT want RTF inflated by ONNX runtime warm-up.)
    try:
        warm_wav, warm_sr = sf.read(args.source, always_2d=False)
        if warm_wav.ndim > 1:
            warm_wav = warm_wav[:, 0]
        if warm_wav.dtype == np.int16:
            warm_wav = warm_wav.astype(np.float32) / 32767.0
        warm_wav = warm_wav[: int(warm_sr * 0.1)]  # 100 ms warmup
        _wrapped(warm_wav, warm_sr)
        print(f"[benchmark] warm-up OK", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[benchmark] warm-up failed (continuing): {e}", file=sys.stderr)

    rtf, out_wav = measure_rtf(_wrapped, args.source, duration_s=args.duration)
    rss_mb = measure_rss()
    out_path = args.output or "/tmp/benchmark_out.wav"
    sf.write(out_path, out_wav, 24000, subtype="PCM_16")

    # Speaker similarity — only attempt if ref exists.
    similarity = None
    if Path(args.ref).exists():
        try:
            similarity = measure_speaker_similarity(args.ref, out_path)
        except Exception as e:  # noqa: BLE001
            similarity = None
            print(f"[benchmark] similarity failed: {e}", file=sys.stderr)

    result = {
        "rtf": round(float(rtf), 4),
        "rss_mb": round(float(rss_mb), 2),
        "similarity": (None if similarity is None
                       else round(float(similarity), 4)),
        "source": str(args.source),
        "voice_id": int(args.voice_id),
        "infer_script": args.infer_script,
        "output": str(out_path),
        "duration_s": float(args.duration),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
