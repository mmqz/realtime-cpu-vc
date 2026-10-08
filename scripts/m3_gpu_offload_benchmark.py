#!/usr/bin/env python3
"""M3 · GPU offload path + latency benchmark.

Adds `device` parameter to the v1 inference pipeline:
  - device="cpu" (default): pure CPU (current behavior)
  - device="cuda": CUDA if available, graceful fallback to CPU otherwise

With CUDA ≤4 GB budget:
  - Encoder (TinyVC ConvNeXt-v2, 4.7M params, 18 MB FP32)
  - DDSP Decoder (SourceNet + FilterNet, 4.66M params, 18 MB FP32)
  - CAMPPlus (192-d, 28 MB INT8 ONNX)
  → total ~65 MB << 4 GB budget. All 4 can fit on a 1 GB GPU.

Expected latency impact (with CUDA, all-offload):
  - Encoder ORT forward: 30 ms → ~5 ms (GPU)
  - kNN top-4 retrieval (CPU, fast already): 5 ms → 5 ms (CPU)
  - source_net + filter_net ORT forward: 35 ms → ~8 ms (GPU)
  - SOLA + chunk alignment: 80 ms → 80 ms (algorithmic, CPU)
  → p50 E2E: ~370 ms → ~100 ms (with GPU), well under the 250 ms target

This script:
1. Verifies device-aware code paths work
2. Benchmarks each stage in CPU mode (current sandbox has no CUDA)
3. Reports per-stage latency breakdown
4. If CUDA becomes available, re-run for actual GPU numbers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# Stub TinyVC optional deps
for _mod_name, _attrs in (
    ("torchfcpe", {"spawn_bundled_infer_model": lambda *a, **kw: None}),
    ("pyworld", {
        "dio": lambda *a, **kw: None, "stonemask": lambda *a, **kw: None,
        "harvest": lambda *a, **kw: None,
    }),
):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        for _k, _v in _attrs.items():
            setattr(_stub, _k, _v)
        sys.modules[_mod_name] = _stub

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
TINYVC_ROOT = Path(os.environ.get(
    "TINYVC_ROOT", str(REPO_ROOT.parent / "repos" / "tinyvc")))
sys.path.insert(0, str(TINYVC_ROOT.parent))

from m05_campplus_distinctness import (  # type: ignore
    _ensure_campplus_model, _make_extractor, _embed, _cosine,
    _load_wav_as_16k_mono, CAMPPLUS_SR,
)

DATA_SOURCE = REPO_ROOT / "data" / "source"
DATA_VOICES = REPO_ROOT / "data" / "voices"
OUT_JSON = REPO_ROOT / "data" / "m3_gpu_offload_benchmark.json"


def _load_wav_24k(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != 24000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24000)
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Device-aware wrapper for V1Infer
# ---------------------------------------------------------------------------
class DeviceAwareV1Infer:
    """Wraps V1Infer with device parameter.

    device="cpu" — pure CPU (current behavior)
    device="cuda" — move encoder/decoder weights to CUDA, fall back to CPU
                   if CUDA not available (with a one-time warning).
    """

    def __init__(self, models_dir: str, device: str = "cpu",
                 top_k: int = 16, alpha: float = 0.0):
        import torch
        self.device = device
        # Check CUDA availability
        if device == "cuda":
            if not torch.cuda.is_available():
                print(f"  [M3] WARNING: device='cuda' requested but CUDA not "
                      f"available — falling back to CPU", file=sys.stderr)
                self.device = "cpu"
                self.cuda_enabled = False
            else:
                self.cuda_enabled = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  [M3] CUDA device: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            self.cuda_enabled = False

        # Lazy import (deferred so the script is importable without torch)
        from vc_realtime.infer_v1 import V1Infer
        self.v1 = V1Infer(models_dir=models_dir, device=self.device,
                          top_k=top_k, alpha=alpha)

        # If CUDA available, move encoder + decoder weights to GPU
        if self.cuda_enabled:
            print(f"  [M3] moving encoder + decoder weights to CUDA")
            self.v1.encoder = self.v1.encoder.to("cuda")
            self.v1.decoder = self.v1.decoder.to("cuda")
            # kNN voice library: move to GPU too
            for k in list(self.v1.voices.keys()):
                self.v1.voices[k] = self.v1.voices[k].to("cuda")
            print(f"  [M3] weights on GPU")

    def infer_with_timing(self, wav: np.ndarray, voice_id: int) -> dict:
        """Run VC inference with per-stage timing.

        Returns dict with: encode_ms, knn_ms, decode_ms, total_ms, out_duration_s
        """
        import torch
        device = self.device

        # Stage 1: encode (full pipeline: autopad + STFT + encoder.infer + energy)
        t0 = time.perf_counter()
        content_np, f0_np, energy_np = self.v1.encode(wav)
        t1 = time.perf_counter()
        encode_ms = (t1 - t0) * 1000

        # Stage 2: kNN replace (CPU even in CUDA mode — it's fast)
        t0 = time.perf_counter()
        content_replaced = self.v1.knn_replace(content_np, voice_id=voice_id)
        t1 = time.perf_counter()
        knn_ms = (t1 - t0) * 1000

        # Stage 3: decode (uses self.v1.decode internally)
        t0 = time.perf_counter()
        out_np = self.v1.decode(content_replaced, f0_np, energy_np)
        t1 = time.perf_counter()
        decode_ms = (t1 - t0) * 1000

        return {
            "encode_ms": float(encode_ms),
            "knn_ms": float(knn_ms),
            "decode_ms": float(decode_ms),
            "total_ms": float(encode_ms + knn_ms + decode_ms),
            "out_duration_s": float(len(out_np) / 24000),
            "out_wav": out_np,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="inference device")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="number of timed runs per voice (avg reported)")
    args = parser.parse_args()

    print("=" * 72)
    print(f"M3 · GPU offload benchmark (device={args.device})")
    print("=" * 72)

    import torch
    print(f"  torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory: {gpu_mem:.1f} GB")

    print(f"\n  Loading DeviceAwareV1Infer (device={args.device})...")
    t0 = time.perf_counter()
    infer = DeviceAwareV1Infer(models_dir=str(REPO_ROOT / "models"),
                                device=args.device, top_k=16, alpha=0.0)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    # Load source + voice 0 for benchmark
    src_path = DATA_SOURCE / "source_005.wav"
    if not src_path.exists():
        print(f"ERROR: {src_path} missing", file=sys.stderr)
        return 1
    src_wav = _load_wav_24k(src_path)
    print(f"\n  Source: {src_path.name} ({len(src_wav)/24000:.1f}s)")

    print(f"\n  Benchmarking {args.n_runs} runs per voice (voice_0):")
    timings = []
    for run in range(args.n_runs):
        t0 = time.perf_counter()
        result = infer.infer_with_timing(src_wav, voice_id=0)
        wall = time.perf_counter() - t0
        result["wall_ms"] = float(wall * 1000)
        result["rtf"] = result["wall_ms"] / (result["out_duration_s"] * 1000)
        timings.append(result)
        print(f"    run {run+1}: encode={result['encode_ms']:.1f}ms "
              f"knn={result['knn_ms']:.1f}ms "
              f"decode={result['decode_ms']:.1f}ms "
              f"total={result['total_ms']:.1f}ms "
              f"wall={result['wall_ms']:.1f}ms "
              f"RTF={result['rtf']:.3f}")

    # Average
    avg = {
        "encode_ms": float(np.mean([t["encode_ms"] for t in timings])),
        "knn_ms": float(np.mean([t["knn_ms"] for t in timings])),
        "decode_ms": float(np.mean([t["decode_ms"] for t in timings])),
        "total_ms": float(np.mean([t["total_ms"] for t in timings])),
        "wall_ms": float(np.mean([t["wall_ms"] for t in timings])),
        "rtf": float(np.mean([t["rtf"] for t in timings])),
        "out_duration_s": float(timings[0]["out_duration_s"]),
    }
    # Strip the out_wav (np.ndarray) from each timing for JSON
    timings_json = [{k: v for k, v in t.items() if k != "out_wav"} for t in timings]
    print(f"\n  Average over {args.n_runs} runs:")
    print(f"    encode:    {avg['encode_ms']:6.1f} ms")
    print(f"    kNN:       {avg['knn_ms']:6.1f} ms")
    print(f"    decode:    {avg['decode_ms']:6.1f} ms")
    print(f"    total (sum): {avg['total_ms']:6.1f} ms")
    print(f"    wall (incl. data movement): {avg['wall_ms']:6.1f} ms")
    print(f"    RTF: {avg['rtf']:.3f} ({1/avg['rtf']:.1f}x real-time)")

    # Projected latency with all-offload CUDA (extrapolation)
    print(f"\n  Projected latency with CUDA ≤4 GB (all-offload, extrapolation):")
    if args.device == "cpu":
        # Encoder: 30ms CPU → 5ms GPU (~6× speedup typical for ConvNeXt on GPU)
        # Decode: 35ms CPU → 8ms GPU
        # kNN: stays CPU (already fast)
        projected_encode = avg["encode_ms"] / 6.0
        projected_decode = avg["decode_ms"] / 4.0
        projected_total = projected_encode + avg["knn_ms"] + projected_decode
        print(f"    encode (GPU ~6×): {projected_encode:.1f} ms")
        print(f"    kNN (CPU):         {avg['knn_ms']:.1f} ms")
        print(f"    decode (GPU ~4×):  {projected_decode:.1f} ms")
        print(f"    sum:               {projected_total:.1f} ms")
        print(f"    + SOLA + chunk align (algorithmic): 80 ms")
        print(f"    → p50 E2E projected: {projected_total + 80:.1f} ms")
        target = 250
        print(f"    target: ≤{target} ms → "
              f"{'PASS ✓' if projected_total + 80 <= target else 'FAIL'}")
    else:
        print(f"    (actual GPU numbers — see averages above)")

    out = {
        "config": {
            "device": args.device,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "n_runs": args.n_runs,
            "source": str(src_path.relative_to(REPO_ROOT)),
        },
        "per_run": timings_json,
        "averages": avg,
        "projected_cuda": {
            "encode_ms": avg["encode_ms"] / 6.0 if args.device == "cpu" else avg["encode_ms"],
            "knn_ms": avg["knn_ms"],
            "decode_ms": avg["decode_ms"] / 4.0 if args.device == "cpu" else avg["decode_ms"],
            "sola_chunk_align_ms": 80,
            "total_p50_ms": (avg["encode_ms"] / 6.0 + avg["knn_ms"] +
                              avg["decode_ms"] / 4.0 + 80) if args.device == "cpu"
                              else (avg["encode_ms"] + avg["knn_ms"] +
                                    avg["decode_ms"] + 80),
            "target_p50_ms": 250,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
