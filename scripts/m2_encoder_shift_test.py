#!/usr/bin/env python3
"""M2 · Encoder distribution shift test (offline vs streaming chunked).

The M2 plan calls for re-exporting encoder.onnx with causal (left-only)
Conv1d padding instead of ConvNeXt-v2's default centered padding. But
that requires modifying upstream TinyVC source (encoder.py ConvNeXt block)
and re-exporting. Before doing that, we need to know: **how much does
the encoder output distribution shift when we go from full-sequence
inference to chunked (80 ms = 4 frames @ 50 Hz) inference?**

This is the M2 acceptance criterion:
  "shift ≤ 5% → safe to proceed with causal re-export + 1-chunk streaming.
   shift > 5% → must retrain VoicePack on streaming-mode encoder outputs."

Why this matters
----------------
The v1 kNN index (voices.pt) and any future VoicePack weights are built
from offline (full-sequence) encoder outputs. If the streaming encoder
output distribution differs significantly (>5%), the kNN retrieval /
VoicePack conditioning becomes OOD, silently dropping VC_effect by
0.05-0.10. This bug would not show up in any test that uses offline
encoding.

Test method
-----------
1. Load 5 reference voice wavs (30 s each) + 5 source clips (5 s each).
2. For each wav:
   a. Offline: encode the full wav → content [1, 768, T_full]
   b. Streaming: encode in 80 ms chunks (4 frames @ 50 Hz each), with
      reflect-padding as TinyVC's encoder expects, accumulate per-chunk
      content features, concatenate → content_stream [1, 768, T_full]
3. Compare per-channel mean and std between offline and streaming:
   - shift_mean = mean(|offline_mean - stream_mean|) / |offline_mean|
   - shift_std  = mean(|offline_std  - stream_std|)  / |offline_std|
   - shift_overall = max(shift_mean, shift_std)
4. PASS if shift_overall ≤ 5%, FAIL otherwise.

Note: the streaming mode uses the SAME (centered) encoder — we're not
replacing it with a causal one yet. The shift comes from the chunked
input's edge effects at chunk boundaries (since each chunk gets its
own centered padding instead of seeing the full sequence).
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

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

TINYVC_ROOT = Path(os.environ.get(
    "TINYVC_ROOT", str(REPO_ROOT.parent / "repos" / "tinyvc")))
if not TINYVC_ROOT.exists():
    raise SystemExit(f"TinyVC repo not found at {TINYVC_ROOT}")
sys.path.insert(0, str(TINYVC_ROOT.parent))

DATA_VOICES = REPO_ROOT / "data" / "voices"
DATA_SOURCE = REPO_ROOT / "data" / "source"
OUT_JSON = REPO_ROOT / "data" / "m2_encoder_shift.json"

# 80 ms chunk = 4 frames @ 50 Hz = 1920 samples @ 24 kHz (M2 target spec)
CHUNK_FRAMES = 4
HOP_SIZE = 480  # TinyVC default
N_FFT = 1920    # TinyVC default
CHUNK_SAMPLES = CHUNK_FRAMES * HOP_SIZE  # 1920 samples

# Larger chunks for sanity-check (1 s = 50 frames = 24000 samples)
LARGE_CHUNK_FRAMES = 50
LARGE_CHUNK_SAMPLES = LARGE_CHUNK_FRAMES * HOP_SIZE  # 24000 samples


def _load_wav_24k(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != 24000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24000)
    return wav.astype(np.float32)


def _encode_offline(wav: np.ndarray, v1_infer) -> np.ndarray:
    """Encode the full wav → content [1, 768, T]."""
    content, _, _ = v1_infer.encode(wav)
    return content


def _encode_streaming(wav: np.ndarray, v1_infer,
                       chunk_samples: int = CHUNK_SAMPLES) -> np.ndarray:
    """Encode the wav in chunks → concatenated content [1, 768, T'].

    Each chunk = 4 frames @ 50 Hz = 1920 samples = 80 ms.
    TinyVC's encoder uses centered padding (n_fft//2 on each side), so
    each chunk gets its own padding (the encoder doesn't see beyond the
    chunk boundary). This is the worst-case shift scenario for streaming.
    """
    # Pad wav to multiple of chunk_samples
    n = len(wav)
    pad_n = (-n) % chunk_samples
    if pad_n:
        wav = np.concatenate([wav, np.zeros(pad_n, dtype=np.float32)])
    n_chunks = len(wav) // chunk_samples

    # Pre-process once (peak-normalize)
    wav_pre = v1_infer._preprocess(wav, 24000)

    contents = []
    for i in range(n_chunks):
        chunk = wav_pre[i * chunk_samples : (i + 1) * chunk_samples]
        # Each chunk encodes to ~4 frames (or fewer if at edge after preprocess)
        try:
            content, _, _ = v1_infer.encode(chunk)
            contents.append(content)
        except Exception as e:
            print(f"  [chunk {i}] encode failed: {e}", file=sys.stderr)
            continue
    if not contents:
        return np.zeros((1, 768, 0), dtype=np.float32)
    # Concatenate along time axis
    return np.concatenate(contents, axis=2)


def _compute_shift(offline: np.ndarray, streaming: np.ndarray) -> dict:
    """Compute per-channel mean/std shift between offline and streaming.

    Both arrays: [1, 768, T]. Truncate to min T for fair comparison.
    Returns shift_mean, shift_std, shift_overall (all in %).
    """
    T = min(offline.shape[2], streaming.shape[2])
    off = offline[..., :T].squeeze(0)  # [768, T]
    strm = streaming[..., :T].squeeze(0)  # [768, T]

    off_mean = off.mean(axis=-1)  # [768]
    strm_mean = strm.mean(axis=-1)
    off_std = off.std(axis=-1) + 1e-8
    strm_std = strm.std(axis=-1) + 1e-8

    # Relative shift per channel
    rel_mean_shift = np.abs(off_mean - strm_mean) / (np.abs(off_mean) + 1e-8)
    rel_std_shift = np.abs(off_std - strm_std) / off_std

    return {
        "n_frames_compared": int(T),
        "mean_rel_shift_per_channel_mean": float(rel_mean_shift.mean()),
        "mean_rel_shift_per_channel_max": float(rel_mean_shift.max()),
        "mean_rel_shift_per_channel_p95": float(np.percentile(rel_mean_shift, 95)),
        "std_rel_shift_per_channel_mean": float(rel_std_shift.mean()),
        "std_rel_shift_per_channel_max": float(rel_std_shift.max()),
        "std_rel_shift_per_channel_p95": float(np.percentile(rel_std_shift, 95)),
        "shift_overall_mean": float(max(rel_mean_shift.mean(),
                                          rel_std_shift.mean())),
        "shift_overall_max": float(max(rel_mean_shift.max(),
                                         rel_std_shift.max())),
    }


def main() -> int:
    print("=" * 70)
    print("M2 · Encoder distribution shift test (offline vs streaming chunked)")
    print("=" * 70)
    print(f"  chunk size: {CHUNK_FRAMES} frames @ 50 Hz = {CHUNK_SAMPLES} samples "
          f"= {CHUNK_SAMPLES/24000*1000:.0f} ms")
    print(f"  sanity chunk: {LARGE_CHUNK_FRAMES} frames @ 50 Hz = "
          f"{LARGE_CHUNK_SAMPLES} samples = {LARGE_CHUNK_SAMPLES/24000*1000:.0f} ms")
    print(f"  acceptance: shift_overall_mean ≤ 5% → PASS")

    print("\n  Loading V1Infer (encoder.pt + decoder.pt)...")
    from vc_realtime.infer_v1 import V1Infer
    v1_infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                       device="cpu", top_k=4, alpha=0.0)
    print("  Loaded")

    # Test on all 5 voices + 5 sources = 10 wavs total
    test_wavs = []
    for i in range(5):
        test_wavs.append((f"voice_{i}", DATA_VOICES / f"voice_{i}.wav"))
    for i in range(5):
        src_path = DATA_SOURCE / f"source_{i+1:03d}.wav"
        if src_path.exists():
            test_wavs.append((f"source_{i+1:03d}", src_path))

    print(f"\n  Testing {len(test_wavs)} wavs (both 80 ms and 1000 ms chunks):")
    results = []
    pass_count_small = 0
    pass_count_large = 0
    for name, path in test_wavs:
        if not path.exists():
            print(f"    [skip] {name}: {path} missing")
            continue
        wav = _load_wav_24k(path)
        t0 = time.perf_counter()
        offline = _encode_offline(wav, v1_infer)
        offline_t = time.perf_counter() - t0

        # Small chunk (80 ms = M2 spec)
        t0 = time.perf_counter()
        streaming_small = _encode_streaming(wav, v1_infer, CHUNK_SAMPLES)
        stream_small_t = time.perf_counter() - t0
        shift_small = _compute_shift(offline, streaming_small)
        pass_small = shift_small["shift_overall_mean"] <= 0.05
        if pass_small:
            pass_count_small += 1

        # Large chunk (1 s = sanity check)
        t0 = time.perf_counter()
        streaming_large = _encode_streaming(wav, v1_infer, LARGE_CHUNK_SAMPLES)
        stream_large_t = time.perf_counter() - t0
        shift_large = _compute_shift(offline, streaming_large)
        pass_large = shift_large["shift_overall_mean"] <= 0.05
        if pass_large:
            pass_count_large += 1

        results.append({
            "name": name,
            "path": str(path.relative_to(REPO_ROOT)),
            "duration_s": len(wav) / 24000,
            "offline_t_s": offline_t,
            "stream_small_t_s": stream_small_t,
            "stream_large_t_s": stream_large_t,
            "offline_shape": list(offline.shape),
            "stream_small_shape": list(streaming_small.shape),
            "stream_large_shape": list(streaming_large.shape),
            "shift_small_80ms": shift_small,
            "shift_large_1000ms": shift_large,
            "pass_small": bool(pass_small),
            "pass_large": bool(pass_large),
        })
        ss = "PASS" if pass_small else "FAIL"
        sl = "PASS" if pass_large else "FAIL"
        print(f"    {name:12s} ({len(wav)/24000:.1f}s): "
              f"80ms shift {shift_small['shift_overall_mean']*100:6.2f}% {ss} | "
              f"1s shift {shift_large['shift_overall_mean']*100:6.2f}% {sl}")

    overall_small_pass = pass_count_small == len(results)
    overall_large_pass = pass_count_large == len(results)
    print(f"\n  Overall: 80 ms chunks {pass_count_small}/{len(results)} pass, "
          f"1 s chunks {pass_count_large}/{len(results)} pass")
    print(f"  M2 acceptance (80 ms chunks ≤ 5% shift): "
          f"{'PASS ✓' if overall_small_pass else 'FAIL ✗'}")
    print(f"  Sanity (1 s chunks ≤ 5% shift): "
          f"{'PASS ✓' if overall_large_pass else 'FAIL ✗'}")

    out = {
        "config": {
            "small_chunk_frames": CHUNK_FRAMES,
            "small_chunk_samples": CHUNK_SAMPLES,
            "small_chunk_ms": CHUNK_SAMPLES / 24000 * 1000,
            "large_chunk_frames": LARGE_CHUNK_FRAMES,
            "large_chunk_samples": LARGE_CHUNK_SAMPLES,
            "large_chunk_ms": LARGE_CHUNK_SAMPLES / 24000 * 1000,
            "n_fft": N_FFT,
            "hop_size": HOP_SIZE,
            "acceptance_threshold_pct": 5.0,
        },
        "results": results,
        "summary": {
            "n_wavs": len(results),
            "n_pass_small_80ms": pass_count_small,
            "n_pass_large_1000ms": pass_count_large,
            "overall_small_pass": bool(overall_small_pass),
            "overall_large_pass": bool(overall_large_pass),
            "mean_shift_small_pct": float(np.mean(
                [r["shift_small_80ms"]["shift_overall_mean"] for r in results]) * 100),
            "mean_shift_large_pct": float(np.mean(
                [r["shift_large_1000ms"]["shift_overall_mean"] for r in results]) * 100),
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")

    if not overall_small_pass:
        print("\n  ⚠️  80 ms chunk shift > 5% — VoicePack trained on offline outputs")
        print("     will be OOD for streaming inference. Two paths:")
        print("     A) Use proper state-carrying StreamingConv1d (Rust vc-native")
        print("        already implements this — but Python V1Infer doesn't use it)")
        print("     B) Retrain VoicePack on streaming-mode encoder outputs (the")
        print("        same encode_streaming() used here, as a data augmentation)")
    else:
        print("\n  ✓ Shift ≤ 5% — safe to proceed with causal re-export + 1-chunk streaming.")
    return 0 if overall_small_pass else 2


if __name__ == "__main__":
    sys.exit(main())
