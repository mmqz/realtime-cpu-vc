#!/usr/bin/env python3
"""M2 Round 15 · Use Rust StreamingConv1dState + Rust STFT to fix the
streaming encoder distribution shift.

Round 13 showed pure-Python streaming STFT alone is insufficient because:
1. My naive impl didn't match torch.stft's center padding
2. Even with correct STFT, ConvNeXt-v2 internal convs need state-carrying

This round uses the Rust primitives from vc_python (just built and installed):
  - linear_stft_magnitude_np: Rust STFT that matches torch.stft(center=True)
  - StreamingPipeline: Rust pipeline with state-carrying conv state machine

Test plan:
1. Verify Rust STFT matches torch.stft offline (sanity check)
2. Run TinyVC encoder on Rust STFT in chunks (4 frames = 80ms)
3. Compare to offline (full audio → torch.stft → encoder)
4. Measure shift; expect ≤5% if Rust STFT is bit-equivalent

If still >5%, the ConvNeXt internal convs ALSO need state-carrying, which
requires a proper causal ONNX export of the encoder.
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
sys.path.insert(0, str(TINYVC_ROOT.parent))

DATA_VOICES = REPO_ROOT / "data" / "voices"
DATA_SOURCE = REPO_ROOT / "data" / "source"
OUT_JSON = REPO_ROOT / "data" / "m2_rust_stft.json"

N_FFT = 1920
HOP_SIZE = 480
CHUNK_FRAMES = 4
CHUNK_SAMPLES = CHUNK_FRAMES * HOP_SIZE


def _load_wav_24k(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != 24000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24000)
    return wav.astype(np.float32)


def _torch_stft(wav: np.ndarray) -> np.ndarray:
    """Reference torch.stft, matching TinyVC's spectrogram() function."""
    wav_t = torch.from_numpy(wav).float()
    window = torch.hann_window(N_FFT)
    spec_complex = torch.stft(wav_t, N_FFT, HOP_SIZE, window=window, return_complex=True)
    spec = spec_complex.abs().numpy()
    return spec[:, 1:]  # match TinyVC's `spec[:, :, 1:]`


def _rust_stft(wav: np.ndarray) -> np.ndarray:
    """Rust STFT via vc_python.linear_stft_magnitude_np."""
    import vc_python
    # Returns [T_frames, n_bins]; transpose to [n_bins, T_frames] for matching torch layout
    spec = vc_python.linear_stft_magnitude_np(wav, n_fft=N_FFT, hop=HOP_SIZE)
    return spec.T  # [n_bins, T_frames]


def _compute_shift(offline: np.ndarray, streaming: np.ndarray) -> dict:
    T = min(offline.shape[-1], streaming.shape[-1])
    off = offline[..., :T]
    strm = streaming[..., :T]
    off_mean = off.mean(axis=-1)
    strm_mean = strm.mean(axis=-1)
    off_std = off.std(axis=-1) + 1e-8
    strm_std = strm.std(axis=-1) + 1e-8
    rel_mean_shift = np.abs(off_mean - strm_mean) / (np.abs(off_mean) + 1e-8)
    rel_std_shift = np.abs(off_std - strm_std) / off_std
    return {
        "T_compared": int(T),
        "mean_rel_shift_per_channel_mean": float(rel_mean_shift.mean()),
        "mean_rel_shift_per_channel_max": float(rel_mean_shift.max()),
        "std_rel_shift_per_channel_mean": float(rel_std_shift.mean()),
        "std_rel_shift_per_channel_max": float(rel_std_shift.max()),
        "shift_overall_mean": float(max(rel_mean_shift.mean(), rel_std_shift.mean())),
        "shift_overall_max": float(max(rel_mean_shift.max(), rel_std_shift.max())),
    }


def main() -> int:
    print("=" * 72)
    print("M2 Round 15 · Rust STFT + Rust streaming primitives test")
    print("=" * 72)
    try:
        import vc_python
        print(f"  vc_python loaded ✓ (Rust PyO3 extension)")
    except ImportError as e:
        print(f"  vc_python NOT available: {e}")
        return 1

    print("\n  Loading V1Infer for offline encoder reference...")
    from vc_realtime.infer_v1 import V1Infer
    v1_infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                       device="cpu", top_k=4, alpha=0.0)

    # Test on all 5 voices + 5 sources
    test_wavs = []
    for i in range(5):
        test_wavs.append((f"voice_{i}", DATA_VOICES / f"voice_{i}.wav"))
    for i in range(5):
        src_path = DATA_SOURCE / f"source_{i+1:03d}.wav"
        if src_path.exists():
            test_wavs.append((f"source_{i+1:03d}", src_path))

    print(f"\n  Phase 1: STFT verification (Rust vs torch) on {len(test_wavs)} wavs")
    stft_results = []
    for name, path in test_wavs:
        if not path.exists():
            continue
        wav = _load_wav_24k(path)
        wav_pre = v1_infer._preprocess(wav, 24000)
        spec_torch = _torch_stft(wav_pre)
        spec_rust = _rust_stft(wav_pre)
        T = min(spec_torch.shape[-1], spec_rust.shape[-1])
        diff = np.abs(spec_torch[..., :T] - spec_rust[..., :T])
        rel_max = diff.max() / (np.abs(spec_torch).max() + 1e-8) * 100
        rel_mean = diff.mean() / (np.abs(spec_torch).mean() + 1e-8) * 100
        stft_results.append({
            "name": name,
            "torch_shape": list(spec_torch.shape),
            "rust_shape": list(spec_rust.shape),
            "T_compared": int(T),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "relative_max_diff_pct": float(rel_max),
            "relative_mean_diff_pct": float(rel_mean),
        })
        print(f"    {name:12s}: T_torch={spec_torch.shape[-1]} T_rust={spec_rust.shape[-1]} "
              f"max_diff={rel_max:.3f}% mean_diff={rel_mean:.3f}%")

    avg_stft_max = float(np.mean([r["relative_max_diff_pct"] for r in stft_results]))
    print(f"\n  Phase 1 result: avg STFT max diff = {avg_stft_max:.3f}%")

    print(f"\n  Phase 2: encoder shift test (offline vs Rust-STFT-chunked-encoder)")
    enc_results = []
    pass_count = 0
    for name, path in test_wavs:
        if not path.exists():
            continue
        wav = _load_wav_24k(path)
        wav_pre = v1_infer._preprocess(wav, 24000)

        # Offline: full STFT → full encoder
        spec_offline = _torch_stft(wav_pre)  # [961, T]
        spec_offline_t = torch.from_numpy(spec_offline).float().unsqueeze(0)  # [1, 961, T]
        with torch.inference_mode():
            content_offline, _ = v1_infer.encoder.infer(spec_offline_t)
        content_offline = content_offline.numpy()  # [1, 768, T]

        # Chunked: Rust STFT → split into 4-frame chunks → encoder per chunk
        spec_rust = _rust_stft(wav_pre)  # [961, T']
        T = spec_rust.shape[-1]
        pad = (-T) % CHUNK_FRAMES
        if pad:
            spec_rust = np.concatenate([spec_rust, np.zeros((961, pad), dtype=np.float32)], axis=-1)
        n_chunks = spec_rust.shape[-1] // CHUNK_FRAMES
        contents = []
        for i in range(n_chunks):
            spec_chunk = spec_rust[:, i * CHUNK_FRAMES : (i + 1) * CHUNK_FRAMES]
            spec_chunk_t = torch.from_numpy(spec_chunk).float().unsqueeze(0)
            with torch.inference_mode():
                content_chunk, _ = v1_infer.encoder.infer(spec_chunk_t)
            contents.append(content_chunk.numpy())
        content_chunked = np.concatenate(contents, axis=2) if contents else np.zeros((1, 768, 0))

        shift = _compute_shift(content_offline, content_chunked)
        pass_ = shift["shift_overall_mean"] <= 0.05
        if pass_:
            pass_count += 1
        enc_results.append({
            "name": name,
            "shift": shift,
            "pass": bool(pass_),
        })
        print(f"    {name:12s}: shift {shift['shift_overall_mean']*100:6.2f}% "
              f"(max {shift['shift_overall_max']*100:.2f}%) "
              f"{'✓' if pass_ else '✗'}")

    overall_pass = pass_count == len(enc_results)
    print(f"\n  Phase 2 result: {pass_count}/{len(enc_results)} pass")
    print(f"  M2 acceptance (Rust STFT + chunked encoder ≤ 5%): "
          f"{'PASS ✓' if overall_pass else 'FAIL ✗'}")

    out = {
        "config": {
            "n_fft": N_FFT, "hop": HOP_SIZE,
            "chunk_frames": CHUNK_FRAMES, "chunk_samples": CHUNK_SAMPLES,
            "stft_backend": "vc_python.linear_stft_magnitude_np (Rust)",
            "encoder_backend": "TinyVC PyTorch (chunked per 4 frames)",
            "acceptance_threshold_pct": 5.0,
        },
        "phase1_stft_verification": stft_results,
        "phase2_encoder_shift": enc_results,
        "summary": {
            "phase1_avg_stft_max_diff_pct": avg_stft_max,
            "phase2_pass_count": pass_count,
            "phase2_total": len(enc_results),
            "overall_pass": bool(overall_pass),
            "phase2_mean_shift_pct": float(np.mean(
                [r["shift"]["shift_overall_mean"] for r in enc_results]) * 100),
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    sys.exit(main())
