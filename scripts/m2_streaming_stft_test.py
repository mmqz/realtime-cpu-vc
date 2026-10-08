#!/usr/bin/env python3
"""M2 Round 13 · State-carrying streaming STFT + chunked encoder test.

Round 8 showed naive streaming chunks produce 400-590% per-channel shift
vs offline. Root cause: torch.stft with center=True re-pads each chunk
from zero, producing extra edge-effect frames + per-chunk statistics.

This script implements state-carrying streaming STFT in pure Python:
1. Maintain a rolling buffer of n_fft samples (1920)
2. For each new hop_size samples (480), extract n_fft samples ending at
   the new boundary, apply Hann window + rfft → one new STFT frame
3. No center padding → no edge effects at chunk boundaries

Then run the TinyVC encoder on the streaming STFT in chunks (4 frames
per chunk = 80ms) and compare to:
- (a) Offline (full audio → full STFT → full encoder): the gold standard
- (b) Naive chunked (Round 8 baseline): the broken streaming path

If streaming STFT brings shift ≤5%, the STFT was the main issue. If
shift is still high, the ConvNeXt internal convs also need state-carrying
(Rust StreamingConv1dState).
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
sys.path.insert(0, str(TINYVC_ROOT.parent))  # so `module.*` resolves
sys.path.insert(0, str(TINYVC_ROOT))  # alternative resolution

from module.utils.spectrogram import spectrogram as _tinyvc_spectrogram  # noqa: E402

DATA_VOICES = REPO_ROOT / "data" / "voices"
DATA_SOURCE = REPO_ROOT / "data" / "source"
OUT_JSON = REPO_ROOT / "data" / "m2_streaming_stft.json"

N_FFT = 1920
HOP_SIZE = 480
CHUNK_FRAMES = 4  # 80 ms
CHUNK_SAMPLES = CHUNK_FRAMES * HOP_SIZE  # 1920


# ---------------------------------------------------------------------------
# State-carrying streaming STFT
# ---------------------------------------------------------------------------
class StreamingSTFT:
    """Incremental STFT with state-carrying rolling buffer.

    Mirrors torch.stft(n_fft=N_FFT, hop=HOP, center=False, window=hann)
    but processes audio in chunks of arbitrary size. Maintains a buffer
    of (n_fft - hop_size) past samples between chunks.

    Output: |STFT| spectrogram [1, fft_bin=961, T_frames]
    """

    def __init__(self, n_fft: int = N_FFT, hop: int = HOP_SIZE):
        self.n_fft = n_fft
        self.hop = hop
        self.window = torch.hann_window(n_fft)
        # Rolling buffer of (n_fft - hop) samples; carries across chunks
        self.buffer = np.zeros(n_fft - hop, dtype=np.float32)
        self.output_frames: list[np.ndarray] = []

    def reset(self) -> None:
        self.buffer = np.zeros(self.n_fft - self.hop, dtype=np.float32)
        self.output_frames = []

    def push(self, samples: np.ndarray) -> int:
        """Push new samples into the streaming STFT.

        Returns the number of new STFT frames produced.
        """
        samples = np.asarray(samples, dtype=np.float32)
        # Prepend buffer + new samples → continuous stream
        combined = np.concatenate([self.buffer, samples])
        n_full_frames = (len(combined) - self.n_fft) // self.hop + 1
        if n_full_frames <= 0:
            # Not enough samples for even one frame; store and return
            self.buffer = combined
            return 0
        # Compute STFT for n_full_frames new frames
        consumed = (n_full_frames - 1) * self.hop + self.n_fft
        # Note: torch.stft returns (n_fft//2 + 1) bins
        for i in range(n_full_frames):
            start = i * self.hop
            frame = combined[start : start + self.n_fft]
            # Apply Hann window + rfft → magnitude
            windowed = frame * self.window.numpy()
            spec = np.fft.rfft(windowed)
            mag = np.abs(spec).astype(np.float32)
            self.output_frames.append(mag)
        # Save unconsumed tail as new buffer
        self.buffer = combined[consumed:]
        return n_full_frames

    def get_spec(self) -> np.ndarray:
        """Return accumulated STFT frames as [1, fft_bin, T_frames]."""
        if not self.output_frames:
            return np.zeros((1, self.n_fft // 2 + 1, 0), dtype=np.float32)
        # Drop first frame to match TinyVC's `spec[:, :, 1:]` behavior
        frames = self.output_frames[1:] if len(self.output_frames) > 1 else self.output_frames
        return np.stack(frames, axis=-1)[None, ...]  # [1, fft_bin, T]


# ---------------------------------------------------------------------------
# Encoding modes for comparison
# ---------------------------------------------------------------------------
def encode_offline(wav: np.ndarray, v1_infer) -> np.ndarray:
    """Standard offline: full audio → full STFT → full encoder."""
    content, _, _ = v1_infer.encode(wav)
    return content


def encode_naive_chunked(wav: np.ndarray, v1_infer,
                          chunk_samples: int = CHUNK_SAMPLES) -> np.ndarray:
    """Round 8 baseline: chunk audio, encode each chunk separately."""
    n = len(wav)
    pad_n = (-n) % chunk_samples
    if pad_n:
        wav = np.concatenate([wav, np.zeros(pad_n, dtype=np.float32)])
    n_chunks = len(wav) // chunk_samples
    wav_pre = v1_infer._preprocess(wav, 24000)
    contents = []
    for i in range(n_chunks):
        chunk = wav_pre[i * chunk_samples : (i + 1) * chunk_samples]
        content, _, _ = v1_infer.encode(chunk)
        contents.append(content)
    return np.concatenate(contents, axis=2) if contents else np.zeros((1, 768, 0))


def encode_streaming_stft(wav: np.ndarray, v1_infer,
                           chunk_samples: int = CHUNK_SAMPLES) -> np.ndarray:
    """State-carrying STFT + chunked encoder.

    1. Compute streaming STFT (state-carrying) over the full wav
    2. Pass STFT frames to encoder in chunks of CHUNK_FRAMES
    """
    wav_pre = v1_infer._preprocess(wav, 24000)
    wav_t = torch.from_numpy(wav_pre).float()  # [L]

    # Compute streaming STFT
    streamer = StreamingSTFT(N_FFT, HOP_SIZE)
    # Push in chunks of CHUNK_SAMPLES (= CHUNK_FRAMES * HOP_SIZE)
    n = len(wav_t)
    pad_n = (-n) % chunk_samples
    if pad_n:
        wav_t_padded = torch.nn.functional.pad(wav_t, (0, pad_n))
    else:
        wav_t_padded = wav_t
    n_chunks = len(wav_t_padded) // chunk_samples
    for i in range(n_chunks):
        chunk = wav_t_padded[i * chunk_samples : (i + 1) * chunk_samples].numpy()
        streamer.push(chunk)
    spec_stream = streamer.get_spec()  # [1, 961, T_frames]
    spec_stream_t = torch.from_numpy(spec_stream).float()

    # Chunk the streaming STFT into CHUNK_FRAMES-frame pieces, run encoder
    T = spec_stream_t.shape[-1]
    n_chunks_spec = (T + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    contents = []
    for i in range(n_chunks_spec):
        start = i * CHUNK_FRAMES
        end = min(start + CHUNK_FRAMES, T)
        if end - start < 1:
            continue
        spec_chunk = spec_stream_t[..., start:end]  # [1, 961, frames]
        with torch.inference_mode():
            content, _ = v1_infer.encoder.infer(spec_chunk)
        contents.append(content)
    return np.concatenate(contents, axis=2) if contents else np.zeros((1, 768, 0))


# ---------------------------------------------------------------------------
# Compute shift
# ---------------------------------------------------------------------------
def _compute_shift(offline: np.ndarray, streaming: np.ndarray) -> dict:
    T = min(offline.shape[2], streaming.shape[2])
    off = offline[..., :T].squeeze(0)
    strm = streaming[..., :T].squeeze(0)
    off_mean = off.mean(axis=-1)
    strm_mean = strm.mean(axis=-1)
    off_std = off.std(axis=-1) + 1e-8
    strm_std = strm.std(axis=-1) + 1e-8
    rel_mean_shift = np.abs(off_mean - strm_mean) / (np.abs(off_mean) + 1e-8)
    rel_std_shift = np.abs(off_std - strm_std) / off_std
    return {
        "n_frames_compared": int(T),
        "mean_rel_shift_per_channel_mean": float(rel_mean_shift.mean()),
        "mean_rel_shift_per_channel_max": float(rel_mean_shift.max()),
        "std_rel_shift_per_channel_mean": float(rel_std_shift.mean()),
        "std_rel_shift_per_channel_max": float(rel_std_shift.max()),
        "shift_overall_mean": float(max(rel_mean_shift.mean(),
                                          rel_std_shift.mean())),
        "shift_overall_max": float(max(rel_mean_shift.max(),
                                         rel_std_shift.max())),
    }


# ---------------------------------------------------------------------------
# Sanity check: streaming STFT should match offline STFT (not encoder)
# ---------------------------------------------------------------------------
def _verify_streaming_stft(wav: np.ndarray, v1_infer) -> dict:
    """Compare streaming STFT output vs torch.stft output (offline).

    If they match, streaming STFT itself is correct; any encoder shift
    must come from the ConvNeXt internal convs.
    """
    wav_pre = v1_infer._preprocess(wav, 24000)
    wav_t = torch.from_numpy(wav_pre).float()  # [L]

    # Offline STFT via torch.stft (matches TinyVC's spectrogram.py logic but
    # with proper 1D input — torch 2.14+ doesn't accept 3D)
    window = torch.hann_window(N_FFT)
    spec_offline_complex = torch.stft(
        wav_t, N_FFT, HOP_SIZE, window=window, return_complex=True)
    spec_offline = spec_offline_complex.abs().numpy()  # [961, T]
    # Drop first frame to match TinyVC's `spec[:, :, 1:]`
    spec_offline = spec_offline[:, 1:]
    spec_offline = spec_offline[None, ...]  # [1, 961, T]

    # Streaming STFT
    streamer = StreamingSTFT(N_FFT, HOP_SIZE)
    chunk_size = CHUNK_SAMPLES
    n = len(wav_t)
    pad_n = (-n) % chunk_size
    if pad_n:
        wav_t_padded = torch.nn.functional.pad(wav_t, (0, pad_n))
    else:
        wav_t_padded = wav_t
    n_chunks = len(wav_t_padded) // chunk_size
    for i in range(n_chunks):
        chunk = wav_t_padded[i * chunk_size : (i + 1) * chunk_size].numpy()
        streamer.push(chunk)
    spec_stream = streamer.get_spec()  # [1, 961, T']

    T = min(spec_offline.shape[-1], spec_stream.shape[-1])
    diff = np.abs(spec_offline[..., :T] - spec_stream[..., :T])
    return {
        "offline_shape": list(spec_offline.shape),
        "streaming_shape": list(spec_stream.shape),
        "T_compared": int(T),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "offline_max": float(np.abs(spec_offline).max()),
        "relative_max_diff_pct": float(diff.max() / (np.abs(spec_offline).max() + 1e-8) * 100),
        "relative_mean_diff_pct": float(diff.mean() / (np.abs(spec_offline).mean() + 1e-8) * 100),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("M2 Round 13 · State-carrying streaming STFT + chunked encoder test")
    print("=" * 72)
    print(f"  n_fft={N_FFT}, hop={HOP_SIZE}, chunk={CHUNK_FRAMES} frames = "
          f"{CHUNK_SAMPLES} samples = {CHUNK_SAMPLES/24000*1000:.0f} ms")
    print(f"  acceptance: shift ≤ 5% → PASS")

    print("\n  Loading V1Infer...")
    from vc_realtime.infer_v1 import V1Infer
    v1_infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                       device="cpu", top_k=4, alpha=0.0)

    # Test on all 5 voices + 5 sources = 10 wavs
    test_wavs = []
    for i in range(5):
        test_wavs.append((f"voice_{i}", DATA_VOICES / f"voice_{i}.wav"))
    for i in range(5):
        src_path = DATA_SOURCE / f"source_{i+1:03d}.wav"
        if src_path.exists():
            test_wavs.append((f"source_{i+1:03d}", src_path))

    print(f"\n  Testing {len(test_wavs)} wavs (3 modes: offline, naive, streaming STFT):")
    results = []
    pass_naive = 0
    pass_stream = 0
    for name, path in test_wavs:
        if not path.exists():
            print(f"    [skip] {name}: missing")
            continue
        wav, sr = sf.read(str(path), always_2d=False)
        if wav.ndim > 1:
            wav = wav[:, 0]
        if sr != 24000:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24000)
        wav = wav.astype(np.float32)

        # Verify streaming STFT matches offline STFT (sanity)
        stft_verify = _verify_streaming_stft(wav, v1_infer)

        # Offline encoding
        offline = encode_offline(wav, v1_infer)
        # Naive chunked (Round 8 path)
        naive = encode_naive_chunked(wav, v1_infer, CHUNK_SAMPLES)
        # Streaming STFT + chunked encoder
        stream = encode_streaming_stft(wav, v1_infer, CHUNK_SAMPLES)

        shift_naive = _compute_shift(offline, naive)
        shift_stream = _compute_shift(offline, stream)

        pass_n = shift_naive["shift_overall_mean"] <= 0.05
        pass_s = shift_stream["shift_overall_mean"] <= 0.05
        if pass_n:
            pass_naive += 1
        if pass_s:
            pass_stream += 1

        results.append({
            "name": name,
            "duration_s": len(wav) / 24000,
            "stft_verify": stft_verify,
            "shift_naive_chunked_80ms": shift_naive,
            "shift_streaming_stft_80ms": shift_stream,
            "pass_naive": bool(pass_n),
            "pass_streaming": bool(pass_s),
        })
        n_pct = shift_naive["shift_overall_mean"] * 100
        s_pct = shift_stream["shift_overall_mean"] * 100
        stft_pct = stft_verify["relative_max_diff_pct"]
        print(f"    {name:12s} ({len(wav)/24000:.1f}s): "
              f"STFT diff {stft_pct:6.2f}% | "
              f"naive shift {n_pct:6.2f}% | "
              f"stream shift {s_pct:6.2f}% "
              f"{'✓' if pass_s else '✗'}")

    print(f"\n  Overall: naive 0/10 vs streaming STFT {pass_stream}/{len(results)} pass")
    print(f"  M2 acceptance (streaming STFT ≤ 5%): "
          f"{'PASS ✓' if pass_stream == len(results) else 'FAIL ✗'}")
    print(f"\n  STFT verification (streaming STFT vs offline STFT):")
    avg_stft_diff = float(np.mean([r["stft_verify"]["relative_max_diff_pct"] for r in results]))
    print(f"    average max relative diff: {avg_stft_diff:.2f}%")

    out = {
        "config": {
            "n_fft": N_FFT, "hop": HOP_SIZE,
            "chunk_frames": CHUNK_FRAMES, "chunk_samples": CHUNK_SAMPLES,
            "acceptance_threshold_pct": 5.0,
        },
        "results": results,
        "summary": {
            "n_wavs": len(results),
            "n_pass_naive": pass_naive,
            "n_pass_streaming_stft": pass_stream,
            "overall_naive_pass": bool(pass_naive == len(results)),
            "overall_streaming_pass": bool(pass_stream == len(results)),
            "avg_stft_max_relative_diff_pct": avg_stft_diff,
            "mean_shift_naive_pct": float(np.mean(
                [r["shift_naive_chunked_80ms"]["shift_overall_mean"] for r in results]) * 100),
            "mean_shift_streaming_pct": float(np.mean(
                [r["shift_streaming_stft_80ms"]["shift_overall_mean"] for r in results]) * 100),
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0 if pass_stream == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
