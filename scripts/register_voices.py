#!/usr/bin/env python3
"""
scripts/register_voices.py — Offline speaker pre-registration pipeline
=========================================================================
Walks through Steps 1-5 of analysis.md § 7:
  Step 1: Collect (assumed already done — wavs in --voices-dir)
  Step 2: Feature extraction (via TinyVC encoder)
  Step 3: Pack into voices.safetensors
  Step 4: Validate distinctness (cosine similarity < 0.85)
  Step 5: Warmup dummy forward (done at runtime startup, not here)

Usage:
    python scripts/register_voices.py \\
        --voices-dir data/voices \\
        --encoder-onnx models/encoder.onnx \\
        --output models/voices.safetensors
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

# vc_realtime is installed via `pip install -e .` (see pyproject.toml).
from vc_realtime.encoder import Encoder, make_mel_spec


def load_wav(path: str, sr: int = 24000) -> np.ndarray:
    """Load and validate a reference wav: 24 kHz, mono, ~30 s."""
    import librosa
    wav, file_sr = librosa.load(path, sr=sr, mono=True)
    if file_sr != sr:
        print(f"WARNING: {path} was {file_sr} Hz, resampled to {sr}")
    if len(wav) < sr * 25:
        print(f"WARNING: {path} is only {len(wav)/sr:.1f}s, expected ~30s")
    if len(wav) > sr * 30:
        wav = wav[: sr * 30]  # truncate to 30 s
    # Normalize to -3 dBFS
    peak = float(np.max(np.abs(wav)) + 1e-8)
    target_peak = 10 ** (-3.0 / 20.0)  # -3 dBFS
    wav = wav * (target_peak / peak)
    return wav.astype(np.float32)


def extract_voice_feature(wav: np.ndarray, encoder: Encoder) -> np.ndarray:
    """Encode 30 s wav → [1, 768, 1500] content feature tensor."""
    mel_spec = make_mel_spec(wav[None, :])  # [1, 128, T]
    content_feat, _, _ = encoder.encode(mel_spec)
    return content_feat  # [1, 768, T]


def validate_distinctness(voices: dict, threshold: float = 0.85):
    """Compute pairwise cosine similarity matrix; assert all pairs < threshold."""
    means = []
    keys = sorted(voices.keys())
    for k in keys:
        v = voices[k]  # [1, 768, T]
        means.append(v.mean(dim=-1).squeeze())  # [768]
    means = torch.stack(means)  # [N, 768]
    # Normalize
    means_n = means / (means.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(means_n, means_n.t())  # [N, N]
    print("\nPairwise cosine similarity matrix:")
    print(sim.numpy().round(3))
    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s >= threshold:
                print(f"  FAIL: {keys[i]} vs {keys[j]} similarity = {s:.3f} "
                      f"(>= threshold {threshold}). "
                      f"Re-record one of these voices.")
                return False
    print(f"  OK: all pairs < {threshold}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--voices-dir', required=True, help='dir containing voice_{id}.wav files')
    p.add_argument('--encoder-onnx', default='models/encoder.onnx',
                   help='TinyVC encoder ONNX path (or .int8.onnx)')
    p.add_argument('--output', default='models/voices.safetensors')
    p.add_argument('--threshold', type=float, default=0.85,
                   help='max cosine similarity allowed between any two voices')
    args = p.parse_args()

    voices_dir = Path(args.voices_dir)
    wav_files = sorted(voices_dir.glob('voice_*.wav'))
    if len(wav_files) == 0:
        print(f"ERROR: no voice_*.wav files found in {voices_dir}")
        sys.exit(1)
    print(f"Found {len(wav_files)} reference voices:")
    for w in wav_files:
        print(f"  - {w}")

    print(f"\nLoading encoder: {args.encoder_onnx}")
    encoder = Encoder(args.encoder_onnx)

    print("\nExtracting content features (one-time offline cost):")
    voices = {}
    for w in wav_files:
        voice_id = w.stem  # e.g. "voice_0"
        print(f"  Encoding {voice_id} from {w}...")
        wav = load_wav(str(w))
        feat = extract_voice_feature(wav, encoder)
        voices[voice_id] = torch.from_numpy(feat).half()  # FP16 storage
        print(f"    shape: {voices[voice_id].shape}, dtype: {voices[voice_id].dtype}")

    # Validate
    print("\nValidating distinctness...")
    if not validate_distinctness(voices, args.threshold):
        print("\nREFUSING to write registry — please re-record the flagged voices.")
        sys.exit(2)

    # Save as safetensors
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(voices, str(out_path))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {out_path} ({size_mb:.1f} MB)")
    print(f"  Voices registered: {len(voices)}")
    print(f"  Per-voice size: {size_mb / len(voices):.2f} MB")
    print("\nDone. Ready for runtime: python scripts/realtime_infer.py --voice-id 0")


if __name__ == '__main__':
    main()
