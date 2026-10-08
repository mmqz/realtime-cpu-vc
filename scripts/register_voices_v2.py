"""
scripts/register_voices_v2.py — v2 hybrid speaker pre-registration
==================================================================
Uses OpenVoice v2 ReferenceEncoder (256-d) instead of TinyVC's kNN feature
extraction. Output is one se_<voice_id>.pth file per voice (256-d vector, ~1 KB
each, vs v1's 11.5 MB kNN index for 5 voices — a 4600× storage reduction).

Pipeline:
  Step 1: Collect 5 × 30 s reference wavs in data/voices/voice_*.wav
  Step 2: Resample to 22050 Hz (OpenVoice v2's expected rate)
  Step 3: Compute mel-spec with OpenVoice's params (80 mel, 200 Hz hop, 1024 n_fft)
  Step 4: Run ReferenceEncoder ONNX -> 256-d embedding per voice
  Step 5: Save as models/se_<id>.pth
  Step 6: Validate pairwise cosine similarity < 0.85 (same as v1)

Usage:
  python scripts/register_voices_v2.py \
      --voices-dir data/voices \
      --ref-encoder-onnx models/openvoice_ref_encoder.onnx \
      --output-dir models/
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vc_realtime.speaker_encoder import SpeakerEncoder, make_openvoice_mel_spec


def load_wav(path: str, target_sr: int = 22050) -> np.ndarray:
    """Load wav, resample to target_sr (default 22050 for OpenVoice v2)."""
    import librosa
    wav, file_sr = librosa.load(path, sr=target_sr, mono=True)
    if file_sr != target_sr:
        print(f"  Resampled {path} from {file_sr} Hz to {target_sr} Hz")
    # Normalize to -3 dBFS
    peak = float(np.max(np.abs(wav)) + 1e-8)
    target_peak = 10 ** (-3.0 / 20.0)
    wav = wav * (target_peak / peak)
    # Truncate to 30 s if longer
    if len(wav) > target_sr * 30:
        wav = wav[: target_sr * 30]
    return wav.astype(np.float32)


def validate_distinctness(embeddings: dict, threshold: float = 0.85):
    """Cosine similarity matrix; assert all pairs < threshold."""
    keys = sorted(embeddings.keys())
    se = np.stack([embeddings[k] for k in keys])  # [N, 256]
    se_n = se / (np.linalg.norm(se, axis=1, keepdims=True) + 1e-8)
    sim = se_n @ se_n.T  # [N, N]
    print("\nPairwise cosine similarity matrix:")
    print(sim.round(3))
    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s >= threshold:
                print(f"  FAIL: {keys[i]} vs {keys[j]} = {s:.3f} (>= {threshold})")
                return False
    print(f"  OK: all pairs < {threshold}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--voices-dir', required=True,
                   help='dir containing voice_*.wav files')
    p.add_argument('--ref-encoder-onnx', default='models/openvoice_ref_encoder.onnx',
                   help='OpenVoice v2 ReferenceEncoder ONNX path')
    p.add_argument('--output-dir', default='models/',
                   help='output directory for se_<id>.pth files')
    p.add_argument('--threshold', type=float, default=0.85,
                   help='max cosine similarity allowed between any two voices')
    args = p.parse_args()

    voices_dir = Path(args.voices_dir)
    wav_files = sorted(voices_dir.glob('voice_*.wav'))
    if not wav_files:
        print(f"ERROR: no voice_*.wav files in {voices_dir}")
        sys.exit(1)
    print(f"Found {len(wav_files)} reference voices:")
    for w in wav_files:
        print(f"  - {w}")

    print(f"\nLoading OpenVoice v2 ReferenceEncoder: {args.ref_encoder_onnx}")
    enc = SpeakerEncoder(args.ref_encoder_onnx)

    print("\nEncoding 256-d speaker embeddings (offline, one-time cost):")
    embeddings = {}
    for w in wav_files:
        # e.g. "voice_0" -> voice_id=0
        stem = w.stem
        try:
            voice_id = int(stem.split('_')[-1])
        except ValueError:
            voice_id = len(embeddings)
        print(f"  Encoding voice {voice_id} from {w}...")
        wav = load_wav(str(w), target_sr=SpeakerEncoder.OPENVOICE_SAMPLE_RATE)
        mel_spec = make_openvoice_mel_spec(wav)
        se = enc.encode_reference(mel_spec)  # [256]
        embeddings[voice_id] = se
        print(f"    shape: {se.shape}, dtype: {se.dtype}, "
              f"norm: {float(np.linalg.norm(se)):.3f}")

    # Validate
    print("\nValidating distinctness...")
    if not validate_distinctness(embeddings, args.threshold):
        print("\nREFUSING to write registry — please re-record the flagged voices.")
        sys.exit(2)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for voice_id, se in embeddings.items():
        out_path = out_dir / f'se_{voice_id}.pth'
        # Save as FP16 to halve storage (256 × 2 bytes = 512 bytes per voice!)
        torch.save(torch.from_numpy(se).half(), out_path)
        print(f"  Wrote {out_path} ({out_path.stat().st_size} bytes)")
    total_size = sum(
        (out_dir / f'se_{vid}.pth').stat().st_size for vid in embeddings
    )
    print(f"\nTotal registry: {total_size} bytes ({len(embeddings)} voices)")
    print(f"v1 kNN index would be ~12 MB; v2 OpenVoice embeddings: {total_size} bytes")
    print(f"  Storage reduction: ~{12 * 1024 * 1024 / total_size:.0f}× smaller")
    print("\nReady for v2 runtime: "
          "python scripts/realtime_infer.py --voice-id 0 --config configs/v2_hybrid.yaml")


if __name__ == '__main__':
    main()
