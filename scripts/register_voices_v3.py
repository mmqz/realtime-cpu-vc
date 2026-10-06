"""
scripts/register_voices_v3.py — v3 hybrid speaker pre-registration
====================================================================
Uses Spark-TTS BiCodec SpeakerEncoder (ECAPA-TDNN c512 + Perceiver + ResidualFSQ)
to encode 5 voices into 48-byte FSQ discrete codes + 512-d float embeddings.

Storage comparison:
  v1: voices.safetensors 12 MB (5 × 2.4 MB kNN index, FP16)
  v2: se_*.pth 2.5 KB total (5 × 256-d × 2 bytes, FP16)
  v3: se_*.fsq 240 bytes total (5 × 48 bytes, hash-retrievable)
       + se_*.pth 5 KB total (5 × 512-d × 2 bytes for runtime conditioning)
       → v3 = 5.24 KB total, still 2300× smaller than v1, 2× smaller than v2 alone

Pipeline:
  Step 1: Collect 5 × 30 s reference wavs in data/voices/voice_*.wav
  Step 2: Resample to 16 kHz (Spark-TTS expected rate)
  Step 3: Compute mel-spec with Spark params (80 mel, ~80 Hz hop, 1024 n_fft)
  Step 4: Run BiCodec SpeakerEncoder ONNX → 512-d latent + 192-d FSQ codes
  Step 5: Save as models/se_<id>.fsq (48 bytes packed) + models/se_<id>.pth (512-d FP16)
  Step 6: Validate all FSQ codes are distinct (2^48 space, near-impossible collision)

Usage:
  python scripts/register_voices_v3.py \\
      --voices-dir data/voices \\
      --speaker-encoder-onnx models/spark_speaker_encoder.onnx \\
      --output-dir models/
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vc_realtime.speaker_encoder_v3 import BiCodecSpeakerEncoder, make_spark_mel_spec


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load wav, resample to target_sr (default 16 kHz for Spark-TTS BiCodec)."""
    import librosa
    wav, file_sr = librosa.load(path, sr=target_sr, mono=True)
    if file_sr != target_sr:
        print(f"  Resampled {path} from {file_sr} Hz to {target_sr} Hz")
    peak = float(np.max(np.abs(wav)) + 1e-8)
    target_peak = 10 ** (-3.0 / 20.0)
    wav = wav * (target_peak / peak)
    if len(wav) > target_sr * 30:
        wav = wav[: target_sr * 30]
    return wav.astype(np.float32)


def pack_fsq_to_48_bytes(fsq_codes: np.ndarray) -> bytes:
    """Pack 192 int values (each 0-3, 2 bits) into 48 bytes.

    Spark-TTS BiCodec uses ResidualFSQ with levels=[4,4,4,4,4,4] — each code
    is 2 bits (0-3). 192 codes × 2 bits = 384 bits = 48 bytes.
    """
    # fsq_codes is [192] int32 with values 0-3
    flat = fsq_codes.astype(np.uint8).flatten()
    assert flat.max() < 4, f"FSQ code values out of range [0,3]: max={flat.max()}"
    # Pack 4 values per byte (2 bits each)
    packed = np.zeros(48, dtype=np.uint8)
    for i in range(0, 192, 4):
        byte = (flat[i] << 6) | (flat[i+1] << 4) | (flat[i+2] << 2) | flat[i+3]
        packed[i // 4] = byte
    return packed.tobytes()


def validate_distinct_fsq(fsq_codes: dict):
    """Check all 48-byte FSQ codes are distinct."""
    keys = sorted(fsq_codes.keys())
    seen = {}
    for vid in keys:
        code = fsq_codes[vid]
        h = code.tobytes()
        if h in seen:
            print(f"  WARNING: voice {vid} FSQ code identical to voice {seen[h]}!")
            return False
        seen[h] = vid
    print(f"  OK: all {len(keys)} FSQ codes distinct (2^48 space)")
    return True


def validate_distinct_embeddings(embeddings: dict, threshold: float = 0.85):
    """Cosine similarity matrix on 512-d embeddings."""
    keys = sorted(embeddings.keys())
    se = np.stack([embeddings[k] for k in keys])
    se_n = se / (np.linalg.norm(se, axis=1, keepdims=True) + 1e-8)
    sim = se_n @ se_n.T
    print("\nPairwise cosine similarity (512-d embeddings):")
    print(sim.round(3))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            s = float(sim[i, j])
            if s >= threshold:
                print(f"  FAIL: voice {keys[i]} vs {keys[j]} = {s:.3f} (>= {threshold})")
                return False
    print(f"  OK: all pairs < {threshold}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--voices-dir', required=True)
    p.add_argument('--speaker-encoder-onnx', default='models/spark_speaker_encoder.onnx',
                   help='Spark-TTS BiCodec SpeakerEncoder ONNX path')
    p.add_argument('--output-dir', default='models/')
    p.add_argument('--threshold', type=float, default=0.85)
    args = p.parse_args()

    voices_dir = Path(args.voices_dir)
    wav_files = sorted(voices_dir.glob('voice_*.wav'))
    if not wav_files:
        print(f"ERROR: no voice_*.wav files in {voices_dir}")
        sys.exit(1)
    print(f"Found {len(wav_files)} reference voices:")
    for w in wav_files:
        print(f"  - {w}")

    print(f"\nLoading Spark-TTS BiCodec SpeakerEncoder: {args.speaker_encoder_onnx}")
    enc = BiCodecSpeakerEncoder(args.speaker_encoder_onnx)

    print("\nEncoding 512-d + 48-byte FSQ per voice (offline, one-time):")
    embeddings = {}
    fsq_codes = {}
    for w in wav_files:
        stem = w.stem
        try:
            voice_id = int(stem.split('_')[-1])
        except ValueError:
            voice_id = len(embeddings)
        print(f"  Encoding voice {voice_id} from {w}...")
        wav = load_wav(str(w), target_sr=BiCodecSpeakerEncoder.SPARK_SAMPLE_RATE)
        mel_spec = make_spark_mel_spec(wav)
        se = enc.encode_reference(mel_spec)  # [512] float32
        embeddings[voice_id] = se
        fsq_codes[voice_id] = enc.fsq_codes[voice_id]  # [192] int32
        print(f"    512-d embedding norm: {float(np.linalg.norm(se)):.3f}")
        print(f"    FSQ codes: shape={fsq_codes[voice_id].shape}, "
              f"unique={len(np.unique(fsq_codes[voice_id]))}")

    # Validate FSQ distinctness
    print("\nValidating FSQ code distinctness (collision-resistant in 2^48 space)...")
    if not validate_distinct_fsq(fsq_codes):
        sys.exit(2)

    # Validate embedding distinctness
    print("\nValidating 512-d embedding distinctness (cosine sim < 0.85)...")
    if not validate_distinct_embeddings(embeddings, args.threshold):
        sys.exit(3)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for voice_id, se in embeddings.items():
        # Save 512-d embedding as FP16 .pth (for runtime conditioning, 1 KB per voice)
        emb_path = out_dir / f'se_{voice_id}.pth'
        torch.save(torch.from_numpy(se).half(), emb_path)
        # Save 48-byte FSQ code as .fsq (for hash-keyed O(1) retrieval)
        fsq_path = out_dir / f'se_{voice_id}.fsq'
        with open(fsq_path, 'wb') as f:
            f.write(pack_fsq_to_48_bytes(fsq_codes[voice_id]))
        print(f"  Wrote {emb_path} ({emb_path.stat().st_size} bytes) + "
              f"{fsq_path} ({fsq_path.stat().st_size} bytes)")

    total_bytes = sum(
        (out_dir / f'se_{vid}').with_suffix('.pth').stat().st_size +
        (out_dir / f'se_{vid}').with_suffix('.fsq').stat().st_size
        for vid in embeddings
    )
    print(f"\nTotal registry: {total_bytes} bytes ({len(embeddings)} voices)")
    print(f"  v1 kNN index would be ~12 MB; v3 (embeddings + FSQ codes): {total_bytes} bytes")
    print(f"  Storage reduction vs v1: ~{12 * 1024 * 1024 / total_bytes:.0f}× smaller")
    print(f"  Storage reduction vs v2 (~2.5 KB): ~{2500 / total_bytes:.2f}× smaller (or {total_bytes / 2500:.2f}× if embeddings included)")
    print("\nReady for v3 runtime: "
          "python scripts/realtime_infer.py --voice-id 0 --config configs/v3_hybrid.yaml")


if __name__ == '__main__':
    main()
