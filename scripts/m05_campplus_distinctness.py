#!/usr/bin/env python3
"""M0.5 · CAMPPlus distinctness check for the 5 real VCTK voices.

Verifies the M0.5 acceptance criterion "CAMPPlus pairwise cosine < 0.70".
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

REPO_ROOT = Path(__file__).resolve().parent.parent

CAMPPLUS_HF_REPO = "bitsydarel/campplus-onnx"
CAMPPLUS_ONNX_NAME = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
CAMPPLUS_DIM = 192
CAMPPLUS_SR = 16000

DATA_VOICES = REPO_ROOT / "data" / "voices"
OUT_PATH = DATA_VOICES / "campplus_distinctness.json"


def _ensure_campplus_model() -> Path:
    local = REPO_ROOT / "models" / CAMPPLUS_ONNX_NAME
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[campplus] downloading from HF {CAMPPLUS_HF_REPO}...", flush=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=CAMPPLUS_HF_REPO, filename=CAMPPLUS_ONNX_NAME,
        local_dir=str(local.parent),
    )
    if not local.exists():
        raise RuntimeError(f"download did not produce {local}")
    print(f"[campplus] cached at {local}")
    return local


def _make_extractor(model_path: Path):
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path), num_threads=1, debug=False, provider="cpu",
    )
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    if not cfg.validate():
        raise RuntimeError("SpeakerEmbeddingExtractorConfig.validate() failed")
    if ext.dim != CAMPPLUS_DIM:
        raise RuntimeError(f"Expected CAMPPlus dim={CAMPPLUS_DIM}, got {ext.dim}")
    return ext


def _load_wav_as_16k_mono(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != CAMPPLUS_SR:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=CAMPPLUS_SR)
    return wav.astype(np.float32)


def _embed(ext, wav: np.ndarray) -> np.ndarray:
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32767.0
    elif wav.dtype == np.int32:
        wav = wav.astype(np.float32) / 2147483647.0
    elif wav.dtype == np.float64:
        wav = wav.astype(np.float32)
    min_samples = int(CAMPPLUS_SR * 0.5)
    if len(wav) < min_samples:
        wav = np.concatenate([wav, np.zeros(min_samples - len(wav), dtype=np.float32)])
    stream = ext.create_stream()
    stream.accept_waveform(CAMPPLUS_SR, wav)
    stream.input_finished()
    if not ext.is_ready(stream):
        for _ in range(100):
            if ext.is_ready(stream):
                break
            time.sleep(0.001)
    emb = ext.compute(stream)
    return np.asarray(emb, dtype=np.float32)


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    if not DATA_VOICES.exists():
        print(f"ERROR: {DATA_VOICES} does not exist; run m05_real_voices.py first",
              file=sys.stderr)
        return 1
    voice_paths = [DATA_VOICES / f"voice_{i}.wav" for i in range(5)]
    missing = [p for p in voice_paths if not p.exists()]
    if missing:
        print(f"ERROR: missing voice files: {missing}", file=sys.stderr)
        return 1

    print(f"\n[CAMPPlus] computing 192-d embeddings for 5 voices")
    model_path = _ensure_campplus_model()
    ext = _make_extractor(model_path)

    embeddings = []
    for i, p in enumerate(voice_paths):
        wav = _load_wav_as_16k_mono(p)
        min_len = CAMPPLUS_SR * 3
        if len(wav) < min_len:
            pad = np.zeros(min_len - len(wav), dtype=np.float32)
            wav = np.concatenate([wav, pad])
        emb = _embed(ext, wav)
        embeddings.append(emb)
        print(f"  voice_{i} ({p.name}): dim={emb.shape[0]} norm={np.linalg.norm(emb):.3f}")

    n = len(embeddings)
    matrix = np.eye(n, dtype=np.float32)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine(embeddings[i], embeddings[j])
            matrix[i, j] = sim
            matrix[j, i] = sim
            pairs.append((i, j, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    mean_sim = float(np.mean([p[2] for p in pairs]))
    max_sim = float(np.max([p[2] for p in pairs]))
    min_sim = float(np.min([p[2] for p in pairs]))
    max_pair = pairs[0]
    min_pair = pairs[-1]

    print("\n=== CAMPPlus Pairwise Cosine (real VCTK speakers) ===")
    print(f"  Mean:  {mean_sim:.3f}")
    print(f"  Max:   {max_sim:.3f}  (voice_{max_pair[0]} vs voice_{max_pair[1]})")
    print(f"  Min:   {min_sim:.3f}  (voice_{min_pair[0]} vs voice_{min_pair[1]})")
    print(f"\n  All pairs (sorted desc):")
    for i, j, sim in pairs:
        print(f"    voice_{i} vs voice_{j}: {sim:.3f}")

    print("\n=== Comparison to M0.5 baseline ===")
    print(f"  path3 (pitch_shifted fixtures): mean=0.450  range 0.184–0.867")
    print(f"  M0.5 (real VCTK p225-p229):     mean={mean_sim:.3f}  "
          f"range {min_sim:.3f}–{max_sim:.3f}")
    pass_threshold = mean_sim < 0.70
    print(f"\n  Acceptance: mean cosine < 0.70 → "
          f"{'PASS ✓' if pass_threshold else 'FAIL ✗'}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "model": CAMPPLUS_ONNX_NAME, "dim": CAMPPLUS_DIM, "n_voices": n,
        "voices": [str(p.name) for p in voice_paths],
        "pairwise_cosine": {f"voice_{i}_vs_voice_{j}": float(s) for i, j, s in pairs},
        "matrix": matrix.tolist(),
        "summary": {
            "mean": mean_sim, "max": max_sim,
            "max_pair": [int(max_pair[0]), int(max_pair[1])],
            "min": min_sim, "min_pair": [int(min_pair[0]), int(min_pair[1])],
        },
        "acceptance_pass": bool(pass_threshold),
        "baseline_comparison": {
            "path3_pitch_shifted_mean": 0.450,
            "path3_pitch_shifted_range": [0.184, 0.867],
        },
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_PATH}")
    return 0 if pass_threshold else 2


if __name__ == "__main__":
    sys.exit(main())
