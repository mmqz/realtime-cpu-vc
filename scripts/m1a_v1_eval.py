#!/usr/bin/env python3
"""M1a · Run VoicePack inference on M0.5 baseline + measure delta.

Replaces KNNRetrieval with VoicePackConditioner in the v1 pipeline,
runs VC on the same source_001.wav → 5 VCTK target voices, and
computes CAMPPlus target_sim / source_sim / VC_effect per voice.

Compares to v1 baseline (M0.5):
  mean VC_effect = +0.001 (M0.5 baseline, kNN)
  target: M1a delta ≥ +0.10 (so M1a mean VC_effect ≥ +0.10)
  stretch: M1a mean VC_effect ≥ +0.15
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from m05_campplus_distinctness import (  # type: ignore
    _ensure_campplus_model, _make_extractor, _embed, _cosine,
    _load_wav_as_16k_mono, CAMPPLUS_SR,
)

OUTPUT_DIR = REPO_ROOT / "download"
DATA_SOURCE = REPO_ROOT / "data" / "source"
DATA_VOICES = REPO_ROOT / "data" / "voices"
OUT_JSON = REPO_ROOT / "data" / "m1a_v1_eval.json"
TARGET_SPEAKERS = ["p225", "p226", "p227", "p228", "p229"]


def _load_wav_24k(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != 24000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24000)
    return wav.astype(np.float32)


def main() -> int:
    # Lazy: infer_v1 needs TinyVC upstream
    TINYVC_ROOT = Path(os.environ.get(
        "TINYVC_ROOT", str(REPO_ROOT.parent / "repos" / "tinyvc")))
    if not TINYVC_ROOT.exists():
        print(f"ERROR: TinyVC repo not found at {TINYVC_ROOT}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(TINYVC_ROOT.parent))

    from vc_realtime.infer_v1 import V1Infer
    from vc_realtime.voicepack import VoicePackConditioner

    # Check VoicePack weights exist
    vp_path = REPO_ROOT / "models" / "voicepack_v1.safetensors"
    if not vp_path.exists():
        print(f"ERROR: {vp_path} not found; run scripts/train_voicepack_joint.py first",
              file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src_path = DATA_SOURCE / "source_001.wav"
    src_wav = _load_wav_24k(src_path)
    print(f"\n[M1a eval] source: {src_path.name} ({len(src_wav)/24000:.1f}s)")

    # Load v1 for encoder + decoder (we'll bypass kNN)
    print("  Loading V1Infer (encoder.pt + decoder.pt)...")
    t0 = time.perf_counter()
    infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                    device="cpu", top_k=4, alpha=0.0)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    # Load VoicePack conditioner
    print(f"  Loading VoicePack from {vp_path}...")
    cond = VoicePackConditioner(vp_path, device="cpu")

    # CAMPPlus
    print("  Loading CAMPPlus (for speaker similarity scoring)...")
    campplus_path = _ensure_campplus_model()
    ext = _make_extractor(campplus_path)

    src_emb = _embed(ext, _load_wav_as_16k_mono(src_path))
    print(f"  source emb norm={np.linalg.norm(src_emb):.3f}")

    target_embs = []
    for i in range(5):
        emb = _embed(ext, _load_wav_as_16k_mono(DATA_VOICES / f"voice_{i}.wav"))
        target_embs.append(emb)

    # Run VC for each target voice using VoicePack instead of kNN
    print("\n  Running VoicePack-VC inference per target voice:")
    rows = []
    for vid in range(5):
        cond.select_voice(vid)
        # Re-use V1Infer's encode → knn_replace (we'll override) → decode
        # But V1Infer's process_audio() always uses internal knn — we need
        # to run encode+kNN+decode manually.
        wav_24k = src_wav
        # 1. preprocess
        wav_pre = infer._preprocess(wav_24k, 24000)
        # 2. encode
        content, f0, energy = infer.encode(wav_pre)
        # 3. VoicePack replace (instead of kNN)
        content_voiced = cond.replace(content)  # [1, 768, T]
        # 4. decode
        t1 = time.perf_counter()
        out_wav = infer.decode(content_voiced, f0, energy)
        decode_time = time.perf_counter() - t1

        out_dur = len(out_wav) / 24000
        # CAMPPlus on the output
        out_path = OUTPUT_DIR / f"vc_m1a_voice_{vid}.wav"
        peak = max(np.max(np.abs(out_wav)), 1e-8)
        norm = out_wav * (10 ** (-3 / 20) / peak)
        sf.write(str(out_path),
                 (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)

        # Compute similarities
        out_emb = _embed(ext, _load_wav_as_16k_mono(out_path))
        target_sim = _cosine(target_embs[vid], out_emb)
        source_sim = _cosine(src_emb, out_emb)
        vc_effect = target_sim - source_sim

        print(f"    voice_{vid}: target_sim={target_sim:.3f} "
              f"source_sim={source_sim:.3f} VC_effect={vc_effect:+.3f} "
              f"(decode {decode_time*1000:.0f}ms)")
        rows.append({
            "voice_id": vid, "speaker": TARGET_SPEAKERS[vid],
            "target_sim": float(target_sim), "source_sim": float(source_sim),
            "vc_effect": float(vc_effect),
            "decode_time_ms": float(decode_time * 1000),
        })

    # Summary
    mean_target = float(np.mean([r["target_sim"] for r in rows]))
    mean_source = float(np.mean([r["source_sim"] for r in rows]))
    mean_effect = float(np.mean([r["vc_effect"] for r in rows]))

    print("\n=== M1a VoicePack eval summary ===")
    print(f"  mean target_sim:  {mean_target:.3f}  (M0.5 baseline 0.261)")
    print(f"  mean source_sim:  {mean_source:.3f}  (M0.5 baseline 0.260)")
    print(f"  mean VC_effect:   {mean_effect:+.3f}  (M0.5 baseline +0.001)")
    print(f"  delta vs M0.5:    {mean_effect - 0.001:+.3f}")
    print(f"  acceptance (delta ≥ +0.10): "
          f"{'PASS' if mean_effect - 0.001 >= 0.10 else 'FAIL'}")
    print(f"  stretch (delta ≥ +0.15):    "
          f"{'PASS' if mean_effect - 0.001 >= 0.15 else 'FAIL'}")

    out = {
        "source": str(src_path.relative_to(REPO_ROOT)),
        "model_weights": str(vp_path.relative_to(REPO_ROOT)),
        "voices": rows,
        "summary": {
            "mean_target_sim": mean_target,
            "mean_source_sim": mean_source,
            "mean_vc_effect": mean_effect,
            "delta_vs_m05": mean_effect - 0.001,
        },
        "baseline_comparison": {
            "m05_v1_baseline_mean_target_sim": 0.261,
            "m05_v1_baseline_mean_source_sim": 0.260,
            "m05_v1_baseline_mean_vc_effect": 0.001,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
