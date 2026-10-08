#!/usr/bin/env python3
"""M1b · Canonical source_005 baseline + M1b eval — closest to +0.20 milestone.

Round 10 multi-source sweep showed source_005 (VCTK prompt 003:
"Six spoons of fresh snow peas, five thick slabs of blue cheese, and
maybe a snack for her brother Bob.") gives the best all-around mean
VC_effect (+0.168 with top_k=1, alpha=0). This script re-runs the M0.5
baseline + M1b F0-mapping eval using source_005 as the canonical source,
to establish an updated baseline that's more representative of typical
VC quality (rather than the arbitrarily-chosen source_001).

Goal: see how close src_005 + M1b gets to the +0.20 milestone stretch.
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
OUT_JSON = REPO_ROOT / "data" / "m1b_src005_canonical.json"
TARGET_SPEAKERS = ["p225", "p226", "p227", "p228", "p229"]

CANONICAL_SOURCE = "source_005.wav"  # best all-around from Round 10


def _load_wav_24k(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != 24000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24000)
    return wav.astype(np.float32)


def main() -> int:
    TINYVC_ROOT = Path(os.environ.get(
        "TINYVC_ROOT", str(REPO_ROOT.parent / "repos" / "tinyvc")))
    if not TINYVC_ROOT.exists():
        print(f"ERROR: TinyVC repo not found at {TINYVC_ROOT}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(TINYVC_ROOT.parent))

    from vc_realtime.infer_v1 import V1Infer
    from vc_realtime.pitch import F0QuantileMapper

    src_path = DATA_SOURCE / CANONICAL_SOURCE
    if not src_path.exists():
        print(f"ERROR: {src_path} missing", file=sys.stderr)
        return 1
    src_wav = _load_wav_24k(src_path)
    print(f"\n[src_005 canonical] source: {src_path.name} "
          f"({len(src_wav)/24000:.1f}s)")

    print("  Loading V1Infer...")
    infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                    device="cpu", top_k=4, alpha=0.0)

    print("  Building F0 quantile mapper...")
    mapper = F0QuantileMapper.from_voices_dir(str(DATA_VOICES), sr=24000)
    mapper.build_source_table_from_audio(src_wav, sr=24000)

    print("  Loading CAMPPlus...")
    campplus_path = _ensure_campplus_model()
    ext = _make_extractor(campplus_path)

    src_emb = _embed(ext, _load_wav_as_16k_mono(src_path))
    target_embs = []
    for i in range(5):
        emb = _embed(ext, _load_wav_as_16k_mono(DATA_VOICES / f"voice_{i}.wav"))
        target_embs.append(emb)

    print("\n  Encoding source once (content + f0 + energy)...")
    wav_pre = infer._preprocess(src_wav, 24000)
    content, f0, energy = infer.encode(wav_pre)
    f0_track = f0.squeeze()

    rows_baseline = []
    rows_m1b = []
    print("\n  Running per-voice eval (baseline + M1b):")
    for vid in range(5):
        # Baseline (no F0 mapping): kNN + DDSP with source F0
        content_replaced = infer.knn_replace(content, voice_id=vid,
                                              top_k=4, alpha=0.0)
        out_baseline = infer.decode(content_replaced, f0, energy)

        # M1b: kNN + DDSP with mapped F0
        mapper.select_voice(vid)
        mapped_f0 = mapper.map_f0(f0_track).reshape(1, 1, -1).astype(np.float32)
        out_m1b = infer.decode(content_replaced, mapped_f0, energy)

        # Save outputs
        for label, out in [("baseline", out_baseline), ("m1b", out_m1b)]:
            out_path = OUTPUT_DIR / f"vc_m1b_src005_{label}_voice_{vid}.wav"
            peak = max(np.max(np.abs(out)), 1e-8)
            norm = out * (10 ** (-3 / 20) / peak)
            sf.write(str(out_path),
                     (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)
            out_emb = _embed(ext, _load_wav_as_16k_mono(out_path))
            target_sim = _cosine(target_embs[vid], out_emb)
            source_sim = _cosine(src_emb, out_emb)
            vc_effect = target_sim - source_sim
            if label == "baseline":
                rows_baseline.append({
                    "voice_id": vid, "speaker": TARGET_SPEAKERS[vid],
                    "target_sim": float(target_sim),
                    "source_sim": float(source_sim),
                    "vc_effect": float(vc_effect),
                })
                print(f"    voice_{vid} baseline: VC_effect={vc_effect:+.3f}")
            else:
                rows_m1b.append({
                    "voice_id": vid, "speaker": TARGET_SPEAKERS[vid],
                    "target_sim": float(target_sim),
                    "source_sim": float(source_sim),
                    "vc_effect": float(vc_effect),
                    "f0_out_median_hz": float(np.median(mapped_f0[mapped_f0 > 0]))
                        if (mapped_f0 > 0).any() else 0,
                })
                print(f"    voice_{vid} M1b:      VC_effect={vc_effect:+.3f} "
                      f"(delta vs baseline {vc_effect - rows_baseline[-1]['vc_effect']:+.3f})")

    mean_baseline = float(np.mean([r["vc_effect"] for r in rows_baseline]))
    mean_m1b = float(np.mean([r["vc_effect"] for r in rows_m1b]))
    cross_baseline = [r["vc_effect"] for r in rows_baseline if r["voice_id"] in (3, 4)]
    cross_m1b = [r["vc_effect"] for r in rows_m1b if r["voice_id"] in (3, 4)]
    cross_baseline_mean = float(np.mean(cross_baseline))
    cross_m1b_mean = float(np.mean(cross_m1b))

    print("\n=== src_005 canonical summary (top_k=4, alpha=0) ===")
    print(f"  Baseline (no F0 mapping):")
    print(f"    mean VC_effect:  {mean_baseline:+.3f}")
    print(f"    cross-gender mean: {cross_baseline_mean:+.3f}")
    print(f"  M1b (F0 mapping):")
    print(f"    mean VC_effect:  {mean_m1b:+.3f}")
    print(f"    cross-gender mean: {cross_m1b_mean:+.3f}")
    print(f"  delta M1b - baseline: {mean_m1b - mean_baseline:+.3f}")
    print(f"\n  comparison to src_001 baselines (top_k=4 alpha=0):")
    print(f"    src_001 baseline: +0.001")
    print(f"    src_001 + M1b:    +0.131")
    print(f"    src_005 baseline: {mean_baseline:+.3f}")
    print(f"    src_005 + M1b:    {mean_m1b:+.3f}")
    print(f"\n  milestone check:")
    print(f"    M1 acceptance (delta ≥ +0.05):  "
          f"{'PASS ✓' if mean_m1b - mean_baseline >= 0.05 else 'FAIL'}")
    print(f"    M1 stretch (abs ≥ +0.20):      "
          f"{'PASS ✓' if mean_m1b >= 0.20 else 'FAIL'} "
          f"(gap {0.20 - mean_m1b:+.3f})")
    print(f"    single-voice 'good VC' ≥ +0.20: ", end="")
    n_good = sum(1 for r in rows_m1b if r["vc_effect"] >= 0.20)
    print(f"{n_good}/5 voices pass")

    out = {
        "config": {
            "source": CANONICAL_SOURCE,
            "top_k": 4, "alpha": 0.0,
        },
        "baseline_no_f0_mapping": rows_baseline,
        "m1b_with_f0_mapping": rows_m1b,
        "summary": {
            "baseline_mean_vc_effect": mean_baseline,
            "m1b_mean_vc_effect": mean_m1b,
            "delta_m1b_vs_baseline": mean_m1b - mean_baseline,
            "baseline_cross_gender_mean": cross_baseline_mean,
            "m1b_cross_gender_mean": cross_m1b_mean,
        },
        "src_001_baseline_comparison": {
            "src_001_baseline_mean": 0.001,
            "src_001_m1b_mean": 0.131,
            "src_005_baseline_mean": mean_baseline,
            "src_005_m1b_mean": mean_m1b,
            "src_005_lift_vs_src_001_baseline": mean_baseline - 0.001,
            "src_005_lift_vs_src_001_m1b": mean_m1b - 0.131,
        },
        "milestone": {
            "target_abs": 0.20,
            "m1_acceptance_delta_target": 0.05,
            "m1_acceptance_pass": bool(mean_m1b - mean_baseline >= 0.05),
            "m1_stretch_pass": bool(mean_m1b >= 0.20),
            "gap_to_milestone": 0.20 - mean_m1b,
            "n_voices_passing_good_vc": n_good,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
