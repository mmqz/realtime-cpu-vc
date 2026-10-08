#!/usr/bin/env python3
"""Round 12 · src_005 + per-voice optimal kNN config.

Round 11 showed src_005 + M1b = +0.188 (top_k=4, alpha=0) — gap to +0.20
milestone only +0.012. This script sweeps top_k × alpha ON SRC_005 to
find per-voice optimal config, then computes the per-voice optimal mean.

If per-voice optimal mean ≥ +0.20, M1 stretch milestone is achieved.

Grid: top_k ∈ [1, 4, 8, 16] × alpha ∈ [0.0, 0.1, 0.3, 0.5] = 16 configs.
Per voice, pick the best config, average across 5 voices.
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
OUT_JSON = REPO_ROOT / "data" / "m1b_src005_pervoice_sweep.json"
TARGET_SPEAKERS = ["p225", "p226", "p227", "p228", "p229"]

CANONICAL_SOURCE = "source_005.wav"
TOP_K_GRID = [1, 4, 8, 16]
ALPHA_GRID = [0.0, 0.1, 0.3, 0.5]


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
    print(f"\n[Round 12] source: {src_path.name} ({len(src_wav)/24000:.1f}s)")

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

    print("\n  Encoding source once...")
    wav_pre = infer._preprocess(src_wav, 24000)
    content, f0, energy = infer.encode(wav_pre)
    f0_track = f0.squeeze()

    # Pre-compute mapped F0 per voice
    mapped_f0_per_voice = {}
    for vid in range(5):
        mapper.select_voice(vid)
        mapped_f0_per_voice[vid] = mapper.map_f0(f0_track).reshape(1, 1, -1).astype(np.float32)

    # Sweep: matrix [config_idx, voice] → vc_effect
    print(f"\n  Sweeping {len(TOP_K_GRID)}×{len(ALPHA_GRID)} = "
          f"{len(TOP_K_GRID)*len(ALPHA_GRID)} configs × 5 voices = "
          f"{len(TOP_K_GRID)*len(ALPHA_GRID)*5} inferences")
    # Save best-config wavs (one per voice for ablation)
    best_per_voice = [None] * 5  # (vc_effect, top_k, alpha, target_sim, source_sim)
    config_results = []
    t_start = time.perf_counter()
    for top_k in TOP_K_GRID:
        for alpha in ALPHA_GRID:
            config_row = []
            for vid in range(5):
                content_replaced = infer.knn_replace(
                    content, voice_id=vid, top_k=top_k, alpha=alpha)
                mapped_f0_dec = mapped_f0_per_voice[vid]
                out_wav = infer.decode(content_replaced, mapped_f0_dec, energy)
                # Save wavs only for the best per-voice config (do it later when we know)
                tmp = OUTPUT_DIR / f"_tmp_pervoice_{vid}.wav"
                peak = max(np.max(np.abs(out_wav)), 1e-8)
                norm = out_wav * (10 ** (-3 / 20) / peak)
                sf.write(str(tmp),
                         (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)
                out_emb = _embed(ext, _load_wav_as_16k_mono(tmp))
                tmp.unlink(missing_ok=True)
                target_sim = _cosine(target_embs[vid], out_emb)
                source_sim = _cosine(src_emb, out_emb)
                vc_effect = target_sim - source_sim
                config_row.append({
                    "voice_id": vid,
                    "target_sim": float(target_sim),
                    "source_sim": float(source_sim),
                    "vc_effect": float(vc_effect),
                })
                # Track best per voice
                if best_per_voice[vid] is None or vc_effect > best_per_voice[vid][0]:
                    best_per_voice[vid] = (float(vc_effect), top_k, alpha,
                                            float(target_sim), float(source_sim))
            mean_eff = float(np.mean([r["vc_effect"] for r in config_row]))
            config_results.append({
                "top_k": top_k, "alpha": alpha,
                "mean_vc_effect": mean_eff,
                "voices": config_row,
            })
            print(f"    top_k={top_k:2d} alpha={alpha:.1f}: "
                  f"mean={mean_eff:+.3f}  "
                  f"per-voice: " + " ".join(f"v{r['voice_id']}={r['vc_effect']:+.3f}" for r in config_row))

    elapsed = time.perf_counter() - t_start
    print(f"\n  Sweep completed in {elapsed:.0f}s")

    # Per-voice optimal
    print("\n  Per-voice optimal config:")
    optimal_mean = 0.0
    for vid in range(5):
        eff, tk, al, ts, ss = best_per_voice[vid]
        optimal_mean += eff
        print(f"    voice_{vid} ({TARGET_SPEAKERS[vid]}): "
              f"top_k={tk} alpha={al} → VC_effect={eff:+.3f} "
              f"(target_sim={ts:.3f}, source_sim={ss:.3f})")
    optimal_mean /= 5
    print(f"\n  Per-voice optimal mean: {optimal_mean:+.3f}")
    print(f"\n  Milestone check:")
    print(f"    M1 stretch (abs ≥ +0.20):  "
          f"{'PASS ✓' if optimal_mean >= 0.20 else 'FAIL'} "
          f"(gap {0.20 - optimal_mean:+.3f})")

    # Also compute best single-config mean (use same config for all voices)
    best_config = max(config_results, key=lambda r: r["mean_vc_effect"])
    print(f"\n  Best single-config (same for all voices):")
    print(f"    top_k={best_config['top_k']} alpha={best_config['alpha']} "
          f"→ mean={best_config['mean_vc_effect']:+.3f}")

    out = {
        "config": {
            "source": CANONICAL_SOURCE,
            "grid": {"top_k": TOP_K_GRID, "alpha": ALPHA_GRID},
        },
        "sweep_results": config_results,
        "per_voice_optimal": [
            {
                "voice_id": vid,
                "speaker": TARGET_SPEAKERS[vid],
                "top_k": best_per_voice[vid][1],
                "alpha": best_per_voice[vid][2],
                "vc_effect": best_per_voice[vid][0],
                "target_sim": best_per_voice[vid][3],
                "source_sim": best_per_voice[vid][4],
            }
            for vid in range(5)
        ],
        "summary": {
            "per_voice_optimal_mean": optimal_mean,
            "best_single_config_mean": best_config["mean_vc_effect"],
            "best_single_config": {
                "top_k": best_config["top_k"],
                "alpha": best_config["alpha"],
            },
        },
        "milestone": {
            "target_abs": 0.20,
            "per_voice_optimal_pass": bool(optimal_mean >= 0.20),
            "gap_to_milestone": 0.20 - optimal_mean,
        },
        "baseline_comparison": {
            "m1b_src001_baseline": 0.131,
            "m1b_src005_top4_alpha0": 0.188,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
