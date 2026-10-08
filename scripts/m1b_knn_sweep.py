#!/usr/bin/env python3
"""M1b · kNN parameter sweep: top_k × alpha grid search.

Tests whether tuning kNN retrieval parameters can break through the
+0.131 delta ceiling achieved by M1b alone (top_k=4, alpha=0).

Hypothesis: The cross-gender laggard (voice_3 p228 at +0.005) might
benefit from:
- Higher top_k (8, 16): more retrieval candidates → smoother content
  replacement, less variance per frame
- Non-zero alpha (0.1, 0.3, 0.5): blend source + target features →
  preserve naturalness while still injecting target identity

Grid: top_k ∈ [1, 4, 8, 16] × alpha ∈ [0.0, 0.1, 0.3, 0.5] = 16 configs
× 5 voices = 80 inferences (~5 min CPU). For each config, compute mean
VC_effect + per-voice breakdown.

Acceptance: any config with delta > +0.131 (M1b baseline).
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
OUT_JSON = REPO_ROOT / "data" / "m1b_knn_sweep.json"
TARGET_SPEAKERS = ["p225", "p226", "p227", "p228", "p229"]

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src_path = DATA_SOURCE / "source_001.wav"
    src_wav = _load_wav_24k(src_path)
    print(f"\n[M1b kNN sweep] source: {src_path.name} ({len(src_wav)/24000:.1f}s)")

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
    print(f"  content: {content.shape}, f0_track: {f0_track.shape}")

    # Pre-compute mapped F0 per voice (independent of kNN params)
    mapped_f0_per_voice = {}
    for vid in range(5):
        mapper.select_voice(vid)
        mapped_f0_per_voice[vid] = mapper.map_f0(f0_track).reshape(1, 1, -1).astype(np.float32)

    # Sweep
    print(f"\n  Sweeping {len(TOP_K_GRID)}×{len(ALPHA_GRID)} = "
          f"{len(TOP_K_GRID)*len(ALPHA_GRID)} configs × 5 voices = "
          f"{len(TOP_K_GRID)*len(ALPHA_GRID)*5} inferences")
    all_results = []
    t_start = time.perf_counter()
    for top_k in TOP_K_GRID:
        for alpha in ALPHA_GRID:
            config_rows = []
            for vid in range(5):
                # kNN replace with this top_k/alpha
                content_replaced = infer.knn_replace(
                    content, voice_id=vid, top_k=top_k, alpha=alpha)
                mapped_f0_dec = mapped_f0_per_voice[vid]
                t1 = time.perf_counter()
                out_wav = infer.decode(content_replaced, mapped_f0_dec, energy)
                decode_time = time.perf_counter() - t1
                # Save best-config wavs only (to save disk)
                out_path = None  # don't save for all configs
                # Compute CAMPPlus inline (need to write to temp file for sherpa)
                tmp_out = OUTPUT_DIR / f"_tmp_sweep_voice_{vid}.wav"
                peak = max(np.max(np.abs(out_wav)), 1e-8)
                norm = out_wav * (10 ** (-3 / 20) / peak)
                sf.write(str(tmp_out),
                         (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)
                out_emb = _embed(ext, _load_wav_as_16k_mono(tmp_out))
                tmp_out.unlink(missing_ok=True)
                target_sim = _cosine(target_embs[vid], out_emb)
                source_sim = _cosine(src_emb, out_emb)
                vc_effect = target_sim - source_sim
                config_rows.append({
                    "voice_id": vid, "speaker": TARGET_SPEAKERS[vid],
                    "target_sim": float(target_sim),
                    "source_sim": float(source_sim),
                    "vc_effect": float(vc_effect),
                    "decode_time_ms": float(decode_time * 1000),
                })
            mean_effect = float(np.mean([r["vc_effect"] for r in config_rows]))
            cross_gender = [r["vc_effect"] for r in config_rows if r["voice_id"] in (3, 4)]
            cross_mean = float(np.mean(cross_gender))
            all_results.append({
                "top_k": top_k, "alpha": alpha,
                "mean_vc_effect": mean_effect,
                "delta_vs_m05": mean_effect - 0.001,
                "delta_vs_m1b_baseline": mean_effect - 0.131,
                "cross_gender_mean": cross_mean,
                "voices": config_rows,
            })
            print(f"    top_k={top_k:2d} alpha={alpha:.1f}: "
                  f"mean VC_effect={mean_effect:+.3f} "
                  f"(delta vs M1b baseline {mean_effect - 0.131:+.3f}) "
                  f"cross-gender={cross_mean:+.3f}")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Sweep completed in {elapsed:.0f}s ({elapsed/len(all_results):.1f}s per config)")

    # Find best config
    best = max(all_results, key=lambda r: r["mean_vc_effect"])
    print(f"\n  BEST: top_k={best['top_k']} alpha={best['alpha']} "
          f"→ mean VC_effect={best['mean_vc_effect']:+.3f} "
          f"(delta vs M1b baseline {best['delta_vs_m1b_baseline']:+.3f})")
    print(f"\n  per-voice for best:")
    for r in best["voices"]:
        print(f"    voice_{r['voice_id']} ({r['speaker']}): "
              f"VC_effect={r['vc_effect']:+.3f} "
              f"target_sim={r['target_sim']:.3f} "
              f"source_sim={r['source_sim']:.3f}")

    pass_threshold = best["delta_vs_m1b_baseline"] > 0
    print(f"\n  acceptance (best delta > M1b baseline +0.131): "
          f"{'PASS ✓' if pass_threshold else 'FAIL ✗'}")

    out = {
        "source": str(src_path.relative_to(REPO_ROOT)),
        "grid": {"top_k": TOP_K_GRID, "alpha": ALPHA_GRID},
        "results": all_results,
        "best": best,
        "baseline_m1b_delta": 0.131,
        "pass": bool(pass_threshold),
        "sweep_elapsed_s": elapsed,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
