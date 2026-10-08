#!/usr/bin/env python3
"""M1b v2 · F0 quantile mapping with voice_2 (male 30s) as source table.

The original m1b_v1_eval.py builds source table from the 5s source_001.wav
clip (p232 reading prompt 001). That's a small sample of one speaker's F0
distribution — pyin may have missed low-probability voiced frames, leaving
the quantile table biased toward the few F0 values in that short clip.

This variant uses voice_2.wav (p227, male, 30s concat of 10 utterances)
as the source table — much larger and more representative of a typical
male F0 distribution. Should give a smoother, less-biased mapping.

Hypothesis: better source table → smoother F0 remap → small lift on
voice_3 (p228, the laggard at +0.005 in v1).
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
OUT_JSON = REPO_ROOT / "data" / "m1b_v2_eval.json"
TARGET_SPEAKERS = ["p225", "p226", "p227", "p228", "p229"]


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
    print(f"\n[M1b v2] source: {src_path.name} ({len(src_wav)/24000:.1f}s)")

    print("  Loading V1Infer...")
    infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                    device="cpu", top_k=4, alpha=0.0)

    print("  Building F0 quantile mapper (DO NOT override source table)...")
    mapper = F0QuantileMapper.from_voices_dir(str(DATA_VOICES), sr=24000)
    # voice_2 (p227) source table is the default — leave it alone
    src_table = mapper.source_table
    print(f"  source table: {len(src_table)} quantiles, "
          f"range [{np.exp(src_table.min()):.1f}, "
          f"{np.exp(src_table.max()):.1f}] Hz")

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
    print(f"  source F0: voiced={int((f0_track>0).sum())}/{len(f0_track)}, "
          f"median={float(np.median(f0_track[f0_track>0])):.1f}Hz")

    print("\n  Running v1 + F0 mapping (voice_2 source table) per voice:")
    rows = []
    for vid in range(5):
        mapper.select_voice(vid)
        mapped_f0 = mapper.map_f0(f0_track)
        mapped_f0_dec = mapped_f0.reshape(1, 1, -1).astype(np.float32)

        content_replaced = infer.knn_replace(content, voice_id=vid, top_k=4, alpha=0.0)

        t1 = time.perf_counter()
        out_wav = infer.decode(content_replaced, mapped_f0_dec, energy)
        decode_time = time.perf_counter() - t1

        out_path = OUTPUT_DIR / f"vc_m1b_v2_voice_{vid}.wav"
        peak = max(np.max(np.abs(out_wav)), 1e-8)
        norm = out_wav * (10 ** (-3 / 20) / peak)
        sf.write(str(out_path),
                 (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)

        out_emb = _embed(ext, _load_wav_as_16k_mono(out_path))
        target_sim = _cosine(target_embs[vid], out_emb)
        source_sim = _cosine(src_emb, out_emb)
        vc_effect = target_sim - source_sim

        med_out = float(np.median(mapped_f0[mapped_f0 > 0])) if (mapped_f0 > 0).any() else 0

        print(f"    voice_{vid}: target_sim={target_sim:.3f} "
              f"source_sim={source_sim:.3f} VC_effect={vc_effect:+.3f} "
              f"F0→{med_out:.0f}Hz (decode {decode_time*1000:.0f}ms)")
        rows.append({
            "voice_id": vid, "speaker": TARGET_SPEAKERS[vid],
            "target_sim": float(target_sim), "source_sim": float(source_sim),
            "vc_effect": float(vc_effect),
            "f0_out_median_hz": med_out,
            "decode_time_ms": float(decode_time * 1000),
        })

    mean_target = float(np.mean([r["target_sim"] for r in rows]))
    mean_source = float(np.mean([r["source_sim"] for r in rows]))
    mean_effect = float(np.mean([r["vc_effect"] for r in rows]))
    cross_gender = [r for r in rows if r["voice_id"] in (3, 4)]
    cross_gender_mean = float(np.mean([r["vc_effect"] for r in cross_gender]))

    print("\n=== M1b v2 (voice_2 source table) summary ===")
    print(f"  mean VC_effect:   {mean_effect:+.3f}  (M0.5 baseline +0.001)")
    print(f"  delta vs M0.5:    {mean_effect - 0.001:+.3f}")
    print(f"  cross-gender mean: {cross_gender_mean:+.3f}  (M0.5 -0.114)")
    print(f"\n  ablation:")
    print(f"    M1b v1 (source_001 5s table):  delta +0.131, voice_3 +0.005")
    print(f"    M1b v2 (voice_2 30s table):    delta {mean_effect - 0.001:+.3f}, "
          f"voice_3 {rows[3]['vc_effect']:+.3f}")

    out = {
        "source": str(src_path.relative_to(REPO_ROOT)),
        "voices": rows,
        "summary": {
            "mean_target_sim": mean_target,
            "mean_source_sim": mean_source,
            "mean_vc_effect": mean_effect,
            "delta_vs_m05": mean_effect - 0.001,
            "cross_gender_mean_vc_effect": cross_gender_mean,
        },
        "ablation": {
            "m1b_v1_delta": 0.131,
            "m1b_v2_delta": mean_effect - 0.001,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
