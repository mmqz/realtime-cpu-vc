#!/usr/bin/env python3
"""M1b · Re-run v1 + F0 quantile mapping (without VoicePack).

Tests the hypothesis: M0.5 baseline negative cross-gender VC_effect is
caused by source F0 not being remapped. If M1b works, the cross-gender
cases (voice_3, voice_4) should flip from negative to positive even
WITHOUT M1a VoicePack — proving F0 mapping is the missing piece.

Pipeline:
  source_audio → TinyVC encoder → content + f0 + energy
                ↓ (NEW M1b)
                F0QuantileMapper.map_f0(f0) → mapped_f0 [target voice distribution]
                ↓
  kNN-VC replace (alpha=0) → content_replaced
                ↓
  TinyVC DDSP decoder(content_replaced, mapped_f0, energy) → output

Compares to:
  M0.5 v1 baseline (no F0 mapping): mean VC_effect +0.001
    cross-gender voice_3 -0.156, voice_4 -0.072
  Target: cross-gender cases flip positive, mean delta ≥ +0.05
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
OUT_JSON = REPO_ROOT / "data" / "m1b_v1_eval.json"
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
    print(f"\n[M1b eval] source: {src_path.name} ({len(src_wav)/24000:.1f}s)")

    print("  Loading V1Infer (encoder.pt + decoder.pt + voices.pt)...")
    t0 = time.perf_counter()
    infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                    device="cpu", top_k=4, alpha=0.0)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    print("  Building F0 quantile mapper from 5 voice references...")
    t0 = time.perf_counter()
    mapper = F0QuantileMapper.from_voices_dir(str(DATA_VOICES), sr=24000)
    print(f"  Built in {time.perf_counter()-t0:.1f}s")
    # Build source table from the actual source audio (p232 male reading)
    mapper.build_source_table_from_audio(src_wav, sr=24000)
    print(f"  Source table built from {src_path.name}")

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

    # Pre-encode source (same for all 5 voices)
    print("\n  Encoding source once (content + f0 + energy)...")
    wav_pre = infer._preprocess(src_wav, 24000)
    content, f0, energy = infer.encode(wav_pre)
    # f0 shape is [1, 1, T_frames]; squeeze to [T_frames] for the mapper
    f0_track = f0.squeeze()  # [T_frames]
    print(f"  content: {content.shape}, f0: {f0.shape} → track {f0_track.shape}, "
          f"energy: {energy.shape}")
    print(f"  source F0 stats: voiced={int((f0_track>0).sum())}/{len(f0_track)}, "
          f"median={float(np.median(f0_track[f0_track>0]) if (f0_track>0).any() else 0):.1f}Hz")

    print("\n  Running v1 + F0 mapping inference per target voice:")
    rows = []
    for vid in range(5):
        mapper.select_voice(vid)
        # Map F0
        mapped_f0 = mapper.map_f0(f0_track)
        # Reshape back to [1, 1, T] for decoder
        mapped_f0_dec = mapped_f0.reshape(1, 1, -1).astype(np.float32)

        # kNN-VC replace (alpha=0, same as M0.5 baseline)
        content_replaced = infer.knn_replace(content, voice_id=vid, top_k=4, alpha=0.0)

        # Decode with MAPPED F0 (this is the M1b change)
        t1 = time.perf_counter()
        out_wav = infer.decode(content_replaced, mapped_f0_dec, energy)
        decode_time = time.perf_counter() - t1

        out_dur = len(out_wav) / 24000
        out_path = OUTPUT_DIR / f"vc_m1b_voice_{vid}.wav"
        peak = max(np.max(np.abs(out_wav)), 1e-8)
        norm = out_wav * (10 ** (-3 / 20) / peak)
        sf.write(str(out_path),
                 (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)

        out_emb = _embed(ext, _load_wav_as_16k_mono(out_path))
        target_sim = _cosine(target_embs[vid], out_emb)
        source_sim = _cosine(src_emb, out_emb)
        vc_effect = target_sim - source_sim

        # Track F0 mapping stats per voice
        voiced_in = int((f0_track > 0).sum())
        voiced_out = int((mapped_f0 > 0).sum())
        med_in = float(np.median(f0_track[f0_track > 0])) if voiced_in else 0
        med_out = float(np.median(mapped_f0[mapped_f0 > 0])) if voiced_out else 0

        print(f"    voice_{vid}: target_sim={target_sim:.3f} "
              f"source_sim={source_sim:.3f} VC_effect={vc_effect:+.3f} "
              f"F0 med {med_in:.0f}→{med_out:.0f}Hz "
              f"(decode {decode_time*1000:.0f}ms)")
        rows.append({
            "voice_id": vid, "speaker": TARGET_SPEAKERS[vid],
            "target_sim": float(target_sim), "source_sim": float(source_sim),
            "vc_effect": float(vc_effect),
            "f0_in_median_hz": med_in, "f0_out_median_hz": med_out,
            "decode_time_ms": float(decode_time * 1000),
        })

    # Summary
    mean_target = float(np.mean([r["target_sim"] for r in rows]))
    mean_source = float(np.mean([r["source_sim"] for r in rows]))
    mean_effect = float(np.mean([r["vc_effect"] for r in rows]))
    # Cross-gender subset (voice_3, voice_4)
    cross_gender = [r for r in rows if r["voice_id"] in (3, 4)]
    cross_gender_mean = float(np.mean([r["vc_effect"] for r in cross_gender]))

    print("\n=== M1b F0-mapping eval summary ===")
    print(f"  mean target_sim:  {mean_target:.3f}  (M0.5 baseline 0.261)")
    print(f"  mean source_sim:  {mean_source:.3f}  (M0.5 baseline 0.260)")
    print(f"  mean VC_effect:   {mean_effect:+.3f}  (M0.5 baseline +0.001)")
    print(f"  delta vs M0.5:    {mean_effect - 0.001:+.3f}")
    print(f"  cross-gender (v3,v4) mean VC_effect: {cross_gender_mean:+.3f} "
          f"(M0.5 baseline -0.114)")
    print(f"  acceptance (delta ≥ +0.05): "
          f"{'PASS' if mean_effect - 0.001 >= 0.05 else 'FAIL'}")

    out = {
        "source": str(src_path.relative_to(REPO_ROOT)),
        "voices": rows,
        "summary": {
            "mean_target_sim": mean_target,
            "mean_source_sim": mean_source,
            "mean_vc_effect": mean_effect,
            "delta_vs_m05": mean_effect - 0.001,
            "cross_gender_mean_vc_effect": cross_gender_mean,
            "cross_gender_m05_baseline": -0.114,
        },
        "baseline_comparison": {
            "m05_v1_baseline_mean_vc_effect": 0.001,
            "m05_v1_baseline_cross_gender_mean": -0.114,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
