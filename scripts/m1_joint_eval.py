#!/usr/bin/env python3
"""M1 (joint) · VoicePack (M1a) + F0 quantile mapping (M1b) — additive eval.

Tests the M1 milestone acceptance criterion:
  "mean VC_effect ≥ +0.20 on the M0.5 real-speaker validation set"

Combines:
- M1a: VoicePack conditioning (replace per-frame content features with
  speaker-conditioned features via proj_in + FiLM_v + proj_out)
- M1b: F0 quantile mapping (remap source F0 track to target voice's
  F0 distribution in log-Hz domain, preserving unvoiced frames)

Hypothesis: the two mechanisms are *additive*:
- M1a handles formant/timbre transfer via content feature modulation
- M1b handles pitch range transfer via F0 remapping
- Together they should give delta ≈ +0.131 (M1b alone) + 0.05-0.10 (M1a)
  = +0.18-0.23, hitting the +0.20 milestone target.

Pipeline:
  source_audio
    → TinyVC encoder → content [1,768,T] + f0 [1,1,T] + energy
    → VoicePack.replace(content, voice_id) → content_voiced [1,768,T]   [M1a]
    → F0QuantileMapper.map_f0(f0, voice_id) → mapped_f0 [1,1,T]         [M1b]
    → TinyVC DDSP decoder(content_voiced, mapped_f0, energy) → output
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
OUT_JSON = REPO_ROOT / "data" / "m1_joint_eval.json"
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
    from vc_realtime.voicepack import VoicePackConditioner
    from vc_realtime.pitch import F0QuantileMapper

    # Check weights exist
    vp_path = REPO_ROOT / "models" / "voicepack_v1.safetensors"
    if not vp_path.exists():
        print(f"ERROR: {vp_path} missing; run scripts/train_voicepack_joint.py",
              file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src_path = DATA_SOURCE / "source_001.wav"
    src_wav = _load_wav_24k(src_path)
    print(f"\n[M1 joint eval] source: {src_path.name} ({len(src_wav)/24000:.1f}s)")

    print("  Loading V1Infer (encoder.pt + decoder.pt + voices.pt)...")
    t0 = time.perf_counter()
    infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                    device="cpu", top_k=4, alpha=0.0)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    print(f"  Loading VoicePack from {vp_path}...")
    cond = VoicePackConditioner(vp_path, device="cpu")

    print("  Building F0 quantile mapper from 5 voice references...")
    t0 = time.perf_counter()
    mapper = F0QuantileMapper.from_voices_dir(str(DATA_VOICES), sr=24000)
    print(f"  Built in {time.perf_counter()-t0:.1f}s")
    mapper.build_source_table_from_audio(src_wav, sr=24000)

    print("  Loading CAMPPlus (for speaker similarity scoring)...")
    campplus_path = _ensure_campplus_model()
    ext = _make_extractor(campplus_path)

    src_emb = _embed(ext, _load_wav_as_16k_mono(src_path))
    print(f"  source emb norm={np.linalg.norm(src_emb):.3f}")

    target_embs = []
    for i in range(5):
        emb = _embed(ext, _load_wav_as_16k_mono(DATA_VOICES / f"voice_{i}.wav"))
        target_embs.append(emb)

    # Encode source once
    print("\n  Encoding source once (content + f0 + energy)...")
    wav_pre = infer._preprocess(src_wav, 24000)
    content, f0, energy = infer.encode(wav_pre)
    f0_track = f0.squeeze()  # [T_frames]
    print(f"  content: {content.shape}, f0_track: {f0_track.shape}")

    print("\n  Running M1a VoicePack + M1b F0-mapping inference per voice:")
    rows = []
    for vid in range(5):
        cond.select_voice(vid)
        mapper.select_voice(vid)

        # M1a: VoicePack replace
        content_voiced = cond.replace(content)  # [1, 768, T]

        # M1b: F0 quantile mapping
        mapped_f0 = mapper.map_f0(f0_track)  # [T_frames]
        mapped_f0_dec = mapped_f0.reshape(1, 1, -1).astype(np.float32)

        # Decode with both mods
        t1 = time.perf_counter()
        out_wav = infer.decode(content_voiced, mapped_f0_dec, energy)
        decode_time = time.perf_counter() - t1

        out_path = OUTPUT_DIR / f"vc_m1_joint_voice_{vid}.wav"
        peak = max(np.max(np.abs(out_wav)), 1e-8)
        norm = out_wav * (10 ** (-3 / 20) / peak)
        sf.write(str(out_path),
                 (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)

        out_emb = _embed(ext, _load_wav_as_16k_mono(out_path))
        target_sim = _cosine(target_embs[vid], out_emb)
        source_sim = _cosine(src_emb, out_emb)
        vc_effect = target_sim - source_sim

        med_in = float(np.median(f0_track[f0_track > 0])) if (f0_track > 0).any() else 0
        med_out = float(np.median(mapped_f0[mapped_f0 > 0])) if (mapped_f0 > 0).any() else 0

        print(f"    voice_{vid}: target_sim={target_sim:.3f} "
              f"source_sim={source_sim:.3f} VC_effect={vc_effect:+.3f} "
              f"F0 {med_in:.0f}→{med_out:.0f}Hz "
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
    cross_gender = [r for r in rows if r["voice_id"] in (3, 4)]
    cross_gender_mean = float(np.mean([r["vc_effect"] for r in cross_gender]))

    print("\n=== M1 (joint: M1a VoicePack + M1b F0 mapping) summary ===")
    print(f"  mean target_sim:  {mean_target:.3f}  (M0.5 baseline 0.261)")
    print(f"  mean source_sim:  {mean_source:.3f}  (M0.5 baseline 0.260)")
    print(f"  mean VC_effect:   {mean_effect:+.3f}  (M0.5 baseline +0.001)")
    print(f"  delta vs M0.5:    {mean_effect - 0.001:+.3f}")
    print(f"  cross-gender mean VC_effect: {cross_gender_mean:+.3f} "
          f"(M0.5 baseline -0.114)")
    print(f"\n  Comparison to ablations:")
    print(f"    M1a alone (5-ep VoicePack):     delta -0.098")
    print(f"    M1b alone (F0 mapping):        delta +0.131")
    print(f"    M1a + M1b joint (this run):    delta {mean_effect - 0.001:+.3f}")
    print(f"\n  acceptance:")
    print(f"    delta ≥ +0.05: "
          f"{'PASS' if mean_effect - 0.001 >= 0.05 else 'FAIL'}")
    print(f"    M1 milestone (delta ≥ +0.199): "
          f"{'PASS' if mean_effect >= 0.20 else 'FAIL'}")

    out = {
        "source": str(src_path.relative_to(REPO_ROOT)),
        "weights": str(vp_path.relative_to(REPO_ROOT)),
        "voices": rows,
        "summary": {
            "mean_target_sim": mean_target,
            "mean_source_sim": mean_source,
            "mean_vc_effect": mean_effect,
            "delta_vs_m05": mean_effect - 0.001,
            "cross_gender_mean_vc_effect": cross_gender_mean,
        },
        "ablation_comparison": {
            "m05_baseline": 0.001,
            "m1a_alone_delta": -0.098,
            "m1b_alone_delta": 0.131,
            "m1_joint_delta": mean_effect - 0.001,
        },
        "milestone_target": 0.20,
        "milestone_pass": bool(mean_effect >= 0.20),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
