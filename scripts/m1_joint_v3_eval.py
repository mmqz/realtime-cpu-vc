#!/usr/bin/env python3
"""M1a V3 + M1b F0 mapping joint eval.

Tests whether the V3 VoicePack (trained with mel-spec reconstruction loss
on decoded audio) actually adds value on top of M1b F0 mapping alone.

If delta > 0, V3 is the first M1a architecture that doesn't hurt M1b.
If delta ≤ 0, V3 also fails (like V1 and V2).
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
OUT_JSON = REPO_ROOT / "data" / "m1_joint_v3_eval.json"
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
    from vc_realtime.voicepack_v2 import VoicePackConditionerV2  # V3 uses V2 architecture
    from vc_realtime.pitch import F0QuantileMapper

    vp_path = REPO_ROOT / "models" / "voicepack_v3.safetensors"
    if not vp_path.exists():
        print(f"ERROR: {vp_path} missing; run scripts/train_voicepack_v3.py",
              file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src_path = DATA_SOURCE / "source_005.wav"  # canonical best source from Round 11
    src_wav = _load_wav_24k(src_path)
    print(f"\n[M1a V3 + M1b] source: {src_path.name} ({len(src_wav)/24000:.1f}s)")

    print("  Loading V1Infer...")
    t0 = time.perf_counter()
    infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                    device="cpu", top_k=16, alpha=0.0)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    print(f"  Loading VoicePack V3 (mel-spec loss trained) from {vp_path}...")
    cond = VoicePackConditionerV2(vp_path, device="cpu")

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

    print("\n  Running M1a V3 + M1b per voice:")
    rows = []
    for vid in range(5):
        cond.select_voice(vid)
        mapper.select_voice(vid)
        # M1a V3: VoicePack replace
        content_voiced = cond.replace(content)
        # M1b: F0 mapping
        mapped_f0 = mapper.map_f0(f0_track)
        mapped_f0_dec = mapped_f0.reshape(1, 1, -1).astype(np.float32)
        # Decode with both mods
        t1 = time.perf_counter()
        out_wav = infer.decode(content_voiced, mapped_f0_dec, energy)
        decode_time = time.perf_counter() - t1

        out_path = OUTPUT_DIR / f"vc_m1_v3_voice_{vid}.wav"
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

    print("\n=== M1a V3 (mel-spec loss) + M1b summary ===")
    print(f"  mean VC_effect:   {mean_effect:+.3f}")
    print(f"  cross-gender mean: {cross_gender_mean:+.3f}")
    print(f"\n  ablation comparison:")
    print(f"    M0.5 baseline:                  +0.001")
    print(f"    M1b alone (top_k=16 α=0):       +0.203  ← M1 milestone PASS")
    print(f"    M1a V1 + M1b joint (30ep):      -0.014  (V1 hurts)")
    print(f"    M1a V2 + M1b joint (30ep):      -0.095  (V2 hurts)")
    print(f"    M1a V3 + M1b joint (3ep mel):   {mean_effect:+.3f}  ← THIS")
    print(f"\n  acceptance (V3 + M1b > M1b alone): "
          f"{'PASS ✓' if mean_effect > 0.203 else 'FAIL ✗'}")

    out = {
        "source": str(src_path.relative_to(REPO_ROOT)),
        "weights": str(vp_path.relative_to(REPO_ROOT)),
        "voices": rows,
        "summary": {
            "mean_target_sim": mean_target,
            "mean_source_sim": mean_source,
            "mean_vc_effect": mean_effect,
            "cross_gender_mean_vc_effect": cross_gender_mean,
        },
        "ablation_comparison": {
            "m05_baseline": 0.001,
            "m1b_alone_top16_a0": 0.203,
            "m1a_v1_plus_m1b_30ep": -0.014,
            "m1a_v2_plus_m1b_30ep": -0.095,
            "m1a_v3_plus_m1b_3ep_mel": mean_effect,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
