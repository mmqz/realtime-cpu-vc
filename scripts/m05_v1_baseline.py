#!/usr/bin/env python3
"""M0.5 · Re-run v1 baseline on real VCTK 5-speaker validation set.

Replaces the legacy path3 baseline numbers
(mean VC_effect +0.053, mean target_sim 0.420, RTF 0.073) with numbers
measured on the M0.5 fixture set (real VCTK p225-p229 targets, p232 source).
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
OUT_JSON = REPO_ROOT / "data" / "m05_v1_baseline.json"
TARGET_SPEAKERS = ["p225", "p226", "p227", "p228", "p229"]


def _load_wav_24k(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != 24000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24000)
    return wav.astype(np.float32)


def _campplus_embed_file(ext, path: Path) -> np.ndarray:
    return _embed(ext, _load_wav_as_16k_mono(path))


def main() -> int:
    import torch
    torch.set_num_threads(2)

    TINYVC_ROOT = Path(os.environ.get(
        "TINYVC_ROOT", str(REPO_ROOT.parent / "repos" / "tinyvc")))
    if not TINYVC_ROOT.exists():
        print(f"ERROR: TinyVC repo not found at {TINYVC_ROOT}.", file=sys.stderr)
        return 1
    sys.path.insert(0, str(TINYVC_ROOT.parent))

    from vc_realtime.infer_v1 import V1Infer  # noqa: E402

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src_path = DATA_SOURCE / "source_001.wav"
    if not src_path.exists():
        print(f"ERROR: {src_path} missing", file=sys.stderr)
        return 1
    for i in range(5):
        if not (DATA_VOICES / f"voice_{i}.wav").exists():
            print(f"ERROR: {DATA_VOICES / f'voice_{i}.wav'} missing",
                  file=sys.stderr)
            return 1

    src_wav = _load_wav_24k(src_path)
    print(f"\n[M0.5 v1 baseline] source: {src_path.name} "
          f"({len(src_wav)/24000:.1f}s)")

    print("\n  Loading V1Infer (encoder.pt + decoder.pt + voices.pt)...")
    t0 = time.perf_counter()
    infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                    device="cpu", top_k=4, alpha=0.0)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    print("  Loading CAMPPlus (for speaker similarity scoring)...")
    campplus_path = _ensure_campplus_model()
    ext = _make_extractor(campplus_path)

    src_emb = _campplus_embed_file(ext, src_path)
    print(f"  source embedding norm={np.linalg.norm(src_emb):.3f}")

    print("  Computing target embeddings...")
    target_embs = []
    for i in range(5):
        emb = _campplus_embed_file(ext, DATA_VOICES / f"voice_{i}.wav")
        target_embs.append(emb)
        print(f"    voice_{i} target_emb norm={np.linalg.norm(emb):.3f}")

    print("\n  Running v1 kNN-VC inference per target voice:")
    rows = []
    for vid in range(5):
        t0 = time.perf_counter()
        out_wav = infer.process_audio(src_wav, sr=24000, voice_id=vid)
        infer_time = time.perf_counter() - t0
        out_dur = len(out_wav) / 24000
        rtf = infer_time / out_dur if out_dur > 0 else 0.0

        out_path = OUTPUT_DIR / f"vc_m05_voice_{vid}.wav"
        peak = max(np.max(np.abs(out_wav)), 1e-8)
        norm = out_wav * (10 ** (-3 / 20) / peak)
        sf.write(str(out_path),
                 (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)

        out_emb = _campplus_embed_file(ext, out_path)
        target_sim = _cosine(target_embs[vid], out_emb)
        source_sim = _cosine(src_emb, out_emb)
        vc_effect = target_sim - source_sim

        print(f"    voice_{vid}: target_sim={target_sim:.3f} "
              f"source_sim={source_sim:.3f} VC_effect={vc_effect:+.3f} "
              f"RTF={rtf:.3f} ({infer_time*1000:.0f}ms / {out_dur:.1f}s)")
        rows.append({
            "voice_id": vid, "speaker": TARGET_SPEAKERS[vid],
            "target_sim": float(target_sim), "source_sim": float(source_sim),
            "vc_effect": float(vc_effect), "rtf": float(rtf),
            "infer_time_ms": float(infer_time * 1000),
            "out_duration_s": float(out_dur),
        })

    mean_target = float(np.mean([r["target_sim"] for r in rows]))
    mean_source = float(np.mean([r["source_sim"] for r in rows]))
    mean_effect = float(np.mean([r["vc_effect"] for r in rows]))
    mean_rtf = float(np.mean([r["rtf"] for r in rows]))

    print("\n=== M0.5 v1 baseline summary (real VCTK speakers) ===")
    print(f"  mean target_sim:  {mean_target:.3f}  (path3 baseline 0.420)")
    print(f"  mean source_sim:  {mean_source:.3f}  (path3 baseline 0.367)")
    print(f"  mean VC_effect:   {mean_effect:+.3f}  (path3 baseline +0.053)")
    print(f"  mean RTF:          {mean_rtf:.3f}  (path3 baseline 0.073)")

    out = {
        "source": str(src_path.relative_to(REPO_ROOT)),
        "source_speaker": "p232",
        "source_text_id": "001",
        "voices": rows,
        "summary": {
            "mean_target_sim": mean_target, "mean_source_sim": mean_source,
            "mean_vc_effect": mean_effect, "mean_rtf": mean_rtf,
        },
        "baseline_comparison": {
            "path3_pitch_shifted_mean_target_sim": 0.420,
            "path3_pitch_shifted_mean_source_sim": 0.367,
            "path3_pitch_shifted_mean_vc_effect": 0.053,
            "path3_pitch_shifted_mean_rtf": 0.073,
            "delta_vc_effect": mean_effect - 0.053,
            "delta_target_sim": mean_target - 0.420,
        },
        "campplus_distinctness": json.load(
            open(DATA_VOICES / "campplus_distinctness.json"))
            if (DATA_VOICES / "campplus_distinctness.json").exists() else None,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
