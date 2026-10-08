#!/usr/bin/env python3
"""M1b · Multi-source × multi-voice evaluation.

Tests whether different source audios (source_001-005, all from p232
reading different VCTK prompts) give better VC_effect for the laggard
voices (voice_3 p228 +0.005, voice_4 p229 +0.125 in M1b baseline).

Hypothesis: the laggard voices might benefit from a source audio whose
phoneme content overlaps better with the target voice's reference audio.
VCTK prompts 002 (long sentence) and 003 (snow peas / blue cheese) have
more diverse phonemes than prompt 001 (short "Please call Stella").

Best config from Round 9: top_k=1, alpha=0.0 (matches baseline within
noise). Use that to maximize speaker transfer.

Output:
- per-(source, voice) VC_effect matrix [5 sources × 5 voices]
- best source per voice
- best mean across all 5 sources

Acceptance: best (source, voice) matrix entry > +0.131 (M1b baseline mean).
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
OUT_JSON = REPO_ROOT / "data" / "m1b_multi_source.json"
TARGET_SPEAKERS = ["p225", "p226", "p227", "p228", "p229"]

BEST_TOP_K = 1
BEST_ALPHA = 0.0


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

    # Find available sources
    src_paths = sorted(DATA_SOURCE.glob("source_*.wav"))
    if len(src_paths) < 1:
        print(f"ERROR: no source_*.wav in {DATA_SOURCE}", file=sys.stderr)
        return 1
    print(f"\n[M1b multi-source] {len(src_paths)} sources × 5 voices "
          f"(top_k={BEST_TOP_K}, alpha={BEST_ALPHA})")

    print("  Loading V1Infer...")
    infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                    device="cpu", top_k=BEST_TOP_K, alpha=BEST_ALPHA)

    print("  Building F0 quantile mapper...")
    mapper = F0QuantileMapper.from_voices_dir(str(DATA_VOICES), sr=24000)

    print("  Loading CAMPPlus...")
    campplus_path = _ensure_campplus_model()
    ext = _make_extractor(campplus_path)

    # Compute target embeddings once (independent of source)
    target_embs = []
    for i in range(5):
        emb = _embed(ext, _load_wav_as_16k_mono(DATA_VOICES / f"voice_{i}.wav"))
        target_embs.append(emb)

    # For each source × each voice: encode → kNN replace → F0 map → decode
    matrix = np.zeros((len(src_paths), 5), dtype=np.float32)
    t_start = time.perf_counter()
    for s_idx, src_path in enumerate(src_paths):
        src_wav = _load_wav_24k(src_path)
        # Build source-specific F0 table
        mapper.build_source_table_from_audio(src_wav, sr=24000)
        # Source embedding
        src_emb = _embed(ext, _load_wav_as_16k_mono(src_path))
        # Encode source once
        wav_pre = infer._preprocess(src_wav, 24000)
        content, f0, energy = infer.encode(wav_pre)
        f0_track = f0.squeeze()
        print(f"\n  source {s_idx+1}/{len(src_paths)}: {src_path.name} "
              f"({len(src_wav)/24000:.1f}s)")
        for vid in range(5):
            mapper.select_voice(vid)
            mapped_f0 = mapper.map_f0(f0_track).reshape(1, 1, -1).astype(np.float32)
            content_replaced = infer.knn_replace(
                content, voice_id=vid,
                top_k=BEST_TOP_K, alpha=BEST_ALPHA)
            out_wav = infer.decode(content_replaced, mapped_f0, energy)
            # Save best-config outputs (only first source as ablation reference)
            if s_idx == 0:
                out_path = OUTPUT_DIR / f"vc_m1b_multi_voice_{vid}_src{s_idx+1}.wav"
                peak = max(np.max(np.abs(out_wav)), 1e-8)
                norm = out_wav * (10 ** (-3 / 20) / peak)
                sf.write(str(out_path),
                         (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)
            # Compute CAMPPlus (inline, write to temp file)
            tmp = OUTPUT_DIR / f"_tmp_ms_{s_idx}_{vid}.wav"
            peak = max(np.max(np.abs(out_wav)), 1e-8)
            norm = out_wav * (10 ** (-3 / 20) / peak)
            sf.write(str(tmp),
                     (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)
            out_emb = _embed(ext, _load_wav_as_16k_mono(tmp))
            tmp.unlink(missing_ok=True)
            target_sim = _cosine(target_embs[vid], out_emb)
            source_sim = _cosine(src_emb, out_emb)
            vc_effect = target_sim - source_sim
            matrix[s_idx, vid] = vc_effect
            print(f"    voice_{vid} ({TARGET_SPEAKERS[vid]}): "
                  f"VC_effect={vc_effect:+.3f} "
                  f"target_sim={target_sim:.3f} source_sim={source_sim:.3f}")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Sweep completed in {elapsed:.0f}s")

    # Summary
    print("\n=== M1b multi-source × voice matrix (VC_effect) ===")
    print(f"           " + "  ".join(f"v{i} ({spk})" for i, spk in enumerate(TARGET_SPEAKERS)))
    for s_idx, src_path in enumerate(src_paths):
        row_str = "  ".join(f"{matrix[s_idx, v]:+7.3f}" for v in range(5))
        print(f"  {src_path.stem[-4:]}:  {row_str}  mean={matrix[s_idx].mean():+.3f}")
    print(f"           " + "  ".join(f"mean={matrix[:, v].mean():+7.3f}" for v in range(5)))
    print(f"\n  Overall mean: {matrix.mean():+.3f}")
    print(f"  Best single cell: source={src_paths[np.unravel_index(matrix.argmax(), matrix.shape)[0]].name} "
          f"voice_{np.unravel_index(matrix.argmax(), matrix.shape)[1]} "
          f"= {matrix.max():+.3f}")

    # Best source per voice
    print("\n  Best source per voice:")
    for v in range(5):
        best_src_idx = int(np.argmax(matrix[:, v]))
        print(f"    voice_{v} ({TARGET_SPEAKERS[v]}): "
              f"best src = {src_paths[best_src_idx].name} "
              f"VC_effect={matrix[best_src_idx, v]:+.3f}")

    # M1b baseline (source_001 mean) vs multi-source mean
    src1_mean = float(matrix[0].mean())
    multi_mean = float(matrix.mean())
    print(f"\n  source_001 mean (M1b baseline): {src1_mean:+.3f}")
    print(f"  multi-source mean:              {multi_mean:+.3f}")
    print(f"  delta (multi - src1):            {multi_mean - src1_mean:+.3f}")
    print(f"\n  acceptance (delta > 0): "
          f"{'PASS ✓' if multi_mean - src1_mean > 0 else 'FAIL ✗'}")

    out = {
        "config": {"top_k": BEST_TOP_K, "alpha": BEST_ALPHA, "n_sources": len(src_paths)},
        "sources": [str(p.relative_to(REPO_ROOT)) for p in src_paths],
        "matrix": matrix.tolist(),
        "voices": TARGET_SPEAKERS,
        "summary": {
            "src1_mean": src1_mean,
            "multi_mean": multi_mean,
            "delta_multi_vs_src1": multi_mean - src1_mean,
            "best_cell_value": float(matrix.max()),
            "best_cell_pos": [int(np.unravel_index(matrix.argmax(), matrix.shape)[0]),
                              int(np.unravel_index(matrix.argmax(), matrix.shape)[1])],
        },
        "baseline_comparison": {
            "m1b_baseline_delta": 0.131,
            "src1_mean_delta_vs_m05": src1_mean - 0.001,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
