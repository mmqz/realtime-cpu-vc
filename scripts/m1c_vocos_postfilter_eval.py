#!/usr/bin/env python3
"""M1c · Vocos as DDSP postfilter.

Tests the hypothesis: Vocos's ISTFT-head reconstruction produces cleaner
audio than DDSP's raw additive sine synthesis, so running Vocos as a
postfilter on the M1b output mel should improve CAMPPlus speaker
similarity by reducing DDSP artifacts (metallic/buzzy texture from the
harmonic synth that may bias CAMPPlus toward source).

Pipeline:
  source_audio
    → TinyVC encoder → content [1,768,T] + f0 [1,1,T] + energy
    → kNN replace (alpha=0)  [M0.5 baseline]
    → F0 quantile map        [M1b]
    → TinyVC DDSP decoder    → DDSP audio
    → Vocos MelSpectrogram extractor (24kHz, n_fft=1024, hop=256, n_mels=100)
    → Vocos ONNX             → Vocos-refined audio
    → CAMPPlus scoring

If Vocos postfilter improves CAMPPlus by ≥ +0.02 over M1b alone
(current +0.131), it's worth doing the proper M1c (192→100 projection
+ Vocos as primary vocoder). If it doesn't, the DDSP artifacts aren't
the bottleneck and we should pivot to M2 latency work.
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
OUT_JSON = REPO_ROOT / "data" / "m1c_vocos_postfilter_eval.json"
TARGET_SPEAKERS = ["p225", "p226", "p227", "p228", "p229"]


def _load_wav_24k(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != 24000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24000)
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Vocos mel-spec extractor + ONNX session
# ---------------------------------------------------------------------------
class VocosPostfilter:
    """Compute 100-mel at 93.75 Hz from 24kHz audio, run Vocos ONNX."""

    def __init__(self, vocos_onnx_path: Path):
        import onnxruntime as ort
        from vocos.feature_extractors import MelSpectrogramFeatures
        self.mel_extractor = MelSpectrogramFeatures(
            sample_rate=24000, n_fft=1024, hop_length=256,
            n_mels=100, padding="center",
        )
        self.mel_extractor.eval()
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(vocos_onnx_path), sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def refine(self, audio_24k: np.ndarray) -> np.ndarray:
        """audio_24k [N] → mel [1,100,T_93.75Hz] → waveform [N']."""
        with torch.no_grad():
            x = torch.from_numpy(audio_24k).float().unsqueeze(0)
            mel = self.mel_extractor(x).numpy()  # [1, 100, T]
        # Run Vocos ONNX
        wave = self.session.run(None, {self.input_name: mel.astype(np.float32)})[0]
        # wave shape: [B, T_samples]
        return wave[0]  # [T_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    TINYVC_ROOT = Path(os.environ.get(
        "TINYVC_ROOT", str(REPO_ROOT.parent / "repos" / "tinyvc")))
    if not TINYVC_ROOT.exists():
        print(f"ERROR: TinyVC repo not found at {TINYVC_ROOT}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(TINYVC_ROOT.parent))

    from vc_realtime.infer_v1 import V1Infer
    from vc_realtime.pitch import F0QuantileMapper

    vocos_path = REPO_ROOT / "models" / "vocos.onnx"
    if not vocos_path.exists():
        print(f"ERROR: {vocos_path} missing; run scripts/export_vocos_onnx.py",
              file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src_path = DATA_SOURCE / "source_001.wav"
    src_wav = _load_wav_24k(src_path)
    print(f"\n[M1c Vocos postfilter] source: {src_path.name} "
          f"({len(src_wav)/24000:.1f}s)")

    print("  Loading V1Infer...")
    t0 = time.perf_counter()
    infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                    device="cpu", top_k=4, alpha=0.0)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    print("  Building F0 quantile mapper...")
    mapper = F0QuantileMapper.from_voices_dir(str(DATA_VOICES), sr=24000)
    mapper.build_source_table_from_audio(src_wav, sr=24000)

    print(f"  Loading Vocos ONNX from {vocos_path}...")
    vocos = VocosPostfilter(vocos_path)

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

    print("\n  Running M1b + Vocos postfilter per voice:")
    rows = []
    for vid in range(5):
        mapper.select_voice(vid)
        mapped_f0 = mapper.map_f0(f0_track)
        mapped_f0_dec = mapped_f0.reshape(1, 1, -1).astype(np.float32)
        content_replaced = infer.knn_replace(content, voice_id=vid, top_k=4, alpha=0.0)

        # Stage 1: DDSP decode (M1b path)
        t1 = time.perf_counter()
        ddsp_audio = infer.decode(content_replaced, mapped_f0_dec, energy)
        ddsp_time = time.perf_counter() - t1

        # Stage 2: Vocos postfilter on the DDSP audio
        t2 = time.perf_counter()
        vocos_audio = vocos.refine(ddsp_audio.squeeze())
        vocos_time = time.perf_counter() - t2

        # The Vocos output may be a slightly different length than the input
        # (because center padding + ISTFT); truncate to the smaller length
        n_target = min(len(ddsp_audio.squeeze()), len(vocos_audio))
        vocos_audio = vocos_audio[:n_target].astype(np.float32)

        out_path = OUTPUT_DIR / f"vc_m1c_vocos_voice_{vid}.wav"
        peak = max(np.max(np.abs(vocos_audio)), 1e-8)
        norm = vocos_audio * (10 ** (-3 / 20) / peak)
        sf.write(str(out_path),
                 (np.clip(norm, -1, 1) * 32767).astype(np.int16), 24000)

        out_emb = _embed(ext, _load_wav_as_16k_mono(out_path))
        target_sim = _cosine(target_embs[vid], out_emb)
        source_sim = _cosine(src_emb, out_emb)
        vc_effect = target_sim - source_sim

        med_out = float(np.median(mapped_f0[mapped_f0 > 0])) if (mapped_f0 > 0).any() else 0

        print(f"    voice_{vid}: target_sim={target_sim:.3f} "
              f"source_sim={source_sim:.3f} VC_effect={vc_effect:+.3f} "
              f"F0→{med_out:.0f}Hz "
              f"(DDSP {ddsp_time*1000:.0f}ms + Vocos {vocos_time*1000:.0f}ms)")
        rows.append({
            "voice_id": vid, "speaker": TARGET_SPEAKERS[vid],
            "target_sim": float(target_sim), "source_sim": float(source_sim),
            "vc_effect": float(vc_effect),
            "f0_out_median_hz": med_out,
            "ddsp_time_ms": float(ddsp_time * 1000),
            "vocos_time_ms": float(vocos_time * 1000),
        })

    mean_target = float(np.mean([r["target_sim"] for r in rows]))
    mean_source = float(np.mean([r["source_sim"] for r in rows]))
    mean_effect = float(np.mean([r["vc_effect"] for r in rows]))
    cross_gender = [r for r in rows if r["voice_id"] in (3, 4)]
    cross_gender_mean = float(np.mean([r["vc_effect"] for r in cross_gender]))

    print("\n=== M1c Vocos postfilter summary ===")
    print(f"  mean VC_effect:   {mean_effect:+.3f}  (M0.5 baseline +0.001)")
    print(f"  delta vs M0.5:    {mean_effect - 0.001:+.3f}")
    print(f"  cross-gender mean: {cross_gender_mean:+.3f}  (M0.5 -0.114)")
    print(f"\n  ablation:")
    print(f"    M0.5 baseline:                delta +0.001")
    print(f"    M1b alone (no Vocos):         delta +0.131  ← previous best")
    print(f"    M1b + Vocos postfilter:       delta {mean_effect - 0.001:+.3f}  ← THIS")
    print(f"    delta M1c - M1b:              {mean_effect - 0.131:+.3f}")
    print(f"\n  acceptance (delta M1c - M1b ≥ +0.02): "
          f"{'PASS' if mean_effect - 0.131 >= 0.02 else 'FAIL'}")

    out = {
        "source": str(src_path.relative_to(REPO_ROOT)),
        "voices": rows,
        "summary": {
            "mean_target_sim": mean_target,
            "mean_source_sim": mean_source,
            "mean_vc_effect": mean_effect,
            "delta_vs_m05": mean_effect - 0.001,
            "delta_vs_m1b": mean_effect - 0.131,
            "cross_gender_mean_vc_effect": cross_gender_mean,
        },
        "ablation": {
            "m05_baseline": 0.001,
            "m1b_alone": 0.131,
            "m1c_vocos_postfilter": mean_effect - 0.001,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Written → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
