#!/usr/bin/env python3
"""
SolH: kNN alpha parameter sweep.

alpha=0: pure target replacement (current default)
alpha=0.1: 10% source + 90% target
alpha=0.3: 30% source + 70% target
alpha=0.5: 50/50 blend
alpha=0.7: 70% source + 30% target

Pipeline (TinyVC baseline, NOT adapted HuBERT):
    source_audio (24 kHz)
        |  TinyVC Encoder (combined SSL + pitch ConvNeXt)  -> content [1, 768, T]
        |                                                  -> f0      [1, 1, T]
        |  TinyVC estimate_energy                          -> energy  [1, 1, L]
        v
    kNN-VC replace(content, target_voice_features, k=4, alpha=X)
        |
        v
    TinyVC DDSP Decoder (SourceNet + FilterNet) -> 24 kHz mono audio

Higher alpha => more source character preserved => potentially more
naturalness but less VC identity transfer. The sweep finds the sweet spot.

Notes
-----
- `module/utils/__init__.py` eagerly imports `f0_estimation`, which in turn
  imports `torchfcpe` and `pyworld`. We only use the ConvNeXt pitch estimator
  inside Encoder.infer(), never the dio/harvest/fcpe paths, so we stub those
  optional deps (same pattern as `vc_adapted_features_test.py`).
- `estimate_energy` expects the *waveform* (not the spectrogram) — we mirror
  the working pattern from `vc_adapted_features_test.py`.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa

# -----------------------------------------------------------------------------
# Optional-deps stubbing. `module/utils/__init__.py` imports `f0_estimation`,
# which imports `torchfcpe` and `pyworld`. We only use the ConvNeXt pitch
# estimator inside Encoder.infer(), never the dio/harvest/fcpe paths.
# -----------------------------------------------------------------------------
for _mod_name, _attrs in (
    ("torchfcpe", {"spawn_bundled_infer_model": lambda *a, **kw: None}),
    (
        "pyworld",
        {
            "dio": lambda *a, **kw: None,
            "stonemask": lambda *a, **kw: None,
            "harvest": lambda *a, **kw: None,
        },
    ),
):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        for _k, _v in _attrs.items():
            setattr(_stub, _k, _v)
        sys.modules[_mod_name] = _stub

# Make upstream TinyVC importable.
TINYVC_ROOT = Path(os.environ.get("TINYVC_ROOT", "/home/z/my-project/repos/tinyvc"))
if str(TINYVC_ROOT) not in sys.path:
    sys.path.insert(0, str(TINYVC_ROOT))
sys.path.insert(0, "src")  # local prototype helpers (harmless if unused)

from module.tinyvc.encoder import Encoder  # noqa: E402
from module.tinyvc.decoder import Decoder  # noqa: E402
from module.tinyvc.feature_retrieval import match_features  # noqa: E402
from module.utils.spectrogram import spectrogram  # noqa: E402
from module.utils.auto_padding import autopad_waveform  # noqa: E402
from module.utils.energy_estimation import estimate_energy  # noqa: E402

torch.set_num_threads(2)

SAMPLE_RATE = 24000
DEFAULT_NORM_DB = -3.0
OUTPUT_DIR = "/home/z/my-project/download"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_peak(wav: np.ndarray) -> np.ndarray:
    wav = wav.astype(np.float32)
    peak = max(float(np.max(np.abs(wav))), 1e-8)
    return wav * (10 ** (DEFAULT_NORM_DB / 20.0) / peak)


def encode_wav(wav_24k: np.ndarray, enc: Encoder):
    """Returns (content [1, 768, T], f0 [1, 1, T], energy [1, 1, L])."""
    wav_n = normalize_peak(wav_24k)
    wf = torch.from_numpy(wav_n.astype(np.float32)).unsqueeze(0)  # [1, L]
    wf = autopad_waveform(wf)
    spec = spectrogram(wf, 1920, 480)
    with torch.no_grad():
        content, f0 = enc.infer(spec)
    energy = estimate_energy(wf)  # [1, 1, L]
    return content, f0, energy


def main():
    import sherpa_onnx

    print("=" * 70)
    print("SolH: kNN alpha parameter sweep (TinyVC baseline)")
    print("  alpha=0.0 -> pure target (default)")
    print("  alpha=0.1/0.3/0.5/0.7 -> source/target blend ratios")
    print("=" * 70)

    # --- Load TinyVC models ----------------------------------------------
    print("\n[1/4] Loading TinyVC models ...")
    enc = Encoder()
    enc.load_state_dict(
        torch.load("models/encoder.pt", map_location="cpu")
    )
    enc.eval()
    dec = Decoder()
    dec.load_state_dict(
        torch.load("models/decoder.pt", map_location="cpu")
    )
    dec.eval()
    enc_params = sum(p.numel() for p in enc.parameters())
    dec_params = sum(p.numel() for p in dec.parameters())
    print(f"  Encoder = {enc_params/1e6:.2f}M params")
    print(f"  Decoder = {dec_params/1e6:.2f}M params")

    # --- Encode voice references -----------------------------------------
    print("\n[2/4] Encoding voice references ...")
    voices: dict[int, torch.Tensor] = {}
    for i in range(5):
        wav, sr = sf.read(f"data/voices/voice_{i}.wav")
        if sr != SAMPLE_RATE:
            wav = librosa.resample(
                wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
            )
        content, _, _ = encode_wav(wav, enc)
        voices[i] = content
        print(
            f"  voice_{i}: content={tuple(content.shape)}, "
            f"mean={content.mean():+.4f}, std={content.std():.4f}"
        )

    # --- CAMPPlus speaker embedding extractor ----------------------------
    print("\n[3/4] Loading CAMPPlus extractor ...")
    campplus_path = (
        "models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
    )
    if not os.path.exists(campplus_path):
        from huggingface_hub import hf_hub_download
        import shutil

        shutil.copy(
            hf_hub_download(
                repo_id="bitsydarel/campplus-onnx",
                filename="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
            ),
            campplus_path,
        )
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
    config.model = campplus_path
    config.num_threads = 1
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def get_emb(wav, sr=SAMPLE_RATE):
        if sr != 16000:
            wav = librosa.resample(
                wav.astype(np.float32), orig_sr=sr, target_sr=16000
            )
        stream = extractor.create_stream()
        stream.accept_waveform(16000, wav.tolist())
        stream.input_finished()
        return np.asarray(extractor.compute(stream), dtype=np.float32)

    def cosine(a, b):
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        )

    # --- Source + target embeddings --------------------------------------
    src, sr = sf.read("data/source/source_real_001.wav")
    src = src.astype(np.float32)[: sr * 5]  # 5 seconds
    if sr != SAMPLE_RATE:
        src = librosa.resample(src, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    src_emb = get_emb(src, sr)
    target_embs = [
        get_emb(sf.read(f"data/voices/voice_{i}.wav")[0]) for i in range(5)
    ]
    print(
        f"  source: data/source/source_real_001.wav "
        f"({len(src)/sr:.1f}s @ {sr}Hz, emb dim={src_emb.shape[0]})"
    )

    # --- Encode source content + f0 + energy ----------------------------
    src_content, src_f0, src_energy = encode_wav(src, enc)
    print(
        f"  source features: content={tuple(src_content.shape)}, "
        f"f0={tuple(src_f0.shape)}, energy={tuple(src_energy.shape)}"
    )

    # --- Alpha sweep -----------------------------------------------------
    ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7]
    print(f"\n[4/4] kNN Alpha Sweep ({ALPHAS})")
    print("-" * 70)

    summary: list[tuple[float, float, float, float]] = []

    # Content + f0 are at 50 Hz (frame rate). Energy is at 24 kHz (sample rate).
    # The decoder's SourceNet does max_pool1d(energy, frame_size=480, stride=480)
    # to bring energy down to frame rate. So we must NOT slice energy by the
    # frame count; we slice it to match content frames * frame_size instead.
    T_content = src_content.shape[-1]  # 250 frames for 5 sec
    target_samples = T_content * 480  # 120000 samples
    energy_full = src_energy
    if energy_full.shape[-1] > target_samples:
        energy_full = energy_full[:, :, :target_samples]
    elif energy_full.shape[-1] < target_samples:
        # Pad energy to match content frames
        pad = target_samples - energy_full.shape[-1]
        energy_full = torch.cat(
            [energy_full, torch.zeros(1, 1, pad)], dim=-1
        )
    print(
        f"  energy aligned: {tuple(energy_full.shape)} "
        f"({T_content} frames x 480 = {target_samples} samples)"
    )

    for alpha in ALPHAS:
        results = []
        for vid in range(5):
            target = voices[vid]
            content = src_content
            f0 = src_f0
            energy = energy_full

            content_replaced = match_features(content, target, k=4, alpha=alpha)
            with torch.no_grad():
                out = dec.infer(content_replaced, f0, energy)
            out = out.squeeze().numpy()

            out_emb = get_emb(out, sr)
            t_sim = cosine(out_emb, target_embs[vid])
            s_sim = cosine(out_emb, src_emb)
            results.append((t_sim, s_sim))

        avg_t = float(np.mean([r[0] for r in results]))
        avg_s = float(np.mean([r[1] for r in results]))
        avg_eff = avg_t - avg_s
        summary.append((alpha, avg_t, avg_s, avg_eff))
        print(
            f"alpha={alpha:.1f}: target={avg_t:.3f}, source={avg_s:.3f}, "
            f"VC_effect={avg_eff:+.3f}"
        )
        for vid, (t_sim, s_sim) in enumerate(results):
            print(
                f"  voice_{vid}: target={t_sim:.3f}, source={s_sim:.3f}, "
                f"VC_effect={t_sim-s_sim:+.3f}"
            )

        # Save sample outputs for selected alphas
        if alpha in [0.0, 0.3, 0.5]:
            for vid in range(5):
                target = voices[vid]
                content = src_content
                f0 = src_f0
                energy = energy_full
                content_replaced = match_features(
                    content, target, k=4, alpha=alpha
                )
                with torch.no_grad():
                    out = dec.infer(content_replaced, f0, energy)
                sf.write(
                    f"{OUTPUT_DIR}/vc_alpha{alpha}_voice_{vid}.wav",
                    out.squeeze().numpy(),
                    sr,
                )

    # --- Summary ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: VC_effect (target_sim - source_sim) by alpha")
    print("=" * 70)
    print(f"{'alpha':>6} | {'target':>7} | {'source':>7} | {'VC_effect':>10}")
    print("-" * 40)
    best_alpha, best_eff = None, -1e9
    for alpha, avg_t, avg_s, avg_eff in summary:
        print(
            f"{alpha:>6.1f} | {avg_t:>7.3f} | {avg_s:>7.3f} | {avg_eff:>+10.3f}"
        )
        if avg_eff > best_eff:
            best_eff = avg_eff
            best_alpha = alpha
    print("-" * 40)
    print(
        f"BEST alpha={best_alpha:.1f}  (VC_effect={best_eff:+.3f})  "
        f"target={dict((a, t) for a, t, _, _ in summary)[best_alpha]:.3f}, "
        f"source={dict((a, s) for a, _, s, _ in summary)[best_alpha]:.3f}"
    )
    print(f"\n[done] Sample outputs in {OUTPUT_DIR}/vc_alpha*_voice_*.wav")


if __name__ == "__main__":
    main()
