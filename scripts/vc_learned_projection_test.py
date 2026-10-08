#!/usr/bin/env python3
"""
SolE: Learned 768x768 affine projection.

Train a `nn.Linear(768, 768)` to map HuBERT-base layer-4 features →
TinyVC's distilled ConvNeXt features, then run VC through TinyVC's
DDSP decoder.

Method
------
- For each training wav: run HuBERT (layer 4) AND TinyVC encoder on the
  SAME audio → paired features at 50 Hz.
- Train proj: minimize MSE between `proj(hubert)` and `tinyvc`.
- Initialise proj.weight = identity, proj.bias = 0 (so the untrained
  baseline is the HuBERT-as-TinyVC baseline, comparable to Path 2).
- 200 epochs of minibatch SGD over 50-frame chunks; Adam, lr=1e-3.

Pipeline at test time
--------------------
  source (24 kHz)
    │ HuBERT-base (layer 4, 768-d @ 50Hz) → proj(768→768)
    │ TinyVC PitchEstimator → f0
    │ TinyVC estimate_energy → energy
    v
  kNN-VC replace(content, target_voice_proj_features, k=4, alpha=0)
    v
  TinyVC DDSP Decoder (SourceNet + FilterNet)
    v
  output (24 kHz)

Notes
-----
- `module/utils/__init__.py` imports `f0_estimation`, which imports
  `torchfcpe` and `pyworld`. We only use the ConvNeXt pitch estimator
  inside Encoder.infer(), never the dio/harvest/fcpe paths, so we stub
  those optional deps (same pattern as `vc_distilhubert_test.py`).
- `module.tinyvc.__init__` re-exports Encoder, Decoder, match_features.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa

# -----------------------------------------------------------------------------
# Optional-deps stubbing. `module/utils/__init__.py` imports `f0_estimation`,
# which imports `torchfcpe` and `pyworld`. Those are only needed for the
# fcpe/dio/harvest algorithms. We never call those here.
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
TINYVC_ROOT = Path(os.environ.get("TINYVC_ROOT", "../repos/tinyvc"))
if str(TINYVC_ROOT) not in sys.path:
    sys.path.insert(0, str(TINYVC_ROOT))
sys.path.insert(0, "src")

from transformers import HubertModel, Wav2Vec2FeatureExtractor  # noqa: E402

from module.tinyvc import Encoder as TinyVCEncoder  # noqa: E402
from module.tinyvc import Decoder as TinyVCDecoder  # noqa: E402
from module.tinyvc import match_features as tinyvc_match_features  # noqa: E402
from module.utils.spectrogram import spectrogram as tinyvc_spectrogram  # noqa: E402
from module.utils.auto_padding import autopad_waveform as tinyvc_autopad  # noqa: E402
from module.utils.energy_estimation import estimate_energy as tinyvc_estimate_energy  # noqa: E402

torch.set_num_threads(2)

SAMPLE_RATE = 24000
HUBERT_LAYER = 4  # TinyVC's ConvNeXt was distilled from WavLM layer 4
DEFAULT_NORM_DB = -3.0
OUTPUT_DIR = "./download"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _normalize_peak(wav: np.ndarray) -> np.ndarray:
    wav = wav.astype(np.float32)
    peak = max(float(np.max(np.abs(wav))), 1e-8)
    return wav * (10 ** (DEFAULT_NORM_DB / 20.0) / peak)


def main() -> int:
    # --- Load models ------------------------------------------------------
    print("Loading models...")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()
    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    tinyvc_enc = TinyVCEncoder()
    tinyvc_enc.load_state_dict(
        torch.load("models/encoder.pt", map_location="cpu")
    )
    tinyvc_enc.eval()
    decoder = TinyVCDecoder()
    decoder.load_state_dict(
        torch.load("models/decoder.pt", map_location="cpu")
    )
    decoder.eval()

    # --- Step 1: Collect paired (HuBERT, TinyVC) features ----------------
    print("\nCollecting paired features (HuBERT <-> TinyVC)...")

    def encode_pair(wav: np.ndarray):
        """Run both encoders on the same wav; return (h_feat, t_feat)."""
        wav = wav.astype(np.float32)
        # HuBERT path (needs 16 kHz mono)
        wav_16k = librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=16000)
        inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            h_out = hubert(**inputs, output_hidden_states=True)
            h_feat = h_out.hidden_states[HUBERT_LAYER].transpose(1, 2)  # [1,768,T]
        # TinyVC path (24 kHz, peak-normalised). autopad/spectrogram expect
        # 2D [BatchSize, Length], NOT 3D [B, 1, L].
        wav_n = _normalize_peak(wav)
        wav_t = tinyvc_autopad(torch.from_numpy(wav_n).unsqueeze(0))
        spec = tinyvc_spectrogram(wav_t, 1920, 480)
        with torch.no_grad():
            t_feat, _ = tinyvc_enc.infer(spec)
        return h_feat, t_feat

    hubert_feats: list[torch.Tensor] = []
    tinyvc_feats: list[torch.Tensor] = []
    for i in range(5):
        wav, sr = sf.read(f"data/voices/voice_{i}.wav")
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        h_feat, t_feat = encode_pair(wav)
        min_T = min(h_feat.shape[-1], t_feat.shape[-1])
        hubert_feats.append(h_feat[:, :, :min_T])
        tinyvc_feats.append(t_feat[:, :, :min_T])
        print(
            f"  voice_{i}: HuBERT {tuple(h_feat.shape)}, TinyVC {tuple(t_feat.shape)}, "
            f"paired_T={min_T}"
        )

    # Also include source fixtures so the projection sees source-side audio.
    for i in range(1, 11):
        p = f"data/source/source_{i:03d}.wav"
        if not os.path.exists(p):
            continue
        wav, sr = sf.read(p)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        h_feat, t_feat = encode_pair(wav)
        min_T = min(h_feat.shape[-1], t_feat.shape[-1])
        hubert_feats.append(h_feat[:, :, :min_T])
        tinyvc_feats.append(t_feat[:, :, :min_T])
    print(f"  total paired samples: {len(hubert_feats)}")

    # --- Step 2: Train 768x768 affine projection -------------------------
    print("\nTraining 768x768 affine projection (200 epochs)...")
    proj = torch.nn.Linear(768, 768, bias=True)
    torch.nn.init.eye_(proj.weight)  # start as identity
    torch.nn.init.zeros_(proj.bias)
    optimizer = torch.optim.Adam(proj.parameters(), lr=1e-3)

    chunk_size = 50  # frames per minibatch (CPU-friendly)
    for epoch in range(200):
        total_loss = 0.0
        n_steps = 0
        # Iterate over paired samples in fixed order (small dataset).
        for h, t in zip(hubert_feats, tinyvc_feats):
            T = h.shape[-1]
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                h_chunk = h[:, :, start:end]  # [1,768,chunk]
                t_chunk = t[:, :, start:end]
                # Linear acts on last dim → transpose [1,chunk,768] → [1,chunk,768]
                h_proj = proj(h_chunk.transpose(1, 2)).transpose(1, 2)
                loss = F.mse_loss(h_proj, t_chunk)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_steps += 1
        if (epoch + 1) % 50 == 0:
            print(
                f"  Epoch {epoch + 1}: loss={total_loss / max(n_steps, 1):.6f}"
            )

    torch.save(proj.state_dict(), "models/learned_projection_768x768.pt")
    print("Saved projection to models/learned_projection_768x768.pt")

    # --- Step 3: VC test with learned projection -------------------------
    print("\n=== VC Test with Learned Projection ===")
    import sherpa_onnx

    campplus_path = "models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
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
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # Source + target embeddings
    src, sr = sf.read("data/source/source_real_001.wav")
    src = src.astype(np.float32)[: sr * 5]  # 5 seconds
    src_emb = get_emb(src, sr)
    target_embs = [get_emb(sf.read(f"data/voices/voice_{i}.wav")[0]) for i in range(5)]
    print(
        f"  Source: source_real_001.wav ({len(src) / sr:.1f}s @ {sr}Hz, "
        f"emb_dim={src_emb.shape[0]})"
    )

    # Build voice index with PROJECTED HuBERT features (mirrors test-time path)
    print("\n  Building projected voice index...")
    proj_voices: dict[int, torch.Tensor] = {}
    for i in range(5):
        wav, vsr = sf.read(f"data/voices/voice_{i}.wav")
        if vsr != SAMPLE_RATE:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=vsr, target_sr=SAMPLE_RATE)
        wav_16k = librosa.resample(wav.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=16000)
        inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            h_out = hubert(**inputs, output_hidden_states=True)
            h_feat = h_out.hidden_states[HUBERT_LAYER].transpose(1, 2)
            projected = proj(h_feat.transpose(1, 2)).transpose(1, 2)
        proj_voices[i] = projected
        print(
            f"    voice_{i}: projected {tuple(projected.shape)}, "
            f"mean={projected.mean():+.4f}, std={projected.std():.4f}"
        )

    print()
    results = []
    for vid in range(5):
        try:
            # Encode source with HuBERT + project
            wav_16k = librosa.resample(src, orig_sr=sr, target_sr=16000)
            inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                h_out = hubert(**inputs, output_hidden_states=True)
                h_feat = h_out.hidden_states[HUBERT_LAYER].transpose(1, 2)
                content = proj(h_feat.transpose(1, 2)).transpose(1, 2)

            # F0 from TinyVC (on the same source audio). autopad/spectrogram
            # expect 2D [BatchSize, Length], not [B, 1, L].
            wav_n = _normalize_peak(src)
            wav_t = tinyvc_autopad(torch.from_numpy(wav_n).unsqueeze(0))
            spec = tinyvc_spectrogram(wav_t, 1920, 480)
            with torch.no_grad():
                _, f0 = tinyvc_enc.infer(spec)

            # kNN against the projected target voice index
            target = proj_voices[vid]
            min_T = min(content.shape[-1], target.shape[-1], f0.shape[-1])
            content = content[:, :, :min_T]
            target = target[:, :, :min_T]
            f0 = f0[:, :, :min_T]
            content_replaced = tinyvc_match_features(
                content, target, k=4, alpha=0.0
            )

            # Energy at sample rate (decoder SourceNet max-pools to frame rate).
            # train_projection.py + vc_adapted_features_test.py use the same
            # convention: pass the 2-D padded waveform to estimate_energy.
            energy = tinyvc_estimate_energy(wav_t)

            # Decode
            with torch.no_grad():
                out = decoder.infer(content_replaced, f0, energy)
            out = out.squeeze().numpy()

            out_path = f"{OUTPUT_DIR}/vc_learned_proj_voice_{vid}.wav"
            sf.write(out_path, out, SAMPLE_RATE)
            out_emb = get_emb(out, SAMPLE_RATE)
            t_sim = cosine(out_emb, target_embs[vid])
            s_sim = cosine(out_emb, src_emb)
            results.append(
                {
                    "voice_id": vid,
                    "target_sim": t_sim,
                    "source_sim": s_sim,
                    "vc_effect": t_sim - s_sim,
                }
            )
            print(
                f"  voice_{vid}: target={t_sim:.3f}, source={s_sim:.3f}, "
                f"VC_effect={t_sim - s_sim:+.3f}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  voice_{vid}: FAILED - {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    if results:
        avg_t = float(np.mean([r["target_sim"] for r in results]))
        avg_s = float(np.mean([r["source_sim"] for r in results]))
        print(
            f"\n  AVG: target={avg_t:.3f}, source={avg_s:.3f}, "
            f"VC_effect={avg_t - avg_s:+.3f}"
        )

    print(f"\n[done] Outputs in {OUTPUT_DIR}/vc_learned_proj_voice_*.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
