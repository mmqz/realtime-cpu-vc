#!/usr/bin/env python3
"""
SolG: Multi-layer HuBERT feature fusion.

Different HuBERT layers capture different information: lower=phonetic,
higher=semantic. Using only layer 4 may miss useful content features. This
script tests concatenating layers 4+6+8 (and other combinations) → kNN-VC
replace on the concatenated representation → take the first 768 channels
(L4 of the matched neighbors) for the TinyVC DDSP decoder.

Pipeline per layer config:
    HuBERT-base (selected layers, concat → [1, 768*N, T] @ 50Hz)
        → moment-match adapt (std=0.175, matching SolC's tinyvc_std target)
        → kNN-VC replace against voice references (also adapted)
        → slice first 768 channels (L4 of matched neighbors)
        → TinyVC DDSP Decoder (SourceNet + FilterNet)
        → 24 kHz mono audio

Notes
-----
- The task spec's `sys.modules['torchfcpe'] = type(sys)('torchfcpe')` stub does
  not work because `module/utils/__init__.py` does
  `from torchfcpe import spawn_bundled_infer_model`. We mirror the working stub
  pattern from `vc_adapted_features_test.py` / `vc_distilhubert_test.py`.
- `Decoder.infer(content, f0, energy)` requires 3 positional args; we pass
  TinyVC's `estimate_energy` output for the third one.
- CAMPPlus (3D-Speaker) speaker embedding extractor is used for similarity
  scoring (target_sim / source_sim / VC_effect = target_sim - source_sim).
"""
import sys, os, types, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa

# -----------------------------------------------------------------------------
# Optional-deps stubbing. `module/utils/__init__.py` imports `f0_estimation`,
# which imports `torchfcpe` and `pyworld`. We only use the ConvNeXt pitch
# estimator inside Encoder.infer(), never the dio/harvest/fcpe paths.
# (The spec's `type(sys)('torchfcpe')` form alone is insufficient — those
#  modules need the named attributes below to import cleanly.)
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

# Make upstream TinyVC importable + local `src` importable.
TINYVC_ROOT = Path(os.environ.get("TINYVC_ROOT", "../repos/tinyvc"))
if str(TINYVC_ROOT) not in sys.path:
    sys.path.insert(0, str(TINYVC_ROOT))
sys.path.insert(0, "src")

from transformers import HubertModel, Wav2Vec2FeatureExtractor  # noqa: E402

from module.tinyvc.encoder import Encoder  # noqa: E402
from module.tinyvc.decoder import Decoder  # noqa: E402
from module.tinyvc.feature_retrieval import match_features  # noqa: E402
from module.utils.spectrogram import spectrogram  # noqa: E402
from module.utils.auto_padding import autopad_waveform  # noqa: E402
from module.utils.energy_estimation import estimate_energy  # noqa: E402

torch.set_num_threads(2)

SAMPLE_RATE = 24000
TARGET_STD = 0.175  # ≈ TinyVC distilled ConvNeXt feature std (SolC baseline)

print("Loading models...")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960"); hubert.eval()
fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
tinyvc_enc = Encoder()
tinyvc_enc.load_state_dict(torch.load('models/encoder.pt', map_location='cpu'))
tinyvc_enc.eval()
decoder = Decoder()
decoder.load_state_dict(torch.load('models/decoder.pt', map_location='cpu'))
decoder.eval()

# Configurations to test
LAYER_CONFIGS = [
    ([4], "L4 only"),
    ([6], "L6 only"),
    ([8], "L8 only"),
    ([4, 6], "L4+L6"),
    ([4, 6, 8], "L4+L6+L8"),
    ([4, 8], "L4+L8"),
]


def encode_hubert_layers(wav, sr, layers):
    """Extract specified HuBERT layers, concatenate, return [1, 768*len(layers), T]."""
    if sr != 16000:
        wav_16k = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    else:
        wav_16k = wav.astype(np.float32)
    inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = hubert(**inputs, output_hidden_states=True)
        feats = [outputs.hidden_states[l].transpose(1, 2) for l in layers]  # each [1, 768, T]
        concat = torch.cat(feats, dim=1)  # [1, 768*len(layers), T]
    return concat


def encode_tinyvc(wav, sr):
    """TinyVC encoder for f0 + spectrogram + energy.

    Returns (ssl, f0, spec, energy). Note: `autopad_waveform` expects a
    2D `[Batch, Length]` tensor (not 3D), and `estimate_energy` expects the
    waveform (not the spectrogram). Energy stays at waveform rate; the
    TinyVC decoder max-pools it internally to frame rate.
    """
    wav = wav.astype(np.float32)
    peak = max(np.max(np.abs(wav)), 1e-8)
    wav = wav * (10**(-3/20) / peak)
    wav_t = autopad_waveform(torch.from_numpy(wav).unsqueeze(0))  # [1, L_padded]
    spec = spectrogram(wav_t, 1920, 480)
    with torch.no_grad():
        ssl, f0 = tinyvc_enc.infer(spec)
    energy = estimate_energy(wav_t)  # [1, 1, L_padded] @ waveform rate
    return ssl, f0, spec, energy


def adapt_features(content, target_std=TARGET_STD):
    """Simple adaptation: normalize to target std (SolC-style moment matching
    against TinyVC distilled feature std)."""
    c_std = content.std()
    c_mean = content.mean()
    return (content - c_mean) / (c_std + 1e-8) * target_std + c_mean


# Build voice indices for each layer config
print("\nBuilding voice indices...")
src, sr = sf.read('data/source/source_real_001.wav')
src = src.astype(np.float32)[:sr*5]

# CAMPPlus speaker embedding extractor for similarity scoring.
import sherpa_onnx
campplus_path = 'models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'
if not os.path.exists(campplus_path):
    from huggingface_hub import hf_hub_download
    import shutil
    shutil.copy(hf_hub_download(repo_id='bitsydarel/campplus-onnx',
                                filename='3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'), campplus_path)
config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
config.model = campplus_path; config.num_threads = 1
extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)


def get_emb(wav, sr=24000):
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    stream = extractor.create_stream()
    stream.accept_waveform(16000, wav.tolist())
    stream.input_finished()
    return np.asarray(extractor.compute(stream), dtype=np.float32)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


src_emb = get_emb(src, sr)
target_embs = [get_emb(sf.read(f'data/voices/voice_{i}.wav')[0]) for i in range(5)]
print(f"  source={len(src)/sr:.1f}s @ {sr}Hz, src_emb dim={src_emb.shape[0]}")

# Aggregate summary across configs
summary_rows = []

for layers, label in LAYER_CONFIGS:
    print(f"\n=== {label} (layers={layers}) ===")

    # Build voice index
    voices = {}
    for vid in range(5):
        wav, sr_v = sf.read(f'data/voices/voice_{vid}.wav')
        feat = encode_hubert_layers(wav, sr_v, layers)
        feat = adapt_features(feat)
        voices[vid] = feat

    # Run VC
    results = []
    for vid in range(5):
        try:
            # Encode source
            content = encode_hubert_layers(src, sr, layers)
            content = adapt_features(content)
            _, f0, _, energy = encode_tinyvc(src, sr)

            # kNN
            target = voices[vid]
            # Align T of content ↔ f0 (50Hz nominal, but HuBERT padding adds/
            # removes 1-2 frames vs TinyVC). Energy stays at waveform rate —
            # the decoder's SourceNet max-pools it by frame_size=480 down to
            # frame rate, so we just need content/f0 to match f0_T =
            # L_padded // 480 (which the TinyVC encoder guarantees).
            t_c, t_f = content.shape[-1], f0.shape[-1]
            if t_c != t_f:
                if t_c < t_f:
                    content = F.interpolate(content, size=t_f, mode="linear")
                else:
                    f0 = F.interpolate(f0, size=t_c, mode="linear")
            # Align target T to content T (target was way longer — ~30s of
            # voice ref → ~1500 frames; kNN match handles it fine at any T).
            if target.shape[-1] != content.shape[-1]:
                target = F.interpolate(
                    target, size=content.shape[-1], mode="linear"
                )

            # For multi-layer, kNN with matching dims (768*N each);
            # then take the first 768 channels (= L4 of matched neighbors)
            # for the TinyVC decoder (which was trained on 768-d content).
            if content.shape[1] > 768:
                content_for_knn = content
                target_for_knn = target
                content_replaced = match_features(content_for_knn, target_for_knn, k=4, alpha=0.0)
                # Take first 768 channels for decoder
                content_for_decode = content_replaced[:, :768, :]
            else:
                content_replaced = match_features(content, target, k=4, alpha=0.0)
                content_for_decode = content_replaced

            with torch.no_grad():
                out = decoder.infer(content_for_decode, f0, energy)
            out = out.squeeze().numpy()

            out_emb = get_emb(out, SAMPLE_RATE)
            t_sim = cosine(out_emb, target_embs[vid])
            s_sim = cosine(out_emb, src_emb)
            results.append((t_sim, s_sim))
            print(f"  voice_{vid}: target={t_sim:.3f}, source={s_sim:.3f}, VC_effect={t_sim-s_sim:.3f}")
        except Exception as e:
            print(f"  voice_{vid}: FAILED - {type(e).__name__}: {e}")

    if results:
        avg_t = float(np.mean([r[0] for r in results]))
        avg_s = float(np.mean([r[1] for r in results]))
        print(f"  AVG: target={avg_t:.3f}, source={avg_s:.3f}, VC_effect={avg_t-avg_s:.3f}")
        summary_rows.append((label, layers, avg_t, avg_s, avg_t - avg_s, len(results)))
    else:
        summary_rows.append((label, layers, float('nan'), float('nan'), float('nan'), 0))

# Final cross-config summary
print("\n" + "=" * 70)
print("SUMMARY: Multi-layer HuBERT fusion (SolG)")
print("=" * 70)
print(f"{'config':<12} {'layers':<14} {'target':>8} {'source':>8} {'VC_effect':>10} {'n':>3}")
print("-" * 70)
for label, layers, t, s, eff, n in summary_rows:
    layers_str = "+".join(f"L{l}" for l in layers)
    print(f"{label:<12} {layers_str:<14} {t:>8.3f} {s:>8.3f} {eff:>+10.3f} {n:>3}")
print("-" * 70)
