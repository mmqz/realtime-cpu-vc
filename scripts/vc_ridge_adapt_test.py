#!/usr/bin/env python3
"""
SolJ: Ridge-regularized per-channel adaptation.
Fixes SolD's noise blowup by adding λ to denominator.
adapted[c] = (hubert[c] - μ_h[c]) / (σ_h[c] + λ) × σ_t[c] + μ_t[c]
"""
import sys, os, time, numpy as np, torch, soundfile as sf, librosa
sys.path.insert(0, 'src'); sys.path.insert(0, '/home/z/my-project/repos/tinyvc')
sys.modules['torchfcpe'] = type(sys)('torchfcpe'); sys.modules['torchfcpe'].spawn_bundled_infer_model = lambda *a: None
sys.modules['pyworld'] = type(sys)('pyworld'); sys.modules['pyworld'].dio = lambda *a: None; sys.modules['pyworld'].harvest = lambda *a: None; sys.modules['pyworld'].stonemask = lambda *a: None

from transformers import HubertModel, Wav2Vec2FeatureExtractor
from module.tinyvc.encoder import Encoder
from module.tinyvc.decoder import Decoder
from module.tinyvc.feature_retrieval import match_features
from module.utils.spectrogram import spectrogram
from module.utils.auto_padding import autopad_waveform
from module.utils.energy_estimation import estimate_energy
import torch.nn.functional as F

torch.set_num_threads(2)

hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960"); hubert.eval()
fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
enc = Encoder(); enc.load_state_dict(torch.load('models/encoder.pt', map_location='cpu')); enc.eval()
dec = Decoder(); dec.load_state_dict(torch.load('models/decoder.pt', map_location='cpu')); dec.eval()

# Compute per-channel stats
print("Computing per-channel stats...")
all_h, all_t = [], []
for i in range(5):
    wav, sr = sf.read(f'data/voices/voice_{i}.wav')
    wav_16k = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        h = hubert(**inputs, output_hidden_states=True).hidden_states[4].transpose(1,2)
    peak = max(float(np.max(np.abs(wav))), 1e-8); wav_n = (wav.astype(np.float32) * (10**(-3/20)/peak)).astype(np.float32)
    spec = spectrogram(autopad_waveform(torch.from_numpy(wav_n).unsqueeze(0)), 1920, 480)
    with torch.no_grad():
        t, _ = enc.infer(spec)
    mT = min(h.shape[-1], t.shape[-1])
    all_h.append(h[:,:,:mT]); all_t.append(t[:,:,:mT])

all_h = torch.cat(all_h, dim=-1); all_t = torch.cat(all_t, dim=-1)
h_mean = all_h.mean(dim=(0,2))  # [768]
h_std = all_h.std(dim=(0,2))    # [768]
t_mean = all_t.mean(dim=(0,2))   # [768]
t_std = all_t.std(dim=(0,2))     # [768]

# Also scalar stats
h_mu_s, h_sig_s = all_h.mean().item(), all_h.std().item()
t_mu_s, t_sig_s = all_t.mean().item(), all_t.std().item()

print(f"HuBERT per-ch: std range [{h_std.min():.4f}, {h_std.max():.4f}]")
print(f"Scalar: mu={h_mu_s:.4f}, sigma={h_sig_s:.4f} -> mu={t_mu_s:.4f}, sigma={t_sig_s:.4f}")

# CAMPPlus
import sherpa_onnx
cp = 'models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'
if not os.path.exists(cp):
    from huggingface_hub import hf_hub_download; import shutil
    shutil.copy(hf_hub_download(repo_id='bitsydarel/campplus-onnx', filename='3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'), cp)
cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(); cfg.model = cp; cfg.num_threads = 1
ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
def emb(w, sr=24000):
    if sr != 16000: w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
    s = ext.create_stream(); s.accept_waveform(16000, w.tolist()); s.input_finished()
    return ext.compute(s)
def cos(a, b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

src, sr = sf.read('data/source/source_real_001.wav'); src = src.astype(np.float32)[:sr*5]
src_e = emb(src); tgt_e = [emb(sf.read(f'data/voices/voice_{i}.wav')[0]) for i in range(5)]

def encode_hubert(wav, sr):
    wav_16k = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        h = hubert(**inputs, output_hidden_states=True).hidden_states[4].transpose(1,2)
    return h

def encode_tinyvc_f0(wav, sr):
    peak = max(float(np.max(np.abs(wav))), 1e-8); wav_n = (wav.astype(np.float32) * (10**(-3/20)/peak)).astype(np.float32)
    wav_t = autopad_waveform(torch.from_numpy(wav_n).unsqueeze(0))
    spec = spectrogram(wav_t, 1920, 480)
    with torch.no_grad(): _, f0 = enc.infer(spec)
    energy = estimate_energy(wav_t)
    return f0, energy, wav_t

def adapt_ridge(content, lam):
    """Ridge-regularized per-channel adaptation."""
    mean = h_mean.unsqueeze(0).unsqueeze(-1)
    std = h_std.unsqueeze(0).unsqueeze(-1)
    tm = t_mean.unsqueeze(0).unsqueeze(-1)
    ts = t_std.unsqueeze(0).unsqueeze(-1)
    return (content - mean) / (std + lam) * ts + tm

def adapt_scalar(content):
    return (content - h_mu_s) / (h_sig_s + 1e-8) * t_sig_s + t_mu_s

# Build voice indices for each method
LAMBDAS = [0.01, 0.05, 0.1, 0.2, 0.5]

print("\n=== SolJ: Ridge-Regularized Per-Channel ===\n")

# Scalar baseline
print("--- Scalar (SolC baseline) ---")
scalar_voices = {}
for i in range(5):
    wav, sr_v = sf.read(f'data/voices/voice_{i}.wav')
    h = encode_hubert(wav, sr_v)
    scalar_voices[i] = adapt_scalar(h)

scalar_results = []
for vid in range(5):
    h = encode_hubert(src, sr)
    content = adapt_scalar(h)
    target = scalar_voices[vid]
    f0, energy, wav_t = encode_tinyvc_f0(src, sr)
    mT = min(content.shape[-1], target.shape[-1], f0.shape[-1])
    content, target, f0 = content[:,:,:mT], target[:,:,:mT], f0[:,:,:mT]
    energy = energy[:,:,:mT*480]
    cr = match_features(content, target, k=4, alpha=0.0)
    with torch.no_grad(): out = dec.infer(cr, f0, energy)
    out = out.squeeze().numpy()
    oe = emb(out, sr)
    t_s = cos(oe,tgt_e[vid]); s_s = cos(oe,src_e)
    scalar_results.append((t_s, s_s))
    print(f"  voice_{vid}: target={t_s:.3f}, source={s_s:.3f}, VC_effect={t_s-s_s:.3f}")
avg_t = np.mean([r[0] for r in scalar_results]); avg_s = np.mean([r[1] for r in scalar_results])
print(f"  AVG: target={avg_t:.3f}, source={avg_s:.3f}, VC_effect={avg_t-avg_s:.3f}")

# Ridge sweep
for lam in LAMBDAS:
    print(f"\n--- Ridge lambda={lam} ---")
    r_voices = {}
    for i in range(5):
        wav, sr_v = sf.read(f'data/voices/voice_{i}.wav')
        h = encode_hubert(wav, sr_v)
        r_voices[i] = adapt_ridge(h, lam)
    
    results = []
    for vid in range(5):
        h = encode_hubert(src, sr)
        content = adapt_ridge(h, lam)
        target = r_voices[vid]
        f0, energy, _ = encode_tinyvc_f0(src, sr)
        mT = min(content.shape[-1], target.shape[-1], f0.shape[-1])
        content, target, f0 = content[:,:,:mT], target[:,:,:mT], f0[:,:,:mT]
        energy = energy[:,:,:mT*480]
        cr = match_features(content, target, k=4, alpha=0.0)
        with torch.no_grad(): out = dec.infer(cr, f0, energy)
        out = out.squeeze().numpy()
        oe = emb(out, sr)
        t_s = cos(oe, tgt_e[vid]); s_s = cos(oe, src_e)
        results.append((t_s, s_s))
        print(f"  voice_{vid}: target={t_s:.3f}, source={s_s:.3f}, VC_effect={t_s-s_s:.3f}")
    avg_t = np.mean([r[0] for r in results]); avg_s = np.mean([r[1] for r in results])
    print(f"  AVG: target={avg_t:.3f}, source={avg_s:.3f}, VC_effect={avg_t-avg_s:.3f}")
