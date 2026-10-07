"""SolE+I: Quick 768x768 learned projection (50 steps) + multi-layer scalar-adapted combo.

Task A (SolE): Train nn.Linear(768,768) on 3 voices, 50 epochs, to map
HuBERT-base L4 -> TinyVC ConvNeXt distilled features. Then run VC.

Task B (SolI): Combine SolG multi-layer fusion (L4+L6+L8) with SolC scalar
adaptation applied to the L4 channel block.
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

torch.set_num_threads(2)

hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960"); hubert.eval()
fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
enc = Encoder(); enc.load_state_dict(torch.load('models/encoder.pt', map_location='cpu')); enc.eval()
dec = Decoder(); dec.load_state_dict(torch.load('models/decoder.pt', map_location='cpu')); dec.eval()

# Collect paired features (just 3 voices for speed)
print("Collecting paired features...")
h_feats, t_feats = [], []
for i in [0, 1, 2]:
    wav, sr = sf.read(f'data/voices/voice_{i}.wav')
    wav_16k = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        h = hubert(**inputs, output_hidden_states=True).hidden_states[4].transpose(1,2)
    peak = max(np.max(np.abs(wav)), 1e-8); wav_n = (wav.astype(np.float32) * float(10**(-3/20)/peak)).astype(np.float32)
    spec = spectrogram(autopad_waveform(torch.from_numpy(wav_n).unsqueeze(0)), 1920, 480)
    with torch.no_grad():
        t, _ = enc.infer(spec)
    min_T = min(h.shape[-1], t.shape[-1])
    h_feats.append(h[:,:,:min_T]); t_feats.append(t[:,:,:min_T])

# Train 768x768 (only 50 steps for speed)
proj = torch.nn.Linear(768, 768, bias=True)
torch.nn.init.eye_(proj.weight); torch.nn.init.zeros_(proj.bias)
opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
print("Training 768x768 (50 steps)...")
for epoch in range(50):
    loss_sum = 0
    for h, t in zip(h_feats, t_feats):
        for s in range(0, h.shape[-1], 50):
            e = min(s+50, h.shape[-1])
            hc, tc = h[:,:,s:e], t[:,:,s:e]
            hp = proj(hc.transpose(1,2)).transpose(1,2)
            loss = torch.nn.functional.mse_loss(hp, tc)
            opt.zero_grad(); loss.backward(); opt.step(); loss_sum += loss.item()
    if (epoch+1) % 25 == 0: print(f"  Epoch {epoch+1}: loss={loss_sum/len(h_feats):.6f}")

# Test VC
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

# Build projected voice index
pv = {}
for i in range(5):
    wav, sr_v = sf.read(f'data/voices/voice_{i}.wav')
    wav_16k = librosa.resample(wav.astype(np.float32), orig_sr=sr_v, target_sr=16000)
    inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        h = hubert(**inputs, output_hidden_states=True).hidden_states[4].transpose(1,2)
        pv[i] = proj(h.transpose(1,2)).transpose(1,2)

print("\n=== SolE: 768x768 Learned Projection ===")
for vid in range(5):
    wav_16k = librosa.resample(src, orig_sr=sr, target_sr=16000)
    inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        h = hubert(**inputs, output_hidden_states=True).hidden_states[4].transpose(1,2)
        content = proj(h.transpose(1,2)).transpose(1,2)
    peak = max(np.max(np.abs(src)), 1e-8); wav_n = (src.astype(np.float32) * float(10**(-3/20)/peak)).astype(np.float32)
    spec = spectrogram(autopad_waveform(torch.from_numpy(wav_n).unsqueeze(0)), 1920, 480)
    with torch.no_grad(): _, f0 = enc.infer(spec)
    energy = estimate_energy(autopad_waveform(torch.from_numpy(wav_n).unsqueeze(0)))
    target = pv[vid]
    mT = min(content.shape[-1], target.shape[-1], f0.shape[-1])
    content, target, f0 = content[:,:,:mT], target[:,:,:mT], f0[:,:,:mT]
    energy = energy[:,:,:mT*480]
    cr = match_features(content, target, k=4, alpha=0.0)
    with torch.no_grad(): out = dec.infer(cr, f0, energy)
    out = out.squeeze().numpy()
    sf.write(f'/home/z/my-project/download/vc_solE_voice_{vid}.wav', out, sr)
    oe = emb(out, sr)
    print(f"  voice_{vid}: target={cos(oe,tgt_e[vid]):.3f}, source={cos(oe,src_e):.3f}, VC_effect={cos(oe,tgt_e[vid])-cos(oe,src_e):.3f}")

# SolI: Combine multi-layer fusion (L4+L6+L8) + scalar adaptation
print("\n=== SolI: Multi-layer (L4+L6+L8) + Scalar Adaptation ===")

# Compute scalar stats from L4 features (same as SolC)
all_h, all_t = [], []
for i in range(5):
    wav, sr_v = sf.read(f'data/voices/voice_{i}.wav')
    wav_16k = librosa.resample(wav.astype(np.float32), orig_sr=sr_v, target_sr=16000)
    inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        h = hubert(**inputs, output_hidden_states=True).hidden_states[4].transpose(1,2)
    peak = max(np.max(np.abs(wav)), 1e-8); wav_n = (wav.astype(np.float32) * float(10**(-3/20)/peak)).astype(np.float32)
    spec = spectrogram(autopad_waveform(torch.from_numpy(wav_n).unsqueeze(0)), 1920, 480)
    with torch.no_grad():
        t, _ = enc.infer(spec)
    mT = min(h.shape[-1], t.shape[-1])
    all_h.append(h[:,:,:mT]); all_t.append(t[:,:,:mT])
all_h = torch.cat(all_h, dim=-1); all_t = torch.cat(all_t, dim=-1)
h_mu, h_sig = all_h.mean().item(), all_h.std().item()
t_mu, t_sig = all_t.mean().item(), all_t.std().item()

def encode_multilayer_adapted(wav, sr, layers=[4,6,8]):
    wav_16k = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    inputs = fe(wav_16k, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = hubert(**inputs, output_hidden_states=True)
        feats = [out.hidden_states[l].transpose(1,2) for l in layers]
        concat = torch.cat(feats, dim=1)  # [1, 768*3, T]
    # Adapt only the first 768 channels (L4) with scalar normalization
    l4 = concat[:, :768, :]
    l4_adapted = (l4 - h_mu) / (h_sig + 1e-8) * t_sig + t_mu
    # Replace L4 with adapted, keep L6+L8 raw
    concat[:, :768, :] = l4_adapted
    return concat

# Build voice index with multi-layer + adapted
ml_voices = {}
for i in range(5):
    wav, sr_v = sf.read(f'data/voices/voice_{i}.wav')
    ml_voices[i] = encode_multilayer_adapted(wav, sr_v)

for vid in range(5):
    content = encode_multilayer_adapted(src, sr)
    peak = max(np.max(np.abs(src)), 1e-8); wav_n = (src.astype(np.float32) * float(10**(-3/20)/peak)).astype(np.float32)
    spec = spectrogram(autopad_waveform(torch.from_numpy(wav_n).unsqueeze(0)), 1920, 480)
    with torch.no_grad(): _, f0 = enc.infer(spec)
    energy = estimate_energy(autopad_waveform(torch.from_numpy(wav_n).unsqueeze(0)))
    target = ml_voices[vid]
    mT = min(content.shape[-1], target.shape[-1], f0.shape[-1])
    content, target, f0 = content[:,:,:mT], target[:,:,:mT], f0[:,:,:mT]
    energy = energy[:,:,:mT*480]
    cr = match_features(content, target, k=4, alpha=0.0)
    # Use first 768 channels for decoder
    with torch.no_grad(): out = dec.infer(cr[:, :768, :], f0, energy)
    out = out.squeeze().numpy()
    sf.write(f'/home/z/my-project/download/vc_solI_voice_{vid}.wav', out, sr)
    oe = emb(out, sr)
    print(f"  voice_{vid}: target={cos(oe,tgt_e[vid]):.3f}, source={cos(oe,src_e):.3f}, VC_effect={cos(oe,tgt_e[vid])-cos(oe,src_e):.3f}")
