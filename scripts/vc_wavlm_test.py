#!/usr/bin/env python3
"""
SolB: WavLM-base as content encoder (TinyVC's teacher model).

Background
----------
TinyVC's content encoder is a 4.7M ConvNeXt-v2 distilled FROM WavLM-Base-Plus
layer 4. It produces 768-d features at 50Hz. Using raw WavLM-base layer-4
features should therefore be the closest possible distribution to what the
TinyVC DDSP decoder was trained on (since WavLM layer 4 IS the training target).

Pipeline
--------
  source_audio (24 kHz mono)
      │  TinyVC PitchEstimator (ConvNeXt over STFT) -> f0 [1, 1, T] @ 50Hz
      │  TinyVC estimate_energy                        -> energy [1, 1, L] @ 24kHz
      │  WavLM-base layer-4 hidden_state (16kHz)      -> content [1, 768, T] @ 50Hz
      v  (align T_wavlm to T_tinyvc via F.interpolate)
  kNN-VC replace(content, target_voice_wavlm_features, k=4, alpha=0)
      │
      v
  TinyVC DDSP Decoder (SourceNet + FilterNet)
      │
      v
  output_audio (24 kHz mono)

Also compares WavLM layer-4 vs TinyVC ConvNeXt-v2 feature distributions
(per-frame cosine, mean/std) to verify they are indeed close (same distillation
target).
"""
from __future__ import annotations

import os
import sys
import time
import types
import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Optional-deps stubbing (mirrors scripts/vc_distilhubert_test.py).
# `module/utils/__init__.py` imports `f0_estimation`, which imports `torchfcpe`
# and `pyworld`. Those are only needed for the `fcpe`/`dio`/`harvest` algorithms.
# We never call those here, so stub them so `from module.tinyvc import ...`
# works without heavy optional deps.
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

import librosa  # noqa: E402

# Make upstream TinyVC importable.
TINYVC_ROOT = Path(os.environ.get("TINYVC_ROOT", "/home/z/my-project/repos/tinyvc"))
if str(TINYVC_ROOT) not in sys.path:
    sys.path.insert(0, str(TINYVC_ROOT))

from transformers import WavLMModel, Wav2Vec2FeatureExtractor  # noqa: E402

from module.tinyvc import Decoder as _TinyVCDecoder  # noqa: E402
from module.tinyvc import Encoder as _TinyVCEncoder  # noqa: E402
from module.tinyvc import match_features as _tinyvc_match_features  # noqa: E402
from module.utils.auto_padding import autopad_waveform as _tinyvc_autopad  # noqa: E402
from module.utils.energy_estimation import estimate_energy as _tinyvc_estimate_energy  # noqa: E402
from module.utils.spectrogram import spectrogram as _tinyvc_spectrogram  # noqa: E402

torch.set_num_threads(2)

SAMPLE_RATE = 24000
FRAME_SIZE = 480  # 50Hz @ 24kHz
CONTENT_DIM = 768
DEFAULT_NORM_DB = -3.0
WAVLM_LAYER = 4  # TinyVC's ConvNeXt-v2 was distilled FROM WavLM layer 4.

OUTPUT_DIR = "/home/z/my-project/download"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class WavLMTinyVCHybrid:
    """WavLM-base (TinyVC's teacher) content + TinyVC F0/energy + TinyVC DDSP decoder."""

    def __init__(self, models_dir: str = "models", wavlm_layer: int = WAVLM_LAYER):
        # --- WavLM-base (teacher model of TinyVC's ConvNeXt) -----------------
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.wavlm.eval()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
        self.wavlm_layer = wavlm_layer
        self.ssl_params = sum(p.numel() for p in self.wavlm.parameters())
        print(
            f"  WavLM-base = {self.ssl_params/1e6:.1f}M params, "
            f"hidden_size={self.wavlm.config.hidden_size}, "
            f"num_layers={self.wavlm.config.num_hidden_layers}, "
            f"using layer {wavlm_layer}"
        )

        # --- TinyVC encoder (only used for F0 + ConvNeXt comparison) ----------
        tinyvc_enc_path = Path(f"{models_dir}/encoder.pt")
        if not tinyvc_enc_path.exists():
            raise FileNotFoundError(f"{tinyvc_enc_path} not found")
        self._tinyvc_encoder = _TinyVCEncoder()
        self._tinyvc_encoder.load_state_dict(
            torch.load(str(tinyvc_enc_path), map_location="cpu")
        )
        self._tinyvc_encoder.eval()
        self.pitch_estimator = self._tinyvc_encoder.pitch_estimator
        self.ssl_feature_estimator = self._tinyvc_encoder.ssl_feature_estimator
        self.pitch_params = sum(p.numel() for p in self.pitch_estimator.parameters())
        self.tinyvc_enc_params = sum(p.numel() for p in self._tinyvc_encoder.parameters())
        print(
            f"  TinyVC Encoder = {self.tinyvc_enc_params/1e6:.2f}M params "
            f"(PitchEstimator={self.pitch_params/1e6:.2f}M)"
        )

        # --- TinyVC DDSP decoder ---------------------------------------------
        dec_path = Path(f"{models_dir}/decoder.pt")
        if not dec_path.exists():
            raise FileNotFoundError(f"{dec_path} not found")
        self.decoder = _TinyVCDecoder()
        self.decoder.load_state_dict(
            torch.load(str(dec_path), map_location="cpu")
        )
        self.decoder.eval()
        self.dec_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"  TinyVC Decoder = {self.dec_params/1e6:.2f}M params")

        # kNN index of voice references, encoded with WavLM
        self.voices: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _encode_wavlm(self, wav_24k: np.ndarray) -> torch.Tensor:
        """WavLM-base forward on 16kHz-resampled input.

        Returns content [1, 768, T_wav] @ 50Hz using hidden_states[layer].
        """
        wav_16k = librosa.resample(
            wav_24k.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=16000
        )
        inputs = self.fe(wav_16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.wavlm(**inputs, output_hidden_states=True)
        # hidden_states is a tuple of (num_layers+1,) tensors [B, T, D].
        content = outputs.hidden_states[self.wavlm_layer]  # [1, T, 768]
        content = content.transpose(1, 2)  # [1, 768, T_wav]
        return content

    def _encode_tinyvc_content(self, wav_24k: np.ndarray) -> torch.Tensor:
        """TinyVC ConvNeXt-v2 SSLFeatureEstimator forward on 24kHz STFT.

        Returns content [1, 768, T_tiny] @ 50Hz.
        """
        wf = torch.from_numpy(wav_24k.astype(np.float32)).unsqueeze(0)  # [1, L]
        wf = _tinyvc_autopad(wf)
        spec = _tinyvc_spectrogram(wf)  # [1, fft_bin, T]
        with torch.no_grad():
            content = self.ssl_feature_estimator.infer(spec)  # [1, 768, T]
        return content

    def _encode_f0(self, wav_24k: np.ndarray) -> torch.Tensor:
        """TinyVC PitchEstimator forward on 24kHz STFT.

        Returns f0 [1, 1, T_tiny] @ 50Hz.
        """
        wf = torch.from_numpy(wav_24k.astype(np.float32)).unsqueeze(0)
        wf = _tinyvc_autopad(wf)
        spec = _tinyvc_spectrogram(wf)
        with torch.inference_mode():
            f0 = self.pitch_estimator.infer(spec)  # [1, 1, T]
        return f0

    def _estimate_energy(self, wav_24k: np.ndarray) -> torch.Tensor:
        """TinyVC energy estimation. Returns energy [1, 1, L] @ 24kHz."""
        wf = torch.from_numpy(wav_24k.astype(np.float32)).unsqueeze(0)
        wf = _tinyvc_autopad(wf)
        return _tinyvc_estimate_energy(wf)

    @staticmethod
    def _align_frames(content: torch.Tensor, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Align T_wavlm to T_tinyvc via F.interpolate on time dim.

        Both are nominally 50Hz, but WavLM may produce 1-2 extra/missing frames
        due to its 16kHz stride + padding. We resample the smaller to the larger
        so the decoder sees a consistent frame count.
        """
        t_wav = content.shape[-1]
        t_tiny = f0.shape[-1]
        if t_wav == t_tiny:
            return content, f0
        if t_wav < t_tiny:
            content = F.interpolate(content, size=t_tiny, mode="linear")
        else:
            f0 = F.interpolate(f0, size=t_wav, mode="linear")
        return content, f0

    def encode_wavlm(self, wav_24k: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode: WavLM content + TinyVC F0 + TinyVC energy."""
        content = self._encode_wavlm(wav_24k)      # [1, 768, T_wav]
        f0 = self._encode_f0(wav_24k)              # [1, 1, T_tiny]
        energy = self._estimate_energy(wav_24k)    # [1, 1, L]
        content, f0 = self._align_frames(content, f0)
        return content, f0, energy

    def encode_tinyvc(self, wav_24k: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TinyVC encode (ConvNeXt content + PitchEstimator f0 + energy) for comparison."""
        wf = torch.from_numpy(wav_24k.astype(np.float32)).unsqueeze(0)
        wf = _tinyvc_autopad(wf)
        spec = _tinyvc_spectrogram(wf)
        with torch.no_grad():
            content, f0 = self._tinyvc_encoder.infer(spec)  # [1, 768, T], [1, 1, T]
        energy = self._estimate_energy(wav_24k)
        return content, f0, energy

    def compare_features(self, wav_24k: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare WavLM layer-4 vs TinyVC ConvNeXt-v2 feature distributions."""
        wl_content, _, _ = self.encode_wavlm(wav_24k)
        tv_content, _, _ = self.encode_tinyvc(wav_24k)

        # Match time dimension
        t_min = min(wl_content.shape[-1], tv_content.shape[-1])
        wl = wl_content[:, :, :t_min]
        tv = tv_content[:, :, :t_min]

        # Per-frame cosine similarity
        wl_flat = wl.squeeze(0).T  # [T, 768]
        tv_flat = tv.squeeze(0).T  # [T, 768]
        cosine_per_frame = F.cosine_similarity(wl_flat, tv_flat, dim=-1)

        # Per-channel magnitude stats (768-d)
        wl_mean = float(wl.mean())
        wl_std = float(wl.std())
        tv_mean = float(tv.mean())
        tv_std = float(tv.std())

        print(f"  WavLM L4 : mean={wl_mean:+.4f}, std={wl_std:.4f}, shape={tuple(wl.shape)}")
        print(f"  TinyVC   : mean={tv_mean:+.4f}, std={tv_std:.4f}, shape={tuple(tv.shape)}")
        print(
            f"  Per-frame cosine (WavLM L4 vs TinyVC): "
            f"mean={float(cosine_per_frame.mean()):.4f}  "
            f"std={float(cosine_per_frame.std()):.4f}  "
            f"min={float(cosine_per_frame.min()):.4f}  "
            f"max={float(cosine_per_frame.max()):.4f}"
        )
        # Ratio of std (WavLM features are typically 5-10x larger in magnitude than
        # ConvNeXt-v2 distilled features because ConvNeXt is learned to be a
        # low-rank compression of WavLM layer 4).
        print(f"  std ratio (WavLM/TinyVC) = {wl_std/tv_std:.3f}")

        return wl, tv

    # ------------------------------------------------------------------
    # kNN voice index (rebuild with WavLM features)
    # ------------------------------------------------------------------
    def build_voice_index(self, voice_dir: str = "data/voices"):
        print(f"\n[kNN index] Rebuilding with WavLM-base layer-{self.wavlm_layer} features ...")
        t_start = time.perf_counter()
        for path in sorted(glob.glob(f"{voice_dir}/voice_*.wav")):
            wav, sr = sf.read(path)
            if sr != SAMPLE_RATE:
                wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
            else:
                wav = wav.astype(np.float32)
            # Peak-normalize to -3 dBFS (TinyVC convention) for fair kNN distances.
            peak = max(float(np.max(np.abs(wav))), 1e-8)
            wav = wav * (10 ** (DEFAULT_NORM_DB / 20.0) / peak)
            content, _, _ = self.encode_wavlm(wav)
            voice_id = int(path.split("voice_")[-1].split(".")[0])
            key = f"voice_{voice_id}"
            self.voices[key] = content
            print(f"  {key}: content {tuple(content.shape)}  (src len {len(wav)} samples)")
        print(f"  built {len(self.voices)} voice indices in {time.perf_counter()-t_start:.2f}s")

    # ------------------------------------------------------------------
    # Full pipeline: WavLM → kNN → TinyVC DDSP
    # ------------------------------------------------------------------
    def process_audio(self, wav: np.ndarray, sr: int, voice_id: int = 0) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        peak = max(float(np.max(np.abs(wav))), 1e-8)
        wav = wav * (10 ** (DEFAULT_NORM_DB / 20.0) / peak)

        # 1. Encode (WavLM content + TinyVC F0 + TinyVC energy)
        content, f0, energy = self.encode_wavlm(wav)

        # 2. kNN retrieval (WavLM → WavLM matched frames)
        key = f"voice_{voice_id}"
        if key not in self.voices:
            raise KeyError(f"{key} not in voice index (have: {list(self.voices.keys())})")
        target = self.voices[key]
        if target.shape[-1] != content.shape[-1]:
            target = F.interpolate(target, size=content.shape[-1], mode="linear")
        content_replaced = _tinyvc_match_features(content, target, k=4, alpha=0.0, metrics="cos")

        # 3. Decode (TinyVC DDSP)
        with torch.inference_mode():
            out = self.decoder.infer(content_replaced, f0, energy)
        return out.squeeze(0).squeeze(0).cpu().numpy()


# ===========================================================================
# Main test driver
# ===========================================================================
def main():
    import sherpa_onnx

    print("=" * 70)
    print("SolB: WavLM-base Content Encoder (TinyVC's teacher model)")
    print("=" * 70)
    print(
        "TinyVC's 4.7M ConvNeXt-v2 was distilled FROM WavLM-Base-Plus layer 4 →\n"
        "raw WavLM-base layer-4 features should be CLOSEST to what the\n"
        "TinyVC DDSP decoder expects."
    )

    print("\n[step 1] Loading models ...")
    infer = WavLMTinyVCHybrid(models_dir="models", wavlm_layer=WAVLM_LAYER)
    print(
        f"\n  Model sizes: SSL(WavLM)={infer.ssl_params/1e6:.1f}M  "
        f"Pitch={infer.pitch_params/1e6:.2f}M  Decoder={infer.dec_params/1e6:.2f}M  "
        f"Total(inference path)={(infer.ssl_params+infer.pitch_params+infer.dec_params)/1e6:.1f}M"
    )

    # ---- Feature distribution comparison: WavLM L4 vs TinyVC ConvNeXt-v2 ----
    print(f"\n{'='*70}")
    print("Feature distribution comparison: WavLM layer 4 vs TinyVC ConvNeXt-v2")
    print(f"{'='*70}")
    wav, sr = sf.read("data/voices/voice_0.wav")
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    else:
        wav = wav.astype(np.float32)
    infer.compare_features(wav[: sr * 5])

    # ---- Build WavLM voice index ----
    infer.build_voice_index()

    # ---- CAMPPlus speaker similarity ----
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
    config.model = "models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
    config.num_threads = 1
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def get_emb(wav_in, sr=SAMPLE_RATE):
        if sr != 16000:
            wav_in = librosa.resample(wav_in.astype(np.float32), orig_sr=sr, target_sr=16000)
        stream = extractor.create_stream()
        stream.accept_waveform(16000, wav_in.tolist())
        stream.input_finished()
        return np.asarray(extractor.compute(stream), dtype=np.float32)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # Source: real human speech
    src_path = "data/source/source_real_001.wav"
    if not os.path.exists(src_path):
        src_path = "data/source/source_006.wav"
    src, sr = sf.read(src_path)
    src = src.astype(np.float32)[: sr * 5]
    src_emb = get_emb(src, sr)
    print(f"\nSource: {src_path}  ({len(src)/sr:.1f}s @ {sr}Hz)")
    print(f"  source CAMPPlus embedding dim = {src_emb.shape}")

    # Target embeddings from original voice WAVs
    target_embs = []
    for i in range(5):
        wav, vsr = sf.read(f"data/voices/voice_{i}.wav")
        if vsr != SAMPLE_RATE:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=vsr, target_sr=SAMPLE_RATE)
        else:
            wav = wav.astype(np.float32)
        target_embs.append(get_emb(wav, sr))

    # Sanity: target distinctness
    print("\nTarget voice pairwise similarity (CAMPPlus cosine):")
    for i in range(5):
        for j in range(i + 1, 5):
            s = cosine(target_embs[i], target_embs[j])
            print(f"  voice_{i} vs voice_{j}: {s:.3f}")

    # ---- Run VC for all 5 voices ----
    print(f"\n{'='*70}")
    print(f"VC quality test: WavLM-base → kNN-VC → TinyVC decoder")
    print(f"{'='*70}")

    results = []
    for vid in range(5):
        try:
            t0 = time.perf_counter()
            out = infer.process_audio(src, sr, voice_id=vid)
            t1 = time.perf_counter()

            out_path = f"{OUTPUT_DIR}/vc_wavlm_voice_{vid}.wav"
            sf.write(out_path, out, SAMPLE_RATE)

            out_emb = get_emb(out, SAMPLE_RATE)
            t_sim = cosine(out_emb, target_embs[vid])
            s_sim = cosine(out_emb, src_emb)

            rtf = (t1 - t0) / 5.0
            rms = float(np.sqrt(np.mean(out ** 2)))
            peak_db = 20 * np.log10(max(float(np.max(np.abs(out))), 1e-8))
            n_nan = int(np.sum(~np.isfinite(out)))

            results.append({
                "voice_id": vid,
                "target_sim": round(t_sim, 3),
                "source_sim": round(s_sim, 3),
                "vc_effect": round(t_sim - s_sim, 3),
                "rtf": round(rtf, 4),
                "rms": round(rms, 4),
                "peak_db": round(peak_db, 2),
                "n_nan": n_nan,
            })
            print(
                f"  voice_{vid}: target={t_sim:.3f}  source={s_sim:.3f}  "
                f"VC_effect={t_sim-s_sim:+.3f}  RTF={rtf:.4f}  "
                f"RMS={rms:.4f}  peak={peak_db:.2f}dB  NaN={n_nan}"
            )
        except Exception as e:
            print(f"  voice_{vid}: FAILED - {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            results.append({"voice_id": vid, "error": str(e)})

    if results and "target_sim" in results[0]:
        avg_t = float(np.mean([r["target_sim"] for r in results if "target_sim" in r]))
        avg_s = float(np.mean([r["source_sim"] for r in results if "source_sim" in r]))
        avg_rtf = float(np.mean([r["rtf"] for r in results if "rtf" in r]))
        print(
            f"\n  AVG: target={avg_t:.3f}  source={avg_s:.3f}  "
            f"VC_effect={avg_t-avg_s:+.3f}  RTF={avg_rtf:.4f}"
        )

    print(f"\n[done] Output files in {OUTPUT_DIR}/vc_wavlm_voice_*.wav")


if __name__ == "__main__":
    main()
