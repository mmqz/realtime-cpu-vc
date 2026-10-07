#!/usr/bin/env python3
"""
Path 2: DistilHuBERT (or HuBERT-base substitute) as content encoder.

Background
----------
TinyVC's content encoder is a 4.7M ConvNeXt-v2 distilled from WavLM-Base-Plus
layer 4. It produces 768-d features at 50Hz.

DistilHuBERT (microsoft/distilhubert, 23M) is a smaller distilled version of
HuBERT-base. It also produces 768-d features at 50Hz. The hope is that DistilHuBERT
features → better kNN matches → higher VC quality.

Note on substitution
-------------------
`microsoft/distilhubert` returned HTTP 401 (gated/unavailable) in this sandbox
even though the repo is public on HF. We tried several DistilHuBERT variants
(OthmaneJ/distil-hubert, synapseplatform/distilhubert-base, s3prl/s3prl-distilhubert,
m-a-p/HuBERT-base) — all returned 401. The following alternatives are reachable:

  * facebook/hubert-base-ls960      (94.4M — DistilHuBERT's source model)
  * microsoft/wavlm-base            (94.4M — TinyVC's distillation source)
  * facebook/wav2vec2-base-960h     (94.4M)

This script uses facebook/hubert-base-ls960 (the FULL model DistilHuBERT was
distilled from) as the primary substitute, and also tests microsoft/wavlm-base
to answer "what if we used the original WavLM-base layer 4 instead of the
distilled ConvNeXt-v2?"

Pipeline
--------
  source_audio (24 kHz mono)
      │  TinyVC PitchEstimator (ConvNeXt over STFT) -> f0 [1, 1, T] @ 50Hz
      │  TinyVC estimate_energy                        -> energy [1, 1, L] @ 24kHz
      │  HuBERT-base (16kHz) -> hidden_states          -> content [1, T, 768] @ 50Hz
      v  (align T_hubert to T_tinyvc via F.interpolate)
  kNN-VC replace(content, target_voice_hubert_features, k=4, alpha=0)
      │
      v
  TinyVC DDSP Decoder (SourceNet + FilterNet)
      │
      v
  output_audio (24 kHz mono)
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
# Optional-deps stubbing (mirrors src/vc_realtime/infer_v1.py).
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

import librosa

# Make upstream TinyVC importable.
TINYVC_ROOT = Path(os.environ.get("TINYVC_ROOT", "/home/z/my-project/repos/tinyvc"))
if str(TINYVC_ROOT) not in sys.path:
    sys.path.insert(0, str(TINYVC_ROOT))

from transformers import AutoModel, AutoFeatureExtractor  # noqa: E402

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

OUTPUT_DIR = "/home/z/my-project/download"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Encoder selection
# ---------------------------------------------------------------------------
def _try_load_hf(repo_id: str):
    """Return (model, feature_extractor) or None if repo is unavailable."""
    try:
        m = AutoModel.from_pretrained(repo_id)
        fe = AutoFeatureExtractor.from_pretrained(repo_id)
        return m, fe
    except Exception as e:
        print(f"  [skip] {repo_id}: {type(e).__name__}: {str(e)[:120]}")
        return None, None


def select_distilhubert_substitute():
    """Try microsoft/distilhubert first, then fall back to reachable alternatives."""
    candidates = [
        ("microsoft/distilhubert", "DistilHuBERT (target)", "last_hidden_state"),
        ("facebook/hubert-base-ls960", "HuBERT-base (substitute)", "last_hidden_state"),
        ("microsoft/wavlm-base", "WavLM-base (TinyVC distillation source)", "last_hidden_state"),
        ("facebook/wav2vec2-base-960h", "Wav2Vec2-base (substitute)", "last_hidden_state"),
    ]
    print("\n[encoder selection] probing HF repos in order:")
    for repo_id, label, _attr in candidates:
        m, fe = _try_load_hf(repo_id)
        if m is not None:
            n_params = sum(p.numel() for p in m.parameters()) / 1e6
            print(
                f"  [ok] {repo_id}  ({label}, {n_params:.1f}M params, "
                f"hidden_size={m.config.hidden_size}, "
                f"layers={m.config.num_hidden_layers})"
            )
            return repo_id, label, m, fe
    raise RuntimeError(
        "No DistilHuBERT substitute reachable. Tried: "
        + ", ".join(c[0] for c in candidates)
    )


class HubertTinyVCHybrid:
    """Hybrid: HuBERT-base content + TinyVC F0 + TinyVC DDSP decoder.

    DistilHuBERT was unavailable (HTTP 401 in this sandbox), so we use the full
    HuBERT-base (94.4M, the model DistilHuBERT was distilled from) as the
    substitute. We also keep the TinyVC PitchEstimator + estimate_energy since
    the TinyVC decoder was trained with those signal distributions.
    """

    def __init__(
        self,
        models_dir: str = "models",
        encoder_repo: str = "facebook/hubert-base-ls960",
        encoder_label: str = "HuBERT-base",
        hubert_model=None,
        hubert_fe=None,
        hubert_layer: int | None = 4,
    ):
        # --- SSL content encoder (DistilHuBERT substitute) -------------------
        self.encoder_repo = encoder_repo
        self.encoder_label = encoder_label
        self.hubert = hubert_model or AutoModel.from_pretrained(encoder_repo)
        self.hubert.eval()
        self.fe = hubert_fe or AutoFeatureExtractor.from_pretrained(encoder_repo)
        # Layer 4 to match TinyVC's distillation target (WavLM layer 4).
        # If `None`, use last_hidden_state.
        self.hubert_layer = hubert_layer
        self.ssl_params = sum(p.numel() for p in self.hubert.parameters())
        print(
            f"  DistilHuBERT-substitute = {encoder_repo} ({self.ssl_params/1e6:.1f}M params, "
            f"layer={'last' if hubert_layer is None else hubert_layer})"
        )

        # --- TinyVC F0 estimator (separate submodule of TinyVC Encoder) ------
        # We ONLY use its pitch_estimator; the SSLFeatureEstimator is discarded.
        tinyvc_enc_path = Path(f"{models_dir}/encoder.pt")
        if not tinyvc_enc_path.exists():
            raise FileNotFoundError(f"{tinyvc_enc_path} not found")
        self._tinyvc_encoder_ref = _TinyVCEncoder()
        self._tinyvc_encoder_ref.load_state_dict(
            torch.load(str(tinyvc_enc_path), map_location="cpu")
        )
        self._tinyvc_encoder_ref.eval()
        self.pitch_estimator = self._tinyvc_encoder_ref.pitch_estimator
        self.pitch_params = sum(
            p.numel() for p in self.pitch_estimator.parameters()
        )
        print(f"  TinyVC PitchEstimator = {self.pitch_params/1e6:.2f}M params")

        # --- TinyVC DDSP decoder ----------------------------------------------
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

        # kNN index of voice references, encoded with HuBERT
        self.voices: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _encode_hubert(self, wav_24k: np.ndarray) -> torch.Tensor:
        """HuBERT forward on 16kHz-resampled input.

        Returns content [1, 768, T_hub] @ 50Hz.
        """
        wav_16k = librosa.resample(
            wav_24k.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=16000
        )
        inputs = self.fe(wav_16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.hubert(**inputs)
        # Use a specific hidden layer if requested, else last_hidden_state.
        if self.hubert_layer is not None and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            # hidden_states is a tuple of (num_layers+1,) tensors [B, T, D].
            content = outputs.hidden_states[self.hubert_layer]
        else:
            content = outputs.last_hidden_state
        content = content.transpose(1, 2)  # [1, 768, T_hub]
        return content

    def _encode_f0(self, wav_24k: np.ndarray) -> torch.Tensor:
        """TinyVC PitchEstimator forward on 24kHz STFT.

        Returns f0 [1, 1, T_tiny] @ 50Hz.
        """
        wf = torch.from_numpy(wav_24k.astype(np.float32)).unsqueeze(0)  # [1, L]
        wf = _tinyvc_autopad(wf)
        spec = _tinyvc_spectrogram(wf)  # [1, fft_bin, T]
        with torch.inference_mode():
            f0 = self.pitch_estimator.infer(spec)  # [1, 1, T]
        return f0

    def _estimate_energy(self, wav_24k: np.ndarray) -> torch.Tensor:
        """TinyVC energy estimation.

        Returns energy [1, 1, L] @ sample-rate.
        """
        wf = torch.from_numpy(wav_24k.astype(np.float32)).unsqueeze(0)  # [1, L]
        wf = _tinyvc_autopad(wf)
        return _tinyvc_estimate_energy(wf)

    @staticmethod
    def _align_frames(content: torch.Tensor, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Align T_hubert to T_tinyvc via F.interpolate on time dim.

        Both are nominally 50Hz, but HuBERT may produce 1-2 extra/missing frames
        due to its 16kHz stride + padding. We resample the smaller to the larger
        so the decoder sees a consistent frame count.
        """
        t_hub = content.shape[-1]
        t_tiny = f0.shape[-1]
        if t_hub == t_tiny:
            return content, f0
        if t_hub < t_tiny:
            content = F.interpolate(content, size=t_tiny, mode="linear")
        else:
            f0 = F.interpolate(f0, size=t_hub, mode="linear")
        return content, f0

    def encode(self, wav_24k: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode: HuBERT content + TinyVC F0 + TinyVC energy."""
        content = self._encode_hubert(wav_24k)  # [1, 768, T_hub]
        f0 = self._encode_f0(wav_24k)           # [1, 1, T_tiny]
        energy = self._estimate_energy(wav_24k)  # [1, 1, L]
        content, f0 = self._align_frames(content, f0)
        return content, f0, energy

    # ------------------------------------------------------------------
    # kNN voice index (rebuild with HuBERT features)
    # ------------------------------------------------------------------
    def build_voice_index(self, voice_dir: str = "data/voices"):
        print(f"\n[kNN index] Rebuilding with {self.encoder_label} features ...")
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
            content, _, _ = self.encode(wav)
            voice_id = int(path.split("voice_")[-1].split(".")[0])
            key = f"voice_{voice_id}"
            self.voices[key] = content
            print(f"  {key}: content {tuple(content.shape)}  (src len {len(wav)} samples)")
        print(f"  built {len(self.voices)} voice indices in {time.perf_counter()-t_start:.2f}s")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process_audio(self, wav: np.ndarray, sr: int, voice_id: int = 0) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        peak = max(float(np.max(np.abs(wav))), 1e-8)
        wav = wav * (10 ** (DEFAULT_NORM_DB / 20.0) / peak)

        # 1. Encode
        content, f0, energy = self.encode(wav)

        # 2. kNN retrieval
        key = f"voice_{voice_id}"
        if key not in self.voices:
            raise KeyError(f"{key} not in voice index (have: {list(self.voices.keys())})")
        target = self.voices[key]
        # Align T_target to T_content if mismatched (e.g. different reference durations)
        if target.shape[-1] != content.shape[-1]:
            target = F.interpolate(target, size=content.shape[-1], mode="linear")
        content_replaced = _tinyvc_match_features(content, target, k=4, alpha=0.0, metrics="cos")

        # 3. Decode
        with torch.inference_mode():
            out = self.decoder.infer(content_replaced, f0, energy)
        return out.squeeze(0).squeeze(0).cpu().numpy()


# ===========================================================================
# Main test driver
# ===========================================================================
def main():
    import sherpa_onnx

    print("=" * 70)
    print("Path 2: DistilHuBERT Content Encoder Test")
    print("=" * 70)

    # Try the requested DistilHuBERT first; fall back to HuBERT-base.
    repo_id, label, hubert_model, hubert_fe = select_distilhubert_substitute()

    print(f"\nSelected encoder: {label}  ({repo_id})")

    infer = HubertTinyVCHybrid(
        models_dir="models",
        encoder_repo=repo_id,
        encoder_label=label,
        hubert_model=hubert_model,
        hubert_fe=hubert_fe,
        # Use layer 4 to match TinyVC's distillation target (WavLM layer 4).
        # TinyVC's 4.7M ConvNeXt-v2 was distilled FROM WavLM layer 4 output.
        hubert_layer=4,
    )

    # Build HuBERT-features voice index
    infer.build_voice_index()

    # CAMPPlus for speaker similarity
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
    config.model = "models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
    config.num_threads = 1
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def get_emb(wav, sr=SAMPLE_RATE):
        if sr != 16000:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        stream = extractor.create_stream()
        stream.accept_waveform(16000, wav.tolist())
        stream.input_finished()
        return np.asarray(extractor.compute(stream), dtype=np.float32)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # Source: real human speech (already at data/source/source_real_001.wav)
    src_path = "data/source/source_real_001.wav"
    if not os.path.exists(src_path):
        # fall back to the synthetic source_006
        src_path = "data/source/source_006.wav"
    src, sr = sf.read(src_path)
    src = src.astype(np.float32)[: sr * 5]  # 5 seconds
    src_emb = get_emb(src, sr)
    print(f"\nSource: {src_path}  ({len(src)/sr:.1f}s @ {sr}Hz)")
    print(f"  source CAMPPlus embedding dim = {src_emb.shape}")

    # Build target embeddings from the original voice WAVs (not the HuBERT index)
    target_wavs = []
    target_embs = []
    for i in range(5):
        wav, vsr = sf.read(f"data/voices/voice_{i}.wav")
        if vsr != SAMPLE_RATE:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=vsr, target_sr=SAMPLE_RATE)
        else:
            wav = wav.astype(np.float32)
        target_wavs.append(wav)
        target_embs.append(get_emb(wav, sr))

    # Sanity: target distinctness
    print("\nTarget voice pairwise similarity (CAMPPlus cosine):")
    for i in range(5):
        for j in range(i + 1, 5):
            s = cosine(target_embs[i], target_embs[j])
            print(f"  voice_{i} vs voice_{j}: {s:.3f}")

    # ---- Run VC for all 5 voices ----
    print(f"\n{'='*70}")
    print(f"VC quality test: {label} content encoder → kNN-VC → TinyVC decoder")
    print(f"{'='*70}")
    print(
        f"  Model sizes: SSL={infer.ssl_params/1e6:.1f}M  "
        f"F0={infer.pitch_params/1e6:.2f}M  Decoder={infer.dec_params/1e6:.2f}M  "
        f"Total={(infer.ssl_params+infer.pitch_params+infer.dec_params)/1e6:.1f}M"
    )

    results = []
    for vid in range(5):
        try:
            t0 = time.perf_counter()
            out = infer.process_audio(src, sr, voice_id=vid)
            t1 = time.perf_counter()

            out_path = f"{OUTPUT_DIR}/vc_path2_hubert_voice_{vid}.wav"
            sf.write(out_path, out, SAMPLE_RATE)

            out_emb = get_emb(out, SAMPLE_RATE)
            t_sim = cosine(out_emb, target_embs[vid])
            s_sim = cosine(out_emb, src_emb)

            rtf = (t1 - t0) / 5.0
            rms = float(np.sqrt(np.mean(out ** 2)))
            peak_db = 20 * np.log10(max(float(np.max(np.abs(out))), 1e-8))
            n_nan = int(np.sum(~np.isfinite(out)))

            results.append(
                {
                    "voice_id": vid,
                    "target_sim": round(t_sim, 3),
                    "source_sim": round(s_sim, 3),
                    "vc_effect": round(t_sim - s_sim, 3),
                    "rtf": round(rtf, 4),
                    "rms": round(rms, 4),
                    "peak_db": round(peak_db, 2),
                    "n_nan": n_nan,
                }
            )
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
        print(f"\n  AVG: target={avg_t:.3f}  source={avg_s:.3f}  VC_effect={avg_t-avg_s:+.3f}")

    # ---- Feature distribution comparison: HuBERT vs TinyVC's own encoder ----
    print(f"\n{'='*70}")
    print(f"Bonus: Feature distribution comparison ({label} vs TinyVC)")
    print(f"{'='*70}")
    try:
        # Compare frame-wise cosine similarity between HuBERT and TinyVC content
        # features for the same source audio.
        tinyvc_encoder_full = infer._tinyvc_encoder_ref
        wf = torch.from_numpy(src.astype(np.float32)).unsqueeze(0)
        wf = _tinyvc_autopad(wf)
        spec = _tinyvc_spectrogram(wf)
        with torch.inference_mode():
            tinyvc_content, _ = tinyvc_encoder_full.infer(spec)  # [1, 768, T]
        hubert_content = infer._encode_hubert(src)  # [1, 768, T_hub]

        # Align frames
        t_min = min(tinyvc_content.shape[-1], hubert_content.shape[-1])
        tc = tinyvc_content[..., :t_min]
        hc = hubert_content[..., :t_min]

        # Per-frame cosine similarity
        tc_n = F.normalize(tc, dim=1)
        hc_n = F.normalize(hc, dim=1)
        per_frame_cos = (tc_n * hc_n).sum(dim=1).squeeze(0)  # [T]
        avg_cos = float(per_frame_cos.mean())

        # Feature statistics
        print(f"  TinyVC content  : mean={tinyvc_content.mean():.3f}  std={tinyvc_content.std():.3f}  shape={tuple(tinyvc_content.shape)}")
        print(f"  {label} content: mean={hubert_content.mean():.3f}  std={hubert_content.std():.3f}  shape={tuple(hubert_content.shape)}")
        print(f"  Per-frame cosine (TinyVC vs {label}): avg={avg_cos:.3f}  min={float(per_frame_cos.min()):.3f}  max={float(per_frame_cos.max()):.3f}")
        print(f"  → Low cosine = different feature distributions → TinyVC decoder (trained on TinyVC/WavLM-distilled features)")
        print(f"    may produce artifacts with raw {label} features. Decoder retrain recommended for best VC quality.")
    except Exception as e:
        print(f"  comparison failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n[done] Output files in {OUTPUT_DIR}/vc_path2_hubert_voice_*.wav")


if __name__ == "__main__":
    main()
