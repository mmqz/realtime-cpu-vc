#!/usr/bin/env python3
"""
SolC: Feature distribution adaptation.

Path 2 found that raw HuBERT-base layer-4 features (mean≈0, std≈0.43) have a
much wider spread than TinyVC's distilled 4.7M ConvNeXt features (mean≈0,
std≈0.18). The TinyVC DDSP decoder was trained on TinyVC-distribution features,
so raw HuBERT features lead to decoder artifacts.

This script adapts HuBERT features to match TinyVC's first two moments
(mean + std), making them look statistically like TinyVC features to the
decoder, WITHOUT any retraining.

Adaptation formula:
    adapted = (hubert - hubert_mean) / hubert_std * tinyvc_std + tinyvc_mean

Pipeline:
    HuBERT-base (layer 4, 768-d @ 50Hz) → moment-match adapt
        → kNN-VC replace against voice references (also adapted)
        → TinyVC PitchEstimator (ConvNeXt over STFT) → f0
        → TinyVC estimate_energy → energy
        → TinyVC DDSP Decoder (SourceNet + FilterNet)
        → 24 kHz mono audio

Notes
-----
- The task spec referenced `SSLFeatureEstimator` and `MelSpec`, but the
  saved `encoder.pt` is the *combined* `Encoder` (which returns `(ssl, f0)`
  from `infer()`), and `module.utils.spectrogram` exposes `spectrogram`
  (no `MelSpec`). Both substitutions are equivalent in spirit.
- `Decoder.infer(content, f0, energy)` requires 3 positional args; we pass
  TinyVC's `estimate_energy` output for the third one.
- `module/utils/__init__.py` eagerly imports `f0_estimation`, which in turn
  imports `torchfcpe` and `pyworld`. We only use the bundled ConvNeXt pitch
  estimator inside the Encoder, so we stub those optional deps (same pattern
  as `vc_distilhubert_test.py`).
"""
from __future__ import annotations

import os
import sys
import types
import time
import glob
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
# Make local `src` importable for prototype-side helpers (harmless if unused).
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
FRAME_SIZE = 480  # 50 Hz @ 24 kHz
DEFAULT_NORM_DB = -3.0
HUBERT_LAYER = 4  # TinyVC's ConvNeXt was distilled from WavLM layer 4
STATS_DUR_SEC = 10.0  # seconds per voice used to estimate distribution stats

OUTPUT_DIR = "/home/z/my-project/download"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class AdaptedHuBERTV1Infer:
    """HuBERT-base content + TinyVC F0/energy + TinyVC DDSP decoder, with
    moment-matching adaptation of HuBERT features to the TinyVC distribution.
    """

    def __init__(self, models_dir: str = "models"):
        # --- HuBERT-base content encoder (SSL) ------------------------------
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert.eval()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        self.ssl_params = sum(p.numel() for p in self.hubert.parameters())
        print(
            f"  HuBERT-base = facebook/hubert-base-ls960 "
            f"({self.ssl_params/1e6:.1f}M params, layer={HUBERT_LAYER})"
        )

        # --- TinyVC Encoder (combined SSL + pitch) + Decoder ---------------
        enc_path = Path(f"{models_dir}/encoder.pt")
        if not enc_path.exists():
            raise FileNotFoundError(f"{enc_path} not found")
        self.tinyvc_enc = TinyVCEncoder()
        self.tinyvc_enc.load_state_dict(
            torch.load(str(enc_path), map_location="cpu")
        )
        self.tinyvc_enc.eval()
        enc_params = sum(p.numel() for p in self.tinyvc_enc.parameters())
        print(f"  TinyVC Encoder = {enc_params/1e6:.2f}M params")

        dec_path = Path(f"{models_dir}/decoder.pt")
        if not dec_path.exists():
            raise FileNotFoundError(f"{dec_path} not found")
        self.decoder = TinyVCDecoder()
        self.decoder.load_state_dict(
            torch.load(str(dec_path), map_location="cpu")
        )
        self.decoder.eval()
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"  TinyVC Decoder = {dec_params/1e6:.2f}M params")

        # --- Adaptation parameters (filled by _compute_*_stats) -----------
        self.hubert_mean: float | None = None
        self.hubert_std: float | None = None
        self.tinyvc_mean: float | None = None
        self.tinyvc_std: float | None = None

        # Voice kNN index: voice_id -> adapted content [1, 768, T]
        self.voices: dict[str, torch.Tensor] = {}

        # --- Compute target (TinyVC) + source (HuBERT) feature stats ------
        # TinyVC stats first because adaptation maps HuBERT→TinyVC.
        self._compute_tinyvc_stats()
        self._compute_hubert_stats()
        print(
            f"  Adaptation: "
            f"adapted = (hubert - {self.hubert_mean:.4f}) / {self.hubert_std:.4f} "
            f"* {self.tinyvc_std:.4f} + {self.tinyvc_mean:.4f}"
        )

    # ------------------------------------------------------------------
    # Distribution statistics
    # ------------------------------------------------------------------
    def _normalize_peak(self, wav: np.ndarray) -> np.ndarray:
        wav = wav.astype(np.float32)
        peak = max(float(np.max(np.abs(wav))), 1e-8)
        return wav * (10 ** (DEFAULT_NORM_DB / 20.0) / peak)

    def _compute_tinyvc_stats(self):
        """Compute mean + std of TinyVC SSL features across all voice fixtures."""
        all_features = []
        for i in range(5):
            wav, sr = sf.read(f"data/voices/voice_{i}.wav")
            if sr != SAMPLE_RATE:
                wav = librosa.resample(
                    wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
                )
            # Use first STATS_DUR_SEC seconds for speed.
            wav = self._normalize_peak(wav)[: int(SAMPLE_RATE * STATS_DUR_SEC)]
            wf = torch.from_numpy(wav).unsqueeze(0)
            wf = tinyvc_autopad(wf)
            spec = tinyvc_spectrogram(wf)
            with torch.no_grad():
                content, _ = self.tinyvc_enc.infer(spec)  # [1, 768, T]
            all_features.append(content)
        all_features = torch.cat(all_features, dim=-1)  # [1, 768, total_T]
        self.tinyvc_mean = all_features.mean().item()
        self.tinyvc_std = all_features.std().item()
        print(
            f"  TinyVC feature stats: mean={self.tinyvc_mean:+.4f}, "
            f"std={self.tinyvc_std:.4f}  (n={all_features.numel():,} values)"
        )

    def _compute_hubert_stats(self):
        """Compute mean + std of HuBERT layer-4 features across all voice fixtures."""
        all_features = []
        for i in range(5):
            wav, sr = sf.read(f"data/voices/voice_{i}.wav")
            if sr != SAMPLE_RATE:
                wav = librosa.resample(
                    wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
                )
            wav = self._normalize_peak(wav)[: int(SAMPLE_RATE * STATS_DUR_SEC)]
            wav_16k = librosa.resample(
                wav.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=16000
            )
            inputs = self.fe(wav_16k, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                outputs = self.hubert(**inputs, output_hidden_states=True)
                content = outputs.hidden_states[HUBERT_LAYER].transpose(1, 2)
            all_features.append(content)
        all_features = torch.cat(all_features, dim=-1)
        self.hubert_mean = all_features.mean().item()
        self.hubert_std = all_features.std().item()
        print(
            f"  HuBERT feature stats: mean={self.hubert_mean:+.4f}, "
            f"std={self.hubert_std:.4f}  (n={all_features.numel():,} values)"
        )

    # ------------------------------------------------------------------
    # Adaptation
    # ------------------------------------------------------------------
    def adapt_features(self, content: torch.Tensor) -> torch.Tensor:
        """Moment-match HuBERT content to the TinyVC feature distribution."""
        # adapted = (hubert - hubert_mean) / hubert_std * tinyvc_std + tinyvc_mean
        adapted = (
            (content - self.hubert_mean) / self.hubert_std * self.tinyvc_std
            + self.tinyvc_mean
        )
        return adapted

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode_hubert(self, wav_24k: np.ndarray) -> torch.Tensor:
        wav_16k = librosa.resample(
            wav_24k.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=16000
        )
        inputs = self.fe(wav_16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.hubert(**inputs, output_hidden_states=True)
            content = outputs.hidden_states[HUBERT_LAYER].transpose(1, 2)
        return content  # [1, 768, T_hub]

    def _encode_tinyvc_f0_energy(
        self, wav_24k: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wf = torch.from_numpy(wav_24k.astype(np.float32)).unsqueeze(0)
        wf = tinyvc_autopad(wf)
        spec = tinyvc_spectrogram(wf)
        with torch.no_grad():
            _, f0 = self.tinyvc_enc.infer(spec)  # [1, 1, T_tiny]
        energy = tinyvc_estimate_energy(wf)  # [1, 1, L]
        return f0, energy

    @staticmethod
    def _align_frames(
        content: torch.Tensor, f0: torch.Tensor, energy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align T_hubert to T_tinyvc (and energy) via F.interpolate on time dim.

        Both content and f0 are nominally 50Hz, but HuBERT's 16kHz stride +
        padding produces 1-2 extra/missing frames. Energy is sample-rate;
        it gets max-pooled inside the decoder to frame rate.
        """
        t_hub = content.shape[-1]
        t_tiny = f0.shape[-1]
        if t_hub == t_tiny:
            return content, f0, energy
        if t_hub < t_tiny:
            content = F.interpolate(content, size=t_tiny, mode="linear")
        else:
            f0 = F.interpolate(f0, size=t_hub, mode="linear")
        return content, f0, energy

    def encode(
        self, wav_24k: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """HuBERT content (adapted) + TinyVC F0 + TinyVC energy."""
        content = self._encode_hubert(wav_24k)        # [1, 768, T_hub]
        content = self.adapt_features(content)         # ADAPT
        f0, energy = self._encode_tinyvc_f0_energy(wav_24k)
        content, f0, energy = self._align_frames(content, f0, energy)
        return content, f0, energy

    # ------------------------------------------------------------------
    # kNN voice index (rebuild with ADAPTED HuBERT features)
    # ------------------------------------------------------------------
    def build_voice_index(self, voice_dir: str = "data/voices"):
        print("\n[kNN index] Building with adapted HuBERT features ...")
        t_start = time.perf_counter()
        for path in sorted(glob.glob(f"{voice_dir}/voice_*.wav")):
            wav, sr = sf.read(path)
            if sr != SAMPLE_RATE:
                wav = librosa.resample(
                    wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
                )
            wav = self._normalize_peak(wav)
            content, _, _ = self.encode(wav)
            vid = int(path.split("voice_")[-1].split(".")[0])
            self.voices[f"voice_{vid}"] = content
            print(
                f"  voice_{vid}: {tuple(content.shape)}, "
                f"stats: mean={content.mean():+.4f}, std={content.std():.4f}"
            )
        print(
            f"  built {len(self.voices)} voice indices in "
            f"{time.perf_counter()-t_start:.2f}s"
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process_audio(
        self, wav: np.ndarray, sr: int, voice_id: int = 0
    ) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        wav = self._normalize_peak(wav)

        content, f0, energy = self.encode(wav)

        key = f"voice_{voice_id}"
        if key not in self.voices:
            raise KeyError(
                f"{key} not in voice index (have: {list(self.voices.keys())})"
            )
        target = self.voices[key]
        # Align T_target to T_content if mismatched (different reference durations)
        if target.shape[-1] != content.shape[-1]:
            target = F.interpolate(
                target, size=content.shape[-1], mode="linear"
            )
        content_replaced = tinyvc_match_features(
            content, target, k=4, alpha=0.0, metrics="cos"
        )

        with torch.no_grad():
            wav_out = self.decoder.infer(content_replaced, f0, energy)
        return wav_out.squeeze(0).squeeze(0).cpu().numpy()


def main():
    import sherpa_onnx

    print("=" * 70)
    print("SolC: Feature Distribution Adaptation")
    print("  HuBERT-base → moment-match (mean+std) → kNN-VC → TinyVC decode")
    print("=" * 70)

    print("\n[1/3] Loading models:")
    infer = AdaptedHuBERTV1Infer(models_dir="models")

    print("\n[2/3] Building adapted voice index:")
    infer.build_voice_index()

    # ---- CAMPPlus speaker embedding extractor for similarity scoring --------
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
    config.model = "models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
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

    # ---- Source + target embeddings -------------------------------------
    src, sr = sf.read("data/source/source_real_001.wav")
    src = src.astype(np.float32)[: sr * 5]  # 5 seconds
    src_emb = get_emb(src, sr)
    target_embs = [
        get_emb(sf.read(f"data/voices/voice_{i}.wav")[0]) for i in range(5)
    ]
    print(
        f"\nSource: data/source/source_real_001.wav ({len(src)/sr:.1f}s @ {sr}Hz, "
        f"emb dim={src_emb.shape[0]})"
    )

    print("\n[3/3] VC test (HuBERT-adapted → kNN → TinyVC decode):")
    print(
        f"  Model sizes: HuBERT={infer.ssl_params/1e6:.1f}M  "
        f"TinyVC_enc+dec={(sum(p.numel() for p in infer.tinyvc_enc.parameters()) + sum(p.numel() for p in infer.decoder.parameters()))/1e6:.1f}M"
    )

    results = []
    for vid in range(5):
        try:
            t0 = time.perf_counter()
            out = infer.process_audio(src, sr, voice_id=vid)
            t1 = time.perf_counter()
            out_path = f"{OUTPUT_DIR}/vc_adapted_voice_{vid}.wav"
            sf.write(out_path, out, SAMPLE_RATE)
            out_emb = get_emb(out, SAMPLE_RATE)
            t_sim = cosine(out_emb, target_embs[vid])
            s_sim = cosine(out_emb, src_emb)
            rtf = (t1 - t0) / 5.0
            rms = float(np.sqrt(np.mean(out ** 2)))
            n_nan = int(np.sum(~np.isfinite(out)))
            results.append(
                {
                    "voice_id": vid,
                    "target_sim": t_sim,
                    "source_sim": s_sim,
                    "vc_effect": t_sim - s_sim,
                    "rtf": rtf,
                    "rms": rms,
                    "n_nan": n_nan,
                }
            )
            print(
                f"  voice_{vid}: target={t_sim:.3f}, source={s_sim:.3f}, "
                f"VC_effect={t_sim-s_sim:+.3f}, RTF={rtf:.4f}, "
                f"RMS={rms:.4f}, NaN={n_nan}"
            )
        except Exception as e:
            print(f"  voice_{vid}: FAILED - {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    if results:
        avg_t = float(np.mean([r["target_sim"] for r in results]))
        avg_s = float(np.mean([r["source_sim"] for r in results]))
        print(
            f"\n  AVG: target={avg_t:.3f}, source={avg_s:.3f}, "
            f"VC_effect={avg_t-avg_s:+.3f}"
        )

    # ---- Sanity: adapted vs raw HuBERT feature distribution --------------
    print("\n[adaptation verification]")
    try:
        wav, _ = sf.read("data/voices/voice_0.wav")
        wav = infer._normalize_peak(wav)[: SAMPLE_RATE * 5]
        raw = infer._encode_hubert(wav)
        adapted = infer.adapt_features(raw)
        print(
            f"  raw HuBERT   : mean={raw.mean():+.4f}, std={raw.std():.4f} "
            f"(target: mean={infer.tinyvc_mean:+.4f}, std={infer.tinyvc_std:.4f})"
        )
        print(
            f"  adapted      : mean={adapted.mean():+.4f}, std={adapted.std():.4f} "
            f"(should ≈ TinyVC stats)"
        )
    except Exception as e:
        print(f"  verification failed: {type(e).__name__}: {e}")

    print(f"\n[done] Output files in {OUTPUT_DIR}/vc_adapted_voice_*.wav")


if __name__ == "__main__":
    main()
