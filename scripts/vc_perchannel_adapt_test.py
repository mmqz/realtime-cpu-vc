#!/usr/bin/env python3
"""
SolD+F: Per-channel moment matching + learnable BatchNorm1d.

SolC (baseline): adapted = (hubert - μ_h) / σ_h × σ_t + μ_t  (single scalar)
SolD:            adapted[c] = (hubert[c] - μ_h[c]) / σ_h[c] × σ_t[c] + μ_t[c]  (768 indep.)
SolF:            BatchNorm1d with frozen running stats (μ_h[c], σ_h[c]^2) and
                 learnable affine (weight, bias). Init from SolD, then fine-tune
                 with SGD against TinyVC features. Optimal weight → per-channel
                 cross-cov, not per-channel std ratio → should beat SolD.

Pipeline (all 4 methods):
    HuBERT-base (layer 4, 768-d @ 50Hz) → adapt (scalar / per-channel / BN)
        → kNN-VC replace against (similarly adapted) voice references
        → TinyVC PitchEstimator → f0  (TinyVC encoder)
        → TinyVC estimate_energy       → energy
        → TinyVC DDSP Decoder          → 24 kHz mono audio

Notes
-----
- Bug fixes vs. task spec:
  * `autopad_waveform` expects `[N, L]`, not `[N, 1, L]`. We use `[1, L]`.
  * `estimate_energy` takes raw wave `[N, L]`, not a spectrogram. We pass `wf`.
- `_encode_tinyvc` does NOT peak-normalize; `process_audio`/`build_indices`
  do it once upstream so HuBERT and TinyVC encoders see the same audio.
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
import librosa

# -----------------------------------------------------------------------------
# Optional-deps stubbing (same as vc_adapted_features_test.py).
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
sys.path.insert(0, "src")

from transformers import HubertModel, Wav2Vec2FeatureExtractor  # noqa: E402

from module.tinyvc.encoder import Encoder as TinyVCEncoder  # noqa: E402
from module.tinyvc.decoder import Decoder as TinyVCDecoder  # noqa: E402
from module.tinyvc.feature_retrieval import match_features as tinyvc_match_features  # noqa: E402
from module.utils.spectrogram import spectrogram as tinyvc_spectrogram  # noqa: E402
from module.utils.auto_padding import autopad_waveform as tinyvc_autopad  # noqa: E402
from module.utils.energy_estimation import estimate_energy as tinyvc_estimate_energy  # noqa: E402

torch.set_num_threads(2)

SAMPLE_RATE = 24000
HUBERT_LAYER = 4
DEFAULT_NORM_DB = -3.0
STATS_DUR_SEC = 10.0


class PerChannelAdaptedInfer:
    """HuBERT-base content + TinyVC F0/energy + TinyVC DDSP decoder, with
    scalar / per-channel / learnable-BatchNorm1d adaptation of HuBERT features.
    """

    def __init__(self, models_dir: str = "models"):
        # --- HuBERT-base content encoder (SSL) ------------------------------
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert.eval()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )

        # --- TinyVC Encoder (combined SSL + pitch) + Decoder ---------------
        self.tinyvc_enc = TinyVCEncoder()
        self.tinyvc_enc.load_state_dict(
            torch.load(f"{models_dir}/encoder.pt", map_location="cpu")
        )
        self.tinyvc_enc.eval()
        self.decoder = TinyVCDecoder()
        self.decoder.load_state_dict(
            torch.load(f"{models_dir}/decoder.pt", map_location="cpu")
        )
        self.decoder.eval()

        # --- Compute per-channel + scalar stats ----------------------------
        self._compute_stats()

        # --- Build BatchNorm1d (SolF) initialized to match SolD ------------
        self._build_batchnorm()

        # --- Voice kNN indices (one per method) ---------------------------
        self.voices: dict[str, dict[int, torch.Tensor]] = {
            "tinyvc": {},
            "scalar": {},
            "perchannel": {},
            "batchnorm": {},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_peak(wav: np.ndarray) -> np.ndarray:
        wav = wav.astype(np.float32)
        peak = max(float(np.max(np.abs(wav))), 1e-8)
        return wav * (10 ** (DEFAULT_NORM_DB / 20.0) / peak)

    def _encode_tinyvc(
        self, wav_24k: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TinyVC encoder. Returns (ssl[1,768,T], f0[1,1,T], wf[1,L'])."""
        wav_24k = np.asarray(wav_24k, dtype=np.float32)
        wf = torch.from_numpy(wav_24k).unsqueeze(0)  # [1, L]
        wf = tinyvc_autopad(wf)                       # [1, L']
        spec = tinyvc_spectrogram(wf)                 # [1, fft, frames]
        with torch.no_grad():
            ssl, f0 = self.tinyvc_enc.infer(spec)
        return ssl, f0, wf

    def _encode_hubert(self, wav_24k: np.ndarray) -> torch.Tensor:
        wav_16k = librosa.resample(
            wav_24k.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=16000
        )
        inputs = self.fe(wav_16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.hubert(**inputs, output_hidden_states=True)
            content = outputs.hidden_states[HUBERT_LAYER].transpose(1, 2)
        return content  # [1, 768, T_hub]

    # ------------------------------------------------------------------
    # Distribution statistics
    # ------------------------------------------------------------------
    def _compute_stats(self) -> None:
        """Compute per-channel (768) and scalar mean/std for HuBERT + TinyVC."""
        all_h: list[torch.Tensor] = []
        all_t: list[torch.Tensor] = []
        for i in range(5):
            wav, sr = sf.read(f"data/voices/voice_{i}.wav")
            if sr != SAMPLE_RATE:
                wav = librosa.resample(
                    wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
                )
            wav = self._normalize_peak(wav)[: int(SAMPLE_RATE * STATS_DUR_SEC)]
            h = self._encode_hubert(wav)
            t, _, _ = self._encode_tinyvc(wav)
            min_t = min(h.shape[-1], t.shape[-1])
            all_h.append(h[:, :, :min_t])
            all_t.append(t[:, :, :min_t])
        all_h_t = torch.cat(all_h, dim=-1)  # [1, 768, total_T]
        all_t_t = torch.cat(all_t, dim=-1)

        # Per-channel stats [768]
        self.h_mean_pc = all_h_t.mean(dim=(0, 2))
        self.h_std_pc = all_h_t.std(dim=(0, 2))
        self.t_mean_pc = all_t_t.mean(dim=(0, 2))
        self.t_std_pc = all_t_t.std(dim=(0, 2))

        # Scalar stats
        self.h_mean_scalar = float(self.h_mean_pc.mean().item())
        self.h_std_scalar = float(self.h_std_pc.mean().item())
        self.t_mean_scalar = float(self.t_mean_pc.mean().item())
        self.t_std_scalar = float(self.t_std_pc.mean().item())

        print(
            f"  Per-channel HuBERT: mean[{self.h_mean_pc.min():+.4f}, "
            f"{self.h_mean_pc.max():+.4f}], std[{self.h_std_pc.min():.4f}, "
            f"{self.h_std_pc.max():.4f}]"
        )
        print(
            f"  Per-channel TinyVC: mean[{self.t_mean_pc.min():+.4f}, "
            f"{self.t_mean_pc.max():+.4f}], std[{self.t_std_pc.min():.4f}, "
            f"{self.t_std_pc.max():.4f}]"
        )
        print(
            f"  Scalar: HuBERT μ={self.h_mean_scalar:+.4f} "
            f"σ={self.h_std_scalar:.4f}, TinyVC μ={self.t_mean_scalar:+.4f} "
            f"σ={self.t_std_scalar:.4f}"
        )

        # Cache full tensors for BatchNorm1d fine-tuning
        self._all_h = all_h_t
        self._all_t = all_t_t

    # ------------------------------------------------------------------
    # BatchNorm1d (SolF)
    # ------------------------------------------------------------------
    def _build_batchnorm(self) -> None:
        """SolF: BatchNorm1d initialized to mathematically equal SolD.

        eval-mode forward:
            y[c] = (x[c] - running_mean[c]) / sqrt(running_var[c] + eps)
                   * weight[c] + bias[c]
        Set running_mean = μ_h[c], running_var = σ_h[c]^2,
            weight       = σ_t[c], bias        = μ_t[c].
        """
        self.bn = torch.nn.BatchNorm1d(
            768, eps=1e-8, affine=True, track_running_stats=True
        )
        with torch.no_grad():
            self.bn.running_mean.copy_(self.h_mean_pc)
            self.bn.running_var.copy_(self.h_std_pc ** 2)
            self.bn.weight.copy_(self.t_std_pc)
            self.bn.bias.copy_(self.t_mean_pc)
        self.bn.eval()  # use frozen running stats, not batch stats

        # Closed-form optimal affine (per-channel OLS of standardized HuBERT
        # → TinyVC features). For y = w * z + b with z = (h - μ_h)/σ_h, the
        # MSE-optimal solution is:
        #     w*[c] = Cov(z[c], t[c]) / Var(z[c])  =  cross-cov(z, t)  (Var=1)
        #     b*[c] = μ_t[c]
        # This is what an unfrozen-affine BatchNorm1d trained with sufficient
        # gradient descent would converge to (and is strictly better than the
        # moment-matching σ_t init when corr(h,t) < 1 per channel).
        h = self._all_h  # [1, 768, T]
        t = self._all_t
        h_z = (h - self.h_mean_pc.view(1, -1, 1)) / (
            self.h_std_pc.view(1, -1, 1) + 1e-8
        )
        t_centered = t - self.t_mean_pc.view(1, -1, 1)
        cross_cov = (h_z * t_centered).mean(dim=(0, 2))  # [768]
        self._cross_cov = cross_cov.detach()
        self._opt_weight = cross_cov.detach()
        self._opt_bias = self.t_mean_pc.detach()

        # Replace affine with closed-form optimum.
        with torch.no_grad():
            self.bn.weight.copy_(self._opt_weight)
            self.bn.bias.copy_(self._opt_bias)

        # Diagnostic: how different is closed-form from moment-matching?
        ratio = (self._opt_weight / (self.t_std_pc + 1e-8)).clamp(-5, 5)
        print(
            f"  [SolF closed-form] w/σ_t ratio: "
            f"min={ratio.min():.3f}, median={ratio.median():.3f}, "
            f"max={ratio.max():.3f}  (<1 means moment-match over-amplifies)"
        )

    def train_batchnorm_sgd(
        self, n_steps: int = 100, lr: float = 5e-2
    ) -> None:
        """Sanity check: fine-tune BatchNorm1d affine params with Adam and
        verify that SGD approaches the closed-form optimum.

        Running stats are frozen (eval-mode forward uses running_mean/var).
        We restore the moment-matching init first so the trajectory starts
        from SolD's solution and demonstrates convergence to SolF's optimum.
        """
        # Reset to moment-matching init.
        with torch.no_grad():
            self.bn.weight.copy_(self.t_std_pc)
            self.bn.bias.copy_(self.t_mean_pc)
        self.bn.weight.requires_grad = True
        self.bn.bias.requires_grad = True
        self.bn.eval()
        opt = torch.optim.Adam([self.bn.weight, self.bn.bias], lr=lr)
        h = self._all_h.detach()
        t = self._all_t.detach()
        init_loss = F.mse_loss(self.bn(h), t).item()
        target_loss = F.mse_loss(
            self._opt_weight.view(1, -1, 1) * (
                (h - self.h_mean_pc.view(1, -1, 1))
                / (self.h_std_pc.view(1, -1, 1) + 1e-8)
            ) + self._opt_bias.view(1, -1, 1),
            t,
        ).item()
        print(
            f"  [SolF SGD sanity] init MSE={init_loss:.6f}, "
            f"closed-form target MSE={target_loss:.6f}"
        )
        for step in range(n_steps):
            opt.zero_grad()
            out = self.bn(h)
            loss = F.mse_loss(out, t)
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == n_steps - 1:
                w_drift = float(
                    (self.bn.weight - self._opt_weight).abs().mean()
                )
                print(
                    f"  [SolF SGD sanity] step={step:03d} loss={loss.item():.6f} "
                    f"w_drift_from_optimal={w_drift:.4f}"
                )
        # Restore the closed-form optimum for downstream inference.
        with torch.no_grad():
            self.bn.weight.copy_(self._opt_weight)
            self.bn.bias.copy_(self._opt_bias)
        self.bn.eval()

    # ------------------------------------------------------------------
    # Adaptation variants
    # ------------------------------------------------------------------
    def adapt_scalar(self, content: torch.Tensor) -> torch.Tensor:
        """SolC: scalar moment match."""
        return (
            (content - self.h_mean_scalar) / self.h_std_scalar * self.t_std_scalar
            + self.t_mean_scalar
        )

    def adapt_perchannel(self, content: torch.Tensor) -> torch.Tensor:
        """SolD: per-channel moment match (768 indep. μ/σ)."""
        mean = self.h_mean_pc.view(1, -1, 1)
        std = self.h_std_pc.view(1, -1, 1)
        t_mean = self.t_mean_pc.view(1, -1, 1)
        t_std = self.t_std_pc.view(1, -1, 1)
        return (content - mean) / (std + 1e-8) * t_std + t_mean

    def adapt_batchnorm(self, content: torch.Tensor) -> torch.Tensor:
        """SolF: BatchNorm1d with (frozen) HuBERT running stats + learned affine."""
        return self.bn(content)

    # ------------------------------------------------------------------
    # Voice kNN indices (per method)
    # ------------------------------------------------------------------
    def build_indices(self) -> None:
        print("\n[kNN] Building voice indices (4 methods) ...")
        t_start = time.perf_counter()
        for path in sorted(glob.glob("data/voices/voice_*.wav")):
            wav, sr = sf.read(path)
            if sr != SAMPLE_RATE:
                wav = librosa.resample(
                    wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
                )
            wav = self._normalize_peak(wav)
            vid = int(path.split("voice_")[-1].split(".")[0])

            t_content, _, _ = self._encode_tinyvc(wav)
            h_content = self._encode_hubert(wav)
            h_scalar = self.adapt_scalar(h_content)
            h_perchannel = self.adapt_perchannel(h_content)
            h_batchnorm = self.adapt_batchnorm(h_content)

            min_t = min(
                t_content.shape[-1],
                h_scalar.shape[-1],
                h_perchannel.shape[-1],
                h_batchnorm.shape[-1],
            )
            self.voices["tinyvc"][vid] = t_content[:, :, :min_t]
            self.voices["scalar"][vid] = h_scalar[:, :, :min_t]
            self.voices["perchannel"][vid] = h_perchannel[:, :, :min_t]
            self.voices["batchnorm"][vid] = h_batchnorm[:, :, :min_t]
        print(
            f"  built {len(self.voices['tinyvc'])} voices × 4 methods in "
            f"{time.perf_counter()-t_start:.2f}s"
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process_audio(
        self,
        wav: np.ndarray,
        sr: int,
        voice_id: int,
        method: str = "perchannel",
    ) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        wav = self._normalize_peak(wav)

        # Encode content + f0 + wf (sample-level wave for energy)
        if method == "tinyvc":
            content, f0, wf = self._encode_tinyvc(wav)
        else:
            h_content = self._encode_hubert(wav)
            _, f0, wf = self._encode_tinyvc(wav)
            if method == "scalar":
                content = self.adapt_scalar(h_content)
            elif method == "perchannel":
                content = self.adapt_perchannel(h_content)
            elif method == "batchnorm":
                content = self.adapt_batchnorm(h_content)
            else:
                raise ValueError(f"unknown method: {method}")

        # Align content & f0 to a common frame length (SolC pattern).
        # Energy stays at sample level; the decoder pools it itself.
        if content.shape[-1] != f0.shape[-1]:
            if content.shape[-1] < f0.shape[-1]:
                content = F.interpolate(
                    content, size=f0.shape[-1], mode="linear"
                )
            else:
                f0 = F.interpolate(f0, size=content.shape[-1], mode="linear")

        # kNN-VC replace (target can be any length).
        target = self.voices[method][voice_id]
        content_replaced = tinyvc_match_features(
            content, target, k=4, alpha=0.0, metrics="cos"
        )

        # Energy at sample level (decoder's max_pool1d(480, 480) yields frames).
        energy = tinyvc_estimate_energy(wf)

        with torch.no_grad():
            wav_out = self.decoder.infer(content_replaced, f0, energy)
        return wav_out.squeeze(0).squeeze(0).cpu().numpy()


def main() -> None:
    import sherpa_onnx

    print("=" * 72)
    print("SolD+F: Per-Channel Adaptation + Learnable BatchNorm1d")
    print("  4 methods: tinyvc (baseline) | scalar (SolC) | "
          "perchannel (SolD) | batchnorm (SolF)")
    print("=" * 72)

    print("\n[1/4] Loading models:")
    infer = PerChannelAdaptedInfer(models_dir="models")

    print("\n[2/4] Sanity-checking SGD fine-tune of BatchNorm1d affine "
          "(should approach closed-form optimum):")
    infer.train_batchnorm_sgd(n_steps=100, lr=5e-2)

    print("\n[3/4] Building voice indices:")
    infer.build_indices()

    # ---- CAMPPlus speaker embedding extractor ----------------------------
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
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    src, sr = sf.read("data/source/source_real_001.wav")
    src = src.astype(np.float32)[: sr * 5]
    src_emb = get_emb(src, sr)
    target_embs = [
        get_emb(sf.read(f"data/voices/voice_{i}.wav")[0]) for i in range(5)
    ]
    print(f"\n  Source: data/source/source_real_001.wav "
          f"({len(src)/sr:.1f}s @ {sr}Hz, emb dim={src_emb.shape[0]})")

    print("\n[4/4] VC test (4 methods × 5 voices):")
    summary: dict[str, tuple[float, float, float]] = {}
    for method in ["tinyvc", "scalar", "perchannel", "batchnorm"]:
        print(f"\n--- Method: {method} ---")
        results: list[tuple[float, float]] = []
        for vid in range(5):
            try:
                t0 = time.perf_counter()
                out = infer.process_audio(src, sr, vid, method=method)
                t1 = time.perf_counter()
                out_emb = get_emb(out, SAMPLE_RATE)
                t_sim = cosine(out_emb, target_embs[vid])
                s_sim = cosine(out_emb, src_emb)
                results.append((t_sim, s_sim))
                print(
                    f"  voice_{vid}: target={t_sim:.3f}, source={s_sim:.3f}, "
                    f"VC_effect={t_sim-s_sim:+.3f}, RTF={(t1-t0)/5:.4f}, "
                    f"RMS={float(np.sqrt(np.mean(out**2))):.4f}, "
                    f"NaN={int(np.sum(~np.isfinite(out)))}"
                )
            except Exception as e:
                print(f"  voice_{vid}: FAILED - {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
        if results:
            avg_t = float(np.mean([r[0] for r in results]))
            avg_s = float(np.mean([r[1] for r in results]))
            print(
                f"  AVG: target={avg_t:.3f}, source={avg_s:.3f}, "
                f"VC_effect={avg_t-avg_s:+.3f}"
            )
            summary[method] = (avg_t, avg_s, avg_t - avg_s)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Method':<14} {'Target':>8} {'Source':>8} {'VC_effect':>10}")
    print("-" * 42)
    for m, (t, s, e) in summary.items():
        print(f"{m:<14} {t:>8.3f} {s:>8.3f} {e:>+10.3f}")
    print("-" * 42)

    # Quick comparison vs SolC scalar baseline
    if "scalar" in summary and "perchannel" in summary:
        delta = summary["perchannel"][2] - summary["scalar"][2]
        print(f"\nSolD - SolC (per-channel − scalar) = {delta:+.3f}")
    if "perchannel" in summary and "batchnorm" in summary:
        delta = summary["batchnorm"][2] - summary["perchannel"][2]
        print(f"SolF - SolD (BatchNorm1d − per-channel) = {delta:+.3f}")


if __name__ == "__main__":
    main()
