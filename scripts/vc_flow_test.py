#!/usr/bin/env python3
"""Path 1: OpenVoice flow + 768→192 projection.

Pipeline
--------
TinyVC encoder (768) → Linear(768→192) → OpenVoice flow (disentangle src,
apply tgt, 192-d) → Linear(192→768) → TinyVC decoder (DDSP) → output wav.

Reuses :class:`vc_realtime.infer_v1.V1Infer` for the TinyVC encoder +
decoder + STFT + energy plumbing, so the only new pieces here are:
  * the two projection layers (initialised as a truncated identity so a
    zero-trained baseline is at least numerically well-behaved);
  * the OpenVoice flow ONNX call (single graph: forward + reverse in one);
  * the OpenVoice ReferenceEncoder ONNX call (linear spec → 256-d SE).

Outputs a CAMPPlus comparison of flow-path vs kNN baseline across the 5
target voices for the same real-human source.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

# Make `vc_realtime` importable as a top-level package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Make the upstream TinyVC `module.*` package importable
sys.path.insert(0, "../repos/tinyvc")

torch.set_num_threads(2)

from vc_realtime.infer_v1 import V1Infer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = Path("./download")
SOURCE_WAV = ROOT / "data" / "source" / "source_real_001.wav"
VOICE_WAV_TPL = str(ROOT / "data" / "voices" / "voice_{}.wav")


# ---------------------------------------------------------------------------
# Projection + flow inference wrapper
# ---------------------------------------------------------------------------
class FlowV2Infer:
    """Path-1 pipeline: TinyVC encode → project → flow → project → TinyVC decode.

    Composition over inheritance: we instantiate a :class:`V1Infer` and
    reuse its TinyVC encoder/decoder + STFT/energy plumbing instead of
    re-implementing those. Only the projection layers and the two ONNX
    calls (flow + ReferenceEncoder) are new.
    """

    def __init__(self, models_dir: str | os.PathLike = MODELS_DIR) -> None:
        models_dir = Path(models_dir)

        # --- TinyVC encoder + decoder via V1Infer (no re-implementation) ---
        # We use V1Infer as a host for .encode() / .decode() so we don't
        # duplicate the STFT/padding/energy/autopad/trim bookkeeping.
        self._host = V1Infer(models_dir=str(models_dir))
        self.encoder = self._host.encoder
        self.decoder = self._host.decoder

        # --- OpenVoice flow ONNX (single graph: forward + reverse) ---
        flow_path = models_dir / "openvoice_residual_flow.onnx"
        if not flow_path.exists():
            raise FileNotFoundError(
                f"{flow_path} missing — run scripts/export_openvoice_onnx.py first"
            )
        self.flow_session = ort.InferenceSession(
            str(flow_path), providers=["CPUExecutionProvider"]
        )

        # --- OpenVoice ReferenceEncoder ONNX (linear spec → 256-d SE) ---
        ref_path = models_dir / "openvoice_ref_encoder.onnx"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"{ref_path} missing — run scripts/export_openvoice_onnx.py first"
            )
        self.ref_encoder = ort.InferenceSession(
            str(ref_path), providers=["CPUExecutionProvider"]
        )

        # --- Projection layers: 768 ↔ 192 ---
        # Initialised as a truncated identity so the untrained baseline is
        # numerically stable (the OpenVoice flow expects features that look
        # like its posterior latents — random Gaussian init would explode).
        self.proj_down = torch.nn.Linear(768, 192, bias=False)
        self.proj_up = torch.nn.Linear(192, 768, bias=False)
        with torch.no_grad():
            self.proj_down.weight.zero_()
            for i in range(192):
                self.proj_down.weight[i, i] = 1.0
            self.proj_up.weight.zero_()
            for i in range(192):
                self.proj_up.weight[i, i] = 1.0
        self.proj_down.eval()
        self.proj_up.eval()

    # --- OpenVoice ReferenceEncoder wrapper -------------------------------
    def get_speaker_embedding(self, wav_path: str | os.PathLike) -> np.ndarray:
        """Compute a 256-d OpenVoice speaker embedding from an audio file.

        Replicates the OpenVoice ``extract_se`` linear-spectrogram
        computation: n_fft=1024, hop=256, win=1024, Hann, centered. The
        ONNX graph expects ``[B, T_frames, 513]`` (time-first, NOT mel).
        """
        wav, sr = sf.read(wav_path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 22050:
            wav = librosa.resample(
                wav.astype(np.float32), orig_sr=sr, target_sr=22050
            )
        # Linear spectrogram |STFT|; OpenVoice uses spectrogram_torch which
        # is the centered-Hann |STFT| with these exact params.
        linear = np.abs(
            librosa.stft(
                wav.astype(np.float32),
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                center=True,
                window="hann",
            )
        ).T  # → [T_frames, 513]
        linear = linear[None].astype(np.float32)  # → [1, T, 513]
        input_name = self.ref_encoder.get_inputs()[0].name
        se = self.ref_encoder.run(None, {input_name: linear})[0]  # [1, 256]
        return se

    # --- Main pipeline ----------------------------------------------------
    def process_audio(
        self,
        wav: np.ndarray,
        sr: int,
        voice_id: int = 0,
        source_wav_path: str | os.PathLike = SOURCE_WAV,
    ) -> np.ndarray:
        """End-to-end VC: TinyVC encode → project → flow → project → TinyVC decode."""
        # 1. TinyVC encode → content [1,768,T] + f0 [1,1,T] + energy [1,1,L]
        #    V1Infer.encode() handles resample-to-24k, mono, autopad, STFT.
        content, f0, energy = self._host.encode(wav)

        # 2. Speaker embeddings from OpenVoice ReferenceEncoder
        se_src = self.get_speaker_embedding(source_wav_path)  # [1,256]
        se_tgt = self.get_speaker_embedding(VOICE_WAV_TPL.format(voice_id))

        # 3. Project 768 → 192 (along the channel dim, not the time dim)
        z = torch.from_numpy(content)  # [1, 768, T]
        with torch.inference_mode():
            # Linear acts on the last dim; transpose to [1, T, 768] then back.
            z192 = self.proj_down(z.transpose(1, 2)).transpose(1, 2)  # [1, 192, T]
        content_192 = z192.numpy().astype(np.float32)

        # 4. OpenVoice flow: disentangle src speaker, apply tgt speaker
        T = content_192.shape[-1]
        x_mask = np.ones((1, 1, T), dtype=np.float32)
        se_src_3d = se_src.reshape(1, 256, 1).astype(np.float32)
        se_tgt_3d = se_tgt.reshape(1, 256, 1).astype(np.float32)
        flow_out = self.flow_session.run(
            None,
            {
                "content": content_192,
                "x_mask": x_mask,
                "se_src": se_src_3d,
                "se_tgt": se_tgt_3d,
            },
        )[0]  # [1, 192, T]

        # 5. Project back 192 → 768
        z_p = torch.from_numpy(flow_out)  # [1, 192, T]
        with torch.inference_mode():
            z768 = self.proj_up(z_p.transpose(1, 2)).transpose(1, 2)  # [1, 768, T]
        content_back = z768.numpy().astype(np.float32)

        # 6. TinyVC decode (SourceNet + FilterNet, DDSP)
        out = self._host.decode(content_back, f0, energy)
        return out


# ---------------------------------------------------------------------------
# CAMPPlus speaker similarity helpers
# ---------------------------------------------------------------------------
def _campplus_extractor():
    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
    cfg.model = str(MODELS_DIR / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
    cfg.num_threads = 1
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def _embed(extractor, wav: np.ndarray, sr: int = 24000) -> np.ndarray:
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    stream = extractor.create_stream()
    stream.accept_waveform(16000, wav.tolist())
    stream.input_finished()
    return extractor.compute(stream)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extractor = _campplus_extractor()

    # Load source + reference embeddings
    src, sr = sf.read(SOURCE_WAV)
    if src.ndim == 2:
        src = src.mean(axis=1)
    src = src.astype(np.float32)[: sr * 5]  # 5 s clip for fair RTF comparison
    src_emb = _embed(extractor, src, sr)
    target_embs = [
        _embed(extractor, sf.read(VOICE_WAV_TPL.format(i))[0], sr)
        for i in range(5)
    ]

    print("=== Path 1: OpenVoice flow + 768→192 projection ===")
    print(f"  source: {SOURCE_WAV.name}  ({len(src)/sr:.1f}s)")
    print(f"  target voices: 0..4  ({[Path(VOICE_WAV_TPL.format(i)).name for i in range(5)]})")
    print(f"  flow ONNX size : {MODELS_DIR / 'openvoice_residual_flow.onnx'}")
    print(f"  ref  ONNX size : {MODELS_DIR / 'openvoice_ref_encoder.onnx'}")
    print()

    # --- Instantiate both pipelines ---
    infer_flow = FlowV2Infer(models_dir=MODELS_DIR)
    infer_knn = V1Infer(models_dir=str(MODELS_DIR))

    def run_one(pipeline, label: str, vid: int):
        t0 = time.perf_counter()
        try:
            out = pipeline.process_audio(src, sr, vid)
        except Exception as exc:  # noqa: BLE001
            print(f"  voice_{vid} ({label}): FAILED - {exc}")
            return
        t1 = time.perf_counter()
        out_path = OUTPUT_DIR / f"vc_flow_{label}_voice_{vid}.wav"
        sf.write(str(out_path), out, sr)
        out_emb = _embed(extractor, out, sr)
        t_sim = _cosine(out_emb, target_embs[vid])
        s_sim = _cosine(out_emb, src_emb)
        print(
            f"  voice_{vid} ({label}): target_sim={t_sim:.3f}, "
            f"source_sim={s_sim:.3f}, VC_effect={t_sim - s_sim:.3f}, "
            f"RTF={(t1 - t0) / (len(src) / sr):.4f}"
        )

    # Flow path first (this is the new contribution)
    print("--- Flow path (project → OpenVoice flow → project back) ---")
    flow_rows = []
    for vid in range(5):
        t0 = time.perf_counter()
        try:
            out = infer_flow.process_audio(src, sr, vid)
        except Exception as exc:  # noqa: BLE001
            print(f"  voice_{vid} (flow): FAILED - {exc}")
            flow_rows.append(None)
            continue
        t1 = time.perf_counter()
        out_path = OUTPUT_DIR / f"vc_flow_voice_{vid}.wav"
        sf.write(str(out_path), out, sr)
        out_emb = _embed(extractor, out, sr)
        t_sim = _cosine(out_emb, target_embs[vid])
        s_sim = _cosine(out_emb, src_emb)
        flow_rows.append((t_sim, s_sim, t_sim - s_sim, (t1 - t0) / (len(src) / sr)))
        print(
            f"  voice_{vid} (flow): target_sim={t_sim:.3f}, "
            f"source_sim={s_sim:.3f}, VC_effect={t_sim - s_sim:.3f}, "
            f"RTF={(t1 - t0) / (len(src) / sr):.4f}"
        )
    flow_valid = [r for r in flow_rows if r is not None]
    if flow_valid:
        avg_t = np.mean([r[0] for r in flow_valid])
        avg_s = np.mean([r[1] for r in flow_valid])
        avg_e = np.mean([r[2] for r in flow_valid])
        avg_rtf = np.mean([r[3] for r in flow_valid])
        print(
            f"  AVERAGE (flow): target={avg_t:.3f}, source={avg_s:.3f}, "
            f"VC_effect={avg_e:.3f}, RTF={avg_rtf:.4f}"
        )

    # kNN baseline (no flow)
    print()
    print("--- Baseline: kNN-VC (no flow) ---")
    knn_rows = []
    for vid in range(5):
        t0 = time.perf_counter()
        out = infer_knn.process_audio(src, sr, vid)
        t1 = time.perf_counter()
        out_emb = _embed(extractor, out, sr)
        t_sim = _cosine(out_emb, target_embs[vid])
        s_sim = _cosine(out_emb, src_emb)
        knn_rows.append((t_sim, s_sim, t_sim - s_sim, (t1 - t0) / (len(src) / sr)))
        print(
            f"  voice_{vid} (kNN): target_sim={t_sim:.3f}, "
            f"source_sim={s_sim:.3f}, VC_effect={t_sim - s_sim:.3f}, "
            f"RTF={(t1 - t0) / (len(src) / sr):.4f}"
        )
    if knn_rows:
        avg_t = np.mean([r[0] for r in knn_rows])
        avg_s = np.mean([r[1] for r in knn_rows])
        avg_e = np.mean([r[2] for r in knn_rows])
        avg_rtf = np.mean([r[3] for r in knn_rows])
        print(
            f"  AVERAGE (kNN): target={avg_t:.3f}, source={avg_s:.3f}, "
            f"VC_effect={avg_e:.3f}, RTF={avg_rtf:.4f}"
        )

    # Final delta
    print()
    print("=== Delta (flow − kNN) ===")
    if flow_valid and knn_rows:
        print(
            f"  Δtarget_sim = {np.mean([r[0] for r in flow_valid]) - avg_t:+.3f}   "
            f"Δsource_sim = {np.mean([r[1] for r in flow_valid]) - avg_s:+.3f}   "
            f"ΔVC_effect  = {np.mean([r[2] for r in flow_valid]) - avg_e:+.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
