#!/usr/bin/env python3
"""Path 1 (SolA): TRAINED 768→192→768 projection + OpenVoice flow.

Compared to the untrained baseline (`scripts/vc_flow_test.py`, which uses a
truncated-identity projection), this script loads the autoencoder-trained
projection saved at ``models/trained_projection.pt`` and re-runs the full
flow path on the same 5 target voices, reporting CAMPPlus target/source
similarity for each.

Bugs in the original task spec fixed here:
  * `Decoder.infer(content, f0, energy)` takes **3** args, not 2.
  * Same `torchfcpe` / `pyworld` stub trick as `vc_realtime.infer_v1`.
  * `Linear(768, 192)` must be applied on the last dim (transpose
    channels-first → channels-last → back) — done via `V1Infer` host.
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


class FlowV2InferTrained:
    """Path-1 pipeline with TRAINED projection layers.

    Same shape as ``FlowV2Infer`` in ``scripts/vc_flow_test.py`` but loads
    the trained projection weights instead of using truncated identity.
    """

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        # TinyVC encoder + decoder via V1Infer host (handles STFT / autopad /
        # energy bookkeeping — we don't reimplement that here).
        self._host = V1Infer(models_dir=str(models_dir))
        self.encoder = self._host.encoder
        self.decoder = self._host.decoder

        # OpenVoice flow ONNX (single graph: forward + reverse)
        flow_path = models_dir / "openvoice_residual_flow.onnx"
        if not flow_path.exists():
            raise FileNotFoundError(
                f"{flow_path} missing — run scripts/export_openvoice_onnx.py first"
            )
        self.flow_session = ort.InferenceSession(
            str(flow_path), providers=["CPUExecutionProvider"]
        )

        # OpenVoice ReferenceEncoder ONNX (linear spec → 256-d SE)
        ref_path = models_dir / "openvoice_ref_encoder.onnx"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"{ref_path} missing — run scripts/export_openvoice_onnx.py first"
            )
        self.ref_encoder = ort.InferenceSession(
            str(ref_path), providers=["CPUExecutionProvider"]
        )

        # TRAINED projection layers (autoencoder 768→192→768)
        proj_path = models_dir / "trained_projection.pt"
        if not proj_path.exists():
            raise FileNotFoundError(
                f"{proj_path} missing — run scripts/train_projection.py first"
            )
        ckpt = torch.load(str(proj_path), map_location="cpu")
        self.proj_down = torch.nn.Linear(768, 192, bias=False)
        self.proj_up = torch.nn.Linear(192, 768, bias=False)
        self.proj_down.load_state_dict(ckpt["proj_down"])
        self.proj_up.load_state_dict(ckpt["proj_up"])
        self.proj_down.eval()
        self.proj_up.eval()

    def get_speaker_embedding(self, wav_path: str | os.PathLike) -> np.ndarray:
        """256-d OpenVoice speaker embedding from a wav (linear |STFT|)."""
        wav, sr = sf.read(wav_path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 22050:
            wav = librosa.resample(
                wav.astype(np.float32), orig_sr=sr, target_sr=22050
            )
        linear = np.abs(
            librosa.stft(
                wav.astype(np.float32),
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                center=True,
                window="hann",
            )
        ).T  # [T_frames, 513]
        linear = linear[None].astype(np.float32)  # [1, T, 513]
        input_name = self.ref_encoder.get_inputs()[0].name
        se = self.ref_encoder.run(None, {input_name: linear})[0]  # [1, 256]
        return se

    def process_audio(
        self,
        wav: np.ndarray,
        sr: int,
        voice_id: int = 0,
        source_wav_path: str | os.PathLike = SOURCE_WAV,
    ) -> np.ndarray:
        """End-to-end: TinyVC encode → project → flow → project → TinyVC decode."""
        # 1. TinyVC encode → content [1,768,T], f0 [1,1,T], energy [1,1,L]
        content, f0, energy = self._host.encode(wav)

        # 2. Speaker embeddings
        se_src = self.get_speaker_embedding(source_wav_path)  # [1, 256]
        se_tgt = self.get_speaker_embedding(VOICE_WAV_TPL.format(voice_id))

        # 3. Trained projection 768 → 192 (along channel dim)
        z = torch.from_numpy(content)  # [1, 768, T]
        with torch.inference_mode():
            z192 = self.proj_down(z.transpose(1, 2)).transpose(1, 2)  # [1, 192, T]
        content_192 = z192.numpy().astype(np.float32)

        # 4. OpenVoice flow: disentangle src, apply tgt
        T = content_192.shape[-1]
        x_mask = np.ones((1, 1, T), dtype=np.float32)
        se_src_3d = se_src.reshape(1, 256, 1).astype(np.float32)
        se_tgt_3d = se_tgt.reshape(1, 256, 1).astype(np.float32)

        inputs = self.flow_session.get_inputs()
        feed: dict[str, np.ndarray] = {}
        for i, inp in enumerate(inputs):
            if "content" in inp.name or i == 0:
                feed[inp.name] = content_192
            elif "mask" in inp.name or i == 1:
                feed[inp.name] = x_mask
            elif "src" in inp.name or i == 2:
                feed[inp.name] = se_src_3d
            elif "tgt" in inp.name or i == 3:
                feed[inp.name] = se_tgt_3d
        flow_out = self.flow_session.run(None, feed)[0]  # [1, 192, T]

        # 5. Trained projection 192 → 768
        z_p = torch.from_numpy(flow_out)  # [1, 192, T]
        with torch.inference_mode():
            z768 = self.proj_up(z_p.transpose(1, 2)).transpose(1, 2)  # [1, 768, T]
        content_back = z768.numpy().astype(np.float32)

        # 6. TinyVC decode (SourceNet + FilterNet, DDSP) — needs energy too
        out = self._host.decode(content_back, f0, energy)
        return out


# ---------------------------------------------------------------------------
# CAMPPlus speaker similarity helpers
# ---------------------------------------------------------------------------
def _campplus_extractor():
    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
    cfg.model = str(
        MODELS_DIR / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
    )
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


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor = _campplus_extractor()

    src, sr = sf.read(SOURCE_WAV)
    if src.ndim == 2:
        src = src.mean(axis=1)
    src = src.astype(np.float32)[: sr * 5]  # 5s clip
    src_emb = _embed(extractor, src, sr)
    target_embs = [
        _embed(extractor, sf.read(VOICE_WAV_TPL.format(i))[0], sr)
        for i in range(5)
    ]

    print("=== SolA: Trained projection (autoencoder) + OpenVoice flow ===")
    print(f"  source: {SOURCE_WAV.name}  ({len(src)/sr:.1f}s)")
    print(f"  projection: {MODELS_DIR / 'trained_projection.pt'}")
    print(f"  flow ONNX  : {MODELS_DIR / 'openvoice_residual_flow.onnx'}")
    print(f"  ref  ONNX  : {MODELS_DIR / 'openvoice_ref_encoder.onnx'}")
    print()

    infer = FlowV2InferTrained(models_dir=MODELS_DIR)

    rows = []
    print("--- Flow path (trained project → OpenVoice flow → project back) ---")
    for vid in range(5):
        t0 = time.perf_counter()
        try:
            out = infer.process_audio(src, sr, vid)
        except Exception as exc:  # noqa: BLE001
            print(f"  voice_{vid} (flow-trained): FAILED - {exc}")
            import traceback

            traceback.print_exc()
            rows.append(None)
            continue
        t1 = time.perf_counter()
        out_path = OUTPUT_DIR / f"vc_trained_flow_voice_{vid}.wav"
        sf.write(str(out_path), out, sr)
        out_emb = _embed(extractor, out, sr)
        t_sim = _cosine(out_emb, target_embs[vid])
        s_sim = _cosine(out_emb, src_emb)
        rtf = (t1 - t0) / (len(src) / sr)
        rows.append((t_sim, s_sim, t_sim - s_sim, rtf))
        print(
            f"  voice_{vid} (flow-trained): target_sim={t_sim:.3f}, "
            f"source_sim={s_sim:.3f}, VC_effect={t_sim - s_sim:.3f}, RTF={rtf:.4f}"
        )

    valid = [r for r in rows if r is not None]
    if valid:
        avg_t = float(np.mean([r[0] for r in valid]))
        avg_s = float(np.mean([r[1] for r in valid]))
        avg_e = float(np.mean([r[2] for r in valid]))
        avg_rtf = float(np.mean([r[3] for r in valid]))
        print(
            f"  AVERAGE (flow-trained): target={avg_t:.3f}, source={avg_s:.3f}, "
            f"VC_effect={avg_e:.3f}, RTF={avg_rtf:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
