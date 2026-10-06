#!/usr/bin/env python3
"""Export F5-TTS Vocos vocoder as ONNX (opset 18, legacy exporter).

Architecture
------------
- Vocos: mel-spectrogram -> waveform (Fourier-based neural vocoder).
- Input  : mel-spec [B, 100, T_mel]   (100 mel bins, frame rate ~93.75 Hz at 24 kHz)
- Output : waveform   [B, T_samples]   (T_samples = T_mel * hop_length = T_mel * 256)
- ~13.5M params (matches the ~13M target).

Reference: official F5-TTS export script at
    repos/f5-tts/src/f5_tts/runtime/triton_trtllm/scripts/export_vocoder_to_onnx.py

The official Vocos ``ISTFTHead`` (vocos.heads.ISTFTHead) uses ``torch.fft.istft``
with a complex tensor. That path exports cleanly only when padding == "center",
but the conv_stft-based head used by the official F5-TTS export script swaps in
a pure Conv1D / ConvTranspose1D ISTFT (``conv_stft.STFT.inverse``), which is
fully ONNX-friendly. We follow the same approach here.

Notes
-----
- Opset 18 (not 17) is required because torch 2.14's new dynamo exporter
  emits a Split node using the ``num_outputs`` attribute (introduced in opset
  18). Down-conversion to opset 17 fails with
  ``Unrecognized attribute: num_outputs for operator Split``.
  onnxruntime 1.30 fully supports opset 18.
- We use the legacy TorchScript exporter (``dynamo=False``) to avoid
  the dynamo exporter's reliance on onnxscript-specific operators.
- CPU-only: we never call ``.cuda()``. The checkpoint is from HF repo
  ``charactr/vocos-mel-24khz`` (the same repo F5-TTS downloads from).

Licenses
--------
- vocos (PyPI): MIT (c) 2024 Yushen CHEN.
- conv_stft.py: MIT (c) 2020 Shimin Zhang  +  Apache-2.0 NVIDIA overlay.
- F5-TTS source: CC-BY-NC-4.0 (we only READ the source; we do not ship any
  F5-TTS code).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Make F5-TTS's vendored conv_stft importable.
# We deliberately do NOT import the F5-TTS package itself (CC-BY-NC-4.0),
# only the standalone MIT-licensed conv_stft module.
_TRITON_SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / ".." / "repos" / "f5-tts" / "src" / "f5_tts"
    / "runtime" / "triton_trtllm" / "scripts"
).resolve()
sys.path.insert(0, str(_TRITON_SCRIPTS_DIR))

# torch.onnx.export will warn about the deprecation of the legacy exporter
# under torch>=2.9. We intentionally keep the legacy exporter (dynamo=False)
# because it avoids the Split/num_outputs opset-18-only downgrade problem.
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx.*")

from conv_stft import STFT  # noqa: E402
from vocos import Vocos  # noqa: E402

# ---------------------------------------------------------------------------
# Vocos export wrapper (adapted from official F5-TTS export script).
# MIT-licensed (NVIDIA overlay) + MIT conv_stft.
# ---------------------------------------------------------------------------


class ISTFTHead(nn.Module):
    """Conv-based ISTFT head — ONNX-exportable replacement for the native head.

    Mirrors ``vocos.heads.ISTFTHead.forward`` but computes the inverse STFT
    via ConvTranspose1D (from ``conv_stft.STFT.inverse``) instead of
    ``torch.fft.istft``, so the entire forward pass uses only Conv / MatMul
    / Elementwise ops.
    """

    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        # ``out`` is wired up by ``VocosVocoder`` after construction to share
        # weights with the original ISTFTHead's final Linear layer.
        self.out: nn.Linear | None = None
        self.stft = STFT(fft_len=n_fft, win_hop=hop_length, win_len=n_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H] -> Linear -> [B, L, n_fft+2] -> transpose -> [B, n_fft+2, L]
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard against explode-y magnitudes
        real = mag * torch.cos(p)
        imag = mag * torch.sin(p)
        # conv_stft.inverse returns [B, T_samples] (already squeezed).
        audio = self.stft.inverse(input1=real, input2=imag, input_type="realimag")
        return audio


class VocosVocoder(nn.Module):
    """Wraps a Vocos model so its ISTFTHead is replaced with the Conv-based one.

    The rest of the model (feature_extractor is unused here, VocosBackbone,
    ISTFTHead.out Linear) is kept intact. Calling ``forward(mel)`` runs
    ``vocos.decode(mel)`` = backbone(mel) -> head(x).
    """

    def __init__(self, vocos_vocoder: Vocos) -> None:
        super().__init__()
        self.vocos_vocoder = vocos_vocoder
        istft_head_out = self.vocos_vocoder.head.out
        n_fft = self.vocos_vocoder.head.istft.n_fft
        hop_length = self.vocos_vocoder.head.istft.hop_length
        export_head = ISTFTHead(n_fft=n_fft, hop_length=hop_length)
        export_head.out = istft_head_out  # share weights with original head
        self.vocos_vocoder.head = export_head

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.vocos_vocoder.decode(mel)


# ---------------------------------------------------------------------------
# Checkpoint loading.
# ---------------------------------------------------------------------------


def load_vocos_hf(repo_id: str = "charactr/vocos-mel-24khz") -> Vocos:
    """Download config.yaml + pytorch_model.bin from HuggingFace and load Vocos."""
    print(f"[vocos] Downloading from HF repo '{repo_id}' ...")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    print(f"[vocos]   config: {config_path}")
    print(f"[vocos]   model:  {model_path}")

    vocos = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    vocos.load_state_dict(state_dict)
    vocos = vocos.eval().cpu()
    return vocos


# ---------------------------------------------------------------------------
# Main export routine.
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "vocos.onnx"

    # 1. Load Vocos from HF (charactr/vocos-mel-24khz).
    vocos = load_vocos_hf()

    # 2. Inspect + log.
    n_params = sum(p.numel() for p in vocos.parameters())
    head = vocos.head
    print(
        f"[vocos] params={n_params} ({n_params / 1e6:.2f}M) | "
        f"head={type(head).__name__} | "
        f"istft n_fft={head.istft.n_fft} hop_length={head.istft.hop_length} "
        f"win_length={head.istft.win_length} padding={head.istft.padding}"
    )

    # 3. Wrap with the Conv-based ISTFTHead (ONNX-friendly).
    model = VocosVocoder(vocos).cpu().eval()

    # 4. Dummy input: 1 batch, 100 mel bins, 100 mel frames (~1.067 s @ 24 kHz).
    dummy_mel = torch.randn(1, 100, 100, dtype=torch.float32)

    # Sanity-check the torch forward pass before exporting.
    with torch.no_grad():
        dummy_out = model(dummy_mel)
    expected_samples = 100 * head.istft.hop_length
    print(
        f"[vocos] torch forward OK | "
        f"mel={tuple(dummy_mel.shape)} -> wave={tuple(dummy_out.shape)} "
        f"(expected [{dummy_mel.shape[0]}, {expected_samples}])"
    )
    assert dummy_out.shape == (1, expected_samples), (
        f"unexpected torch output shape {tuple(dummy_out.shape)}; "
        f"expected (1, {expected_samples})"
    )

    # 5. Export to ONNX (opset 18, legacy TorchScript exporter).
    #    - opset 18 is required: torch 2.14's Split uses the `num_outputs`
    #      attribute (introduced in opset 18); down-converting to opset 17
    #      produces an invalid Split node that onnxruntime rejects.
    #    - dynamo=False: use the legacy TorchScript exporter to avoid
    #      onnxscript-specific ops emitted by the new dynamo path.
    print(f"[vocos] exporting ONNX -> {out_path} (opset 18, legacy exporter)")
    torch.onnx.export(
        model,
        dummy_mel,
        str(out_path),
        opset_version=18,
        do_constant_folding=True,
        input_names=["mel"],
        output_names=["waveform"],
        dynamic_axes={
            "mel": {0: "batch_size", 2: "input_length"},
            "waveform": {0: "batch_size", 1: "output_length"},
        },
        verbose=False,
        dynamo=False,
    )

    # 6. Verify with onnxruntime (CPUExecutionProvider).
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"[vocos] ONNX file size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"[vocos]   inputs : {[(i.name, list(i.shape)) for i in inputs]}")
    print(f"[vocos]   outputs: {[(o.name, list(o.shape)) for o in outputs]}")

    # 7. Sample inference + numerical parity check.
    mel_np = np.random.randn(1, 100, 100).astype(np.float32)
    out_np = sess.run(None, {inputs[0].name: mel_np})[0]
    rms = float(np.sqrt(np.mean(out_np**2)))
    print(
        f"[vocos] sample inference: mel={mel_np.shape} -> wave={out_np.shape} "
        f"range=[{out_np.min():.3f}, {out_np.max():.3f}] RMS={rms:.4f}"
    )

    with torch.no_grad():
        out_torch = model(torch.from_numpy(mel_np)).numpy()
    max_diff = float(np.abs(out_torch - out_np).max())
    print(f"[vocos] max |torch - onnx| = {max_diff:.6f}")

    # 8. Dynamic axes check.
    mel_dyn = np.random.randn(2, 100, 200).astype(np.float32)
    out_dyn = sess.run(None, {inputs[0].name: mel_dyn})[0]
    print(
        f"[vocos] dynamic check: B=2,T=200 mel -> wave={out_dyn.shape} "
        f"(expected (2, {200 * head.istft.hop_length}))"
    )
    assert out_dyn.shape == (2, 200 * head.istft.hop_length)

    print("[vocos] DONE")


if __name__ == "__main__":
    main()
