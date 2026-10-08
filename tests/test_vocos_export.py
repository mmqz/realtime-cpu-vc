"""Tests for the F5-TTS Vocos ONNX vocoder export (P2-2).

Verifies that ``models/vocos.onnx``:
  1. Exists and loads in onnxruntime.
  2. Has exactly one input (mel-spec [B, 100, T_mel]) and one output (waveform
     [B, T_samples]).
  3. Produces an output whose length matches T_mel * hop_length (hop=256).
  4. Produces non-silent audio (RMS > 1e-3) for a typical random mel input.
  5. Honours dynamic batch + time axes.
  6. Numerical parity with the torch export wrapper within tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"
VOCOS_ONNX = MODELS / "vocos.onnx"

# Vocos (charactr/vocos-mel-24khz) constants, verified at export time.
N_MEL = 100
HOP_LENGTH = 256  # 24 kHz / 256 = 93.75 Hz frame rate
T_MEL = 100  # ~1.067 s of audio
EXPECTED_SAMPLES = T_MEL * HOP_LENGTH  # = 25600


# Make the export wrapper importable for the parity test. We import lazily
# inside the test so that simply collecting tests doesn't require vocos.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def session() -> ort.InferenceSession:
    """Load vocos.onnx once per test module."""
    if not VOCOS_ONNX.exists():
        pytest.skip(f"{VOCOS_ONNX} not found; run scripts/export_vocos_onnx.py")
    return ort.InferenceSession(str(VOCOS_ONNX), providers=["CPUExecutionProvider"])


def test_vocos_onnx_loads(session: ort.InferenceSession) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    assert len(inputs) == 1, f"expected 1 input, got {len(inputs)}"
    assert len(outputs) == 1, f"expected 1 output, got {len(outputs)}"
    assert inputs[0].name == "mel"
    assert outputs[0].name == "waveform"
    # Static dim = mel channels (100); batch + time are dynamic.
    assert inputs[0].shape[1] == N_MEL, (
        f"expected {N_MEL} mel channels, got {inputs[0].shape[1]}"
    )


def test_vocos_onnx_produces_audio(session: ort.InferenceSession) -> None:
    """Output should be 2D [B, T_samples], length == T_mel * hop, and non-silent."""
    mel_input = np.random.randn(1, N_MEL, T_MEL).astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: mel_input})[0]

    # Output is 2D: [batch, T_samples].
    assert out.ndim == 2, f"expected 2D output, got {out.ndim}D ({out.shape})"
    assert out.shape[0] == 1
    actual_samples = out.shape[-1]
    assert abs(actual_samples - EXPECTED_SAMPLES) < 0.1 * EXPECTED_SAMPLES, (
        f"output length {actual_samples} != expected {EXPECTED_SAMPLES}"
    )

    rms = float(np.sqrt(np.mean(out**2)))
    assert rms > 1e-3, f"Vocos output silent (RMS={rms})"


def test_vocos_onnx_dynamic_batch(session: ort.InferenceSession) -> None:
    """Vary batch size; verify output batches scale accordingly."""
    mel = np.random.randn(3, N_MEL, T_MEL).astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: mel})[0]
    assert out.shape == (3, EXPECTED_SAMPLES)


def test_vocos_onnx_dynamic_length(session: ort.InferenceSession) -> None:
    """Vary mel length; verify output length scales linearly with hop_length."""
    t_mel = 50
    mel = np.random.randn(1, N_MEL, t_mel).astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: mel})[0]
    assert out.shape == (1, t_mel * HOP_LENGTH)


def test_vocos_onnx_numerical_parity(session: ort.InferenceSession) -> None:
    """ONNX output should match the torch export wrapper within ~1e-4."""
    if not _SCRIPTS_DIR.exists():
        pytest.skip("scripts/ directory not found")
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        import export_vocos_onnx as exp  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot import export_vocos_onnx: {e}")

    vocos = exp.load_vocos_hf()
    model = exp.VocosVocoder(vocos).cpu().eval()
    mel_np = np.random.randn(1, N_MEL, T_MEL).astype(np.float32)

    import torch

    with torch.no_grad():
        out_torch = model(torch.from_numpy(mel_np)).numpy()
    out_onnx = session.run(None, {session.get_inputs()[0].name: mel_np})[0]
    max_diff = float(np.abs(out_torch - out_onnx).max())
    assert max_diff < 1e-4, f"max |torch - onnx| = {max_diff}, expected < 1e-4"
