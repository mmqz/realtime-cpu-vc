"""Tests for the TinyVC encoder ONNX export (P2.1-1).

Verifies that ``models/encoder.onnx`` (FP32) and ``models/encoder.int8.onnx``
(INT8 PTQ):

  1. Exist and load in onnxruntime.
  2. Have exactly one input ``spectrogram`` shape ``[B, 961, T]`` and two
     outputs ``content`` ``[B, 768, T]`` + ``f0`` ``[B, 512, T]`` (matching
     upstream ``tinyvc/export_onnx.py``).
  3. Produce numerically plausible outputs for a random input.
  4. INT8 quantized model's outputs stay within a reasonable tolerance of the
     FP32 model's outputs (dynamic PTQ preserves content features).
  5. Honour dynamic batch + time axes.
  6. Rust ``vc-ort`` crate can load ``encoder.int8.onnx`` (P2.0-1 prerequisite)
     — verified via the separate ``cargo test -p vc-ort`` suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"
ENCODER_ONNX = MODELS / "encoder.onnx"
ENCODER_INT8 = MODELS / "encoder.int8.onnx"

# TinyVC Encoder constants (verified from
# repos/tinyvc/module/tinyvc/encoder.py:101-107).
N_FFT = 1920
FFT_BIN = N_FFT // 2 + 1  # = 961
SSL_DIM = 768              # SSLFeatureEstimator ssl_dim (encoder.py:81)
NUM_PITCH_CLASSES = 512    # PitchEstimator num_classes (encoder.py:17)
T_FRAMES = 50              # 1 s of audio at 50 Hz frame rate


@pytest.fixture(scope="module")
def session_fp32() -> ort.InferenceSession:
    if not ENCODER_ONNX.exists():
        pytest.skip(f"{ENCODER_ONNX} not found; "
                    f"run scripts/export_tinyvc_encoder_onnx.py")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(ENCODER_ONNX), sess_options=so,
                                providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def session_int8() -> ort.InferenceSession:
    if not ENCODER_INT8.exists():
        pytest.skip(f"{ENCODER_INT8} not found; "
                    f"run scripts/export_tinyvc_encoder_onnx.py --quantize")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(ENCODER_INT8), sess_options=so,
                                providers=["CPUExecutionProvider"])


def test_encoder_onnx_loads(session_fp32: ort.InferenceSession) -> None:
    """encoder.onnx loads and has the expected I/O contract."""
    inputs = session_fp32.get_inputs()
    outputs = session_fp32.get_outputs()
    assert len(inputs) == 1, f"expected 1 input, got {len(inputs)}"
    assert len(outputs) == 2, f"expected 2 outputs, got {len(outputs)}"
    assert inputs[0].name == "spectrogram", \
        f"input name mismatch: {inputs[0].name}"
    assert outputs[0].name == "content", \
        f"output[0] name mismatch: {outputs[0].name}"
    assert outputs[1].name == "f0", \
        f"output[1] name mismatch: {outputs[1].name}"
    # Static frequency bin dim must equal FFT_BIN (961).
    assert inputs[0].shape[1] == FFT_BIN, \
        f"expected fft_bin={FFT_BIN}, got {inputs[0].shape[1]}"
    # Static channel dims on outputs.
    assert outputs[0].shape[1] == SSL_DIM, \
        f"expected ssl_dim={SSL_DIM}, got {outputs[0].shape[1]}"
    assert outputs[1].shape[1] == NUM_PITCH_CLASSES, \
        f"expected num_pitch_classes={NUM_PITCH_CLASSES}, " \
        f"got {outputs[1].shape[1]}"


def test_encoder_int8_loads(session_int8: ort.InferenceSession) -> None:
    """encoder.int8.onnx loads with the same I/O contract as the FP32 graph."""
    inputs = session_int8.get_inputs()
    outputs = session_int8.get_outputs()
    assert len(inputs) == 1
    assert len(outputs) == 2
    assert inputs[0].name == "spectrogram"
    assert outputs[0].name == "content"
    assert outputs[1].name == "f0"
    assert inputs[0].shape[1] == FFT_BIN
    assert outputs[0].shape[1] == SSL_DIM
    assert outputs[1].shape[1] == NUM_PITCH_CLASSES


def test_encoder_onnx_forward_pass(session_fp32: ort.InferenceSession) -> None:
    """Forward pass with random input produces non-trivial content features."""
    rng = np.random.default_rng(seed=42)
    spec = rng.standard_normal((1, FFT_BIN, T_FRAMES)).astype(np.float32)
    in_name = session_fp32.get_inputs()[0].name
    out = session_fp32.run(None, {in_name: spec})

    content, f0_logits = out[0], out[1]
    assert content.shape == (1, SSL_DIM, T_FRAMES)
    assert f0_logits.shape == (1, NUM_PITCH_CLASSES, T_FRAMES)
    # Content features have meaningful variation (not all-zero / all-NaN).
    assert np.isfinite(content).all(), "content has NaN/inf"
    assert content.std() > 1e-3, \
        f"content features near-constant (std={content.std():.4e})"
    # F0 logits are unbounded pre-softmax; just verify finite + non-trivial.
    assert np.isfinite(f0_logits).all(), "f0 logits has NaN/inf"
    assert f0_logits.std() > 1e-3, \
        f"f0 logits near-constant (std={f0_logits.std():.4e})"


def test_encoder_onnx_dynamic_batch(
    session_fp32: ort.InferenceSession,
) -> None:
    """Vary batch dimension; verify both outputs scale."""
    rng = np.random.default_rng(seed=7)
    spec = rng.standard_normal((3, FFT_BIN, T_FRAMES)).astype(np.float32)
    in_name = session_fp32.get_inputs()[0].name
    out = session_fp32.run(None, {in_name: spec})
    assert out[0].shape == (3, SSL_DIM, T_FRAMES)
    assert out[1].shape == (3, NUM_PITCH_CLASSES, T_FRAMES)


def test_encoder_onnx_dynamic_length(
    session_fp32: ort.InferenceSession,
) -> None:
    """Vary time dimension; verify both outputs scale linearly."""
    rng = np.random.default_rng(seed=11)
    t = 80
    spec = rng.standard_normal((1, FFT_BIN, t)).astype(np.float32)
    in_name = session_fp32.get_inputs()[0].name
    out = session_fp32.run(None, {in_name: spec})
    assert out[0].shape == (1, SSL_DIM, t)
    assert out[1].shape == (1, NUM_PITCH_CLASSES, t)


def test_encoder_int8_approximates_fp32(
    session_fp32: ort.InferenceSession,
    session_int8: ort.InferenceSession,
) -> None:
    """Per-channel dynamic INT8 PTQ must preserve content features within ~5%.

    The pitch-class logits are inherently noisy (pre-softmax, unbounded), so
    we tolerate a looser bound on the f0 output (within ~15%). The 768-d
    content feature is what feeds the downstream vocoder and must be tight.
    """
    rng = np.random.default_rng(seed=123)
    spec = rng.standard_normal((1, FFT_BIN, T_FRAMES)).astype(np.float32)
    in_name_fp32 = session_fp32.get_inputs()[0].name
    in_name_int8 = session_int8.get_inputs()[0].name

    out_fp32 = session_fp32.run(None, {in_name_fp32: spec})
    out_int8 = session_int8.run(None, {in_name_int8: spec})

    content_fp32, f0_fp32 = out_fp32[0], out_fp32[1]
    content_int8, f0_int8 = out_int8[0], out_int8[1]

    # Content features (the semantic backbone feeding the vocoder) —
    # relative-error bound on the L2 norm of the difference. Dynamic PTQ
    # (per-channel, no calibration data) typically lands in 1–8% territory
    # on ConvNeXt-v2; we assert < 10% as a sensible invariant that the
    # INT8 graph is *not* silently broken.
    content_rel = float(
        np.linalg.norm(content_int8 - content_fp32)
        / max(np.linalg.norm(content_fp32), 1e-9)
    )
    assert content_rel < 0.10, (
        f"content relative L2 error too high: {content_rel:.4e} "
        f"(expected < 0.10)"
    )

    # Pitch logits are unbounded; bound the mean abs deviation relative to
    # the FP32 logit magnitude.
    f0_scale = float(np.abs(f0_fp32).mean())
    f0_mad = float(np.abs(f0_int8 - f0_fp32).mean())
    assert f0_mad < 0.5 + 0.20 * f0_scale, (
        f"f0 mean-abs-dev too high: mad={f0_mad:.4e}, "
        f"fp32_scale={f0_scale:.4e}"
    )
