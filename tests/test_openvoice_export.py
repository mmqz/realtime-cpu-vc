"""Tests for the OpenVoice v2 ReferenceEncoder + ResidualCouplingBlock ONNX export (P2-1).

Verifies that ``models/openvoice_ref_encoder.onnx`` and
``models/openvoice_residual_flow.onnx``:
  1. Exist on disk.
  2. Load cleanly in onnxruntime (CPUExecutionProvider).
  3. Have the expected input/output signatures and shapes (verified at
     export time against the actual OpenVoice v2 model:
     - ReferenceEncoder: [B, T, 513] linear spec → [B, 256] speaker emb.
     - Flow             : (content [B,192,T], mask [B,1,T],
                           se_src [B,256,1], se_tgt [B,256,1])
                         → content_disentangled [B,192,T].
  4. Run inference without error.
  5. Produce outputs with sensible magnitudes (not NaN / not all-zero).
  6. Honour dynamic batch + time axes (different shapes than the export
     dummy shapes still produce valid output of matching shape).
  7. Flow identity roundtrip: ``src==tgt`` → output ≈ input (the mean-only
     affine coupling is invertible by construction, so forward-then-reverse
     with the same speaker embedding must recover the input).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"
REF_ENC_ONNX = MODELS / "openvoice_ref_encoder.onnx"
FLOW_ONNX = MODELS / "openvoice_residual_flow.onnx"

# Constants from the OpenVoice v2 config (filter_length=1024 → spec_channels=513).
SPEC_CHANNELS = 513
GIN_CHANNELS = 256
INTER_CHANNELS = 192


# ---------------------------------------------------------------------------
# ReferenceEncoder ONNX
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REF_ENC_ONNX.exists(), reason="ref_encoder ONNX not exported yet")
def test_ref_encoder_onnx_loads() -> None:
    sess = ort.InferenceSession(str(REF_ENC_ONNX), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    # Single input (linear_spec), single output (speaker_embedding).
    assert len(inputs) == 1, f"expected 1 input, got {len(inputs)}"
    assert len(outputs) == 1, f"expected 1 output, got {len(outputs)}"

    # Names match the export contract.
    assert inputs[0].name == "linear_spec"
    assert outputs[0].name == "speaker_embedding"

    # Fixed mel-dim (513), dynamic batch + time.
    in_shape = inputs[0].shape  # ['batch', 'time', 513]
    out_shape = outputs[0].shape  # ['batch', 256]
    assert in_shape[-1] == SPEC_CHANNELS, f"input mel-dim mismatch: {in_shape}"
    assert out_shape[-1] == GIN_CHANNELS, f"output emb-dim mismatch: {out_shape}"


@pytest.mark.skipif(not REF_ENC_ONNX.exists(), reason="ref_encoder ONNX not exported yet")
def test_ref_encoder_runs_and_shape() -> None:
    sess = ort.InferenceSession(str(REF_ENC_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    # 200 frames ≈ 2.3 s at 86 Hz hop; matches the export dummy shape.
    spec = np.random.randn(1, 200, SPEC_CHANNELS).astype(np.float32)
    out = sess.run(None, {in_name: spec})[0]

    assert out.shape == (1, GIN_CHANNELS), f"output shape mismatch: {out.shape}"
    assert np.all(np.isfinite(out)), "output contains NaN/Inf"
    # Embedding should be non-zero (random input → non-degenerate embedding).
    assert float(np.abs(out).max()) > 1e-3, f"output near-zero: max|out|={float(np.abs(out).max())}"


@pytest.mark.skipif(not REF_ENC_ONNX.exists(), reason="ref_encoder ONNX not exported yet")
def test_ref_encoder_dynamic_batch_and_time() -> None:
    """Verify the dynamic axes on batch and time dimensions actually work."""
    sess = ort.InferenceSession(str(REF_ENC_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    for batch, t in [(2, 100), (3, 250), (1, 50)]:
        spec = np.random.randn(batch, t, SPEC_CHANNELS).astype(np.float32)
        out = sess.run(None, {in_name: spec})[0]
        assert out.shape == (batch, GIN_CHANNELS), (
            f"dynamic axes broken for batch={batch}, t={t}: got {out.shape}"
        )


@pytest.mark.skipif(not REF_ENC_ONNX.exists(), reason="ref_encoder ONNX not exported yet")
def test_ref_encoder_batch_consistency() -> None:
    """Embedding for batch element i should equal embedding for batch as a singleton."""
    sess = ort.InferenceSession(str(REF_ENC_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    spec_b2 = np.random.randn(2, 200, SPEC_CHANNELS).astype(np.float32)
    out_b2 = sess.run(None, {in_name: spec_b2})[0]
    out_i0_alone = sess.run(None, {in_name: spec_b2[:1]})[0]
    out_i1_alone = sess.run(None, {in_name: spec_b2[1:2]})[0]

    # GRU with batch_first=True is batch-independent, so embeddings must match.
    np.testing.assert_allclose(out_b2[0], out_i0_alone[0], atol=1e-5,
                                err_msg="batch elem 0 mismatch")
    np.testing.assert_allclose(out_b2[1], out_i1_alone[0], atol=1e-5,
                                err_msg="batch elem 1 mismatch")


# ---------------------------------------------------------------------------
# ResidualCouplingBlock flow ONNX
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not FLOW_ONNX.exists(), reason="flow ONNX not exported yet")
def test_flow_onnx_loads() -> None:
    sess = ort.InferenceSession(str(FLOW_ONNX), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    # Four inputs: content, x_mask, se_src, se_tgt — one output: content_disentangled.
    assert len(inputs) == 4, f"expected 4 inputs, got {len(inputs)}"
    assert len(outputs) == 1, f"expected 1 output, got {len(outputs)}"

    in_names = [i.name for i in inputs]
    assert in_names == ["content", "x_mask", "se_src", "se_tgt"], (
        f"input names mismatch: {in_names}"
    )
    assert outputs[0].name == "content_disentangled"

    # Fixed channel dims (192 for content; 256 for speaker embeddings),
    # dynamic batch + time on content/mask, dynamic batch on embeddings.
    content_shape = inputs[0].shape  # ['batch', 192, 'time']
    mask_shape = inputs[1].shape     # ['batch', 1, 'time']
    se_src_shape = inputs[2].shape   # ['batch', 256, 1]
    se_tgt_shape = inputs[3].shape   # ['batch', 256, 1]
    out_shape = outputs[0].shape     # ['batch', 192, 'time']

    assert content_shape[1] == INTER_CHANNELS, f"content ch mismatch: {content_shape}"
    assert mask_shape[1] == 1, f"mask ch != 1: {mask_shape}"
    assert se_src_shape[1] == GIN_CHANNELS, f"se_src dim mismatch: {se_src_shape}"
    assert se_tgt_shape[1] == GIN_CHANNELS, f"se_tgt dim mismatch: {se_tgt_shape}"
    assert out_shape[1] == INTER_CHANNELS, f"output ch mismatch: {out_shape}"
    # T-broadcast dim of speaker embedding must be 1.
    assert se_src_shape[2] == 1, f"se_src T-dim != 1: {se_src_shape}"
    assert se_tgt_shape[2] == 1, f"se_tgt T-dim != 1: {se_tgt_shape}"


@pytest.mark.skipif(not FLOW_ONNX.exists(), reason="flow ONNX not exported yet")
def test_flow_runs_and_shape() -> None:
    sess = ort.InferenceSession(str(FLOW_ONNX), providers=["CPUExecutionProvider"])
    feed = _make_flow_feed(batch=1, t=50)
    out = sess.run(None, feed)[0]
    assert out.shape == (1, INTER_CHANNELS, 50), f"output shape mismatch: {out.shape}"
    assert np.all(np.isfinite(out)), "output contains NaN/Inf"
    # Flow preserves shape; output magnitude should be in the same order as input.
    in_mag = float(np.abs(feed["content"]).max())
    out_mag = float(np.abs(out).max())
    assert out_mag > 1e-4, f"output near-zero: max|out|={out_mag}"
    # Don't expect strict equality of magnitudes (coupling applies affine),
    # but they should be within a factor of 100 of each other.
    ratio = out_mag / max(in_mag, 1e-6)
    assert 0.01 < ratio < 100.0, f"flow magnitude drift suspicious: in={in_mag}, out={out_mag}, ratio={ratio}"


@pytest.mark.skipif(not FLOW_ONNX.exists(), reason="flow ONNX not exported yet")
def test_flow_dynamic_batch_and_time() -> None:
    sess = ort.InferenceSession(str(FLOW_ONNX), providers=["CPUExecutionProvider"])
    for batch, t in [(2, 80), (1, 100), (3, 30)]:
        feed = _make_flow_feed(batch=batch, t=t)
        out = sess.run(None, feed)[0]
        assert out.shape == (batch, INTER_CHANNELS, t), (
            f"dynamic axes broken for batch={batch}, t={t}: got {out.shape}"
        )


@pytest.mark.skipif(not FLOW_ONNX.exists(), reason="flow ONNX not exported yet")
def test_flow_identity_roundtrip_same_speaker() -> None:
    """With src==tgt, the disentangle-then-retarget roundtrip must recover input.

    The OpenVoice ResidualCouplingBlock uses mean-only affine coupling layers
    interleaved with Flip layers, both of which are invertible. The forward
    pass strips the source speaker; the reverse pass with the SAME speaker
    embedding must apply the exact inverse transform — recovering the input
    to within FP32 numerical precision.
    """
    sess = ort.InferenceSession(str(FLOW_ONNX), providers=["CPUExecutionProvider"])
    feed = _make_flow_feed(batch=1, t=50)
    feed["se_tgt"] = feed["se_src"]  # same speaker both sides
    out = sess.run(None, feed)[0]
    diff = float(np.abs(out - feed["content"]).max())
    assert diff < 1e-3, f"identity roundtrip |Δ|={diff:.2e} exceeds 1e-3"


@pytest.mark.skipif(not FLOW_ONNX.exists(), reason="flow ONNX not exported yet")
def test_flow_different_speakers_change_output() -> None:
    """With different src/tgt speaker embeddings, the output must differ from input."""
    sess = ort.InferenceSession(str(FLOW_ONNX), providers=["CPUExecutionProvider"])
    feed = _make_flow_feed(batch=1, t=50)
    out = sess.run(None, feed)[0]
    diff = float(np.abs(out - feed["content"]).max())
    # The disentangle+retarget with different speakers MUST change the content
    # (otherwise the flow would be a no-op, which would defeat the purpose).
    assert diff > 1e-3, (
        f"different-speaker VC did not modify content: |Δ|={diff:.2e}"
    )


# ---------------------------------------------------------------------------
# Cross-cutting: file size sanity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REF_ENC_ONNX.exists(), reason="ref_encoder ONNX not exported yet")
def test_ref_encoder_onnx_size() -> None:
    size_kb = REF_ENC_ONNX.stat().st_size / 1024
    # RefEncoder is ~0.81M params → FP32 ≈ 3.2 MB on disk. Allow some slack
    # for ONNX overhead but flag anything absurd.
    assert 100 < size_kb < 10_000, f"ref_encoder ONNX size {size_kb:.0f} KB out of expected range"


@pytest.mark.skipif(not FLOW_ONNX.exists(), reason="flow ONNX not exported yet")
def test_flow_onnx_size() -> None:
    size_kb = FLOW_ONNX.stat().st_size / 1024
    # Flow is ~8.69M params → FP32 ≈ 34 MB on disk. Allow slack.
    assert 5_000 < size_kb < 100_000, f"flow ONNX size {size_kb:.0f} KB out of expected range"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_flow_feed(batch: int, t: int) -> dict[str, np.ndarray]:
    """Build a feed dict for the flow ONNX graph with the given batch and time."""
    return {
        "content": np.random.randn(batch, INTER_CHANNELS, t).astype(np.float32),
        "x_mask": np.ones((batch, 1, t), dtype=np.float32),
        "se_src": np.random.randn(batch, GIN_CHANNELS, 1).astype(np.float32),
        "se_tgt": np.random.randn(batch, GIN_CHANNELS, 1).astype(np.float32),
    }
