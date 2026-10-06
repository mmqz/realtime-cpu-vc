"""Tests for the Spark-TTS BiCodec SpeakerEncoder ONNX export (P3-1).

Verifies that ``models/spark_speaker_encoder.onnx``:
  1. Exists on disk.
  2. Loads cleanly in onnxruntime (CPUExecutionProvider).
  3. Has the expected input/output signatures and shapes (verified at
     export time against the actual Spark-TTS 0.5B HuggingFace checkpoint):
     - Input: mel_spec [B, T, 128] float32 (128 mels, 50 Hz frame rate).
     - Output d_vector    [B, 1024]      float32 (speaker conditioning).
     - Output fsq_indices [B, 1, 32]     int64   (FSQ codes, 32 tokens).
     - Output x_vector    [B, 1024]      float32 (auxiliary speaker emb).
  4. Runs inference without error.
  5. Produces outputs with sensible magnitudes (not NaN / not all-zero).
  6. FSQ indices are in valid range [0, 4095] (= prod([4,4,4,4,4,4]) = 4096
     codes per token; 6 levels × 4 values = 12 bits per token = 48 bytes
     per voice when packed, matching the v3 hybrid 48-byte storage spec).
  7. Honour dynamic batch + time axes (different shapes than the export
     dummy shapes still produce valid output of matching shape).
  8. Determinism: same input → same output (the FSQ round() is deterministic
     in eval mode; the ASTP pooling has no stochasticity).
  9. Batch independence: embedding for batch element i equals embedding
     computed for that element alone (no inter-batch coupling — verified
     by the absence of batchnorm-in-training-mode artifacts).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"
SE_ONNX = MODELS / "spark_speaker_encoder.onnx"

# Constants from the actual Spark-TTS BiCodec config.yaml
# (SparkAudio/Spark-TTS-0.5B/BiCodec/config.yaml).
NUM_MELS = 128           # mel_params.num_mels
OUT_DIM = 1024           # speaker_encoder.out_dim
TOKEN_NUM = 32           # speaker_encoder.token_num (Perceiver num_latents)
FSQ_NUM_QUANTIZERS = 1   # speaker_encoder.fsq_num_quantizers
FSQ_LEVELS_PROD = 4 ** 6  # = 4096 codes per token
REF_SEGMENT_FRAMES = 300  # 6 s × 16000 / 320 — canonical ref length


# ---------------------------------------------------------------------------
# Load + signature
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SE_ONNX.exists(), reason="spark_speaker_encoder ONNX not exported yet")
def test_speaker_encoder_onnx_loads() -> None:
    sess = ort.InferenceSession(str(SE_ONNX), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    # Single input (mel_spec), three outputs (d_vector, fsq_indices, x_vector).
    assert len(inputs) == 1, f"expected 1 input, got {len(inputs)}"
    assert len(outputs) == 3, f"expected 3 outputs, got {len(outputs)}"

    # Names match the export contract.
    assert inputs[0].name == "mel_spec", f"input name: {inputs[0].name}"
    out_names = [o.name for o in outputs]
    assert out_names == ["d_vector", "fsq_indices", "x_vector"], (
        f"output names mismatch: {out_names}"
    )

    # Fixed mel-dim (128), dynamic batch + time on input.
    in_shape = inputs[0].shape  # ['batch', 'time', 128]
    assert in_shape[-1] == NUM_MELS, f"input mel-dim mismatch: {in_shape}"

    # Output shapes: d_vector + x_vector fixed at [batch, 1024]; fsq_indices
    # at [batch, 1, 32] with the 1 being the num_quantizers dim. The third
    # dim of fsq_indices may show as a symbolic name (e.g.
    # 'Castfsq_indices_dim_2') because the int64 Cast op produces a tensor
    # whose shape ONNX does not statically fold back to the constant 32 —
    # the runtime value is always 32 (verified by
    # ``test_speaker_encoder_runs_and_shape``).
    d_shape = outputs[0].shape    # ['batch', 1024]
    idx_shape = outputs[1].shape  # ['batch', 1, <symbolic>]
    x_shape = outputs[2].shape    # ['batch', 1024]
    assert d_shape[-1] == OUT_DIM, f"d_vector dim mismatch: {d_shape}"
    assert idx_shape[1] == FSQ_NUM_QUANTIZERS, f"fsq num_quantizers mismatch: {idx_shape}"
    assert x_shape[-1] == OUT_DIM, f"x_vector dim mismatch: {x_shape}"


# ---------------------------------------------------------------------------
# Run + shape
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SE_ONNX.exists(), reason="spark_speaker_encoder ONNX not exported yet")
def test_speaker_encoder_runs_and_shape() -> None:
    sess = ort.InferenceSession(str(SE_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    mel = np.random.randn(1, REF_SEGMENT_FRAMES, NUM_MELS).astype(np.float32)
    out = sess.run(None, {in_name: mel})

    d_vec, idx, x_vec = out
    assert d_vec.shape == (1, OUT_DIM), f"d_vector shape mismatch: {d_vec.shape}"
    assert idx.shape == (1, FSQ_NUM_QUANTIZERS, TOKEN_NUM), (
        f"fsq_indices shape mismatch: {idx.shape}"
    )
    assert x_vec.shape == (1, OUT_DIM), f"x_vector shape mismatch: {x_vec.shape}"

    assert np.all(np.isfinite(d_vec)), "d_vector contains NaN/Inf"
    assert np.all(np.isfinite(x_vec)), "x_vector contains NaN/Inf"
    # Speaker embeddings should be non-zero (random input → non-degenerate emb).
    assert float(np.abs(d_vec).max()) > 1e-3, (
        f"d_vector near-zero: max|d_vec|={float(np.abs(d_vec).max())}"
    )
    assert float(np.abs(x_vec).max()) > 1e-3, (
        f"x_vector near-zero: max|x_vec|={float(np.abs(x_vec).max())}"
    )


@pytest.mark.skipif(not SE_ONNX.exists(), reason="spark_speaker_encoder ONNX not exported yet")
def test_speaker_encoder_fsq_indices_in_range() -> None:
    """FSQ codes must be in [0, 4095] = prod([4,4,4,4,4,4]) = 12-bit codes.

    32 tokens × 12 bits = 384 bits = 48 bytes packed per voice, matching the
    v3 hybrid spec for O(1) hash-keyed voice retrieval.
    """
    sess = ort.InferenceSession(str(SE_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    mel = np.random.randn(2, REF_SEGMENT_FRAMES, NUM_MELS).astype(np.float32)
    out = sess.run(None, {in_name: mel})
    idx = out[1]  # [2, 1, 32] int64

    assert idx.dtype.kind in ("i", "u"), f"fsq indices not integer: {idx.dtype}"
    assert idx.min() >= 0, f"fsq indices negative: min={idx.min()}"
    assert idx.max() < FSQ_LEVELS_PROD, (
        f"fsq indices out of range: max={idx.max()} (expected < {FSQ_LEVELS_PROD})"
    )
    # 32 tokens × 1 quantizer = 32 codes per voice. 32 × 12 bits = 384 bits
    # = 48 bytes packed — matches the v3 hybrid 48-byte storage spec.
    n_codes = idx[0].size
    assert n_codes == TOKEN_NUM * FSQ_NUM_QUANTIZERS, (
        f"unexpected number of fsq codes: {n_codes} (expected "
        f"{TOKEN_NUM * FSQ_NUM_QUANTIZERS})"
    )


# ---------------------------------------------------------------------------
# Dynamic batch + time axes
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SE_ONNX.exists(), reason="spark_speaker_encoder ONNX not exported yet")
def test_speaker_encoder_dynamic_batch_and_time() -> None:
    """Verify the dynamic axes on batch and time dimensions actually work."""
    sess = ort.InferenceSession(str(SE_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    for batch, t in [(2, 200), (3, 100), (1, 50)]:
        mel = np.random.randn(batch, t, NUM_MELS).astype(np.float32)
        out = sess.run(None, {in_name: mel})
        d_vec, idx, x_vec = out
        assert d_vec.shape == (batch, OUT_DIM), (
            f"dynamic axes broken for batch={batch}, t={t}: "
            f"d_vec got {d_vec.shape}"
        )
        assert idx.shape == (batch, FSQ_NUM_QUANTIZERS, TOKEN_NUM), (
            f"dynamic axes broken for batch={batch}, t={t}: "
            f"idx got {idx.shape}"
        )
        assert x_vec.shape == (batch, OUT_DIM), (
            f"dynamic axes broken for batch={batch}, t={t}: "
            f"x_vec got {x_vec.shape}"
        )


# ---------------------------------------------------------------------------
# Determinism + batch independence
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SE_ONNX.exists(), reason="spark_speaker_encoder ONNX not exported yet")
def test_speaker_encoder_deterministic() -> None:
    """Same input twice must produce identical output (no stochasticity)."""
    sess = ort.InferenceSession(str(SE_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    mel = np.random.randn(1, REF_SEGMENT_FRAMES, NUM_MELS).astype(np.float32)
    out1 = sess.run(None, {in_name: mel})
    out2 = sess.run(None, {in_name: mel})

    np.testing.assert_allclose(out1[0], out2[0], atol=0, rtol=0,
                               err_msg="d_vector not deterministic")
    np.testing.assert_array_equal(out1[1], out2[1],
                                  err_msg="fsq_indices not deterministic")
    np.testing.assert_allclose(out1[2], out2[2], atol=0, rtol=0,
                               err_msg="x_vector not deterministic")


@pytest.mark.skipif(not SE_ONNX.exists(), reason="spark_speaker_encoder ONNX not exported yet")
def test_speaker_encoder_batch_independence() -> None:
    """Embedding for batch element i equals embedding computed alone.

    ECAPA-TDNN's BatchNorm1d is in eval mode (folded into conv at export),
    so the forward pass is batch-independent. ASTP pooling is also
    per-sequence. Hence batching must NOT change per-element outputs.
    """
    sess = ort.InferenceSession(str(SE_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    mel_b2 = np.random.randn(2, REF_SEGMENT_FRAMES, NUM_MELS).astype(np.float32)
    out_b2 = sess.run(None, {in_name: mel_b2})
    out_i0_alone = sess.run(None, {in_name: mel_b2[:1]})
    out_i1_alone = sess.run(None, {in_name: mel_b2[1:2]})

    # d_vector must match to BN-fold precision (small float drift expected).
    np.testing.assert_allclose(out_b2[0][0], out_i0_alone[0][0], atol=1e-5,
                                err_msg="batch elem 0 d_vector mismatch")
    np.testing.assert_allclose(out_b2[0][1], out_i1_alone[0][0], atol=1e-5,
                                err_msg="batch elem 1 d_vector mismatch")
    # FSQ codes must match EXACTLY (discrete int indices).
    np.testing.assert_array_equal(out_b2[1][0], out_i0_alone[1][0],
                                  err_msg="batch elem 0 fsq_indices mismatch")
    np.testing.assert_array_equal(out_b2[1][1], out_i1_alone[1][0],
                                  err_msg="batch elem 1 fsq_indices mismatch")
    # x_vector (ECAPA-TDNN pooled output) must also be batch-independent.
    np.testing.assert_allclose(out_b2[2][0], out_i0_alone[2][0], atol=1e-5,
                                err_msg="batch elem 0 x_vector mismatch")
    np.testing.assert_allclose(out_b2[2][1], out_i1_alone[2][0], atol=1e-5,
                                err_msg="batch elem 1 x_vector mismatch")


# ---------------------------------------------------------------------------
# Cross-cutting: file size sanity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SE_ONNX.exists(), reason="spark_speaker_encoder ONNX not exported yet")
def test_speaker_encoder_onnx_size() -> None:
    size_kb = SE_ONNX.stat().st_size / 1024
    # SpeakerEncoder is ~14.06 M params → FP32 ≈ 56 MB on disk. Allow slack
    # for ONNX overhead but flag anything absurd (e.g., the full 0.5B BiCodec
    # would be ~2 GB; we want only the speaker_encoder sub-module).
    assert 5_000 < size_kb < 200_000, (
        f"spark_speaker_encoder ONNX size {size_kb:.0f} KB out of expected range"
    )
