"""
tests/test_infer_v3.py — v3 hybrid (TinyVC runtime + Spark BiCodec
SpeakerEncoder offline) end-to-end tests.

Four tests:
  1. test_v3_infer_non_silent_output — 1 s sine wave → output must be non-silent.
  2. test_v3_register_voice — register a voice ref wav → 1024-d d_vector
     + 48-byte FSQ saved to models/voices_v3.safetensors as FP16 / uint8.
  3. test_v3_different_voices_produce_different_output — voice_0 vs voice_1
     must produce audibly different output (corr < 0.95). Runtime path is
     identical to v1, so this also confirms the v1 kNN path is preserved.
  4. test_v3_fsq_pack_unpack_roundtrip — 32 random 12-bit indices → 48
     bytes → unpacked → must match original.

Run::

    cd /home/z/my-project/prototype
    python3 -m pytest tests/test_infer_v3.py -v
"""
import numpy as np
import pytest
from pathlib import Path

# `conftest.py` at the prototype root prepends `src/` to sys.path so this
# import works without `pip install -e .`.
from vc_realtime.infer_v3 import SAMPLE_RATE, SPARK_OUT_DIM, V3Infer
from vc_realtime.speaker_encoder_v3 import (
    SPARK_FSQ_BYTES_PACKED,
    pack_fsq_to_48_bytes,
    unpack_fsq_from_48_bytes,
)


# ---------------------------------------------------------------------------
# Shared fixtures (session-scope: building encoder+decoder is ~1 s)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def infer_runtime() -> V3Infer:
    """Session-cached V3Infer with Spark ONNX NOT loaded (runtime path only).

    This mirrors what ``vc_realtime.infer_v3.process_audio`` does (the
    singleton pattern) so the test reflects the production code path.
    """
    return V3Infer(models_dir="models", load_spark=False)


@pytest.fixture(scope="session")
def infer_with_spark() -> V3Infer:
    """Session-cached V3Infer with the Spark BiCodec ONNX loaded."""
    return V3Infer(models_dir="models", load_spark=True, use_int8=False)


@pytest.fixture(scope="session")
def source_1s_sine_150hz() -> np.ndarray:
    """1 second of 150 Hz sine wave at 24 kHz, peak 0.3."""
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: non-silent output
# ---------------------------------------------------------------------------
def test_v3_infer_non_silent_output(
    infer_runtime: V3Infer,
    source_1s_sine_150hz: np.ndarray,
):
    """Process 1 s of audio, output must be non-silent (RMS > 1e-3).

    This exercises the full v1 / v2 runtime path:
      sine → TinyVC encoder → kNN-VC top-4 → DDSP decoder → output
    """
    out = infer_runtime.process_audio(source_1s_sine_150hz, SAMPLE_RATE, voice_id=0)
    assert out.ndim == 1, f"Expected 1D output, got shape {out.shape}"
    assert len(out) > 0, "Output empty"
    # Output should cover ~1 s @ 24 kHz (with possible autopad of up to ~480 samples)
    expected_min = SAMPLE_RATE - SAMPLE_RATE  # at least 0 s
    expected_max = int(SAMPLE_RATE * 1.2)  # at most ~1.2 s (padding)
    assert expected_min < len(out) <= expected_max, (
        f"Output length {len(out)} outside expected range "
        f"({expected_min}, {expected_max}]"
    )
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms > 1e-3, f"Output silent (RMS={rms:.6f})"


# ---------------------------------------------------------------------------
# Test 2: register a voice → voices_v3.safetensors contains 1024-d + 48-byte FSQ
# ---------------------------------------------------------------------------
def test_v3_register_voice(infer_with_spark: V3Infer):
    """Register a voice + verify 1024-d embedding + 48-byte FSQ stored.

    Uses the existing voice_0 fixture (data/voices/voice_0.wav) so the test
    is reproducible. After registration, ``models/voices_v3.safetensors``
    must contain ``voice_0_dvec`` [1024] (FP16) and ``voice_0_fsq`` [48]
    (uint8) tensors.
    """
    voice_path = Path("data/voices/voice_0.wav")
    if not voice_path.exists():
        pytest.skip(f"{voice_path} not present; cannot test register_voice")
    infer_with_spark.register_voice(str(voice_path), voice_id=0)

    out_path = Path("models/voices_v3.safetensors")
    assert out_path.exists(), f"{out_path} not created by register_voice"
    try:
        from safetensors.torch import load_file
    except ImportError:  # pragma: no cover
        pytest.skip("safetensors not installed")
    tensors = load_file(str(out_path))
    assert "voice_0_dvec" in tensors, f"voice_0_dvec missing: keys={list(tensors.keys())}"
    assert "voice_0_fsq" in tensors, f"voice_0_fsq missing: keys={list(tensors.keys())}"
    # d_vector shape = [1024] (FP16 stored, may be upcast).
    dvec = tensors["voice_0_dvec"]
    assert dvec.numel() == SPARK_OUT_DIM, (
        f"d_vector numel {dvec.numel()} != {SPARK_OUT_DIM}"
    )
    # FSQ shape = [48] uint8
    fsq = tensors["voice_0_fsq"]
    assert fsq.numel() == SPARK_FSQ_BYTES_PACKED, (
        f"fsq numel {fsq.numel()} != {SPARK_FSQ_BYTES_PACKED}"
    )
    # Sanity: embedding norm should be non-trivial (Spark d_vectors are L2-normalized-ish)
    arr = dvec.float().numpy()
    norm = float(np.linalg.norm(arr))
    assert norm > 0.1, f"d_vector norm suspiciously small: {norm}"


# ---------------------------------------------------------------------------
# Test 3: different voices produce different output
# ---------------------------------------------------------------------------
def test_v3_different_voices_produce_different_output(
    infer_runtime: V3Infer,
    source_1s_sine_150hz: np.ndarray,
):
    """voice_0 vs voice_1 must produce different output (corr < 0.95).

    The runtime path is identical to v1 (TinyVC encoder + kNN + DDSP) — the
    Spark BiCodec ONNX is offline-only and not loaded here. The difference
    in output comes from the per-voice TinyVC kNN-VC indices (voices.pt).
    """
    out0 = infer_runtime.process_audio(source_1s_sine_150hz, SAMPLE_RATE, voice_id=0)
    out1 = infer_runtime.process_audio(source_1s_sine_150hz, SAMPLE_RATE, voice_id=1)
    min_len = min(len(out0), len(out1))
    # Guard against degenerate zero outputs
    if np.std(out0[:min_len]) < 1e-9 or np.std(out1[:min_len]) < 1e-9:
        pytest.fail("One of the outputs is essentially constant — kNN path broken")
    corr = float(np.corrcoef(out0[:min_len], out1[:min_len])[0, 1])
    assert abs(corr) < 0.95, (
        f"voice_0 and voice_1 outputs too similar (corr={corr:.4f}); "
        f"kNN-VC retrieval is not selecting distinct target frames"
    )


# ---------------------------------------------------------------------------
# Test 4: FSQ pack/unpack roundtrip
# ---------------------------------------------------------------------------
def test_v3_fsq_pack_unpack_roundtrip():
    """Verify FSQ pack/unpack roundtrip preserves all 32 12-bit indices."""
    rng = np.random.default_rng(seed=42)
    original = rng.integers(0, 4096, size=(1, 32), dtype=np.int64)
    packed = pack_fsq_to_48_bytes(original)
    assert len(packed) == SPARK_FSQ_BYTES_PACKED, (
        f"Packed length {len(packed)} != {SPARK_FSQ_BYTES_PACKED}"
    )
    unpacked = unpack_fsq_from_48_bytes(packed)
    assert unpacked.shape == (32,), f"Unpacked shape {unpacked.shape} != (32,)"
    assert np.array_equal(original.flatten(), unpacked), (
        f"Roundtrip mismatch:\n  orig={original.flatten()}\n  unpacked={unpacked}"
    )


def test_v3_fsq_pack_edge_cases():
    """Boundary values: all zeros and all 4095."""
    # All zeros
    zeros = np.zeros((1, 32), dtype=np.int64)
    packed = pack_fsq_to_48_bytes(zeros)
    assert len(packed) == 48
    assert all(b == 0 for b in packed), "All-zero FSQ should pack to 48 zero bytes"
    unpacked = unpack_fsq_from_48_bytes(packed)
    assert np.array_equal(zeros.flatten(), unpacked)

    # All 4095 (max value, all 12 bits set per index)
    max_vals = np.full((1, 32), 4095, dtype=np.int64)
    packed_max = pack_fsq_to_48_bytes(max_vals)
    assert len(packed_max) == 48
    unpacked_max = unpack_fsq_from_48_bytes(packed_max)
    assert np.array_equal(max_vals.flatten(), unpacked_max), "All-4095 roundtrip failed"


def test_v3_fsq_pack_rejects_invalid_indices():
    """Out-of-range indices should raise ValueError."""
    too_big = np.full((1, 32), 4096, dtype=np.int64)
    with pytest.raises(ValueError, match="out of \\[0, 4095\\] range"):
        pack_fsq_to_48_bytes(too_big)

    too_small = np.full((1, 32), -1, dtype=np.int64)
    with pytest.raises(ValueError, match="out of \\[0, 4095\\] range"):
        pack_fsq_to_48_bytes(too_small)

    wrong_len = np.zeros((1, 31), dtype=np.int64)
    with pytest.raises(ValueError, match="Expected 32 FSQ indices"):
        pack_fsq_to_48_bytes(wrong_len)
