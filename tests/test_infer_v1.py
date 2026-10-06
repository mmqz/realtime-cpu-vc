"""
tests/test_infer_v1.py — v1 baseline (TinyVC + kNN + DDSP) end-to-end tests.

Three functional tests verify the v1 baseline produces a non-silent
waveform of the correct length and that two distinct target voices yield
audibly different outputs.

Run::

    cd /home/z/my-project/prototype
    python3 -m pytest tests/test_infer_v1.py -v
"""
import numpy as np
import pytest

# `conftest.py` at the prototype root prepends `src/` to sys.path so this
# import works without `pip install -e .`.
from vc_realtime.infer_v1 import SAMPLE_RATE, V1Infer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope='session')
def infer() -> V1Infer:
    """Session-cached V1Infer — building encoder+decoder once is expensive."""
    return V1Infer(models_dir='models', device='cpu')


@pytest.fixture(scope='session')
def source_1s_sine_150hz() -> np.ndarray:
    """1 second of 150 Hz sine wave at 24 kHz, peak 0.3 (well above the
    -3 dBFS peak-normalization threshold applied inside process_audio)."""
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1 — non-silent output
# ---------------------------------------------------------------------------
def test_v1_infer_non_silent_output(infer: V1Infer,
                                     source_1s_sine_150hz: np.ndarray):
    """Process 1 s of audio; output must be non-silent (RMS > 0.001)."""
    out = infer.process_audio(source_1s_sine_150hz, SAMPLE_RATE,
                              voice_id=0)
    assert out.ndim == 1, f"Output must be 1-D, got shape {out.shape}"
    assert len(out) > 0, "Output is empty"
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms > 0.001, f"Output is silent (RMS={rms:.6f})"


# ---------------------------------------------------------------------------
# Test 2 — output length matches input length (within ±50 ms)
# ---------------------------------------------------------------------------
def test_v1_infer_output_length_matches(infer: V1Infer,
                                         source_1s_sine_150hz: np.ndarray):
    """Output length must be within 50 ms (1200 samples at 24 kHz) of the
    input length. TinyVC's decoder reconstructs sample-rate output from the
    frame-rate content feature, so |Δ| < 480 (= 1 frame = 20 ms) in
    practice; we allow 1200 for safety margin."""
    out = infer.process_audio(source_1s_sine_150hz, SAMPLE_RATE,
                              voice_id=0)
    expected = len(source_1s_sine_150hz)
    actual = len(out)
    slack = int(SAMPLE_RATE * 0.05)  # 50 ms = 1200 samples
    assert abs(actual - expected) < slack, (
        f"Output length {actual} != input {expected} "
        f"(slack={slack}). Δ={actual - expected} samples.")


# ---------------------------------------------------------------------------
# Test 3 — different voices produce audibly different output
# ---------------------------------------------------------------------------
def test_v1_different_voices_produce_different_output(
        infer: V1Infer, source_1s_sine_150hz: np.ndarray):
    """voice_id=0 vs voice_id=1 must produce audibly different outputs
    (Pearson correlation < 0.95).

    voice_0 is a synthetic 100 Hz reference and voice_1 a 150 Hz reference;
    their kNN feature banks are sufficiently distinct that the same source
    (150 Hz sine) retrieves different target frames per voice, which the
    DDSP decoder renders as waveforms with different harmonic/energy
    envelopes."""
    out0 = infer.process_audio(source_1s_sine_150hz, SAMPLE_RATE,
                               voice_id=0)
    out1 = infer.process_audio(source_1s_sine_150hz, SAMPLE_RATE,
                               voice_id=1)
    min_len = min(len(out0), len(out1))
    a = out0[:min_len].astype(np.float64)
    b = out1[:min_len].astype(np.float64)
    # Pearson correlation — both arrays have non-zero variance since they
    # are non-silent (verified by Test 1).
    corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(corr) < 0.95, (
        f"voice_0 and voice_1 outputs too similar (corr={corr:.4f}). "
        f"Expected corr < 0.95 to confirm voice-distinctness.")
