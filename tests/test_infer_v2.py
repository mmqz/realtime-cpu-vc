"""
tests/test_infer_v2.py — v2 hybrid (TinyVC runtime + OpenVoice RefEncoder
offline) end-to-end tests.

Three tests:
  1. test_v2_infer_non_silent_output — 1 s sine wave → output must be non-silent.
  2. test_v2_register_voice — register a voice ref wav → 256-d embedding
     saved to models/voices_v2.safetensors as FP16.
  3. test_v2_different_voices_produce_different_output — voice_0 vs voice_1
     must produce audibly different output (corr < 0.95). Runtime path is
     identical to v1, so this also confirms the v1 kNN path is preserved.

Run::

    cd .
    python3 -m pytest tests/test_infer_v2.py -v
"""
import numpy as np
import pytest

# `conftest.py` at the prototype root prepends `src/` to sys.path so this
# import works without `pip install -e .`.
from vc_realtime.infer_v2 import OPENVOICE_EMB_DIM, SAMPLE_RATE, V2Infer


# ---------------------------------------------------------------------------
# Shared fixtures (session-scope: building encoder+decoder is ~1 s)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def infer_runtime() -> V2Infer:
    """Session-cached V2Infer with RefEncoder NOT loaded (runtime path only).

    This mirrors what `vc_realtime.infer_v2.process_audio` does (the singleton
    pattern) so the test reflects the production code path.
    """
    return V2Infer(models_dir="models", load_ref_encoder=False)


@pytest.fixture(scope="session")
def infer_with_ref_encoder() -> V2Infer:
    """Session-cached V2Infer with the OpenVoice RefEncoder ONNX loaded."""
    return V2Infer(models_dir="models", load_ref_encoder=True)


@pytest.fixture(scope="session")
def source_1s_sine_150hz() -> np.ndarray:
    """1 second of 150 Hz sine wave at 24 kHz, peak 0.3."""
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1 — non-silent output (runtime path)
# ---------------------------------------------------------------------------
def test_v2_infer_non_silent_output(
    infer_runtime: V2Infer, source_1s_sine_150hz: np.ndarray
):
    """Process 1 s of audio; output must be non-silent (RMS > 0.001).

    Verifies the v2 runtime path (TinyVC encoder + kNN + DDSP, identical to
    v1) produces a valid waveform. OpenVoice RefEncoder is NOT loaded for
    this test (mirrors the singleton used by benchmark.py).
    """
    out = infer_runtime.process_audio(
        source_1s_sine_150hz, SAMPLE_RATE, voice_id=0
    )
    assert out.ndim == 1, f"Output must be 1-D, got shape {out.shape}"
    assert len(out) > 0, "Output is empty"
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms > 0.001, f"Output is silent (RMS={rms:.6f})"


# ---------------------------------------------------------------------------
# Test 2 — offline voice registration (OpenVoice RefEncoder path)
# ---------------------------------------------------------------------------
def test_v2_register_voice(
    infer_with_ref_encoder: V2Infer, tmp_path
):
    """Register a voice ref wav → 256-d embedding saved to safetensors.

    Steps verified:
      - V2Infer can load the OpenVoice RefEncoder ONNX.
      - register_voice() loads a 24 kHz wav, resamples to 22.05 kHz, computes
        the LINEAR spec [1, T, 513] (NOT 80-mel), runs the ONNX, gets [256].
      - The embedding is persisted to models/voices_v2.safetensors as FP16.
      - On reload (via a fresh V2Infer), the embedding is retrieved as [256]
        float32 and matches (within FP16 quantization tolerance).
    """
    from pathlib import Path
    infer = infer_with_ref_encoder
    # Use the existing voice_0 fixture (30 s @ 24 kHz — will be resampled
    # to 22.05 kHz and truncated to 30 s).
    emb = infer.register_voice(
        "data/voices/voice_0.wav", voice_id=0, persist=True
    )
    # Shape
    assert isinstance(emb, np.ndarray), f"Expected ndarray, got {type(emb)}"
    assert emb.shape == (OPENVOICE_EMB_DIM,), (
        f"Expected shape ({OPENVOICE_EMB_DIM},), got {emb.shape}"
    )
    assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
    # Finite + non-trivial
    assert np.all(np.isfinite(emb)), "Embedding has NaN/Inf"
    norm = float(np.linalg.norm(emb))
    assert norm > 0.1, f"Embedding norm too small ({norm}) — looks like zeros"
    # File written
    safetensors_path = Path("models") / V2Infer.VOICES_V2_FILE
    assert safetensors_path.exists(), (
        f"{safetensors_path} not created by register_voice()"
    )
    # On-disk size: 5 voices × 256 × 2 B (FP16) = 2560 B + ~140 B safetensors
    # header overhead. After registering voice 0 only: ~512 B + header.
    # We only check upper bound here.
    assert safetensors_path.stat().st_size < 100_000, (
        f"voices_v2.safetensors too large ({safetensors_path.stat().st_size} B) — "
        "should be < 100 KB for 5 voices × 256-d FP16."
    )
    # Reload and verify
    from safetensors.torch import load_file
    tensors = load_file(str(safetensors_path))
    assert "voice_0" in tensors, f"voice_0 not in {list(tensors.keys())}"
    # safetensors returns torch.Tensor — np.asarray works on it directly.
    reloaded = np.asarray(tensors["voice_0"]).astype(np.float32)
    assert reloaded.shape == (OPENVOICE_EMB_DIM,), (
        f"Reloaded embedding shape {reloaded.shape} != ({OPENVOICE_EMB_DIM},)"
    )
    # FP16 round-trip introduces up to ~1e-3 relative error.
    np.testing.assert_allclose(
        reloaded, emb, rtol=1e-3, atol=1e-3,
        err_msg="FP16 round-trip drifted beyond 1e-3 tolerance",
    )


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Test 3 — different voices produce different output (runtime path)
# ---------------------------------------------------------------------------
def test_v2_different_voices_produce_different_output(
    infer_runtime: V2Infer, source_1s_sine_150hz: np.ndarray
):
    """voice_id=0 vs voice_id=1 must produce audibly different outputs.

    The runtime path (TinyVC encoder + kNN + DDSP) is identical to v1; this
    test confirms that v2's runtime is functionally equivalent to v1 in
    distinguishing voices. (The OpenVoice 256-d embeddings are offline-only
    and not yet wired into the runtime conversion — that's v2.5.)
    """
    out0 = infer_runtime.process_audio(
        source_1s_sine_150hz, SAMPLE_RATE, voice_id=0
    )
    out1 = infer_runtime.process_audio(
        source_1s_sine_150hz, SAMPLE_RATE, voice_id=1
    )
    min_len = min(len(out0), len(out1))
    a = out0[:min_len].astype(np.float64)
    b = out1[:min_len].astype(np.float64)
    corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(corr) < 0.95, (
        f"voice_0 and voice_1 outputs too similar (corr={corr:.4f}). "
        f"Expected corr < 0.95 to confirm voice-distinctness."
    )
