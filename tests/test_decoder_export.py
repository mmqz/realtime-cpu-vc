"""Tests for the TinyVC decoder ONNX export (P2.1-F).

Verifies that ``models/source_net.onnx``, ``models/source_net.int8.onnx``,
``models/filter_net.onnx`` and ``models/filter_net.int8.onnx``:

  1. Exist and load in onnxruntime.
  2. Have the expected I/O contract (matching upstream
     ``tinyvc/export_onnx.py`` for source_net/filter_net):
       - source_net: 3 inputs (content, f0, energy) → 2 outputs
         (amplitudes, kernel)
       - filter_net: 4 inputs (content, f0, energy, source) → 1 output
         (waveform)
  3. Produce numerically plausible outputs for a random input.
  4. INT8 quantized model's outputs stay within a reasonable tolerance
     of the FP32 model's outputs (dynamic PTQ preserves semantic content).
  5. Honour dynamic batch + time axes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"

SOURCE_NET_FP32 = MODELS / "source_net.onnx"
SOURCE_NET_INT8 = MODELS / "source_net.int8.onnx"
FILTER_NET_FP32 = MODELS / "filter_net.onnx"
FILTER_NET_INT8 = MODELS / "filter_net.int8.onnx"

# TinyVC decoder constants (verified from
# repos/tinyvc/module/tinyvc/decoder.py:103-124 + 193-233).
N_FFT = 1920
FRAME_SIZE = 480
FFT_BIN = N_FFT // 2 + 1  # = 961 — kernel output channels
NUM_HARMONICS = 14
CONTENT_CHANNELS = 768
SOURCE_CHANNELS = NUM_HARMONICS + 2  # harmonics(15) + noise(1) = 16
T_FRAMES = 25  # 0.5 s @ 50 Hz frame rate — small enough to be fast
                # but long enough to exercise ConvNeXt dilations


def _session(path: Path) -> ort.InferenceSession:
    if not path.exists():
        pytest.skip(f"{path} not found; "
                    f"run scripts/export_tinyvc_decoder_onnx.py")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def src_fp32() -> ort.InferenceSession:
    return _session(SOURCE_NET_FP32)


@pytest.fixture(scope="module")
def src_int8() -> ort.InferenceSession:
    return _session(SOURCE_NET_INT8)


@pytest.fixture(scope="module")
def flt_fp32() -> ort.InferenceSession:
    return _session(FILTER_NET_FP32)


@pytest.fixture(scope="module")
def flt_int8() -> ort.InferenceSession:
    return _session(FILTER_NET_INT8)


def _src_feed(t: int = T_FRAMES, batch: int = 1, seed: int | None = None
              ) -> dict[str, np.ndarray]:
    """Build a source_net feed dict with the correct shape contract.

    content : [batch, 768, T]
    f0      : [batch, 1,   T]            (raw Hz; positive)
    energy  : [batch, 1,   T * frame_size]   (per-sample energy; positive)
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    randn = (lambda shape: rng.standard_normal(shape).astype(np.float32)
             if rng is not None
             else np.zeros(shape, dtype=np.float32))
    return {
        "content": randn((batch, CONTENT_CHANNELS, t)),
        "f0":      np.abs(randn((batch, 1, t))) + 1e-3,
        "energy":  np.abs(randn((batch, 1, t * FRAME_SIZE))),
    }


def _flt_feed(t: int = T_FRAMES, batch: int = 1, seed: int | None = None
              ) -> dict[str, np.ndarray]:
    """Build a filter_net feed dict with the correct shape contract.

    content : [batch, 768, T]
    f0      : [batch, 1,   T]
    energy  : [batch, 1,   T * frame_size]
    source  : [batch, 16,  T * frame_size]
    """
    feed = _src_feed(t=t, batch=batch, seed=seed)
    rng = np.random.default_rng(seed + 1) if seed is not None else None
    src_shape = (batch, SOURCE_CHANNELS, t * FRAME_SIZE)
    if rng is not None:
        feed["source"] = rng.standard_normal(src_shape).astype(np.float32)
    else:
        feed["source"] = np.zeros(src_shape, dtype=np.float32)
    return feed


# ---------------------------------------------------------------------------
# source_net contract
# ---------------------------------------------------------------------------

def test_source_net_onnx_loads(src_fp32: ort.InferenceSession) -> None:
    """source_net.onnx loads with the expected I/O contract."""
    inputs = src_fp32.get_inputs()
    outputs = src_fp32.get_outputs()
    assert len(inputs) == 3, f"expected 3 inputs, got {len(inputs)}"
    assert len(outputs) == 2, f"expected 2 outputs, got {len(outputs)}"

    # Input names match upstream tinyvc/export_onnx.py:54.
    expected_in = ["content", "f0", "energy"]
    assert [i.name for i in inputs] == expected_in, \
        f"input names mismatch: {[i.name for i in inputs]}"

    # Output names match upstream tinyvc/export_onnx.py:55.
    expected_out = ["amplitudes", "kernel"]
    assert [o.name for o in outputs] == expected_out, \
        f"output names mismatch: {[o.name for o in outputs]}"

    # Static channel dims.
    assert inputs[0].shape[1] == CONTENT_CHANNELS, \
        f"content channels mismatch: {inputs[0].shape[1]}"
    assert inputs[1].shape[1] == 1, \
        f"f0 channels mismatch: {inputs[1].shape[1]}"
    assert inputs[2].shape[1] == 1, \
        f"energy channels mismatch: {inputs[2].shape[1]}"
    # amplitudes has num_harmonics+1 channels (15); kernel has fft_bin (961).
    assert outputs[0].shape[1] == NUM_HARMONICS + 1, \
        f"amplitudes channels mismatch: {outputs[0].shape[1]}"
    assert outputs[1].shape[1] == FFT_BIN, \
        f"kernel channels mismatch: {outputs[1].shape[1]}"


def test_source_net_int8_loads(src_int8: ort.InferenceSession) -> None:
    """source_net.int8.onnx loads with the same I/O contract as FP32."""
    inputs = src_int8.get_inputs()
    outputs = src_int8.get_outputs()
    assert len(inputs) == 3
    assert len(outputs) == 2
    assert [i.name for i in inputs] == ["content", "f0", "energy"]
    assert [o.name for o in outputs] == ["amplitudes", "kernel"]
    assert inputs[0].shape[1] == CONTENT_CHANNELS
    assert outputs[0].shape[1] == NUM_HARMONICS + 1
    assert outputs[1].shape[1] == FFT_BIN


def test_source_net_forward_pass(src_fp32: ort.InferenceSession) -> None:
    """Forward pass with random input produces non-trivial + non-negative outputs.

    SourceNet applies ``F.elu(...) + 1.0`` to both amplitudes and kernel, so
    both must be non-negative (mathematically ≥ 0; ELU(x)+1 approaches 0 in
    the limit x → -∞ and equals exactly 0 in fp32 when exp underflows).
    """
    feed = _src_feed(seed=42)
    outs = src_fp32.run(None, feed)

    amps, kernel = outs[0], outs[1]
    t = T_FRAMES
    assert amps.shape == (1, NUM_HARMONICS + 1, t)
    assert kernel.shape == (1, FFT_BIN, t)
    # Both outputs are ELU+1, so non-negative.
    assert np.isfinite(amps).all() and (amps >= 0).all(), \
        "amplitudes must be finite & >= 0 (ELU+1)"
    assert np.isfinite(kernel).all() and (kernel >= 0).all(), \
        "kernel must be finite & >= 0 (ELU+1)"
    # Non-trivial variation across frames + harmonic bins.
    assert amps.std() > 1e-4, \
        f"amplitudes near-constant (std={amps.std():.4e})"


def test_source_net_dynamic_batch(src_fp32: ort.InferenceSession) -> None:
    """Vary batch dimension; verify both outputs scale."""
    feed = _src_feed(t=T_FRAMES, batch=3, seed=7)
    outs = src_fp32.run(None, feed)
    assert outs[0].shape == (3, NUM_HARMONICS + 1, T_FRAMES)
    assert outs[1].shape == (3, FFT_BIN, T_FRAMES)


def test_source_net_dynamic_length(src_fp32: ort.InferenceSession) -> None:
    """Vary time dimension; verify both outputs scale linearly.

    Energy input length must be T * frame_size (per-sample); content/f0 length
    is T (per-frame). MaxPool1d(frame_size, frame_size) down-samples energy to
    match the frame-rate content.
    """
    t = 40
    feed = _src_feed(t=t, seed=11)
    outs = src_fp32.run(None, feed)
    assert outs[0].shape == (1, NUM_HARMONICS + 1, t)
    assert outs[1].shape == (1, FFT_BIN, t)


def test_source_net_int8_approximates_fp32(
    src_fp32: ort.InferenceSession,
    src_int8: ort.InferenceSession,
) -> None:
    """Per-channel dynamic INT8 PTQ preserves amplitudes + kernel within ~15%.

    SourceNet has only ~1.7 M params (3 ConvNeXt layers + 1×1 conv heads) —
    small model with strong ELU+1 output non-negativity. Dynamic PTQ without
    calibration typically lands in 1–10% L2 territory here.
    """
    feed = _src_feed(seed=123)
    out_fp32 = src_fp32.run(None, feed)
    out_int8 = src_int8.run(None, feed)

    for name, fp32, int8 in [("amplitudes", out_fp32[0], out_int8[0]),
                            ("kernel", out_fp32[1], out_int8[1])]:
        rel = float(
            np.linalg.norm(int8 - fp32)
            / max(np.linalg.norm(fp32), 1e-9)
        )
        assert rel < 0.15, (
            f"{name} relative L2 error too high: {rel:.4e} "
            f"(expected < 0.15)"
        )


# ---------------------------------------------------------------------------
# filter_net contract
# ---------------------------------------------------------------------------

def test_filter_net_onnx_loads(flt_fp32: ort.InferenceSession) -> None:
    """filter_net.onnx loads with the expected I/O contract."""
    inputs = flt_fp32.get_inputs()
    outputs = flt_fp32.get_outputs()
    assert len(inputs) == 4, f"expected 4 inputs, got {len(inputs)}"
    assert len(outputs) == 1, f"expected 1 output, got {len(outputs)}"

    expected_in = ["content", "f0", "energy", "source"]
    assert [i.name for i in inputs] == expected_in, \
        f"input names mismatch: {[i.name for i in inputs]}"

    assert outputs[0].name == "waveform", \
        f"output name mismatch: {outputs[0].name}"

    # Static channel dims.
    assert inputs[0].shape[1] == CONTENT_CHANNELS
    assert inputs[1].shape[1] == 1
    assert inputs[2].shape[1] == 1
    assert inputs[3].shape[1] == SOURCE_CHANNELS, \
        f"source channels mismatch: {inputs[3].shape[1]}"
    # Output waveform is mono.
    assert outputs[0].shape[1] == 1


def test_filter_net_int8_loads(flt_int8: ort.InferenceSession) -> None:
    """filter_net.int8.onnx loads with the same I/O contract as FP32."""
    inputs = flt_int8.get_inputs()
    outputs = flt_int8.get_outputs()
    assert len(inputs) == 4
    assert len(outputs) == 1
    assert [i.name for i in inputs] == ["content", "f0", "energy", "source"]
    assert outputs[0].name == "waveform"
    assert inputs[0].shape[1] == CONTENT_CHANNELS
    assert inputs[3].shape[1] == SOURCE_CHANNELS


def test_filter_net_forward_pass(flt_fp32: ort.InferenceSession) -> None:
    """Forward pass with random input produces a finite waveform.

    FilterNet output is the final 7-tap Conv1d with replicate padding; output
    length = T_frames * frame_size (per-sample rate).
    """
    feed = _flt_feed(seed=42)
    outs = flt_fp32.run(None, feed)

    wf = outs[0]
    assert wf.shape == (1, 1, T_FRAMES * FRAME_SIZE)
    assert np.isfinite(wf).all(), "waveform has NaN/inf"
    # Random inputs produce a non-zero output (final conv has bias term).
    assert wf.std() > 1e-6, \
        f"waveform near-constant (std={wf.std():.4e})"


def test_filter_net_dynamic_batch(flt_fp32: ort.InferenceSession) -> None:
    """Vary batch dimension; verify waveform scales."""
    feed = _flt_feed(t=T_FRAMES, batch=3, seed=7)
    outs = flt_fp32.run(None, feed)
    assert outs[0].shape == (3, 1, T_FRAMES * FRAME_SIZE)


def test_filter_net_dynamic_length(flt_fp32: ort.InferenceSession) -> None:
    """Vary time dimension; verify waveform length scales linearly."""
    t = 40
    feed = _flt_feed(t=t, seed=11)
    outs = flt_fp32.run(None, feed)
    assert outs[0].shape == (1, 1, t * FRAME_SIZE)


def test_filter_net_int8_approximates_fp32(
    flt_fp32: ort.InferenceSession,
    flt_int8: ort.InferenceSession,
) -> None:
    """Per-channel dynamic INT8 PTQ preserves waveform within ~25%.

    FilterNet is a 5-stage U-Net with dilated ConvNeXt layers (~9 M params).
    Dynamic PTQ without calibration typically lands in 5–20% L2 territory;
    we assert < 25% as a sensible invariant that the INT8 graph is not
    silently broken.
    """
    feed = _flt_feed(seed=123)
    out_fp32 = flt_fp32.run(None, feed)[0]
    out_int8 = flt_int8.run(None, feed)[0]
    rel = float(
        np.linalg.norm(out_int8 - out_fp32)
        / max(np.linalg.norm(out_fp32), 1e-9)
    )
    assert rel < 0.25, (
        f"waveform relative L2 error too high: {rel:.4e} "
        f"(expected < 0.25)"
    )
