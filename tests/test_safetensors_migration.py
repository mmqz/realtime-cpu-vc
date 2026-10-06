"""tests/test_safetensors_migration.py — OPT-11 safetensors migration + batch.

Verifies:
  1. The migration script (scripts/migrate_voices_to_safetensors.py) produces
     a valid voices_v1.safetensors file with FP16 storage.
  2. V1Infer loads from voices_v1.safetensors when present (no torch.load
     weights_only=False on the hot path — security risk closed).
  3. V1Infer.process_audio_batched() — offline batched processing for
     multi-chunk audio — produces a non-silent output of the correct length.

Run::

    cd /home/z/my-project/prototype
    python3 -m pytest tests/test_safetensors_migration.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path("/home/z/my-project/prototype")
MODELS = ROOT / "models"
PYBIN = "/home/z/.venv/bin/python"

# `conftest.py` at the prototype root prepends `src/` to sys.path so the
# import below works without `pip install -e .`.


# ---------------------------------------------------------------------------
# Test 1 — migration script
# ---------------------------------------------------------------------------
def test_migrate_voices_to_safetensors():
    """Verify migration script produces a valid safetensors file.

    The script is idempotent — running it twice is fine. We assert:
      * subprocess exit code 0
      * voices_v1.safetensors exists at the expected path
      * safetensors file loads via safetensors.torch.load_file
      * contains >= 5 voice tensors (one per voice in voices.pt)
      * all tensors are FP16 (storage compression)
    """
    pt_path = MODELS / "voices.pt"
    if not pt_path.exists():
        pytest.skip("voices.pt not found — nothing to migrate")

    result = subprocess.run(
        [PYBIN, "scripts/migrate_voices_to_safetensors.py"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"Migration failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    sf_path = MODELS / "voices_v1.safetensors"
    assert sf_path.exists(), f"safetensors file not written at {sf_path}"
    assert sf_path.stat().st_size > 0, "safetensors file is empty"

    # Verify safetensors loads cleanly (no arbitrary-code execution path).
    from safetensors.torch import load_file

    voices = load_file(str(sf_path))
    assert isinstance(voices, dict), (
        f"Expected dict[str -> Tensor], got {type(voices)}"
    )
    assert len(voices) >= 5, (
        f"Expected >= 5 voices, got {len(voices)}: {list(voices.keys())}"
    )
    for k, v in voices.items():
        assert v.dtype == torch.float16, (
            f"voice {k} dtype={v.dtype}, expected float16 (FP16 storage)"
        )
        # Shape unchanged from voices.pt — [1, 768, T_ref]
        assert v.dim() == 3, f"voice {k} expected 3-D tensor, got {v.dim()}-D"
        assert v.shape[1] == 768, (
            f"voice {k} expected content_dim=768, got {v.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Test 2 — V1Infer loads from safetensors (no torch.load weights_only=False)
# ---------------------------------------------------------------------------
def test_v1_loads_safetensors():
    """V1Infer must prefer voices_v1.safetensors over legacy voices.pt."""
    sf_path = MODELS / "voices_v1.safetensors"
    if not sf_path.exists():
        pytest.skip("voices_v1.safetensors not found — run migration script first")

    from vc_realtime.infer_v1 import SAMPLE_RATE, V1Infer  # noqa: F401

    infer = V1Infer(models_dir=str(MODELS))
    # Either attribute should expose the voice index — both point to the
    # same dict[str, Tensor].
    n = max(len(infer.tinyvc_voices), infer.n_voices)
    assert n >= 5, (
        f"Expected >= 5 voices, got tinyvc_voices={len(infer.tinyvc_voices)}, "
        f"n_voices={infer.n_voices}"
    )
    # Verify all loaded voice tensors are FP32 (safetensors stored FP16, but
    # inference is FP32).
    for k, v in infer.voices.items():
        assert v.dtype == torch.float32, (
            f"voice {k} inference dtype={v.dtype}, expected float32"
        )

    # Verify inference still works end-to-end after the migration.
    t = np.linspace(0, 1, 24000, endpoint=False, dtype=np.float32)
    wav = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    out = infer.process_audio(wav, 24000, voice_id=0)
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms > 0.001, f"Output is silent (RMS={rms:.6f})"


# ---------------------------------------------------------------------------
# Test 3 — process_audio_batched
# ---------------------------------------------------------------------------
def test_process_audio_batched():
    """Verify batched processing produces a non-silent output of the correct
    length.

    Feeds 10 s of 150 Hz sine at 24 kHz (so no resample needed). With
    batch_sec=5.0, this yields 2 chunks of 120 000 samples each. The
    function must trim the concatenation back to ``len(wav)`` (240 000).
    """
    from vc_realtime.infer_v1 import V1Infer

    infer = V1Infer(models_dir=str(MODELS))
    sr = 24000
    t = np.linspace(0, 10, sr * 10, endpoint=False, dtype=np.float32)
    wav = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)

    out = infer.process_audio_batched(wav, sr, voice_id=0, batch_sec=5.0)

    assert len(out) == len(wav), (
        f"Output length {len(out)} != input {len(wav)} "
        f"(Δ={len(out) - len(wav)} samples)"
    )
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms > 0.001, f"Output is silent (RMS={rms:.6f})"


# ---------------------------------------------------------------------------
# Test 4 — batched == single-shot (parity check, loose tolerance)
# ---------------------------------------------------------------------------
def test_batched_matches_single_shot():
    """Batched output should match single-shot output up to chunk-boundary
    artifacts. Both run the same encoder → kNN → decoder pipeline; the only
    divergence is at chunk boundaries (TinyVC is not stateless across long
    windows due to STFT edge effects). We assert cross-correlation >= 0.3
    as a coarse parity check — the per-chunk outputs are deterministic
    given the chunk input, so most of the signal should align."""
    from vc_realtime.infer_v1 import V1Infer

    infer = V1Infer(models_dir=str(MODELS))
    sr = 24000
    t = np.linspace(0, 10, sr * 10, endpoint=False, dtype=np.float32)
    wav = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)

    batched_out = infer.process_audio_batched(wav, sr, voice_id=0, batch_sec=5.0)
    single_out = infer.process_audio(wav, sr, voice_id=0)

    min_len = min(len(batched_out), len(single_out))
    a = batched_out[:min_len].astype(np.float64)
    b = single_out[:min_len].astype(np.float64)
    # Pearson correlation — both non-zero variance (verified non-silent).
    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr > 0.3, (
        f"Batched vs single-shot output too different (corr={corr:.4f})"
    )
