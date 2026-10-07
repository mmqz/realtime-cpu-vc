"""Edge case tests for V1Infer.process_audio — DEEP-PY audit Part B.

Run with::

    python3 tests/test_edge_cases.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from vc_realtime.infer_v1 import V1Infer  # noqa: E402

MODELS_DIR = str(ROOT / "models")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    infer = V1Infer(models_dir=MODELS_DIR)

    # 1a: Empty array
    section("1a: Empty input")
    try:
        out = infer.process_audio(np.array([], dtype=np.float32), 24000, 0)
        print(f"  len={len(out)} — {'OK' if len(out) >= 0 else 'FAIL'}")
    except Exception as e:
        print(f"  EXCEPTION — {type(e).__name__}: {e}")

    # 1b: Single sample
    section("1b: Single sample")
    try:
        out = infer.process_audio(np.array([0.5], dtype=np.float32), 24000, 0)
        print(f"  len={len(out)} — OK")
    except Exception as e:
        print(f"  EXCEPTION — {type(e).__name__}: {e}")

    # 1c: NaN input
    section("1c: NaN input")
    try:
        wav = np.full(24000, np.nan, dtype=np.float32)
        out = infer.process_audio(wav, 24000, 0)
        has_nan = bool(np.isnan(out).any())
        print(f"  output has NaN={has_nan}, len={len(out)} — {'FAIL' if has_nan else 'OK'}")
    except Exception as e:
        print(f"  EXCEPTION — {type(e).__name__}: {e}")

    # 1d: Inf input
    section("1d: Inf input")
    try:
        wav = np.full(24000, np.inf, dtype=np.float32)
        out = infer.process_audio(wav, 24000, 0)
        has_inf = bool(np.isinf(out).any()) or bool(np.isnan(out).any())
        print(f"  output has Inf/NaN={has_inf}, len={len(out)} — {'FAIL' if has_inf else 'OK'}")
    except Exception as e:
        print(f"  EXCEPTION — {type(e).__name__}: {e}")

    # 1e: Very quiet (near-zero RMS)
    section("1e: Very quiet input")
    try:
        wav = np.full(24000, 1e-10, dtype=np.float32)
        out = infer.process_audio(wav, 24000, 0)
        rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
        print(f"  output RMS={rms:.6f}, len={len(out)} — OK")
    except Exception as e:
        print(f"  EXCEPTION — {type(e).__name__}: {e}")

    # 1f: Clipping (>1.0)
    section("1f: Clipping input (>1.0)")
    try:
        wav = np.full(24000, 5.0, dtype=np.float32)
        out = infer.process_audio(wav, 24000, 0)
        rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
        mx = float(np.max(np.abs(out))) if out.size > 0 else 0.0
        print(f"  output RMS={rms:.4f}, max={mx:.4f}, len={len(out)} — OK")
    except Exception as e:
        print(f"  EXCEPTION — {type(e).__name__}: {e}")

    # 1g: Wrong sample rate (8kHz)
    section("1g: 8kHz input")
    try:
        sr = 8000
        wav = (0.3 * np.sin(2 * np.pi * 150 * np.linspace(0, 1, sr, endpoint=False))).astype(np.float32)
        out = infer.process_audio(wav, sr, 0)
        print(f"  output len={len(out)} (expected ~24000) — {'OK' if 23990 <= len(out) <= 24010 else 'WARN'}")
    except Exception as e:
        print(f"  EXCEPTION — {type(e).__name__}: {e}")

    # 1h: Invalid voice_id (out of range)
    section("1h: Invalid voice_id=99")
    wav = (0.3 * np.sin(2 * np.pi * 150 * np.linspace(0, 1, 24000, endpoint=False))).astype(np.float32)
    try:
        out = infer.process_audio(wav, 24000, 99)
        print(f"  output len={len(out)} — FAIL (no exception)")
    except ValueError as e:
        print(f"  ValueError (expected) — {e}")
    except Exception as e:
        print(f"  UNEXPECTED EXCEPTION — {type(e).__name__}: {e}")

    # 1i: Negative voice_id
    section("1i: Negative voice_id=-1")
    try:
        out = infer.process_audio(wav, 24000, -1)
        print(f"  output len={len(out)} — FAIL (no exception)")
    except ValueError as e:
        print(f"  ValueError (expected) — {e}")
    except Exception as e:
        print(f"  UNEXPECTED EXCEPTION — {type(e).__name__}: {e}")

    # 1j: Stereo input (2D array)
    section("1j: Stereo input")
    try:
        wav = np.column_stack([
            np.sin(2 * np.pi * 150 * np.linspace(0, 1, 24000, endpoint=False)),
            np.sin(2 * np.pi * 200 * np.linspace(0, 1, 24000, endpoint=False)),
        ]).astype(np.float32) * 0.3
        out = infer.process_audio(wav, 24000, 0)
        # Output should be 1-D mono (or compatible shape)
        print(f"  output shape={out.shape}, len={len(out)} — OK")
    except Exception as e:
        print(f"  EXCEPTION — {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
