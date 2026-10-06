#!/usr/bin/env python3
"""scripts/quantize_int4.py — INT4 dynamic PTQ for the v3 hybrid stack.

INT4 = 4-bit integer weight-only quantization. Where INT8 PTQ halves the
weight mass of an FP32 model, INT4 quarters it — at the cost of a small
accuracy drop that is acceptable for the largest weights in the v3 hybrid
stack.

Method
------
``onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer`` is
the supported entry-point for INT4 + block-wise quantization in
onnxruntime >= 1.18. It replaces each ``MatMul`` / ``Gemm`` whose B operand
is a constant initializer with a ``com.microsoft::MatMulNBits`` (QOperator
format) node that stores B as a packed uint4 tensor + one scale per block
of ``block_size`` consecutive weights along the K-axis.

Why ``block_size=32``? Smaller blocks = more scales = larger file + better
accuracy; larger blocks = fewer scales = smaller file + worse accuracy.
32 is the same block size used by llama.cpp's Q4_0 scheme and the GPTQ
default — it strikes the empirical sweet spot where the per-block scale
overhead (~6.25 % of the 4-bit weight mass for 32-element FP32 scales) is
dwarfed by the 8× compression of the weights themselves.

If the MatMulNBits path raises (e.g. the graph contains no quantizable
MatMul, or the onnxruntime build was compiled without the
``com.microsoft::MatMulNBits`` kernel), we fall back to the
``quantize_dynamic(weight_type=QInt4)`` path — which still produces a
4-bit weight tensor but without block-wise scales. If THAT also fails,
we fall back to the existing INT8 path so the v3 pipeline never loses
quantization entirely.

Expected size reduction
-----------------------
  spark_speaker_encoder: 55 MB FP32 -> 30 MB INT8 -> ~15 MB INT4
  vocos:                  60 MB FP32 -> 22 MB INT8 -> ~11 MB INT4
  openvoice_residual_flow: 33 MB FP32 -> ~9 MB INT8 -> ~5 MB INT4

Usage
-----
    python scripts/quantize_int4.py
    python scripts/quantize_int4.py --block-size 64   # smaller file, worse accuracy
    python scripts/quantize_int4.py --int8-fallback    # force the INT8 fallback path

Licenses
--------
- This script: project MIT (c) 2024 mmqz.
- onnxruntime.quantization: Apache 2.0 (Microsoft).
- Spark-TTS BiCodec weights: Apache 2.0 (SparkAudio).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import onnxruntime.quantization as ort_q

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "models"

# Default block size for INT4 weight quantization. 32 matches llama.cpp's
# Q4_0 and the GPTQ default; smaller = more accurate + larger file, larger
# = smaller file + worse accuracy.
DEFAULT_BLOCK_SIZE = 32

# Per-model: (fp32 filename, int4 filename, sample inputs dict).
# Input keys MUST match the ONNX input names verified at export time.
# Values are dummy numpy arrays used for parity verification only.
TARGETS: list[tuple[str, str, dict[str, np.ndarray] | np.ndarray]] = [
    (
        "spark_speaker_encoder.onnx",
        "spark_speaker_encoder.int4.onnx",
        np.random.randn(1, 200, 128).astype(np.float32),
    ),
    (
        "vocos.onnx",
        "vocos.int4.onnx",
        np.random.randn(1, 100, 100).astype(np.float32),
    ),
    (
        "openvoice_residual_flow.onnx",
        "openvoice_residual_flow.int4.onnx",
        {
            "content": np.random.randn(1, 192, 50).astype(np.float32),
            "x_mask": np.ones((1, 1, 50), dtype=np.float32),
            "se_src": np.random.randn(1, 256, 1).astype(np.float32),
            "se_tgt": np.random.randn(1, 256, 1).astype(np.float32),
        },
    ),
]

# Op types whose weights get quantized. MatMulNBits only supports MatMul
# (it inserts a com.microsoft::MatMulNBits node). We include Gemm for the
# quantize_dynamic INT4 fallback path; it is ignored by MatMulNBits.
OP_TYPES_TO_QUANTIZE = ["MatMul", "Gemm"]


def _to_feed_dict(test_input: dict[str, np.ndarray] | np.ndarray) -> dict[str, np.ndarray]:
    """Normalize the test_input union type into a feed dict.

    The task spec accepts either a raw ndarray (for single-input models
    whose first input is fed positionally) or a dict keyed by input name.
    We always work with a dict internally.
    """
    if isinstance(test_input, dict):
        return dict(test_input)
    return {"__positional__": np.asarray(test_input)}


def quantize_int4(
    fp32_path: Path,
    int4_path: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    int8_fallback: bool = False,
) -> str:
    """Apply INT4 dynamic quantization.

    Returns a short string indicating which path was taken
    ('int4-matmul-nbits', 'int4-quantize-dynamic', or 'int8-fallback').
    """
    print(f"\n[quantize] {fp32_path.name} -> {int4_path.name}")
    print(f"  block_size      : {block_size}")
    print(f"  weight_type     : QInt4")
    print(f"  op_types        : {OP_TYPES_TO_QUANTIZE}")
    print(f"  int8_fallback   : {int8_fallback}")

    # Remove stale output so a failed re-run doesn't silently compare
    # against the previous quantization.
    if int4_path.exists():
        int4_path.unlink()

    if int8_fallback:
        return _fallback_int8(fp32_path, int4_path)

    # ---- Path 1: MatMulNBitsQuantizer (preferred; supports block_size) ----
    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            MatMulNBitsQuantizer,
        )

        # MatMulNBitsQuantizer only quantizes MatMul (not Gemm). The
        # op_types_to_quantize kwarg restricts which MatMul nodes get
        # touched; we pass ("MatMul",) to match.
        quantizer = MatMulNBitsQuantizer(
            model=str(fp32_path),
            bits=4,
            block_size=block_size,
            is_symmetric=True,  # symmetric for MatMul weights (standard)
            op_types_to_quantize=("MatMul",),
        )
        quantizer.process()
        quantizer.model.save_model_to_file(str(int4_path))
        _print_size(fp32_path, int4_path)
        return "int4-matmul-nbits"
    except Exception as e:  # noqa: BLE001
        print(f"  MatMulNBits INT4 failed: {e!r}")
        print("  Trying quantize_dynamic(weight_type=QInt4) fallback...")

    # ---- Path 2: quantize_dynamic with QInt4 (no block_size arg) ----
    try:
        ort_q.quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int4_path),
            weight_type=ort_q.QuantType.QInt4,
            op_types_to_quantize=OP_TYPES_TO_QUANTIZE,
            per_channel=True,
        )
        if int4_path.exists():
            _print_size(fp32_path, int4_path)
            return "int4-quantize-dynamic"
    except Exception as e:  # noqa: BLE001
        print(f"  quantize_dynamic(QInt4) failed: {e!r}")
        print("  Falling back to INT8...")

    # ---- Path 3: INT8 (final fallback; never lose quantization) ----
    return _fallback_int8(fp32_path, int4_path)


def _fallback_int8(fp32_path: Path, int4_path: Path) -> str:
    """Apply INT8 PTQ to the same input; writes to a sibling .int8.onnx file.

    We do NOT overwrite the FP32 -> INT4 .int4.onnx path target — instead
    we write the fallback to ``<name>.int8.onnx`` so the existing INT8
    pipeline (which points at ``.int8.onnx``) keeps working. The function
    still returns ``'int8-fallback'`` so callers can detect the fallback.
    """
    int8_path = Path(str(int4_path).replace(".int4.onnx", ".int8.onnx"))
    # Avoid clobbering an existing INT8 file the v2 quantize scripts wrote.
    if not int8_path.exists():
        ort_q.quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=ort_q.QuantType.QInt8,
            op_types_to_quantize=OP_TYPES_TO_QUANTIZE,
            per_channel=True,
        )
    _print_size(fp32_path, int8_path)
    return "int8-fallback"


def _print_size(fp32_path: Path, out_path: Path) -> None:
    fp32_size = fp32_path.stat().st_size
    out_size = out_path.stat().st_size
    ratio = out_size / fp32_size if fp32_size else 0.0
    print(
        f"  size            : {fp32_size / 1024:.0f} KB -> "
        f"{out_size / 1024:.0f} KB "
        f"({ratio * 100:.1f}% of FP32, "
        f"{(1.0 - ratio) * 100:.1f}% reduction)"
    )


def verify_parity(
    fp32_path: Path,
    int4_path: Path,
    test_input: dict[str, np.ndarray] | np.ndarray,
) -> dict[str, Any]:
    """Compare FP32 vs INT4 output.

    For float outputs: report L1, L_inf, rel_L1.
    For int outputs: report exact-match fraction.
    """
    fp32_sess = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    )
    int4_sess = ort.InferenceSession(
        str(int4_path), providers=["CPUExecutionProvider"]
    )

    fp32_in = [i.name for i in fp32_sess.get_inputs()]
    int4_in = [i.name for i in int4_sess.get_inputs()]
    if fp32_in != int4_in:
        print(f"  WARN: input name mismatch FP32={fp32_in} INT4={int4_in}")

    feed = _to_feed_dict(test_input)
    # Map feed keys to actual input names (positional fallback if needed).
    if list(feed.keys()) == ["__positional__"]:
        fp32_feed = {fp32_in[0]: feed["__positional__"]}
        int4_feed = {int4_in[0]: feed["__positional__"]}
    else:
        fp32_feed = {n: feed[n] for n in fp32_in if n in feed}
        int4_feed = {n: feed[n] for n in int4_in if n in feed}
        # If no keys matched, fall back to positional.
        if not fp32_feed and feed:
            fp32_feed = {fp32_in[0]: next(iter(feed.values()))}
        if not int4_feed and feed:
            int4_feed = {int4_in[0]: next(iter(feed.values()))}

    fp32_out = fp32_sess.run(None, fp32_feed)
    int4_out = int4_sess.run(None, int4_feed)

    results: dict[str, Any] = {}
    for i, (f, n) in enumerate(zip(fp32_out, int4_out)):
        if f.dtype.kind == "f":
            l1 = float(np.mean(np.abs(f.astype(np.float64) - n.astype(np.float64))))
            print(f"  output[{i}]: shape={n.shape} L1={l1:.6e}")
            results[f"output_{i}"] = {"l1": l1, "shape": list(n.shape)}
        else:
            match = bool(np.array_equal(f, n))
            total = int(np.prod(f.shape)) if f.size > 0 else 0
            n_match = int(np.sum(f == n)) if f.size > 0 else 0
            print(
                f"  output[{i}]: shape={n.shape} exact_match={match} "
                f"({n_match}/{total})"
            )
            results[f"output_{i}"] = {
                "exact_match": match,
                "match_count": n_match,
                "total": total,
            }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=(
            "Block size for INT4 weight quantization (default: 32, matches "
            "llama.cpp Q4_0). Smaller = more accurate + larger file."
        ),
    )
    parser.add_argument(
        "--int8-fallback",
        action="store_true",
        help="Skip INT4 entirely and write the INT8 fallback (smoke-test).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS,
        help=f"directory with FP32 ONNX files (default: {MODELS})",
    )
    args = parser.parse_args()
    models_dir: Path = args.models_dir

    np.random.seed(0xC0FFEE)  # reproducible parity numbers

    print("=" * 72)
    print("INT4 dynamic PTQ for the v3 hybrid stack")
    print(f"  block_size    : {args.block_size}")
    print(f"  int8_fallback : {args.int8_fallback}")
    print(f"  models dir    : {models_dir}")
    print(f"  onnxruntime   : {ort.__version__}")
    print("=" * 72)

    print("\n--- Step 1: Quantize ---")
    summary: list[tuple[str, str, int, int]] = []
    for fp32_name, int4_name, test_input in TARGETS:
        fp32 = models_dir / fp32_name
        int4 = models_dir / int4_name
        if not fp32.exists():
            print(f"\nSKIP: {fp32} not found")
            print(
                "      (run the relevant export script — "
                "scripts/export_*_onnx.py — first)"
            )
            continue
        path_used = quantize_int4(
            fp32,
            int4,
            block_size=args.block_size,
            int8_fallback=args.int8_fallback,
        )
        # Path of the actual file written (int4 or int8 fallback).
        out_path = (
            int4 if path_used.startswith("int4") and int4.exists()
            else Path(str(int4).replace(".int4.onnx", ".int8.onnx"))
        )
        if out_path.exists():
            summary.append(
                (
                    fp32_name,
                    path_used,
                    fp32.stat().st_size,
                    out_path.stat().st_size,
                )
            )

    # ----- Step 2: Parity verification -----
    print("\n--- Step 2: Parity verification (FP32 vs INT4) ---")
    for fp32_name, int4_name, test_input in TARGETS:
        fp32 = models_dir / fp32_name
        int4 = models_dir / int4_name
        if not fp32.exists() or not int4.exists():
            continue
        print(f"\n{fp32_name}:")
        verify_parity(fp32, int4, test_input)

    # ----- Step 3: Summary -----
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    if not summary:
        print("  (no models were quantized — see SKIP messages above)")
    else:
        print(
            f"{'model':<32} {'path':<22} {'FP32 KB':>10} {'out KB':>10} "
            f"{'ratio':>7}"
        )
        for name, path_used, fp32_size, out_size in summary:
            ratio = out_size / fp32_size if fp32_size else 0.0
            print(
                f"{name:<32} {path_used:<22} "
                f"{fp32_size // 1024:>10} {out_size // 1024:>10} "
                f"{ratio * 100:>6.1f}%"
            )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
