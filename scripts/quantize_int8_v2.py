#!/usr/bin/env python3
"""scripts/quantize_int8_v2.py — ONNX dynamic INT8 PTQ for the v3 hybrid stack.

Extends P1's ``quantize_int8.py`` (which handled TinyVC's 3 sub-graphs) to the
3 ONNX models exported by P2-1 and P2-2:

  - ``openvoice_ref_encoder.onnx``     (3.1 MB  FP32, by P2-1)
        linear_spec      [B, T, 513]  -> speaker_embedding [B, 256]
  - ``openvoice_residual_flow.onnx``   (33.3 MB FP32, by P2-1)
        content, x_mask, se_src, se_tgt -> content_disentangled
  - ``vocos.onnx``                     (59.8 MB FP32, by P2-2)
        mel [B, 100, T_mel]            -> waveform [B, T_samples]

Method: ``onnxruntime.quantization.quantize_dynamic`` with
``weight_type=QInt8``, ``per_channel=True``, ``reduce_range=False``, and the
4 op types that dominate these graphs: ``MatMul, Gemm, Conv, ConvTranspose``.

Per-channel is preferred because OpenVoice's flow and Vocos' backbone rely
heavily on depthwise / grouped convs whose per-tensor ranges are wide; the
extra scale/zero-point overhead (1 pair per output channel) is dwarfed by
the 4× weight reduction.

Outputs are written next to the FP32 source with ``.int8.onnx`` suffix.

Usage
-----
    python scripts/quantize_int8_v2.py
    python scripts/quantize_int8_v2.py --per-tensor   # fallback if parity fails

Licenses
--------
- This script: project MIT (c) 2024 mmqz.
- onnxruntime.quantization: Apache 2.0 (Microsoft).
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
# Configuration — matches the names used by P2-1 / P2-2 exporters verbatim.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

# Per-model: (fp32 filename, int8 filename, sample inputs dict)
# Input keys MUST match the ONNX input names verified at export time.
# Values are dummy numpy arrays used for parity verification only.
TARGETS: list[tuple[str, str, dict[str, np.ndarray]]] = [
    (
        "openvoice_ref_encoder.onnx",
        "openvoice_ref_encoder.int8.onnx",
        {"linear_spec": np.random.randn(1, 200, 513).astype(np.float32)},
    ),
    (
        "openvoice_residual_flow.onnx",
        "openvoice_residual_flow.int8.onnx",
        {
            "content": np.random.randn(1, 192, 50).astype(np.float32),
            "x_mask": np.ones((1, 1, 50), dtype=np.float32),
            "se_src": np.random.randn(1, 256, 1).astype(np.float32),
            "se_tgt": np.random.randn(1, 256, 1).astype(np.float32),
        },
    ),
    (
        "vocos.onnx",
        "vocos.int8.onnx",
        {"mel": np.random.randn(1, 100, 100).astype(np.float32)},
    ),
]

# Op types whose weights get quantized. These are the four heavy-matmul ops
# that account for >95 % of FLOPs in OpenVoice + Vocos.
OP_TYPES_TO_QUANTIZE = ["MatMul", "Gemm", "Conv", "ConvTranspose"]


# ---------------------------------------------------------------------------
# Quantize one model
# ---------------------------------------------------------------------------
def quantize_one(fp32_path: Path, int8_path: Path, per_channel: bool) -> None:
    """Apply dynamic INT8 PTQ to a single ONNX model and print the size delta."""
    print(f"\n[quantize] {fp32_path.name} -> {int8_path.name}")
    print(f"  per_channel   : {per_channel}")
    print("  weight_type   : QInt8")
    print(f"  op_types      : {OP_TYPES_TO_QUANTIZE}")
    print("  reduce_range  : False")

    # Remove any stale int8 output so a failed run doesn't silently leave an
    # old artifact behind (the parity check would then compare against the
    # previous quantization, not the current per-tensor fallback).
    if int8_path.exists():
        int8_path.unlink()

    ort_q.quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=ort_q.QuantType.QInt8,
        op_types_to_quantize=OP_TYPES_TO_QUANTIZE,
        per_channel=per_channel,
        reduce_range=False,
    )

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    print(
        f"  size          : {fp32_size / 1024:.0f} KB -> "
        f"{int8_size / 1024:.0f} KB "
        f"({int8_size / fp32_size * 100:.1f}% of FP32, "
        f"{(1 - int8_size / fp32_size) * 100:.1f}% reduction)"
    )


# ---------------------------------------------------------------------------
# Parity verification
# ---------------------------------------------------------------------------
def verify_parity(
    fp32_path: Path, int8_path: Path, test_input: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Run the same input through FP32 and INT8 ONNX sessions and compare.

    Returns a dict per output index: {l1, l_inf, fp32_shape, int8_shape}.
    """
    fp32_sess = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    )
    int8_sess = ort.InferenceSession(
        str(int8_path), providers=["CPUExecutionProvider"]
    )

    # Sanity-check input name alignment: INT8 input names MUST match FP32
    # names exactly (quantize_dynamic preserves them). If not, we map by
    # order — but warn loudly because that's a sign the exporter was changed.
    fp32_in = [i.name for i in fp32_sess.get_inputs()]
    int8_in = [i.name for i in int8_sess.get_inputs()]
    if fp32_in != int8_in:
        print(f"  WARN: input name mismatch FP32={fp32_in} INT8={int8_in}")
        # Map by order:
        fp32_feed = {
            name: test_input[k]
            for name, k in zip(fp32_in, test_input.keys())
        }
        int8_feed = {
            name: test_input[k]
            for name, k in zip(int8_in, test_input.keys())
        }
    else:
        fp32_feed = {name: test_input[name] for name in fp32_in if name in test_input}
        int8_feed = fp32_feed  # identical

    fp32_out = fp32_sess.run(None, fp32_feed)
    int8_out = int8_sess.run(None, int8_feed)

    results: dict[str, Any] = {}
    for i, (f, n) in enumerate(zip(fp32_out, int8_out)):
        l1 = float(np.mean(np.abs(f.astype(np.float64) - n.astype(np.float64))))
        l_inf = float(np.max(np.abs(f.astype(np.float64) - n.astype(np.float64))))
        rel = (
            l1 / max(float(np.mean(np.abs(f.astype(np.float64)))), 1e-12)
        )
        print(
            f"  output[{i}]   : shape={n.shape} L1={l1:.6e} "
            f"L_inf={l_inf:.6e} rel_L1={rel:.4e}"
        )
        results[f"output_{i}"] = {
            "l1": l1,
            "l_inf": l_inf,
            "rel_l1": rel,
            "fp32_shape": list(f.shape),
            "int8_shape": list(n.shape),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--per-tensor",
        action="store_true",
        help=(
            "Use per-tensor quantization (faster, smaller file, but lower "
            "accuracy). Fallback if per-channel parity fails."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"directory with FP32 ONNX files (default: {MODELS_DIR})",
    )
    args = parser.parse_args()
    per_channel = not args.per_tensor
    models_dir: Path = args.models_dir

    # Use a fixed RNG seed so the parity numbers are reproducible across runs.
    np.random.seed(0xC0FFEE)

    print("=" * 72)
    print(f"INT8 dynamic PTQ for v3 hybrid stack (per_channel={per_channel})")
    print(f"  models dir: {models_dir}")
    print(f"  onnxruntime: {ort.__version__}")
    print("=" * 72)

    missing: list[str] = []
    for fp32_name, _, _ in TARGETS:
        if not (models_dir / fp32_name).exists():
            missing.append(fp32_name)
    if missing:
        print(f"ERROR: missing FP32 ONNX sources: {missing}")
        print("       run scripts/export_openvoice_onnx.py + export_vocos_onnx.py first")
        return 1

    # ----- Step 1: Quantize all 3 -----
    print("\n--- Step 1: Quantize ---")
    for fp32_name, int8_name, _ in TARGETS:
        quantize_one(
            models_dir / fp32_name, models_dir / int8_name, per_channel
        )

    # ----- Step 2: Smoke-test loading each INT8 ONNX -----
    print("\n--- Step 2: Smoke-test load ---")
    for _, int8_name, _ in TARGETS:
        int8_path = models_dir / int8_name
        try:
            sess = ort.InferenceSession(
                str(int8_path), providers=["CPUExecutionProvider"]
            )
            in_meta = [(i.name, list(i.shape)) for i in sess.get_inputs()]
            out_meta = [(o.name, list(o.shape)) for o in sess.get_outputs()]
            print(f"  {int8_name}: OK")
            print(f"    inputs : {in_meta}")
            print(f"    outputs: {out_meta}")
        except Exception as e:  # noqa: BLE001
            print(f"  {int8_name}: FAIL - {e}")
            return 2

    # ----- Step 3: Parity verification -----
    print("\n--- Step 3: Parity verification (FP32 vs INT8) ---")
    all_results: dict[str, dict[str, Any]] = {}
    worst_l1 = 0.0
    worst_name = ""
    for fp32_name, int8_name, test_input in TARGETS:
        fp32_path = models_dir / fp32_name
        int8_path = models_dir / int8_name
        print(f"\n{fp32_name}:")
        res = verify_parity(fp32_path, int8_path, test_input)
        all_results[fp32_name] = res
        for k, v in res.items():
            if v["l1"] > worst_l1:
                worst_l1 = v["l1"]
                worst_name = f"{fp32_name}/{k}"

    # ----- Step 4: Summary -----
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'model':<32} {'FP32 KB':>10} {'INT8 KB':>10} {'ratio':>8} {'L1':>12}")
    for fp32_name, int8_name, _ in TARGETS:
        fp32_p = models_dir / fp32_name
        int8_p = models_dir / int8_name
        if not (fp32_p.exists() and int8_p.exists()):
            continue
        l1 = all_results[fp32_name]["output_0"]["l1"]
        print(
            f"{fp32_name:<32} "
            f"{fp32_p.stat().st_size / 1024:>10.0f} "
            f"{int8_p.stat().st_size / 1024:>10.0f} "
            f"{int8_p.stat().st_size / fp32_p.stat().st_size:>7.2f} "
            f"{l1:>12.4e}"
        )
    print(f"\nworst L1: {worst_l1:.4e} ({worst_name})")

    # Parity guidance: MatMul/Gemm-heavy models should be < 1e-3; the
    # OpenVoice flow (depthwise WaveNet convs) and Vocos (ConvTranspose
    # ISTFT head) can drift higher. We DO NOT fail the script if parity is
    # above some arbitrary threshold — the test suite asserts a generous
    # 1e-1 upper bound, and we report actual numbers honestly.
    if worst_l1 > 1e-2:
        print(
            f"\nNOTE: worst L1 = {worst_l1:.4e} > 1e-2. Consider re-running with"
            f" --per-tensor for a smaller (less accurate) fallback, OR keep "
            f"per-channel — per-channel is usually MORE accurate, not less."
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
