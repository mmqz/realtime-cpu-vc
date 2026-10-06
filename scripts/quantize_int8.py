#!/usr/bin/env python3
"""
scripts/quantize_int8.py — ONNX dynamic INT8 PTQ for TinyVC's 3 sub-graphs
=============================================================================
Implements P1-2 from docs/checklist.md.

For each sub-graph (encoder, source_net, filter_net):
  - Read FP32 ONNX
  - Apply per-channel dynamic INT8 quantization to Conv, MatMul, Gemm weights
  - Write to <name>.int8.onnx

Expected size reduction:
  encoder.onnx     18 MB -> 9 MB
  source_net.onnx  9 MB  -> 4.5 MB
  filter_net.onnx  9 MB  -> 4.5 MB
  total            36 MB -> 18 MB

Usage:
    python scripts/quantize_int8.py --models-dir models/
    python scripts/quantize_int8.py --models-dir models/ --per-tensor  # faster, less accurate
"""
import argparse
import os
from pathlib import Path
import onnxruntime.quantization as ort_q


def quantize_subgraph(fp32_path: Path, int8_path: Path, per_channel: bool = True):
    """Apply dynamic INT8 PTQ to a single ONNX sub-graph."""
    print(f"  Quantizing {fp32_path.name} -> {int8_path.name}")
    print(f"    per_channel: {per_channel}")
    print(f"    op_types: ['MatMul', 'Gemm', 'Conv']")
    ort_q.quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=ort_q.QuantType.QInt8,
        op_types_to_quantize=['MatMul', 'Gemm', 'Conv'],
        per_channel=per_channel,
        reduce_range=False,
    )
    fp32_size = fp32_path.stat().st_size / 1024
    int8_size = int8_path.stat().st_size / 1024
    print(f"    Size: {fp32_size:.0f} KB -> {int8_size:.0f} KB "
          f"({int8_size / fp32_size * 100:.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models-dir', default='models/', help='directory with *.onnx files')
    p.add_argument('--per-tensor', action='store_true',
                   help='use per-tensor (faster but lower accuracy) instead of per-channel')
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    sub_graphs = ['encoder', 'source_net', 'filter_net']
    per_channel = not args.per_tensor

    print(f"INT8 dynamic quantization on TinyVC sub-graphs in {models_dir}")
    print(f"per_channel: {per_channel}")
    print()

    for name in sub_graphs:
        fp32 = models_dir / f"{name}.onnx"
        int8 = models_dir / f"{name}.int8.onnx"
        if not fp32.exists():
            print(f"  SKIP: {fp32} not found (run tinyvc/export_onnx.py first)")
            continue
        quantize_subgraph(fp32, int8, per_channel=per_channel)
        print()

    # Verify: smoke test loading
    print("Smoke test: loading each INT8 ONNX...")
    import onnxruntime as ort
    for name in sub_graphs:
        int8 = models_dir / f"{name}.int8.onnx"
        if not int8.exists():
            continue
        try:
            s = ort.InferenceSession(str(int8),
                                      providers=['CPUExecutionProvider'])
            inputs = [(i.name, i.shape) for i in s.get_inputs()]
            outputs = [(o.name, o.shape) for o in s.get_outputs()]
            print(f"  {int8.name}: OK")
            print(f"    inputs:  {inputs}")
            print(f"    outputs: {outputs}")
        except Exception as e:
            print(f"  {int8.name}: FAIL - {e}")
            return 1

    print("\nDone. Update configs/default.yaml to point to the .int8.onnx files.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
