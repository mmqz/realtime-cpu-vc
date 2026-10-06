#!/usr/bin/env python3
"""Export TinyVC Encoder (SSLFeatureEstimator + PitchEstimator) as ONNX.

Architecture (verified against ``repos/tinyvc/module/tinyvc/encoder.py``)
------------------------------------------------------------------------
``Encoder`` (``module.tinyvc.Encoder``) composes two sub-estimators:

  * ``SSLFeatureEstimator`` (~4.7 M params):
      - 6-layer ConvNeXt-v2 (384 internal channels, dilations=[1,3,9,1,1,1],
        depthwise k=7, GRN).
      - input_layer: Conv1d(fft_bin -> 384, k=1)  where fft_bin = n_fft//2+1
      - output_layer: Conv1d(384 -> 768, k=1)     (ssl_dim=768)
  * ``PitchEstimator`` (~460 K params):
      - 4-layer ConvNeXt (128 internal channels, default dilations).
      - output_layer: Conv1d(128 -> 512, k=1)     (num_classes=512)
      - classes_per_octave=48, min_frequency=20 Hz
      - The ONNX graph emits the *raw logits* (NOT decoded f0); downstream
        decoding uses ``topk(logits, k=4)`` + softmax-weighted ``id2freq``.

The full ``Encoder.forward(spec)`` returns ``(ssl, f0_logits)`` — 2 outputs.
This faithfully matches upstream ``tinyvc/export_onnx.py``.

I/O contract
------------
  Input  name  : ``spectrogram``
          shape: ``[B, fft_bin=961, T_frames]``  (raw STFT magnitude spectrum,
                                                 n_fft=1920, hop=480 @ 24 kHz
                                                 → 50 Hz frame rate)
          dtype: float32
  Output name 0: ``content``    shape ``[B, 768, T_frames]``
  Output name 1: ``f0``        shape ``[B, 512, T_frames]``  (pitch logits)
  Dynamic axes : batch (axis 0) + length (axis 2) for all tensors.
  Opset        : 17  (matches upstream ``tinyvc/export_onnx.py`` default).

Why faithful to upstream (not 3-output ``content+f0+energy``):
  Some downstream Python wrappers (``src/vc_realtime/encoder.py``) describe a
  speculative 3-output ``(content, f0, energy)`` API. That is **incorrect**
  w.r.t. the actual TinyVC source — energy is produced by the *decoder*'s
  ``source_net`` (it takes f0 + content as inputs and outputs amplitudes +
  kernel). The faithful export is the 2-output graph upstream ships. This
  script matches upstream exactly.

Usage
-----
  /home/z/.venv/bin/python scripts/export_tinyvc_encoder_onnx.py
  /home/z/.venv/bin/python scripts/export_tinyvc_encoder_onnx.py --quantize
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# Make the TinyVC package importable. We import only the encoder module
# (MIT-licensed TinyVC) and never touch the CC-BY-NC F5-TTS code.
TINYVC_ROOT = Path("/home/z/my-project/repos/tinyvc")
if str(TINYVC_ROOT) not in sys.path:
    sys.path.insert(0, str(TINYVC_ROOT))

from module.tinyvc import Encoder  # noqa: E402

# Suppress torch.onnx legacy-exporter deprecation warnings (we intentionally
# keep the legacy exporter — same approach as ``export_vocos_onnx.py``).
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx.*")

# ---------------------------------------------------------------------------
# Constants (verified from ``repos/tinyvc/module/tinyvc/encoder.py``)
# ---------------------------------------------------------------------------
N_FFT = 1920          # TinyVC Encoder default (encoder.py:101)
HOP_SIZE = 480         # 24 kHz / 480 = 50 Hz frame rate (encoder.py:104)
FFT_BIN = N_FFT // 2 + 1  # = 961 — input frequency bins
SSL_DIM = 768          # content feature dim (encoder.py:81)
NUM_PITCH_CLASSES = 512  # pitch classifier bins (encoder.py:17)
DEFAULT_T_FRAMES = 100  # ~2 s of audio at 50 Hz — ample for tracing


def load_encoder(encoder_pt: Path) -> Encoder:
    """Instantiate the TinyVC ``Encoder`` and load PyTorch weights from disk."""
    if not encoder_pt.exists():
        raise FileNotFoundError(
            f"encoder.pt not found at {encoder_pt}. Run scripts/build_voices_index.py "
            "or download the encoder checkpoint first."
        )
    enc = Encoder(n_fft=N_FFT, hop_size=HOP_SIZE)
    state_dict = torch.load(encoder_pt, map_location="cpu")
    enc.load_state_dict(state_dict)
    enc.eval()
    return enc


def export_encoder_onnx(encoder: Encoder, out_path: Path,
                        t_frames: int = DEFAULT_T_FRAMES) -> None:
    """Trace + export the Encoder to ONNX (opset 17, legacy exporter)."""
    # Input: raw STFT magnitude spectrum, [B=1, fft_bin=961, T_frames].
    dummy = torch.randn(1, FFT_BIN, t_frames, dtype=torch.float32)

    torch.onnx.export(
        encoder,
        dummy,
        str(out_path),
        opset_version=17,
        input_names=["spectrogram"],
        output_names=["content", "f0"],
        dynamic_axes={
            "spectrogram": {0: "batch_size", 2: "length"},
            "content":     {0: "batch_size", 2: "length"},
            "f0":          {0: "batch_size", 2: "length"},
        },
        do_constant_folding=True,
        # Use the legacy TorchScript exporter — same approach as
        # ``export_vocos_onnx.py``. Avoids the torch 2.14 dynamo exporter's
        # hard dependency on onnxscript (which is not installed).
        dynamo=False,
    )


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """Apply per-channel dynamic INT8 PTQ to Conv/Gemm/MatMul weights."""
    import onnxruntime.quantization as ort_q

    ort_q.quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=ort_q.QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        per_channel=True,
        reduce_range=False,
    )


def smoke_test(onnx_path: Path) -> None:
    """Load via ORT and run a forward pass with zeros to verify the graph."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"  {onnx_path.name}: "
          f"{onnx_path.stat().st_size / 1024:.0f} KB")
    print(f"    inputs : {[(i.name, i.shape) for i in inputs]}")
    print(f"    outputs: {[(o.name, o.shape) for o in outputs]}")

    # Forward pass with zeros — verifies the full ORT load → run pipeline.
    in_name = inputs[0].name
    t = 50
    spec = np.zeros((1, FFT_BIN, t), dtype=np.float32)
    outs = sess.run(None, {in_name: spec})
    print(f"    forward({t} frames) -> "
          f"{[(o.name, str(o.shape)) for o in outputs]}")
    for name, arr in zip([o.name for o in outputs], outs):
        print(f"      {name}: shape={arr.shape}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}, "
              f"mean={arr.mean():.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--encoder-pt",
                   default="/home/z/my-project/prototype/models/encoder.pt",
                   help="Path to TinyVC encoder PyTorch weights")
    p.add_argument("--models-dir",
                   default="/home/z/my-project/prototype/models",
                   help="Output directory for *.onnx")
    p.add_argument("--t-frames", type=int, default=DEFAULT_T_FRAMES,
                   help=f"Number of frames in the tracing input "
                        f"(default {DEFAULT_T_FRAMES})")
    p.add_argument("--quantize", action="store_true", default=True,
                   help="Also produce encoder.int8.onnx via dynamic INT8 PTQ "
                        "(default: True)")
    p.add_argument("--no-quantize", dest="quantize", action="store_false",
                   help="Skip INT8 PTQ — only produce encoder.onnx")
    args = p.parse_args()

    encoder_pt = Path(args.encoder_pt)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TinyVC Encoder from PyTorch...")
    enc = load_encoder(encoder_pt)
    print(f"  n_fft={enc.n_fft}, hop_size={enc.hop_size}, "
          f"fft_bin={enc.n_fft // 2 + 1}")

    print("\nExporting encoder.onnx (opset 17, dynamic batch+length)...")
    out_fp32 = models_dir / "encoder.onnx"
    export_encoder_onnx(enc, out_fp32, t_frames=args.t_frames)
    print(f"  wrote {out_fp32} ({out_fp32.stat().st_size / 1024:.0f} KB)")

    if args.quantize:
        print("\nApplying per-channel dynamic INT8 PTQ "
              "(MatMul, Gemm, Conv -> QInt8)...")
        out_int8 = models_dir / "encoder.int8.onnx"
        quantize_int8(out_fp32, out_int8)
        print(f"  wrote {out_int8} ({out_int8.stat().st_size / 1024:.0f} KB)")

    print("\nSmoke test (ORT load + forward pass with zeros):")
    smoke_test(out_fp32)
    if args.quantize:
        print()
        smoke_test(models_dir / "encoder.int8.onnx")

    print("\nDone. Rust vc-ort can now load encoder.int8.onnx "
          "(5/5 v3-hybrid ONNX models).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
