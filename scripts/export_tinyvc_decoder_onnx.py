#!/usr/bin/env python3
"""Export TinyVC Decoder sub-graphs (SourceNet + FilterNet) as ONNX.

Architecture (verified against ``repos/tinyvc/module/tinyvc/decoder.py``)
------------------------------------------------------------------------
``Decoder`` composes two sub-networks plus a non-trainable DSP block:

  * ``SourceNet`` (~1.7 M params):
      - Input:  content  [B, 768, T_frames]    (content_channels=768)
                f0      [B, 1,   T_frames]    (raw pitch, Hz)
                energy  [B, 1,   T_frames * frame_size]  (per-sample energy)
      - Internal: 1×1 convs fuse content + max-pooled energy + log(f0),
        then 3 ConvNeXt layers (128 channels, k=7).
      - Output: amplitudes [B, num_harmonics+1=15, T_frames]  (ELU+1, >0)
                kernel     [B, fft_bin=961,        T_frames]  (ELU+1, >0)
      - The kernel is the per-frame frequency-domain noise filter that
        ``oscillate_noise`` multiplies with random-phase noise + ``istft``
        to synthesise the aperiodic component (NOT inside this sub-graph —
        the DSP block is implemented in Rust in P2.0-2 / P2.1).

  * ``FilterNet`` (~9 M params):
      - Input:  content [B, 768, T_frames]
                f0      [B, 1,   T_frames]
                energy  [B, 1,   T_frames * frame_size]
                source  [B, num_harmonics+2=16, T_frames * frame_size]
                        (harmonics [15 ch] + noise [1 ch], produced by
                         the DSP block from SourceNet's outputs)
      - 5-stage U-Net: Downsample (factor 5,4,4,3,2) + Upsample (factor
        2,3,4,4,5) with FiLM conditioning on the downsampled source.
        Channels [384, 192, 96, 48, 24].
      - Output: waveform [B, 1, T_frames * frame_size=480]
                 (final 7-tap Conv1d, padding=3, replicate mode).

This faithfully matches upstream ``tinyvc/export_onnx.py`` I/O contract:

  source_net.onnx:
    inputs : content, f0, energy        (all dynamic batch + length)
    outputs: amplitudes, kernel

  filter_net.onnx:
    inputs : content, f0, energy, source  (all dynamic batch + length)
    outputs: waveform

INT8 PTQ
--------
Per-channel dynamic INT8 quantization on MatMul, Gemm, Conv weights — same
recipe as ``export_tinyvc_encoder_onnx.py``. The two sub-graphs are pure
conv stacks (no RNN / attention), so dynamic PTQ lands within a few % of
FP32 L2 norm without calibration data.

Usage
-----
  python3 scripts/export_tinyvc_decoder_onnx.py
  python3 scripts/export_tinyvc_decoder_onnx.py --no-quantize
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# Make the TinyVC package importable (MIT-licensed TinyVC upstream).
TINYVC_ROOT = Path("../repos/tinyvc")
if str(TINYVC_ROOT) not in sys.path:
    sys.path.insert(0, str(TINYVC_ROOT))

from module.tinyvc import Decoder  # noqa: E402

# Suppress torch.onnx legacy-exporter deprecation warnings (we intentionally
# keep the legacy exporter — same approach as ``export_vocos_onnx.py``).
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx.*")

# ---------------------------------------------------------------------------
# Constants (verified from ``repos/tinyvc/module/tinyvc/decoder.py``)
# ---------------------------------------------------------------------------
N_FFT = 1920                  # decoder.n_fft default
FRAME_SIZE = 480              # decoder.frame_size default (= hop_size)
FFT_BIN = N_FFT // 2 + 1      # = 961 — kernel output channels
NUM_HARMONICS = 14            # decoder.num_harmonics default
CONTENT_CHANNELS = 768       # ssl_dim from encoder; source_net.content_channels
SOURCE_CHANNELS = NUM_HARMONICS + 2   # harmonics(15) + noise(1) = 16
DEFAULT_T_FRAMES = 100       # 2 s @ 50 Hz frame rate — ample for tracing


def load_decoder(decoder_pt: Path) -> Decoder:
    """Instantiate the TinyVC ``Decoder`` and load PyTorch weights from disk."""
    if not decoder_pt.exists():
        raise FileNotFoundError(
            f"decoder.pt not found at {decoder_pt}. Run scripts/build_voices_index.py "
            "or download the decoder checkpoint first."
        )
    dec = Decoder()
    state_dict = torch.load(decoder_pt, map_location="cpu")
    dec.load_state_dict(state_dict)
    dec.eval()
    return dec


def export_source_net_onnx(
    decoder: Decoder,
    out_path: Path,
    t_frames: int = DEFAULT_T_FRAMES,
) -> None:
    """Trace + export SourceNet to ONNX (opset 17, legacy exporter).

    Input shapes (matches upstream ``tinyvc/export_onnx.py:45-47``):
      content : [1, 768, T_frames]
      f0      : [1, 1,   T_frames]
      energy  : [1, 1,   T_frames * frame_size]   (per-sample energy;
                                                   max-pooled to T_frames
                                                   inside SourceNet.forward)
    """
    sn = decoder.source_net
    assert sn.frame_size == FRAME_SIZE
    assert sn.num_harmonics == NUM_HARMONICS
    assert sn.content_channels == CONTENT_CHANNELS

    content = torch.randn(1, CONTENT_CHANNELS, t_frames, dtype=torch.float32)
    f0 = torch.randn(1, 1, t_frames, dtype=torch.float32).clamp(min=0.0) + 1e-3
    energy = torch.randn(1, 1, t_frames * FRAME_SIZE, dtype=torch.float32).abs()

    torch.onnx.export(
        sn,
        (content, f0, energy),
        str(out_path),
        opset_version=17,
        input_names=["content", "f0", "energy"],
        output_names=["amplitudes", "kernel"],
        dynamic_axes={
            "content":    {0: "batch_size", 2: "length"},
            "f0":         {0: "batch_size", 2: "length"},
            "energy":     {0: "batch_size", 2: "length"},
            "amplitudes": {0: "batch_size", 2: "length"},
            "kernel":     {0: "batch_size", 2: "length"},
        },
        do_constant_folding=True,
        dynamo=False,
    )


def export_filter_net_onnx(
    decoder: Decoder,
    out_path: Path,
    t_frames: int = DEFAULT_T_FRAMES,
) -> None:
    """Trace + export FilterNet to ONNX (opset 17, legacy exporter).

    Input shapes (matches upstream ``tinyvc/export_onnx.py:62-64``):
      content : [1, 768, T_frames]
      f0      : [1, 1,   T_frames]
      energy  : [1, 1,   T_frames * frame_size]
      source  : [1, source_channels=16, T_frames * frame_size]
                 (harmonics [15 ch] + noise [1 ch] — produced by the DSP
                 block in P2.0-2 / P2.1 from SourceNet's amps + kernel)
    """
    fn = decoder.filter_net

    content = torch.randn(1, CONTENT_CHANNELS, t_frames, dtype=torch.float32)
    f0 = torch.randn(1, 1, t_frames, dtype=torch.float32).clamp(min=0.0) + 1e-3
    energy = torch.randn(1, 1, t_frames * FRAME_SIZE, dtype=torch.float32).abs()
    source = torch.randn(1, SOURCE_CHANNELS, t_frames * FRAME_SIZE,
                         dtype=torch.float32)

    torch.onnx.export(
        fn,
        (content, f0, energy, source),
        str(out_path),
        opset_version=17,
        input_names=["content", "f0", "energy", "source"],
        output_names=["waveform"],
        dynamic_axes={
            "content":  {0: "batch_size", 2: "length"},
            "f0":       {0: "batch_size", 2: "length"},
            "energy":   {0: "batch_size", 2: "length"},
            "source":   {0: "batch_size", 2: "length"},
            "waveform": {0: "batch_size", 2: "length"},
        },
        do_constant_folding=True,
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

    # Build a feed dict from declared input shapes (use small T for speed).
    # Only axis 2 (length) is dynamic; axis 0 (batch_size) is fixed to 1 and
    # axis 1 (channels) is the static channel dim. The length axis on
    # energy/source inputs must be T * frame_size (per-sample rate); the
    # length axis on content/f0 inputs is T (per-frame rate).
    t = 20
    feed = {}
    for i in inputs:
        name = i.name
        shape = list(i.shape)
        # Replace only the LAST dim (axis 2 = length) with a concrete value.
        # Batch (axis 0) and channels (axis 1) stay at their static dims
        # (batch_size symbol → 1, channels = literal int from the graph).
        shape[0] = 1
        if name in ("energy", "source"):
            shape[2] = t * FRAME_SIZE
        else:
            shape[2] = t
        feed[name] = np.zeros(shape, dtype=np.float32)

    outs = sess.run(None, feed)
    for name, arr in zip([o.name for o in outputs], outs):
        print(f"    {name}: shape={arr.shape}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}, "
              f"mean={arr.mean():.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--decoder-pt",
                   default="./models/decoder.pt",
                   help="Path to TinyVC decoder PyTorch weights")
    p.add_argument("--models-dir",
                   default="./models",
                   help="Output directory for *.onnx")
    p.add_argument("--t-frames", type=int, default=DEFAULT_T_FRAMES,
                   help=f"Number of frames in the tracing input "
                        f"(default {DEFAULT_T_FRAMES})")
    p.add_argument("--quantize", action="store_true", default=True,
                   help="Also produce *.int8.onnx via dynamic INT8 PTQ "
                        "(default: True)")
    p.add_argument("--no-quantize", dest="quantize", action="store_false",
                   help="Skip INT8 PTQ — only produce *.onnx")
    args = p.parse_args()

    decoder_pt = Path(args.decoder_pt)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TinyVC Decoder from PyTorch...")
    dec = load_decoder(decoder_pt)
    print(f"  source_net.frame_size={dec.source_net.frame_size}, "
          f"num_harmonics={dec.source_net.num_harmonics}, "
          f"content_channels={dec.source_net.content_channels}, "
          f"n_fft={dec.source_net.n_fft}")
    print(f"  filter_net.input_layer expects content_channels="
          f"{CONTENT_CHANNELS}, source_channels={SOURCE_CHANNELS}")

    print("\nExporting source_net.onnx (opset 17, dynamic batch+length)...")
    out_src_fp32 = models_dir / "source_net.onnx"
    export_source_net_onnx(dec, out_src_fp32, t_frames=args.t_frames)
    print(f"  wrote {out_src_fp32} "
          f"({out_src_fp32.stat().st_size / 1024:.0f} KB)")

    print("\nExporting filter_net.onnx (opset 17, dynamic batch+length)...")
    out_flt_fp32 = models_dir / "filter_net.onnx"
    export_filter_net_onnx(dec, out_flt_fp32, t_frames=args.t_frames)
    print(f"  wrote {out_flt_fp32} "
          f"({out_flt_fp32.stat().st_size / 1024:.0f} KB)")

    if args.quantize:
        print("\nApplying per-channel dynamic INT8 PTQ "
              "(MatMul, Gemm, Conv -> QInt8)...")
        for name, fp32 in [("source_net", out_src_fp32),
                           ("filter_net", out_flt_fp32)]:
            int8 = models_dir / f"{name}.int8.onnx"
            quantize_int8(fp32, int8)
            print(f"  {name}: {fp32.stat().st_size / 1024:.0f} KB -> "
                  f"{int8.stat().st_size / 1024:.0f} KB "
                  f"({int8.stat().st_size / max(fp32.stat().st_size, 1) * 100:.0f}%)")

    print("\nSmoke test (ORT load + forward pass with zeros):")
    smoke_test(out_src_fp32)
    print()
    smoke_test(out_flt_fp32)
    if args.quantize:
        print()
        smoke_test(models_dir / "source_net.int8.onnx")
        print()
        smoke_test(models_dir / "filter_net.int8.onnx")

    print("\nDone. All 7 ONNX models for the v2.0 Rust path are now exported:")
    print("  encoder.onnx / encoder.int8.onnx  (P2.1-1)")
    print("  source_net.onnx / source_net.int8.onnx  (P2.1-F, this task)")
    print("  filter_net.onnx / filter_net.int8.onnx  (P2.1-F, this task)")
    print("  + previously exported: openvoice_ref_encoder, "
          "spark_speaker_encoder, openvoice_residual_flow, vocos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
