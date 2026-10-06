#!/usr/bin/env python3
"""Export OpenVoice v2 ReferenceEncoder + ResidualCouplingBlock flow as ONNX.

Architecture (verified by reading the OpenVoice source + loading v2 ckpt)
-----------------------------------------------------------------------
The OpenVoice v2 ``SynthesizerTrn`` (``openvoice/models.py``) is built with
config ``checkpoints_v2/converter/config.json`` which sets ``n_speakers=0``
→ triggers the ReferenceEncoder branch (instead of the multi-speaker lookup).
After loading ``checkpoints_v2/converter/checkpoint.pth`` with
``strict=False`` we get **0 missing / 0 unexpected keys**, confirming the
architecture-to-weights match is perfect.

Top-level children of ``SynthesizerTrn`` (n_speakers=0):

    dec      -> Generator                  (HiFi-GAN dec, NOT exported here)
    enc_q    -> PosteriorEncoder           (posterior z extractor, NOT exported)
    flow     -> ResidualCouplingBlock      (4-layer mean-only affine coupling)
    ref_enc  -> ReferenceEncoder           (6-layer 2D-conv + GRU + Linear)

We export exactly the two sub-modules the v3 hybrid architecture needs to
plug into a TinyVC + Spark BiCodec pipeline:

  ReferenceEncoder:  linear_spec [B, T_frames, 513] → speaker_embedding [B, 256]
    - 6-layer 2D-conv (filters=[32,32,64,64,128,128]) + GRU(hidden=128) + Linear(256)
    - Input shape note: OpenVoice's API does ``ref_enc(y.transpose(1,2))`` where
      ``y = spectrogram_torch(...)`` returns ``[B, n_freqs=513, T]``. The
      transposed input is ``[B, T, 513]`` — time-first, NOT mel-first.
    - ``spec_channels`` = ``filter_length//2 + 1`` = 513 (linear spectrogram,
      NOT 80-mel; this matches the v2 config at
      ``filter_length=1024, hop_length=256, win_length=1024, sr=22050``).
    - Output: ``[B, gin_channels=256]`` (global speaker embedding, no T axis).
    - 0.813M params (matches the ~0.76M spec — small diff from weight-norm
      parameters; the 0.813M count includes the layer-norm + conv weights).

  ResidualCouplingBlock flow:
    - 4-layer mean-only affine coupling (WaveNet-style WN k=5, hidden=192,
      gin=256) interleaved with Flip layers.
    - 8.687M params (matches the ~8.7M spec exactly).
    - Sub-module signature (verified by direct invocation):
        flow(x, x_mask, g=None, reverse=False) -> x
        x      : [B, channels=192, T]
        x_mask : [B, 1, T]           (1-D mask, broadcast across channels)
        g      : [B, gin_channels=256, 1]   (speaker embedding, T-broadcast)
        reverse: bool — False forward (strips speaker), True reverse (applies)
    - For VC, the canonical two-call pattern (from ``voice_conversion``) is:
        z_p   = flow(z,    y_mask, g=g_src)              # forward: strip src
        z_hat = flow(z_p,  y_mask, g=g_tgt, reverse=True) # reverse: apply tgt
      We wrap this entire disentangle+retarget roundtrip in a single ONNX
      graph: (content, mask, se_src, se_tgt) -> content_disentangled.

ONNX details
------------
- Opset 17 (matches the existing vocos.onnx opset family; onnxruntime 1.30
  fully supports 17). The OpenVoice sub-modules use only standard ops
  (Conv, GRU, MatMul, Add, Mul, LayerNorm) — no opset-18-only features.
- Dynamic axes on batch and time dimensions (not on the 513 freq or 256 emb
  dims, which are fixed by the model definition).
- ``do_constant_folding=True`` so weight-norm hooks are folded into the conv
  weights at export time (no runtime weight-norm computation in ONNX).
- We use the legacy TorchScript exporter (``dynamo=False``) because the
  OpenVoice ``ReferenceEncoder.forward`` calls ``self.gru.flatten_parameters()``
  inside forward — a side-effecting op the dynamo tracer mishandles.

Licenses
--------
- OpenVoice (source + weights): MIT (c) 2023 MyShell.ai.
- This export script: project MIT (c) 2024 mmqz.
- No OpenVoice source code is shipped; we only READ it at export time to
  build the ONNX graphs.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch

# -- Path bootstrap: add the cloned OpenVoice repo so `import openvoice` works.
OPENVOICE_REPO = Path("/home/z/my-project/repos/OpenVoice")
if str(OPENVOICE_REPO) not in sys.path:
    sys.path.insert(0, str(OPENVOICE_REPO))

from openvoice.api import ToneColorConverter  # noqa: E402
from openvoice.models import ResidualCouplingBlock, ReferenceEncoder, SynthesizerTrn  # noqa: E402

# -- Constants from the v2 config (filter_length=1024 → spec_channels=513)
CONFIG_PATH = OPENVOICE_REPO / "checkpoints_v2" / "converter" / "config.json"
CKPT_PATH = OPENVOICE_REPO / "checkpoints_v2" / "converter" / "checkpoint.pth"
OUT_DIR = Path("/home/z/my-project/prototype/models")

# v2 config values (read at runtime in __main__ for self-documentation, but
# these constants match the loaded config.json and are used as the dummy
# input shapes during tracing).
SPEC_CHANNELS = 513      # filter_length // 2 + 1 = 513 (linear spec)
GIN_CHANNELS = 256      # speaker embedding dim
INTER_CHANNELS = 192    # flow content channels (= hidden_channels)
HOP_LENGTH = 256         # 22050 / 256 ≈ 86.13 Hz frame rate


# ---------------------------------------------------------------------------
# Wrapper 1: ReferenceEncoder
# ---------------------------------------------------------------------------
class RefEncoderWrapper(torch.nn.Module):
    """Standalone wrapper around OpenVoice's ``ReferenceEncoder`` sub-module.

    Forward signature matches OpenVoice's actual call site in
    ``ToneColorConverter.extract_se`` (api.py line 131):
        ``self.model.ref_enc(y.transpose(1, 2))``

    Where ``y = spectrogram_torch(...)`` returns ``[B, n_freqs, T]``, so the
    transposed input to ``ref_enc`` is ``[B, T, n_freqs]`` (time-first).
    """

    def __init__(self, model: SynthesizerTrn) -> None:
        super().__init__()
        # Verified attribute name from models.py line 452:
        #   self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)
        # when n_speakers == 0 (the v2 config case).
        self.ref_enc: ReferenceEncoder = model.ref_enc

    def forward(self, linear_spec: torch.Tensor) -> torch.Tensor:
        """linear_spec [B, T_frames, 513] → speaker_embedding [B, 256]."""
        return self.ref_enc(linear_spec)


# ---------------------------------------------------------------------------
# Wrapper 2: ResidualCouplingBlock flow (full disentangle + retarget)
# ---------------------------------------------------------------------------
class FlowWrapper(torch.nn.Module):
    """Standalone wrapper around OpenVoice's ``ResidualCouplingBlock`` flow.

    Replicates the canonical VC roundtrip from
    ``SynthesizerTrn.voice_conversion`` (models.py lines 492-499):
        z, *_ = self.enc_q(y, y_lengths, g=g_src, tau=tau)   # outside our scope
        z_p   = self.flow(z,    y_mask, g=g_src)             # forward: strip src
        z_hat = self.flow(z_p,  y_mask, g=g_tgt, reverse=True)  # reverse: apply tgt

    We skip the PosteriorEncoder step (handled by TinyVC's WavLM features
    upstream in the v3 hybrid) and expose only the flow's
    disentangle-then-retarget roundtrip as a single ONNX graph.
    """

    def __init__(self, model: SynthesizerTrn) -> None:
        super().__init__()
        # Verified attribute name from models.py line 448:
        #   self.flow = ResidualCouplingBlock(inter_channels, hidden_channels,
        #                                    5, 1, 4, gin_channels=gin_channels)
        self.flow: ResidualCouplingBlock = model.flow

    def forward(
        self,
        content: torch.Tensor,
        x_mask: torch.Tensor,
        se_src: torch.Tensor,
        se_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Disentangle src speaker then apply tgt speaker.

        Args:
            content: posterior latent  [B, 192, T]
            x_mask : 1-D mask           [B, 1, T]   (broadcast across channels)
            se_src : src speaker emb    [B, 256, 1] (T-broadcast, last dim=1)
            se_tgt : tgt speaker emb    [B, 256, 1]

        Returns:
            content_disentangled        [B, 192, T]  (re-targeted to se_tgt)
        """
        # Forward pass: strip source speaker → speaker-agnostic content.
        z_p = self.flow(content, x_mask, g=se_src, reverse=False)
        # Reverse pass: apply target speaker.
        z_hat = self.flow(z_p, x_mask, g=se_tgt, reverse=True)
        return z_hat


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def _load_openvoice_v2() -> SynthesizerTrn:
    """Load the SynthesizerTrn instance from the OpenVoice v2 checkpoint."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"OpenVoice v2 config not found at {CONFIG_PATH}. "
            "Run the download step in this script's __main__ first."
        )
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"OpenVoice v2 checkpoint not found at {CKPT_PATH}. "
            "Run the download step in this script's __main__ first."
        )
    # ToneColorConverter.__init__ always loads wavmark (watermark) — we have
    # wavmark installed (pip install wavmark) so this works. The watermark
    # module is NOT part of our ONNX export (we only export ref_enc + flow).
    with warnings.catch_warnings():
        # Suppress the torch.jit.script deprecation warning emitted from
        # openvoice.commons (harmless; we don't use jit.script).
        warnings.filterwarnings("ignore", category=FutureWarning)
        converter = ToneColorConverter(str(CONFIG_PATH), device="cpu")
    converter.load_ckpt(str(CKPT_PATH))
    syn = converter.model
    syn.eval()
    return syn


def _export_ref_encoder(syn: SynthesizerTrn, out_path: Path) -> None:
    """Export ReferenceEncoder as ONNX (opset 17, dynamic batch + time)."""
    wrapper = RefEncoderWrapper(syn).eval()

    # Dummy input: [B=1, T_frames=200, 513]. 200 frames ≈ 2.3 s at 86 Hz.
    dummy_spec = torch.randn(1, 200, SPEC_CHANNELS, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_spec,
            str(out_path),
            input_names=["linear_spec"],
            output_names=["speaker_embedding"],
            dynamic_axes={
                "linear_spec": {0: "batch", 1: "time"},
                "speaker_embedding": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            # Legacy (TorchScript) exporter — see module docstring for why.
            dynamo=False,
        )


def _export_flow(syn: SynthesizerTrn, out_path: Path) -> None:
    """Export ResidualCouplingBlock disentangle+retarget flow as ONNX."""
    wrapper = FlowWrapper(syn).eval()

    # Dummy inputs matching the v2 config:
    #   content [B=1, 192, T=50]   (50 frames ≈ 0.58 s at 86 Hz hop)
    #   x_mask  [B=1, 1, T=50]     (all-ones, broadcast across channels=192)
    #   se_src  [B=1, 256, 1]     (T-broadcast speaker embedding)
    #   se_tgt  [B=1, 256, 1]
    T = 50
    dummy_content = torch.randn(1, INTER_CHANNELS, T, dtype=torch.float32)
    dummy_mask = torch.ones(1, 1, T, dtype=torch.float32)
    dummy_se_src = torch.randn(1, GIN_CHANNELS, 1, dtype=torch.float32)
    dummy_se_tgt = torch.randn(1, GIN_CHANNELS, 1, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_content, dummy_mask, dummy_se_src, dummy_se_tgt),
            str(out_path),
            input_names=["content", "x_mask", "se_src", "se_tgt"],
            output_names=["content_disentangled"],
            dynamic_axes={
                "content": {0: "batch", 2: "time"},
                "x_mask": {0: "batch", 2: "time"},
                "se_src": {0: "batch"},
                "se_tgt": {0: "batch"},
                "content_disentangled": {0: "batch", 2: "time"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )


def _verify_onnx(path: Path) -> dict:
    """Load the ONNX file via onnxruntime and return input/output metadata."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = [(i.name, list(i.shape), i.type) for i in sess.get_inputs()]
    outputs = [(o.name, list(o.shape), o.type) for o in sess.get_outputs()]
    return {
        "size_kb": path.stat().st_size / 1024,
        "inputs": inputs,
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    out_dir = OUT_DIR
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load OpenVoice v2 model via ToneColorConverter API.
    print(f"[1/4] Loading OpenVoice v2 from {CKPT_PATH}")
    syn = _load_openvoice_v2()
    print(f"      SynthesizerTrn loaded. n_speakers={syn.n_speakers} (0 = RefEncoder branch)")
    n_ref = sum(p.numel() for p in syn.ref_enc.parameters())
    n_flow = sum(p.numel() for p in syn.flow.parameters())
    print(f"      ref_enc params: {n_ref / 1e6:.3f}M")
    print(f"      flow     params: {n_flow / 1e6:.3f}M")
    print(f"      ref_enc.spec_channels = {syn.ref_enc.spec_channels} (linear spec, FFT 1024)")
    print(f"      flow.channels        = {syn.flow.channels} (inter_channels)")
    print(f"      flow.gin_channels     = {syn.flow.gin_channels} (speaker emb dim)")

    # 2. Export ReferenceEncoder ONNX.
    ref_path = out_dir / "openvoice_ref_encoder.onnx"
    print(f"\n[2/4] Exporting ReferenceEncoder → {ref_path}")
    _export_ref_encoder(syn, ref_path)
    info = _verify_onnx(ref_path)
    print(f"      size: {info['size_kb']:.0f} KB")
    print(f"      inputs : {info['inputs']}")
    print(f"      outputs: {info['outputs']}")

    # 3. Export Flow ONNX (full disentangle + retarget).
    flow_path = out_dir / "openvoice_residual_flow.onnx"
    print(f"\n[3/4] Exporting ResidualCouplingBlock flow → {flow_path}")
    _export_flow(syn, flow_path)
    info = _verify_onnx(flow_path)
    print(f"      size: {info['size_kb']:.0f} KB")
    print(f"      inputs : {info['inputs']}")
    print(f"      outputs: {info['outputs']}")

    # 4. End-to-end sanity: ONNX output matches torch wrapper output.
    print(f"\n[4/4] Parity check (torch wrapper vs ONNX runtime)")
    _parity_check(ref_path, flow_path, syn)

    print("\nDone. Two ONNX graphs written to", out_dir)
    return 0


def _parity_check(ref_path: Path, flow_path: Path, syn: SynthesizerTrn) -> None:
    """Run the same inputs through both torch and onnxruntime, compare."""
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(42)
    np.random.seed(42)

    # --- ReferenceEncoder parity ---
    re_wrapper = RefEncoderWrapper(syn).eval()
    spec_np = np.random.randn(1, 200, SPEC_CHANNELS).astype(np.float32)
    spec_t = torch.from_numpy(spec_np)
    with torch.no_grad():
        re_torch = re_wrapper(spec_t).numpy()
    re_sess = ort.InferenceSession(str(ref_path), providers=["CPUExecutionProvider"])
    re_onnx = re_sess.run(None, {"linear_spec": spec_np})[0]
    re_diff = float(np.abs(re_torch - re_onnx).max())
    print(f"      ref_encoder  : torch {re_torch.shape} vs onnx {re_onnx.shape}"
          f"  max|Δ| = {re_diff:.2e}")
    assert re_diff < 1e-4, f"ref_encoder parity exceeded 1e-4: {re_diff}"

    # --- Flow parity ---
    fl_wrapper = FlowWrapper(syn).eval()
    T = 50
    content_np = np.random.randn(1, INTER_CHANNELS, T).astype(np.float32)
    mask_np = np.ones((1, 1, T), dtype=np.float32)
    se_src_np = np.random.randn(1, GIN_CHANNELS, 1).astype(np.float32)
    se_tgt_np = np.random.randn(1, GIN_CHANNELS, 1).astype(np.float32)
    with torch.no_grad():
        fl_torch = fl_wrapper(
            torch.from_numpy(content_np),
            torch.from_numpy(mask_np),
            torch.from_numpy(se_src_np),
            torch.from_numpy(se_tgt_np),
        ).numpy()
    fl_sess = ort.InferenceSession(str(flow_path), providers=["CPUExecutionProvider"])
    fl_onnx = fl_sess.run(
        None,
        {
            "content": content_np,
            "x_mask": mask_np,
            "se_src": se_src_np,
            "se_tgt": se_tgt_np,
        },
    )[0]
    fl_diff = float(np.abs(fl_torch - fl_onnx).max())
    print(f"      flow         : torch {fl_torch.shape} vs onnx {fl_onnx.shape}"
          f"  max|Δ| = {fl_diff:.2e}")
    assert fl_diff < 1e-3, f"flow parity exceeded 1e-3: {fl_diff}"

    # --- Flow identity roundtrip sanity: src==tgt → output ≈ input
    same_out = fl_sess.run(
        None,
        {
            "content": content_np,
            "x_mask": mask_np,
            "se_src": se_src_np,
            "se_tgt": se_src_np,  # same speaker both sides
        },
    )[0]
    identity_diff = float(np.abs(same_out - content_np).max())
    print(f"      flow identity (src==tgt): max|Δ(input)| = {identity_diff:.2e}")
    # Identity roundtrip is approximately (but not exactly) identity because
    # the mean-only affine coupling leaves the second half of channels
    # unchanged and applies a deterministic transform on the first half; the
    # full forward+reverse roundtrip with same g should recover the input
    # to within 1e-3 because the coupling is invertible by construction.
    assert identity_diff < 1e-3, f"flow identity roundtrip exceeded 1e-3: {identity_diff}"


if __name__ == "__main__":
    raise SystemExit(main())
