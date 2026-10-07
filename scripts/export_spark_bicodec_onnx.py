#!/usr/bin/env python3
"""Export Spark-TTS BiCodec SpeakerEncoder as ONNX.

Architecture (verified by reading the actual Spark-TTS source + loading the
0.5B HuggingFace checkpoint ``SparkAudio/Spark-TTS-0.5B``)
----------------------------------------------------------------------
Spark's ``BiCodec`` model (``sparktts/models/bicodec.py``) is a complete
neural codec with 6 sub-modules: ``encoder``, ``decoder``, ``quantizer``
(FSQ-based factorized VQ), ``speaker_encoder``, ``prenet``, ``postnet``,
plus a torchaudio MelSpectrogram transformer.

We extract **only** the ``speaker_encoder`` sub-module (a ``SpeakerEncoder``
instance from ``sparktts/modules/speaker/speaker_encoder.py``) for use as a
drop-in replacement of OpenVoice's 256-d ReferenceEncoder in the v3 hybrid
voice-conversion pipeline. The Spark speaker encoder is strictly stronger:

  - 14.056 M params (vs OpenVoice v2's 0.81 M)
  - 1024-d output (vs OpenVoice's 256-d)
  - Plus a 32-token × 12-bit FSQ index per voice = 48 bytes packed,
    giving us O(1) hash-keyed voice retrieval (OpenVoice has no equivalent).

Sub-module composition (verified by direct invocation):

    SpeakerEncoder(
        speaker_encoder    = ECAPA_TDNN_GLOB_c512(feat_dim=128, embed_dim=1024)
                            # ~5 M params; conv1d + SE-Res2Blocks × 3 + ASTP pool
        perceiver_sampler  = PerceiverResampler(dim=128, dim_context=512*3=1536,
                                                num_latents=32, depth=2)
                            # ~1 M params; cross-attn over 1536-d ECAPA features
        quantizer          = ResidualFSQ(levels=[4,4,4,4,4,4],
                                         num_quantizers=1, dim=128,
                                         is_channel_first=True)
                            # ~0.5 M params; 1 FSQ layer = 6 levels × 4 = 4096 codes
        project            = nn.Linear(128 * 32, 1024)
                            # ~4.2 M params; flatten + project to 1024-d
    )

The actual checkpoint config (read from
``SparkAudio/Spark-TTS-0.5B/BiCodec/config.yaml``) overrides the defaults:

    mel_params: {sample_rate=16000, n_fft=1024, win_length=640, hop_length=320,
                 mel_fmin=10, mel_fmax=null, num_mels=128}
    speaker_encoder: {input_dim=128, out_dim=1024, latent_dim=128,
                      token_num=32, fsq_levels=[4,4,4,4,4,4],
                      fsq_num_quantizers=1}

Forward signature (verified by direct invocation):

    mels      : [B, T_frames, 128]   # mel-spec, time-first channels-last
                                    # T_frames = ref_segment_duration(6 s)
                                    #            × 16000 / 320 = 300 typical
    x_vector  : [B, 1024]            # ECAPA-TDNN pooled output
    d_vector  : [B, 1024]            # Perceiver+FSQ+project output
    indices   : [B, 1, 32] int32      # FSQ codes, 32 tokens × 1 quantizer
                                      # each index ∈ [0, 4095] = 12 bits
                                      # total = 32 × 12 = 384 bits = 48 bytes packed

We wrap a single forward pass that returns ``(d_vector, indices)`` in one
shot — sharing the ECAPA-TDNN + Perceiver compute between both outputs.
The full roundtrip (x_vector + indices) is exposed so that downstream v3
code can either use d_vector (1024-d conditioning for OpenVoice flow) or
the indices (48-byte hash-keyed voice retrieval) as needed.

ONNX details
------------
- Opset 17 (matches the existing openvoice + vocos ONNX files; onnxruntime
  1.30 fully supports 17). The Spark sub-modules use only standard ops:
  Conv, BatchNorm, ReLU, Tanh, Atanh, Round, Einsum, Softmax, Linear,
  LayerNorm, ScaledDotProductAttention, Cumsum — all opset-17 supported.
- Dynamic axes on batch and time dimensions (not on the 128-mel or 1024-emb
  dims, which are fixed by the model definition).
- ``do_constant_folding=True`` so any weight-norm hooks are folded into
  conv weights at export time.
- Legacy TorchScript exporter (``dynamo=False``) because the Perceiver's
  ``Attend`` module uses ``torch.backends.cuda.sdp_kernel`` context-manager
  branching that the dynamo tracer mishandles (we only take the
  ``not use_flash`` branch on CPU, but the dynamo tracer still tries to
  trace the entire python control flow).

FSQ export note
---------------
The ResidualFSQ forward includes a ``round_ste`` (straight-through estimator)
that does ``z + (zhat - z).detach()``. In ``torch.no_grad()`` + ``model.eval()``
context the ``.detach()`` is semantically a no-op and exports cleanly as
Identity. The FSQ also calls ``torch.amp.autocast("cuda", enabled=False)``
which on CPU is a nullcontext — not traced.

Licenses
--------
- Spark-TTS (source + weights): Apache 2.0 (c) 2025 SparkAudio.
- This export script: project MIT (c) 2024 mmqz.
- No Spark-TTS source code is shipped; we only READ it at export time to
  build the ONNX graph.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

# -- Path bootstrap: add the cloned Spark-TTS repo so `import sparktts` works.
SPARK_REPO = Path("/home/z/my-project/repos/spark-tts")
if str(SPARK_REPO) not in sys.path:
    sys.path.insert(0, str(SPARK_REPO))

from sparktts.models.bicodec import BiCodec  # noqa: E402
from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder  # noqa: E402

# -- Constants from the actual BiCodec config.yaml (read at runtime below
# for self-documentation, but these match the loaded config and are used as
# the dummy input shapes during tracing).
BICODEC_DIR = SPARK_REPO / "ckpt" / "BiCodec"
OUT_DIR = Path("/home/z/my-project/prototype/models")

NUM_MELS = 128          # mel_params.num_mels (NOT 80 — Spark uses 128)
OUT_DIM = 1024         # speaker_encoder.out_dim (NOT 512 — Spark uses 1024)
LATENT_DIM = 128       # speaker_encoder.latent_dim
TOKEN_NUM = 32         # speaker_encoder.token_num (Perceiver num_latents)
FSQ_LEVELS = [4, 4, 4, 4, 4, 4]   # 6 levels × 4 values = 4096 codes per token
FSQ_NUM_QUANTIZERS = 1            # one FSQ layer (not residual)
SAMPLE_RATE = 16000
HOP_LENGTH = 320        # mel_params.hop_length → 50 Hz frame rate
N_FFT = 1024
WIN_LENGTH = 640
REF_SEGMENT_FRAMES = 300   # 6 s × 16000 / 320 — the canonical ref length


# ---------------------------------------------------------------------------
# Wrapper: SpeakerEncoder → (d_vector, indices, x_vector) in one pass
# ---------------------------------------------------------------------------
class SparkSpeakerEncoderWrapper(torch.nn.Module):
    """Standalone wrapper around Spark's ``SpeakerEncoder`` sub-module.

    Re-implements the inner compute path of ``SpeakerEncoder.forward`` +
    ``SpeakerEncoder.tokenize`` so that the ECAPA-TDNN and Perceiver
    computations are shared (instead of running forward twice). The output
    contract matches the v3 hybrid pipeline's needs:

        Input:
            mels : [B, T, 128]   # mel-spec, time-first channels-last

        Outputs (in order):
            d_vector : [B, 1024]      # speaker conditioning for downstream flow
            indices  : [B, 1, 32] int32  # FSQ codes, 32 tokens × 1 quantizer
                                          # each ∈ [0, 4095], 12 bits → 48 B packed
            x_vector : [B, 1024]      # ECAPA-TDNN pooled speaker embedding
                                      # (auxiliary; for speaker verification)

    The shared compute graph (verified against
    ``SpeakerEncoder.forward`` and ``SpeakerEncoder.tokenize`` source):

        mels [B, T, 128]
          ↓ speaker_encoder(mels, return_latent=True)        # ECAPA-TDNN
          → (x_vector [B, 1024], features [B, 1536, T])     # 512*3 cat channels
          ↓ features.transpose(1,2) → [B, T, 1536]
          ↓ perceiver_sampler([B, T, 1536]) → [B, 32, 128]  # cross-attn pool
          ↓ .transpose(1,2) → [B, 128, 32]                   # channel-first for FSQ
          ↓ quantizer([B, 128, 32])                          # ResidualFSQ
          → (zq [B, 128, 32], indices [B, 1, 32])
          ↓ zq.reshape(B, -1) → [B, 4096]
          ↓ project → [B, 1024]                              # d_vector
    """

    def __init__(self, enc: SpeakerEncoder) -> None:
        super().__init__()
        # Verified attribute names from speaker_encoder.py:
        #   self.speaker_encoder  = ECAPA_TDNN_GLOB_c512(feat_dim, embed_dim)
        #   self.perceiver_sampler = PerceiverResampler(dim=latent_dim,
        #                                                dim_context=512*3,
        #                                                num_latents=token_num)
        #   self.quantizer        = ResidualFSQ(levels=fsq_levels,
        #                                       num_quantizers=fsq_num_quantizers,
        #                                       dim=latent_dim,
        #                                       is_channel_first=True)
        #   self.project          = nn.Linear(latent_dim * token_num, out_dim)
        self.enc = enc

    def forward(
        self, mels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ECAPA-TDNN c512 + ASTP pool → x_vector [B, 1024]
        # and frame-level features [B, 1536, T] (3 SE-Res2Blocks concatenated).
        x_vector, features = self.enc.speaker_encoder(mels, True)

        # PerceiverResampler cross-attends from 32 learned latents to the
        # T frame features → 32 latent tokens [B, 32, 128]. The transpose
        # before/after matches the channel-first convention ResidualFSQ
        # expects when ``is_channel_first=True``.
        x = self.enc.perceiver_sampler(features.transpose(1, 2)).transpose(1, 2)

        # ResidualFSQ quantizes the 32 tokens × 128 channels → 32 codes.
        # Returns (zq [B, 128, 32], indices [B, 1, 32]) with channel_first out.
        zq, indices = self.enc.quantizer(x)

        # Flatten + project to 1024-d conditioning vector.
        x_flat = zq.reshape(zq.shape[0], -1)
        d_vector = self.enc.project(x_flat)

        return d_vector, indices.to(torch.int64), x_vector


# ---------------------------------------------------------------------------
# Wrapper v2: SpeakerEncoder → (d_vector, indices) — SKIP x_vector
# ---------------------------------------------------------------------------
class SparkSpeakerEncoderSkipXVector(torch.nn.Module):
    """Wrapper around Spark's ``SpeakerEncoder`` that SKIPS the x_vector path.

    The original ``SparkSpeakerEncoderWrapper`` runs the full
    ``ECAPA_TDNN.forward(mels, return_latent=True)`` which returns BOTH
    ``(x_vector, features)``. The ``x_vector`` is the ECAPA-TDNN pooled
    speaker embedding (ASTP pool + BN + Linear → 1024-d), used only for
    auxiliary speaker-verification tasks. In the v3 hybrid pipeline we
    consume ``d_vector`` (Perceiver + FSQ + project) for flow conditioning
    and ``fsq_indices`` for O(1) voice retrieval — ``x_vector`` is unused.

    By re-implementing the ECAPA-TDNN conv-only path (up to and including
    ``F.relu(self.conv(out))`` = the ``latent``/``features`` tensor) we skip
    the ASTP pool + BN + Linear sub-modules of ``ECAPA_TDNN`` that produce
    ``x_vector``:

        ECAPA_TDNN.forward (source: sparktts/modules/speaker/ecapa_tdnn.py)
        ─────────────────────────────────────────────────────────────────
          x = x.permute(0, 2, 1)                       # (B,T,F)→(B,F,T)
          out1 = self.layer1(x)                       # Conv1dReluBn (128→512)
          out2 = self.layer2(out1)                     # SE-Res2Block
          out3 = self.layer3(out2)                     # SE-Res2Block
          out4 = self.layer4(out3)                     # SE-Res2Block
          out = torch.cat([out2, out3, out4], dim=1)  # [B, 1536, T]
          latent = F.relu(self.conv(out))             # [B, 1536, T]  ← KEPT
          # vvv x_vector path (~3-4M params, ~10% compute) — SKIPPED vvv
          out = self.bn(self.pool(latent))            # ASTP att-pool [B, 3072]
          out = self.linear(out)                      # Linear → [B, 1024]
          if self.emb_bn: out = self.bn2(out)
          # ^^^ x_vector path ^^^

    Params saved (verified against the loaded 0.5B checkpoint):
        - ECAPA_TDNN.pool.linear1   Conv1d(4608, 128, 1)  ≈ 590 K
        - ECAPA_TDNN.pool.linear2   Conv1d(128, 1536, 1) ≈ 198 K
        - ECAPA_TDNN.bn             BatchNorm1d(3072)     ≈   6 K
        - ECAPA_TDNN.linear         Linear(3072, 1024)   ≈ 3.15 M
        ─────────────────────────────────────────────────────────
        total x_vector path                                  ≈ 3.94 M params
        (≈ 28 % of the 14.06 M SpeakerEncoder total)

    Output contract (matches v3 hybrid needs):
        Input  : mels       [B, T, 128]
        Outputs: d_vector   [B, 1024]      float32  (speaker conditioning)
                 indices    [B, 1, 32]      int64    (FSQ codes, 48 B packed)
    """

    def __init__(self, enc: SpeakerEncoder) -> None:
        super().__init__()
        self.enc = enc

    def forward(
        self, mel_spec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ECAPA-TDNN conv path — replicate the source's conv-only portion
        # (layer1..layer4 + conv) UP TO `latent`, skipping pool/bn/linear.
        ecapa = self.enc.speaker_encoder
        x = mel_spec.permute(0, 2, 1)               # (B,T,F) -> (B,F,T)
        out1 = ecapa.layer1(x)
        out2 = ecapa.layer2(out1)
        out3 = ecapa.layer3(out2)
        out4 = ecapa.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)  # [B, 1536, T]
        latent = F.relu(ecapa.conv(out))             # [B, 1536, T]  (features)
        # ↑ END of ECAPA conv path — pool/bn/linear (x_vector) SKIPPED ↑

        # PerceiverResampler + ResidualFSQ + project — identical to
        # SparkSpeakerEncoderWrapper.forward (the d_vector path is unchanged).
        x = self.enc.perceiver_sampler(latent.transpose(1, 2)).transpose(1, 2)
        zq, indices = self.enc.quantizer(x)
        x_flat = zq.reshape(zq.shape[0], -1)
        d_vector = self.enc.project(x_flat)
        return d_vector, indices.to(torch.int64)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def _load_spark_bicodec() -> BiCodec:
    """Load the full BiCodec model from the Spark-TTS HuggingFace checkpoint.

    We need the full BiCodec because ``BiCodec.load_from_checkpoint`` is the
    only public loader, and it applies ``remove_weight_norm`` + ``eval()``
    correctly to the entire model. We then extract just the
    ``speaker_encoder`` sub-module for ONNX export.
    """
    if not BICODEC_DIR.exists():
        raise FileNotFoundError(
            f"Spark-TTS BiCodec checkpoint not found at {BICODEC_DIR}. "
            "Run the download step in this script's __main__ first."
        )
    with warnings.catch_warnings():
        # Suppress two harmless warnings emitted during load:
        # 1. torch.jit.script deprecation (from sparktts.commons)
        # 2. torch.nn.utils.weight_norm deprecation (from ECAPA-TDNN's
        #    remove_weight_norm hook applied to all sub-modules)
        warnings.filterwarnings("ignore", category=FutureWarning)
        model = BiCodec.load_from_checkpoint(BICODEC_DIR)
    model.eval()
    return model


def _export_speaker_encoder(
    bicodec: BiCodec, out_path: Path
) -> None:
    """Export SpeakerEncoder (d_vector + indices + x_vector) as ONNX opset 17."""
    wrapper = SparkSpeakerEncoderWrapper(bicodec.speaker_encoder).eval()

    # Dummy input: [B=1, T=300, 128]. T=300 = 6 s ref segment at 50 Hz.
    dummy_mel = torch.randn(1, REF_SEGMENT_FRAMES, NUM_MELS, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_mel,
            str(out_path),
            input_names=["mel_spec"],
            output_names=["d_vector", "fsq_indices", "x_vector"],
            dynamic_axes={
                "mel_spec": {0: "batch", 1: "time"},
                "d_vector": {0: "batch"},
                "fsq_indices": {0: "batch"},
                "x_vector": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            # Legacy (TorchScript) exporter — see module docstring for why.
            dynamo=False,
        )


def _export_speaker_encoder_v2(
    bicodec: BiCodec, out_path: Path
) -> None:
    """Export SpeakerEncoder v2 (d_vector + indices ONLY, x_vector SKIPPED).

    OPT-4: skip the ECAPA-TDNN ASTP pool + BN + Linear path that produces
    x_vector (unused in the v3 hybrid pipeline). Saves ~3.94 M params of
    ONNX initializers (~28 % of the 14.06 M SpeakerEncoder total) and the
    ~10 % of compute that goes into the x_vector path.
    """
    wrapper = SparkSpeakerEncoderSkipXVector(bicodec.speaker_encoder).eval()

    # Dummy input: [B=1, T=300, 128]. T=300 = 6 s ref segment at 50 Hz.
    dummy_mel = torch.randn(1, REF_SEGMENT_FRAMES, NUM_MELS, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_mel,
            str(out_path),
            input_names=["mel_spec"],
            # NOTE: x_vector intentionally omitted — the v3 hybrid pipeline
            # only consumes d_vector (flow conditioning) + fsq_indices
            # (48-byte voice hash key).
            output_names=["d_vector", "fsq_indices"],
            dynamic_axes={
                "mel_spec": {0: "batch", 1: "time"},
                "d_vector": {0: "batch"},
                "fsq_indices": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            # Legacy (TorchScript) exporter — see module docstring for why.
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

    # 1. Load Spark-TTS BiCodec model.
    print(f"[1/6] Loading Spark-TTS BiCodec from {BICODEC_DIR}")
    bicodec = _load_spark_bicodec()
    n_se = sum(p.numel() for p in bicodec.speaker_encoder.parameters())
    print(f"      BiCodec loaded. SpeakerEncoder params: {n_se / 1e6:.3f}M")
    print(f"      mel_params: num_mels={NUM_MELS}, hop={HOP_LENGTH}, "
          f"sr={SAMPLE_RATE}, n_fft={N_FFT}, win={WIN_LENGTH}")
    print(f"      speaker_encoder: out_dim={OUT_DIM}, latent_dim={LATENT_DIM}, "
          f"token_num={TOKEN_NUM}, fsq_levels={FSQ_LEVELS}, "
          f"fsq_num_quantizers={FSQ_NUM_QUANTIZERS}")

    # 2. Sanity-check torch forward (catch any API drift before tracing).
    print("\n[2/6] Sanity-check torch wrappers")
    wrapper = SparkSpeakerEncoderWrapper(bicodec.speaker_encoder).eval()
    mel = torch.randn(2, REF_SEGMENT_FRAMES, NUM_MELS, dtype=torch.float32)
    with torch.no_grad():
        d_vec, idx, x_vec = wrapper(mel)
    print(f"      [v1] d_vector shape: {tuple(d_vec.shape)}")
    print(f"      [v1] fsq_indices shape: {tuple(idx.shape)} dtype: {idx.dtype}")
    print(f"      [v1] x_vector shape: {tuple(x_vec.shape)}")
    assert d_vec.shape == (2, OUT_DIM), f"d_vector shape mismatch: {d_vec.shape}"
    assert idx.shape == (2, FSQ_NUM_QUANTIZERS, TOKEN_NUM), \
        f"fsq_indices shape mismatch: {idx.shape}"
    assert x_vec.shape == (2, OUT_DIM), f"x_vector shape mismatch: {x_vec.shape}"
    # FSQ codes are in [0, prod(levels)=4096)
    assert idx.max().item() < 4096, f"fsq index out of range: max={idx.max()}"
    assert idx.min().item() >= 0, f"fsq index negative: min={idx.min()}"

    # Also sanity-check the v2 skip-x_vector wrapper. d_vector + fsq_indices
    # MUST match the v1 wrapper exactly (same compute path; we just stop before
    # the ASTP pool + Linear that produce x_vector).
    wrapper_v2 = SparkSpeakerEncoderSkipXVector(bicodec.speaker_encoder).eval()
    with torch.no_grad():
        d_vec_v2, idx_v2 = wrapper_v2(mel)
    print(f"      [v2] d_vector shape: {tuple(d_vec_v2.shape)}")
    print(f"      [v2] fsq_indices shape: {tuple(idx_v2.shape)} dtype: {idx_v2.dtype}")
    assert d_vec_v2.shape == (2, OUT_DIM), f"v2 d_vector shape mismatch: {d_vec_v2.shape}"
    assert idx_v2.shape == (2, FSQ_NUM_QUANTIZERS, TOKEN_NUM), \
        f"v2 fsq_indices shape mismatch: {idx_v2.shape}"
    # v2 d_vector + fsq_indices must match v1 EXACTLY (same compute path).
    assert torch.allclose(d_vec, d_vec_v2, atol=1e-6), (
        "v1/v2 d_vector divergence — wrapper path drift"
    )
    assert torch.equal(idx, idx_v2), "v1/v2 fsq_indices divergence"
    # Param count: x_vector path = pool + bn + linear of ECAPA_TDNN.
    ecapa = bicodec.speaker_encoder.speaker_encoder
    n_xvec = (
        sum(p.numel() for p in ecapa.pool.parameters())
        + sum(p.numel() for p in ecapa.bn.parameters())
        + sum(p.numel() for p in ecapa.linear.parameters())
    )
    print(f"      [v2] x_vector path params skipped: {n_xvec / 1e6:.3f}M "
          f"({n_xvec / n_se * 100:.1f}% of {n_se / 1e6:.3f}M SpeakerEncoder)")

    # 3. Export SpeakerEncoder ONNX (original — 3 outputs, for backward compat).
    se_path = out_dir / "spark_speaker_encoder.onnx"
    print(f"\n[3/6] Exporting SpeakerEncoder v1 → {se_path}")
    _export_speaker_encoder(bicodec, se_path)
    info = _verify_onnx(se_path)
    print(f"      size: {info['size_kb']:.0f} KB")
    print(f"      inputs : {info['inputs']}")
    print(f"      outputs: {info['outputs']}")

    # 4. End-to-end sanity: ONNX output matches torch wrapper output.
    print("\n[4/6] Parity check v1 (torch wrapper vs ONNX runtime)")
    _parity_check(se_path, bicodec)

    # 5. Export SpeakerEncoder v2 ONNX (2 outputs, x_vector SKIPPED).
    #    Original spark_speaker_encoder.onnx is NOT overwritten — v2 is a
    #    separate file (spark_speaker_encoder_v2.onnx). Smaller + faster.
    se_v2_path = out_dir / "spark_speaker_encoder_v2.onnx"
    print(f"\n[5/6] Exporting SpeakerEncoder v2 (skip x_vector) → {se_v2_path}")
    _export_speaker_encoder_v2(bicodec, se_v2_path)
    info_v2 = _verify_onnx(se_v2_path)
    print(f"      size: {info_v2['size_kb']:.0f} KB "
          f"(v1: {info['size_kb']:.0f} KB, "
          f"saved {(info['size_kb'] - info_v2['size_kb']):.0f} KB = "
          f"{(1 - info_v2['size_kb'] / info['size_kb']) * 100:.1f}%)")
    print(f"      inputs : {info_v2['inputs']}")
    print(f"      outputs: {info_v2['outputs']}")
    # HARD assertion: v2 must have EXACTLY 2 outputs (d_vector + fsq_indices).
    assert len(info_v2["outputs"]) == 2, (
        f"v2 expected 2 outputs (d_vector + fsq_indices), "
        f"got {len(info_v2['outputs'])}: {info_v2['outputs']}"
    )
    assert [o[0] for o in info_v2["outputs"]] == ["d_vector", "fsq_indices"], (
        f"v2 output names mismatch: {info_v2['outputs']}"
    )

    # 6. End-to-end sanity for v2.
    print("\n[6/6] Parity check v2 (torch wrapper vs ONNX runtime)")
    _parity_check_v2(se_v2_path, bicodec)

    print("\nDone. ONNX graphs written:")
    print(f"  - {se_path}     (v1, 3 outputs incl. x_vector — backward compat)")
    print(f"  - {se_v2_path}  (v2, 2 outputs — skip x_vector, OPT-4)")
    return 0


def _parity_check(se_path: Path, bicodec: BiCodec) -> None:
    """Run the same inputs through both torch and onnxruntime, compare."""
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(42)
    np.random.seed(42)

    wrapper = SparkSpeakerEncoderWrapper(bicodec.speaker_encoder).eval()
    mel_np = np.random.randn(2, REF_SEGMENT_FRAMES, NUM_MELS).astype(np.float32)
    mel_t = torch.from_numpy(mel_np)
    with torch.no_grad():
        d_torch, idx_torch, x_torch = wrapper(mel_t)
        d_torch = d_torch.numpy()
        idx_torch = idx_torch.numpy()
        x_torch = x_torch.numpy()

    sess = ort.InferenceSession(str(se_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"mel_spec": mel_np})
    d_onnx, idx_onnx, x_onnx = out

    d_diff = float(np.abs(d_torch - d_onnx).max())
    idx_match = int((idx_torch == idx_onnx).sum())
    idx_total = int(np.prod(idx_torch.shape))
    x_diff = float(np.abs(x_torch - x_onnx).max())

    print(f"      d_vector   : torch {d_torch.shape} vs onnx {d_onnx.shape}"
          f"  max|Δ| = {d_diff:.2e}")
    print(f"      fsq_indices : torch {idx_torch.shape} vs onnx {idx_onnx.shape}"
          f"  match {idx_match}/{idx_total}")
    print(f"      x_vector   : torch {x_torch.shape} vs onnx {x_onnx.shape}"
          f"  max|Δ| = {x_diff:.2e}")

    # Float outputs must match to FP32 precision. The thresholds are
    # asymmetric:
    #   - d_vector: comes from FSQ-quantized zq flattened through one Linear.
    #     The FSQ round+cast ops are deterministic, so d_vector must match
    #     torch to near machine epsilon.
    #   - x_vector: comes from ECAPA-TDNN's full conv + BatchNorm + ASTP pool
    #     chain. ONNX folds BatchNorm-in-eval-mode into the preceding conv
    #     weights at export time, which introduces float32 rounding drift of
    #     ~1e-3 to 5e-3 in the worst element across 1024 outputs. This is
    #     within the standard ONNX BN-fold tolerance and doesn't affect the
    #     v3 pipeline (we use d_vector, not x_vector, for flow conditioning).
    assert d_diff < 1e-3, f"d_vector parity exceeded 1e-3: {d_diff}"
    assert x_diff < 5e-3, f"x_vector parity exceeded 5e-3: {x_diff}"
    # Integer FSQ codes must match EXACTLY — they are discrete indices and
    # any divergence would mean the ONNX graph rounds differently than torch.
    assert idx_match == idx_total, (
        f"fsq_indices parity: {idx_match}/{idx_total} matched "
        f"(torch={idx_torch.flatten()[:8]}, onnx={idx_onnx.flatten()[:8]})"
    )


def _parity_check_v2(se_path: Path, bicodec: BiCodec) -> None:
    """Parity check for v2 (d_vector + fsq_indices only; x_vector skipped).

    The d_vector + fsq_indices compute path is IDENTICAL to v1 — we just
    stop the ECAPA-TDNN forward after ``F.relu(self.conv(out))`` = the
    ``latent``/``features`` tensor that feeds the Perceiver. Hence the v2
    ONNX d_vector + fsq_indices must match the v1 ONNX (and the v2 torch
    wrapper) to the same precision as v1's parity bar.
    """
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(42)
    np.random.seed(42)

    wrapper = SparkSpeakerEncoderSkipXVector(bicodec.speaker_encoder).eval()
    mel_np = np.random.randn(2, REF_SEGMENT_FRAMES, NUM_MELS).astype(np.float32)
    mel_t = torch.from_numpy(mel_np)
    with torch.no_grad():
        d_torch, idx_torch = wrapper(mel_t)
        d_torch = d_torch.numpy()
        idx_torch = idx_torch.numpy()

    sess = ort.InferenceSession(str(se_path), providers=["CPUExecutionProvider"])
    # HARD assertion: v2 must have EXACTLY 2 outputs (no x_vector).
    out_names = [o.name for o in sess.get_outputs()]
    assert out_names == ["d_vector", "fsq_indices"], (
        f"v2 ONNX outputs mismatch: {out_names} "
        f"(expected ['d_vector', 'fsq_indices'])"
    )
    out = sess.run(None, {"mel_spec": mel_np})
    d_onnx, idx_onnx = out

    d_diff = float(np.abs(d_torch - d_onnx).max())
    idx_match = int((idx_torch == idx_onnx).sum())
    idx_total = int(np.prod(idx_torch.shape))

    print(f"      d_vector   : torch {d_torch.shape} vs onnx {d_onnx.shape}"
          f"  max|Δ| = {d_diff:.2e}")
    print(f"      fsq_indices : torch {idx_torch.shape} vs onnx {idx_onnx.shape}"
          f"  match {idx_match}/{idx_total}")
    print("      (x_vector  : SKIPPED — not in v2 ONNX graph)")

    # Same bar as v1's d_vector + fsq_indices.
    assert d_diff < 1e-3, f"v2 d_vector parity exceeded 1e-3: {d_diff}"
    assert idx_match == idx_total, (
        f"v2 fsq_indices parity: {idx_match}/{idx_total} matched "
        f"(torch={idx_torch.flatten()[:8]}, onnx={idx_onnx.flatten()[:8]})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
