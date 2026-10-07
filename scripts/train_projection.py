#!/usr/bin/env python3
"""
Train 768→192→768 projection layers as autoencoder (self-supervised).

The OpenVoice residual flow operates on 192-d posteriors. TinyVC content
features are 768-d (distilled WavLM). We need two linear maps that preserve
as much information as possible through the 192-d bottleneck, so the flow
has something to act on.

Self-supervised: no paired data needed, just reconstruct TinyVC content
features computed on the existing voice/source fixtures.

Bugs in the original task spec fixed here:
  * `models/encoder.pt` is the full `Encoder` (SSL + Pitch), not
    `SSLFeatureEstimator` — must use `module.tinyvc.Encoder`.
  * No `MelSpec` class exists upstream; the actual entry point is
    `module.utils.spectrogram.spectrogram(wav, n_fft, hop_size)`.
  * `nn.Linear(768, 192)` acts on the *last* dim; content is channels-first
    `[1, 768, T]`, so we must transpose to `[1, T, 768]` before/after.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# Make `module.*` (upstream TinyVC) importable
sys.path.insert(0, "../repos/tinyvc")

# Stub optional TinyVC deps we don't exercise. `module/utils/__init__.py`
# imports `f0_estimation` → `torchfcpe` + `pyworld`. We only need the
# encoder + spectrogram, so stub the missing modules (same trick as
# `vc_realtime.infer_v1`).
import types
for _mod_name, _attrs in (
    ("torchfcpe", {"spawn_bundled_infer_model": lambda *a, **kw: None}),
    (
        "pyworld",
        {
            "dio": lambda *a, **kw: None,
            "stonemask": lambda *a, **kw: None,
            "harvest": lambda *a, **kw: None,
        },
    ),
):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        for _k, _v in _attrs.items():
            setattr(_stub, _k, _v)
        sys.modules[_mod_name] = _stub

torch.set_num_threads(2)

from module.tinyvc import Encoder  # noqa: E402
from module.utils.auto_padding import autopad_waveform  # noqa: E402
from module.utils.spectrogram import spectrogram  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
VOICE_DIR = ROOT / "data" / "voices"
SOURCE_DIR = ROOT / "data" / "source"


def encode_wav(encoder: Encoder, wav: np.ndarray) -> torch.Tensor:
    """Run TinyVC SSL encoder on a 1-D float32 wav → content [1, 768, T]."""
    wav = wav.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wf = torch.from_numpy(wav).unsqueeze(0)  # [1, L]
    wf = autopad_waveform(wf)  # pad to multiple of 480
    spec = spectrogram(wf, encoder.n_fft, encoder.hop_size)  # [1, 961, T]
    with torch.no_grad():
        content, _f0 = encoder.infer(spec)  # [1, 768, T]
    return content


def main() -> int:
    # --- Load TinyVC encoder ---
    encoder = Encoder()
    encoder.load_state_dict(
        torch.load(str(MODELS_DIR / "encoder.pt"), map_location="cpu")
    )
    encoder.eval()

    # --- Projection layers (init: truncated identity so the untrained
    # baseline is at least numerically well-behaved) ---
    proj_down = torch.nn.Linear(768, 192, bias=False)
    proj_up = torch.nn.Linear(192, 768, bias=False)
    with torch.no_grad():
        proj_down.weight.zero_()
        proj_up.weight.zero_()
        for i in range(192):
            proj_down.weight[i, i] = 1.0
            proj_up.weight[i, i] = 1.0

    optimizer = torch.optim.Adam(
        list(proj_down.parameters()) + list(proj_up.parameters()), lr=1e-3
    )

    # --- Generate training data: encode all voice + source fixtures ---
    print("Generating training data...")
    train_features: list[torch.Tensor] = []
    for i in range(5):
        wav, _sr = sf.read(str(VOICE_DIR / f"voice_{i}.wav"))
        content = encode_wav(encoder, wav)
        train_features.append(content)
        print(f"  voice_{i}: {tuple(content.shape)}")
    for i in range(1, 11):
        wav, _sr = sf.read(str(SOURCE_DIR / f"source_{i:03d}.wav"))
        content = encode_wav(encoder, wav)
        train_features.append(content)
    # Also the real human source (so the projection sees test-time data)
    for name in ("source_real_001.wav", "source_real_003.wav"):
        wav, _sr = sf.read(str(SOURCE_DIR / name))
        content = encode_wav(encoder, wav)
        train_features.append(content)
    print(f"Training data: {len(train_features)} chunks")

    # --- Training loop (autoencoder: 768→192→768 reconstruction) ---
    # Apply Linear on the last dim, so transpose content [1,768,T] → [1,T,768]
    # → proj_down → [1,T,192] → transpose → [1,192,T] (matches flow input).
    print("Training projection layers (autoencoder)...")
    chunk_size = 50  # frames per minibatch (memory-friendly on CPU)
    for epoch in range(200):
        total_loss = 0.0
        n_steps = 0
        for content in train_features:
            T = content.shape[-1]
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                chunk = content[:, :, start:end]  # [1, 768, chunk]
                # channels-first → channels-last, project, back
                x = chunk.transpose(1, 2)  # [1, chunk, 768]
                z = proj_down(x)  # [1, chunk, 192]
                recon = proj_up(z)  # [1, chunk, 768]
                loss = F.mse_loss(recon, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_steps += 1
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss / max(n_steps, 1):.6f}")

    # --- Save trained projections ---
    out_path = MODELS_DIR / "trained_projection.pt"
    torch.save(
        {
            "proj_down": proj_down.state_dict(),
            "proj_up": proj_up.state_dict(),
        },
        out_path,
    )
    print(f"Saved trained projection to {out_path}")

    # --- Verify reconstruction quality ---
    with torch.no_grad():
        test = train_features[0][:, :, :50].transpose(1, 2)  # [1, 50, 768]
        z = proj_down(test)
        recon = proj_up(z)
        recon_err = F.mse_loss(recon, test).item()
        print(f"Reconstruction MSE: {recon_err:.6f}")
        print(
            f"Original std: {test.std():.4f}, "
            f"Projected std: {z.std():.4f}, "
            f"Reconstructed std: {recon.std():.4f}"
        )
        # Relative error: how much of the variance is preserved
        rel_err = recon_err / (test.var() + 1e-8)
        print(f"Relative reconstruction error (var fraction lost): {rel_err:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
