"""vc_realtime.voicepack — M1a closed-set VoicePack conditioner.

Replaces the per-frame open-set kNN retrieval (KNNRetrieval) with a
closed-set learned conditioner: for each of the 5 pre-registered target
voices, a small FiLM (Feature-wise Linear Modulation) layer applies an
affine transform to a 192-d latent projection of the source content
features, then projects back to 768-d for the downstream DDSP decoder.

Architecture (verified against the audit patch in download/issues-m0.5-to-m3.md)
-----------------------------------------------------------------------------
::

    source content feat  [B, 768, T]
        │
        │ transpose to [B, T, 768]
        ▼
    proj_in: Linear(768 → 192)         # SHARED across all voices
        │
        ▼
    z  [B, T, 192]                     # latent content
        │
        │  + per-voice FiLM: γ_v ⊙ z + β_v   (γ, β ∈ R^192)
        ▼
    z_cond  [B, T, 192]                # speaker-conditioned latent
        │
        ▼
    proj_out: Linear(192 → 768)        # SHARED across all voices
        │
        │ transpose back to [B, 768, T]
        ▼
    target-voiced content feat  [B, 768, T]

Why this beats kNN (SolA failure root cause)
--------------------------------------------
SolA trained the projection as an autoencoder (L2 reconstruction of 768-d
from 192-d), which is the wrong objective. The 192-d latent does not need
to be a faithful reconstruction of 768-d — it needs to be a *separable*
space where speaker identity can be injected by FiLM. The joint training
objective (see scripts/train_voicepack_joint.py) is:

    L = α · (1 - cos(FiLM(proj(content_src)), content_tgt))    # main
      + β · CE(speaker_classifier(z), tgt_speaker_id)           # auxiliary

The auxiliary classifier forces speaker info to flow through the FiLM
conditioning (rather than leaking into the projection).

Memory footprint (vs v1 kNN)
---------------------------
- v1 kNN: 5 voices × [1, 768, 1500] FP16 = 11.5 MB
- M1a VoicePack: 3 tensors (proj_in, proj_out, 5×FiLM(192-d × 2))
  = 768*192 + 192*768 + 5*(2*192) = ~295 K params FP16 ≈ 600 KB total
  → ~19× smaller than v1 kNN index.

Protocol conformance
--------------------
VoicePackConditioner implements the `SpeakerConditioner` Protocol from
`vc_realtime.interfaces`:
- `select_voice(voice_id: int) -> None` — O(1) pointer swap
- `replace(source_feat: ContentFeat) -> ContentFeat` — forward pass
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Torch imports kept lazy — package remains importable without torch installed
# ---------------------------------------------------------------------------
def _import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
def build_model(
    n_voices: int = 5,
    content_dim: int = 768,
    latent_dim: int = 192,
) -> Any:
    """Construct the joint projection + per-voice FiLM + back-projection.

    Returns a torch.nn.Module with:
      - self.proj_in   Linear(content_dim → latent_dim)
      - self.film      ModuleList of FiLM(latent_dim) per voice
      - self.proj_out  Linear(latent_dim → content_dim)
    """
    torch, nn, F = _import_torch()

    class FiLM(nn.Module):
        """Feature-wise Linear Modulation: γ * x + β (per-channel affine)."""

        def __init__(self, dim: int):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))

        def forward(self, x):
            # x: [B, T, D] or [B, D, T] — works for either since broadcasting
            return x * self.gamma + self.beta

    class VoicePack(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = nn.Linear(content_dim, latent_dim)
            self.film = nn.ModuleList([FiLM(latent_dim) for _ in range(n_voices)])
            self.proj_out = nn.Linear(latent_dim, content_dim)
            # Init proj_out weights so the untrained baseline ≈ identity
            # (helps M1a warmup; proj_in is a random init, but the FiLM
            # starts as identity, so the only noise is from proj_in↔proj_out).
            with torch.no_grad():
                # Truncated identity: keep first 192 of 768 → 192 → 768
                w = torch.zeros(content_dim, latent_dim)
                for i in range(min(latent_dim, content_dim)):
                    w[i, i] = 1.0
                self.proj_out.weight.copy_(w)
                self.proj_out.bias.zero_()

        def forward(self, x_content_first: torch.Tensor,
                    voice_id: int) -> torch.Tensor:
            """x_content_first: [B, C, T] → [B, C, T] (target-voiced)."""
            B, C, T = x_content_first.shape
            # [B, C, T] → [B, T, C]
            x = x_content_first.transpose(1, 2)
            z = self.proj_in(x)               # [B, T, latent]
            z = self.film[voice_id](z)        # apply per-voice FiLM
            out = self.proj_out(z)            # [B, T, C]
            return out.transpose(1, 2)        # back to [B, C, T]

    return VoicePack()


# ---------------------------------------------------------------------------
# Runtime wrapper (Protocol-conformant)
# ---------------------------------------------------------------------------
class VoicePackConditioner:
    """Runtime wrapper around a trained VoicePack model.

    Implements the `SpeakerConditioner` Protocol from vc_realtime.interfaces.
    Loads weights from a safetensors file (saved by train_voicepack_joint.py).

    Usage
    -----
        cond = VoicePackConditioner("models/voicepack_v1.safetensors")
        cond.select_voice(2)
        target_feat = cond.replace(source_feat)  # [1, 768, T] → [1, 768, T]
    """

    def __init__(self, weights_path: str | Path, device: str = "cpu"):
        torch, nn, F = _import_torch()
        self.torch = torch
        self.device = device
        # Build model skeleton, then load weights
        self.model = build_model()
        weights = self._load_weights(weights_path)
        self.model.load_state_dict(weights)
        self.model.eval()
        self.model.to(device)
        self._active_voice: int = 0

    @staticmethod
    def _load_weights(path: str | Path) -> dict:
        from safetensors.torch import load_file
        return load_file(str(path))

    # ----- Protocol: SpeakerConditioner ---------------------------------
    def select_voice(self, voice_id: int) -> None:
        """Hot-swap active target voice. O(1) — just sets an int."""
        if not 0 <= voice_id < len(self.model.film):
            raise ValueError(
                f"voice_id {voice_id} out of range [0, {len(self.model.film)})")
        self._active_voice = int(voice_id)

    def replace(self, source_feat: np.ndarray) -> np.ndarray:
        """Apply the active voice's FiLM conditioning to source features.

        Args:
            source_feat: [B, 768, T] float32 (matches ContentFeat type alias)
        Returns:
            [B, 768, T] float32 (target-voiced content features)
        """
        torch = self.torch
        # Accept both numpy and torch tensors
        if isinstance(source_feat, np.ndarray):
            x = torch.from_numpy(source_feat.astype(np.float32))
        else:
            x = source_feat
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x, self._active_voice)
        if isinstance(source_feat, np.ndarray):
            return out.cpu().numpy()
        return out


# ---------------------------------------------------------------------------
# Convenience: register + hot-swap (mirrors KNNRetrieval's API)
# ---------------------------------------------------------------------------
__all__ = ["VoicePackConditioner", "build_model"]
