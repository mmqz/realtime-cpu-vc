"""vc_realtime.voicepack_v2 — M1a redesign: per-voice FiLM at 768-d (no projection).

V1 (voicepack.py) used a 768→192→768 bottleneck + per-voice FiLM(192). The
bottleneck loses 75% of channel variance, and the disentanglement loss
(speaker classifier) never actually converged — the latent z didn't carry
speaker info because proj_in learned to drop it (which is what autoencoder-
style training does). So V1 actively hurt VC quality even with 30 epochs of
training (M1a+M1b joint delta -0.014 vs M1b alone +0.131).

V2 redesign: skip the projection entirely. Apply per-voice FiLM directly
at 768-d. This is mathematically equivalent to "per-voice learned per-channel
mean/std bias" applied to the source content features. Much simpler, no
information loss, and the per-voice FiLM(768) has 2×768 = 1536 params per
voice (vs V1's proj_in 768×192 + proj_out 192×768 + FiLM(192) = ~296K shared
+ 384 per voice).

Trade-off: V2 loses the "separable latent" property — the disentanglement
auxiliary loss is no longer meaningful because there's no latent to
classify. So V2 training uses only the main loss (cosine to target content).

Architecture (V2)
----------------
::

    source content feat  [B, 768, T]
        │
        │ transpose to [B, T, 768]
        ▼
    + per-voice FiLM(768): γ_v ⊙ x + β_v   (γ, β ∈ R^768)
        │
        │ transpose back to [B, 768, T]
        ▼
    target-voiced content feat  [B, 768, T]

Memory: 5 voices × (2 × 768) FP16 = 15.4 KB total (vs V1 582 KB).

Protocol conformance: same as V1 — implements SpeakerConditioner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_model_v2(n_voices: int = 5, content_dim: int = 768) -> Any:
    """Build V2 VoicePack: per-voice FiLM at 768-d (no projection)."""
    torch, nn, F = _import_torch()

    class FiLM768(nn.Module):
        """Per-voice feature-wise linear modulation at content_dim."""
        def __init__(self, dim: int):
            super().__init__()
            # Init γ=1, β=0 (identity → untrained = no-op, safe baseline)
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))

        def forward(self, x):
            return x * self.gamma + self.beta

    class VoicePackV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.film = nn.ModuleList([FiLM768(content_dim) for _ in range(n_voices)])

        def forward(self, x_content_first: "torch.Tensor", voice_id: int) -> "torch.Tensor":
            # x: [B, C, T] → transpose to [B, T, C] for FiLM broadcast
            x = x_content_first.transpose(1, 2)
            x = self.film[voice_id](x)
            return x.transpose(1, 2)

    return VoicePackV2()


class VoicePackConditionerV2:
    """V2 runtime wrapper. Same SpeakerConditioner Protocol as V1.

    Usage:
        cond = VoicePackConditionerV2("models/voicepack_v2.safetensors")
        cond.select_voice(2)
        target_feat = cond.replace(source_feat)  # [1, 768, T] → [1, 768, T]
    """

    def __init__(self, weights_path: str | Path, device: str = "cpu"):
        torch, nn, F = _import_torch()
        self.torch = torch
        self.device = device
        self.model = build_model_v2()
        weights = self._load_weights(weights_path)
        self.model.load_state_dict(weights)
        self.model.eval()
        self.model.to(device)
        self._active_voice: int = 0

    @staticmethod
    def _load_weights(path: str | Path) -> dict:
        from safetensors.torch import load_file
        return load_file(str(path))

    def select_voice(self, voice_id: int) -> None:
        if not 0 <= voice_id < len(self.model.film):
            raise ValueError(
                f"voice_id {voice_id} out of range [0, {len(self.model.film)})")
        self._active_voice = int(voice_id)

    def replace(self, source_feat: np.ndarray) -> np.ndarray:
        torch = self.torch
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


__all__ = ["VoicePackConditionerV2", "build_model_v2"]
