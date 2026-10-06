"""
modules/knn_retrieval.py — kNN-VC speaker conditioning
=========================================================
Source: tinyvc/module/tinyvc/feature_retrieval.py:15-33

Given a source content feature tensor [B, 768, T] and a pre-extracted
target voice feature library [n_voices, 1, 768, T_ref], do per-frame
cosine-similarity top-k weighted-average replacement.

Memory footprint:
    5 voices × [1, 768, 1500] FP16 = 5 × 2.3 MB = 11.5 MB
"""

import numpy as np


class KNNRetrieval:
    """kNN-VC feature replacement.

    Loads voice feature library from a safetensors file at construction time.
    Voice hot-swap = O(1) pointer change.
    """

    def __init__(self, voices_path: str, top_k: int = 4):
        import torch
        from safetensors.torch import load_file

        self.top_k = top_k

        # Load all voice tensors into one tensor [n_voices, 768, T_ref]
        state = load_file(voices_path)
        voice_keys = sorted([k for k in state if "voice_" in k])
        if len(voice_keys) == 0:
            raise ValueError(f"No voice tensors found in {voices_path}")
        # Squeeze the batch dim, stack
        voices = [state[k].squeeze(0) for k in voice_keys]  # each [768, T_ref]
        self.voices = torch.stack(voices, dim=0).to(torch.float32)  # [N, 768, T_ref]
        # Normalize along channel dim for cosine similarity
        self.voices_norm = self.voices / (self.voices.norm(dim=1, keepdim=True) + 1e-8)
        self.n_voices = self.voices.shape[0]
        self.current_voice_idx = 0

    def select_voice(self, voice_id: int):
        """Hot-swap to voice_id. O(1)."""
        assert 0 <= voice_id < self.n_voices, f"voice_id {voice_id} out of range"
        self.current_voice_idx = voice_id

    def replace(self, source_feat: np.ndarray) -> np.ndarray:
        """Per-frame top-k cosine retrieval and replacement.

        Args:
            source_feat: [B=1, 768, T] float32
        Returns:
            target_feat: [B=1, 768, T] float32 (same shape, content replaced)
        """
        import torch

        src = torch.from_numpy(source_feat).squeeze(0)  # [768, T]
        src_norm = src / (src.norm(dim=0, keepdim=True) + 1e-8)  # [768, T]

        # Target voice: [768, T_ref]
        target = self.voices_norm[self.current_voice_idx]  # [768, T_ref]

        # Cosine similarity per frame: src[:, t] · target[:, r] for all t, r
        # Compute via einsum: [T, T_ref]
        sim = torch.einsum("ct,cr->tr", src_norm, target)  # [T, T_ref]

        # Top-k weighted average: 1/score^2 weighting (kNN-VC standard)
        topk_vals, topk_idx = torch.topk(sim, k=self.top_k, dim=-1)  # [T, k]
        weights = 1.0 / (topk_vals**2 + 1e-8)  # [T, k]
        weights = weights / weights.sum(dim=-1, keepdim=True)
        # Gather target features for top-k indices
        # target[topk_idx] shape: [T, k, 768]
        target_expanded = target.t()  # [T_ref, 768]
        selected = target_expanded[topk_idx]  # [T, k, 768]
        # Weighted average
        out = (selected * weights.unsqueeze(-1)).sum(dim=1)  # [T, 768]
        out = out.t().unsqueeze(0)  # [1, 768, T]
        return out.numpy().astype(np.float32)
