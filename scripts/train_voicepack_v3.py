#!/usr/bin/env python3
"""M1a V3 · VoicePack training with mel-spectrogram reconstruction loss.

V1 (proj+FiLM@192) and V2 (FiLM@768) both failed because cosine-to-target-
content loss learns text-specific per-channel bias that doesn't generalize
to novel text (SolA-SolK trap).

V3 takes a different angle: use mel-spectrogram reconstruction loss on the
DECODED AUDIO. The training loop becomes:

  for each (src_audio, tgt_audio, voice_id):
    1. content = V1Infer.encode(src_audio)         # frozen
    2. content_voiced = VoicePack(content, voice_id) # TRAINABLE FiLM@768
    3. out_audio = V1Infer.decode(content_voiced, f0, energy)  # differentiable
    4. out_mel = MelSpec(out_audio)                 # differentiable
    5. tgt_mel = MelSpec(tgt_audio)                 # precomputed, frozen
    6. L = MSE(out_mel, tgt_mel)                    # PyTorch-native loss
    7. Backprop → update VoicePack FiLM params

Why this might work where V1/V2 didn't:
- MSE on mel-spec at the AUDIO level forces the decoder to produce
  audio that matches the target's spectral envelope (formants) AND
  harmonic structure (F0 + overtones).
- Content feature cosine (V1/V2) only matches per-channel statistics
  which encode both text and speaker — easy to overfit to training text.
- Audio-level loss captures the speaker's actual acoustic signature
  (formant frequencies, spectral tilt, breathiness) regardless of text.

Trade-off:
- Slower training (~250ms per decode + 50ms per mel-spec + 50ms backward
  = ~350ms per iteration on CPU vs ~5ms for V1/V2 content-only loss)
- For 80 pairs × 10 epochs = 800 iterations × 350ms = ~5 minutes
- Tractable on CPU

Output:
- models/voicepack_v3.safetensors (~15 KB FP16)
- data/m1a_v3_train_log.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

for _mod_name, _attrs in (
    ("torchfcpe", {"spawn_bundled_infer_model": lambda *a, **kw: None}),
    ("pyworld", {
        "dio": lambda *a, **kw: None, "stonemask": lambda *a, **kw: None,
        "harvest": lambda *a, **kw: None,
    }),
):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        for _k, _v in _attrs.items():
            setattr(_stub, _k, _v)
        sys.modules[_mod_name] = _stub

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

TINYVC_ROOT = Path(os.environ.get(
    "TINYVC_ROOT", str(REPO_ROOT.parent / "repos" / "tinyvc")))
if not TINYVC_ROOT.exists():
    raise SystemExit(f"TinyVC repo not found at {TINYVC_ROOT}")
sys.path.insert(0, str(TINYVC_ROOT.parent))

from vc_realtime.voicepack_v2 import build_model_v2  # noqa: E402

TARGET_SPEAKERS = ["p225", "p226", "p227", "p228", "p229"]


# ---------------------------------------------------------------------------
# Mel-spec extractor (PyTorch, differentiable)
# ---------------------------------------------------------------------------
class DifferentiableMelSpec(nn.Module):
    """PyTorch-native mel-spec extractor matching Vocos's MelSpectrogramFeatures.

    Used as the loss target — audio → 100-mel at 93.75 Hz frame rate.
    Differentiable, so gradient flows back through decode → VoicePack.
    """
    def __init__(self, sample_rate: int = 24000, n_fft: int = 1024,
                 hop_length: int = 256, n_mels: int = 100):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels,
            center=True, power=1.0,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, T_samples] → mel: [B, n_mels, T_frames]
        mel = self.mel_spec(audio)
        return torch.log1p(mel)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_paired_hard_pairs(v1_infer) -> list[dict]:
    """Load (src_audio, tgt_audio, voice_id) tuples from paired_hard."""
    index_path = REPO_ROOT / "data" / "paired_hard" / "index.json"
    if not index_path.exists():
        raise SystemExit(
            f"paired_hard index missing at {index_path}; run "
            f"scripts/m05_real_voices.py + scripts/m1a_patch_paired_hard_sources.py first")
    with open(index_path) as f:
        index = json.load(f)
    pairs_dir = REPO_ROOT / "data" / "paired_hard"

    out = []
    skipped = 0
    print(f"[load] loading {len(index)} paired_hard pairs...")
    for i, (key, entry) in enumerate(index.items()):
        tgt_spk = entry["tgt_speaker"]
        if tgt_spk not in TARGET_SPEAKERS:
            skipped += 1
            continue
        voice_id = TARGET_SPEAKERS.index(tgt_spk)
        src_path = pairs_dir / entry["src_file"]
        tgt_path = pairs_dir / entry["tgt_file"]
        if not src_path.exists() or not tgt_path.exists():
            skipped += 1
            continue
        try:
            src_wav, sr_s = sf.read(str(src_path), always_2d=False)
            tgt_wav, sr_t = sf.read(str(tgt_path), always_2d=False)
            if sr_s != 24000:
                src_wav = librosa.resample(src_wav.astype(np.float32),
                                            orig_sr=sr_s, target_sr=24000)
            if sr_t != 24000:
                tgt_wav = librosa.resample(tgt_wav.astype(np.float32),
                                            orig_sr=sr_t, target_sr=24000)
            if src_wav.ndim > 1:
                src_wav = src_wav[:, 0]
            if tgt_wav.ndim > 1:
                tgt_wav = tgt_wav[:, 0]
            src_wav = src_wav.astype(np.float32)
            tgt_wav = tgt_wav.astype(np.float32)
        except Exception as e:
            print(f"  [skip] {key}: {e}", file=sys.stderr)
            skipped += 1
            continue
        if len(src_wav) < 12000 or len(tgt_wav) < 12000:
            skipped += 1
            continue
        out.append({
            "key": key, "voice_id": voice_id,
            "src_wav": src_wav, "tgt_wav": tgt_wav,
        })
        if (i + 1) % 20 == 0:
            print(f"  loaded {i+1}/{len(index)} ({len(out)} kept, {skipped} skipped)")
    print(f"[load] kept {len(out)}/{len(index)} pairs (skipped {skipped})")
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(pairs: list[dict], v1_infer, epochs: int = 5, lr: float = 1e-3,
          out_weights: Path = REPO_ROOT / "models" / "voicepack_v3.safetensors",
          out_log: Path = REPO_ROOT / "data" / "m1a_v3_train_log.json",
          device: str = "cpu") -> dict:
    torch.set_num_threads(2)
    n_voices = len(TARGET_SPEAKERS)
    model = build_model_v2(n_voices=n_voices).to(device)
    mel_extractor = DifferentiableMelSpec().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    log = {"epochs": [], "config": {
        "epochs": epochs, "lr": lr, "n_pairs": len(pairs),
        "n_voices": n_voices, "loss": "mel-spec MSE",
        "architecture": "V2 (FiLM@768) + audio-level mel-spec loss",
    }}

    print(f"\n[train V3] {len(pairs)} pairs × {epochs} epochs, lr={lr}")
    print(f"  loss = MSE(MelSpec(decode(FiLM(content_src))), MelSpec(tgt_audio))")
    t0 = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_mel_mse = 0.0
        np.random.shuffle(pairs)
        for p in pairs:
            voice_id = p["voice_id"]
            src_wav = p["src_wav"]
            tgt_wav = p["tgt_wav"]

            # Stage 1: encode source (frozen)
            with torch.no_grad():
                content, f0, energy = v1_infer.encode(src_wav)
            content_t = torch.from_numpy(content).to(device).float()
            f0_t = torch.from_numpy(f0).to(device).float()
            energy_t = torch.from_numpy(energy).to(device).float()

            # Stage 2: VoicePack FiLM (TRAINABLE)
            content_voiced = model(content_t, voice_id)  # [1, 768, T]

            # Stage 3: decode (PyTorch, differentiable)
            # Need to call V1Infer.decode with grad enabled
            with torch.enable_grad():
                # V1Infer.decode internally calls decoder.infer
                # which uses torch ops — should be differentiable
                out_audio = v1_infer.decode_t(content_voiced, f0_t, energy_t)
                # out_audio: numpy or tensor depending on path
                if isinstance(out_audio, np.ndarray):
                    out_audio_t = torch.from_numpy(out_audio).to(device).float()
                else:
                    out_audio_t = out_audio

            # Stage 4: mel-spec loss
            # Truncate to common length
            min_len = min(out_audio_t.shape[-1], len(tgt_wav))
            out_audio_t = out_audio_t[..., :min_len]
            tgt_t = torch.from_numpy(tgt_wav).to(device).float()[..., :min_len]

            # Add batch dim for MelSpec
            out_mel = mel_extractor(out_audio_t.unsqueeze(0) if out_audio_t.dim() == 1 else out_audio_t)
            tgt_mel = mel_extractor(tgt_t.unsqueeze(0) if tgt_t.dim() == 1 else tgt_t)
            # Match T
            T_mel = min(out_mel.shape[-1], tgt_mel.shape[-1])
            L = F.mse_loss(out_mel[..., :T_mel], tgt_mel[..., :T_mel])

            opt.zero_grad()
            L.backward()
            opt.step()
            epoch_loss += L.item()
            epoch_mel_mse += L.item()

        n = len(pairs)
        avg_loss = epoch_loss / n
        elapsed = time.perf_counter() - t0
        log["epochs"].append({
            "epoch": epoch, "loss_mel_mse": avg_loss, "elapsed_s": elapsed,
        })
        print(f"  epoch {epoch:3d}  mel_mse={avg_loss:.4f}  t={elapsed:.0f}s")
    log["total_seconds"] = time.perf_counter() - t0

    # Save weights
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.half() for k, v in model.state_dict().items()}
    from safetensors.torch import save_file
    save_file(state, str(out_weights))
    print(f"\n[train V3] saved → {out_weights} ({out_weights.stat().st_size} bytes)")
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[train V3] log → {out_log}")
    return log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("[m1a V3] loading V1Infer for content + decode...")
    import torchaudio  # noqa: F401 — needed by DifferentiableMelSpec
    from vc_realtime.infer_v1 import V1Infer
    v1_infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                       device="cpu", top_k=4, alpha=0.0)

    # Add a decode_t method that returns a tensor (differentiable path)
    # instead of numpy. The original V1Infer.decode returns numpy.
    def decode_t(self, content, f0, energy):
        """Differentiable decode path — returns torch tensor."""
        # Reproduce V1Infer.decode logic but keep tensors throughout
        # so gradient flows back to VoicePack.
        # Original: content → decoder.infer(content, f0, energy) → wav
        # decoder.infer is itself differentiable (PyTorch ops).
        device = content.device
        # decoder.infer expects content [B, 768, T], f0 [B, 1, T], energy [B, 1, L]
        # energy needs to match T_samples = T * 480
        T_frames = content.shape[-1]
        # Pad/truncate energy to match decoder expectation
        L_samples = T_frames * 480
        if energy.shape[-1] != L_samples:
            energy = F.interpolate(energy.unsqueeze(0).unsqueeze(0) if energy.dim() == 2 else energy,
                                    size=L_samples, mode='linear')
            if energy.dim() == 4:
                energy = energy.squeeze(0).squeeze(0)
        # Decode (no torch.inference_mode — we want gradients)
        wav = self.decoder.infer(content, f0, energy)
        return wav.squeeze(0) if wav.dim() == 3 else wav
    v1_infer.decode_t = decode_t.__get__(v1_infer, type(v1_infer))

    pairs = _load_paired_hard_pairs(v1_infer)
    if len(pairs) < 10:
        print(f"ERROR: only {len(pairs)} pairs loaded", file=sys.stderr)
        return 1

    train(pairs, v1_infer, epochs=args.epochs, lr=args.lr, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
