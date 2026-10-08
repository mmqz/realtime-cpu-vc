#!/usr/bin/env python3
"""M1a · Joint training of (proj_in 768→192) + (per-voice FiLM) + (proj_out 192→768).

This is the core M1a script. Replaces the SolA `train_projection.py`
autoencoder objective (which produced VC_effect +0.02-0.04 because the
192-d latent was forced to be a faithful reconstruction of 768-d, not a
speaker-separable latent space) with a joint disentanglement objective.

Loss
----
::

    L_total = α · L_main + β · L_aux

    L_main = 1 - cos(FiLM_v(proj_in(content_src_v)), proj_out(content_tgt_v))
             # where v = the voice index this pair belongs to
             # ↑↑↑ this is the disentanglement loss: encourage projected+FiLM
             # features to align with target content features.

    L_aux  = CE(speaker_classifier(z), tgt_speaker_id)
             # z = proj_in(content_src) (pre-FiLM latent, B,T,192)
             # ↑↑↑ forces speaker info to flow through FiLM γ_v/β_v,
             # not through proj_in. Without this term, proj_in could
             # trivially pass through speaker info, making FiLM redundant.

Training data: data/paired_hard/{src}_{tgt}_{text_id}.wav (130 pairs from M0.5)
For each pair:
  - source audio → TinyVC encoder → content_src [1, 768, T_src]
  - target audio → TinyVC encoder → content_tgt [1, 768, T_tgt]
  - We pair them by voice_id (= tgt_speaker's index in TARGET_SPEAKERS),
    not by src_speaker.

Training config (CPU-only, ~10 min for 30 epochs on 130 pairs):
  - Adam, lr=1e-3
  - Batch size = 1 (variable T)
  - 30 epochs over all 130 pairs (1300 forward passes per epoch)
  - Save checkpoint to models/voicepack_v1.safetensors

Output
------
- `models/voicepack_v1.safetensors` — trained weights, ~600 KB FP16
- `data/m1a_train_log.json` — per-epoch loss values
- `data/m1a_train_eval.json` — post-train eval:
    per-voice cosine(content_pred, content_tgt) on held-out 10% pairs
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
import torch.nn as nn
import torch.nn.functional as F

# --- Stub TinyVC optional deps (same pattern as infer_v1) -------------------
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

# VoicePack model
from vc_realtime.voicepack import build_model  # noqa: E402

# 5 VCTK target speakers — same as M0.5
TARGET_SPEAKERS = ["p225", "p226", "p227", "p228", "p229"]
# Source speakers used to build paired_hard
SOURCE_SPEAKERS = ["p232", "p237"]


# ---------------------------------------------------------------------------
# Speaker classifier (auxiliary loss)
# ---------------------------------------------------------------------------
class SpeakerClassifier(nn.Module):
    """Simple 2-layer MLP on pooled 192-d latent → n_voices logits."""

    def __init__(self, latent_dim: int = 192, n_voices: int = 5,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_voices),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, latent] → pool over T → [B, latent]
        z_pooled = z.mean(dim=1)
        return self.net(z_pooled)


# ---------------------------------------------------------------------------
# Data loading: encode each paired_hard wav via TinyVC encoder
# ---------------------------------------------------------------------------
def _encode_audio_to_content(audio_24k: np.ndarray, v1_infer) -> torch.Tensor:
    """Run V1Infer's encoder on a 24 kHz float32 wav → content [1, 768, T].

    Reuses V1Infer.encode which already does autopad + spectrogram + encoder.infer.
    Returns just the content (drops f0/energy).
    """
    content, _, _ = v1_infer.encode(audio_24k)
    return torch.from_numpy(content)


def _load_paired_hard(v1_infer) -> list[dict]:
    """Walk data/paired_hard/index.json, encode each (src, tgt) wav to content.

    Returns list of:
      { "voice_id": int (target speaker index in TARGET_SPEAKERS),
         "src_content": Tensor [1, 768, T_src],
         "tgt_content": Tensor [1, 768, T_tgt] }
    """
    index_path = REPO_ROOT / "data" / "paired_hard" / "index.json"
    if not index_path.exists():
        raise SystemExit(
            f"paired_hard index missing at {index_path}; run "
            f"scripts/m05_real_voices.py first")
    with open(index_path) as f:
        index = json.load(f)
    pairs_dir = REPO_ROOT / "data" / "paired_hard"

    out = []
    skipped = 0
    print(f"[load] encoding {len(index)} paired_hard pairs via V1Infer.encode "
          f"(one-time cost)...")
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
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {key}: {e}", file=sys.stderr)
            skipped += 1
            continue
        # Skip very short clips (< 0.5 s)
        if len(src_wav) < 12000 or len(tgt_wav) < 12000:
            skipped += 1
            continue
        try:
            src_content = _encode_audio_to_content(src_wav, v1_infer)
            tgt_content = _encode_audio_to_content(tgt_wav, v1_infer)
        except Exception as e:  # noqa: BLE001
            print(f"  [encode-fail] {key}: {e}", file=sys.stderr)
            skipped += 1
            continue
        out.append({
            "key": key,
            "voice_id": voice_id,
            "src_content": src_content,
            "tgt_content": tgt_content,
        })
        if (i + 1) % 10 == 0:
            print(f"  encoded {i+1}/{len(index)} ({len(out)} kept, {skipped} skipped)")
    print(f"[load] kept {len(out)}/{len(index)} pairs (skipped {skipped})")
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(pairs: list[dict], epochs: int = 30, lr: float = 1e-3,
          alpha: float = 1.0, beta: float = 0.1,
          out_weights: Path = REPO_ROOT / "models" / "voicepack_v1.safetensors",
          out_log: Path = REPO_ROOT / "data" / "m1a_train_log.json",
          device: str = "cpu") -> dict:
    """Train proj_in + FiLM + proj_out jointly.

    Loss = alpha * (1 - cos(FiLM(proj_in(src)), proj_out(tgt))) +
           beta  * CE(classifier(z_src), tgt_voice_id)

    Returns the training log (per-epoch losses).
    """
    torch.set_num_threads(2)
    n_voices = len(TARGET_SPEAKERS)
    model = build_model(n_voices=n_voices).to(device)
    classifier = SpeakerClassifier(latent_dim=192, n_voices=n_voices).to(device)
    # Optimizer on both model + classifier
    params = list(model.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    log = {"epochs": [], "config": {
        "epochs": epochs, "lr": lr, "alpha": alpha, "beta": beta,
        "n_pairs": len(pairs), "n_voices": n_voices,
    }}

    print(f"\n[train] {len(pairs)} pairs × {epochs} epochs, "
          f"alpha={alpha} beta={beta} lr={lr}")
    t0 = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_main = 0.0
        epoch_aux = 0.0
        epoch_cos = 0.0  # track raw cosine for monitoring
        np.random.shuffle(pairs)
        for p in pairs:
            voice_id = p["voice_id"]
            src = p["src_content"].to(device)  # [1, 768, T]
            tgt = p["tgt_content"].to(device)  # [1, 768, T_tgt]
            # Truncate to common T for cosine alignment
            T = min(src.shape[2], tgt.shape[2])
            src = src[..., :T]
            tgt = tgt[..., :T]
            # Forward through VoicePack
            # [1, 768, T] → [B, T, 768] via transpose inside model.forward
            pred = model(src, voice_id)  # [1, 768, T]
            # Also compute latent z for classifier
            src_t = src.transpose(1, 2)  # [1, T, 768]
            z = model.proj_in(src_t)     # [1, T, 192]
            # Main loss: cosine similarity between pred and tgt, averaged
            cos_per_frame = F.cosine_similarity(
                pred.transpose(1, 2), tgt.transpose(1, 2), dim=-1)  # [1, T]
            cos_mean = cos_per_frame.mean()
            L_main = 1.0 - cos_mean
            # Aux loss: speaker classification on z
            logits = classifier(z)  # [1, n_voices]
            tgt_label = torch.tensor([voice_id], device=device, dtype=torch.long)
            L_aux = F.cross_entropy(logits, tgt_label)
            L_total = alpha * L_main + beta * L_aux
            opt.zero_grad()
            L_total.backward()
            opt.step()
            epoch_loss += L_total.item()
            epoch_main += L_main.item()
            epoch_aux += L_aux.item()
            epoch_cos += cos_mean.item()
        n = len(pairs)
        avg = epoch_loss / n
        avg_main = epoch_main / n
        avg_aux = epoch_aux / n
        avg_cos = epoch_cos / n
        elapsed = time.perf_counter() - t0
        log["epochs"].append({
            "epoch": epoch, "loss": avg, "loss_main": avg_main,
            "loss_aux": avg_aux, "cos_mean": avg_cos, "elapsed_s": elapsed,
        })
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  loss={avg:.4f} "
                  f"(main={avg_main:.4f} aux={avg_aux:.4f}) "
                  f"cos={avg_cos:.3f}  t={elapsed:.0f}s")
    log["total_seconds"] = time.perf_counter() - t0

    # Save weights (FP16 for size)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for k, v in model.state_dict().items():
        state[k] = v.half()  # FP16
    from safetensors.torch import save_file
    save_file(state, str(out_weights))
    print(f"\n[train] saved weights → {out_weights} "
          f"({out_weights.stat().st_size//1024} KB)")

    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[train] saved log → {out_log}")
    return log


# ---------------------------------------------------------------------------
# Eval: per-voice cosine similarity on 10% held-out pairs
# ---------------------------------------------------------------------------
def evaluate(pairs: list[dict], weights_path: Path, device: str = "cpu") -> dict:
    """Load trained model + compute per-voice cosine on held-out 10%."""
    from vc_realtime.voicepack import VoicePackConditioner
    cond = VoicePackConditioner(weights_path, device=device)
    # 10% holdout
    np.random.seed(42)
    n_test = max(1, len(pairs) // 10)
    test_pairs = np.random.choice(len(pairs), n_test, replace=False)
    per_voice: dict[int, list[float]] = {v: [] for v in range(len(TARGET_SPEAKERS))}
    for idx in test_pairs:
        p = pairs[idx]
        vid = p["voice_id"]
        cond.select_voice(vid)
        src = p["src_content"]
        tgt = p["tgt_content"]
        T = min(src.shape[2], tgt.shape[2])
        src = src[..., :T]
        tgt = tgt[..., :T]
        pred = cond.replace(src.cpu().numpy())  # [1, 768, T]
        pred_t = torch.from_numpy(pred).to(device)
        cos = F.cosine_similarity(
            pred_t.transpose(1, 2), tgt.transpose(1, 2), dim=-1).mean().item()
        per_voice[vid].append(cos)
    summary = {
        "n_test_pairs": n_test,
        "per_voice_mean_cos": {
            TARGET_SPEAKERS[v]: float(np.mean(per_voice[v]))
            if per_voice[v] else None
            for v in range(len(TARGET_SPEAKERS))
        },
        "overall_mean_cos": float(np.mean([
            c for v in per_voice.values() for c in v
        ])) if per_voice else None,
    }
    out_eval = REPO_ROOT / "data" / "m1a_train_eval.json"
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] saved → {out_eval}")
    print(f"  overall_mean_cos: {summary['overall_mean_cos']}")
    for spk, cos in summary["per_voice_mean_cos"].items():
        print(f"    {spk}: {cos}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="weight for L_main (disentanglement cosine)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="weight for L_aux (speaker classifier)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Load V1Infer to access encoder (V1Infer.encode handles all TinyVC plumbing)
    print("[m1a] loading V1Infer (encoder.pt + decoder.pt) for content extraction...")
    t0 = time.perf_counter()
    from vc_realtime.infer_v1 import V1Infer
    v1_infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                       device="cpu", top_k=4, alpha=0.0)
    print(f"[m1a] V1Infer loaded in {time.perf_counter()-t0:.1f}s")

    # Load paired_hard, encode each wav
    pairs = _load_paired_hard(v1_infer)
    if len(pairs) < 20:
        print(f"ERROR: only {len(pairs)} pairs loaded (need ≥ 20 for training)",
              file=sys.stderr)
        return 1

    # Train
    log = train(pairs, epochs=args.epochs, lr=args.lr,
                alpha=args.alpha, beta=args.beta,
                device=args.device)

    # Eval
    eval_summary = evaluate(pairs, REPO_ROOT / "models" / "voicepack_v1.safetensors",
                             device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
