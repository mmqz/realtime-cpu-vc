#!/usr/bin/env python3
"""M1a V2 · Train per-voice FiLM(768) directly (no projection bottleneck).

V1 used 768→192→768 + FiLM(192). The 75% channel bottleneck destroyed
content info, and the speaker classifier never learned (CE stuck at log(5)
= 1.61, random guessing on 5 voices — latent z didn't carry speaker info).
M1a+M1b joint delta was -0.014 (vs M1b alone +0.131) — V1 hurt.

V2 skips the projection, applies per-voice FiLM(768) directly. The
disentanglement auxiliary loss is removed (no latent to classify).
Only the main cosine loss is used.

Loss: L = 1 - cos(FiLM_v(content_src_v), content_tgt_v)
       (where v = voice index = tgt speaker)

Training:
  - Adam, lr=1e-3
  - 30 epochs over 80 paired_hard pairs
  - Saves models/voicepack_v2.safetensors (~15 KB FP16)
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
SOURCE_SPEAKERS = ["p232", "p237"]


def _encode_audio_to_content(audio_24k: np.ndarray, v1_infer) -> torch.Tensor:
    content, _, _ = v1_infer.encode(audio_24k)
    return torch.from_numpy(content)


def _load_paired_hard(v1_infer) -> list[dict]:
    """Same as train_voicepack_joint.py — encode each paired_hard pair."""
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
    print(f"[load] encoding {len(index)} paired_hard pairs...")
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
        try:
            src_content = _encode_audio_to_content(src_wav, v1_infer)
            tgt_content = _encode_audio_to_content(tgt_wav, v1_infer)
        except Exception as e:
            print(f"  [encode-fail] {key}: {e}", file=sys.stderr)
            skipped += 1
            continue
        out.append({
            "key": key, "voice_id": voice_id,
            "src_content": src_content, "tgt_content": tgt_content,
        })
        if (i + 1) % 10 == 0:
            print(f"  encoded {i+1}/{len(index)} ({len(out)} kept, {skipped} skipped)")
    print(f"[load] kept {len(out)}/{len(index)} pairs (skipped {skipped})")
    return out


def train(pairs: list[dict], epochs: int = 30, lr: float = 1e-3,
          out_weights: Path = REPO_ROOT / "models" / "voicepack_v2.safetensors",
          out_log: Path = REPO_ROOT / "data" / "m1a_v2_train_log.json",
          device: str = "cpu") -> dict:
    torch.set_num_threads(2)
    n_voices = len(TARGET_SPEAKERS)
    model = build_model_v2(n_voices=n_voices).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = {"epochs": [], "config": {
        "epochs": epochs, "lr": lr, "n_pairs": len(pairs), "n_voices": n_voices,
        "architecture": "FiLM(768) per voice, no projection",
    }}
    print(f"\n[train V2] {len(pairs)} pairs × {epochs} epochs, lr={lr}")
    t0 = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_cos = 0.0
        np.random.shuffle(pairs)
        for p in pairs:
            voice_id = p["voice_id"]
            src = p["src_content"].to(device)
            tgt = p["tgt_content"].to(device)
            T = min(src.shape[2], tgt.shape[2])
            src = src[..., :T]
            tgt = tgt[..., :T]
            pred = model(src, voice_id)
            cos_per_frame = F.cosine_similarity(
                pred.transpose(1, 2), tgt.transpose(1, 2), dim=-1)
            cos_mean = cos_per_frame.mean()
            L = 1.0 - cos_mean
            opt.zero_grad()
            L.backward()
            opt.step()
            epoch_loss += L.item()
            epoch_cos += cos_mean.item()
        n = len(pairs)
        avg = epoch_loss / n
        avg_cos = epoch_cos / n
        elapsed = time.perf_counter() - t0
        log["epochs"].append({
            "epoch": epoch, "loss": avg, "cos_mean": avg_cos,
            "elapsed_s": elapsed,
        })
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  loss={avg:.4f} cos={avg_cos:.3f}  t={elapsed:.0f}s")
    log["total_seconds"] = time.perf_counter() - t0

    out_weights.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.half() for k, v in model.state_dict().items()}
    from safetensors.torch import save_file
    save_file(state, str(out_weights))
    print(f"\n[train V2] saved → {out_weights} ({out_weights.stat().st_size} bytes)")
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[train V2] log → {out_log}")
    return log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("[m1a V2] loading V1Infer for content extraction...")
    t0 = time.perf_counter()
    from vc_realtime.infer_v1 import V1Infer
    v1_infer = V1Infer(models_dir=str(REPO_ROOT / "models"),
                       device="cpu", top_k=4, alpha=0.0)
    print(f"[m1a V2] V1Infer loaded in {time.perf_counter()-t0:.1f}s")

    pairs = _load_paired_hard(v1_infer)
    if len(pairs) < 20:
        print(f"ERROR: only {len(pairs)} pairs loaded", file=sys.stderr)
        return 1

    train(pairs, epochs=args.epochs, lr=args.lr, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
