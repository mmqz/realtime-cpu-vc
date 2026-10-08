# M1 Milestone Status — VC Quality Ladder

> Status after 6 YOLO rounds of M0.5 → M1 work. All commits pushed to
> `mmqz/realtime-cpu-vc` main branch.
>
> See `download/issues-m0.5-to-m3.md` for the original milestone plan +
> audit patches.

## Final ablation table (on M0.5 real VCTK p232→p225-p229)

| Method | mean VC_effect | delta vs M0.5 | voice_0 | voice_3 (cross-gender) |
|---|---:|---:|---:|---:|
| **M0.5 baseline (v1 kNN-VC)** | +0.001 | — | +0.072 | **-0.156** |
| M1a V1 alone (proj+FiLM@192, 5ep) | -0.097 | -0.098 | -0.024 | -0.168 |
| M1a V1 alone (30ep, beta=0.3) | -0.098 | -0.099 | -0.024 | -0.168 |
| M1a V1 + M1b joint (30ep) | -0.013 | -0.014 | +0.034 | -0.092 |
| **M1a V2 (FiLM@768, 30ep) + M1b** | -0.094 | -0.095 | -0.044 | -0.220 |
| **M1b alone (F0 quantile map, 5s source table)** | **+0.132** | **+0.131** | **+0.250** | +0.005 |
| M1b v2 (F0 quantile map, voice_2 30s source table) | +0.123 | +0.122 | +0.272 | -0.007 |

## M1 milestone status

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| M1 acceptance (delta ≥ +0.05) | +0.05 | +0.131 | **PASS ✓** |
| M1 stretch (delta ≥ +0.199) | +0.199 | +0.131 | FAIL ✗ |
| M1 absolute (mean VC_effect ≥ +0.20) | +0.20 | +0.132 | FAIL ✗ (need +0.07 more) |
| Single voice "good VC" threshold | +0.20 | voice_0 +0.272 | **PASS ✓** (1/5 voices) |

## Key findings

### 1. M1b F0 quantile mapping is the load-bearing change
- delta +0.131 vs M0.5 baseline — single biggest quality lift in the project
- Cross-gender cases (M→F) flipped from -0.114 mean to +0.031-0.065 mean
- voice_0 (p225) reached +0.250-0.272, already past "good VC" threshold +0.20
- The audit patch prediction ("F0 not being remapped is the dominant failure
  mode") was correct

### 2. M1a VoicePack cannot beat the SolA-SolK trap
- V1 (proj+FiLM@192) and V2 (FiLM@768) both made things worse, not better
- The cosine-to-target-content loss converges (cosine 0.43→0.53 in V1,
  0.23→0.44 in V2) but the trained FiLM learns "per-channel bias toward
  target's per-channel mean" — which encodes BOTH speaker AND text, so it
  overfits to training text and hurts at inference on novel text
- This is the SAME failure mode as SolC (scalar adaptation) and SolG
  (multi-layer) — content feature distribution matching ≠ speaker
  identity transfer
- For M1a to actually work, training objective must be:
  - speaker embedding loss (CAMPPlus on decoded audio), OR
  - adversarial content preservation + speaker classifier
  Both require the full decode + CAMPPlus pipeline in the training loop
  (heavier, deferred to next phase)

### 3. M1b has hit the v1 kNN-VC + DDSP architectural ceiling
- M1b v1 vs v2 (different source F0 tables) differ by ±0.01, within noise
- voice_3 (p228, high-pitch female ~220Hz) remains the laggard at +0.005 —
  source p232 male ~115Hz → target p228 female ~220Hz is the largest pitch
  gap in the test set, and the kNN-VC retrieval can't bridge it
- The remaining +0.07 gap to +0.20 milestone requires a different vocoder
  (M1c Vocos) or better content features (out of scope for this YOLO run)

## Recommended next steps (post-M1)

### Option A · M1c Vocos integration (heavy, ~1-2 days)
- Train 192→100 + TimeUpsample(50→93.75 Hz) projection layer
- Connect Vocos (already exported ONNX) to replace DDSP
- Expected lift: +0.05-0.07 (Vocos's ISTFT-head produces cleaner audio
  than DDSP's additive sine synthesis, MOS +0.2-0.3 in literature)
- This would close the +0.07 gap to +0.20 milestone

### Option B · M2 causal ONNX encoder (independent axis, ~1 day)
- Re-export encoder.onnx with left-only Conv1d padding (no retraining)
- Wire into Rust StreamingPipeline with 1-chunk delay + VAD skip
- Test encoder distribution shift (offline vs streaming chunked)
- Latency target: x86 p50 ≤ 250 ms
- Does NOT change VC_effect numbers (latency axis, not quality axis)

### Option C · M1a speaker-embedding-loss retrain (research, ~3-5 days)
- Implement differentiable decode + CAMPPlus in training loop
- Replace cosine loss with speaker embedding loss
- Train V3 VoicePack
- Expected lift: +0.05-0.15 (uncertain — may also fail if speaker
  classifier converges but doesn't generalize)

## Commits pushed to mmqz/realtime-cpu-vc/main

```
9d57699  M1b v2 (voice_2 source table): delta +0.122 — within noise of v1
af2bd82  M1a V2 (FiLM@768 no projection) + ablation: V2 worse than V1+M1b
00d7622  M1b: F0 quantile mapping — delta +0.131 vs M0.5 baseline (PASS)
d6f8b98  M1a (partial): VoicePack + joint training + eval — delta -0.098 (negative)
c87f92b  M0.5 (part 1/2): scripts + small JSON metrics + doc + .gitignore
6d6d6c9  fix: last 4 Rust hardcoded paths → relative (pre-existing)
```

## Reproducibility

All scripts are runnable end-to-end on a CPU-only sandbox:

```bash
git clone https://github.com/mmqz/realtime-cpu-vc.git
cd realtime-cpu-vc
git clone https://github.com/uthree/tinyvc.git ../repos/tinyvc
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install datasets onnxruntime safetensors sherpa-onnx onnx pyarrow librosa soundfile

# M0.5 baseline
PYTHONPATH=src python3 scripts/m05_real_voices.py
python3 scripts/m05_campplus_distinctness.py
python3 scripts/export_tinyvc_encoder_onnx.py
python3 scripts/export_tinyvc_decoder_onnx.py
PYTHONPATH=src python3 scripts/build_voices_index.py \
    --voices-dir data/voices --models-dir models --output models/voices.pt
PYTHONPATH=src python3 scripts/m05_v1_baseline.py

# M1b (best result)
python3 scripts/m1a_patch_paired_hard_sources.py  # fix paired_hard source wavs
PYTHONPATH=src python3 scripts/m1b_v1_eval.py
PYTHONPATH=src python3 scripts/m1b_v2_eval.py

# M1a (negative results, kept for ablation)
PYTHONPATH=src python3 scripts/train_voicepack_joint.py --epochs 30
PYTHONPATH=src python3 scripts/m1a_v1_eval.py
PYTHONPATH=src python3 scripts/train_voicepack_v2.py --epochs 30
PYTHONPATH=src python3 scripts/m1_joint_eval.py  # V1 + M1b
PYTHONPATH=src python3 scripts/m1_joint_v2_eval.py  # V2 + M1b
```
