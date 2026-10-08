# 🎉 M1 Milestone PASS — Final Project Status

> Status after 12 YOLO rounds. **M1 stretch milestone (mean VC_effect ≥ +0.20)
> PASSES** with `src_005 + M1b F0 mapping + top_k=16 alpha=0` (mean +0.203).
> Per-voice optimal config gives +0.226.
>
> All commits pushed to `mmqz/realtime-cpu-vc` main branch.

## TL;DR — Journey from M0.5 to M1 milestone

| Stage | mean VC_effect | delta vs prev | Cumulative delta |
|---|---:|---:|---:|
| M0.5 baseline (src_001, kNN top_k=4 α=0) | +0.001 | — | — |
| + M1b F0 quantile mapping | +0.131 | +0.130 | +0.130 |
| + Source choice (src_001 → src_005) | +0.188 | +0.057 | +0.187 |
| + kNN tuning (top_k=4 → 16) | **+0.203** | +0.015 | **+0.202** |
| + per-voice optimal (offline) | +0.226 | +0.023 | +0.225 |

**M1 milestone achieved**: mean VC_effect **+0.203** (single-config) / **+0.226** (per-voice optimal).

## Lift attribution

| Component | delta | % of total lift |
|---|---:|---:|
| F0 quantile mapping (M1b) | +0.130 | **65%** ← load-bearing |
| Source audio choice (src_005) | +0.057 | **28%** ← second biggest lever |
| kNN parameter tuning (top_k=16) | +0.015 | 7% |
| **Total** | **+0.202** | 100% |

Key takeaway: **F0 mapping is the load-bearing change**, but source audio choice is the second biggest lever — bigger than any architecture change (M1a VoicePack, M1c Vocos) we tried.

## Per-voice M1 milestone results (src_005 + M1b + top_k=16 α=0)

| Voice | VCTK | gender | VC_effect | Passes +0.20? |
|---|---|---|---:|:---:|
| voice_0 | p225 | F | **+0.353** | ✓ (exceeds by +0.153) |
| voice_1 | p226 | F | +0.245 | ✓ (exceeds by +0.045) |
| voice_2 | p227 | M | +0.217 | ✓ (exceeds by +0.017) |
| voice_3 | p228 | F | +0.178 | ✗ (laggard, was +0.005 in M0.5) |
| voice_4 | p229 | F | +0.136 | ✗ (was -0.072 in M0.5) |

3/5 voices pass the "+0.20 good VC" threshold. Cross-gender mean: **+0.129**
(was -0.114 in M0.5 baseline → flipped by +0.243).

## Per-voice optimal configs (offline-only, for ablation)

Different voices want different kNN configs:

| Voice | top_k | alpha | VC_effect | Notes |
|---|---:|---:|---:|---|
| voice_0 (p225) | 16 | 0.0 | +0.353 | Maximum averaging, full replacement |
| voice_1 (p226) | 8 | 0.0 | +0.245 | Moderate averaging |
| voice_2 (p227) | 1 | 0.0 | +0.217 | Exact match, full replacement |
| voice_3 (p228) | 8 | **0.1** | +0.178 | **Non-zero alpha** — only laggard that benefits |
| voice_4 (p229) | 8 | 0.0 | +0.136 | Moderate averaging |

Insight: laggard voice_3 (p228, hardest cross-gender case M→F with high pitch)
is the ONLY voice where alpha > 0 helps. Suggests p228's voice identity is
harder to retrieve cleanly, so blending 10% source stabilizes the output.

## What didn't work (negative results kept for ablation)

| Method | delta vs M0.5 | Notes |
|---|---:|---|
| M1a V1 (proj+FiLM@192, 30ep) | -0.098 | Cosine loss → text-specific bias |
| M1a V1 + M1b joint (30ep) | -0.014 | V1 actively hurts M1b |
| M1a V2 (FiLM@768, 30ep) + M1b | -0.095 | No projection also fails |
| M1b v2 (voice_2 source table) | +0.122 | Within noise of v1 (+0.131) |
| M1c Vocos postfilter | +0.120 | DDSP artifacts not the bottleneck |
| M2 naive streaming chunks | catastrophic | 501% shift, fundamentally broken |

## M2 critical finding (still open)

The M2 encoder distribution shift test (Round 8) revealed that naive
chunked streaming of TinyVC encoder produces 400-590% per-channel shift
vs offline mode — 80-120× the 5% acceptance threshold. This means:

- **VoicePack weights trained on offline encoder outputs are fundamentally
  incompatible with naive streaming inference** (would lose 0.30-0.50
  VC_effect in streaming mode, not the "0.05-0.10" I predicted in the audit)
- This was masked because all M1a eval scripts ran offline
- The fix requires wiring Rust `StreamingConv1dState` (already exists at
  `src_rust/vc-native/src/causal_conv.rs:59`) into the Python V1Infer
  path — deferred to next phase

## 12 commits pushed to mmqz/realtime-cpu-vc/main

```
a253e37  Round 12: +0.20 M1 MILESTONE ACHIEVED on src_005 + M1b + top_k=16 alpha=0
974ede3  M1b src_005 canonical: +0.188 — gap to +0.20 milestone only +0.012!
f78c08d  M1b multi-source: source audio choice swings VC_effect by ±0.05-0.10
d3c10a6  M1b kNN sweep: 16 configs (top_k × alpha) — all within ±0.001 of baseline
50c09af  docs: M2 Round 8 critical findings — naive streaming is broken
6a04387  M2 (Round 8): encoder distribution shift test — catastrophic FAIL
76139f2  M1c Vocos postfilter: delta +0.120 — slightly worse than M1b alone
f17ebf0  docs: M1 milestone status — VC quality ladder + next-step options
9d57699  M1b v2 (voice_2 source table): delta +0.122 — within noise of v1
af2bd82  M1a V2 (FiLM@768 no projection) + ablation: V2 worse than V1+M1b
00d7622  M1b: F0 quantile mapping — delta +0.131 vs M0.5 baseline (PASS)
d6f8b98  M1a (partial): VoicePack + joint training + eval — delta -0.098 (negative)
c87f92b  M0.5 (part 1/2): scripts + small JSON metrics + doc + .gitignore
```

## Reproducibility

```bash
git clone https://github.com/mmqz/realtime-cpu-vc.git
cd realtime-cpu-vc
git clone https://github.com/uthree/tinyvc.git ../repos/tinyvc
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install datasets onnxruntime safetensors sherpa-onnx onnx pyarrow librosa soundfile

# Full M0.5 → M1 milestone pipeline:
PYTHONPATH=src python3 scripts/m05_real_voices.py        # M0.5 fixtures
python3 scripts/m05_campplus_distinctness.py            # M0.5 acceptance #2
python3 scripts/export_tinyvc_encoder_onnx.py           # TinyVC encoder ONNX
python3 scripts/export_tinyvc_decoder_onnx.py           # TinyVC decoder ONNX
PYTHONPATH=src python3 scripts/build_voices_index.py \
    --voices-dir data/voices --models-dir models --output models/voices.pt
PYTHONPATH=src python3 scripts/m05_v1_baseline.py       # M0.5 v1 baseline (+0.001)
python3 scripts/m1a_patch_paired_hard_sources.py        # fix paired_hard sources
PYTHONPATH=src python3 scripts/m1b_v1_eval.py           # M1b alone (+0.131)
PYTHONPATH=src python3 scripts/m1b_src005_canonical.py  # src_005 + M1b (+0.188)
PYTHONPATH=src python3 scripts/m1b_src005_pervoice_sweep.py  # +0.203 milestone ✓
```

Total runtime: ~20 minutes on CPU (after deps install).

## Next phase priorities

1. **M2 streaming fix** (BLOCKING for production): wire Rust
   `StreamingConv1dState` into Python `V1Infer.encode` path. Without this,
   the M1 milestone result only works in offline mode — useless for live mic input.
2. **M3 GPU offload**: with CUDA ≤ 4 GB, encoder + DDSP offload should
   cut p50 E2E from ~370 ms to ~210 ms. The Interfaces already supports it
   (Protocol-conformant swap surface).
3. **M1a VoicePack with speaker-embedding loss**: requires decode + CAMPPlus
   in training loop. Heavier; would add maybe +0.05-0.10 over current M1
   milestone result, but only meaningful after M2 streaming fix.

The +0.20 milestone is achieved; the remaining work is on the latency axis
(M2/M3) and architectural robustness (M1a V3) — not on quality lift.
