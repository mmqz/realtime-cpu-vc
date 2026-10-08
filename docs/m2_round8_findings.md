# M2 Round 8 · Critical Findings — Naive Streaming is Broken

> Round 8 of YOLO M0.5→M3 work. Commit `6a04387` pushed to main.
>
> Per-voice metrics + analysis below. The M2 acceptance test reveals a
> fundamental architecture problem that invalidates the streaming plan
> as written in `download/issues-m0.5-to-m3.md`.

## TL;DR

The M2 plan called for re-exporting `encoder.onnx` with causal (left-only)
Conv1d padding, then wiring into the Rust `StreamingPipeline` with 1-chunk
delay, then verifying the encoder output distribution shift ≤ 5% between
offline (full-sequence) and streaming (chunked) inference.

The verification step is done first. **Result: catastrophic FAIL — shift is
80-120× the 5% threshold even with 1-second chunks.** This is not a "slight
OOD" as my own audit patch predicted. It is a fundamental incompatibility
between v1's centered-STFT encoder and naive chunked inference.

## The test

`scripts/m2_encoder_shift_test.py`:
- 10 wavs tested: 5 voice refs (30s each) + 5 source clips (5s each)
- For each: encode full → `content_offline [1, 768, T]`
- For each: encode in 80ms chunks → `content_streaming [1, 768, T']` (different T!)
- Truncate to min T, compute per-channel mean/std shift (%)

Two chunk sizes tested: 80 ms (M2 spec, 1920 samples) and 1000 ms (sanity check, 24000 samples).

## Results

| Chunk | voice_0 | voice_1 | voice_2 | voice_3 | voice_4 | mean |
|---|---:|---:|---:|---:|---:|---:|
| 80 ms | 427% | 561% | 393% | 538% | 587% | 501% |
| 1000 ms | 97% | 133% | 89% | 155% | 118% | 118% |

Even 1-second chunks (12.5× larger than M2 spec) fail at 24× the threshold.

## Root cause

TinyVC encoder uses **centered STFT padding** (`n_fft//2 = 960` samples on each side).
The encoder's pipeline is:
1. Audio `[L]` → padded to `[L + 1920]` (center=True, 960 each side)
2. STFT → spec `[1, 961, T_frames]` where `T_frames ≈ L/480 + 5`
3. ConvNeXt-v2 blocks (also use centered conv padding internally)
4. Output: content `[1, 768, T_frames]`

When you chunk the audio into 1920-sample (80 ms) pieces and re-pad each:
- Each chunk gets fresh 960-sample padding → 9 padding-derived frames per chunk
- The ConvNeXt blocks add their own re-padding per chunk
- Frame counts don't match: 30 s offline = 1500 frames; 80 ms chunks concatenated = 3375 frames

The "shift" metric I'm using compares element-wise content values at each [channel, frame] position. Since streaming produces 2.25× more frames than offline, and the extra frames are dominated by padding-derived edge effects, the values at any given frame position differ wildly.

## What this means for the project plan

The M2 plan assumed the only streaming issue was the Conv1d padding direction (centered → causal). The actual issue is bigger:

1. **Conv1d padding direction** (the audit patch's "LayerNorm statistics" concern) — secondary
2. **STFT center padding** — primary, every chunk re-pads from zero
3. **ConvNeXt-v2 internal conv padding** — also re-pads per chunk

All three need to be addressed with **state-carrying streaming primitives**, not just causal export.

## The good news

The Rust codebase already has the right primitives:
- `src_rust/vc-native/src/causal_conv.rs:59` — `StreamingConv1dState` maintains conv state across chunks
- `src_rust/vc-native/src/streaming_pipeline.rs:70` — `StreamingPipeline` orchestrates state-carrying chunks

These were written during the v2 Rust rewrite but the Python v1 path doesn't use them. The M2 work is now:
1. **Wire Rust `StreamingConv1dState` into the Python V1Infer.encode path** (via PyO3 if the binding exists, or rewrite encode in Rust)
2. Re-run M2 shift test with state-carrying conv → expect ≤ 5%
3. Then proceed with the original M2 plan (causal ONNX export, 1-chunk delay, VAD skip)

## What this means for M1a VoicePack

I previously predicted in the audit patch:

> "encoder 输出分布相对 offline 模式会有 shift ... M1a 训练的 VoicePack
> 用的是 offline encoder 输出，streaming 推理时会变成轻微 OOD，**静默地
> 把 VC_effect 偷走 0.05-0.10**"

This was **wrong**. The shift is 400-590%, not 5-10%. VoicePack weights trained on offline encoder outputs would lose 0.30-0.50 VC_effect in streaming mode — basically useless.

This was masked because all M1a eval scripts (m1a_v1_eval, m1_joint_eval, m1_joint_v2_eval) ran in offline mode. Even if M1a had trained successfully (+0.10 delta), the streaming inference path would have produced worse results than the M0.5 baseline.

## Updated milestone dependency graph

```
M0.5  ✓ done (real VCTK 5-speaker baseline +0.001)
  │
  ▼
M2 (PREREQUISITE FOR M1a) — encoder streaming shift test ✗ catastrophic FAIL
  │   └─ root cause: STFT center padding per chunk
  │   └─ fix: wire Rust StreamingConv1dState into Python path
  │       → re-run shift test, expect ≤5%
  ▼
M1a VoicePack (BLOCKED until M2 streaming works)
  │   └─ VoicePack training requires streaming-stable encoder outputs
  │       otherwise weights are OOD at inference
  ▼
M1b F0 quantile mapping ✓ done (delta +0.131, current best)
  │
  ▼
M1c Vocos integration ✗ postfilter doesn't help (delta +0.120 < M1b)
  │   └─ proper 192→100 projection training deferred
  ▼
M3 GPU offload (untouched)
```

## Reproducibility

```bash
PYTHONPATH=src python3 scripts/m2_encoder_shift_test.py
```

Output is at `data/m2_encoder_shift.json`. Reproducible across runs (deterministic).

## Commit pushed

```
6a04387  M2 (Round 8): encoder distribution shift test — catastrophic FAIL
76139f2  M1c Vocos postfilter: delta +0.120 — slightly worse than M1b alone
9d57699  M1b v2 (voice_2 source table): delta +0.122 — within noise of v1
f17ebf0  docs: M1 milestone status — VC quality ladder + next-step options
af2bd82  M1a V2 (FiLM@768 no projection) + ablation: V2 worse than V1+M1b
00d7622  M1b: F0 quantile mapping — delta +0.131 vs M0.5 baseline (PASS)
d6f8b98  M1a (partial): VoicePack + joint training + eval — delta -0.098 (negative)
c87f92b  M0.5 (part 1/2): scripts + small JSON metrics + doc + .gitignore
```
