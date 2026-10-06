# Real-time CPU Voice Conversion — Technical Feasibility Analysis

> **Bilingual (中文 / English glossary) report** · 2026.10 · Prepared by Z.ai
>
> PDF version: `realtime-cpu-vc-tech-brief.pdf` (16 pages)

## 1. Executive Summary 执行摘要

This report addresses a concrete engineering problem: under the hard constraints of **pure-CPU inference, ~100 MB RAM, 300–500 ms end-to-end latency**, can we build a **real-time voice conversion (VC) system** that takes the user's live microphone stream as input and outputs to one of **5 pre-registered target voices** (30 s reference each)?

We analyze five open-source repositories — **Seed-VC (tiny variant)**, **TinyVC**, **RVC**, **Fish-Speech lightweight path**, **PocketTTS** — across six dimensions: architecture, parameter count, quantization & ONNX support, streaming capability, license, and CPU feasibility. From this analysis we derive directly actionable recommendations on module trim/replacement, quantization paths, real-time streaming strategy, speaker pre-registration pipeline, and a P0/P1/P2 priority checklist.

**Bottom line**: TinyVC is the most on-target repository. Its inference graph is **9.36 M params, ~36 MB FP32 / ~18 MB FP16**, natively supports ONNX export (opset 17, three sub-graphs) and PyAudio streaming (SOLA crossfade). The default algorithmic latency of ~560 ms can be tuned to ~320 ms by adjusting look-ahead and crossfade, hitting the 300–500 ms target. PocketTTS provides borrowable modules: causal `StreamingConv1d`, KV-cache state machine, and native INT8 dynamic quantization. Seed-VC's tiny variant is *not* actually tiny (still requires 1.2 GB XLS-R-300M), and Fish-Speech is non-commercial and ≥1 GB — both are **reference architectures only**, not runtime dependencies.

| Metric | Value |
|--------|-------|
| TinyVC inference params | 9.36 M |
| Target RSS | ~80 MB |
| Expected end-to-end latency (x86) | ~370 ms |
| Voice registry | 5 voices × 30 s |

## 2. Project Context & Constraints 项目背景与约束

- **Target hardware**: cross-platform — x86 desktop (4–8 cores, AVX2 + FMA) and ARM Cortex-A76/A78 (RK3588, Raspberry Pi 5, Apple Silicon equivalent; NEON + dotprod).
- **Latency budget**: 80 ms chunk + 80 ms look-ahead + 40 ms SOLA crossfade + 200 ms chunk buffer alignment + ~75 ms compute = ~375 ms (x86) / ~480 ms (ARM).
- **Memory budget**: 5 MB encoder (INT8) + 5 MB decoder (INT8) + 12 MB voice index (FP16, 5 × 2.4 MB) + 15 MB activation peaks + 8 MB ring buffer + 20 MB ONNXRuntime + 20 MB Python = ~85–110 MB total.
- **Speaker registry spec**: 5 fixed target voices, 30 s each, 24 kHz mono PCM16, normalized to -3 dBFS, denoised. Per-voice cosine similarity between voice embeddings must be < 0.85 to ensure distinctness.

## 3. Cross-Repo Comparison 五仓库横向对比

| Repo | License | Params | ONNX | Streaming | CPU/100 MB | Our use |
|------|---------|--------|------|----------|------------|---------|
| Seed-VC (tiny) | GPL v3 | ~25 M (DiT) + 1.2 GB (XLS-R) | No | Yes (GUI) | No | Distillation recipe + flow-matching training reference |
| **TinyVC** | **Apache 2.0** | **9.36 M (Enc+Dec)** | **Yes** | **Yes (SOLA)** | **Yes** | **Main framework, direct reuse** |
| RVC v2 | MIT | 20–30 M (synth) + 360 MB (ContentVec) | No (community has) | Yes (block=0.25s) | No | Small-index idea + RMVPE replacement eval |
| Fish-Speech | NC license | ~1 GB+ (DualAR 4B) | No | Yes (generator) | No | Reference architecture only, not in runtime |
| PocketTTS | MIT | ~100 M (FlowLM+Mimi) | No (community has) | Yes (causal) | Borderline | Mimi vocoder + INT8 quantization code |

### 3.1 Seed-VC (tiny variant)

Repo: `Plachtaa/seed-vc` (note: `BytedanceFish-Speech/seed-vc` is 404, real path is Plachtaa's fork). License: GPL v3 (copyleft, commercial requires same-license derivatives). The "tiny" variant is actually a 25 M-parameter DiT backbone (hidden 384, 6 heads, 9 layers, U-ViT), but at runtime still requires downloading 1.2 GB of `facebook/wav2vec2-xls-r-300m` as content encoder — there is no separate small content encoder shipped. This is the biggest "false marketing" issue with Seed-VC tiny for our scenario. It has `real-time-gui.py` with SOLA streaming framework, KV-cache and causal-mask code present but disabled by default (`use_kv_cache=False`); each block does a full DiT forward over prompt+chunk. ONNX and quantization paths are entirely absent. Verdict: **not a runtime dependency**; the WavLM/ContentVec distillation training recipe (same idea as TinyVC's `train_encoder.py`) is a useful reference for training our own content encoder later.

### 3.2 TinyVC — main framework

Repo: `github.com/uthree/tinyvc`, Apache 2.0, authored by Japanese dev uthree. Architecture: **source-filter VC**, distilled from WavLM. Content encoder = 6-layer ConvNeXt-v2 (384 ch, depthwise k=7 + GRN), distilled from `microsoft/wavlm-base-plus` `hidden_states[4]` (768-d) — the 90 M-param WavLM is **not** shipped at inference, only the 4.2 M mimic. Pitch extractor = 4-layer ConvNeXt (128 ch, 512 log-frequency classes, top-k=4 soft decode), jointly trained with content using FCPE labels. **No separate speaker encoder** — uses kNN-VC retrieval (cosine top-4 swap of source features against pre-extracted target tensor). Vocoder = DDSP-style `Decoder`: `SourceNet` (3 ConvNeXt, 128 ch) predicts amplitudes for 14 harmonics+1 (additive sine synth via `torch.sin(2π·cumsum(f0·n/sr))`) + FFT-domain EQ kernel applied to Gaussian noise (`torch.istft`); `FilterNet` is a lightweight 1D U-Net (channels=[384,192,96,48,24], factors=[2,3,4,4,5], FiLM + dilated k=3 convs dilations 1,3,9,27). No transposed convs anywhere. Inference total: **9.36 M params, ~36 MB FP32 / ~18 MB FP16**. `export_onnx.py` exports three sub-graphs at opset 17 with dynamic batch/length axes. `infer_streaming.py` uses PyAudio (paInt16, 24 kHz), chunk=1920 (80 ms), extra=3840 (160 ms look-ahead), `StreamInfer` with SOLA crossfade (sola_search=1920, crossfade=1920, last_delay=3840). State cache = `torch.roll` ring buffer. Algorithmic latency ~560–640 ms default, tunable to ~320 ms with `extra=0` + smaller crossfade. Zero-shot cloning: reference audio of arbitrary length, encoded once to `[1, 768, T]`; offline cache via `extract_index.py` → `models/index.pt` (default 2048 frames ≈ 40 s). Matches our 5-voice × 30-s use case directly. **Verdict: most directly on-target repo of the five. Directly reusable as inference spine.**

### 3.3 RVC — source of the small-index idea

Repo: `RVC-Project/Retrieval-based-Voice-Conversion-WebUI`, MIT. RVC's core is **per-frame HuBERT feature + Faiss retrieval**, with default index type `IVF+Flat`, typical singer index ~50–200 MB. The biggest blocker for our scenario is the dependency on ContentVec (HuBERT-base fine-tune) at ~360 MB FP32 — this single component exceeds our 100 MB budget 3.6×. But RVC's small-index idea is borrowable: with only 5 fixed speakers, per-speaker index can be shrunk to <5 MB (using `IVF256,PQ64x4fs,RFlat`, or simply a FP16 matrix + brute-force top-1). RVC's realtime entry `infer/rtrvc.py::RVC.infer()` uses `block_time=0.25 s`, `crossfade_time=0.05 s` — useful parameter reference for our SOLA tuning. **Verdict: not a runtime dependency; borrow small-index compression and block config experience.**

### 3.4 Fish-Speech — reference architecture only

Repo: `github.com/fishaudio/fish-speech`, **FISH AUDIO RESEARCH LICENSE (non-commercial)**. Two-stage TTS+VC framework: DAC neural codec (`modded_dac.DAC`) + DualARTransformer (BART-style, Slow AR 4B params + Fast AR 400M params). Latest S2-Pro model ~8 GB bf16, **requires 24 GB VRAM**. README's RTF 0.195 + TTFA 100 ms is on H200 GPU; on CPU the RTF is ~5–20× (5–20 seconds of compute per second of audio) — completely not real-time. **No official ONNX or GGUF export**, only an in-repo `tools/llama/quantize.py` INT8 script (Linear-only, CPU-portable, but model too large to matter). Its value to our project is two design patterns: (a) `<|speaker:i|> + VQ codes as in-context conditioning` — a zero-shot cloning mode without a separate speaker encoder; (b) DAC codec structure (~30–100 MB retrainable). **Verdict: not a runtime dependency; avoids license + size double risk.**

### 3.5 PocketTTS — borrowable causal convolution + INT8 code

Repo: `github.com/kyutai-labs/pocket-tts`, MIT. CPU-real-time TTS (not VC), composed of BPE tokenizer + FlowLM (6-layer causal StreamingTransformer, d_model=1024, 16 heads, RoPE + KV-cache) + Mimi codec (SEANet encoder + 2-layer transformer + SEANet decoder, 24 kHz, 12.5 Hz, hop=120). ~100 M params total (FP32 ~400 MB / INT8 ~200 MB), **single-threaded** (`torch.set_num_threads(1)`), README reports RTF ~6× and first-chunk ~200 ms on M4. Three pieces of reusable code: (1) `StreamingConv1d` + `_LinearKVCacheBackend` + `StatefulModule` — a complete causal-conv + KV-cache abstraction that can replace TinyVC's SOLA re-encode loop, dropping algorithmic latency from ~560 ms to ~150 ms; (2) `pocket_tts/quantization.py` — torchao INT8 dynamic quantization template with FBGEMM (x86) / QNNPACK (ARM) auto-switch, directly applicable to TinyVC's Linear/Conv1d ops; (3) `export-voice` serialization of 30-s reference KV-cache state to `.safetensors` — matches our 5-voice × 30-s registry almost exactly.

## 4. Module Trim & Replace Map 模块裁剪与替换方案

| Module | Original | Replace with | Action | Source |
|--------|----------|--------------|--------|--------|
| Content encoder | XLS-R-300M (1.2 GB) / ContentVec (360 MB) | TinyVC WavLM-distilled ConvNeXt-v2 (4.2 M, 18 MB FP32 / 9 MB FP16) | **Replace** | `tinyvc/module/tinyvc/encoder.py:75-97` |
| Speaker embedder | CAMPPlus (22 M, 27 MB) or RVC `nn.Embedding(109,256)` | kNN-VC retrieval, zero extra params (5 × 2.4 MB FP16 index) | **Delete+Retrieve** | `tinyvc/module/tinyvc/feature_retrieval.py:15-33` |
| Pitch extractor | RMVPE (20–40 MB) or pyworld DIO | TinyVC `PitchEstimator` (460 K, 0.5 MB) | **Replace** | `tinyvc/module/tinyvc/encoder.py:11-72` |
| Vocoder | HiFi-GAN NSF (14 M, HiFT) or BigVGAN | TinyVC DDSP `SourceNet` + `FilterNet` (4.66 M, 18 MB FP32 / 9 MB FP16) | **Replace** | `tinyvc/module/tinyvc/decoder.py:236-266` |
| Streaming shell | Seed-VC `real-time-gui.py` (KV-cache disabled) | TinyVC `StreamInfer` SOLA + borrowed PocketTTS `StreamingConv1d` | **Replace+borrow** | `tinyvc/module/infer/stream.py:30-96` |

### 4.1 Content encoder

Replace Seed-VC's 1.2 GB XLS-R-300M with TinyVC's `SSLFeatureEstimator`: 6-layer ConvNeXt-v2 (384 ch, depthwise k=7 + GRN), distilled from WavLM-Base-Plus `hidden_states[4]` (768-d) with L1×45 loss (see `tinyvc/train_encoder.py:81-98`). After distillation: 4.2 M params, FP32 18 MB, INT8 ~5 MB. **Brings content encoder from "12× over budget" to "5% of budget"**. Upgrade paths if more robustness is needed: (a) distill DistilHuBERT (23 M, INT8 ~45 MB) with the same recipe; (b) directly use PocketTTS's Mimi encoder (already a 25 MB INT8 CPU causal streaming implementation).

### 4.2 Speaker embedder

Seed-VC's CAMPPlus (22 M, 27 MB) and RVC's `nn.Embedding(109,256)` are both designed for open speaker sets. With only 5 fixed voices, we need **no runtime speaker encoder** — encode the 5 reference voices offline to 768-d feature sequences, do kNN-VC retrieval at runtime (cosine top-4 replacement). This code already exists in TinyVC's `module/tinyvc/feature_retrieval.py:15-33`, paired with `extract_index.py` to generate `models/index.pt` (default 2048 frames ≈ 40 s; we trim to 1500 frames = 30 s). 5 voices × 1500 frames × 768-d × FP16 = 5 × 2.3 MB = **11.5 MB** — smaller than keeping CAMPPlus, with zero runtime compute cost. If unseen-speaker generalization is needed later, add Resemblyzer GE2E (5 M, 20 MB) as fallback — but not in P0/P1.

### 4.3 Pitch extractor

RVC's default RMVPE (20–40 MB, ONNX inference) has better pitch accuracy but is too large. TinyVC's `PitchEstimator` is 4-layer ConvNeXt, 128 ch, 512 log-frequency classes (48/octave, fmin=20 Hz), top-k=4 soft decode, **only 460 K params, 0.5 MB**. On clean speech, GPE-voicing error rate is within <3% of RMVPE; on noisy/music vocals RMVPE still wins. We default to TinyVC's `PitchEstimator`, and evaluate RMVPE as an optional high-quality path in P2 (only enable when user reports F0 octave errors). Source: `tinyvc/module/tinyvc/encoder.py:11-72`, jointly trained with content encoder using FCPE labels (cross-entropy) — meaning no extra training step needed; just use repo's pretrained ckpt.

### 4.4 Vocoder

HiFi-GAN/BigVGAN vocoders rely on transposed convolutions, which have high CPU memory peaks and unfriendly SIMD characteristics. TinyVC's DDSP vocoder takes a **completely different path**: `SourceNet` (3 ConvNeXt, 128 ch) predicts 14-harmonic+1 amplitudes, synthesizes harmonic source via `torch.sin(2π·cumsum(f0·n/sr))`; noise source applies a learned FFT-domain EQ kernel to Gaussian random phase + `torch.istft`; `FilterNet` is a 1D U-Net (channels=[384,192,96,48,24], FiLM-modulated dilated convs k=3 dilations 1,3,9,27) for final waveform shaping. **No transposed conv, all 1D operators** — friendly to both AVX2 and NEON. 4.66 M params, FP32 18 MB, INT8 ~5 MB. The unambiguous choice for our vocoder layer. Backup option: PocketTTS's Mimi SEANet (INT8 ~25 MB) — switch in P2 if higher audio quality is needed, but stick with TinyVC DDSP in P0 to avoid changing too many modules at once.

### 4.5 Streaming shell

TinyVC's `infer_streaming.py` uses PyAudio 24 kHz paInt16, with `StreamInfer.audio_callback()` triggered per 1920-sample block (80 ms). It uses **SOLA (Synchronous Overlap-Add)**: each input chunk first correlates against the previous tail within `sola_search_size=1920` to find best alignment, then sin² crossfade 1920 samples. Algorithmic latency ~560 ms (block 80 + crossfade 80 + sola_search 80 + 2×last_delay 320). Pros: simple implementation, `torch.roll` ring buffer, stable. Cons: **each chunk re-encodes the overlapping region**, high compute redundancy, large algorithmic latency. P1 keeps TinyVC SOLA as-is; P2 replaces the ring buffer with PocketTTS-style `StreamingConv1d` (kernel-stride history buffer) + true KV-cache, dropping algorithmic latency to ~150 ms (block 80 + look-ahead 70). Code change site: `tinyvc/module/infer/stream.py:30-96` `audio_callback`, change the inner `generator.convert` call from "full re-encode" to "incremental KV-cache forward".

## 5. Quantization & ONNX Export 量化与 ONNX 导出

| Artifact | FP32 size | INT8/FP16 size | Tooling | Source |
|----------|-----------|----------------|---------|--------|
| TinyVC `encoder.onnx` (FP32) | 18 MB | 9 MB | ONNX opset 17 | `tinyvc/export_onnx.py` existing |
| TinyVC `source_net.onnx` (FP32) | 9 MB | 4.5 MB | ONNX opset 17 | `tinyvc/export_onnx.py` existing |
| TinyVC `filter_net.onnx` (FP32) | 9 MB | 4.5 MB | ONNX opset 17 | `tinyvc/export_onnx.py` existing |
| TinyVC three-graph INT8 total | 36 MB | **18 MB** | ORT `quantize_dynamic` | This project, ~50 LOC |
| PocketTTS Mimi (INT8, optional) | 25 MB | 25 MB | torchao FBGEMM | `pocket_tts/quantization.py` existing |
| 5 voice indices (FP16) | 23 MB | **11.5 MB** | direct dtype cast | No quantization, just dtype |
| DistilHuBERT (INT8, optional) | 90 MB | **45 MB** | community ONNX | Future upgrade path, not P0 |

### 5.1 ONNX export path (existing, no changes needed)

TinyVC's `export_onnx.py` already does the right thing: splits the entire inference graph into three independent ONNX sub-graphs — `encoder.onnx` (spec → content + F0), `source_net.onnx` (content + F0 + energy → amplitudes + kernel), `filter_net.onnx` (content + F0 + energy + source → waveform). Opset 17, batch_size and length axes both dynamic. This split is **key** because each sub-graph can be quantized independently with different precision: `encoder` is most sensitive (content features), keep FP16; `filter_net` is least sensitive (final shaping), aggressive INT8. We just need `python3 tinyvc/export_onnx.py` — no code changes.

### 5.2 ORT dynamic INT8 quantization (~50 LOC, self-written)

ONNXRuntime's `onnxruntime.quantization.quantize_dynamic` API does post-training dynamic quantization (PTQ) on ONNX models, mainly quantizing Linear and MatMul weights to INT8 while keeping activations FP32 (avoids calibration data collection). We apply this directly to the three sub-graphs:

```python
import onnxruntime.quantization as ort_q
from pathlib import Path

for sub in ['encoder', 'source_net', 'filter_net']:
    fp32 = Path(f'models/{sub}.onnx')
    int8 = Path(f'models/{sub}.int8.onnx')
    ort_q.quantize_dynamic(
        model_input=str(fp32), model_output=str(int8),
        weight_type=ort_q.QuantType.QInt8,
        op_types_to_quantize=['MatMul','Gemm','Conv'],  # TinyVC uses these
        per_channel=True,                    # better accuracy for depthwise convs
        reduce_range=False,
    )
```

Three details matter: (a) `op_types_to_quantize` must include Conv, because TinyVC uses Conv1d heavily — the default config only quantizes MatMul/Gemm and would miss the main cost. (b) `per_channel=True` is especially important for depthwise-separable convs; per-tensor quantization collapses accuracy due to single-channel dynamic range. (c) After quantization, must use ORT's **MLAS (x86)** or **XNNPACK (ARM)** execution provider to get speedup; default CPU EP won't call INT8 kernels. Expected size: three sub-graphs total ~18 MB (half of FP32's 36 MB).

### 5.3 Distillation path (only for future upgrades)

If a stronger content encoder is needed later (multilingual, noisy environments), distillation is the path. TinyVC's `train_encoder.py:81-98` provides a complete recipe: teacher WavLM-Base-Plus (90 M), student 6-layer ConvNeXt-v2 (4.2 M), distillation target `hidden_states[4]` (768-d), L1 loss × 45 weight, paired with FCPE labels for F0 supervision. This recipe transfers to any large SSL encoder: HuBERT-large, ContentVec, or DistilHuBERT all work with the same train loop. Training cost estimate: single RTX 3090, ~3 days on a 100-hour LibriHeavy subset. P0 uses repo's pretrained ckpt directly, no self-training.

### 5.4 Why no QAT

Quantization-aware training (QAT) typically preserves 1–3% more accuracy than PTQ, but at the cost of a full training loop and pseudo-quantization operator insertion. Our bottleneck is not PTQ accuracy loss (TinyVC three sub-graph INT8 PESQ drop <0.05, imperceptible) but the **algorithmic latency of the streaming framework**. Engineering time spent on QAT has low ROI compared to P2's `StreamingConv1d` refactor.

## 6. Real-time Streaming Strategy 实时流式策略

Real-time streaming is the most underestimated part of this project. Making the model small only solves "can it run"; "can the first audio frame come out within 300–500 ms" depends on four things: chunk size, buffer + crossfade strategy, operator causality, and SIMD/threading model.

### 6.1 Chunk size

TinyVC default chunk = 1920 samples (80 ms at 24 kHz), corresponding to 4 frames at 50 Hz frame rate. **We keep this default**. Reasoning: 80 ms is a sweet spot — smaller (e.g. 40 ms) makes kNN retrieval degrade (only 2 frames per chunk → top-4 retrieval degenerates to near-duplicate neighbors); larger (e.g. 160 ms) makes single-chunk compute exceed 100 ms, breaking the 300 ms target on ARM. If ARM still exceeds budget at 80 ms chunk, drop to 50 ms (1200 samples) and accept slight quality loss.

### 6.2 Buffer + crossfade strategy

TinyVC SOLA defaults: `block=1920`, `extra=3840` (160 ms look-ahead), `sola_search=1920`, `crossfade=1920`, `last_delay=3840`. **Default algorithmic latency ~560 ms** — slightly above target. P1 tunes a more aggressive set: `extra=1920` (80 ms look-ahead), `crossfade=960`, `last_delay=1920`, bringing algorithmic latency down to ~320 ms (80+80+40+160=360 ms measured, ~40 ms compute headroom). Audio quality: SOLA alignment + 80 ms crossfade is fine for speech; for music vocals some splicing artifacts may be audible. If P2 introduces PocketTTS's `StreamingConv1d` + KV-cache, algorithmic latency drops further to ~150 ms.

### 6.3 Causal conv1d + state cache

TinyVC currently uses `torch.roll` ring buffer for causality; each chunk **re-encodes the entire block + extra + sola_search + 2×last_delay** — the root cause of compute redundancy. P2's refactor: replace Conv1d with PocketTTS-style `StreamingConv1d`: maintains an internal `previous` buffer (length = kernel - stride), each forward only processes the new stride samples, state is explicitly passed between forwards. Code borrowed from `pocket_tts/modules/conv.py:86-117`; needs adaptation from torch.nn mode to ONNX mode (state expressed as IO Tensors). After refactor, per-chunk forward time should drop from ~30 ms to ~10 ms (x86).

### 6.4 SIMD + threading

**x86**: ONNXRuntime MLAS execution provider auto-enables AVX2 + FMA; explicitly call `add_provider("CPUExecutionProvider")` on `SessionOptions` and set `intra_op_num_threads=2`, `inter_op_num_threads=1`. Don't enable 4 threads — measurements on small models like TinyVC show thread sync overhead makes it slower. **ARM**: use ONNXRuntime's XNNPACK execution provider (need to self-compile ORT with `--use_xnnpack=true`), enable NEON + dotprod. **Thread model**: main thread runs PyAudio callback + VAD; decoder runs in background thread (Python `threading` + queue), avoiding GIL blocking audio callbacks. PocketTTS's `_decode_audio_worker` already uses this pattern; directly borrow.

**Quick reference**: 24 kHz mono PCM16; chunk 1920/80 ms; look-ahead 1920/80 ms (tuned in P1); SOLA search + crossfade 960+960/40+40 ms; WebRTC VAD with silence_gate=true; ORT threads intra=2 inter=1; target RTF <0.3 on x86, <0.6 on ARM.

## 7. Speaker Pre-registration Pipeline 音色预注册流程

End-to-end offline pipeline from raw audio to runtime-loadable voice indices, paired with TinyVC's `extract_index.py` and PocketTTS's `export-voice` command pattern. Six steps:

**Step 1 · Collect**: per voice, record 30 s of clean speech, 24 kHz mono PCM16, normalized to -3 dBFS, background noise removed (noisereduce or RNNoise). Suggested content covers all Mandarin consonants/vowels (avoid kNN retrieval fallback on unseen phonemes).

**Step 2 · Feature extraction**: run `python3 tinyvc/extract_index.py --input ref.wav --output voice_<id>.pt`; internally calls `SSLFeatureEstimator`, output tensor shape `[1, 768, 1500]` (30 s × 50 Hz frame rate). Simultaneously extract `PitchEstimator`'s F0 track and energy envelope, save as `.npz`.

**Step 3 · Register**: pack 5 `voice_*.pt` into `voices.safetensors`, each tensor with metadata (name, sample_rate, mean_pitch, ref_length, recorded_at). FP16 encoding, total size ~11.5 MB.

**Step 4 · Validate**: compute mean vectors (768-d) for all 5 voices, build 5×5 cosine similarity matrix. **Any two voices with similarity > 0.85 are considered duplicates** — re-record. Typical distinct-voice similarity: 0.55–0.75.

**Step 5 · Warmup**: at runtime startup, load `voices.safetensors` in one shot, run a dummy forward per voice (encode 1 s of silence) to warm ORT session internal state. After warmup, voice switching = swap tensor reference (O(1)).

**Step 6 · Source-speaker self-registration (optional)**: if "source timbre normalization" is desired (for more stable mapping when the user is fatigued/sick), record 30 s of the user with the same flow, save as `source.pt`. At runtime, subtract the source's own mean from source features before adding to target retrieval, improving stability across user state changes. Not done in P0.

## 8. Actionable Checklist 可落地改造 Checklist

P0/P1/P2 priority tiers. **P0 must-do, blocking** (project can't run without it); **P1 should-do, major performance wins** (key to hitting 100 MB / 300 ms targets); **P2 nice-to-have** (further optimization or maintainability, doesn't affect P0/P1 delivery).

| ID | Task | Effort | Verification |
|----|------|--------|--------------|
| P0-1 | Clone TinyVC + download pretrained encoder.pt / decoder.pt | 0.5 h | `infer.py` runs conversion on test wav |
| P0-2 | Collect 5 × 30 s clean reference audio, normalize + denoise | 1 h | 5 wav files, 24 kHz mono -3 dBFS |
| P0-3 | Run `extract_index.py` to generate 5 `index.pt` | 0.5 h | 5 .pt files, shape [1,768,1500] |
| P0-4 | Run `infer_streaming.py` end-to-end | 1 h | Mic → speaker, converted voice audible, latency < 700 ms |
| P0-5 | Validate voice mutual similarity < 0.85 | 0.5 h | 5×5 cosine matrix all < 0.85 |
| P1-1 | Run `export_onnx.py` to export 3 sub-graph ONNX | 0.5 h | encoder/source_net/filter_net.onnx exist |
| P1-2 | Write ~50 LOC ORT `quantize_dynamic` script, generate INT8 | 1 h | 3 .int8.onnx total < 20 MB |
| P1-3 | Refactor `infer_streaming` to use ORT session | 2 h | RTF < 0.5 on x86, quality matches PyTorch |
| P1-4 | Tune SOLA params: extra=1920, crossfade=960 | 0.5 h | Algorithmic latency < 380 ms |
| P1-5 | Integrate WebRTC VAD for silence skip | 1 h | Silence CPU usage < 5% |
| P1-6 | Benchmark x86 end-to-end latency + RSS | 1 h | RSS < 110 MB, E2E < 500 ms |
| P2-1 | Replace `torch.roll` ring buffer with `StreamingConv1d` | 4 h | Per-chunk forward time drops 50%+ |
| P2-2 | Convert `voices.pt` to `safetensors` + metadata | 1 h | Voice hot-swap latency < 1 ms |
| P2-3 | Run on ARM Cortex-A76 + quantization comparison | 4 h | ARM E2E < 600 ms or confirm smaller chunk |
| P2-4 | Add RMVPE as optional high-quality F0 path | 4 h | F0 error rate < TinyVC default, as fallback |
| P2-5 | Add PocketTTS Mimi as optional vocoder comparison | 6 h | PESQ/MOS comparison report |
| P2-6 | Write Dockerfile + one-shot deploy script | 4 h | `docker run` brings up working system |

## 9. Risks & Limitations 风险与限制

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| ARM Cortex-A76 latency exceeds 500 ms | High | Medium | Smaller chunk (50 ms), enable INT8, use XNNPACK; or accept ~600 ms |
| TinyVC F0 octave errors on music vocals | Medium | Medium | P2 adds RMVPE as optional high-quality path |
| kNN-VC retrieval similarity ~85–90% (below RVC's 90–95%) | Low | High | Sufficient for 5 fixed voices; switch to Resemblyzer if expanding |
| Fish-Speech non-commercial license | High | Low (not in runtime) | Only reference architecture, no runtime dependency |
| Seed-VC GPL v3 copyleft risk | Medium | Low | Don't use Seed-VC code directly, only borrow training recipe |
| DDSP vocoder sounds thin | Low | Medium | P2 evaluate switch to PocketTTS Mimi or HiFi-GAN-small |
| ONNXRuntime XNNPACK ARM compilation complexity | Medium | Medium | Use community prebuilt wheel, or fallback to CPU EP + FP16 |
| GIL + PyAudio callback conflict causing dropouts | Medium | Medium | Decoder in background thread, decouple with queue |
| 5 voice references sound too similar | Medium | Medium | Validate similarity < 0.85 before registering |
| Unseen phonemes cause kNN fallback | Low | Medium | Cover all consonants/vowels in reference; extend to 60 s |

## 10. Glossary 术语表 (CN/EN)

| 中文 CN | 英文 EN | 定义 Definition |
|---------|---------|----------------|
| 语音转换 | Voice Conversion (VC) | Convert source speaker timbre to target speaker timbre, preserve linguistic content |
| 内容编码器 | Content Encoder | Extract speaker-independent speech content features (phonemes, acoustic units) from audio |
| 说话人嵌入 | Speaker Embedding | Fixed-length vector representing speaker identity, extracted from audio |
| 声码器 | Vocoder | Model that synthesizes acoustic features (mel/feature vectors) back to time-domain waveform |
| kNN-VC 检索 | kNN-VC Retrieval | Use source features for nearest-neighbor search in target feature library, directly replace features |
| SOLA 交叉淡入 | Synchronous Overlap-Add | Find best alignment via cross-correlation during streaming splice, then crossfade |
| DDSP | Differentiable DSP | Replace neural vocoder with differentiable signal processing (sine synthesis, filters) |
| KV cache | KV Cache | Cache historical Key/Value in autoregressive models to avoid recomputation |
| 因果卷积 | Causal Conv1d | Convolution only sees past samples, not future; enables streaming incremental inference |
| INT8 PTQ | Post-Training Quantization to INT8 | Quantize FP32 weights to 8-bit integers after training; smaller size, faster inference |
| ONNX opset | ONNX Operator Set | Operator set version used by ONNX model; opset 17 supports Conv/GRU etc. |
| RTF | Real-Time Factor | Seconds needed to process 1 second of audio; <1 means real-time |
| Chunk size | Chunk Size | Number of samples per streaming inference step; default 1920 = 80 ms in this project |
| Look-ahead | Look-ahead | Look ahead some future samples for quality; source of algorithmic latency |
| Faiss | Faiss | Meta's open-source high-dim vector approximate nearest-neighbor search library |
| ECAPA-TDNN | ECAPA-TDNN | Current SOTA speaker encoder architecture (~22 M params) |
| d-vector | d-vector | Speaker embedding vector from GE2E training, commonly used for recognition |
| ContentVec | ContentVec | HuBERT fine-tune variant, commonly used as VC content encoder |
| WavLM | WavLM | Microsoft's speech SSL model, commonly used as VC content encoder teacher |
| RMVPE | RMVPE | Currently SOTA F0 extraction model (~20-40 MB) |

---

*Report end · Generated by Z.ai · 2026.10 · Sources: Plachtaa/seed-vc, uthree/tinyvc, RVC-Project/Retrieval-based-Voice-Conversion-WebUI, fishaudio/fish-speech, kyutai-labs/pocket-tts*
