# Real-time CPU Voice Conversion (VC) — Lightweight Prototype

> **Goal**: Real-time voice conversion on pure CPU, ~100 MB RAM, 300–500 ms latency, 5 fixed target voices × 30 s reference.
>
> **v1 spine**: TinyVC (Apache-2.0) + INT8 PTQ + optional PocketTTS modules.
>
> **v2 hybrid**: Replace kNN-VC with OpenVoice v2 256-d ReferenceEncoder (0.76 M, 1 MB INT8) + ResidualCoupling flow (8.7 M, 9 MB INT8); replace DDSP with F5-TTS Vocos vocoder (13 M, 7 MB INT8). Expected: +3-5% similarity, MOS +0.2-0.3, RSS ~85 MB.
>
> **v3 incremental upgrade** (NEW, Round 3): Replace OpenVoice v2 256-d ReferenceEncoder with Spark-TTS BiCodec SpeakerEncoder (ECAPA-TDNN c512 + Perceiver + ResidualFSQ, 6-12 M, 5 MB INT8, 48-byte FSQ code per voice). Expected: +1-2% similarity (→ 90-94%), RSS ~89 MB.

After analyzing **17 open-source VC/TTS repositories** (Round 1: 5 repos, Round 2: 6 repos, Round 3: 6 repos), the v3 hybrid architecture sits at the optimal point of the Pareto curve for "CPU ≤100 MB + real-time + good VC quality". This repo contains the full analysis + Python prototype skeleton.

## Repository Layout

```
.
├── README.md                              ← you are here
├── analysis.md                            ← v1 bilingual analysis (Markdown mirror of PDF 1)
├── realtime-cpu-vc-tech-brief.pdf         ← v1 report (16 pages, Round 1: Seed-VC / TinyVC / RVC / Fish-Speech / PocketTTS)
├── realtime-cpu-vc-supplement-v2.pdf      ← v2 supplement (10 pages, Round 2: OpenVoice v2 / FreeVC / GPT-SoVITS / Applio / F5-TTS / CosyVoice2 + v2 hybrid)
├── realtime-cpu-vc-supplement-v3.pdf      ← v3 supplement (7 pages, Round 3: Spark-TTS / IndexTTS / MaskGCT / Orpheus / StyleTTS-2 / Piper + v3 hybrid)
├── docs/
│   ├── checklist.md                       ← P0/P1/P2 actionable checklist (v1 + v2 + v3 tasks)
│   ├── architecture.md                     ← module-level architecture + data flow (v1 + v2 + v3)
│   └── deployment.md                      ← x86 / ARM deployment notes
├── requirements.txt
├── configs/
│   ├── default.yaml                       ← v1 runtime config (TinyVC + kNN + DDSP)
│   ├── v2_hybrid.yaml                     ← v2 runtime config (OpenVoice encoder + flow + Vocos)
│   └── v3_hybrid.yaml                     ← v3 runtime config (Spark BiCodec encoder + OpenVoice flow + Vocos)
├── modules/
│   ├── __init__.py
│   ├── encoder.py                         ← v1: TinyVC SSLFeatureEstimator skeleton
│   ├── decoder.py                         ← v1: TinyVC DDSP decoder skeleton
│   ├── knn_retrieval.py                   ← v1: kNN-VC feature replacement (default.yaml)
│   ├── speaker_encoder.py                 ← v2 NEW: OpenVoice v2 ReferenceEncoder (256-d)
│   ├── speaker_encoder_v3.py              ← v3 NEW: Spark-TTS BiCodec SpeakerEncoder (512-d + FSQ)
│   ├── flow.py                            ← v2 NEW: OpenVoice v2 ResidualCouplingBlock flow
│   ├── vocoder_v2.py                      ← v2 NEW: F5-TTS Vocos vocoder (replaces DDSP)
│   ├── pitch.py                           ← v1: TinyVC PitchEstimator wrapper
│   └── streaming.py                       ← v1+v2+v3: SOLA streaming shell + ORT session runner
└── scripts/
    ├── register_voices.py                 ← v1: 5 voices × 30 s → voices.safetensors (kNN index)
    ├── register_voices_v2.py              ← v2 NEW: 5 voices → se_<id>.pth (256-d embeddings)
    ├── register_voices_v3.py              ← v3 NEW: 5 voices → se_<id>.fsq (48-byte FSQ codes) + se_<id>.pth (512-d)
    ├── quantize_int8.py                   ← ONNX dynamic INT8 quantization
    └── realtime_infer.py                  ← runtime entry (PyTorch or ORT mode)
```

## v1 → v2 → v3 Evolution

| Dimension | v1 Baseline | v2 Hybrid | v3 Incremental | Delta v3 vs v2 |
|-----------|-------------|-----------|-----------------|-----------------|
| Content encoder (INT8) | TinyVC ConvNeXt 4.7M | unchanged | unchanged | 0 |
| Speaker encoder (INT8) | kNN index 12 MB | OpenVoice 256-d, 1 MB | **Spark BiCodec 512-d, 5 MB** | +4 MB |
| Speaker disentanglement | none | OpenVoice flow 9 MB | unchanged | 0 |
| Vocoder (INT8) | TinyVC DDSP 5 MB | F5-TTS Vocos 7 MB | unchanged | 0 |
| 5-voice registry storage | 12 MB (kNN FP16) | 2.5 KB (256-d FP16) | **240 B (FSQ) + 5 KB (512-d)** | -10× vs v2 |
| Total RSS | ~85-95 MB | ~85 MB | **~89 MB** | +4 MB (still fits) |
| Speaker similarity | ~85-90% | ~88-93% | **~90-94%** | +1-2% (est) |
| Audio quality (subjective MOS) | DDSP thin | Vocos rich | Vocos rich | flat |
| x86 E2E latency | ~370 ms | ~370-420 ms | ~380-430 ms | +10 ms |
| ARM E2E latency | ~480 ms | ~500-580 ms | ~520-600 ms | borderline (P2 tune) |
| License stack | Apache + Apache | Apache + MIT + MIT | Apache + MIT + Apache | flat (Spark is Apache) |
| Training cost | 0 (pretrained) | 0 (pretrained) | 0 (pretrained) | flat |

## Quick Start

### v3 hybrid (latest recommended)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Clone TinyVC + OpenVoice + Spark-TTS + F5-TTS source repos
git clone --depth 1 https://github.com/uthree/tinyvc /tmp/tinyvc
git clone --depth 1 https://github.com/myshell-ai/OpenVoice /tmp/openvoice
git clone --depth 1 https://github.com/SparkAudio/Spark-TTS /tmp/spark-tts
git clone --depth 1 https://github.com/SWivid/F5-TTS /tmp/f5-tts

# 3. Export + INT8 quantize all ONNX models (TBD per upstream repo)
#    See docs/checklist.md P0-1 to P0-11 for the exact sequence.

# 4. Register 5 voices via v3 BiCodec SpeakerEncoder (FSQ codes + 512-d embeddings)
python scripts/register_voices_v3.py \
    --voices-dir data/voices \
    --speaker-encoder-onnx models/spark_speaker_encoder.onnx \
    --output-dir models/

# 5. Run with v3 config
python scripts/realtime_infer.py --voice-id 0 --config configs/v3_hybrid.yaml
```

### v2 hybrid (older, smaller RAM)

```bash
python scripts/register_voices_v2.py \
    --voices-dir data/voices \
    --ref-encoder-onnx models/openvoice_ref_encoder.onnx \
    --output-dir models/
python scripts/realtime_infer.py --voice-id 0 --config configs/v2_hybrid.yaml
```

### v1 baseline (TinyVC only, simplest)

```bash
python scripts/register_voices.py --voices-dir data/voices --output models/voices.safetensors
python scripts/realtime_infer.py --voice-id 0 --config configs/default.yaml
```

## 17-Repo Pareto Curve (final, after Round 3)

| Option | Speaker sim / Quality | INT8 weight | 100 MB feasible? | Notes |
|--------|-----------------------|-------------|-------------------|-------|
| TinyVC (v1 baseline) | ~85-90% | ~18 MB | ✓ ✓ | Lightest, low similarity |
| v2 hybrid | ~88-93% | ~25 MB | ✓ ✓ | Round 2 recommendation |
| **v3 hybrid** | **~90-94%** | **~29 MB** | **✓ ✓** | **Round 3 recommendation** |
| RVC v2 (DistilHuBERT) | ~90-95% | ~65 MB | ✗ borderline | DistilHuBERT needed |
| OpenVoice v2 (whole) | ~3.6-3.8 MOS | ~33 MB | ✗ | No streaming, non-causal HiFi-GAN |
| StyleTTS-2 (whole) | ~4.14 MOS | ~22 MB | ✗ borderline | TTS not VC |
| Applio (DistilHuBERT) | ~90-95% | ~65 MB | ✗ borderline | Same blocker as RVC |
| Piper | n/a (no zero-shot) | ~62 MB | ✗ (no VC) | Pure TTS, requires per-voice training |
| FreeVC | ~3.7-3.8 MOS | ~310 MB | ✗✗ | WavLM-Large blocker |
| GPT-SoVITS | ~90-95% | ~480 MB | ✗✗ | BERT+HuBERT mandatory |
| CosyVoice2 | ~75% SS | ~520 MB | ✗✗ | Qwen2-0.5B LM |
| F5-TTS | ~4.5 UTMOS | ~346 MB | ✗✗ | NC license, DiT too big |
| Spark-TTS | n/a (TTS) | ~830 MB | ✗✗ | Qwen2.5-0.5B + Wav2Vec2-L (we only borrow BiCodec SpeakerEncoder) |
| IndexTTS | ~73% SS | ~1.7 GB | ✗✗ | GPT-2 0.8B + w2v-bert-2.0 |
| MaskGCT | ~0.805 SIM | ~1.5 GB | ✗✗✗ | 122 forwards, CPU not real-time |
| Orpheus-TTS | n/a | ~1.8 GB (Q4) | ✗✗ | Llama-3.2-3B (watch for future 400M variant) |
| Fish-Speech | n/a | ~1 GB+ | ✗✗✗ | NC license, DualAR 4B |

## Detailed Analysis (3 reports, 33 pages total)

- [`realtime-cpu-vc-tech-brief.pdf`](./realtime-cpu-vc-tech-brief.pdf) — v1 report (16 pp, Round 1: 5 repos)
- [`realtime-cpu-vc-supplement-v2.pdf`](./realtime-cpu-vc-supplement-v2.pdf) — v2 supplement (10 pp, Round 2: 6 new repos + v2 hybrid)
- [`realtime-cpu-vc-supplement-v3.pdf`](./realtime-cpu-vc-supplement-v3.pdf) — v3 supplement (7 pp, Round 3: 6 new repos + v3 hybrid)
- [`analysis.md`](./analysis.md) — Markdown mirror of v1 report
- [`docs/checklist.md`](./docs/checklist.md) — P0/P1/P2 actionable checklist (v1 + v2 + v3 new tasks)
- [`docs/architecture.md`](./docs/architecture.md) — module-by-module architecture (v1 + v2 + v3)
- [`docs/deployment.md`](./docs/deployment.md) — x86 / ARM deployment notes

## Acknowledgements

This prototype builds on the work of:
- [uthree/tinyvc](https://github.com/uthree/tinyvc) — Apache 2.0, the primary spine
- [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) — MIT, v2 ReferenceEncoder + ResidualCoupling flow
- [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) — Apache 2.0, v3 BiCodec SpeakerEncoder
- [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) — code MIT, weights CC-BY-NC (we only borrow the Vocos code + ONNX export)
- [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) — MIT, StreamingConv1d + INT8 quantization code
- [rhasspy/piper](https://github.com/rhasspy/piper) — MIT, SpeechStreamer streaming pattern reference
- [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) — GPL v3, distillation recipe reference
- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — MIT, small-index idea
- [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) — non-commercial, reference architecture only
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — Apache 2.0, flow recipe + HiFT vocoder reference
- [open-mmlab/Amphion (MaskGCT)](https://github.com/open-mmlab/Amphion) — MIT, future-GPU reference
- [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) — Apache 2.0, future 400M variant watch
- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) — MIT, StyleEncoder reference (architecturally equivalent to OpenVoice RefEncoder)
- [index-tts/index-tts (Bilibili)](https://github.com/index-tts/index-tts) — non-commercial, reference only

## License

Prototype code in this repo is MIT-licensed. See [`LICENSE`](./LICENSE). The analysis reports (PDF + Markdown) are CC-BY 4.0.

> ⚠️ **License alert**: The v3 hybrid path uses TinyVC (Apache 2.0), OpenVoice v2 (MIT), Spark-TTS BiCodec SpeakerEncoder (Apache 2.0), and F5-TTS code/export scripts (MIT, not the weights) — keeping the system permissive for commercial use. Avoid incorporating Seed-VC (GPL v3), Fish-Speech (non-commercial), F5-TTS weights (CC-BY-NC), or IndexTTS code (Bilibili non-OSS) into the runtime.
