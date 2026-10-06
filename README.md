# Real-time CPU Voice Conversion (VC) — Lightweight Prototype

> **Goal**: Real-time voice conversion on pure CPU, ~100 MB RAM, 300–500 ms latency, 5 fixed target voices × 30 s reference.
>
> **Architecture spine (v1)**: TinyVC (Apache-2.0) + INT8 PTQ via ONNXRuntime + optional PocketTTS modules.
>
> **v2 Hybrid upgrade**: Replace kNN-VC with OpenVoice v2's 256-d ReferenceEncoder (0.76 M, ~1 MB INT8) + ResidualCoupling flow (8.7 M, ~9 MB INT8) for learned speaker disentanglement; replace DDSP with F5-TTS Vocos vocoder (13 M, ~7 MB INT8, ONNX-ready). Expected: +3–5% speaker similarity lift, MOS +0.2-0.3, RSS unchanged ~85 MB.

This repository contains:
- **v1 analysis report** (`analysis.md` + `realtime-cpu-vc-tech-brief.pdf`, 16 pages, covers Seed-VC / TinyVC / RVC / Fish-Speech / PocketTTS)
- **v2 supplement report** (`realtime-cpu-vc-supplement-v2.pdf`, 10 pages, covers OpenVoice v2 / FreeVC / GPT-SoVITS / Applio / F5-TTS / CosyVoice2 + v2 hybrid architecture)
- **Python prototype skeleton** (modules + scripts, ~1700 LOC v1, +600 LOC v2)

The skeleton is *not* a runnable system out-of-the-box — it provides the directory layout, key file stubs, and copy-pasteable code patterns for the P0/P1/P2 checklists in `docs/checklist.md`.

## Repository Layout

```
.
├── README.md                       ← you are here
├── analysis.md                    ← full bilingual analysis (mirrors v1 PDF)
├── realtime-cpu-vc-tech-brief.pdf       ← v1 report, 16 pages
├── realtime-cpu-vc-supplement-v2.pdf    ← v2 supplement, 10 pages
├── docs/
│   ├── checklist.md                ← P0/P1/P2 actionable checklist (v1 + v2 new tasks)
│   ├── architecture.md              ← module-level architecture + data flow (v1 + v2)
│   └── deployment.md               ← x86 / ARM deployment notes
├── requirements.txt
├── configs/
│   ├── default.yaml                ← v1 runtime config (TinyVC spine + kNN + DDSP)
│   └── v2_hybrid.yaml              ← v2 runtime config (OpenVoice encoder + flow + Vocos)
├── modules/
│   ├── __init__.py
│   ├── encoder.py                  ← v1: TinyVC SSLFeatureEstimator skeleton
│   ├── decoder.py                  ← v1: TinyVC DDSP decoder skeleton
│   ├── knn_retrieval.py            ← v1: kNN-VC feature replacement (used by default.yaml)
│   ├── speaker_encoder.py          ← v2 NEW: OpenVoice v2 ReferenceEncoder (256-d)
│   ├── flow.py                     ← v2 NEW: OpenVoice v2 ResidualCouplingBlock flow
│   ├── vocoder_v2.py               ← v2 NEW: F5-TTS Vocos vocoder (replaces DDSP)
│   ├── pitch.py                    ← v1: TinyVC PitchEstimator wrapper
│   └── streaming.py                ← v1+v2: SOLA streaming shell + ORT session runner
└── scripts/
    ├── register_voices.py          ← v1: 5 voices × 30 s → voices.safetensors (kNN index)
    ├── register_voices_v2.py        ← v2 NEW: 5 voices → se_<id>.pth (256-d embeddings)
    ├── quantize_int8.py             ← ONNX dynamic INT8 quantization
    └── realtime_infer.py            ← runtime entry (PyTorch or ORT mode)
```

## Quick Start

### v1 baseline (TinyVC + kNN + DDSP)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Clone TinyVC + download pretrained weights
git clone --depth 1 https://github.com/uthree/tinyvc /tmp/tinyvc
mkdir -p models && cp /tmp/tinyvc/models/{encoder,decoder}.pt models/

# 3. Register 5 voices (kNN index path, v1)
python scripts/register_voices.py --voices-dir data/voices --output models/voices.safetensors

# 4. Export + INT8 quantize ONNX
python -m tinyvc.export_onnx --models-dir models/
python scripts/quantize_int8.py --models-dir models/

# 5. Run
python scripts/realtime_infer.py --voice-id 0 --config configs/default.yaml
```

### v2 hybrid (OpenVoice v2 + Vocos)

```bash
# 1. Clone OpenVoice v2 + extract ReferenceEncoder + ResidualCoupling flow ONNX
git clone --depth 1 https://github.com/myshell-ai/OpenVoice /tmp/openvoice
# (export scripts TBD — see docs/checklist.md P0-6)

# 2. Clone F5-TTS + export Vocos ONNX
git clone --depth 1 https://github.com/SWivid/F5-TTS /tmp/f5-tts
python /tmp/f5-tts/runtime/triton_trtllm/scripts/export_vocoder_to_onnx.py \
    --ckpt_path /path/to/vocos.pth --out models/vocos.onnx
python scripts/quantize_int8.py --models-dir models/  # INT8 quantize Vocos too

# 3. Register 5 voices via OpenVoice 256-d embeddings (v2 path)
python scripts/register_voices_v2.py \
    --voices-dir data/voices \
    --ref-encoder-onnx models/openvoice_ref_encoder.onnx \
    --output-dir models/

# 4. Run with v2 config
python scripts/realtime_infer.py --voice-id 0 --config configs/v2_hybrid.yaml
```

## v1 vs v2 Comparison

| Dimension | v1 Baseline | v2 Hybrid | Delta |
|-----------|-------------|-----------|-------|
| Total INT8 weight size | ~18 MB | ~25 MB | +7 MB |
| 5-voice registry storage | 11.5 MB (kNN index) | 2.5 KB (256-d embeddings) | **-4600×** |
| Total RSS | ~85–95 MB | ~85 MB | flat (offsetting) |
| Speaker similarity | ~85–90% | **~88–93%** | **+3–5%** (learned flow > kNN) |
| Audio quality (subjective) | DDSP thin | Vocos rich | **+0.2-0.3 MOS** |
| x86 E2E latency | ~370 ms | ~370–420 ms | slight (Vocos ~5ms) |
| ARM E2E latency | ~480 ms | ~500–580 ms | borderline (P2 may need smaller chunk) |
| License stack | Apache + Apache | Apache + MIT + MIT | improved (no NC) |
| Training cost | 0 (pretrained) | 0 (pretrained) | flat |

## Detailed Analysis

See:
- [`realtime-cpu-vc-tech-brief.pdf`](./realtime-cpu-vc-tech-brief.pdf) — v1 report (16 pages, 5 repos analyzed)
- [`realtime-cpu-vc-supplement-v2.pdf`](./realtime-cpu-vc-supplement-v2.pdf) — v2 supplement (10 pages, 6 new repos + v2 hybrid arch)
- [`analysis.md`](./analysis.md) — Markdown mirror of v1 report (v2 sections to be added)
- [`docs/checklist.md`](./docs/checklist.md) — P0/P1/P2 actionable checklist (v1 + v2 new tasks)
- [`docs/architecture.md`](./docs/architecture.md) — module-by-module architecture (v1 + v2)
- [`docs/deployment.md`](./docs/deployment.md) — x86 / ARM deployment notes

## Pareto curve (quality vs INT8 weight size)

| Option | Speaker similarity / Quality | INT8 weight | 100 MB feasible? | Notes |
|--------|-----------------------------|-------------|-------------------|-------|
| TinyVC (v1 baseline) | ~85–90% | ~18 MB | ✓ ✓ | Lightest, low similarity |
| **v2 hybrid** | **~88–93%** | **~25 MB** | **✓ ✓** | **Recommended** |
| RVC v2 (DistilHuBERT) | ~90–95% | ~65 MB | ✗ borderline | Switch to DistilHuBERT |
| OpenVoice v2 (whole) | ~3.6–3.8 MOS | ~33 MB | ✗ | No streaming, non-causal HiFi-GAN |
| GPT-SoVITS | ~90–95% | ~480 MB | ✗✗ | BERT+HuBERT mandatory |
| CosyVoice2 | ~75% SS | ~520 MB | ✗✗ | Qwen2-0.5B LM blocker |
| F5-TTS | ~4.5 UTMOS | ~346 MB | ✗✗ | NC license, DiT too big |

## Acknowledgements

This prototype builds on the work of:
- [uthree/tinyvc](https://github.com/uthree/tinyvc) — Apache 2.0, the primary spine
- [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) — MIT, v2 ReferenceEncoder + ResidualCoupling flow
- [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) — code MIT, weights CC-BY-NC (we only borrow the Vocos code + ONNX export, not the DiT weights)
- [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) — MIT, StreamingConv1d + INT8 quantization code
- [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) — GPL v3, distillation recipe reference
- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — MIT, small-index idea
- [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) — non-commercial, reference architecture only
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — Apache 2.0, flow recipe + HiFT vocoder reference

## License

Prototype code in this repo is MIT-licensed. See [`LICENSE`](./LICENSE). The analysis reports (PDF + Markdown) are CC-BY 4.0.

> ⚠️ **License alert**: If you incorporate code from Seed-VC (GPL v3), Fish-Speech (non-commercial), or F5-TTS weights (CC-BY-NC), the resulting system inherits those licenses. The v2 hybrid path in this prototype only uses TinyVC (Apache 2.0), OpenVoice v2 (MIT), and F5-TTS code/export scripts (MIT, not the weights) — keeping the system permissive for commercial use.
