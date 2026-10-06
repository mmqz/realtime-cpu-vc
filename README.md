# Real-time CPU Voice Conversion (VC) — Lightweight Prototype

> **Goal**: Real-time voice conversion on pure CPU, ~100 MB RAM, 300–500 ms latency, 5 fixed target voices × 30 s reference.
>
> **Architecture spine**: TinyVC (Apache-2.0) + INT8 PTQ via ONNXRuntime + optional PocketTTS modules.

This repository contains the **analysis report (Markdown + PDF)** and a **Python prototype skeleton** for the lightweight real-time VC system described in the accompanying technical brief. The skeleton is *not* a runnable system out-of-the-box — it provides the directory layout, key file stubs, and copy-pasteable code patterns for the P0/P1/P2 checklist in `docs/checklist.md`.

## Repository Layout

```
.
├── README.md                  ← you are here
├── analysis.md                ← full bilingual analysis (mirrors the PDF)
├── docs/
│   ├── checklist.md           ← P0/P1/P2 actionable checklist
│   ├── architecture.md        ← module-level architecture + data flow
│   └── deployment.md          ← x86 / ARM deployment notes
├── requirements.txt           ← Python dependencies for the prototype
├── configs/
│   └── default.yaml           ← runtime config (chunk, look-ahead, etc.)
├── modules/
│   ├── __init__.py
│   ├── encoder.py             ← TinyVC SSLFeatureEstimator skeleton
│   ├── decoder.py             ← TinyVC DDSP decoder skeleton
│   ├── knn_retrieval.py       ← kNN-VC feature replacement
│   ├── pitch.py               ← TinyVC PitchEstimator skeleton
│   └── streaming.py           ← SOLA streaming shell + ORT session runner
└── scripts/
    ├── register_voices.py     ← offline: 5 voices × 30 s → voices.safetensors
    ├── quantize_int8.py       ← ONNX dynamic INT8 quantization on 3 sub-graphs
    └── realtime_infer.py      ← runtime: PyAudio mic → ORT → PyAudio speaker
```

## Quick Start (P0 checklist)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Clone TinyVC pretrained weights (we only need encoder.pt + decoder.pt)
git clone --depth 1 https://github.com/uthree/tinyvc /tmp/tinyvc
mkdir -p models && cp /tmp/tinyvc/models/{encoder,decoder}.pt models/

# 3. Register 5 target voices (30 s each, 24 kHz mono, -3 dBFS)
#    Put reference wavs in data/voices/voice_{id}.wav
python scripts/register_voices.py --voices-dir data/voices --output models/voices.safetensors

# 4. Export ONNX sub-graphs (uses TinyVC's own export_onnx.py)
python -m tinyvc.export_onnx --models-dir models/

# 5. INT8 dynamic quantization on the three exported ONNX graphs
python scripts/quantize_int8.py --models-dir models/

# 6. Run real-time inference
python scripts/realtime_infer.py --config configs/default.yaml --voice-id 0
```

## Key Findings (executive summary)

| Repo | License | Params | ONNX | Streaming | CPU/100 MB | Use in this project |
|------|---------|--------|------|----------|------------|---------------------|
| Seed-VC (tiny) | GPL v3 | ~25 M (DiT) + 1.2 GB (XLS-R) | No | Yes | No | Distillation recipe reference only |
| **TinyVC** | **Apache 2.0** | **9.36 M** | **Yes (opset 17)** | **Yes (SOLA)** | **Yes** | **Main framework, direct reuse** |
| RVC v2 | MIT | 20–30 M + 360 MB (ContentVec) | No (community) | Yes | No | Small-index idea, not runtime |
| Fish-Speech | NC license | ~1 GB+ (DualAR 4B) | No | Yes | No | Reference architecture only |
| PocketTTS | MIT | ~100 M | No (community) | Yes (causal) | Borderline | Borrow StreamingConv1d + INT8 code |

Bottom line: **TinyVC + INT8 PTQ + 5 voice × 30 s kNN index ≈ 80–95 MB RAM, ~370 ms x86 latency, hits all targets.**

## Detailed Analysis

See:
- [`analysis.md`](./analysis.md) — full bilingual report (mirrors `realtime-cpu-vc-tech-brief.pdf`)
- [`docs/checklist.md`](./docs/checklist.md) — P0/P1/P2 actionable checklist with verification criteria
- [`docs/architecture.md`](./docs/architecture.md) — module-by-module architecture
- [`docs/deployment.md`](./docs/deployment.md) — x86 / ARM deployment notes

## Acknowledgements

This prototype builds on the work of:
- [uthree/tinyvc](https://github.com/uthree/tinyvc) — Apache 2.0, the primary spine
- [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) — GPL v3, distillation recipe reference
- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — MIT, small-index idea
- [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) — non-commercial, reference architecture
- [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) — MIT, StreamingConv1d + INT8 quantization code

## License

Prototype code in this repo is MIT-licensed. See [`LICENSE`](./LICENSE). The analysis report (`analysis.md` and the PDF) is CC-BY 4.0.

> ⚠️ **License alert**: If you incorporate code from Seed-VC (GPL v3) or Fish-Speech (non-commercial), the resulting system inherits those licenses. The P0/P1 path in this prototype only uses TinyVC (Apache 2.0) and PocketTTS modules (MIT), keeping the system permissive.
