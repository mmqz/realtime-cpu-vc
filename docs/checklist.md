# Actionable Checklist · P0/P1/P2 改造步骤

> Each row has: **ID · Task · Estimated effort · Verification criterion**.
>
> All paths assume the working directory is the repo root and the prototype code lives under `prototype/`.

## P0 — Must-do, blocking (project cannot run without these)

| ID | Task | Effort | Verification criterion |
|----|------|--------|-------------------------|
| P0-1 | Clone TinyVC + download pretrained `encoder.pt` / `decoder.pt` from HuggingFace `uthree/tinyvc` into `models/` | 0.5 h | `python -c "from modules.encoder import SSLFeatureEstimator; import torch; m=SSLFeatureEstimator(); m.load_state_dict(torch.load('models/encoder.pt'))"` exits 0 |
| P0-2 | Collect 5 × 30 s clean reference audio in `data/voices/voice_{id}.wav` (24 kHz mono, normalized to -3 dBFS, denoised via `noisereduce` or `RNNoise`) | 1 h | 5 wav files exist; `sox --i data/voices/*.wav` reports 24000 Hz, 1 channel, ~30 s each |
| P0-3 | Run `python scripts/register_voices.py --voices-dir data/voices --output models/voices.safetensors` | 0.5 h | `voices.safetensors` exists, ~12 MB; each tensor shape `[1, 768, 1500]` FP16 |
| P0-4 | Run `python scripts/realtime_infer.py --voice-id 0` for end-to-end smoke test (PyTorch path, no ONNX yet) | 1 h | Mic input → speaker output is converted voice; latency < 700 ms (rough) |
| P0-5 | Compute 5×5 cosine similarity matrix of registered voices | 0.5 h | All pairs < 0.85; if any exceeds, re-record the offending voice |

**P0 completion criterion**: system runs end-to-end on a laptop microphone, switching among 5 target voices, with audibly distinct output. Latency may exceed the 300–500 ms target at this stage.

## P1 — Should-do, hits the 100 MB / 300–500 ms targets

| ID | Task | Effort | Verification criterion |
|----|------|--------|-------------------------|
| P1-1 | Run `python -m tinyvc.export_onnx --models-dir models/` to export 3 ONNX sub-graphs | 0.5 h | `models/encoder.onnx`, `source_net.onnx`, `filter_net.onnx` exist (~36 MB total) |
| P1-2 | Write `scripts/quantize_int8.py` (~50 LOC) using ORT `quantize_dynamic` | 1 h | 3 `.int8.onnx` files exist; total size < 20 MB; loading smoke test passes |
| P1-3 | Refactor `modules/streaming.py` to use ORT session instead of PyTorch | 2 h | RTF < 0.5 on x86; audio quality unchanged vs PyTorch baseline (PESQ diff < 0.05) |
| P1-4 | Tune SOLA params in `configs/default.yaml`: `extra=1920, crossfade=960, last_delay=1920` | 0.5 h | Algorithmic latency (block + extra + crossfade + 2×last_delay) < 380 ms |
| P1-5 | Integrate WebRTC VAD in `modules/streaming.py` `audio_callback` (silence skip) | 1 h | Silence segments consume < 5% CPU (verified via `perf` or `top`) |
| P1-6 | Benchmark x86 end-to-end latency + RSS | 1 h | `python scripts/realtime_infer.py --benchmark` reports RSS < 110 MB, E2E < 500 ms |

**P1 completion criterion**: system runs end-to-end on x86 CPU within the 100 MB / 500 ms hard targets.

## P2 — Nice-to-have, further optimization or maintainability

| ID | Task | Effort | Verification criterion |
|----|------|--------|-------------------------|
| P2-1 | Replace `torch.roll` ring buffer in `modules/streaming.py` with `StreamingConv1d` (borrowed from PocketTTS) | 4 h | Per-chunk forward time drops 50%+ vs P1; algorithmic latency < 200 ms |
| P2-2 | Convert `voices.pt` to `voices.safetensors` with full metadata (name, sample_rate, mean_pitch, ref_length, recorded_at) | 1 h | Voice hot-swap latency < 1 ms (vs ~50 ms cold-start) |
| P2-3 | Run on ARM Cortex-A76 / RK3588 with INT8 vs FP16 comparison | 4 h | ARM E2E < 600 ms OR confirm need for smaller chunk (50 ms) |
| P2-4 | Add RMVPE as optional high-quality F0 path in `modules/pitch.py` | 4 h | F0 error rate (GPE) < TinyVC default by ≥ 1%; loadable as fallback only when user reports octave errors |
| P2-5 | Add PocketTTS Mimi as optional vocoder in `modules/decoder.py` (compare against TinyVC DDSP) | 6 h | PESQ/MOS comparison report; switchable at config time |
| P2-6 | Write `Dockerfile` + `docker-compose.yml` + one-shot `deploy.sh` | 4 h | `docker run` brings up working system; documented in `docs/deployment.md` |

**P2 completion criterion**: system is production-grade, cross-platform, well-documented.

## Verification commands (cheat sheet)

```bash
# P0-1 verify TinyVC weights load
python -c "import torch; from modules.encoder import SSLFeatureEstimator; \
           m=SSLFeatureEstimator(); m.load_state_dict(torch.load('models/encoder.pt')); \
           print('OK', sum(p.numel() for p in m.parameters()))"
# Expected: OK 4704256

# P0-3 verify safetensors shape
python -c "from safetensors.torch import load_file; \
           d=load_file('models/voices.safetensors'); \
           [print(k, v.shape, v.dtype) for k,v in d.items()]"
# Expected: voice_0 torch.Size([1, 768, 1500]) torch.float16, ...voice_4...

# P0-5 cosine similarity matrix
python -c "import torch; from safetensors.torch import load_file; \
           d=load_file('models/voices.safetensors'); \
           means=torch.stack([d[k].mean(dim=2).squeeze() for k in d if 'voice_' in k]); \
           sim=torch.nn.functional.cosine_similarity(means.unsqueeze(0), means.unsqueeze(1), dim=-1); \
           print((sim<0.85).all())"
# Expected: True

# P1-2 INT8 quantization smoke test
python -c "import onnxruntime as ort; \
           s=ort.InferenceSession('models/encoder.int8.onnx', providers=['CPUExecutionProvider']); \
           print('OK inputs:', [(i.name,i.shape) for i in s.get_inputs()])"

# P1-6 benchmark
python scripts/realtime_infer.py --benchmark --duration 30 | tee benchmark.txt
# Expected output: avg_latency_ms < 500, peak_rss_mb < 110, rtf < 0.5
```

## Total estimated effort

| Tier | Tasks | Effort |
|------|-------|--------|
| P0 | 5 | 3.5 h |
| P1 | 6 | 6 h |
| P2 | 6 | 23 h |
| **Total** | **17** | **~32.5 h** (~1 work-week for one engineer) |
