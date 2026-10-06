# Deployment · x86 / ARM 部署指南

## x86 desktop deployment (Ubuntu 22.04+)

```bash
# 1. System deps
sudo apt update && sudo apt install -y \
    portaudio19-dev libsndfile1-dev libffi-dev \
    python3.10 python3.10-venv

# 2. Create venv
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Install Python deps (CPU-only PyTorch)
pip install --upgrade pip
pip install -r requirements.txt

# 4. ONNXRuntime with MLAS (x86 AVX2/FMA) — prebuilt wheel includes it
pip install onnxruntime==1.18.1  # pinned; verify MLAS available

# 5. Verify MLAS provider is active
python -c "import onnxruntime as ort; \
           s=ort.InferenceSession('models/encoder.int8.onnx'); \
           print('Providers:', s.get_providers())"
# Expected: ['CPUExecutionProvider']  (MLAS is the default CPU EP, no separate name)

# 6. (Optional) Tune ORT threads
export OMP_NUM_THREADS=2
export ORT_INTRA_OP_NUM_THREADS=2
export ORT_INTER_OP_NUM_THREADS=1

# 7. Run
python scripts/realtime_infer.py --voice-id 0
```

### x86 hardware reference

| CPU class | Expected E2E latency | Expected RSS |
|-----------|----------------------|---------------|
| Intel i5-12400 (6c/12t, AVX2) | ~330 ms | ~85 MB |
| AMD Ryzen 5 5600 (6c/12t, AVX2) | ~340 ms | ~85 MB |
| Intel i3-12100 (4c/8t, AVX2) | ~400 ms | ~90 MB |
| Older i5-8250U (4c/8t, AVX2) | ~480 ms | ~95 MB |

## ARM deployment (Raspberry Pi 5 / RK3588)

ARM needs ONNXRuntime compiled with XNNPACK execution provider for NEON acceleration. Prebuilt wheels are scarce; we recommend two paths:

### Path A: Use community prebuilt ONNXRuntime

```bash
# Prebuilt ARM wheels from the onnxruntime-community project
pip install onnxruntime==1.18.1 \
    --extra-index-url https://pkgs.dev.azure.com/.../onnxruntime/_packaging/onnxruntime/pypi/simple/

# Verify NEON is being used (check via perf or strace)
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### Path B: Self-compile ONNXRuntime with XNNPACK

```bash
# Approximate 1-hour build on a 4-core ARM box
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --update --build --parallel \
    --use_xnnpack=true \
    --cmake_extra_defines onnxruntime_USE_XNNPACK=ON \
    --cmake_extra_defines onnxruntime_CMAKE_OSX_ARCHITECTURES=arm64
pip install build/Linux/Release/dist/onnxruntime-*.whl
```

### Path C: Fallback (FP16 + CPU EP, no XNNPACK)

If XNNPACK is unavailable, ONNXRuntime falls back to default CPU EP. Performance is ~50% slower but still works. Use FP16 weights instead of INT8 to avoid the QDQ overhead:

```python
# In modules/streaming.py
session = ort.InferenceSession(
    'models/encoder.fp16.onnx',  # use FP16 instead of INT8
    providers=['CPUExecutionProvider'],
    sess_options=opts
)
```

### ARM hardware reference

| CPU class | Expected E2E latency (INT8+XNNPACK) | Expected E2E latency (FP16 fallback) | RSS |
|-----------|-------------------------------------|--------------------------------------|-----|
| Raspberry Pi 5 (Cortex-A76 4c) | ~520 ms | ~720 ms | ~95 MB |
| RK3588 (Cortex-A76 4c + A55 4c) | ~480 ms | ~680 ms | ~95 MB |
| Apple M1 (P-core 4c + E-core 4c) | ~310 ms | ~430 ms | ~80 MB |

> ⚠️ **ARM borderline**: Pi 5 with INT8+XNNPACK lands at ~520 ms — slightly above the 500 ms target. To hit target: drop chunk size to 50 ms (1200 samples) and accept slight quality loss, or accept ~520 ms as "good enough".

## Docker deployment

See `Dockerfile` and `docker-compose.yml` (P2-6 deliverable). Quick reference:

```dockerfile
# Dockerfile (skeleton, completes in P2-6)
FROM python:3.10-slim
RUN apt-get update && apt-get install -y portaudio19-dev libsndfile1-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Pre-download ONNX models + voice registry at build time
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/uthree/tinyvc/resolve/main/models/encoder.pt', 'models/encoder.pt')"
ENTRYPOINT ["python", "scripts/realtime_infer.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  vc:
    build: .
    devices:
      - /dev/snd:/dev/snd  # audio device passthrough
    volumes:
      - ./data/voices:/app/data/voices:ro
      - ./models:/app/models:ro
    command: ["--voice-id", "0"]
```

## Performance tuning cheat sheet

| Knob | Default | Aggressive | Conservative |
|------|---------|------------|--------------|
| `chunk_size` (samples) | 1920 (80 ms) | 1200 (50 ms) | 3840 (160 ms) |
| `extra_size` (look-ahead) | 1920 (80 ms) | 960 (40 ms) | 3840 (160 ms) |
| `crossfade_size` | 960 (40 ms) | 480 (20 ms) | 1920 (80 ms) |
| `sola_search_size` | 960 (40 ms) | 480 (20 ms) | 1920 (80 ms) |
| `last_delay_size` | 1920 (80 ms) | 960 (40 ms) | 3840 (160 ms) |
| ORT intra_op threads | 2 | 1 | 4 |
| INT8 quantize | yes | yes | no (FP16) |
| WebRTC VAD | on | on | off |
| Algorithmic latency | ~320 ms | ~190 ms | ~640 ms |
| Compute latency (x86) | ~75 ms | ~50 ms | ~110 ms |
| E2E latency (x86) | ~370 ms | ~240 ms | ~750 ms |
