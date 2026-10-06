# Architecture · 模块级架构

> Companion to `analysis.md` § 4 (Module Trim & Replace Map). This file describes the *target* runtime architecture after P0+P1 are complete. For the *current* skeleton code, see `modules/`.

## Data flow (runtime, per 80 ms chunk)

```
┌─────────┐  16-bit    ┌─────────┐  PCM  ┌──────────────┐  mel-spec  ┌─────────────┐  content feat
│ PyAudio │ ─────────> │ WebRTC  │ ────> │ MelSpec pad  │ ─────────> │ ORT session │ ───────────┐
│ mic in  │  24 kHz    │ VAD     │ yes?  │ (1920→Tspec) │            │ encoder.int8│             │
└─────────┘            └────┬────┘       └──────────────┘            └─────────────┘             │
                            │ silence?                                                                           │
                            ▼ yes → return silence chunk (skip encode)                            │
                                                                                                                  │
   ┌──────────────────────────────────────────────────────────────────────────────────────────────────┘
   │ 768-d × T frames
   ▼
┌──────────────────┐  kNN top-4   ┌──────────────────┐  replaced feat   ┌──────────────────┐  F0
│ kNN retrieval    │ <─────────── │ voices.safetensors│ <─────────────── │ PitchEstimator    │ <──┐
│ (cosine top-4)   │              │ (5 × [1,768,1500])│                  │ ORT int8          │    │
└────────┬─────────┘              └──────────────────┘                  └──────────────────┘    │
         │  target_feat + F0 + energy                                                          │
         ▼                                                                                       │
┌──────────────────┐  harmonic source   ┌──────────────────┐ filtered source ┌──────────────────┐
│ ORT session      │ ─────────────────> │ ORT session      │ ──────────────> │ ORT session       │
│ source_net.int8  │                    │ filter_net.int8  │                 │ (post-EQ, optional)│
└──────────────────┘                    └──────────────────┘                 └──────────────────┘
                                                                                    │
                                                                                    ▼
                                                                       ┌─────────────────┐  PCM
                                                                       │ SOLA crossfade  │ ────> PyAudio speaker
                                                                       │ + ring buffer   │
                                                                       └─────────────────┘
```

## Module responsibilities

### `modules/encoder.py` — Content + F0 encoder (ORT wrapper)

- **Input**: `mel_spec` tensor `[B, n_mels=128, T_frames]` (FP16 or FP32)
- **Output**: `content_feat` `[B, 768, T_frames]` + `f0` `[B, T_frames]` + `energy` `[B, T_frames]`
- **Backend**: ONNXRuntime session, `encoder.int8.onnx` (P1+) or `encoder.onnx` (P0)
- **Code skeleton**: see `modules/encoder.py`

### `modules/knn_retrieval.py` — Speaker conditioning

- **Input**: source `content_feat` `[1, 768, T]` + voice_id `int`
- **Output**: target-swapped `content_feat` `[1, 768, T]`
- **Backend**: pure PyTorch (cosine similarity + top-4 weighted average), no ONNX needed (small enough)
- **Voice library**: loaded once at startup from `voices.safetensors` (5 × `[1, 768, 1500]` FP16 = 11.5 MB total)
- **Hot-swap**: O(1), just swap tensor reference

### `modules/decoder.py` — DDSP vocoder (2 ORT sessions)

- **Sub-graph 1** `source_net.int8.onnx`: takes content+F0+energy → amplitudes (14+1 harmonics) + FFT noise kernel
- **Sub-graph 2** `filter_net.int8.onnx`: takes content+F0+energy+source_signal → final waveform
- **Output**: `[B, 1, T_samples]` at 24 kHz
- **Harmonic synth** (in Python, between the two ORT calls):
  ```python
  # Source: tinyvc/module/tinyvc/decoder.py:24-54
  def oscillate_harmonics(f0, amp, n_harmonics=14, sr=24000):
      # f0: [B, T_frames] -> [B, T_samples]
      f0_up = torch.nn.functional.interpolate(f0.unsqueeze(1), scale_factor=sr//50, mode='linear').squeeze(1)
      n = torch.arange(f0_up.shape[-1], device=f0.device).float()
      phase = torch.cumsum(f0_up / sr, dim=-1) * 2 * math.pi  # [B, T]
      harmonics = torch.stack([torch.sin((k+1) * phase) for k in range(n_harmonics+1)], dim=1)  # [B, K, T]
      return (amp * harmonics).sum(dim=1, keepdim=True)  # [B, 1, T]
  ```
- **FFT-domain noise** (in Python): learned kernel applied to Gaussian random phase + `torch.istft`

### `modules/streaming.py` — Real-time shell

- **Thread model**: 1 main thread (PyAudio callback + VAD), 1 worker thread (ORT inference + SOLA)
- **Audio buffer**: ring buffer of length `block_size + crossfade_size + sola_search_size + 2*last_delay_size`
- **Crossfade**: SOLA (find best cross-correlation shift in `sola_search`, then `sin²` fade)
- **State cache**: `torch.roll(input_wav, -block_size)` per chunk (P1) → replaced by `StreamingConv1d` with explicit state (P2)
- **Config**: see `configs/default.yaml`

### `modules/pitch.py` — PitchEstimator wrapper

- **Input**: mel-spec
- **Output**: F0 track `[B, T_frames]` in Hz
- **Backend**: same ORT session as encoder (PitchEstimator is part of `encoder.onnx` in TinyVC's export) OR a separate session if we export independently
- **P2 extension**: optional RMVPE fallback path

## Memory budget after P1

| Component | RAM | Notes |
|-----------|-----|-------|
| TinyVC encoder (INT8 ONNX) | 5 MB | loaded once at startup |
| TinyVC source_net (INT8 ONNX) | 4.5 MB | loaded once |
| TinyVC filter_net (INT8 ONNX) | 4.5 MB | loaded once |
| ORT runtime (3 sessions) | 20 MB | intra_op=2, inter_op=1 |
| 5 voice indices (FP16) | 12 MB | loaded once at startup |
| Audio ring buffer (PyAudio + SOLA) | 8 MB | 1920 + 1920 + 1920 + 2×3840 samples × 2 bytes |
| Activation peak (single chunk forward) | 15 MB | FP16 mel-spec + intermediate tensors |
| Python interpreter + stdlib | 20 MB | base footprint |
| WebRTC VAD + utils | 2 MB | |
| **Total RSS** | **~91 MB** | within 100 MB target |

## Latency budget after P1 (x86)

| Stage | Time |
|-------|------|
| Audio ADC + ring buffer fill | 5 ms |
| WebRTC VAD decision | 2 ms |
| Encoder ORT forward (4.7M params, INT8, AVX2 MLAS) | 30 ms |
| kNN top-4 retrieval (2048 frames × 768-d × 5 voices) | 5 ms |
| F0 extraction (inside encoder session) | 0 ms |
| source_net ORT forward | 15 ms |
| Python harmonic synth + FFT noise | 5 ms |
| filter_net ORT forward | 20 ms |
| SOLA crossfade | 80 ms (algorithmic) |
| Chunk buffer alignment (2.5 × 80 ms block) | 200 ms (algorithmic) |
| Audio DAC output | 5 ms |
| **Total** | **~367 ms** (within 300–500 ms target) |

## Failure modes & mitigations

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| F0 octave error (TinyVC PitchEstimator on music vocals) | Pitch jumps an octave mid-utterance | P2 add RMVPE fallback; or accept on speech-only use case |
| kNN retrieval fallback on unseen phonemes | Output sounds muffled/whispery on rare syllables | Cover all consonants in reference audio; extend to 60 s if needed |
| SOLA crossfade artifacts | Audible "zipper" at chunk boundaries | Increase `crossfade_size` to 1920 (full block); accept higher latency |
| GIL contention with PyAudio callback | Audio dropouts / underruns | Decoder in background thread, decouple with queue |
| ORT session cold-start lag | First chunk takes 500+ ms | Warmup pass at startup (encode 1 s of silence per voice) |
| ARM XNNPACK not available | ORT falls back to slow CPU EP | Use prebuilt wheel; or compile ORT with `--use_xnnpack=true`; or accept FP16 + CPU EP |
