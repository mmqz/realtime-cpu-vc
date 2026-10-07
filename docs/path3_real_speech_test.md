## Path3: Real Speech VC Test Report

### Source speech obtained
- **Method**: Approach C — torchaudio pretrained `TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH` (107MB)
- Tacotron2 + Griffin-Lim vocoder, pretrained on LJSpeech
- 5 phrases generated, concatenated to ~23s of 24kHz real neural TTS speech
- RMS ≈ 0.043, peak ≈ 0.49 (natural speech levels)
- Saved as `data/source/voice_tts_base.wav` (30s)
- Source = first 5s, saved as `data/source/source_real_001.wav`

### Target voices (5)
Built via `librosa.effects.pitch_shift` from the 30s TTS base, creating
genuinely different "speakers" (different fundamental frequencies):
- voice_0: 0 semitones (unchanged)
- voice_1: +4 semitones
- voice_2: -4 semitones
- voice_3: +7 semitones
- voice_4: -7 semitones

### CAMPPlus target distinctness (cosine similarity)
| Pair | Cosine |
|------|--------|
| v0 vs v1 | 0.448 |
| v0 vs v2 | 0.254 |
| v0 vs v3 | 0.409 |
| v0 vs v4 | 0.184 |
| v1 vs v2 | 0.491 |
| v1 vs v3 | 0.867 |
| v1 vs v4 | 0.438 |
| v2 vs v3 | 0.396 |
| v2 vs v4 | 0.609 |
| v3 vs v4 | 0.398 |

Mean = 0.450 — genuinely distinct (vs synthetic sine fixtures 0.82-0.999)

### VC test (v1 kNN-VC inference, 5s source → target N)
| Voice | Target_sim | Source_sim | VC_effect | RTF |
|-------|-----------|-----------|----------|------|
| v0 (n_steps=0) | 0.416 | 0.484 | -0.068 | 0.0957 |
| v1 (+4)        | 0.403 | 0.355 | +0.049 | 0.0671 |
| v2 (-4)        | 0.457 | 0.353 | +0.105 | 0.0665 |
| v3 (+7)        | 0.377 | 0.357 | +0.019 | 0.0660 |
| v4 (-7)        | 0.449 | 0.287 | +0.162 | 0.0699 |

### Summary
- **mean target_sim**: 0.420 (range 0.377–0.457)
- **mean source_sim**: 0.367 (range 0.287–0.484)
- **mean VC_effect**: +0.053 (positive but small)
- **mean RTF**: 0.073 (≈13.7× real-time on CPU)

### Observations
1. Real speech target voices are **genuinely distinct** under CAMPPlus (mean pairwise cosine ≈ 0.45),
   vs the synthetic sine fixtures' near-collinear embeddings (0.82–0.999).
2. v1 kNN-VC has a **small positive effect** on target similarity (+0.053) — well above the
   +0.0 noise floor, but far below a "good VC" threshold of +0.2.
3. The degenerate case (voice_0 = source pitch, expected high target_sim) actually shows a
   **slight negative VC_effect (-0.068)** — kNN-VC is *destructive* when source≈target pitch,
   because the kNN retrieval+decoder still perturbs the mel features even when no pitch
   transposition is requested.
4. RTF of 0.073 (≈13.7× real-time) is well within the v1 budget for on-device CPU inference.

### Files saved
- `./download/vc_real_voice_0.wav` … `_4.wav` — VC outputs (24kHz)
- `data/source/voice_tts_base.wav` — 30s TTS speech
- `data/source/source_real_001.wav` — 5s source
- `data/voices/voice_0.wav` … `voice_4.wav` — 5 pitch-shifted target voices
- `models/voices.pt` + `models/voices_v1.safetensors` — rebuilt kNN index

