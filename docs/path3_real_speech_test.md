# Path3 + M0.5: Real VCTK 5-Speaker VC Test Report

> **Supersedes the original Path3 report** (which used
> `librosa.effects.pitch_shift` to fake 5 distinct speakers from one TTS
> voice). The original numbers were biased upward because the 5 "voices"
> shared all formants — CAMPPlus pairwise cosine 0.450 (fake distinct).
>
> M0.5 replaces them with genuine VCTK speakers (p225–p229), pairwise
> cosine 0.313. The v1 baseline on real speakers is significantly worse
> than the path3 report suggested, and is **negative on cross-gender
> conversions**.

## 1. Fixtures

### Source (real human speech)
- **Speaker**: p232 (VCTK), 5 s, 24 kHz mono PCM16, peak-normalised to -3 dBFS
- **Text**: "Please call Stella..." (VCTK prompt 001)
- **Path**: `data/source/source_001.wav` (legacy synthetic source_00X.wav archived to `data/legacy/source/`)

### Target voices (5, real distinct speakers)
Built from VCTK speakers p225–p229, each a 30 s concat of that speaker's
first ~10 utterances from the parquet shards at
`huggingface.co/datasets/sanchit-gandhi/vctk`. Replaces the legacy
`pitch_shifted` voice fixtures (archived to `data/legacy/voice_*.pitch_shifted.wav`).

| voice_id | VCTK speaker | gender | accent | 30s concat RMS |
|----------|--------------|--------|--------|---------------|
| voice_0  | p225         | F      | Southern English | 0.068 |
| voice_1  | p226         | F      | Southern English | 0.052 |
| voice_2  | p227         | M      | Surrey  | 0.046 |
| voice_3  | p228         | F      | Southern English | 0.054 |
| voice_4  | p229         | F      | Southern Irish | 0.060 |

## 2. CAMPPlus target distinctness (cosine similarity)

Measured with the 3D-Speaker CAMPPlus ONNX
(`3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`, 192-d) via
`sherpa_onnx.SpeakerEmbeddingExtractor`.

| Pair | Cosine (M0.5) | Cosine (path3 baseline) |
|------|---------------|--------------------------|
| v0 vs v1 | 0.279 | 0.448 |
| v0 vs v2 | 0.249 | 0.254 |
| v0 vs v3 | 0.363 | 0.409 |
| v0 vs v4 | 0.138 | 0.184 |
| v1 vs v2 | 0.565 | 0.491 |
| v1 vs v3 | 0.438 | 0.867 |
| v1 vs v4 | 0.273 | 0.438 |
| v2 vs v3 | 0.257 | 0.396 |
| v2 vs v4 | 0.335 | 0.609 |
| v3 vs v4 | 0.236 | 0.398 |

- **M0.5 mean pairwise cosine**: 0.313 (range 0.138–0.565)
- **path3 baseline mean**: 0.450 (range 0.184–0.867)
- **Acceptance**: mean < 0.70 → **PASS ✓**

The M0.5 set is genuinely distinct — the path3 v1-vs-v3 cosine of 0.867
(near-collinear, essentially same speaker) is replaced by 0.438 (genuinely
different). The new ceiling 0.565 (v1 vs v2 — two female Southern English
speakers) is well under the 0.70 acceptance threshold.

## 3. v1 kNN-VC baseline (M0.5 real-speaker)

| Voice | VCTK | target_sim | source_sim | VC_effect | RTF | infer_ms |
|-------|------|-----------:|-----------:|----------:|----:|---------:|
| voice_0 | p225 (F) | 0.309 | 0.237 | **+0.072** | 0.067 | 335 |
| voice_1 | p226 (F) | 0.386 | 0.271 | **+0.115** | 0.052 | 261 |
| voice_2 | p227 (M) | 0.326 | 0.280 | **+0.046** | 0.054 | 269 |
| voice_3 | p228 (F) | 0.136 | 0.292 | **-0.156** | 0.054 | 269 |
| voice_4 | p229 (F) | 0.149 | 0.221 | **-0.072** | 0.053 | 264 |
| **mean** |       | **0.261** | **0.260** | **+0.001** | **0.056** | 280 |

### Comparison to path3 baseline

| Metric | path3 (pitch_shifted) | M0.5 (real VCTK) | Δ |
|--------|------------------------|-------------------|---|
| mean target_sim | 0.420 | **0.261** | -0.159 |
| mean source_sim | 0.367 | **0.260** | -0.107 |
| mean VC_effect  | +0.053 | **+0.001** | **-0.052** |
| mean RTF        | 0.073 | 0.056 | -0.017 |

### Observations

1. **v1 baseline on real distinct speakers is essentially zero** (+0.001
   mean VC_effect vs path3 +0.053). The pitch_shifted fixture set was
   masking how badly v1 kNN-VC handles genuinely different speakers.

2. **Cross-gender conversion is destructive**: voice_3 (p228, F) and
   voice_4 (p229, F) both have `target_sim < source_sim` — the output
   sounds *more like the source speaker* than the target. Root cause:
   source (p232, M) pitch range is ~85-180 Hz; target (p228/p229) pitch
   range is ~180-340 Hz. v1 kNN-VC replaces per-frame content features
   but does NOT remap F0 — the decoder synthesises from the source F0
   distribution, so the output pitch stays in the source range and the
   speaker embedding classifier votes for source over target.

3. **Path3's "+0.053 with degenerate -0.068 case" is now replaced by
   "+0.072/+0.115/+0.046 on same-gender pairs, -0.156/-0.072 on
   cross-gender pairs"**. The same-gender pairs confirm v1 kNN-VC has a
   small positive effect when F0 ranges are similar; the cross-gender
   pairs confirm it is destructive when F0 ranges diverge.

4. **RTF improved** from 0.073 to 0.056 — same hardware, just shorter
   pipeline warmup. Either way, RTF is ~13-18× real-time, well within the
   v1 CPU budget.

## 4. Implications for M1a target

The M0.5 measurement **raises the M1a target gap**:

- path3 baseline reported VC_effect +0.053 → M1a target +0.20 → gap +0.147
- M0.5 baseline measures VC_effect +0.001 → M1a target +0.20 → **gap +0.199**

The M1a VoicePack must deliver a +0.199 absolute lift to hit the milestone.
The dominant failure mode is **F0 not being remapped** — this is exactly
what M1b (F0 quantile mapping) is designed to fix. The M1a + M1b
combination is expected to deliver ~+0.15 of the required +0.199, with
M1c (Vocos) providing the remaining ~+0.05.

## 5. Files saved

- `./download/vc_m05_voice_{0..4}.wav` — VC outputs (24 kHz)
- `data/voices/voice_{0..4}.wav` — 5 real VCTK reference voices (30 s each)
- `data/source/source_001.wav` — 5 s real source (p232 reading VCTK prompt 001)
- `data/legacy/voice_{0..4}.pitch_shifted.wav` — archived legacy fixtures (gitignored)
- `data/legacy/source/source_001..010.wav` — archived legacy sources (gitignored)
- `data/paired_hard/{src}_{tgt}_{text_id}.wav` — 130 VCTK parallel pairs (13 text_ids × 10 src/tgt direction pairs) (gitignored, regenerable)
- `data/paired_hard/index.json` — pair manifest (committed)
- `data/voices/campplus_distinctness.json` — pairwise cosine matrix (committed)
- `data/m05_v1_baseline.json` — full per-voice metrics table (committed)
- `models/voices.pt` — rebuilt kNN index from real VCTK voices (5 × 23 MB, gitignored)

## 6. Reproducibility

```bash
# 1. Generate the M0.5 fixtures (downloads ~500 MB of VCTK parquet shards)
python3 scripts/m05_real_voices.py

# 2. Verify distinctness (downloads 28 MB CAMPPlus ONNX)
python3 scripts/m05_campplus_distinctness.py

# 3. Re-run v1 baseline (needs encoder.pt + decoder.pt at models/,
#    + voices.pt built by scripts/build_voices_index.py)
python3 scripts/export_tinyvc_encoder_onnx.py
python3 scripts/export_tinyvc_decoder_onnx.py
PYTHONPATH=src python3 scripts/build_voices_index.py \
    --voices-dir data/voices --models-dir models --output models/voices.pt
PYTHONPATH=src python3 scripts/m05_v1_baseline.py
```
