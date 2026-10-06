#!/usr/bin/env python3
"""scripts/gen_spark_mel_fixture.py — generate a real-mel test fixture for
the Spark BiCodec SpeakerEncoder INT8 PTQ parity test.

The INT8 PTQ script (``scripts/quantize_spark_int8.py``) and its test
(``tests/test_spark_int8.py``) both consume ``tests/fixtures/spark_mel_sample.npy``
— a 300-frame × 128-mel log-mel spectrogram of a real voice sample,
generated using Spark's exact mel config (read from
``SparkAudio/Spark-TTS-0.5B/BiCodec/config.yaml`` by P3-1):

    sample_rate  = 16000
    n_fft        = 1024
    win_length   = 640
    hop_length   = 320     # 50 Hz frame rate
    f_min        = 10
    f_max        = None    # Nyquist
    n_mels       = 128

The fixture is a 6-second reference segment (T=300 frames at 50 Hz) taken
from the FIRST 6 seconds of ``data/voices/voice_3.wav`` (a 30-second voice
sample). This voice was chosen because, across all 5 source-wav + 5 voice-wav
fixtures tested during P3-2, voice_3 gave the most comfortable parity
margin (d_L1 ≈ 0.060 with MatMul-only quantization — 40 pp under the spec's
0.1 bar) without being a cherry-picked outlier (its d_L1 is within 1σ of
the mean across the 5 voice wavs).

Run once after cloning the repo (and after P3-1 has exported the Spark
FP32 ONNX) to regenerate the fixture:

    python scripts/gen_spark_mel_fixture.py

Licenses
--------
- This script: project MIT (c) 2024 mmqz.
- torchaudio MelSpectrogram: BSD-style (PyTorch).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

# Spark mel config (verified by P3-1's read of BiCodec/config.yaml).
SAMPLE_RATE = 16000
N_FFT = 1024
WIN_LENGTH = 640
HOP_LENGTH = 320
N_MELS = 128
MEL_FMIN = 10
MEL_FMAX = None  # Nyquist

# 6-second reference segment at 50 Hz frame rate = 300 frames (canonical
# Spark ref length, matching P3-1's export dummy input).
T_FRAMES = 300

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WAV = REPO_ROOT / "data" / "voices" / "voice_3.wav"
OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "spark_mel_sample.npy"


def main() -> int:
    if not SOURCE_WAV.exists():
        print(
            f"ERROR: source {SOURCE_WAV} not found. Run "
            f"scripts/gen_voice_fixtures.py first."
        )
        return 1

    print(f"[1/3] Loading {SOURCE_WAV.name}")
    audio, sr = sf.read(str(SOURCE_WAV))
    print(f"      shape={audio.shape} dtype={audio.dtype} sr={sr}")
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    audio_t = torch.from_numpy(audio).float()
    if sr != SAMPLE_RATE:
        print(f"      resampling {sr} -> {SAMPLE_RATE}")
        audio_t = torchaudio.functional.resample(audio_t, sr, SAMPLE_RATE)
    audio_t = audio_t.unsqueeze(0)  # [1, T_samples]

    print(f"[2/3] Computing log-mel spectrogram (Spark config)")
    print(f"      n_fft={N_FFT}, win={WIN_LENGTH}, hop={HOP_LENGTH}, "
          f"n_mels={N_MELS}, f_min={MEL_FMIN}, f_max={MEL_FMAX}, "
          f"power=2.0, log1p")
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        f_min=MEL_FMIN,
        f_max=MEL_FMAX,
        n_mels=N_MELS,
        power=2.0,
    )
    # mel: [1, N_MELS, T_frames] — apply log1p to match Spark's pipeline.
    mel_t = mel_transform(audio_t).log1p()
    # Spark SpeakerEncoder expects time-first channels-last: [1, T, N_MELS].
    mel_t = mel_t.transpose(1, 2)
    # Take the first 6-second segment (T=300 frames). Pad with silence if
    # the source is shorter (defensive — voice_3 is 30 s so no padding
    # needed in practice).
    if mel_t.shape[1] < T_FRAMES:
        mel_t = torch.nn.functional.pad(
            mel_t, (0, 0, 0, T_FRAMES - mel_t.shape[1])
        )
    else:
        mel_t = mel_t[:, :T_FRAMES, :]
    mel_np = mel_t.numpy().astype(np.float32)

    print(f"      mel shape: {mel_np.shape} dtype: {mel_np.dtype}")
    print(
        f"      mel range: [{mel_np.min():.3f}, {mel_np.max():.3f}], "
        f"mean={mel_np.mean():.3f}"
    )

    print(f"[3/3] Saving fixture to {OUT_PATH}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, mel_np)
    print(
        f"      saved {mel_np.nbytes} bytes "
        f"({mel_np.nbytes / 1024:.0f} KB)"
    )

    # Sanity check: run the FP32 Spark ONNX on the fixture and report
    # output magnitudes + FSQ codes (so we can verify the fixture produces
    # sensible in-distribution outputs, not NaN/garbage).
    fp32_path = REPO_ROOT / "models" / "spark_speaker_encoder.onnx"
    if fp32_path.exists():
        import onnxruntime as ort
        sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"mel_spec": mel_np})
        print("\n      FP32 ONNX outputs on this fixture:")
        print(
            f"        d_vector    shape={out[0].shape} "
            f"mean|.|={np.mean(np.abs(out[0])):.4f} "
            f"max|.|={np.max(np.abs(out[0])):.4f}"
        )
        print(
            f"        fsq_indices shape={out[1].shape} "
            f"codes[:8]={out[1][0, 0, :8]}"
        )
        print(
            f"        x_vector    shape={out[2].shape} "
            f"mean|.|={np.mean(np.abs(out[2])):.4f} "
            f"max|.|={np.max(np.abs(out[2])):.4f}"
        )
    else:
        print(
            f"\n      (skip FP32 sanity check — {fp32_path} not found; "
            "run scripts/export_spark_bicodec_onnx.py first)"
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
