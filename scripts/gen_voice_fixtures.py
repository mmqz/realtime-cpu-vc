#!/usr/bin/env python3
"""Generate 5 voice reference wavs (30 s each) at 24 kHz mono PCM16.

Primary path:  download 5 distinct VCTK speakers from HuggingFace.
Fallback path: synthesise 5 distinct "speakers" parametrically.

  voice_0:  F0=100 Hz male          formants 800 / 1150 / 2900 Hz
  voice_1:  F0=220 Hz female        formants 1220 / 2810 Hz
  voice_2:  F0=180 Hz neutral       formants 800 / 1700 / 2600 Hz
  voice_3:  F0=140 Hz deep male    formants 600 / 900  / 2400 Hz
  voice_4:  F0=260 Hz high female  formants 850 / 1220 / 3100 Hz

Each voice = glottal pulse train at F0 + 6 harmonics (-12 dB/oct)
              -> 3-pole IIR formant filter (Q=15)
              -> high-pass 80 Hz (remove DC)
              -> 1 % white noise (naturalness)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR: int = 24_000
DURATION_S: float = 30.0
N: int = int(SR * DURATION_S)
TARGET_PEAK_DBFS: float = -3.0

OUT_DIR: Path = Path("data/voices")

# Voice definitions for synthetic fallback
SYNTHETIC_VOICES: list[dict] = [
    {"name": "voice_0", "f0": 100.0, "formants": [800.0, 1150.0, 2900.0]},
    {"name": "voice_1", "f0": 220.0, "formants": [1220.0, 2810.0]},
    {"name": "voice_2", "f0": 180.0, "formants": [800.0, 1700.0, 2600.0]},
    {"name": "voice_3", "f0": 140.0, "formants": [600.0, 900.0, 2400.0]},
    {"name": "voice_4", "f0": 260.0, "formants": [850.0, 1220.0, 3100.0]},
]

# VCTK target speakers and repo (HuggingFace dataset)
VCTK_REPO_ID: str = "vctk_vctk"
VCTK_SPEAKERS: list[str] = ["p225", "p226", "p227", "p228", "p229"]
VCTK_FILE_TEMPLATE: str = "{spk}/{spk}_001.wav"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_to_peak_dbfs(x: np.ndarray, dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak < 1e-12:
        raise ValueError("Signal is silent; cannot normalise.")
    target = 10.0 ** (dbfs / 20.0)
    return (x * (target / peak)).astype(np.float32)


def _to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def _write(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), _to_pcm16(x), SR, subtype="PCM_16")


def _assert_spec(x: np.ndarray, name: str) -> None:
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    dur = len(x) / SR
    if rms < 0.05:
        raise RuntimeError(f"{name}: RMS {rms:.4f} < 0.05 (too quiet)")
    if peak > 1.0 + 1e-6:
        raise RuntimeError(f"{name}: peak {peak:.3f} > 1.0 (clip)")
    if dur < DURATION_S - 0.5:
        raise RuntimeError(f"{name}: duration {dur:.2f}s < {DURATION_S}s")


def _iir_formant(x: np.ndarray, freqs: list[float], q: float = 15.0,
                 sr: int = SR) -> np.ndarray:
    """Cascade of 2-pole resonators (one per formant)."""
    y = x.astype(np.float64).copy()
    for f0 in freqs:
        bw = f0 / q
        a = np.exp(-np.pi * bw / sr)
        b = np.sqrt(1.0 - a * a)
        w = 2.0 * np.pi * f0 / sr
        c = 2.0 * a * np.cos(w)
        a2 = a * a
        y_prev1 = 0.0
        y_prev2 = 0.0
        out = np.empty_like(y)
        for n in range(len(y)):
            val = b * y[n] + c * y_prev1 - a2 * y_prev2
            out[n] = val
            y_prev2, y_prev1 = y_prev1, val
        y = out
    return y.astype(np.float32)


def _highpass_80hz(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Simple 1-pole high-pass at fc=80 Hz."""
    rc = 1.0 / (2.0 * np.pi * 80.0)
    dt = 1.0 / sr
    alpha = rc / (rc + dt)  # standard 1-pole HPF coefficient
    y = np.empty_like(x, dtype=np.float32)
    prev_x = 0.0
    prev_y = 0.0
    for n in range(len(x)):
        cur = alpha * (prev_y + x[n] - prev_x)
        y[n] = cur
        prev_x, prev_y = x[n], cur
    return y


def _glottal_source(f0: float, n: int, sr: int = SR,
                    n_harmonics: int = 6) -> np.ndarray:
    """Glottal pulse train at F0 with `n_harmonics` decaying at 12 dB/oct."""
    # Build per-period exponential-decay pulse, then add harmonic enrichment
    # for the source excitation.
    period = int(round(sr / f0))
    src = np.zeros(n, dtype=np.float32)
    decay = int(0.001 * sr)  # 1 ms glottal pulse decay
    for k in range(0, n, period):
        end = min(k + decay, n)
        src[k:end] = np.exp(-np.linspace(0, 6.0, end - k)).astype(np.float32)
    # Add harmonics on top so the formant filter has rich content.
    # 12 dB/oct decay = amplitude ratio 1/sqrt(4)=0.5 per octave
    t = np.arange(n) / sr
    harmonic = np.zeros(n, dtype=np.float32)
    for h in range(1, n_harmonics + 1):
        amp = 1.0 / (h ** 1.0)  # -12 dB/octave  (h^1 -> 12 dB/oct for power spectral)
        harmonic += amp * np.sin(2.0 * np.pi * f0 * h * t)
    return (src + 0.3 * harmonic).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic voice path
# ---------------------------------------------------------------------------
def synthesize_voice(spec: dict, seed: int = 1234) -> np.ndarray:
    """Build a 30 s synthetic voice with the given F0 + formant spec."""
    f0 = spec["f0"]
    formants = spec["formants"]
    src = _glottal_source(f0, N, SR, n_harmonics=6)
    y = _iir_formant(src, formants, q=15.0)
    y = _highpass_80hz(y)
    rng = np.random.default_rng(seed + hash(spec["name"]) & 0xFFFF)
    noise = rng.standard_normal(N).astype(np.float32) * 0.01  # 1% noise
    out = y + noise
    return _normalize_to_peak_dbfs(out)


# ---------------------------------------------------------------------------
# VCTK download path
# ---------------------------------------------------------------------------
def _try_vctk() -> bool:
    """Attempt to download 5 VCTK speakers. Return True on success."""
    try:
        import librosa
        from huggingface_hub import hf_hub_download
    except Exception as e:  # noqa: BLE001
        print(f"  [vctk] cannot import deps: {e}", file=sys.stderr)
        return False
    ok_count = 0
    for i, spk in enumerate(VCTK_SPEAKERS):
        fname = VCTK_FILE_TEMPLATE.format(spk=spk)
        try:
            t0 = time.perf_counter()
            path = hf_hub_download(
                repo_id=VCTK_REPO_ID,
                filename=fname,
                repo_type="dataset",
            )
            wav, sr = sf.read(path, always_2d=False)
            if wav.ndim > 1:
                wav = wav[:, 0]
            if sr != SR:
                wav = librosa.resample(
                    wav.astype(np.float32), orig_sr=sr, target_sr=SR)
            wav = wav.astype(np.float32)
            target_n = int(SR * DURATION_S)
            if len(wav) < target_n:
                # pad if short
                pad = np.zeros(target_n - len(wav), dtype=np.float32)
                wav = np.concatenate([wav, pad])
            else:
                wav = wav[:target_n]
            wav = _normalize_to_peak_dbfs(wav)
            _assert_spec(wav, f"voice_{i}")
            _write(OUT_DIR / f"voice_{i}.wav", wav)
            print(f"  [vctk]  voice_{i}  ({spk})  "
                  f"{time.perf_counter()-t0:.1f}s  "
                  f"rms={float(np.sqrt(np.mean(wav*wav))):.3f}")
            ok_count += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [vctk]  {spk} failed: {type(e).__name__}: "
                  f"{str(e)[:120]}", file=sys.stderr)
    return ok_count == len(VCTK_SPEAKERS)


def _synthesize_all() -> None:
    """Synthesise 5 distinct synthetic voices."""
    for i, spec in enumerate(SYNTHETIC_VOICES):
        x = synthesize_voice(spec, seed=1234 + i * 17)
        _assert_spec(x, spec["name"])
        _write(OUT_DIR / f"voice_{i}.wav", x)
        rms = float(np.sqrt(np.mean(x * x)))
        print(f"  [synth] voice_{i}  ({spec['name']})  f0={spec['f0']:.0f}Hz  "
              f"formants={spec['formants']}  rms={rms:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating 5 voice reference wavs ({DURATION_S}s each) at {OUT_DIR}/")
    if _try_vctk():
        print("\n  VCTK download succeeded.")
        source = "vctk"
    else:
        print("\n  VCTK download failed/unavailable — using synthetic fallback.")
        _synthesize_all()
        source = "synthetic"
    # Final verification
    print(f"\n  Source: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
