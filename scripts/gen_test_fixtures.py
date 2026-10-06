#!/usr/bin/env python3
"""Generate 10 synthetic source test wavs at 24 kHz mono PCM16.

Layout (5 s each, normalised to -3 dBFS, RMS > 0.01):
  data/source/source_001.wav  100 Hz pure sine (low male pitch)
  data/source/source_002.wav  150 Hz sine + 2nd harmonic
  data/source/source_003.wav  200 Hz sine + vibrato (modulated frequency)
  data/source/source_004.wav  250 Hz sine + white noise (SNR 20 dB)
  data/source/source_005.wav  300 Hz sine + formant-like filter (3-pole IIR)
  data/source/source_006.wav  3 harmonics (F0=120 / F1=240 / F2=360) + 5 Hz AM
  data/source/source_007.wav  100 Hz glottal pulse train + formant filter
                               (800 / 1200 / 2800 Hz)
  data/source/source_008.wav  filtered noise burst (mimicking unvoiced consonant)
  data/source/source_009.wav  2 s sustained "ah" vowel + 3 s silence
  data/source/source_010.wav  composite (vowel + noise + formant transitions)

All wavs MUST: 24 kHz mono, PCM16, -3 dBFS, RMS > 0.01.

Run:  python3 scripts/gen_test_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR: int = 24_000
DURATION_S: float = 5.0
N: int = int(SR * DURATION_S)
T: np.ndarray = np.arange(N) / SR  # time axis in seconds
TARGET_PEAK_DBFS: float = -3.0  # -3 dBFS peak


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_to_peak_dbfs(x: np.ndarray, dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
    """Scale x so its peak equals `dbfs` dBFS (relative to full-scale sine)."""
    peak = float(np.max(np.abs(x)))
    if peak < 1e-12:
        raise ValueError("Signal is silent; cannot normalise.")
    target_peak = 10.0 ** (dbfs / 20.0)
    return (x * (target_peak / peak)).astype(np.float32)


def _to_pcm16(x: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16 with safe clipping."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def _write(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), _to_pcm16(x), SR, subtype="PCM_16")


def _assert_spec(x: np.ndarray, name: str) -> None:
    """Validate the generated signal meets RMS + amplitude constraints."""
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    if rms < 0.01:
        raise RuntimeError(f"{name}: RMS {rms:.4f} < 0.01 (too quiet)")
    if peak > 1.0 + 1e-6:
        raise RuntimeError(f"{name}: peak {peak:.3f} > 1.0 (clip)")


def _iir_formant(x: np.ndarray, freqs: list[float], q: float = 15.0,
                 sr: int = SR) -> np.ndarray:
    """Cascade of 2-pole resonators (one per formant). State-space IIR."""
    y = x.astype(np.float64).copy()
    w = 2.0 * np.pi * np.array(freqs) / sr
    for f_idx in range(len(freqs)):
        # Each pole pair is a 2nd-order bandpass with centre `freqs[f_idx]`, Q.
        bw = freqs[f_idx] / q
        a = np.exp(-np.pi * bw / sr)
        b = np.sqrt(1.0 - a * a)  # gain so peak ~1 at centre
        # difference equation (transposed direct form II):
        #   y[n] = b * x[n] + 2 a cos(w) y[n-1] - a^2 y[n-2]
        c = 2.0 * a * np.cos(w[f_idx])
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


def _gauss_noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


# ---------------------------------------------------------------------------
# Source generators (1 per file)
# ---------------------------------------------------------------------------
def gen_001() -> np.ndarray:
    """100 Hz pure sine."""
    x = np.sin(2.0 * np.pi * 100.0 * T)
    return _normalize_to_peak_dbfs(x)


def gen_002() -> np.ndarray:
    """150 Hz sine + 2nd harmonic."""
    x = (0.7 * np.sin(2.0 * np.pi * 150.0 * T)
         + 0.3 * np.sin(2.0 * np.pi * 300.0 * T))
    return _normalize_to_peak_dbfs(x)


def gen_003() -> np.ndarray:
    """200 Hz sine with 5 Hz vibrato (±15 Hz frequency deviation)."""
    base_f = 200.0
    vibrato_rate = 5.0  # Hz
    vibrato_depth = 15.0  # Hz
    # instantaneous phase = ∫ 2π f(t) dt ; for vibrato this is analytic:
    inst_f = base_f + vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * T)
    phase = 2.0 * np.pi * np.cumsum(inst_f) / SR
    x = np.sin(phase)
    return _normalize_to_peak_dbfs(x)


def gen_004() -> np.ndarray:
    """250 Hz sine + white noise at 20 dB SNR."""
    sig = np.sin(2.0 * np.pi * 250.0 * T)
    noise = _gauss_noise(N, seed=7)
    # Scale noise so that signal-to-noise ratio (RMS) is 20 dB.
    sig_rms = float(np.sqrt(np.mean(sig * sig)))
    noise_rms = float(np.sqrt(np.mean(noise * noise)))
    desired_noise_rms = sig_rms / (10.0 ** (20.0 / 20.0))
    noise = noise * (desired_noise_rms / (noise_rms + 1e-12))
    x = sig + noise
    return _normalize_to_peak_dbfs(x)


def gen_005() -> np.ndarray:
    """300 Hz sine through a 3-pole IIR formant-like filter."""
    x = np.sin(2.0 * np.pi * 300.0 * T)
    # 3 formant poles
    x = _iir_formant(x, [800.0, 1500.0, 2500.0], q=12.0)
    return _normalize_to_peak_dbfs(x)


def gen_006() -> np.ndarray:
    """3 harmonics (120 / 240 / 360 Hz) with AM at 5 Hz."""
    base = (np.sin(2.0 * np.pi * 120.0 * T)
            + 0.6 * np.sin(2.0 * np.pi * 240.0 * T)
            + 0.36 * np.sin(2.0 * np.pi * 360.0 * T))
    am = 0.5 + 0.5 * np.sin(2.0 * np.pi * 5.0 * T)  # 0..1 AM at 5 Hz
    x = base * am
    return _normalize_to_peak_dbfs(x)


def gen_007() -> np.ndarray:
    """Glottal pulse train at 100 Hz + formant filter (800/1200/2800 Hz)."""
    # Build glottal pulse: short positive burst every 1/100 s, with 1ms decay.
    f0 = 100.0
    period = int(SR / f0)
    pulse = np.zeros(N, dtype=np.float32)
    decay = int(0.001 * SR)  # 1 ms decay
    for k in range(0, N, period):
        end = min(k + decay, N)
        rng_len = end - k
        # exponential decay pulse (LF model simplification)
        pulse[k:end] = np.exp(-np.linspace(0, 6.0, rng_len)).astype(np.float32)
    x = _iir_formant(pulse, [800.0, 1200.0, 2800.0], q=15.0)
    return _normalize_to_peak_dbfs(x)


def gen_008() -> np.ndarray:
    """Filtered noise burst (mimicking an unvoiced consonant)."""
    noise = _gauss_noise(N, seed=42)
    # Bandpass 1500-4000 Hz to mimic /s/ like frication
    from numpy.fft import fft, ifft
    spec = fft(noise)
    freqs = np.fft.fftfreq(N, 1.0 / SR)
    mask = (np.abs(freqs) >= 1500.0) & (np.abs(freqs) <= 4000.0)
    spec[~mask] = 0.0
    burst = np.real(ifft(spec)).astype(np.float32)
    # Make it bursty: 5 bursts of 500 ms
    envelope = np.zeros(N, dtype=np.float32)
    burst_dur = int(0.5 * SR)
    for start in range(0, N - burst_dur, int(1.0 * SR)):
        envelope[start:start + burst_dur] = (
            np.hanning(burst_dur) * 0.6 + 0.4 * np.hanning(burst_dur))
    x = burst * envelope
    return _normalize_to_peak_dbfs(x)


def gen_009() -> np.ndarray:
    """2 s sustained vowel-like 'ah' + 3 s silence."""
    sustained_n = int(2.0 * SR)
    t_short = np.arange(sustained_n) / SR
    # Glottal source at 110 Hz
    f0 = 110.0
    period = int(SR / f0)
    pulse = np.zeros(sustained_n, dtype=np.float32)
    decay = int(0.001 * SR)
    for k in range(0, sustained_n, period):
        end = min(k + decay, sustained_n)
        pulse[k:end] = np.exp(-np.linspace(0, 6.0, end - k)).astype(np.float32)
    vowel = _iir_formant(pulse, [800.0, 1150.0, 2900.0], q=15.0)
    silence = np.zeros(int(3.0 * SR), dtype=np.float32)
    x = np.concatenate([vowel, silence])
    # Normalise using only the vowel part (silence drags RMS down).
    nonzero = x[:sustained_n]
    peak = float(np.max(np.abs(nonzero)))
    target = 10.0 ** (-3.0 / 20.0)
    x = x * (target / (peak + 1e-12))
    return x.astype(np.float32)


def gen_010() -> np.ndarray:
    """Composite: vowel + noise + formant transitions."""
    f0 = 130.0
    period = int(SR / f0)
    pulse = np.zeros(N, dtype=np.float32)
    decay = int(0.001 * SR)
    for k in range(0, N, period):
        end = min(k + decay, N)
        pulse[k:end] = np.exp(-np.linspace(0, 6.0, end - k)).astype(np.float32)
    # Slowly varying formants (a->i->u transition over 5 s)
    out = np.zeros(N, dtype=np.float32)
    for seg_i in range(5):
        s = seg_i * (N // 5)
        e = (seg_i + 1) * (N // 5)
        # formant interpolations: a=(800,1200,2800), i=(300,2200,2900), u=(350,800,2600)
        formants = [
            ([800, 1200, 2800],  # 'a'
             [600, 1700, 2600]),  # midway
            ([600, 1700, 2600],
             [300, 2200, 2900]),  # 'i'
            ([300, 2200, 2900],
             [350, 800, 2600]),  # 'u'
            ([350, 800, 2600],
             [800, 1200, 2800]),  # back to 'a'
            ([800, 1200, 2800],
             [800, 1200, 2800]),  # hold 'a'
        ][seg_i]
        chunk = pulse[s:e]
        y = _iir_formant(chunk, formants[0], q=15.0)
        out[s:e] = y
    # Add background noise
    noise = _gauss_noise(N, seed=99) * 0.05
    x = out + noise
    return _normalize_to_peak_dbfs(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
GENS = [
    ("source_001.wav", gen_001),
    ("source_002.wav", gen_002),
    ("source_003.wav", gen_003),
    ("source_004.wav", gen_004),
    ("source_005.wav", gen_005),
    ("source_006.wav", gen_006),
    ("source_007.wav", gen_007),
    ("source_008.wav", gen_008),
    ("source_009.wav", gen_009),
    ("source_010.wav", gen_010),
]


def main() -> int:
    out_dir = Path("data/source")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for fname, fn in GENS:
        try:
            x = fn()
            _assert_spec(x, fname)
            _write(out_dir / fname, x)
            rms = float(np.sqrt(np.mean(x * x)))
            peak = float(np.max(np.abs(x)))
            print(f"  OK  {fname:<22}  rms={rms:.4f}  peak={peak:.3f}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERR {fname}: {e}", file=sys.stderr)
    if failures:
        print(f"\nFAILED: {failures}/{len(GENS)}", file=sys.stderr)
        return 1
    print(f"\nAll {len(GENS)} source fixtures generated at {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
