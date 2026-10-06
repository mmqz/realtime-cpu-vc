"""Verify generated test fixtures (source wavs + voice wavs).

Run:  python3 -m pytest tests/test_fixtures.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "data" / "source"
VOICE_DIR = ROOT / "data" / "voices"


# ---------------------------------------------------------------------------
# Source wavs (10 × 5 s × 24 kHz × PCM16 mono)
# ---------------------------------------------------------------------------
def test_source_wavs_exist():
    assert SOURCE_DIR.exists(), f"{SOURCE_DIR} does not exist"
    wavs = sorted(SOURCE_DIR.glob("source_*.wav"))
    assert len(wavs) == 10, f"Expected 10 source wavs, got {len(wavs)}"
    for w in wavs:
        info = sf.info(str(w))
        assert info.samplerate == 24000, f"{w.name}: sr {info.samplerate} != 24000"
        assert info.channels == 1, f"{w.name}: channels {info.channels} != 1"
        assert info.subtype == "PCM_16", (
            f"{w.name}: subtype {info.subtype} != PCM_16")
        assert info.duration >= 4.5, (
            f"{w.name}: duration {info.duration:.2f}s < 4.5s")


def test_source_wavs_rms_above_threshold():
    """RMS > 0.01 for all source wavs (no silent / near-silent fixtures)."""
    for w in sorted(SOURCE_DIR.glob("source_*.wav")):
        wav, _ = sf.read(str(w))
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32767.0
        rms = float(np.sqrt(np.mean(wav * wav)))
        assert rms > 0.01, f"{w.name}: RMS {rms:.4f} <= 0.01"


def test_source_wavs_peak_normalised():
    """Peak amplitude ~ -3 dBFS for all source wavs (0.707 ± 0.05)."""
    for w in sorted(SOURCE_DIR.glob("source_*.wav")):
        wav, _ = sf.read(str(w))
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32767.0
        peak = float(np.max(np.abs(wav)))
        assert 0.65 < peak <= 0.72, (
            f"{w.name}: peak {peak:.4f} not in [-3 dBFS ±0.05] range")


# ---------------------------------------------------------------------------
# Voice wavs (5 × 30 s × 24 kHz × PCM16 mono)
# ---------------------------------------------------------------------------
def test_voice_wavs_exist():
    assert VOICE_DIR.exists(), f"{VOICE_DIR} does not exist"
    wavs = sorted(VOICE_DIR.glob("voice_*.wav"))
    assert len(wavs) == 5, f"Expected 5 voice wavs, got {len(wavs)}"
    for w in wavs:
        info = sf.info(str(w))
        assert info.samplerate == 24000, f"{w.name}: sr {info.samplerate} != 24000"
        assert info.channels == 1, f"{w.name}: channels {info.channels} != 1"
        assert info.subtype == "PCM_16", (
            f"{w.name}: subtype {info.subtype} != PCM_16")
        assert info.duration >= 28.0, (
            f"{w.name}: duration {info.duration:.2f}s < 28s")


def test_voice_wavs_rms_above_threshold():
    """RMS > 0.05 for all voice wavs."""
    for w in sorted(VOICE_DIR.glob("voice_*.wav")):
        wav, _ = sf.read(str(w))
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32767.0
        rms = float(np.sqrt(np.mean(wav * wav)))
        assert rms > 0.05, f"{w.name}: RMS {rms:.4f} <= 0.05"


def test_voice_wavs_are_distinct():
    """Pairwise zero-lag correlation must be |corr| < 0.95
    (i.e. voices are audibly different, not duplicates)."""
    voices = sorted(VOICE_DIR.glob("voice_*.wav"))
    assert len(voices) == 5
    for i, v1 in enumerate(voices):
        for v2 in voices[i + 1:]:
            w1, _ = sf.read(str(v1))
            w2, _ = sf.read(str(v2))
            min_len = min(len(w1), len(w2))
            # correlation is scale- and mean-invariant, so int16 is fine here.
            a = w1[:min_len].astype(np.float64)
            b = w2[:min_len].astype(np.float64)
            a -= a.mean()
            b -= b.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            corr = 0.0 if denom < 1e-8 else float(np.dot(a, b) / denom)
            assert abs(corr) < 0.95, (
                f"{v1.name} and {v2.name} are too similar (corr={corr:.3f})")
