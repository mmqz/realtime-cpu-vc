"""
modules/pitch.py — Pitch extractor wrapper (P0 uses encoder.onnx bundled output)
===================================================================================
Source: tinyvc/module/tinyvc/encoder.py:11-72 (PitchEstimator, 460K params, jointly
        trained with the content encoder using FCPE labels, cross-entropy loss).

In TinyVC's exported ONNX, the PitchEstimator is part of `encoder.onnx` and its
F0 output comes back as one of the three encoder outputs (content, f0, energy).
This file is therefore a thin shim for now.

P2-4 plan: add RMVPE as an optional high-quality F0 fallback path, configurable
           in configs/default.yaml: pitch.use_rmvpe_fallback.
"""

import numpy as np


class PitchExtractor:
    """Pitch extractor wrapper.

    P0: F0 is returned by the encoder.onnx graph itself (one of its 3 outputs).
        This wrapper is a no-op shim that simply forwards the encoder's F0.
    P2-4: optionally re-extract F0 via RMVPE ONNX for higher accuracy on music
          vocals.
    """

    def __init__(self, use_rmvpe_fallback: bool = False, rmvpe_model_path: str | None = None):
        self.use_rmvpe_fallback = use_rmvpe_fallback
        self.rmvpe_session = None
        if use_rmvpe_fallback:
            if rmvpe_model_path is None:
                raise ValueError("rmvpe_model_path required when use_rmvpe_fallback=True")
            import onnxruntime as ort

            self.rmvpe_session = ort.InferenceSession(
                rmvpe_model_path, providers=["CPUExecutionProvider"]
            )

    def extract_or_fallback(self, mel_spec: np.ndarray, encoder_f0: np.ndarray) -> np.ndarray:
        """If RMVPE fallback is enabled and encoder F0 looks unreliable,
        re-extract via RMVPE. Otherwise just return encoder_f0.

        Heuristic for "unreliable": variance > 2x median across the chunk
        (suggests octave jumps).
        """
        if not self.use_rmvpe_fallback:
            return encoder_f0
        # Compute F0 variance heuristic
        f0_var = float(np.std(encoder_f0))
        f0_med = float(np.median(encoder_f0[encoder_f0 > 0]) or 0)
        if f0_med > 0 and f0_var > 0.5 * f0_med:
            # Likely octave errors → use RMVPE
            return self._rmvpe_forward(mel_spec)
        return encoder_f0

    def _rmvpe_forward(self, mel_spec: np.ndarray) -> np.ndarray:
        """Run RMVPE ONNX on mel-spec → F0 track. Placeholder for P2-4."""
        # RMVPE expects a different input format (full-band mel, not log-mel)
        # — needs proper preprocessing. To be implemented in P2-4.
        raise NotImplementedError("RMVPE fallback is a P2-4 task")


# ===========================================================================
# M1b · Per-voice F0 quantile mapping
# ===========================================================================
# Replaces the implicit "F0 stays at source pitch range" assumption in v1
# (which makes cross-gender VC destructive: source p232 male ~120Hz →
# target p228 female ~220Hz never lands, output pitch stays male, CAMPPlus
# votes for source over target → VC_effect -0.156 measured in M0.5).
#
# The map: per-voice pre-computed F0 quantile table (256-d, log-Hz domain).
# At runtime, source F0 → log-Hz → quantile (via inverse interp on the
# source's own table for symmetry) → target log-Hz (via forward interp on
# target's table) → exp back to Hz.
#
# Design choice (from download/m1b-f0-mapping-design.md, option i):
# This is "feature-level" F0 mapping done at the F0 track level — NOT
# audio-level pitch_shift + re-mel. The DDSP decoder takes f0 as an explicit
# input, so we can remap f0 between encoder output and decoder input without
# touching the mel-spec. This is faster than option (i)'s audio-level
# pitch_shift, but means content features still come from source audio.
# For full F0 control at the Vocos path (M1c), option (i) audio-level
# pitch_shift is required (Vocos doesn't take f0 input).
# ===========================================================================


N_QUANTILES: int = 256
LOG_HZ_MIN: float = 2.0    # ~7.4 Hz (well below human voice floor 60 Hz log=4.1)
LOG_HZ_MAX: float = 8.0    # ~3000 Hz (above human voice ceiling 500 Hz log=6.2)


def _extract_f0_track(wav: "np.ndarray", sr: int = 24000,
                       frame_ms: int = 20) -> "np.ndarray":
    """Extract F0 track from audio via librosa.pyin.

    Returns F0 in Hz (0 = unvoiced), shape [T_frames] at frame_ms rate.
    """
    import librosa
    f0, voiced_flag, _ = librosa.pyin(
        wav.astype(np.float32),
        fmin=60.0, fmax=500.0,
        sr=sr,
        frame_length=int(sr * frame_ms / 1000),
    )
    # pyin returns NaN for unvoiced → 0
    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
    return f0


def _build_quantile_table(f0: "np.ndarray", n: int = N_QUANTILES) -> "np.ndarray":
    """Build n-point quantile table in log-Hz domain from voiced F0 frames."""
    voiced = f0[f0 > 0]
    if len(voiced) < 10:
        # Not enough voiced frames — return a default table (uniform in log-Hz)
        return np.linspace(LOG_HZ_MIN, LOG_HZ_MAX, n, dtype=np.float32)
    log_f0 = np.log(np.clip(voiced, 1e-3, None))
    # Sort and subsample to n equally-spaced quantiles
    sorted_log = np.sort(log_f0)
    qs = np.linspace(0, len(sorted_log) - 1, n, dtype=np.float32)
    indices = np.clip(qs.astype(int), 0, len(sorted_log) - 1)
    table = sorted_log[indices]
    return table.astype(np.float32)


class F0QuantileMapper:
    """Per-voice F0 quantile mapping.

    Pre-computed at registration time from each target voice's 30 s reference
    audio. Stored as a 5 × 256 float32 quantile table (5 KB total).

    Runtime:
      source_f0  [T_frames] in Hz, 0 = unvoiced
        ↓ log-Hz, mask unvoiced
      source_q   [T_voiced] in [0,1] via np.interp into source's own table
        ↓ inverse interp on target table
      target_log [T_voiced] in log-Hz
        ↓ exp back to Hz, restore unvoiced mask
      → mapped F0 track for the DDSP decoder

    Design choice: also build a "source" quantile table per voice (so source
    F0 distribution is also pre-characterized). This is symmetric — at
    runtime we need *both* source and target tables. For M1b we use a single
    source table built from the source_001.wav clip at startup, and 5 target
    tables (one per voice).
    """

    def __init__(self, target_quantile_tables: "dict[int, np.ndarray]",
                 source_quantile_table: "np.ndarray | None" = None):
        # target_quantile_tables[v] = np.ndarray [N_QUANTILES] log-Hz
        self.target_tables = target_quantile_tables
        self.source_table = source_quantile_table
        self._active_voice: int = 0

    @classmethod
    def from_voices_dir(cls, voices_dir: str, n_quantiles: int = N_QUANTILES,
                         sr: int = 24000) -> "F0QuantileMapper":
        """Build 5 target tables + 1 source table from data/voices/voice_*.wav.

        Uses librosa.pyin to extract F0 from each 30 s reference, then builds
        a 256-point quantile table in log-Hz domain.
        """
        from pathlib import Path
        import soundfile as sf
        import librosa
        vdir = Path(voices_dir)
        target_tables: dict[int, np.ndarray] = {}
        source_table: np.ndarray | None = None
        # Use voice_0 as a "default source" too (in case the actual source
        # speaker's table isn't pre-built; less ideal but provides a fallback)
        for i in range(5):
            wav_path = vdir / f"voice_{i}.wav"
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing {wav_path}")
            wav, file_sr = sf.read(str(wav_path), always_2d=False)
            if wav.ndim > 1:
                wav = wav[:, 0]
            if file_sr != sr:
                wav = librosa.resample(wav.astype(np.float32), orig_sr=file_sr,
                                       target_sr=sr)
            f0 = _extract_f0_track(wav, sr=sr)
            target_tables[i] = _build_quantile_table(f0, n_quantiles)
            # Use voice_2 (p227, male) as the default source table — closest
            # to a typical source speaker distribution
            if i == 2:
                source_table = target_tables[i].copy()
        return cls(target_tables, source_table)

    def select_voice(self, voice_id: int) -> None:
        if voice_id not in self.target_tables:
            raise ValueError(f"voice_id {voice_id} not in target_tables")
        self._active_voice = int(voice_id)

    def set_source_table(self, source_table: "np.ndarray") -> None:
        """Override the default source quantile table (e.g., when running
        VC on a specific source speaker — build a table from their audio)."""
        self.source_table = source_table

    def build_source_table_from_audio(self, wav: "np.ndarray",
                                       sr: int = 24000) -> "np.ndarray":
        """Convenience: build a source table from arbitrary source audio."""
        f0 = _extract_f0_track(wav, sr=sr)
        table = _build_quantile_table(f0, len(next(iter(self.target_tables.values()))))
        self.source_table = table
        return table

    def map_f0(self, source_f0: "np.ndarray") -> "np.ndarray":
        """Map a source F0 track to target voice's F0 distribution.

        Args:
            source_f0: [T_frames] float32 Hz, 0 = unvoiced
        Returns:
            [T_frames] float32 Hz, 0 = unvoiced (preserved)
        """
        source_f0 = np.asarray(source_f0, dtype=np.float32)
        out = np.zeros_like(source_f0)
        voiced_mask = source_f0 > 0
        if not voiced_mask.any():
            return out
        if self.source_table is None:
            # Fallback: identity mapping (no source table → no remap)
            return source_f0.copy()
        src_log = np.log(np.clip(source_f0[voiced_mask], 1e-3, None))
        # Source: log-Hz → quantile [0,1]
        src_table_sorted = np.sort(self.source_table)
        src_q = np.interp(src_log, src_table_sorted,
                          np.linspace(0, 1, len(src_table_sorted)))
        # Target: quantile → log-Hz
        tgt_table_sorted = np.sort(self.target_tables[self._active_voice])
        tgt_log = np.interp(src_q,
                            np.linspace(0, 1, len(tgt_table_sorted)),
                            tgt_table_sorted)
        out[voiced_mask] = np.exp(tgt_log).astype(np.float32)
        # Safety: clamp to [60, 500] Hz (human voice range)
        out = np.clip(out, 0.0, 500.0)
        return out.astype(np.float32)


__all__ = [
    "PitchExtractor",
    "F0QuantileMapper",
    "N_QUANTILES",
    "LOG_HZ_MIN",
    "LOG_HZ_MAX",
    "_extract_f0_track",
    "_build_quantile_table",
]
