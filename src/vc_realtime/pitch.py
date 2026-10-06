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
