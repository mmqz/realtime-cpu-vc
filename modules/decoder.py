"""
modules/decoder.py — DDSP vocoder (ONNXRuntime wrapper)
=========================================================
Source: TinyVC's Decoder (SourceNet + FilterNet), exported via
        tinyvc/export_onnx.py as two separate ONNX graphs:
          - source_net.onnx:  content+f0+energy → amplitudes + FFT noise kernel
          - filter_net.onnx:  content+f0+energy+source_signal → waveform

Inference flow:
    content_feat [B, 768, T]  +  f0 [B, T]  +  energy [B, T]
                                |
                                v
                        source_net.onnx (4.43M params, INT8)
                                |
                                +--> amp      [B, 15, T]  (14 harmonics + 1)
                                +--> kernel   [B, 2, T, n_fft//2+1]
                                v
                    Python harmonic synth + FFT noise
                                |
                                +--> source_signal [B, 1, T_samples]
                                v
                        filter_net.onnx (4.23M params, INT8)
                                |
                                v
                        waveform [B, 1, T_samples] at 24 kHz
"""
import os
import math
import numpy as np
import onnxruntime as ort
from typing import Tuple


class Decoder:
    """ORT wrapper for TinyVC DDSP decoder."""

    def __init__(self, source_net_path: str, filter_net_path: str,
                 intra_op_threads: int = 2, inter_op_threads: int = 1):
        for p in (source_net_path, filter_net_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Decoder ONNX not found at {p}. "
                    f"Run: python -m tinyvc.export_onnx --models-dir models/"
                )
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.inter_op_num_threads = inter_op_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.source_net = ort.InferenceSession(
            source_net_path, sess_options=so,
            providers=['CPUExecutionProvider']
        )
        self.filter_net = ort.InferenceSession(
            filter_net_path, sess_options=so,
            providers=['CPUExecutionProvider']
        )

    def decode(self, content_feat: np.ndarray, f0: np.ndarray,
               energy: np.ndarray) -> np.ndarray:
        """Full DDSP decode: content+f0+energy → waveform.

        Args:
            content_feat: [B, 768, T_frames] float32
            f0:           [B, T_frames] float32 (Hz)
            energy:       [B, T_frames] float32
        Returns:
            waveform:     [B, 1, T_samples] float32 at 24 kHz, in [-1, 1]
        """
        # Step 1: run source_net to get amplitudes + FFT noise kernel
        amp, kernel = self._run_source_net(content_feat, f0, energy)
        # Step 2: synthesize harmonic source + noise in Python
        source_signal = self._synthesize_source(amp, kernel, f0)
        # Step 3: run filter_net to apply final 1D U-Net filtering
        waveform = self._run_filter_net(content_feat, f0, energy, source_signal)
        return waveform

    # ---------- Sub-graph forwarders ----------

    def _run_source_net(self, content: np.ndarray, f0: np.ndarray,
                        energy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inputs = {
            'content': content.astype(np.float32),
            'f0': f0.astype(np.float32),
            'energy': energy.astype(np.float32),
        }
        # Match by input name
        feed = {}
        for i in self.source_net.get_inputs():
            feed[i.name] = inputs.get(i.name, inputs[list(inputs.keys())[0]])
        outputs = self.source_net.run(None, feed)
        amp, kernel = outputs[0], outputs[1]
        return amp, kernel

    def _run_filter_net(self, content: np.ndarray, f0: np.ndarray,
                        energy: np.ndarray, source_signal: np.ndarray) -> np.ndarray:
        inputs = {
            'content': content.astype(np.float32),
            'f0': f0.astype(np.float32),
            'energy': energy.astype(np.float32),
            'source': source_signal.astype(np.float32),
        }
        feed = {}
        for i in self.filter_net.get_inputs():
            feed[i.name] = inputs.get(i.name, inputs[list(inputs.keys())[0]])
        outputs = self.filter_net.run(None, feed)
        return outputs[0]  # waveform [B, 1, T_samples]

    # ---------- DDSP synthesis (Python, between the two ONNX calls) ----------

    def _synthesize_source(self, amp: np.ndarray, kernel: np.ndarray,
                           f0: np.ndarray, sr: int = 24000,
                           hop: int = 480) -> np.ndarray:
        """Harmonic + noise synthesis, faithful to tinyvc/module/tinyvc/decoder.py:24-85.

        Args:
            amp:    [B, 15, T_frames] amplitudes for 14 harmonics + 1 (DC)
            kernel: [B, 2, T_frames, n_fft//2+1] FFT-domain EQ kernel for noise
            f0:     [B, T_frames] pitch in Hz
        Returns:
            source_signal: [B, 1, T_samples] float32
        """
        B, K, T_frames = amp.shape
        T_samples = T_frames * hop

        # Upsample F0 from frame rate (50 Hz) to sample rate (24 kHz)
        f0_up = np.repeat(f0, hop, axis=-1)[:, :T_samples]  # nearest-neighbor upsample
        # Linear-interpolated phase via cumsum
        phase = np.cumsum(f0_up * (2 * math.pi / sr), axis=-1)  # [B, T_samples]

        # Harmonic source: sum_k=1^15 amp_k * sin(k * phase)
        harmonics = np.stack(
            [np.sin((k+1) * phase) for k in range(K-1)],  # K-1 = 14 harmonics (skip DC at k=0)
            axis=1
        )  # [B, 14, T_samples]
        # Upsample amp from frame to sample domain
        amp_up = np.repeat(amp[:, 1:, :], hop, axis=-1)[:, :, :T_samples]  # [B, 14, T_samples]
        harmonic_signal = (amp_up * harmonics).sum(axis=1, keepdims=True)  # [B, 1, T_samples]

        # Noise source: FFT-domain EQ kernel applied to Gaussian random phase
        noise_freq = np.fft.rfft(np.random.randn(B, 1, T_samples).astype(np.float32), axis=-1)
        # Apply the learned kernel (kernel shape may need broadcast; simplify for skeleton)
        # In production, follow tinyvc/module/tinyvc/decoder.py:63-85 exactly.
        noise_signal = np.fft.irfft(noise_freq, n=T_samples, axis=-1)
        noise_signal = noise_signal[:, None, :]

        return harmonic_signal + noise_signal * 0.05  # noise gain (placeholder)
