"""
modules/encoder.py — Content + F0 encoder (ONNXRuntime wrapper)
================================================================
Source: TinyVC's SSLFeatureEstimator + PitchEstimator, exported via
        tinyvc/export_onnx.py as encoder.onnx (opset 17, dynamic axes).

Inference flow:
    mel-spec [B, n_mels=128, T_frames]
        |
        v
    encoder.onnx (ConvNeXt-v2 6-layer, 4.7M params, INT8)
        |
        +--> content_feat [B, 768, T_frames]
        +--> f0          [B, T_frames]  (PitchEstimator 460K params)
        +--> energy      [B, T_frames]

Note: in TinyVC's exported ONNX, content + F0 + energy are returned as
multiple outputs of a single graph. This wrapper decodes them.
"""
import os
import numpy as np
import onnxruntime as ort
from typing import Tuple


class Encoder:
    """ORT wrapper for TinyVC encoder + pitch + energy.

    The exported ONNX expects mel-spec input with shape [B, n_mels=128, T_frames]
    where T_frames = audio_samples / 480 (hop=480 at 24 kHz).
    """

    def __init__(self, model_path: str, intra_op_threads: int = 2,
                 inter_op_threads: int = 1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Encoder ONNX not found at {model_path}. "
                f"Run: python -m tinyvc.export_onnx --models-dir models/"
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.inter_op_num_threads = inter_op_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=so,
            providers=['CPUExecutionProvider']
        )

    def encode(self, mel_spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run content + F0 + energy extraction.

        Args:
            mel_spec: float32 or float16 array, shape [B, 128, T_frames]
        Returns:
            content_feat: [B, 768, T_frames]
            f0:           [B, T_frames]
            energy:       [B, T_frames]
        """
        # TinyVC's export expects a single input "spec" with shape [B, 128, T]
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: mel_spec.astype(np.float32)})
        # Output order: content, f0, energy (matches tinyvc/export_onnx.py)
        content_feat, f0, energy = outputs[0], outputs[1], outputs[2]
        return content_feat, f0, energy


def make_mel_spec(audio: np.ndarray, sr: int = 24000, n_fft: int = 1920,
                  hop: int = 480, n_mels: int = 128) -> np.ndarray:
    """Compute mel-spectrogram for TinyVC encoder input.

    Args:
        audio: [B, T_samples] float32 in [-1, 1]
        sr: sample rate, default 24000
        n_fft: STFT window, default 1920 (80 ms)
        hop: STFT hop, default 480 (20 ms, 50 Hz frame rate)
        n_mels: mel bands, default 128

    Returns:
        mel_spec: [B, n_mels, T_frames] float32
    """
    import librosa
    # librosa.power_to_db gives the log-mel TinyVC expects
    mel = librosa.feature.melspectrogram(
        y=audio.squeeze().astype(np.float32), sr=sr,
        n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        fmin=20, fmax=12000, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # normalize to roughly [-4, +4] range
    mel_db = (mel_db + 80.0) / 20.0
    return mel_db[None, ...]  # add batch dim
