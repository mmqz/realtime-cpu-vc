"""
modules/speaker_encoder.py — OpenVoice v2 ReferenceEncoder (256-d) wrapper
============================================================================
Source: myshell-ai/OpenVoice/models.py:301-359

Architecture: 6-layer 2D-conv + GRU(128) + Linear(256) on reference mel-spec,
              outputs a 256-d tone-color embedding per reference voice.

Why this replaces TinyVC's kNN-VC retrieval in v2 hybrid:
  v1 kNN: 5 voices × 1500 frames × 768-d × FP16 = 11.5 MB storage
  v2 ReferenceEncoder: 5 voices × 256-d × FP16 = 2.5 KB storage   (-4600×)
  Plus: per-inference compute saved (no cosine top-4 search)

Usage:
  enc = SpeakerEncoder('models/openvoice_ref_encoder.onnx')
  # Offline: encode 30 s reference wav -> 256-d vector
  se = enc.encode_reference(ref_mel_spec)  # [256]
  # Save as se_<voice_id>.pth
  torch.save(se, 'models/se_0.pth')
  # Runtime: load all 5 se vectors once
  enc.load_voice_registry('models/se_*.pth')
  # Per inference: lookup by voice_id
  se_target = enc.get_speaker_embedding(voice_id=0)
"""

import os
from glob import glob

import numpy as np
import onnxruntime as ort


class SpeakerEncoder:
    """OpenVoice v2 ReferenceEncoder wrapper (ORT).

    Takes mel-spec of reference audio (typically 30 s, 24 kHz, 80 mel bins)
    and outputs a 256-d speaker embedding. Used offline for registration
    and at runtime as a FiLM conditioning vector.
    """

    # OpenVoice v2 default ref encoder input shape:
    #   mel-spec [B, n_mels=80, T_ref_frames] at 200 Hz hop, 1024 n_fft
    OPENVOICE_SAMPLE_RATE = 22050  # OpenVoice v2 was trained at 22.05 kHz
    OPENVOICE_HOP = 256  # 200 Hz frame rate
    OPENVOICE_N_FFT = 1024
    OPENVOICE_N_MELS = 80

    def __init__(self, model_path: str, intra_op_threads: int = 2):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Speaker encoder ONNX not found at {model_path}. Export from OpenVoice v2 source."
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.voice_registry = {}  # voice_id -> 256-d np.float32 array

    def encode_reference(self, ref_mel_spec: np.ndarray) -> np.ndarray:
        """Run ReferenceEncoder on a mel-spec to get a 256-d embedding.

        Args:
            ref_mel_spec: [B, 80, T_frames] float32 at 200 Hz frame rate
                          (use OPENVOICE_HOP and OPENVOICE_N_MELS to compute)
        Returns:
            se: [256] float32 (or [B, 256] if batched)
        """
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: ref_mel_spec.astype(np.float32)})
        # ReferenceEncoder outputs a single 256-d tensor (after GRU final state + Linear)
        se = outputs[0]
        if se.ndim == 2 and se.shape[0] == 1:
            se = se[0]  # [256]
        return se

    def load_voice_registry(self, pattern: str = "models/se_*.pth"):
        """Load all se_*.pth files into self.voice_registry.

        File names should follow the convention se_<voice_id>.pth where voice_id
        is an integer. Total storage per voice: 256 × 2 bytes = 512 bytes (FP16)
        or 256 × 4 bytes = 1024 bytes (FP32).
        """
        import torch

        self.voice_registry.clear()
        for path in sorted(glob(pattern)):
            # Extract voice_id from filename: se_0.pth -> 0
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                voice_id = int(stem.split("_")[-1])
            except ValueError:
                continue
            se = torch.load(path).numpy().astype(np.float32)
            if se.shape == (256,):
                self.voice_registry[voice_id] = se
                print(f"  Loaded voice {voice_id}: {path}")
        print(f"  Voice registry: {len(self.voice_registry)} voices loaded")

    def get_speaker_embedding(self, voice_id: int) -> np.ndarray:
        """O(1) lookup of the 256-d embedding for the given voice.

        Args:
            voice_id: integer index from 0..N-1
        Returns:
            se: [256] float32
        """
        if voice_id not in self.voice_registry:
            raise KeyError(
                f"voice_id {voice_id} not in registry (loaded: {list(self.voice_registry.keys())})"
            )
        return self.voice_registry[voice_id]


def make_openvoice_mel_spec(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Compute mel-spec for OpenVoice v2 ReferenceEncoder input.

    OpenVoice uses 80 mel bins, 200 Hz frame rate, 1024-pt FFT, 22.05 kHz.

    Args:
        audio: [T_samples] float32 in [-1, 1] at 22050 Hz
        sr: sample rate, must be 22050 for OpenVoice v2
    Returns:
        mel_spec: [1, 80, T_frames] float32
    """
    import librosa

    if sr != SpeakerEncoder.OPENVOICE_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SpeakerEncoder.OPENVOICE_SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=SpeakerEncoder.OPENVOICE_SAMPLE_RATE,
        n_fft=SpeakerEncoder.OPENVOICE_N_FFT,
        hop_length=SpeakerEncoder.OPENVOICE_HOP,
        n_mels=SpeakerEncoder.OPENVOICE_N_MELS,
        fmin=0,
        fmax=8000,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # OpenVoice uses normalized log-mel
    mel_db = (mel_db + 80.0) / 20.0
    return mel_db[None, ...]  # [1, 80, T_frames]
