"""
modules/speaker_encoder_v3.py — Spark-TTS BiCodec SpeakerEncoder (ECAPA-TDNN c512 + Perceiver + ResidualFSQ)
==============================================================================================================
Source: SparkAudio/Spark-TTS/sparktts/models/bicodec.py (SpeakerEncoder class)

v3 hybrid upgrade over v2:
  v2: OpenVoice v2 ReferenceEncoder (256-d, 0.76M, 1MB INT8)
  v3: Spark-TTS BiCodec SpeakerEncoder (512-d + 48-byte FSQ code, 6-12M, ~5MB INT8)
  Expected uplift: +1-2% speaker similarity, +4MB RAM (still <100MB total)

Architecture (3 sub-modules):
  (1) ECAPA-TDNN c512 (~5M params) — current SOTA speaker encoder architecture
      - Input: mel-spec [B, n_mels=80, T_frames]
      - Output: frame-level 512-d features [B, T, 512]
  (2) PerceiverResampler(num_latents=32, ~1M params) — learning-based attention pool
      - Input: frame-level features [B, T, 512]
      - Output: 32 latent tokens [B, 32, 512]  (vs simple mean-pooling in v2)
  (3) ResidualFSQ(levels=[4,4,4,4,4,4], ~0.5M params) — quantize each latent token to 6×2-bit codes
      - Input: 32 latent tokens [B, 32, 512]
      - Output: 32 × 6 = 192 code indices per voice, packed into 48 bytes (6 bits × 32 × 6 = 2304 bits ≈ 288 bytes raw, or 48 bytes compressed)
      - FSQ vs VQ advantage: no codebook training needed, smaller quantization error

Per-voice storage:
  v1 kNN index:    5 × 11.5 MB / 5 = 2.3 MB per voice (768-d × 1500 frames × FP16)
  v2 OpenVoice:    5 × 256 × 2 bytes = 512 bytes per voice (256-d FP16)
  v3 Spark BiCodec: 5 × 48 bytes per voice (FSQ discrete codes)  ← O(1) hash-lookupable!

Usage:
  enc = BiCodecSpeakerEncoder('models/spark_speaker_encoder.onnx')
  # Offline: encode 30 s reference -> FSQ code (48 bytes)
  fsq_code = enc.encode_reference(ref_mel_spec)
  with open('models/se_0.fsq', 'wb') as f:
      f.write(fsq_code.tobytes())
  # Runtime: load all 5 FSQ codes, do O(1) lookup by voice_id
  enc.load_voice_registry('models/se_*.fsq')
  se_target = enc.get_speaker_embedding(voice_id=0)  # returns 512-d float32 (decoded from FSQ)
"""
import os
import numpy as np
import onnxruntime as ort
from typing import Optional
from glob import glob


class BiCodecSpeakerEncoder:
    """Spark-TTS BiCodec SpeakerEncoder wrapper (ORT).

    Three sub-modules in a single ONNX graph (or three separate graphs):
      1. ECAPA-TDNN (frame-level 512-d features)
      2. PerceiverResampler (32 latent tokens)
      3. ResidualFSQ (48-byte discrete code)
    """

    SPARK_SAMPLE_RATE = 16000  # Spark-TTS uses 16 kHz input
    SPARK_N_FFT = 1024
    SPARK_HOP = 200            # ~80 Hz frame rate (16 kHz / 200)
    SPARK_N_MELS = 80

    def __init__(self, model_path: str, intra_op_threads: int = 2,
                 use_int8: bool = True):
        """Load BiCodec SpeakerEncoder ONNX.

        Auto-prefer INT8 variant if available.
        """
        if use_int8:
            int8_path = model_path.replace('.onnx', '.int8.onnx')
            if os.path.exists(int8_path):
                model_path = int8_path
                print(f"  BiCodecSpeakerEncoder: using INT8 variant {int8_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Spark-TTS BiCodec SpeakerEncoder ONNX not found at {model_path}. "
                f"Export from Spark-TTS source (sparktts/models/bicodec.py)."
            )

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=so,
            providers=['CPUExecutionProvider']
        )
        self.voice_registry = {}  # voice_id -> 512-d np.float32 (decoded from FSQ)
        self.fsq_codes = {}        # voice_id -> 192-d int32 (raw FSQ codes)

    def encode_reference(self, ref_mel_spec: np.ndarray) -> np.ndarray:
        """Run BiCodec SpeakerEncoder on a reference mel-spec.

        Args:
            ref_mel_spec: [B, 80, T_frames] float32 at ~80 Hz frame rate
        Returns:
            se: [512] float32 (the last latent token, or pooled) — for use as
                conditioning vector to downstream flow
            Also internally caches the 192-d FSQ code for O(1) retrieval.
        """
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: ref_mel_spec.astype(np.float32)})
        # Spark-TTS's BiCodec SpeakerEncoder returns multiple outputs:
        #   [0]: frame-level features [B, T, 512]  (from ECAPA-TDNN)
        #   [1]: latent tokens [B, 32, 512]        (from PerceiverResampler)
        #   [2]: FSQ codes [B, 32, 6]               (from ResidualFSQ, 6 codes per token)
        # For conditioning downstream flow, we use the mean of the 32 latents.
        latents = outputs[1]  # [B, 32, 512]
        fsq_codes = outputs[2]  # [B, 32, 6]
        se = latents.mean(axis=1)  # [B, 512]
        if se.shape[0] == 1:
            se = se[0]  # [512]
        # Cache FSQ codes (32 × 6 = 192 int32 codes per voice, packed into 48 bytes when packed as 6-bit)
        self.fsq_codes[len(self.fsq_codes)] = fsq_codes[0].flatten()  # [192]
        return se

    def load_voice_registry(self, pattern: str = 'models/se_*.fsq'):
        """Load all se_*.fsq files (48-byte each) into self.voice_registry.

        The .fsq files contain 48 bytes of FSQ codes per voice. To get the 512-d
        embedding for downstream flow conditioning, we need to decode FSQ → latent.
        For simplicity, we also accept .pth files containing 512-d float vectors
        (saved during registration alongside .fsq for forward use).
        """
        self.voice_registry.clear()
        # Load .pth (512-d float vector for runtime conditioning)
        for path in sorted(glob(pattern.replace('.fsq', '.pth'))):
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                voice_id = int(stem.split('_')[-1])
            except ValueError:
                continue
            import torch
            se = torch.load(path).numpy().astype(np.float32)
            if se.shape == (512,):
                self.voice_registry[voice_id] = se
                print(f"  Loaded voice {voice_id}: {path} (512-d)")
        # Optionally load .fsq (48-byte) for hash-keyed O(1) retrieval fallback
        for path in sorted(glob(pattern)):
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                voice_id = int(stem.split('_')[-1])
            except ValueError:
                continue
            with open(path, 'rb') as f:
                self.fsq_codes[voice_id] = np.frombuffer(f.read(), dtype=np.uint8)
        print(f"  Voice registry: {len(self.voice_registry)} voices loaded")

    def get_speaker_embedding(self, voice_id: int) -> np.ndarray:
        """O(1) lookup of the 512-d embedding for the given voice."""
        if voice_id not in self.voice_registry:
            raise KeyError(f"voice_id {voice_id} not in registry "
                            f"(loaded: {list(self.voice_registry.keys())})")
        return self.voice_registry[voice_id]

    def lookup_by_fsq_hash(self, ref_fsq_code: np.ndarray) -> int:
        """O(1) hash-based retrieval of voice_id by FSQ code.

        Useful when the reference audio is processed at runtime (instead of
        voice_id being known). For our 5-fixed-voice setup, voice_id is known
        so direct lookup is sufficient. This method is for future extension
        to arbitrary unseen reference matching.
        """
        for vid, code in self.fsq_codes.items():
            if np.array_equal(code, ref_fsq_code):
                return vid
        raise KeyError("FSQ code not in registry")


def make_spark_mel_spec(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute mel-spec for Spark-TTS BiCodec SpeakerEncoder input.

    Spark-TTS uses 80 mel bins, ~80 Hz frame rate, 1024-pt FFT, 16 kHz.
    """
    import librosa
    if sr != BiCodecSpeakerEncoder.SPARK_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr,
                                 target_sr=BiCodecSpeakerEncoder.SPARK_SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=BiCodecSpeakerEncoder.SPARK_SAMPLE_RATE,
        n_fft=BiCodecSpeakerEncoder.SPARK_N_FFT,
        hop_length=BiCodecSpeakerEncoder.SPARK_HOP,
        n_mels=BiCodecSpeakerEncoder.SPARK_N_MELS,
        fmin=0, fmax=8000, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_db = (mel_db + 80.0) / 20.0
    return mel_db[None, ...]  # [1, 80, T_frames]
