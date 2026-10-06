"""
modules/vocoder_v2.py — F5-TTS Vocos vocoder (ONNX) wrapper for v2 hybrid
==========================================================================
Source: SWivid/F5-TTS/runtime/triton_trtllm/scripts/export_vocoder_to_onnx.py
        F5-TTS Vocos uses ISTFTHead (inverse STFT head) — outputs magnitude
        + phase directly, then iSTFT to waveform. 24 kHz / 100-mel / 100 Hz
        frame rate. ~13M params, ONNX opset 17.

v2 hybrid role:
  Replaces TinyVC's DDSP SourceNet + FilterNet (4.66M, INT8 ~5MB).
  Vocos is heavier (13M, INT8 ~7MB) but produces significantly richer audio
  thanks to its ISTFT-head design (compared to DDSP's additive sine synthesis
  which sounds thin).

Quality: F5-TTS paper reports UTMOS 4.5 (vs GT 4.6) — among the best published
         TTS/VC decoders. Compared to DDSP: MOS +0.2-0.3.

Usage:
  vocoder = VocosV2('models/vocos.onnx', 'models/vocos.int8.onnx')
  # At inference: feed content features [B, 100, T_mel_frames] -> waveform
  waveform = vocoder.decode(mel_pred)
"""
import os
import numpy as np
import onnxruntime as ort


class VocosV2:
    """F5-TTS Vocos vocoder ONNX wrapper.

    Input:  mel-spec (predicted, [B, 100, T_mel_frames] at 100 Hz frame rate)
    Output:  waveform [B, 1, T_samples] at 24 kHz

    Note: The TinyVC content encoder outputs at 50 Hz frame rate, but Vocos
    expects 100 Hz. We need an upsample step before Vocos:
      1. Repeat/interpolate content features 2× along the time axis
      2. Run Vocos on the upsampled features
    OR alternatively, train Vocos at 50 Hz (smaller dataset, faster).
    P0/P1 uses option (1) — simple linear interpolation.
    """

    VOCOS_SAMPLE_RATE = 24000
    VOCOS_FRAME_RATE = 100   # Hz — Vocos is trained at 100 Hz
    VOCOS_N_MELS = 100       # mel bins

    def __init__(self, model_path: str, intra_op_threads: int = 2,
                 use_int8: bool = True):
        """Load Vocos ONNX.

        Args:
            model_path: path to vocos.onnx or vocos.int8.onnx
            intra_op_threads: ORT intra-op thread count
            use_int8: if True, prefer the .int8.onnx variant
        """
        # Auto-prefer INT8 variant if available
        if use_int8:
            int8_path = model_path.replace('.onnx', '.int8.onnx')
            if os.path.exists(int8_path):
                model_path = int8_path
                print(f"  VocosV2: using INT8 variant {int8_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Vocos ONNX not found at {model_path}. "
                f"Export from F5-TTS source via "
                f"runtime/triton_trtllm/scripts/export_vocoder_to_onnx.py"
            )

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=so,
            providers=['CPUExecutionProvider']
        )

    def decode(self, mel_pred: np.ndarray) -> np.ndarray:
        """Run Vocos on predicted mel-spec to produce waveform.

        Args:
            mel_pred: [B, 100, T_mel_frames] float32 — predicted mel features
                      (from F5-TTS or from a mel-projection of TinyVC content)
        Returns:
            waveform: [B, 1, T_samples] float32 at 24 kHz
        """
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: mel_pred.astype(np.float32)})
        # Vocos outputs waveform directly via ISTFTHead
        return outputs[0]  # [B, 1, T_samples]

    def upsample_50hz_to_100hz(self, content_feat: np.ndarray) -> np.ndarray:
        """Linear interpolation 50 Hz → 100 Hz to match Vocos frame rate.

        Args:
            content_feat: [B, 768, T_50hz] — TinyVC encoder output
        Returns:
            upsampled: [B, 768, T_100hz] — interpolated
        """
        # Simple linear interpolation via numpy
        B, C, T = content_feat.shape
        # Use np.interp along time axis
        t_in = np.linspace(0, 1, T)
        t_out = np.linspace(0, 1, T * 2 - 1)
        out = np.zeros((B, C, T * 2 - 1), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                out[b, c] = np.interp(t_out, t_in, content_feat[b, c])
        return out
