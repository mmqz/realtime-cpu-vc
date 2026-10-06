"""
modules/streaming.py — Real-time streaming shell (PyAudio + ORT)
===================================================================
Source: tinyvc/infer_streaming.py + tinyvc/module/infer/stream.py:30-96

P1 design: SOLA crossfade with torch.roll ring buffer + ORT session
P2 plan:  replace ring buffer with PocketTTS StreamingConv1d + KV-cache
          (borrowed from pocket_tts/modules/conv.py:86-117)
"""
import numpy as np
import pyaudio
from typing import Optional
from collections import deque
import webrtcvad

from .encoder import Encoder, make_mel_spec
from .decoder import Decoder
from .knn_retrieval import KNNRetrieval


class StreamingInfer:
    """Real-time VC streaming inference.

    Audio flow:
        PyAudio mic callback -> ring buffer
        -> WebRTC VAD -> if speech: Encoder -> KNN -> Decoder -> SOLA crossfade
        -> PyAudio speaker callback
    """

    def __init__(self, config: dict, encoder: Encoder, decoder: Decoder,
                 knn: KNNRetrieval):
        self.cfg = config
        self.enc = encoder
        self.dec = decoder
        self.knn = knn

        sr = config['audio']['sample_rate']
        block = config['audio']['block_size']
        extra = config['audio']['extra_size']
        crossfade = config['audio']['crossfade_size']
        sola_search = config['audio']['sola_search_size']
        last_delay = config['audio']['last_delay_size']
        self.sr = sr
        self.block_size = block
        self.extra_size = extra
        self.crossfade_size = crossfade
        self.sola_search_size = sola_search
        self.last_delay_size = last_delay

        # Ring buffer: must hold block + extra + sola_search + 2*last_delay
        buf_size = max(block + crossfade + sola_search + 2 * last_delay,
                       block + extra)
        self.input_buf = np.zeros(buf_size, dtype=np.float32)
        self.last_output_tail = np.zeros(crossfade + sola_search + last_delay,
                                          dtype=np.float32)

        # VAD
        self.vad = None
        if config['vad']['enabled']:
            self.vad = webrtcvad.Vad(config['vad']['aggressiveness'])

        # PyAudio streams (opened on start)
        self.pa = None
        self.in_stream = None
        self.out_stream = None

    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for input stream.

        Receives `block_size` samples of int16 audio from mic, runs VC, writes
        to the output stream (sync callback model for simplicity).
        """
        # Decode int16 -> float32 [-1, 1]
        chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0

        # VAD silence gate (skip if silent)
        if self.vad is not None and self._is_silence(in_data):
            # Pass-through silence
            return (in_data, pyaudio.paContinue)

        # Roll ring buffer: shift in the new chunk
        self.input_buf = np.concatenate([self.input_buf[len(chunk):], chunk])

        # Compute mel-spec from input_buf (use the last block+extra samples)
        audio_for_mel = self.input_buf[-(self.block_size + self.extra_size):]
        mel_spec = make_mel_spec(audio_for_mel[None, :], sr=self.sr)

        # Encoder: content + F0 + energy
        content_feat, f0, energy = self.enc.encode(mel_spec)

        # kNN retrieval: replace source content with target voice
        content_replaced = self.knn.replace(content_feat)

        # Decoder: DDSP synthesis
        waveform = self.dec.decode(content_replaced, f0, energy)  # [1, 1, T_samples]

        # SOLA crossfade with previous tail
        out_chunk = self._sola_crossfade(waveform.squeeze())

        # Convert back to int16
        out_int16 = (out_chunk * 32768.0).clip(-32768, 32767).astype(np.int16)
        return (out_int16.tobytes(), pyaudio.paContinue)

    def _is_silence(self, in_bytes: bytes) -> bool:
        """WebRTC VAD on 30 ms frame."""
        try:
            # WebRTC VAD needs 8/16/32kHz mono PCM16, 30 ms frames
            frame = in_bytes[: self.sr * 30 // 1000 * 2]  # 30 ms of int16 bytes
            return not self.vad.is_speech(frame, self.sr)
        except Exception:
            return False

    def _sola_crossfade(self, new_chunk: np.ndarray) -> np.ndarray:
        """Synchronous Overlap-Add: find best alignment via cross-correlation.

        Algorithm (faithful to tinyvc/module/infer/stream.py:30-96):
            1. Take sola_search_size samples from the end of last_output_tail.
            2. For each offset in [0, sola_search_size], compute cross-correlation
               between (new_chunk first crossfade_size samples shifted by offset)
               and (last_output_tail last crossfade_size samples).
            3. Pick the offset with max cross-correlation.
            4. Crossfade new_chunk and last_output_tail at that offset with sin² window.
        """
        # Simplified for skeleton — full impl in P0-4 task
        cf = self.crossfade_size
        ss = self.sola_search_size
        ld = self.last_delay_size
        tail = self.last_output_tail

        # Naive: just crossfade at offset 0 (P0 baseline)
        # TODO: implement proper SOLA search in P0-4
        cf_window = 0.5 * (1 - np.cos(np.linspace(0, np.pi, cf)))
        out = new_chunk.copy()
        out[:cf] = cf_window * new_chunk[:cf] + (1 - cf_window) * tail[:cf]

        # Save tail for next chunk
        self.last_output_tail = np.concatenate([
            tail[cf:],
            new_chunk[cf: cf + ss + ld]
        ])
        return out[: self.block_size]

    def start(self):
        """Open PyAudio streams and start streaming."""
        self.pa = pyaudio.PyAudio()
        a = self.cfg['audio']
        self.in_stream = self.pa.open(
            format=getattr(pyaudio, a['format']),
            channels=a['channels'],
            rate=a['sample_rate'],
            input=True,
            input_device_index=a['input_device'],
            frames_per_buffer=a['block_size'],
            stream_callback=self.audio_callback,
        )
        self.out_stream = self.pa.open(
            format=getattr(pyaudio, a['format']),
            channels=a['channels'],
            rate=a['sample_rate'],
            output=True,
            output_device_index=a['output_device'],
            frames_per_buffer=a['block_size'],
        )
        self.in_stream.start_stream()
        self.out_stream.start_stream()
        print(f"Streaming started. voice_id={self.knn.current_voice_idx}")
        try:
            while self.in_stream.is_active():
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        if self.in_stream:
            self.in_stream.stop_stream()
            self.in_stream.close()
        if self.out_stream:
            self.out_stream.stop_stream()
            self.out_stream.close()
        if self.pa:
            self.pa.terminate()
