"""
Build the kNN voice index (voices.pt) for the v1 baseline.

For each voice WAV in data/voices/voice_*.wav, run the TinyVC encoder to obtain
a target feature tensor of shape [1, 768, T_ref] (T_ref = ~1500 frames for 30s
of 24kHz audio at hop=480). Combine into a single voices.pt file as a dict
{voice_N: tensor[1, 768, T_ref_N]}.

If data/voices/voice_*.wav is missing (P1-2 hasn't completed), generate 5
synthetic sine waves at 100/150/200/250/300 Hz so the kNN index has something
distinctive to retrieve from. These are NOT real speech — only used to verify
the v1 pipeline plumbing end-to-end.

Usage:
    python3 scripts/build_voices_index.py
        --voices-dir data/voices
        --models-dir  models
        --output       models/voices.pt
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# --- Stub out optional TinyVC dependencies that we don't need for v1 ----
# `module/utils/__init__.py` imports `f0_estimation`, which in turn imports
# `torchfcpe` and `pyworld`. These are only used for the alternative
# `f0_estimation='dio'|'harvest'|'fcpe'` paths; the default v1 pipeline uses
# the TinyVC PitchEstimator (built-in). Stub them out so the package imports.
for _mod_name, _attrs in (
    ('torchfcpe', {'spawn_bundled_infer_model': lambda *a, **kw: None}),
    ('pyworld',   {'dio': lambda *a, **kw: None,
                   'stonemask': lambda *a, **kw: None,
                   'harvest': lambda *a, **kw: None}),
):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        for _k, _v in _attrs.items():
            setattr(_stub, _k, _v)
        sys.modules[_mod_name] = _stub

# Make TinyVC importable
TINYVC_ROOT = Path('../repos/tinyvc')
sys.path.insert(0, str(TINYVC_ROOT))

from module.tinyvc import Encoder  # noqa: E402
from module.utils.auto_padding import autopad_waveform  # noqa: E402
from module.utils.spectrogram import spectrogram  # noqa: E402

SAMPLE_RATE = 24000


def synth_voice_wav(path: Path, freq: float, duration: float = 30.0):
    """Synthesize a distinctive sine wave with a couple of harmonics.

    We add a 2nd and 3rd harmonic and slow amplitude modulation so the
    resulting STFT (1920-pt FFT, 480 hop, 12.5 Hz/bin) has rich spectral
    content and the encoder produces clearly distinct features per voice.
    """
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    # fundamental + harmonics (low amplitude to keep sine-ish)
    wav = (
        0.35 * np.sin(2 * np.pi * freq * t)
        + 0.12 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.06 * np.sin(2 * np.pi * 3 * freq * t)
    ).astype(np.float32)
    # slow AM so the temporal axis is non-trivial — features differ across frames
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t).astype(np.float32)
    wav = wav * am
    # peak-normalize to -3 dBFS
    peak = float(np.max(np.abs(wav))) + 1e-8
    wav = wav * (10 ** (-3 / 20) / peak)
    sf.write(str(path), (wav * 32767).astype(np.int16), SAMPLE_RATE)


def ensure_voice_wavs(voices_dir: Path) -> list[Path]:
    """Return list of voice wav paths, generating 5 synthetic ones if absent."""
    voices_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(voices_dir.glob('voice_*.wav'))
    if existing and len(existing) >= 5:
        return existing[:5]
    print(f"[build_voices_index] No voice_*.wav in {voices_dir}; "
          f"generating 5 synthetic sine voices.")
    freqs = [100.0, 150.0, 200.0, 250.0, 300.0]
    paths = []
    for i, f in enumerate(freqs):
        p = voices_dir / f'voice_{i}.wav'
        synth_voice_wav(p, f)
        paths.append(p)
    return paths


def build_index(voices_dir: Path, models_dir: Path, output: Path,
                device: str = 'cpu'):
    enc_path = models_dir / 'encoder.pt'
    if not enc_path.exists():
        raise FileNotFoundError(
            f"encoder.pt not found at {enc_path}. Download from HF 'uthree/tinyvc'.")

    voice_paths = ensure_voice_wavs(voices_dir)
    print(f"[build_voices_index] {len(voice_paths)} voice wavs to encode.")

    dev = torch.device(device)
    encoder = Encoder().to(dev).eval()
    encoder.load_state_dict(torch.load(str(enc_path), map_location=dev))
    print(f"[build_voices_index] encoder loaded "
          f"({sum(p.numel() for p in encoder.parameters())/1e6:.3f}M params).")

    voices: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for i, vp in enumerate(voice_paths):
            wav, sr = sf.read(str(vp))
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                wav = librosa.resample(wav.astype(np.float32),
                                       orig_sr=sr, target_sr=SAMPLE_RATE)
            wav = wav.astype(np.float32)
            # peak normalize to -3 dBFS (same convention as infer.py)
            peak = float(np.max(np.abs(wav))) + 1e-8
            wav = wav * (10 ** (-3 / 20) / peak)
            # Generator.encode expects [Batch, Length] (2D), as torchaudio.load returns [C, L]
            wf = torch.from_numpy(wav).unsqueeze(0)  # [1, L]
            wf = wf.to(dev)
            wf = autopad_waveform(wf)
            spec = spectrogram(wf, encoder.n_fft, encoder.hop_size)
            z, _f0 = encoder.infer(spec)
            z = z.cpu()
            voices[f'voice_{i}'] = z
            print(f"  voice_{i}: {vp.name}  -> z {tuple(z.shape)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(voices, str(output))
    total_bytes = output.stat().st_size
    print(f"[build_voices_index] saved {output} "
          f"({len(voices)} voices, {total_bytes/1e6:.2f} MB).")
    return voices


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--voices-dir', default='data/voices')
    p.add_argument('--models-dir',  default='models')
    p.add_argument('--output',      default='models/voices.pt')
    p.add_argument('--device',      default='cpu')
    args = p.parse_args()
    build_index(Path(args.voices_dir), Path(args.models_dir),
                Path(args.output), args.device)


if __name__ == '__main__':
    main()
