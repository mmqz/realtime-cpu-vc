//! Rust STFT + mel filter bank, replacing librosa for the streaming path.
//!
//! # OPT-3 context
//! v1.0 prototype computed STFT in Python via `librosa.stft(...)` (~15-25 ms
//! per chunk on a 24 kHz / 1920-sample block). This module moves that work to
//! Rust via `realfft` (a thin wrapper around `rustfft`'s complex FFT that
//! exploits Hermitian symmetry to output only `n_fft/2 + 1` bins — half the
//! cost of a full C2C FFT for real input). Target: ~1-2 ms per chunk.
//!
//! # Layout conventions
//! - `stft_magnitude` returns `[T_frames, n_bins]` row-major (frames-major),
//!   the natural output of the frame loop.
//! - `MelFilterBank::apply` transposes to `[n_mels, T_frames]` row-major to
//!   match librosa `melspectrogram(...)`'s output shape — so the downstream
//!   ONNX encoder sees the same layout as it did under v1.
//! - `linear_stft_magnitude` (TinyVC encoder path) returns `[T_frames, n_bins]`
//!   raw magnitude, no mel / no log.

use num_complex::Complex;
use std::f32::consts::PI;

// ============================================================
// STFT magnitude — rustfft via realfft's R2C FFT
// ============================================================
// Replaces: librosa.stft(y, n_fft, hop_length, window='hann')
// v1.0 baseline: ~15-25ms per chunk (Python + scipy FFT + numpy windowing)
// v2.0 target:   ~1-2ms per chunk (Rust + realfft R2C FFT + cached window)

/// Compute STFT magnitude spectrogram.
///
/// Uses `realfft`'s `RealToComplex<f32>` FFT — R2C exploits Hermitian
/// symmetry of real input to write only the `n_fft/2 + 1` non-redundant
/// bins (vs. `n_fft` for a full C2C FFT), roughly halving FFT cost.
///
/// # Arguments
/// - `wav`:   `[T_samples]` float32, mono PCM in `[-1, 1]`.
/// - `n_fft`: FFT size (must match `win.len()`).
/// - `hop`:   STFT hop in samples (frame stride).
/// - `win`:   Window of length `n_fft` (e.g. Hann). Caller supplies so the
///            window can be cached across chunks.
///
/// # Returns
/// `[T_frames * n_bins]` float32 in **frames-major** row-major order, where
/// `n_bins = n_fft/2 + 1` and `T_frames = max(1, (T_samples - n_fft) / hop + 1)`.
/// Element `[f * n_bins + b]` is the magnitude of FFT bin `b` in frame `f`.
///
/// # Degenerate inputs
/// - Empty / shorter-than-`n_fft` input produces 1 zero-padded frame (so the
///   downstream pipeline doesn't have to special-case the chunk tail).
pub fn stft_magnitude(wav: &[f32], n_fft: usize, hop: usize, win: &[f32]) -> Vec<f32> {
    assert_eq!(win.len(), n_fft, "window length must equal n_fft");
    assert!(n_fft > 0, "n_fft must be > 0");
    assert!(hop > 0, "hop must be > 0");

    let n_bins = n_fft / 2 + 1;
    let n_frames = if wav.len() < n_fft {
        1
    } else {
        (wav.len() - n_fft) / hop + 1
    };
    let mut output = vec![0.0f32; n_bins * n_frames];

    // `RealFftPlanner` caches the FFT twiddle factors; for a single `n_fft`
    // we only plan once outside the frame loop. The returned `Arc<RealToComplex>`
    // is cheap to clone if you want to share it across threads (we don't here).
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);

    let mut frame = vec![0.0f32; n_fft];
    // R2C output has `n_bins = n_fft/2 + 1` complex samples (Nyquist + DC
    // are real-valued but stored as Complex for a uniform API).
    let mut spectrum: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_bins];

    for f in 0..n_frames {
        let start = f * hop;
        // Apply window + copy frame (zero-pad past end of input). The window
        // multiply is the second hot kernel after the FFT itself — librosa
        // does the same `y * win` element-wise.
        for i in 0..n_fft {
            let idx = start + i;
            frame[i] = if idx < wav.len() { wav[idx] * win[i] } else { 0.0 };
        }
        // Real→Complex FFT. `unwrap` is safe — realfft only returns `Err` on
        // length mismatch (impossible here: we allocate `frame` and `spectrum`
        // with the exact sizes the planner expects).
        r2c.process(&mut frame, &mut spectrum).unwrap();
        // Magnitude = `|z| = sqrt(re² + im²)` — librosa's `np.abs(stft)`.
        // `Complex::norm()` is `sqrt(re*re + im*im)` (one `sqrt` per bin).
        for b in 0..n_bins {
            output[f * n_bins + b] = spectrum[b].norm();
        }
    }

    output
}

// ============================================================
// Cached mel filter bank (Slaney-style triangular filters, HTK mel scale)
// ============================================================
// Replaces: librosa.filters.mel(sr, n_fft, n_mels=n_mels, fmin, fmax, htk=True)
// Construction is O(n_mels * n_bins) — for n_mels=128, n_bins=961 that's
// ~123k multiplications, ~50µs. The streaming path calls this once per
// (sr, n_fft, n_mels, fmin, fmax) configuration and reuses it across all
// chunks → cost amortizes to ~0 across the session.

/// Cached mel filter bank. Construct once per (n_mels, n_fft, sr, fmin, fmax)
/// configuration and reuse across all chunks.
///
/// # Layout
/// `weights` is `[n_mels, n_bins]` row-major, where `n_bins = n_fft/2 + 1`.
/// Element `[m * n_bins + b]` is the weight of FFT bin `b` in mel filter `m`.
pub struct MelFilterBank {
    /// Number of mel bands (typically 80 or 128).
    pub n_mels: usize,
    /// Number of FFT bins (`n_fft / 2 + 1`).
    pub n_bins: usize,
    /// Lower frequency bound in Hz.
    pub fmin: f32,
    /// Upper frequency bound in Hz.
    pub fmax: f32,
    /// Sample rate in Hz.
    pub sr: u32,
    /// Weight matrix `[n_mels, n_bins]` row-major. Non-negative triangular
    /// filters (Slaney-style). Kept private — callers should use [`apply`].
    weights: Vec<f32>,
}

impl MelFilterBank {
    /// Build a mel filter bank with HTK mel scale and Slaney-style triangular
    /// filters.
    ///
    /// # Panics
    /// Panics if `fmax <= fmin` or `n_fft == 0` (degenerate configs — caller
    /// bug, fail fast rather than returning a bank of NaNs).
    pub fn new(n_mels: usize, n_fft: usize, sr: u32, fmin: f32, fmax: f32) -> Self {
        assert!(n_fft > 0, "n_fft must be > 0");
        assert!(fmax > fmin, "fmax must be > fmin");
        let n_bins = n_fft / 2 + 1;
        let mut weights = vec![0.0f32; n_mels * n_bins];

        // HTK mel scale (matches librosa.filters.mel(htk=True) — the streaming
        // path uses HTK since the v1 ONNX encoder was trained on HTK mels).
        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);
        let mel_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        let hz_points: Vec<f32> = mel_points.iter().map(|m| mel_to_hz(*m)).collect();

        // FFT bin center frequencies: `bin_b = b * sr / n_fft`.
        let fft_freqs: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * sr as f32 / n_fft as f32)
            .collect();

        // Slaney-style triangular filters: for each mel band `m`, the triangle
        // has vertices at `(left, 0)`, `(center, 1)`, `(right, 0)` in (Hz,
        // weight) space. Linear ramp up from left→center, linear ramp down
        // from center→right, zero outside `[left, right]`.
        for m in 0..n_mels {
            let left = hz_points[m];
            let center = hz_points[m + 1];
            let right = hz_points[m + 2];
            for b in 0..n_bins {
                let freq = fft_freqs[b];
                let weight = if freq < left || freq > right {
                    0.0
                } else if freq <= center {
                    (freq - left) / (center - left + 1e-8)
                } else {
                    (right - freq) / (right - center + 1e-8)
                };
                weights[m * n_bins + b] = weight;
            }
        }

        Self {
            n_mels,
            n_bins,
            fmin,
            fmax,
            sr,
            weights,
        }
    }

    /// Apply mel filter bank to STFT magnitude.
    ///
    /// # Arguments
    /// - `stft_mag`: `[T_frames * n_bins]` float32 in **frames-major**
    ///   row-major order (the output of [`stft_magnitude`]). Element
    ///   `[f * n_bins + b]` is the magnitude of FFT bin `b` in frame `f`.
    /// - `n_frames`: number of frames (the slice alone doesn't encode the row
    ///   count, so the caller supplies it).
    ///
    /// # Returns
    /// `[n_mels * T_frames]` float32 in **mels-major** row-major order, i.e.
    /// librosa `melspectrogram(...)`'s layout. Element `[m * n_frames + f]`
    /// is the mel energy of band `m` in frame `f`.
    pub fn apply(&self, stft_mag: &[f32], n_frames: usize) -> Vec<f32> {
        // The natural layout from `stft_magnitude` is frames-major — we
        // transpose to mels-major on the fly to match librosa's output shape,
        // so the downstream ONNX encoder sees the same memory layout as v1.
        let mut output = vec![0.0f32; self.n_mels * n_frames];
        for f in 0..n_frames {
            // Slice the input row once per frame (one bounds check vs.
            // n_bins per-element indexing).
            let mag_row = &stft_mag[f * self.n_bins..(f + 1) * self.n_bins];
            for m in 0..self.n_mels {
                let w_row = &self.weights[m * self.n_bins..(m + 1) * self.n_bins];
                // Inner dot product. For n_bins=961, this is 961 fma's per
                // (frame, mel) — could be vectorized with `wide::f32x4`
                // (matches `dot_product_simd` in lib.rs) but the FFT dominates
                // the chunk cost; we keep it scalar for clarity. Future work.
                let mut sum = 0.0f32;
                for b in 0..self.n_bins {
                    sum += w_row[b] * mag_row[b];
                }
                output[m * n_frames + f] = sum;
            }
        }
        output
    }
}

/// Compute log-mel spectrogram (full pipeline: STFT → mel → log dB).
///
/// `mel_bank` is caller-supplied so it can be cached across calls (avoids
/// re-computing the mel triangle weights each chunk — see [`MelFilterBank`]
/// docstring).
///
/// # Returns
/// `[n_mels * T_frames]` float32 in **mels-major** row-major order
/// (librosa-compatible layout, ready to hand to the ONNX encoder).
///
/// # Normalization
/// `power_to_db(mel) = 10 * log10(mel + 1e-10)`, then folded with the
/// `(db + 80) / 20` affine shift to roughly normalize to `[-4, +4]` —
/// matches the v1 prototype's preprocessing (which the encoder was trained
/// against). `1e-10` floor mirrors librosa's default `amin`.
pub fn log_mel_spec(
    wav: &[f32],
    sr: u32,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
    mel_bank: &MelFilterBank,
) -> Vec<f32> {
    // Defensive: caller should pass mel_bank matching the other params (the
    // bank is supposed to be cached across chunks for the same config, but
    // a mismatch would silently produce wrong-shaped output). Using these
    // also suppresses the dead-code warning on the redundant-looking args.
    assert_eq!(mel_bank.n_mels, n_mels, "mel_bank.n_mels != n_mels arg");
    assert_eq!(mel_bank.n_bins, n_fft / 2 + 1, "mel_bank.n_bins != n_fft/2+1");
    assert_eq!(mel_bank.sr, sr, "mel_bank.sr != sr arg");
    assert!(
        (mel_bank.fmin - fmin).abs() < 1e-6 && (mel_bank.fmax - fmax).abs() < 1e-6,
        "mel_bank (fmin,fmax)=({}, {}) != args ({}, {})",
        mel_bank.fmin,
        mel_bank.fmax,
        fmin,
        fmax
    );

    // 1. Hann window. librosa's default `window='hann'`:
    //    `0.5 - 0.5 * cos(2π * i / n_fft)` for i in [0, n_fft).
    let win: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n_fft as f32).cos())
        .collect();

    // 2. STFT magnitude (frames-major).
    let n_frames = if wav.len() < n_fft {
        1
    } else {
        (wav.len() - n_fft) / hop + 1
    };
    let stft_mag = stft_magnitude(wav, n_fft, hop, &win);

    // 3. Apply cached mel filter bank (frames-major → mels-major).
    let mel_spec = mel_bank.apply(&stft_mag, n_frames);

    // 4. Power→dB with +80 top_db floor and /20 scale (v1 normalization).
    let mut output = vec![0.0f32; mel_spec.len()];
    for i in 0..mel_spec.len() {
        let db = 10.0 * (mel_spec[i] + 1e-10).log10();
        output[i] = (db + 80.0) / 20.0;
    }
    output
}

/// Compute linear STFT magnitude (no mel filter, no log).
///
/// Used by the TinyVC encoder path which takes `[n_fft/2+1, T_frames]` raw
/// magnitude (the encoder was trained on linear spectrogram, not log-mel).
///
/// # Returns
/// `[T_frames * n_bins]` float32 in **frames-major** row-major order.
/// Element `[f * n_bins + b]` is the magnitude of FFT bin `b` in frame `f`.
pub fn linear_stft_magnitude(wav: &[f32], n_fft: usize, hop: usize) -> Vec<f32> {
    let win: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n_fft as f32).cos())
        .collect();
    stft_magnitude(wav, n_fft, hop, &win)
}

// ============================================================
// Mel scale (HTK formula — matches librosa.filters.mel(htk=True))
// ============================================================

/// Hz → mel (HTK formula: `2595 * log10(1 + hz/700)`).
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// mel → Hz (inverse HTK: `700 * (10^(mel/2595) - 1)`).
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_magnitude_shape() {
        let sr = 24000u32;
        let wav = vec![0.5; sr as usize / 10]; // 100ms at 24kHz = 2400 samples
        let n_fft = 1920;
        let hop = 480;
        let win = vec![1.0; n_fft]; // rectangular (so we test the FFT path, not windowing)
        let mag = stft_magnitude(&wav, n_fft, hop, &win);
        let n_bins = n_fft / 2 + 1;
        let n_frames = (wav.len() - n_fft) / hop + 1;
        assert_eq!(mag.len(), n_bins * n_frames);
        // All magnitudes of a constant signal should be ≥ 0 (sanity).
        let min_val = mag.iter().cloned().fold(0.0f32, f32::min);
        assert!(min_val >= 0.0, "magnitude should be non-negative, got min {min_val}");
        // For a constant-0.5 input with rect window, DC bin (b=0) of every
        // frame = 0.5 * n_fft = 960 (sum of all samples, no FFT phase).
        for f in 0..n_frames {
            let dc = mag[f * n_bins + 0];
            assert!(
                (dc - (0.5 * n_fft as f32)).abs() < 1.0,
                "frame {f}: DC bin expected ~{}, got {dc}",
                0.5 * n_fft as f32
            );
        }
    }

    #[test]
    fn test_mel_filter_bank_shape() {
        let bank = MelFilterBank::new(128, 1920, 24000, 20.0, 12000.0);
        assert_eq!(bank.n_mels, 128);
        assert_eq!(bank.n_bins, 961);
        // Weights are non-negative (triangular filter coefficients).
        for &w in &bank.weights {
            assert!(w >= 0.0, "mel weight must be non-negative, got {w}");
        }
        // Every mel band (except possibly the edge ones at fmin/fmax) should
        // have at least one positive weight — otherwise it's an empty filter
        // and contributes nothing. Count bands with zero total weight.
        let empty_bands = (0..bank.n_mels)
            .filter(|m| {
                let row = &bank.weights[m * bank.n_bins..(m + 1) * bank.n_bins];
                row.iter().all(|&w| w == 0.0)
            })
            .count();
        assert_eq!(empty_bands, 0, "no mel band should be entirely zero");
    }

    #[test]
    fn test_log_mel_spec_output() {
        let sr = 24000u32;
        // 1s of 150 Hz sine — bin 12 (150 * 1920 / 24000 = 12) should have
        // the most energy. n_mels=128 means band ~2 should peak (150 Hz is
        // near the bottom of the mel scale).
        let wav: Vec<f32> = (0..sr as usize)
            .map(|i| 0.5 * (2.0 * PI * 150.0 * i as f32 / sr as f32).sin())
            .collect();
        let bank = MelFilterBank::new(128, 1920, 24000, 20.0, 12000.0);
        let mel = log_mel_spec(&wav, sr, 1920, 480, 128, 20.0, 12000.0, &bank);
        // Should have 128 mels × T_frames.
        let n_frames = (wav.len() - 1920) / 480 + 1;
        assert_eq!(mel.len(), 128 * n_frames);
        // Should be non-zero (150 Hz sine has energy).
        let max_val = mel.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_val > 0.0, "mel spec all zero — sine input should produce energy");
    }

    #[test]
    fn test_linear_stft_magnitude_shape() {
        let wav = vec![0.5; 24000]; // 1s at 24kHz
        let mag = linear_stft_magnitude(&wav, 1920, 480);
        let n_bins = 961;
        let n_frames = (24000 - 1920) / 480 + 1;
        assert_eq!(mag.len(), n_bins * n_frames);
        // Constant input → DC bin (b=0) of every frame should peak.
        for f in 0..n_frames {
            let dc = mag[f * n_bins + 0];
            assert!(dc > 0.0, "frame {f}: DC should be > 0 for constant input");
        }
    }

    /// Edge case: very short input (less than n_fft) should produce 1 frame
    /// without panic.
    #[test]
    fn test_stft_short_input_single_frame() {
        let wav = vec![0.5; 100]; // 100 samples, n_fft=1920 → zero-pad
        let win = vec![1.0; 1920];
        let mag = stft_magnitude(&wav, 1920, 480, &win);
        let n_bins = 961;
        assert_eq!(mag.len(), n_bins * 1);
        // DC bin = 0.5 * 100 (only 100 samples, rest zero-padded).
        assert!((mag[0] - 50.0).abs() < 1.0, "DC = {}, expected ~50", mag[0]);
    }

    /// hz_to_mel / mel_to_hz round-trip sanity.
    #[test]
    fn test_mel_roundtrip() {
        for &hz in &[20.0_f32, 100.0, 440.0, 1000.0, 8000.0, 12000.0] {
            let mel = hz_to_mel(hz);
            let back = mel_to_hz(mel);
            assert!(
                (back - hz).abs() < 0.5,
                "roundtrip hz={hz} → mel={mel} → hz={back} (diff {})",
                (back - hz).abs()
            );
        }
        // HTK formula: `2595 * log10(1 + 1000/700) = 2595 * log10(2.4286) ≈ 999.99`.
        // Not exactly 1000 (the "1000 Hz = 1000 mel" mnemonic is approximate),
        // but very close — sanity-check the formula against the closed form.
        let mel_1000 = hz_to_mel(1000.0);
        let expected: f32 = 2595.0 * (1.0 + 1000.0_f32 / 700.0).log10();
        assert!(
            (mel_1000 - expected).abs() < 1e-3,
            "hz_to_mel(1000)={mel_1000}, expected closed-form {expected}"
        );
        assert!(
            (mel_1000 - 1000.0).abs() < 0.1,
            "hz_to_mel(1000)={mel_1000}, expected ≈1000 (HTK mnemonic)"
        );
    }
}
