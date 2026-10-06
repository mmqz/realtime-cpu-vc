//! vc-native: Pure-Rust hot paths for v2.0
//!
//! Replaces v1.0's Python implementations of:
//! - SOLA cross-correlation search (modules/streaming.py:_sola_crossfade)
//! - DDSP harmonic synthesis (modules/decoder.py:_synthesize_source)
//! - SPSC ring buffer for audio streaming (modules/streaming.py:input_buf)
//! - miniaudio audio I/O callback (replaces PyAudio)
//!
//! All hot-path code is pure Rust + `wide::f32x4` SIMD (AVX2 on x86, NEON on ARM).
//! miniaudio.h is vendored at `src_c/miniaudio/` and compiled via `cc::Build` in
//! `build.rs` so the audio callback runs GIL-free.

use wide::f32x4;

// ============================================================
// 1. SOLA cross-correlation search — Rust SIMD (AVX2 / NEON)
// ============================================================
// Replaces: prototype/modules/streaming.py:_sola_crossfade
// v1.0 baseline: ~30ms per chunk (Python + numpy)
// v2.0 target: ~5ms (Rust + wide f32x4 SIMD)

/// Find the best alignment offset for SOLA crossfade.
///
/// `new_chunk`: `[N]` float32 — the new audio block's first crossfade_size samples.
/// `tail`: `[N + sola_search_size]` float32 — the previous chunk's tail (crossfade_size
///   samples followed by sola_search_size candidates).
/// Returns: offset in `[0, sola_search_size)` maximizing cross-correlation between
///   `new_chunk` and `tail[offset..offset+N]`.
pub fn sola_find_best_offset(
    new_chunk: &[f32],
    tail: &[f32],
    sola_search_size: usize,
) -> usize {
    let crossfade_size = new_chunk.len();
    if tail.len() < crossfade_size + sola_search_size || crossfade_size == 0 {
        return 0;
    }
    let mut best_offset = 0usize;
    let mut best_corr = f32::NEG_INFINITY;
    for offset in 0..sola_search_size {
        let corr = dot_product_simd(new_chunk, &tail[offset..offset + crossfade_size]);
        if corr > best_corr {
            best_corr = corr;
            best_offset = offset;
        }
    }
    best_offset
}

/// SIMD-accelerated dot product using `wide::f32x4` (4 floats per iteration).
/// On x86: AVX2/FMA via `wide`. On ARM: NEON via `wide`.
#[inline(always)]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = f32x4::from([0.0f32; 4]);
    let chunks = a.chunks_exact(4).zip(b.chunks_exact(4));
    for (ac, bc) in chunks {
        // Safe, branchless load — `chunks_exact(4)` guarantees 4 elements.
        let av = f32x4::from([ac[0], ac[1], ac[2], ac[3]]);
        let bv = f32x4::from([bc[0], bc[1], bc[2], bc[3]]);
        sum += av * bv;
    }
    let mut total = sum.reduce_add();
    // Handle remainder (samples not divisible by 4).
    let remainder = a.len() % 4;
    let tail_start = a.len() - remainder;
    for i in 0..remainder {
        total += a[tail_start + i] * b[tail_start + i];
    }
    total
}

/// SOLA crossfade: apply sin² window crossfade at the best offset.
///
/// `new_chunk`: full new block (length >= crossfade_size).
/// `tail`: previous chunk's tail (length >= crossfade_size + sola_search_size).
/// Returns the crossfaded output (same length as `new_chunk`).
pub fn sola_crossfade(
    new_chunk: &[f32],
    tail: &[f32],
    sola_search_size: usize,
    crossfade_size: usize,
) -> Vec<f32> {
    let cf = crossfade_size.min(new_chunk.len());
    let offset = sola_find_best_offset(&new_chunk[..cf], tail, sola_search_size);
    let mut out = new_chunk.to_vec();
    for i in 0..cf {
        // sin² window crossfade (raised-cosine variant): w = 0.5*(1 - cos(pi*i/cf))
        let w = 0.5 * (1.0 - (std::f32::consts::PI * i as f32 / cf as f32).cos());
        let tail_idx = offset + i;
        if tail_idx < tail.len() {
            out[i] = w * new_chunk[i] + (1.0 - w) * tail[tail_idx];
        }
    }
    out
}

// ============================================================
// 2. DDSP harmonic synthesis — Rust scalar (SIMD-able)
// ============================================================
// Replaces: prototype/modules/decoder.py:_synthesize_source (harmonic part)
// v1.0 baseline: ~25ms per chunk (Python + numpy + torch.sin)
// v2.0 target: ~3ms (Rust + wide f32x4 + cumulative phase)
//
// Faithful to tinyvc/module/tinyvc/decoder.py:24-54 (oscillate_harmonics).
// Differences from the original Torch impl:
//   - We use `cumsum` of `f0 * 2π/sr` directly (no modulo-1 wrap, sin() handles it).
//   - Voiced/unvoiced mask is applied upstream via amp_k (zero on unvoiced frames).
//   - Linear interpolation upsample happens upstream (matching decoder.py:132).

/// Generate harmonic source signal: `sum_{k=1..n_harmonics} amp_k * sin(k * phase)`.
///
/// # Arguments
/// * `f0_upsampled` — `[T_samples]` fundamental frequency, already upsampled from frame
///   rate to sample rate (e.g. via `np.repeat(f0, hop)`).
/// * `amplitudes` — `[n_harmonics + 1, T_samples]` per-harmonic amplitudes in row-major
///   layout. Index 0 is the DC component (skipped per DDSP convention).
/// * `n_harmonics` — number of harmonics above DC (e.g. 14 for tinyvc).
/// * `sr` — sample rate in Hz.
/// # Returns
/// * `[T_samples]` float32 harmonic source signal.
pub fn synth_harmonics(
    f0_upsampled: &[f32],
    amplitudes: &[f32],
    n_harmonics: usize,
    sr: u32,
) -> Vec<f32> {
    let t = f0_upsampled.len();
    let mut output = vec![0.0f32; t];

    // 1. Compute phase via cumulative sum: phase[i] = phase[i-1] + 2π·f0[i]/sr.
    //    This mirrors numpy's `np.cumsum(f0_up * (2π/sr))` from decoder.py:134.
    let mut phase = vec![0.0f32; t];
    let phase_step = 2.0 * std::f32::consts::PI / sr as f32;
    let mut acc = 0.0f32;
    for i in 0..t {
        acc += f0_upsampled[i] * phase_step;
        phase[i] = acc;
    }

    // 2. For each harmonic k=1..=n_harmonics, accumulate amp_k[i] * sin(k * phase[i]).
    //    Skip k=0 (DC component, per DDSP convention — amp[0..T] is the DC row).
    let amp_stride = t; // amplitudes shape [n_harmonics+1, T]
    for k in 1..=n_harmonics {
        let amp_k = &amplitudes[k * amp_stride..(k + 1) * amp_stride];
        let kf = k as f32;
        for i in 0..t {
            output[i] += amp_k[i] * (kf * phase[i]).sin();
        }
    }
    output
}

// ============================================================
// 3. SPSC ring buffer for lock-free audio streaming
// ============================================================
// Replaces: prototype/modules/streaming.py:input_buf (numpy roll-based)
// v1.0 baseline: ~5ms per chunk + GIL contention
// v2.0 target: <0.5ms (lock-free, no GIL)
//
// We just re-export `rtrb::RingBuffer` — it's a battle-tested SPSC ring with
// `Producer`/`Consumer` halves that can be sent across threads. The audio
// callback (miniaudio thread) holds the Producer; the decoder thread holds the
// Consumer. Push/pop are O(1), lock-free, wait-free.

pub use rtrb::RingBuffer;

/// Audio ring buffer with separate producer (audio callback) and consumer (decoder thread).
///
/// Capacity is rounded up to the next power of two by `rtrb` internally. The producer
/// and consumer halves can be sent to different threads without synchronization.
pub fn new_audio_ring_buffer(
    capacity_samples: usize,
) -> (rtrb::Producer<f32>, rtrb::Consumer<f32>) {
    RingBuffer::new(capacity_samples)
}

// ============================================================
// 4. miniaudio audio I/O (built via cc::Build in build.rs)
// ============================================================
// Replaces: PyAudio
// v1.0 baseline: 10-20ms callback overhead per chunk (GIL contention)
// v2.0 target: <2ms (WASAPI exclusive mode on Windows, default low_latency elsewhere)
//
// Note: miniaudio.h + miniaudio.c are vendored into src_c/miniaudio/.
// build.rs compiles them via cc::Build. The actual `ma_device` FFI declarations
// and Rust-safe wrapper land in P2.1 (audio thread wiring); this crate currently
// exposes the `AudioConfig` struct so vc-python can already configure the device.

/// Audio device config — mirrors miniaudio's `ma_device_config`.
#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    /// Samples per chunk (default 1920 @ 24kHz = 80ms).
    pub block_size: u32,
    /// Use miniaudio's low-latency profile.
    pub low_latency: bool,
    /// WASAPI exclusive mode (Windows only; bypasses the shared-mode mixer).
    pub exclusive_mode: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            channels: 1,
            block_size: 1920,
            low_latency: true,
            exclusive_mode: false,
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sola_find_best_offset_zero_offset() {
        // When new_chunk matches tail at offset 0, best offset should be 0.
        let chunk = vec![1.0, 2.0, 3.0, 4.0];
        let mut tail = vec![0.0; 10];
        tail[0..4].copy_from_slice(&chunk);
        let offset = sola_find_best_offset(&chunk, &tail, 6);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_sola_find_best_offset_nonzero() {
        // When best alignment is at offset 3, should find it.
        let chunk = vec![1.0, 2.0, 3.0, 4.0];
        let mut tail = vec![0.0; 10];
        tail[3..7].copy_from_slice(&chunk);
        let offset = sola_find_best_offset(&chunk, &tail, 6);
        assert_eq!(offset, 3);
    }

    #[test]
    fn test_dot_product_simd_correctness_small() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = dot_product_simd(&a, &b);
        // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
        assert!((result - 70.0).abs() < 1e-5, "got {result}");
    }

    #[test]
    fn test_dot_product_simd_correctness_aligned() {
        // Length exactly divisible by 4 — no remainder path.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = dot_product_simd(&a, &b);
        // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120
        assert!((result - 120.0).abs() < 1e-5, "got {result}");
    }

    #[test]
    fn test_dot_product_simd_correctness_large() {
        // 1000 elements — exercises SIMD loop heavily.
        // SIMD sums in groups of 4 (vectorized) vs scalar sum has different
        // rounding order, so we use a relative tolerance.
        let a: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.2).collect();
        let simd_result = dot_product_simd(&a, &b);
        // Scalar reference
        let scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let abs_err = (simd_result - scalar).abs();
        let rel_err = abs_err / scalar.abs().max(1e-6);
        assert!(
            rel_err < 1e-4,
            "abs_err={abs_err}, rel_err={rel_err}, simd={simd_result}, scalar={scalar}"
        );
    }

    #[test]
    fn test_synth_harmonics_basic() {
        // 1 harmonic, 240 samples, F0=100Hz at 24kHz => exactly 1 cycle.
        // Output: sin(2π * 100 * t / 24000) for t=0..240 — covers full sine cycle.
        let sr = 24000u32;
        let n = 240;
        let f0 = vec![100.0; n];
        // amplitudes shape [n_harmonics+1, T] = [2, 240] (row 0 = DC, row 1 = harmonic 1)
        let mut amps = vec![0.0f32; 2 * n];
        for i in 0..n {
            amps[n + i] = 1.0; // harmonic 1 amplitude = 1.0 everywhere
        }
        let out = synth_harmonics(&f0, &amps, 1, sr);
        assert_eq!(out.len(), n);
        let max_val = out.iter().cloned().fold(0.0f32, f32::max);
        let min_val = out.iter().cloned().fold(0.0f32, f32::min);
        assert!(max_val > 0.9, "max should be > 0.9, got {max_val}");
        assert!(min_val < -0.9, "min should be < -0.9, got {min_val}");
        // Sanity: at i=0, phase = f0[0] * 2π/sr (cumsum's first element is not 0),
        // so out[0] = sin(2π * 100 / 24000) ≈ 0.0262. Match the formula exactly.
        let expected_0 = (2.0 * std::f32::consts::PI * 100.0 / sr as f32).sin();
        assert!(
            (out[0] - expected_0).abs() < 1e-5,
            "out[0] should be {expected_0}, got {}",
            out[0]
        );
    }

    #[test]
    fn test_synth_harmonics_matches_python_formula() {
        // Verify our formula matches the Python reference exactly:
        //   output[i] = amp_k[i] * sin(k * cumsum(f0 * 2π/sr))[i]
        let sr = 24000u32;
        let t = 480; // one hop
        let f0: Vec<f32> = (0..t).map(|i| 100.0 + (i as f32) * 0.01).collect();
        // 2 harmonics (k=0 DC + k=1 + k=2)
        let n_h = 2;
        let mut amps = vec![0.0f32; (n_h + 1) * t];
        for i in 0..t {
            amps[1 * t + i] = 0.5; // harmonic 1 amp = 0.5
            amps[2 * t + i] = 0.25; // harmonic 2 amp = 0.25
        }
        let out = synth_harmonics(&f0, &amps, n_h, sr);

        // Reference: scalar reimplementation of the algorithm
        let phase_step = 2.0 * std::f32::consts::PI / sr as f32;
        let mut acc = 0.0f32;
        let phase: Vec<f32> = f0
            .iter()
            .map(|f| {
                acc += f * phase_step;
                acc
            })
            .collect();
        let reference: Vec<f32> = (0..t)
            .map(|i| {
                0.5 * (1.0 * phase[i]).sin() + 0.25 * (2.0 * phase[i]).sin()
            })
            .collect();

        for i in 0..t {
            assert!(
                (out[i] - reference[i]).abs() < 1e-5,
                "mismatch at {i}: rust={}, ref={}",
                out[i],
                reference[i]
            );
        }
    }

    #[test]
    fn test_sola_crossfade_preserves_length() {
        let chunk = vec![1.0; 100];
        let tail = vec![0.0; 200];
        let out = sola_crossfade(&chunk, &tail, 50, 30);
        assert_eq!(out.len(), 100);
        // After the crossfade region, output should equal the new_chunk.
        for i in 30..100 {
            assert!((out[i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ring_buffer_basic() {
        // rtrb 0.4: push returns Result<(), PushError>, pop returns Result<T, PopError>.
        // We use `.ok()` to convert to Option for ergonomic comparison.
        let (mut producer, mut consumer) = new_audio_ring_buffer(100);
        assert!(producer.push(1.0).is_ok());
        assert!(producer.push(2.0).is_ok());
        assert_eq!(consumer.pop().ok(), Some(1.0));
        assert_eq!(consumer.pop().ok(), Some(2.0));
        assert_eq!(consumer.pop().ok(), None);
    }

    #[test]
    fn test_ring_buffer_full_then_drain() {
        let cap = 16;
        let (mut producer, mut consumer) = new_audio_ring_buffer(cap);
        // rtrb rounds capacity up; just push until full.
        let mut pushed = 0usize;
        while producer.push(0.5).is_ok() {
            pushed += 1;
        }
        assert!(pushed >= cap, "pushed={pushed}, cap={cap}");
        let mut popped = 0usize;
        while consumer.pop().is_ok() {
            popped += 1;
        }
        assert_eq!(pushed, popped);
    }

    #[test]
    fn test_audio_config_default() {
        let cfg = AudioConfig::default();
        assert_eq!(cfg.sample_rate, 24000);
        assert_eq!(cfg.channels, 1);
        assert_eq!(cfg.block_size, 1920);
        assert!(cfg.low_latency);
        assert!(!cfg.exclusive_mode);
    }
}
