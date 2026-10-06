//! vc-native: Pure-Rust hot paths for v2.0
//!
//! Replaces v1.0's Python implementations of:
//! - SOLA cross-correlation search (modules/streaming.py:_sola_crossfade)
//! - DDSP harmonic synthesis (modules/decoder.py:_synthesize_source)
//! - SPSC ring buffer for audio streaming (modules/streaming.py:input_buf)
//! - miniaudio audio I/O callback (replaces PyAudio)

use wide::f32x4;

// ============================================================
// 1. SOLA cross-correlation search — Rust SIMD (AVX2 / NEON)
// ============================================================
// Replaces: prototype/modules/streaming.py:_sola_crossfade
// v1.0 baseline: ~30ms per chunk (Python + numpy)
// v2.0 target: ~5ms (Rust + wide f32x4 SIMD)

/// Find the best alignment offset for SOLA crossfade.
/// `new_chunk`: [N] float32 (the new audio block first crossfade_size samples)
/// `tail`: [N] float32 (the previous chunk's last crossfade_size + sola_search_size samples)
/// Returns: offset in [0, sola_search_size] maximizing cross-correlation.
pub fn sola_find_best_offset(new_chunk: &[f32], tail: &[f32], sola_search_size: usize) -> usize {
    let crossfade_size = new_chunk.len();
    if tail.len() < crossfade_size + sola_search_size {
        return 0;
    }
    let mut best_offset = 0;
    let mut best_corr = f32::NEG_INFINITY;
    for offset in 0..sola_search_size {
        let corr = dot_product_simd(&new_chunk, &tail[offset..offset + crossfade_size]);
        if corr > best_corr {
            best_corr = corr;
            best_offset = offset;
        }
    }
    best_offset
}

/// SIMD-accelerated dot product using wide::f32x4 (4 floats per iteration).
/// On x86: AVX2/FMA via wide. On ARM: NEON via wide.
#[inline(always)]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = f32x4::from([0.0; 4]);
    let chunks = a.chunks_exact(4).zip(b.chunks_exact(4));
    for (ac, bc) in chunks {
        let av = f32x4::from(*ac.as_ptr().cast::<[f32;4]>().read());
        let bv = f32x4::from(*bc.as_ptr().cast::<[f32;4]>().read());
        sum += av * bv;
    }
    let mut total = sum.reduce_add();
    let remainder = a.len() % 4;
    let tail_start = a.len() - remainder;
    for i in 0..remainder {
        total += a[tail_start + i] * b[tail_start + i];
    }
    total
}

// ============================================================
// 2. DDSP harmonic synthesis — Rust SIMD
// ============================================================
// Replaces: prototype/modules/decoder.py:_synthesize_source (harmonic part)
// v1.0 baseline: ~25ms per chunk (Python + numpy + torch.sin)
// v2.0 target: ~3ms (Rust + wide f32x4 + cumulative phase)

/// Generate harmonic source signal: sum_{k=1..n_harmonics} amp_k * sin(k * phase)
///
/// # Arguments
/// * `f0_upsampled` - [T_samples] fundamental frequency, upsampled from frame rate to sample rate
/// * `amplitudes` - [n_harmonics + 1, T_samples] per-harmonic amplitudes (index 0 = DC, skip)
/// * `sr` - sample rate
/// # Returns
/// * [T_samples] float32 harmonic source signal
pub fn synth_harmonics(f0_upsampled: &[f32], amplitudes: &[f32], n_harmonics: usize, sr: u32) -> Vec<f32> {
    let t = f0_upsampled.len();
    let mut output = vec![0.0f32; t];

    // 1. Compute phase via cumulative sum: phase[i] = phase[i-1] + 2π·f0[i]/sr
    //    SIMD-friendly: scan and accumulate
    let mut phase = vec![0.0f32; t];
    let phase_step = 2.0 * std::f32::consts::PI / sr as f32;
    let mut acc = 0.0f32;
    for i in 0..t {
        acc += f0_upsampled[i] * phase_step;
        phase[i] = acc;
    }

    // 2. For each harmonic k=1..n_harmonics, compute sin(k * phase) * amp_k,
    //    accumulate into output.
    //    Skip k=0 (DC component, per DDSP convention).
    let amp_stride = t;  // amplitudes shape [n_harmonics+1, T]
    for k in 1..=n_harmonics {
        let amp_k = &amplitudes[k * amp_stride..(k + 1) * amp_stride];
        // SIMD vectorized: sin(k*phase) can be batched with wide f32x4
        // For simplicity here we use std::f32::sin scalar; P2 will swap to wide
        for i in 0..t {
            output[i] += amp_k[i] * (k as f32 * phase[i]).sin();
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

pub use rtrb::RingBuffer;

/// Audio ring buffer with separate producer (audio callback) and consumer (decoder thread).
pub fn new_audio_ring_buffer(capacity_samples: usize) -> (rtrb::Producer<f32>, rtrb::Consumer<f32>) {
    RingBuffer::new(capacity_samples)
}

// ============================================================
// 4. miniaudio audio I/O (built via cc::Build in build.rs)
// ============================================================
// Replaces: PyAudio
// v1.0 baseline: 10-20ms callback overhead per chunk (GIL contention)
// v2.0 target: <2ms (WASAPI exclusive mode on Windows, default low_latency elsewhere)
//
// Note: miniaudio.h + miniaudio.c are vendor'd into src_c/miniaudio/.
// build.rs compiles them via cc::Build.

/// Audio device config — matches miniaudio's ma_device_config
#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub block_size: u32,      // samples per chunk (default 1920 @ 24kHz = 80ms)
    pub low_latency: bool,   // miniaudio profile
    pub exclusive_mode: bool, // WASAPI exclusive (Windows only)
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

// TODO P1: miniaudio FFI declarations and Rust-safe wrapper
// Currently in build.rs we compile miniaudio.c into the crate; the actual
// ma_device callbacks will be wired in P1.
