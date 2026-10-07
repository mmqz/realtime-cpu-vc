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

// miniaudio FFI + safe wrapper (compiled from src_c/miniaudio/miniaudio.c
// + miniaudio_shim.c via build.rs). Provides `AudioDevice` (RAII handle for
// the C-side `ma_device`) with a safe closure-based `open` API — no
// `unsafe extern "C" fn` is exposed publicly. See `miniaudio_ffi.rs` for
// the design rationale (C shim instead of direct 1:1 FFI over the
// multi-KB miniaudio structs).
mod miniaudio_ffi;
pub use miniaudio_ffi::AudioDevice;

// STFT + cached mel filter bank (OPT-3): Rust replacement for librosa on the
// streaming path. `realfft` (R2C FFT) + cached Slaney mel triangles → ~1-2ms
// per chunk vs. ~15-25ms for v1.0's Python librosa.stft. See `stft.rs`.
mod stft;
pub use stft::{linear_stft_magnitude, log_mel_spec, stft_magnitude, MelFilterBank};

// StreamingPipeline (OPT-8): 2 SPSC rings + decoder thread for pipeline
// parallelism between audio I/O (miniaudio GIL-free callback) and the heavy
// encoder/decoder compute. See `streaming_pipeline.rs` for the architecture
// diagram + the (deferred) miniaudio callback wiring plan.
mod streaming_pipeline;
pub use streaming_pipeline::StreamingPipeline;

// Causal StreamingConv1d (OPT-14): stateful causal 1D convolution that
// eliminates the algorithmic look-ahead latency of v1.0's centered-padding
// Conv1d, removing the need for SOLA chunk-boundary crossfade. Maintains a
// `(kernel_size - 1) * channels` sample state buffer across `process()` calls
// — each call returns the same length as its input (zero added latency). The
// placeholder convolution is identity; the real ONNX `Conv1d` with left-only
// padding lands in OPT-15. See `causal_conv.rs` for the design rationale.
mod causal_conv;
pub use causal_conv::StreamingConv1dState;

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

/// Load 4 f32s from a slice as an `f32x4` via a single unaligned SIMD load.
///
/// This compiles to a single `movups` (x86 SSE/AVX), `vld1q_f32` (ARM NEON),
/// or `v128.load` (wasm) — one instruction instead of the 4-scalar-loads +
/// `setps`/`insertps` sequence emitted by `f32x4::from([s[0], s[1], s[2], s[3]])`.
///
/// # Safety
/// Caller must guarantee `slice.len() >= 4` (use with `chunks_exact(4)` —
/// the resulting chunks are provably 4 elements wide).
#[inline(always)]
unsafe fn load_f32x4(slice: &[f32]) -> f32x4 {
    // SAFETY: `f32x4` is `#[repr(C, align(16))]` on x86 SSE, `#[repr(C)]` on
    // ARM NEON, and `#[repr(transparent)]` over `v128` on wasm — all 16
    // bytes (4 × sizeof(f32)) and `Copy`. `read_unaligned` emits a single
    // unaligned SIMD load instruction (no memcpy fallback for 16-byte Copy
    // types when the LLVM backend can lower to movups/vld1q).
    std::ptr::read_unaligned(slice.as_ptr() as *const f32x4)
}

/// Store an `f32x4` to a slice via a single unaligned SIMD store.
///
/// Counterpart to [`load_f32x4`] — emits `movups`/`st1q`/`v128.store`.
///
/// # Safety
/// Caller must guarantee `slice.len() >= 4`.
#[inline(always)]
unsafe fn store_f32x4(v: f32x4, slice: &mut [f32]) {
    std::ptr::write_unaligned(slice.as_mut_ptr() as *mut f32x4, v);
}

/// SIMD-accelerated dot product using `wide::f32x4` (4 floats per iteration).
/// On x86: AVX2/FMA via `wide`. On ARM: NEON via `wide`.
#[inline(always)]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = f32x4::from([0.0f32; 4]);
    let chunks = a.as_chunks::<4>().0.iter().zip(b.chunks_exact(4));
    for (ac, bc) in chunks {
        // Safe + branchless load — `chunks_exact(4)` guarantees 4 elements.
        // Single `movups`/`vld1q` SIMD load (vs. 4 scalar loads + insertps
        // that `f32x4::from([ac[0], ac[1], ac[2], ac[3]])` would emit).
        let av = unsafe { load_f32x4(ac) };
        let bv = unsafe { load_f32x4(bc) };
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
// 2. DDSP harmonic synthesis — Rust SIMD (wide::f32x4 + Agner-Fog sin)
// ============================================================
// Replaces: prototype/modules/decoder.py:_synthesize_source (harmonic part)
// v1.0 baseline: ~25ms per chunk (Python + numpy + torch.sin)
// v2.0 target: ~0.1ms (Rust + wide f32x4 + Agner-Fog SIMD sin)
//
// Faithful to tinyvc/module/tinyvc/decoder.py:24-54 (oscillate_harmonics).
// Differences from the original Torch impl:
//   - We use `cumsum` of `f0 * 2π/sr` directly (no modulo-1 wrap, sin() handles it).
//   - Voiced/unvoiced mask is applied upstream via amp_k (zero on unvoiced frames).
//   - Linear interpolation upsample happens upstream (matching decoder.py:132).
//
// SIMD strategy:
//   - `wide::f32x4::sin()` is a polynomial approximation based on Agner Fog's
//     vector class library — 4-wide SIMD sin (~10ns for 4 samples vs ~40ns for
//     4× std::f32::sin which calls glibc's sin). Same algorithm internally
//     as the one numpy's vectorized sin uses.
//   - Inner loop processes 4 samples at a time via chunks_exact, then a scalar
//     `fast_sin` (Taylor 4-term) handles the 0–3 sample remainder.
//   - The harmonic accumulation step uses `mul_add` (FMA) where available.

/// Fast scalar sin approximation — Agner-Fog polynomial (same as `wide::f32x4::sin`).
///
/// Algorithm:
///   1. Take |x|, find quadrant `q = round(|x| * 2/π)` (integer multiple of π/2).
///   2. Range-reduce: `x_r = |x| - q * (DP1 + DP2 + DP3)` → x_r ∈ ~[-π/4, π/4].
///      Split-mantissa subtraction (3 constants summing to π/2) preserves accuracy.
///   3. Evaluate minimax polynomial sin/cos of `x_r` (degree-3 in `x²`).
///   4. Pick sin or cos polynomial based on quadrant parity (sin(π/2 + x) = cos(x)).
///   5. Sign flip via bit manipulation: `(q << 30) ^ x_bits` at bit 31.
///   6. Overflow protection: if q > 2^25, x is too large for the algorithm; return 0.
///
/// Accuracy: ~1e-6 globally (matches `wide::f32x4::sin` exactly — same polynomial,
/// same range reduction, same sign logic).
/// Used only for the 0–3 sample remainder in `synth_harmonics`. The main path
/// uses `wide::f32x4::sin` directly (4 samples per call).
#[inline(always)]
fn fast_sin(x: f32) -> f32 {
    // Split-mantissa π/2 constants — sum exactly to π/2 with ~3 mantissa worth
    // of precision (one is exact, two absorb the rounding error).
    const DP1: f32 = 0.78515625 * 2.0; // 1.5703125 — exact
    const DP2: f32 = 2.418_756_5E-4 * 2.0; // ~4.8e-4
    const DP3: f32 = 3.774_895E-8 * 2.0; // ~7.5e-8

    // Minimax polynomial coefficients (sin & cos, degree 2 in x²).
    const P0_SIN: f32 = -1.666_665_5E-1;
    const P1_SIN: f32 = 8.332_161E-3;
    const P2_SIN: f32 = -1.951_529_6E-4;
    const P0_COS: f32 = 4.166_664_6E-2;
    const P1_COS: f32 = -1.388_731_6E-3;
    const P2_COS: f32 = 2.443_315_7E-5;

    const TWO_OVER_PI: f32 = 2.0 / core::f32::consts::PI;
    // Beyond this magnitude, the algorithm breaks down — return 0 (mirrors
    // `wide::f32x4::sin_cos` overflow protection).
    const OVERFLOW_Q: i32 = 0x2000000; // 2^25

    let xa = x.abs();

    // Quadrant index (integer nearest to |x| * 2/π).
    let y = (xa * TWO_OVER_PI).round();
    let q = y as i32;

    // Range reduction via split-mantissa subtraction.
    // Result: x_r = xa - y*DP1 - y*DP2 - y*DP3 ≈ x mod π/2.
    // Split-mantissa preserves precision: DP1 is exact, DP2/DP3 absorb rounding.
    // (For scalar f32 there's no `mul_neg_add`; we write direct subtraction and
    // let the compiler emit FMA when targeting `+fma`.)
    let x_r = xa - y * DP1 - y * DP2 - y * DP3;

    let x2 = x_r * x_r;

    // sin(x_r) = x_r + x_r³ * (P0 + x²*(P1 + x²*P2))
    //         = x_r + (x_r * x²) * poly_sin
    let poly_sin = P0_SIN + x2 * (P1_SIN + x2 * P2_SIN);
    let s = (x_r * x2).mul_add(poly_sin, x_r);

    // cos(x_r) = 1 - x²/2 + x⁴ * (P0 + x²*(P1 + x²*P2))
    let poly_cos = P0_COS + x2 * (P1_COS + x2 * P2_COS);
    let c = (x2 * x2).mul_add(poly_cos, 1.0 - 0.5 * x2);

    // Overflow protection: very large inputs break the quadrant logic.
    let (s, c) = if q > OVERFLOW_Q && xa.is_finite() {
        (0.0f32, 1.0f32)
    } else {
        (s, c)
    };

    // Pick sin or cos based on quadrant parity.
    // q odd → sin(π/2 * q + x_r) = cos(x_r) if (q-1)/2 is even, else -cos(x_r).
    // q even → sin(π/2 * q + x_r) = sin(x_r) if q/2 is even, else -sin(x_r).
    // All of this collapses to: pick s or c, then flip sign per quadrant.
    let sin1 = if q & 1 != 0 { c } else { s };

    // Sign flip via bit-twiddling — mirrors `wide::f32x4::flip_signs`.
    // `sign_bits = (q << 30) ^ x_bits`, but we only care about bit 31 (the float
    // sign bit). Bit 31 of `(q << 30)` = bit 1 of q, so:
    //   - q mod 4 ∈ {0, 1} → bit 1 of q = 0 → sign bit clear
    //   - q mod 4 ∈ {2, 3} → bit 1 of q = 1 → sign bit set
    // Combined with x's own sign bit: result is negative iff (q mod 4 ∈ {2, 3})
    // XOR (x < 0), which is the correct sign for sin in each quadrant.
    let q_sign_bit = (((q as u32) & 2) << 30); // bit 1 of q → bit 31
    let x_sign_bit = x.to_bits() & 0x8000_0000;
    let result_bits = sin1.to_bits() ^ q_sign_bit ^ x_sign_bit;
    f32::from_bits(result_bits)
}

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
    //    Sequential (data dependency between iterations) — cannot be SIMD'd.
    let mut phase = vec![0.0f32; t];
    let phase_step = 2.0 * core::f32::consts::PI / sr as f32;
    let mut acc = 0.0f32;
    for i in 0..t {
        acc += f0_upsampled[i] * phase_step;
        phase[i] = acc;
    }

    // 2. For each harmonic k=1..=n_harmonics, accumulate amp_k[i] * sin(k * phase[i]).
    //    Skip k=0 (DC component, per DDSP convention — amp[0..T] is the DC row).
    //
    //    Inner loop vectorizes 4 samples at a time using `wide::f32x4`:
    //      - 1 SIMD sin call per 4 samples (vs 4× std::f32::sin in the old code)
    //      - mul_add (FMA) for `amp_k * sin + output` accumulation
    //    Remainder (0..3 samples) falls back to scalar `fast_sin`.
    let amp_stride = t; // amplitudes shape [n_harmonics+1, T]
    for k in 1..=n_harmonics {
        let amp_k = &amplitudes[k * amp_stride..(k + 1) * amp_stride];
        let kf = f32x4::from(k as f32);

        // SIMD chunk loop: 4 samples per iteration.
        // `chunks_exact(4)` guarantees each chunk has exactly 4 elements, so
        // indexing [0..4] is provably in-bounds — bounds-check elided.
        let phase_chunks = phase.chunks_exact(4);
        let amp_chunks = amp_k.chunks_exact(4);
        let out_chunks = output.chunks_exact_mut(4);
        for (out_chunk, (amp_chunk, phase_chunk)) in
            out_chunks.zip(amp_chunks.zip(phase_chunks))
        {
            // Single SIMD load per chunk (movups / vld1q) — replaces 4
            // scalar `f32x4::from([s[0], s[1], s[2], s[3]])` loads.
            let phase_v = unsafe { load_f32x4(phase_chunk) };
            let amp_v = unsafe { load_f32x4(amp_chunk) };
            // sin(k * phase) — 1 SIMD sin call per 4 samples (Agner-Fog polynomial).
            let sin_v = (phase_v * kf).sin();
            // FMA-accumulate: out_v += amp_v * sin_v.
            let new_out = amp_v.mul_add(sin_v, unsafe { load_f32x4(out_chunk) });
            // Single SIMD store (movups / st1q) — replaces 4 scalar writes.
            unsafe { store_f32x4(new_out, out_chunk) };
        }

        // Scalar remainder (0..3 samples) — uses fast_sin (Taylor 4-term).
        let n_chunks = t / 4;
        let chunked_len = n_chunks * 4;
        let kf_scalar = k as f32;
        for i in chunked_len..t {
            output[i] += amp_k[i] * fast_sin(kf_scalar * phase[i]);
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
// 5. F0 logits → Hz decoding (TinyVC PitchEstimator.decode)
// ============================================================
// Replaces: repos/tinyvc/module/tinyvc/encoder.py:PitchEstimator.decode
// Input:  f0_logits [B, 512, T] (flattened, row-major)
// Output: f0 in Hz   [B, T]
//
// Matches Python's `decode(logits, k=4)`:
//   probs, indices = torch.topk(logits, k, dim=1)       # top-k along bin axis
//   probs = F.softmax(probs, dim=1)                      # softmax over the k topk only
//   freqs = self.id2freq(indices)                         # fmin * 2^(idx / cpo), 0 if <= fmin
//   f0 = (probs * freqs).sum(dim=1)                       # weighted average
//   f0[f0 <= self.min_frequency] = 0                      # silence floor
//
// Defaults from `PitchEstimator.__init__`:
//   num_classes = 512, classes_per_octave = 48, min_frequency = 20.0, k = 4.

/// Convert f0 logits `[B, n_bins, T]` to F0 in Hz `[B, T]` via top-k softmax
/// weighted average, matching TinyVC's `PitchEstimator.decode`.
///
/// Parameters:
/// - `f0_logits`: `&[f32]` shape `[batch, n_bins, time]` (flattened, row-major).
/// - `batch`: B.
/// - `n_bins`: 512 (`PitchEstimator.num_classes`).
/// - `time`: T.
/// - `bins_per_octave`: 48 (`PitchEstimator.classes_per_octave`).
/// - `fmin`: 20.0 (`PitchEstimator.min_frequency`, Hz).
/// - `top_k`: 4 (`PitchEstimator.decode` default k).
///
/// Returns `Vec<f32>` shape `[batch, time]` with F0 in Hz; frames whose decoded
/// F0 is `<= fmin` are set to 0 (matches `f0[f0 <= min_frequency] = 0`).
pub fn decode_f0_logits(
    f0_logits: &[f32],
    batch: usize,
    n_bins: usize,
    time: usize,
    bins_per_octave: usize,
    fmin: f32,
    top_k: usize,
) -> Vec<f32> {
    // Precompute bin → frequency mapping (matches PitchEstimator.id2freq):
    //   freq(i) = fmin * 2^(i / cpo);  freq = 0 if freq <= fmin (bin 0).
    let bin_to_freq: Vec<f32> = (0..n_bins)
        .map(|i| {
            let f = fmin * 2.0_f32.powf(i as f32 / bins_per_octave as f32);
            if f <= fmin { 0.0 } else { f }
        })
        .collect();

    let mut output = vec![0.0f32; batch * time];

    for b in 0..batch {
        for t in 0..time {
            // Logits for this (batch, time) frame: shape [n_bins].
            let base = b * n_bins * time + t * n_bins;
            let logits = &f0_logits[base..base + n_bins];

            // top-k indices via repeated maximum scan (k=4 → O(k·n) = 2048 ops).
            // For larger k, a partial-sort would be better, but k=4 is small and
            // the simple scan keeps this dependency-free.
            let mut top_indices: Vec<usize> = Vec::with_capacity(top_k);
            let mut top_logits: Vec<f32> = Vec::with_capacity(top_k);
            for _ in 0..top_k {
                let mut best_idx = 0usize;
                let mut best_logit = f32::NEG_INFINITY;
                for (i, &l) in logits.iter().enumerate() {
                    if !top_indices.contains(&i) && l > best_logit {
                        best_logit = l;
                        best_idx = i;
                    }
                }
                top_indices.push(best_idx);
                top_logits.push(best_logit);
            }

            // Numerically stable softmax over the k topk logits only (matches
            // `F.softmax(probs, dim=1)` where `probs` is the [B, k, T] topk tensor).
            let max_logit = top_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = top_logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

            // Weighted average: sum(prob_i * freq_i) (matches `(probs*freqs).sum(dim=1)`).
            let weighted_sum: f32 = top_indices
                .iter()
                .zip(probs.iter())
                .map(|(&idx, &p)| p * bin_to_freq[idx])
                .sum();

            // Silence floor: f0[f0 <= fmin] = 0.
            output[b * time + t] = if weighted_sum > fmin { weighted_sum } else { 0.0 };
        }
    }

    output
}

// ============================================================
// 6. kNN-VC feature retrieval — Rust SIMD cosine + top-k
// ============================================================
// Replaces: prototype/src/vc_realtime/infer_v1.py:V1Infer::knn_replace
//           (wraps tinyvc/module/tinyvc/feature_retrieval.py:match_features)
// v1.0 baseline: torch.bmm + torch.topk (Python + torch, ~10ms per chunk on CPU)
// v2.0 target: ~1ms (Rust + wide f32x4 SIMD dot product + partial sort)
//
// Faithful to tinyvc's `match_features(source, reference, k=4, alpha=0, metrics='cos')`:
//   1. For each source frame `t` (length-T_src sequence of dim-D vectors):
//      a. Compute cosine similarity with every target frame:
//            sim[t, r] = dot(src[t], tgt[r]) / ((||src[t]|| + 1e-6) * (||tgt[r]|| + 1e-6))
//         IMPORTANT: the +1e-6 is added to each norm BEFORE the division — this
//         matches the Python reference exactly:
//           `source_norm = torch.norm(source, dim=2, keepdim=True, p=2) + 1e-6`
//           `sims = torch.bmm(source / source_norm, (reference / reference_norm).T)`
//         Do NOT fold the epsilon into the final cosine sim — that would diverge.
//      b. Take the top-k target frames by similarity descending (k defaults to 4,
//         matching upstream TinyVC). Ties broken by lower index (matches
//         `torch.topk`'s ascending-index ordering on ties).
//      c. Replace `src[t]` with the SIMPLE AVERAGE of those top-k target frames
//         (matches Python `.mean(dim=2)`).
//
//         NOTE: The task spec mentioned "1/score² weighting", but the ACTUAL
//         Python reference (`feature_retrieval.py:30`) uses `torch.stack(...).mean(dim=2)`
//         — a SIMPLE AVERAGE, not a similarity-weighted average. We follow the
//         Python algorithm to ensure bit-equivalent v1↔v2 output. A similarity-
//         weighted variant would diverge numerically and is intentionally NOT
//         provided here.
//
//      d. Optional alpha blend: `out = result * (1 - alpha) + source * alpha`.
//         Default alpha=0.0 = full target replace (matches `V1Infer::knn_replace`).
//
// Memory layout (frame-major, contiguous per frame — cache-friendly for the
// SIMD dot product loop):
//   - `source`: `&[f32]` flat row-major, shape `[T_src, dim]`. Frame `t` is
//     `source[t * dim .. (t + 1) * dim]`.
//     (Numpy equivalent: `np.ascontiguousarray(content.transpose(0, 2, 1))` for
//      `[B=1, C=768, T]` → `[T, 768]` C-order — the PyO3 wrapper does this.)
//   - `target`: same layout, shape `[T_ref, dim]`.
//   - Returns: `Vec<f32>` of length `t_src * dim`, same layout as `source`.

/// kNN-VC feature retrieval: replace each source content frame with the simple
/// average of the top-k most cosine-similar target voice frames.
///
/// Matches the v1.0 Python path
/// (`V1Infer::knn_replace(content, voice_id)` →
/// `tinyvc.match_features(metrics='cos', alpha=0)`) numerically — bit-equivalent
/// up to f32 reduction order.
///
/// See the section-6 docs above for the exact algorithm and the rationale for
/// using a simple average (not 1/sim² weighting — that's NOT what Python does).
///
/// # Arguments
/// * `source` — flat row-major `[T_src * dim]`; source frame `t` at
///   `source[t*dim .. (t+1)*dim]`.
/// * `target` — flat row-major `[T_ref * dim]`; target frame `r` at
///   `target[r*dim .. (r+1)*dim]`.
/// * `dim` — feature dimensionality (768 for distilled WavLM-Base-Plus).
/// * `t_src` — number of source frames.
/// * `t_ref` — number of pre-stored target voice frames.
/// * `top_k` — kNN top-k (typically 4, matching upstream TinyVC default).
///
/// # Returns
/// `Vec<f32>` of length `t_src * dim` — replaced content features, same layout
/// as `source`. Each frame `t` is the simple average of the `min(top_k, t_ref)`
/// most cosine-similar target frames (with the same `+1e-6` norm epsilon as the
/// Python reference).
pub fn knn_retrieve(
    source: &[f32],
    target: &[f32],
    dim: usize,
    t_src: usize,
    t_ref: usize,
    top_k: usize,
) -> Vec<f32> {
    knn_retrieve_with_alpha(source, target, dim, t_src, t_ref, top_k, 0.0)
}

/// `knn_retrieve` with an explicit `alpha` blend factor
/// (0 = full target replace, 1 = identity).
/// Mirrors `tinyvc.match_features`'s `alpha` parameter.
pub fn knn_retrieve_with_alpha(
    source: &[f32],
    target: &[f32],
    dim: usize,
    t_src: usize,
    t_ref: usize,
    top_k: usize,
    alpha: f32,
) -> Vec<f32> {
    // Degenerate cases — return empty / source unchanged.
    if dim == 0 || t_src == 0 {
        return Vec::new();
    }
    if t_ref == 0 {
        // No target frames to match against → identity (alpha = 1.0 effectively).
        return source.to_vec();
    }

    let effective_k = top_k.min(t_ref);
    let one_minus_alpha = 1.0 - alpha;
    let inv_k = 1.0 / effective_k as f32;

    // Pre-compute target norms (with +1e-6 epsilon — matches Python reference).
    // Reused across all source frames, so we compute once.
    let target_norms: Vec<f32> = (0..t_ref)
        .map(|r| {
            let frame = &target[r * dim..(r + 1) * dim];
            let sumsq: f32 = frame.iter().map(|x| x * x).sum();
            sumsq.sqrt() + 1e-6
        })
        .collect();

    let mut output = vec![0.0f32; dim * t_src];

    for t in 0..t_src {
        let src_frame = &source[t * dim..(t + 1) * dim];
        let src_norm = {
            let sumsq: f32 = src_frame.iter().map(|x| x * x).sum();
            sumsq.sqrt() + 1e-6
        };

        // Compute cosine sim with each target frame; collect (idx, sim) pairs.
        // The dot product is the hot path — uses the SIMD `dot_product_simd`
        // helper (4-wide `wide::f32x4` with remainder fallback).
        let mut sims: Vec<(usize, f32)> = (0..t_ref)
            .map(|r| {
                let tgt_frame = &target[r * dim..(r + 1) * dim];
                let dot = dot_product_simd(src_frame, tgt_frame);
                let sim = dot / (src_norm * target_norms[r]);
                (r, sim)
            })
            .collect();

        // Sort by similarity descending. For ties, prefer lower index (matches
        // `torch.topk`'s ascending-index tie-breaking).
        //
        // O(N) partial sort: `select_nth_unstable_by(k - 1, cmp)` partitions
        // `sims` so the top-k largest are at indices [0, k) (unordered), then we
        // sort just that small slice (k log k — typically k=4 ⇒ 4*log2(4) ≈ 8
        // comparisons vs. N*log2(N) ≈ 1500*10 = 15000 for the old full sort,
        // i.e. ~1800× fewer comparisons when N=1500).
        //
        // Also switch from `partial_cmp().unwrap_or()` to `f32::total_cmp`
        // (Rust 1.62+) — branchless (just an integer compare on the bit
        // representations) and handles NaN correctly (no `.unwrap_or` panic
        // guard). Typically ~2× faster per comparison.
        let k = effective_k; // >= 1 since t_ref >= 1 (early-return above).
        let partition_cmp = |a: &(usize, f32), b: &(usize, f32)| {
            b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
        };
        // Partition: top-k sims end up at sims[0..k] (unordered).
        sims.select_nth_unstable_by(k - 1, partition_cmp);
        // Sort just the top-k slice — k=4 typically.
        sims[0..k].sort_by(partition_cmp);

        // Simple average of the top-k target frames (matches Python `.mean(dim=2)`).
        // Apply the alpha blend inline to avoid a second pass over the output.
        for d in 0..dim {
            let mut acc = 0.0f32;
            for (idx, _) in sims.iter().take(effective_k) {
                acc += target[idx * dim + d];
            }
            let mean_topk = acc * inv_k;
            output[t * dim + d] = one_minus_alpha * mean_topk + alpha * src_frame[d];
        }
    }

    output
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

        // Reference: scalar reimplementation of the algorithm using std::f32::sin.
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

        // Tolerance: 1e-4 — `wide::f32x4::sin` (Agner-Fog polynomial) vs
        // `std::f32::sin` (glibc) differ by ~1e-6 per sample; for 2 harmonics
        // summed this is well within 1e-4. The polynomial approximation is
        // MORE accurate than the audio application requires (DDSP targets
        // 16-bit PCM = 1.5e-5 SNR floor).
        let mut max_err = 0.0f32;
        for i in 0..t {
            let err = (out[i] - reference[i]).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 1e-4,
                "mismatch at {i}: rust={}, ref={}, err={}",
                out[i],
                reference[i],
                err
            );
        }
        // Sanity: the actual max error should be much smaller than the threshold.
        // If this fires, it means the SIMD sin accuracy regressed — investigate
        // before relaxing the threshold above.
        assert!(
            max_err < 5e-5,
            "max_err={max_err} unexpectedly high — SIMD sin accuracy regressed?"
        );
    }

    #[test]
    fn test_fast_sin_accuracy_vs_std_sin() {
        // Verify fast_sin (Taylor 4-term) matches std::f32::sin within 1e-4
        // across the typical audio phase range [-2000, 2000] rad.
        // (2000 rad covers ~14 harmonics × 1920 samples × 150Hz × 2π/24000 = ~1055 rad.)
        let mut max_err = 0.0f32;
        let mut worst_x = 0.0f32;
        // Test 0.01-step points over [-2000, 2000] — 400k samples.
        let n = 200_000i32;
        for i in -n..=n {
            let x = (i as f32) * 0.01;
            let reference = x.sin();
            let approx = fast_sin(x);
            let err = (approx - reference).abs();
            if err > max_err {
                max_err = err;
                worst_x = x;
            }
            assert!(
                err < 1e-4,
                "fast_sin({x}) = {approx}, std = {reference}, err = {err} (> 1e-4)"
            );
        }
        // Sanity check: report worst case so future regressions are caught.
        eprintln!(
            "test_fast_sin_accuracy_vs_std_sin: max_err={max_err} at x={worst_x}"
        );
    }

    #[test]
    fn test_synth_harmonics_unaligned_length() {
        // T not divisible by 4 (e.g. 479 = 119 chunks + 3 remainder)
        // exercises the scalar fallback in synth_harmonics.
        let sr = 24000u32;
        let t = 479; // 4*119 + 3
        let f0 = vec![150.0f32; t];
        let n_h = 3;
        let mut amps = vec![0.0f32; (n_h + 1) * t];
        for i in 0..t {
            amps[1 * t + i] = 0.5;
            amps[2 * t + i] = 0.25;
            amps[3 * t + i] = 0.125;
        }
        let out = synth_harmonics(&f0, &amps, n_h, sr);
        assert_eq!(out.len(), t);

        // Reference scalar computation using std::f32::sin.
        let phase_step = 2.0 * std::f32::consts::PI / sr as f32;
        let mut acc = 0.0f32;
        let phase: Vec<f32> = f0.iter().map(|f| { acc += f * phase_step; acc }).collect();
        let reference: Vec<f32> = (0..t).map(|i| {
            0.5 * (1.0 * phase[i]).sin() + 0.25 * (2.0 * phase[i]).sin() + 0.125 * (3.0 * phase[i]).sin()
        }).collect();
        let mut max_err = 0.0f32;
        for i in 0..t {
            let err = (out[i] - reference[i]).abs();
            if err > max_err { max_err = err; }
            assert!(err < 1e-4, "mismatch at {i}: rust={}, ref={}, err={}", out[i], reference[i], err);
        }
        eprintln!("test_synth_harmonics_unaligned_length: max_err={max_err}");
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

    /// Pure-Rust microbenchmark — measures `synth_harmonics` itself, no PyO3
    /// list-Vec conversion overhead. Used to verify the SIMD sin optimization
    /// actually speeds up the inner compute (vs the scalar baseline).
    /// Run with `cargo test --release --bench -- --nocapture` or just
    /// `cargo test --release test_synth_harmonics_bench -- --nocapture --ignored`.
    #[test]
    fn test_synth_harmonics_bench() {
        let t = 1920usize;
        let n_harmonics = 14usize;
        let f0 = vec![150.0f32; t];
        let mut amps = vec![0.0f32; (n_harmonics + 1) * t];
        for k in 1..=n_harmonics {
            for i in 0..t {
                amps[k * t + i] = 1.0 / k as f32;
            }
        }
        // Warmup (JIT-like cache effects in OS/allocator).
        for _ in 0..5 {
            let _ = synth_harmonics(&f0, &amps, n_harmonics, 24000);
        }
        let n_iter = 1000;
        let start = std::time::Instant::now();
        for _ in 0..n_iter {
            std::hint::black_box(synth_harmonics(
                std::hint::black_box(&f0),
                std::hint::black_box(&amps),
                std::hint::black_box(n_harmonics),
                std::hint::black_box(24000),
            ));
        }
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let per_call_ms = total_ms / n_iter as f64;
        eprintln!(
            "test_synth_harmonics_bench: {n_iter} iters, total={total_ms:.2}ms, per_call={per_call_ms:.4}ms"
        );
        // Sanity bound: even with all the SIMD, we shouldn't exceed 1ms per call
        // (14 harmonics × 1920 samples × 1 sin call per 4 = 6720 sin calls × ~10ns = 67μs).
        // The bound is generous (10×) to avoid flakiness on shared CI runners.
        assert!(
            per_call_ms < 1.0,
            "synth_harmonics took {per_call_ms}ms/call — expected <1ms with SIMD sin"
        );
    }

    // ============================================================
    // decode_f0_logits tests
    // ============================================================

    #[test]
    fn test_decode_f0_logits_basic() {
        // Bin 96 → freq = 20 * 2^(96/48) = 20 * 2^2 = 80 Hz.
        // A single dominant logit at bin 96 should yield F0 ≈ 80 Hz.
        let n_bins = 512;
        let time = 10;
        let mut logits = vec![0.0f32; n_bins * time];
        for t in 0..time {
            logits[96 + t * n_bins] = 10.0;
        }
        let f0 = decode_f0_logits(&logits, 1, n_bins, time, 48, 20.0, 4);
        assert_eq!(f0.len(), time);
        for t in 0..time {
            assert!(
                (f0[t] - 80.0).abs() < 5.0,
                "f0[{}]={}, expected ~80 Hz",
                t,
                f0[t]
            );
        }
    }

    #[test]
    fn test_decode_f0_logits_interpolation() {
        // Two adjacent bins with equal high logits → F0 should land between them.
        //   bin 48 → 20 * 2^(48/48) = 40 Hz
        //   bin 49 → 20 * 2^(49/48) ≈ 40.58 Hz
        let n_bins = 512;
        let time = 1;
        let mut logits = vec![0.0f32; n_bins];
        logits[48] = 5.0;
        logits[49] = 5.0;
        let f0 = decode_f0_logits(&logits, 1, n_bins, time, 48, 20.0, 4);
        assert!(
            f0[0] > 39.0 && f0[0] < 41.0,
            "f0={}, expected ~40.3 Hz (interpolation between bins 48 and 49)",
            f0[0]
        );
    }

    #[test]
    fn test_decode_f0_logits_silence_floor() {
        // When the weighted F0 ≤ fmin, Python sets it to 0 (`f0[f0 <= min_frequency] = 0`).
        // Force top-k to be dominated by bin 0 (whose freq = 0 by id2freq), so the
        // weighted average collapses to 0 → silence floor → output 0.
        let n_bins = 512;
        let time = 1;
        let mut logits = vec![f32::NEG_INFINITY; n_bins];
        logits[0] = 100.0;
        logits[1] = -100.0;
        logits[2] = -100.0;
        logits[3] = -100.0;
        let f0 = decode_f0_logits(&logits, 1, n_bins, time, 48, 20.0, 4);
        assert_eq!(f0[0], 0.0, "silence floor should produce 0 Hz, got {}", f0[0]);
    }

    #[test]
    fn test_decode_f0_logits_batch_and_time() {
        // B=2, T=3: each (b,t) frame has a distinct dominant bin → distinct F0.
        //   bin 96  → 80 Hz
        //   bin 144 → 20 * 2^(144/48) = 20 * 2^3 = 160 Hz
        //   bin 192 → 20 * 2^(192/48) = 20 * 2^4 = 320 Hz
        let n_bins = 512;
        let batch = 2;
        let time = 3;
        let mut logits = vec![0.0f32; batch * n_bins * time];
        let targets = [96usize, 144, 192];
        for b in 0..batch {
            for t in 0..time {
                let base = b * n_bins * time + t * n_bins;
                logits[base + targets[t]] = 10.0;
            }
        }
        let f0 = decode_f0_logits(&logits, batch, n_bins, time, 48, 20.0, 4);
        assert_eq!(f0.len(), batch * time);
        let expected = [80.0f32, 160.0, 320.0];
        for b in 0..batch {
            for t in 0..time {
                let got = f0[b * time + t];
                assert!(
                    (got - expected[t]).abs() < 5.0,
                    "b={b} t={t}: f0={got}, expected ~{}",
                    expected[t]
                );
            }
        }
    }

    // ============================================================
    // knn_retrieve tests
    // ============================================================

    #[test]
    fn test_knn_retrieve_basic() {
        // Single source frame, single target frame = source → top-1 returns target
        // verbatim. With k=4 > t_ref=1, effective k clamps to 1.
        let dim = 768;
        let t_src = 1;
        let t_ref = 1;

        let mut source = vec![0.0f32; dim];
        source[0] = 1.0; // source frame = [1, 0, 0, ..., 0]

        // Target is identical to source.
        let target = source.clone();

        let out = knn_retrieve(&source, &target, dim, t_src, t_ref, 4);
        assert_eq!(out.len(), dim);
        // Top-1 = target frame 0 = source. Output (alpha=0) = mean_topk = target.
        assert!(
            (out[0] - 1.0).abs() < 1e-5,
            "out[0]={}, expected ~1.0",
            out[0]
        );
        assert!(
            (out[1] - 0.0).abs() < 1e-5,
            "out[1]={}, expected ~0.0",
            out[1]
        );
    }

    #[test]
    fn test_knn_retrieve_similar_dominates() {
        // 3 target frames: 2 identical-to-source (sim ≈ 1) + 1 orthogonal (sim ≈ 0).
        // Simple average of top-3 (k=4 > t_ref=3 → effective k=3) =
        //   (frame0 + frame1 + frame2) / 3
        //   = ([1,0,...] + [1,0,...] + [0,1,0,...]) / 3 = [2/3, 1/3, 0, ...]
        // The two similar frames together contribute 2/3 to out[0] (> 0.5).
        let dim = 768;
        let t_src = 1;
        let t_ref = 3;

        let mut source = vec![0.0f32; dim];
        source[0] = 1.0;

        let mut target = vec![0.0f32; dim * t_ref];
        // Frame 0: identical to source (cos sim ≈ 1.0)
        target[0] = 1.0;
        // Frame 1: identical to source (cos sim ≈ 1.0)
        target[dim + 0] = 1.0;
        // Frame 2: orthogonal to source (cos sim ≈ 0.0)
        target[2 * dim + 1] = 1.0;

        let out = knn_retrieve(&source, &target, dim, t_src, t_ref, 4);
        // mean of [1,0,0,...] + [1,0,0,...] + [0,1,0,...] = [2/3, 1/3, 0, ...]
        assert!(
            (out[0] - 2.0 / 3.0).abs() < 1e-5,
            "out[0]={}, expected 2/3 ≈ 0.667",
            out[0]
        );
        assert!(
            (out[1] - 1.0 / 3.0).abs() < 1e-5,
            "out[1]={}, expected 1/3 ≈ 0.333",
            out[1]
        );
        assert!(
            out[2].abs() < 1e-5,
            "out[2]={}, expected 0.0",
            out[2]
        );
    }

    #[test]
    fn test_knn_retrieve_preserves_shape() {
        let dim = 128;
        let t_src = 10;
        let t_ref = 50;
        let source: Vec<f32> = (0..dim * t_src).map(|i| (i as f32) * 0.001).collect();
        let target: Vec<f32> = (0..dim * t_ref).map(|i| (i as f32) * 0.001).collect();
        let out = knn_retrieve(&source, &target, dim, t_src, t_ref, 4);
        assert_eq!(out.len(), dim * t_src);
    }

    #[test]
    fn test_knn_retrieve_matches_python_reference() {
        // Bit-equivalence vs the Python `tinyvc.match_features(metrics='cos', alpha=0)`
        // scalar reference. This is the strongest correctness signal — if the Rust
        // SIMD impl ever diverges from the algorithm, this test fires.
        //
        // The reference impl below is the SAME algorithm as `knn_retrieve` but
        // written out scalar-style without SIMD. It catches:
        //   - SIMD dot product bugs (different rounding/reduction order)
        //   - Top-k sort/tie-breaking bugs
        //   - Epsilon placement bugs (e.g., +1e-6 on final sim vs on norms)
        //   - Average/alpha-blend bugs
        let dim = 16; // small dim → many SIMD-vs-scalar reduction-order differences
        let t_src = 5;
        let t_ref = 12;
        let top_k = 4;

        // Deterministic pseudo-random input (so test failures are reproducible).
        let mut seed = 0x1234u32;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };

        let source: Vec<f32> = (0..dim * t_src).map(|_| rng()).collect();
        let target: Vec<f32> = (0..dim * t_ref).map(|_| rng()).collect();

        // Scalar reference impl of Python `match_features(metrics='cos', alpha=0)`:
        //   norm(p) = sqrt(sum(p_i²)) + 1e-6
        //   sim[t, r] = dot(src[t], tgt[r]) / (norm(src[t]) * norm(tgt[r]))
        //   top_k_idx = top-k indices by sim (descending; ties → lower index)
        //   out[t] = mean(tgt[top_k_idx])
        let scalar_ref: Vec<f32> = {
            let mut out = vec![0.0f32; dim * t_src];
            for t in 0..t_src {
                let src_frame = &source[t * dim..(t + 1) * dim];
                let src_norm = {
                    let s: f32 = src_frame.iter().map(|x| x * x).sum();
                    s.sqrt() + 1e-6
                };
                let mut sims: Vec<(usize, f32)> = (0..t_ref)
                    .map(|r| {
                        let tgt_frame = &target[r * dim..(r + 1) * dim];
                        let dot: f32 = src_frame
                            .iter()
                            .zip(tgt_frame.iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        let tgt_norm = {
                            let s: f32 = tgt_frame.iter().map(|x| x * x).sum();
                            s.sqrt() + 1e-6
                        };
                        (r, dot / (src_norm * tgt_norm))
                    })
                    .collect();
                sims.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });

                let k = top_k.min(t_ref);
                let inv_k = 1.0 / k as f32;
                for d in 0..dim {
                    let mut acc = 0.0f32;
                    for (idx, _) in sims.iter().take(k) {
                        acc += target[idx * dim + d];
                    }
                    out[t * dim + d] = acc * inv_k;
                }
            }
            out
        };

        let rust_out = knn_retrieve(&source, &target, dim, t_src, t_ref, top_k);

        // Tolerance: f32 reduction order differences (SIMD vs scalar) → ~1e-6 per
        // element. For dim=16 with ~unit-magnitude random values, 1e-5 is generous.
        let mut max_err = 0.0f32;
        for i in 0..scalar_ref.len() {
            let err = (rust_out[i] - scalar_ref[i]).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 1e-5,
                "mismatch at i={i}: rust={}, ref={}, err={}",
                rust_out[i],
                scalar_ref[i],
                err
            );
        }
        eprintln!("test_knn_retrieve_matches_python_reference: max_err={max_err}");
    }

    #[test]
    fn test_knn_retrieve_alpha_identity() {
        // alpha=1.0 → output = source unchanged (identity blend).
        let dim = 32;
        let t_src = 2;
        let t_ref = 4;
        let source: Vec<f32> = (0..dim * t_src).map(|i| (i as f32) * 0.01).collect();
        let target: Vec<f32> = (0..dim * t_ref).map(|i| (i as f32) * 0.05).collect();
        let out = knn_retrieve_with_alpha(&source, &target, dim, t_src, t_ref, 4, 1.0);
        for i in 0..source.len() {
            assert!(
                (out[i] - source[i]).abs() < 1e-5,
                "alpha=1 should return source unchanged: out[{}] = {}, source = {}",
                i,
                out[i],
                source[i]
            );
        }
    }

    #[test]
    fn test_knn_retrieve_alpha_half_blend() {
        // alpha=0.5, k=1, single target frame:
        //   out = 0.5 * mean_topk + 0.5 * source
        //       = 0.5 * target + 0.5 * source
        let dim = 8;
        let t_src = 1;
        let t_ref = 1;
        let source: Vec<f32> = vec![2.0; dim]; // all 2.0
        let target: Vec<f32> = vec![4.0; dim]; // all 4.0
        let out = knn_retrieve_with_alpha(&source, &target, dim, t_src, t_ref, 1, 0.5);
        // top-1 mean = target = 4.0; output = 0.5*4.0 + 0.5*2.0 = 3.0.
        for i in 0..dim {
            assert!(
                (out[i] - 3.0).abs() < 1e-5,
                "alpha=0.5 mix failed at i={}: got {}, expected 3.0",
                i,
                out[i]
            );
        }
    }

    #[test]
    fn test_knn_retrieve_empty_target_returns_source() {
        // Edge case: t_ref=0 → no targets to match → return source unchanged.
        let dim = 8;
        let t_src = 3;
        let source: Vec<f32> = (0..dim * t_src).map(|i| i as f32).collect();
        let out = knn_retrieve(&source, &[], dim, t_src, 0, 4);
        assert_eq!(out.len(), source.len());
        for i in 0..source.len() {
            assert_eq!(out[i], source[i], "empty target should return source unchanged");
        }
    }

    #[test]
    fn test_knn_retrieve_topk_clamped_to_t_ref() {
        // top_k > t_ref → effective k = t_ref (Python's `torch.topk` semantics).
        // With t_ref=2 and top_k=4, the top-2 frames are used (all of them).
        let dim = 4;
        let t_src = 1;
        let t_ref = 2;
        let source: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let target: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, // frame 0: matches source
                                    0.0, 1.0, 0.0, 0.0]; // frame 1: orthogonal
        let out = knn_retrieve(&source, &target, dim, t_src, t_ref, 4);
        // top-2 = both frames; mean = ([1,0,0,0] + [0,1,0,0]) / 2 = [0.5, 0.5, 0, 0]
        assert!((out[0] - 0.5).abs() < 1e-5, "out[0]={}, expected 0.5", out[0]);
        assert!((out[1] - 0.5).abs() < 1e-5, "out[1]={}, expected 0.5", out[1]);
        assert!(out[2].abs() < 1e-5, "out[2]={}, expected 0", out[2]);
        assert!(out[3].abs() < 1e-5, "out[3]={}, expected 0", out[3]);
    }

    /// Cross-validation vs the REAL Python `tinyvc.match_features(metrics='cos')`
    /// on randomly-generated input. **Ignored by default** — requires a Python
    /// fixture dump at `/tmp/knn_verify/{source,target,output_py}.f32`.
    ///
    /// To regenerate the fixture:
    /// ```bash
    /// cd /home/z/my-project/prototype
    /// /home/z/.venv/bin/python -c "
    /// import sys; sys.path.insert(0,'src'); sys.path.insert(0,'/home/z/my-project/repos/tinyvc')
    /// import torch, numpy as np
    /// from module.tinyvc import match_features
    /// np.random.seed(0x1234_5678)
    /// dim, t_src, t_ref, top_k = 768, 5, 12, 4
    /// source_np = np.random.randn(t_src, dim).astype(np.float32)
    /// target_np = np.random.randn(t_ref, dim).astype(np.float32)
    /// src_t = torch.from_numpy(source_np).T.unsqueeze(0).contiguous()
    /// tgt_t = torch.from_numpy(target_np).T.unsqueeze(0).contiguous()
    /// out_t = match_features(src_t, tgt_t, k=top_k, alpha=0.0, metrics='cos')
    /// out_fm = out_t.squeeze(0).T.contiguous().numpy()
    /// import os; os.makedirs('/tmp/knn_verify', exist_ok=True)
    /// source_np.tofile('/tmp/knn_verify/source.f32')
    /// target_np.tofile('/tmp/knn_verify/target.f32')
    /// out_fm.tofile('/tmp/knn_verify/output_py.f32')
    /// print('dumped')
    /// "
    /// ```
    ///
    /// Then run with:
    /// ```bash
    /// cargo test --release -p vc-native knn_match_features_real -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn knn_match_features_real() {
        use std::fs;
        let dir = "/tmp/knn_verify";
        let src_bytes = match fs::read(format!("{dir}/source.f32")) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipped: fixture {dir}/source.f32 missing ({e}) — see docstring to regenerate"
                );
                return;
            }
        };
        let tgt_bytes = fs::read(format!("{dir}/target.f32")).expect("target.f32");
        let out_bytes = fs::read(format!("{dir}/output_py.f32")).expect("output_py.f32");

        // Reinterpret bytes as f32 (little-endian on x86/ARM).
        let to_f32 = |b: Vec<u8>| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let source = to_f32(src_bytes);
        let target = to_f32(tgt_bytes);
        let py_out = to_f32(out_bytes);

        let dim = 768;
        let t_src = source.len() / dim;
        let t_ref = target.len() / dim;
        let top_k = 4;

        assert_eq!(py_out.len(), dim * t_src);
        assert_eq!(source.len(), dim * t_src);
        assert_eq!(target.len(), dim * t_ref);

        let rust_out = knn_retrieve(&source, &target, dim, t_src, t_ref, top_k);

        // Tolerance: torch.bmm + torch.topk + torch.mean use BLAS/SIMD with
        // different reduction order than our Rust f32x4 dot product, plus
        // numpy randn is the same on both sides (so inputs are bit-identical,
        // eliminating input-generation noise). For ~unit-norm 768-d vectors
        // averaged in groups of 4, max abs diff should be well under 1e-4.
        let mut max_err = 0.0f32;
        let mut max_err_i = 0;
        for i in 0..py_out.len() {
            let err = (rust_out[i] - py_out[i]).abs();
            if err > max_err {
                max_err = err;
                max_err_i = i;
            }
        }
        eprintln!(
            "knn_match_features_real: dim={dim}, t_src={t_src}, t_ref={t_ref}, top_k={top_k}, max_err={max_err:.3e} at i={max_err_i} (rust={}, py={})",
            rust_out.get(max_err_i).copied().unwrap_or(f32::NAN),
            py_out.get(max_err_i).copied().unwrap_or(f32::NAN),
        );
        assert!(
            max_err < 1e-4,
            "Rust vs Python match_features max_err={max_err:.3e} at i={max_err_i} exceeds 1e-4"
        );
    }
}
