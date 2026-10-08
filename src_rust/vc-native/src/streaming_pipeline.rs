//! StreamingPipeline — pipeline parallelism for real-time voice conversion.
//!
//! Wires together two SPSC ring buffers (`rtrb`) + a decoder thread so that
//! audio I/O (mic capture, speaker playback) overlaps with the heavy
//! encoder/decoder compute. This is the v2.0 replacement for v1.0's
//! `modules/streaming.py:input_buf` (numpy roll-based) + Python-side
//! `process_chunk()` call-per-block design, which suffered 5–10ms of GIL
//! contention + NumPy dispatch overhead per chunk.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────────┐   input ring   ┌──────────────┐   output ring  ┌──────────────┐
//!   │ audio thread │──(SPSC, f32)──▶│ decoder thd  │──(SPSC, f32)──▶│ audio thread │
//!   │ miniaudio cb │   push_input   │ process_fn() │  pop_output    │ miniaudio cb │
//!   └──────────────┘                └──────────────┘                └──────────────┘
//! ```
//!
//! - **input ring** (capacity `block_size * 4`): the audio thread (or, in
//!   tests, `push_input`) pushes mono f32 mic samples in; the decoder thread
//!   pulls them out block-by-block.
//! - **output ring** (capacity `block_size * 4`): the decoder thread pushes
//!   the processed f32 speaker samples; the audio thread (or `pop_output`)
//!   pulls them out.
//!
//! ## Threading
//!
//! The decoder thread owns the input consumer + output producer halves of the
//! two rings (moved into the thread on `start_decoder`). The pipeline struct
//! retains the other two halves (`input_producer` + `output_consumer`) so
//! `push_input` / `pop_output` can be called from any thread with `&mut` to
//! the pipeline. The `running` flag is an `Arc<AtomicBool>` shared with the
//! decoder thread for cooperative shutdown — `stop()` clears it and joins.
//!
//! ## GIL-free
//!
//! The hot path (decoder thread + future miniaudio callback) never acquires
//! the Python GIL. Python only calls `start()` / `stop()` / `push_input()` /
//! `pop_output()` infrequently (once per chunk or once per session). This is
//! the v2.0 design rationale for moving audio I/O out of PyAudio.
//!
//! ## miniaudio callback wiring (OPT-10)
//!
//! `start_with_audio` opens a duplex `AudioDevice` with a GIL-free
//! `FnMut(&[f32], &mut [f32]) + Send + 'static` closure that runs on
//! miniaudio's audio thread. The closure captures the input ring's
//! `Producer` (mic capture push) + the output ring's `Consumer` (speaker
//! playback pull) — both `rtrb` halves are `Send`, so the closure is too.
//! No raw `*mut c_void` user-data pointer is exposed to the caller; the
//! `AudioDevice` owns the boxed closure internally and frees it in `Drop`
//! after the audio device has been stopped (so the callback can no longer
//! fire while we're tearing down shared state). The current `new` +
//! `start_decoder` path (manual push/pop) is still supported for tests +
//! audio-hardware-less environments.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use rtrb::{Consumer, Producer, RingBuffer};

// AudioDevice (miniaudio FFI wrapper) + AudioConfig — used by
// `start_with_audio` to wire the GIL-free audio callback to the rings.
use crate::miniaudio_ffi::AudioDevice;
use crate::AudioConfig;

/// Full streaming pipeline: SPSC ring (input) → decoder thread → SPSC ring (output).
///
/// See the module-level doc for the architecture diagram + design rationale.
pub struct StreamingPipeline {
    /// Samples per decoder block (the decoder pulls exactly this many samples
    /// before invoking `process_fn`).
    block_size: usize,
    /// Push side of input ring — caller (audio thread / test) pushes mic samples in.
    /// `None` after `start_with_audio` (the audio callback owns the producer
    /// then; calling `push_input` silently drops to preserve the SPSC
    /// contract — only one producer per ring).
    input_producer: Option<Producer<f32>>,
    /// Pull side of output ring — caller (audio thread / test) pulls speaker samples out.
    /// `None` after `start_with_audio` (the audio callback owns the consumer
    /// then; calling `pop_output` returns `Empty` to preserve the SPSC
    /// contract — only one consumer per ring).
    output_consumer: Option<Consumer<f32>>,
    /// Decoder-thread side of input ring — moved into the decoder thread on
    /// `start_decoder`. `None` after `start_decoder` has been called (until
    /// `stop` resets it — see `stop`'s implementation for the no-restart
    /// policy).
    decoder_input_consumer: Option<Consumer<f32>>,
    /// Decoder-thread side of output ring — moved into the decoder thread on
    /// `start_decoder`.
    decoder_output_producer: Option<Producer<f32>>,
    /// Cooperative shutdown flag shared with the decoder thread.
    running: Arc<AtomicBool>,
    /// Decoder thread handle. `None` if not started or already joined.
    decoder_thread: Option<JoinHandle<()>>,
    /// miniaudio duplex device — set by `start_with_audio`, dropped in `stop`
    /// (which calls `shim_stop` + `shim_close`). `None` on the manual-push path.
    audio_device: Option<AudioDevice>,
}

// `StreamingPipeline` is `Send` automatically: every field is `Send`.
// - `Option<Producer/Consumer<f32>>`: rtrb halves are `Send` (single-producer /
//   single-consumer contract — each end owned by exactly one thread).
// - `Arc<AtomicBool>`, `Option<JoinHandle<()>>`: `Send + Sync`.
// - `Option<AudioDevice>`: `AudioDevice: Send` (see `miniaudio_ffi.rs`).
//
// No `unsafe impl Send` is required — moving the pipeline across threads is
// sound because the audio device owns its boxed callback (and frees it on
// `Drop` only after the audio thread is torn down), and the rtrb halves are
// moved — not aliased — between the audio callback and the decoder thread.

impl StreamingPipeline {
    /// Allocate two SPSC rings (each `block_size * 4` capacity) and the
    /// shared shutdown flag. Does NOT spawn the decoder thread — call
    /// `start_decoder` to do that.
    ///
    /// `block_size == 0` is rejected to avoid a divide-by-zero / empty-block
    /// infinite loop in the decoder.
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        let ring_cap = block_size.checked_mul(4).expect("block_size * 4 overflow");
        let (input_prod, input_cons) = RingBuffer::<f32>::new(ring_cap);
        let (output_prod, output_cons) = RingBuffer::<f32>::new(ring_cap);
        Self {
            block_size,
            input_producer: Some(input_prod),
            output_consumer: Some(output_cons),
            decoder_input_consumer: Some(input_cons),
            decoder_output_producer: Some(output_prod),
            running: Arc::new(AtomicBool::new(false)),
            decoder_thread: None,
            audio_device: None,
        }
    }

    /// Push a single mic sample into the input ring for the decoder to consume.
    ///
    /// Returns `Err(PushError::Full(sample))` if the input ring is full
    /// (decoder hasn't kept up). The caller may drop the sample, backpressure
    /// the producer, or retry — typical miniaudio callbacks just drop it.
    pub fn push_input(&mut self, sample: f32) -> Result<(), rtrb::PushError<f32>> {
        match self.input_producer.as_mut() {
            Some(p) => p.push(sample),
            // Audio callback owns the producer (`start_with_audio` path) —
            // silently drop to preserve the SPSC contract (single producer).
            None => Ok(()),
        }
    }

    /// Pull a single processed sample from the output ring (speaker side).
    ///
    /// Returns `Ok(sample)` if available, `Err(PopError::Empty)` if the
    /// decoder hasn't pushed anything yet (caller should emit silence).
    pub fn pop_output(&mut self) -> Result<f32, rtrb::PopError> {
        match self.output_consumer.as_mut() {
            Some(c) => c.pop(),
            // Audio callback owns the consumer (`start_with_audio` path) —
            // return `Empty` so the caller emits silence. Preserves the SPSC
            // contract (single consumer).
            None => Err(rtrb::PopError::Empty),
        }
    }

    /// Spawn the decoder thread, which loops: pull `block_size` samples from
    /// the input ring, call `process_fn(&[f32]) -> Vec<f32>`, push the
    /// returned samples to the output ring. The thread runs cooperatively —
    /// it yields the CPU when either ring is empty/full, and exits cleanly
    /// when `stop()` (or `Drop`) clears the `running` flag.
    ///
    /// If called twice without an intervening `stop`, the second call is a
    /// silent no-op (the ring halves have already been moved into the first
    /// thread). This is a defensive choice — the alternative (panic) would
    /// abort the process under `panic = "abort"`.
    ///
    /// `process_fn` is `FnMut` (not `Fn`) so callers can mutate state across
    /// chunks (e.g. a streaming SOLA tail buffer or a DDSP phase accumulator).
    pub fn start_decoder<F>(&mut self, process_fn: F)
    where
        F: FnMut(&[f32]) -> Vec<f32> + Send + 'static,
    {
        // Move the decoder-side ring halves out of the pipeline. If either
        // is None, the decoder was already started — bail (no-op).
        let input_consumer = match self.decoder_input_consumer.take() {
            Some(c) => c,
            None => return,
        };
        let output_producer = match self.decoder_output_producer.take() {
            Some(p) => p,
            None => {
                // Restore the consumer we just took so a future restart could
                // work (current contract says start is one-shot, but be tidy).
                self.decoder_input_consumer = Some(input_consumer);
                return;
            }
        };

        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();
        let block_size = self.block_size;
        let mut input_consumer = input_consumer;
        let mut output_producer = output_producer;
        let mut process_fn = process_fn;

        let handle = thread::spawn(move || {
            let mut input_buf = vec![0.0f32; block_size];
            'main: loop {
                if !running.load(Ordering::SeqCst) {
                    break 'main;
                }
                // Pull exactly `block_size` samples from the input ring.
                // On empty, yield + re-check `running` (cooperative shutdown).
                for slot in input_buf.iter_mut() {
                    loop {
                        match input_consumer.pop() {
                            Ok(s) => {
                                *slot = s;
                                break;
                            }
                            Err(rtrb::PopError::Empty) => {
                                if !running.load(Ordering::SeqCst) {
                                    // Shutdown before a full block arrived —
                                    // exit cleanly without calling process_fn
                                    // on a partial buffer.
                                    break 'main;
                                }
                                thread::yield_now();
                            }
                        }
                    }
                }
                // Compute the output block (CPU-heavy: STFT + ONNX + DDSP).
                let output_buf = process_fn(&input_buf);
                // Push all output samples to the output ring. On full, yield
                // + re-check `running` (the speaker side may be slow).
                for &s in &output_buf {
                    while output_producer.push(s).is_err() {
                        if !running.load(Ordering::SeqCst) {
                            break 'main;
                        }
                        thread::yield_now();
                    }
                }
            }
        });

        self.decoder_thread = Some(handle);
    }

    /// Signal the decoder thread to shut down (clears `running`) and block
    /// until it has joined. Safe to call when not started (no-op). Safe to
    /// call twice — the second call sees `decoder_thread == None`.
    ///
    /// After `stop`, the ring halves moved into the decoder thread are gone
    /// (consumed by the join); calling `start_decoder` again would silently
    /// no-op (the ring halves are None). A fresh pipeline requires
    /// `StreamingPipeline::new`.
    pub fn stop(&mut self) {
        // Signal the decoder thread to exit its main loop.
        self.running.store(false, Ordering::SeqCst);
        // Stop the audio device BEFORE joining the decoder thread, so the
        // audio callback stops touching the rings. `take()` + drop calls
        // `shim_stop` + `shim_close` (in `AudioDevice::drop`), which also
        // frees the boxed closure handed to the C shim in `open`.
        if let Some(mut dev) = self.audio_device.take() {
            let _ = dev.stop();
            // `dev` dropped here → `shim_close` called → audio thread joined.
        }
        // Join the decoder thread. Should exit promptly (it polls
        // `running` every yield cycle).
        if let Some(handle) = self.decoder_thread.take() {
            let _ = handle.join();
        }
    }

    /// True iff the decoder thread is currently running (started + not yet
    /// stopped). Mainly for diagnostics / Python-side getters.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// The configured block size (samples per decoder invocation).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Open audio device + spawn decoder thread in one call.
    ///
    /// This is the full audio path: the audio callback (running on
    /// miniaudio's thread) is the producer of the input ring + consumer of
    /// the output ring; the decoder thread is the consumer of input +
    /// producer of output. `process_fn` runs on the decoder thread, never on
    /// the audio thread (so it can take 5–20ms without glitching the audio
    /// callback, as long as the rings have ~4 blocks of capacity for slack).
    ///
    /// The audio callback is a safe `FnMut(&[f32], &mut [f32]) + Send + 'static`
    /// closure that captures the input ring's `Producer` + output ring's
    /// `Consumer`. `AudioDevice::open` boxes the closure internally and
    /// frees it in `Drop` after the device has been stopped — no raw pointer
    /// is exposed to the caller.
    ///
    /// On error (e.g., headless env with no audio backend), returns
    /// `Err(String)` naming the failing FFI call. Safe to call in tests on
    /// machines without audio hardware — the test should accept either Ok
    /// (audio available) or Err (headless).
    ///
    /// After `start_with_audio`, `push_input`/`pop_output` become silent
    /// no-ops (the producer/consumer moved to the audio thread); the user
    /// drives audio I/O purely through `process_fn` (decoder thread) + the
    /// audio device itself.
    pub fn start_with_audio<F>(config: AudioConfig, process_fn: F) -> Result<Self, String>
    where
        F: FnMut(&[f32]) -> Vec<f32> + Send + 'static,
    {
        let block_size = config.block_size as usize;
        if block_size == 0 {
            return Err("AudioConfig.block_size must be > 0".into());
        }
        let ring_cap = block_size
            .checked_mul(4)
            .ok_or_else(|| "block_size * 4 overflow".to_string())?;

        // Split ring ownership: the audio callback owns input_prod + output_cons,
        // the decoder thread owns input_cons + output_prod. This preserves rtrb's
        // SPSC contract — exactly one producer + one consumer per ring.
        let (input_prod, input_cons) = RingBuffer::<f32>::new(ring_cap);
        let (output_prod, output_cons) = RingBuffer::<f32>::new(ring_cap);

        // Build the audio callback as a SAFE closure that captures the ring
        // halves. `Producer<f32>` and `Consumer<f32>` are `Send`, so the
        // closure is `Send` — satisfying `AudioDevice::open`'s bound. No raw
        // pointer / `unsafe impl Send` needed.
        let mut input_producer = input_prod;
        let mut output_consumer = output_cons;
        let audio_callback = move |input: &[f32], output: &mut [f32]| {
            // Push interleaved input samples (mic capture) to the input ring
            // for the decoder thread to consume. Non-blocking: drop if full
            // (typical real-time policy — better to drop input than stall
            // the audio thread).
            for &s in input {
                let _ = input_producer.push(s);
            }
            // Pull interleaved output samples (speaker playback) from the
            // output ring that the decoder thread produced. Non-blocking:
            // emit silence if empty (startup underrun).
            for slot in output {
                *slot = output_consumer.pop().unwrap_or(0.0);
            }
        };

        // Open the audio device with the SAFE closure-based API — no `unsafe`
        // required from this call site. `AudioDevice::open` boxes the closure
        // internally and frees it in `Drop` after the audio device is stopped.
        let mut audio_device =
            AudioDevice::open(&config, audio_callback).map_err(|e| format!("AudioDevice::open failed: {}", e))?;

        // Start the audio device (begins firing the callback on miniaudio's
        // thread). On failure, `audio_device` is dropped (which calls
        // `shim_stop` + `shim_close` + frees the boxed closure).
        audio_device
            .start()
            .map_err(|e| format!("AudioDevice::start failed: {}", e))?;

        // Spawn the decoder thread with the decoder-side ring halves.
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();
        let mut input_consumer = input_cons;
        let mut output_producer = output_prod;
        let mut process_fn = process_fn;

        let decoder = thread::spawn(move || {
            let mut input_buf = vec![0.0f32; block_size];
            'main: loop {
                if !running_clone.load(Ordering::SeqCst) {
                    break 'main;
                }
                // Pull exactly `block_size` samples from the input ring.
                for slot in input_buf.iter_mut() {
                    loop {
                        match input_consumer.pop() {
                            Ok(s) => {
                                *slot = s;
                                break;
                            }
                            Err(rtrb::PopError::Empty) => {
                                if !running_clone.load(Ordering::SeqCst) {
                                    break 'main;
                                }
                                thread::yield_now();
                            }
                        }
                    }
                }
                // Compute the output block (CPU-heavy: STFT + ONNX + DDSP).
                let output_buf = process_fn(&input_buf);
                // Push all output samples to the output ring.
                for &s in &output_buf {
                    while output_producer.push(s).is_err() {
                        if !running_clone.load(Ordering::SeqCst) {
                            break 'main;
                        }
                        thread::yield_now();
                    }
                }
            }
        });

        Ok(Self {
            block_size,
            // Producer/consumer moved into the audio callback — calling
            // `push_input`/`pop_output` on this instance silently drops /
            // returns silence (audio thread is the SPSC counterpart now).
            input_producer: None,
            output_consumer: None,
            decoder_input_consumer: None,
            decoder_output_producer: None,
            running,
            decoder_thread: Some(decoder),
            audio_device: Some(audio_device),
        })
    }
}

impl Drop for StreamingPipeline {
    /// Defensive `Drop`: if the user forgot to call `stop`, signal shutdown
    /// + stop the audio device + join the decoder thread. Safe to call
    /// after an explicit `stop` (it's a no-op then — all `Option<...>`
    /// fields are `None`).
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================
// Multi-channel (de)interleave helpers (OPT-16)
// ============================================================

/// Deinterleave multi-channel audio.
///
/// Input: interleaved samples `[L, R, L, R, ...]` (one f32 per channel per
/// frame, total `n_frames * channels` samples).
/// Output: planar samples `[[L, L, ...], [R, R, ...]]` (one `Vec<f32>` per
/// channel, each of length `n_frames`).
///
/// Used by the decoder thread when the audio device is configured for >1
/// channel (e.g. stereo) but the model / processing expects per-channel
/// (planar) data. Round-trips with [`interleave`].
///
/// # Panics
///
/// Panics if `channels == 0`.
#[allow(dead_code)] // public helpers for future decoder-thread planar↔interleaved conversion
pub fn deinterleave(input: &[f32], channels: usize) -> Vec<Vec<f32>> {
    assert!(channels > 0, "channels must be > 0");
    let n_frames = input.len() / channels;
    let mut output = vec![vec![0.0f32; n_frames]; channels];
    for frame in 0..n_frames {
        for ch in 0..channels {
            output[ch][frame] = input[frame * channels + ch];
        }
    }
    output
}

/// Interleave multi-channel audio.
///
/// Input: planar samples `[[L, L, ...], [R, R, ...]]` (one slice per channel).
/// Output: interleaved samples `[L, R, L, R, ...]` (length `n_frames * channels`).
///
/// Used by the decoder thread to convert per-channel (planar) model output
/// back into the interleaved format the miniaudio playback buffer expects.
/// Round-trips with [`deinterleave`].
///
/// # Panics
///
/// Panics if `channels == 0` or `input` is empty.
#[allow(dead_code)] // public helpers for future decoder-thread planar↔interleaved conversion
pub fn interleave(input: &[Vec<f32>], channels: usize) -> Vec<f32> {
    assert!(channels > 0, "channels must be > 0");
    assert!(!input.is_empty(), "input must have at least one channel");
    let n_frames = input[0].len();
    let mut output = vec![0.0f32; n_frames * channels];
    for frame in 0..n_frames {
        for ch in 0..channels.min(input.len()) {
            output[frame * channels + ch] = input[ch][frame];
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
    use std::time::Duration;

    /// End-to-end passthrough: push 100 samples through a `|x| x.to_vec()`
    /// decoder, then drain the output ring and verify the values match.
    /// Exercises the full push → SPSC → decoder → SPSC → pop pipeline.
    #[test]
    fn test_streaming_pipeline_passthrough() {
        let block = 100usize;
        let mut pipeline = StreamingPipeline::new(block);
        pipeline.start_decoder(|input: &[f32]| input.to_vec());

        // Push 100 samples (exactly one decoder block).
        for i in 0..block {
            pipeline
                .push_input(i as f32)
                .expect("input ring should not be full (cap = 4*block)");
        }

        // Give the decoder thread a moment to process the block + push outputs.
        std::thread::sleep(Duration::from_millis(100));

        // Drain the output ring.
        let mut outputs = vec![];
        while let Ok(s) = pipeline.pop_output() {
            outputs.push(s);
        }

        pipeline.stop();
        assert!(
            !outputs.is_empty(),
            "no output received — decoder thread didn't run?"
        );
        // Passthrough: every output sample should equal its index in the block.
        for (i, &out) in outputs.iter().take(block).enumerate() {
            assert!(
                (out - i as f32).abs() < 1e-5,
                "output[{}]={}, expected {} (passthrough)",
                i,
                out,
                i
            );
        }
    }

    /// Silence / idle: start the decoder but never push anything. The
    /// decoder should block on `input_consumer.pop()` (yielding), and
    /// `stop()` should wake it + join cleanly within microseconds. The only
    /// assertion is "no hang, no crash" — there's no output to check.
    #[test]
    fn test_streaming_pipeline_silence_idle() {
        let mut pipeline = StreamingPipeline::new(100);
        pipeline.start_decoder(|_input: &[f32]| vec![]);
        // Idle — decoder should be blocked on the input ring.
        std::thread::sleep(Duration::from_millis(50));
        // stop() must clear `running` + join promptly. If it doesn't, the
        // test will time out (default 60s in cargo test).
        pipeline.stop();
        assert!(!pipeline.is_running());
        // No output expected.
        let mut n = 0;
        while pipeline.pop_output().is_ok() {
            n += 1;
        }
        assert_eq!(n, 0, "should have zero output after silence + stop");
    }

    /// Double-start is a silent no-op (not a panic). Verify the second call
    /// doesn't spawn a second thread or otherwise misbehave.
    #[test]
    fn test_streaming_pipeline_double_start_no_op() {
        let mut pipeline = StreamingPipeline::new(50);
        pipeline.start_decoder(|input: &[f32]| input.to_vec());
        // Second start should be a no-op (decoder_input_consumer is None now).
        pipeline.start_decoder(|input: &[f32]| input.to_vec());
        // Push one block, drain, verify it round-trips.
        for i in 0..50 {
            pipeline.push_input(i as f32).unwrap();
        }
        std::thread::sleep(Duration::from_millis(50));
        let mut out = vec![];
        while let Ok(s) = pipeline.pop_output() {
            out.push(s);
        }
        pipeline.stop();
        assert_eq!(out.len(), 50, "expected exactly 50 outputs, got {}", out.len());
    }

    /// Stop without ever starting the decoder must be a no-op (don't panic
    /// on `decoder_thread.take()` returning None).
    #[test]
    fn test_streaming_pipeline_stop_without_start() {
        let mut pipeline = StreamingPipeline::new(10);
        pipeline.stop();
        assert!(!pipeline.is_running());
    }

    /// Verify `running` is true after `start_decoder` and false after `stop`.
    #[test]
    fn test_streaming_pipeline_running_flag_lifecycle() {
        let mut pipeline = StreamingPipeline::new(10);
        assert!(!pipeline.is_running());
        pipeline.start_decoder(|input: &[f32]| input.to_vec());
        assert!(pipeline.is_running());
        pipeline.stop();
        assert!(!pipeline.is_running());
    }

    /// `block_size` getter returns the configured value.
    #[test]
    fn test_streaming_pipeline_block_size_getter() {
        let p = StreamingPipeline::new(1920);
        assert_eq!(p.block_size(), 1920);
    }

    /// `start_with_audio` end-to-end: in a headless env (no audio backend),
    /// `AudioDevice::open` should fail gracefully with `Err(String)` — not
    /// panic, not abort, not leak the `Box`-allocated callback state. On a
    /// machine WITH audio hardware, the device may open + start; both
    /// outcomes are acceptable (the test just verifies no panic + clean
    /// shutdown).
    #[test]
    fn test_start_with_audio_headless() {
        let config = AudioConfig::default();
        let result = StreamingPipeline::start_with_audio(config, |input: &[f32]| input.to_vec());
        match result {
            Ok(mut p) => {
                // Let the audio callback + decoder thread run briefly.
                std::thread::sleep(Duration::from_millis(50));
                // `stop` must stop the audio device, join the decoder, +
                // free the callback state without panicking.
                p.stop();
                println!("Audio pipeline started + stopped OK (hardware available)");
            }
            Err(e) => {
                // Expected in headless env. The error must mention the FFI
                // call so future debuggers can look up the miniaudio code.
                println!(
                    "Audio open failed (expected in headless env): {}",
                    e
                );
                assert!(
                    e.contains("shim_open_duplex failed") || e.contains("shim_start failed"),
                    "error should name the failing FFI call: got {e:?}"
                );
            }
        }
    }

    /// `deinterleave` with 3 frames × 2 channels: input is interleaved
    /// `[L, R, L, R, L, R]`, output must be `[[L, L, L], [R, R, R]]`.
    #[test]
    fn test_deinterleave_stereo() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 frames × 2 channels
        let output = deinterleave(&input, 2);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], vec![1.0, 3.0, 5.0]); // Left channel
        assert_eq!(output[1], vec![2.0, 4.0, 6.0]); // Right channel
    }

    /// `interleave` with 2 channels: input is planar
    /// `[[L, L, L], [R, R, R]]`, output must be `[L, R, L, R, L, R]`.
    #[test]
    fn test_interleave_stereo() {
        let input = vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]]; // 2 channels
        let output = interleave(&input, 2);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// `deinterleave` ↔ `interleave` round-trip: any interleaved input must
    /// survive deinterleave → interleave unchanged.
    #[test]
    fn test_deinterleave_interleave_roundtrip() {
        let original = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // 4 frames × 2 ch
        let planar = deinterleave(&original, 2);
        let back = interleave(&planar, 2);
        assert_eq!(original, back);
    }

    /// `deinterleave` with 1 channel must be a no-op copy (mono case).
    #[test]
    fn test_deinterleave_mono() {
        let input = vec![1.0f32, 2.0, 3.0];
        let output = deinterleave(&input, 1);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], input);
    }
}
