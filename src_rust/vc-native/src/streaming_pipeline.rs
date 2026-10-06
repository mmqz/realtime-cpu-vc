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
//! ## What's NOT wired up yet
//!
//! The miniaudio callback wiring (`AudioDevice::open` + a raw-pointer
//! `user_data` trampoline that pushes/pops from the rings on miniaudio's
//! real-time thread) is deferred to a follow-up task. It needs careful
//! `unsafe` design: the callback is `extern "C"`, so it cannot capture Rust
//! closures directly — it must read a raw `*mut c_void` user_data pointer
//! (typically `Box::into_raw(Box::new(PipelineState { ... }))`) and unsafely
//! dereference it. The current implementation is fully testable without
//! audio hardware by manually pushing + popping samples.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use rtrb::{Consumer, Producer, RingBuffer};

/// Full streaming pipeline: SPSC ring (input) → decoder thread → SPSC ring (output).
///
/// See the module-level doc for the architecture diagram + design rationale.
pub struct StreamingPipeline {
    /// Samples per decoder block (the decoder pulls exactly this many samples
    /// before invoking `process_fn`).
    block_size: usize,
    /// Push side of input ring — caller (audio thread / test) pushes mic samples in.
    input_producer: Producer<f32>,
    /// Pull side of output ring — caller (audio thread / test) pulls speaker samples out.
    output_consumer: Consumer<f32>,
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
}

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
            input_producer: input_prod,
            output_consumer: output_cons,
            decoder_input_consumer: Some(input_cons),
            decoder_output_producer: Some(output_prod),
            running: Arc::new(AtomicBool::new(false)),
            decoder_thread: None,
        }
    }

    /// Push a single mic sample into the input ring for the decoder to consume.
    ///
    /// Returns `Err(PushError::Full(sample))` if the input ring is full
    /// (decoder hasn't kept up). The caller may drop the sample, backpressure
    /// the producer, or retry — typical miniaudio callbacks just drop it.
    pub fn push_input(&mut self, sample: f32) -> Result<(), rtrb::PushError<f32>> {
        self.input_producer.push(sample)
    }

    /// Pull a single processed sample from the output ring (speaker side).
    ///
    /// Returns `Ok(sample)` if available, `Err(PopError::Empty)` if the
    /// decoder hasn't pushed anything yet (caller should emit silence).
    pub fn pop_output(&mut self) -> Result<f32, rtrb::PopError> {
        self.output_consumer.pop()
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
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.decoder_thread.take() {
            // Join: blocks until the decoder sees `running == false` and
            // exits its loop. Should be at most one yield_now() cycle (~µs).
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
}

impl Drop for StreamingPipeline {
    /// Defensive `Drop`: if the user forgot to call `stop`, signal shutdown
    /// + join the decoder thread so we don't leak a thread that touches freed
    /// memory. Safe to call after an explicit `stop` (it's a no-op then).
    fn drop(&mut self) {
        self.stop();
    }
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
}
