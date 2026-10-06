//! vc-python: PyO3 module exposing v2.0 Rust pipeline to Python.
//!
//! Usage (Python):
//!   from vc_python import RealtimeInfer, AudioConfig
//!   from vc_python import sola_find_best_offset_py, sola_crossfade_py, synth_harmonics_py
//!   infer = RealtimeInfer(models_dir="models/", voice_id=0)
//!   infer.start()  # opens miniaudio mic→speaker, runs Rust pipeline
//!
//! Build wheel: `cd src_rust/vc-python && maturin build --release`
//! Install:    `pip install ./target/wheels/vc_python-*.whl`
//!
//! Design: GIL-free audio hot path. The miniaudio callback runs in Rust
//! without acquiring the Python GIL. Python side just calls .start() / .stop()
//! and optionally polls the latency stats via a lock-free ring.
//!
//! NOTE: `RealtimeInfer::start()` and `RealtimeInfer::process_chunk()` are
//! stubs in this task — actual miniaudio device wiring and ONNX inference
//! orchestration land in P2.1. The KEY requirement of P2.0-3 is that the
//! wheel builds, installs, and Python can import all 5 symbols
//! (`RealtimeInfer`, `AudioConfig`, `sola_find_best_offset_py`,
//! `sola_crossfade_py`, `synth_harmonics_py`).

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use vc_native::{
    new_audio_ring_buffer, sola_crossfade, sola_find_best_offset, synth_harmonics, AudioConfig,
};
use vc_ort::{Backend, V3HybridSessions};

// ===========================================================================
// RealtimeInfer — top-level Python class for the v2.0 Rust pipeline
// ===========================================================================

/// Top-level Python class wrapping the v2.0 Rust real-time voice-conversion
/// pipeline. Construction eagerly loads all 5 v3-hybrid ONNX sessions via
/// `vc_ort::V3HybridSessions::load_all`. `start()` (stub) will eventually
/// spawn the miniaudio callback + decoder threads; `process_chunk()` (stub)
/// will eventually call `vc_ort` + `vc_native` to process audio.
#[pyclass]
struct RealtimeInfer {
    /// Audio device config (sample rate, channels, block size, latency flags).
    config: AudioConfig,
    /// Voice ID (0..N) — selects which reference embedding / fine-tune to use.
    voice_id: u8,
    /// Started flag — atomic so future miniaudio callback thread can read it.
    started: Arc<AtomicBool>,
    /// Latest end-to-end latency estimate (ms) — atomic for cross-thread reads.
    latency_ms_atomic: Arc<AtomicU32>,
    /// Loaded v3-hybrid ONNX sessions (None iff construction failed — but
    /// since `new` returns PyResult, this is always Some on success).
    sessions: Option<V3HybridSessions>,
    /// Producer side of SPSC ring (audio callback pushes mic PCM in).
    /// Allocated on `start()` — unused until miniaudio wiring lands.
    _audio_producer: Option<rtrb::Producer<f32>>,
    /// Consumer side (decoder thread pops and processes). Allocated on
    /// `start()` — unused until decoder thread wiring lands.
    _audio_consumer: Option<rtrb::Consumer<f32>>,
}

#[pymethods]
impl RealtimeInfer {
    /// Construct a new `RealtimeInfer`. Eagerly loads all 5 v3-hybrid ONNX
    /// sessions from `models_dir` via `vc_ort::V3HybridSessions::load_all`
    /// using the `Ort` backend (ONNXRuntime 1.27 C API).
    ///
    /// # Python signature
    /// ```python
    /// RealtimeInfer(models_dir="models/", voice_id=0, config=None)
    /// ```
    ///
    /// # Errors
    /// Raises `RuntimeError` if any of the 5 ONNX files is missing or fails
    /// to load (file names: `encoder.int8.onnx`,
    /// `openvoice_ref_encoder.int8.onnx`, `spark_speaker_encoder.int8.onnx`,
    /// `openvoice_residual_flow.int8.onnx`, `vocos.int8.onnx`).
    #[new]
    #[pyo3(signature = (models_dir="models/", voice_id=0, config=None))]
    fn new(
        models_dir: &str,
        voice_id: u8,
        config: Option<AudioConfigPy>,
    ) -> PyResult<Self> {
        let config = config.map(AudioConfig::from).unwrap_or_default();
        eprintln!(
            "[vc-python] init models_dir={models_dir} voice_id={voice_id} config={config:?}"
        );
        // Eagerly load all 5 v3-hybrid ONNX sessions via vc-ort.
        let sessions = V3HybridSessions::load_all(models_dir, Backend::Ort).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to load v3-hybrid ONNX sessions from `{models_dir}`: {e}"
            ))
        })?;
        eprintln!("[vc-python] loaded 5 v3-hybrid ONNX sessions (Ort backend)");
        Ok(Self {
            config,
            voice_id,
            started: Arc::new(AtomicBool::new(false)),
            latency_ms_atomic: Arc::new(AtomicU32::new(0)),
            sessions: Some(sessions),
            _audio_producer: None,
            _audio_consumer: None,
        })
    }

    /// Start the real-time pipeline: open miniaudio mic→speaker, spawn the
    /// decoder thread. Stub for P2.0-3 — just marks the pipeline as started
    /// and allocates the SPSC ring. The actual miniaudio device + decoder
    /// thread wiring lands in P2.1 (audio thread wiring).
    fn start(&self) -> PyResult<()> {
        if self.started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err("already started"));
        }
        // Allocate SPSC ring buffer (block_size + look-ahead + crossfade + 2×last_delay).
        let cap = (self.config.block_size as usize) * 4;
        let (producer, consumer) = new_audio_ring_buffer(cap);
        // SAFETY: RealtimeInfer is `&self` here (PyO3 signature) — we mutate
        // through the Option<_> slots, which works because interior mutability
        // via Option::take/replace doesn't require &mut self on the slot itself
        // when accessed through a Cell or AtomicUsize ... actually we need
        // &mut self here. Switch signature to `&mut self` would break the
        // template; instead we leak the ring halves into static-side state
        // (acceptable for stub purposes — actual implementation will move to
        // &mut self or use a Mutex<Option<...>>).
        //
        // For P2.0-3 we just mark started; the ring halves are dropped on the
        // floor (still validates the API contract).
        let _ = (producer, consumer); // drop — stub.
        self.started.store(true, Ordering::SeqCst);
        eprintln!("[vc-python] started voice_id={}", self.voice_id);
        Ok(())
    }

    /// Stop the pipeline. Stub — signals shutdown via the `started` atomic.
    /// Future implementation will join the decoder thread + close the
    /// miniaudio device.
    fn stop(&self) -> PyResult<()> {
        if !self.started.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.started.store(false, Ordering::SeqCst);
        eprintln!("[vc-python] stopped");
        Ok(())
    }

    /// Current end-to-end latency estimate (ms). Reads from an atomic that
    /// the decoder thread updates. Returns 0 if not started. Stub returns a
    /// fixed 50ms when started (no real decoder thread is running yet).
    #[getter]
    fn latency_ms(&self) -> f32 {
        let stored = self.latency_ms_atomic.load(Ordering::Relaxed);
        if stored == 0 && self.started.load(Ordering::SeqCst) {
            // Stub: report 50ms when started but no real measurement yet.
            50.0
        } else {
            stored as f32
        }
    }

    /// Current RSS memory usage (MB). Reads `/proc/self/status` on Linux
    /// (VmRSS line). Returns 0.0 on non-Linux platforms (macOS / Windows).
    #[getter]
    fn rss_mb(&self) -> f32 {
        // Read /proc/self/status VmRSS on Linux — this is the canonical way
        // to get the kernel's view of resident set size.
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u32>() {
                            return kb as f32 / 1024.0;
                        }
                    }
                }
            }
        }
        0.0
    }

    /// Test method: process an audio chunk via the Rust pipeline (no audio
    /// I/O). Stub for P2.0-3 — returns the input unchanged. Actual
    /// implementation will route through the v3-hybrid ONNX sessions +
    /// SOLA crossfade + DDSP harmonic synthesis via vc_ort + vc_native.
    fn process_chunk(&mut self, wav: Vec<f32>) -> PyResult<Vec<f32>> {
        if !self.started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "RealtimeInfer not started — call .start() first",
            ));
        }
        // Stub: passthrough. Real impl will:
        //  1. Resample mic PCM → 24 kHz mono.
        //  2. Run content encoder ONNX → content_feat + f0 + energy.
        //  3. Run speaker_encoder ONNX → spk_emb (if not cached).
        //  4. Run residual_flow ONNX → converted content.
        //  5. Run vocoder ONNX → 48 kHz waveform.
        //  6. SOLA-crossfade into the output ring.
        Ok(wav)
    }
}

// ===========================================================================
// AudioConfig — Python-visible mirror of vc_native::AudioConfig
// ===========================================================================

/// Audio device config — Python-visible mirror of `vc_native::AudioConfig`.
///
/// Fields:
///   - `sample_rate` (int, default 24000)
///   - `channels` (int, default 1)
///   - `block_size` (int, default 1920 — 80ms at 24 kHz)
///   - `low_latency` (bool, default True)
///   - `exclusive_mode` (bool, default False — WASAPI exclusive on Windows)
#[pyclass(name = "AudioConfig")]
#[derive(Clone, Debug)]
struct AudioConfigPy {
    sample_rate: u32,
    channels: u16,
    block_size: u32,
    low_latency: bool,
    exclusive_mode: bool,
}

#[pymethods]
impl AudioConfigPy {
    #[new]
    #[pyo3(signature = (sample_rate=24000, channels=1, block_size=1920, low_latency=true, exclusive_mode=false))]
    fn new(
        sample_rate: u32,
        channels: u16,
        block_size: u32,
        low_latency: bool,
        exclusive_mode: bool,
    ) -> Self {
        Self {
            sample_rate,
            channels,
            block_size,
            low_latency,
            exclusive_mode,
        }
    }

    /// Sample rate in Hz (default 24000).
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of channels (1 = mono).
    #[getter]
    fn channels(&self) -> u16 {
        self.channels
    }

    /// Samples per block (default 1920 = 80ms at 24 kHz).
    #[getter]
    fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Low-latency miniaudio profile flag.
    #[getter]
    fn low_latency(&self) -> bool {
        self.low_latency
    }

    /// WASAPI exclusive mode flag (Windows only).
    #[getter]
    fn exclusive_mode(&self) -> bool {
        self.exclusive_mode
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioConfig(sample_rate={}, channels={}, block_size={}, low_latency={}, exclusive_mode={})",
            self.sample_rate,
            self.channels,
            self.block_size,
            self.low_latency,
            self.exclusive_mode
        )
    }
}

impl From<AudioConfigPy> for AudioConfig {
    fn from(c: AudioConfigPy) -> Self {
        Self {
            sample_rate: c.sample_rate,
            channels: c.channels,
            block_size: c.block_size,
            low_latency: c.low_latency,
            exclusive_mode: c.exclusive_mode,
        }
    }
}

// ===========================================================================
// Top-level test functions — direct wrappers around vc_native hot paths
// ===========================================================================

/// Find the best SOLA alignment offset for crossfade.
///
/// # Python signature
/// ```python
/// sola_find_best_offset_py(new_chunk: list[float], tail: list[float], sola_search_size: int) -> int
/// ```
///
/// `new_chunk`: `[N]` float32 — the new audio block's first `crossfade_size` samples.
/// `tail`: `[N + sola_search_size]` float32 — the previous chunk's tail.
/// Returns: offset in `[0, sola_search_size)` maximizing cross-correlation.
#[pyfunction]
#[pyo3(signature = (new_chunk, tail, sola_search_size))]
fn sola_find_best_offset_py(
    new_chunk: Vec<f32>,
    tail: Vec<f32>,
    sola_search_size: usize,
) -> usize {
    sola_find_best_offset(&new_chunk, &tail, sola_search_size)
}

/// SOLA crossfade: apply sin² window crossfade at the best offset.
///
/// # Python signature
/// ```python
/// sola_crossfade_py(new_chunk: list[float], tail: list[float], sola_search_size: int, crossfade_size: int) -> list[float]
/// ```
#[pyfunction]
#[pyo3(signature = (new_chunk, tail, sola_search_size, crossfade_size))]
fn sola_crossfade_py(
    new_chunk: Vec<f32>,
    tail: Vec<f32>,
    sola_search_size: usize,
    crossfade_size: usize,
) -> Vec<f32> {
    sola_crossfade(&new_chunk, &tail, sola_search_size, crossfade_size)
}

/// Generate DDSP harmonic source: `sum_{k=1..n_harmonics} amp_k * sin(k * phase)`.
///
/// # Python signature
/// ```python
/// synth_harmonics_py(f0: list[float], amps: list[float], n_harmonics: int, sr: int) -> list[float]
/// ```
///
/// `f0`: `[T]` fundamental frequency, upsampled to sample rate.
/// `amps`: `[(n_harmonics + 1) * T]` per-harmonic amplitudes in row-major
///   layout. Index 0 is the DC component (skipped per DDSP convention).
/// `n_harmonics`: number of harmonics above DC.
/// `sr`: sample rate in Hz.
#[pyfunction]
#[pyo3(signature = (f0, amps, n_harmonics, sr))]
fn synth_harmonics_py(f0: Vec<f32>, amps: Vec<f32>, n_harmonics: usize, sr: u32) -> Vec<f32> {
    synth_harmonics(&f0, &amps, n_harmonics, sr)
}

/// NumPy-array variant of `synth_harmonics_py` — zero-copy input + output.
///
/// Accepts `np.ndarray[float32]` (1-D) for both `f0` and `amps` (no per-element
/// list-Vec conversion overhead — passes a slice view straight into the Rust
/// SIMD hot path). Returns a freshly-allocated `np.ndarray[float32]` whose
/// backing buffer is moved from the Rust `Vec<f32>` (no element-by-element
/// copy).
///
/// # Python signature
/// ```python
/// synth_harmonics_np(f0: np.ndarray, amps: np.ndarray, n_harmonics: int, sr: int) -> np.ndarray
/// ```
///
/// # Errors
/// Raises `RuntimeError` if either input array is not C-contiguous. Convert
/// non-contiguous arrays with `np.ascontiguousarray(...)` before calling.
#[pyfunction]
#[pyo3(signature = (f0, amps, n_harmonics, sr))]
fn synth_harmonics_np<'py>(
    py: Python<'py>,
    f0: PyReadonlyArray1<'_, f32>,
    amps: PyReadonlyArray1<'_, f32>,
    n_harmonics: usize,
    sr: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let f0_slice = f0
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("f0 must be a contiguous numpy array — use np.ascontiguousarray()"))?;
    let amps_slice = amps
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("amps must be a contiguous numpy array — use np.ascontiguousarray()"))?;
    let output = synth_harmonics(f0_slice, amps_slice, n_harmonics, sr);
    // `from_vec_bound` moves the Vec's allocation into a NumPy array (zero-copy).
    Ok(PyArray1::from_vec_bound(py, output))
}

// ===========================================================================
// Module registration
// ===========================================================================

/// Module init: registers `RealtimeInfer`, `AudioConfig`, and the three
/// top-level `*_py` functions with the Python interpreter.
#[pymodule]
fn vc_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RealtimeInfer>()?;
    m.add_class::<AudioConfigPy>()?;
    m.add_function(wrap_pyfunction!(sola_find_best_offset_py, m)?)?;
    m.add_function(wrap_pyfunction!(sola_crossfade_py, m)?)?;
    m.add_function(wrap_pyfunction!(synth_harmonics_py, m)?)?;
    m.add_function(wrap_pyfunction!(synth_harmonics_np, m)?)?;
    m.add("__doc__", "vc-python: v2.0 Rust pipeline PyO3 bindings (vc-ort + vc-native)")?;
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_config_py_defaults() {
        let c = AudioConfigPy::new(24000, 1, 1920, true, false);
        assert_eq!(c.sample_rate, 24000);
        assert_eq!(c.channels, 1);
        assert_eq!(c.block_size, 1920);
        assert!(c.low_latency);
        assert!(!c.exclusive_mode);
    }

    #[test]
    fn test_audio_config_py_to_native() {
        let c = AudioConfigPy::new(48000, 2, 960, false, true);
        let native: AudioConfig = c.into();
        assert_eq!(native.sample_rate, 48000);
        assert_eq!(native.channels, 2);
        assert_eq!(native.block_size, 960);
        assert!(!native.low_latency);
        assert!(native.exclusive_mode);
    }

    #[test]
    fn test_sola_find_best_offset_py_wraps_native() {
        // Mirror of vc-native test_sola_find_best_offset_nonzero.
        let chunk = vec![1.0, 2.0, 3.0, 4.0];
        let mut tail = vec![0.0; 10];
        tail[3..7].copy_from_slice(&chunk);
        let offset = sola_find_best_offset_py(chunk, tail, 6);
        assert_eq!(offset, 3);
    }

    #[test]
    fn test_synth_harmonics_py_wraps_native() {
        let sr = 24000u32;
        let n = 240;
        let f0 = vec![100.0; n];
        let mut amps = vec![0.0f32; 2 * n];
        for i in 0..n {
            amps[n + i] = 1.0;
        }
        let out = synth_harmonics_py(f0, amps, 1, sr);
        assert_eq!(out.len(), n);
        let max_val = out.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_val > 0.9, "max should be > 0.9, got {max_val}");
    }

    #[test]
    fn test_sola_crossfade_py_preserves_length() {
        let chunk = vec![1.0; 100];
        let tail = vec![0.0; 200];
        let out = sola_crossfade_py(chunk, tail, 50, 30);
        assert_eq!(out.len(), 100);
        for i in 30..100 {
            assert!((out[i] - 1.0).abs() < 1e-6);
        }
    }

    /// Type-inference sanity: `RealtimeInfer::new` with a missing models dir
    /// should yield a PyRuntimeError (we can't call it here because it needs
    /// a Python interpreter, but we can verify the conversion logic via
    /// `From<InferError>` for `PyResult` indirectly).
    ///
    /// NOTE: can't use `unwrap_err()` here because `V3HybridSessions` does
    /// not implement `Debug` (and `unwrap_err` requires `T: Debug`).
    #[test]
    fn test_v3_hybrid_load_all_missing_dir_errors() {
        // Direct vc-ort call (no PyO3) — verifies that the underlying
        // error path produces InferError::NotFound, which we convert to
        // PyRuntimeError in the Python wrapper.
        let result = V3HybridSessions::load_all("/nonexistent", Backend::Ort);
        let msg = match result {
            Ok(_) => panic!("expected error for missing models dir"),
            Err(err) => format!("{err}"),
        };
        assert!(
            msg.contains("not found") || msg.contains("No such file") || msg.contains("ONNX"),
            "unexpected error message: {msg}"
        );
    }
}
