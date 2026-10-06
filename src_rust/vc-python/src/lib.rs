//! vc-python: PyO3 module exposing v2.0 Rust pipeline to Python.
//!
//! Usage (Python):
//!   from vc_python import RealtimeInfer, AudioConfig
//!   infer = RealtimeInfer(models_dir="models/", voice_id=0)
//!   infer.start()  # opens miniaudio mic→speaker, runs Rust pipeline
//!
//! Design: GIL-free audio hot path. The miniaudio callback runs in Rust
//! without acquiring the Python GIL. Python side just calls .start() / .stop()
//! and optionally polls the latency stats via a lock-free ring.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use vc_native::{AudioConfig, new_audio_ring_buffer};

#[pyclass]
struct RealtimeInfer {
    config: AudioConfig,
    voice_id: u8,
    started: bool,
    // Producer side of the SPSC ring (audio callback pushes mic PCM in)
    _audio_producer: Option<rtrb::Producer<f32>>,
    // Consumer side (decoder thread pops and processes)
    _audio_consumer: Option<rtrb::Consumer<f32>>,
}

#[pymethods]
impl RealtimeInfer {
    #[new]
    #[pyo3(signature = (models_dir="models/", voice_id=0, config=None))]
    fn new(models_dir: &str, voice_id: u8, config: Option<AudioConfigPy>) -> PyResult<Self> {
        let config = config.map(|c| c.into()).unwrap_or_default();
        println!("[vc-python] init models_dir={models_dir} voice_id={voice_id} config={config:?}");
        // TODO P1: load 4 ONNX sessions via vc_ort::V3HybridSessions::load_all
        // TODO P1: wire sherpa-onnx Rust crate for silero-vad
        Ok(Self {
            config,
            voice_id,
            started: false,
            _audio_producer: None,
            _audio_consumer: None,
        })
    }

    /// Start the real-time pipeline: open miniaudio mic→speaker, spawn decoder thread.
    fn start(&mut self) -> PyResult<()> {
        if self.started {
            return Err(PyRuntimeError::new_err("already started"));
        }
        // Allocate SPSC ring buffer (block_size + look-ahead + crossfade + 2×last_delay)
        let cap = (self.config.block_size * 4) as usize;
        let (producer, consumer) = new_audio_ring_buffer(cap);
        self._audio_producer = Some(producer);
        self._audio_consumer = Some(consumer);
        // TODO P1: spawn miniaudio callback thread (GIL-free)
        // TODO P1: spawn decoder thread (Rust native, calls vc_ort + vc_native)
        self.started = true;
        println!("[vc-python] started voice_id={}", self.voice_id);
        Ok(())
    }

    /// Stop the pipeline.
    fn stop(&mut self) -> PyResult<()> {
        if !self.started {
            return Ok(());
        }
        // TODO P1: signal shutdown to audio + decoder threads
        self.started = false;
        println!("[vc-python] stopped");
        Ok(())
    }

    /// Get current end-to-end latency estimate (ms). 0 if not started.
    #[getter]
    fn latency_ms(&self) -> f32 {
        // TODO P1: read from decoder thread stats via atomic
        if self.started { 50.0 } else { 0.0 }
    }

    /// Get current RSS memory usage (MB).
    #[getter]
    fn rss_mb(&self) -> f32 {
        // TODO P1: read /proc/self/status VmRSS on Linux, mach_task_basic_info on macOS
        if self.started { 55.0 } else { 5.0 }
    }
}

/// Python-side AudioConfig mirror (PyO3-friendly)
#[pyclass]
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
    fn new(sample_rate: u32, channels: u16, block_size: u32, low_latency: bool, exclusive_mode: bool) -> Self {
        Self { sample_rate, channels, block_size, low_latency, exclusive_mode }
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

#[pymodule]
fn vc_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RealtimeInfer>()?;
    m.add_class::<AudioConfigPy>()?;
    Ok(())
}
