//! vc-ort: ONNX inference orchestration in Rust for v2.0
//!
//! Primary backend: `ort` crate (Rust bindings to ONNXRuntime 1.28+)
//! Fallback backend: `tract-onnx` (pure-Rust ONNX interpreter by Sonos)
//!
//! This module replaces v1.0's `prototype/modules/encoder.py`, `decoder.py`,
//! `flow.py`, `vocoder_v2.py`, `speaker_encoder_v3.py` (which all use Python
//! onnxruntime) with Rust-native equivalents. Same ONNX models, same I/O shape.
//!
//! Streaming optimization: tract's `pulsification` converts the full-sequence
//! ONNX graph into a chunk-streaming variant (see tract-pulse), eliminating
//! the need for our custom SOLA overlap-add ring buffer in v1.0.

use ndarray::ArrayD;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferError {
    #[error("ORT inference failed: {0}")]
    Ort(String),
    #[error("tract inference failed: {0}")]
    Tract(String),
    #[error("Tensor shape mismatch: expected {expected:?}, got {got:?}")]
    Shape { expected: Vec<usize>, got: Vec<usize> },
    #[error("ONNX model not found at {0}")]
    NotFound(String),
}

pub type InferResult<T> = Result<T, InferError>;

/// Backend selection — runtime configurable.
#[derive(Clone, Copy, Debug)]
pub enum Backend {
    /// Rust → ONNXRuntime C API (default, fastest on x86 AVX2 MLAS + ARM XNNPACK)
    Ort,
    /// Pure-Rust ONNX (tract by Sonos, ~5-15MB static binary, no C++ dep)
    Tract,
}

pub struct InferenceSession {
    backend: Backend,
    // ort::Session or tract::Runnable — wrapped behind Box<dyn Any> for now
    _session_inner: (),  // placeholder, real impl in P1
}

impl InferenceSession {
    /// Load an ONNX model from disk. Chooses backend based on the `Backend` enum.
    pub fn load(model_path: &str, backend: Backend) -> InferResult<Self> {
        if !std::path::Path::new(model_path).exists() {
            return Err(InferError::NotFound(model_path.to_string()));
        }
        // TODO P1: actual ort::Session::builder().commit_from_file(model_path)?
        // TODO P1: actual tract_onnx::onnx().load_model_for_layering(model_path)?
        Ok(Self { backend, _session_inner: () })
    }

    /// Run inference. input: [B, C, T] float32. Returns: [B, C', T'] float32.
    pub fn run(&self, input: ArrayD<f32>) -> InferResult<ArrayD<f32>> {
        // TODO P1: dispatch by self.backend
        let _ = (input, self.backend);
        Ok(ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 1]), vec![0.0]).unwrap())
    }

    /// Pulsify: convert a full-sequence ONNX graph to a chunk-streaming variant
    /// (tract-only feature). Returns a new session that accepts chunked input
    /// and maintains internal state.
    pub fn pulsify(&self, chunk_size: usize) -> InferResult<Self> {
        // TODO P2: tract-pulse: graph.with_pulse_steps(chunk_size)?
        let _ = chunk_size;
        Ok(Self { backend: Backend::Tract, _session_inner: () })
    }
}

/// Helper: load all 4 v3 hybrid ONNX graphs as InferenceSessions.
pub struct V3HybridSessions {
    pub encoder: InferenceSession,            // TinyVC ConvNeXt-v2 (4.7M, INT8 5MB)
    pub speaker_encoder: InferenceSession,    // Spark BiCodec (6-12M, INT8 5MB)
    pub speaker_flow: InferenceSession,       // OpenVoice ResidualCoupling (8.7M, INT8 9MB)
    pub vocoder: InferenceSession,            // F5-TTS Vocos (13M, INT8 7MB)
}

impl V3HybridSessions {
    pub fn load_all(models_dir: &str, backend: Backend) -> InferResult<Self> {
        Ok(Self {
            encoder: InferenceSession::load(&format!("{models_dir}/encoder.int8.onnx"), backend)?,
            speaker_encoder: InferenceSession::load(&format!("{models_dir}/spark_speaker_encoder.int8.onnx"), backend)?,
            speaker_flow: InferenceSession::load(&format!("{models_dir}/openvoice_residual_flow.int8.onnx"), backend)?,
            vocoder: InferenceSession::load(&format!("{models_dir}/voscos.int8.onnx"), backend)?,
        })
    }
}
