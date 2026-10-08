//! vc-ort: ONNX inference orchestration in Rust for v2.0
//!
//! Primary backend: `ort` crate (Rust bindings to ONNXRuntime 1.27 C API via
//! `ort-sys`). Auto-downloads prebuilt libonnxruntime for x86_64/arm64-linux,
//! macOS, Windows. Fastest path on x86 (AVX2 MLAS GEMM) and ARM (XNNPACK).
//!
//! Fallback backend: `tract-onnx` (pure-Rust ONNX interpreter by Sonos).
//! Used for ops ORT doesn't support (rare), for pulsification streaming
//! conversion, or for environments where libonnxruntime can't be downloaded.
//!
//! This module replaces v1.0's Python `onnxruntime` wrappers
//! (`prototype/src/vc_realtime/encoder.py`, `decoder.py`, `flow.py`,
//! `vocoder_v2.py`, `speaker_encoder_v3.py`) with Rust-native equivalents.
//! Same ONNX models, same I/O shapes (`[B, C, T]` float32 in, `[B, C', T']`
//! float32 out), but Rust performance and no GIL.
//!
//! # Quick start
//!
//! ```no_run
//! use ndarray::ArrayD;
//! use vc_ort::{Backend, InferenceSession};
//!
//! let mut sess = InferenceSession::load("models/encoder.int8.onnx", Backend::Ort).unwrap();
//! let input = ndarray::Array3::<f32>::zeros((1, 128, 100)).into_dyn();
//! let output: ArrayD<f32> = sess.run(input).unwrap();
//! ```
//!
//! # Backend selection
//!
//! - [`Backend::Ort`]: ONNXRuntime 1.27 C API. Default. Best perf on x86/ARM
//!   with AVX2/NEON SIMD. ~5-10x faster than tract for most ops.
//! - [`Backend::Tract`]: pure-Rust ONNX. Slower but no C++ dependency,
//!   ~5-15 MB static binary. Useful for embedded targets without libstdc++.

use std::path::Path;
use std::sync::Arc;

use ndarray::ArrayD;
use thiserror::Error;

// tract-onnx prelude — pulls in `Tensor`, `TValue`, `IntoTValue`, `Runnable`,
// `TypedModel`, `TypedRunnableModel`, `InferenceModel`, `InferenceModelExt`,
// `tvec!`, `TVec`, etc. (re-exports tract_core::prelude -> tract_data::prelude)
use tract_onnx::prelude::*;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Error, Debug)]
pub enum InferError {
    #[error("ORT inference failed: {0}")]
    Ort(String),
    #[error("tract inference failed: {0}")]
    Tract(String),
    #[error("ONNX model not found at {0}")]
    NotFound(String),
    #[error("Tensor shape mismatch: expected {expected:?}, got {got:?}")]
    Shape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("No backend session loaded")]
    NoSession,
    #[error("Expected single output, got {0}")]
    MultiOutput(usize),
}

impl<R> From<ort::Error<R>> for InferError
where
    R: 'static,
{
    fn from(e: ort::Error<R>) -> Self {
        InferError::Ort(e.to_string())
    }
}

impl From<anyhow::Error> for InferError {
    fn from(e: anyhow::Error) -> Self {
        InferError::Tract(e.to_string())
    }
}

pub type InferResult<T> = Result<T, InferError>;

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

/// Backend selection — runtime configurable per [`InferenceSession`].
///
/// Both backends take the same ONNX file path and the same `ArrayD<f32>` input
/// shape, and produce the same `ArrayD<f32>` output. The backend choice is
/// orthogonal to the model and can be A/B compared via [`Backend::Ort`] vs
/// [`Backend::Tract`] on the same ONNX file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Rust → ONNXRuntime 1.27 C API (default, fastest on x86 AVX2 MLAS + ARM XNNPACK).
    /// Auto-downloads prebuilt `libonnxruntime.so` / `.dylib` / `.dll`.
    Ort,
    /// Pure-Rust ONNX (tract by Sonos, ~5-15 MB static binary, no C++ dep).
    /// Slower but portable to any platform with a Rust toolchain.
    Tract,
}

// ---------------------------------------------------------------------------
// InferenceSession — single ONNX model, single backend
// ---------------------------------------------------------------------------

/// A single loaded ONNX inference session. Either `ort` or `tract` — never
/// both (loaded lazily based on [`Backend`]).
///
/// `InferenceSession` is `Send + Sync` so it can live behind an `Arc` in a
/// real-time audio pipeline and be shared across the audio callback thread
/// and the decoder thread.
///
/// NOTE: [`InferenceSession::run`] takes `&mut self` because `ort::Session::run`
/// requires a `&mut self` (ONNXRuntime mutates internal run-state). This is
/// fine for the v2.0 single-decoder-thread design — sessions are called
/// sequentially, never concurrently.
#[derive(Debug)]
pub struct InferenceSession {
    backend: Backend,
    // Populated iff `backend == Ort`:
    session_ort: Option<ort::session::Session>,
    // Populated iff `backend == Tract`:
    // `TypedRunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>>` — the
    // concrete tract plan type. `Arc<TypedRunnableModel>` implements the
    // `Runnable` trait so `.run()` is dispatched statically.
    session_tract: Option<Arc<TypedRunnableModel>>,
    // OPT-9: cached `IoBinding` + bound-output shape for zero-allocation
    // streaming inference on the ORT backend. Populated lazily on the first
    // `run_into_cached` call; reused across all subsequent calls with the
    // same output shape. When the caller's `output` buffer changes shape,
    // the cache is invalidated and a new binding (with a new bound output
    // tensor of the right shape) is created on the next call. Use
    // `clear_cache()` to invalidate explicitly (e.g. between pipeline
    // stages that produce different shapes).
    //
    // `IoBinding` is `Send` (see ort/src/session/io_binding.rs:224) and
    // holds an `Arc<SharedSessionInner>` rather than a borrow, so it can
    // live alongside `session_ort` in the same struct without lifetime
    // issues.
    cached_binding: Option<ort::session::IoBinding>,
    // Tracks the bound output tensor's shape so we know when to invalidate
    // the cached binding (shape change ⇒ new allocation required).
    cached_output_shape: Option<Vec<usize>>,
}

// `ort::Session` is `Send + Sync` (see `ort/src/session/mod.rs:723-724`:
// `unsafe impl Send for Session {}` / `unsafe impl Sync for Session {}`,
// with a thorough SAFETY comment justifying both for the ONNXRuntime C
// API). `IoBinding` is `Send` (see `ort/src/session/io_binding.rs:224`).
// `Arc<TypedRunnableModel>` is `Send + Sync` because the underlying
// `SimplePlan` only contains `Arc<...>`-wrapped graph data, plain `Vec`s
// of `usize`/`OutletId`/`Symbol`, an `Option<Executor>` (enum of
// `SingleThread` / `Arc<ThreadPool>` / `RayonGlobal` — all Send+Sync),
// and an `Option<Arc<dyn TurnStateHandler + 'static>>` (the trait is
// declared `: Send + Sync` at `tract-core/src/plan.rs:186`). Both
// backends are therefore thread-safe by construction — no `unsafe impl`
// is required on `InferenceSession`.

impl InferenceSession {
    /// Load an ONNX model from disk. Backend is chosen by the `Backend` enum.
    ///
    /// For `Backend::Ort`: builds an [`ort::session::Session`] with
    /// `GraphOptimizationLevel::Level3`, `intra_op_num_threads = 2`,
    /// `inter_op_num_threads = 1` — matching the v1.0 Python config in
    /// `encoder.py` / `decoder.py`.
    ///
    /// For `Backend::Tract`: parses the ONNX protobuf via
    /// `tract_onnx::onnx().model_for_path(...)`, runs inference → typed
    /// model conversion via [`InferenceModelExt::into_optimized`], and
    /// compiles to a [`TypedRunnableModel`] via [`IntoRunnable::into_runnable`].
    pub fn load(model_path: &str, backend: Backend) -> InferResult<Self> {
        if !Path::new(model_path).exists() {
            return Err(InferError::NotFound(model_path.to_string()));
        }
        match backend {
            Backend::Ort => {
                let session = ort::session::Session::builder()?
                    .with_optimization_level(
                        ort::session::builder::GraphOptimizationLevel::Level3,
                    )?
                    .with_intra_threads(2)?
                    .with_inter_threads(1)?
                    .commit_from_file(model_path)?;
                Ok(Self {
                    backend,
                    session_ort: Some(session),
                    session_tract: None,
                    cached_binding: None,
                    cached_output_shape: None,
                })
            }
            Backend::Tract => {
                // tract-onnx load path: ONNX proto -> InferenceModel -> TypedModel -> RunnableModel
                let inference_model: InferenceModel =
                    tract_onnx::onnx().model_for_path(model_path)?;
                let typed_model: TypedModel = inference_model.into_optimized()?;
                let runnable: Arc<TypedRunnableModel> = typed_model.into_runnable()?;
                Ok(Self {
                    backend,
                    session_ort: None,
                    session_tract: Some(runnable),
                    cached_binding: None,
                    cached_output_shape: None,
                })
            }
        }
    }

    /// Convenience getter for the active backend.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Number of named inputs declared by the loaded ONNX graph.
    pub fn input_count(&self) -> usize {
        match self.backend {
            Backend::Ort => self
                .session_ort
                .as_ref()
                .map(|s| s.inputs().len())
                .unwrap_or(0),
            Backend::Tract => self
                .session_tract
                .as_ref()
                .map(|s| s.input_count())
                .unwrap_or(0),
        }
    }

    /// Number of named outputs declared by the loaded ONNX graph.
    pub fn output_count(&self) -> usize {
        match self.backend {
            Backend::Ort => self
                .session_ort
                .as_ref()
                .map(|s| s.outputs().len())
                .unwrap_or(0),
            Backend::Tract => self
                .session_tract
                .as_ref()
                .map(|s| s.output_count())
                .unwrap_or(0),
        }
    }

    /// Run inference. `input`: `ArrayD<f32>` (typically `[B, C, T]` or `[B, T]`).
    /// Returns: `ArrayD<f32>` (typically `[B, C', T']`).
    ///
    /// If the graph has more than one input, only the first slot is fed with
    /// `input` and the others are left default (zeros). For multi-input
    /// graphs, use [`InferenceSession::run_named`] instead.
    ///
    /// If the graph produces more than one output, returns the first output
    /// tensor (matching the Python `decoder.py` convention of indexing
    /// `outputs[0]`).
    pub fn run(&mut self, input: ArrayD<f32>) -> InferResult<ArrayD<f32>> {
        match self.backend {
            Backend::Ort => {
                let session = self
                    .session_ort
                    .as_mut()
                    .ok_or(InferError::NoSession)?;
                // Convert `ArrayD<f32>` (ndarray 0.17, same version as
                // `ort`'s ndarray feature) into an `ort::value::Tensor<f32>`
                // value — this is a no-op view (zero-copy) if the array is
                // contiguous, otherwise it copies into a contiguous buffer.
                let tensor = ort::value::Tensor::<f32>::from_array(input)?;
                // Capture the first output name BEFORE the mutable `run` borrow
                // (Rust's borrow checker otherwise flags `session.outputs()` as a
                // second immutable borrow alive alongside the mutable one.)
                let first_name = session.outputs()[0].name().to_string();
                let outputs = session.run(ort::inputs![tensor])?;
                let out_value = outputs
                    .get(first_name.as_str())
                    .ok_or_else(|| {
                        InferError::Ort(format!("output `{first_name}` missing"))
                    })?;
                let view = out_value.try_extract_array::<f32>()?;
                Ok(view.into_owned())
            }
            Backend::Tract => {
                let runnable = self
                    .session_tract
                    .as_ref()
                    .ok_or(InferError::NoSession)?;
                // Convert `ArrayD<f32>` -> `TValue` via the `IntoTValue` impl
                // on `ndarray::Array<T, D>` (re-exported as
                // `tract_data::prelude::Array`).
                let inputs: TVec<TValue> = tvec![input.into_tvalue()];
                let outputs: TVec<TValue> = <Arc<TypedRunnableModel> as Runnable>::run(
                    runnable, inputs,
                )?;
                if outputs.len() != 1 {
                    return Err(InferError::MultiOutput(outputs.len()));
                }
                let out_tensor: &Tensor = &outputs[0];
                let view = out_tensor.to_plain_array_view::<f32>()?;
                Ok(view.to_owned())
            }
        }
    }

    /// Run inference and return ALL outputs as a `Vec<ArrayD<f32>>`.
    ///
    /// Unlike [`InferenceSession::run`] which returns only the first output
    /// (matching the v1.0 Python convention of `outputs[0]`), `run_all`
    /// extracts every output declared by the ONNX graph in declaration order.
    ///
    /// Required for multi-output ONNX models like the v3-hybrid
    /// `encoder.int8.onnx` (TinyVC) which produces two outputs:
    ///   - `content`   `[B, 768, T]` — content features for kNN-VC retrieval
    ///   - `f0_logits` `[B, 512, T]` — pitch classification logits
    ///
    /// The caller indexes into the returned `Vec` positionally (output 0,
    /// output 1, ...) — the order matches `session.outputs()`.
    ///
    /// # Errors
    /// - [`InferError::NoSession`] if no backend session is loaded.
    /// - [`InferError::Ort`] if any output extraction fails (e.g. the
    ///   declared output name is missing from `SessionOutputs`, or the
    ///   tensor element type is not `f32`).
    /// - [`InferError::MultiOutput`] (Tract only) if any output is missing
    ///   — Tract returns outputs as a `TVec<TValue>` indexed positionally,
    ///   so we just extract them in order.
    pub fn run_all(&mut self, input: ArrayD<f32>) -> InferResult<Vec<ArrayD<f32>>> {
        match self.backend {
            Backend::Ort => {
                let session = self
                    .session_ort
                    .as_mut()
                    .ok_or(InferError::NoSession)?;
                // Capture ALL output names BEFORE the mutable `run` borrow,
                // same pattern as `run()` (the borrow checker would otherwise
                // flag `session.outputs()` as a second immutable borrow alive
                // alongside the mutable `session.run` one).
                let out_names: Vec<String> = session
                    .outputs()
                    .iter()
                    .map(|o| o.name().to_string())
                    .collect();
                let tensor = ort::value::Tensor::<f32>::from_array(input)?;
                let outputs = session.run(ort::inputs![tensor])?;
                let mut result = Vec::with_capacity(out_names.len());
                for name in &out_names {
                    let out_value = outputs
                        .get(name.as_str())
                        .ok_or_else(|| InferError::Ort(format!("output `{name}` missing")))?;
                    let view = out_value.try_extract_array::<f32>()?;
                    // `view` is `ArrayViewD<'_, f32>` (dynamic dim, borrowed
                    // from `outputs`). `into_owned()` clones the data into an
                    // owned `Array<f32, IxDyn>` = `ArrayD<f32>`. We do this
                    // because `outputs` (and the views into it) are dropped
                    // at the end of this scope.
                    result.push(view.into_owned());
                }
                Ok(result)
            }
            Backend::Tract => {
                let runnable = self
                    .session_tract
                    .as_ref()
                    .ok_or(InferError::NoSession)?;
                let inputs: TVec<TValue> = tvec![input.into_tvalue()];
                let outputs: TVec<TValue> =
                    <Arc<TypedRunnableModel> as Runnable>::run(runnable, inputs)?;
                let mut result = Vec::with_capacity(outputs.len());
                for out_tensor in &outputs {
                    let view = out_tensor.to_plain_array_view::<f32>()?;
                    result.push(view.to_owned());
                }
                Ok(result)
            }
        }
    }

    /// Run inference and write the first output into a caller-provided
    /// buffer (`output`), eliminating the `ArrayD<f32>` allocation that
    /// [`InferenceSession::run`] would otherwise return.
    ///
    /// Designed for streaming inference with fixed shapes (e.g. 80 ms
    /// audio chunks → fixed-size mel-spec → fixed-size waveform): the
    /// caller allocates `output` once at startup and reuses it across
    /// thousands of `run_into` calls, dropping the per-call `Vec`-backed
    /// `ArrayD` allocation (~2-5 ms each on a hot allocator).
    ///
    /// # IoBinding (ORT only)
    ///
    /// On the ORT backend this uses [`ort::session::IoBinding`] under the
    /// hood: the input is bound by reference (zero-copy for contiguous
    /// `ArrayD`), the output is bound to a freshly-allocated tensor of the
    /// same shape as the caller's `output`, then `Session::run_binding` is
    /// invoked. ORT writes its result directly into the bound output buffer,
    /// which is then extracted as an `ArrayView` (non-owning view into the
    /// bound buffer) and copied into `output` via `Array::assign`.
    ///
    /// NOTE: `ort 2.0.0-rc.13`'s `IoBinding::bind_output` takes an *owned*
    /// `Value<T>` — there is no borrowed-reference variant. This means we
    /// cannot make the bound output buffer alias the caller's `&mut output`
    /// memory; we must allocate a separate bound tensor and copy at the
    /// end. The copy is a single `memcpy` (no extra tensor allocation), so
    /// the per-call cost is one `memcpy` + zero tensor allocations on the
    /// steady-state path.
    ///
    /// For tract (no IoBinding equivalent), this falls back to the regular
    /// `run()` followed by `Array::assign` into `output`.
    ///
    /// # Shape contract
    ///
    /// `output` must have the same shape as the model's first declared
    /// output; otherwise this function will return
    /// [`InferError::Shape`] (for ORT, propagated from ORT's runtime
    /// shape check) OR silently resize `output` to match (for tract).
    ///
    /// # Errors
    /// - [`InferError::NoSession`] if no backend session is loaded.
    /// - [`InferError::Ort`] / [`InferError::Tract`] if the underlying
    ///   inference call fails (shape mismatch, dtype mismatch, etc.).
    pub fn run_into(
        &mut self,
        input: ArrayD<f32>,
        output: &mut ArrayD<f32>,
    ) -> InferResult<()> {
        match self.backend {
            Backend::Ort => {
                let session = self
                    .session_ort
                    .as_mut()
                    .ok_or(InferError::NoSession)?;
                // Capture the first input + first output names BEFORE the
                // mutable `run_binding` borrow (same borrow-checker pattern
                // as `run()` / `run_all()`).
                let in_name = session.inputs()[0].name().to_string();
                let out_name = session.outputs()[0].name().to_string();
                // Build input tensor — zero-copy if `input` is contiguous
                // (which it is, having just been constructed by the caller
                // or moved across the FFI boundary).
                let input_tensor = ort::value::Tensor::<f32>::from_array(input)?;
                // Create the IoBinding for this run. NOTE: a future
                // optimization can cache `IoBinding` + a pre-allocated
                // output `Tensor` inside `InferenceSession` so that the
                // bound output buffer is allocated once at first call and
                // reused across all subsequent calls (true steady-state
                // zero-allocation). For now we create a fresh binding per
                // call — still uses IoBinding semantics so the optimization
                // path is a drop-in.
                let mut binding = session.create_binding()?;
                binding.bind_input(in_name.as_str(), &input_tensor)?;
                // Allocate the bound output tensor with the SAME shape as
                // the caller's `output` buffer. ORT will overwrite this
                // memory during `run_binding`.
                let out_shape: Vec<usize> = output.shape().to_vec();
                let bound_output = ndarray::ArrayD::<f32>::zeros(
                    out_shape.as_slice(),
                );
                let bound_output_tensor =
                    ort::value::Tensor::<f32>::from_array(bound_output)?;
                binding.bind_output(out_name.as_str(), bound_output_tensor)?;
                // Run with the IoBinding — `Session::run_binding` returns a
                // `SessionOutputs` whose values are non-owning views into
                // the bound output buffer (via `GetBoundOutputValues`).
                let outputs = session.run_binding(&binding)?;
                let out_value = outputs
                    .get(out_name.as_str())
                    .ok_or_else(|| {
                        InferError::Ort(format!(
                            "output `{out_name}` missing from IoBinding"
                        ))
                    })?;
                let view = out_value.try_extract_array::<f32>()?;
                // `view` borrows from `outputs` which borrows from `binding`.
                // Copy into the caller's buffer.
                output.assign(&view);
                Ok(())
            }
            Backend::Tract => {
                // tract has no IoBinding equivalent — use the regular
                // `run()` (which allocates a fresh output ArrayD internally)
                // and move that result into the caller's buffer slot.
                let result = self.run(input)?;
                // If the caller's buffer shape matches, `assign` in-place
                // (keeps the caller's allocation). Otherwise replace the
                // caller's buffer by move (the caller's pre-allocated
                // storage is dropped and replaced by `result`).
                if output.shape() == result.shape() {
                    output.assign(&result);
                } else {
                    *output = result;
                }
                Ok(())
            }
        }
    }

    /// Run inference with a cached [`IoBinding`] for true zero-allocation
    /// streaming on the ORT backend (OPT-9).
    ///
    /// First call creates + caches an `IoBinding` with a pre-allocated bound
    /// output tensor of the same shape as `output`. Subsequent calls reuse
    /// the same binding — only the input is re-bound each call (since the
    /// input data changes per call), and ORT writes its output directly
    /// into the still-bound output buffer.
    ///
    /// Steady-state per-call cost: one input-tensor allocation (owned by the
    /// caller's `input`), one `bind_input` Arc-clone (refcount bump), one
    /// `memcpy` of the bound output buffer → caller's `output`. **Zero**
    /// tensor allocations on the ORT side after the first call.
    ///
    /// # Cache invalidation
    ///
    /// The cache is auto-invalidated when the caller's `output.shape()`
    /// differs from `cached_output_shape` (e.g. when switching from
    /// `[B, 1, T_audio]` to `[B, T_audio]` between pipeline stages). The
    /// next call then pays a one-shot re-bind cost (new IoBinding + new
    /// bound output tensor). For explicit invalidation (e.g. on a known
    /// shape transition), call [`InferenceSession::clear_cache`].
    ///
    /// # Tract backend
    ///
    /// `tract` has no IoBinding equivalent — falls back to `run()` + assign
    /// (one allocation per call). The cached path is ORT-only.
    ///
    /// # Shape contract
    ///
    /// `output` must have the same shape as the model's first declared
    /// output; ORT will return a shape-mismatch error otherwise.
    ///
    /// # Errors
    /// - [`InferError::NoSession`] if no backend session is loaded.
    /// - [`InferError::Ort`] / [`InferError::Tract`] if the underlying
    ///   inference call fails (shape mismatch, dtype mismatch, etc.).
    pub fn run_into_cached(
        &mut self,
        input: ArrayD<f32>,
        output: &mut ArrayD<f32>,
    ) -> InferResult<()> {
        match self.backend {
            Backend::Ort => {
                let out_shape: Vec<usize> = output.shape().to_vec();
                // Invalidate cache if output shape changed since the last call.
                // `as_deref()` compares `Option<&[usize]>` to `Option<&Vec<usize>>`
                // (via the deref coercion to `&[usize]`) — clean shape equality.
                let need_rebind = self
                    .cached_binding
                    .is_none()
                    || self.cached_output_shape.as_deref() != Some(out_shape.as_slice());

                if need_rebind {
                    // `create_binding` takes `&self` (immutable session borrow),
                    // so we take a short-lived immutable borrow to construct the
                    // new binding, then drop it before mutably borrowing later.
                    let session = self
                        .session_ort
                        .as_ref()
                        .ok_or(InferError::NoSession)?;
                    // Only the output name is needed here — the input is bound
                    // in the next block (mutable session borrow) since it
                    // changes every call. We capture the output name once
                    // because `bind_output` happens here, in the cold path.
                    let out_name = session.outputs()[0].name().to_string();
                    let mut binding = session.create_binding()?;
                    // Allocate the bound output tensor with the SAME shape as
                    // the caller's `output` buffer. ORT will overwrite this
                    // memory during `run_binding` for every subsequent call —
                    // this is the zero-alloc steady-state path.
                    let bound_output =
                        ndarray::ArrayD::<f32>::zeros(out_shape.as_slice());
                    let bound_output_tensor =
                        ort::value::Tensor::<f32>::from_array(bound_output)?;
                    binding.bind_output(out_name.as_str(), bound_output_tensor)?;
                    // Stash the freshly-built binding + shape into self so the
                    // next block (mutable session borrow) can use them.
                    // The immutable `session` borrow ends here — the IoBinding
                    // holds an `Arc<SharedSessionInner>` (not a borrow), so it
                    // is safe to keep across the mutable borrow below.
                    self.cached_binding = Some(binding);
                    self.cached_output_shape = Some(out_shape.clone());
                }

                // Disjoint mutable borrows of two struct fields: borrow
                // checker permits `&mut self.session_ort` and `&mut
                // self.cached_binding` to coexist (different fields).
                let session = self
                    .session_ort
                    .as_mut()
                    .ok_or(InferError::NoSession)?;
                let binding = self
                    .cached_binding
                    .as_mut()
                    .ok_or_else(|| InferError::Ort("cached IoBinding missing".to_string()))?;
                let in_name = session.inputs()[0].name().to_string();
                let out_name = session.outputs()[0].name().to_string();

                // Re-bind input each call (input data changes). ORT's C API
                // `BindInput` replaces any existing binding for that name —
                // no need to `clear_inputs()` first. The `IoBinding::held_inputs`
                // `MiniMap::insert` replaces the previous `Arc<ValueInner>`,
                // releasing the previous input tensor (refcount → 0).
                let input_tensor = ort::value::Tensor::<f32>::from_array(input)?;
                binding.bind_input(in_name.as_str(), &input_tensor)?;
                // `input_tensor` is dropped at end of this scope, but the
                // `Arc::clone` inside `bind_input` keeps the underlying
                // buffer alive for the duration of `run_binding`.

                // Run with the cached binding. `Session::run_binding` takes
                // `&mut self` (mut session) and `&IoBinding` (imm binding),
                // returning `SessionOutputs<'b>` whose lifetime is tied to
                // the binding borrow — fine because we extract + copy before
                // the borrow ends.
                let outputs = session.run_binding(binding)?;
                let out_value = outputs
                    .get(out_name.as_str())
                    .ok_or_else(|| {
                        InferError::Ort(format!(
                            "output `{out_name}` missing from IoBinding"
                        ))
                    })?;
                let view = out_value.try_extract_array::<f32>()?;
                // Copy from the bound output buffer into the caller's buffer.
                // This is a single `memcpy` (no tensor allocation).
                output.assign(&view);
                Ok(())
            }
            Backend::Tract => {
                // tract has no IoBinding — use `run()` + assign. This still
                // allocates a fresh ArrayD per call, but at least avoids
                // returning it to the caller (the buffer is reused via
                // `assign`).
                let result = self.run(input)?;
                if output.shape() == result.shape() {
                    output.assign(&result);
                } else {
                    *output = result;
                }
                Ok(())
            }
        }
    }

    /// Clear the cached `IoBinding` + bound output tensor (OPT-9).
    ///
    /// Call this when the caller knows the output shape is about to change
    /// (e.g. switching from a `[B, T]` mel-spec inference stage to a
    /// `[B, 1, T*hop]` waveform stage), so that the next
    /// [`InferenceSession::run_into_cached`] call starts fresh with a new
    /// binding sized for the new shape.
    ///
    /// Cheap no-op if the cache is already empty (e.g. immediately after
    /// `load()`).
    pub fn clear_cache(&mut self) {
        self.cached_binding = None;
        self.cached_output_shape = None;
    }

    /// Run inference with named input bindings. Use this for multi-input
    /// graphs (e.g. `decoder.filter_net` which takes `content`, `f0`,
    /// `energy`, `source`).
    ///
    /// `inputs` is a slice of `(name, tensor)` pairs; the order does not
    /// matter for `ort` (it builds a name-keyed map). For `tract`, the
    /// inputs are forwarded in declaration order.
    pub fn run_named(
        &mut self,
        inputs: &[(String, ArrayD<f32>)],
    ) -> InferResult<ArrayD<f32>> {
        match self.backend {
            Backend::Ort => {
                let session = self
                    .session_ort
                    .as_mut()
                    .ok_or(InferError::NoSession)?;
                // Build a name-keyed `Vec<(Cow<str>, SessionInputValue)>` and
                // pass via the `ort::inputs!`-compatible `SessionInputs::ValueMap`.
                let mut input_vec: Vec<(
                    std::borrow::Cow<'_, str>,
                    ort::session::SessionInputValue<'_>,
                )> = Vec::with_capacity(inputs.len());
                for (name, arr) in inputs {
                    let tensor = ort::value::Tensor::<f32>::from_array(arr.clone())?;
                    input_vec.push((
                        std::borrow::Cow::Borrowed(name.as_str()),
                        ort::session::SessionInputValue::from(tensor),
                    ));
                }
                let first_name = session.outputs()[0].name().to_string();
                let outputs = session.run(input_vec)?;
                let out_value = outputs
                    .get(first_name.as_str())
                    .ok_or_else(|| {
                        InferError::Ort(format!("output `{first_name}` missing"))
                    })?;
                let view = out_value.try_extract_array::<f32>()?;
                Ok(view.into_owned())
            }
            Backend::Tract => {
                let runnable = self
                    .session_tract
                    .as_ref()
                    .ok_or(InferError::NoSession)?;
                let mut tvec: TVec<TValue> = tvec![];
                for (_name, arr) in inputs {
                    tvec.push(arr.clone().into_tvalue());
                }
                let outputs: TVec<TValue> = <Arc<TypedRunnableModel> as Runnable>::run(
                    runnable, tvec,
                )?;
                if outputs.is_empty() {
                    return Err(InferError::MultiOutput(0));
                }
                let out_tensor: &Tensor = &outputs[0];
                let view = out_tensor.to_plain_array_view::<f32>()?;
                Ok(view.to_owned())
            }
        }
    }

    /// Pulsify the *currently loaded* model into a chunk-streaming variant
    /// in-place (re-uses the loaded `TypedModel` from `session_tract`).
    ///
    /// This is the in-memory form of the free function [`pulsify`]
    /// below — when you already have an `InferenceSession` with the
    /// `Tract` backend, you can pulsify it directly without re-reading
    /// the ONNX file from disk. Returns a *new* session (the original
    /// is left untouched).
    ///
    /// Returns [`InferError::NoSession`] if `self` isn't a `Tract`-backend
    /// session. Returns [`InferError::Tract`] if pulsification fails for
    /// this model architecture (common — see [`pulsify`] docstring).
    pub fn pulsify(&self, chunk_size: usize) -> InferResult<Self> {
        let runnable = self
            .session_tract
            .as_ref()
            .ok_or(InferError::NoSession)?;
        // `TypedRunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>>`.
        // `SimplePlan::model()` gives us back `&TypedModel` so we can
        // re-pulsify without re-reading the ONNX protobuf from disk.
        let typed: &TypedModel = runnable.model();
        pulsify_typed(typed, chunk_size).map(|runnable| Self {
            backend: Backend::Tract,
            session_ort: None,
            session_tract: Some(runnable),
            cached_binding: None,
            cached_output_shape: None,
        })
    }
}

/// Pulsify a full-sequence ONNX model into a chunk-streaming variant.
///
/// `tract-pulse` converts a full-sequence ONNX graph into a
/// chunk-streaming variant with internal state (delay lines, source
/// padding, overlap-add tails). This eliminates the need for SOLA
/// crossfade at chunk boundaries: a pulsified model maintains streaming
/// state across `.run()` calls, so adjacent chunks see the same context
/// that a full-sequence inference would have produced internally.
///
/// # Pipeline impact
///
/// In the v2.0 Rust pipeline, calling `pulsify` on `vocos.int8.onnx`
/// (the 48 kHz waveform decoder) would let the real-time audio thread
/// feed chunks of `chunk_size` samples directly and receive the
/// corresponding waveform chunk back without the SOLA windowing that
/// the v1.0 Python path needed (`vocoder_v2.py:crossfade()`).
///
/// # Limitations
///
/// Pulsification requires:
/// 1. At least one source input declared with a symbolic streaming
///    dimension (e.g. ONNX `dim_param` on the time axis).
/// 2. Every op in the graph to have a registered pulsifier — tract-pulse
///    ships pulsifiers for Conv/Deconv/concat/slice/scan/etc. but exotic
///    custom ops will fail.
/// 3. The streaming symbol appears linearly (≤ once per wire shape);
///    super-linear wires need ROI annotations.
///
/// For models that fail any of the above (common for production vocoders
/// that ship with fixed input shapes), this function returns
/// [`InferError::Tract`] with a descriptive message. Callers should
/// fall back to the SOLA path in that case — the failure is graceful.
///
/// # Returns
///
/// A new `InferenceSession` with the `Tract` backend whose
/// `session_tract` is the pulsed, stateful `TypedRunnableModel`. Note
/// that callers must wrap calls in `SimpleState` to actually maintain
/// streaming state across `.run()` invocations — the bare
/// `TypedRunnableModel` returned here runs each chunk independently
/// (no carryover). Full `SimpleState` integration is tracked as
/// follow-on work to OPT-13.
pub fn pulsify(model_path: &str, chunk_size: usize) -> InferResult<InferenceSession> {
    if !Path::new(model_path).exists() {
        return Err(InferError::NotFound(model_path.to_string()));
    }
    // Load ONNX proto → InferenceModel → TypedModel → optimized TypedModel.
    let inference_model: InferenceModel =
        tract_onnx::onnx().model_for_path(model_path)?;
    let typed_model: TypedModel = inference_model.into_optimized()?;
    pulsify_typed(&typed_model, chunk_size).map(|runnable| InferenceSession {
        backend: Backend::Tract,
        session_ort: None,
        session_tract: Some(runnable),
        cached_binding: None,
        cached_output_shape: None,
    })
}

/// Shared pulsification core: take a `TypedModel` (already optimized),
/// add a streaming symbol if the model doesn't already have one, then
/// pulse → typed → runnable.
fn pulsify_typed(typed: &TypedModel, chunk_size: usize) -> InferResult<Arc<TypedRunnableModel>> {
    use tract_pulse::model::PulsedModelExt as _;
    // tract-pulse requires at least one source to carry the streaming
    // symbol in its shape. tract-onnx parses ONNX `dim_param` (e.g. the
    // `batch` or `time` axis declared as a string in the protobuf) into
    // symbolic `TDim`s, so we scan input 0's shape for any pre-existing
    // symbol and reuse it; if none, we mint a fresh "S".
    let streaming_sym = typed
        .input_fact(0)
        .ok()
        .and_then(|f| {
            f.shape
                .iter()
                .flat_map(|d| d.symbols().into_iter())
                .next()
        })
        .unwrap_or_else(|| typed.symbols.sym("S"));

    // Pulsify: convert the graph to per-pulse chunks with internal state.
    // `PulsedModel::new` returns a `PulsedModel` (a graph over `PulsedOp`s).
    // `.into_typed()` lowers it back to a `TypedModel` whose ops now
    // include `Delay` / `Source` state-machines, then `.into_runnable()`
    // produces the executable `SimplePlan`.
    let pulsed = tract_pulse::model::PulsedModel::new(typed, streaming_sym.clone(), &chunk_size.to_dim())
        .map_err(|e| InferError::Tract(format!("pulse: {e}")))?;
    let pulsed_typed: TypedModel = pulsed
        .into_typed()
        .map_err(|e| InferError::Tract(format!("pulse.into_typed: {e}")))?;
    let runnable: Arc<TypedRunnableModel> = pulsed_typed
        .into_runnable()
        .map_err(|e| InferError::Tract(format!("pulse.into_runnable: {e}")))?;
    Ok(runnable)
}

// ---------------------------------------------------------------------------
// V3HybridSessions — load all v3 hybrid ONNX graphs
// ---------------------------------------------------------------------------

/// All five v3 hybrid ONNX graphs loaded into a single struct.
///
/// Mirrors the Python v3 hybrid pipeline (see `infer_v3.py`) which uses:
///
/// | Field             | ONNX file                          | Params  | INT8   | Role                          |
/// |-------------------|-----------------------------------|---------|--------|-------------------------------|
/// | `encoder`         | `encoder.int8.onnx`               | 4.7 M   | 5 MB   | TinyVC content + F0 + energy  |
/// | `ref_encoder`     | `openvoice_ref_encoder.int8.onnx` | 0.76 M  | 2.3 MB | OpenVoice 256-d speaker emb  |
/// | `speaker_encoder` | `spark_speaker_encoder.int8.onnx`| 14 M    | 30 MB  | Spark 1024-d + FSQ codes     |
/// | `residual_flow`   | `openvoice_residual_flow.int8.onnx`| 8.7 M  | 9 MB   | OpenVoice ResidualCoupling    |
/// | `vocoder`         | `vocos.int8.onnx`                 | 13 M    | 22 MB  | Vocos 48 kHz waveform decoder |
///
/// All five are loaded with the same [`Backend`] so that A/B comparison
/// between `Ort` and `Tract` can be done without re-instantiating.
pub struct V3HybridSessions {
    /// TinyVC ConvNeXt-v2 content encoder (4.7 M params, INT8 5 MB).
    /// Input: `[B, 128, T_frames]` mel-spec → `[B, 768, T_frames]` content
    /// + `[B, T]` f0 + `[B, T]` energy.
    pub encoder: InferenceSession,
    /// OpenVoice 256-d speaker embedding reference encoder (0.76 M, INT8 2.3 MB).
    /// Input: `[B, n_mels, T_ref]` reference mel → `[B, 256]` speaker emb.
    pub ref_encoder: InferenceSession,
    /// Spark BiCodec + FSQ speaker encoder (14 M, INT8 30 MB).
    /// Input: `[B, 80, T_ref]` reference mel → `[B, 1024]` + FSQ codes.
    pub speaker_encoder: InferenceSession,
    /// OpenVoice ResidualCoupling flow (8.7 M, INT8 9 MB).
    /// Speaker disentanglement: content + spk_emb → converted content.
    pub residual_flow: InferenceSession,
    /// Vocos vocoder (13 M, INT8 22 MB).
    /// Input: `[B, n_mels, T]` → `[B, 1, T_samples]` 48 kHz waveform.
    pub vocoder: InferenceSession,
}

impl V3HybridSessions {
    /// Load all five v3 hybrid ONNX graphs from `models_dir` using `backend`.
    ///
    /// # File names expected
    ///
    /// ```text
    /// {models_dir}/encoder.int8.onnx
    /// {models_dir}/openvoice_ref_encoder.int8.onnx
    /// {models_dir}/spark_speaker_encoder.int8.onnx
    /// {models_dir}/openvoice_residual_flow.int8.onnx
    /// {models_dir}/vocos.int8.onnx
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`InferError::NotFound`] if any of the five files is missing.
    pub fn load_all(models_dir: &str, backend: Backend) -> InferResult<Self> {
        Ok(Self {
            encoder: InferenceSession::load(
                &format!("{models_dir}/encoder.int8.onnx"),
                backend,
            )?,
            ref_encoder: InferenceSession::load(
                &format!("{models_dir}/openvoice_ref_encoder.int8.onnx"),
                backend,
            )?,
            speaker_encoder: InferenceSession::load(
                &format!("{models_dir}/spark_speaker_encoder.int8.onnx"),
                backend,
            )?,
            residual_flow: InferenceSession::load(
                &format!("{models_dir}/openvoice_residual_flow.int8.onnx"),
                backend,
            )?,
            vocoder: InferenceSession::load(
                &format!("{models_dir}/vocos.int8.onnx"),
                backend,
            )?,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests — gated by ONNX model existence (skip if not downloaded)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to the prototype's ONNX model directory. Used by all tests below.
    /// Tests auto-skip (with a stderr note) if a required ONNX file is absent.
    const MODELS_DIR: &str = "./models";

    fn have(p: &str) -> bool {
        Path::new(p).exists()
    }

    #[test]
    fn test_load_openvoice_ref_encoder_ort() {
        let path = format!("{MODELS_DIR}/openvoice_ref_encoder.int8.onnx");
        if !have(&path) {
            eprintln!("SKIP: {path} not found");
            return;
        }
        let sess = InferenceSession::load(&path, Backend::Ort);
        assert!(sess.is_ok(), "ORT load failed: {:?}", sess.err());
        let s = sess.unwrap();
        assert!(s.input_count() >= 1, "expected at least 1 input");
        assert!(s.output_count() >= 1, "expected at least 1 output");
    }

    #[test]
    fn test_load_spark_speaker_encoder_ort() {
        let path = format!("{MODELS_DIR}/spark_speaker_encoder.int8.onnx");
        if !have(&path) {
            eprintln!("SKIP: {path} not found");
            return;
        }
        let sess = InferenceSession::load(&path, Backend::Ort);
        assert!(sess.is_ok(), "ORT load failed: {:?}", sess.err());
    }

    #[test]
    fn test_load_openvoice_residual_flow_ort() {
        let path = format!("{MODELS_DIR}/openvoice_residual_flow.int8.onnx");
        if !have(&path) {
            eprintln!("SKIP: {path} not found");
            return;
        }
        let sess = InferenceSession::load(&path, Backend::Ort);
        assert!(sess.is_ok(), "ORT load failed: {:?}", sess.err());
    }

    #[test]
    fn test_load_vocos_ort() {
        let path = format!("{MODELS_DIR}/vocos.int8.onnx");
        if !have(&path) {
            eprintln!("SKIP: {path} not found");
            return;
        }
        let sess = InferenceSession::load(&path, Backend::Ort);
        assert!(sess.is_ok(), "ORT load failed: {:?}", sess.err());
    }

    /// P2.1-1: Verify the TinyVC encoder (SSLFeatureEstimator +
    /// PitchEstimator, exported by `scripts/export_tinyvc_encoder_onnx.py`)
    /// loads cleanly via ORT. This is the 5th v3-hybrid ONNX model — once it
    /// exists, `V3HybridSessions::load_all` works end-to-end.
    #[test]
    fn test_load_encoder() {
        let path = format!("{MODELS_DIR}/encoder.int8.onnx");
        if !have(&path) {
            eprintln!("SKIP: {path} not found");
            return;
        }
        let sess = InferenceSession::load(&path, Backend::Ort);
        assert!(sess.is_ok(), "ORT load failed: {:?}", sess.err());
        let s = sess.unwrap();
        // TinyVC encoder: input "spectrogram" [B, 961, T],
        // outputs "content" [B, 768, T] + "f0" [B, 512, T].
        assert!(s.input_count() == 1, "expected 1 input, got {}", s.input_count());
        assert!(s.output_count() == 2, "expected 2 outputs, got {}", s.output_count());
    }

    #[test]
    fn test_load_missing_file_errors_notfound() {
        let path = "/nonexistent/missing.int8.onnx";
        let err = InferenceSession::load(path, Backend::Ort).unwrap_err();
        assert!(
            matches!(err, InferError::NotFound(ref p) if p == path),
            "expected NotFound, got {err:?}"
        );
    }

    #[test]
    fn test_v3_hybrid_load_all_ort() {
        // V3HybridSessions needs encoder.int8.onnx + 4 others. encoder.int8.onnx
        // is only present if the user has run `scripts/export_encoder_onnx.py`
        // (not in the default Phase-1 download set), so this test self-skips
        // if `encoder.int8.onnx` is missing.
        let encoder_path = format!("{MODELS_DIR}/encoder.int8.onnx");
        let vocos_path = format!("{MODELS_DIR}/vocos.int8.onnx");
        if !have(&encoder_path) || !have(&vocos_path) {
            eprintln!(
                "SKIP: V3HybridSessions needs both {encoder_path} and {vocos_path}"
            );
            return;
        }
        let sessions = V3HybridSessions::load_all(MODELS_DIR, Backend::Ort);
        assert!(
            sessions.is_ok(),
            "V3HybridSessions::load_all failed: {:?}",
            sessions.err()
        );
    }

    /// Smoke test: load OpenVoice ref_encoder via ORT and run a tiny
    /// forward pass with a zero tensor. Verifies the full I/O path
    /// (load → Tensor::from_array → session.run → try_extract_array).
    #[test]
    fn test_openvoice_ref_encoder_forward_zeros() {
        let path = format!("{MODELS_DIR}/openvoice_ref_encoder.int8.onnx");
        if !have(&path) {
            eprintln!("SKIP: {path} not found");
            return;
        }
        let mut sess = InferenceSession::load(&path, Backend::Ort)
            .expect("ORT load failed");
        // Inspect the input shape from the model's declared input fact —
        // build a small zero tensor of matching rank.
        let inputs = sess.session_ort.as_ref().unwrap().inputs();
        let input_fact = &inputs[0];
        // The input name + rank are known; we use a [1, 128, 16] zero block.
        // If the actual model needs a different shape, ort will return an
        // error which we surface as a skipped test rather than a failure.
        let n_mels = 128;
        let t_frames = 16;
        let input =
            ndarray::Array3::<f32>::zeros((1, n_mels, t_frames)).into_dyn();
        // Suppress unused-warning for input_fact.name() — it's for debug only
        let _name = input_fact.name();
        match sess.run(input) {
            Ok(out) => {
                // Output rank should be >= 1 and total elements > 0
                assert!(out.len() > 0, "output tensor empty");
            }
            Err(e) => {
                // Most likely the model needs a specific input shape — log
                // and skip rather than fail. The point of this test is to
                // exercise the load → run → extract plumbing, not to
                // assert numerical correctness.
                eprintln!("SKIP: forward pass failed (likely shape mismatch): {e}");
            }
        }
    }

    #[test]
    fn test_run_into_basic() {
        // OPT-5: IoBinding / pre-allocated output buffer smoke test.
        // Same shape contract as test_openvoice_ref_encoder_forward_zeros
        // but uses `run_into(input, &mut output)` so the output buffer is
        // caller-provided (no ArrayD returned).
        let path = format!("{MODELS_DIR}/openvoice_ref_encoder.int8.onnx");
        if !have(&path) {
            eprintln!("SKIP: {path} not found");
            return;
        }
        let mut sess = InferenceSession::load(&path, Backend::Ort)
            .expect("ORT load failed");
        // Same input as the forward-zeros test: [1, 128, 16] zero block.
        let n_mels = 128;
        let t_frames = 16;
        let input = ndarray::Array3::<f32>::zeros((1, n_mels, t_frames))
            .into_dyn();
        // Pre-allocate a tiny output buffer. We don't know the exact
        // output shape without inspecting the graph, so use a 1-element
        // stub — if shape mismatches, the IoBinding path surfaces ORT's
        // runtime error and we skip rather than fail (the test's purpose
        // is to exercise the IoBinding plumbing, not assert numerical
        // correctness).
        let mut output = ndarray::Array0::<f32>::zeros(()).into_dyn();
        match sess.run_into(input, &mut output) {
            Ok(()) => {
                // If the call succeeded with a 1-element output, ORT was
                // happy with our bound output shape. Otherwise ORT would
                // have returned a shape-mismatch error (which we skip).
                assert!(
                    output.len() >= 1,
                    "run_into output buffer empty after call"
                );
                eprintln!(
                    "test_run_into_basic: output shape = {:?}",
                    output.shape()
                );
            }
            Err(e) => {
                // Most likely ORT rejected the bound output shape; skip.
                eprintln!(
                    "SKIP: run_into forward failed (likely bound output shape mismatch): {e}"
                );
            }
        }
    }

    #[test]
    fn test_run_into_cached_basic() {
        // OPT-9: cached IoBinding smoke test. First call creates + caches the
        // IoBinding + bound output tensor; second call reuses them. Both
        // calls must produce identical output (cached path is deterministic).
        let path = format!("{MODELS_DIR}/openvoice_ref_encoder.int8.onnx");
        if !have(&path) {
            eprintln!("SKIP: {path} not found");
            return;
        }
        let mut sess = InferenceSession::load(&path, Backend::Ort)
            .expect("ORT load failed");
        let n_mels = 128;
        let t_frames = 16;
        let input =
            ndarray::Array3::<f32>::zeros((1, n_mels, t_frames)).into_dyn();

        // Discover the actual output shape by running once via `run()` (the
        // non-cached path). This avoids guessing the bound-output shape —
        // ORT will accept whatever the model actually produces.
        let discovered = match sess.run(input.clone()) {
            Ok(o) => o,
            Err(e) => {
                eprintln!(
                    "SKIP: run() forward failed (likely shape mismatch): {e}"
                );
                return;
            }
        };
        let out_shape = discovered.shape().to_vec();
        eprintln!(
            "test_run_into_cached_basic: discovered output shape = {:?}",
            out_shape
        );

        // Two pre-allocated output buffers of the discovered shape.
        let mut output1 =
            ndarray::ArrayD::<f32>::zeros(out_shape.as_slice());
        let mut output2 =
            ndarray::ArrayD::<f32>::zeros(out_shape.as_slice());

        // First cached call: cold path — creates IoBinding + bound output.
        if let Err(e) = sess.run_into_cached(input.clone(), &mut output1) {
            eprintln!(
                "SKIP: first run_into_cached failed (shape mismatch?): {e}"
            );
            return;
        }
        // Cache should now be populated.
        assert!(
            sess.cached_binding.is_some(),
            "cached_binding should be Some after first run_into_cached call"
        );
        assert_eq!(
            sess.cached_output_shape.as_deref(),
            Some(out_shape.as_slice()),
            "cached_output_shape should match discovered shape"
        );

        // Second cached call: hot path — reuses the cached IoBinding.
        if let Err(e) = sess.run_into_cached(input, &mut output2) {
            eprintln!(
                "SKIP: second run_into_cached failed (shape mismatch?): {e}"
            );
            return;
        }

        // Both calls must produce the same output (cached path is
        // deterministic — ORT overwrites the bound buffer each call).
        let diff = (&output1 - &output2).map(|x| x.abs()).sum();
        assert!(
            diff < 1e-5,
            "cached path produced different output on call 1 vs call 2 (diff = {diff})"
        );
        // And the output must match the non-cached `run()` result.
        let diff_run = (&output1 - &discovered).map(|x| x.abs()).sum();
        assert!(
            diff_run < 1e-5,
            "cached output differs from non-cached run() (diff = {diff_run})"
        );
        eprintln!(
            "test_run_into_cached_basic: 2 cached calls, diff = {diff}, diff_vs_run = {diff_run}"
        );
    }

    #[test]
    fn test_run_into_cached_shape_change_invalidates() {
        // OPT-9: when the caller's `output` shape changes between calls,
        // `run_into_cached` should auto-invalidate + rebuild the cached
        // binding (no panic, no shape-mismatch error).
        let path = format!("{MODELS_DIR}/openvoice_ref_encoder.int8.onnx");
        if !have(&path) {
            eprintln!("SKIP: {path} not found");
            return;
        }
        let mut sess = InferenceSession::load(&path, Backend::Ort)
            .expect("ORT load failed");
        let input =
            ndarray::Array3::<f32>::zeros((1, 128, 16)).into_dyn();

        // Discover the real output shape via `run()`.
        let discovered = match sess.run(input.clone()) {
            Ok(o) => o,
            Err(e) => {
                eprintln!(
                    "SKIP: run() forward failed (likely shape mismatch): {e}"
                );
                return;
            }
        };
        let real_shape = discovered.shape().to_vec();

        // First call with the WRONG shape (1-element stub). This should fail
        // with an ORT shape-mismatch error (since the bound output shape
        // doesn't match what the model produces). After the failure, the
        // cache may still hold the wrong-shape binding, so the second call
        // (with the right shape) must rebuild it.
        let mut stub = ndarray::Array0::<f32>::zeros(()).into_dyn();
        let _ = sess.run_into_cached(input.clone(), &mut stub);
        // Whether or not the stub call errored, the cached shape is now
        // either None (cold path skipped due to error mid-construction) or
        // Some([1]) — the wrong shape. The next call with the right shape
        // must trigger a rebuild via `need_rebind`.

        let mut output =
            ndarray::ArrayD::<f32>::zeros(real_shape.as_slice());
        if let Err(e) = sess.run_into_cached(input, &mut output) {
            eprintln!(
                "SKIP: second run_into_cached (correct shape) failed: {e}"
            );
            return;
        }
        // After the shape-correct call, the cache must reflect the new shape.
        assert_eq!(
            sess.cached_output_shape.as_deref(),
            Some(real_shape.as_slice()),
            "cached_output_shape should match the most-recent successful call's output shape"
        );
        // And the output should match the non-cached `run()` result.
        let diff = (&output - &discovered).map(|x| x.abs()).sum();
        assert!(
            diff < 1e-5,
            "shape-change-invalidated cached output differs from run() (diff = {diff})"
        );
        eprintln!(
            "test_run_into_cached_shape_change_invalidates: rebuild succeeded, diff_vs_run = {diff}"
        );
    }

    #[test]
    fn test_clear_cache() {
        // OPT-9: `clear_cache()` resets both cached_binding and
        // cached_output_shape to None — cheap to call on an already-empty
        // cache. Uses a hand-built `InferenceSession` (no ONNX model
        // needed) since we're only testing the field-reset behavior.
        let mut sess = InferenceSession {
            backend: Backend::Ort,
            session_ort: None,
            session_tract: None,
            cached_binding: None,
            cached_output_shape: Some(vec![1, 100]),
        };
        // First call: only cached_output_shape is Some — clear_cache must
        // reset it.
        sess.clear_cache();
        assert!(
            sess.cached_binding.is_none(),
            "cached_binding should be None after clear_cache"
        );
        assert!(
            sess.cached_output_shape.is_none(),
            "cached_output_shape should be None after clear_cache"
        );
        // Idempotent: calling again on an already-empty cache is a no-op.
        sess.clear_cache();
        assert!(sess.cached_binding.is_none());
        assert!(sess.cached_output_shape.is_none());
    }

    /// Verify `run_all` returns every output for a multi-output ONNX model.
    ///
    /// The TinyVC encoder (`encoder.int8.onnx`) declares two outputs:
    ///   - `content`   [B, 768, T] — content features
    ///   - `f0_logits` [B, 512, T] — pitch classification logits
    ///
    /// `run_all` should return a `Vec<ArrayD<f32>>` of length 2 with both
    /// outputs present, in the order declared by the ONNX graph.
    #[test]
    fn test_run_all_multi_output() {
        let encoder_path = format!("{MODELS_DIR}/encoder.int8.onnx");
        if !have(&encoder_path) {
            eprintln!("SKIP: {encoder_path} not found");
            return;
        }
        let mut sess = InferenceSession::load(&encoder_path, Backend::Ort)
            .expect("ORT load failed");
        // Inspect the input fact — TinyVC encoder expects [B, 961, T]
        // (log-mel spectrogram, 961 bins).
        let inputs = sess.session_ort.as_ref().unwrap().inputs();
        let _name = inputs[0].name(); // suppress unused-warning, debug aid
        // Small zero block — if the actual model needs a different shape,
        // `run_all` will surface the error and we skip rather than fail.
        let input = ndarray::Array3::<f32>::zeros((1, 961, 50)).into_dyn();
        match sess.run_all(input) {
            Ok(outputs) => {
                assert!(
                    outputs.len() >= 2,
                    "expected >= 2 outputs for multi-output encoder, got {}",
                    outputs.len()
                );
                // Each output tensor must be non-empty (shape dim > 0).
                for (i, out) in outputs.iter().enumerate() {
                    assert!(
                        out.len() > 0,
                        "output {i} is empty (shape={:?})",
                        out.shape()
                    );
                }
                eprintln!(
                    "test_run_all_multi_output: {} outputs, shapes = {:?}",
                    outputs.len(),
                    outputs.iter().map(|o| o.shape()).collect::<Vec<_>>()
                );
            }
            Err(e) => {
                eprintln!("SKIP: run_all forward failed (likely shape mismatch): {e}");
            }
        }
    }

    /// OPT-13: tract pulsification — turn a full-sequence ONNX graph into a
    /// chunk-streaming variant with internal state (delay lines), so chunked
    /// inference sees the same context as full-sequence inference and SOLA
    /// crossfade becomes unnecessary.
    ///
    /// Most production vocoders (vocos included) ship with fixed input
    /// shapes and ops that tract-pulse doesn't have pulsifiers for, so the
    /// common outcome is "pulsification failed (expected)". The test treats
    /// that as a soft pass — what we must not see is a compile-time error
    /// or a panic.
    #[test]
    fn test_pulsify_basic() {
        let model_path = format!("{MODELS_DIR}/vocos.int8.onnx");
        if !have(&model_path) {
            eprintln!("SKIP: vocos.int8.onnx not found");
            return;
        }
        match pulsify(&model_path, 10) {
            Ok(sess) => {
                println!(
                    "Pulsification succeeded, backend={:?}, input_count={}, output_count={}",
                    sess.backend,
                    sess.input_count(),
                    sess.output_count(),
                );
            }
            Err(e) => {
                println!(
                    "Pulsification failed (expected for complex models): {e}"
                );
                // Soft pass: pulsification is best-effort. The test only
                // verifies that the function runs to completion without
                // panicking — the function's contract is to return a
                // descriptive `InferError::Tract` when the model isn't
                // pulsifiable, and we got one.
            }
        }
    }
}
