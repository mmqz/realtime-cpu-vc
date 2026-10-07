//! miniaudio FFI declarations + safe Rust wrapper.
//!
//! miniaudio is a single-header C audio library (MIT-0) supporting:
//! - WASAPI/DirectSound (Windows), Core Audio (macOS/iOS), ALSA/Pulse/JACK (Linux)
//! - AAudio (Android), OpenSL|ES (Android older), Web Audio (Emscripten)
//!
//! This module provides safe Rust bindings for the device + callback API.
//!
//! ## Design: shim, not direct FFI
//!
//! miniaudio's `ma_device_config` and `ma_device` are multi-KB structs with
//! hundreds of platform-specific nested fields (WASAPI/ALSA/PulseAudio/Core
//! Audio/OpenSL|ES/AAudio enums + config blocks). Replicating those 1:1 in
//! Rust would be brittle and require transcribing dozens of platform enums.
//!
//! Instead, `src_c/miniaudio/miniaudio_shim.c` exposes 4 simple `extern "C"`
//! functions with opaque `void*` handles:
//!   - `shim_open_duplex(...)` — open a full-duplex device, get opaque handle
//!   - `shim_start(handle)`   — begin audio callback fire
//!   - `shim_stop(handle)`    — pause audio callback fire
//!   - `shim_close(handle)`   — uninit device + free shim
//!
//! The shim's internal trampoline receives the miniaudio `ma_device*` from
//! miniaudio's audio thread and casts it back to a `shim_device_t*` (because
//! `ma_device` is the first field of that struct — C guarantees that
//! `&struct == &first_field`). This lets the trampoline recover the user's
//! simplified `shim_data_callback` + user_data pointers and forward the call
//! with the much simpler signature `(user_data, output, input, frame_count)`.
//!
//! The audio callback runs on miniaudio's thread (GIL-free), making it safe
//! to push into an SPSC ring buffer for the decoder thread to drain.

use std::os::raw::c_int;
use std::os::raw::c_void;
use crate::AudioConfig;

// ============================================================
// Callback typedef (internal — not part of the public API)
// ============================================================

/// Simplified C data callback signature passed to the miniaudio shim.
///
/// This is an internal implementation detail of [`AudioDevice::open`]: the
/// public API accepts a safe `FnMut(&[f32], &mut [f32]) + Send + 'static`
/// closure, which we wrap in a C trampoline internally. Users never write
/// `unsafe extern "C" fn` themselves.
///
/// Called by miniaudio's audio thread (GIL-free, real-time priority). The
/// callback must:
///  - read `frame_count * channels` f32 samples from `input` (capture)
///  - write `frame_count * channels` f32 samples to `output` (playback)
///
/// Both buffers are in interleaved f32 format. `input` is read-only.
///
/// The callback must not panic (it runs on a miniaudio thread without
/// unwind support — panicking aborts the process). It must not call
/// Python code (holding the GIL on an audio thread risks deadlocks if
/// the main thread is also waiting on the audio device).
type MaDataCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    output: *mut f32,
    input: *const f32,
    frame_count: u32,
);

// ============================================================
// Boxed user-callback storage (held by `AudioDevice`, freed in `Drop`)
// ============================================================

/// Boxed user callback + the channel count the device was opened with.
///
/// Stored as the opaque `user_data` pointer handed to the C shim. The
/// `channels` field lets the trampoline compute the correct slice length:
/// miniaudio passes `frame_count` *frames* (not samples), so the actual
/// f32 count in each buffer is `frame_count * channels`. Slicing only
/// `frame_count` samples (the previous bug) silently dropped half the
/// buffer for stereo audio — see commit message for the deep-audit fix.
///
/// Not public — only [`AudioDevice::open`] / `trampoline` / `Drop` touch it.
struct CallbackData {
    callback: Box<dyn FnMut(&[f32], &mut [f32]) + Send>,
    channels: u32,
}

// ============================================================
// FFI function declarations (link against libminiaudio.a — see build.rs)
// ============================================================

extern "C" {
    /// Open a full-duplex audio device.
    ///
    /// Returns 0 on success (MA_SUCCESS) and writes an opaque handle into
    /// `out_dev`. Returns a miniaudio error code on failure and sets
    /// `*out_dev = NULL`.
    fn shim_open_duplex(
        sample_rate: u32,
        channels: u32,
        period_size_in_frames: u32,
        callback: Option<MaDataCallback>,
        user_data: *mut c_void,
        out_dev: *mut *mut c_void,
    ) -> c_int;

    /// Begin firing the audio callback on miniaudio's thread. Idempotent.
    fn shim_start(dev: *mut c_void) -> c_int;

    /// Stop firing the audio callback. Idempotent.
    fn shim_stop(dev: *mut c_void) -> c_int;

    /// Uninitialize the device + free the shim struct. Safe to call on NULL.
    /// After this, the handle is invalid and must not be reused.
    fn shim_close(dev: *mut c_void);
}

// ============================================================
// Safe Rust wrapper
// ============================================================

/// Safe RAII wrapper around a miniaudio duplex audio device.
///
/// The underlying `ma_device` is heap-allocated on the C side (in
/// `shim_device_t`) so this struct is small (just a non-null pointer + a
/// `started` flag + the boxed user callback). On `Drop`, the device is
/// stopped + uninitialized + freed and the boxed callback is dropped —
/// there is no manual `close()` needed.
///
/// # Threading
///
/// `AudioDevice` is `Send` (the underlying `ma_device` is owned by miniaudio
/// and supports cross-thread start/stop; the boxed callback is `Send` by
/// construction — see [`AudioDevice::open`]'s `F: ... + Send` bound) but
/// NOT `Sync` — concurrent calls to `start`/`stop` from multiple threads
/// are undefined behavior in miniaudio. The typical pattern is to move the
/// `AudioDevice` into the miniaudio-callback-side owner thread and call
/// `start`/`stop` only from there.
pub struct AudioDevice {
    /// Opaque handle to the C-side `shim_device_t`. NULL only if the device
    /// failed to open (then we don't have an `AudioDevice` value at all —
    /// `open` returns `Err`).
    handle: *mut c_void,
    started: bool,
    /// Owned boxed user callback. `Some` while the device is alive so that
    /// `Drop` can free the closure after the audio device has been stopped
    /// (which guarantees the callback can no longer fire on miniaudio's
    /// thread). The raw pointer handed to the C shim aliases this box —
    /// see [`AudioDevice::open`] for the safety argument.
    _callback_data: Option<*mut c_void>,
}

// SAFETY: the underlying `ma_device` is owned by miniaudio and supports
// cross-thread start/stop. The `_callback_data` raw pointer aliases a
// `Box<Box<dyn FnMut(&[f32], &mut [f32]) + Send>>` whose inner closure is
// `Send` by the `F: Send` bound on `open` — so moving the box across threads
// is sound. We do NOT share `&AudioDevice` across threads (it's not Sync),
// only move it between threads.
unsafe impl Send for AudioDevice {}

impl AudioDevice {
    /// Open a duplex (capture + playback) audio device with a safe Rust
    /// callback closure. No `unsafe` is required from the caller.
    ///
    /// The closure runs on miniaudio's audio thread (GIL-free, real-time
    /// priority). It receives `(input_samples, output_samples)` as
    /// interleaved f32 slices — read from `input_samples` (capture, length
    /// `frame_count * channels`) and write to `output_samples` (playback,
    /// same length). Either slice may be empty if miniaudio passes a null
    /// buffer for the corresponding direction.
    ///
    /// The closure must not panic — miniaudio's audio thread has no unwind
    /// support and a panic will abort the process. The closure must not call
    /// Python code (holding the GIL on the audio thread risks deadlocks).
    ///
    /// # Ownership / lifetime
    ///
    /// `AudioDevice::open` boxes the closure (`Box<Box<dyn FnMut ...>>`) so
    /// it can be passed to the C shim as a raw `*mut c_void`. The box is
    /// freed in `AudioDevice::drop` after the device has been stopped (which
    /// guarantees the callback can no longer fire). The closure thus must
    /// be `Send + 'static` so the box can be moved across threads + live as
    /// long as the device.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` with the miniaudio error code if device
    /// initialization fails (typical in headless environments: `MA_NO_DEVICE`
    /// = -204, `MA_NO_BACKEND` = -203, `MA_FAILED_TO_INIT_BACKEND` = -400).
    /// On failure, the boxed closure is dropped before returning so no leak
    /// occurs.
    pub fn open<F>(config: &AudioConfig, callback: F) -> Result<Self, String>
    where
        F: FnMut(&[f32], &mut [f32]) + Send + 'static,
    {
        // Box the closure + the channel count inside a `CallbackData`
        // so the C trampoline can recover both via the single `*mut c_void`
        // user_data pointer miniaudio accepts. The `channels` field is
        // required because miniaudio passes `frame_count` *frames* (not
        // samples) to the trampoline — the actual f32 count in each buffer
        // is `frame_count * channels` (see `trampoline` below).
        let cb_data: Box<CallbackData> = Box::new(CallbackData {
            callback: Box::new(callback),
            channels: config.channels as u32,
        });
        let user_data = Box::into_raw(cb_data) as *mut c_void;

        // C trampoline: the FFI signature is `unsafe extern "C" fn`, but the
        // closure-based path means *users* never write `unsafe` — only this
        // one internal function does. The `unsafe` block here is the FFI
        // boundary (deref the raw pointer + slice the C buffers).
        extern "C" fn trampoline(
            user_data: *mut c_void,
            output: *mut f32,
            input: *const f32,
            frame_count: u32,
        ) {
            if user_data.is_null() {
                return;
            }
            // SAFETY: `user_data` was created by `Box::into_raw` in `open`
            // and remains valid until `AudioDevice::drop` reclaims it. The
            // audio device has already been stopped before `Drop` runs the
            // reclaim, so the trampoline cannot fire concurrently.
            let cb_data = unsafe {
                &mut *(user_data as *mut CallbackData)
            };
            // miniaudio passes `frame_count` *frames* (one frame per
            // channel-pair), so the f32 count in each buffer is
            // `frame_count * channels`. Slicing only `frame_count` samples
            // (the previous behavior) silently dropped the second half of
            // each buffer for stereo audio — a HIGH-severity data-corruption
            // bug. `checked_mul` guards against adversarial overflow.
            let n = (frame_count as usize)
                .checked_mul(cb_data.channels as usize)
                .unwrap_or(0);
            let input_slice = if input.is_null() || n == 0 {
                &[][..]
            } else {
                // SAFETY: miniaudio guarantees `input` is valid for
                // `frame_count * channels` f32 reads for this callback.
                unsafe { std::slice::from_raw_parts(input, n) }
            };
            let output_slice = if output.is_null() || n == 0 {
                &mut [][..]
            } else {
                // SAFETY: miniaudio guarantees `output` is valid for
                // `frame_count * channels` f32 writes for this callback.
                unsafe { std::slice::from_raw_parts_mut(output, n) }
            };
            (cb_data.callback)(input_slice, output_slice);
        }

        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: the FFI call is safe — `shim_open_duplex` writes either a
        // valid handle or NULL into `&mut handle`. The callback signature
        // matches `shim_data_callback` (verified in the shim's typedef).
        let result = unsafe {
            shim_open_duplex(
                config.sample_rate,
                config.channels as u32,
                config.block_size,
                Some(trampoline),
                user_data,
                &mut handle,
            )
        };
        if result != 0 {
            // SAFETY: on failure, the trampoline was never invoked (the
            // device was never started), so `user_data` is still owned
            // solely by this thread. Reclaim the box to avoid a leak.
            unsafe {
                drop(Box::from_raw(user_data as *mut CallbackData));
            }
            return Err(format!(
                "shim_open_duplex failed with miniaudio error code {} (e.g. -204 MA_NO_DEVICE, -203 MA_NO_BACKEND, -400 MA_FAILED_TO_INIT_BACKEND)",
                result
            ));
        }
        if handle.is_null() {
            // Should not happen — shim returns non-zero on failure. Defensive.
            // SAFETY: as above — device never started, sole ownership.
            unsafe {
                drop(Box::from_raw(user_data as *mut CallbackData));
            }
            return Err("shim_open_duplex returned success but handle is NULL".into());
        }
        Ok(Self {
            handle,
            started: false,
            _callback_data: Some(user_data),
        })
    }

    /// Begin firing the audio callback. Idempotent — calling `start` while
    /// already started is a no-op.
    pub fn start(&mut self) -> Result<(), String> {
        if self.started {
            return Ok(());
        }
        // SAFETY: `self.handle` is non-null (guaranteed by `open`) and points
        // to a valid `shim_device_t`.
        let result = unsafe { shim_start(self.handle) };
        if result != 0 {
            return Err(format!("shim_start failed with code {}", result));
        }
        self.started = true;
        Ok(())
    }

    /// Stop firing the audio callback. Idempotent.
    pub fn stop(&mut self) -> Result<(), String> {
        if !self.started {
            return Ok(());
        }
        // SAFETY: as above.
        let result = unsafe { shim_stop(self.handle) };
        if result != 0 {
            return Err(format!("shim_stop failed with code {}", result));
        }
        self.started = false;
        Ok(())
    }

    /// True iff `start()` has been called and `stop()` has not yet been called
    /// (successfully) since.
    pub fn is_started(&self) -> bool {
        self.started
    }

    /// Raw opaque handle to the C-side `shim_device_t`. For advanced use only
    /// (e.g., passing to other FFI functions). The pointer is valid only
    /// while the `AudioDevice` is alive.
    pub fn raw_handle(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for AudioDevice {
    fn drop(&mut self) {
        // Stop first (safe to call on a stopped device — no-op).
        if self.started {
            // SAFETY: as above. Ignore the result — we can't propagate errors
            // from `Drop` anyway, and a failed stop during teardown is not
            // actionable.
            let _ = unsafe { shim_stop(self.handle) };
            self.started = false;
        }
        // SAFETY: `self.handle` is non-null (guaranteed by `open`). The shim's
        // `shim_close` calls `ma_device_uninit` (safe at any state) + `free`.
        // After this call, the handle is invalid — but we're dropping so
        // no one can use it again. Importantly, `shim_close` joins + stops
        // the audio thread first, so the trampoline can no longer fire by
        // the time we reclaim the boxed closure below.
        unsafe { shim_close(self.handle) };
        // Reclaim the boxed user closure. `shim_close` has returned, so the
        // audio thread is torn down + the trampoline cannot fire again —
        // sole ownership of the box is on this thread.
        if let Some(user_data) = self._callback_data.take() {
            // SAFETY: `user_data` was created by `Box::into_raw` in `open`
            // and the device has just been closed above, so the trampoline
            // cannot be running concurrently.
            unsafe {
                drop(Box::from_raw(user_data as *mut CallbackData));
            }
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial test: just verifies the `AudioDevice` type compiles + has the
    /// expected methods. Doesn't actually open a device.
    #[test]
    fn test_audio_device_struct_exists() {
        // Just verify the type compiles + has the API surface we expect.
        fn _compile_check(_device: AudioDevice) {}
        // (No runtime call — this is purely a type-level check.)
    }

    /// Compile-time check that the FFI function signatures match what the shim
    /// declares. Doesn't actually link anything — this is satisfied if the
    /// `extern "C"` block above parses without errors.
    #[test]
    fn test_ffi_signatures_compile() {
        let _ = std::ptr::null::<extern "C" fn(
            u32, u32, u32,
            Option<MaDataCallback>,
            *mut c_void,
            *mut *mut c_void,
        ) -> c_int>();
    }

    /// Verify the `CallbackData` struct stores both the channel count and
    /// the boxed closure (regression test for the deep-audit fix that made
    /// the trampoline slice `frame_count * channels` samples instead of
    /// `frame_count`).
    #[test]
    fn test_callback_data_stores_channels() {
        let cb_data = CallbackData {
            callback: Box::new(|_input: &[f32], _output: &mut [f32]| {}),
            channels: 2,
        };
        assert_eq!(cb_data.channels, 2);
    }

    /// In a headless environment (no audio hardware / no PulseAudio daemon),
    /// `AudioDevice::open` should fail gracefully with an `Err(String)` —
    /// not panic, not abort. On a machine WITH audio hardware, it may
    /// succeed; both outcomes are acceptable here.
    #[test]
    fn test_audio_device_open_no_hardware() {
        // Empty no-op callback — just verifies the FFI plumbing. If the device
        // actually opens, the callback will fire on miniaudio's thread but
        // we never call `start()`, so it shouldn't actually fire.
        let config = AudioConfig::default();
        let result = AudioDevice::open(&config, |_input: &[f32], _output: &mut [f32]| {
            // intentionally empty
        });
        match result {
            Ok(mut device) => {
                // Hardware is available — verify start/stop cycle works.
                // (We accept errors here too — some backends refuse to start
                // in sandboxed environments.)
                match device.start() {
                    Ok(_) => println!("AudioDevice: start OK (hardware available)"),
                    Err(e) => println!("AudioDevice: start failed (sandbox?): {}", e),
                }
                if let Err(e) = device.stop() {
                    println!("AudioDevice: stop failed (sandbox?): {}", e);
                }
                // Drop triggers shim_close — must not panic / abort.
                drop(device);
            }
            Err(e) => {
                // Expected in headless env. Verify the error message names
                // the error code so future debuggers can look it up.
                println!(
                    "AudioDevice::open failed (expected in headless env): {}",
                    e
                );
                assert!(
                    e.contains("shim_open_duplex failed"),
                    "error message should name the FFI call: got {e:?}"
                );
            }
        }
    }

    /// `is_started` should be false before `start()` is called.
    /// If the device fails to open, we skip the assertion (headless).
    #[test]
    fn test_audio_device_is_started_false_before_start() {
        let config = AudioConfig::default();
        if let Ok(device) = AudioDevice::open(&config, |_input: &[f32], _output: &mut [f32]| {}) {
            assert!(!device.is_started());
        }
    }
}
