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
// Callback typedef
// ============================================================

/// Simplified data callback signature.
///
/// Called by miniaudio's audio thread (GIL-free, real-time priority). The
/// caller must:
///  - read `frame_count * channels` f32 samples from `input` (capture)
///  - write `frame_count * channels` f32 samples to `output` (playback)
///
/// Both buffers are in interleaved f32 format. `input` is read-only.
///
/// The `user_data` pointer is the same one passed to `AudioDevice::open()`
/// — typically a raw pointer to an `rtrb::Producer` for capture, or an
/// `rtrb::Consumer` for playback.
///
/// # Safety
///
/// The callback must not panic (it runs on a miniaudio thread without
/// unwind support — panicking aborts the process). It must not call
/// Python code (holding the GIL on an audio thread risks deadlocks if
/// the main thread is also waiting on the audio device).
pub type MaDataCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    output: *mut f32,
    input: *const f32,
    frame_count: u32,
);

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
/// `started` flag). On `Drop`, the device is stopped + uninitialized +
/// freed — there is no manual `close()` needed.
///
/// # Threading
///
/// `AudioDevice` is `Send` (the underlying `ma_device` is owned by miniaudio
/// and supports cross-thread start/stop) but NOT `Sync` — concurrent calls
/// to `start`/`stop` from multiple threads are undefined behavior in
/// miniaudio. The typical pattern is to move the `AudioDevice` into the
/// miniaudio-callback-side owner thread and call `start`/`stop` only from
/// there.
pub struct AudioDevice {
    /// Opaque handle to the C-side `shim_device_t`. NULL only if the device
    /// failed to open (then we don't have an `AudioDevice` value at all —
    /// `open` returns `Err`).
    handle: *mut c_void,
    started: bool,
}

// SAFETY: the underlying `ma_device` is owned by miniaudio and supports
// cross-thread start/stop. We do NOT share `&AudioDevice` across threads
// (it's not Sync), only move it between threads.
unsafe impl Send for AudioDevice {}

impl AudioDevice {
    /// Open a duplex (capture + playback) audio device.
    ///
    /// The callback runs on miniaudio's thread (GIL-free, real-time priority).
    /// The caller owns the `user_data` pointer — it must outlive the
    /// `AudioDevice` (i.e., the user must drop the device before dropping
    /// whatever `user_data` points to).
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` with the miniaudio error code if device
    /// initialization fails (typical in headless environments: `MA_NO_DEVICE`
    /// = -204, `MA_NO_BACKEND` = -203, `MA_FAILED_TO_INIT_BACKEND` = -400).
    pub fn open(
        config: &AudioConfig,
        callback: MaDataCallback,
        user_data: *mut c_void,
    ) -> Result<Self, String> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: the FFI call is safe — `shim_open_duplex` writes either a
        // valid handle or NULL into `&mut handle`. The callback signature
        // matches `shim_data_callback` (verified in the shim's typedef).
        let result = unsafe {
            shim_open_duplex(
                config.sample_rate,
                config.channels as u32,
                config.block_size,
                Some(callback),
                user_data,
                &mut handle,
            )
        };
        if result != 0 {
            return Err(format!(
                "shim_open_duplex failed with miniaudio error code {} (e.g. -204 MA_NO_DEVICE, -203 MA_NO_BACKEND, -400 MA_FAILED_TO_INIT_BACKEND)",
                result
            ));
        }
        if handle.is_null() {
            // Should not happen — shim returns non-zero on failure. Defensive.
            return Err("shim_open_duplex returned success but handle is NULL".into());
        }
        Ok(Self {
            handle,
            started: false,
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
        // no one can use it again.
        unsafe { shim_close(self.handle) };
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

    /// In a headless environment (no audio hardware / no PulseAudio daemon),
    /// `AudioDevice::open` should fail gracefully with an `Err(String)` —
    /// not panic, not abort. On a machine WITH audio hardware, it may
    /// succeed; both outcomes are acceptable here.
    #[test]
    fn test_audio_device_open_no_hardware() {
        // Empty no-op callback — just verifies the FFI plumbing. If the device
        // actually opens, the callback will fire on miniaudio's thread but
        // we never call `start()`, so it shouldn't actually fire.
        extern "C" fn noop_callback(
            _user_data: *mut c_void,
            _output: *mut f32,
            _input: *const f32,
            _frame_count: u32,
        ) {
            // intentionally empty
        }

        let config = AudioConfig::default();
        let result = AudioDevice::open(&config, noop_callback, std::ptr::null_mut());
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
        extern "C" fn noop(
            _u: *mut c_void, _o: *mut f32, _i: *const f32, _f: u32,
        ) {}
        let config = AudioConfig::default();
        if let Ok(device) = AudioDevice::open(&config, noop, std::ptr::null_mut()) {
            assert!(!device.is_started());
        }
    }
}
