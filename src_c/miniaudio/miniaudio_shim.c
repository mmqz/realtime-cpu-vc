// miniaudio_shim.c - Thin C wrapper around miniaudio's ma_device API.
//
// Why this shim exists:
// - miniaudio's `ma_device_config` and `ma_device` are massive (~multi-KB)
//   structs with hundreds of platform-specific nested fields (WASAPI/ALSA/
//   PulseAudio/CoreAudio/OpenSL|ES/AAudio enums and config blocks). Replicating
//   those 1:1 in Rust would be brittle and require transcribing dozens of
//   platform enums.
// - Instead, this shim exposes 4 simple `extern "C"` functions with opaque
//   `void*` handles. The Rust side never touches the miniaudio structs
//   directly.
//
// The shim defines `shim_device_t` with `ma_device` as its FIRST field. This
// makes `&shim->device == &shim`, so the internal trampoline can cast the
// `ma_device*` it receives from miniaudio back into a `shim_device_t*` and
// recover the user's simplified callback + user_data pointer.
//
// All public functions use the C ABI (no name mangling) so they can be linked
// from Rust via `extern "C"` declarations.

#include "miniaudio.h"  // declarations only — implementation lives in miniaudio.c
#include <stdint.h>
#include <stdlib.h>

// Forward-declare the shim_device_t struct so the callback typedef can refer
// to `void*` for user_data (we never pass the device itself into the user
// callback — Rust holds the device handle in `AudioDevice`).
typedef struct shim_device_t shim_device_t;

// Simplified callback signature — user_data + f32 buffers + frame count.
// This is what Rust will pass in (after the GIL is released). It does NOT
// receive the `ma_device*` because Rust never needs to inspect that opaque
// handle (it already has the shim handle in `AudioDevice`).
typedef void (*shim_data_callback)(void* user_data,
                                   float* output,
                                   const float* input,
                                   uint32_t frame_count);

// Full struct definition. `ma_device` is the FIRST field, so casting
// `ma_device*` -> `shim_device_t*` is valid (C guarantees &struct == &first).
struct shim_device_t {
    ma_device           device;     // MUST be first
    void*               user_data;
    shim_data_callback  callback;
};

// Internal trampoline: converts miniaudio's callback signature into the
// simplified `shim_data_callback` signature. miniaudio calls this from its
// audio thread (GIL-free).
static void shim_trampoline(ma_device* pDevice,
                            void* pOutput,
                            const void* pInput,
                            ma_uint32 frameCount) {
    // ma_device is the first field of shim_device_t, so this cast is valid.
    shim_device_t* self = (shim_device_t*)pDevice;
    if (self != NULL && self->callback != NULL) {
        self->callback(self->user_data,
                       (float*)pOutput,
                       (const float*)pInput,
                       (uint32_t)frameCount);
    }
}

// Open a full-duplex (capture + playback) audio device.
//
// `sample_rate`            in Hz (e.g. 24000, 48000)
// `channels`               1 = mono, 2 = stereo
// `period_size_in_frames`  callback block size (e.g. 1920 @ 24kHz = 80ms)
// `callback`              simplified data callback (may be NULL = silence)
// `user_data`             passed back to the callback (may be NULL)
// `out_dev`               receives the opaque device handle on success
//
// Returns: 0 (MA_SUCCESS) on success, miniaudio error code otherwise.
// On failure, `*out_dev` is set to NULL and no resources are leaked.
int shim_open_duplex(uint32_t sample_rate,
                     uint32_t channels,
                     uint32_t period_size_in_frames,
                     shim_data_callback callback,
                     void* user_data,
                     shim_device_t** out_dev) {
    if (out_dev == NULL) {
        return -2;  // MA_INVALID_ARGS
    }
    *out_dev = NULL;

    shim_device_t* self = (shim_device_t*)calloc(1, sizeof(shim_device_t));
    if (self == NULL) {
        return -4;  // MA_OUT_OF_MEMORY
    }
    self->callback  = callback;
    self->user_data = user_data;

    // ma_device_config_init takes ONLY a device type — all other fields are
    // filled in directly on the returned struct. We then patch the few fields
    // we actually care about (sample rate, period size, f32 format, channels,
    // callback + user data). All other fields keep miniaudio's safe defaults.
    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    config.sampleRate           = sample_rate;
    config.periodSizeInFrames  = period_size_in_frames;
    config.playback.format      = ma_format_f32;
    config.playback.channels    = channels;
    config.capture.format       = ma_format_f32;
    config.capture.channels     = channels;
    config.dataCallback         = shim_trampoline;
    // Note: we set pUserData to `self` so the trampoline can recover the user's
    // callback pointer + user_data. The user's actual user_data is stored in
    // `self->user_data` and surfaced through the trampoline.
    config.pUserData            = self;

    // ma_device_init_ex with NULL backend list + 0 backend count uses the
    // default context (tries platform backends in priority order: WASAPI on
    // Windows, CoreAudio on macOS, PulseAudio -> ALSA -> JACK on Linux, etc.).
    ma_result result = ma_device_init_ex(/*backends=*/NULL,
                                         /*backendCount=*/0,
                                         /*pContextConfig=*/NULL,
                                         &config,
                                         &self->device);
    if (result != MA_SUCCESS) {
        // On init failure, miniaudio internally calls its own uninit path,
        // so we must NOT call ma_device_uninit here — just free the shim.
        free(self);
        return (int)result;
    }

    *out_dev = self;
    return (int)MA_SUCCESS;
}

int shim_start(shim_device_t* dev) {
    if (dev == NULL) return -2;  // MA_INVALID_ARGS
    return (int)ma_device_start(&dev->device);
}

int shim_stop(shim_device_t* dev) {
    if (dev == NULL) return -2;  // MA_INVALID_ARGS
    return (int)ma_device_stop(&dev->device);
}

void shim_close(shim_device_t* dev) {
    if (dev == NULL) return;
    ma_device_uninit(&dev->device);
    free(dev);
}
