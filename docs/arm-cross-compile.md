# ARM Cross-Compile Guide

This crate stack targets both x86_64 (dev / CI) and aarch64 (Pi 5,
RK3588, Apple Silicon Linux VM). The Rust code itself is fully portable
to ARM64; only the C cross-linker for `miniaudio` needs extra setup on
the dev host.

## Status (verified in this repo)

| Step | x86_64 (host) | aarch64 (cross) |
|------|---------------|-----------------|
| `cargo check` | ✅ passes | ✅ passes (Rust code ARM-compatible) |
| `cargo build`  | ✅ passes | ⚠️ requires `gcc-aarch64-linux-gnu` for miniaudio |
| `cargo test`   | ✅ passes | n/a (run on device) |

The `cargo check --target aarch64-unknown-linux-gnu -p vc-native` step
verifies the full Rust dependency tree (rustfft, realfft, num-complex,
rtrb, wide, …) compiles cleanly for ARM64. The only blocker for a full
`cargo build` is the missing C cross-compiler for the `cc`-crate build
script that compiles `miniaudio.c`.

## Prerequisites

```bash
# 1. Install Rust target
rustup target add aarch64-unknown-linux-gnu

# 2. Install C cross-compiler (needed by miniaudio build.rs)
sudo apt-get update && sudo apt-get install -y gcc-aarch64-linux-gnu

# 3. (Optional) Docker-based alternative — useful if you can't apt install
cargo install cross
```

## Build

```bash
cd prototype/src_rust

# Check only — verifies Rust code is ARM-compatible, no C linker needed.
# (Use this in CI if the cross-toolchain isn't installed.)
cargo check --target aarch64-unknown-linux-gnu -p vc-native

# Full build — needs the C cross-linker for miniaudio.
# .cargo/config.toml already wires `aarch64-linux-gnu-gcc` as the linker.
cargo build --target aarch64-unknown-linux-gnu -p vc-native --release

# Or with `cross` (Docker-based, fully reproducible):
cross build --target aarch64-unknown-linux-gnu -p vc-native --release
```

### Skipping miniaudio for an inference-only ARM build

If you don't need audio I/O on the ARM target, you can move the
vendored miniaudio aside and build.rs will emit a warning and skip the
C compilation — vc-native still compiles as inference-only:

```bash
# Temporarily hide miniaudio.c so build.rs takes its "skip C" path
mv src_c/miniaudio/miniaudio.c src_c/miniaudio/miniaudio.c.skip
cargo build --target aarch64-unknown-linux-gnu -p vc-native --release
mv src_c/miniaudio/miniaudio.c.skip src_c/miniaudio/miniaudio.c
```

## Deploy to Pi 5 / RK3588

```bash
# Copy the static lib (and/or .so if changed) to the ARM device
scp target/aarch64-unknown-linux-gnu/release/libvc_native.a pi@device:~/

# On the ARM device, link against the local aarch64 libonnxruntime.so
# (download the aarch64 build from
#  https://github.com/microsoft/onnxruntime/releases — pick the
#  `onnxruntime-linux-aarch64-*.tgz` artifact).
```

## `.cargo/config.toml` notes

The repo's `.cargo/config.toml` was previously:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

That's correct for x86_64 host builds, but when cross-compiling to
aarch64 the `target-cpu=native` flag resolves to the *host* CPU's
(x86) feature set and LLVM emits ~80 "not a recognized feature for this
target" warnings on every `cargo` invocation. It now scopes the flag to
`[target.x86_64-unknown-linux-gnu]` and adds an explicit
`[target.aarch64-unknown-linux-gnu]` section pointing at
`aarch64-linux-gnu-gcc`.

## Known Issues

- **vc-python (PyO3)** — can't cross-compile easily; PyO3 needs Python
  dev headers for the target arch (`aarch64` Python). Build natively on
  the device, or use `pyo3-build-config` overrides with a sysroot.
- **vc-ort** — needs `libonnxruntime.so` for aarch64 (download the
  aarch64 build from the ORT releases page). The build script looks for
  it via `ORT_LIB_DIR`.
- **miniaudio** — needs the C cross-compiler (`gcc-aarch64-linux-gnu`).
  Without it, `cargo check` still works (build.rs takes its "skip C"
  fallback) but `cargo build` fails in the `cc`-crate build script.
- **`target-cpu=native`** — must NOT be set globally; it must be scoped
  to `[target.x86_64-unknown-linux-gnu]` or it pollutes the aarch64
  invocation with x86 feature flags (warnings, no functional impact).
