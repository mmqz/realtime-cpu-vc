# ARM Cross-Compile Guide

This crate stack targets both x86_64 (dev / CI) and aarch64 (Pi 5,
RK3588, Apple Silicon Linux VM). The Rust code itself is fully portable
to ARM64; only the C cross-linker for `miniaudio` needs extra setup on
the dev host.

## Status (verified in this repo)

| Step | x86_64 (host) | aarch64 (cross) |
|------|---------------|-----------------|
| `cargo check` | ✅ passes | ✅ passes (no C cross-compiler needed — OPT-17) |
| `cargo build`  | ✅ passes | ⚠️ requires `gcc-aarch64-linux-gnu` for miniaudio |
| `cargo test`   | ✅ passes | n/a (run on device) |

The `cargo check --target aarch64-unknown-linux-gnu -p vc-native` step
verifies the full Rust dependency tree (rustfft, realfft, num-complex,
rtrb, wide, …) compiles cleanly for ARM64. As of OPT-17 the build.rs
detects cross-compilation and emits `cargo:rustc-cfg=no_miniaudio` when
no C cross-linker is available, so `cargo check` works on a bare dev
host. The only blocker for a full `cargo build` is the missing C
cross-compiler for the `cc`-crate build script that compiles
`miniaudio.c`.

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

As of OPT-17, `cargo check --target aarch64` skips miniaudio
automatically when `TARGET != HOST` and no C cross-compiler is found —
no manual intervention needed. The build.rs probes for:

1. `CC_<target>` env var (e.g. `CC_aarch64_unknown_linux_gnu`)
2. `TARGET_CC` env var
3. The `aarch64-linux-gnu-gcc` / `arm-linux-gnueabihf-gcc` binary on PATH

If none are present, it emits `cargo:rustc-cfg=no_miniaudio` and exits
without invoking the `cc` crate, so the build script can't fail on a
missing C toolchain. The Rust side still compiles — `extern "C"`
declarations in `miniaudio_ffi.rs` are unresolved only at link time,
which doesn't affect `cargo check` (which doesn't link).

If you also want to skip miniaudio for a *full* `cargo build` (so the
linker doesn't fail on unresolved symbols), you can still move the
vendored file aside:

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

## vc-python (PyO3) Cross-Compile

PyO3 cross-compilation requires a Python sysroot for the target
architecture. This is more complex than pure Rust crates (PyO3 embeds
the build-host Python's interpreter ABI into the extension module).

### Option A: Build on-device (recommended)

The simplest path — no cross-toolchain hassles, the resulting wheel
matches the device's native Python ABI.

```bash
# On the ARM device (Pi 5 / RK3588) running a 64-bit OS:
# 1. Install Rust + Python dev headers
sudo apt update && sudo apt install -y python3-dev cargo rustc
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 2. Clone + build from source
git clone <repo> && cd <repo>/prototype/src_rust/vc-python
maturin build --release
pip install --user target/wheels/*.whl
```

### Option B: Cross-compile with Docker (maturin)

Maturin publishes official Docker images with the right target Python
already inside. This is the easiest way to produce a manylinux wheel
from an x86 host.

```bash
# From the repo root (where src_rust/vc-python/Cargo.toml lives):
docker run --rm -v "$(pwd):/io" \
    ghcr.io/pyo3/maturin:main-aarch64 \
    build --release --out /io/dist
# Produces: dist/vc_python-<ver>-cp<py>-cp<py>-linux_aarch64.whl
```

### Option C: Cross-compile with `cross` (requires Docker)

For a raw `.so` (not a wheel), `cross` is a thin Docker wrapper around
`cargo build`. You'll need to extend the image with Python dev headers
for the target arch — see `Cross.toml`.

```bash
cargo install cross
cross build --target aarch64-unknown-linux-gnu -p vc-python --release
# Note: needs Python dev headers (libpython3-dev:arm64) inside the
# Docker image. The default `cross` image does NOT include them —
# build a custom image extending ghcr.io/cross-rs/aarch64-unknown-linux-gnu
# with `apt-get install -y python3-dev:arm64` + multiarch setup.
```

### Known Limitations

- PyO3 `extension-module` feature pins the wheel to the build Python's
  ABI (CPython 3.11 wheels won't load on 3.12, etc.).
- For broader compatibility, use the `abi3-py310` feature in
  `vc-python/Cargo.toml` — abi3 wheels load on any CPython ≥ 3.10:
  ```toml
  pyo3 = { version = "0.22", features = ["extension-module", "abi3-py310"] }
  ```
- ONNXRuntime aarch64 binary: download the
  `onnxruntime-linux-aarch64-*.tgz` artifact from
  <https://github.com/microsoft/onnxruntime/releases> and place its
  `libonnxruntime.so` somewhere `LD_LIBRARY_PATH` (or the rpath) can
  find it on the device.
- miniaudio for vc-python's audio I/O path is *not* needed for the
  Python wheel itself — vc-python calls vc-native via PyO3, and vc-native
  is the crate that links miniaudio. So building vc-python with maturin
  will *not* need the C cross-compiler for miniaudio unless you also
  build `-p vc-native` for aarch64 (see OPT-17 above).

## Known Issues

- **vc-python (PyO3)** — see the dedicated section above; the easiest
  path is on-device builds via `maturin build`. The Docker option
  (`ghcr.io/pyo3/maturin:main-aarch64`) produces a manylinux wheel from
  an x86 host.
- **vc-ort** — needs `libonnxruntime.so` for aarch64 (download the
  aarch64 build from the ORT releases page). The build script looks for
  it via `ORT_LIB_DIR`.
- **miniaudio** — needs the C cross-compiler (`gcc-aarch64-linux-gnu`)
  for a full `cargo build` link. As of OPT-17, `cargo check` no longer
  needs it: build.rs detects `TARGET != HOST` with no `CC_<target>` /
  `TARGET_CC` / cross-gcc binary and emits `cargo:rustc-cfg=no_miniaudio`
  instead of running the `cc`-crate build script.
- **`target-cpu=native`** — must NOT be set globally; it must be scoped
  to `[target.x86_64-unknown-linux-gnu]` or it pollutes the aarch64
  invocation with x86 feature flags (warnings, no functional impact).
