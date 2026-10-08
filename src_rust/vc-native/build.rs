// build.rs for vc-native — compile vendored miniaudio.c via cc crate
// miniaudio.h + miniaudio.c live in ../../src_c/miniaudio/
//
// Run `bash scripts/vendor_c_deps.sh` from the prototype root to download
// miniaudio.h (~95K lines, MIT-0) before `cargo build`. If the file is missing,
// build.rs emits a warning and skips the C compilation (vc-native still
// compiles, but the miniaudio FFI symbols are not linked — P2.1 audio wiring
// will then need the file present).
//
// Cross-compile note (OPT-17): when `TARGET != HOST` (e.g. building for
// aarch64-unknown-linux-gnu on an x86_64 host) and no C cross-compiler is
// available, build.rs emits `cargo:rustc-cfg=no_miniaudio` and skips the
// C compilation. This makes `cargo check --target aarch64` work without
// requiring `gcc-aarch64-linux-gnu` to be installed on the dev host.

fn main() {
    // Vendor miniaudio as a single-header C library. miniaudio_shim.c is a
    // thin C wrapper (4 extern "C" functions with opaque void* handles) that
    // hides the complex ma_device_config/ma_device structs from Rust.
    let miniaudio_dir = "../../src_c/miniaudio";
    let shim_path = format!("{miniaudio_dir}/miniaudio_shim.c");
    let miniaudio_c = format!("{miniaudio_dir}/miniaudio.c");

    // ---- Case 1: miniaudio.c not vendored yet ------------------------------
    if !std::path::Path::new(&miniaudio_c).exists() {
        println!(
            "cargo:warning=miniaudio.c not found at {miniaudio_dir}/ — \
             run `bash scripts/vendor_c_deps.sh` to fetch it, then rebuild. \
             Skipping C compilation; vc-native will be inference-only."
        );
        // Skip C compilation; miniaudio_ffi.rs still compiles (it only
        // declares `extern "C"` symbols — they're unresolved until link,
        // which is fine for `cargo check` and for inference-only builds).
        return;
    }

    // ---- Case 2: cross-compiling without a C cross-linker ------------------
    // When `TARGET != HOST`, the cc crate needs a C cross-compiler. If the
    // user hasn't installed one (e.g. `gcc-aarch64-linux-gnu`) and hasn't
    // set CC_<target> / TARGET_CC, skip the C compile entirely so that
    // `cargo check --target aarch64` works on a bare dev host.
    //
    // For `cargo build` (full link), the user still needs the cross-gcc —
    // otherwise the link step will fail with unresolved miniaudio symbols.
    // The `no_miniaudio` cfg flag lets downstream code conditionally stub
    // out the audio I/O path if desired.
    let target = std::env::var("TARGET").unwrap_or_default();
    let host = std::env::var("HOST").unwrap_or_default();
    let is_cross = !target.is_empty() && !host.is_empty() && target != host;

    if is_cross {
        let cc_var = format!("CC_{}", target.replace('-', "_"));
        let cross_cc = std::env::var(&cc_var)
            .or_else(|_| std::env::var("TARGET_CC"))
            .unwrap_or_default();

        if cross_cc.is_empty() && !has_cross_gcc(&target) {
            println!(
                "cargo:warning=Cross-compiling to {target} without a C cross-linker — \
                 skipping miniaudio (audio I/O disabled). \
                 Install `gcc-{cross_pkg}` or set {cc_var} for a full audio build.",
                cross_pkg = cross_pkg_name(&target),
                cc_var = cc_var,
            );
            println!("cargo:rustc-cfg=no_miniaudio");
            return;
        }
    }

    // ---- Case 3: normal compilation (x86 host OR cross with toolchain) ------
    // Compile the shim in the same archive (so both miniaudio + shim
    // symbols live in libminiaudio.a and Rust can link them together).
    let mut build = cc::Build::new();
    build
        .file(&miniaudio_c)
        .include(miniaudio_dir)
        .flag_if_supported("-O3")
        .flag_if_supported("-mavx2") // x86 AVX2
        .flag_if_supported("-mfpu=neon"); // ARM NEON
    if std::path::Path::new(&shim_path).exists() {
        build.file(&shim_path);
        println!("cargo:rerun-if-changed={shim_path}");
    }
    build.compile("miniaudio");
    println!("cargo:rerun-if-changed={miniaudio_c}");
    println!("cargo:rerun-if-changed={miniaudio_dir}/miniaudio.h");
}

/// Map a Rust target triple to the Debian/Ubuntu `gcc-<triplet>` package name
/// (used only for the warning message — does not affect compilation).
fn cross_pkg_name(target: &str) -> &'static str {
    if target.contains("aarch64") && target.contains("linux") {
        "aarch64-linux-gnu"
    } else if target.contains("arm") && target.contains("linux") {
        "arm-linux-gnueabihf"
    } else {
        "<target-triplet>"
    }
}

/// Probe for the cross-gcc binary on PATH (best-effort — false negatives OK).
fn has_cross_gcc(target: &str) -> bool {
    let cross_gcc = match target {
        t if t.contains("aarch64") && t.contains("linux") => "aarch64-linux-gnu-gcc",
        t if t.contains("arm") && t.contains("linux") && !t.contains("aarch64") => {
            "arm-linux-gnueabihf-gcc"
        }
        _ => return false,
    };
    std::process::Command::new(cross_gcc)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
