// build.rs for vc-native — compile vendored miniaudio.c via cc crate
// miniaudio.h + miniaudio.c live in ../../src_c/miniaudio/
//
// Run `bash scripts/vendor_c_deps.sh` from the prototype root to download
// miniaudio.h (~95K lines, MIT-0) before `cargo build`. If the file is missing,
// build.rs emits a warning and skips the C compilation (vc-native still
// compiles, but the miniaudio FFI symbols are not linked — P2.1 audio wiring
// will then need the file present).

fn main() {
    // Vendor miniaudio as a single-header C library. miniaudio_shim.c is a
    // thin C wrapper (4 extern "C" functions with opaque void* handles) that
    // hides the complex ma_device_config/ma_device structs from Rust.
    let miniaudio_dir = "../../src_c/miniaudio";
    let shim_path = format!("{miniaudio_dir}/miniaudio_shim.c");
    if std::path::Path::new(&format!("{miniaudio_dir}/miniaudio.c")).exists() {
        let mut build = cc::Build::new();
        build
            .file(format!("{miniaudio_dir}/miniaudio.c"))
            .include(miniaudio_dir)
            .flag_if_supported("-O3")
            .flag_if_supported("-mavx2") // x86 AVX2
            .flag_if_supported("-mfpu=neon"); // ARM NEON
        // Compile the shim in the same archive (so both miniaudio + shim
        // symbols live in libminiaudio.a and Rust can link them together).
        if std::path::Path::new(&shim_path).exists() {
            build.file(&shim_path);
            println!("cargo:rerun-if-changed={shim_path}");
        }
        build.compile("miniaudio");
        println!("cargo:rerun-if-changed={miniaudio_dir}/miniaudio.c");
        println!("cargo:rerun-if-changed={miniaudio_dir}/miniaudio.h");
    } else {
        println!(
            "cargo:warning=miniaudio.c not found at {miniaudio_dir}/ — \
             run `bash scripts/vendor_c_deps.sh` to fetch it, then rebuild. \
             Skipping C compilation; vc-native will be inference-only."
        );
    }
}
