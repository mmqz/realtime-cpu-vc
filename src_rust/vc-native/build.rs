// build.rs for vc-native — compile vendored miniaudio.c via cc crate
// miniaudio.h + miniaudio.c live in ../../src_c/miniaudio/
//
// Run `bash scripts/vendor_c_deps.sh` from the prototype root to download
// miniaudio.h (~95K lines, MIT-0) before `cargo build`. If the file is missing,
// build.rs emits a warning and skips the C compilation (vc-native still
// compiles, but the miniaudio FFI symbols are not linked — P2.1 audio wiring
// will then need the file present).

fn main() {
    // Vendor miniaudio as a single-header C library.
    let miniaudio_dir = "../../src_c/miniaudio";
    if std::path::Path::new(&format!("{miniaudio_dir}/miniaudio.c")).exists() {
        cc::Build::new()
            .file(format!("{miniaudio_dir}/miniaudio.c"))
            .include(miniaudio_dir)
            .flag_if_supported("-O3")
            .flag_if_supported("-mavx2") // x86 AVX2
            .flag_if_supported("-mfpu=neon") // ARM NEON
            .compile("miniaudio");
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
