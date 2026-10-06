// build.rs for vc-native — compile vendored miniaudio.c via cc crate
// miniaudio.h + miniaudio.c live in ../../src_c/miniaudio/

fn main() {
    // Vendor miniaudio as a single-header C library.
    // The actual miniaudio.h is ~95K lines; we download it in P1 (placeholder for now).
    let miniaudio_dir = "../../src_c/miniaudio";
    if std::path::Path::new(&format!("{miniaudio_dir}/miniaudio.c")).exists() {
        cc::Build::new()
            .file(format!("{miniaudio_dir}/miniaudio.c"))
            .flag_if_supported("-O3")
            .flag_if_supported("-mavx2")    // x86 AVX2
            .flag_if_supported("-mfpu=neon") // ARM NEON
            .compile("miniaudio");
        println!("cargo:rerun-if-changed={miniaudio_dir}/miniaudio.c");
        println!("cargo:rerun-if-changed={miniaudio_dir}/miniaudio.h");
    } else {
        // Placeholder until P1 vendors the actual file
        println!("cargo:warning=miniaudio.c not found at {miniaudio_dir}/ — skip audio I/O (vc-native will be inference-only)");
    }
}
