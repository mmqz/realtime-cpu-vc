# src_c/ — vendored C/C++ headers for v2.0 Rust FFI
#
# Files to vendor in P1:
#   miniaudio/miniaudio.h + miniaudio.c    (95K lines, MIT-0, from mackron/miniaudio)
#   ggml/ggml.h + ggml.c                   (~2MB compiled, MIT, from ggml-org/ggml)
#
# These are NOT included in the repo (size + licensing hygiene). They are downloaded
# by `scripts/vendor_c_deps.sh` (P1 task) before `cargo build`.

This file serves as a marker. Actual vendored files land here in P1 via:

    bash scripts/vendor_c_deps.sh

which fetches:
- miniaudio.h from https://raw.githubusercontent.com/mackron/miniaudio/master/miniaudio.h
- ggml.h from https://github.com/ggml-org/ggml (vendored as git submodule or tarball)
