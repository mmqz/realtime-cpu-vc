#!/bin/bash
# scripts/vendor_c_deps.sh — fetch vendored C/C++ headers for v2.0 Rust FFI
# Run before: cd src_rust && cargo build --release

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_C_DIR="$PROJECT_ROOT/src_c"

mkdir -p "$SRC_C_DIR/miniaudio" "$SRC_C_DIR/ggml"

echo "==> Vendoring miniaudio.h (single-header, ~95K lines, MIT-0)..."
if [ ! -f "$SRC_C_DIR/miniaudio/miniaudio.h" ]; then
    curl -fsSL https://raw.githubusercontent.com/mackron/miniaudio/master/miniaudio.h \
        -o "$SRC_C_DIR/miniaudio/miniaudio.h"
    echo "    miniaudio.h OK"
fi

# miniaudio needs a .c stub to compile
if [ ! -f "$SRC_C_DIR/miniaudio/miniaudio.c" ]; then
    echo '// miniaudio stub — vendor by including the header' > "$SRC_C_DIR/miniaudio/miniaudio.c"
    echo '#include "miniaudio.h"' >> "$SRC_C_DIR/miniaudio/miniaudio.c"
    echo "    miniaudio.c stub OK"
fi

echo ""
echo "==> Vendoring ggml (submodule, ~2MB compiled, MIT)..."
if [ ! -d "$SRC_C_DIR/ggml/src" ]; then
    git clone --depth 1 https://github.com/ggml-org/ggml "$SRC_C_DIR/ggml" 2>&1 | tail -3
    echo "    ggml OK"
fi

echo ""
echo "==> Done. Sources vendored at $SRC_C_DIR"
ls -la "$SRC_C_DIR"/miniaudio/ "$SRC_C_DIR"/ggml/ 2>/dev/null | head -20
