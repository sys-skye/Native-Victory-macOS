#!/bin/bash
# cargo test for the native macOS host, failing fast when the Metal Shader Converter is
# missing: without it the shader-translating tests panic inside the converter's dlopen
# and look like GPU bugs (#379). Package subsets that need no converter (victory-shader
# alone, say) can still call cargo test directly.
# Usage: tools/macos/test-native.sh [cargo test arguments]
set -euo pipefail
cd "$(dirname "$0")/../.."

MSC="${VICTORY_MSC_DYLIB:-/usr/local/lib/libmetalirconverter.dylib}"
if [[ ! -f "$MSC" ]]; then
  {
    echo "fail: Metal Shader Converter not found at $MSC"
    echo "      Download \"Metal shader converter\" from https://developer.apple.com/metal/shader-converter/"
    echo "      and run its installer (free with an Apple ID), or point VICTORY_MSC_DYLIB at Apple's"
    echo "      libmetalirconverter.dylib. A copy bundled with another tool is not a substitute (#379)."
  } >&2
  exit 1
fi

# The toolchain's own bin first: a Homebrew rustc earlier on PATH has no arm64 std.
sysroot=$(rustup run 1.98.1 rustc --print sysroot)
export PATH="$sysroot/bin:$PATH"
cargo test "$@"
