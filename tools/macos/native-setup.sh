#!/bin/bash
# One-time setup for the native macOS port: checks the prerequisites, installs
# the pinned Rust toolchain with its x86_64 target, and builds `victory`.
# Usage: tools/macos/native-setup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mfail\033[0m  %s\n' "$1"; exit 1; }

echo "Checking your Mac"
[[ "$(uname -s)" == Darwin ]] || fail "this script is for macOS"
[[ "$(uname -m)" == arm64 ]] || fail "an Apple Silicon Mac (M1 or newer) is required"
ok "Apple Silicon, macOS $(sw_vers -productVersion)"

if arch -x86_64 /usr/bin/true 2>/dev/null; then ok "Rosetta 2"
else fail "Rosetta 2 is missing: run  softwareupdate --install-rosetta --agree-to-license"; fi

if xcode-select -p >/dev/null 2>&1; then ok "Xcode command line tools"
else fail "Xcode command line tools are missing: run  xcode-select --install"; fi

source tools/macos/native-binary.sh
if ! find_rustup; then
  fail "rustup is missing: install it from https://rustup.rs (Homebrew's plain 'rust' package will not work)"
elif [[ -n "$rustup_found" ]]; then
  warn "rustup is in $rustup_found but not on your PATH; using it. Open a new terminal, or add that folder to your PATH"
else ok "rustup"; fi

MSC="${VICTORY_MSC_DYLIB:-/usr/local/lib/libmetalirconverter.dylib}"
msc_exports=$([[ -f "$MSC" ]] && nm -gU "$MSC" 2>/dev/null || true)
if [[ -f "$MSC" && "$msc_exports" != *_IRCompilerSetFunctionConstantResourceSpace* ]]; then
  # Converters before 3.0 lack this export and need array textures, which victory does not bind (#295).
  warn "the Metal Shader Converter at $MSC is older than 3.0, and victory refuses it"
  echo "        Install Apple's 3.0 or newer from https://developer.apple.com/metal/shader-converter/"
elif [[ -f "$MSC" ]]; then ok "Metal Shader Converter ($MSC)"
else
  warn "Metal Shader Converter not found at $MSC"
  echo "        Download \"Metal shader converter\" from https://developer.apple.com/metal/shader-converter/"
  echo "        and run its installer. It is free with an Apple ID. Without it shaders cannot be translated"
  echo "        (tools/macos/test-native.sh fails fast on the same check, #379)."
fi

echo
echo "Installing the pinned Rust toolchain (1.98.1) and the x86_64 target"
rustup toolchain install 1.98.1 --profile minimal --component rustfmt,clippy >/dev/null
rustup target add --toolchain 1.98.1 x86_64-apple-darwin >/dev/null
ok "toolchain ready"

echo
echo "Building victory (release, x86_64, runs under Rosetta). The first build takes a few minutes."
# The toolchain's own bin first: a Homebrew rustc earlier on PATH has no x86_64 std.
sysroot=$(rustup run 1.98.1 rustc --print sysroot)
export PATH="$sysroot/bin:$PATH"
bin=$(native_binary) || fail "could not locate the Cargo build output"
cargo build --release --target x86_64-apple-darwin -p victory
ok "built $bin"

echo
echo "Done. Run the game with:"
echo "  tools/macos/run-native.sh /path/to/your/game/folder"
