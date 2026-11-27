#!/bin/bash
# test-native.sh fails fast without the Metal Shader Converter and runs cargo with it.
# Run: /bin/bash tools/macos/tests/msc-check.sh
set -euo pipefail
script="$(cd "$(dirname "$0")/.." && pwd)/test-native.sh"
root=$(mktemp -d "${TMPDIR:-/tmp}/victory msc.XXXXXX")
trap 'rm -rf "$root"' EXIT

# Stand-ins for rustup and cargo: this test must not run a real build.
mkdir -p "$root/bin"
printf '#!/bin/sh\nif [ "$1" = run ]; then echo /stub-sysroot; exit 0; fi\nexit 1\n' > "$root/bin/rustup"
printf '#!/bin/sh\necho "cargo-stub $*"\n' > "$root/bin/cargo"
chmod +x "$root/bin/rustup" "$root/bin/cargo"
export PATH="$root/bin:$PATH"

# The converter named by VICTORY_MSC_DYLIB but absent: fail fast, cargo not run.
out=$(VICTORY_MSC_DYLIB="$root/nope.dylib" /bin/bash "$script" -p victory-gpu 2>&1) && rc=0 || rc=$?
if [ "$rc" -eq 0 ]; then
  echo "FAIL: without the converter the run succeeded" >&2
  exit 1
fi
case "$out" in
  *"Metal Shader Converter not found"*) ;;
  *) echo "FAIL: unclear message: $out" >&2; exit 1 ;;
esac
case "$out" in
  *cargo-stub*) echo "FAIL: cargo ran without the converter" >&2; exit 1 ;;
  *) ;;
esac

# The converter present at the override path: cargo runs with the arguments.
touch "$root/msc.dylib"
out=$(VICTORY_MSC_DYLIB="$root/msc.dylib" /bin/bash "$script" -p victory-gpu --locked 2>&1) \
  || { echo "FAIL: with the converter the run failed: $out" >&2; exit 1; }
case "$out" in
  *"cargo-stub test -p victory-gpu --locked"*) ;;
  *) echo "FAIL: cargo did not get the arguments: $out" >&2; exit 1 ;;
esac

# Unset: the installed default decides, as on a Mac with or without the installer.
default=/usr/local/lib/libmetalirconverter.dylib
out=$(env -u VICTORY_MSC_DYLIB /bin/bash "$script" -p victory-gpu 2>&1) && rc=0 || rc=$?
if [ -f "$default" ]; then
  if [ "$rc" -ne 0 ]; then
    echo "FAIL: with $default present the run failed: $out" >&2
    exit 1
  fi
else
  if [ "$rc" -eq 0 ]; then
    echo "FAIL: without $default the run succeeded" >&2
    exit 1
  fi
  case "$out" in
    *"Metal Shader Converter not found at $default"*) ;;
    *) echo "FAIL: unclear message: $out" >&2; exit 1 ;;
  esac
fi
echo "test-native converter check: ok"
