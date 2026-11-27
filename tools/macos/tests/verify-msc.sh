#!/bin/bash
# verify.sh fails fast when a Mac has no Metal Shader Converter dylib, and
# skips that check off a Mac (#379). Game-free; stubs `uname` and `rustup`.
# Run: /bin/bash tools/macos/tests/verify-msc.sh
set -euo pipefail
scripts=$(cd "$(dirname "$0")/.." && pwd)
root=$(mktemp -d "${TMPDIR:-/tmp}/victory verify-msc.XXXXXX")
trap 'rm -rf "$root"' EXIT
mkdir -p "$root/bin"

# `uname` answers the test's fake OS; `rustup` records that a gate ran.
printf '#!/bin/sh\necho "${FAKE_UNAME:-Linux}"\n' > "$root/bin/uname"
printf '#!/bin/sh\necho "RUSTUP-RAN $*"\nexit 1\n' > "$root/bin/rustup"
chmod +x "$root/bin/uname" "$root/bin/rustup"
touch "$root/present.dylib"

run() { # run <fake-os> <dylib-var-or-empty> -> verify.sh's output and status
  local out rc=0
  if [ -n "$2" ]; then
    out=$(PATH="$root/bin:$PATH" FAKE_UNAME="$1" VICTORY_MSC_DYLIB="$2" \
      sh "$scripts/verify.sh" quick 2>&1) || rc=$?
  else
    out=$(PATH="$root/bin:$PATH" FAKE_UNAME="$1" \
      sh "$scripts/verify.sh" quick 2>&1) || rc=$?
  fi
  echo "$out"
  return "$rc"
}

# A Mac without the dylib: fails before any gate, naming the converter.
out=$(run Darwin "$root/missing.dylib") && { echo "FAIL: missing dylib did not fail" >&2; exit 1; }
case "$out" in
  *"no Metal Shader Converter"*"install Apple's Metal Shader Converter"*) ;;
  *) echo "FAIL: Darwin without the dylib: $out" >&2; exit 1 ;;
esac
case "$out" in
  *suite-hle*|*RUSTUP-RAN*) echo "FAIL: gates ran despite the missing dylib: $out" >&2; exit 1 ;;
esac

# Off a Mac the check is skipped: the gates run (and fail on the stub).
out=$(run Linux "$root/missing.dylib") || true
case "$out" in
  *suite-hle*|*RUSTUP-RAN*) ;;
  *) echo "FAIL: Linux skipped the gates: $out" >&2; exit 1 ;;
esac
case "$out" in
  *"no Metal Shader Converter"*) echo "FAIL: Linux checked the dylib: $out" >&2; exit 1 ;;
esac

# A Mac with the dylib (named or at the default): the gates run.
out=$(run Darwin "$root/present.dylib") || true
case "$out" in
  *suite-hle*|*RUSTUP-RAN*) ;;
  *) echo "FAIL: Darwin with the dylib did not run the gates: $out" >&2; exit 1 ;;
esac
case "$out" in
  *"no Metal Shader Converter"*) echo "FAIL: Darwin with the dylib still failed: $out" >&2; exit 1 ;;
esac

echo "verify msc check: ok"
