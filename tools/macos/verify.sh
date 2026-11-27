#!/bin/sh
# The verification gate for native VIctory on a Mac: every check a change
# must pass before it goes into a pull request.
#
# usage: tools/macos/verify.sh [quick|full]
#
#   quick  the test suites: CONTRIBUTING.md's two, and victory-gpu natively
#   full   quick + cargo xtask policy + the CI clippy gate for the Windows target (default)
#
# Prints one PASS/FAIL line per gate and exits non-zero if any gate failed,
# so a caller (CI, a hook, a human) can gate a pull request on it.
set -u
cd "$(dirname "$0")/../.." || exit 1

GATE="${1:-full}"
RUNNER="rustup run 1.98.1"
fail=0

run_gate() {
    name="$1"; shift
    printf '%-14s ... ' "$name"
    log=$(mktemp)
    if "$@" >"$log" 2>&1; then
        # Summarize: last "test result:" lines carry the counts.
        counts=$(grep -h "^test result:" "$log" | awk '{p+=$4; f+=$6} END {printf "%d passed, %d failed", p, f}')
        echo "PASS (${counts:-ok})"
    else
        echo "FAIL"
        grep -E "^(error|failures:|---- )" "$log" | head -10 | sed 's/^/    /'
        tail -3 "$log" | sed 's/^/    /'
        fail=1
    fi
    rm -f "$log"
}

echo "== native VIctory verification gate ($GATE) =="

# The HLE suite builds for x86_64-apple-darwin on an Apple Silicon Mac (the
# tests run x86_64 binaries under Rosetta); on any other host the hostless
# build (#316) lets it run for the host target.
if [ "$(uname)" = "Darwin" ]; then
    HLE_TARGET="--target x86_64-apple-darwin"
else
    HLE_TARGET=""
fi

# The native victory-gpu tests convert DXIL through Apple's Metal Shader
# Converter; without its dylib they fail with an indirect panic that looks
# like a GPU bug, so fail fast on the prerequisite first on a Mac (#379).
# (GPTK's bundled copy of the dylib is not a substitute for Apple's.)
MSC="${VICTORY_MSC_DYLIB:-/usr/local/lib/libmetalirconverter.dylib}"
if [ "$(uname)" = "Darwin" ] && [ ! -f "$MSC" ]; then
    echo "FAIL: no Metal Shader Converter at $MSC"
    echo "install Apple's Metal Shader Converter (https://developer.apple.com/metal/metal-shader-converter/)"
    echo "or set VICTORY_MSC_DYLIB to its libmetalirconverter.dylib"
    exit 1
fi

run_gate "suite-hle" sh -c \
    "$RUNNER cargo test $HLE_TARGET -p victory-hle -p victory-loader -p victory"

run_gate "suite-shader" sh -c \
    "$RUNNER cargo test -p victory-shader -p victory-hle"

run_gate "suite-gpu" sh -c \
    "$RUNNER cargo test -p victory-gpu"

if [ "$GATE" = "full" ]; then
    run_gate "policy" sh -c \
        "$RUNNER cargo xtask policy"
    run_gate "clippy-win" sh -c \
        "$RUNNER cargo clippy --workspace --all-targets --locked --target x86_64-pc-windows-gnullvm -- -D warnings"
fi

echo "== gate $([ $fail -eq 0 ] && echo PASSED || echo FAILED) =="
exit $fail
