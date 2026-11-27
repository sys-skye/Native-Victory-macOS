#!/bin/bash
# verify.sh builds the HLE suite for x86_64-apple-darwin on a Mac and for the
# host target elsewhere (the hostless build, #316). Game-free; stubs `uname`
# and `rustup`.
# Run: /bin/bash tools/macos/tests/verify-target.sh
set -euo pipefail
scripts=$(cd "$(dirname "$0")/.." && pwd)
root=$(mktemp -d "${TMPDIR:-/tmp}/victory verify-target.XXXXXX")
trap 'rm -rf "$root"' EXIT
mkdir -p "$root/bin"

printf '#!/bin/sh\necho "${FAKE_UNAME:-Linux}"\n' > "$root/bin/uname"
# Record each gate's command line instead of running cargo.
printf '#!/bin/sh\necho "$*" >> "$MARKER"\nexit 1\n' > "$root/bin/rustup"
chmod +x "$root/bin/uname" "$root/bin/rustup"
# verify.sh stops early on a Mac without the shader converter (#379): give it a stand-in.
touch "$root/msc.dylib"

check() { # check <fake-os> <expected-substring> <forbidden-substring>
  local os="$1" want="$2" forbid="$3"
  : > "$root/marker"
  PATH="$root/bin:$PATH" FAKE_UNAME="$os" MARKER="$root/marker" VICTORY_MSC_DYLIB="$root/msc.dylib" \
    sh "$scripts/verify.sh" quick >/dev/null 2>&1 || true
  if ! grep -qF -- "$want" "$root/marker"; then
    echo "FAIL: $os: no '$want' in:" >&2; cat "$root/marker" >&2; exit 1
  fi
  if [ -n "$forbid" ] && grep -qF -- "$forbid" "$root/marker"; then
    echo "FAIL: $os: unexpected '$forbid' in:" >&2; cat "$root/marker" >&2; exit 1
  fi
}

# On a Mac the HLE suite is built for x86_64-apple-darwin.
check Darwin "cargo test --target x86_64-apple-darwin -p victory-hle -p victory-loader -p victory" ""
# Elsewhere (Linux agents, #316) it runs for the host target: no --target.
check Linux "cargo test -p victory-hle -p victory-loader -p victory" "--target"

echo "verify host target: ok"
