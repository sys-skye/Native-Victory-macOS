# Shared by native-setup.sh and run-native.sh, from the workspace root.

# Puts rustup on PATH when it is installed but not on PATH yet. That happens
# right after the rustup.rs installer (until a new terminal opens) and with
# Homebrew's rustup formula, which is keg-only. Sets rustup_found to the
# folder it added, or leaves it empty when rustup was already on PATH.
# Returns 1 when there is no rustup.
find_rustup() {
  rustup_found=""
  command -v rustup >/dev/null 2>&1 && return 0
  local dir
  for dir in "${CARGO_HOME:-$HOME/.cargo}/bin" "${HOMEBREW_PREFIX:-/opt/homebrew}/opt/rustup/bin"; do
    if [[ -x "$dir/rustup" ]]; then
      export PATH="$dir:$PATH"
      rustup_found="$dir"
      return 0
    fi
  done
  return 1
}

# The console model run-native.sh reports when --console is not given, from the
# Mac's RAM in whole GiB. Series X declares 13.5 GiB of title memory: more than
# Metal's working set on 16 GB Macs (#8), and it can still run out of GPU
# memory at the world load on 24 GB (#79, #110). Series S declares 7.6 GiB.
default_console() {
  if (( $1 <= 24 )); then echo series-s; else echo series-x; fi
}

# Cargo resolves environment overrides and local/global build.target-dir config.
# plutil is part of macOS and decodes JSON strings without losing path quoting.
native_binary() {
  local target_dir
  if ! command -v cargo >/dev/null 2>&1 && ! find_rustup; then
    echo "Cargo is not on your PATH. Install rustup from https://rustup.rs, then run tools/macos/native-setup.sh." >&2
    return 1
  fi
  if ! target_dir=$(set -o pipefail
    cargo metadata --format-version 1 --no-deps --offline |
      /usr/bin/plutil -extract target_directory raw -expect string -o - -
  ); then
    echo "Cannot resolve Cargo's target directory. Check the Cargo error above and your Cargo configuration/toolchain." >&2
    return 1
  fi
  if [[ -z "$target_dir" || "$target_dir" != /* ]]; then
    echo "Cargo metadata did not provide an absolute target directory." >&2
    return 1
  fi
  printf '%s/x86_64-apple-darwin/release/victory\n' "$target_dir"
}
