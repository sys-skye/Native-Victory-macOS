#!/bin/bash
# run-native.sh's default console model for each RAM size, without a game.
# Run: /bin/bash tools/macos/tests/console-default.sh
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/native-binary.sh"

check() {
  local got
  got=$(default_console "$1")
  if [ "$got" != "$2" ]; then
    echo "FAIL: $1 GB defaults to $got, expected $2" >&2
    exit 1
  fi
}

check 8 series-s
check 16 series-s
check 18 series-s
check 24 series-s
check 32 series-x
check 36 series-x
check 48 series-x
check 128 series-x
echo "console default: ok"
