#!/bin/bash
# Runs the game natively and keeps a log you can attach to a tester report.
# Usage: tools/macos/run-native.sh [--headless] [--seconds N] [--console series-x|series-s] <game folder or .exe>
#   --headless            no window; frames go to an offscreen layer
#   --seconds N, --seconds=N stop the run after N decimal seconds (default: run until you close it)
#   --console MODEL       console model reported to the game (default: series-s on <=24 GB Macs, else series-x)
# Extra VICTORY_* variables in your environment are passed through
# (see docs/native-host.md, "Bring-up diagnostics").
set -uo pipefail

headless=()
seconds=0
console=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless) headless=(--headless); shift ;;
    --seconds)
      [[ $# -ge 2 ]] || { echo "--seconds requires a non-negative decimal integer" >&2; exit 2; }
      seconds="$2"; shift 2 ;;
    --seconds=*) seconds="${1#--seconds=}"; shift ;;
    --console)
      [[ -n "${2:-}" && "$2" != -* ]] || { echo "--console requires a model" >&2; exit 2; }
      console="$2"; shift 2 ;;
    --console=) echo "--console requires a model" >&2; exit 2 ;;
    --console=*) console="${1#--console=}"; shift ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) break ;;
  esac
done
[[ "$seconds" =~ ^[0-9]+$ ]] || { echo "--seconds requires a non-negative decimal integer" >&2; exit 2; }
decimal_seconds=$((10#$seconds))
# Bash silently wraps overflowing integers, which could disable or shorten the timer.
[[ "$seconds" =~ ^0*$decimal_seconds$ ]] || { echo "--seconds exceeds the timer range" >&2; exit 2; }
[[ $# -eq 1 ]] || { sed -n 2,9p "$0"; exit 2; }

ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))

target="$1"
if [[ -d "$target" ]]; then
  exe=$(find "$target" -maxdepth 1 -iname 'game_*master.exe' | head -1)
  [[ -n "$exe" ]] || { echo "No game_*master.exe found in $target"; exit 2; }
else
  exe="$target"
fi
[[ -f "$exe" ]] || { echo "Not a file: $exe"; exit 2; }
# Absolute before moving to the repo root, so a relative game path still works.
exe="$(cd "$(dirname "$exe")" && pwd)/$(basename "$exe")"
cd "$(dirname "$0")/../.."

source tools/macos/native-binary.sh
[[ -n "$console" ]] || console=$(default_console "$ram_gb")
bin=$(native_binary) || exit 2
[[ -x "$bin" ]] || { echo "victory is not executable at $bin: run tools/macos/native-setup.sh first"; exit 2; }

mkdir -p runs || exit 1
run_dir=$(mktemp -d "runs/$(date +%Y%m%d-%H%M%S)-XXXXXX") || exit 1
log="$run_dir/victory.log"
{
  echo "# $(sw_vers -productName) $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), ${ram_gb} GB, console=${console}"
  echo "# commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} > "$log"

export VICTORY_GUESS_GRAPHICS="${VICTORY_GUESS_GRAPHICS:-1}"
# Press Enter (the pad's A) every 2 s from 4 s to 80 s unless VICTORY_KEYS is set.
export VICTORY_KEYS="${VICTORY_KEYS:-$(seq 4 2 80 | sed 's/$/:Enter/' | paste -sd, -)}"
echo "Running $exe (console=$console)"
echo "Log: $log"

"$bin" run --execute --console "$console" ${headless[@]+"${headless[@]}"} --unimplemented zero "$exe" >> "$log" 2>&1 &
pid=$!
timer=""
if (( decimal_seconds > 0 )); then
  # The marker tells run-summary.sh that the limit, not a crash, ended the run.
  ( sleep "$decimal_seconds" && { echo "# stopped by --seconds" >> "$log"; kill "$pid" 2>/dev/null; } ) &
  timer=$!
fi
# A background job ignores Ctrl-C in a script, so pass it on: stop the game, its
# children (a restarted game) and the timer, or they outlive the script.
stop_timer() {
  [[ -z "$timer" ]] || { pkill -P "$timer" 2>/dev/null; kill "$timer" 2>/dev/null; }
}
trap 'pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; stop_timer; wait "$pid"; echo; echo "Stopped. Log: $log"; exit 130' INT TERM
wait "$pid"
rc=$?
# For run-summary.sh: a host crash (139) writes no log line of its own.
echo "# exit code $rc" >> "$log"
stop_timer

echo
echo "Exit code $rc. Last lines of the log:"
sed 's/\x1b\[[0-9;]*m//g' "$log" | tail -15 | cut -c1-200
echo
echo "To report: open a Tester report issue and attach $log"
exit "$rc"
