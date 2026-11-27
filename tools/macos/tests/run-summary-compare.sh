#!/bin/bash
# Game-free checks of run-summary.sh --compare, under stock macOS Bash 3.2.
# Run: /bin/bash tools/macos/tests/run-summary-compare.sh
# The logs are synthetic, made up for this test in the format of real log lines.
set -eu
script="$(cd "$(dirname "$0")/.." && pwd)/run-summary.sh"
dir=$(mktemp -d "${TMPDIR:-/tmp}/run-summary.XXXXXX")
trap 'rm -rf "$dir"' EXIT
esc=$(printf '\033')
fail() { echo "FAIL: $*" >&2; cat "$dir/out" >&2; exit 1; }
has() { grep -F -- "$1" "$dir/out" >/dev/null || fail "missing: $1"; }
row() { grep -E -- "$1" "$dir/out" >/dev/null || fail "no row matching: $1"; }

# The logger colours lines even into a file; @ESC@ stands for the escape byte.
sed "s/@ESC@/${esc}/g" > "$dir/before.log" <<'LOG'
# macOS 26.1, Apple M5, 24 GB, console=series-s
# commit aaaa111
console        Series S
@ESC@[2m2026-01-01T00:00:00.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=64 recorded=1
@ESC@[2m2026-01-01T00:00:04.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=128 recorded=1
@ESC@[2m2026-01-01T00:00:04.100000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::pipeline: first bind pipeline=0x1 frame=130 stage="pixel" hash=abc
@ESC@[2m2026-01-01T00:00:04.200000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::pipeline: first bind pipeline=0x2 frame=130 stage="vertex" hash=abd
@ESC@[2m2026-01-01T00:00:04.300000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::pipeline: shaders prewarmed n=8000 pending=8 dxil_mb=30
@ESC@[2m2026-01-01T00:00:05.000000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::native::shaders: title shader refused; its work is dropped shader="pixel aa" reason=translation: code at 0x10: s_lshr_b32 is not supported
@ESC@[2m2026-01-01T00:00:05.100000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::native::shaders: title shader refused; its work is dropped shader="pixel aa" reason=translation: code at 0x10: s_lshr_b32 is not supported
@ESC@[2m2026-01-01T00:00:05.200000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::native::shaders: title shader refused; its work is dropped shader="pixel bb" reason=translation: code at 0x18: s_lshr_b32 is not supported
@ESC@[2m2026-01-01T00:00:05.300000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::native::shaders: title shader refused; its work is dropped shader="compute cc" reason=translation: code at 0x20: a descriptor neither loaded nor built
@ESC@[2m2026-01-01T00:00:06.000000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::list: indirect execution dropped (first time) stride=12 address=Some("0xdead")
@ESC@[2m2026-01-01T00:00:06.100000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::list: indirect execution dropped (first time) stride=16 address=Some("0xbeef")
@ESC@[2m2026-01-01T00:00:06.200000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::list: indirect execution dropped (first time) stride=16 address=Some("0xf00d")
@ESC@[2m2026-01-01T00:00:08.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=192 recorded=1
metal: command buffer failed: NSError { code: 2, "Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)" }
metal: command buffer failed: NSError { code: 2, "Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)" }
@ESC@[2m2026-01-01T00:00:16.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=256 recorded=1
VIctory: unhandled exception 0xc0000005 at 0x1423cf683 [1, 61]
# exit code 1
LOG

sed "s/@ESC@/${esc}/g" > "$dir/after.log" <<'LOG'
# macOS 26.1, Apple M5, 24 GB, console=series-s
# commit bbbb222
console        Series S
@ESC@[2m2026-01-01T23:59:58.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=64 recorded=1
@ESC@[2m2026-01-02T00:00:00.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=128 recorded=1
@ESC@[2m2026-01-02T00:00:00.100000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::pipeline: first bind pipeline=0x1 frame=130 stage="pixel" hash=abc took=5.0ms
@ESC@[2m2026-01-02T00:00:00.200000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::pipeline: shaders prewarmed n=8192 pending=0 dxil_mb=31
@ESC@[2m2026-01-02T00:00:01.000000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::native::shaders: title shader refused; its work is dropped shader="pixel aa" reason=translation: code at 0x10: s_lshr_b32 is not supported
@ESC@[2m2026-01-02T00:00:01.100000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::native::shaders: title shader refused; its work is dropped shader="pixel dd" reason=pixel input: barycentrics read directly
@ESC@[2m2026-01-02T00:00:01.200000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::native::shaders: title shader refused; its work is dropped shader="pixel ee" reason=pixel input: barycentrics read directly
@ESC@[2m2026-01-02T00:00:02.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=192 recorded=1
@ESC@[2m2026-01-02T00:00:02.100000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::list: indirect execution dropped (first time) stride=12 address=Some("0xdead")
@ESC@[2m2026-01-02T00:00:02.200000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::native::host: command dropped without a host pipeline key=0x55
@ESC@[2m2026-01-02T00:00:02.300000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::native::host: command dropped without a host pipeline key=0x56
@ESC@[2m2026-01-02T00:00:04.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=256 recorded=1
@ESC@[2m2026-01-02T00:00:04.500000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::capture: the world's first frame (VICTORY_CAPTURE_WORLD counts from it) frame=300 draws=812
@ESC@[2m2026-01-02T00:00:06.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=320 recorded=1
@ESC@[2m2026-01-02T00:00:10.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=384 recorded=1
# stopped by --seconds
# exit code 143
LOG

printf '# commit cccc333\n# exit code 139\n' > "$dir/segv.log"

# 1. The plain summary is unchanged: these are the outputs of run-summary.sh before --compare.
/bin/bash "$script" "$dir/before.log" | sed 1d > "$dir/out"
cat > "$dir/expected" <<'OUT'
lines 14, log span 16.0s, last frame 256

-- frame rate (per frame block)
  frame    128  +    4.0s    16.0 fps
  frame    192  +    8.0s    16.0 fps
  frame    256  +   16.0s     8.0 fps
  overall frames 64..256: 12.0 fps

-- shader work
  first binds: 2
  prewarm: 8000 shaders, 30 MB of DXIL, from t=+4.3s to t=+4.3s
  no timing fields on first-bind lines in this log; busiest frames by first binds:
    frame    130  2 first binds
  refused shaders: 4 lines, unique by reason:
  at frame~130..130 (first t=+5.0s)
        2  translation: code at H: s_lshr_bN is not supported
        1  translation: code at H: a descriptor neither loaded nor built

-- dropped work / warnings (by template)
       3  victory_gpu::list: indirect execution dropped (first time) stride=N address=Some("H")

-- GPU errors
  Timeout=2
  Timeout        t=+8.0s frame~192
  Timeout        t=+8.0s frame~192

-- audio
  no audio lines

-- how it ended
  VIctory: unhandled exception 0xc0000005 at 0x1423cf683 [1, 61]
  (at +16.0s, frame~256)
  last frame reached: 256

OUT
diff "$dir/expected" "$dir/out" > "$dir/diff" || { cat "$dir/diff" >&2; fail "plain summary of before.log changed"; }
/bin/bash "$script" "$dir/after.log" | sed 1d > "$dir/out"
cat > "$dir/expected" <<'OUT'
lines 15, log span 12.0s, last frame 384

-- frame rate (per frame block)
  frame    128  +    2.0s    32.0 fps
  frame    192  +    4.0s    32.0 fps
  frame    256  +    6.0s    32.0 fps
  frame    320  +    8.0s    32.0 fps
  frame    384  +   12.0s    16.0 fps
  overall frames 64..384: 26.7 fps

-- shader work
  first binds: 1
  prewarm: 8192 shaders, 31 MB of DXIL, from t=+2.2s to t=+2.2s
  stalls with timing: 1, total 5.0 ms; slowest 10:
         5.00 ms  frame=130 stage="pixel" hash=abc
  refused shaders: 3 lines, unique by reason:
  at frame~130..130 (first t=+3.0s)
        2  pixel input: barycentrics read directly
        1  translation: code at H: s_lshr_bN is not supported

-- dropped work / warnings (by template)
       2  victory_gpu::native::host: command dropped without a host pipeline key=H
       1  victory_gpu::list: indirect execution dropped (first time) stride=N address=Some("H")

-- GPU errors
  none

-- audio
  no audio lines

-- how it ended
  no unhandled exception or watchdog line (killed by --seconds or clean exit)
  last frame reached: 384

OUT
diff "$dir/expected" "$dir/out" > "$dir/diff" || { cat "$dir/diff" >&2; fail "plain summary of after.log changed"; }

# 2. The table.
/bin/bash "$script" --compare "$dir/before.log" "$dir/after.log" > "$dir/out"
row '^commit +aaaa111 +bbbb222 +changed$'
row '^console +Series S +Series S +same$'
row '^last frame +256 +384 +\+128$'
row '^fps median +16\.0 +32\.0 +\+16\.0$'
row '^fps last +8\.0 +16\.0 +\+8\.0$'
row '^first world frame +- +300 +changed$'
row '^first binds +2 +1 +-1$'
row '^prewarm shaders +8000 +8192 +\+192$'
row '^refused lines +4 +3 +-1$'
row '^refused \(unique reason\+shader\) +3 +3 +0$'
row '^GPU errors: Timeout +2 +0 +-2$'
row '^GPU errors: OutOfMemory +0 +0 +0$'
row '^dropped: indirect execution dropped \(first time\) stride=N address=\.\.\. +3 +1 +-2$'
row '^dropped: command dropped without a host pipeline key=H +0 +2 +\+2$'
row '^ended +fault at 0x1423cf683 +time limit +changed$'
# Drop rows by the largest change, then by name (the full template, module path first); "ended" last.
[ "$(grep '^dropped: ' "$dir/out" | head -1 | cut -c1-19)" = "dropped: indirect e" ] || fail "drop rows not sorted"
[ "$(grep -E '^(dropped|ended)' "$dir/out" | tail -1 | cut -c1-5)" = ended ] || fail "ended is not the last row"

# 3. Refusals by cause: largest change first, new and gone marked, addresses stripped.
grep -A4 '^-- refusals by cause' "$dir/out" | sed 1,2d > "$dir/causes"
cat > "$dir/expected" <<'OUT'
       0      2      +2  new   pixel input: barycentrics read directly
       1      0      -1  gone  translation: code at H: a descriptor neither loaded nor built
       2      1      -1        translation: code at H: s_lshr_bN is not supported
OUT
diff "$dir/expected" "$dir/causes" > "$dir/diff" || { cat "$dir/diff" >&2; fail "refusals by cause"; }

# 4. Plain text only, and the other endings.
grep -q "$esc" "$dir/out" && fail "colour codes in --compare output"
/bin/bash "$script" --compare "$dir/after.log" "$dir/segv.log" > "$dir/out"
row '^ended +time limit +SIGSEGV +changed$'
row '^fps median +32\.0 +- +changed$'
has 'refusals by cause'
/bin/bash "$script" --compare "$dir/before.log" > "$dir/out" 2>&1 && fail "one log accepted"
has 'usage:'

echo "run-summary compare ok"
