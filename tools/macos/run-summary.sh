#!/usr/bin/env bash
# Summarise a run log: fps, shader work, dropped work, audio, GPU errors, how it ended.
# Usage: run-summary.sh [runs/<date>/victory.log]   (default: newest log under runs/)
#        run-summary.sh --self-test
set -u
export LC_ALL=C

summarise() {
  # The logger colours every line even into a file, and macOS's sed has no `\x1b` escape (it
  # matches a literal "x1b"), so the escape is built here: the timestamp-driven parts below read
  # nothing otherwise.
  esc=$(printf '\033')
  sed "s/${esc}\\[[0-9;]*m//g" "$1" | awk -v file="$1" '
function secs(ts,   h) { # ISO time of day to seconds; wraps at midnight
  h = substr(ts, 12, 2) * 3600 + substr(ts, 15, 2) * 60 + substr(ts, 18, 9)
  if (h < lastsec - 43200) day++
  lastsec = h
  return h + day * 86400
}
function norm(s) {
  gsub(/0x[0-9a-fA-F]+/, "H", s); gsub(/[0-9a-f]{12,}/, "H", s); gsub(/[0-9]+/, "N", s)
  return substr(s, 1, 150)
}
function tomsec(v,   n) {
  n = v + 0
  if (v ~ /us$/) return n / 1000
  if (v ~ /ms$/ || v !~ /[0-9]s$/) return n
  return n * 1000
}
{
  ts = ""
  if ($0 ~ /^20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]T/) { ts = $1; t = secs(ts); if (!tset) { t0 = t; tset = 1 } tlast = t; nlines++ }
  line = $0
  if (match(line, /frame=[0-9]+/)) {
    f = substr(line, RSTART + 6, RLENGTH - 6) + 0
    if (ts != "" && (line ~ /native draws so far/ || line ~ /frame presented/)) {
      if (!(f in ft)) { ft[f] = t; fts[f] = ts }
    }
    if (f > curframe && ts != "") curframe = f
  }
  if (line ~ /victory_gpu::pipeline: first bind/) {
    nbind++
    if (match(line, /(took|stall|elapsed|translat[a-z]*)[a-z_]*=[0-9.]+(us|ms|s)?/)) {
      kv = substr(line, RSTART, RLENGTH); sub(/^[^=]*=/, "", kv); ms = tomsec(kv)
      nstall++; tot += ms; stall[nstall] = ms
      desc = ""
      if (match(line, /frame=[0-9]+/)) desc = substr(line, RSTART, RLENGTH)
      if (match(line, /stage="[a-z]+"/)) desc = desc " " substr(line, RSTART, RLENGTH)
      if (match(line, /hash=[0-9a-f]+/)) desc = desc " " substr(line, RSTART, RLENGTH)
      sdesc[nstall] = desc
    }
    if (match(line, /frame=[0-9]+/)) bindf[substr(line, RSTART + 6, RLENGTH - 6) + 0]++
  }
  if (line ~ /shaders prewarmed/) {
    if (match(line, /n=[0-9]+/)) pre_n = substr(line, RSTART + 2, RLENGTH - 2) + 0
    if (match(line, /dxil_mb=[0-9]+/)) pre_mb = substr(line, RSTART + 8, RLENGTH - 8) + 0
    if (!pre_seen) { pre_seen = 1; pre_from = t - t0 }
    pre_to = t - t0
  }
  if (line ~ /title shader refused/) {
    sh = ""; rs = line
    if (match(line, /shader="[^"]*"/)) sh = substr(line, RSTART, RLENGTH)
    sub(/.*reason=/, "", rs); rs = norm(rs)
    if (!((rs SUBSEP sh) in seen)) { seen[rs SUBSEP sh] = 1; refc[rs]++ }
    if (nref == 0) { reff = curframe; reft = t - t0 }
    refl = curframe
    nref++
  }
  if (line ~ /command buffer failed/ || line ~ /out of memory/ || line ~ /OutOfMemory/) {
    kind = "Other"
    if (line ~ /ErrorHang/) kind = "Hang"
    else if (line ~ /ErrorInnocentVictim/) kind = "InnocentVictim"
    else if (line ~ /ErrorTimeout/) kind = "Timeout"
    else if (line ~ /ErrorOutOfMemory|[Oo]ut of memory|OutOfMemory/) kind = "OutOfMemory"
    gk[kind]++
    if (++ngpu <= 12) gl[ngpu] = sprintf("%-14s t=+%.1fs frame~%d", kind, lastt - t0, curframe)
  }
  if (ts != "") lastt = t
  if (line !~ /title shader refused|frame targets|dstorage|BacktraceBreakpad/ &&
      line ~ /dropped|refused|not in a mapped pool|not mirrored|out of range|drops work|not implemented|not represent|without shaders|pool_corrupt|work held|no program/) {
    x = line
    if (ts != "") { sub(/^[^ ]+ +[A-Z]+ +/, "", x) }
    x = norm(x); drop[x]++
  }
  if (line ~ /unhandled exception/) { exc = line; excf = curframe; exct = lastt - t0 }
  if (line ~ /no audio device: sound is mixed and dropped/) {
    naud++
    aerr = line; sub(/.*error=/, "", aerr)
    # norm() without its long-hex rule (an interval expression that some
    # awks read differently): an error name has no long hex runs in it.
    gsub(/0x[0-9a-fA-F]+/, "H", aerr); gsub(/[0-9]+/, "N", aerr)
    if (naud <= 6) aud[naud] = sprintf("output fell back to silence t=+%.1fs frame~%d (%s)", lastt - t0, curframe, aerr)
  }
  if (line ~ /spatial audio stream activated/) nspat++
  if (line ~ /render client released frames in the last second/) nmix++
  if (line ~ /audio DSP connected/) ndsp++
  if (line ~ /audio DSP stream activated/) ndstream++
  if (line ~ /[Ww]atchdog|[Kk]illed|[Tt]erminated|timed out|time limit/ && line !~ /BacktraceBreakpad|GPU Timeout|dstorage|command buffer failed/) {
    if (++nend <= 3) endl[nend] = substr(line, 1, 160)
  }
  if (line ~ /profile at [0-9.]+s:/) { np = 0; prof = line }
  else if (line ~ /[ ]profile .* calls [0-9]+ total/) { np++; pl[np] = line }
}
function top(arr, n, outn, idx,   i, j, best, bi, used) { # selection sort of arr[1..n] indices by value
  for (i = 1; i <= outn && i <= n; i++) {
    best = -1; bi = 0
    for (j = 1; j <= n; j++) if (!(j in used) && arr[j] > best) { best = arr[j]; bi = j }
    idx[i] = bi; used[bi] = 1
  }
  return i - 1
}
END {
  printf "== %s\n", file
  printf "lines %d, log span %.1fs, last frame %d\n\n", nlines, tlast - t0, curframe

  print "-- frame rate (per frame block)"
  n = 0; for (f in ft) fr[++n] = f + 0
  for (i = 2; i <= n; i++) { v = fr[i]; for (j = i - 1; j >= 1 && fr[j] > v; j--) fr[j + 1] = fr[j]; fr[j + 1] = v }
  if (n < 2) print "  (fewer than two frame counters)"
  step = (n > 41) ? int(n / 40) + 1 : 1
  pf = 0
  for (i = 1; i <= n; i++) {
    f = fr[i]
    if (i > 1 && i % step != 0 && i != n) continue
    if (pf && ft[f] > ft[pf]) printf "  frame %6d  +%7.1fs  %6.1f fps\n", f, ft[f] - t0, (f - pf) / (ft[f] - ft[pf])
    pf = f
  }
  if (n >= 2) printf "  overall frames %d..%d: %.1f fps\n", fr[1], fr[n], (fr[n] - fr[1]) / (ft[fr[n]] - ft[fr[1]] + 1e-9)
  print ""

  print "-- shader work"
  printf "  first binds: %d\n", nbind
  if (pre_seen) printf "  prewarm: %d shaders, %d MB of DXIL, from t=+%.1fs to t=+%.1fs\n", pre_n, pre_mb, pre_from, pre_to
  if (nstall) {
    printf "  stalls with timing: %d, total %.1f ms; slowest 10:\n", nstall, tot
    k = top(stall, nstall, 10, ix)
    for (i = 1; i <= k; i++) printf "    %9.2f ms  %s\n", stall[ix[i]], sdesc[ix[i]]
  } else {
    print "  no timing fields on first-bind lines in this log; busiest frames by first binds:"
    nb = 0; for (f in bindf) { bf[++nb] = bindf[f]; bk[nb] = f }
    k = top(bf, nb, 5, ix)
    for (i = 1; i <= k; i++) printf "    frame %6d  %d first binds\n", bk[ix[i]], bf[ix[i]]
  }
  nr = 0; for (r in refc) nr++
  printf "  refused shaders: %d lines, unique by reason:\n", nref
  if (nref) printf "  at frame~%d..%d (first t=+%.1fs)\n", reff, refl, reft
  nr = 0; for (r in refc) { rc[++nr] = refc[r]; rk[nr] = r }
  k = top(rc, nr, 12, ix)
  for (i = 1; i <= k; i++) printf "    %5d  %s\n", rc[ix[i]], rk[ix[i]]
  print ""

  print "-- dropped work / warnings (by template)"
  nd = 0; for (d in drop) { dc[++nd] = drop[d]; dk[nd] = d }
  k = top(dc, nd, 20, ix)
  if (!k) print "  none"
  for (i = 1; i <= k; i++) printf "  %6d  %s\n", dc[ix[i]], dk[ix[i]]
  print ""

  print "-- GPU errors"
  if (!ngpu) print "  none"
  else {
    s = ""; for (kd in gk) s = s " " kd "=" gk[kd]; print " " s
    for (i = 1; i <= ngpu && i <= 12; i++) print "  " gl[i]
    if (ngpu > 12) printf "  ... %d more\n", ngpu - 12
  }
  print ""

  print "-- audio"
  if (!naud && !nspat && !nmix && !ndsp) print "  no audio lines"
  else {
    for (i = 1; i <= naud && i <= 6; i++) print "  " aud[i]
    if (naud > 6) printf "  ... %d more\n", naud - 6
    if (!naud) print "  the device kept playing (no fallback to silence)"
    printf "  mixer seconds: %d, spatial stream activations: %d, DSP connects: %d, DSP streams: %d\n", nmix, nspat, ndsp, ndstream
  }
  print ""

  print "-- how it ended"
  if (exc != "") printf "  %s\n  (at +%.1fs, frame~%d)\n", substr(exc, 1, 120), exct, excf
  for (i = 1; i <= nend && i <= 3; i++) print "  " endl[i]
  if (exc == "" && !nend) print "  no unhandled exception or watchdog line (killed by --seconds or clean exit)"
  printf "  last frame reached: %d\n\n", curframe

  if (prof != "") {
    print "-- VICTORY_PROFILE (last dump: " substr(prof, index(prof, "profile at"), 40) ")"
    for (i = 1; i <= np && i <= 10; i++) { x = pl[i]; sub(/^[^ ]+ +[A-Z]+ +[^ ]+ +/, "", x); print "  " substr(x, 1, 140) }
  }
}'
}

self_test() {
  local f out ok=1 esc
  f=$(mktemp) || exit 1
  # Synthetic log, made up for this test. Its lines wear the colours the logger writes even into a
  # file (`ESC[2m` … `ESC[0m`); without stripping them the timestamps below read as nothing, so
  # this also covers the strip.
  esc=$(printf '\033')
  sed "s/@ESC@/${esc}/g" > "$f" <<'LOG'
# fake header
@ESC@[2m2026-01-01T00:00:00.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=64 recorded=1
@ESC@[2m2026-01-01T00:00:02.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=128 recorded=1
@ESC@[2m2026-01-01T00:00:03.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::pipeline: first bind pipeline=0x1 frame=100 stage="pixel" size=1 hash=abc took=12.5ms
@ESC@[2m2026-01-01T00:00:03.050000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::pipeline: shaders prewarmed n=8192 pending=8 dxil_mb=31
@ESC@[2m2026-01-01T00:00:03.100000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::native::shaders: title shader refused; its work is dropped shader="vertex aa pixel bb" reason=translation: code at 0x10: a thing
@ESC@[2m2026-01-01T00:00:03.200000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::list: indirect execution dropped (first time) stride=12 address=Some("0xdead")
@ESC@[2m2026-01-01T00:00:03.300000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_gpu::list: indirect execution dropped (first time) stride=16 address=Some("0xbeef")
metal: command buffer failed: NSError { code: 2, "Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)" }
@ESC@[2m2026-01-01T00:00:03.500000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_hle::host_posix::coreaudio: no audio device: sound is mixed and dropped error=AudioQueueStart failed with OSStatus 2003329396
@ESC@[2m2026-01-01T00:00:04.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_hle::host_posix::wasapi::spatial: WASAPI: spatial audio stream activated static_mask=6
@ESC@[2m2026-01-01T00:00:04.500000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_hle::host_posix::wasapi: WASAPI: render client released frames in the last second frames=48000
@ESC@[2m2026-01-01T00:00:04.000000Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_hle::x: profile at 4.0s: 1 methods
@ESC@[2m2026-01-01T00:00:04.000100Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_hle::x: profile k!f0x1 calls 3 total 9.0 ms max 4.00 ms
VIctory: unhandled exception 0xc0000005 at image+0x1234 [0]
LOG
  out=$(summarise "$f")
  rm -f "$f"
  chk() { printf '%s\n' "$out" | grep -q -- "$1" || { echo "self-test FAIL: missing '$1'"; ok=0; }; }
  chk '32.0 fps'; chk '12.50 ms  frame=100 stage="pixel"'; chk '2 .*indirect execution dropped'
  chk 'Hang=1'; chk 'unhandled exception 0xc0000005 at image+0x1234'; chk 'last frame reached: 128'
  chk 'calls 3 total 9.0 ms'; chk '1  translation: code at H: a thing'
  chk 'prewarm: 8192 shaders, 31 MB of DXIL, from t=+3.0s to t=+3.0s'
  chk 'at frame~128..128 (first t=+3.1s)'
  chk 'output fell back to silence t=+3.5s frame~128 (AudioQueueStart failed with OSStatus N)'
  chk 'mixer seconds: 1, spatial stream activations: 1, DSP connects: 0, DSP streams: 0'
  [ "$ok" = 1 ] && echo "self-test ok" || { printf '%s\n' "$out"; exit 1; }
}

case "${1:-}" in
  --self-test) self_test; exit ;;
  -h|--help) sed -n 2,4p "$0"; exit 0 ;;
esac
log=${1:-}
if [ -z "$log" ]; then
  root=$(cd "$(dirname "$0")/../.." && pwd)
  log=$(ls -t "$root"/runs/*/victory.log 2>/dev/null | head -1)
  [ -n "$log" ] || { echo "no runs/*/victory.log found" >&2; exit 2; }
fi
[ -f "$log" ] || { echo "not a file: $log" >&2; exit 2; }
summarise "$log"
