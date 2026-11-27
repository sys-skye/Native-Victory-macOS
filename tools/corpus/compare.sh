#!/usr/bin/env bash
# Tally one shader corpus sweep, or compare two (BASE then NEW).
# Usage: compare.sh BASE.tsv            outcomes per stage, refusal reasons by count
#        compare.sh BASE.tsv NEW.tsv    what changed: newly translated, REGRESSIONS (translated
#                                       before, not now), output changed, other reason now
#          --world FILE  names (<stage>_<hash>.bin, one a line) of the shaders a reference run
#                        bound: adds how many of each count are among them
#          --names TEXT  print the files whose reason (in the last sweep given) contains TEXT
#          --show N      how many names to list per group (default 12)
#        compare.sh --self-test
# A sweep is what `cargo run --release -p victory-shader --example sweep -- <folder>` prints: a
# line `<stage>_<hash>.bin \t ok|refused|panic|timeout|missing \t fingerprint or reason \t ms` per
# shader. Sweeps name a title's shaders: keep them outside the repo (docs/native-host.md,
# "Sweeping a shader corpus").
set -u
export LC_ALL=C

usage() { sed -n 2,10p "$0" >&2; exit 2; }

# compare BASE NEW WORLD MODE TEXT SHOW: NEW and WORLD may be empty; MODE is "names" or empty.
compare() {
  local base=$1 new=$2 world=$3 mode=$4 text=$5 show=$6 f
  for f in "$base" "$new" "$world"; do
    [ -z "$f" ] || [ -f "$f" ] || { echo "not a file: $f" >&2; return 2; }
  done
  # awk takes an operand of the form name=value as an assignment: a path gets a leading "./".
  path() { case "$1" in /*) printf '%s' "$1" ;; *) printf './%s' "$1" ;; esac; }
  local -a sweeps=("$base") operands=(role=base "$(path "$base")")
  [ -z "$new" ] || { sweeps+=("$new"); operands+=(role=new "$(path "$new")"); }
  [ -z "$world" ] || operands+=(role=world "$(path "$world")")
  # Every name of either sweep, sorted: the order of the comparison (awk has no sort of its own).
  cut -f1 -- "${sweeps[@]}" | sort -u |
    T_BASE=$base T_NEW=$new T_MODE=$mode T_TEXT=$text T_SHOW=$show awk -F'\t' '
function trim(s) { gsub(/^[ \t\r\n]+/, "", s); gsub(/[ \t\r\n]+$/, "", s); return s }
# A reason without what differs between two shaders refused for the same thing: the code offset,
# the numbers of a range, the "is not supported" tail.
function norm(r,   out, pre, post, n) {
  gsub(/GFX10 code at 0x[0-9a-fA-F]+: /, "", r)
  out = ""
  while (match(r, /[0-9]+ to [0-9]+/)) {
    pre = RSTART > 1 ? substr(r, RSTART - 1, 1) : ""
    post = substr(r, RSTART + RLENGTH, 1)
    if (pre ~ /[A-Za-z0-9_]/ || post ~ /[A-Za-z0-9_]/) { # part of a longer word: left as it is
      n = RSTART + index(substr(r, RSTART), " ") - 2
      out = out substr(r, 1, n); r = substr(r, n + 1)
    } else {
      out = out substr(r, 1, RSTART - 1) "N to N"; r = substr(r, RSTART + RLENGTH)
    }
  }
  r = out r
  sub(/ is not supported.*$/, "", r)
  return trim(r)
}
function stage(n,   i) { i = index(n, "_"); return i ? substr(n, 1, i - 1) : n }
function per_stage(c, key,   i, s) {
  s = ""
  for (i = 1; i <= ns; i++) if (c[key, stg[i]] > 0) s = s (s == "" ? "" : ", ") stg[i] " " c[key, stg[i]]
  return s == "" ? "none" : s
}
function sort_strings(a, n,   i, j, t) {
  for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && (a[j] "") > (t ""); j--) a[j + 1] = a[j]; a[j + 1] = t }
}
function tally(S, D, O, n, title,   i, j, k, nm, st, s, by, here, ord, line, wn, wb, wl, wk, key, rk, rc, rw, nr, used, best, p) {
  print "== " title ": " n " shaders"
  wn = 0; wk = 0; nr = 0
  for (i = 1; i <= n; i++) {
    nm = O[i]; st = stage(nm); s = S[nm]
    by[st, s]++; here[st] = 1
    if (nm in world) { wn++; if (!(s in wb)) wl[++wk] = s; wb[s]++ }
    if (s != "ok") {
      key = st SUBSEP (s == "refused" ? "" : s) SUBSEP norm(D[nm])
      if (!(key in rc)) rk[++nr] = key
      rc[key]++
      if (nm in world) rw[key]++
    }
  }
  split("ok refused panic timeout missing", ord, " ")
  for (i = 1; i <= ns; i++) {
    st = stg[i]; line = ""
    if (!(st in here)) continue
    for (j = 1; j <= 5; j++) if (by[st, ord[j]] > 0) line = line (line == "" ? "" : ", ") ord[j] " " by[st, ord[j]]
    print "   " st ": " line
  }
  if (hasworld) {
    sort_strings(wl, wk); line = ""
    for (i = 1; i <= wk; i++) line = line (i > 1 ? ", " : "") wl[i] " " wb[wl[i]]
    print "   bound in the world run: " wn " (" line ")"
  }
  print "   refusals by reason (count, of which bound in the world run, stage, reason):"
  for (k = 1; k <= 45 && k <= nr; k++) { # the most frequent first; among equals, the first met
    best = 0
    for (i = 1; i <= nr; i++) if (!(i in used) && (!best || rc[rk[i]] > rc[rk[best]])) best = i
    used[best] = 1; key = rk[best]; split(key, p, SUBSEP)
    printf "   %6d %5d  %-8s %s%s\n", rc[key], rw[key], p[1], (p[2] != "" ? p[2] ": " : ""), substr(p[3], 1, 150)
  }
}
function add(kind, detail, nm,   g, st, k) {
  g = kind SUBSEP detail
  if (!(g in gc)) { gk[++gn] = g; gkind[gn] = kind; gdet[gn] = detail; greg[gn] = (kind ~ /^REGRESSION/) }
  gm[g, ++gc[g]] = nm
  st = stage(nm); gs[g, st]++
  k = (kind ~ /^REGRESSION/) ? "REGRESSION" : kind
  kc[k]++; ks[k, st]++
  if (nm in world) { gw[g]++; kw[k]++ }
}
function before(a, b) { # the order of the detail: regressions, then by kind, the largest group first
  if (greg[a] != greg[b]) return greg[a] > greg[b]
  if ((gkind[a] "") != (gkind[b] "")) return (gkind[a] "") < (gkind[b] "")
  return gc[gk[a]] > gc[gk[b]]
}
function seen_stage(nm,   st) { st = stage(nm); if (!(st in stseen)) { stseen[st] = 1; stg[++ns] = st } }
role == "world" { t = trim($0); if (t != "") { world[t] = 1; hasworld = 1 } next }
role == "base" { if (NF >= 3) { if (!($1 in bs)) bo[++bn] = $1; bs[$1] = $2; bd[$1] = $3; seen_stage($1) } next }
role == "new" { if (NF >= 3) { if (!($1 in vs)) vo[++vn] = $1; vs[$1] = $2; vd[$1] = $3; seen_stage($1) } next }
role == "names" { if (($1 in bs) || ($1 in vs)) sorted[++sn] = $1 }
END {
  two = ENVIRON["T_NEW"] != ""; show = ENVIRON["T_SHOW"] + 0; text = ENVIRON["T_TEXT"]
  sort_strings(stg, ns)
  if (ENVIRON["T_MODE"] == "names") {
    for (i = 1; i <= sn; i++) {
      nm = sorted[i]
      if (two ? !(nm in vs) : !(nm in bs)) continue
      s = two ? vs[nm] : bs[nm]; d = norm(two ? vd[nm] : bd[nm])
      if (s != "ok" && (text == "" || index(d, text))) print nm
    }
    exit
  }
  if (!two) { tally(bs, bd, bo, bn, ENVIRON["T_BASE"]); exit }
  tally(vs, vd, vo, vn, ENVIRON["T_NEW"])
  for (i = 1; i <= sn; i++) {
    nm = sorted[i]
    # Fingerprints and reasons are compared as text: two fingerprints of digits alone would
    # otherwise compare as numbers, and equal past 15 digits.
    if (!(nm in bs) || !(nm in vs)) add("only in one sweep", "", nm)
    else if (bs[nm] == "ok" && vs[nm] == "ok") { if ((bd[nm] "") != (vd[nm] "")) add("output changed (ok before and after)", "", nm) }
    else if (bs[nm] == "ok") add("REGRESSION: translated before, now " vs[nm], norm(vd[nm]), nm)
    else if (vs[nm] == "ok") add("newly translated", "was: " norm(bd[nm]), nm)
    else {
      was = norm(bd[nm]); now = norm(vd[nm])
      if ((was "") != (now "") || (bs[nm] "") != (vs[nm] "")) add("other reason now", was "  ->  " (vs[nm] != "refused" ? vs[nm] ": " : "") now, nm)
    }
  }
  print ""; print "== changes from " ENVIRON["T_BASE"] " to " ENVIRON["T_NEW"]
  nk = split("newly translated|output changed (ok before and after)|other reason now|only in one sweep", kinds, "|")
  for (i = 1; i <= nk; i++)
    printf "   %s: %d (%s)%s\n", kinds[i], kc[kinds[i]], per_stage(ks, kinds[i]), (hasworld ? "; bound in the world run: " (kw[kinds[i]] + 0) : "")
  printf "   REGRESSIONS (translated before, refused or panicking now): %d (%s)%s\n", kc["REGRESSION"], per_stage(ks, "REGRESSION"), (hasworld ? "; bound in the world run: " (kw["REGRESSION"] + 0) : "")
  print ""; print "== detail"
  for (i = 1; i <= gn; i++) ix[i] = i
  for (i = 2; i <= gn; i++) { t = ix[i]; for (j = i - 1; j >= 1 && before(t, ix[j]); j--) ix[j + 1] = ix[j]; ix[j + 1] = t }
  for (i = 1; i <= gn; i++) {
    g = gk[ix[i]]; det = gdet[ix[i]]
    printf "   %6d  %s%s  [%s%s]\n", gc[g], gkind[ix[i]], (det != "" ? " | " substr(det, 1, 170) : ""), per_stage(gs, g), (hasworld ? ", world " (gw[g] + 0) : "")
    if (show > 0) {
      line = ""
      for (j = 1; j <= gc[g] && j <= show; j++) line = line (j > 1 ? " " : "") substr(gm[g, j], 1, length(gm[g, j]) - 4)
      print "           " line (gc[g] > show ? " ..." : "")
    }
  }
}' "${operands[@]}" role=names -
}

self_test() {
  local d out ok=1
  d=$(mktemp -d) || exit 1
  # Two sweeps and a world list made up for this test (no shader has these names); "|" stands for
  # the tab. Between them: one shader newly translated, one whose output changed (its
  # fingerprints differ in the last of 16 digits, which a comparison as numbers would miss), two
  # regressions, one refused for another reason, one refused for the same reason at another
  # offset and range (no change), and one in each sweep only.
  tr '|' '\t' > "$d/base.tsv" <<'TSV'
compute_00000000000000a1.bin|ok|1111111111111111|3
compute_00000000000000a2.bin|refused|GFX10 code at 0x40: a made-up construct is not supported|1
compute_00000000000000a3.bin|refused|GFX10 code at 0x8: a made-up range, dwords 2 to 3|1
pixel_00000000000000b1.bin|ok|9999999999999991|2
pixel_00000000000000b2.bin|ok|3333333333333333|2
pixel_00000000000000b3.bin|refused|GFX10 code at 0x10: s_made_up_b32 is not supported|0
vertex_00000000000000c1.bin|ok|4444444444444444|5
vertex_00000000000000c2.bin|panic|a made-up panic|5
TSV
  tr '|' '\t' > "$d/new.tsv" <<'TSV'
compute_00000000000000a1.bin|ok|1111111111111111|3
compute_00000000000000a2.bin|ok|5555555555555555|4
compute_00000000000000a3.bin|refused|GFX10 code at 0x20: a made-up range, dwords 4 to 5|1
pixel_00000000000000b1.bin|ok|9999999999999992|2
pixel_00000000000000b2.bin|refused|GFX10 code at 0x4: another made-up construct is not supported|1
pixel_00000000000000b3.bin|refused|GFX10 code at 0x10: a third made-up construct|0
vertex_00000000000000c1.bin|timeout|no result after 120 s|120000
vertex_00000000000000c3.bin|ok|6666666666666666|1
TSV
  printf '%s\n' compute_00000000000000a2.bin pixel_00000000000000b1.bin pixel_00000000000000b2.bin > "$d/world.txt"
  chk() { printf '%s\n' "$out" | grep -qF -- "$1" || { echo "self-test FAIL: missing '$1'"; ok=0; }; }
  no() { printf '%s\n' "$out" | grep -qF -- "$1" && { echo "self-test FAIL: unexpected '$1'"; ok=0; }; }

  out=$(compare "$d/base.tsv" "" "$d/world.txt" "" "" 12)
  chk ': 8 shaders'; chk '   compute: ok 1, refused 2'; chk '   pixel: ok 2, refused 1'; chk '   vertex: ok 1, panic 1'
  chk 'bound in the world run: 3 (ok 2, refused 1)'
  chk '        1     1  compute  a made-up construct'; chk '        1     0  compute  a made-up range, dwords N to N'
  chk '        1     0  pixel    s_made_up_b32'; chk '        1     0  vertex   panic: a made-up panic'

  out=$(compare "$d/base.tsv" "$d/new.tsv" "$d/world.txt" "" "" 12)
  chk '   compute: ok 2, refused 1'; chk '   vertex: ok 1, timeout 1'
  chk '   newly translated: 1 (compute 1); bound in the world run: 1'
  chk '   output changed (ok before and after): 1 (pixel 1); bound in the world run: 1'
  chk '   other reason now: 1 (pixel 1); bound in the world run: 0'
  chk '   only in one sweep: 2 (vertex 2); bound in the world run: 0'
  chk '   REGRESSIONS (translated before, refused or panicking now): 2 (pixel 1, vertex 1); bound in the world run: 1'
  chk '        1  REGRESSION: translated before, now refused | another made-up construct  [pixel 1, world 1]'
  chk '        1  REGRESSION: translated before, now timeout | no result after 120 s  [vertex 1, world 0]'
  chk '        1  newly translated | was: a made-up construct  [compute 1, world 1]'
  chk '        1  other reason now | s_made_up_b32  ->  a third made-up construct  [pixel 1, world 0]'
  chk '           vertex_00000000000000c2 vertex_00000000000000c3'
  no 'compute_00000000000000a3'
  # The regressions lead the detail.
  [ "$(printf '%s\n' "$out" | sed -n '/^== detail/{n;p;}' | cut -c1-21)" = "        1  REGRESSION" ] || { echo "self-test FAIL: the detail does not start with the regressions"; ok=0; }

  out=$(compare "$d/base.tsv" "$d/new.tsv" "" "" "" 1)
  chk '   only in one sweep: 2 (vertex 2)'; chk '           vertex_00000000000000c2 ...'; no 'bound in the world run:'; no ', world '

  out=$(compare "$d/base.tsv" "$d/new.tsv" "" names "made-up construct" 12)
  [ "$out" = "pixel_00000000000000b2.bin
pixel_00000000000000b3.bin" ] || { echo "self-test FAIL: --names over two sweeps gave '$out'"; ok=0; }
  out=$(compare "$d/base.tsv" "" "" names "dwords N to N" 12)
  [ "$out" = "compute_00000000000000a3.bin" ] || { echo "self-test FAIL: --names over one sweep gave '$out'"; ok=0; }

  rm -rf "$d"
  [ "$ok" = 1 ] && echo "self-test ok" || exit 1
}

base='' new='' world='' mode='' text='' show=12
while [ $# -gt 0 ]; do
  case "$1" in
    --self-test) self_test; exit ;;
    -h|--help) sed -n 2,14p "$0"; exit 0 ;;
    --world) [ $# -ge 2 ] || usage; world=$2; shift 2 ;;
    --names) [ $# -ge 2 ] || usage; mode=names; text=$2; shift 2 ;;
    --show) [ $# -ge 2 ] || usage; show=$2; shift 2 ;;
    -*) usage ;;
    *) if [ -z "$base" ]; then base=$1; elif [ -z "$new" ]; then new=$1; else usage; fi; shift ;;
  esac
done
[ -n "$base" ] || usage
case "$show" in ''|*[!0-9]*) usage ;; esac
compare "$base" "$new" "$world" "$mode" "$text" "$show"
