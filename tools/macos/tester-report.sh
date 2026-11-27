#!/bin/bash
# Make a tester report: a scrubbed, gzipped copy of a run log and an issue body to paste.
# Usage: tester-report.sh [--out DIR] [runs/<date>/victory.log]   (default: newest log under runs/)
# Copies the log with your home folder, user name and host names replaced by placeholders (the
# original is not changed), gzips the copy into DIR (default ~/victory-reports), and prints its
# path, its sha256 and a body for the "Tester report" issue template. It never uploads anything:
# read the body, then attach the .gz file to the issue yourself.
set -u
export LC_ALL=C # match non-ASCII host names byte for byte

root=$(cd "$(dirname "$0")/../.." && pwd -P)
out="$HOME/victory-reports"
log=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) sed -n 2,7p "$0"; exit 0 ;;
    --out) [ -n "${2:-}" ] || { echo "--out requires a folder" >&2; exit 2; }; out="$2"; shift 2 ;;
    --out=*) out="${1#--out=}"; shift ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) log="$1"; shift ;;
  esac
done
if [ -z "$log" ]; then
  log=$(ls -t "$root"/runs/*/victory.log 2>/dev/null | head -1)
  [ -n "$log" ] || { echo "no runs/*/victory.log found" >&2; exit 2; }
fi
[ -f "$log" ] || { echo "not a file: $log" >&2; exit 2; }

mkdir -p "$out" && out=$(cd "$out" && pwd -P) || { echo "cannot use report folder: $out" >&2; exit 2; }
case "$out/" in "$root/"*) echo "the report folder must be outside the repo: $out" >&2; exit 2 ;; esac

# What to scrub: "value<TAB>placeholder" lines. Longest values are replaced first.
scrub_list=$(
  printf '%s\t<home>\n' "$HOME"
  printf '%s\t<host>\n' "$(hostname 2>/dev/null)" "$(scutil --get ComputerName 2>/dev/null)" \
    "$(scutil --get LocalHostName 2>/dev/null)" "$(scutil --get HostName 2>/dev/null)"
  printf '%s\t<user>\n' "$(id -un 2>/dev/null)" "${USER:-}" "${LOGNAME:-}"
)
# Literal, not regex, replacement (host names have dots); a value only matches where it is not
# part of a longer word, so a short user name like "lee" leaves "release" alone.
scrub() {
  SCRUB="$scrub_list" awk '
BEGIN {
  m = split(ENVIRON["SCRUB"], rows, "\n")
  for (i = 1; i <= m; i++) {
    t = index(rows[i], "\t"); v = substr(rows[i], 1, t - 1)
    if (t < 2 || v == "/" || (v in dup)) continue
    dup[v] = 1; n++; val[n] = v; ph[n] = substr(rows[i], t + 1)
  }
  for (i = 2; i <= n; i++) for (j = i; j > 1 && length(val[j]) > length(val[j - 1]); j--) {
    v = val[j]; val[j] = val[j - 1]; val[j - 1] = v; v = ph[j]; ph[j] = ph[j - 1]; ph[j - 1] = v
  }
}
{
  s = $0
  for (i = 1; i <= n; i++) {
    v = val[i]; o = ""
    while ((p = index(s, v)) > 0) {
      pre = (p > 1) ? substr(s, p - 1, 1) : substr(o, length(o), 1)
      post = substr(s, p + length(v), 1)
      if (pre ~ /[A-Za-z0-9]/ && v ~ /^[A-Za-z0-9]/ || post ~ /[A-Za-z0-9]/ && v ~ /[A-Za-z0-9]$/)
        o = o substr(s, 1, p + length(v) - 1)
      else
        o = o substr(s, 1, p - 1) ph[i]
      s = substr(s, p + length(v))
    }
    s = o s
  }
  print s
}'
}

tmp=$(mktemp -d "${TMPDIR:-/tmp}/tester-report.XXXXXX") || exit 1
trap 'rm -rf "$tmp"' EXIT
scrub < "$log" > "$tmp/victory.log" || exit 1
name="victory-$(date +%Y%m%d-%H%M%S)-$$.log.gz"
gzip -n -c < "$tmp/victory.log" > "$out/$name" || exit 1
sha=$(shasum -a 256 "$out/$name" | awk '{print $1}')

# Facts from the log (colour codes stripped, as run-summary.sh does).
esc=$(printf '\033')
sed "s/${esc}\\[[0-9;]*m//g" "$tmp/victory.log" > "$tmp/plain.log"
log_commit=$(awk '/^# commit /{print $3; exit}' "$tmp/plain.log")
console=$(awk 'match($0, /console=series-[sx]/){print substr($0, RSTART + 8, RLENGTH - 8); exit}
  /^console  /{sub(/^console +/, ""); print; exit}' "$tmp/plain.log")
converter=$(awk 'i = index($0, "metal: shader converter "){print substr($0, i + 24); exit}' "$tmp/plain.log")
grep -q 'the shader converter is not' "$tmp/plain.log" && converter="$converter (warned: not Apple's default copy)"

head=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)
git -C "$root" diff --quiet HEAD -- 2>/dev/null || head="$head, with local changes"
if [ -z "$log_commit" ] || [ "$log_commit" = "${head%%,*}" ]; then commit="$head"
else commit="$log_commit (from the log; the checkout is now at $head)"; fi
mac="$(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, $(sw_vers -productName) $(sw_vers -productVersion)"
options=$(awk '/options:/{o = 1; next} o && /^ *- /{sub(/^ *- /, ""); printf "%s%s", s, $0; s = " / "; next} o{exit}' \
  "$root/.github/ISSUE_TEMPLATE/tester-report.yml" 2>/dev/null)
vars=$(for v in $(compgen -e); do case "$v" in (VICTORY_*) printf '%s=%s\n' "$v" "${!v}" ;; esac; done)

{
  printf '### Mac\n\n%s\n\n' "$mac"
  printf '### Commit\n\n%s\n\n' "$commit"
  printf '### How far did it get?\n\n<pick one: %s>\n\n' "$options"
  printf '### What happened\n\n<In your own words: what you saw, what you pressed, when it stopped.>\n\n'
  printf 'Console: %s\nShader converter: %s\n' "${console:-not in the log}" "${converter:-not in the log}"
  printf 'VICTORY_* variables in this shell: %s\n\n' "${vars:-none}"
  printf '<details><summary>run-summary.sh</summary>\n\n```\n'
  /bin/bash "$root/tools/macos/run-summary.sh" "$tmp/victory.log" | sed 1d
  printf '```\n\n</details>\n\n'
  printf '### Log\n\n%s (sha256 %s), scrubbed by tools/macos/tester-report.sh\n' "$name" "$sha"
} | scrub > "$tmp/body"

echo "Scrubbed log: $out/$name"
echo "sha256: $sha"
echo "Nothing was uploaded. Read the body below, paste it into a new \"Tester report\" issue, and drag the .gz file into its Log field."
echo "----- issue body -----"
cat "$tmp/body"
