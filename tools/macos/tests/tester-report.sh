#!/bin/bash
# tester-report.sh on a synthetic log, with the Mac's tools stubbed so it runs the same on CI.
# Run: /bin/bash tools/macos/tests/tester-report.sh
set -euo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd -P)
root=$(mktemp -d "${TMPDIR:-/tmp}/tester report.XXXXXX")
root=$(cd "$root" && pwd -P)
trap 'rm -rf "$root"' EXIT
fail() { echo "FAIL: $*" >&2; cat "$root/output" >&2; exit 1; }

# Made-up identity; every value must be gone from the report.
user=fakeuser
home="$root/Users/$user"
host=fakehost.local
computer="Fake's MacBook Pro"
local_host=Fakes-MacBook-Pro
mkdir -p "$home" "$root/bin"
stub() { printf '#!/bin/bash\n%s\n' "$2" > "$root/bin/$1"; chmod +x "$root/bin/$1"; }
stub hostname "echo $host"
stub id "echo $user"
stub scutil "case \$2 in ComputerName) echo \"$computer\" ;; LocalHostName) echo $local_host ;; *) exit 1 ;; esac"
stub sysctl 'case $2 in machdep.cpu.brand_string) echo "Apple M9 Test" ;; hw.memsize) echo 17179869184 ;; esac'
stub sw_vers 'case $1 in -productName) echo macOS ;; -productVersion) echo 99.1 ;; esac'

# Synthetic log in the shape run-native.sh writes, with the logger's colours (@ESC@).
log="$root/victory.log"
sed -e "s/@ESC@/$(printf '\033')/g" -e "s|@HOME@|$home|g" -e "s/@HOST@/$host/g" \
  -e "s/@COMPUTER@/$computer/g" -e "s/@LOCAL@/$local_host/g" -e "s/@USER@/$user/g" > "$log" <<'LOG'
# macOS 99.1, Apple M9 Test, 16 GB, console=series-s
# commit 0000000
console        Series S
@ESC@[2m2026-01-01T00:00:00.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory::run: image path=@HOME@/Games/game_scarlett_master.exe user=@USER@
@ESC@[2m2026-01-01T00:00:00.100000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_metal::msc: metal: shader converter 3.0.4 (from VICTORY_MSC_DYLIB)
@ESC@[2m2026-01-01T00:00:00.100100Z@ESC@[0m  @ESC@[33mWARN@ESC@[0m victory_metal::msc: metal: the shader converter is not /usr/local/lib/libmetalirconverter.dylib; a copy other than Apple's package may convert shaders differently
@ESC@[2m2026-01-01T00:00:01.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_hle::host: host=@HOST@ name="@COMPUTER@" bonjour=@LOCAL@.local release build
@ESC@[2m2026-01-01T00:00:02.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=64 recorded=1
@ESC@[2m2026-01-01T00:00:04.000000Z@ESC@[0m  @ESC@[32mINFO@ESC@[0m victory_gpu::native::glue: native draws so far frame=128 recorded=1
LOG
cp "$log" "$root/original"

export HOME="$home" USER="$user" LOGNAME="$user" PATH="$root/bin:/usr/bin:/bin"
export VICTORY_MSC_DYLIB="$home/lib/libmetalirconverter.dylib"
/bin/bash "$repo/tools/macos/tester-report.sh" "$log" > "$root/output" 2>&1 || fail "exited $?"

cmp -s "$log" "$root/original" || fail "the original log changed"
gz=$(sed -n 's/^Scrubbed log: //p' "$root/output")
sha=$(sed -n 's/^sha256: //p' "$root/output")
[ -f "$gz" ] || fail "no report file: $gz"
case "$gz" in "$home/victory-reports/"*) ;; *) fail "report not in ~/victory-reports: $gz" ;; esac
[ "$(shasum -a 256 "$gz" | awk '{print $1}')" = "$sha" ] || fail "printed sha256 does not match $gz"
gzip -dc "$gz" > "$root/scrubbed"
sed '1,/^----- issue body -----$/d' "$root/output" > "$root/body"

for secret in "$home" "$user" "$host" "$computer" "$local_host"; do
  for f in scrubbed body; do
    if grep -F -- "$secret" "$root/$f" >/dev/null; then fail "'$secret' survived in the $f"; fi
  done
done
grep -F 'path=<home>/Games/game_scarlett_master.exe user=<user>' "$root/scrubbed" >/dev/null || fail "home or user placeholder missing"
grep -F 'host=<host> name="<host>" bonjour=<host>.local release build' "$root/scrubbed" >/dev/null || fail "host placeholders or word boundaries wrong"

# Every template section, and the facts the maintainer asks for.
labels=$(sed -n 's/^ *label: *//p' "$repo/.github/ISSUE_TEMPLATE/tester-report.yml")
[ -n "$labels" ] || fail "no labels in the template"
while IFS= read -r label; do
  grep -Fx "### $label" "$root/body" >/dev/null || fail "body lacks section: $label"
done <<< "$labels"
for want in 'Apple M9 Test, 16 GB, macOS 99.1' 'Console: series-s' 'Shader converter: 3.0.4 (from VICTORY_MSC_DYLIB) (warned' \
  'VICTORY_MSC_DYLIB=<home>/lib/libmetalirconverter.dylib' '<pick one: Setup or build failed / ' \
  'last frame reached: 128' "$(basename "$gz") (sha256 $sha)" '0000000 (from the log; the checkout is now at'; do
  grep -F -- "$want" "$root/body" >/dev/null || fail "body lacks: $want"
done

# A report folder inside the repo is refused.
if /bin/bash "$repo/tools/macos/tester-report.sh" --out "$repo/tools" "$log" > "$root/output" 2>&1; then
  fail "wrote a report inside the repo"
fi
grep -F 'outside the repo' "$root/output" >/dev/null || fail "no reason given for refusing the folder"
echo "PASS: tester-report"
