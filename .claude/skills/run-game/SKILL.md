---
name: run-game
description: Build and run GTA VI on the native macOS host (windowed or headless), script its gamepad with VICTORY_KEYS, and read the run log. Use when asked to run, launch or test the game, or to check how far a change gets.
---

# Run the game natively

Needs your own copy of the game (the folder with `game_scarlett_master.exe`). Never copy any game file into the repo.

## Build

```bash
export PATH="$(rustup run 1.98.1 rustc --print sysroot)/bin:$PATH"
nice cargo build --release -j6 --target x86_64-apple-darwin -p victory
```

## Run

- Windowed, for a person: `tools/macos/run-native.sh <game folder>`. Click the window; the keyboard drives the pad (Enter/Space = A, Esc/Backspace = B, arrows = D-pad, WASD = left stick, Tab = Menu, Q/E = shoulders, Z/C = triggers).
- After a run: `tools/macos/run-summary.sh [runs/<date>/victory.log]` prints fps per 64 frames, shader refusals, dropped work, GPU errors and how the run ended (newest log by default). `run-summary.sh --compare <before.log> <after.log>` prints one before/after table of two runs (commit, fps median and last, first world frame, first binds, refusals, dropped work by kind, GPU errors, how it ended) and the refusals by cause, largest change first, with new and gone causes marked; plain text, to paste into an issue in a fenced block. A log from run-native.sh ends with `# exit code N` (and `# stopped by --seconds` when the limit stopped it). That is how the table tells a host SIGSEGV (139) from the time limit; older logs read as unknown.
- For a tester report: `tools/macos/tester-report.sh [--out DIR] [log]` writes a gzipped copy of the log with $HOME, the user name and host names replaced (to `~/victory-reports/` by default, never inside the repo; the original log is untouched), prints its path and sha256, and prints an issue body for the tester-report template (Mac, commit, console model, shader converter, VICTORY_* variables, run-summary.sh output). It uploads nothing.
- Headless, for agents: `tools/macos/run-native.sh --headless --seconds 120 <game folder>`, or the binary directly:

```bash
VICTORY_GUESS_GRAPHICS=1 VICTORY_KEYS="$(seq 4 2 80 | sed 's/$/:Enter/' | paste -sd, -)" \
  timeout 150 nice target/x86_64-apple-darwin/release/victory run --execute --headless \
  --unimplemented zero <game folder>/game_scarlett_master.exe > run.log 2>&1
```

When several agents share a machine, wrap the run in `/usr/bin/lockf -k <dir>/game.lock ...` so only one game runs at a time.

## Scripted input (`VICTORY_KEYS`)

A comma-separated list of `<seconds>:<key>` presses. Key names: a letter, a digit, `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down` and the other arrows. Enter every 2 s gets past "press A". Some setup screens need `Down` then `Enter` (the default choice isn't the one that goes forward), so build the script screen by screen and check each step with an image dump (see the debug-graphics skill).

The game saves setup choices in `~/.cache/victory/data/7952987A/storage`. Once setup has been completed, later runs skip it and go straight to loading the story. Delete that folder to see the setup screens again.

## Reading the log

- `frame presented ... frame=N`: frames keep coming (about 30 fps).
- `native draws so far ... recorded= encoded=`: equal numbers mean no draws were dropped.
- `native host drops work: ...` and unknown-format warnings: logged once per kind.
- `dstorage: last 12.5 s requests= mb_per_s=`: loading throughput. `a status entry is still pending` means a DirectStorage request isn't completing.
- `unhandled exception` or `fatal runtime error`: a crash. The lines just before it say where.

## Milestones so far

Boot movie, "press A", FINISH SETUP, Brightness, UI TEXT SIZE, LEGAL NOTICES, then the story load (currently stalls on a grey loading screen; see docs/ROADMAP.md).
