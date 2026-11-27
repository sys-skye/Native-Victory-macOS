# Working on Native VIctory for macOS (for AI coding agents and humans)

A native macOS port of VIctory, the Xbox Series emulator for GTA VI. The game's x86-64 code runs in our own process under Rosetta. Its Windows/Xbox system calls are reimplemented in Rust (`crates/victory-hle`). Its Direct3D 12 work and GFX10 shaders go to Metal (`crates/victory-gpu`, `crates/victory-metal`, `crates/victory-shader`). Deep notes: `docs/native-host.md`. Open problems in priority order: `docs/ROADMAP.md`.

## Hard rules

- **Nothing from the game goes in the repo:** no executables, data, shaders, disassembly listings, screenshots or audio. `cargo xtask policy` (also a CI test) enforces an allowlist of file types and blocks embedded binary data. Dumps and traces go to a folder outside the repo.
- Don't vendor Apple's Metal Shader Converter headers or libraries.
- Don't send game files or bytes to any external service or model.
- No AI attribution lines (`Co-Authored-By`, "Generated with ...") in commits or PRs.
- Guesses about undocumented console behaviour are marked in the code and explained in `docs/native-host.md`.
- New diagnostics are off by default, behind a `VICTORY_*` environment variable, and documented.

## Build and check

The toolchain is pinned in `rust-toolchain.toml` (1.98.1). Put its own `bin` first on PATH, or Homebrew's rustc breaks the build with "can't find crate for std":

```bash
export PATH="$(rustup run 1.98.1 rustc --print sysroot)/bin:$PATH"
cargo build --release --target x86_64-apple-darwin -p victory
```

Before opening a PR, run what CI runs (CI is Windows, so lint for a Windows target):

```bash
cargo test -p victory-hle -p victory -p victory-gpu -p xtask
cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnullvm -- -D warnings
cargo xtask policy
```

The victory-gpu tests crash when run as x86_64 under Rosetta, so run them on the native arm64 host (no `--target`).

## Running the game

You need your own copy of the game. Windowed, for a person: `tools/macos/run-native.sh <game folder>`. Headless, for agents and scripts: `--headless --seconds N`. Logs go to `runs/<date>/victory.log`. `tools/macos/run-summary.sh [log]` summarises a run log (fps per 64 frames, shader stalls and refusals, dropped work, audio, GPU errors, how it ended). `run-summary.sh --compare <before.log> <after.log>` prints a before/after table of two runs for an issue. Details and diagnostics are in the `run-game` and `debug-graphics` skills (`.claude/skills/`).

- Run one game at a time. If several agents share a machine, serialise runs with a lock: `/usr/bin/lockf -k <some>/game.lock timeout 150 nice <command>`.
- Keep builds polite: `-j6` or fewer, under `nice`.
- The game saves its setup choices in `~/.cache/victory/data/7952987A/storage`. Once setup has been completed, runs skip the setup screens; delete that folder to see them again.

## Checking a translator change on a shader corpus

A change under `crates/victory-shader` can be checked against every shader the game registers, without running it. In the tree before the change and in the tree after it: `cargo run --release -p victory-shader --example sweep -- <corpus folder> -j 4 > <outside the repo>/<name>.tsv`. Then `tools/corpus/compare.sh <before>.tsv <after>.tsv` lists the shaders newly translated, the regressions, those whose output changed and those refused for another reason (`--self-test` checks the script). No regression is the bar; explain every changed output.

- The corpus is your own dump (`VICTORY_DUMP_ALL_SHADERS=1 VICTORY_DUMP_REFUSED=<folder>` on a run): game data. It and the sweep files stay outside the repo and are never committed, attached or sent anywhere; quote counts, shader hashes and refusal messages.
- One target directory per tree, and `-j 4` or fewer for a whole corpus on a shared machine. Details: `docs/native-host.md`, "Sweeping a shader corpus".

## Where things are

| Area | Code |
|---|---|
| Loader, threads, SEH, TEB | `crates/victory-loader/src/native/` |
| System calls (HLE) | `crates/victory-hle/src/modules/` (Xbox/GDK APIs), `src/host_posix/` (Win32, UCRT, WASAPI, XAudio2 on macOS) |
| Input | `crates/victory-hle/src/modules/gameinput.rs`, `apps/victory/src/window.rs` |
| GPU host, pipelines, textures | `crates/victory-gpu/src/native/` |
| Shader translation (GFX10 to DXIL) | `crates/victory-shader/src/emit/` |
| Metal backend | `crates/victory-metal/src/` |
| Setup and run scripts | `tools/macos/` |

## How to work

0. If several agents share this machine or repository, read [docs/multi-agent.md](docs/multi-agent.md) first (branch naming, claiming issues, the game-run lock).

Before any work, book it (details in `CONTRIBUTING.md`, "Claim it before you start"):

- Find or open the issue for the task. Skip it if it has an assignee or the `claimed` label, or if `gh pr list --state open --search "<issue number> in:title,body"` shows a pull request for it.
- Comment `/claim` on the issue, then open a draft pull request with `Refs #<issue>` as soon as you have a first commit. A claim with no pull request after 24 hours expires.
- When the pull request is up and green, pick the next task the same way. Never start a second task on an issue someone else holds.

1. Reproduce with a headless run and the trace variables. Read the log and the image dumps.
2. Hold one hypothesis at a time and prove it with a trace or dump before changing code.
3. Fix it in the shared code, with a unit test that fails without the fix.
4. Update `docs/native-host.md` (what you learned) and, if the status changes, `docs/ROADMAP.md` and the README.
5. Open a PR against `main`. Every PR needs a passing CI run and the maintainer's review.
