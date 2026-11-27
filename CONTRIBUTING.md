# Contributing to Native VIctory for macOS

Thanks for helping. The goal: GTA VI running natively on an Apple Silicon Mac, from a copy the player owns. There is work for testers, Rust programmers, graphics people, reverse engineers and compiler people.

## The one rule

**Nothing from the game is ever committed or posted.** That covers executables, DLLs, shaders, textures, audio, keys, dumps, decrypted files, disassembly listings and screenshots of game content. The same goes for Microsoft's GDK and Apple's SDK files (the Metal Shader Converter is installed by each user, never vendored). `cargo xtask policy` checks this. Pull requests that break it are closed. Findings such as "the boot waits on the record at this address" are fine and welcome; copied code or bytes are not.

## Set up

Follow the [Quick start](README.md#quick-start). The native probe tests build a small Windows program, and the clippy gate checks a Windows target, so also install the cross tools and that target:

```bash
brew install llvm lld mingw-w64 && rustup target add --toolchain 1.98.1 x86_64-pc-windows-gnullvm
```

Then make sure the tests pass before you change anything:

```bash
rustup run 1.98.1 cargo test --target x86_64-apple-darwin -p victory-hle -p victory-loader -p victory
```

```bash
rustup run 1.98.1 cargo test -p victory-shader -p victory-hle
```

The native host tests translate shaders and need Apple's Metal Shader Converter; this wrapper fails fast with the install steps when it is missing, instead of letting the tests panic inside the converter's `dlopen` (#379):

```bash
tools/macos/test-native.sh -p victory-hle -p victory -p victory-gpu -p xtask
```

The CI gate is a strict clippy run for Windows, which you can run on a Mac:

```bash
rustup run 1.98.1 cargo clippy --workspace --all-targets --locked --target x86_64-pc-windows-gnullvm -- -D warnings
```

## Find something to work on

- [docs/ROADMAP.md](docs/ROADMAP.md) lists the open problems, with what is already known about each.
- Issues labelled `good first issue` are small and self-contained, and `help wanted` marks where help moves things most.

## Claim it before you start

The issue tracker is the task queue. Two people have fixed the same thing on the same day more than once, so every piece of work is booked on an issue first.

1. **Check that it is free.** The issue has no assignee and no `claimed` label, and no open pull request mentions it:

   ```bash
   gh pr list --state open --search "<issue number> in:title,body"
   ```

2. **No issue for it yet?** Open one, with what is wrong and what you plan to do.
3. **Claim it.** Comment `/claim` on the issue. That assigns it to you and adds the `claimed` label. If someone else holds it, you are told so; pick another, or ask them in the thread.
4. **Open a draft pull request early**, with `Refs #<issue>` in its description, as soon as you have a first commit. The draft is what tells everyone the work is real.
5. **Claims expire.** With no pull request that mentions the issue after 24 hours, or no activity on that pull request for 72 hours, the claim is removed and the issue is free again. Comment `/release` to give one up sooner.

Then the loop is: develop, test, push, pick the next task, check that it is free, claim it, start.

One issue is one task. If an issue covers several pieces of work (a tracking issue), say in your `/claim` comment which piece you are taking, or open a smaller issue for it and claim that.

## How the code is laid out

| Crate | What it does |
| --- | --- |
| `crates/victory-loader` | Maps the game's PE files, TLS, exceptions, the Windows↔macOS calling-convention bridge (`native/`) |
| `crates/victory-hle` | The Windows and Xbox system calls, reimplemented (`host_posix/` for macOS, `modules/` for Xbox APIs) |
| `crates/victory-gpu` | Console Direct3D 12.X; `native/` is the Metal path |
| `crates/victory-shader` | GFX10 (RDNA2) shader decoding and translation to DXIL |
| `crates/victory-metal` | Metal and Metal Shader Converter glue |
| `crates/victory-platform` | Memory, threads and sync on macOS and Windows |
| `apps/victory` | The command-line front end (`victory run`) |

## Debugging a run

Every diagnostic is an environment variable, documented in [docs/native-host.md](docs/native-host.md) under "Bring-up diagnostics". A few common ones:

| Variable | What it does |
| --- | --- |
| `VICTORY_GUESS_GRAPHICS=1` | Build graphics pipelines from the console descriptions (on by default in `run-native.sh`) |
| `VICTORY_DUMP_TARGETS=<dir>,<s>,<every>` | Save render targets as PNGs |
| `VICTORY_TRACE_FAILS=1` | Log every failing system call |
| `VICTORY_DUMP_THREADS=<s>` | Dump every thread's stack after `<s>` seconds |
| `VICTORY_DEBUG_PS=in<N>` / `VICTORY_DEBUG_VS_POS=1` | Graphics bisection switches |

## Pull requests

- Keep each pull request to one change, with a test where it is testable without the game.
- Run the tests and the clippy gate above.
- Gate new diagnostics behind a `VICTORY_*` variable and document them in docs/native-host.md.
- When you guess at undocumented console behaviour, say so in a comment and in native-host.md's list of guesses.
- By contributing, you agree to license your work under the GPL-3.0.

## Be kind

Be patient with newcomers, assume good faith, and keep discussion about the code. Harassment of any kind isn't tolerated.

## Using an AI coding agent

`AGENTS.md` (read by Codex and most agents; `CLAUDE.md` imports it for Claude Code) has the build, check and run commands and the hard rules. `.claude/skills/` has step-by-step skills: `run-game`, `debug-graphics` and `add-hle-function`.
