<h1 align="center">Native VIctory for macOS</h1>

<p align="center">
A <b>native</b> macOS port of <a href="https://github.com/Marvisak/VIctory">VIctory</a>, the Xbox Series emulator for GTA VI.<br>
No Wine, no CrossOver, no Game Porting Toolkit: the game's code runs in our own process, and its GPU work goes straight to Metal.
</p>

<p align="center">
<img alt="Apple Silicon M1 or newer" src="https://img.shields.io/badge/Apple%20Silicon-M1%2B-555555">
<img alt="macOS 15 or newer" src="https://img.shields.io/badge/macOS-15%2B-blue">
<img alt="Status: the prologue runs, the world is still dark and slow" src="https://img.shields.io/badge/status-prologue%20runs%2C%20dark%20and%20slow-orange">
<img alt="GPL-3.0 licence" src="https://img.shields.io/badge/licence-GPL--3.0-lightgrey">
</p>

> [!IMPORTANT]
> **Not playable yet, but it gets as far as the story.** The game boots natively on a Mac, plays its boot movie, and shows the start screen and every setup screen on Metal at about 30 fps, with sound, keyboard and controllers. Then it loads story mode and runs into the prologue's world: the prologue crash is fixed, and on our test Mac the game keeps running with the dialogue playing. The world is not watchable yet: it now reaches the window as a real picture, but far too dark, with broken pieces of geometry, at about 1 frame per second. Lighting and speed are what everyone is working on now: see [#15](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/15). **No game files are included here, ever.** You need your own legally obtained copy of the game.

📸 **[See the latest screenshots](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/5)** (the game's start screen and setup menus, rendered natively on Metal).

## Why a native port?

The [Wine-based route](https://github.com/pragyaangaur/VIctory-macOS) stacks the game on Wine, a Windows Direct3D 12 layer and Apple's D3DMetal. This project removes those layers:

| Layer | Wine route | Native route (this repo) |
| --- | --- | --- |
| Game code (x86-64) | Wine process under Rosetta | Our own process under Rosetta, or recompiled to arm64 |
| Windows and Xbox system calls | Wine + VIctory | Reimplemented in Rust for macOS (`victory-hle`, about 1,000 functions) |
| Console graphics (D3D12.X) | D3D12 → D3DMetal → Metal | Straight to Metal (`victory-gpu`) |
| Console shaders (GFX10) | → DXIL → D3DMetal | → DXIL → Apple Metal Shader Converter |

Fewer layers means fewer places for things to break, and much better speed: boot frames run at about 28 to 30 fps.

## Where we are

| Area | State |
| --- | --- |
| Loader: maps the game and its DLLs, TLS, exceptions, threads | ✅ Works |
| System calls: memory, files, threads, time, C runtime, sockets, audio (XAudio2), DirectStorage | ✅ Works for everything the boot uses |
| Sound | 🟡 Plays through CoreAudio (the game's Windows Spatial Audio stream, mixed to stereo); reverb is still silent |
| Loading: DirectStorage reads about 9,500 requests at about 79 MB/s | ✅ Works |
| GPU: every draw gets a Metal pipeline, built from the console's own pipeline descriptions | ✅ Works (some parts are inferred, see [docs/native-host.md](docs/native-host.md)) |
| Shaders: GFX10 → DXIL → Metal | 🟡 Most translate, including the world's mesh shaders. A few hundred are still refused once the world loads (most are 64-lane console waves the Mac's 32-lane GPU has to split), and a refused shader's work is skipped. A general fix for that class is being built. Shaders are cached on disk between runs |
| Boot movie and menus in the window | ✅ The boot movie plays smoothly, and the start screen and setup screens show with correct text and blending. The black window on M1/M2 is fixed ([#91](https://github.com/LakshanSivakumar/Native-Victory-macOS/pull/91)) |
| Start screen ("press A") and input | ✅ Passed; the keyboard and Game Controller pads (Xbox, DualShock 4, DualSense) drive the gamepad in the window |
| Running past ~50 s | ✅ Fixed: the old Metal GPU timeout is gone ([#2](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/2)) |
| Story mode load | ✅ Loads, and reaches the prologue |
| Prologue | 🟡 Runs into the world without crashing; the dialogue plays. The crash (GPU work that overwrote the game's memory) is fixed. With a cold shader cache, the first world frame can still take more than 30 s and trip the game's own watchdog; a second run is usually past it ([#15](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/15)) |
| World picture | 🔴 A real picture, but not a good one yet. Since 28 November the world reaches the window with the game's own colour grading, no longer flat grey (the display colour table now gets its input). It is still far too dark, and pieces of geometry are broken where draws are still dropped ([#258](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/258)). Menus, the title cards between scenes and the HUD are right |
| World frame rate and stability | 🔴 About 1 frame per second in the world. Since a guard on runaway indirect dispatches, one windowed run stayed in the world for over 3 minutes without a crash; that is one run, not yet proven stable, and runs can still hang or crash ([#79](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/79)). Stopping the game with Ctrl-C or a stop signal now drains its GPU work first, so GPU memory is not left stranded until a reboot. Most of a frame goes to copying the textures the world streams in, and to memory pressure |
| macOS 15 | 🟡 Boots and loads, but Apple's Metal compiler crashes on a few world pipelines, which holds the first world frame. Some macOS 27 Macs also report this failure ([#181](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/181)) |
| Memory on 16 to 24 GB Macs | 🟡 Use the Series S profile (the default there); Series X can run out of GPU memory at the world load. Macs with 32 GB or more keep GPU memory resident in one set by default, which is much faster; on smaller Macs you can try it with `VICTORY_METAL_RESIDENCY_SETS=heap` and tell us how it goes ([#8](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/8), [#79](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/79)) |
| Apple Silicon recompilation (no Rosetta) | 🟡 The whole game (440,000 functions) recompiles to arm64, and a contributor's branch boots it to the intro at 30 fps ([#115](https://github.com/LakshanSivakumar/Native-Victory-macOS/pull/115)); not on main yet |
| Gameplay | 🔴 Not yet |

The live list of open problems, with everything we already know about each one, is in **[docs/ROADMAP.md](docs/ROADMAP.md)**.

## How far along is the port?

**Roughly 55%**, as of 27 November 2026. This is a judgement, not a measurement: each area gets a share of the whole job and a rough score for how much of it works today.

| Area | Share of the job | Done | What is left |
| --- | --- | --- | --- |
| Running the game's code: loader, threads, exceptions, about 1,000 system calls | 15% | about 95% | Calls that only later parts of the game use |
| Sound, input, files and loading | 5% | about 85% | Reverb, and a sound device that fails in the middle of a run |
| GPU host: the console's D3D12 on Metal | 20% | about 75% | The buffer views the culling shaders read ([#258](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/258), fixes open), mesh dispatches, the parts that are still inferred |
| Shader translation: GFX10 → DXIL → Metal | 20% | about 80% | About four in five of the game's shaders translate. Of the shaders the first world frame binds, about one in five is still refused, and a refused shader's work is skipped |
| A correct world picture | 15% | about 20% | Menus, boot movie and loading screen are right. In the world the geometry and the lighting pass now produce a lit frame in the internal scene buffer; it is torn in places. The window shows the picture plane now, but the passes after lighting turn it into a flat grey |
| Speed | 10% | about 10% | Menus run at about 30 fps, the world at 1 to 3 fps. The goal is 30 |
| Stability and memory | 10% | about 20% | A world run can still end in a GPU time-out or a crash, even with 48 GB: one of our last three did. Macs with 16 to 24 GB run short of GPU memory ([#8](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/8), [#79](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/79)) |
| Gameplay: the prologue and beyond | 5% | 0% | Not started. It needs everything above |

What the number does not say: nothing is playable yet, and the remaining 45% is the harder part. The first half was getting the game to boot, load and reach the story. The second half is making the world look right, run fast and stay up, and that work finds new problems as it goes. We cannot give a date.

## Quick start

You need:

- an Apple Silicon Mac (M1 or newer) with macOS 15 or newer and about 20 GB free;
- [rustup](https://rustup.rs) (not Homebrew's `rust` package);
- Apple's free [Metal Shader Converter](https://developer.apple.com/metal/shader-converter/) 3.0 or newer (download and run the installer; older ones are refused, #295);
- your own copy of the Xbox Series game files.

**1. Get the code and build it.**

```bash
git clone https://github.com/LakshanSivakumar/Native-Victory-macOS.git
```

```bash
cd Native-Victory-macOS && tools/macos/native-setup.sh
```

The setup script checks Rosetta, the Xcode command line tools, rustup and the Metal Shader Converter. It tells you exactly what to install if anything is missing, then builds `victory`.

**2. Run the game.**

```bash
tools/macos/run-native.sh /path/to/your/game/folder
```

Point it at the folder that holds `game_scarlett_master.exe`. A window opens; after a few seconds of black while it loads, you should see the boot movie and then the menus, and the full log is saved under `runs/`. The script also presses Enter for you every two seconds until 80 s to get past the start screen. Click the window and the keyboard drives the game's gamepad:

| Keys | Pad |
| --- | --- |
| Enter, Space | A |
| Escape, Backspace | B |
| X, Y | X, Y |
| Arrow keys | D-pad |
| W A S D | left stick |
| Tab | Menu (Start) |
| Q, E | left, right shoulder |
| Z, C | left, right trigger |

**Controllers.** An Xbox pad, a PS4 DualShock 4 or a PS5 DualSense connected to the Mac (USB, or Bluetooth through System Settings → Bluetooth; on a DualShock 4 hold Share and PS until the light bar flashes to pair) drives the gamepad while the window is in front. The keyboard still works alongside it. On a PlayStation pad Cross, Circle, Square and Triangle are A, B, X and Y, Options is Menu, and Share or a touchpad click is View. With several controllers connected, the one you used last wins. The run log names the controller it picked (`controller: drives the pad`); if it says none, check that macOS sees the controller in System Settings → Game Controllers.

Add `--headless` to run without a window, or `--seconds 60` to stop after a minute.

**3. Tell us what happened.** Open a [Tester report](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/new?template=tester-report.yml) and attach the log. `tools/macos/tester-report.sh` does the work for you: it writes a copy of the newest log with your home folder, user name and Mac's name replaced, gzips it into `~/victory-reports/`, and prints the report text with your Mac, commit and a run summary filled in. Only those exact names are replaced: other paths, such as an external disk named after you (`/Volumes/Alex SSD`), are not changed, so read the file before you attach it. It never uploads anything; you paste the text and attach the `.gz` file yourself. Reports from different Macs (M1 to M5, 8 GB to 128 GB) are very useful.

## Help wanted

Whatever your background, there is something to do:

- **Testers:** run the game on your Mac and file a tester report. No coding needed.
- **Graphics and Metal:** the shader refusals that keep the world unlit ([#4](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/4)), frame rate in the world, and GPU memory use on 16 to 24 GB Macs.
- **Debugging:** runs that reach the prologue's world ([#15](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/15)). Logs from your own Mac help a lot, especially on macOS 15 and on 16 to 24 GB.
- **Compilers:** the arm64 recompiler (jump tables, SSE, exceptions).
- **Rust and systems:** missing Windows and Xbox system calls, speed, tests.

Read **[CONTRIBUTING.md](CONTRIBUTING.md)** to get started, then pick something from [docs/ROADMAP.md](docs/ROADMAP.md) or the issues labelled `good first issue`.

## Documentation

| Document | What it covers |
| --- | --- |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Open problems, what is known, where to start |
| [docs/native-host.md](docs/native-host.md) | The native host in depth: loader, system calls, findings, every diagnostic switch |
| [docs/metal-backend.md](docs/metal-backend.md) | The Metal backend and the Metal Shader Converter |
| [docs/architecture.md](docs/architecture.md) | VIctory's overall design |
| [docs/asset-policy.md](docs/asset-policy.md) | What may and may not be committed |

## Legal

This project contains no game code, assets, keys or Microsoft or Apple SDK files, and never will. Emulation is built from documentation, observation and clean-room analysis. You must supply your own legally obtained copy of the game. Do not post game files or links to them in issues or pull requests.

Not affiliated with or endorsed by Rockstar Games, Take-Two Interactive, Microsoft or Apple. All trademarks belong to their owners.

## Credits

Native VIctory for macOS builds on [VIctory](https://github.com/Marvisak/VIctory) by Marvisak and its contributors, and on [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS) by Pragyaan Gaur and its contributors. Their full history is kept in this repository. Licensed under the [GPL-3.0](LICENSE).
