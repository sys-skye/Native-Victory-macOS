# Roadmap and open problems

Updated 27 November 2026. Each problem lists what is already known, so you don't have to rediscover it. The deep detail, with addresses and every diagnostic switch, is in [native-host.md](native-host.md).

**Where it stands:** the game boots, plays its boot movie, shows the start screen and every setup screen, loads story mode and reaches the prologue. It crashes a few seconds into the prologue. Getting the prologue to play through is the top priority.

## 1. The prologue crash (top priority, [#15](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/15))

Two problems stop it, in this order:

1. **A stall on the first world frame trips the game's own watchdog.**
   - At the first world frame, the game binds about 1,100 pipelines for the first time.
   - About 69 of them are large compute shaders. Each took 0.2–0.6 s to translate, almost all of it in structurisation (`range::build`), and is then refused. Each was done in turn on the game's thread.
   - The game's watchdog fires 30 s into the frame: a deliberate null write at `image+0x23cf683`, then the fault at `image+0x55e56f8`.
   - **Fixed on branch `fix/prologue-crash`:** every shader is translated ahead on host worker threads as the game creates it (`VICTORY_NO_SHADER_PREWARM=1` turns it off). Being checked for memory cost before it lands.
   - A slowdown in queue submission after merging #76 (resources named at every draw) is being fixed on `fix/fps-decay`.
2. **Pointers read from data no shader filled in.**
   - Once in the world, the game reads arrays of *empty* half-float bounding boxes as pointers: `0xfc007c007c007c00`, `0xfc00fc00fc007c00`, or the float32 form `0x7f8000007f800000`. It then reads null+0x10 at `image+0x5592694`.
   - In both runs that showed it, a Metal assertion came just before: "bytes_per_row >= used_bytes_per_row". It followed the first use of a compute shader refused for "a cross-lane operation between host waves in divergent code".
   - **Leading guess (not proven):** a refused shader's output buffer keeps its initial empty boxes, and the CPU reads it.
   - `VICTORY_TRACE_OWNERS=1` names who owns the memory, and `VICTORY_DUMP_REFUSED=<dir>` dumps the refused shaders.
   - **Ruled out:** linear root views (those shaders go through the page table), pool pages owned twice (fixed by #108; a pool audit finds 0 differences), and GPU command buffers that fail (their signals are now paid on the CPU).

**Start here:** `crates/victory-gpu/src/native/shaders.rs`, `crates/victory-gpu/src/pipeline.rs`, `crates/victory-metal/src/gfx.rs`, and the run-game skill. `tools/macos/run-summary.sh <log>` shows how a run ended. `tools/macos/tester-report.sh` turns a log into a scrubbed attachment and a tester-report body.

## 2. Frame rate and memory in the world

- In the world, fps falls from about 15 to about 7 over 4 minutes. Something grows every frame; host texture copies are the main suspect (one tester counted 26,000 by frame 4096). Being investigated on `fix/fps-decay`.
- Series X can run out of GPU memory at the world load, even on 24 GB ([#8](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/8), [#79](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/79), [#110](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/110)). A byte budget on host texture copies ([#294](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/294); `VICTORY_METAL_TEXTURE_BUDGET_MB`, off unless set) gives up copies nothing uses past a byte limit; whether that frees enough on 16 GB is not measured yet. On a 24 GB M4 Pro the world load stopped with the budget off because copies overlap: 8.5 GB of copies over 1.7 GB of distinct title memory (#301). Series S is the default on Macs with 24 GB or less. The native host now logs Metal's own recommended working set size alongside the title's declared budget at startup, so a tester's log carries the numbers needed to set a real threshold.
- The game rewrites about 40,000 descriptor records a frame with plain stores. Patching its store sites instead of remirroring is being measured ([#43](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/43); `VICTORY_TRACE_DESCRIPTORS=sites`).
- Textures the GPU writes and the CPU reads back (the water simulation) are copied back behind `VICTORY_TEXTURE_BACK=1`, off by default ([#46](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/46)). A world run on an M5 Pro names 22 textures at no measurable cost, then crashes before boats can be seen. Needed: a run past that crash.

## 3. Shader translation gaps ([#4](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/4))

Refused shaders drop their draws and dispatches. Unique refused shaders in the story load fell from 526 to about 150 on 27 November, and more fixes have landed since. Refusals now name their exact cause. The largest groups left:
- a descriptor loaded from title memory at an address in a root constant (bindless; needs the host to map the address to a heap index);
- a lane-mask join of an untracked mask;
- cross-lane operations and LDS shared between the two host waves of a console wave, inside real per-lane branches (needs if-conversion so both host waves always arrive);
- a whole-wave section step the simulation does not follow.

Fixed on 27 November: compute root constants, sized from the root signature's `Num32BitValues` (#114); `v_pk_fmac_f16` constants; wave64 lane-mask joins; `s_absdiff_i32`; signed BC6H; ExecuteIndirect with a count buffer. A batch on `fix/shader-refusals-2` also covers samplers built from constant bit masks, the thread-group info SGPR, and LDS behind uniform branches; it is being merged.

Since then: image stores of some channels (#94) and whole-wave float products (`WaveActiveProduct`, `v_mul_f32`).

Translated shaders, pipelines and refusals are cached on disk between runs.

**Start here:** `crates/victory-shader/src/emit/`. Tests use hand-assembled GFX10 code (`gfx_refusal_tests.rs`), so a refusal shape can be fixed without the game.

## 4. Running without Rosetta (arm64 recompilation)

The whole game executable (about 440,000 functions) lifts to C in seconds and compiles to arm64 (about 80 minutes at 6 jobs). A contributor's branch ([#115](https://github.com/LakshanSivakumar/Native-Victory-macOS/pull/115)) boots it to the intro at 30 fps. It is being tested here, and it is 85 commits behind main.

## 5. Everything after that

- Reverb: the audio DSP's convolution has a synthetic CPU fallback behind `VICTORY_GUESS_DSP=1` (a fixed decaying-noise tail, not the title's own impulse response); default behaviour is unchanged (silence) until a real run judges whether it sounds right ([#17](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/17)). The hardware's impulse-response layout is still undocumented; `VICTORY_TRACE_DSP=1` logs the title's own `ir` data to help settle it.
- The Brightness logo draws as half a rectangle ([#16](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/16)).
- Mesh-shader pipelines: direct and indirect mesh dispatches run on the native host (#353 for the direct ones), under a guessed record layout and a guessed `DispatchMeshX` group count.
- Culling, depth, stencil and write mask in guessed pipelines are still partly guessed.

## Solved recently

- **26–27 November:**
  - The Metal GPU timeout at about 53 s ([#2](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/2)): waits stay on the CPU until the GPU work is observed done.
  - Failed command buffers now still signal their fences ([#80](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/80)).
  - The black window on M1/M2 ([#26](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/26), #91).
  - The boot movie's green frames and stutter.
  - Pool pages owned twice after `VirtualFree` (#108); physical page numbers reused (#124); decommit keeps page views (#126).
- **Earlier:**
  - Black content draws.
  - The present pass.
  - Boxed glyphs (#6).
  - The "press A" start screen, now with keyboard and controller input.
  - The story-load deadlock.

## Good first contributions

- Run the game on your Mac and file a tester report; different chips and RAM sizes help, especially runs that reach the prologue.
- Pick a refused shader shape from section 3. It's self-contained and testable without the game.
- Turn a finding in native-host.md into a unit test.
- Improve `tools/macos/native-setup.sh` and `run-native.sh` for setups we haven't seen.
