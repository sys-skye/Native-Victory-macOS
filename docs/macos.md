# macOS port status

This file tracks the work to run VIctory on Apple Silicon Macs. No game files are used or needed for any step that is marked as done.

The Wine route below (Route A) is retired: the game now runs natively (`tools/macos/native-setup.sh`, `tools/macos/run-native.sh`, `docs/native-host.md`). Its setup and run scripts (`setup.sh`, `tools/macos/victory.sh`, `build-crossover-wine.sh`, `build-deps.sh`, `native-build-tools.sh`) and the test fleet (`tools/fleet`) were removed; this file keeps what was learned. `tools/macos/test-wine.sh` still runs the Windows tests under a Wine you supply.

## Done on 24 November 2026

The whole workspace builds natively on macOS (aarch64, Homebrew Rust 1.100). The Windows-only parts are gated with `cfg(windows)`, and a few small stubs stand in for them. The text entry dialog is cancelled on other hosts, a relaunch request stops the process, and the `translate` command reports that it needs Windows. On a Mac M4 the test suite gives 162 passed, 0 failed and 10 ignored. The ignored tests read their own test binary as a PE file or use Windows paths, so they only run on Windows.

A native build proves that the platform code (formats, shader recompiler, PE analysis and most of the HLE layer) is portable. It does not run a title, because the loader maps a Windows PE image into the process and calls Windows APIs.

## Done on 24 November 2026: the Windows build under Wine

VIctory now cross builds on the Mac for `x86_64-pc-windows-gnullvm` and runs under Wine 11.14. The `windows-gnu` target cannot be used, because Rust keeps its thread-locals in OS TLS slots there and the loader only saves and restores static TLS, so std aborted when a thread started. Homebrew has no llvm-mingw, so `tools/macos/gnullvm-cc.sh` links with Homebrew clang and lld against the mingw-w64 runtime.

Run the tests under Wine with this command.

```sh
WINE=/path/to/wine tools/macos/test-wine.sh --workspace --no-fail-fast
```

The loader probe passes, so the stand-in title is mapped, bound, delay loaded and run with its thread callbacks. The HLE tests pass (86 of 86), and `victory.exe --help` runs. Wine lacks three network exports (`WSAConnectByList`, `GetAddrInfoExA` and `WinHttpGetProxySettingsEx`), and VIctory now fails those calls cleanly. Two kinds of test still fail under Wine.

- Seven shader tests need a D3D12 device with shader model 6.6. The Wine build has no D3D12 that works on Metal, so this is the GPU gap that D3DMetal has to fill.
- The xtask policy test starts git from inside Wine. Run `cargo xtask policy` natively instead.

## D3DMetal results, 24 November 2026

With the GPTK cask (`brew install --cask gcenx/wine/game-porting-toolkit`, Wine 7.7 with the D3DMetal of GPTK 1.1), VIctory gets a D3D12 device on Metal, and all 69 shader tests pass on an M4. Metal's sin and cos return 0 for NaN and infinity, where the console GPU returns NaN, so the translator now passes that NaN through. D3DMetal does not check the DXIL container hash.

Wine 7.7 is too old for the rest of VIctory. The console memory tests (xmem and dstorage) and the display test fail on it, and it also lacks `WSAConnectByNameA`. Wine 11.14 passes all of them, but D3DMetal from GPTK 4.0 beta 3 crashes in libsystem_pthread on Wine 11.14, because D3DMetal needs Apple's or CrossOver's patched Wine. A Wine that is new enough for VIctory and also works with D3DMetal is the open problem.

```sh
WINE="/Applications/Game Porting Toolkit.app/Contents/Resources/wine/bin/wine64" WINEDLLOVERRIDES=winedbg.exe=d tools/macos/test-wine.sh --workspace --no-fail-fast
```

## CrossOver Wine with D3DMetal, 24 November 2026

`tools/macos/build-crossover-wine.sh` (since removed) built CodeWeavers' CrossOver 26.3 Wine (Wine 11.0) from its published source into `target/wine-cx`, using x86_64 FreeType and GnuTLS from `DEPS`. Copy the `redist/lib` folder of Apple's "Evaluation environment for Windows games" over `target/wine-cx/lib`, and run the tests with this command.

```sh
WINE="$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine" WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b" tools/macos/test-wine.sh --workspace --no-fail-fast
```

On an M4 with D3DMetal 4.0 beta 3 this gives 260 passed, 4 failed and 4 ignored. Every loader, HLE, console memory and shader test passes on the same Wine. The policy test fails only because it runs git inside Wine. Three GPU tests show real D3DMetal differences that still need work.

- `buffer_data_reaches_the_host_texture`: fixed. The upload was queued as a copy into title memory, and that check uses VirtualQuery, which cannot see D3DMetal's mappings under Wine, so the copy was skipped. It is now queued as a fill of the list's upload memory.
- `argument_clamp_builds`: fixed. Wine's HLSL compiler (vkd3d-shader) cannot compile VIctory's clamp shader. Under Wine, `indirect.rs` now builds a second form of it (`WORDS_SOURCE`) that reads its buffers through structured views and charges the records in one thread, with the same results. This replaced the bytecode that a Windows runner used to compile, because VIctory's asset policy now refuses long runs of embedded hex.
- `a_rejected_blob_gives_an_empty_library`: fixed. D3DMetal accepts any pipeline library bytes, so the cache file now carries a mark and a checksum that VIctory checks itself.

After these fixes the whole suite passed under CrossOver Wine with D3DMetal on an M4: 343 passed, 0 failed and 4 ignored. After the merge of 25 November 2026 and the fixes below it gives 481 passed, 0 failed and 7 ignored.

## Run-time HLSL and textures under Wine and D3DMetal

Under Wine, `D3DCompile` is Wine's HLSL compiler (vkd3d-shader). In CrossOver 26.3 Wine it differs from Microsoft's compilers in four ways that matter here:

- It ignores `[RootSignature]` (warning W5302) and writes no root signature into the bytecode, so `CreateRootSignature` on that bytecode fails with `E_FAIL`. Build run-time compute shaders with `compute::pipeline` in `victory-gpu`, which serializes the root signature itself.
- `RWByteAddressBuffer` has no `Load`, and `RWStructuredBuffer` stores are not implemented. The depth converter in `twin.rs` reads its scratch buffer through a structured view and writes it through a raw one.
- Shader model 5.1 resource arrays are not implemented, bounded or unbounded (`E_NOTIMPL`), and `NonUniformResourceIndex` is not defined. So `lists_share_a_table_and_copies_keep_their_views` skips under Wine. Embedding FXC's bytes instead is ruled out by VIctory's asset policy, which refuses long runs of embedded hex.
- It still compiles plain `cs_5_1` shaders, and D3DMetal runs their bytecode.

D3DMetal has no Metal format for `R10G10B10_XR_BIAS_A2_UNORM`. It refuses any castable format list that names it, and creating a texture in that format stops the process in Metal's validation. `castable` leaves it out, so 4-byte colour textures of one memory share one host texture under D3DMetal too.

It also refuses any list that names the packed formats `R8G8_B8G8_UNORM` and `G8R8_G8B8_UNORM` (68 and 69), which every 2-byte format's list did, so no 2-byte texture took castable formats. After the first refusal, VIctory gave each format of one memory its own host texture, and a run on a 16 GB M5 grew to 6.9 GB of host textures and ran out of GPU memory (issue #15). `castable` leaves both out. With all three left out, D3DMetal 4.0 beta 3 takes all 96 lists `castable` makes, checked by creating each format's texture with its list.

D3DMetal rounds float depth down when it stores 24-bit depth (a clear to 0.5 stores 8388607, not 8388608), where Windows GPUs round to nearest.

D3DMetal 3.0, as bundled with CrossOver 26.2 and 26.3, returns zeros when a 24-bit depth texture (`R24G8_TYPELESS` or `D24_UNORM_S8_UINT`) is filled by a copy or copied out, while a clear reads back correctly in a shader (#33, with a standalone repro). 32-bit depth works. GPTK 4.0 beta 3 does not have this problem, and `a_depth_texture_written_through_unordered_views_reads_as_depth` passes there, so VIctory needs D3DMetal 4.0 or newer for depth copies. The test stays as it is on 3.0 so the failure remains visible.

## Route A: next steps

The first playable route is the upstream Windows build running under Wine, with D3D12 going to Metal through Apple's D3DMetal. DXMT cannot be used because it covers D3D11 only. Rosetta 2 runs the title's x86-64 code and has supported AVX2 since macOS 15.

The steps are as follows.

1. Install rustup and add the `x86_64-pc-windows-gnu` or `x86_64-pc-windows-msvc` target. The MSVC route through `cargo-xwin` downloads the Microsoft CRT and SDK under Microsoft's licence.
2. Build `apps/victory` and the `tests/fixtures/hle-probe` stand-in title for Windows.
3. Run the probe under Wine to test the loader and HLE layer without any game files.
4. Add D3DMetal from Apple's Game Porting Toolkit and check that VIctory creates a D3D12 device with mesh shaders and shader model 6.6.

## Route B: a native macOS host

Later work replaces the Windows host underneath the title. That needs a Rosetta x86-64 process that maps the PE image, a macOS version of the host pass-through in `victory-hle`, a Metal backend for `victory-gpu`, and DXIL to Metal conversion with Apple's Metal shader converter. This only makes sense after Route A shows where things break. `docs/native-host.md` has the design, the first working slice (a PE mapped and run under Rosetta with no Wine) and the interfaces for the Metal backend and the recompiler.

## Running VIctory on a Mac

`./setup.sh` (since removed) checked the Mac, installed the Homebrew tools, built checksum-verified x86_64 GMP, Nettle, GnuTLS and FreeType, built CrossOver Wine, added D3DMetal from the Game Porting Toolkit in Downloads, and built VIctory. The only manual step was downloading the toolkit from Apple. A clean run on an M4 on 24 November 2026 finished without help, and the full suite then passed (343 passed, 0 failed).

## Use D3DMetal 4.0 beta 3

On 25 November 2026 two testers ran the same Series X executable (one SHA-256) with the same profile. With CrossOver 26.3's bundled D3DMetal 3.0 on an M5 Pro, the game stopped about seven seconds in at its own assert at `+0x1dac734`, a subsystem checked during start-up (#24). With D3DMetal 4.0 beta 3 on an M3 Max it never hit that assert and reached the game world (#50). D3DMetal 3.0 also names the adapter `VirtualApple @ 2.50GHz` and refuses an empty pipeline library blob (#30). Since #74 asks for an empty library with a null pointer, host pipelines are kept across runs on 3.0 too: mikebuckets171 saw a library written on a first run and loaded back on the second. Use 4.0 beta 3 or newer for the game. It is not the only cause of that assert, though: later the same day us3r40456 hit `+0x1dac734` with D3DMetal 4.0 beta 3 too, on both the Series X and Series S profiles, so something else in their setup also decides it (#24).

## Stacks during a freeze

In the loading freeze from #50 every thread stops logging, and `VICTORY_STACKS` printed nothing, not even its first line. Printing takes the stderr lock and allocates, and the thread snapshot goes through Wine, so a thread stuck while it holds one of those locks stops the dump too. With `VICTORY_STACKS_FILE=<path>` as well (in a build with `VICTORY_DIAGNOSTICS=1`), the dumps go to that file instead. The module table is written at start-up, and each dump writes a marker line before each step and raw addresses for each frame, with no allocation and no stderr lock. Where the file stops shows which step blocked.

## Presented planes

On Series S the game presents two planes (#15). Plane 0 is the 1920x1080 picture and plane 1 is the 3840x2160 interface: in game it is cleared to transparent black each frame and holds only the HUD and subtitles, which is what the window showed while it showed only the largest plane. The native host shows every plane, bottom first, in the title's order (`queue::planes_shown`): plane 0 is drawn over the whole window, and each later plane is scaled to the window and blended over it by its alpha (straight alpha; `PresentBackend::present` on Metal). Guess: the title lists its planes bottom first, and each fills the screen. The rest of a plane description (placement, scaling, alpha mode) is not decoded. `VICTORY_PRESENT_PLANE=<n>` shows plane n alone. On Windows the window still shows the largest plane. With `VICTORY_PRESENT_COMPOSITE=1`, the other plane is also blended over it by a compute pass (`victory-gpu/src/composite.rs`).

## D3DMetal's shader cache needs unique shader hashes

D3DMetal keeps a compiled-shader cache for each executable, in `$(getconf DARWIN_USER_CACHE_DIR)d3dm/<exe name>/shaders.cache`. It finds shaders by the DXIL container's shader hash (the `HASH` part), apparently together with the shader's length: two shaders collided only when their lengths matched. VIctory used to write the same all-zero shader hash into every container. So two different shaders of the same length shared a cache entry, and the one compiled second ran the first one's code. The cache is on disk, so the wrong code also came back on every later run of that executable.

This was the "signed zero in minimums" difference listed here before. Each `v_min_f32` and `v_min3_f32` test shader is the same length as the maximum built just before it, so the minimum ran the maximum's code and gave +0 where the console gives -0. Selecting on the integer bits could not help, because the shader that ran was never the one being changed. It also made `v_cmp_class_f16_matches_rdna2` fail whenever the max/min test ran before it, as it does with `--test-threads=1`. The `v_max3_f32` shader and that test's first shader are both 2316 bytes.

`container::shader_hash` now writes the MD5 of the program into the `HASH` part. The minimum cases run under Wine again, and both tests pass in every order on an M5 Pro with D3DMetal 4.0 beta 3. Cache entries written under the old zero hash are never looked up again, so old caches need no cleaning.

## Open differences under D3DMetal

The merge of VIctory's changes up to 70564ae on 25 November 2026 brought new shader tests, and three of them find real differences in D3DMetal 4.0 beta 3 on an M4. The tests skip only those checks under Wine, so the rest of each test still runs.

- 64-bit image atomics. D3DMetal reports `AtomicInt64OnTypedResourceSupported` and `AtomicInt64OnDescriptorHeapResourceSupported`, but its 64-bit atomics in `image_atomics_of_64_bits` leave every texel at zero. The game's alpha prepasses use them. Under Wine the translator now emulates them (`Options::atomic_locks`, the module notes' "Atomic locks", `VICTORY_ATOMIC_LOCKS=0` or `1` to override): the lanes of a wave take turns in a loop over lane indices, and in its turn a lane takes a 32-bit lock word, reads the texel, writes the result and frees the lock. Two simpler forms failed under D3DMetal and are worth knowing about. Lanes of one wave spinning on each other's locks never finish, and `WaveIsFirstLane` and `WaveReadLaneFirst` inside a divergent loop return other lanes' values. The spin is bounded at 65536 tries so a fault can never hang the GPU.
- Sample bias in compute programs. D3DMetal ignores the bias of a sample in a compute shader: a bias of 1 reads level 1 where the console reads level 2. Under Wine the translator now runs those samples as gradient samples with the quad gradients scaled by `2^bias` (`Options::bias_as_gradients`, found by nickorlabs in #30; `VICTORY_BIAS_AS_GRADIENTS=0` or `1` overrides). The bias cases in `image_addresses_of_16_bits_and_clamps` run under Wine again and pass. Pixel shaders keep the plain biased sample.
- 24-bit depth. Apple GPUs have no 24-bit depth format, so D3DMetal keeps it as 32-bit float and rounds down when it stores 24 bits. The depth test clears to 0.75, where both roundings agree (PR #6).
- Root signature blobs. D3DMetal accepts a truncated root signature blob and crashes on garbage, where Windows returns `E_INVALIDARG`, so `malformed_blobs_are_rejected_by_the_host` skips under Wine.
- Hard links. `a_kept_blob_past_the_link_limit_is_copied` needs NTFS's limit of 1024 links to a file, and APFS has no such limit, so the test skips where the 1025th link succeeds.
- Gather positions. D3DMetal's own gather rounds the position as the console does, but from u × size taken in f32, where the console takes the exact product. On an M5 Pro it starts a texel over on all 12 lanes of `gathers_at_a_level_match_the_hardware` where the two products fall on either side of a texel's threshold, and matches the console on the others. Truncating, as WARP does, would have kept the console's texel on those lanes. VIctory's gathers at a level follow the console, but its level-0 gathers are the host's own, so they can differ there. The test checks that the host's own gather is either the console's or this one.

Run-time compute shaders are built with explicit root signatures (`compute.rs`, PR #6), which Wine's compiler can build. The one exception is the memory test's shader, which needs shader model 5.1 resource arrays, so that test skips under Wine.

## Host view heaps are filled with other threads' submissions held off

Under Wine on D3DMetal, a list that one thread submits while another thread fills a new host view heap can lose its views: its shaders' writes are dropped and its reads give zeros, whatever heap the list binds (issue #31). Each host view heap has 2,359,296 slots, and `HostHeap::fill_null` makes every one null with 36 copies of 65,536 null views. On an M5 Pro with D3DMetal 4.0 beta 3 that takes 30 to 50 ms, and the victory-gpu tests fill several at once when they start.

A texture that a dispatch wrote in that time read back as zeros. A copy of it by the copy engine was zeros too, and so was the same write and read in fresh lists, with fresh pipelines or with the views made again. It took writes again by itself 31 to 49 ms later, once the fills had ended. Skipping the fill, or holding submissions off while a heap is made and filled, stopped it. Holding them off only while its upload buffers were made did not.

So `Host::quiet` holds the submission lock while a view heap is made and filled (`HostHeap::new`, and the first `Heap::host` of a title view heap, which also writes the views queued before it). The fill waits for the submission running then, and the next submission waits for the fill. That happens once per view heap. Elsewhere than Wine nothing is held. In 100 runs of the victory-gpu library tests on that Mac, the four texture tests this hit (1D and 3D, placed view, depth through unordered views, overlay) failed 62 times before, often several in one run, and not once after.

A standalone D3D12 program did not reproduce it: 16 threads writing and reading textures, while 3 or 4 other threads made heaps of the same size and filled them the same way, bound in submissions or not. Something else VIctory does is needed as well, and it is not known what.

Buffers are another matter, still open. `root_views_reach_their_copy_and_read_zeros_past_it` and the two `heap_views_` tests still fail in a few runs in a hundred, with or without the change. In a failed `root_views` run the source's host copy held the title's values while the output stayed zero, when run again at once, 100 ms later and with its view written again, so it does not come back by itself as the textures do.

## Masked stores into protected pages hang Rosetta

On macOS 27.3 (M3 Max) the game stops at start-up with `rosetta error: unexpectedly need to EmulateForward on a synchronous exception` and the process hangs (issue #1). The faulting instruction is `vpmaskmovd [rax], ymm6, ymm4`, an AVX2 masked store in a loop that steps 32 bytes at a time.

The reproducer from issue #1, extended with `vpmaskmovd`, shows the cause without game files. Under CrossOver Wine 26.3.0 with `ROSETTA_ADVERTISE_AVX=1`, SSE and AVX `vmovups` stores survive in every layout, and so does a masked store wholly inside a protected page. A masked store that starts in writable memory and ends in a protected page hangs Rosetta with the same error, both in a block guarded the way `victory-gpu/src/watch.rs` guards it and in `MEM_WRITE_WATCH` memory, which Wine emulates with page protection. Rosetta seems to have written the first lanes before the fault and cannot finish the instruction afterwards.

VIctory used both kinds of protection to see which title GPU memory changed. Under Rosetta, which `victory_gpu::fault_tracking` detects as Wine on Darwin, it now uses neither. The runtime's pools are allocated without a write watch, and host copies of title memory take it whole every time, as they already do for memory that is not the runtime's. That costs copy time but no fault can reach Rosetta. `VICTORY_FAULT_TRACKING=1` turns tracking back on to test a newer Rosetta, and `0` turns it off anywhere.


The game's executable holds 1130 masked stores (789 `vpmaskmovd`, 160 `vpmaskmovq`, 180 `vmaskmovps` and 1 `vmaskmovpd`), which `victory` reports at start-up. `VICTORY_PATCH_MASKED=all` (or a comma list of hex RVAs) rewrites each one at load time as a jump to a small trampoline that does the same stores one lane at a time with plain `mov`s, which Rosetta can resume after a fault (`victory-loader/src/masked.rs`). With that, `VICTORY_FAULT_TRACKING=1` can be tried under Rosetta, so copies of title memory no longer have to be taken whole (about 129 GB per frame once the world streams in, #50). Both switches are off by default until runs with the game confirm them. The scan reads bytes, so a rare false match could be patched with `all`, and naming RVAs avoids that. Run 11 in #50 showed that `vcvtps2ph` stores to memory hang Rosetta the same way, so the scan and the patch also cover `vcvtps2ph`, `vextractf128` and `vextracti128` stores. Those are redone into the stack and then copied out with plain `mov`s. With the masked stores patched, copies during loading fell from 158.7 MB to 0.7 MB per frame and frames from 125.7 ms to 38.3 ms.