# Native Metal backend

Design notes for running `victory-gpu` on Metal with no Wine, and the results of the first
prototype (`crates/victory-metal`). Measured on an Apple M5 Pro, macOS 27.2, 48 GB, 16 KiB pages.
Nothing here changes the D3D12 path.

## Why

Under Wine and D3DMetal most of a frame goes to the layering, not the GPU work:

- the host GPU cannot read title memory, so `shadow.rs` keeps host copies of title buffers, `watch.rs`
  write-guards title pages (64 KiB blocks, vectored fault handler) to know what changed, `memory.rs`
  and `texture.rs` refresh copies before each submission, and shader writes are copied back
  (`Shadow::copy_back_page`, the page bitmaps of `victory_shader::emit` "Title memory");
- `host.rs` waits for every submission (`shortcut: every submission is waited for`), 17-35 per
  frame, at about 5-8 ms each.

Apple Silicon and the console both have one memory the CPU and GPU share, which is what the title
already assumes ("a resource's GPU address is its CPU address"). On Metal that can be literally true.

## 1. D3D12 concepts onto Metal

| victory-gpu (D3D12.X on host D3D12) | Metal |
|---|---|
| `host.rs`: one device, one direct queue | one `MTLDevice`, one or more `MTLCommandQueue` |
| `queue.rs`, `hold.rs`: queues, Signals, memory-value waits across queues | `MTLCommandQueue`s; Signal/Wait on `MTLSharedEvent`, or `MTLEvent` within one device. Waits on memory values (`Wait32BitValueX`, a graphics list writes the word) stay a host-side hold (`hold.rs` logic) or become shared-event signals when both lists are known |
| `list.rs`: command allocator + list, recorded as the title calls methods | `MTLCommandBuffer` per submission segment; draws and dispatches go into one `MTLRenderCommandEncoder` / `MTLComputeCommandEncoder` per pass. Allocators disappear (command buffers are transient). A list cut at a wait (`hold.rs`) is two command buffers |
| `ExecuteCommandLists` + wait | `commit()` with no wait; completion by `addCompletedHandler` or a shared-event value |
| `fence.rs`: 64-bit value, events | `MTLSharedEvent`: `encodeSignalEvent`, `encodeWaitForEvent`, CPU `waitUntilSignaledValue`, or `notifyListener` instead of a wait |
| root signature (`root.rs`, `pipeline/host.rs`: per-stage root constants, root views, static samplers, directly indexed heaps) | the converter's top-level argument buffer, bound at buffer slot 2: root constants inline, root SRV/UAV/CBV as 64-bit GPU addresses, descriptor tables as GPU addresses of table memory. Static samplers are baked into the pipeline |
| descriptor heaps (`descriptor.rs`, host heap, `descriptor::host`) | a descriptor heap is an `MTLBuffer` of fixed-size entries (GPU address, texture resource ID, metadata, written with the converter runtime's `IRDescriptorTableSet*` helpers), bound at slot 0; samplers likewise at slot 1 (argument-buffer sampler IDs). `ResourceDescriptorHeap[i]` in the DXIL is then an index into that buffer. Heap-wide residency by `useResource` / `MTLResidencySet` |
| committed/placed resources, buffers (`resource.rs`) | buffers: `MTLBuffer` over title memory (below). Textures: `MTLTexture` from `MTLHeap` (replaces `placement.rs`) |
| title GPU address = CPU address | a constant or table offset from title address to GPU address (below) |
| pipeline states (`pipeline.rs`, `pipeline/*`): made at first draw, ~60,000 created, dozens drawn | same laziness. Compute: `MTLComputePipelineState` from the converted function. Graphics: `MTLRenderPipelineState`, with the stage-in function, blending and render-target formats the converter needs at link time; mesh shaders (`ms_6_6`) map to Metal mesh pipelines |
| `shader_cache.rs` (translated DXIL by content hash) | cache the converted `.metallib` next to it; conversion averages 84 ms (compute), 32 ms (pixel) over the corpus below |
| `twin.rs` (depth also written by UAV: colour twin) | Metal textures allow `shaderWrite` on depth-format views only through a pixel-format view; likely still needs the twin until checked on real shaders |
| `query.rs` | `MTLCounterSampleBuffer` / visibility result buffer |
| `signature.rs` (ExecuteIndirect) | Metal indirect draws and dispatches that read the records in title memory; with a count buffer, a one-thread gate kernel per record first copies the record or zeros (`gfx.rs`, `GATE`) |
| present | `CAMetalLayer` |

What is not a mapping but a gain: the per-draw "dynamic slots" of the host heap, the per-submission
refresh, and the wait after every submission exist to feed a GPU that cannot see title memory. They
go (next section).

Submission cost, measured with the probe (below): commit + wait of an empty command buffer is
about 14 us; with a shared-event wait, 14 us; a tiny dispatch end to end about 140 us, but the same
dispatches pipelined (35 command buffers, one wait at the end) cost 10 us each. The ~5-8 ms per
submission under Wine does not exist here; what remains is that the **dependency structure** (one
wait per submission) should still go, because a 140 us round trip times 35 is 5 ms.

## 2. Title memory as GPU memory, with no copy

`newBufferWithBytesNoCopy(pointer:length:options:deallocator:)` makes an `MTLBuffer` whose storage
is the caller's memory. With `StorageModeShared` the buffer's `contents()` is the pointer given, the
GPU reads and writes the same pages, and there is no copy and no synchronisation call
(`didModifyRange` is for managed buffers on discrete GPUs; not needed).

### What the prototype shows (`[1]`-`[4b]` of `cargo run --release -p victory-metal`)

- `buffer.contents()` equals the host pointer, byte for byte.
- A CPU write made after the buffer was created is seen by the next dispatch, with no call in between;
  a GPU write is seen through the host pointer after completion. Checked on 64 K words, twice.
- The same through raw GPU addresses (`buffer.gpuAddress()` passed in a constant struct, `useResource`
  for residency): the shape bindless access takes.
- **One set of physical pages, two virtual addresses**: memory allocated with `mach_vm_allocate` and
  aliased with `mach_vm_remap` (shared, not copied), buffer wrapped over the alias; a GPU write is seen
  at the other address. So the title's addresses and the Metal-facing mapping can be different
  mappings of the same physical pages.
- Several buffers over one range, and sub-range buffers inside a bigger one, are accepted.
- `maxBufferLength` is 28 753 MiB here (of 48 GB); a 16 GiB untouched reservation wraps in 0.5 ms,
  32 GiB is rejected. The console's 16 GB fits in one buffer.

### Constraints

- **Alignment.** Apple documents that the pointer must be page-aligned (16 KiB on Apple Silicon) and the
  length a multiple of it. This macOS/M5 build accepted an unaligned pointer and any length and the
  memory was still shared (CPU wrote 100, GPU added one, CPU read 101). Do not rely on that; page-align
  both. Consequence for the host: **every mapping the GPU may touch is 16 KiB granular**, finer than the
  console's 4 KiB small pages only if the title asks for them; 64 KiB large pages are fine.
- **Lifetime.** With no deallocator the memory must stay mapped until the buffer and every command
  buffer that used it have finished. Unmapping title memory the GPU still reads is the title's bug on
  the console, and a GPU fault here.
- **Size.** One buffer is at most `maxBufferLength` (28.7 GiB here, device dependent); larger memory
  takes several buffers (the shader then needs a table, below).
- **Residency / first use.** Metal wires the pages when a command buffer first uses the buffer, once:
  a dispatch binding a fresh 4 GiB touched buffer took 34 ms the first time and 0.2 ms after; 8 GiB took
  68 ms; 16 GiB untouched 281 ms. About 8-17 ms per GiB, once. So wrap the title's pools **once at
  start-up** and prime them with a dummy dispatch; never create buffers per frame. A
  `MTLResidencySet` on the queue keeps them resident without a `useResource` per encoder.
- **Same address for CPU and GPU: no.** The GPU address (`gpuAddress`) is chosen by Metal
  (`0x10000080000` against a host pointer `0x1135a0000`). The title's address is the CPU address, so
  each region needs `gpu = title + delta`, with one delta per buffer. `useResource` / argument buffers
  take the GPU address; the table that does this is smaller than what exists now.
- Not yet tested: memory from `mach_vm_map` of a memory entry (what the host will create), executable
  or guard-page regions inside a wrapped range (a guard page inside the range must be mapped, or the GPU
  faults if a draw reaches it), GPU access while the CPU holds the same cache line through the aliased
  mapping (the same physical memory, so coherent, but unmeasured under contention).

### What it removes from victory-gpu

| Today | With zero-copy |
|---|---|
| `shadow.rs` (1 828 lines): a persistent host buffer per title buffer a draw uses, ranges refreshed per submission | gone: the title buffer is the GPU buffer, vertex/index/constant data are read in place (`setVertexBuffer` at `title_addr - base`) |
| `watch.rs` (1 677 lines): read-only guards, vectored fault handler, "pages the title wrote since taken", alias tracking, `title_memory_remapped` | gone: nothing needs to know what the title wrote. (The Wine build already runs "nothing tracked" under Rosetta, `fault_tracking`.) The host still tells us about map/unmap, below |
| copy-back: `Shadow::copy_back_page`, the `memory_writes` page bitmap (256 KiB per list), `view_marks`, `VIEW_WRITES` bitmaps, atomic locks for 64-bit image atomics via buffer | gone from the shader contract: shader writes land in title memory. 64-bit image atomics still need a Metal-side answer (atomic on a 2xR32 view is not a Metal image op) |
| `memory.rs` "Title memory at raw addresses": page table of heap indices of raw SRV/UAV pairs of copies, rewritten when copies come and go | simpler: the table entry is an offset into one big raw buffer (or `{buffer, delta}`), or with one contiguous arena, `addr - base` and no table |
| `hold.rs`/`host.rs` wait after every submission to make writes visible | not needed: writes are visible when the command buffer completes; queue dependencies use events, many submissions in flight |
| `texture.rs` detile into staging, then copy to the host texture, per changed page | still needs detiling (the console's tiled layout is not Apple's), but as a **compute detile reading title memory directly** and writing the `MTLTexture`; the CPU staging buffer and compare-per-submission go |
| `placement.rs` heaps | `MTLHeap` |

### Interface needed from the native host agent

The host reserves title memory; the GPU layer needs, for each **physical-page pool** (one mach VM
object per pool, as agreed):

1. `base: *mut u8` and `len`, **page-aligned to 16 KiB**, the pool's mapping in the host process, plus the
   title VA it is the primary mapping of. If the title maps a pool's pages at several addresses (it does:
   "physical pages may be mapped at several addresses"), one mapping per pool that is **permanent for the
   process lifetime** is what we wrap; the other addresses are `mach_vm_remap` aliases (or the same
   memory entry mapped again) and need no GPU buffer.
2. The mach memory entry port, if the pool was made with `mach_make_memory_entry_64`, so we can map our own
   GPU-facing alias without asking for a remap (or the permission to `mach_vm_remap` from the pool address).
3. Map/unmap events, as `title_memory_remapped` is today: `map(title_va, len, pool, pool_offset)` and
   `unmap(title_va, len)`, called before the title can use the address. The GPU layer keeps the
   `title page -> (pool buffer, offset)` table from these; no write tracking.
4. A guarantee that **a pool's address range is never unmapped or `mprotect`ed to no-access while the
   process runs**, and that placeholders (reserved, nothing mapped) are not inside a pool.
5. Ideally one contiguous range for all GPU-visible pools (an arena of up to ~28 GiB): then the shader
   address is `title_va - arena_base` for a single raw buffer view and the page table in
   `victory_shader::emit` becomes one subtraction. If the title VA layout is sparse, make the arena a
   second, contiguous `mach_vm_remap` alias of the pools and keep a per-64 KiB-page offset table.

## 3. Shader path

`victory-shader` already ends at DXIL (an `SM 6.6` container: heap-indexed resources, wave ops, mesh
shaders), and `victory-gpu` hands that to D3D12. Two ways to reach Metal:

**A. Metal Shader Converter (MSC) on the DXIL we already emit.** Apple's `metal-shaderconverter` /
`libmetalirconverter`: DXIL container in, `.metallib` out. Supports SM 6.0-6.6 (wave intrinsics, 64-bit
ints, ray tracing, mesh and amplification shaders, dynamic resources), version 4.0 beta is current. It
is what D3DMetal is built on, so it is the path that handles D3D12 semantics (root signatures,
descriptor heaps, resource binding model) on Metal.

**B. Emit MSL or AIR from victory-shader.** A second back end next to `emit/`: no external dependency, full
control (title memory loads as real 64-bit pointers, 64-bit image atomics, no page table). But all the
D3D12 semantics MSC already implements (binding model, descriptor heaps, UAV barriers, wave semantics, mesh
pipelines) must be re-done, MSL needs the `metal` compiler at run time (or AIR emitted by hand, an
undocumented LLVM dialect), and every shader path would be tested again.

**Recommendation: A first, B later and narrowly.** The DXIL emitter is the product of a long effort and
is tested; MSC takes it as it is. Prototype results:

- The whole cached GTA VI corpus converts except the compute shaders that ask for wave64 (next section:
  "Shader corpus"). Every converted library loads in Metal and builds its pipeline state.
- An end-to-end shader (HLSL compiled to DXIL by dxc, `cs_6_0`, root signature `RootConstants(2),
  SRV(t0), UAV(u0)`) converted and ran over a `bytesNoCopy` buffer: root SRV and UAV are 64-bit GPU
  addresses in the top-level argument buffer, root constants inline. 4096 of 4096 outputs correct
  (`victory-metal --demo`). That is the no-copy path for shaders: a root view *is* a title address plus a
  delta.
- Later, B can replace MSC for the hot or unsupported pieces (page table loads, 64-bit atomics) without
  changing the host side.

### Obtaining MSC

Not on PATH; it is a separate Apple download, not part of Xcode or the Metal toolchain
(`xcrun --find metal` finds the compiler, not `metal-shaderconverter`).
Download "Metal shader converter" (4.0 beta at the time of writing, macOS and Windows, with a PDF
reference) from `developer.apple.com/metal/shader-converter/` (links to
`developer.apple.com/download/all/?q=metal%20shader%20converter`, an Apple ID is needed). It installs
`metal-shaderconverter`, `libmetalirconverter.dylib` and the headers `metal_irconverter.h` and
`metal_irconverter_runtime.h`. Requires macOS 13 and Xcode 15 or later; the output requires argument
buffers tier 2 (every Apple Silicon Mac).

Installed here by Apple's package: `/usr/local/lib/libmetalirconverter.dylib` (a universal binary,
arm64 and x86_64), `/usr/local/bin/metal-shaderconverter`, headers in `/usr/local/include/metal_irconverter*`.
`victory-metal` loads `/usr/local/lib/libmetalirconverter.dylib` natively by `dlopen`; `VICTORY_MSC_DYLIB` overrides the
path. (The copy inside D3DMetal.framework is x86_64 only and was what the first prototype ran under Rosetta;
no longer needed. `--demo` still needs the x86_64 `libdxcompiler` of that bundle to make its HLSL
shader, so it is the one thing that runs under Rosetta; the test shader is now committed as a fixture.)
The library's licence is Apple's: nothing of it, nor its headers, is vendored. Conversion could also run
offline in `victory translate --cache`, caching `.metallib`s.

Facts read from the installed headers (`metal_irconverter.h`, `metal_irconverter_runtime.h`):

- `IRErrorCode`: 0 NoError, 1 ShaderRequiresRootSignature, 2 UnrecognizedRootSignatureDescriptor,
  3 UnrecognizedParameterTypeInRootSignature, 4 ResourceNotReferencedByRootSignature,
  5 ShaderIncompatibleWithDualSourceBlending, **6 UnsupportedWaveSize**, 7 UnsupportedInstruction, 8 CompilationError,
  9 FailedToSynthesizeStageInFunction, 10 ...StreamOutFunction, 11 ...IndirectIntersectionFunction,
  12 UnableToVerifyModule, 13 UnableToLinkModule, 14 UnrecognizedDXILHeader, 15 InvalidRaytracingAttribute,
  16 NullHullShaderInputOutputMismatch, 17 InvalidRaytracingUserAttributeSize, 18 IncorrectHitgroupType, 19 FP64Usage.
  `IRErrorGetPayload` is documented only as "cast to the payload struct for the code"; in practice for these
  codes it points at the message text, which is what `msc.rs` reads.
- `IRShaderStage`: 1 Vertex, 2 Fragment, 3 Hull, 4 Domain, 5 Mesh, 6 Amplification, 7 Geometry, **8 Compute**, ...
- Binding layout, `kIR*BindPoint`: **0 resource descriptor heap, 1 sampler heap, 2 top-level argument buffer**,
  3 hull/domain argument buffer, 4 draw arguments, 5 uniforms, 6 vertex buffer. A descriptor table entry is
  `IRDescriptorTableEntry { u64 gpuVA; u64 textureViewID; u64 metadata; }`, 24 bytes. This matches the
  table in section 1 and what `backend` binds (slot 2 only so far; heaps are step 5 below).
- `IRStageInCodeGenerationMode`: 0 UseMetalVertexFetch (default, a `MTLVertexDescriptor` at pipeline creation),
  1 UseSeparateStageInFunction.

## Prototype: `crates/victory-metal`

macOS-only (target-gated dependencies and modules; builds as an empty binary on the Windows targets, so
the D3D12 workspace and its clippy gate are untouched). `objc2-metal` 0.3, `libc`.

```
cargo run --release -p victory-metal                       # memory + latency probe
cargo run --release -p victory-metal -- shader.dxil...     # convert, load, build a compute PSO
cargo run --release -p victory-metal -- --corpus <cache>/shaders [--threads 8] [--fails out.txt]
cargo test -p victory-metal                                # compute, heap/texture and render tests (skip if MSC is not installed)
```

Submit-to-complete latency (3000 iterations, release, native arm64; Wine/D3D12 today ~5000-8000 us):

| | min | p50 | p99 |
|---|---|---|---|
| empty command buffer, commit + `waitUntilCompleted` | 9.7 us | 14.2 us | 20.4 us |
| tiny compute dispatch (64 threads), same | 66.6 us | 137.5 us | 243.4 us |
| empty command buffer + `MTLSharedEvent` signal, CPU `waitUntilSignaledValue` | 6.0 us | 13.9 us | 19.4 us |
| per dispatch command buffer, 17 in flight, one event wait | 9.9 us | 14.8 us | 25.2 us |
| per dispatch command buffer, 35 in flight, one event wait | 8.6 us | 9.9 us | 12.5 us |
| per dispatch command buffer, 256 in flight, one event wait | 8.0 us | 8.7 us | 9.7 us |

GPU time of the tiny dispatch is 2 us, so the 140 us is wake-up and completion latency, which
pipelining removes: a frame of 35 submissions with one wait is about 0.35 ms of submission overhead, not
175-280 ms.

## Shader corpus

`victory-metal --corpus <cache dir>` walks a victory shader cache (`<build>/<xx>/<hash>`; each file is
`VSH1 stage dxil...` or `VSHE reason`, `victory-gpu/src/shader_cache.rs`), counts each container once (the
DXBC digest and length, across the cache's builds), converts it and builds the pipeline state its stage
allows. It prints counts only, writes nothing but stats (optionally a list of failing hashes with `--fails`)
and keeps no shader or converted output. Run on the cache of the Wine prefix
(`target/wineprefix-play/.../victory/cache/shaders`, written by earlier runs and warm-ups: 29 614 entries
from 12 translator builds; GTA VI was not launched). That is every shader the title has asked for so far,
not necessarily the whole game. Entries are the Wine-flavoured translation (view marks, atomic locks,
narrow waves where those options were on), which is what the host would be given today.

Measured on the M5 Pro, native arm64 converter, 8 threads, 522 s for the lot.

| Stage | unique DXIL | DXIL MB | converted | failed in converter | pipeline state built | failed building |
|---|---|---|---|---|---|---|
| Compute (`cs`) | 3 549 | 678 | 3 021 | 528 (IRError 6) | 3 021 | 0 |
| Pixel (`ps`) | 2 635 | 256 | 2 635 | 0 | 2 635 | 0 |
| Vertex (`vs`) | 367 | 4.1 | 367 | 0 | 367 | 0 |
| Mesh (`ms`) | 721 | 28 | 721 | 0 | 693 | 28 (mesh too large) |
| **Total** | **7 272** | **966** | **6 744 (92.7 %)** | **528** | **6 716** | **28** |

Besides those, the translator itself refused 216 cache entries (9 distinct reasons, all GFX10 constructs:
lane masks read as data 90, descriptors read only as data or neither loaded nor built 59, `v_cvt_f32_f64` 19,
barycentrics read directly 17, a lane-mask join of an untracked mask 11, a descriptor chosen by control flow 11, `v_permlanex16_b32` 9); they never
produce DXIL, so they are not in the table. No hull, domain, geometry or ray stages occur in the cache.

Failures:

- **IRError 6, `IRErrorCodeUnsupportedWaveSize`** (header: `metal_irconverter.h`, enum value 6; message
  "Wave size is not supported. Must be 32."): 528 compute shaders, 14.9 % of compute. The emitter writes
  `WaveSize(64)` into a compute shader whose console wave is 64 lanes and that uses wave operations or LDS
  that is not synchronised (`emit/lower.rs`: `s.wave_size = Some(64)`; `narrow_waves` avoids it only where
  the program fits 32 lanes). Apple GPUs run 32-lane SIMD groups, and the converter accepts only wave size
  32 (the same limit D3DMetal reports, hence `narrow_waves` under Wine; there those shaders simply do not
  run, "nothing written"). Not a converter bug: a translator output to change for a Metal host. Options: emit
  wave32 code for these (two waves per group, lane-mask ops split, group barriers where LDS is shared;
  `narrow_waves` already does part of it), or run the 64 lanes as two simdgroups with explicit exchange.
  This is the largest open item of the shader path.
- Mesh pipeline, 28 shaders: Metal rejects "Total mesh size exceeds the maximum mesh size allowed (32768)"
  at 34 816 to 38 400 bytes (vertex plus primitive output memory of the shader's declared maxima).
  Needs the translator to declare tighter output maxima, or a split.
- Nothing else failed: no `UnsupportedInstruction`, `CompilationError`, root-signature or link errors.

How pipelines were built (a harness, not the real render path, so the pixel and vertex numbers say "Metal
accepts the converted function", not "the right pipeline"): compute: `MTLComputePipelineState`; vertex:
render pipeline with no fragment function and a `MTLVertexDescriptor` made from the function's vertex
attributes; pixel: a generated pass-through vertex function that writes the user varyings the fragment
function reads (their names and types are found from Metal's own mismatch errors), colour targets
of float, uint or sint and a depth target; mesh: a mesh pipeline with rasterisation off.
Conversion costs 84 ms average for compute 32 ms pixel, 13 ms mesh, 3 ms vertex;
Metal pipeline creation under 8 threads averages 0.58 s compute, 0.76 s pixel (device compilation, the cost
the pipeline cache must hide); the converter prints "Shader might rely on forward progress/persistent
thread" for some compute shaders (a warning, not an error).

One harness pitfall worth keeping: **`newLibraryWithURL` caches by URL**. Writing different metallibs to
one temp file name returned stale libraries (wrong functions, "unlowered function call air.read..."). Use
a fresh name per library, or `newLibraryWithData`.

## Compute backend, first slice (`backend.rs`, `metal.rs`)

`victory_metal::backend::ComputeBackend` is the seam: a platform-neutral trait (no Metal types) with
`create_queue`, `create_fence`, `wrap_title_memory`, `gpu_address`, `create_compute_pipeline(dxil)`,
`submit(queue, Submission)`, `signal`, `completed`, `wait`. A `Submission` is what one
`ExecuteCommandLists` is: fence waits, dispatches (pipeline, root arguments, resident buffers, group counts),
fence signals. `submit` never waits for the GPU. `metal::Metal` implements it:

| Trait | Metal |
|---|---|
| queue | `MTLCommandQueue` (any number) |
| fence | `MTLSharedEvent`; GPU `encodeWaitForEvent` / `encodeSignalEvent` in the command buffer, CPU `setSignaledValue`, `signaledValue`, `waitUntilSignaledValue` |
| `wrap_title_memory` | `newBufferWithBytesNoCopy`, shared; `gpu_address` = buffer GPU address + offset of the title address in the range |
| `create_compute_pipeline` | `IRCompilerAllocCompileAndLink` -> metallib -> `MTLComputePipelineState` |
| `submit` | one command buffer: waits, one compute encoder with every dispatch in order, signals, `commit` |
| dispatch | `RootArgs` (inline root constants, root SRV/UAV/CBV as 64-bit GPU addresses, in root-parameter order) set with `setBytes` at slot 2; `useResource` for the wrapped buffers |

Tests (`cargo test -p victory-metal`, `src/metal.rs`; the shader is `tests/affine.dxil`, 3.4 KB made by dxc
from the HLSL of `--demo`, our own, not a title shader; skipped when the converter is not installed):

- `queue_b_waits_for_queue_a_over_the_same_buffer`: the D3D12 pattern. Queue B is submitted first, waits for
  fence value 7 and dispatches `c = b * 3`; the test checks B has not run (`wait` times out after 50 ms), then
  queue A dispatches `b = a * 2 + 1` and signals 7; the CPU waits on B's own fence; `c[i] == (a[i]*2+1)*3`
  for all 4096, over one zero-copy buffer holding a, b and c.
- `cpu_signal_releases_a_waiting_queue`: the title's CPU `Signal` (`hold.rs`) releases a queue held at a
  `Wait`; result checked.
- `many_submissions_in_flight_one_wait`: 64 dispatch submissions over 4 queues, chained by fence values
  `k -> k+1`, all submitted before any is waited for, one CPU wait; each adds 1 over ping-pong arrays,
  64 on every element (about 20 ms in all including first use; a queue runs its command buffers in order,
  so submission is in order and only the fence orders across queues).

### Where victory-gpu calls it

`host.rs` today owns one D3D12 device and direct queue and waits after every submission; `queue.rs` holds
Signals and memory-value waits (`hold.rs`); `fence.rs` keeps the 64-bit value and the waiting events. The
title-facing COM objects stay; what they call changes:

| victory-gpu | calls on the backend |
|---|---|
| `host.rs` device and queues; `queue::create` (`ID3D12CommandQueue`, a per-type queue: direct, compute, copy) | `create_queue` per title queue (GTA VI's four compute queues become four Metal queues) |
| `fence::create`; `Fence` value | `create_fence(initial)` next to the `Fence` struct; the title may still read the value from memory, so the struct keeps its own atomic and the shared event is the GPU-visible twin |
| `fence::signal` (CPU), `fence::set` | `signal`; waiters (`SetEventOnCompletion`) become `completed` plus a `notifyListener`, or `wait` on a helper thread, instead of the event list polled by `pump_held` |
| `ExecuteCommandLists` in `queue.rs`, and `hold.rs` for a wait the GPU cannot express (`Wait32BitValueX` on memory) | `submit`: GPU fence waits and signals go into the `Submission`; only memory-value waits remain host-side holds |
| `list.rs` recorded commands | recorded into `Dispatch` (compute) and, later, render passes; the `Submission` is built at `ExecuteCommandLists` |
| `shader_cache.rs` `HostShader` | `create_compute_pipeline(dxil)` (cache the metallib beside it) |
| `shadow.rs`/`watch.rs`/`memory.rs` | `wrap_title_memory` and `gpu_address` for each pool, from the native host's map events |

The trait would move to a small crate both backends depend on, once victory-gpu uses it; for now it lives
beside the Metal implementation and victory-gpu is untouched.

Not in this slice (the next section adds heaps, textures and render passes): static samplers,
`MTLResidencySet` for the title pools, command-buffer error and timeout handling (a GPU fault is only
seen as a missing signal here), thread-safety of the types (they are not `Send`).

## Heaps, textures, render passes (second slice: `resources.rs`, `gfx.rs`)

`resources.rs` is the platform-neutral half (no Metal types): `ResourceBackend: ComputeBackend` (heaps,
descriptor writes, textures, views, samplers) and `GraphicsBackend: ResourceBackend` (render pipelines,
`submit_passes`). `gfx.rs` implements both for `Metal`; compute `submit` is now a thin wrapper over
`submit_passes` with one compute pass. Tests are integration tests (`tests/resources.rs`,
`tests/render.rs`, helpers in `tests/common`), native arm64, skipped without the converter.

### Fixtures

`tests/fixtures/*.hlsl` are our own shaders; the `.dxil` beside each is what dxc made of it and is
committed (3-4 KB each; `.gitignore` has a `!crates/victory-metal/tests/**/*.dxil` exception: the
`affine.dxil` of the first slice had been ignored and never committed). Regenerate with the x86_64 build
under Rosetta, which can `dlopen` the x86_64 `libdxcompiler` of the D3DMetal bundle:

```
cargo build --target x86_64-apple-darwin -p victory-metal      # toolchain 1.98.1 has the x86_64 std
VICTORY_DXC_DYLIB=<D3DMetal.framework>/Versions/A/Resources/libdxcompiler.dylib \
  target/x86_64-apple-darwin/debug/victory-metal --dxc in.hlsl out.dxil cs_6_0
```

No title shader or title code is used anywhere.

### Descriptor heaps

A heap is an `MTLBuffer` of 24-byte `IRDescriptorTableEntry { gpuVA, textureViewID, metadata }`, written
with the layout of `IRDescriptorTableSetBuffer/Texture/Sampler` (`metal_irconverter_runtime.h`),
re-implemented in a few lines (`Metal::write_buffer/write_texture/write_sampler`), so Apple's header is
not needed to build:

| Entry | gpuVA | textureViewID | metadata |
|---|---|---|---|
| raw buffer SRV/UAV | buffer GPU address (title address + delta) | 0 | size in bytes (low 32 bits) |
| typed / 4–16 B structured buffer SRV/UAV | buffer GPU address (title address + delta) | `MTLTextureTypeTextureBuffer` `gpuResourceID` | size in bytes (low 32 bits) |
| texture SRV/UAV | 0 | `MTLTexture.gpuResourceID` | `f32` bits of min LOD clamp (low 32 bits) |
| sampler | `MTLSamplerState.gpuResourceID` | 0 | `f32` bits of LOD bias |

A root descriptor table is the **GPU address of its first entry** in the top-level argument buffer
(`heap_address(heap, i)` pushed with `RootArgs::address`, like a root view); `DescriptorTable(SRV(t0),
UAV(u0))` is two adjacent entries and a second `DescriptorTable(Sampler(s0))` points into the sampler
heap. The heaps are also bound at slots 0 and 1 for shaders that index `ResourceDescriptorHeap[i]`, for
every stage (`victory-gpu`'s `native::heap_tests` draw with vertex, mesh and pixel shaders that do). Typed buffer views and 4-, 8-, and 16-byte structured buffer views (`Metal::write_typed_buffer`) lazily populate `textureViewID` (`MTLTextureTypeTextureBuffer` over the pool `MTLBuffer`) alongside `gpuVA` only when a bound shader uses DXIL `TypedBuffer` operations, while keeping only the pool `MTLBuffer` in `alloc_counts` / `MTLResidencySet`.

Proven by `srv_and_uav_through_a_descriptor_table` (a compute shader reads an SRV and writes a UAV, both
zero-copy buffers, through a table at heap entry 5; then entry 5 is rewritten to point elsewhere and the
next dispatch follows it) and by every sampling test below.

### Residency (what a heap needs)

Resources reached only through a heap entry are **not** resident unless named. Measured: with neither a
residency set nor `useResource`, every heap test returns wrong data (Apple Silicon does not make an
unnamed resource work by luck).

On a Mac with 32 GiB of memory or more (`hw.memsize`), and on any Mac with
`VICTORY_METAL_RESIDENCY_SETS=heap`, a heap carries its own `MTLResidencySet`: the heap buffer and
each allocation its entries name. An allocation is added when an entry first names it and removed (and the set committed) before
the last entry naming it is overwritten or the heap is dropped; the set does not retain what it holds,
the heap does. The set is committed when entries changed and given to each command buffer that uses
the heap (`useResidencySet`), so what the heap names is resident while that command buffer runs:
residency follows the frame, and nothing is rebuilt per command buffer. `clear_entry` empties an
entry (zero bytes, nothing kept): the host clears the entries of a texture the title released, so
what the heap keeps alive and wired does not grow with play time.

`VICTORY_METAL_RESIDENCY_SETS=0` names every allocation a heap names to every encoder with
`useResource` instead, which is also what a Mac with less than 32 GiB does when the switch is unset.
The 32 GiB line is a guess, not a measurement: the world's heap set holds 17 to 18 GB (`heap_set_mb`
on the frame line), more than all the memory of a 16 or 18 GB Mac, and no such Mac has run it yet;
the process's footprint was the same either way on 48 GB, so it may well be safe, and `heap` is there
to find out. The choice is logged once at start-up (`metal: residency of what descriptor heaps name`,
with the mode, the Mac's memory and the switch). In GTA VI's world the heap names about 37,000
allocations (130,000 entries), and the kernel took about 155 ms to take that list for each render pass
that draws (10 ms for a compute pass), whatever the pass's shaders, with the GPU idle. Measured with
each pass in its own command buffer: 164 drawing passes, 0.153 s each between the kernel taking the
command buffer and the GPU starting it; with the sets, 0. The 35 drawing passes of a world frame's
main submission were 5.4 s. With the same build and only the switch changed, world frames went from
0.07 fps (15 frames in 218 s) to 0.47 fps (101 frames in 213 s; 0.6 once the world's pipelines are
made) on an M5 Pro with 48 GB, and the process's footprint at the first world frame was 16 GB either
way.

`VICTORY_METAL_RESIDENCY_SETS=1` also puts the title's pool buffers in one set on every queue (below).
That keeps all of title memory (about 6.7 GB on a Series S profile) resident for the whole process:
on an M3 Pro with 18 GB, command buffers began to time out after the world load
(`kIOGPUCommandBufferCallbackErrorTimeout`, and `...OutOfMemory` with the host's texture copies). So
by default pool buffers are still named by the encoders that use them (`Dispatch::resources`, a
handful per pass).

With a heap's set, Metal's hazard tracking no longer sees what a pass reaches through the heap, so the
host orders the queue's work itself (the queue event and the fence under **Residency** below).
The heap tests run in all three modes (`common::backends()`).

`VICTORY_METAL_REQUEST_RESIDENCY=1` (off by default) makes each heap's set ask to stay resident
(`requestResidency` when it is made and after each commit; `endResidency` when the heap goes), so
the system does not make it non-resident between the command buffers that use it and resident
again for the next. In a 330 s run on a 48 GB Mac the wired memory fell from about 22 GB to 6 GB
and came back within 10 s twice in the world, and command buffers later timed out (#79); whether
that churn is the stall is a guess this switch tests. The frame line counts nothing for it; the
start-up residency line logs `request_residency`.

Not known: how the default does on a Mac with 16 or 18 GB; it was run with 48 GB. The frame line
(`native draws so far`, every 64 frames) logs `heap_set_mb`, the most bytes one heap's set held
(pool buffers its entries name count whole), `heap_allocations`, and, to see where the memory is
when command buffers stall (#79), `metal_mb` (the device's `currentAllocatedSize`) and
`footprint_mb` (the process's physical footprint). Not done: an entry that names a
texture the title has released keeps it, alive and in the set, until the title overwrites the entry,
as naming per encoder did; `useHeap` for `MTLHeap`-placed textures.

### Textures and samplers

`create_texture` takes `TextureDesc`: 2D, 2D array, 3D; formats RGBA8 (unorm, sRGB), BGRA8, RGBA16F,
R32F, D32F, BC1-BC7; mips; render-target and UAV flags (usage always includes `PixelFormatView`). Shared
storage everywhere (Apple Silicon allows it for render targets and depth too). `upload_texture` /
`read_texture` are `replaceRegion` / `getBytes` of a whole mip, BC in blocks (`Format::row_bytes`).
Findings:

- All seven BC formats create and upload; a BC1 texture samples correctly (four colour blocks). RGBA8,
  RGBA16F, R32F, 2D array (per-slice values) and 3D (per-voxel values) sample exactly at texel centres.
- Views (`newTextureView...levels:slices:`): a mip-1-only view and an sRGB reinterpretation of an RGBA8
  texture sample as expected. **Metal does not allow `Depth32Float` to be viewed as `R32Float`**
  (an assertion abort, not an error return: "not castable"); a depth texture is written to the heap
  itself and the shader samples it as `Texture2D<float4>` (depth in `.x`), which works. So `twin.rs`'s
  colour twin is still the way to a UAV on depth, to be checked on real shaders. `create_texture_view`
  refuses a format Metal cannot cast the texture to (`castable`: uncompressed formats of one texel
  size, or a BC format and its sRGB or signed twin; checked: R32Uint, RGB10A2 and BGRA8 views of an
  RGBA8 texture are made, R8, RGBA16F, BC1 and depth ones are errors).
- Samplers: `MTLSamplerDescriptor` with `supportArgumentBuffers = true` (without it there is no resource
  ID). Linear blends and nearest does not, through a sampler heap.
- **Textures over zero-copy memory**: `newTextureWithDescriptor:offset:bytesPerRow:` on the wrapped title
  buffer. 2D, one mip, one slice only (Metal's rule for buffer-backed textures); offset and row pitch must
  be multiples of `minimumLinearTextureAlignmentForPixelFormat` (`linear_alignment`). A CPU write into
  title memory after the texture exists is seen by the next dispatch. Mips, arrays, 3D, BC and the
  console's tiled layouts are not linear, so those still need the compute detile into an `MTLTexture`.

**Dropping the top mip for memory (`VICTORY_TEXTURE_DROP_TOP_MIP=1`, off by default).** For hosts whose
unified memory cannot hold the title's full texture set (an iPad keeps about 9 GiB before it swaps), a
sampled-only 2D or array texture whose larger side is at least 1024 texels and that has a level below
its top is made without that top level (half the width and height, one level fewer: about 75% of the
texture's memory saved). `detile` (with `skip`) turns title level `m` into host level `m - 1` and never
detiles level 0; `host_mip_range` maps a view's title levels onto the host's. Render targets, UAV and
3D textures are never touched. Shortcut, not console behaviour: implicit-LOD sampling simply uses the
next level (a little blurrier up close), explicit-LOD samples come out one level blurrier, and integer
texel loads and `GetDimensions` see the smaller texture. Code: `native/textures.rs`; tests
`drop_top_mip_tests` and `a_texture_made_without_its_top_level_gets_the_lower_levels`.

### Render passes

`RenderPipelineDesc` (VS and optional PS DXIL, input elements and streams, target and depth formats,
depth test, cull) -> `MTLRenderPipelineState` + `MTLDepthStencilState`. `Pass::Render(RenderPass)` has
colour and depth targets (clear or load) and `Draw`s: pipeline, topology, root arguments, heaps, vertex
buffers, index buffer, `DrawCall::Arrays | Indexed`. A submission interleaves compute and render passes
in order in one command buffer.

Vertex fetch uses the converter's default **Metal vertex fetch**
(`IRStageInCodeGenerationModeUseMetalVertexFetch`): the vertex function has `[[stage_in]]`, so the host
supplies an `MTLVertexDescriptor`. Observed, and what `gfx.rs` relies on: input element *k* of the VS
input signature is attribute `kIRStageInAttributeStartIndex + k` = 11 + k, and vertex buffer slot *s* is
Metal buffer index `kIRVertexBufferBindPoint + s` = 6 + s (indices 0-5 are the converter's: heaps 0/1, root
arguments 2, hull/domain 3, draw arguments 4, uniforms 5). Vertex pulling (a table of
`IRRuntimeVertexBuffer { addr, length, stride }` at slot 6, no descriptor) is the alternative when the
title's input layout is dynamic; not built. A draw must set what `IRRuntimeDrawPrimitives` /
`IRRuntimeDrawIndexedPrimitives` set, which is reimplemented: a 5-word `IRRuntimeDrawParams` (count,
instances, start, base vertex/instance) with `setVertexBytes` at slot 4 and the index type (0 =
non-indexed, 1 = u16, 2 = u32) at slot 5. The root arguments go to both stages (`setVertexBytes` and
`setFragmentBytes` at slot 2), heaps likewise.

`tests/render.rs`: `triangle_offscreen` (clear to blue, one non-indexed triangle in a 16x16 RGBA8 target:
inside red, corners blue, orientation checked); `indexed_draw_with_depth_test_and_depth_sampling` (a
u16-indexed quad at z 0.25 and a later full-screen triangle at z 0.75 on a D32F attachment with a less
test: quad pixels green, triangle only where it is alone; then, in the same command buffer, a compute pass
samples the depth attachment: 0.25 and 0.75 exactly); `depth_test_rejects_a_draw_behind`. The pixel
shader multiplies the vertex colour by a texture sampled through a descriptor table and a sampler heap, so
heaps are exercised by a draw too.

Not covered: blending and colour write masks, stencil, multiple render targets (the code takes several,
untested), MSAA, viewport/scissor from the list (full target only), instancing with a per-instance stream,
topologies beyond triangle lists, primitive restart, amplification pipelines,
hull/domain/geometry emulation, stream-out, indirect draws, UAV writes from pixel shaders,
`ExecuteIndirect`, queries, pipeline caching.

**Mesh pipelines.** When the converter makes a mesh function of the "vertex" DXIL (`IRShaderStageMesh`,
5: the translator's `ms_6_6` output for mesh-style console programs), `create_render_pipeline` makes an
`MTLMeshRenderPipelineDescriptor` with that function and the pixel shader's, the same colour and depth
formats, blend and write masks, and no vertex descriptor. Its thread-group size is the converter's
reflection of the mesh shader (`IRShaderReflectionCopyMeshInfo`, `Msc::mesh_threads`, cached on disk
as conversions are), kept on the pipeline (`RenderPipeline::mesh_threads`) for the draw.
Test: `victory-gpu` `native::tests::mesh_tests::a_mesh_style_vertex_program_makes_a_mesh_pipeline`.
A mesh pipeline draws with `DrawCall::Mesh { groups }` (`NativeList::dispatch_mesh`):
`drawMeshThreadgroups` with that thread-group size and no object stage, or, for an `ExecuteIndirect`
record (three group counts, Metal's `MTLDispatchThreadgroupsIndirectArguments` word for word),
`drawMeshThreadgroupsWithIndirectBuffer`; a counted record goes through the same gate as counted draws
(three words). A mesh call of a vertex pipeline, or the reverse, is dropped. Test:
`mesh_tests::a_mesh_draw_colours_the_target`. The mesh stage is bound as the vertex stage is: root
arguments with `setMeshBytes` at slot 2, the heaps with `setMeshBuffer` at 0 and 1, and every resource
and static-sampler table used for the mesh stage (`mesh_tests::a_mesh_shader_reads_its_root_constants`).

### Mapping into victory-gpu (not done; victory-gpu is untouched)

| victory-gpu today | calls on `ResourceBackend` / `GraphicsBackend` |
|---|---|
| `descriptor.rs` / host heap (`CreateDescriptorHeap`, `HostHeap` writes of RDNA descriptors) | `create_heap(kind, count)`; each title descriptor write becomes `write_buffer` / `write_texture` / `write_sampler` at the heap index. The title writes 32-byte RDNA descriptors itself, so the host decodes them (base, size, format, tiling) into one of the three calls, as `HostHeap` already does for D3D12 |
| `root.rs` / `pipeline/host.rs` root parameters | `RootArgs`: constants inline, views as addresses, tables as `heap_address(heap, first)` |
| `resource.rs` committed/placed textures, `texture.rs` | `create_texture`, `create_texture_view` (SRV/UAV/RTV/DSV formats); the detile compute writes the texture, or `create_texture_over_memory` where the layout is linear |
| `placement.rs` heaps | not yet: textures are individual allocations here; an `MTLHeap` variant of `create_texture` plus `useHeap` residency |
| `pipeline.rs` graphics pipelines made at first draw | `create_render_pipeline` (the shader cache's DXIL, input layout, formats, depth); blend state is the missing field |
| `list.rs` draws, clears, barriers | `RenderPass` (clears are load actions; a mid-pass clear splits the pass), `Draw`; barriers are implicit between passes and encoders |
| `queue.rs` `ExecuteCommandLists` | `submit_passes` |

## Next steps

1. Wave64 compute shaders: decide how the translator serves a wave32 host (528 of 3 549 compute shaders,
   IRError 6). Mesh output sizes (28 shaders).
2. Real render pipelines: pass vertex and fragment functions of one pipeline together (the harness used a
   synthetic partner), stage-in layout, blend and target formats, then mesh and amplification pipelines.
3. Add the host interface above to the native-host agent's work, then wrap real pools and re-run the
   probe on them (memory entry mappings, guard pages, 64 KiB mapping granularity on 16 KiB pages).
4. Implement the seam in victory-gpu behind a feature: D3D12 and Metal implementations of `ComputeBackend`,
   `queue.rs`/`fence.rs` calling it, with `hold.rs` keeping only memory-value waits.
5. Descriptor heaps are in the backend (second slice); remaining: decode the title's 32-byte RDNA
   descriptors into the three `write_*` calls, typed buffer views, `MTLHeap` placement.
6. Replace the page-table "Title memory" contract in `victory_shader::emit` by the arena/offset one,
   and delete copy-back, view marks and the shadow (`shadow.rs`, `watch.rs`) behind a feature once the
   Metal path runs.
7. Compute detile of the console texture layouts reading title memory; present on `CAMetalLayer`.

## The seam in victory-gpu (branch `native/gpu-seam`)

What landed, and what did not. The D3D12 path is unchanged: the only edits to existing victory-gpu
files are in `sampler.rs` (the console sampler fields, now platform neutral; `host()`, `console()` and
the tests stay `cfg(windows)`), `descriptor.rs` (`record` and `increment` are `pub(crate)`) and
`lib.rs` (`mod sampler` always built, `pub mod native`).

- **`crates/victory-backend`**: the traits (`ComputeBackend`, `ResourceBackend`, `GraphicsBackend`) moved
  out of `victory-metal`, which re-exports them as `victory_metal::{backend, resources}`. victory-gpu
  depends on it on every target and on `victory-metal` on macOS only. New: `Pass::Blit(Vec<BlitOp>)`
  (`CopyBufferRegion` and byte fills on title memory, `MTLBlitCommandEncoder`, ordered with the
  passes around it).
- **`victory-gpu/src/native`**, generic over `GraphicsBackend`, so it is built and linted on the
  Windows target as well; `native::Host` is `NativeHost<Metal>` on macOS. Module selection is by
  `cfg`: `host.rs` and everything that touches `shadow`/`watch` is Windows only and cannot be
  reached from `native`.
  - `memory`: `PoolTable`, the title page to `(pool, offset)` table. `PoolInfo` / `PoolObserver` have
    exactly the shape of `victory_hle::modules::xmem_physical` (`pool_created`, `pool_released`,
    `mapped`, `unmapped`; `native/xmem-rest` has them), so the host wires them with a forwarding
    `impl` (victory-gpu does not depend on victory-hle). Pools are wrapped lazily, one buffer per pool,
    and dropped on `pool_released`. **Bypass of shadow copies, copy-back and write guards**: there is
    no code path from `native` to them; a buffer is the pool's own mapping, the title's address is an
    alias of it, and `PoolTable::resolve` is the only thing between a title address and a GPU
    address. Nothing is tracked, refreshed or copied. Not wired yet: the call sites in `xmem` that
    would register the observer (a `Host::memory()` handle is what to pass).
  - `descriptors`: `descriptor::View` records (what `CreateShaderResourceView` and friends leave in
    the 32-byte slot as `[tag, index, generation]`) decoded into `Decoded::{Buffer, Image,
    Sampler}`: raw/typed/structured buffer SRV and UAV ranges (the element size rules of
    `descriptor/host.rs`), constant buffers, placed buffer views, Texture2D mip ranges, and the
    console sampler (the same `sampler::Fields` the D3D12 host builds its description from).
    `NativeHost::mirror(handle)` is the `descriptor::mirror` analogue: decode and `write_buffer` /
    `write_texture` / `write_sampler`.
  - `list`: `NativeList`, the recorded calls (pipelines, root constants/views/tables, heaps,
    dispatch, copy, fill, render targets, clears, vertex/index buffers, draws) with title
    addresses kept as addresses.
  - `host`: queues, fences, pipelines, heaps, textures; `execute(queue, lists, waits, signals)`
    resolves every address, groups commands into compute / blit / render passes and calls
    `submit_passes`. It never waits; completion is a fence value.
- **Tests** (`cargo test -p victory-gpu`, macOS arm64, skipped without the Metal Shader Converter): a
  list recorded as a title would record it, title memory shown at an alias address of the pool, run
  through `NativeHost`, results read back through the title's alias:
  `compute_dispatch_writes_title_memory` (GPU copy, GPU fill, root constants and views, a dispatch
  that writes title memory), `descriptor_table_buffers_decode_into_metal_heap` (descriptors written by
  the `create_placed_*` device calls, decoded, reached through a root table; a second submission
  on the same queue waits on the first's fence value), `clear_and_draw_into_render_target` (clear,
  draw with a texture and sampler through descriptor tables, pixels read back),
  `a_command_at_unmapped_title_memory_is_dropped_alone`; plus unit tests of the pool table and the decoders.

### The title-facing objects on the native host (branch `native/gpu-wire`)

On macOS `queue.rs`, `fence.rs`, `list.rs`, `descriptor.rs` and `resource.rs` call
`crates/victory-gpu/src/native/glue.rs` where they call `crate::host` on Windows (the D3D12 path is
unchanged). `glue` holds the one `Title<Metal>` (`native/title.rs`, generic over the backend) behind a lock:
the backend's objects and the `Rc`s in recorded lists are not thread safe, so every call that touches them
takes it (before a list's own lock), lists are dropped under it, and a timer thread (1 ms) moves held work
and raises fence values.

- **Lists** record into a `TitleList` (`NativeList` + the segments cut off it): root constants (also at an
  offset), root views, tables (the heap is made when a list first names it, from what the title wrote by
  then), `SetDescriptorHeaps`, pipelines, dispatch, copies, topology, viewport, scissor, stencil reference,
  blend factor, vertex and index buffers, draws, render targets and clears, bottom-of-pipe value writes
  (byte fills, GPU ordered). A `Wait32/64BitValueX` after commands cuts the list; the segments of a title
  `ExecuteCommandLists` go through `hold.rs` (unchanged) per title queue: a segment whose waits hold goes to
  `NativeHost::execute` at once, the others stay on the CPU until the memory value is there (read with
  `mach_vm_read_overwrite` where it is not in a pool).
- **Queues and fences**: one backend queue per title queue; `Signal` and `Wait` are `MTLSharedEvent`
  signal and wait in the queue (behind held work they are held, in order); the title's fence keeps its own
  atomic (titles read it from memory) which the timer and every poll raise to the event's value; `Signal`
  from the CPU signals the event; `SetEventOnCompletion` with no event blocks while pumping held work, with
  an event it calls the hook set with `native::set_event_hook`.
- **Pool observer**: `victory_gpu::native::pool_observer()` returns an `Arc<dyn PoolObserver>` over the pool
  table; `victory-hle`'s observer forwards `pool_created`, `pool_released`, `mapped`, `unmapped` to it.
- **Shaders** (`native/shaders.rs`, `native/state.rs`): a title pipeline's containers are translated with
  `Options { wave32_host: true, .. }` on first use, DXIL goes to the converter through the backend (one
  library file name per library, pipelines cached by the BLAKE3 of their DXIL), graphics state is read from
  the pipeline-state stream (blend and write masks per target, depth, stencil, rasterizer, input layout in
  the vertex shader's `ISG1` order, target formats) into `RenderPipelineDesc` (backend: blend, stencil,
  winding, fill, depth bias, depth clip; per draw: viewport, scissor, stencil reference, blend factor).
  A shader the host cannot run is refused with a reason, counted (`glue::shader_stats()`) and its work
  dropped; the submission and its fences go on. Refused today: title memory read at computed addresses in
  a graphics pipeline (compute shaders have it, below), heap indexing by a compute shader (graphics
  shaders may), root views as heap
  views, write marks and view addresses, a pixel stage with arguments of its own (one argument block
  serves a pipeline), multisampling, vertex inputs missing from the layout.
- **Title memory at computed addresses: the pool-offset contract** (`native/addr.rs`, compute shaders). A
  shader that reads or writes through a pointer it computes (`global_*`, `s_load` from a loaded address,
  atomics, BVH nodes) finds the memory through the translator's page table (`emit` "Title memory"): per
  64 KiB title page, the resource-heap index of a raw buffer view and the title address it starts at
  (`offset = address low dword - start`). DXIL cannot dereference a 64-bit address, and the converter's raw
  buffers are `{GPU address, size}` descriptors read from the heap, so this is the construct Metal accepts;
  the translator is unchanged and only the host's side differs from D3D12's (views of host copies):
  - the host's views are of the pools themselves, one SRV/UAV pair of descriptors per `(pool, 2 GiB
    chunk)` (a chunk and a page long: a view's size is 32 bits), at the heap indices just past the title's
    own entries: a resource heap is made `addr::RESERVED` (512) entries longer than the title asks, and a
    shader that runs where the title bound no heap gets a heap of the host's own. Pair 0 is the null view:
    size 0, so with the converter's bounds check (`RootLayout::bounds_checked`) a page no pool shows reads
    0 and its writes are dropped. Each heap has a table of its own, since the table holds its view indices
    (heaps must have room below the translator's clamp of heap indices, 2^21 entries);
  - the table is built from `PoolTable::mappings()` into a buffer of its own whenever the pool table's
    generation changed (never rewritten in place: the GPU may be reading the old one, which is a buffer of
    the backend's own that the submissions made with it keep until they have run; the host used to free a
    table 8 rebuilds after it was replaced, and work a queue held meanwhile then wrote through freed memory)
    and bound as the last root parameter. shortcut: rebuilt whole, not incrementally (8 bytes a page: 2 MiB
    for 16 GiB of mappings); a 64 KiB title page not wholly inside one mapping of one pool is a null page;
    at most 255 view pairs;
  - `Options::zero_copy_memory` (on for the host, off for D3D12): the shader marks no pages, so there is
    no bitmap, no `u0` and no atomics per store: shadow copies are gone, nothing needs to know what was
    written. Misses are not recorded;
  - translated shaders carry no root signature, so `victory_backend::RootLayout` tells the converter
    (`IRCompilerSetGlobalRootSignature`, heaps directly indexed) the layout `NativeHost::translated_args`
    fills: the user-data constants, root CBVs, SRVs and UAVs, then the table.
  A global load and store (SGPR base and VGPR-pair address), a scalar load through a pointer loaded from
  memory, and an atomic add run on the GPU at title addresses no CPU pointer is at, with remaps, a title
  heap and the host's own, and misses (`native/addr_tests.rs`).
- **Textures** (`native/textures.rs`): a title texture object gets a host texture when a view, render target
  or clear first names it; its tiled memory (`victory_formats::texture`, the addressable modes: linear, the S
  and D modes, and the XOR modes `_X` and `_T`, below) is detiled on the CPU, mip by mip, when a hash of the memory changed at a
  submission; block-compressed formats as blocks; textures named by address and description
  (`CreateShaderResourceViewPlaced`) are made by the view; layouts the host cannot address are counted.
- **Present**: `native::set_layer(CAMetalLayer*)` (the app's, or `native::create_layer` for one without a
  window) and `PresentX` draws the first resource of the largest plane over the next drawable with a
  full-screen triangle, after the queue's earlier work (`PresentBackend`).
- **Residency**: with `VICTORY_METAL_RESIDENCY_SETS=1`, pool buffers join one `MTLResidencySet` when first
  used by a submission, the set is added to each queue once, and encoders skip `useResource` for them; a
  buffer leaves the set when dropped. By default they are named by the encoders (see "Residency (what a
  heap needs)"). `useResource` also told Metal's hazard tracking which buffers an encoder touches by GPU
  address; without it nothing orders a copy before the dispatch that reads its result (the one flaky test
  that showed it), so on a queue whose work uses sets (the heaps' own by default, the pool's too with the
  switch) every command buffer waits for the previous one through a queue `MTLEvent` and every encoder
  for the one before through an `MTLFence`: work on a queue is serial. D3D12 barriers
  (`ResourceBarrier`) are still not translated; a title's UAV barriers are covered by that serialisation.

### What is left

1. **The translated-shader contract on Metal.** Compute shaders now run end to end, with title memory at
   computed addresses (above), user data in root constants and root views. Not done: graphics pipelines
   with translator output (the page table, immediate samplers as static samplers of the converter's root
   signature, per-stage argument blocks: the D3D12 host's visibility has no equivalent in one argument
   buffer), heap indexing by a compute shader, view addresses and write marks. The table is rebuilt whole when a
   mapping changes (incremental update is the next step), and the null view relies on the converter's
   bounds check, which was verified for loads and stores of a 64-lane compute shader only.
2. Textures: the XOR modes are addressed by addrlib's RB+ equations (`victory_formats::texture`, tables
   generated by `tools/swizzle/gen_xor_tables.py`, checked against the modes without XOR and against a
   reference evaluated from the header's text by `tools/swizzle/xor_reference.py`), so a texture in
   `_X` or `_T` that the title writes with the CPU or a copy is no longer empty. They depend on the GPU's
   pipe configuration (`PipeConfig::DEFAULT`, eight pipes on four packers, an assumption) and on the
   surface's pipe-bank XOR (`Layout::with_pipe_bank_xor`, 0 until the host reads it from the descriptor's
   address): until both are checked against a captured console texture the pipe and bank bits of a mode
   (bits 8 and up of the block offset) can be wrong, the micro-tile bits are not. `SW_4KB_R_X` and
   multisampled layouts have no equations. Render targets are still the host's, not detiled.
   The detile is CPU and hashes the memory of every sampled texture per submission (a compute
   detile over the pool buffer and a per-page change test are the next step); render targets and depth
   targets are not read back to title memory (`CopyTextureRegion` to a buffer, the title's CPU reads);
   clears ignore their rectangles; array and 3D views are not decoded (`Decoded::Image` has no slices);
   resolves, texture copies and uploads from buffers are ignored.
3. `ExecuteIndirect`, mesh shaders, queries, 64-bit image atomics, a mid-pass clear splitting the pass,
   stencil-only formats and the console's 24-bit depth (shown as 32-bit float with stencil).
4. Plain D3D12 sampler descriptions (13 words, `View::Sampler`) are not decoded.
5. Buffer views across two adjacent mappings of one pool are refused (`PoolTable::resolve`).
6. Present is not held behind work the queue still holds back (as on the D3D12 host), takes one plane, no
   overlay composite, no HDR or colour space, no frame pacing, and `NSWindow` creation is the app's.
7. The Windows build was only checked with clippy (`--target x86_64-pc-windows-gnullvm`, `-D warnings`)
   and the existing tests, which this change cannot affect (no Wine run).

## Wave64 compute on 32-lane Apple GPUs (`Options::wave32_host`)

The translator can now emit wave64 compute shaders without `WaveSize(64)` (off by default; D3D12 unchanged).
Choice: keep one thread per console lane and run a console wave as two host waves (flat ids `64n..64n+31`,
`64n+32..64n+63`); the console lane is the group index modulo 64. What stays inside 32 lanes is native (DPP
rows, quads, `ds_swizzle`, `ds_bpermute`, `v_mbcnt`, `v_writelane`, lane-bit tests of masks held as numbers,
`v_readfirstlane` per half). What crosses the halves (ballots read as data, `v_readlane`, the
reductions and prefix sums of whole-wave sections) goes through a `xchg` group-shared array (one dword per
thread) between two group barriers. This was preferred to two lanes per thread because the emitter lowers one
lane per value. Cost: two barriers and a shared-memory round trip per such operation, 4 bytes per thread of
group-shared memory, `WaveSize(32)`.

Group barriers need every thread, so these operations are lowered only where each thread runs the block exactly
once, or (group of one console wave) behind uniform scalar branches outside loops. In EXEC-divergent code or
per-lane loops the value is left unknown and the program is refused if anything uses it; LDS shared without a
barrier in divergent code is refused too. The barriers that sync such LDS (`split_wave_barriers`) go in the
same places: blocks each thread runs once, or (the group being one console wave) reached by uniform scalar
branches alone outside loops. A scalar branch on an SCC computed from a `v_readfirstlane` does not count as uniform
here: each host wave reads its own first lane, so the halves can take different ways and never meet at the
barrier (`split_scc`); code behind one is treated as divergent. A program refused for either is translated again with each `s_cbranch_execz`
to a bare `s_endpgm` dropped (`ir::keep_lanes_to_end`): the usual `if (id < n) { whole shader }` early exit.
Its lanes then run on to the end with EXEC off, as on the console while their wave has a lane left, and reach
every barrier. Only a wave with no lane left runs that code too, which changes nothing but scalar memory
writes (a program with one is not changed). Of 30 dumped story-load shaders refused for these two reasons,
9 translate with this and the barriers behind uniform branches (27 November). 16-bit wave reductions across the halves are refused. Fixing the
divergent case needs if-conversion of the region so both halves always arrive (not done).

Verified: unit tests with synthetic programs (`emit/wave32_tests.rs`); the lowered containers pass
`metal-shaderconverter` (no IRError 6). Not measured: the 528 corpus shaders (the cache holds translated DXIL,
not console shaders; re-translate with the option to count). Not run on the GPU: the compute backend has no
descriptor heaps or root signatures for translator output yet.
