# Native macOS host

This is the design and status of the host that runs VIctory on macOS with no Wine: a macOS process that maps the title's x86-64 PE image itself, binds its imports to `victory-hle`, and runs it. Today the title's code runs as x86-64 under Rosetta in an x86-64 macOS process. The ARM64 static recompiler will plug into the same host interfaces from an arm64 process, and the Metal backend will take over from D3D12 behind `victory-gpu`.

Wine-based work continues on the `test/run-C*` branches. This branch (`native/host-2`) only adds code beside it. On Windows and Wine the loader and runtime behave as before.

## macOS setup and launcher paths

`native-setup.sh` and `run-native.sh` use Cargo's resolved `target_directory`
from `cargo metadata --format-version 1 --no-deps --offline`. This honors
`CARGO_TARGET_DIR` and local/global `build.target-dir` configuration, including
paths with spaces (the setup finding in [#13](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/13)).
Both select `x86_64-apple-darwin/release/victory` within that directory. Use the
same Cargo configuration when building and launching; a one-off build with
`--target-dir` must be matched by configuration or an environment override when
launching. The launcher needs Cargo available and reports configuration errors
rather than falling back to a possibly stale default binary.

Both scripts also find rustup when it is installed but not on PATH yet: in
`$CARGO_HOME/bin` (default `~/.cargo/bin`), where the rustup.rs installer puts
it before a new terminal opens, and in `$HOMEBREW_PREFIX/opt/rustup/bin`, since
Homebrew's rustup formula is keg-only. Setup warns and uses it. The launcher
uses it only when no `cargo` is on PATH, and says so plainly when there is
neither.

Game-free regressions use real Cargo resolution and fixture executables under
stock macOS Bash 3.2: `/bin/bash tools/macos/tests/target-directory.sh`.
Setup prerequisites and its build are stubbed; no game or SDK files are needed.

## Status

| Piece | State |
|-------|-------|
| `victory-platform`: memory (reserve, commit, protect, query, page pools), threads, events, semaphores, wait-on-address | Done, with a Win32 and a macOS backend and the same conformance tests on both |
| Loader and thunks on the Wine path call `victory-platform` for memory | Done (`windows.rs`, `thunks.rs`, `tls.rs`) |
| `victory-loader::native`: map, relocate, protect, bind, TLS, TEB, calling-convention bridge, run | Done, x86-64 Rosetta host |
| `victory-hle::host_posix`: first kernel32 calls on POSIX | 24 functions; `VirtualAlloc` is 64 KiB granular |
| Probe PE end to end under Rosetta | Done |
| Exceptions: signal handler, vectored handlers, SEH over `.pdata`, `RaiseException` | Done (`native::seh`), probe-tested |
| Threads: `DLL_THREAD_ATTACH/DETACH`, `CREATE_SUSPENDED`, `SuspendThread`/`ResumeThread`/`GetThreadContext` (Mach), `WaitForMultipleObjects`, critical sections, SRW locks, condition variables | Done (`host_posix`, `host_posix/sync.rs`) |
| `xmem` physical pages (`XMemAllocatePhysicalPages`, map, scatter, `map_alias`, DirectStorage writes) on `PagePool`, `modules::xmem_physical` | Done on both hosts: one pool per allocation, one Mach memory entry on macOS; Wine keeps one view per run of consecutive pages. |
| The rest of `xmem` (`XMemVirtualAlloc`, `XMemVirtualQuery`, `XMemAlloc` and the hooks, working-set statistics, transfers), `modules::xmem_posix` | Done on macOS, on `mem::reserve_in` / `next_region` / `footprint`; served from the registry. Hooks are found by reading the mapped title's export table |
| DirectStorage (`modules::dstorage`) | Done on macOS: the scheduler and completion rules are unchanged; file reads are `pread` on its worker threads, destinations go through `write_physical` and `page_window`. Paths go through `host_posix::files::translate`, so virtual drives (`G:`, `T:`) work |
| Files: `CreateFile`, `ReadFile`, `WriteFile`, `SetFilePointer(Ex)`, `GetFileSize(Ex)`, `FindFirstFile`, attributes, virtual drives | Done (`host_posix/files.rs`) |
| Game-save recovery (`modules::xgamesave`) | Recovery selects the newest version with a readable manifest and a regular file for every listed blob. Older versions are removed only after that check. |
| Title-shipped DLLs (`native::dll`): msvcp140, vcruntime140, Bink, libHttpClient, ... | Done: mapped beside the exe in dependency order (a DLL is wanted unless our runtime serves its name, so the C++ runtime comes from the title's folder), relocated, own static TLS, bound through the same resolver (hle, then `host_posix`, then the other title DLLs' exports), `.pdata` added to the exception dispatcher, `DllMain` and TLS callbacks run in dependency order on the main title thread before the exe's own, `DLL_THREAD_ATTACH/DETACH` delivered (`DisableThreadLibraryCalls` honoured), `LoadLibrary`/`GetModuleHandle`/`GetProcAddress`/`GetModuleFileName` see them. Static imports and `GetProcAddress` follow the same export forwarders. Title targets keep their mapped addresses, host and HLE functions use title-callable bridges, and cycles or missing targets fail lookup. Late `LoadLibrary` maps, binds and starts a DLL on the calling thread. `--list-imports` binds them too and runs none of their code. A title DLL that stubbed imports come from but that is not in the folder is named on a `missing dll` line, with a warning to point victory at the whole game folder; `MicrosoftGame.config` is found whatever its case, and a warning says when it is absent (the title ID is then 00000000). Both explained tester report #104 |
| kernel32 and friends beyond the first slice (`host_posix/system.rs`) | Time and time zones, strings and locales, system information, mutexes, waitable timers, the thread pool, FLS, SLists, `InitOnce`, registry/ETW/COM/crypt32/bcrypt/user32 message stubs, `Get`/`SetEnvironmentVariable` over the process environment (names ignore case; the C runtime's `getenv` reads the same one, where Windows' CRT keeps a copy). The C runtime (`api-ms-win-crt-*`) is separate |
| `GetModuleHandle`/`LoadLibrary`/`GetProcAddress` on system DLLs (`kernel32.dll`, `api-ms-win-core-*`, `ucrtbase.dll`, ...) | Done (`host_posix/fake_dll.rs`): one stable handle per normalised name, a 64 KiB read-only region holding a minimal PE header (no sections, no exports) so header reads do not crash; `GetProcAddress` (name or ordinal) resolves through the binder's resolver and returns a cached bridge stub (`bridge::stub_for`), NULL with `ERROR_PROC_NOT_FOUND` otherwise; `GetModuleFileName` gives `C:\Windows\System32\<name>`; `FreeLibrary` succeeds |
| Universal C runtime (`api-ms-win-crt-*`, `ucrtbase.dll`) | Done (`host_posix/ucrt.rs`): about 700 exports, see below |
| Winsock (`host_posix/winsock.rs`) | Done: reference-counted startup/cleanup (final cleanup invalidates all handles and releases sockets once in-flight calls finish), sockets (a `SOCKET` is a handle of ours), connect/bind/listen/accept, send/recv/sendto/recvfrom (a truncated UDP receive copies the prefix, fills the requested sender address and reports `WSAEMSGSIZE`), `select`, `WSAPoll`, `ioctlsocket` (`FIONBIO`, `FIONREAD`), socket options with Windows constants mapped (`SO_LINGER`, timeouts, `TCP_KEEPALIVE`, ...), `getaddrinfo`/`getnameinfo`/`gethostbyname` with Windows layouts, `inet_*`, byte order, `WSAGetLastError` is `GetLastError` with `WSAE*` codes from `errno`. Imports by name and by ordinal (`#n`, Wine's `ws2_32.spec`). Same policy as `modules::net`: loopback only, `localhost` and IP literals only, wildcard binds narrowed. No overlapped I/O |
| WinHTTP (`host_posix/winhttp.rs`) | Done, offline: handles, options, timeouts, headers, status callbacks (`HANDLE_CLOSING` is delivered on close), `WinHttpCrackUrl`; `WinHttpSendRequest` fails with `ERROR_WINHTTP_CANNOT_CONNECT` and the calls that need a sent request fail with `ERROR_WINHTTP_INCORRECT_HANDLE_STATE`. Nothing leaves the machine |
| Media Foundation (`host_posix/mf.rs`) | `MFStartup`/`MFShutdown`/`MFGetSystemTime` work; `MFCreateAttributes`, `MFCreateMediaType`, `MFCreateSourceResolver` and the source readers fail with `MF_E_UNSUPPORTED_BYTESTREAM_TYPE` and a null object, so a title skips its video paths. No COM objects; AVFoundation is the way to a real reader |
| Audio decoder (`xapu.dll`, `modules::apu`) | Done on macOS: the same module as on Windows, its result event is a `host_posix` event handle |
| Spatial audio (`host_posix/wasapi/spatial.rs`, over `wasapi.rs`) | `ISpatialAudioClient` on the default render device: the title's objects are mixed to stereo in 10 ms passes into the queue the CoreAudio `AudioQueue` plays. The title's event is set by a clock of the stream's own (`Shared::tick`): a period apart, half a period apart while fewer than three periods are queued, and not at all while five are. The device does not set it: CoreAudio takes frames in cycles of its own (512 frames at 48 kHz on the Mac measured), so it asks for 10 ms buffers 10.7 ms apart with two back to back every 160 ms, and one event for those two left a period of silence. Panning is by azimuth only. |
| XAudio2 (`host_posix/xaudio2.rs`, `mixer.rs`, `coreaudio.rs`) | `XAudio2Create` (ordinal 1) returns a bridged `extern "win64"` `IXAudio2`. The bridge follows the Windows SDK’s one-byte packing. On x64, a 44-byte source buffer carries its callback context at byte 36, and a 20-byte voice state returns the sample count at byte 12. PCM and float source voices support buffer queues, loops, volume, channel volumes, output matrices and frequency ratios. All `CoreAudio` streams of the same `(rate, channels)` (XAudio2, WASAPI and Spatial Audio all open 48 kHz stereo float) share one hardware `AudioQueue` and mix in userspace (`#336`), so mid-run spatial or engine stream re-activations under world-load pressure attach without calling `AudioQueueNewOutput` / `AudioQueueStart` again or stranding the new stream on the silent clock; when opening the hardware queue itself, `AudioQueueStart` retries three times 250 ms apart, disposing and recreating the queue handle on each retry (`#140`, `#288`, `#336`), and reopens if `AudioQueueEnqueueBuffer` fails at runtime. Each pass calls registered engine start callbacks in order, source voice start callbacks with the byte shortfall, then voice and engine end callbacks after mixing. `StopEngine` pauses passes and `StartEngine` resumes them. Buffer, loop and stream notices still use the callback worker. A runtime queue enqueue failure raises `OnCriticalError`. `OnVoiceError` awaits a decoder that can fail during mixing. Effects, filters, full submix mixing and ADPCM/XMA/xWMA sources remain future work. |
| Graphics objects and window on macOS | Done (`victory-gpu` native host on Metal, see `docs/metal-backend.md`): every object `D3D12XboxCreateDevice` and its methods hand out passes `object::answer`, which gives it a bridged method table (`object::set_bridge`, set by `apps/victory/src/native_run.rs` from `victory_loader::native`; the table's length sits just before its first slot, `ClearDepthStencilView` is `extern "win64"` for its float). Slots nobody wrote log once and answer `E_NOTIMPL`. The runtime's physical-memory observer forwards to `native::pool_observer()`, fence events to `posix::set_event`. `victory run --execute` opens an `NSWindow` (1920x1080, "VIctory") over the host's `CAMetalLayer` on the main thread (`apps/victory/src/window.rs`, AppKit through `objc2::msg_send!`, a hand-driven event loop) while the title runs on its own threads; closing it exits the process. `--headless` presents into an offscreen layer instead |
| Remaining Win32 call sites in `victory-hle` and `victory-gpu` moved to the seam | Not started |

Run the tests with:

```sh
rustup target add x86_64-apple-darwin                 # once
cargo test -p victory-platform                        # arm64 host: memory, threads, sync
cargo test -p victory-platform -p victory-loader -p victory-hle --target x86_64-apple-darwin   # Rosetta host: adds the probe PE
WINE=... tools/macos/test-wine.sh -p victory-platform # the same conformance tests on the Win32 backend
```

`crates/victory-hle/tests/native_probe.rs` builds `tests/fixtures/native-probe/probe.c` with `tools/macos/build-native-probe.sh` (the gnullvm clang and lld, no CRT), maps it, binds it, runs it, and checks 16 things the probe reports in its exit code. The probe also imports a second fixture, `probe_dll.c` (built beside it as `victory-probe-dll.dll`): the test maps it with `native::dll`, and `probe_dll` checks its `DllMain` and TLS callback order, per-thread static TLS, thread notices, an exception raised in the DLL unwinding through its `__finally` to a handler in the exe (and from the exe to a handler in the DLL), and `GetModuleHandle`/`GetProcAddress` by name. The C++ runtime is not ours to build, so the throw is a Windows exception through `__C_specific_handler`, not `_CxxThrowException` through `__CxxFrameHandler`.

`victory-gpu`'s `memory` module and its tests are `#[cfg(windows)]` (`src/lib.rs`), so on macOS `cargo test -p victory-gpu memory::` runs 0 tests and passes; only CI's Windows job (`cargo test --workspace` on WARP) checks them. The two that compile pipelines and submit through the registered copies (`lists_share_a_table_and_copies_keep_their_views`, `a_submission_that_missed_leaves_the_copies_at_once`) hold one test lock, since together they once failed in `CreateRootSignature` and they share the copies' registry and frame counter.

The native probe's waits check must run as `x86_64-apple-darwin` (under Rosetta
on Apple Silicon); the arm64 binary only runs its two image-mapping tests.
Its critical-section try check holds the lock on the calling thread until the
competing thread finishes, rather than assuming it runs within an 80 ms sleep,
then requires a successful try after release.
Under contention, an SRW acquisition CAS can fail and then observe the previous
owner's release. The POSIX host retries acquisition if that observed state is
available to the caller; sleeping on an unlocked word can strand a waiter with
no owner left to wake it. `srw_does_not_sleep_on_an_available_lock_after_a_failed_try`
exercises those states directly, with a bounded failure guard and worker cleanup.
This preserves Windows' [shared-reader/exclusive-writer ownership semantics](https://learn.microsoft.com/en-us/windows/win32/sync/slim-reader-writer--srw--locks);
it does not add a fairness guarantee.

Thread-pool timer workers wake when a deadline is replaced, disarmed, or closed.
`WaitForThreadpoolTimerCallbacks` waits for queued and running callbacks. Its
cancel flag drops queued callbacks and leaves future deadlines armed.

## Function pointers we hand to title code, and title pointers we call

Title code runs with GS = its TEB, ours with GS = the host's TSD, and the two ABIs differ, so every pointer crossing needs a bridge.

* **Objects and method tables.** When an HLE object (first word = method table) is first handed to the title, `victory-hle::title::expose` calls `native::expose_object`, which replaces the table word, in place, with a bridged copy: each slot is a bridge stub for the host function in it (`extern "win64"` ones keep their ABI), null slots stay null, and 64 extra slots answer `E_NOTIMPL` and log the area and slot. A title that calls a slot past the end of one of our tables would otherwise read whatever follows the static (often zero: a jump to address 0). Copies are cached per table address. `this` is unchanged: the title gets the object's own address, so methods find their state as before. The price is that Rust must not call an exposed object through its table word (nothing does outside tests, and `expose` does nothing until an image is mapped, so unit tests are unaffected). Applied to `xgameruntime::Object` (`QueryApiImpl`, DirectStorage objects), GameInput hub, devices and readings, XG texture computers and the ACP. `victory-gpu`'s D3D12 objects exist only on the Windows host path today; the Metal backend must expose its objects the same way.
* **Title callbacks.** `victory-hle::title::call(f, args)` calls a callback pointer: through `call_win64` (GS = TEB) when `f` is in a mapped title image, directly when it is one of ours (tests and internal completions pass Rust functions). Used for XTaskQueue and XAsync callbacks, providers and completion routines, monitors, GameInput device callbacks, game-save enumeration callbacks, `XMemAlloc` hooks and the unhandled-exception filter. Thread-pool, `InitOnce`, atexit, qsort, thread procs, TLS callbacks, SEH handlers already used `call_win64`.

## Which Win32 APIs the loader and HLE depend on

Counts are call sites (a call site can be a test). The macOS column is what the native host uses or will use.

### Memory

| Win32 | Where | macOS |
|-------|-------|-------|
| `VirtualAlloc`, `VirtualFree`, `VirtualProtect`, `VirtualQuery(Ex)` | `victory-loader` (`crash.rs` 18, `masked.rs` 12, `tls.rs`), `victory-hle` `xmem.rs` (31), `dstorage.rs`; `victory-gpu` (`watch.rs` 59, `texture.rs` 14, `host.rs` 13, `shadow.rs` 7, `memory.rs`, `list.rs`, `copystat.rs`, `draw.rs`) | `mach_vm_allocate` (reserve), `mach_vm_protect` (commit and protect), `mach_vm_deallocate`, `mach_vm_region` (query) |
| `VirtualAlloc2` with `MEM_RESERVE_PLACEHOLDER`, `MapViewOfFile3`, `UnmapViewOfFile2`, `CreateFileMappingW` (`SEC_RESERVE`) | `xmem.rs`: `XMemAllocatePhysicalPages`, `XMemMapPhysicalPages` and scatter, `XMemVirtualAlloc` | One Mach memory entry per physical-page pool (`mach_make_memory_entry_64`), mapped with `mach_vm_map` at fixed addresses (`VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE`) |
| `MEM_WRITE_WATCH`, `GetWriteWatch`, guard-page tracking with a vectored handler | `victory-gpu/src/watch.rs` | Not needed once the GPU reads title memory directly (zero-copy). Dirty tracking, if wanted, is `mach_vm_protect` plus a Mach exception handler |
| `FlushInstructionCache` | loader patches | no-op on x86-64; `sys_icache_invalidate` on arm64 |
| `LoadLibraryExW(path)` of the prepared image | `windows.rs` | not used: `native::image` maps the PE itself |

Why the Wine path is slow here: one 64 KiB view per page (`MapViewOfFile3`) makes every `VirtualProtect` cost a wineserver round trip. The platform page pool avoids it, since one pool is one VM object and a protect over a range of mappings is one call (tested in `one_protect_call_covers_many_mappings_of_a_pool`).

### Threads and synchronisation

| Win32 | Where | macOS |
|-------|-------|-------|
| `CreateThread`, `GetCurrentThreadId`, `SuspendThread`, `ResumeThread`, `GetThreadContext`, `SetThreadPriority`, `NtQueryInformationThread`, `Sleep`, `SwitchToThread` | loader (`threads.rs`, `stacks.rs`, `crash.rs`), HLE (`kernel.rs`, `apu.rs`, `audio`, `lifecycle.rs`), GPU (`queue.rs`) | pthreads (through `std::thread` for stack size and name), `pthread_threadid_np`, `thread_suspend` and `thread_get_state` on the Mach thread port for suspend and context, `pthread_setschedparam` |
| `CreateEventW`, `SetEvent`, `ResetEvent`, `CreateSemaphoreW`, `WaitForSingleObject`, `WaitForMultipleObjects`, `CloseHandle`, `DuplicateHandle` | HLE (`lifecycle.rs`, `dstorage.rs`), GPU (`fence.rs`, `host.rs`) | `victory_platform::sync::{Event, Semaphore}` behind a handle table (`host_posix.rs`). `WaitForMultipleObjects` across mixed kinds needs one waiter structure: events and semaphores register on it, files and sockets use `kqueue`, timers use `EVFILT_TIMER` |
| `WaitOnAddress`, `WakeByAddress*` | std and title code | `os_sync_wait_on_address` (macOS 14.4 and later) |
| Critical sections and SRW locks | inline in the title's CRT, over `ntdll` | `RtlEnterCriticalSection` and `RtlAcquireSRWLock*` on `os_unfair_lock` or the wait-on-address primitive; not yet reached |
| DLL thread notices (`DLL_THREAD_ATTACH` through a TLS callback registered in `.CRT$XLY`) | `victory-loader/src/threads.rs` | The Windows loader calls each thread's callbacks for us today. Natively `spawn_title_thread` is the one place threads start in title code, so it must send the notices. Not yet done |

### Files

| Win32 | Where | macOS |
|-------|-------|-------|
| 27 path-taking calls (`CreateFileW/A`, `CreateFile2`, `DeleteFile*`, `FindFirstFile*`, `GetFileAttributes*`, `MoveFileExW`, `CreateDirectoryW`, `GetDiskFreeSpace*` and the C runtime's `fopen`, `_wopen`, `_stat64i32`, ...) | `victory-hle/src/modules/files.rs` rewrites drive paths, then calls the host | `open`, `stat`, `opendir`/`readdir` (or `getattrlistbulk`), `rename`, `statfs`. Paths go through the same drive rewrite; wide to UTF-8 conversion is the new part |
| `ReadFile`, `WriteFile`, `SetFilePointerEx`, `GetFileSizeEx`, `FlushFileBuffers`, overlapped I/O with events or completion ports | `files.rs`, `dstorage.rs`, `storage.rs` | `pread`, `pwrite`, `fcntl(F_NOCACHE)`; overlapped I/O on a worker pool, or `kqueue`/`dispatch_io` for DirectStorage's queues |
| `CreateFileMappingW` on files, `NtQueryVolumeInformationFile` | `dstorage.rs`, `files.rs` | `mmap`, `fstatfs` |
| `CreateHardLink` | blob cache (`victory-gpu`) | `link` (APFS has no 1024-link limit; see `docs/macos.md`) |

The CRT's `rename` and `_wrename` use Darwin's `renamex_np` with `RENAME_EXCL`. If the filesystem returns `ENOTSUP`, the shim checks the destination without following symlinks, then renames. This fallback can overwrite a destination created between the check and the rename. Both paths report `EEXIST` when they detect an existing destination, including a dangling symlink. [MSVC documents `EACCES`.](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/rename-wrename?view=msvc-170)

### Exceptions: SEH and vectored handlers

| Win32 | Where | macOS |
|-------|-------|-------|
| `AddVectoredExceptionHandler` | `crash.rs` (reporter), `watch.rs` (write-guard faults: `EXCEPTION_ACCESS_VIOLATION`, `EXCEPTION_CONTINUE_EXECUTION`), `masked.rs` (resumes patched stores) | `sigaction(SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP)` with `SA_SIGINFO | SA_ONSTACK`, translating `ucontext` into a Windows `CONTEXT` and `EXCEPTION_RECORD`. Where signals are too coarse (a fault inside `os_sync`, debugger interplay), a Mach exception port with `EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES` and `thread_get_state` |
| `SetUnhandledExceptionFilter`, `RaiseException` | `kernel.rs`, `crash.rs`, title CRT | Last in the handler chain; `RaiseException` becomes a `longjmp`-free walk of the same chain on the calling thread |
| `RtlLookupFunctionEntry`, `RtlVirtualUnwind`, `RtlCaptureContext`, `RtlAddFunctionTable`, `RtlCaptureStackBackTrace` | `crash.rs` (stack walks), title SEH and C++ exceptions | Pure computation over the image's `.pdata` and `UNWIND_INFO`; the walker in `crash.rs` already does it from bytes and moves over. Windows' dispatch (`RtlDispatchException`, `RtlUnwindEx`, `__C_specific_handler`) must be ours, since the OS no longer does it |

Known Rosetta constraint, from `docs/macos.md`: a fault inside a masked or `vcvtps2ph` store cannot be resumed, so `masked.rs` patches those stores. The same patching applies natively.

### TLS

| Win32 | Where | macOS |
|-------|-------|-------|
| Static TLS (`gs:[0x58]` array, `_tls_index`, TLS directory, TLS callbacks) | `victory-loader/src/tls.rs` redirects exe loads; `victory-pe` reads the directory | `native::teb` builds the array and per-thread blocks from each image's TLS directory and assigns each image its own index, so the redirect is not needed |
| `TlsAlloc`, `TlsGetValue`, `TlsSetValue`, `TlsFree`, `FlsAlloc`... | title and C runtime | `host_posix.rs`: 64 slots at `TEB+0x1480`; `TlsFree` zeroes the slot in every live TEB (`teb::clear_tls_slot`), so a slot handed out again reads 0. `Fls*` (`host_posix/system.rs`) is per host thread, with a generation per slot so a reused slot reads 0 too; its destructor callbacks and the expansion slots are not done |
| `GetLastError`, `SetLastError` | everywhere | `TEB+0x68` |

### Time

| Win32 | Where | macOS |
|-------|-------|-------|
| `QueryPerformanceCounter` and `Frequency`, `GetTickCount(64)` | HLE `runtime.rs`, `xgameruntime.rs`, `kernel.rs`, GPU `queue.rs` | `mach_absolute_time` with `mach_timebase_info` (or `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`); `host_posix.rs` uses `Instant` with a 1 GHz counter |
| `GetSystemTimePreciseAsFileTime`, `GetLocalTime` | HLE | `clock_gettime(CLOCK_REALTIME)` and the 1601 epoch offset |
| `Sleep`, waitable timers | HLE | `nanosleep`; `EVFILT_TIMER` for waitable timers |

### Not in these groups, still needed

Window and input (`CreateWindowExW`, message loop, XInput; `lifecycle.rs`, `gameinput.rs`) go to AppKit and `GameController`. Media Foundation video (`media.rs`) would go to AVFoundation. (Sockets, audio and the audio decoder are done: see the table above.) Crypto (`bcrypt`, `crypt32`) goes to CommonCrypto and Security. These are each separate pieces of work, after the core is up.

## The platform seam

`crates/victory-platform` has one API and two backends, chosen by `cfg`. It is plain functions and types, not a trait: nothing needs to be generic over the OS at run time, and the title-facing code calls the same names either way.

```rust
// mem: all unsafe, all return io::Result
page_size() / granularity()               // macOS: the page size; Windows: 64 KiB
reserve(at: Option<usize>, len)           // no access; `at` exactly or fail
alloc(at, len, Protect)                   // reserve and commit in one step
commit(addr, len, Protect)                // inside a reservation
protect(addr, len, Protect) -> Protect    // returns the previous protection
decommit(addr, len)   release(addr, len)  // Windows: the whole reservation
query(addr) -> Option<Region>             // VirtualQuery / mach_vm_region
PagePool::new(len)                        // one OS memory object
pool.map(offset, len, at, Protect)        // map any part of it, anywhere or at `at`
unmap(addr, len)   flush_icache(addr, len)
// thread
spawn(name, stack_size, FnOnce() -> u32) -> Thread    Thread::{id, join}    current_id()    sleep()
// sync
Event::{new(manual, initial), set, reset, wait(timeout)}    Semaphore::{new, release, wait}
wait_on_address(&AtomicU32, expected, timeout)    wake_one    wake_all
```

`Protect` wraps the raw OS value (`PAGE_*` or `VM_PROT_*`) so that a value `VirtualProtect` returned can be passed back as it came; use the constants (`READ`, `READ_WRITE`, `READ_EXECUTE`, ...).

Two documented differences between backends. On Windows `PagePool::map` at an address needs that range free (it makes a placeholder to map over), where on macOS it replaces what is there, as `MAP_FIXED` does. And Windows addresses are 64 KiB granular where macOS ones are page granular (4 KiB under Rosetta, 16 KiB natively); `host_posix::VirtualAlloc` makes up for it by reserving a granule more and giving the ends back.

What moved so far: `windows.rs` (IAT slot writes), `thunks.rs` (import stub block) and `tls.rs` (thunk page, code patching) use `mem`, and so does the physical-page half of `xmem`. The Wine tests of `victory-loader` and `victory-hle` pass unchanged. The rest follows the order of the tables above: `crash.rs` and `masked.rs` (protect and query), then threads and events in HLE and GPU, then `watch.rs`, which the native host drops.

`PagePool` also has: `base()` (the pool's permanent whole mapping; lazy on Windows, where it can fail for a huge pool), `window(offset, len)` (bytes through `base`, or a view of their own when `base` failed), `map_over(offset, len, at, prot)` (map over part of the caller's reservation, a placeholder on Windows), `mem::unmap_over` (back to no access, or a placeholder), `new_lazy` (Windows: reserve only, commit with `mem::commit`) and, on macOS, `memory_entry()`.

Under Wine `xmem` still maps one view per run of consecutive pages the title passes in one call, so a title that maps page by page (scatter, `CopyPageMappings`) still pays one view per 64 KiB. Merging those needs a mapping model that is not page granular, which the write-guard code (`watch.rs`) assumes; left as it was.

## The native loader

### Mapping (`native::image`, any macOS host)

`NativeImage::map` parses the PE (`victory-pe`), reserves `SizeOfImage` at the preferred base (`0x1_4000_0000` for the probe, as for a console title) or anywhere if that is taken, copies the headers and sections, and applies base relocations (`DIR64`, `HIGHLOW`). `finalize` sets page protections from section flags once imports are bound, taking the union where a host page holds two sections. The preferred base only works if it is free; the host binary sits at `0x1_0000_0000` and must stay below about 1 GiB. Nothing here runs title code or depends on the CPU, so the recompiler's arm64 host maps images the same way.

For ordinary and delay-loaded imports, the PE parser follows the RVA of each DLL name, lookup table and hint/name entry independently. A linker can place them in different sections from the import descriptor.

### TEB and GS under Rosetta (`native::teb`)

Title code reads the TEB through `gs:`: `gs:[0x30]` (self), `gs:[0x58]` (static TLS array), `gs:[0x60]` (PEB), `gs:[0x10]` (stack limit), `gs:[0x68]` (last error), `gs:[0x1480+]` (TLS slots). What was found out, by experiment (`M5 Pro`, macOS 27, x86-64 process under Rosetta):

1. Rosetta gives an x86-64 process the macOS ABI: `gs:` is the pthread's thread-specific data (TSD) array, with `pthread_self()` in slot 0. Slots 6, 11 and 12 (the offsets `0x30`, `0x58`, `0x60`) read as zero in a bare process, but 10 and up are libSystem's own pthread keys, 8 bytes each. A dynamic key starts at 258. Writing title data into those slots would collide with libc as soon as it uses them. Slot 1 (`0x08`, stack base) is `errno` and slot 2 (`0x10`, stack limit) is the Mach reply port, which are in use.
2. Rosetta implements the machdep call `thread_fast_set_cthread_self` (syscall `0x3000003`, `rdi` = the new GS base) for x86-64 guests, from which Wine's macOS port gets its own TEB. It works from inline assembly, with no entitlement. A pair of calls costs about 370 ns (190 ns each), measured in a loop.
3. The base to restore is not `pthread_self()`: it is the address of the TSD array inside the pthread structure, `pthread_self() + 0xE0` here on the main and secondary threads alike. Restoring `pthread_self()` itself as the base breaks `pthread_setspecific` and the process dies later in unrelated code. The offset is not an API: the structure also holds a stray word equal to `pthread_self()` at `+0xB0`, which a first version of the search took, and which crashed the process at thread exit. `host_tsd_base` therefore takes the address whose first 12 words equal what `gs:[0..96]` reads.
4. `rdgsbase` was not tried; the base is found by the search above.

Decision: each thread has two GS bases, the host's (pthread TSD) while host code runs and its own TEB while title code runs, swapped at each crossing by the bridge. The alternatives and why they lose:

| Option | Cost | Why not |
|--------|------|---------|
| Write TEB fields into free TSD slots, no swap | none | `0x08`, `0x10` and `0x58`/`0x60` collide with `errno`, the reply port and libSystem keys; `0x1480` (TLS slots) lands among dynamic pthread keys |
| Patch every `gs:` access in the title (as `tls.rs` and `masked.rs` patch) | patch time; the game has about 53,000 `gs:[0x58]` loads alone | Fragile, and the recompiler will make it moot: it lowers `gs:` to a register-relative access |
| Swap per crossing (chosen) | 370 ns per host call and per callback | Simple and exact; every `gs:` offset works. The cost is the first thing the recompiler removes |

At 370 ns a crossing, 100,000 host calls per frame cost about 37 ms. The Wine path pays far more per call (see the profile in the task), and a title calls the host mostly for `Sleep`, events, memory and the console API, not per draw. Cheaper paths for hot calls (a TEB-only function such as `GetLastError` or `TlsGetValue` can be inlined into the stub with no swap) are a next step.

`ThreadTeb` (in `teb.rs`) is a `0x2000`-byte block with the fields above, `StackBase` and `StackLimit` from `pthread_get_stackaddr_np`, `ClientId`, a pointer to the thread's TLS array, a shared fake PEB (`ImageBaseAddress` at `0x10`), and, at `0x1F00` (unused by Windows), the thread's host GS base for the bridge to swap back to. Every host thread that calls into title code gets one on first use (`teb::current`).

### Static TLS

`register_tls` gives each image with a TLS directory the next index, writes it to the image's `_tls_index` (the TLS directory's `AddressOfIndex`), and records the template. Each thread's TEB gets an array `[block_for_image_0, block_for_image_1, ...]`, each block a copy of the template plus zero fill, so `mov rax, gs:[0x58]; mov rax, [rax + _tls_index*8]; add rax, SECREL(var)` works for an exe as for a DLL (the Wine path needs `tls.rs` to redirect exe loads because Windows gives an exe index 0 by rule; here there is no such rule). Images must be registered before threads that use them start; a thread made earlier has no block for a later image. The probe checks that a thread gets a fresh copy (`100`, not the main thread's `101`), through `CreateThread`.

### The calling-convention bridge (`native::bridge`)

Title to host (`victory_bridge_to_host`, reached through a 32-byte stub per import: `mov r10, target; mov r11, context; mov rax, bridge; jmp rax`):

- the stub keeps the Windows return address and shadow space; the bridge saves `rbx`, `rdi`, `rsi`, `r12`-`r14` and `xmm6`-`xmm15` (the Windows ABI makes these the callee's to keep, the System V ABI does not), spills `rcx`, `rdx`, `r8`, `r9`;
- it switches GS to the host base held at `TEB+0x1F00`, loads `rdi, rsi, rdx, rcx, r8, r9` from the four spilled registers and the first two stack arguments, copies stack arguments 7 to 12, and calls the target with `al = 0`;
- it switches GS back to the TEB, restores the saved registers and returns. `rax` and `xmm0` pass through unchanged, since both ABIs return there.

The bridge covers up to 12 integer or pointer arguments. A function with `float` or `double` arguments cannot be remapped without its signature, because the two ABIs number the float registers differently once an integer comes first. Those are written `extern "win64"` and registered with `declare_win64(address)`; they go through `victory_bridge_win64`, which swaps GS and nothing else (up to 8 arguments; floats stay in `xmm0..3`). The probe does this with `ProbeScale(double, int)`. A float return needs nothing.

An unresolved import gets a stub whose context word is its `ImportSite`; the bridge hands the site to the handler in place of the first argument, which logs (`ReturnZero`) or aborts (`Trap`), the same policies as the Wine path. The probe checks the names reach the report and that `ReturnZero` returns 0. Once per bound image (late-loaded DLLs included), `native imports bound` names them at warn level as `dll!symbol` (`dll!#n` for an ordinal) in its `unresolved` field, so a log shows what a DLL lacks before any of it is called; with nothing missing the line stays at info.

Host to title (`call_win64(fn, &[u64])`): four register arguments, the rest on the stack above 32 bytes of shadow space, 16-byte alignment, GS to the thread's TEB, call, GS back. Used for TLS callbacks, entry points, thread start routines and every callback title code hands the host. It nests: title, host, title again, with an explicit base each time rather than a toggle. A test (`the_bridge_preserves_windows_callee_saved_registers`) runs hand-assembled Windows-ABI code that sets `rdi`, `rsi` and `xmm6`, calls a host function that clobbers all three, and checks they survive.

### Imports and the HLE

`native::bind_imports` takes an `ImportResolver` (the same trait the Wine path uses, so `victory_hle::HleResolver` works unchanged), writes a stub address into each IAT slot (title exports are bound directly), and reports counts. `HostPolicy` on macOS resolves allowed DLLs (`kernel32`, `kernelbase`, `api-ms-win-core-*`) to `victory-hle::host_posix`, which is the first slice of "the host's own Win32, reimplemented on POSIX": `GetLastError`, `SetLastError`, ids, time (`QueryPerformanceCounter`, `GetTickCount(64)`, `Sleep`), `VirtualAlloc/Free/Protect`, `TlsAlloc/Free/GetValue/SetValue`, events, semaphores, `WaitForSingleObject`, `CloseHandle` and `CreateThread` (title threads), over a handle table. Everything else lands in the unimplemented-import stub, named. This module is x86-64 only for now because it reads the TEB; once the recompiler defines its thread context it moves behind a `current_teb()` that both hosts provide.

The HLE functions are `extern "system"`, which is System V on macOS (x86-64 and arm64 alike), so no change to the 100+ existing HLE modules is needed to bridge them. The ones that compile on macOS today (the portable modules: `kernel`, `xgameruntime`, `gameinput`, `d3d12x`, `audio`, `dsp`, `xg`, `rpcrt4`) are in the registry; `files`, `xmem`, `lifecycle`, `net`, `media`, `dstorage`, `apu`, `xsystem` and `fidelityfx` are `cfg(windows)` and are the porting backlog.

## Interfaces for the Metal backend

Goal: `victory-gpu`'s Metal backend wraps title memory pages as `MTLBuffer` objects with no copy and no shadow.

1. **One VM object per physical-page pool.** `XMemAllocatePhysicalPages` should create one `PagePool` per pool (instead of one section with a 64 KiB view per page). A `PagePool` is one Mach memory entry from `mach_make_memory_entry_64(MAP_MEM_NAMED_CREATE)`: anonymous, zero filled, committed on first touch. `XMemMapPhysicalPages` and `XMemVirtualAlloc` map ranges of it into the title's address space with `PagePool::map(offset, len, Some(address), prot)`. Any number of mappings, at any addresses, of any part of it are the same physical pages.
2. **Stable addresses.** A mapping stays at its address until the title unmaps or remaps it. The title's virtual addresses are the CPU's virtual addresses (identity), so a GPU address the title holds is already a pointer. Remapping a range (`XMemMapPhysicalPages` over mapped pages) replaces the pages at that address, which is the moment a backend must drop and rebuild any `MTLBuffer` over it (`title_memory_remapped` in `victory-gpu`).
3. **Alignment.** Map at and in multiples of 16 KiB (`vm_kernel_page_size`, also under Rosetta, where `getpagesize()` says 4 KiB). Note that Rosetta reports a 4 KiB page size, so two reported 4 KiB pages cannot hold the two 16 KiB buffers used by the descriptor-table test; that test sizes its arena as `(2 * 16 KiB).div_ceil(page_size())` pages, so it always holds 32 KiB. Checked on an M5 Pro with macOS 27 (a short Objective-C program, not in the repository, built for arm64 and for x86-64 under Rosetta): `newBufferWithBytesNoCopy` over a pool mapping returns a buffer whose `contents` is the same pointer, for the whole mapping, for a 4 KiB-aligned and a 16 KiB-aligned sub-range, and for a mapping of a sub-range of the pool at another address. A write through a buffer over mapping B is seen through mapping A at once (one VM object). Only the 16 KiB multiple is guaranteed to stay valid on other OS versions.
4. **What the backend may assume.** Memory is `MTLStorageModeShared`. CPU and GPU see one copy, so the shadow copies and the copy-back disappear, and so does `victory-gpu/src/watch.rs` (write-guard tracking and its page-tracking lock). Ordering is by command buffers and fences only: a CPU write the title makes before it submits is visible to that submission, and the title's own fences order the rest. Create buffers with `hazardTrackingModeUntracked`; the title already tracks hazards.
5. **Sizes.** `device.maxBufferLength` bounds one `MTLBuffer`; wrap each pool mapping (or each large allocation's mapping) separately and keep an address-to-buffer interval map keyed by the mapping's start. Pool mappings that are contiguous in address are still separate VM entries, so a buffer cannot span two mappings; the backend should ask for allocations large enough that they are one mapping (`XMemVirtualAlloc` of a range is). The native host does not wrap title mappings: it wraps each pool's own mapping once (`native/memory.rs`), so a title range may run over several title mappings when each follows the one before it both in title memory and in the same pool, which is how the 64 KiB pages of one allocation are reported. `PoolTable::resolve` follows such mappings and gives one pointer into the pool; a range over a gap, an empty mapping, two pools, or mappings that are not next to each other in the pool is refused.
6. **Protection.** The CPU side can set a mapping `READ` while a buffer over it is live; `PagePool::map(.., Protect::READ)` makes a read-only alias (for CPU-side guards or for the GPU's read-only use) of pages another mapping writes.

Provided (`victory_hle::modules::xmem_physical`, branch `native/host-mem`):

- Every `XMemAllocatePhysicalPages` makes one pool (`PagePool`). `pools()` lists them as `PoolInfo { id, base, len, memory_entry }`: `base` is the pool's own read-write mapping of all of it, 16 KiB aligned (also under Rosetta), made with the pool and never moved, protected or unmapped while it lives; `memory_entry` is the Mach entry (a send right the pool owns).
- `set_observer(Some(Arc<dyn Observer>))` delivers `pool_created`, `mapped(at, len, &PoolInfo, pool_offset)`, `unmapped(at, len)` and `pool_released`, synchronously and under the page-table lock (no calls back into `xmem`). A backend that starts late wraps `pools()` and then hears the rest. On Windows the same events fire, next to `victory_gpu::title_memory_remapped`.
- Title mappings are `mach_vm_map` of a run of the pool over the title's reservation (`FIXED | OVERWRITE`), one call per run. Unmapping a page leaves no access, and cuts a larger mapping in two without touching the rest (no event for what stays). A title address is an alias only: the buffer over `base` serves every address showing that pool, from the table the events build.
- Two points differ from item 4 above. A pool is not process-lifetime: when the title has freed every page and unmapped every view, `pool_released` fires and `base` is unmapped right after it returns, so the backend must drop its buffer there (a pool kept for good would keep the memory of a freed 6 GiB allocation). And `base` is not itself a title address: the GPU address of a page is its title address, translated through the table.
- Only 64 KiB aligned title addresses are accepted by `XMemMapPhysicalPages`, as on Windows.
- `XMemFreePhysicalPages` gives page numbers back, and `XMemAllocatePhysicalPages` takes the lowest freed numbers first, then new ones (`Physical::number`). A number whose page a view still shows waits until the last such view is unmapped (`pinned`), so a title mapping a number never reaches two pages. Numbers used to only grow, and the title keeps them in 25 bits (`number >> 4` in 21): past about 128 GiB allocated since start-up, live numbers were rebuilt as other pages' (issue #15). Now they stay under 25 bits while under 128 GiB of pages live at once, or are kept mapped.

## Interfaces for the recompiler

The recompiler produces arm64 code for the title's functions and runs it in an arm64 host process with the same loader, runtime and HLE. What it can rely on:

1. **Guest addresses are host addresses.** `NativeImage::map` is arch-neutral and maps the image at its preferred base (the arm64 host binary must also stay clear of `0x1_4000_0000` and up; use a linker `-image_base` or a custom segment layout if it does not). Title pointers can be passed to HLE functions as they are, and `PagePool` mappings give the same identity for GPU and physical-page memory. Recompiled code is keyed by guest RVA, so the `.pdata` function table, vtables, jump tables, and the IAT stay valid.
2. **Calling an HLE entry point.** Today the registry holds addresses (`registry()` returns `HleExport { name, address }`) of `extern "system"` Rust functions: System V on x86-64, AAPCS64 on arm64, both the host's C ABI. The recompiled caller reads the Windows x64 argument registers from its virtual register file (`rcx, rdx, r8, r9`, then the stack above the 32-byte shadow space), calls the function with them as native integer arguments (`x0..x7`, then the stack), and writes `rax` from `x0` (and `xmm0` from `d0` for a float return). No GS swap, no register saving beyond what recompiled code does for its own state. To do this for floats the recompiler needs each export's signature class; add `sig: &'static str` to `HleExport` (for example `"pp,i->i"`), filled by `hle_export!` from the Rust signature, with the `extern "win64"` exports marked. Until then only integer and pointer signatures can be called.
3. **Imports.** `bind_imports` and `ImportResolver` are reused as they are; the recompiler gives the resolver its own stub generator instead of `StubBlock` (a native call stub for an arm64 host, or a direct branch).
4. **TLS and the TEB.** Recompiled code lowers `gs:[off]` to `[teb_base + off]`, with `teb_base` in a reserved register or in the thread context struct passed to every recompiled function. The layout is the one in `native::teb` (`OFF_*`): `0x30` self, `0x58` TLS array, `0x60` PEB, `0x68` last error, `0x1480` TLS slots, `0x08` and `0x10` stack bounds. `teb::register_tls` and the per-thread block construction (`ThreadTeb::new`) move to be arch-neutral; they have nothing x86-specific except the `host_tsd_base` search, which an arm64 host does not need (there is no second GS base, so the bridge's swap disappears, and a host call is a plain call).
5. **Threads.** Title threads start through `spawn_title_thread`, which the recompiler replaces with a version that allocates the thread's context, sets the stack, and enters recompiled code at the start routine's address.
6. **Exceptions.** A fault in recompiled code arrives as `SIGSEGV` or `SIGBUS` with a host PC. The recompiler gives the host a side table from host PC ranges to guest RIP and a way to rebuild the guest register file (as `Mach` thread state or from the saved context), so the host can run the vectored-handler chain with a Windows `CONTEXT` and `EXCEPTION_RECORD`, and either resume (`EXCEPTION_CONTINUE_EXECUTION`: re-enter at the recompiled block for the faulting guest instruction) or unwind. C++ and SEH unwinding use `.pdata` and `UNWIND_INFO` over the guest context (the walker in `crash.rs` is portable and moves into `native`), and handler and `__finally` addresses are guest addresses of recompiled functions. `RaiseException` and the HLE's own failures enter the same dispatcher.
7. **Self-modifying and patched code.** `masked.rs` patches (stores Rosetta cannot resume) are unneeded under the recompiler, which lowers them itself.

## Exceptions, threads and files on macOS (`native::seh`, `host_posix`)

**Signals, not a Mach exception port.** A SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGTRAP handler (on the thread's alternate stack) tells title code (GS = TEB) from host code (GS = TSD) by `gs:[0x30]`, copies the thread state into a Windows `CONTEXT` and `EXCEPTION_RECORD` on the thread's own stack, and redirects the thread to `victory_exception_entry`. Nothing runs in signal context, there is no second thread or `thread_get_state`, and the faulting registers are exact. The entry switches GS to the host base, then `dispatch` runs the vectored handlers, then walks the frames (`unwind::lookup`/`virtual_unwind`, the x64 unwinder incl. epilogues and chained info) calling each language handler (`__C_specific_handler` is ours: scope tables, `__except` filters, `__finally` via `RtlUnwindEx`), then the unhandled filter. Title handlers run through `call_win64` (GS = TEB, host base restored after); continuing or unwinding ends in `victory_restore_context`, which sets the TEB last. `RtlCaptureContext` and `RaiseException` are Windows-ABI assembly bound with no bridge (`declare_direct`). Access violations carry the read/write flag and address (`info[0]`, `info[1]`); macOS's `si_code` numbers for SIGFPE differ from Linux's. A fault in host code is not dispatched (chained to Rust's handler, then default); that and debugger interplay are what a Mach exception port would add.

The probe must be built with `-fasynchronous-unwind-tables -fexceptions -Xclang -fasync-exceptions` (gnullvm clang supports `__try/__except/__finally`; without async EH clang drops the `__except` around plain loads and stores).

Limits: `RtlUnwindEx` unwinds from the exception being dispatched on the thread (a call from title code outside a dispatch returns); C++ handlers (`__CxxFrameHandler`) are not provided; unwinding stops at host frames (a host function that calls back into title code is not crossed); no stack-overflow handling; `SetThreadContext` is not done.

**`DebugBreak`** is `RaiseException(EXCEPTION_BREAKPOINT, 0, 0, NULL)` for its caller (`_victory_DebugBreak`, a tail call into `_victory_RaiseException`), so a vectored handler or `__except` takes it and, with none, it is unhandled as on Windows; it is not a no-op. Guess: Windows runs an `int3` inside `DbgBreakPoint`, so its record names that address and has one parameter (0); here the address is the caller's return address, the record has none, and a handler that continues resumes after the call.

**`FormatMessageW`** (`host_posix/system.rs`) does the part Microsoft Learn documents that a library uses: `FROM_SYSTEM` (an `HRESULT` of a Win32 error too), `FROM_STRING` with `%1`..`%99` inserts (`!s!`, `!ls!`, `!ws!`, `!d!`, `!i!`, `!u!`, `!x!`, `!X!`) and the escapes `%0`, `%n`, `%t`, `%%`, `%.`, `%!`, `% `, `IGNORE_INSERTS`, `ARGUMENT_ARRAY` or a `va_list`, `ALLOCATE_BUFFER` (leaked, as `FormatMessageA`'s: `LocalFree` keeps nothing to free) and `MAX_WIDTH_MASK`, which drops the system text's closing line break. Other sources (`FROM_HMODULE`), flags, `printf` forms or a null string argument fail with `ERROR_INVALID_PARAMETER`. System texts are Windows' English ones for a few common errors. Guesses: an id with no text gives `Error 0x%08X` where Windows fails with `ERROR_MR_MID_NOT_FOUND`, since a title logging it is better served by the number; a width other than 0 or `0xFF` does not wrap lines. `_ldsign` is `_dsign` (`long double` is `double` on this ABI): nonzero (`0x8000`) when the sign bit is set, as Microsoft Learn's floating-point primitives promise.

**Threads.** `spawn_title_thread_with` sends `DLL_THREAD_ATTACH` before and `DLL_THREAD_DETACH` after the start routine to every image registered with `register_thread_callbacks` (`run_entry` registers the main image). `CREATE_SUSPENDED` holds the thread at a gate (the thread reads its suspension count and marks itself running under one lock, so a `SuspendThread` while it starts either parks it at the gate or finds it running); once running, `SuspendThread` is `thread_suspend` on the Mach port, retried briefly until the thread is in title code (so it holds no host lock), and `GetThreadContext` is `thread_get_state` (x86 state under Rosetta works). Waits on several objects use one epoch counter that every signal bumps (it wakes sleepers only while a thread is in such a wait, so a signal with none makes no system call); wait-all takes under a lock and gives back what it took if it cannot take all. Critical sections, SRW locks and condition variables live in the title's own structures over `wait_on_address`. Not done: mutexes (`CreateMutexW`), waitable timers, APCs, `SetThreadContext`, thread priorities.

**Files.** `host_posix/files.rs` mirrors the calls `modules/files.rs` hands to the host's kernel32 on Windows, on `std::fs`: drives are mounted folders (`mount`, `mount_at`), a drive path is resolved under its folder (`..` stops at the root), other paths just get `/` for `\`. The Windows path is untouched. Persistent storage (`storage.rs`) mounts its folder through it on macOS. Synchronous reads, writes, seeks and `SetEndOfFile` serialize access to the shared file position on each handle. Overlapped reads and writes complete before returning and leave that position unchanged. `OPEN_ALWAYS` with `GENERIC_READ` can create a file or open an existing read-only file. Read-only handles from `OPEN_ALWAYS`, `CREATE_NEW` and `CREATE_ALWAYS` reject `WriteFile` and `SetEndOfFile` with `ERROR_ACCESS_DENIED`. `TRUNCATE_EXISTING` requires `GENERIC_WRITE` or `GENERIC_ALL`. A request with neither bit returns `ERROR_INVALID_PARAMETER` and preserves the file contents. A handle opened with `FILE_APPEND_DATA` and no other write access writes at the end of the file whatever the offset, as Windows does. The C runtime's `fopen`/`_wopen`/`_stat64` are in `host_posix/ucrt` (below). Not done: `MoveFileEx`, `GetDiskFreeSpace`, `CreateFile2`, file mappings, share modes. A nonzero-length overlapped read starting at or past EOF completes with zero bytes, records `STATUS_END_OF_FILE`, and signals its event if present. Both `ReadFile` and `GetOverlappedResult` report `ERROR_HANDLE_EOF`. Partial reads succeed with the bytes available, and synchronous EOF reads and zero-length requests succeed with zero bytes. `TRUNCATE_EXISTING` requires `GENERIC_WRITE` or `GENERIC_ALL`. A request with neither bit returns `ERROR_INVALID_PARAMETER` and preserves the file contents. A handle opened with `FILE_APPEND_DATA` and no other write access writes at the end of the file whatever the offset, as Windows does. The C runtime's `fopen`/`_wopen`/`_stat64` are in `host_posix/ucrt` (below). Not done: `MoveFileEx`, `GetDiskFreeSpace`, `CreateFile2`, file mappings, share modes.

**dstorage on files.** `open_for_reading` calls `host_posix::files::translate`. dstorage signals no Win32 events today (completion is the status array; the fence methods are not implemented), so nothing there needs `sync::signalled`; when fences are added, a host event set by them must be followed by `sync::signalled` so waits on several objects see it. Still open: `FileExt::read_at` on the worker pool, no `kernel_will_write` once the GPU shares memory, `mmap` paths through `victory-platform`.

**dstorage read errors.** A file read goes on through short reads and tries `EINTR` again, as `read_exact` does (`fill_at`; this process takes many signals), so a signal no longer fails a request. A failed request's log line (`dstorage: request failed`) gives the host's error with its `errno` and kind. Host errors reach the title as the Win32 codes Windows gives for them, the inverse of the UCRT's `_dosmaperr` table: `EBADF` is `ERROR_INVALID_HANDLE`, `EMFILE` and `ENFILE` `ERROR_TOO_MANY_OPEN_FILES`, `ENOMEM` `ERROR_NOT_ENOUGH_MEMORY`, `EINVAL` `ERROR_INVALID_PARAMETER`, `ENOSPC` `ERROR_DISK_FULL`, `ENOENT` and `EACCES` or `EPERM` as before, and `EIO` and the rest `ERROR_READ_FAULT`. Before, everything but the last two was `ERROR_READ_FAULT` and the log dropped the `errno`, so the 179 `hr=0x8007001e` failures before a crash on #110 could not be told apart.

## Running a title natively (`victory run` on macOS)

`victory run [--unimplemented trap|zero] [--strict] <exe>` maps the PE with `native::NativeImage`, registers its TLS, sets `runtime::set_main_image`/`set_image_path`, binds imports (unresolved ones get logging stubs that act on `UnimplementedPolicy` when *called*; loading never stops, and a per-DLL summary is printed), and stops. `--list-imports` also prints every import as `native (hle|host|title)` or `STUB`; `--execute` mounts `G:` (the title's folder) and `T:` (temp under the data dir), sets the title and console, and starts the main thread with `native::run_entry`. `--list-imports` and no-flag runs never execute title code. The host executable is linked with `-pagezero_size 0x1000` (`apps/victory/build.rs`): the default 4 GiB `__PAGEZERO` is the whole range 32-bit `XMemAlloc` results must come from. Not done: the title's own DLLs (`load_title_dlls` equivalent), the window, GPU and audio.

## Next steps, in order

1. Merging Wine's views for scatter maps. (`xmem` and `dstorage` are ported to macOS; a 32-bit `XMemAlloc` or `XMemVirtualAlloc` finds no address on macOS until the host executable is linked with a small `-pagezero_size`, since the default 4 GiB `__PAGEZERO` is the whole low range; and a committed page without access reports as reserved to `XMemVirtualQuery`.)
2. (done) Exceptions on macOS. Left: C++ exceptions (`__CxxFrameHandler`), a Mach exception port for host-code faults and debuggers, stack overflow.
3. (done) Thread notices, suspension, waits, locks. Left: mutexes, timers, `SetThreadContext`.
4. Files: `MoveFileEx`, disk space.
5. Move the remaining `cfg(windows)` HLE modules to compile on macOS with their Win32 calls behind `victory-platform`: `lifecycle` (window, `NSWindow`), `net`, `apu`/`audio` (CoreAudio), `media`.
6. Cheaper crossings: inline the TEB-only host calls (`GetLastError`, `TlsGetValue`, `GetCurrentThreadId`) in the stubs; batch crossings for hot calls.
7. Add `sig` to `HleExport`, generate bridge stubs from it (float arguments through the System V bridge, a recompiler-facing description).
8. Boot the real title loader path on macOS: `prepare`/`load_title_dlls` equivalents for the title's own DLLs, `runtime::set_main_image`, the `apps/victory` front end on macOS (`victory run --native`).
9. Metal backend against `PagePool`, and the recompiler against `NativeImage` and the bridge's register-file calling rule above.

### The C runtime (`host_posix/ucrt`)

`api-ms-win-crt-*.dll` and `ucrtbase.dll` resolve to Rust functions over libSystem, tried before kernel32's table. `msvcp140*` is not served by name: the title's own copy (C++ runtime, `__CxxFrameHandler`) is loaded from its folder. `vcruntime140`'s own copy is loaded the same way when it ships one; when it ships none, the host falls back to `vcruntime140`'s kernel-table functions (`__C_specific_handler`) and, since the ucrt serves it then, its memory and string functions (`memset`, `memcpy`, ...; #104) too. The HLE `exit` runs the `atexit`/`onexit` handlers and flushes streams before ending the process on this path (`abort`/`terminate` do not). A title linked against the dynamic UCRT needs nothing else of it. The rules that matter:

- `wchar_t` is two bytes: wide strings are UTF-16 and the host's `wcs*` is never called. Narrow strings are UTF-8; `mbstowcs` and `wcstombs` convert, and fail with `EILSEQ` on a bad sequence or a lone surrogate.
- The counted copies (`strncpy`, `strncat`, their `wcs` and `_s` forms) read no further than their count: a counted source need not hold a NUL within it, such as a fixed-size field. The `_s` forms also stop at the room left in the destination (that many units without a NUL is already `ERANGE`), and an error empties the whole destination, as the CRT does, so a failed `strcat_s` leaves no old string.
- `printf` and `scanf` are our own engine (`ucrt/format.rs`), not macOS's. `__stdio_common_v{s,sn,f}{w,}printf[_s]` and `...v{s,f}{w,}scanf` read the Windows `va_list` (a pointer to 8-byte slots, floats as doubles), honour the MSVC option bits (standard `snprintf` behaviour, legacy NUL termination, ISO wide specifiers) and the `_s` buffer rules, and format `%ls %hs %S %C %I64d %I32d %zu %p %a`, `inf`, `-nan(ind)` as the CRT does. A buffer too small is handled per mode, as the CRT's headers pick it: legacy NUL termination (`_vsnprintf`, `_vsnwprintf`) returns a text that fills the buffer whole without a NUL and cuts a longer one without a NUL (-1); no option (ISO `vswprintf`, `_vswprintf_c_l`) always ends a cut text in a NUL (-1). No buffer with a count of 0 returns the length in every mode, the sizing idiom.
- `long double` is a 64-bit `double` in the Windows x64 ABI, so the `l` math forms and `nexttoward`'s target are doubles (`nexttowardf` steps toward its double target, not one rounded to a float).
- `strtod` and its relatives scale a hexadecimal float with one `scalbn`, so what `%a` writes reads back exactly down to the subnormals (`0x1p-1074`), and a zero mantissa stays 0 whatever its exponent (`0x0p2000` was read as an infinity).
- A function with a `float` or `double` argument is `extern "win64"` and declared to the bridge (`declare_win64`) when looked up; everything else is `extern "system"`. `lldiv` returns through the hidden pointer Windows passes in `rcx`.
- Callbacks (`qsort`, `bsearch`, `_initterm`, `_execute_onexit_table`, `signal`/`raise`, the invalid-parameter handler) are title functions called with `call_win64`. `qsort` is a merge sort over a copy, so an inconsistent comparator cannot panic the host.
- `strtod`, `strtof`, their wide forms and `scanf`'s floats set `ERANGE` for an underflow to zero (nonzero digits read as 0, such as `1e-400`) as for an overflow, as Microsoft documents; zero digits and subnormals do not.
- `FILE*` is a leaked 16-byte block the title never reads; behind it is a stream shared with the descriptor `_fileno` returns, over `files::translate` paths (so `G:\` works). Reads are buffered, writes are not. Text mode drops `\r` before `\n` on reads and writes `\n` unchanged. `_filelength`, `_tell`, `_eof` and their 64-bit forms return -1 with `EBADF` for a descriptor that is not open, as Microsoft documents. `_read` of a descriptor not open for reading returns -1 with `EBADF`, as Microsoft documents, not the 0 of an end of file. A failed seek leaves the position alone and sets `errno` as Microsoft documents for `_lseek`: `EBADF` for a bad descriptor, `EINVAL` for a bad origin or a position before the start of the file (`fseek` too).
- `_findfirst*`, `_findnext*` and `_findclose` set the `errno` Microsoft documents: `EINVAL` for a null file specification or out structure and for a `_findnext` that fails for a reason other than no more files (such as a bad handle), `ENOENT` when nothing (more) matches and for any `_findclose` failure.
- `errno` is per thread (`_errno()`), with the Windows numbers (`EDOM` 33, `ERANGE` 34, `STRUNCATE` 80). The math functions set `EDOM` for a NaN from numbers and `ERANGE` for an infinity from finite ones. `remquo` stores the sign of `x / y` and the low three bits of the quotient's magnitude (taken from `|x| mod 8|y|`, so they hold for huge quotients too). The checked `float` wrappers (the `unary!`/`binary!` forms such as `expf`, `powf` and `fdimf`), `ldexpf`, `scalbnf`, `scalblnf` and `nextafterf` check the result after rounding it to a float, so a float overflow sets `ERANGE`; `fdim`, `ldexp`, `scalbn`, `scalbln`, `_scalb`, `nextafter` and `nexttoward` set it on overflow too. Underflow (a result rounded to zero or a subnormal) leaves `errno` alone, for all of them.
- `rand` is the CRT's linear generator, per thread, seed 1: `srand(1)` gives 41, 18467, 6334 as on Windows.
- `argv`, `argc`, the command line and the environment come from the host process, or from `victory_hle::posix::set_command_line` (call it before the title's start-up code runs).

Locale is "C" only (`setlocale` rejects anything else; `localeconv` is the C locale's). `vcruntime140*.dll` is served from the ucrt too when the title ships no DLL of that name, so its `memcpy`, `memset` and friends run (they came back as no-op stubs in a failed load, #104); what the ucrt has not still falls through to the host's kernel table (`__C_specific_handler` and the exception functions). A failed invalid-parameter check calls the title's handler if it set one and returns the error; it does not terminate. `_endthread(ex)` is not served (it needs an exit primitive in the loader's thread spawn); `_get_osfhandle`/`_open_osfhandle`, `_ecvt`/`_fcvt`/`_gcvt`, positional `%1$d` formats and the `_mbs*` multibyte-string family are not either. `ucrt::tests::every_ucrt_import_of_a_dry_run_is_served` checks a `--list-imports` output when `VICTORY_UCRT_IMPORTS` names it. The probe (`probe_ucrt` in `tests/fixtures/native-probe/probe.c`) links `ucrtbase.dll` and `api-ms-win-crt-math-l1-1-0.dll` and runs 17 checks under Rosetta.

## Descriptors changed after loading

A buffer or image descriptor whose dwords are not one load's consecutive dwords, though one of
them is loaded, was refused as "a descriptor changed after loading" (34 compute programs in the
story-load log of issue #123). The forms the translator reads stay as they were (a descriptor made
raw for byte accesses, `made_raw`). Any other is now refused naming the first dword that is not the
load's next one and what made it (`Tracer::changed`), after that text and a colon: "dword N is a
constant", "comes from another load or place in it", "masked or merged (s_and, s_or, s_bfi...)",
"added to (s_add, s_sub...)", "shifted, multiplied or extracted (s_lshl, s_mul, s_bfe...)",
"chosen by a select or a join", "is a program input" or "computed otherwise". Nothing new is
lowered: the next tester log tells which forms to lower (a V# base address moved by an offset in
dwords 0 and 1, the record count in dword 2, per the RDNA2 ISA's buffer resource layout).
One form is lowered: a buffer descriptor loaded whole except dword 1, which is the loaded dword 1
ANDed with a constant that clears the whole stride and nothing else (bits 16-29 cleared; the base
address bits 0-15 and the swizzle bits 30-31 kept; a partial clear may leave a stride, so it stays
refused), as in a real run's `s_and_b32` with 0xc000ffff, for a buffer access that
is not typed (`Tracer::stride_cleared`). With stride 0 the address is the base plus the offsets,
index × stride adding nothing (RDNA2 ISA 8.1.5): the access reads the loaded view raw and drops its
index (`Resources::unstrided`). Guess: the bounds stay the host view's size, which is the console's
when the title's V# checks raw bounds in bytes (OOB_SELECT 3 without swizzle: offset + payload >
num_records, in bytes with stride 0, ISA tables 33 and 35); OOB_SELECT 0 would put every access out
of range, and OOB_SELECT 1 would check the index against num_records on the console, a check the
dropped index no longer makes here. Typed accesses and other masks stay refused.
`addr_tests::a_descriptor_with_its_stride_cleared_reads_the_view_raw_on_the_gpu` (`victory-gpu`)
reads a structured view (16-byte elements) by index and offset on Metal: each lane gets dword
`lane`, not `lane × 4 + lane`.
`pipeline::changed_descriptor_tests` (`victory-gpu`) checks the refusal on the native host's own
path: the container made from the title's pieces, the host's options.
A mask or merge by a scalar ALU instruction names it and its constant operands (for a 64-bit one,
the half the dword is), for example "dword 1 masked or merged (s_and_b32 with 0x3fffffff)", or
"with computed values" when none is constant: a real run's 78 refusals at dword 1 (base address
bits 47:32, stride, swizzle in a buffer V#, RDNA2 ISA 8.1.5) need the exact form. These texts are
made at run time; refusals are `&'static str`, so each distinct one is kept for the program's life
(`resource::interned`).

Three forms, grouped from the game's dumped containers (an HDR-lighting run and a prologue
run):

* **"dword 0 computed otherwise" was no descriptor edit** (461 dumped containers, all the guessed
  pixel shaders of the HDR lighting chain). They are wave32 programs (`s_and_saveexec_b32`,
  `exec_lo`, `vcc_hi` as a spare SGPR) that also move the VCC pair as a 64-bit SGPR pair
  (`s_mov_b64 vcc, s[82:83]`). The wave-size inference took that move as a sure wave64 sign, so
  the program was built as wave64, a `v_cmp_* s[15:16]` wrote a 64-bit lane mask over s16, and the
  descriptor in `s[16:19]` read the mask. A 64-bit SOP naming VCC is now a weak wave64 sign
  (`graphics::infer_wave64`; one naming EXEC stays a strong one); with a wave32-only instruction
  the program is wave32. Guess: a wave64 program never uses the wave32-only instructions.
* **A descriptor made raw before a typed load that names its own format**
  (`tbuffer_load_format_*`, 961 dumped compute containers, "dword 1 masked or merged (s_and_b32 with
  0xc000ffff)"): the same made-raw words as above, read by bytes. That load is lowered as a format
  read of the raw view by bytes anyway, so the loaded view serves it. `buffer_load_format_*`, which
  reads the view's elements in the descriptor's format, stays refused: the cleared stride would
  change which element an offset names.
* The third form, a dword swapped into another register by three `s_xor` (about 150 of each
  dump), is lowered on main by `Tracer::xor_undone` (below).

## Known limits of this slice

- Only x86-64 hosts run title code; `native::image` also builds and tests on arm64.
- `host_posix` (kernel32 side) is a few dozen functions, enough for the probe, not a title; the C runtime is complete enough for a title that links it dynamically.
- `VirtualQuery` reports as `AllocationBase` the image or the region of ours (`VirtualAlloc`, `XMemVirtualAlloc`) holding the address, as `XMemVirtualQuery` does; only other memory gets the Mach region's base, which is where a run of one protection starts.
- `VirtualAlloc` reserves 64 KiB granules at 64 KiB boundaries (also for `MEM_COMMIT` alone with no address, as Windows does); `VirtualFree(MEM_RELEASE)` takes a base from `VirtualAlloc` only.
- `VirtualFree(MEM_DECOMMIT)` decommits the pages holding any byte of the range, and with size 0 the rest of the region (all of it from the base), as Windows does. A range that is not inside one region `VirtualAlloc` or `XMemVirtualAlloc` made fails with `ERROR_INVALID_ADDRESS`: it used to decommit whatever the host had there, and size 0 decommitted nothing.
- `VirtualFree(MEM_RELEASE)` of `XMemVirtualAlloc` memory, or of a range `XMemMapPhysicalPages` mapped into, gives its physical-page views back too (`xmem_posix::forget_physical`, as `XMemFreeDefault` does). It used to give back only the address space, so the freed pool stayed in the GPU's pool table and the aliases at an address the next allocation was given: two memories at one title address (issue #15).
- `VirtualFree(MEM_DECOMMIT)` over physical-page views (`XMemVirtualAlloc`'s pool backing, `XMemMapPhysicalPages`) keeps the views and makes them no access, instead of decommitting them (`xmem_physical::decommit`); the rest of the range is decommitted as before. Decommitting put anonymous pages in place of the pool's while the views and the GPU's pool table still showed the pool, so after a recommit the title wrote one memory and the GPU and aliases read another (issue #15). Windows only promises that a recommitted page reads zeros, so a page is zeroed when it is `XMemVirtualAlloc`'s pool backing and no other address shows it: no one else can read it. Other pages keep their bytes, since a page shown at other addresses (aliases, `map_alias`) is still read there, and pages `XMemAllocatePhysicalPages` made outlive their mappings (the title may map them again). Guess: the console keeps those pages' contents too; what it does is unknown, and `VICTORY_TRACE_OWNERS` names such a decommit. Zeroing touches every page it zeroes, also those never written.
- File calls (`CreateFileW` and the rest, `host_posix::files::win32_error`) report host errors as Windows does, from one table shared with other callers (`win32_errors::code`, the UCRT's `_dosmaperr` read backwards): `EMFILE` and `ENFILE` are `ERROR_TOO_MANY_OPEN_FILES`, `EBADF` `ERROR_INVALID_HANDLE`, `ENOMEM` `ERROR_NOT_ENOUGH_MEMORY`, `EXDEV` `ERROR_NOT_SAME_DEVICE`, `EINVAL` `ERROR_INVALID_PARAMETER`, `ENOSPC` `ERROR_DISK_FULL`, and as before `ENOENT` (`ERROR_PATH_NOT_FOUND` when the parent folder is missing too), `EACCES`, `EEXIST` and `ENOTEMPTY`. Any other host error is still `ERROR_ACCESS_DENIED`; those above used to be too.
- `FindFirstFile*` and `_findfirst*` match a pattern as Windows matches long names: it is first rewritten as kernel32 does (trailing dots and spaces dropped; a `.` before `*` or `?` may match the end of the name; every `?` matches one character, or none at a `.` or the end; a `*` before dropped dots stops at the name's last `.`), so `*.*` lists every file, `name.*` finds `name` too, and `*.` lists names without an extension. `<`, `>` and `"` in a pattern are those DOS wildcards. Checked against the long-name results of Wine's `test_FindFirstFile_wildcards` table, which runs on Windows too: the only differences are names Windows also matches by their 8.3 short names, which the host has none of.
- TLS images must be registered before the threads that use them; thread notices go to every registered image on every title thread.
- A `PagePool::map` over an existing mapping replaces it silently on macOS.
- `XMemMapPhysicalPages`, its scatter form and the GPU's `map_alias` map only over a region `VirtualAlloc` or `XMemVirtualAlloc` made, over views of ours, or over free address space (`xmem_physical::may_map`). Anything else there (the host's heap, a library, a stack) was replaced silently by the `VM_FLAGS_OVERWRITE` mapping; now the call fails, with one warning per run.
- The relocated-image test picks a free address and maps there a moment later; another thread could take it in between (it has not).
- The probe's `.tls` handling is hand-written (`_tls_used`, `_tls_start`, `_tls_end`), since it links no CRT. A real MinGW or MSVC title brings its own.

## Constant operands of packed 16-bit instructions

`v_pk_fmac_f16` (VOP2, no `op_sel`) with a constant source was refused ("a constant operand of v_pk_fmac_f16"). It now reads the constant as the 32-bit value its two halves come from, as the VOP3P path (`packed_src`) already did: an inline float constant is its f16 in the low half and 0 in the high one, an integer inline constant its sign-extended 32 bits, a literal its own two halves. Source: LLVM's assembler, not AMD's ISA text, which does not say: `llvm-mc -mcpu=gfx1030` (LLVM 23) encodes `0x3c00` as the inline constant `1.0` but needs a literal for `0x3c003c00`, and `0xffffffff` as `-1` but `0xffff` as a literal. The Windows GPU test `packed_fmac_reads_constant_halves` checks the translation on the host, not the console's result.

## Root signatures without a host `d3d12.dll`

`D3D12SerializeRootSignature`, its versioned form and `D3D12CreateRootSignatureDeserializer` call the host's `d3d12.dll` when there is one (Windows). On macOS `victory-gpu/src/rootser.rs` does both in Rust: it writes `RTS0` blobs (versions 1.0 and 1.1) and reads them back, bare or as the `RTS0` part of a DXBC container, into an `ID3D12RootSignatureDeserializer` whose description lives as long as the object. A malformed blob (truncated, an offset or count outside the blob, more than 4096 parameters or 2048 static samplers, a version other than 1 or 2, a parameter type above 4) gives `E_INVALIDARG` and no object. The AMD FidelityFX upscaler deserializes its root signatures at the first world frame (29 calls in testers' logs, which used to fail with `E_NOTIMPL`).

Guess: a 1.1 blob is returned as the 1.0 description, its range and root descriptor flags dropped, which is believed to be what Microsoft's 1.0 deserializer does. The Windows test `we_read_the_hosts_blobs_as_it_does` has Microsoft's serializer write 1.0 and 1.1 blobs, compares Microsoft's reader and ours on them, and settles it. Microsoft's blobs are DXBC containers with an `RTS0` part, which ours reads by the same path; our serializer writes a bare `RTS0`, which only our reader is given. The container parse rules (header fields, declared total, part bounds, duplicate fourccs) are pinned by `dxbc::tests` in `victory-shader`.

Further forms are taken at a join of several edges (27 November; met in the title's compute passes around its HDR scene colour, 1280x720, and its shadow pass). The console compiler reuses mask registers for descriptors and constants on paths the lanes leave by, so a join often holds data on an edge whose lanes never read it. `Lower::unneeded_edges` finds those (join, edge) pairs as a greatest fixpoint, once per program: every instruction reading the join is an AND or NAND with EXEC (or an `s_orn2` by EXEC) in the join's block with no lane of the edge in that EXEC as the edge brings it (their bits of the result are 0, or 1, whatever the join holds), and every later join taking it is on an edge those lanes do not take from the block, or is in the set itself (a nested if's else part writes the register again before any read; a loop passes it round), propagated backward by a dependency worklist with memoized `(block, pred, succ)` reachability so deep SGPR phi chains in large pixel programs do not re-evaluate `bools.meet` quadratically (`io::PrimMaskUses` likewise indexes SSA uses once across entry SGPR checks). Such an edge brings any bit. A first version searched this recursively to a fixed depth, which went exponential (9 s to minutes a program, time that feeds the watchdog); the fixpoint takes well under a second. A join argument no mask tracks but the structure knows as a lane function without opaque or quad atoms (a constant moved before the edge, `s_cselect` of -1 and 0 by SCC, a loop join that resolved) is each lane's bit of that function, unless another edge of the join brings a mask that may differ from it in a lane of this edge (`Lower::mask_differs_on_edge`): the console wave runs every path some lane takes, so where that edge's block ran the register holds its mask for the skipping lanes too (a constant 5 or 0 on the skip edge of an if whose body copies the saved EXEC: those lanes read 1 on the console; a back edge, from a block the join dominates, is left out, as its block runs after those lanes have gone on). Where the skip edge brings a join made in its own block (an if's then part writes the register and this edge skips the else part), that join is an atom the structure knows nothing of, so where the atom alone does not show the two agree, each edge into that block is weighed for its own lanes: the then lanes against what the then part wrote, the others against what they brought, and lanes brought a value not known differ wherever there are any. Without this the check refused 104 pixel programs whose then part writes -1 and whose else part sets the bits of the lanes outside it (`s_orn2` by EXEC), which read 1 either way (`lower::tests::a_then_parts_constant_on_the_else_skip_edge_is_weighed_for_its_own_lanes`, with the refusal of 5 against the saved EXEC beside it). Known gap: the older rule for a constant of every lane or none on the host's lanes (`s_mov_b64 s, -1`) has no such check, so a constant 0 there is still taken as the skipping lanes' bit (the ignored test `lower::tests::a_skip_edge_constant_of_no_lane_is_not_the_bit_either`). The check was tried on it and taken off: it refused pixel programs translated rightly for long (a loop with a per-lane break whose mask is only ORed with EXEC at the counter's exit, as `pixel_0313649a` at 0x5b8, where the lanes that broke early are in no such EXEC), which the structure cannot prove without knowing a loop's passes. The first version lacked this check and translated that case wrongly, which the Windows-only `synthetic_tests::a_compare_mask_joined_with_data_on_the_skip_edge_is_0_there` caught in CI; `lower::tests::a_skip_edge_constant_is_not_the_bit_where_the_arm_keeps_the_skipping_lanes` is the same case on every host, inside a uniform loop's body besides. With these the HDR pass (`5f8b2e3b`), the small pass beside it, the two indirect passes and the shadow pass's wave32 program translate offline; the shadow pass's wave64 program stops at LDS in a split wave instead. A run on 27 November with the first rules had the HDR target go from 8 draws a frame and black to about 60 draws, lit over 0.2 to 0.4% of the image (max up to 5,800): still dim, as the G-buffer before it is mostly black and 240-odd forward draws into it are refused. The rules were found with an adviser's study of those programs. Tests (`victory-gpu`, Metal), in `split_wave_tests`: the positives `data_joined_into_a_mask_seen_only_anded_off_at_a_later_join_is_not_needed`, `data_joined_into_a_mask_whose_lanes_write_it_again_in_an_else_is_not_needed`, `a_mask_join_read_by_two_ands_with_execs_its_skipping_lanes_are_not_in_is_theirs_at_zero`, `a_mask_a_uniform_loop_carries_round_anded_off_is_not_needed`, `data_joined_with_a_mask_nanded_with_exec_is_set_for_lanes_outside_it`, `a_mask_selected_by_scc_is_scc_for_every_lane`, `a_mask_selected_by_scc_joined_with_data_nanded_with_exec_is_each_lanes_bit`, `a_mask_of_the_lanes_leaving_a_loop_by_a_uniform_exit_holds_them`, `a_mask_moved_to_every_lane_before_a_branch_to_a_join_holds_their_lanes`, `a_mask_anded_with_an_exec_its_skipping_lanes_are_not_in_is_theirs_at_zero`; and negatives that must stay refused (reads by `s_or` or `s_xor` with EXEC, a select of an address by SCC).

The G-buffer's six 32-lane compute programs (`4359f972`, `d8ff071a`, `e456bd43`, `3a137e94`, `5e820fc2`, `c56c4093`; 27 November) stopped at the head of one waterfall loop over descriptors, at about 0x1e4, with the same refusal. Two things the structure did not know caused it. (1) A 32-lane program makes two masks at once with `s_mov_b64 s[10:11], 0` or `s_bfm_b64 s[10:11], 32, 32` (none, and every lane). These are now constants for each half (`const32`), and a 32-lane value that a lane-width operation does not make falls through to that check. (2) VCC holds a mask from before the loop, and the loop copies it to another SGPR and back inside an if. At the loop head, every edge brings either VCC, a copy of it, or a join of those. `Lanes::unchanged` (`structure.rs`) follows that web of joins and `s_mov` copies (at most 64 values), and when it ends in one value, the join is that value. This is plain SSA equality: `s_mov` ignores EXEC, so lanes need no reasoning. With both, all six translate offline. Over the 1,756 refused compute programs, 51 more translate (347 to 398) and none regress. Tests (`victory-gpu`, Metal), in `split_wave_tests`: `a_mask_a_32_lane_wave_starts_as_a_pairs_constant_holds_the_lanes_ored_in` (both constants), `a_mask_a_divergent_loop_copies_away_and_back_is_the_mask_it_entered_with`, and the negative `a_mask_a_divergent_loop_replaces_is_not_taken_as_the_one_it_entered_with`. In that negative, the copy back is of another mask: it stays refused, or runs right.

## Lane masks holding lanes elsewhere, read as data

A lane mask read as a number is the ballot of its lanes' bits where it is written; a mask that also holds lanes not running there (a complement, a copy made where EXEC is not known) is instead the scalar operation that writes it, applied to its operands' numbers (`Lower::mask_number`). Where that is not possible the refusal now says why: "a lane mask holding lanes elsewhere read as data: " then "in a mesh, hull or geometry program" (any stage but compute, pixel and vertex), "a program input", "not written by a scalar ALU instruction", "written by a scalar instruction of the other width" (a `_b32` operation in a wave64 program, such as `s_and_saveexec_b32`), "the EXEC a saveexec keeps, or a mask written with EXEC", "shifted on the host's lanes", "an operand no SGPR or constant gives" or "a scalar operation not modelled". The story-load log of issue #110 (Series S) has three compute programs refused with the old, single message; the next log says which of these they are. Of the forms tried by hand in a single-block wave64 compute program (`s_and_saveexec_b64`, `s_orn1_saveexec_b64`, `s_brev_b64`, `s_wqm_b64`, `s_cselect_b32`, `s_mov_b32` and `s_not_b32` of `vcc_lo`), only `s_and_saveexec_b32` reached this refusal when `EXEC` on entry covered every lane (`st.active` held the mask, and `s_wqm_b64` hit the `Atom::Quad => true` derivative shortcut); once written under a narrowed `EXEC` (`!st.active.contains(&lo)`), `s_*_saveexec`, `s_*_wrexec`, `s_brev_*`, and `s_wqm_*` entered `Lower::mask_number` and were refused.

`Lower::mask_number` now models those non-active compute forms (`MaskData::Console`, `#258`): `s_*_saveexec` (saved incoming `EXEC` in `sdst`, combined mask in `EXEC`), `s_*_wrexec` (including `andn1` and `orn1`), `s_bitset0`/`s_bitset1`, `s_bfm`, `s_brev`, `s_quadmask`, and `s_wqm` (`Lower::wqm_word`, also supported in `Lower::salu` on plain scalar data, with `Lower::mask_dword` routing `Atom::Quad`-bearing masks through `mask_number` so `s_wqm` read as data returns the quad-expanded mask rather than all-1s); pixel and vertex programs (`MaskData::Host`) continue to refuse them (`emit::mask_tests::wqm_bitset_bfm_brev_quadmask_and_saveexec_masks_lower_as_data`, `emit::mask_tests::pixel_and_other_width_mask_number_negatives_stay_refused`, `split_wave_tests::mask_number_bfm_brev_bitset_wqm_saveexec_and_wrexec_run_on_the_gpu`).

"A program input" was also what a loop's join gave, as the join has no defining instruction: a structurized loop keeps the lanes still looping in an SGPR beside EXEC (a copy of EXEC before the loop, `s_andn2_b64 s, s, exec` and its SCC for "does any lane loop on?" at the head, `s_andn2_wrexec_b64 s, s` at the back edge, which writes the same result to both). Where the structuriser could not prove that every lane arriving at the head has its bit set (an outer loop with lanes that broke out makes the arrivals unprovable), the head's SCC was a number of that join and was refused. That join is EXEC at the head: on every edge it takes EXEC's own incoming value (a copy, or the same `_wrexec` result), so it now counts as the head's EXEC mask and a ballot there gives it (`structure::Lanes::exec_copies_at_joins`). Found on the world's lighting compute `9e5a12054881a74d` (#258), the only writer of a 1280x720 R11G11B10 target the host otherwise never wrote; that program then stops at the next refusal, the lanes that broke out of a loop read back through copies ("a lane-mask join of an untracked mask: data whose high dword is not known"). Of 4,129 refused containers dumped at boot (27 November, Series S), the 15 refused as "a program input" are all such joins; only that compute one changes. Test: `mask_tests::a_loop_register_that_is_exec_on_every_edge_is_exec_at_the_head` (two nested loops of this shape; the inner loop's register was not the head's EXEC mask before).

That program's next refusal was a join, not the lanes that broke out: after a loop that an EXEC skip (`s_cbranch_execz`, `s_cmp_eq_u64 exec, 0` and `s_cbranch_scc1` once lockstep rewrites it) can jump over, an SGPR that holds the loop's mask on one edge and leftover data on the skip's (descriptor dwords, the high one with no value) is read only ANDed with EXEC. On the skip's edge that EXEC is the one the skip found empty, so the AND keeps no lane whatever the SGPR holds: the edge is not needed (`Lower::unneeded_edges` now knows that edge, where the lane algebra could not prove it), and a join that lockstep takes as a number no longer asks for that edge's dwords first. Measured offline (27 November) on the 4,129 refused containers of a boot run: with both changes 28 compute programs translate (`9e5a12054881a74d`, `2fcb1ef0e54c7df4`, an HDR writer, and `113a4779d954d6d2`, `a81f1b5bb0d7ba01`, `dd1d8e07445421e6` among them), 2 more stop at a later refusal, nothing else changes; Metal made the compute pipelines of 21 of the 28 (not run) and the host refused the other 7 as "needs view addresses or write marks". Test: `gfx_refusal_tests::a_join_read_only_with_exec_needs_nothing_from_an_exec_skip` (refused before as "data whose high dword is not known"). The rule also follows the value through one unconditional relay: a join that is not read in its own block but passed unchanged by a single jump into the next block's phi, and read there only ANDed with an EXEC that the skip found empty (a compute program, `d3ff71633430b06d`, was refused as "a lane-mask join of an untracked mask" for want of that). Only one hop and only reads in the relayed-to block count. Tests: `gfx_refusal_tests::a_join_relayed_by_a_jump_and_read_only_with_exec_needs_nothing_from_an_exec_skip` (refused before), the two that stay refused, `a_relay_block_that_rewrites_exec_still_needs_the_data` and `a_wave64_relay_skip_on_exec_lo_alone_does_not_make_an_edge_unneeded`, and on Metal `split_wave_tests::a_mask_relayed_by_a_jump_and_read_only_with_exec_is_its_lanes_in_lockstep`.

## Shader and pipeline cache on disk (native path)

On by default; `VICTORY_NO_SHADER_CACHE=1` turns it off (an empty value or `0` leaves it on). Everything lives under the cache folder (`cache_subdir("native-shaders")`, for example `~/.cache/victory/native-shaders/<translator>/`), never in the repo, and is never sent anywhere. Three layers, each written to a temporary name and renamed:

- **Translations** (`translated/`, `victory_gpu::shader_cache::native_shader`): the `HostShader` of a console container, including `table_entries` and `splits`, keyed by the container's bytes, the translation `Options` (rectangle lists, extra and copied parameters), and the build key (`victory_shader::SOURCE_HASH`: the translator's sources, manifest and build script, the compiler's version and the `Cargo.lock` entries of the packages the translator depends on, not the whole lock file, which made every dependency change anywhere in the workspace a cold cache, #334; the text of `shader_cache.rs` which defines the encoding, the loop limit, and the translator's environment switches that are set, `victory_shader::emit::TRANSLATION_SWITCHES`: `VICTORY_DEBUG_*` and `VICTORY_NO_LOCKSTEP`). Before the switches were in the key, a debug switch did nothing on a warm cache, and a translation made with one was served to later runs without it. The folder is named by the start of the build key, so a changed translator reads nothing old and old folders can be deleted whole. It is the same store and encoding as the Wine path's (`use_shader_cache`), in its own folder; the encoding now carries `table_entries` and `splits` too. A refusal is stored with its reason.
- **Libraries** (`metal/`, `victory_metal::disk`): the converter's metallib and shader stage, keyed by the DXIL, the root layout, the entry name, the format and the converter itself (its path, size and modification time). Each file has a magic and a checksum; a damaged file is a miss and is converted again.
- **Which converter.** At start-up the log names the converter it loaded (`metal: shader converter <path> <size> <mtime> (from VICTORY_MSC_DYLIB | the installed default)`, `msc::load_note`) and warns when it is not `/usr/local/lib/libmetalirconverter.dylib`. A tester's run with the copy bundled with Unreal Engine 5.6 (strings say `D3DRendererMetal-30.2`) drew the loading screen's panels but no logo or art, with no refusal or GPU error; Apple's 4.0 beta 3 package drew it (#295). The cause is the converter's generation, not who packaged it. Before 3.0, Apple's converter takes every 1D and 2D texture as an array type (its manual, "Texture arrays"), and 3.0 made plain types the default (its changelog: "Default support for non-array texture types"). The host binds a 2D texture as a plain 2D view, which is what 3.0 and later expect. Through Apple's 1.1 and 2.1, a compute shader's writes to such a texture are dropped and the texture stays zero, with no error: `a_texture_compute_writes_is_copied_back_into_title_memory` fails on 1.1 and 2.1 and passes on 3.0 and 4.0 beta 3 (the rest of the victory-gpu suite, 346 tests, passes on all four). With the views made as 2D arrays it passes on 2.1 and fails on 3.0. Unreal's copy is `D3DRendererMetal-30.2`, older than Apple's 2.1 (`31.2`; 1.1 is `20`, 3.0 is `32.47`), so it is of the generation before 3.0. `Msc::open` now refuses a converter older than 3.0, told by an export that 3.0 added (`IRCompilerSetFunctionConstantResourceSpace`, present in 3.0 and 4.0 beta 3, missing from 1.0, 1.1 and 2.1). The log says why (`metal: no shader converter: ... older than 3.0`), and `native-setup.sh` warns about one. Making array views for an older converter instead is possible but not done: every 2D view and directly bound texture would need the other type, for a converter Apple has replaced. No version is logged because the library gives none: its Mach-O current version is 0.0.0, and the C API (`metal_irconverter.h`) has no version call. Because the library cache's keys hold the converter's path, size and modification time, two converters never share cached metallibs. Tests: `msc::tests`.
- **Pipeline states** (`metal/pipelines-<gpu>.metalarchive`, `victory_metal::archive`): an `MTLBinaryArchive` of the compiled compute, render and mesh pipelines. A pipeline is looked up with `FailOnBinaryArchiveMiss`; on a miss it is compiled as before and added. A `pipeline-archive` thread serialises the archive every 5 s while pipelines are added (a write takes about half a second, and a cold world frame's thread waited for 21 of them, 10 s), and the backend does when it drops. Metal refuses an archive from another GPU or OS version; it is then replaced by an empty one. Limit: a run killed within 5 s of its last new pipeline loses those pipelines. Guess, marked in the code: Apple does not say an archive may be added to while another thread reads it, so adds hold a lock that lookups and the write share (`Archive::changing`), and a compile after a miss does not name the archive.
- **Pipelines off the frame** (`Metal::create_compute_pipeline_within`, `create_render_pipeline_within`, `metal::within`, `Title::compile_wait`): a cold first world frame made about 210 compute pipelines one after another on the render thread, 97 s of Metal compiles and 10 s of library conversions in a 122 s pause; with only those moved off it, the same frame still ran into the title's 30 s watchdog on the library conversions and compiles of some 800 render and mesh pipelines. Every compute, render and mesh pipeline is now converted and compiled on a `pipeline-compile` thread (as many as cores, taking the pipeline asked for last first: that is the one a frame may be waiting for), and the device has `shouldMaximizeConcurrentCompilation` on. The frame that first asks for a pipeline waits for it only while it has budget left (`prewarm::FrameBudget`, the 10 s of the translations above); past it, it waits 20 ms a pipeline, which is enough for one the disk cache and the archive hold, so a warm frame that outlasts its budget still puts none off, until one is not made in that time; then not at all (`glue::compile_wait`; 20 ms each for 800 would be 16 s, and a fixed 2 s of them made each later cold frame 12 s instead of 10). A guessed pipeline is logged (`guessed pipeline`) once it is made, not at each draw that asks for it meanwhile. Past that the draw or dispatch is dropped that time, nothing is cached (`ShaderStats::put_off`), and a later one picks the pipeline up once it is made, without waiting (a first version waited again inside each later frame's budget, and the frames after the first stalled 10 s each in turn); the ones put off compile side by side meanwhile. The log says `pipelines put off for their compiles so far put_off=<n>` at each power of two. As with translations this is a host policy, not console behaviour: a pass the title runs once in such a frame stays undone. The 10 s compile deadline and crash memory of render pipelines cover compute and mesh pipelines too (#243); the deadline is 60 s now that compiles run side by side off the frame (a cold world frame's took up to 15 s each), and the slow-compile and at-submission archive writes of #243 are the `pipeline-archive` thread's. Tests: `metal::tests::a_pipeline_still_compiling_is_picked_up_later`, `shader_tests::a_compute_pipeline_put_off_is_made_by_a_later_use`, `shader_tests::render_pipelines_come_from_the_translated_shaders_and_the_streams_state`.
- **Old builds** are deleted: each run stamps its build folder (`last-used`), and a background thread at start-up deletes all but the three used last, its own always kept, and never one stamped within the hour (both `game-run.sh` slots share one cache folder, so another run may still be using it) (`shader_cache::prune`; each is about 400 MB). Only folders named by 16 hex digits are touched. The Wine path's folders are not pruned, since asking Wine for file times has hung its workers (#50). Test: `shader_cache::tests::old_build_folders_are_pruned`.
- **Reading the log.** The `native shader translation ms=.. hit=..` and `metal: library conversion .. ms (.. from the disk cache: ..)` lines are written only for work of 20 ms or more, which a hit never takes, so every such line says `hit=false`, even when nearly everything hits. That is what made a tester's warm runs look as if they missed everything. `VICTORY_TRACE_SHADER_CACHE=1` adds a `native shader cache so far` line to each frame log (every 64 frames), with `shaders_cached` and `shaders_translated` (translations from the cache and made, `shader_cache::counts`) and `libraries_cached` and `libraries_converted` (Metal library lookups that found or missed their entry, `victory_metal::disk::counts`). We checked the keys on our own logs (27 November, about 20 groups of runs that share a slot and a build folder). Each group's first run had 3,600 to 4,600 slow translations, and the later runs of the same build had 0 or 1. The exceptions (52 to 790) were shaders the first run never reached: the cold run was killed by the watchdog before prewarming finished, or the warm run went further. Most of those containers had never missed before. No key part changes between runs: the container's bytes, the `Options` text, the build key (the translator's sources, `Cargo.lock`, the compiler, this encoding, the loop limit), and, for libraries, the DXIL, the root layout text, the entry and the converter's path, size and mtime. A run is cold again after any change to `victory-shader` (except its `*_tests.rs` modules, which `build.rs` leaves out of the hash; `tests::test_modules_are_test_only` checks each is declared under a `cfg(test)`), `Cargo.lock` or `shader_cache.rs` (a new build folder, Metal libraries included), and in each `game-run.sh` slot, which has its own cache folder. Warm runs that still hit the title's watchdog (+0x23cf683, then +0x55e56f8) had no slow translations or conversions. Their world frames' first binds ended 23 to 27 s before the fault, and nothing was presented after that, so the stall there is not shader work. Test: `shader_cache::tests::a_second_run_reads_the_first_runs_translations`.
- **Compiler crashes** (`metal/crash-<key>`, `victory_metal::disk::remember_crash`): Metal's compiler service crashes on some pipelines (`XPC_ERROR_CONNECTION_INTERRUPTED`, or `Compiler encountered an internal error` on macOS 15; see [#181](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/181)), and Metal retries before it reports the crash: about 0.7 s a pipeline on macOS 27, about 30 s on macOS 15, on the frame. The same seven G-buffer pipelines crashed it in every run of the world (`vertex a1785729710c22b3 pixel 80b896aafd2ec5f4` and six others), so the crash is the pipeline's, not the load's. Render, mesh and compute pipelines each compile on a thread of their own (`archive::compile_render`, `compile_mesh`, `compile_compute`); one that takes over 60 s is refused for the run (10 s while the frame's thread waited for it; see "Pipelines off the frame"). A crash, reported in time or after the deadline, is remembered by the stage DXIL and the OS build (`kern.osversion`), so later runs refuse the pipeline at once ("Metal's compiler crashed on it on an earlier run"). An OS update, or deleting the files, tries them again. One crash of the compiler service fails every compile it had in flight, so a single failure does not say which pipeline caused it: in runs of 28 November one crash report took down about 15 G-buffer pipelines a run, each then refused for good, and some 600 G-buffer draws a frame were dropped. A pipeline is now remembered only at its third crash in a run (`victory_metal::CRASHES_REMEMBERED`); before that its compile is put off (`ShaderStats::crashed`, `put_off`) and a later use tries it again. Test: `shaders::compiles_the_compiler_crash_took_down_are_tried_again`. Shortcut: a pipeline that compiles correctly but takes over 60 s is refused for that run and isn't archived. A compile that ends in a crash also spends the rest of that frame's pipeline budget (`glue::compile_failed`, `prewarm::FrameBudget::spend`): in a run on 27 November (commit 38786fd) nine `XPC_ERROR_CONNECTION_INTERRUPTED` refusals, all in `frame~2259` between 05:19:27.027 and 05:19:34.304, spent 7.3 s of one frame waiting for compiles that died, about 0.85 s each. With the budget spent the frame's other new pipelines are put off as they are past its budget, and made in later frames, where the crash memory refuses the bad ones at once. A frame a crash spent is handed no new pipeline at all: `glue::compile_wait` gives it not even the 20 ms window a pipeline the archive holds takes (`FrameBudget::spent_by_crash`), because the compiler service is down for the rest of that frame. In the run after the fix (commit d52860e7) the first crash of `frame~2347` (`vertex a1785729710c22b3 pixel 80b896aafd2ec5f4`, after a 677 ms mesh compile) spent the budget, and the same frame's next pipelines were put off within 22 ms (`put_off` 1, 2, 4); 15 frames logged the warning and none paid a second crash. Earlier, with only the budget spent (d52860e7's first form), 12 pipelines of one frame were still tried in 0.22 s and each failed at once, refused and remembered as a crash of its own although the service being down was not the pipeline's fault. It is a host policy, as above: the crashing pipeline's own compile still costs the frame what Metal spent on it (677 ms in that run; up to the frame's remaining budget, about 10 s on macOS 15). One warning a frame says so: `a compile crashed Metal's compiler; no new pipelines this frame`. Tests: `prewarm::tests::a_spent_frame_starts_the_next_one_whole`, `native::glue::a_compiler_crash_spends_the_frame_budget`.

- **Prewarming** (`VICTORY_NO_SHADER_PREWARM=1` turns it off): each distinct shader the title registers is handed to the `shader-prewarm` threads (the performance cores but two, at least two, at the lowest priority of the default QoS class so the title's threads go first; `VICTORY_PREWARM_WORKERS=<n>` sets the number; a title thread that waits on a translation one of them holds lends it its priority) that call `Shader::native()` and drop the result, so translation happens during the title's load, off its threads. Without it the first world frame binds about 1,300 pipelines for the first time and translates each inline on the render thread; large compute shaders take 190-600 ms each (mostly structurisation), 60-70 of them add up to 25-29 s, and the title's own 30 s watchdog kills the game. `native()` is a `OnceLock`, so a bind that beats the pool waits for or does the translation itself, as before. The log says `shaders prewarmed n=<n> pending=<queued> dxil_mb=<held>` at each power of two; `dxil_mb` is the DXIL the prewarmed translations keep. Measured cold on an M5 Pro (series-s, 2026-11-27): the title registers about 21,600 shaders, the pool spends about 640 CPU-seconds on them (two thirds in the ~700 compute shaders of 64 KB or more; there are no larger pixel or vertex shaders), and the translations keep about 40 MB of DXIL while their containers are let go, so the prewarm adds no measurable peak memory (peak RSS 9.1 and 13.9 GB with it, 16.0 and 16.4 GB without; the spread between runs is larger than any effect). The world frame came 3-87 s later with it in those runs, within the spread of runs without it (158-202 s). On a cold cache, a pool of all cores but two took the cores the title's own threads needed: on an M5 Max (6 performance cores of 18; series-s, 27 November, a fresh cache folder per run) 16 workers dropped about 216 frames of the boot movie and menus in their first 50 s (39 seconds had a gap over 50 ms, the longest 1.9 s), and 4 workers about 50 (7 seconds, the longest 0.85 s), against about 70 for a warm cache with 16. With 4 the world still loaded (frame 2538 at 170 s, 1024 pipelines put off, no watchdog). Twelve of the 16 in the utility class was worse (freezes of up to 7 s): a title thread waiting on a translation a starved utility worker holds waits on its `OnceLock`, and the utility class keeps a thread on the slower cores. The pool takes first the shaders an earlier run bound (`victory_gpu::prewarm`). Each first bind adds the shader's short hash to `first-binds.txt` in the cache folder, beside the build folders. It names containers, not translations, so it outlives a translator change. The list is rewritten every 64 new entries through a temporary name, so a run the watchdog ends loses at most 63, and it holds at most 16,384. The next run reads it and queues those shaders ahead of the others, in the title's order within each queue. The log says `shader prewarm pool started threads=<n> first_binds=<listed>`. Tests: `prewarm::tests::shaders_bound_before_come_first`, `the_pool_is_the_performance_cores_but_two`, `the_list_keeps_earlier_runs_and_adds_this_ones`. For a run with no list (a tester's very first run), each frame has a budget of 10 s of wall time from its first new host pipeline (`glue::pipe`, `prewarm::FrameBudget`). Wall time, because the frame's thread also translates inline elsewhere (rectangle lists, extra parameters, thread counts): a first version that counted only the time spent making pipelines began putting them off 23 s into the frame. Once the budget is spent, a pipeline whose shaders are not translated yet is put off: its draw or dispatch is dropped that time, the result is not cached (a later draw makes it), and its shaders go to the front of the pool. The log says `pipelines put off for their translations so far put_off=<n>` at each power of two. This is a host policy, not console behaviour. It drops work only in a frame that would otherwise be held past the title's 30 s watchdog. A pass the title runs only once in such a frame stays undone. Guessed graphics pipelines' rectangle-list and extra-parameter translations (`glue::guess`) and thread-count compute translations (`glue::thread_pipe`) are budgeted the same way. Once the budget is spent, one that is not made yet (`Shader::rect_ready`, `Shader::threads_ready`, or not on disk, `shader_cache::native_present`) runs on a `shader-put-off` thread, one per variant at a time, and the draw or dispatch is dropped until it is done. In a warm run on 27 November, one such extra-parameter translation took 114 s inline in the world frame. Test: `prewarm::tests::each_frame_has_its_own_budget`.
- **Warm-up ahead of a run** (`victory translate --cache`, no paths, on a Mac; `apps/victory/src/warm.rs`): translates the containers earlier runs kept, on every core (`--jobs`), with no title running. It is meant for the first run after the translator changed: that run's cache is cold, and its prewarm otherwise translates everything while the title loads (about 21,600 shaders, 100 s on 12 cores of an M5 Pro, measured 27 November). Each container a run translates is kept as the title passed it, once, under its short hash in `<cache>/native-shaders/containers` (`prewarm::keep`, from `Shader::native`): beside the list of first binds, so it outlives the build folders. A file whose bytes are not its name's is skipped. The warm-up does not read the game's files, as the Windows one does: the title hands the native host pieces (code, metadata, `graphics_containers` and `compute_container`) and the container is put together here, so the containers in the game's shader archive are other bytes with other cache keys. Measured on a full cache (27 November): of 25,985 distinct containers found in the files, 58 were the run's. Not covered: the translations a run asks for by how a shader is drawn (rectangle lists, extra parameters, thread counts), and the Metal libraries and pipelines, which need a pipeline's root layout and state; the run still makes those, and `--pipelines` stays Windows-only. No switch of its own: `VICTORY_NO_SHADER_CACHE` turns the keeping off with the cache. Measured on an M5 Pro, 24 GB (27 November): a run to the world keeps 21,596 containers, 322 MB. With four open pull requests merged that change the translator (a new build folder, empty), the warm-up went through all of them in 67.9 s on every core: 17,469 translated, 4,127 refused, none damaged. The cold run's prewarm took about 100 s for the same work, with the title loading beside it. The caches earlier runs filled hold as many entries with 6,141 and 5,429 refusals, on older translators. The run after it found them: `shaders_cached=21564 shaders_translated=2` at frame 1920 (`VICTORY_TRACE_SHADER_CACHE=1`), and its prewarm was done 19 s in, as on a warm cache. Tests: `warm::tests::the_warm_up_translates_what_a_run_kept`, `prewarm::tests::kept_containers_read_back_and_damaged_ones_do_not`.

A program with register-relative moves (`v_movrel*`, `v_swaprel`) is structured once fewer (`range::build`): its form with unbounded moves was structured first even when the bounded form was used. On synthetic programs of about 18,000 words with one such move, `range::build` went from 195-204 to 160-165 ms (nested branches) and 78-85 to 65-68 ms (shallow ones), release build, M5 Pro. Programs without such moves were structured once already. A program whose unbounded form does not structure is now tried bounded too.

Slow work shows in the log: a translation or a conversion of 20 ms or more logs `native shader translation` / `library conversion` with whether the disk had it.

Translations of 20 to 40 s (27 November; 44 to 48 KB pixel shaders) were lowering done over: `lower_graphics` lowered the whole program again for each join with an incoming value that has no DXIL form (574 times for one shader), and `fill_phis` now reports all the joins a pass finds, so two passes do. The shaders were refused in 30 ms until descriptor tracing followed dwords swapped by XORs, and then reached this. `descriptor_data` also searched every descriptor's sources for each load read as data: one table (`load_use`), worth about a tenth of the heaviest translations. The walk in `live_edges` is bounded separately (see "Liveness of phi edges"). A program with several such joins may now be refused for another of them than before, so tallies of refusals by reason shift without any shader changing between translated and refused. Test: `joins_without_a_value_are_found_in_one_pass` (`gfx_refusal_tests`) bounds the passes, not the time.

## Buffer atomic subtraction

DXIL's `AtomicBinOp` has no subtraction, so `buffer_atomic_sub` was refused by name (3 compute
programs in a 12,020-shader corpus, Series S boot to the menus, 27 November). It is now an add of
the negated data (`0 - d`): modulo 2^32 the dword left is the same, and the value returned with
`glc` is the old one either way. Over that corpus the 3 translate and no other output changes.
`heap_tests::a_buffer_atomic_sub_leaves_the_difference_and_returns_the_old_value` (`victory-gpu`)
runs it on Metal, wrapping below zero (1 - 3). The 64-bit, image and global forms are not
covered: no title shader in the corpus uses them.

## Clamped integer sums and differences

The VOP3 clamp bit on `v_add_nc_u32`, `v_sub_nc_u32` and `v_subrev_nc_u32` saturates the result
as an unsigned number. LLVM's AMDGPU backend selects `uaddsat` and `usubsat` to these with clamp:
a sum that wraps gives 0xffffffff, and a difference below zero gives 0 (`Lower::integer`,
`SATURATING`). It was refused for every integer operation as "integer output modifiers": 8
compute programs in a 12,020-shader corpus (Series S boot to the menus, 27 November), each a
`v_sub_nc_u32 ... clamp` whose result goes to LDS. Over that corpus the 8 translate and no other
output changes. Clamp on other integer operations (the signed `_i32` forms, `v_mad_u32_u24` and
others) and `omod` stay refused. `split_wave_tests::clamped_unsigned_sums_and_differences_saturate`
(`victory-gpu`) runs both on Metal.

## A half kept by one `v_fma_mix` and replaced by the other

The title's pixel programs often pack two f16 results into a register that still holds an input
they are done with, a barycentric (`v0`/`v1`) or the first half of an interpolation.
`v_fma_mixlo_f16 v1, ...` writes the low half and keeps the high half, then
`v_fma_mixhi_f16 v1, ...` replaces it (or the two come in the other order), and `exp ... compr`
writes the pair. The kept half has no DXIL form. A `v_fma_mixlo_f16` already recorded such a value
as unknown, with the low half it made in `low16`, but `v_fma_mixhi_f16` refused it outright.

Now either one records the half it writes (`low16` or `high16`) when the half it keeps is unknown,
and the other one, finding that entry, keeps the known half and makes the value whole. Entries
exist only for those `mix` results, so anything else that reads such a value is refused as before.
So is a kept half that is exported or read: 2 shaders now name the other half's input, the first
half of an interpolation, where they named the barycentric before, and some of the corpus's
shaders export `v0` after a `mixlo` alone.

Over a 12,020-shader corpus (Series S boot to the menus, 27 November), 91 shaders now translate
(90 pixel, 1 vertex), and no other output changes. They were refused as:
- "barycentrics read directly": 69;
- "the first half of an interpolation": 18;
- "a lane mask holding lanes elsewhere read as data: not written by a scalar ALU instruction": 3;
- "a descriptor built around a table in the code": 1 (vertex).

Each of the 91 has `mix` pairs. `mesh_tests::a_half_kept_by_mixlo_and_replaced_by_mixhi_draws_without_the_barycentric`
(`victory-gpu`) draws the shape on Metal in both orders: a pixel program that interpolates with
`v0`/`v1` and then packs (1.0, 0.5) into `v1` draws (1.0, 0.5, 1.0, 1.0).

## LDS instructions addressed by M0

`ds_append`, `ds_consume`, `ds_read_addtid_b32` and `ds_write_addtid_b32` have no address
operand and were refused as "an LDS address" (three compute programs in the story-load log of
issue #123). They are now lowered as the RDNA2 ISA gives them (12.13, 10.4):
- `ds_append` (`ds_consume`): one atomic add (subtract) of the count of the lanes running at the
  LDS byte address M0 + offset, done by the first of them, and its old value returned to all of
  them. The ISA calls the address "M0.base"; for LDS, M0 is the byte address itself, as LLVM sets
  it (`AMDGPUDAGToDAGISel::SelectDSAppendConsume` copies the pointer to M0 unchanged).
- `ds_read_addtid_b32` (`ds_write_addtid_b32`): the dword at offset + M0[15:0] + 4 × the lane's id
  in the console's wave.
They make the program use the console's lanes (`console_lanes`). Where the host splits a console
wave (wave64 on 32-lane host waves, a group of more than 32 lanes) an append or consume is still
refused, "ds_append or ds_consume in a console wave the host splits": the console makes one atomic
for the whole wave. As for other LDS accesses, addresses are taken as aligned and in bounds (the
ISA clamps append and consume by M0's size). The `ds_gws` group (GDS) stays refused, by name.
`addr_tests::lds_append_consume_and_addtid_follow_m0_on_the_gpu` (`victory-gpu`) runs them on
Metal, wave32 and wave64.

Guess: M0 that a compute program reads before writing it is LDS address 0. Three of the title's
compute programs append to a counter with `ds_append` at offset 0 or 4 and never set M0; one zeroes
LDS bytes 0 to 7 at address 0 and reads the counter back at 0 + 4, so the base must be 0. These
were refused as "a hardware SGPR input".

## Legacy multiply-adds on RDNA2 (#321)

VOP3 0x140 and VOP2 0x006 were named and lowered as GFX10.1's `v_mad_legacy_f32` and `v_mac_legacy_f32`: the legacy product (a zero factor gives +0 against anything), rounded, then the add. On GFX10.3, which is RDNA2, the same encodings are `v_fma_legacy_f32` and `v_fmac_legacy_f32`, one rounding: `llvm-mc -mcpu=gfx1030` assembles `v_fma_legacy_f32` to the bytes gfx1010 gives `v_mad_legacy_f32` and refuses the latter, and gfx1010 and gfx1013 do the opposite (`GFX10_3Insts` brings `FeatureFmaLegacy32Insts`). They are now lowered as DXIL's `fmad` of both factors made 0 where either is 0, which Metal fuses: `legacy_fma_tests` reads (1 + 2^-12)^2 - 1 as 0x3a000400, as `v_fma_f32` gives, where rounding the product first gave 0x3a000000.

## Scalar absolute difference

`s_absdiff_i32` was decoded but refused ("s_absdiff_i32 is not supported", one compute program in
the story-load log of issue #123). It is lowered as the RDNA2 ISA gives it (SOP2): the 32-bit
difference `S0 - S1`, wrapping, negated when negative as a signed number, and SCC set when the
result is not 0. `addr_tests::absdiff_is_the_magnitude_of_the_signed_difference_on_the_gpu`
(`victory-gpu`) runs it on Metal, wrapping cases included.

## 16-bit reads of inputs not marked 16-bit

A pixel shader that interpolates an attribute with `v_interp_p1ll_f16`/`v_interp_p2_f16`, or
reads its high half, when the stage data does not mark that input 16-bit (input flag bit 3) was
refused as "a 16-bit read of a 32-bit input" (three guessed pipelines in the story-load log of
issue #123). The refusal now says which, after a colon: "an f16 interpolation of an input not
marked 16-bit" or "the high half of an input not marked 16-bit". Nothing is lowered: what such a
read returns from a 32-bit attribute is not documented, and in a guessed pipeline the missing mark
may be the stage data's (guessed) rather than the program's.

### 16-bit parameters of guessed pipelines

The G-buffer passes' pixel programs interpolate attributes with `v_interp_p1ll_f16`/`v_interp_p2_f16`, low and `high` halves, while their mesh programs pack those parameters themselves (`v_fma_mixlo_f16`/`v_fma_mixhi_f16`, but also a packed dword moved by `v_cndmask_b32`). The two stages of a guessed pipeline are translated apart, and neither says which parameter dwords hold two halves, so they share a convention instead (`emit::graphics::HALF`; a guess of ours, not the console's layout): a guessed vertex or mesh program exports each parameter channel three times, as itself (`TEXCOORD<n>`) and as its low and high halves each widened to an f32 (`HALF<2n>` and `HALF<2n+1>`, rows after `SV_Position`), all interpolated. A guessed pixel program's f16 interpolation of attribute n reads the half it names from the matching `HALF` element and rounds it to f16; its 32-bit reads stay on `TEXCOORD<n>`. Interpolating each half as its own f32 is what interpolating that half computes, so this is right whatever the vertex side packed; it costs two more interpolated rows per parameter. Halves go only to parameters that may hold them (`packed_params`, a guess), judged by the sources of their channels' values, followed back through moves, `v_cndmask_b32` and joins: 32-bit operations (`*_f32` but 16-bit conversions, `*_u32`, `*_i32`) and literals with a low half of 0 are plain; packing and 16-bit instructions (`v_fma_mix*`, `v_cvt_pk*`, `v_cvt_f16_*`, `v_pack_b32_f16`, `v_pk_*`, `*_f16`) and literals that are no normal f32 (0x3c00, 0xffffbc00: halves) hold halves; loads, SGPRs, other literals and instructions are unknown; inline constants and program inputs count as no source (a mesh program's exports read, through joins, what lanes that export nothing on a path hold). A write to one word of a parameter's dword that keeps the other (`sdwa.dst_sel` with `dst_unused:UNUSED_PRESERVE`) is followed through the value it keeps: the console's vertex programs finish a packed half by zeroing the word a lane does not export (`v_cvt_f16_f32 v3, v9`, then `v_mov_b32 v3, 0 dst_sel:WORD_1`), and read by that last write alone the parameter looked plain, so its `HALF` rows were left out and the pixel program's `user(half10)` linked to nothing (one pipeline, the world's particle pass, 27 November). A parameter of plain sources gets no halves. The others get them in order: those of only 16-bit sources first, then those with some, then the unknown ones; the dwords (`TEXCOORD`) of the first kind come last, if there is room (a pixel program may read them as 32 bits, flat). The order matters because the title's G-buffer mesh programs export six to eight parameters, only one or two of which are packed (`v_fma_mixhi_f16` for .xyz and a `v_cndmask_b32` of 0x3c00 or 0xffffbc00 for .w), and the rest come from loads and float arithmetic. And only as many as fit: 31 elements for a vertex program (Metal's 124 scalar fragment inputs), and for a mesh program 32 KiB of outputs at 16 bytes an element per vertex (the converter's mesh size limit: 16 elements for the title's 128-vertex groups, measured by refusals of 3k + 1 elements at 2,048 bytes each). A pixel program reading a half the vertex side left out does not link. The first run with halves for every parameter had 36 mesh pipelines refused for their size; with halves only within the limit, 24 G-buffer pipelines still lacked the halves of parameter 5, 6 or 7 their pixel program reads (all six to eight parameters judged unknown, with halves for the first four). With the order above, a check of every guessed pipeline pair of a run (167, the containers dumped to a scratch folder with `VICTORY_DUMP_ALL_SHADERS=1 VICTORY_DUMP_REFUSED=<dir>` and their signatures inferred offline) found every pixel input among its vertex side's outputs, the 24 G-buffer pairs and 4 earlier `user(texcoord<n>)` refusals included. Both stages declare every `TEXCOORD` and `HALF` element four channels wide, whatever channels the code exports or reads: Metal links a fragment input only to an output of the same type, and the converter refused pairs whose pixel side read fewer channels than the vertex side exported (24 G-buffer pipelines, `user(half<n>)` "mismatching ... type(s)", in the second run; the pre-existing `user(texcoord<n>)` refusals are the same). Channels an export leaves out are written a copy of one it has, unread: the converter crashes (in its linker) on a mesh output whose channels are partly unwritten, or written a constant (found with hand-assembled pairs). shortcut: a parameter exported twice with different channels keeps only the last export's. A raw dword interpolated as bits would be wrong, and a flat one would lose the interpolation. Not covered: a 32-bit flat read of a packed dword keeps reading `TEXCOORD<n>` (exact bits only if the interpolation keeps them). Tests: `emit::graphics::tests::halves_go_to_parameters_that_may_hold_them`, `emit::graphics::tests::a_half_kept_below_a_zeroed_word_still_counts`, `victory-gpu` `mesh_tests::halves_of_many_parameters_stay_within_the_mesh_size`, `mesh_tests::halves_of_some_channels_link`, `mesh_tests::a_parameter_of_halves_only_reaches_the_pixel_program` and `mesh_tests::packed_half_parameters_reach_the_pixel_program` (a hand-assembled mesh program exports two packed dwords, the pixel program reads all four halves; the colours are checked on Metal).

`HALF` and `TEXCOORD` elements have different names but overlapping indices: `HALF<2n>` is param n's low half, not `TEXCOORD<2n>`. An extra `TEXCOORD` is added beside a `HALF` of its index (`emit::mod`, by name). Its values are copied from the original parameter (`Options::copy_params`), and that copy took `HALF` elements of the original index as sources too: it then wrote the original's `HALF` over the `HALF` of the new index, so the pixel shader read another parameter's half whenever that export came last. The copy now takes `TEXCOORD`s only (`Lower`'s copies; `heap_tests::an_extra_parameter_beside_a_half_of_its_index_links_with_its_own_values` checks the values on Metal).

A mesh program's outputs take 16 bytes an element for each vertex, in at most 32 KiB, so at 128 vertices the halves above fit only for some parameters (16 elements, the position included). GTA VI's G-buffer mesh programs export nine parameters and the halves of three; their pixel shaders read the halves of the ninth too (`user(half16)`, `user(half17)`). Metal's compiler crashed on every such pipeline (`XPC_ERROR_CONNECTION_INTERRUPTED`; the compiler service's report: `AGCLLVMUserMeshShaderSW::getBasePointer` in `MeshShaderSWAuxiliaryShadersPass`, a fragment input the mesh function has no output for). Both stages' AIR is valid (`metal-opt -passes=verify`); a dump of three crashing pipelines (28 November) showed the same two missing inputs in each. Now when a guessed pipeline's pixel shader reads a half its vertex or mesh shader does not export, the vertex or mesh shader is translated with the parameter elements that pixel shader reads and no others (`Options::pixel_reads`, from `state::halves` and `state::texcoords` in `glue::guess`): the missing halves are added, and the parameters it does not read (here the 32-bit copies of the four parameters of halves) are left out, which makes room. Exports to an element left out are not stored. Test: `emit::graphics::tests::a_guessed_program_exports_the_parameter_elements_its_pixel_shader_reads`.

## Bring-up diagnostics (native path)

Off by default, environment variables of the host process. The native host's switches that draws and submissions check are read once, at their first check (`env_set!`, and `VICTORY_TRACE_CONTENT`'s options), so setting one after the process started has no effect:

- `VICTORY_COUNT_GREEN=1`: each presented frame has nine pixels copied out after it is drawn (`victory_metal::present`); a frame whose pixels are mostly pure green (zero YCbCr planes, a movie frame sampled blank) is counted. Once a second stderr gets `present pacing t=<s> presents=<n> longest_ms=<gap> green=<n> partly_green=<n> green_total=<n>`: presents in that second, the longest gap between two, and green frames. Costs one small blit a frame.
- `VICTORY_TRACE_FAILS=1`: every title-to-host call of a bridged method table (the COM-style objects) whose result in `eax` is negative is logged with the object area, the slot offset, `this` and the title's calling address as an offset into the image (`from`). A method returning a pointer or a count shows up as noise. `=2` logs every such call.
- `VICTORY_PROFILE=<secs>`: times every title-to-host call (imports and the methods of bridged objects: count, total and longest, by `rdtsc` in the bridge) and logs the 40 with the longest total every that many seconds. Time inside a call includes waiting for the host's locks and any title code it calls back, so methods that are long everywhere (all of `0x118`, `0x428`, `0xc0` at about 150 ms) point at a lock someone else holds (here the native host's `TITLE` lock, held by `sync_textures`).
- `VICTORY_TRACE_COMMITS=1` (with `VICTORY_TRACE_GPU_FAILS=1`): on macOS, a line for each command buffer as it is committed (`cb commit #<serial> q<queue> t=<s>`, then its encoders: compute pipelines by shader hash with their group counts, render passes with their draw count and each pipeline's vertex and pixel hashes with its draws, blits with their fills) and one as it completes (`cb done #<serial>`, GPU time, failed or not). With `VICTORY_PASSES_PER_BUFFER=1` the command buffer the GPU never finishes, and so the pass, is the one committed and never done (#79).
- `VICTORY_TRACE_RESIDENCY=1`: times the residency-set calls (`add_commit`, `remove_commit`, `request`, `end`, `made`, `drop_commit`, `pool_commit`) and command buffer commits (`cb_commit`): each call over 50 ms is logged at once ("residency: slow call"), and once a second a line gives count, total and longest per kind, with the bytes and allocations of the last set seen. Lines the host's residency work up with wired memory samples (#79).
- `VICTORY_TRACE_HEAP_SET=<frames>`: every 64 frames, with the native-draws line, what the descriptor heaps' residency sets hold: textures and buffers apart, each allocation counted once over all heaps: all of them, those no table a draw or dispatch bound has named for that many frames (`idle_*`; each entry's last use is noted where tables are remirrored, `NativeHeap::used`), and those never named by one (`never_*`). Entries a shader reaches by indexing the heap itself are never seen, so `idle` and `never` are upper bounds. Measures what an eviction of idle allocations from the sets could save (#79).
- `VICTORY_PASSES_PER_BUFFER=<n>`: on macOS, the most passes of a submission in one command buffer (default 0: no limit, the whole submission in one; 1 is the safest value to try). See "Command buffers of at most N passes".
- `VICTORY_TRACE_GPU_FAILS=1`: on macOS, command buffers are made with encoder execution status, labelled (`q<queue> #<serial> t=<secs> waits signals passes`) and so are their encoders (compute: pipeline name and group counts per dispatch; render: draw count; blit: op count), and a command buffer the GPU fails is logged with `tracing::warn!` (timestamped, up to 200) with each encoder's label and state (Completed, Pending, Faulted, Affected). Without it the first 20 failures are an untimestamped `eprintln!`. It also logs each command buffer that took over 1 s to complete (`metal: slow command buffer`, with GPU and kernel times) and, at each submission, each one still in flight after 3 s (`metal: command buffer in flight`, once each; both capped at 100 lines). While work stays held (the `work held for a wait that does not hold` warning, when nothing is submitted any more) the same in-flight report is made, with a line giving how many command buffers are in flight, each queue whose order event is behind the command buffers encoded with it (`queue <title queue> at <reached> of <encoded>`), and the commands dropped so far at unmapped memory. Each piece of a split submission (`VICTORY_PASSES_PER_BUFFER`) is traced as a whole command buffer is, under its own passes: `metal: slow piece` or `metal: failed piece` with its time, and it is in the in-flight report (a time-out carries no encoder states, so the labels are what names the pass). An indirect dispatch's label shows its record as title memory held it at encode time (`indirect[x, y, z]`), and every indirect draw goes through the gate so that the lines above can end with what the GPU read (`gpu read <compute shader, or "draw k" / "mesh draw k" by place in the pass> [counts]`, zeros for a gate that has not run yet); `metal: an indirect dispatch past the most groups ran nothing` names the first four records of each pipeline that asked for more than the gate runs (see "ExecuteIndirect on the native host").
- `VICTORY_TRACE_WRITES=1`: on macOS, a warning for every bottom-of-pipe value write a list records (value, address, whether a pool holds it, the list), every memory-value wait (address, comparison, reference, the list) and every `ExecuteCommandLists` (queue and lists): which list of which queue owes the word another queue waits for. About 100 lines a world frame. Without it, a value write the host has to drop (its address in no mapped pool, or not preparable) is still a warning, once per page (`native host drops a value write`): a queue may wait for that word for good.
- `VICTORY_TRACE_FREED=1`: names host and GPU uses of title memory the title's allocator has freed. `XMemAllocDefault` notes every range it hands out, `XMemFreeDefault` the ones it takes back (the block header holds the size), and `Host::resolve` — the way every host and GPU use of title memory begins — logs any range it is asked to map that overlaps a freed span, with both ranges and a serial. Part-reuse leaves the freed ends beside it. What it cannot see: the title can hook the allocator, and pools it sub-allocates inside a live block free and reuse memory without any call here; this names the `XMem`-granularity class only (#339: a host copy or a GPU write into freed memory, one candidate for the corrupted pool links).
- `VICTORY_TRACE_TEXTURE_SYNC=1`: on macOS, once a second, what keeping the host's textures in step with title memory cost (`texture sync in the last ..`): the submissions that read textures (batches), the textures and MiB they read (those never uploaded before, those read and found unchanged, the blank ones, those with memory that could not be read at once), the time under the host's lock (probing the hot textures, with how many of them went to be read whole; probing the blank ones; uploading), the time the submitting thread spent reading without the lock and the reading threads' time by step (read, hash, detile), and the placements that replaced another at their address (with how many of those had been at that address before). See "Texture sync without the host's lock". It also logs the texture budget once a second (`host texture copies: ..`; "A byte budget on host texture copies" below), whose size `VICTORY_METAL_TEXTURE_BUDGET_MB=<n>` sets (off unless set). The figures are logged with the budget off too.
- `VICTORY_TRACE_RECORD_WRITERS=1`: for indirect mesh draws that leave no pixels. At each present with indirect mesh executions it logs `mesh records` (the frame's records, and how many had a group count of 0 in title memory as the execution was recorded; a culling pass that ran fills them on the GPU later, one that was dropped never does), `mesh record buffer holder` for each compute program, `ran` or `dropped`, whose descriptor tables hold a record buffer for unordered access (with the records in those buffers and the textures it reads), and `texture read by a mesh record buffer holder` with the compute programs that hold that texture for unordered access. Tables are read 64 entries deep whatever their length, so a holder is a candidate writer, not a proven one; the texture keys are those of the `VICTORY_TRACE_TARGETS` frame line. With compute shader hashes for a value (`VICTORY_TRACE_RECORD_WRITERS=<hash>[,<hash>...]`) the records watched are those of these shaders' indirect dispatches instead of the mesh records: the holders are then the candidate writers of their group counts.
- `VICTORY_TRACE_PASSES=<dir>,<skip>,<presents>`: on macOS, an ordered trace of a few world frames, to find the first pass that writes nothing (`dive-lighting` research). It lets `<skip>` world presents go by (a present after 500 or more recorded draws; default 100), then for the next `<presents>` presents (default 4) writes to stderr, in the order each submission queues the work: `PASS exec=<submission> q=<queue> li=<list> ci=<command> kind=<draw|mesh|dispatch[..]|clear-rtv|copy-region|...> out=<"ran" or the drop reason> sh=[shaders] tg=[targets] rd=[shader-resource views] wr=[unordered-access views] none=<empty table entries> topo=<topology>` for every draw, dispatch, clear and copy recorded in the window; `PASSFAIL exec= li= ci= err=` for each of those the submission could not prepare (dropped then, e.g. "not in a mapped pool"); `PASSSUB exec= li= ci= changed=[entries]` when a noted draw's or dispatch's table entries read differently at submission than when it was recorded (the title rewrote them in between; D3D12 reads them at execution); `PASSUPLOAD` for each upload of title memory over a render-target or unordered-access texture; and, at the window's last present, `PASSTEX key= fmt= WxH at= nonzero=` (plus range, mean and non-finite counts for DXGI 2, 10, 24, 26) for every texture the window named, with a `.ppm` of each in `<dir>` (only formats the dump decodes; arrays skipped). A texture is written `<key>:f<DXGI format>:<W>x<H>@<title address>`, a shader-resource view of a buffer the host knows `<key>:buf`, of a null resource `null:f<format>:dim<view dimension>` (from its view description; it reads as zeros), and of anything else without a host texture `<key>:untracked`; a view of placed memory that is a buffer `placed:buf`, and of any other dimension that is not a texture `placed-dim<dimension>`; tables are read to the entries the shader's binding names (24 for a dropped draw or a dispatch, and never past the heap). Set `VICTORY_TRACE_PASSES_DUMP_EACH=1` to dump at every present of the window. The listing also writes a constant-buffer view as `cbv:<address>+<size>:<dwords>` and a placed buffer view as `placed:buf:<address>+<size>:desc[<first ten description words>]:<dwords>` (the first 512 bytes, in hex), and `<slot>.past-heap=<n>@<index>/<heap size>` when a table names entries past its heap; `PASSTEX` reads a 3D texture whole and adds `depth=`, `slices_nonzero=` and `distinct_texels=`, and for a DXGI 2 texture of 16 texels or fewer, `values=[...]`. Constant buffers and buffer contents are game data too. Off unless set: the switch is read once, at the first present; outside the window every hook is one relaxed load of a flag and nothing is formatted or kept, and the commands are prepared once either way (a failure is reported from the submission's own preparation). Images hold game content: keep `<dir>` outside the repository. Runs that end with exit code 134 a minute or two after the window were seen with and without the trace on the same build (the dive's runs and the untraced c2comb run), so that abort is not the trace's.
- `VICTORY_TRACE_DISPATCH_FLAGS=1`: logs the first `DispatchX` with each flags value (`DispatchX: the first dispatch with these flags`, with its counts and the bits the host does not know: it knows 0x20 and 0x8000, see "`DispatchX` with thread counts"), and the first `ExecuteIndirect` of a type-105 dispatch, whose counts run as group counts; and the first three indirect dispatches of each pipeline (`an indirect dispatch`: its shader, the record's title address and the three words title memory holds there as the call is recorded, which is before the GPU writes them).
- `VICTORY_TRACE_IMPORTS=1`: the same for every import bound to a host function (name and DLL), whatever the result. Slow (millions of lines in a title's startup).
- `VICTORY_KEYS=<seconds>:<key>,...`: on macOS, scripted input next to the live keyboard (below; the two are OR-ed): each key is held for 0.2 s that long after `GameInputCreate`, and drives the stand-in pad (`Enter` is A, `Esc` is B, `Tab` is Menu; letters, digits, `Tab`, `Back` and the arrows as in the README). **Live keys (windowed run):** `window.rs` takes key-down and key-up events (macOS key codes, not characters; Cmd shortcuts are left to AppKit) and calls `gameinput::set_live_key` with the Windows virtual key; the macOS `Source::keys` returns them with the scripted ones, so the stand-in pad (`FALLBACK_BUTTONS`, shared with Windows) sees both. Held while down, released on key-up or when the window loses key status. Enter or Space is A, Esc or Backspace B, X, Y, arrows the D-pad, W/A/S/D the left stick (I/J/K/L the right, F/H the stick clicks), Tab Menu, V View, Q/E the shoulders, Z/C the triggers. **Controllers (windowed run):** while the window has key status, each loop turn `window.rs` reads one `GCController`: `GCController.current` (the one used last) when it has an `extendedGamepad`, otherwise the first connected one that has. The Xbox layout comes from the extended gamepad's elements, so a DualShock 4 (`GCDualShockGamepad`) and a DualSense (`GCDualSenseGamepad`) are read like an Xbox pad: Cross/Circle/Square/Triangle are A/B/X/Y, Options (`buttonMenu`) is Menu, Share (`buttonOptions`) is View. Those two profiles also have `touchpadButton` (found with `respondsToSelector:`), and its click is View too: our choice, as PlayStation titles use the touchpad for back and map. The PS button (`buttonHome`) is not mapped: GameInput has no Guide bit for titles, and macOS keeps it for its own overlay. `gameinput::set_live_pad` OR-s the result with the keyboard's stand-in pad. Each change of controller is logged once at info level (`controller: drives the pad`, with `vendor`, `category` such as `DualShock 4`, and `touchpad`). The window tests read a software gamepad (`+[GCController controllerWithExtendedGamepad]`, whose elements take `setValue:`), so the read path is checked without hardware. A headless run needs it to get past the title's engagement screen (below); `VICTORY_KEYS=$(seq 4 2 80 | sed 's/$/:Enter/' | paste -sd, -)` presses Enter every 2 s.
- `VICTORY_HOLD_BYTE=<hex address>:<hex byte>`: keeps one byte of title memory at that value, rewritten every 250 ms. Off unless set. For the build we test, `1471b6330:1` holds the pause flag of the title's init watchdog: when init takes over 30 s (game files on a network share or a slow disk), the watchdog otherwise ends the run before the start screen. The address is for that build only.
- `VICTORY_READ_CACHE=<dir>:<prefix>`: on macOS, keeps every 64 KiB block that DirectStorage reads from a file under `<prefix>` (checked after resolving links) in `<dir>`, as `<name>-<length>-<mtime>.data` (sparse) plus a `.map` with one byte per block, so a patched file gets a fresh entry, and a later run reads those blocks from local disk. For game files on a slow network share, where the first boot can spend minutes on reads. A missing block is fetched together with the missing blocks after it that the same read covers (up to 64). A run adds at most 12 GB; past that, reads go straight to the source. Each fetched run of blocks is synced before its map bytes are written, so a crash can lose blocks but never leave one marked present with no data. Nothing is evicted: delete `<dir>` to reclaim the space.
- `VICTORY_DUMP_THREADS=0`: the thread dump (and `VICTORY_DUMP_MEM`, `VICTORY_FIND_QWORD`) is taken once, instead of after a time: when the title first hits an `int3` (its asserts, at the first chance), or at its first access violation that no handler takes (its fatal-error path writes to address 0), just before the unhandled-exception filter. An access violation that a title handler takes does not use the dump up. A thread of its own writes the dump while the faulting thread waits for it, for at most 60 s (a long `VICTORY_FIND_QWORD` scan may be cut short by a fatal fault). The faulting thread shows in it as in host code; its registers are in the first-chance log line. Each thread is stopped only while its registers and up to 1 MiB of its stack are copied, and is unwound from the copy once it runs again, so a thread stopped inside a lock (the unwind tables', the allocator's) cannot hang the dump; this holds for the timed dump too.
- `VICTORY_FIND_QWORD=<hex>,<hex>,...` (with `VICTORY_DUMP_THREADS`): logs every aligned qword of resident readable memory equal to one of the values, with the 0x100 bytes from it; `@<hex>` stands for the qword at that address. Finds heap objects by their method table (data only).
- `VICTORY_PEEK=rva,rva,...`: at the first-chance exception log, prints those qwords of the faulting image (an operand above `0x10000000` is an absolute address), the qwords `rcx` points at, and a call chain by the unwind tables. Use it with `llvm-objdump` on the title (analysis only).
- `VICTORY_TRACE_OWNERS=1`: who made the memory that comes to share pages (#15), logged as a warning (`VICTORY_TRACE_OWNERS`) with both owners when it happens: a new physical page view over another view (`xmem_physical::record`; the mapping paths unmap first, so it should not happen), page numbers past the 25 bits Grand Theft Auto VI keeps (`number >> 4` in 21 bits) with the page whose number it would rebuild, `VirtualFree(MEM_DECOMMIT)` of a range that still shows views (`MEM_RELEASE` gives them back first, `forget_physical`), and a title mapping (`XMemMapPhysicalPages`, its scatter form, the GPU's `map_alias`) outside every region `VirtualAlloc` or `XMemVirtualAlloc` made (macOS only). Each owner is the public call that made it (the outermost one on that thread), its range, pool id and page index, a sequence number and the title's return address from the host call (`image+offset`; on other threads 0 or a stale one). Off, nothing is recorded: the cost is one read of the switch per view, allocation and free.
- `VICTORY_DEBUG_PS=nanid`: a translated pixel program's export (one output's channels) with a channel that is NaN or infinite writes a marker instead: red, green and blue `-(256 + byte)` of the program's `id` colour (below), alpha -1024; finite exports are unchanged. The pass trace's texture dumps of float targets then add `nanids=[("rrggbb", texels), ...]` (the programs whose markers a target holds, most first) and `nanmixed=` (texels too negative to be one marker, such as two blended). Made for the world's oversized coloured glows, which come from NaNs in the HDR buffers spread by the bloom blur. `VICTORY_DEBUG_PS=nanin` puts the same marker down where any float input the program loaded (an interpolated vertex output) is NaN or infinite instead, whatever it exports, to tell bad inputs from bad arithmetic. Where a lane writes a marker, its program's other exports write 0, so that a blend reading a second source (dual-source: colour + source 1 x target) leaves the marker readable. Markers are negative, so a target of an unsigned float format (R11G11B10) cannot hold them: there they clear the NaN instead, and read the signed targets before them. `VICTORY_DEBUG_PS=nanpc:<rrggbb>[:<from>-<to>]` marks every program as `nanid` does, and in the one program of that `id` colour says where the value went bad: per lane, the byte offset of the last float VALU result that came out NaN or infinite from finite vector operands (marker bytes `01`, offset high, offset low), from finite vector operands but a non-finite scalar one, a constant (`03`...), or of the last image or buffer load that returned one (`02`...); `000000` when none did. With `<from>-<to>` (hex byte offsets) only the instructions in that range are watched. With `:math` in place of a range, the whole program is watched but only for loads and the operations that make a NaN or infinity of finite operands other than by overflow (reciprocals, roots, logarithms, exponentials, sines, divide helpers and interpolation), which keeps a large program small enough to load; a multiply of 0 by an infinity then shows as the operation that made the infinity. A trap temporary carries the record (`ir::NAN_LAST`, which pixel programs never use) through the program's control flow, so the program grows: GTA VI's 31 KB G-buffer program went from 147 KB of DXIL to about three times that watched whole (runs of it did not get past the loading screen's stuck-buffer guard), 203 KB watched over its last quarter. `VICTORY_DEBUG_PS=nanborn` marks only the NaNs and infinities a program makes: in every program (compute too), the record watches loads alone (image, buffer, LDS and global loads, not arithmetic, which is too much to watch everywhere), and an export or image store with a channel that is not finite writes the program's marker (as `nanid`) if no load before it returned one, and zeros if one did (the value is taken to be passed on). Passed-on values then stop spreading, and the frame's markers name the programs that made them: the earliest in the frame is the source, as a later program can make one from an earlier marker. A load of integer data with all exponent bits set (-1) counts as a bad load too. `VICTORY_DEBUG_PS=nanzero` writes 0 in place of every NaN or infinity an export or image store of any program would write, with no marker: whether a frame looks right once none of them spread (through history buffers from frame to frame, and the bloom's blur). A translation switch like `id`.
- `VICTORY_DEBUG_PS=id`: every float export of every translated pixel program writes one colour, the low three bytes (red, green, blue) of the 32-bit FNV-1a hash of its console code (a container's `ACCC` code piece, as the `disasm` example takes it; alpha 1), so a target dump (`VICTORY_TRACE_PASSES`, `VICTORY_DUMP_TARGETS`) says which pixel program drew each pixel, and through the pass trace which draw and vertex program. Read a UNORM target (an sRGB one stores the encoded value). A translation switch (`VICTORY_DEBUG_PS` is in the cache key). Made for the world's G-buffer shards ("World geometry drawn as shards (fixed)"): the 40 pixel programs of the traced G-buffer draws have 40 different colours.
- `VICTORY_DEBUG_MESH=<rrggbb>[,<rrggbb>...]` (or `all`, every mesh program) with `VICTORY_DEBUG_MESH_AT=<present>[:<file>]`: on the native host, the mesh programs named by their debug id colour (the `id` colour above, of a mesh program's code) record what they export, to tell wrong vertex positions from wrong triangles. Each lane that exports a position (`exp pos0`) or a primitive (`exp prim`) claims a 32-byte record with an atomic add and writes it through the page table to title address `0xff_0000_0000` (`emit::MESH_CAPTURE`), which the host maps for the capture (256 MiB, about 8 million records; nothing is captured if the title maps that address itself). Record dwords: 0 the kind (1 position, 2 primitive) with the lane's thread in the group from bit 8, 1 the group, 2 a hash of the user data the program reads (one value per draw), 3 the program's 32-bit FNV-1a code hash, 4-7 the exported dwords (x, y, z, w; a primitive's packed indices in dword 4). The capture holds still until present `<present>` (counting `PresentX` calls from the start), then fills; once its records are full, or some are claimed and four presents have gone by, the claimed count (dword 0) and the records are written to `<file>` (default `mesh-capture.bin` in the working directory) and nothing more is captured that run. `VICTORY_DEBUG_MESH` is a translation switch (in the cache key); `VICTORY_DEBUG_MESH_AT` is not. Exported positions are game data: keep the file outside the repository.
- `VICTORY_TRACE_TABLE_HOLES`: at each rebuild of the page table ("Title memory at computed addresses", `native::addr`) whose count changed, logs how many 64 KiB title pages hold mapped bytes but no one mapping wholly (`addr::split_pages`: a mapping that starts or ends inside the page, or several 16 KiB mappings making it up), and how many of their bytes are mapped: the pages that read as zeros through the table though the title mapped them. Made for the world's G-buffer shards. Off, the cost is one read of the switch per rebuild.
- `VICTORY_TRACE_ROOT_SPAN[=<bytes>]`: root SRVs and UAVs whose title range stops being linear in its pool within `<bytes>` (default 1 MiB) of the root address (#15). A shader that does not read title memory reaches a root view through a view pair (`addr::root_view`): the pool from the mapping holding the address on, linearly, so past that mapping's end it reads and writes whatever the pool has next, not the title's next mapping. At each draw and dispatch a translated shader's root views are measured (`addr::linear_run`: following mappings that start where the last ended, in the same pool, at the next pool offset), and each one whose run is shorter is logged once per pipeline, root parameter and mapping (`a root view's range leaves its pool's linear run`, at most 2000): the pipeline's shader hashes, SRV or UAV, the dispatch's groups, the run, the first mapping, the mapping the title has where the run ends (`title_next`) and the title address of the pool bytes a pair view reaches there (`pool_next_is`). Root views of programs that read title memory (`through = "table"`) look each access up in the page table; they are logged too. Heap buffer views need no check: a range that is not linear in one pool wraps the title's own mapping instead (see "Heap buffer views not linear in one pool"). Off, the cost is one read of the switch per draw and dispatch. Run of 2026-11-26 (M5 Pro, Series S, crash at frame 2682): the 117 lines are all vertex shaders' root SRVs (root parameter 3, vertex fetch), and 30 of them do have a next title page elsewhere in pool 10 (pages mapped in descending pool order); no dispatch and no root UAV has a run under 1 MiB, so root views do not write the empty BVH nodes over other resources. Those vertex shaders read the wrong vertices past the first page, though: vertex and pixel programs now read root SRVs and UAVs through the page table (`root_memory` in `emit::lower`, with `zero_copy_memory`), one lookup per access, so they log as `through = "table"`; compute programs keep the views (no short run seen there). Tested by `a_vertex_root_view_follows_the_title_across_pages_apart_in_the_pool` (`addr_tests.rs`): a triangle whose first vertex ends one 64 KiB page and whose others start the next one, mapped elsewhere in the pool, was not drawn. Limit: the table has 64 KiB pages, so a page not wholly inside one mapping (mappings of 16 KiB) reads as zeros through it, where a view read it.
- `VICTORY_TRACE_DESCRIPTORS=1`: how many descriptor records the title rewrites with plain stores, which no device call reports (#43). `NativeHost::remirror_tables` compares each bound table's records with those it last mirrored at every draw and dispatch; with the switch on, the frame log (`native draws so far`, every 64 frames) is followed by a `descriptor remirror` line with the records changed and checked since the previous one, per frame, per heap (by its first descriptor address, most changed first), and the most changed in one submission (`NativeHost::execute`). It counts only what that check sees: the first 16 entries of each bound table (more when the shader's metadata says so), at the draws that bind them; it does not say which code wrote the records. Off, the cost is one read of the switch per table and per submission. `VICTORY_TRACE_DESCRIPTORS=sites` also finds which title instructions store the records, for the choice between patching those store sites and remirroring: at each `ExecuteCommandLists` the pages of every descriptor heap's records (the title's own view, not the GPU's) are made read-only (`victory_platform::watch`). The loader's signal handler hands every write fault to the watch first, whoever wrote: a fault in a watched page makes that page writable again and the store runs again. It counts as a site (the instruction as an image offset) only when the instruction lies in the title image and the address in a heap's records, so the device's own view calls and title stores that only share the page go on unseen. The handler takes no lock and allocates nothing: a fixed table of 4096 sites (overflow counted), and the watched ranges (64 at most) in two tables behind sequence counts, so that a store faulting while a submission replaces the ranges is still the watch's: the handler reads the table in use and the one before it, and pages are opened only once neither holds them. Opened pages become read-write, which descriptor records' pages always are (the device's view calls write them). The frame log then adds a `descriptor store sites` line: the distinct sites so far and the 64 with the most stores, most first. Limit: a page stays writable until the next submission, so what is counted is the first store into each page per submission, not every store; a site that only stores after another one opened the page is seen less often, or not at all. A kernel write into a watched page (a `read` into it) would fail with `EFAULT` instead of faulting. Off, the handler's cost is one relaxed load per fault.
- `VICTORY_WASAPI_WAV=<file>`: the audio device (`host_posix/wasapi.rs`) writes everything that reaches it as a float WAV — hearing exactly what the title's mixer produced, dry sound and (until #17 lands) silent reverb alike.
- `VICTORY_SHADER_LOOP_LIMIT=<n>`: a translated loop stops after this many iterations per entry, and a lane that reaches the limit leaves the program as killed (a pixel discards; a genuine loop of more iterations is cut short too). The default is 2^14 (a lane doing 2^20 passes took 60 to 195 ms on an Apple M5 GPU, where 2^14 take 1 to 3 ms). A lower limit shows whether such a loop is what makes a frame slow, a higher one whether the limit cut a genuine loop short. The value is part of the translation, so the shader cache keys on it.
- Diagnostics build only (`--features diagnostics`): `VICTORY_DUMP_REGISTRY=<dir>` writes the title's subsystem-registry state (the singleton, its tables and their raw memory) to files in that directory at the first title `int3`, so an assert that checks the registry can be read after the fact (#24), and `VICTORY_XMA_DUMP=<dir>` saves the context words (`<n>-<index>.ctx`) and input buffer 0 (`.xma`) of the first 32 XMA voices enabled, to test the decoder on a title's streams outside the title.

The first-chance log (`seh::dispatch`) shows a breakpoint's `rip` as the `int3` itself, as Windows x64 does.

Instructions Rosetta does not run are skipped or emulated by the SIGILL path in `seh::signal`: `clwb`/`clflushopt` (skipped; a whole `clwb [p]; add p, 0x40; cmp p, limit; jbe/jb` flush loop is skipped in one signal, since the title flushes buffers of many megabytes a line at a time), `clzero` (zeroes the line, or the whole clearing loop), `monitorx`/`mwaitx` (skipped, the latter yields), and the SSE4a `insertq`/`extrq` (run on the saved XMM registers; the title uses `insertq` once its world loads).

## Sweeping a shader corpus

A translator change can be checked against every shader a title registers without running the title again: dump the containers once, translate them all with the tree before the change and with the tree after it, and compare what came out. Counts alone hide a wrong accept (a shader that still translates, to something else), so a sweep keeps a fingerprint of each translation's output beside its outcome.

**The corpus is game data.** The dumped containers are the title's shaders: the folder lives outside the repo and is never committed, attached to an issue or pull request, or sent to any service. Sweep files and `--show` output describe those shaders and stay outside the repo too. What goes in an issue or pull request is counts, shader hashes and refusal messages.

1. **Dump your own corpus** (needs the game; once per title build). Run with `VICTORY_DUMP_ALL_SHADERS=1 VICTORY_DUMP_REFUSED=<folder>`, the folder made beforehand: each shader the host translates is written there as `<stage>_<hash>.bin` (`Shader::native`, `victory-gpu`'s `pipeline.rs`). The prewarm threads translate every shader as the title creates it, so a run leaves all the shaders registered up to where it got, bound or not.
2. **Sweep each build**, in that build's tree (a `git worktree` for the base and one for the change):

   ```bash
   cargo run --release -p victory-shader --example sweep -- <folder> -j 4 > <outside the repo>/base.tsv
   ```

   The example (`crates/victory-shader/examples/sweep.rs`) is built with the translator of the tree it sits in and translates each container with the native host's options (`native::shaders::options`: `wave32_host`, `zero_copy_memory`). It prints one line a shader, sorted by name: `<file> \t ok|refused|panic|timeout|missing \t <fingerprint or reason> \t <ms>`, and the counts on stderr. `--list FILE` sweeps only the files named in `FILE`, one a line (the shaders of one refusal take seconds; `compare.sh --names` makes such a list). `--timeout SECONDS` (default 120) is how long one shader may take: past it its line says `timeout` and the sweep goes on without it. `--show` prints the host-side description (the `HostShader` but its DXIL) in place of the fingerprint.
3. **Compare.** `tools/corpus/compare.sh base.tsv` tallies one sweep: outcomes per stage, and the refusal reasons by count (code offsets and the numbers of a range taken out, so one reason is one line). `tools/corpus/compare.sh base.tsv new.tsv` says what changed, in four groups: **newly translated**; **REGRESSIONS** (translated before; refused, panicking or timing out now); **output changed** (translated before and after, with another fingerprint); **other reason now**. `--world FILE` takes the names of the shaders a reference run bound (`<stage>_<hash>.bin`, one a line) and adds how many of each count are among them: a change that helps those shows in the game at once. `--names TEXT` lists the files whose reason contains `TEXT`, `--show N` sets how many names a group lists, and `--self-test` runs the script on made-up sweeps (CI runs it).

Reading a comparison: no regression is the bar, and every "output changed" shader needs an explanation. It is either the point of the change or a side effect to understand before the change is kept.

The fingerprint is a hash (std's `DefaultHasher`) of the DXIL and of the `Debug` text of the rest of the `HostShader`. Two sweeps of one build are identical but for the times. Fingerprints compare only between builds of one toolchain with the same `HostShader` fields: a change that adds a field changes every fingerprint, and the comparison then reports every translated shader as "output changed". Give each tree its own target directory (cargo's default does that; one `CARGO_TARGET_DIR` for both does not): cargo decides what to rebuild from file times, and with one directory for two trees the second sweep can run the first tree's translator without saying so.

Not swept: the host's other translations of a container (`thread_counts` dispatches, rectangle lists, vertex shaders widened with `extra_params`), and everything after translation (the Metal Shader Converter and pipeline creation; `victory-metal --corpus` in docs/metal-backend.md covers those for a shader cache).

## Lane masks joined with data

A lane mask joined at a block entry with a register that holds data on one edge (the BVH walks
pop masks from memory) is, in a lane, that lane's bit of the number the edge brings: bit `i` of a
mask is lane `i` (RDNA2 ISA, EXEC and VCC). It is taken only where the wave takes one edge (a
uniform join) and with the console's lanes. Wave64 programs now take it too, the low dword for
lanes below 32 and the high dword above, from the high register's join on the same edge; they
were refused as "a lane-mask join of an untracked mask". What stays refused says why, after that
text and a colon: "data, on the host's lanes", "data, at a join of several edges", "data whose
high dword is not known", "a value the translation could not make", "an SCC or lane bit" or "no
value on that edge". `addr_tests::a_wave64_mask_joined_with_data_takes_each_lanes_bit_on_the_gpu`
(`victory-gpu`) runs such a loop on Metal with the host's options (a console wave as two host
waves): lanes 32 to 35 take their bit from the high dword. The story-load log of issue #110
(Series S) has three compute programs refused with the old message; whether they are this wave64
form is not known.
One kind of data is a lane bit on the host's lanes too: a constant of every lane or none
(`s_mov_b64 s, -1` or `0`, both dwords in wave64), whose bit is the same in every lane
(`Lower::all_or_none`). The G-buffer's pixel programs carry such a mask round a loop. It is
SCC's all-or-nothing (`s_cselect_b64 s, -1, 0`) on entry and -1 on the way round, which the
structure cannot join because it is no function of fixed lane bits. Of the 9,995 dumped pixel
programs, 338 more translate (7,172 to 7,510), and none regress. Test:
`emit::gfx_refusal_tests::a_pixel_program_joins_a_loop_mask_of_all_lanes_or_none_on_the_hosts_lanes`.
Data an edge brings is not needed at all where no lane of that edge reads its bit
(`Lower::unneeded_edges`). That now covers the else part of an if that reuses a data register as
a mask: the then part writes the mask, the lanes that skipped it bring the data to the join, and
the else part, a block further down, first clears its own lanes' bits (`s_andn2` by EXEC, where
every lane of the edge is in EXEC). A read in another block counts only where neither block is on
a cycle. Of the 9,995 dumped pixel programs, 273 more translate (7,606 to 7,879), none regress and
24 lane-mask joins stay refused (285 before); the vertex corpus is unchanged. Test:
`emit::gfx_refusal_tests::a_pixel_program_joins_data_an_else_part_clears_with_its_exec`.

A join of SGPR data whose every edge brings copies of one value is that value, whatever edges
the lanes took (`Lower::copy_leaf`, `copied_number`): the copies followed are `s_mov`, a swap's
last XOR (`structure::xored_back`) and a spill read back that the structure proved is the same
value on every path (`restored`). This follows from the instructions alone: a copy keeps every
bit, so the register holds the one number on each path. With console lanes (compute) a lane mask
leaf gives its ballot. The 32-lane programs that rotate a value through a mask's register by two
XOR swaps hit it: the second swap's last XOR gives back the first one's, which the structure
counts as a mask (it XORed the mask's register), so the copy had no number and the join was
refused as "a lane mask read as data". Joins of two different values stay refused. On the
tmi-agent corpus (21,600, `local/pr-stack` 4cd9e726 with the tile-class commits): 9 more
translate (compute `28a2afab9c479d55`, `825365a94651e7ea`, `9171e6576a99ef14`,
`b797dd042ea5b43c`, three pixel and two vertex programs), none regress, 3 outputs change (joins
now numbers: copies of an EXEC mask in `0ef1184e45342d20`, of a constant-buffer dword in two
vertex programs), and 137 compute programs move on to "a descriptor read only as data". Tests: `gfx_refusal_tests::a_join_of_swapped_copies_of_one_value_is_that_value`,
`a_join_of_copies_of_two_values_stays_refused`.

In a mesh shader, a uniform branch can skip a block that replaces a scalar mask with a
vector compare result. At the join each mesh thread uses its group index modulo the console
wave size to read its bit. Wave64 takes lanes 32–63 from the high dword on the same edge.
A branch or operand that depends on a first-lane read can differ between the host waves and
stays refused.

## Liveness of phi edges: a bound on the paths walked

`Lower::live_edges` decides which incoming edges of a join can bring a value to a read: it walks back from the read phis, carrying the last four branch conditions a lane took (`MAX_KNOWN`), and an edge is dead if they contradict. It kept every distinct set of conditions it had walked a phi with and compared each new one with all of them. Six of the title's pixel shaders (`0c138380aac06bee`, `1dc240e3ad667c4a`, `3b005093d9f023ac`, `6569ab9190d883bc`, `b84cdfce7b2dcc1e`, `cf5c1203d7f09fa0`) are branchy enough that a phi gathered thousands of sets (150 phis, 3,520 sets on one after 32,768 steps) and the translation never finished. Once the root-view and descriptor changes of late November let these shaders get that far, the world frame's pipeline of vertex `fd0cfdb9141c858c` and pixel `b84cdfce7b2dcc1e` stopped every run: the thread making it waited on the translation under the title lock, nothing else logged but the audio thread, and the title's watchdog fired 30 s later (+120-133 s, warm or cold). Now a phi walked with 64 sets (`MAX_PATHS`) is walked once more knowing nothing, which covers every later set; knowing less only keeps more edges live (a refusal at worst, never wrong code), as the existing rule of four conditions already does. Over the 12,279 containers one run translated (dumped to a scratch folder, translated offline by the `guess` example, arm64 release), the six now finish in 2 s each (two translate, four are refused for a lane mask read as data) and the other 12,273 give byte-identical results; the slowest translation is a compute shader of 10-11 s, before and after. Test: `emit::lower::tests::liveness_through_joins_of_many_conditions_ends` (six joins of twelve branches each: 0.02 s, 80 s without the bound).

## SGPR spills as scalar registers

The console compiler spills SGPRs it runs short of to the lanes of VGPRs that nothing else
touches (`v_writelane_b32 vN, sX, k` and `v_readlane_b32 sY, vN, k`), and on different paths it
parks different values in the same lane. As lanes of one vector value, no join can tell those
apart: `structure`'s `restored` and `Lower::spill_slot` follow a lane back to one value, or a phi
of the spilled SGPRs, but a mask spilled on two paths came back as an untracked value ("a lane
mask holding lanes elsewhere read as data: a program input", "a lane-mask join of an untracked
mask: ..."). `ir::build` now makes each lane of such a VGPR a scalar register of its own (a spill
slot, codes from `ir::SPILL_SLOTS` = 128 up; `ir::spill_slots`): the spill becomes
`s_mov_b32 slot, sX` and the read back `s_mov_b32 sY, slot`, so SSA joins the spilled values
like any other SGPR's, and the mask analyses follow them as copies. Only VGPRs every access to
which is a spill or read back of a constant lane qualify, and only if every lane read back is
written by some spill (a lane read with no spill holds the VGPR's own data, an input say); up to
128 slots, VGPR by VGPR. A slot read before any spill on some path reads 0, as an unset VGPR does
(`Lower::inputs`, `mask_number`). The pixel prim-mask analysis (`io::PrimMaskUses`) follows the
copies through slots. Measured on the world's deferred-light programs (wave32, 32 threads, two
or three spill VGPRs of 58 to 72 lanes in all): see "Lockstep for one wave32 wave" below for
the totals with the other changes of 28 November. Test:
`structure::tests::a_mask_spilled_to_a_vgpr_lane_comes_back_as_itself` (now through a slot).

## Lockstep: refused one-wave compute on a 32-lane host

The fast translation runs a program as one lane does (`structure`, decision 0003): a branch on EXEC is a branch on the lane's own bit. On a 32-lane host a console wave of 64 lanes is two host waves, and a cross-lane operation reaching across them (`lower/halves.rs`) needs both at a group barrier, which lanes that went separate ways never reach together; each such shape needed its own proof, and the rest were refused ("a cross-lane operation between host waves in divergent code", "LDS shared in a wave the host splits, in divergent code" and others).

In compute, `ds_bpermute_b32` reads within each 32-lane half. Address bits 6 through 2 select the source after the instruction offset is added (RDNA2 ISA §10.4.4). An EXEC-disabled source supplies zero. A split wave needs no group-shared exchange or barrier for this instruction.

A wave64 compute program refused that way whose group is one console wave (64 threads or fewer) is translated again in lockstep (`emit/lockstep.rs`): every host thread follows the console wave's one program counter. Each `s_cbranch_execz`/`execnz`/`vccz`/`vccnz` becomes `s_cmp_{eq,lg}_u64 mask, 0` and an SCC branch (`lockstep::branches`), and the structuriser (`structure::lockstep`) takes SCC of a mask operation as a test of the whole mask. A mask read as a number is the console's ballot (`wide_ballot`), the same in every thread, and so is every lane-to-scalar value: `v_readlane_b32` (`wide_read_lane`), and `v_readfirstlane_b32`, which in lockstep reads the lowest lane of the wide EXEC (`lockstep_first_lane`), not each host wave's own first lane. Every branch is then uniform across the group, every thread reaches every block and barrier the same number of times (loops too), and the exchanges are allowed everywhere. EXEC is only a predicate: what reaching a block no longer proves stays each vector instruction's guard (a register write keeps its old value where EXEC is off; a store is skipped). LDS takes a group barrier before and after each write (`lockstep::lds_barriers`), outside its guard: the other half's next read sees the write, and no half overwrites what the other has yet to read. Whole-wave code runs literally, not through the section simulation (`wave.rs`): every thread is there, DPP, `v_permlane16`/`x16` and `ds_swizzle` stay within the host's 32 lanes, and `v_readlane_b32` crosses the halves through the exchange. A program is refused by lockstep (its message is then the refusal) when a branch it cannot make uniform remains ("a branch not uniform across a console wave run in lockstep"). Groups of several console waves are not run this way: one wave's branches are not another's. Before lowering, a static check (`lockstep::split_branch`) follows what each branch's SCC is made from: scalar inputs, lane masks (their number is the ballot), `v_readlane_b32` of a uniform lane, `v_readfirstlane_b32` and scalar work on those are the same in every thread; a VGPR input, a waterfall loop's first-lane read (each lane keeps its own descriptor dword) or any other scalar a vector instruction writes is not, and a branch on one is refused ("a branch in lockstep on a waterfall loop's first-lane read, each lane's own", "... on a scalar a vector instruction writes per lane", "... on a vector input") rather than left to hang at a barrier. Lockstep reuses the fast path's resource tracking: its program has the same values and sites, and a waterfall loop is proved on the program's own `s_cbranch_execnz`, which lockstep rewrites. None of the dumps measured below trips the check.

`VICTORY_NO_LOCKSTEP=1` turns the fallback off (refusals as before), to compare. Tests (`victory-gpu`, Metal): `split_wave_tests::a_lane_read_behind_an_exec_branch_runs_in_lockstep`, `a_first_lane_read_in_lockstep_is_the_console_waves_first_lane`, `a_per_lane_loop_shares_lds_across_the_halves_in_lockstep`, `a_whole_wave_prefix_sum_behind_an_exec_branch_runs_in_lockstep`, `whole_wave_code_that_is_no_reduction_runs_literally_in_lockstep`; no GPU: `gfx_refusal_tests::a_lane_read_behind_an_exec_branch_translates_in_lockstep`, `lockstep::tests::a_branch_on_a_waterfall_read_would_split_the_halves_and_is_refused`.

Measured offline (27 November) on the 1,765 compute containers the HDR and prologue runs dumped as refused (Series S, `wave32_host` and `zero_copy_memory` as the game translates): main translates 189 of them today; with lockstep 285 (96 more, all wave64 groups of 64 threads or fewer; none of the 189 changes, since lockstep runs only after a refusal). Of the 96, 35 were refused for a cross-lane operation or LDS after the split, 34 for lane-mask joins "on the host's lanes" or "at a join of several edges" (every join is uniform in lockstep) and 27 for whole-wave code the section simulation did not follow (run literally in lockstep). What lockstep still refuses in that class (188): lane masks as data it cannot express, "a lane-mask join of an untracked mask: a value the translation could not make" (53) and "a lane mask holding lanes elsewhere read as data: a scalar operation not modelled" (51), both from masks still being per-lane booleans in the structuriser; then "a hardware SGPR input" (15), others under 8 each. Not run this way: 74 refused programs whose group is several console waves (64 not for a descriptor), and 93 wave32 programs (one host wave each, no split; their refusals are lane-mask joins and descriptors).

### Lockstep for one wave32 wave

A wave32 compute program whose group is one console wave (32 threads or fewer) runs in lockstep
too (`lockstep::fits`, 28 November): there the console wave is exactly one host wave, so ballots
and lane reads need no exchange, and every branch on a mask, being a test of the ballot, is the
same in all of the group's threads. `lockstep::branches` makes a wave32 program's EXEC and VCC
branches `s_cmp_{eq,lg}_u32 mask, 0`. In lockstep every thread runs every block, so EXEC joined
at a block is the join of the EXEC masks coming in, not "the lanes that are here": a lane-mask
join whose edge brings such an EXEC phi (a loop's back edge) takes it as `Atom::Phi` of it
(`fill_phis`) where it was "a lane-mask join of an untracked mask: a value the translation could
not make". This gives the console's own meaning to what the fast path, running each lane on its
own path, has to refuse: SGPRs a wave writes on one side of a divergent branch hold the same
number in every lane after the join, here as on the console.

The world's deferred-light programs are such groups: 200 of the 236 compute programs one world
frame (run 54) refused are wave32 with 32 threads (`[32, 1, 1]`), and they dropped some 470 of
the frame's dispatches (the passes into its 1280x720 lighting targets, about 80 programs run 1 to
11 times each). With this, the spill slots and the descriptor dwords above, 146 of the 236
translate offline (with the game's options, `wave32_host` and `zero_copy_memory`); they ran 343
of the frame's dropped dispatches. Lockstep translations of these programs take up to 78 s each
(eight at once on an M2 Max), after the fast path's refusal: the warm-up's cache takes them off
the game's path. Left refused, by count: "a lane-mask join of an untracked mask: a value the
translation could not make" 48, "a cross-lane operation between host waves in divergent code"
12, then six reasons of 2 to 6 each. Tests: `gfx_refusal_tests::a_join_of_copies_of_two_values_stays_refused`
(refused in a group of two waves, translated in lockstep in a group of one).

A lockstep program's masks are console numbers (`console_lanes`, wave32 too), and a lane-mask
join reads data joined into it as the mask's number (`fill_phis`). Such data was not counted as
read (`live_edges` started only from instructions' reads), so the join under it took the edge as
dead and joined 0: a loop copying another mask into VCC on one arm (`s_mov_b64 vcc, s[22:23]`)
lost every lane after its first pass, in wave64 lockstep as well. `live_edges` now starts from
the data phis mask joins take too. Test:
`split_wave_tests::a_mask_copied_back_in_a_lockstep_loop_holds_its_lanes_on_every_pass` (wave64
and wave32, on the GPU). Only joins that read the data as a number count (every lane enters
along one edge, which is not one whose value goes unread anyway).

Three more rules from the same programs (all of them in `Lower`):

- A lane-mask join read as data in lockstep (`mask_number`, which had no rule for joins and
  refused "a lane mask holding lanes elsewhere read as data: a program input") is the join of
  the numbers its edges bring: every thread takes the same edge. The joins are filled after
  the masks' own (`next_pending_mask`); an edge whose number cannot be made retries the
  lowering with the join marked, so what reads it falls back as before.
- A wave32 program's 64-bit bitwise operation on masks (the console compiler swaps and chains
  register pairs by `s_xor_b64`, the mask in either register of the pair) gives dword `h` of its
  result from dword `h` of its operands (was "written by a scalar instruction of the other
  width"). Test: `split_wave_tests::a_mask_xored_into_a_pairs_high_register_by_a_64_bit_xor_reads_as_its_number`
  checks the shape on the GPU; only the world's lockstep programs reach the rule itself so far
  (their operands' numbers are joins).
- VCC read before any write holds what the last wave left, as SGPRs above the initialised ones
  do, and reads 0 (`inputs`; a wave32 program's VCC_HI joined into a lane mask).

Masks no mask tracks, in lockstep (`Lower::untracked_numbers`). Some lockstep wave32 programs
join lane bits that no mask tracks: a compare's or carry's result, or the low dword of an
`s_xor_b64` / `s_mov_b64` of mask pairs, which the console compiler uses to swap registers.
These were refused with "lane-mask join of an untracked mask: a value the translation could not
make". Every thread runs every block, so such a value's number is exact:

- a compare's or carry's bits are its ballot where it ran, ANDed with its guard
  (`untracked_lane_bit`);
- a scalar operation's result is the operation on its operands' numbers (`mask_number`, read
  through `get`);
- a join of either kind with data reads that number (`lockstep_number`).

A lockstep join whose edge cannot give a number is marked in its own set (`no_number`, retry
reason `NO_NUMBER`). Reading the join's number then answers `NUMBER_JOIN`, and that error going
up through a data read is not mistaken for the mark. Before this split, a data phi carrying the
error was marked as a join, and the lowering ended in "a phi that does not settle".

These numbers can fail in programs that translated without them: a failed speculative number
marks joins that used to work. So the rules run only on a second try, after a first lowering
without them refuses on a lane-mask join or a phi that does not settle (`lower_graphics`). The
first try is the lowering as before. Over the 236 frame compute programs (`refused` example),
190 translate (was 181), and none that translated regress.

### If-conversion for groups of several console waves (#376, design note)

Lockstep makes control flow uniform across the group because the group is one console wave: the wave's ballot is the same number in every thread, so a branch on it is taken by all or none. A group of W console waves has W different ballots, so one program counter cannot follow them; making the group's control flow uniform means removing the branches, not testing them differently. Proposed shape (a predication pass, not an extension of `structure::lockstep` — the single-wave ballot trick is what lockstep *is*, and multi-wave branches have no group-wide scalar to branch on):

* **Branches.** A divergent `if`/`else` becomes both arms executed straight-line, guarded per lane: stores, LDS writes, atomics and exports keep their EXEC guard; register writes are predicated selects at the join (`v_cndmask`-style), which the fast path already emits for EXEC-guarded writes. The branch's own condition must still be computed, but as a per-lane mask, never as an SCC branch. Loops: the loop runs while *any lane in the group* is active — a group-wide ballot (the OR of the waves' ballots, one `ds`/barrier reduction per iteration) instead of the wave's own — with per-lane continue masks; every thread executes the same number of iterations, so barriers inside the loop are reached by all.
* **Barriers and exchanges.** With no branches left, every group barrier is uniform, and the existing half-pair `xchg` exchange extends to W waves as a tree of pairwise exchanges (log2 W barriers) or W-1 sequential barriers; cross-wave LDS needs the same barrier discipline as `lockstep::lds_barriers`, but across all waves, not two halves.
* **Where it runs.** Only after a refusal, as lockstep runs today, so nothing that translates now changes. The static check `lockstep::split_branch` generalises: a branch is convertible when its condition is a per-lane mask computable from uniform or per-lane data; a branch on a scalar a vector instruction writes per lane (waterfall first-lane reads) stays refused unless the value is wave-uniform.
* **Cost bound (guess).** Groups of at most 4 console waves (256 threads on a 32-lane host): Metal threadgroups allow up to 1024 threads, but each predicated arm costs every thread, and the exchange tree grows with W; beyond the bound, refuse as today. Both arms running is the main cost for unbalanced branches — measure on the corpus before raising the bound.
* **Open:** the current count of multi-wave refusals (the 74 was measured 27 November; a fresh `VICTORY_DUMP_REFUSED` corpus sweep answers it, #282), and whether any of them branch on per-lane scalars (those stay refused even under predication).

## Console pipeline descriptions (device slots 0x1c8 and 0x218)

Observed at run time (`VICTORY_DUMP_DESC=1` dumps the first descriptions and the bytes behind every pointer and size pair). Grand Theft Auto VI creates most of its pipelines through these two device methods, `(this, desc, iid, out)` with the pipeline state's IID.

* **0x218, compute.** `desc` is a root signature object pointer and one 0x60-byte record: pairs (pointer, size) of the pipeline registers (0x1c bytes), the hash (0x14), the GFX10 code and the metadata, then two pointers and a 64-bit hash. These are the entries of kinds 0, 1, 6 and 7 of the `FSPX` layout `victory_shader::native::fspx_program` reads, so the host builds that part and a container (`pipeline::create_compute_pieces`) and the compute path translates it. When `desc`'s root signature is one of ours, its blob goes in as one more entry, which sizes root constants (below, "Compute root constants").
* **0x1c8, graphics.** `desc` is a root signature pointer, then stage slots of 0x160 bytes from +0x08. A slot holds 16-byte (pointer, size) pairs: pair 0 a register image (0x17c to 0x1d8 bytes, starting `08d80540` or `00d80540`), pair 1 a blob that holds an input signature (element count, then rows with the semantic names, "POSITION", "COLOR", "TEXCOORD", as an `ISG1` part does), then up to three (code, metadata) pairs at pairs 2/3, 4/5 and 6/7, which of them filled varies. At slot+0x108 there is a block of host pointers, `0x0401000000000160` (the slot size), a 40-byte hash and `0x8000000000000000`. The metadata is the compute metadata with a 0x10-byte header, `u32 2, u32 size - 0xc, u32 (0x76 seen), u8 0x40, u8 first user-data dword, u8 count of 0x41 records, u8 0`, then the same 32-byte tagged records (0x41 resource, 0x42 view declaration, 0x43 hash). The slot layout is therefore known; what is not: which of the code pairs is the vertex shader, the pixel shader or a variant, how the register image maps to the stage records (`SPI_SHADER_*`, `SPI_PS_INPUT_*`, `DB_SHADER_CONTROL`), where a pixel shader's export patches are, and the graphics state (blend, rasterizer, formats), which is not in the description. Until the translator takes these pieces, `pipeline::create_unread_1c8` hands out a pipeline with no shaders and its draws are dropped.

**The graphics state is device method 0x1b8.** Found by `VICTORY_TRACE_MISSING=<bytes>` (every unwritten slot logs its first calls, `VICTORY_TRACE_MISSING_CALLS=n` more, with its arguments and the bytes behind pointers): the title calls device slot 0x1b8 about 42,000 times at start-up, `(this, base, count, records, iid, out)`, `base` a pipeline 0x1c8 made, `iid` the pipeline state's, and uses the result as the pipeline it binds (`VICTORY_TRACE_BIND=1` counts binds by kind). Each of the `count` (1 to 3) records is 83 dwords, a tag then the D3D12 structure of its kind: 0 `D3D12_DEPTH_STENCIL_DESC`, 1 `D3D12_RASTERIZER_DESC`, 2 `D3D12_BLEND_DESC` (tag 5 is rare and not understood). A record left out means D3D12's default. The register image (pair 0) holds no blend or depth state (it has `CB_COLOR_CONTROL` and the shader registers). Target formats are in no record: a draw takes them from the targets `OMSetRenderTargets` bound, and the topology from `IASetPrimitiveTopology`. `native::console_state` decodes the records into `native::state::Graphics` (`apply`, `graphics_state_for`), `variant.rs` is the device method, and `glue::pipe` makes a console pipeline again for each set of targets. Command lists have no Xbox-specific state methods (every slot the title calls is written), and the title writes no packets into list memory while our lists keep it on the call path. `VICTORY_TRACE_STREAM=1` shows that the stream pipelines (device 0x2b0, PC style) carry all their state, and `VICTORY_DUMP_STATE=1` dumps what the host pointers at the end of each 0x1c8 slot point at (not understood).

Labelling the code pairs (by scanning the code for `exp` and `v_interp` encodings, 12 graphics descriptions, `VICTORY_DUMP_DESC_DIR=<dir>` writes every piece to a file): pair 2/3 (`c0`) is the pixel shader (it has `v_interp`, and `exp` to mrt0 and others), pair 4/5 (`c1`) the vertex shader (`exp pos0` and `exp param`), and pair 6/7 (`c2`) a second vertex shader (more exports, larger: probably the variant that runs with a geometry stage or as a primitive shader). A slot with no `c0` is a pipeline without a pixel shader (depth only). So a description is a root signature and one slot per pipeline, the slots after the first being the next pipelines of an array the title keeps (as for compute, whose descriptions are 0x68 apart). What is still open is the vertex input mapping (the signature blob says which semantics, not which VGPRs), the pixel shader's export patch source, and the register image fields.

## Guessed graphics pipelines (`VICTORY_GUESS_GRAPHICS=1`)

Without ground truth for the 0x1c8 register image, `VICTORY_GUESS_GRAPHICS=1` builds pipelines from what the code and metadata say; without it nothing changes (pipelines have no shaders, draws are dropped). Every item is a guess:

* **Which pieces.** Pixel shader: pair 2/3. Vertex shader: pair 4/5, or pair 6/7 when 4/5 is empty (most descriptions: a large pixel shader, empty 4/5, the vertex shader in 6/7). Only the first slot of a description is read. Container: `victory_shader::emit::graphics::container` (an `FSPG` part, code and metadata, plus `ISG1`/`OSG1` made from the inference below), read by `native_programs`.
* **Metadata.** v2 records start at +0x10 and share `native::metadata_resources` with compute (kind 7). Root constants (root type 0, observed as b500, the sampler index) are one SGPR, widened up to the next slot's first SGPR (`native::expand_constants`, a guess); a `SamplerHeap` parameter is added one SGPR after the last declared user data (pixel shaders only; s9 constant, s11 heap, s12 primitive mask). Records of other root types are refused, each with its own message naming the root type number (not the record, so refusals tally by cause): `root type 5` or `6: a descriptor heap (SM 6.6 dynamic resources)` (a declared `ResourceHeap` or `SamplerHeap` record; graphics only infers heap SGPRs from the code), and `root type <n>: an unknown root type` (2, or 7 and above). No shader with root type 2 or 7+ has been seen: before the messages were split (commit `c3e1ba8`), all of these refusals read "an FSPX resource of an unknown root type", 266 unique compute shaders in a story-load run (Series S, windowed, 26 November). With root constants accepted (27 November, same setup, `VICTORY_DUMP_REFUSED`), 222 of them translate, and each of the other 44 has a root type 0 record and is refused for something in its code; none of the 146 refused shaders dumped (139 compute, 7 pixel) has a record of type 2, 5, 6 or 7+, and a translated one cannot. So that whole group was compute root constants; what 2 and 7+ mean is still unknown (the FSPX numbering is not D3D12's: 0 constants, 1 a root descriptor, 3 a table, 4 an immediate sampler, 5 and 6 the heaps).
* **Compute root constants.** A metadata record gives a parameter's first user-data dword, not its size, so a constant of N dwords takes user SGPRs `user_data .. user_data + N`. The count does not move the other parameters (each record names its own dword, the group ids follow `COMPUTE_PGM_RSRC2`'s user SGPR count, and the host fills each dword the shader reads by slot and dword); it only decides which SGPRs are the constant's. Too small, the shader's read of a later dword is refused (`a hardware SGPR input`); too large, the constant would take another parameter's SGPRs or the group ids (parameters are matched first), which reads wrong values. So compute takes the exact count: device method 0x218 puts the blob of its root signature object (kept by `root::create`, read back by `root::blob_of`) into the `FSPX` part as an entry of our own, `native::FSPX_ROOT_SIGNATURE` (kind `RTS0`), and `fspx_program` takes `Num32BitValues` of the record's slot from it (`RTS0` versions 1.0 to 1.2, bare or in a DXBC container). Guess (not observed): the console driver puts all N dwords of a constant in consecutive user SGPRs from the record's dword; the checks below refuse a count that disagrees with the other records or the SGPR count. Without a readable root signature (no entry: containers not made by 0x218, a root signature that is not ours, or a blob that does not read as `RTS0`), compute falls back to the graphics guess, `native::expand_constants` (d02869f): one SGPR, widened up to the first SGPR of the next slot's record. That guess is wrong when a constant is followed by a gap (it takes the gap's SGPRs) or when slot + 1 has no record, as for the last one (it stays one dword). With a readable root signature, these are refused with `root type 0`: `root constants the root signature does not size` (slot absent, not root constants, 0 or more than 64 dwords), `root constants overlapping another root parameter`, and `root constants past the user SGPRs`. Graphics keeps the guess (`emit::graphics`).
* **Vertex shaders** fetch for themselves: v0 is the vertex id, `s[0:1]` the table of 16-byte vertex buffer descriptors (`s_load` before `buffer_load`/`tbuffer_load` with `idxen`), base vertex and instance (s2, s3) read 0. The translator turns such a load into a stage-in element `VB<F|U|N><n>` (float, uint, unorm8x4; n channels) of index `slot << 16 | byte offset`; the host maps those to attributes of vertex buffer `slot`, with the stride the title's `IASetVertexBuffers` gave (the `system_gaps` placeholders keep the converter's numbering by signature position). Only bound streams with nonzero addresses and sizes are submitted to Metal, each at its original input slot. One index per vertex: no per-instance streams. Other formats are refused. Outputs: `exp param<n>` is `TEXCOORD<n>` of register n (channels as enabled), `SV_Position` last.
* **Pixel shaders.** Attribute n is `TEXCOORD<n>` (identity with the vertex parameters), channels as read; flat when only `v_interp_mov` reads it. `SPI_PS_INPUT_ADDR` comes from the description's register image when it gives one (see "Pixel input enables"); otherwise it is inferred from the VGPRs read on entry: perspective centre (bit 1) when it interpolates, then position x.. front face (bits 8-12). Colour exports `exp mrt<n>` go to `SV_Target<n>`, declared float unless target n has an integer format: then the pixel shader is translated for the draw's targets (`Options::integer_targets`, `glue::make_host_pipe`, any pipeline with a pixel stage) and the output is uint, storing the exported dword's bits (the console keeps integer colours in its float registers; Metal refuses a float output for an integer attachment, which dropped the whole deferred-lighting pass); a compressed export is taken as FP16_ABGR (two halves per dword, as `v_cvt_pkrtz_f16_f32` makes them); no export patches. Wave size (`graphics::infer_wave64`) is 64 when the code names `exec_hi` or runs a 64-bit SOP on EXEC or a wave64-only instruction (`*_saveexec_b64`, `s_wqm_b64`...); else 32 when it runs a wave32-only one (`*_saveexec_b32`...); else 64 when it names `vcc_hi` or runs a 64-bit SOP on VCC (weak signs: a wave32 program may use the VCC pair as spare SGPRs), 32 when a 32-bit SOP reads or writes `exec_lo`, and 64 by default.
* **Pixel input enables.** Guess: a 0x1c8 slot's register image (pair 0) holds `SPI_PS_INPUT_ENA` at byte 0x58 when its word 1 is 4, and at 0xac when word 1 is 6 (`graphics::ps_input_ena`). Found from values across 23,022 dumped images, not a console document: a 16-bit word next to the interpolant-count word, one value per pixel program, covering the VGPRs the program reads on entry in about 96% of programs (the rest read implausibly high VGPRs), and with a barycentric bit set wherever the code interpolates. The image holds one such word; it is taken as both `SPI_PS_INPUT_ENA` and `SPI_PS_INPUT_ADDR` (the VGPR layout), so a console that packs VGPRs by a different ADDR would place them elsewhere. When present it replaces the guess from the VGPRs read (`graphics::container_with_inputs` carries the word after the metadata in the FSPG part; `pipeline::graphics_containers` decodes it). Guess: `POS_FIXED_PT` (bit 15) is the pixel's integer x in bits 0-15 and y in bits 16-31, made as `uint(pos.x) | uint(pos.y) << 16` from `SV_Position` (the programs read it as 16-bit halves and as an `a16` image address); a sample-rate or sub-pixel meaning is not modelled. Guess: the line stipple coordinate (bit 7) reads 0, as for a triangle; lines are not modelled.
* **Root layout.** Each stage is converted with the host's layout of its arguments (`Binding::layout`: user data inline, then root views, then the page table), as compute is; with its immediate samplers as static samplers.
* **Typed image stores.** `image_store` with a dmask of fewer than four channels (and its `d16` form) writes the whole texel, with 0 in the channels dmask leaves out; the data VGPRs hold the enabled channels in turn. For a prefix dmask (0x1, 0x3, 0x7) this is not a guess: on GFX11 and older the hardware stores 0 in the missing trailing channels (LLVM 88421ea "[AMDGPU] Trim zero components from buffer and image stores"; Mesa 4a6ee2c4833 "aco: shrink buffer stores with undef/zero components"), as for typed buffer stores. For a gapped dmask (such as 0x5, or one without red) it is a guess: that evidence only covers trailing channels, and the 0 in the gaps is assumed the same way. A dmask of 0, whose effect is not documented, stays refused ("an image store of no channel").
* **State.** Blend, culling, depth, stencil, write masks and polygon offset from the register image (`Graphics::guess_from`, below; which polygon offset float is which is a guess), formats from the targets bound at the draw, strides from the vertex buffers bound; the host pipeline is made at the draw (`Pipe::Guess`, cached per formats and strides). Topology 17 (rectangle list) is drawn as a triangle list, so half of each rectangle: the fourth corner needs a geometry stage.
* **Copy propagation that does not settle.** The tracer's copy propagation (`Tracer::propagate_copies`) is an optimistic fixpoint whose phi rule is not monotone, so phis that feed each other across nested loops can cycle instead of settling (a register saved and restored inside an inner loop). After `MAX_ROUNDS` sweeps a phi that still changes is kept as a join, standing for itself. That never names a wrong origin, but it can retain less copy information than an exact answer would: a descriptor or address traced through such a phi may then be refused later where an exact propagation would have resolved it. Programs that settle within `MAX_ROUNDS` are not affected.
* **Vertex user data.** If a v2 vertex stage has a nonzero first user-data dword and a 0x41 register index below it, try those record indices relative to the header's start. Accept that layout only when it resolves an otherwise unbound address to a declared table or root view and agrees with existing root-address uses. The title's root slot and descriptor table offsets come from the record. Absolute indices remain the default until the register image is decoded.
* **SCC of `s_add_i32`, `s_sub_i32` and `s_addk_i32`.** The signed overflow, from the sign bits (RDNA2 ISA, SOP2 2 and 3, SOPK 15): for the adds, both sources of one sign and the result of the other (`s_addk_i32` takes its constant sign-extended); for the subtraction, sources of different signs and a result whose sign differs from the first source's. A shader that read it was refused before ("the carry of a scalar operation").
* **Undeclared SGPR addresses.** A load or store whose address base is a live-in SGPR pair no root parameter fills is taken as a title-memory address (as an address loaded from memory is), and lowers through the memory-space page table. The guessed layout declares such a pair as root constants so lowering can read it from the user-data block. Guesses: such a pair always holds a title address (the console's user data carries raw addresses), and its root-signature slot is the first free one — the register image does not yet say which root parameter holds the pair, so a draw whose title never set that slot fails loudly ("root parameter was never set"). A descriptor built around such an address is still refused, and a shader translated with a real (non-guessed) layout still refuses the undeclared read itself ("a hardware SGPR input").

* **Pixel primitive mask.** Scalar masks, shifts and unsigned bit-field extraction (`s_bfe_u32`) may feed M0 for interpolation. When the resulting value and SCC have no other use, the inferred primitive-mask SGPR anchors the existing POPS collision-id mapping in the next SGPR. A hand-assembled regression covers extraction and rejects exports of either the original or extracted mask, as well as observable SCC. This extends the guessed-path heuristic; it has not been verified against the seven remaining title shaders in issue #4.
* **Sample and gather variants.** Each `image_sample*` and `image_gather4*` suffix adds address VGPRs in the order of the RDNA2 ISA (section 8.2.5, "VGPR Usage"; LLVM's `AMDGPUSampleVariant` agrees): offset, bias, z-compare, derivatives, then the coordinates and the level or clamp. With `a16` the coordinates, level and clamp are f16 pairs and the derivatives stay 32-bit, one VGPR each: only the `_g16` opcodes (not translated) pack derivatives, each direction from a new VGPR. LLVM sizes the address so from gfx1010 to gfx1030; before #320 the derivatives were read as f16 pairs, and 1D and 3D gradient samples with `a16` were refused. The offset word holds three 6-bit signed offsets (x in bits 0-5, y 8-13, z 16-21). So `_o` is accepted on every sample the translator takes (`sampleLevel`, `sampleBias`, `sampleCmp`, `sampleCmpLevelZero`, `sampleGrad`): the offsets must be constant and within DXIL's -8..7, and a cube sample with offsets is refused. `image_gather4_c_lz` and `_c_lz_o` become `textureGatherCmp` (DXIL op 74). Guess: a comparison gather compares the channel its dmask picks, as a plain gather gathers it; DXIL compares red only, so any other channel is refused. Still refused: comparisons at an explicit level or with bias or gradients (`image_sample_c_l`, `_c_b`, `_c_d`: DXIL 1.7 and 1.8 operations), and comparison gathers at a computed level (`image_gather4_c`, `_c_l`, `_c_b`).
* **Described pipelines read title memory too.** A pipeline whose state the title described (not guessed) was refused when a stage read title memory through the page table ("reads title memory through the page table in a graphics pipeline"), though it binds the same way: `make_render` builds both kinds alike and the draw gives either stage the table (`memory_args`). It now gets the table as a guessed pipeline does; a stage that marks the pages it writes is still refused. In a run of 27 November one of the five pipelines refused this way drew into the 2560x1440 half-float scene colour target every frame of the prison scene, which the final composite samples (`VICTORY_TRACE_INTO`). Tested by `shaders_the_host_cannot_run_are_refused_and_counted`.
* **Computed pixel and vertex buffers.** In a guessed pixel or vertex shader, scalar and vector dword loads through linear descriptors with a known control word use the pool page table at the computed address. The descriptor supplies the stride, record count and bounds mode. The Metal host binds that table for both render stages, as for compute: each stage's root arguments get the table's address (`memory_args` in `native/host.rs`), and the heap and the pool buffers are made resident for the vertex and fragment stages (`bind_root_render` in `victory-metal`'s `gfx.rs`). Vertex shaders were refused ("a buffer descriptor built around a computed address") until the tester log in #123 showed four guessed ones; `addr_tests::guessed_vertex_buffer_reads_the_pool` draws with positions read this way. The translator requires proof that the descriptor's swizzle flags are clear. Vector byte/halfword loads (`buffer_load_ubyte`, `sbyte`, `ushort`, `sshort`) also use this path, with zero/sign extension and bounds checked at the element width. Each byte is resolved separately, so halfwords can cross dword/page boundaries and carry into the address high word. Bounds mode 0 checks each component's starting offset against the stride; mode 3 checks its end. Bounds mode 2 skips range checks but still treats a zero record count as out of bounds. Alignment is a guess: these subword reads assume `SH_MEM_CONFIG.alignment_mode = UNALIGNED`; the register is not yet decoded, so DWORD/strict modes are not emulated. See the [AMD RDNA2 ISA](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture), sections 8.1.5 and 8.1.7 and the MUBUF opcode table. Hand-assembled translation tests and Windows GPU/WARP value tests cover this extension; it is not evidence that any of issue #4's seven title shaders now works. Any four SGPR dwords the translator cannot trace (last dword constant) are classed this way, so an image instruction can meet one too: as its sampler it is refused with "a sampler built from computed words" (all 17 such shaders of a story-load run, 27 November, built a constant sampler with `s_bfm_b64` and a 64-bit literal move; the descriptor tracer now folds `s_bfm_b64` of inline constants as it did `s_bfm_b32`, so those samplers are immediate ones), as a 128-bit image descriptor with "a computed buffer descriptor used as a view". Buffer accesses never give the second message. A refusal of an instruction without side effects is reported at the first instruction that needs its result, not at the image instruction.
* **Buffer loads through a descriptor around the code's constant table.** The console compiler builds a raw buffer descriptor around the table it appends to the code (`s_getpc_b64`, `s_add_u32` of the table's offset, a byte count in dword 2, control word `0x31016fac`) and reads it per lane with `buffer_load_dword ... offen` (the spline tables of an ACES output transform: the world's 64^3 display LUT, compute `1b1365c6711630be`). The translator read such tables through scalar and global loads but refused this vector buffer load. A load has no side effect, so the refusal was not reported: its result became a missing value and the program ran on without the table, writing one value (0.2278) to all 262,144 texels of the LUT; the tone-map then turned the whole world into a flat grey. `code_buffer_load` in `lower.rs` now reads the table's dwords, and 0 for a dword whose end passes the byte count (RDNA2 ISA table 35, mode 3; SOFFSET does not move the bound). Shortcut: every bounds mode is taken as mode 3, and an indexed (`idxen`) load is still refused. `addr_tests::a_buffer_load_through_a_descriptor_around_the_code_table_reads_it_on_the_gpu` covers it.
* **Thread-group info SGPR.** The compute system SGPR `COMPUTE_PGM_RSRC2` bit 10 enables (`TG_SIZE`) is synthesised: bits 0-5 the group's console wave count, bits 6-11 the console wave's index in the group (from the console-lane flat id when lanes are relabelled), and for a wave32 program the index again in bits 20-24; the rest 0. GTA VI's temporal resolve (64 threads, two waves of 32) numbers its threads `bfe(info, 20, 5) * 32 + lane` to fill a 10x10 tile in LDS; with those bits 0 both waves filled the same half, the rest of the tile was never written, and the resolved scene came out as sparkle noise. Tested by `a_wave32_programs_group_info_numbers_its_waves_from_bit_20`. Guess: the bit layout is RADV's reading of the register (`ac_unpack_param(tg_size, 6, 6)` for the wave index); the title's shaders only mask bits 6-11 and add `v_mbcnt` to get the thread's index. Before, these were refused as "the thread-group info SGPR" (5 unique shaders in a story-load run, 27 November). Tested on the GPU: `split_wave_tests::the_thread_group_info_sgpr_numbers_the_console_waves`.
* **Computed compute buffers.** A compute shader's buffer descriptor built around an address it computes (loaded from memory or passed in root constants) is treated like the pixel case above (same limits: loads only, linear, no swizzle) when the host uses title memory (`zero_copy_memory`). Before, such a load was refused ("a buffer descriptor built around a computed address"). Guess: compute shaders get the same subword alignment assumption (UNALIGNED). Only translation is tested (a hand-assembled shader in `gfx_refusal_tests.rs`); no value test runs it on a GPU. Stores, atomics and typed access through such a descriptor stay refused. Root constants in FSPX compute metadata (root type 0) are accepted as in graphics; a trailing block fills the user SGPRs up to the count `COMPUTE_PGM_RSRC2` declares (guess, see `native::constants_to_the_end`: the words after the first were read as "a hardware SGPR input", 19 unique shaders in a story-load run of 27 November, all past it now).
* **Indexed table buffers without a stride.** FSPX metadata gives a used resource no stride (0). A structured access by `idxen` through a buffer loaded from a descriptor table, with stride 0, used to be refused ("a structured buffer of unknown stride"). It now takes the element size from the view-stride table, as a buffer loaded from the resource heap does (the table entry's heap index is its table offset plus the load's, in the same heap). Guess: the host's view-stride table has an entry for every view slot, table entries included, and the views are raw (`descriptor/host.rs`). Only translation is tested (`gfx_refusal_tests.rs`); no GPU value test. A stride the metadata gives is kept.
* **Typed buffer stores.** `buffer_store_format_x`, `_xy` and `_xyz` (and their `_d16_` forms) write the whole element, with 0 in the channels the instruction does not name. This is not a guess: on GFX11 and older the hardware stores 0 for them (LLVM 88421ea "[AMDGPU] Trim zero components from buffer and image stores"; Mesa 4a6ee2c4833 "aco: shrink buffer stores with undef/zero components"). A typed view cannot address bytes inside an element, so only a store at offset 0 lowers: no immediate offset, no `offen`, and a scalar offset register that is absent or proved to hold 0 by moves of the constant (`s_mov_b32 s2, 0`). Any other offset, also one that may be 0 at run time but is not proved so, is refused ("a typed buffer store at an offset") rather than dropped. The `idxen` index is the element and is not part of this.
* **SCC of `s_lshl1_add_u32`..`s_lshl4_add_u32`.** SCC is whether `(S0 << N) + S1`, taken in 64 bits, reaches 2^32: the add's carry or a bit the shift moves out (RDNA2 ISA, SOP2 46-49; the RDNA3 manual spells the 64-bit sum out). Mesa ACO ignores the bits shifted out (it fuses `s_lshl` and `s_add_u32` even when SCC is read); the manual is followed here. SCC was `result != 0` before.
* **Mesh-style vertex programs.** A vertex piece that exports primitives (`exp prim`) is an NGG program (7,718 of the 8,841 vertex pieces of a prologue run, 27 November), translated as a mesh shader (`emit::graphics::mesh_io`). Guesses, the register image being undecoded: the group size is `1 << k` lanes for the program's `v_lshrrev_b32 vN, k, v5` of the group input (k = 7 in every one sampled), the maximum vertex and primitive counts are the group size (the programs count both in 7 bits plus one), triangles, every output per vertex, LDS up to the highest `ds_*` offset plus 16 bytes (they share three dwords at offsets 0, 4 and 8). User data starts at s8, as for the console's merged stages (`HwStage::user_sgpr`); SGPRs the tracer infers as heaps or memory addresses are renumbered so. A constant EXEC (the programs' `s_mov_b64 exec, 1`) is lane 0 of each console wave: mesh lanes are the console's. Test: `emit::graphics::tests::a_vertex_program_exporting_primitives_is_a_mesh_program`.
* **Masked carry in a descriptor heap address.** After adding a byte offset to a table pointer, a shader can form the high word with `s_addc_u32 hi, 0, 0` and mask it to eight bits. The carry is either 0 or 1, so this mask preserves it. The address tracer removes a scalar AND when the mask retains every possibly set input bit, then checks the underlying high word. Regression: `resource::tests::a_mask_that_preserves_a_heap_address_carry_keeps_the_table_offset`.
* **`src_scc` carry in a descriptor heap or table address (`#258`).** When adding a 32-bit byte offset to a 32-bit descriptor heap or table base (`s_add_u32 lo, <base>, <offset>`), the compiler also forms the high word with `s_add_u32 hi, 0, src_scc` (or `src_scc, 0`) instead of `s_addc_u32`. `Tracer::scc_carry` (`resource.rs`) accepts that high word when `base` is a table or heap, one operand of the low `s_add_u32` is `base` itself (a copy stripped), and the high `s_add_u32` adds constant `0` and the exact `SCC` value left by that low `s_add_u32`; `s_add_i32`, non-zero constants, chained low adds, overwritten `SCC`, and 64-bit root descriptors stay refused (`resource::tests::a_heap_or_table_address_carry_added_as_scc_source_tracks_the_offset`).

* **Graphics shaders that index the heap themselves.** A guessed vertex, mesh or pixel shader that loads descriptors from an SGPR the tracer infers as a `ResourceHeap` base (a 32-bit address, high half 0; heap index = byte offset / 32 from 0) used to be refused ("indexes the descriptor heap itself"), although it translated: the translator reads those descriptors with `createHandleFromHeap`, and the host already binds the title's resource and sampler heaps at the converter's heap slots 0 and 1 for the vertex (or mesh) and fragment stages (`bind_root_render` in `victory-metal`), with the converter told the heaps are directly indexed (`msc.rs`). Only the refusal went (`native::shaders::unsupported`). Tests: `heap_tests` (a vertex, a mesh and a pixel shader each load the image descriptor at heap entry 1 and draw with its texel). shortcut: the host still keeps current only the heap entries device calls write and the first entries of bound tables (`remirror_tables`); a record the title copies with plain stores into an entry no bound table covers is not mirrored for a shader that indexes it, and reads what the entry held before. Compute shaders that index the heap themselves stayed refused until a compute value test covered them (below).
* **Compute programs that index the heap themselves.** A pair of the title's compute programs (wave32 and wave64, 160x90 groups, about 100 image reads and two four-channel image stores each; first guessed to be the deferred lighting, but the views their tables hold are 2048x1824 and 7680x90, so more likely a shadow pass, still a guess) loads a shadow map descriptor with `s_mov_b32 sN, s6; s_mov_b32 sN+1, 0; s_load_dwordx8 ..., s[N:N+1], <index * 32>`: s6 is a user SGPR no FSPX metadata record names (the records are two tables, at s4 and s5). It was refused as "a descriptor outside the heaps" (the address traced to title memory). FSPX compute programs now take the guessed graphics programs' heap inference (`native::infer_heaps`, guess: an undeclared user SGPR below the program's user SGPR count, used as a 32-bit descriptor base, is the resource or sampler heap), and the host no longer refuses compute shaders that index the heap (`Title::compute_pipe`): the heaps were already bound for compute at the converter's heap slots. Over the 1,756 compute containers a run's refusals dumped (`VICTORY_DUMP_REFUSED`), 74 now translate, all heap-indexing (translated offline by the `guess` example); those two programs move on to their next refusals ("a lane-mask join of an untracked mask: data, at a join of several edges" for the wave32 one, LDS in a split wave for the wave64 one). Test: `heap_tests::a_compute_shader_loads_a_texture_by_heap_index` (the title's shape: a UAV table at s4, the heap at s6; reads heap entry 1 on Metal).
* **Mesh programs: uniform lane masks and root constants.** Two guesses let most of the title's mesh programs translate (27 November; of the 4,153 vertex programs a prologue run to the first world frame refused, 2,965 mesh programs now translate). (1) The G-buffer programs test a wave-uniform value as a lane mask: `v_cmp` of SGPRs, `s_and_b64 sN, vcc, exec`, then `s_mov_b32 sY, src_scc` and a branch on sY much later, inside an EXEC region of `lane < count` (`v_cmpx`). A mesh program has no ballot over the console's wave, but none is needed: when the mask is a VALU compare of SGPRs and constants made under the same EXEC, SCC is the compare's bit, in every lane, those outside EXEC included, as the console's wave holds it (`Lower::uniform_and_exec`; whatever EXEC is, a join of lane conditions included). Guess: EXEC is not empty there (the programs branch around their EXEC regions with `s_cbranch_execz`); with an empty EXEC the console's SCC is 0. Test: `emit::graphics::tests::a_mesh_program_keeps_the_scc_of_a_uniform_lane_mask`. (2) "A hardware SGPR input" in the mesh programs was not s0-s7 (of those only s3, the merged wave info, is read) but s22, user data 14: the b500 root constants are read as two dwords (13 and 14) with the descriptor heap's SGPR inferred at 15, while the metadata gives one. Root constants with no parameter after them but an inferred heap now reach up to the heap (guess: the heap SGPRs follow the last parameter's). Test: `root_constants_reach_up_to_an_inferred_heap`. `s_sext_i32_i8`/`_i16` are lowered too. Neither sets SCC (RDNA2 ISA, SOP1), so a compare before them still decides a select after them; the Windows GPU test `emit::valu_tests::scalar_sign_extensions_keep_scc` checks that SCC is kept. Value tests on Metal (`victory-gpu`): `mesh_tests::a_mesh_program_branches_on_the_scc_of_a_uniform_lane_mask` (the SCC rule decides whether a mesh triangle is drawn), `addr_tests::scalar_sign_extensions_on_the_gpu` and `addr_tests::a_compare_with_dpp_reads_the_quad_neighbour_on_the_gpu` (`quad_perm:[2,3,0,1]` of a compare, per lane).
  Later the same day: the NGG vertex programs ahead of the G-buffer (`08173be1`, `d7447e18`, `f537f5a1`) test EXEC without such a mask instead. The compare is ANDed with an `s_cselect` of -1 and 0 by SCC, then `s_andn2_b64 sN, exec, mask`, and its SCC is kept and branched on. The atoms of the mask are not all uniform: a compare holds 0 outside its own EXEC, which shows up as that EXEC's atoms. On the current EXEC, though, those atoms are decided (`Bools::decides`), and what is left is uniform. So SCC is the mask's bit with those atoms fixed, or its complement for `s_andn2` (`Lower::uniform_with_exec`; same guess as above: EXEC is not empty). The three programs now get from 0xf30 to about 0x7c2c. There they stop at "a cross-lane operation in a mesh shader": a `v_readlane` of an SGPR spill whose lane was written on two paths into a join. The spill analysis (`restored`) maps a read back to one value only. Of 8,843 dumped vertex programs, 37 more translate (8,121 to 8,158), and none regress. Tests: `emit::graphics::tests::a_mesh_program_keeps_the_scc_of_exec_without_a_uniform_lane_mask`, and on Metal `mesh_tests::a_mesh_program_branches_on_the_scc_of_exec_without_a_uniform_lane_mask` (both atoms decide whether the triangle is drawn).
  Then three more rules take all three programs through. (1) A `v_readlane_b32` of a constant lane reads a VGPR that only `v_writelane_b32` fills on every path (an SGPR spill). It is the value written there, and where paths wrote different SGPRs into a VGPR join, a phi of them (`Lower::spill_slot`; the phis are filled with the other joins). The structure's `restored` map holds one value per read-back, so it could not take this. (2) The compiler swaps two SGPRs by three XORs, a mask's and data's say. The last XOR is a copy of what it XORs back out (`structure::xored_back`). In lowering, that gives the data its number back (`Lower::xor_copies`), where it used to be "a lane mask read as data". (3) `Lanes::unchanged` follows those XOR copies too, and the two halves of a wave64 copy each on their own (one half moved by `s_mov_b32`, the other swapped). All three programs now translate offline. Of the dumped corpora, 26 more vertex programs translate (8,163 to 8,189 on main's merged code) and 5 more compute programs (398 to 403), with none regressing. Tests:
  * `emit::graphics::tests::a_mesh_program_reads_back_an_sgpr_spilled_on_two_paths`
  * `emit::graphics::tests::a_mesh_program_swaps_a_mask_and_data_by_three_xors`
  * on Metal, `mesh_tests::a_mesh_program_reads_back_a_spill_joined_and_swapped_with_a_mask`, where the spilled scale decides whether the triangle is drawn
  * `split_wave_tests::a_mask_a_divergent_loop_swaps_away_and_back_by_xors_is_the_mask_it_entered_with`

  Each test fails without its rule. The per-half copy has no test of its own; the vertex programs above need it.
* **A mask swapped by `s_xor_b64` in a 32-lane program is the mask it gives back.** The tile classifiers (`f1a0f42fc6c79ad7`, `3ade34337c5736e1`, two of the indirect dispatches that write the 80x45 tile texture the output draw reads) leave a waterfall loop by swapping SGPR pairs with three `s_xor_b64`, though they are 32-lane programs, then join the swapped loop masks with what the loop's EXEC-zero skip brings and AND them with EXEC. The structure modelled no 64-bit scalar operation of a 32-lane program, so the swapped value was an opaque lane function and the join was refused ("a lane-mask join of an untracked mask: data, at a join of several edges"). `Lanes::mask` now takes a swap's last XOR (`structure::xored_back`, of either width, each dword on its own) as the mask it gives back. Moves stay `Lanes::sop`'s: taking 64-bit moves of a 32-lane program as mask copies too made 8 pixel programs read such a copy as a host-lane mask number, which they cannot ("not written by a scalar ALU instruction"). Not a guess: the last XOR of a swap holds the other register's bits exactly, in every lane, so each lane's bit is the copied mask's (proof as for `xored_back` above); with console lanes or split host waves the bits are then lowered as for that mask itself. A third XOR with any other register is no swap and stays refused. In these two programs the copied masks are ORed in a waterfall loop only with bits of the pass's EXEC, and the loop leaves by `s_cbranch_scc1` of `s_andn2_wrexec`, a wave-wide branch, so every lane leaves on the same pass and reads the final mask (the #257 shape does not arise). Tests: `split_wave_tests::a_mask_swapped_into_a_pair_by_64_bit_xors_in_a_32_lane_wave_is_that_mask_at_a_join` (Metal values, and the look-alike refused).
* **Missing vertex stage.** When both vertex piece pairs are empty, skip the slot. A vertex-only slot can still draw depth. Mesh pipelines still need their stage registers decoded. Pixel shaders with opaque lane masks, system inputs the signature lacks, and vertex fetches with other formats are refused (see the frame log).
* **`v_cvt_pknorm_u16_f32`, `v_cvt_pknorm_i16_f32` and NaN.** Assumption: a NaN source packs as 0. No public source states what the hardware gives; this follows Direct3D's float-to-normalized conversion rule, and the code comment marks it as a guess. The rest is not a guess: src0 in the low half and src1 in the high half (RDNA2 ISA, Mesa ACO), clamped, scaled and rounded to nearest even (Mesa NIR `pack_unorm_1x16` / `pack_snorm_1x16`).
* **`s_pack_{ll,lh,hh}_b32_b16`, `s_bfm_b64`, `s_bitreplicate_b64_b32`, `v_mad_u32_u16`.** Lowered from the RDNA2 ISA opcode tables (SOP2 50-52 and 37, SOP1 59, VOP3 883); none writes SCC. `s_bfm_b64` reads six bits of each count, as its opcode entry says (the summary table's `[4:0]` is the 32-bit form's). `v_mad_u32_u16` takes op_sel bits 0 and 1 (table 21) to pick the high halves of src0 and src1; op_sel bits 2 and 3 (the 32-bit addend and result) and source modifiers have no documented meaning and are refused rather than guessed.
* **Refused shaders.** `VICTORY_DUMP_REFUSED=<dir>` writes the container of each shader the native host cannot translate as `<stage>_<hash>.bin` (the hash the refusal log line names); with `VICTORY_DUMP_ALL_SHADERS=1` every shader translated. A compute container carries the FSPX part (code, kind 6; metadata, kind 7); a guessed graphics one the `FSPG` part. Game data: keep the folder outside the repo.
* **Debugging.** `cargo run -p victory-shader --example disasm` (`VICTORY_PIECE=<piece>`) disassembles a dumped code piece; `--example guess` (`VICTORY_STAGE=vs|ps VICTORY_CODE VICTORY_META`) runs the guessed translation on a pair; `--example sweep` translates a whole folder of dumped containers ("Sweeping a shader corpus").

### Why guessed draws leave the target black (findings)

Tools (all off by default): `VICTORY_TRACE_TARGETS=1` logs per present which title textures were drawn to or cleared (address, format, size) and which was presented, plus texture copies and resolves the host drops, and a warning the first time a draw, dispatch or clear is dropped at submission ("Dropped work is logged loudly" below); `VICTORY_DUMP_TARGETS=<dir>,<seconds>` writes the 8-bit colour targets drawn since the last present as PPMs; `VICTORY_DEBUG_PS_RED=1` makes every translated pixel shader write red, `VICTORY_DEBUG_VS_TRI=1` makes every translated vertex shader position a triangle covering the target (translator debugging, `emit/lower.rs`). `VICTORY_TRACE_PRESENT=<every>[,<seconds>]` logs draws into a texture the title has presented (the first, then every `<every>`th, at most 12; with `<seconds>`, only from that long after the first present, to skip the menus): shaders, static samplers, root parameters and table entries (textures with how much of their memory is non-zero, constant and placed buffers with their first bytes); `VICTORY_TRACE_TARGETS` does not reach the present pass. `VICTORY_TRACE_INTO=<title address, hex>[,<seconds>]` (with `VICTORY_TRACE_TARGETS`) logs at each present what wrote the title texture at that address since the last present, by shaders: draws that went on to the host list or were dropped, compute dispatches holding it for unordered access (ran or dropped), and texture copies into it; from `<seconds>` after start (0 when left out), the first 3 drawn draws of each kind also log what they read, as `VICTORY_TRACE_PRESENT` does. `VICTORY_TRACE_TABLE_ENTRIES=<n>` sets how many entries of each table those two traces log (4 if unset).

* The first frames are 3840x2160, one MRT draw per frame into the swapchain buffer, a second RGBA8 target and a depth target; no texture copy or resolve was ever dropped, no UAV or compute work touches them before frame ~700. The presented buffer is the one drawn to.
* Host textures are separate host allocations (`Textures::realize`), not views of pool memory; only sampled textures are uploaded from title memory, render targets never are.
* With the real shaders every target dumps all black. With `VICTORY_DEBUG_PS_RED` and `VICTORY_DEBUG_VS_TRI` together, a target dumps solid red: pipeline, targets, formats, viewport (3840x2160) and the draw path work. With only `VICTORY_DEBUG_PS_RED` it stays black (the real vertex shader's triangle covers no pixel); with only `VICTORY_DEBUG_VS_TRI` it stays black (the real pixel shader writes black or transparent there). So two things remain: the vertex shader's position (its matrix comes from a constant buffer read through a descriptor table, likely zeros) and the pixel shader's sample (its texture and sampler).
* The title draws rectangle lists (topology 17) from vertex buffers of 24-byte vertices (float3 position, unorm8x4 colour, float2 uv): fullscreen rectangles in clip space (-1, 1). Only the half-rectangle triangle is drawn; a proper expansion is not done.
* `descriptor_table_buffers_decode_into_metal_heap` also fails on c542e9d (title address not in a mapped pool): not caused by this work.

#### The fullscreen quad draws (trace of what they bind)

`VICTORY_TRACE_TARGETS=1` also logs, for the first draws with vertex buffers, the root parameters and the views of their descriptor tables (`native/host.rs`, `trace_draw`). Observed at start-up (frames 2 to ~700), per frame three fullscreen draws (a 3-vertex triangle with no vertex buffer, and rectangle lists of 24-byte vertices) into 3840x2160 RGBA8 targets:

* The descriptor tables hold the title's own view records (`vVIE` tag, view index, generation; 32 bytes per entry), not GFX10 descriptors. For the quad's pipeline the table is three entries: a placed depth view (R32, format 39), a 2D SRV of a 3840x2160 RGBA8 render target (the previous pass's output), and a 480-byte constant buffer whose first rows are the identity matrix. The pipeline's metadata offsets (SRV at 0 and 8, CBV at 16, in dwords) are these entries at the host's 32-byte step, so the vertex shader's matrix is the identity and the binding is right; the SGPR-above-s1-reads-0 guess does not zero anything here.
* The sampled render target and the dumped targets are all zero: every pass of these frames samples a target that only these passes (or clears to black) write. No mesh dispatch and no compute work were seen in these frames (dispatch and draw counters agree), so nothing ever puts content into that chain. The black is therefore consistent with a real black input, not with a binding error found so far; the scene is drawn by pipelines this path does not run (mesh dispatches are dropped on the native host) or later than the frames examined.

#### Rectangle lists, dropped-command counters, variants (guessed path)

* Rectangle lists (topology 17) of a guessed pipeline are drawn as instances of four vertices (a triangle strip): the vertex shader is translated again with `Options::rect_list`, which reads each fetched attribute for the rectangle's three corners (stage-in elements of corner 0 to 2, index `corner << 28 | slot << 16 | offset`, instance-stepped with three times the bound stride) and picks a corner by vertex id, the fourth being the two other corners less the right-angle one (so only affine vertex shaders are right; a uint attribute takes a stored corner; see "Rectangle-list corners" below). Indexed rectangle lists are dropped.
* The 64-frame log line counts dropped command kinds: `mesh`, `mesh_indirect`, `indirect`, `texture_copies`. Frames 2 to ~1000 show 0 of each: no mesh dispatches, indirect executions, texture copies or resolves, in the boot frames. The Bink DLL (`bink2scarlett.dll`) is loaded at start-up, but no compute dispatch or texture copy happens before frame ~695 (so no GPU or upload-based video decode before then); calls into its exports are title code and are not counted.
* State variants (device slot 0x1b8, `native/gfx-state`) of a guessed base inherit its shaders; their decoded blend/raster/depth state replaces the guessed defaults, and `Graphics::for_draw` adds the inferred vertex layout and bound targets. With them every draw of the boot frames has a host pipeline (`no_pipeline=0`), still all black through frame ~1000.
* Keep large-frame code out of `PresentX`'s call chain (`queue.rs`, `glue::present`): enlarging its stack frame crashed the title under Rosetta ("unexpectedly need to EmulateForward"); the draw-counter log and the present trace are `#[inline(never)]` helpers for that reason.

#### The sampled target's title memory, and the compute at frame ~600

* `VICTORY_DUMP_TITLE=<dir>,<s>,<s>` reads the title memory behind each 8-bit colour target drawn to, whole, and detiles it into a scratch texture. At 20 s and 120 s every one is entirely zero (33,423,360 bytes, 0 non-zero), including the 3840x2160 RGBA8 target the quad passes sample. So the black chain is not a CPU, DMA or DirectStorage write the host misses.
* Render-target textures are nevertheless watched now (`textures.rs`, the scanner of `native/perf`): a change of their title memory after its first sighting is uploaded.
* The first compute dispatch (frame ~580) is groups [5,5,33] of 8x8x1 writing a 33x33x(33) R16G16B16A16_FLOAT unordered view (`VICTORY_TRACE_TARGETS=1` logs dispatches): a colour-grading LUT, not video or a splash.

#### The int3 at image+0x4dcbc9d (analysis, `native/perf`)

* It is an overflow assert in an append routine (`0x4dcbc40`: used + length past a 0x18000-byte capture buffer), reached when a profiling scope ends (`0x4e529a0`, started in `0x5ab12e0` with a 32 KiB stack buffer and the scope hash `0xf0a244ef`). The scope is one iteration of the main thread's boot wait loop (`0x21b7070`): update the main loop (`0x21b7970`), `Sleep(33)`, wait 100 ms on a semaphore (`0x5ab12e0`), test again. Each iteration appends 0x6c0 bytes, so the buffer holds 56 and the 57th asserts (about 8 s). It is not a clock limit: running QPC and the tick counts at half speed (`VICTORY_CLOCK_SLOW=2`) moves nothing.
* The loop ends when `0x41e8c0(&0x1478ac9c8, [0x1478acba0])` is true: record `[0x1478acba0]` (4) of a ring of 0x50-byte records at `0x1478ac9c8` has a non-zero dword at +0x18 (`0x1478acb20`). Other threads stamp their own record's first dword each time they poll (the two counters that grow 1000 a second are such stamps). `VICTORY_TRAP_WRITES=<page>,<addr>,...` (the page is kept read-only; title writes fault, are logged with the title call chain and resumed) shows nothing ever writes `0x1478acb20` or the other records' +0x18, and the cell `0x1478acba0` only once, at static initialisation (`0x222db2a`). A host write would have crashed on the read-only page. So the code that sets the flag never runs here, or sets it before the trap starts and something clears it.
* Ruled out as the flag's writer: frame statistics (device 0x288 now answers; they feed a different ring at `0x1475de4b0`), GPU list writes (all 2.6k land in pools, none in the image), fences, DirectStorage, and the shared semaphores (the two semaphores of the loop are never released; they are only sleeps).
* Diagnostics added: `VICTORY_WATCH_MEM`, `VICTORY_DUMP_MEM`, `VICTORY_DUMP_STACK`, `VICTORY_DUMP_REGS`, `VICTORY_TRAP_WRITES`, `VICTORY_TRACE_HANDLES`, `VICTORY_WATCH_HANDLE`, `VICTORY_TRACE_WRITES`, `VICTORY_CLOCK_SLOW`.
* The "mark ready" function is `0x145862370(this = ring, &cell)`: if the ring parent gate (`0x1458623e0` on `[ring+0x1a0]`, null for this ring) allows, it stores the tick (`0x144e4ba80`) into `[ring + cell*0x50 + 0x18]` when `[ring+0x190] == cell`. Its callers are `0x140dcbc08`, `0x141cfc349`, `0x1421b741e`, `0x1421b751d`, `0x142207863`, `0x142355d6d`, `0x1427ed3bb`, `0x14289e5b2`, `0x1429df27f`, `0x1458622b4`; those passing another object (`0x147bd2140`, cell `0x147bd2318`) are other rings of the same class. For ring `0x1478ac9c8` the only caller is `0x142355d6d`, in slot 12 (`0x142355d40`) of the class whose vtable is `0x1461b2790` (constructor `0x145a8fbe0`); slot 13 (`0x142355dd0`) resets the ring. Slot 12 is never called in a run (the write trap sees no store); who calls it, and on what event, is not found yet. The title app state is not it: its state byte (`0x1471b9048`, context `0x1471b8c78`) is already 2 (running) at 10 s, and delivering the app-state and constrained-change routines the title registers (`0x142217c70`, `0x142217c90`, later `0x1455f8720`) with "running" did not set the flag (the state routine blocks waiting for a handshake bit, by design).
* Frame count or time: presents delayed by 100 ms (`VICTORY_PRESENT_DELAY_MS=100`) still hit the int3 at the same wall time (about 18 s) at frame 64-128 instead of 384, so the boot wait runs on time (its iterations), not on frames. Bink: `movies.rpf` is opened (title RPF device, `CreateFileW`) but never read, by `ReadFile` or DirectStorage (`VICTORY_TRACE_FILES=1`, `VICTORY_TRACE_DSTORAGE_FILES=1`), and no XAudio2 engine, voice or buffer is created before the int3, so the wait is before the boot movie starts, not a stalled movie. `VICTORY_TRACE_CALLBACKS=1` lists every title code address passed to a host import or method and not called by the host when `VICTORY_DUMP_THREADS=<s>` fires (7,655 registered, 38 called in the first 10 s; most are timestamps in structures, so read it with a filter for names such as Register, Create and Set).

* **Found (native/perf):** the wait is the title's engagement screen, a front-end state waiting for a button press. The state object (method table `0x1461b2790`, inside a wrapper with table `0x1461bc490`, state name hash `0xbaab6434`, found on the heap at the int3 with `VICTORY_FIND_QWORD=1461b2790` and `VICTORY_DUMP_THREADS=0`) has run slot 13 (enter): its listener at +0x80 points at itself and is registered for UI event 0xe of the event hub `[0x147f00518]` with handler `0x142355ef0`, which takes menu actions (up, down, toggle, accept). Slot 12 (exit, which marks ring `0x1478ac9c8` ready) is then left to the player accepting (inferred: not traced to the call). The macOS input source had no keys and no pads, and `VICTORY_KEYS` was Windows-only, so nothing ever accepted; under Wine the runs pressed Enter (`VICTORY_KEYS`). With Enter pressed (`VICTORY_KEYS`, now served on macOS) the wait ends, the main thread leaves the wait for the frame loop at `0x1421b7530` (it runs while state `0xbaab6434` is current), the world loads (`core_restofworld.rpf`), and the run reaches frame ~1400 at about 53 s. After it, in the same run: `insertq` at `0x1402e7ff1` (now emulated), a texture view with more levels than its texture (Metal asserts; views are now cut to the texture), the render thread in `clwb` flush loops at one signal per line (now one signal per loop), and then a Metal command buffer GPU time-out at about 53 s (`kIOGPUCommandBufferCallbackErrorTimeout`, a fence stuck 4 behind) that stops presents: the next problem, on the GPU side.
* **Found (Metal): an event wait that lasts about 5 s ends its command buffer.** Measured on an Apple M4 (macOS 15.7) with our own kernels: a command buffer whose `encodeWaitForEvent` is not satisfied within about 5 s fails with `kIOGPUCommandBufferCallbackErrorTimeout` (5.006 s for waits of 5, 10 and 20 s; a 2 s wait completes). Its work does not run and the signals encoded in it are not delivered, so a fence it would have moved stays behind, and buffers behind it on the queue fail the same way. Buffers queued behind a long-running one fail too (a single 30 s dispatch completed; the two buffers queued after it failed). D3D12 lets a queue's `Wait` last until its signal comes, so `native/title.rs` keeps every unsatisfied fence wait on the CPU until observed backend completion, even when its producer has committed: that producer can still be delayed or fail asynchronously. This trades GPU-side wait scheduling for CPU polling: an already-committed producer completing between polls can add roughly one nominal 1 ms timer interval before dependent work is submitted (potentially longer under host scheduling or lock contention; title calls also pump). A before/after headless frame-rate comparison is not yet available because the development machine has no game installation; the functional tests do not establish the frame-rate impact. Each queued wait retains its specific backend fence through title release and address reuse, releasing it after consumption. Presents join the same held FIFO as signals, retaining their texture and surface until submission, so they cannot overtake a CPU-held fence wait or rendering; a later signal stays behind the present. Drawable acquisition runs on a per-surface worker, outside the shared title lock. A pending drawable leaves its present at the FIFO head while other queues continue; per-frame autorelease pools release drawable references when draining multiple presents. Deferred presentation failures are logged by the pump. The standalone blocking `Metal::present` helper polls at 1 ms intervals and returns an error after a 2 s drawable wait; the title uses only nonblocking `try_present`. Unmapped GPU reads and writes do not fail a command buffer on this hardware (they are dropped), so they cannot be used to provoke a failure in tests. Whether this was the 53 s time-out above is not confirmed; the regression test is `a_queue_wait_longer_than_metals_event_wait_limit_still_runs_the_work`.
* **Found (Metal, macOS 26.5): a process stopped while its command buffers wait can leave their GPU memory allocated** (#357). Twice on an M5 Max (36 GB), a world run stopped by a watcher (SIGTERM, then SIGKILL) right after its first command-buffer time-out left about 20 GB "In use system memory" on the GPU driver (`ioreg -c IOAccelerator`), owned by no process, until a reboot. The kernel logged two of the game's command queues "timed out waiting for prior signals" 3.5 s after the process was gone. A run that ended on the title's own fault after its time-outs got its memory back, and a 1 GB standalone repro (`kill -9` of a process with a command buffer asleep on an event) never stranded anything, so scale or memory pressure matters too. So SIGTERM, SIGINT and SIGHUP now stop the GPU's work first (`victory_metal::gfx::stop_gpu`, from a `sigwait` thread in `apps/victory`): every shared event the backend made (queue order events, fences, piece events; tracked weakly by `track_event`) is raised past any awaited value, the command buffers in flight get 3 s to complete, and what is left is paid on the CPU. Then every residency set the backend made (each heap's and the pool's; tracked weakly by `track_residency_set`) is emptied and committed, since process exit skips `Heap`'s `Drop` and a run that hung the GPU at world entry with ~16 GB in its heaps' sets left 19.7 GB in use after it exited. Tests: `a_stop_wakes_command_buffers_waiting_on_events_nobody_signals`, `a_stop_empties_the_residency_sets`.

#### What the boot waits on (20 to 50 s runs, `VICTORY_PROFILE`, `VICTORY_TRACE_FAILS`)

* The boot is not waiting on sign-in, store, packages or a missing file: the title's GDK calls are `InitializeApiImplEx` and GameInput only, no failed file open is logged, nothing polls an XAsync block. Calls by time: `WaitForSingleObject` (70k calls in 18 s over many threads), `Sleep`, `d3d12` slot 0x1c8 (pipeline creation, 23k), 0x1b8 (state variants, 41k), `qsort`, string compares.
* The one long poll is DirectStorage: the title loads about 9,500 requests (`G:\data\ttdb\...ttpf`, the `.rpf` archives) through one queue and polls entry 0 of its status array in a `Sleep(1)` loop (20,000 polls in 20 s, `dstorage status array` slot 0x18 returning `E_PENDING`). The queue is not stuck (`VICTORY_*` log of its lanes: reads 3,361 falling to 27, no failures, `held 0`), it drains at about 20 MB/s with workers 90% idle (the title submits in small batches); it finishes at about 30 s into a run (frame ~770 on the fast build). Right after it, the draw count jumps from 3 to about 21 per frame and new pipelines appear: that is the first content-producing work, and with the shader fixes merged it encodes (frame 1024: recorded 5749, encoded 5747).
* The title then hits its int3 (the perf agent's) at frame ~850 to ~1030, before a frame dump of the new work could be taken.

#### Extra vertex outputs, periodic dumps, first content draws

* Guess: a pixel shader's `TEXCOORD` n that the vertex shader does not write is declared as an (undefined) vertex output (`Options::extra_params`, made at the draw by `Shader::translate_extra`), as the converter refuses a pipeline whose fragment inputs the vertex stage lacks. After it every draw of the boot has a host pipeline (`no_pipeline=0` to frame ~900).
* `VICTORY_DUMP_TARGETS=<dir>,<seconds>,<every>` dumps repeatedly (files `t<n>_<second>.ppm`).
* First content draws (frame ~830 on, ~20 per frame into a 3840x2160 RGBA8 target with depth, after the DirectStorage load): with the real shaders the target stays black; with `VICTORY_DEBUG_PS_RED=1` a red band covers the bottom ~9% of the target, its right edge slanting (a ground-like surface seen from a camera: real geometry reaching the screen through the real vertex shaders). So the vertex path draws; the real pixel shaders write black there (probably sampling textures not streamed in yet, not examined).

#### The content draws (`VICTORY_TRACE_CONTENT=<dir>[,<seconds>[,<format>]]`)

* The ~20 draws per frame after the DirectStorage load (frame ~830) sample no texture at all: each has a descriptor table of constant buffers only (64 or 1536 bytes, 62 to 460 non-zero bytes: colours such as (1,1,1,1) and (0.976,...), scale and offset values like 3.05e-5, 0.748 and -0.92) and two root constants [0, 0]. They look like a batch of screen-space quads (UI, a fade or a mask pass), not world geometry.
* The boot passes (3 per frame) sample the 3840x2160 RGBA8 target these draws write (`0x73c7520000`), which is all zero including its alpha: so everything downstream stays black as long as these draws write (0,0,0,0).
* With the real pixel shaders the target and its alpha are 0 everywhere; with every pixel shader forced to (1,0,0,1) a band shows, so coverage, blend, depth and formats let pixels through. The cause is the real pixel shaders' output (their math, an input the vertex shader supplies, or the undefined extra vertex outputs), not yet identified.
* `D32_FLOAT_S8X24_UINT` (DXGI 19) now maps to the host's depth-stencil format; before, the scene's depth target had no host texture, so its draws ran without depth.

#### Rectangle corner order, a content pixel shader, and the depth suspect

* Rectangle lists come as top-left, bottom-left, bottom-right (observed in the title's vertex data), so the missing corner is `v0 + v2 - v1`, and the strip is v0, v1, missing, v2 so both triangles wind alike (the first try, `v1 + v2 - v0` last, covered half the target: the second triangle was back-face culled by the console's rasterizer state). With every pixel shader forced red the rectangle passes now cover 100% of the target.
* A content pixel shader (`825a5674`): `rgb = clamp(attr0.xyz * attr2.yzw + attr1.yzw)`, `a = clamp(attr0.w * attr3.x + attr2.x) * attr1.x`, one compressed FP16 `exp mrt0`, no kill. Its vertex shader (`2927a7de`) exports param0 = vertex colour (unorm8x4), param1 = (an R8 byte at offset 7, then constants), params 2 and 3 constants, position = a matrix from the constant buffer times the vertex's float2 at offset 8 (values near +-32768, scale 3.05e-5). The vertices are 16-byte, indexed (u16, 30 to 4500 indices per draw) with white or dark colours, the constants near (1,1,1,1): so the expected output is vertex colour, i.e. white UI geometry, and nothing in the shaders explains zero.
* Forcing the vertex parameters 0 to 3 to 1.0 (`VICTORY_DEBUG_VS_PARAM=0,1,2,3`) with the real pixel shader still gives black targets, and with the pixel shader forced red the content draws showed a band only in the run before the depth-stencil format mapping (DXGI 19) was added, not after. So the prime suspect is the depth test now active on these draws (their depth state comes from 0x1b8 variants or the guessed default less-equal; a reversed-Z title cleared to 0 with a less-equal test passes nothing). A `VICTORY_DEBUG_NO_DEPTH=1` run was inconclusive (all dumps black, boot passes included: the run probably died early).

#### Split pixel inputs, depth, video (guessed path)

* The merged `lower.rs` splits a pixel input read two ways (perspective and flat) onto a fresh semantic index. That index was the "texcoord4" the vertex shader never wrote; declaring it as an undefined output made every such input read zero, which is why the content pixel shader (`rgb = colour * constant + constant` with the constant inputs flat) wrote (0,0,0,0). `HostShader::splits` now reports (original, new) indices and `Options::copy_params` makes the vertex shader write the new index with the original's values (`Guessed` passes them at the draw).
* No Bink video before frame ~1000: the draws of frames 695 to 830 sample no texture that changes (the boot passes sample the 4K target, the content draws sample none), and the only compute is the LUT.
* Debug switches added: `VICTORY_DEBUG_PS=white|alpha`, `VICTORY_DEBUG_PS_INPUTS` (not wired after the merge), `VICTORY_DEBUG_NO_DEPTH`, `VICTORY_DEBUG_NO_DEPTH_ATTACH`, `VICTORY_DEBUG_VS_PARAM`. `MTL_DEBUG_LAYER=1` overflows the title threads' stacks (a Metal call on a title thread), so it cannot be used; command buffers that fail now log `metal: command buffer failed`.

#### The logo, and why the content draws are invisible (constant buffer reads zero in their vertex shader)

* Debug switches `VICTORY_DEBUG_PS=in<N>` (raw pixel input N as the colour), `VICTORY_DEBUG_ONLY_INDEXED=1` (skip the non-indexed boot passes, so the content target holds the content draws alone) and `VICTORY_DEBUG_VS_POS=1` (vertex positions from the raw vertex xy times 1/32768 instead of the constant-buffer matrix).
* Content draws alone, real constant buffers: nothing is drawn even with every pixel shader forced red, so their vertex positions are degenerate. With `VICTORY_DEBUG_VS_POS=1` they draw: a dark background quad and then a large white vector logo (a "Y" crossing an ellipse and an "X", the first content the title shows), filling the target at that scale (the real matrix scales it down and places it).
* The vertex shader's constant-buffer dwords read zero on the GPU although the title memory holds them at submission (`VICTORY_TRACE_CONTENT`): `in2` (param 2 = s11..s14, expected (0,1,1,1)) is black; `in1.x` (the vertex's R8 byte) is correct. The boot passes' constant buffers (an identity matrix, 480 bytes) read fine with the same machinery, so the open difference is these content draws' tables (1536-byte buffers in a table at heap index ~680k to 980k, two root constants [0,0]), or the memory being reused before the GPU reads it. (Resolved below.)

#### First real frame: two causes of the zero constant buffers (resolved)

* **Root layout.** Guessed render pipelines gave the converter the host's root layout (`Binding::layout`) only for stages that read title memory. Without it the converter makes the user-data block a root CBV, a pointer in the argument buffer, while `translated_args` writes the dwords inline; so every guessed vertex and pixel shader read its user data (descriptor table offsets, root constants) through the address formed by user-data dwords 0 and 1, here 0. Seen by converting the translated vertex shader both ways and reading the AIR (`llvm-dis` on the bitcode in the metallib): without a root signature it loads a pointer from the argument buffer, with one it reads dword 8 inline. Compute pipelines always had the layout. Now render stages get it too. A stage with immediate samplers needs them in the layout (the converter refuses it otherwise, "Resource SMP ... not referenced by the root signature"): `RootLayout::static_samplers` carries the decoded `StaticSampler`s (`s<n>`, space 0) and `msc.rs` passes them as the root signature's static samplers. With it the boot passes' pixel shader (`9406103f`, which samples the content target) runs, and the presented image shows the menus.
* **Copied descriptor records.** The title copies view records into its shader-visible heap with plain stores, which no device call reports: at submission the host heap entry of a content table was all zero (read back from the Metal heap) while the title's record named the constant buffer. `NativeHost::remirror_tables` now checks, at each draw and dispatch, the first 16 entries of every bound table against the record each was last mirrored from (`NativeHeap::mirrored`) and skips unchanged records after a successful mirror. A buffer view whose address is still unmapped gets a zero-length null view, and any entry whose mirror attempt could not complete yet (`NativeHeap::pending_mirror`) is retried only once the record, the pool table's generation, or the registered-resource epoch changes (rather than re-encoding the null view at every draw). Root SRV/UAV lookups (`NativeHost::root_view`) likewise query `PoolTable::mapping_at` directly in `O(log n)` without rebuilding the full mapping snapshot when streaming updates the pool table between draws. The scan extends past 16 entries when shader metadata requires it. Views of resources the host has not realized yet are not made there.
* **Views of memory in no pool.** A buffer view whose address the pool table cannot resolve now writes the null view (size 0) into its host heap entry, and `mirror` still returns the error; it used to leave the entry as it was, so a descriptor the title reused for such a buffer kept showing the buffer it named before, and shaders read and wrote that memory, possibly given to something else since (a cause of writes landing elsewhere, issue #15). The entry is not marked mirrored by `mirror`. Image and sampler views the host cannot make still leave the entry as it was: a buffer is not something to put in a texture or sampler entry. A buffer view of address 0, which is what D3D12 writes for a null view, is the null view for good, whatever its size, and is marked mirrored: nothing can be mapped at 0 in this process (`__PAGEZERO` keeps page 0) and no buffer object is at 0, so no pool or wrap can ever resolve it. As memory in no pool it was tried again after every change to the pool table, which streaming makes between almost every submission in the world. Measured on one Mac (M4 Pro), on a branch that also checks whole heaps for shaders that index them (main checks only the bound tables' entries): 21,000 to 23,000 such tries a frame in the world in two runs without it, none in two runs with it, and the world at about 2.0 fps instead of 0.9 to 1.1. The gain on main has not been measured and is likely smaller.
* Ruled out on the way: the pool view of the constant buffer equals the title's bytes (no aliasing problem for the low, sub-4 GiB linear allocator), no mirror errors for it, the table's heap is the heap bound, residency (`VICTORY_METAL_RESIDENCY_SETS=1` changes nothing there).

Residency itself: what a descriptor heap names is resident through the heap's own `MTLResidencySet` (the heap buffer plus each allocation its entries name, added when first named and removed when the last entry naming it is overwritten), given to every command buffer that uses the heap, so residency follows the frame. Before, every allocation was named to every encoder with `useResource`: the world's heap names about 37,000 allocations, the kernel took about 155 ms per drawing render pass for that list whatever the pass's shaders, and a world frame with 35 such passes in one submission took over 5 s with the GPU idle (0.07 fps on an M5 Pro; 0.47 with the sets). The sets are the default only on a Mac with 32 GiB or more (a guess: the world's heap set holds 17 to 18 GB and no smaller Mac has run it); a smaller Mac keeps the per-encoder naming unless `VICTORY_METAL_RESIDENCY_SETS=heap` asks for the sets, and the choice is logged once at start-up. `VICTORY_METAL_RESIDENCY_SETS=0` gives the per-encoder naming back on any Mac. `VICTORY_METAL_RESIDENCY_SETS=1` also keeps the title's pool buffers in a set on every queue, which holds all of title memory (about 6.7 GB on a Series S profile) resident for the whole process and on an 18 GB Mac ended in `kIOGPUCommandBufferCallbackErrorTimeout` after the world load; by default the encoders name the pool buffers they use. A texture the title releases for good (its object's last reference, or the pool it was placed in) has its heap entries cleared (`NativeHost::release_resource`, counted as `heap_cleared` on the frame line): left alone they kept the host texture alive and resident until the title happened to write another view there. A cleared entry reads as one never written. It is written again when the title writes another record there, or makes a texture at the released object's address again (`register_texture`: the record is then checked against the new texture, as the title's record for it may equal the old one's). The old record alone is not mirrored again: in a run it named the next object at that address, an RGBA8 texture under a stale R8 view, and Metal asserts on such a view (the backend now refuses a view format Metal cannot cast the texture to, `gfx::castable`). A texture only replaced by another description at its address keeps its entries (the title may still read them; that release is a guess). The frame line logs the heap set's size (`heap_set_mb`). Not checked on a 16 or 18 GB Mac. Details in `docs/metal-backend.md`, "Residency (what a heap needs)".

Why the wired memory flips (#79, 48 GB M5 Pro, `VICTORY_METAL_REQUEST_RESIDENCY=1`, `VICTORY_TRACE_RESIDENCY=1`, memory sampled every 2 s): in a 220 s run the heap set grew to 18 to 19 GB (81,000 allocations) and the wired memory stayed at 22 to 23 GB for over 100 s of world frames at about one frame a second. It fell once, to 4.5 GB, with the IOAccelerator's "In use system memory" at 1.3 GB (all GPU memory, not only the set), in a 3.6 s window where the host committed one command buffer; it was back at 22 GB 4 s later when submissions resumed. In that window the frame was creating render pipelines one after another, 0.2 s each (about 30 `first bind` lines in frame 2262); the earlier 4.3 s stall (t = 79) had 1,400 `first bind` lines in 5 s. No residency call took over 0.28 s (removals: up to 6,500 commits a second, 30 ms in all); the slow ones (0.14 and 0.22 s) came right after the fall, when the set was made resident again. So the system un-wires an idle device's memory whatever `requestResidency` asked: the fall is a sign of a stall, not its cause, and shrinking the set or keeping it requested does not stop it. `recommendedMaxWorkingSetSize` on that Mac is 38 GB, twice the set, so the set's size is not over the driver's limit either. The stalls are on the CPU: pipeline creation, and the title (the earlier run's last 30 s had no command buffer in flight and no host call logged before the title's own watchdog fault at image+0x55e56f8, which this run hit too at t = 218). Not checked: what the title waits on in that last silence; whether the 0.2 s pipelines are misses of the binary archive (that run's cache was filled by another branch).
* With both, a normal run (no debug switch) draws the content target: a dark (32, 32, 44) background and the white button prompts "ACCOUNT PICKER (Y)  SCREEN NARRATION (OFF) (X)  PLAYER (A)" at the bottom right, placed and scaled by the real matrix (the "logo" under `VICTORY_DEBUG_VS_POS=1` was these glyphs at 1/32768 scale). With static samplers in the layout the presented buffer shows the same frames (scratchpad `presented-finish-setup.png`).
* Texture formats added for the menus: DXGI 2 (`R32G32B32A32_FLOAT`), 56 (`R16_UNORM`), 61 (`R8_UNORM`) and 65 (`A8_UNORM`, a 2048x2048 UI atlas). Views of them had no host texture before. With them, runs reach the legal-notices menu (about 90 s with `VICTORY_KEYS`). Still open: the body font's x-height-only letters (a c e m n o r s u v w x z) draw as filled boxes, while letters with ascenders or descenders are right. The title never calls `ClearDepthStencilView`, and stencil-only clears would be dropped anyway (`glue::clear_dsv` acts on flag 1 only), so stencil-then-cover text was the first suspect; ruled out since: no pipeline enables stencil (`VICTORY_TRACE_PIPES=1` logs each guessed pipeline's depth, stencil and blend state), and the atlas reaches the host (below). The cause was the blend state (below).
* Debug runs: the content draws start about 5 s before the boot-wait breakpoint, so dump every second (`VICTORY_DUMP_TARGETS=<dir>,30,1` with `VICTORY_TRACE_TARGETS=1`, which the dumps need) and check that the target was drawn before the last dump. `VICTORY_DEBUG_PS=in<N>` reads the original input row, which a split input may have moved: a black result does not mean the vertex shader wrote 0.

#### Green movie frames: textures written after their first upload (resolved)

* The boot movie (Bink, about 40 to 55 s with `VICTORY_KEYS`) is one draw a frame into a 3840x2160 sRGB target (DXGI 29), sampling three `R8_UNORM` planes (Y 3840x2160, U and V 1920x1080), three sets of them in turn. Each plane was realized and uploaded blank (all zero) about 3 s before the title decoded into it; the scanner then saw the change one round later, and by then a round over every watched texture took about 2 s (900 textures, 2.4 GB). So the first frames of each set showed zero planes, which convert to solid (0, 193, 0), and later frames an old set: the green screens and flicker.
* Fix: a texture whose memory changed after its first upload, or that was blank at it, is hot for the next 600 submissions: each submission probes 8 pages spread over its memory (`textures::probe`) and syncs it when they changed. A movie's planes, rewritten whole, are caught at the submission that samples them. A hot texture of at most 4 MiB that changed more than once (a glyph cache) is probed whole instead, so a few new glyphs are seen at the next submission too; one written once after a blank upload (streaming) is sampled, as probing all of those whole cost frames. Shortcut: a small write into a sampled texture still waits for the scanner's round. The probes of one submission are bounded by time (`PROBE_BUDGET`, 2 ms, at least 32 textures, the cursor resuming where the last submission stopped), not by a count: a limit of 32 probes a submission came back as green flicker, because the boot movie runs with 500 to 700 hot textures (blank uploads of textures the title streams in), so a plane written after its blank upload waited 16 to 20 submissions to be seen; thousands of hot textures later in the load are what the bound is for. With it, no dump from 36 to 55 s has a green pixel and the movie plays (scratchpad `movie_t3_41_c655e0000.png`).
* Tests: the `native::texture_tests` share one `Title`, and two things in it broke the hot-texture tests' "at once" and "at the next submission" asserts now and then (8% of full-suite runs under a parallel build for `a_texture_rewritten_whole_is_read_whole_only_now_and_then`, 1.5% for `a_texture_uploaded_blank_is_probed_at_each_submission`). First, hot textures outlive the test that made them (the 200 of `a_hot_texture_is_probed_at_each_submission_among_hundreds` stay hot for `HOT` submissions after it releases `AGING`): under load a round cut short by `PROBE_BUDGET` after 32 probes skipped the test's texture (traced: 215 hot keys, 61 probed, 154 not, that one among them). Second, other tests' submissions run between a test's own `with_title` calls and probe its texture too: one uploaded the write the samples missed, the test's next submission spent the follow-up whole read on it (unchanged, so back to `WHOLE_EVERY`), and the next write then waited. A test that asserts what its next submission sees holds the title lock from its first write to its last assert and syncs with `sync_dirty_without_probe_deadline`; holding `AGING` longer would not help, since the textures stay hot after it and the tests that submit in between do not take it.
* Came back (Series S, 2026-11): the planes are realized and uploaded blank 25 to 40 s before the movie decodes into them (the movie starts anywhere from 34 to 65 s, and in some runs not within 2 minutes), so by then they had spent their 600 hot submissions and gone cold, and the first frames waited for the scanner again: 35 green frames counted at the movie's start (`VICTORY_COUNT_GREEN=1`), with the planes' first content in memory 1.5 and 2.5 s before it was uploaded (a plane trace comparing memory changes with uploads). Fix: a texture uploaded blank is kept in a list of its own (`Title::blank_keys`) until it is written, however long that takes, and probed at every submission after the hot ones with a budget of its own (2 ms, at least 32), by one 256-byte read in its middle (`probe_nonzero`), so a few hundred blank textures are all reached in one submission. Shortcut: a first write that misses the middle of a blank texture waits for the scanner.
* Cooling by submission serial (`Title::sync_serial`, `Tracked::hot_until`): when world streaming puts thousands of textures into `Title::hot_keys` (18,900 observed), decrementing a per-texture counter only when `probe_round` reached it kept each texture hot for `600 * (hot_keys / probed)` submissions (minutes to over an hour). Each non-blank re-upload now records `sync_serial + HOT`, and `probe_round` evicts entries whose `hot_until < sync_serial` without reading memory or spending a probe while keeping wrapped removals inside the unvisited prefix (`native::texture_tests::hot_textures_cool_by_submissions_even_when_probe_round_is_capped`).
* The movie also ran at 12 to 20 presents a second (30 before and after it): probing the six 2 MiB chroma planes whole at every submission took about 15 ms a submission, on top of the uploads (about 0.4 s a second for 250 MiB of planes). A hot texture whose writes show in its samples (a movie's planes, rewritten whole) is now read whole only every `WHOLE_EVERY` (8) probes; a change found then that the samples missed (a glyph) turns whole reads at every probe back on, as does a change only the scanner saw. (Since then the choice is made at each upload and covers streamed textures: "Texture sync without the host's lock" below.) Its whole hash is compared with the hash of its last upload. Headless Series S, both slots taken, during the movie's 16 s: gaps over 50 ms between presents went from 72 to 5 (longest 134 to 112 ms), green frames from 1 to 0 (that run's movie started at 30 s, before its planes cooled).
* The UI atlas (A8 2048x2048) is not affected: it is blank until the title draws its first text, and is uploaded within 0.1 s of that.
* `VICTORY_TRACE_CONTENT=<dir>,<seconds>,<format>` traces only the draws into render targets of that title format, boot passes included (the movie's draws are skipped otherwise), and writes 8-bit textures (A8, R8) as grey PPMs: the title's memory once (`<key>_title_<draw>.ppm`) and the host texture at each draw (`<key>_host.ppm`).
* Dumps of the presented image are read without waiting for the GPU: some show the top of the frame black in 1024-pixel steps while the movie target is complete. That is the read racing the draw, not a rendering fault.

#### Texture sync without the host's lock (27 November, Series S, headless)

* In the world a submission (`ExecuteCommandLists`, some 30 a frame) spent nearly all its time bringing textures in step with title memory, under the host's `TITLE` lock that every other title thread's call waits for. `VICTORY_TRACE_TEXTURE_SYNC=1` showed about 1,000 textures and 500 MiB a second read whole, hashed and detiled there for minutes, 25 to 70% of them unchanged.
* A submission now lists the textures it must read under the lock (`Title::sync_jobs`: those never uploaded, and those the scanner or a probe found changed), reads, hashes and detiles them without it on up to 6 threads (`textures::prepare`), and takes the lock again only to upload them (`Title::sync_apply`) and to run its lists. `glue::submit` keeps submissions in order with a lock of its own, taken before `TITLE` and never inside it. A texture released while it was read is told by a serial number (its key, an object's address or a placement, names another texture later) and its reading is dropped. The unit tests share one title and count their own uploads, so there the lock is kept throughout (`cfg!(test)`).
* What a submission read is what the scanner compares with (`Watch::seen`). The scanner used to report every new texture and every write a probe had caught already, and each report was the whole texture read once more. A texture made anew under a released key starts over there, so a render target's first sight is its baseline again.
* A hot texture's whole check (the read that catches a write its 8 sampled pages miss) is one of those reads without the lock, not a hash under it. Whether a texture's samples showed its last write is decided at each upload, whoever found the write. If they did (a movie's planes, a streamed texture written whole after its blank upload), it is read whole once in `WHOLE_EVERY` (8) probes, and only within `WHOLE_SPARE` (1 MiB a submission; one that is due and not read stays due). If they did not (a glyph, or a streamed GFX10 mipmapped texture whose coarse mip tail at the start of the allocation has unused padding at offset 0 and unmapped fine mips across the upper 75%), it is read whole at the very next probe to catch an immediate burst; once a follow-up whole read finds nothing changed, `sampled_sees_writes` returns to periodic `WHOLE_EVERY` checks within `WHOLE_SPARE`, doubling after each further unchanged whole check up to `WHOLE_EVERY << WHOLE_BACKOFF_MAX` (256 probes) while the scanner continues watching every texture. Until now every texture written once after a blank upload was read whole at every probe, and then every texture whose write missed the samples stayed on every-probe whole reads across all 600 hot probes even when unchanged.
* A texture whose whole read met an unmapped or protected range (`Tracked::partial`, `Watch::partial`) skips the initial whole-span `mach_vm_read_overwrite` on later whole reads and scanner rounds, walking readable `mach_vm_region` spans directly and zeroing only the unmapped gaps instead of failing the full-range read, zeroing the whole buffer, and reading the mapped prefix a second time; once every page in its range is readable again, `partial` clears.
* Shortcut: an atlas whose first glyphs reach a sampled page gets its next ones up to 8 probes late, once (per-sample hashes would tell a partial write from a whole one); after repeated unchanged whole checks, a solitary write that misses every sampled page waits for the backed-off periodic check or the scanner round. Guesses, marked in the code: `WHOLE_BUDGET` (8 MiB) and `WHOLE_SPARE` (1 MiB) of whole reads a submission, `WHOLE_BACKOFF_MAX` (5 doubling steps, up to 256 probes), `SYNC_THREADS` (6), `SYNC_BATCH` (256 MiB read and detiled at a time).
* Measured on warm runs with `VICTORY_HOLD_BYTE` on the init watchdog's flag: world frames presented from 30 to 150 s after the world starts (its first indirect rectangle-list draw). 0.38 a second before (46 frames), 1.17 with the reads off the lock (140), 1.73 with the whole-check rule and the scanner's baseline as well (208; that build also has main's G-buffer and upscale fixes). A second of world in the last one's trace: 45 submissions read 1,670 textures and 840 MiB (290 for the first time, 680 unchanged, 780 with memory that could not be read at once) in 0.41 s without the lock (the reading threads: 0.5 s reading, 0.44 s hashing, 0.24 s detiling); under the lock, 28 ms probing the hot textures (18,900 of them at most), 81 ms probing the blank ones, 94 ms uploading. Whole checks: 790 and 410 MiB a second, 1,630 and 330 MiB before the rule (17 and 9 MiB a submission, 44 and 9 MiB before). `WHOLE_BUDGET` bounds both: the rule halved their number, not their bytes, and what is still read at every probe are textures whose last write their samples did not show.
* `victory_formats::texture::Layout::walk` precomputes column block offsets and low-bit contiguous micro-runs (`1` to `32` bytes) for XOR swizzle modes (`_X`, `_T`) and hoists the per-slice and per-row address/XOR terms out of the inner column loop, cutting a `2048x2048` `SW_64KB_S_X` detile to ~1.3 ms at 4 B/element and ~5.6 ms at 16 B/element (`texture::xor_tests::walk_matches_per_element_xor_address_across_modes`, `texture::xor_tests::partial_xor_runs_with_pipe_bank_xor_round_trip_every_element_size`).
* The suspect that two descriptions placed at one address release each other in turn ("Placed textures are released" below): the trace counts the placements that replace another at their address, and those of them that had been at that address before. In 260 s of world, 32,291 were replaced and 5,715 of them (18%) were returns, against 62,656 first uploads. It happens, but it is under a tenth of the first uploads and not what makes the world slow.
* Still open: whole checks of textures whose writes the samples miss fill `WHOLE_BUDGET` at every submission and mostly find nothing (the scanner reads the same textures on its own thread); nearly half the textures read have memory that cannot be read at once and are read page by page; the submitting thread waits for its reads; the probes are budgets a submission (2 ms for the hot textures, 2 ms for the blank ones, at least 32 of each), which at 30 submissions a frame is over 100 ms a frame; hot textures cool one probe at a time, so thousands stay hot for minutes; `mach_vm_read_overwrite` on 6 threads reads about twice as fast as on one, not six times.

#### Text drawn as boxes: blend from the 0x1c8 register image (resolved)

* The text pixel shaders (`ac967781...` and the body font's) write the vertex colour unmultiplied and the atlas coverage in alpha, so they need blending: guessed pipelines had none, and every glyph quad wrote a solid box into the UI target's colour while its alpha held the right glyph. The glyphs that looked right were the ones whose quads later draws covered. The title draws text with the base pipeline 0x1c8 made, not one of the variants 0x1b8 makes (43,000 of those at boot, few ever bound), so the state is in the base's register image.
* Pair 0 of a description slot is the register image (0x17c to 0x208 bytes). Decoded from 23,000 images: word 1 is 4 or 6, and the eight `CB_BLEND<n>_CONTROL` registers start at byte 0xd0 or 0x124 (every image has DISABLE_ROP3, bit 31, in all eight, and 0xffffffff after them; then three words and the eight `SX_MRT<n>_BLEND_OPT`, which agree: 0x06000600 is "blend disabled"). The text pipeline's is `e5010504`, straight alpha. Common values: `80000000` (off), `e5010504`, `f1010f01` (dual source), `e5040504`, `e0010161` (max). `Graphics::guess_from` reads them into the guessed state (hardware factors are D3D12's less one; combine functions add, subtract, min, max, reverse subtract), and variants' blend records apply over it.
* With it the legal-notices screen is right: text, the purple gradient background and the faded art behind it (scratchpad `legal-notices-blend.png`); the white block and stepped black areas were opaque quads too. Still guessed: culling, depth, stencil and the write mask (the word before the blend registers is not the target mask: it is 0 for the text pipeline).
* **Dual-source blending** (`f1010f01` above, 27 November). Target 0's blend can name `SRC1_*` / `INV_SRC1_*` factors: the pixel shader then gives the target two outputs, the second (`SV_Target1`) a blend source only, so it writes one more output than the pipeline has targets. Metal writes that as `[[color(0) index(1)]]`, which its converter only emits when `IRCompilerSetDualSourceBlendingConfiguration` enables it; without that, Metal refuses the pipeline with "Fragment shader does not write to render target color(0), index(1) that is required for blending" and every draw with it is dropped (two such pipelines, every run, before the fix; their pixel shaders write targets 0 and 1 into one `Rgba16Float` target). `state::dual_source` decides per pipeline (one more shader output than targets, in order, and target 0's blend reads it), and the converter's cache is keyed by the answer: the same shader converts differently for the two cases.
* Two fixes found on the way: title threads get at least 8 MiB of stack (`host_posix::MIN_STACK`; the title's 1 MiB overflowed while the host built a Metal pipeline at the first guessed draw), and a released pipeline object's host pipelines are forgotten (`glue::pipeline_released`), as a new object at a freed one's address got the old one's pipeline.
* `VICTORY_DUMP_DESC_DIR=<dir>` (with `VICTORY_GUESS_GRAPHICS=1`) writes each guessed pipeline's register image as `regs_<vs>_<ps>_<address>.bin`.

#### Indirect mesh dispatches (`DispatchMeshX`)

`DispatchMeshX(x, flags, indirect, payload)` (list method 0x310) with a non-zero `indirect`
runs as a one-record `ExecuteIndirect` of a mesh dispatch: on the native host through
`glue::execute_indirect` (a Metal indirect mesh draw reading the record in title memory), on the
D3D12 host through `State::execute_indirect` (kind 10). GPU addresses are title addresses, so
`indirect` is used as it is. Guesses, not confirmed on a console: the record is
`D3D12_DISPATCH_MESH_ARGUMENTS` (three 32-bit group counts, 12 bytes), `x` is unused when
`indirect` is set, and `flags` and `payload` change nothing the host needs. "indirect mesh
dispatches dropped" (D3D12) and "an indirect mesh dispatch" (native) now mean the host could not
run it.

Direct mesh dispatches run too (#353): `DispatchMesh(x, y, z)` (method 0x3f8) and `DispatchMeshX` with `indirect` 0 go to `glue::dispatch_mesh`, which checks the bound pipeline as the indirect path does (`graphics(.., mesh = true)`) and encodes `NativeList::dispatch_mesh`. `DispatchMeshX` runs `[x, 1, 1]` groups, the D3D12 host's guess (`flags` and `payload` ignored). Without a mesh pipeline the dispatch is dropped for the reason `graphics` logs and counted in the frame line's `mesh=`. Before this, every direct one was dropped: a tester's prologue run (#343, Series S) counted `mesh=5` by frame 2048. Tests: `a_direct_dispatch_mesh_x_draws_its_groups`, `a_direct_dispatch_mesh_draws_its_groups`.

#### Dropped work is logged loudly

Every kind of work the native host drops now logs one `warn!` the first time it
happens ("native host drops work: ..."), and each drop still counts in the
periodic "native draws so far" line. The kinds are: a draw or dispatch with no
host pipeline (refused or not translated), a topology the host does not draw,
an indexed rectangle list, a guessed mesh pipeline, mesh and indirect mesh
dispatches, texture copies and resolves, and indirect executions (one warning
per signature). A texture in a format the host cannot realize warns once per
format value, with its size and flags. No switch is needed; `RUST_LOG` filters
them as usual.

Two more kinds are dropped one command at a time when a piece is submitted
(`NativeHost::execute`), where the whole piece used to be refused: a draw or
dispatch that reads a root parameter the title never set, and a draw or clear
none of whose targets has a host texture (Metal opens no render pass without
an attachment). The rest of the piece still runs, and so do its value writes
and fence signals: a queue waiting for one of those values used to be held for
good. A dropped command still ends the pass it would have ended. Two draws
into the same targets with a dropped dispatch between them stay two render
passes (likewise two dispatches around a dropped draw, and blits), so the
backend still orders their pool-buffer accesses; a dropped command of the same
kind as both its neighbours parts nothing, as before. These two are counted
per host (`NativeHost::unset_roots`, `NativeHost::no_targets`). The unset-root
drops are in the "native draws so far" line as `dropped_unset_root` (with
`dropped_unmapped` and `dropped_failed` for the kinds below) and warn without
a switch, the first time for each root parameter (at most 16); the no-target
drops warn once with `VICTORY_TRACE_TARGETS=1`. Unknown: what the console
gives a shader in a root slot the title set nothing in; the host draws nothing
for it, which is a fallback, not the console's behaviour.

Root parameter 255 was not such a slot (#258). It is the translator's own
view-stride table (`victory_shader::emit::VIEW_STRIDES`, root SRV `t255`): a
shader that takes a buffer view from the resource heap itself looks up the
entry's stride there before it reads the entry, and reads the host's null
buffer view where the table holds 0 (`emit`, "Buffer views at computed heap
indices"). The native host bound no such table, so every draw and dispatch of
such a shader was dropped at submission, counted only in `unset_roots` and
warned about only with `VICTORY_TRACE_TARGETS=1`; the title's culling shaders
are among them. Now each resource heap has a stride table
(`NativeHost::view_strides`: a dword per heap index up to `HEAP_VIEWS`, 8 MiB,
from `descriptors::stride_word`, made the first time such a shader is bound
with the heap, from the title's records, and kept current by
`NativeHost::mirror`), bound where the shader's layout has `t255`. A resource
heap is as long as the translator's index clamp (`HEAP_VIEWS` entries of 24
bytes, 48 MiB of zeroed host memory a heap, as the D3D12 host's heaps are),
with null buffer views at `NULL_BUFFER_VIEW` and below it: a shader's clamped
index and its null view must be entries the heap has. shortcut: an entry the
title copies as bytes outside a bound table keeps its old stride until it is
mirrored, and a record the title zeroes keeps its stride (the table check
skips empty records). Not measured: how much of the longer heap and of the
table becomes resident. Tests:
`heap_tests::a_compute_shader_stores_through_a_buffer_view_it_takes_from_the_heap`
(a compute program stores through a buffer view it loads from the heap by
index, on Metal) and `descriptors::tests::stride_words_of_buffer_views`.

A third kind is dropped the same way: any command (draw, dispatch, copy, fill
or value write) that names title memory outside every mapped pool ("title
address ... is not in a mapped pool"). The GPU cannot reach that memory, so
the command is still not encoded and causes no GPU fault, but the rest of the
piece runs. Refusing the whole piece also dropped its valid dispatches, copies
and value writes, so the title read their buffers stale (issue #15: 229 such
refusals in one story load). These are counted per host
(`NativeHost::unmapped`). The first one warns without any switch, as the
refused piece did, and each one is logged at debug level with its address.

Every other error the host meets preparing one command (root arguments that are not what the
shader reads them as, a resource heap with no room for the view pairs, a buffer or page table
the backend did not make) now drops that command alone as well, counted in
`NativeHost::failed`, with a warning the first time and each at debug level. It used to refuse
the whole piece, with its value writes, and a queue waiting for one of those values was held.

#### Title memory view pairs of released pools are reused

The pool-offset contract (`native/addr.rs`) has 255 view pairs per resource heap, one per
`(pool, 2 GiB chunk)`. They used to be handed out for good: once 255 were taken, pages of every
later pool read as the null view (zeros, writes dropped), and a released pool kept its pair, whose
buffer a heap that wrote its pairs later tried to wrap over the freed mapping. Now a released
pool's pairs are freed (`Pairs::release`, when the host next takes the pool table's released
pools) and the next pool takes the first free one. The pairs carry an epoch that changes whenever
an existing pair changes; a heap whose pairs were written in another epoch writes them all again
(a freed one as the null view). When all are taken the host warns once.

#### Placed textures are released

A texture the title places in its own memory (`CreateShaderResourceViewPlaced`) has no resource
object, so nothing released it: its host texture and its entry in the scanner's watch list stayed
for the whole run. The host now keeps the key it placed at each address
(`Title::texture_placed`). When a view places another description at the same address, the old
texture is released (guess: the title is done with it once it places another there; a view of the
old description still in a heap would name a released texture). Every texture placed in a pool is
released at the first submission after the pool is (`NativeHost::released_pools`).

#### A byte budget on host texture copies (#294)

Every host texture copy of a title texture is made when a view first names it (`Title::realize`,
from `glue::mirror` for every view the device writes, staging heaps included) and stayed until the
title released the texture. Nothing bounded their total: on a 16 GB M4, Series S (#8), the host made
about 13,000 copies during the first load after the menus while the footprint grew about 7.6 GB,
and Metal then failed ordinary command buffers with `kIOGPUCommandBufferCallbackErrorOutOfMemory`.

* **The budget.** Off by default, until there are runs on more machines:
  `VICTORY_METAL_TEXTURE_BUDGET_MB=<n>` turns it on at `n` MiB (`0` or anything not a number: off;
  `texture_budget_mib` on the `native host: Metal` startup line). The figure to try is logged next
  to it as `suggested_texture_budget_mib`: Metal's `recommendedMaxWorkingSetSize` less the title's
  memory, at least 1536 MiB (`MIN_TEXTURE_BUDGET`; guess: about what a world frame samples). 4348
  MiB on that M4 with Series S; Series X on 16 GB gets the floor. A copy counts its size in title
  memory.
* **What may go.** At each submission over the budget (`Title::keep_to_budget`), before its
  textures are synced, the least recently used copies that may go are given up until under it.
  A copy may go when it is a sampled texture filled only from title memory (not a render target,
  depth or GPU-copied texture, nor one ever shown through an unordered access view, whose host
  texture may hold what title memory does not), no heap entry shows it, and no recorded or
  submitted list holds it (the host's own two references are all there are). Shaders index the
  heaps themselves (#79), so a copy an entry shows counts as used however long no bound table
  named it; D3D12 forbids rewriting an entry that submitted work still reads, so a copy no entry
  shows is not reached by work in flight, and Metal retains what a committed command buffer names
  directly. A check that cannot get under the budget waits 64 submissions before the next
  (`RECHECK`). "Least recently used" is the last submission a view, a copy or a check saw it
  shown at.
* **Coming back.** A given-up copy goes the way a released one does (`NativeHost::evict_texture`,
  through `release_resource`), so it leaves the heap residency sets with its last reference and
  does not work against `VICTORY_METAL_RESIDENCY_SETS=heap`. Its description goes back to
  `Title::pending`: the next view the device writes makes it again and it is uploaded at the next
  submission. A record the title copies into a heap with plain stores is mirrored only when a
  list binding its table is encoded (`remirror_tables`, which makes no texture), after the sync.
  An entry naming a copy not made then is cleared, and a shader samples a cleared entry as
  transparent black: `(0, 0, 0, 0)` on an M2 Pro, macOS 26.6
  (`sampling_a_cleared_entry_reads_transparent_black` in `victory-metal/tests/resources.rs`; a
  measurement, not a documented Metal guarantee). So before a submission's sync, when over the
  budget or with any copy given up, the host reads the records of every table the submission's
  draws and dispatches bind, and those of work held behind fences (`Title::keep_named`,
  `NativeHost::resources_named`, the same entries `remirror_tables` checks): a given-up copy they
  name is made again and uploaded at that sync, and none they name is given up at it. A draw then
  samples it in the same submission, at the cost of that upload. The names read stay out of the
  budget's reach until the submission is with its queue (`Title::let_named_go` after
  `submit_synced`; counted per submission, so one submission letting go does not release what
  another still names): `glue::submit` lets the title lock go while it reads textures between the
  two. Today `glue::submit` also runs one submission at a time (`ORDER`), so no other sync could
  come in between; the title no longer depends on that
  (`a_texture_named_by_a_submission_in_flight_is_not_given_up_by_another`). The lists that
  `keep_named` reads are snapshots (`TitleList::segments`), and a snapshot clones thousands of
  `Rc`s (heaps, pipelines, textures) that the recording threads clone and drop under the title
  lock. `Rc` counts are not atomic, so a snapshot taken or dropped without that lock corrupts them
  (a double free or use after free: the menus crashed in 43-54 s with a malloc abort or a SIGSEGV
  when an earlier version of this change took the snapshot before the lock). Each snapshot is now
  taken and dropped under the lock: once for `keep_named`, once for the submission.
  `TitleList::segments` takes a `&Title`, which exists only inside `glue::with_title`, so taking
  one without the lock does not compile. The tests cannot show the race: the `cfg(test)` submit
  keeps the lock throughout, and nothing stops a snapshot made under the lock from being moved out
  through a captured variable and dropped later. Under the budget with
  nothing given up, nothing is read (except at the submission the trace logs at, once a second,
  so its figures match a run with the budget on). Left: a record the title changes after that read (between submit and
  encoding, for work held behind a fence) and an entry mirrored outside a submission. Those entries
  ask for the copy (`NativeHost::take_wanted`) and are cleared until the next submission makes it.
* **Trace.** With `VICTORY_TRACE_TEXTURE_SYNC=1`, once a second: `host texture copies: <MiB>,
  budget <MiB>; <MiB> shown by heap entries, named by the submission's tables or viewed for
  unordered access, <MiB> could be given up; given up in the last second <n> (<MiB>), made again <n>;
  <MiB> of distinct title memory under the copies, <MiB> of copies outside the title's pools; in
  the last second <n> textures shared an identical copy (<n> share one now), <n> placed copies
  another placement covered were given up (<MiB>), <n> covered still in use` ("One copy per
  range" below). If
  the copies are mostly shown by heap entries, the budget frees little and the levers are the ones
  #79 names (one copy instead of two, fewer top mips).
* **What the count counts, and why it can pass the title's memory.** One copy per title texture,
  by its layout's size in title memory (all mips, slices and planes; never more than the
  footprint the title allocated, so untiling, format expansion and Metal's padding do not enter
  it). A title texture is a resource object (its address) or, for a view of memory the title
  placed itself, its address and resource description (`placed_key`; the view's own description,
  format or mip range, is not part of it). Views of one object, or of one placed description, so
  share one host texture and are counted once; the entry a view writes is a Metal texture view
  (`create_texture_view`), with no storage of its own. But the host keeps no map from memory to copies:
  two objects over the same memory (placed resources aliasing one heap offset), or a placed
  description starting inside another texture's memory, each get a host texture of their own and
  each are counted. A placed copy goes only when another description is placed at its exact
  address (the texture sync line's "replaced" counts) or with its pool, so memory the title
  streams something else into at other addresses keeps the old copy too, and an entry still
  holding the old record keeps it shown. So the copies can add up to more than the title's
  memory (10.1 GB against 7.8 GB on a 24 GB M5 Pro, Series S, #301) without a counting error: the
  host then holds those bytes more than once. The last two trace figures tell which: distinct
  memory well below the copies means overlapping copies; copies outside the pools are textures
  in memory the host allocated itself. Copies in a pool are placed by their pool offset
  (`Title::memory_under_copies`), so pages the title maps at two addresses count once.
* Tests (`texture_tests`): `the_least_recently_used_textures_go_first_until_the_excess_is_freed`,
  `the_texture_budget_is_the_working_set_the_title_leaves`,
  `over_budget_only_textures_nothing_uses_are_given_up_and_come_back` (a copy a heap entry shows,
  one a recorded list holds, a render target and one viewed for unordered access stay; the held one
  goes once the list is gone; the one given up comes back with its texels),
  `an_entry_copied_with_plain_stores_gets_an_evicted_texture_back` (the fallback),
  `a_texture_given_up_is_back_for_the_draw_whose_table_names_it` and
  `a_texture_the_submission_names_is_not_given_up_under_its_draw` (a draw through a record stored
  with plain stores reads the texture, not black, in the same submission),
  `views_of_one_texture_share_one_copy_counted_once`,
  `placed_views_of_one_description_share_one_key`,
  `textures_over_one_memory_each_hold_a_copy_of_their_own`, `distinct_bytes_counts_overlaps_once`,
  `a_copy_outside_the_titles_pools_is_logged_so`,
  and `a_textures_counted_size_is_never_more_than_its_footprint` (in `resource.rs`).

#### One copy per range of title memory (#356)

The copies held the same title memory several times over: during a world load on a 24 GB M4 Pro
(Series S, budget off), 8.5 GB of copies over 1.7 GB of distinct title memory, about 11 bytes of
copies per new byte in the last 12 s, until macOS ran out of memory (#301). Two changes:

* **Identical copies share one texture** (`Title::realize`, `identical_copy`). A title texture
  laid out exactly as a live copy that starts at the same memory (same size, layout, dimensions,
  unordered-access flag and host format; neither or both B4G4R4A4, which is converted on upload;
  not a render target, depth or GPU-written) registers that copy's texture under its own key
  (`Title::copies_of`) instead of making one. Both would have been detiled from the same bytes into
  the same format, so a draw samples the same texels either way. It is not counted again. A GPU
  write through either key makes the shared copy the GPU's (`gpu_written` goes to the owner), as
  the R9G9B9E5 view of an R32_UINT object already did; before, the other key's own copy went on
  being filled from title memory. When the owner's title texture goes, the first key that shared
  it takes the copy over (`texture_released`): its entries already show that texture, so nothing
  is cleared, made or uploaded again, and the others share the new owner's. An owner that others
  share is never given up by the budget (the extra references make `evictable` false). Sharing
  goes by title address: a copy left over memory the title freed and mapped again would be shared
  too. The title releases its objects before their memory, and placed copies go with their pool;
  the tests, whose pools reuse addresses under one host, release what earlier tests left there
  (`texture_tests::pool`).
* **A placed copy goes when another placement covers it at another start, once nothing uses it**
  (`Title::give_up_overlapped` marks it from `texture_placed`; `give_up_covered` gives it up at a
  sync). Before, it went only when a description was placed at its exact address, or with its
  pool, so memory the title streamed something else into kept the old copies. Guess: the title is
  done with a placed view once it places another over its memory; for resource objects D3D12
  makes an older alias's contents undefined once another is used, but for views of raw memory
  the console's rule is not known. A covered copy is given up only under the budget's own rule
  (`evictable`: no heap entry shows it, no recorded or submitted list holds it) and when no
  submission in flight names it. Giving up a copy an entry shows would clear that entry (and drop
  the Metal reference it holds) under work the GPU may still be running; so one still shown waits
  until the title writes another record over the entry. A view or table that names a copy given
  up gets it made again (one late upload; the draw in the same submission reads it through
  `keep_named`), and a view written again at the copy's own address is not a new placement, so
  two placements in use do not keep covering each other. Never marked: copies that hold what
  title memory does not (render targets, depth, GPU-written, ever viewed for unordered access).
  Placements are the only trigger: an object created over placed copies does not cover them. The
  trace's "covered still in use" counts those waiting on an entry.
* What it cannot share: copies at other starts (separately placed mips or slices of one chain)
  and copies of one memory in other layouts or formats still hold it once each; the trace's
  distinct figure shows what is left.
* Tests (`texture_tests`): `an_identical_texture_over_the_same_memory_shares_its_copy`,
  `a_texture_sharing_a_released_copy_takes_it_over`,
  `a_placed_copy_another_placement_covers_is_given_up_once_unused`.

#### Texture views over 2D arrays

Object and placed `TEXTURE2DARRAY` SRVs select the requested mip levels and array slices.
UAVs select one mip and the requested slices. Level 0 and slice 0 of the Metal view refer to
the first mip and slice in the descriptor. A count of `UINT_MAX` includes the remaining mips
or slices from that starting index. The host refuses an array view with an empty range.

A shader resource view created with a null description binds the registered texture itself,
including its native format, all mips and all slices. This also preserves the format chosen
when a typeless texture is realized as a depth target. Reusing a slot for such a view replaces
the previous texture.

A 2D shader resource view with `MipLevels = UINT_MAX` exposes every mip from
`MostDetailedMip` onward. The native decoder preserves that count for ordinary and placed
views so Metal clamps it to the remaining levels. Previously the decoder returned zero and
the host bound the whole texture, so a shader reading level 0 saw the base mip even when
the view started farther down the chain.

#### Typed and structured buffer views lazily carry a Metal `texture_buffer`

`victory-shader` lowers MUBUF `buffer_load_format_*`, `buffer_store_format_*`, and their
`getDimensions` size queries to DXIL `TypedBuffer` (`bufferLoad.f32` / `bufferStore.f32`), which the
Metal Shader Converter compiles (tested on a small synthetic DXIL shader) to
`air.read_texture_buffer_1d.v4f32`, `air.write_texture_buffer_1d.v4f32`, and
`air.get_width_texture_buffer_1d` reading `textureViewID` (word 1) of the 24-byte
`IRDescriptorTableEntry`. `Metal::write_buffer` previously left word 1 at `0` (null
`MTLResourceID`) for every buffer view, so formatted buffer stores wrote nothing and formatted
buffer loads and size queries returned `0` (`#255`).

To keep `NativeHost::mirror` free of Metal texture allocations across ~40,000-entry world heaps,
`mirror` continues to write `[gpu, 0, size]` via `write_buffer` and marks two pending bits per
entry (`list::PendingTyped`, ~10 KiB for 40,000 entries, `descriptors::typed_buffer`) for typed
buffer SRVs/UAVs and for 4-, 8-, and 16-byte structured buffer SRVs/UAVs (guess, marked in
`descriptors.rs`: on RDNA2 a structured `V#` of 4-, 8-, or 16-byte elements carries a 1-, 2-, or
4-dword 32-bit data format, so shaders may access the same descriptor with both
`buffer_store_format_*` / `buffer_load_format_*` and `buffer_store_dword*` / `buffer_load_dword*`).
`Binding::of` scans the module's DXIL symbol and constant tables once
(`list::dxil_uses_typed_buffers`) to flag shaders that use `bufferLoad`, `bufferStore`, or
`TypedBuffer` `getDimensions`. At draw/dispatch submission, `NativeHost::materialize_typed_buffers`
is a no-op unless a bound stage sets `typed_buffers`; when set, it calls `Metal::write_typed_buffer`
only for pending bits inside the draw or dispatch's bound `Root::Table` ranges (plus explicit
`stride == 0` typed buffer views in the bound heap if the shader also indexes
`ResourceDescriptorHeap` directly), leaving unbound structured buffers unmaterialized. If a view's pool offset is not a multiple of
`minimumTextureBufferAlignmentForPixelFormat` (16 bytes on Apple Silicon), `write_typed_buffer`
warns once and falls back to `write_buffer` so raw accesses in word 0 still work without tripping
Metal validation. If `Binding`'s DXIL scan ever missed a `TypedBuffer` operation (or a view fell
back to `write_buffer`), `write_buffer` leaves word 1 (`textureViewID`) at `0` (also zeroing it and
dropping the `MTLTexture` whenever an entry is re-mirrored), so `texture_buffer` loads and
`get_width()` read `0` and `texture_buffer` stores are no-ops without faulting (`wrong_typed_buffer_scan_reads_zero_without_crashing`).
`NativeHost::live_typed_views` (`Metal::heap_typed_views`) counts the entries currently holding a
`texture_buffer` view alongside cumulative promotions (`NativeHost::typed_views`), logged together
in `native typed buffer views`. 32-bit `SINT` views (`4`, `18`, `43`) map to `Uint` because `victory-shader`
passes raw 32-bit lane bits through `TypedBuffer<float4>` (`bitcast i32 <-> float`) while Metal's
texture unit unpacks/packs by the view's `MTLPixelFormat` with no sub-32-bit extension; 8- and
16-bit `SINT` stay unmapped so `Uint` does not zero-extend signed channels. Tested on the GPU in
`native::tests::heap_tests::formatted_buffer_stores_and_loads_through_structured_and_typed_uavs`,
`native::tests::heap_tests::wrong_typed_buffer_scan_reads_zero_without_crashing`, and
`victory-metal` `gfx::heap_set_tests::a_heaps_set_follows_its_entries`.
The "native draws so far" line counts the promotions so far as `typed_views`
(an entry mirrored again and promoted again counts twice) and the title's
resource heaps as `resource_heaps`, to weigh what the typed views and the
longer heaps cost.

#### A command buffer the GPU fails still pays its signals and value writes

Observed: the GPU failed a command buffer (`kIOGPUCommandBufferCallbackErrorHang`), and the buffers
after it with `...InnocentVictim`. A failed buffer runs none of its encoded tail, so the queue's
order event, the title's fence signals and the bottom-of-pipe value writes (blit fills of title
memory) never happened. The queue's next buffer waited for the order event for good, the title's
fences never reached their values, other queues stayed held on the barrier words the fills would
have written, and 30 s later the title's own watchdog killed the process.

Now: `submit_passes` records what the submission owes (`gfx::Owed`: the order event and its value,
each `(fence, value)` signal, each fill's buffer, offset, length and byte). When the completed
handler sees an error it writes the fills on the CPU, in order, then raises those events (never
lowers one; the order event is an `MTLSharedEvent` for this), so a waiter that sees its fence
reached also sees the fills before it, and warns for the first few
times. Draws and dispatches of the failed buffer are lost, and its copies are not replayed (one may
have partly run, and its source may have changed since). Guess: the console has no equivalent of a
discarded command buffer, so finishing the signals and barrier writes on the CPU is our recovery
policy, not console behaviour.

#### A command buffer the GPU never reports on is paid too

Observed, with memory running out (two games in the world at once on a 48 GB Mac: 41 GB wired, 5%
free; one game alone wires about 22.5 GB in either residency mode): after 30 to 60 world frames one
game stops for good with no crash. Metal fails some command buffers of the graphics queue with
`kIOGPUCommandBufferCallbackErrorTimeout` 5 and 10 s after they were committed, and never calls the
completed handler of others committed before them: in one run #11485, #11486, #11489 and #11490
failed while #11483, #11484, #11487 and #11488 were neither completed nor failed for the remaining
minute. The unreported ones' value writes never land (one held the fills of three barrier words),
so every title queue stays held on its barrier word, no fence moves, and the title spins in `Sleep`
with no graphics call at all. The queue's order event is not what is stuck (it had reached every
value encoded). The same error is what an 18 GB Mac got with the pool's set on the queues.

Now: every submission is listed until Metal reports on it (`gfx::Flight`). While work stays held
(`refresh`, at most every 5 s) `gfx::recover_lost` pays what the lost ones owe, as for a failed
command buffer: a command buffer is lost when one committed after it on the same queue failed 5 s
ago or more, or when it has gone unreported for 30 s. It logs an error the first time and a
warning with the count each time. Guess: such a command buffer never runs. If one does, its draws
land late and its signals and writes repeat what was paid. Lost draws are missing from the
picture; the title goes on, and recovers when memory is there again.

Two runs with `VICTORY_METAL_RESIDENCY_SETS=0` under the same pressure (40.5 GB wired, 6% free,
four minutes, 21 world frames each) had no failed command buffer. That is fewer frames than the
set-mode runs reached before stalling (31 and 54), so it does not show that per-encoder naming
survives pressure better.

#### A refused submission still pays its value writes

Observed on 16 GB Macs (tester reports #265 and #280): after Metal failed command buffers with
`kIOGPUCommandBufferCallbackErrorOutOfMemory`, queues stayed held on their per-frame barrier words
(`work held for a wait that does not hold`: `0x18437f9d8 cmp 7 ref 0x883 ... now Some(2178)`), the
frame rate decayed to 0.4 fps and the title's watchdog ended the run (`0xc0000005` at
`image+0x55e56f8`). A pass `submit_passes` cannot encode is dropped while the command buffer still
commits, so the fills of the dropped pass never run and nothing paid them; a piece refused before a
command buffer exists at all loses them the same way.

Now: `submit_passes` reports what it lost (`victory_backend::resources::SubmitFailure`): whether the
command buffer committed, and which pass indices were dropped from it. When the call in
`NativeHost::execute` returns an error, the host makes only the lost passes' fills on the CPU, in
order, with a release fence after them (`crates/victory-gpu/src/native/host.rs`), and warns the first
time (`native host pays a lost pass's value writes on the CPU`). A committed submission's kept passes
still run on the GPU: paying their fills too would let a waiting queue go on before this queue's
earlier GPU passes have run, and the GPU's own later fill could overwrite a newer value the title
wrote after seeing the CPU write. What a waiting queue can read early from a paid fill is nothing
harmful: the fill is a plain store of the word range into coherent title memory plus a release
fence, so a queue polling the barrier word reads the old value (keeps waiting) or the filled one; a
partial multi-word fill just means it waits on the next word. The hazard closed is the opposite:
the word staying 0 forever after a refused submission, which held the queue for good. Host policy,
not console behaviour, like the recoveries above: the console has no refused submissions. Tests:
`vertex_tests::a_submission_that_never_committed_pays_its_value_writes_on_the_cpu`,
`vertex_tests::only_the_fills_of_refused_passes_are_paid_on_the_cpu` and
`vertex_tests::a_submitted_piece_leaves_its_value_writes_to_the_gpu`.

#### A pass refused mid-encoding ends its encoder

`submit_passes` refuses a pass it cannot encode and still encodes the passes after it. A refusal after the pass's encoder was opened (a counted record whose gate fails, an indexed indirect draw without an index buffer) now ends that encoder first; it used to leave it open, and Metal aborted the process (`A command encoder is already encoding to this command buffer`) when the next pass opened its own.

#### Heap samplers live as long as the backend

A sampler heap entry holds only the sampler's resource ID. The host used to drop its `MTLSamplerState` right after writing the entry, and Metal gives a freed sampler's ID to the next sampler made, so an older entry then sampled with a newer descriptor's filter and address modes. `create_sampler` now keeps every sampler it makes, one per distinct `SamplerDesc` (few exist), and hands the same one back for the same descriptor.

Dynamic `CreateSamplerX` descriptors preserve minification and magnification independently. Changing either filter selects a different cached Metal sampler. These descriptors use Metal's border mode for transparent black, opaque black, and opaque white. Other border colors retain edge clamp, since Metal offers three fixed colors.

#### CPU texture transfers are bounds-checked

Texture uploads, the background scanner and submit-time probes first try one Mach read per chunk (up to 1 MiB). If a read spanning multiple pages fails, the host walks the VM regions that overlap the chunk. It leaves unmapped gaps and regions without read permission zero-filled and copies each readable span in one read. If that read also fails, for example because the title changed the mapping after the query, the host retries just that span at page boundaries. A 1 MiB chunk entirely inside a hole or a single no-access region costs one failed read and one region query. A failed probe contained in one page needs only its original read.

`upload_texture` and `read_texture` compute what the mip needs from the texture's format (blocks wide times bytes per block, rows, slice pitch) and skip the Metal call, with a warning naming the format, region, pitches and buffer length (and a backtrace the first time), when the row pitch, slice pitch or buffer is too small. Metal's `AGX: Texture read/write assertion failed: bytes_per_row >= used_bytes_per_row` was seen 0.6-2 s before a burst of title-heap corruption, so a short pitch is suspected to let the driver run past the buffer; the check keeps that out of host memory (hypothesis, not yet proven).

#### Rectangle-list corners and the Brightness logo (open)

* The right-angle corner is no longer assumed to be v1: the first two-channel-or-wider float fetch (the position) picks it as the corner opposite the longest side, and the strip is p, r, p + q - r, q (r the right angle). With the title's data (top-left, bottom-left, bottom-right) this is the old rule. A value two corners share is copied to the fourth exactly, not through float arithmetic (packed or NaN bits survive).
* A rectangle-list draw whose pipeline is not guessed (a described pipeline) is drawn with its fourth corner too since 27 Nov 2026 ("Described pipeline rectangle-list draws"); only when its vertex shader does not translate for rectangle lists, or the draw is indexed, is it drawn as plain triangles (half of each rectangle), warning once.
* `VICTORY_TRACE_RECTS=1` logs the first three rectangle-list draws of each pipeline with the first three vertices as floats.
* The Brightness logo (half a rectangle, the cut along one diagonal) is a guessed pipeline (`vertex 511484f3`: float3 position, unorm8x4 colour at 12, float2 uv at 16, 24-byte vertices) drawing one rectangle at (+-0.2, +-0.325) with corners top-left, bottom-left, bottom-right, which is the order the rule expects. Its matrices flip y, so the visible half is the strip's first triangle (which holds the computed corner), and the second triangle is lost. Both triangles wind alike, so culling does not explain it; the next suspects are the guessed depth state (less-equal with writes, `VICTORY_DEBUG_NO_DEPTH`) and the pixel shader on the second triangle. Not settled: the game runs needed were stopped.

#### Raster, depth, stencil and write mask from the register image

In all 23,022 images the registers just before the eight `CB_BLEND<n>_CONTROL` sit at the same
offsets back from them (the field layouts below match the values seen):

| Bytes before the blend block | Register | Seen |
|---|---|---|
| 0x30 | `PA_SU_SC_MODE_CNTL` | 0x201a40 no cull (11,341), 0x201a42 cull back (11,621), 0x201a41 cull front (58), 0x201a46 cull back with clockwise front (2) |
| 0x2c, 0x28 | constant 0x8, 0x1000 (probably `PA_SU_LINE_CNTL` and `PA_SC_LINE_CNTL`, whose bit 12 is DX10_DIAMOND_TEST_ENA) | |
| 0x24, 0x20, 0x1c | floats, polygon offset: `PA_SU_POLY_OFFSET_CLAMP`, `_FRONT_SCALE`, `_FRONT_OFFSET` (order guessed, below) | 0.003, 29.0, 16.0, 1.0 |
| 0x18 | `DB_DEPTH_CONTROL` | 0x7007f0 depth off (6,895), 0x7007e2 test GREATER_EQUAL without writes (4,999), 0x7007e7 GREATER_EQUAL with writes and stencil (3,325), 0x700796 LESS with writes, 0x7007b6 LESS_EQUAL with writes, 0x2002f1 stencil EQUAL only |
| 0x14 | `DB_STENCIL_CONTROL` | 0 (keep), 0x30030 pass REPLACE, 0x80080 pass increment-wrap |
| 0x10, 0xc | `DB_STENCILREFMASK`, `_BF` | 0x100ff00 (read mask 0xff, write mask 0), 0x1ffff00 (both 0xff) |
| 0x4 | `CB_TARGET_MASK` | 0xffffffff, 0 (depth only), 0xf, 0x3ffff |

`Graphics::guess_from` now reads all of these over the guessed defaults: cull (FACE 0 = counter-clockwise front), depth enable, write and function (hardware + 1 = D3D12), stencil enable, read and write masks, front and back ops and functions (the back face's only with BACKFACE_ENABLE), and each target's write mask. Most depth tests are GREATER_EQUAL (reversed Z), where the guess was LESS_EQUAL with writes. The polygon offset is read too (below). `VICTORY_GUESS_BLEND_ONLY=1` keeps the earlier blend-only reading, for comparison. A run past setup to the loading screen gave the same presented images, byte for byte, as without this change. The setup screens were not checked: runs now skip them, probably because the profile is saved.

**Polygon offset ([#18](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/18)), a guess.** Which of the three floats is which was not checked against the images. They are taken in register order (`PA_SU_POLY_OFFSET_CLAMP` at 0x28b7c, then `_FRONT_SCALE`, `_FRONT_OFFSET`), and read the way AMD's PAL writes them (`CmdSetDepthBiasState`): the clamp as it is, the slope scale times 16 (the hardware takes Z gradients in 1/16ths), D3D12's integer bias as a float with no scaling for the depth format, and the back face the same as the front. So `guess_from` sets `DepthBias` to the offset, `SlopeScaledDepthBias` to the scale / 16 and `DepthBiasClamp` to the clamp, when `POLY_OFFSET_FRONT_ENABLE` (bit 11 of `PA_SU_SC_MODE_CNTL`) is set. Every value seen sets it, with the back face's (bit 12). A variant's rasterizer record replaces them. To check the guess in images dumped with `VICTORY_DUMP_DESC_DIR`: the word at 0x1c back should hold only whole numbers, and 16.0 (a slope scale of 1) should sit at 0x20 back.

Metal binds the pipeline's depth-stencil state on every draw. When both tests are disabled,
passing `nil` clears the previous draw's comparison and depth writes within the same render pass.

#### The first world frame is black (27 November, Series S, headless)

In each of five runs the prologue presented exactly one world frame (about 80 targets drawn or cleared; the game renders at 1280x720 there) before the crash, or none when the first-frame stall tripped the watchdog. With `VICTORY_TRACE_TARGETS=1` and `VICTORY_DUMP_TARGETS` at that frame:

* Every colour target of the 3D passes reads zero: the G-buffer (three 1280x720 RGBA8 targets and an RG16F, cleared, no draw encoded), the HDR scene colour (1280x720 RGBA16F, 3 draws encoded, 209 dropped), the 1920x1080 R10G10B10A2 target after it, and the presented 3840x2160 image. Only small targets written by compute (two 256x256 lookup tables) hold data. So the image is black from its first link, the geometry, not at tonemap or present.
* Dropped in that frame: about 5,200 indirect executions of the title's mesh signature (stride 12, argument type 10), 960 to 1,140 dispatches without a host compute pipeline, about 300 draws without a host render pipeline, 25 indirect rectangle-list draws, and about 270 texture copies.
* The type-10 executions run under graphics pipelines described by the console's register image whose "vertex" stage is an NGG program (`s_sendmsg MSG_GS_ALLOC_REQ`, `exp prim`, the group id read from LDS): the world's opaque geometry is mesh shading in console pipelines, which the native host has no path for (no mesh pipelines in `victory-metal`, and the translator is never asked for a mesh program from a guessed pipeline). Apple's converter does convert the translator's mesh output (the synthetic `triangle_mesh` of `emit` tests: stage 5, loads).
* The draws without a pipeline, by refusal (one run): 117 "a cross-lane operation outside compute" (pixel shaders into RGBA16F + R32_UINT), 43 "a descriptor changed after loading", 60 "indexes the descriptor heap itself" (34 of them G-buffer draws, 21 a D16 shadow pass), 47 lane-mask joins, 19 `v_ldexp_f32` (now lowered). "A cross-lane operation outside compute" now names the operation and the stage after its prefix (for example `...: ds_bpermute_b32 in a pixel program`, `...: DPP row_shl, row_shr or row_ror in a pixel program`), since which one the title uses decides how to lower it: 127 of the 639 refusals in the M5 Ultra log of [#147](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/147), all in graphics pipelines. That log is from 57b7ebd, before the names, and at that commit a constant-lane `v_readlane_b32` of an SGPR spill (a VGPR lane that only `v_writelane_b32` fills, on two paths into a join) counted under this cause. `Lower::spill_slot` (269e7d5) now makes that read a phi of the spilled SGPRs in a pixel program too, so part of the 127 may already translate on main. A tally on #4 (27 November, c1b9587 plus #253, #267 and #264, none of which touch graphics cross-lane lowering; Series S, one cold and one warm run to frame 2560) found no pixel refusal of this cause left: the only cross-lane refusals were 10 "between host waves in divergent code", all compute. So nothing in the pixel shape is lowered for now. Tests: `emit::gfx_refusal_tests::a_pixel_program_reads_back_an_sgpr_spilled_on_two_paths` (fails with the old refusal when the spill rule is off) and `a_pixel_program_with_dpp8_is_refused_by_name`.
* With mesh pipelines (guessed vertex programs that export primitives run as mesh shaders, type-10 records as `drawMeshThreadgroupsWithIndirectBuffer`; 27 November, one run that reached the world frame and one that tripped the watchdog): no type-10 execution drops as an unrun signature any more, but the world frame is still black (G-buffer and HDR targets read zero, 0 draws encoded into the G-buffer). Every type-10 record now drops at its pipeline instead: of the frame's 5,178 draws without a host pipeline, 4,924 are mesh pipelines, refused because the mesh program indexes the descriptor heap itself (2,962, the D16 and D32 depth passes), reads a lane mask as data (944, the G-buffer passes: a VALU compare of a uniform SGPR ANDed with EXEC and kept in an SGPR) or reads a hardware SGPR input (824). Those three are the next blockers.
* With heap-indexing graphics shaders accepted (`heap_tests`, 27 November, Series S headless, `VICTORY_TRACE_TARGETS=1`, one run each before and after, both reaching the world frame): "indexes the descriptor heap itself" refusals went from 62 shaders to 0, and the world frame's draws without a host render pipeline from 5,143 to 2,238. The 22 D16 shadow targets (DXGI 53, 256x256 to 640x2560) went from 326 draws encoded and 2,725 dropped to 2,633 encoded and 456 dropped; the 1280x720 D32S8 scene depth (DXGI 19) from 8 encoded and 2,229 dropped to 675 and 1,593; two G-buffer targets now get 52 draws each and an RG16F one 54 (were 0 and 2), still with about 1,500 dropped each. The G-buffer colour and the presented image still read zero: what remains dropped there is the lane-mask and hardware-SGPR mesh refusals.

* With the mesh refusal fixes (uniform lane masks, root constants up to the heap, compares with DPP, `s_sext`; 27 November, branch `fix/mesh-refusals`, headless): of the world frame's ~5,150 dropped draws, those refused for a lane mask read as data fell from 951 to 78 and those refused for a hardware SGPR input from 824 to 0. The G-buffer passes' mesh programs now translate, and the passes are refused one step later, by their pixel programs: they interpolate 16-bit attributes (`v_interp_p1ll_f16`/`v_interp_p2_f16`, `high` halves; 766 draws "the high half of an input not marked 16-bit", 181 "an f16 interpolation ..."), which the guessed signatures cannot carry: the mesh program packs those parameters itself (`v_fma_mixlo_f16`/`v_fma_mixhi_f16` into one dword, but also moves a packed dword from a `v_cndmask_b32`), so neither stage alone says which channels hold two halves. The G-buffer still gets no draw; the depth-only passes (2,769 draws) are refused because their programs index the descriptor heap. With the first world frame now compiling about 2,000 more Metal pipelines, one cold-cache run crashed (at the same title address as the usual post-frame crash) 32 s after the last present, before presenting it; the next two, with the Metal library cache warm, presented it.
* With the 16-bit parameter convention ("16-bit parameters of guessed pipelines"; 27 November, branch `feat/f16-interp`, Series S headless, `VICTORY_TRACE_TARGETS=1`, `VICTORY_DUMP_TARGETS`): no pipeline is refused for a 16-bit read any more (947 draws in the world frame before: 181 "an f16 interpolation ...", 766 "the high half ..."), nor for a half its vertex side does not export. Six runs with the translator changing in between tripped the first-frame watchdog with a cold Metal cache; the seventh, warm, ran the full 320 s and presented 17 world frames. In the first, the G-buffer targets (1280x720) got draws for the first time: three RGBA8 targets 1,348 to 1,367 draws each (55 to 151 dropped), the RGBA8 sRGB one 1,367 (151 dropped), the RG16F one 1,361 (144 dropped), the D32S8 scene depth 1,990 (243 dropped); the RGBA16F HDR scene colour 8 (224 dropped). Dumps: one RGBA8 G-buffer target holds content over 15 to 39% of the image in later frames (red and blue channels, means up to about 50 of 255), the others mostly zero, a fourth with traces; the HDR target is lit over about 1% (maximum 2,360); the presented 3840x2160 image is still black. The world frame's remaining drops are about 900 dispatches without a host compute pipeline and about 570 draws without a host render pipeline (top refusals: lane-mask joins of untracked masks, descriptors outside the heaps, lane masks read as data).
* Compares with DPP (`v_cmp_*` / `v_cmpx_*`, `#258`): full-mask 16-bit compares (`f16`, `i16`, `u16`, `v_cmp_class_f16`, `v_cmpx_*`) now lower with DPP through `Lower::half_src` (`self.src0`), 64-bit compares with DPP are explicitly refused (`"a 64-bit compare with DPP"`), and partial `row_mask`/`bank_mask` or unguarded `bound_ctrl:0` DPP compares stay refused (`"a compare with DPP row masks or bounds"`, matching LLVM `GCNDPPCombine.cpp`'s `MaskAllLanes` constraint; tests: `emit::graphics::tests::pixel_compares_with_16bit_dpp_and_64bit_or_partial_mask_refusals`, `native::addr_tests::compares_with_16bit_dpp_vcmpx_and_class_run_on_the_gpu`).

#### Render targets of different sizes

At the story load the title draws a 1920x1080 colour target with its 3840x2160 depth target bound. The console draws into the area all bound targets cover; the Metal pass used to be refused instead ("render targets of different sizes"), which dropped the whole piece of title work around it, three pieces at the switch into the world. The pass now sets `renderTargetWidth`/`renderTargetHeight` to the smallest attachment and bounds each draw's scissor by it (`gfx.rs`, `encode_render`); only differing sample counts are still refused. Test: `state_tests::a_depth_target_larger_than_the_colour_target_still_draws`.

#### Independent depth and stencil clears

The native `ClearDepthStencilView` path now carries the stencil value and both clear flags
through the recorded command to Metal. Depth-only clears preserve stencil; stencil-only
clears preserve depth; a combined clear sets both to the requested values. Previously the
native bridge dropped stencil-only clears and the stencil value, while Metal reset stencil
to zero whenever depth was cleared. The backend's `DepthTarget` has independent depth and
stencil clear values (`None` means load/preserve), including passes that only draw.

Synthetic Metal tests draw into a `D32FloatS8` target and read back color selected by depth
and stencil comparisons. They cover depth-only preservation after a stencil-writing draw,
stencil-only and combined clears to 0, 173 and 255, command snapshots, and rejection of
mismatched depth/stencil. No title run is needed. This does not establish the cause of the
brightness-screen problem. The native bridge skips rectangle-limited clears and views it
cannot represent: explicit views must be TEXTURE1D or TEXTURE2D at mip 0, and the resource must be a
single-layer, single-sampled 1D or 2D texture (also for default/null descriptions).
1D textures use height-one Metal backing. Sample count is checked on the original
resource description because the native texture description does not retain it. These guards apply to all
clear flags, so unsupported requests cannot wipe the entire base depth/stencil attachment.
Bridge regression coverage calls the title-facing clear ABI with real descriptor records,
checks rectangle and mip-1 clears record no command, and verifies full base-view clears
still record commands for depth-only, stencil-only and combined flags. Array/unknown views
and default views of multisampled resources are also rejected; explicit and default
single-layer 1D views still record full mip-0 clears. Rectangle and subresource clear support remains unimplemented.

#### Large sampled textures without their top mip (`VICTORY_TEXTURE_DROP_TOP_MIP=1`)

For Macs whose unified memory cannot hold the title's texture set (#8: 16 GB, out of GPU memory at the world load). Off by default. A sampled-only 2D or array texture whose larger side is at least 1024 texels, with a level below its top, is made on the host without that level: half the width and height, one level fewer, about 75% of its memory saved. `detile` skips the top level (title level `m` becomes host level `m - 1`) and `host_mip_range` maps each view's levels onto the host texture. Render targets, UAV and 3D textures are left as they are. Not console behaviour: implicit-LOD sampling uses the next level (a little blurrier up close), explicit-LOD samples come out one level blurrier, and integer loads and `GetDimensions` see the smaller texture. It complements #301 (a byte budget on how many copies are kept) by making each large copy smaller. Details in [metal-backend.md](metal-backend.md); tests `drop_top_mip_tests` and `a_texture_made_without_its_top_level_gets_the_lower_levels`.

#### GPU-written textures copied back into title memory (`VICTORY_TEXTURE_BACK=1`)

Host textures are Metal's own allocations in Apple's layout, so what a compute pass writes never shows in the title's bytes, and a title that reads such a texture on the CPU (the ocean's height and displacement maps, [#46](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/46)) gets the old contents. Off unless `VICTORY_TEXTURE_BACK=1`:

* **Which textures.** At each dispatch, the textures named by unordered access views in the descriptor tables it binds (`NativeHost::note_written`): more than the shader writes, which costs time but changes nothing (the read-back equals what is there). Of those, textures with the unordered access flag, not render or depth targets, with at most 4 MiB of title memory. The limit is a guess until a world run shows what is written; each texture is logged once, copied or too large, with format, size and layout.
* **When.** Each piece of title work also signals a fence of the host's own, in the same command buffer; the title's signals go in later ones. The copy happens once that fence is reached, before the title is shown any fence value (`Title::completed`, `Title::fence_value`: read the values, then copy back, so the work a value covers is already in memory). No page protection and no read faults: the native host has none, and Rosetta cannot resume some faults (`watch.rs`). Guess: the title waits on a fence, or reads a frame late; a title polling a label the GPU writes into memory, or another queue that waits on such a label and then signals a fence, could get there before the copy.
* **Layout.** The reverse of the upload: `read_texture` per mip and array slice (a 3D mip is read with all its depth), `Layout::tile` into the current title bytes (the same address walk as `detile`, so every layout the upload handles, including the XOR modes; padding between subresources stays), then written through the pool's own mapping. B4G4R4A4 (115) is rotated back by 4 bits. Layouts `detile` refuses never had a host copy to read.
* **The title's own writes win.** If the memory no longer holds what the host texture was filled from (or last copied back as), nothing is written and the next submission uploads the title's bytes. The hash of the bytes written back is recorded as the last upload, so the scanner's sighting of them uploads nothing.
* **Not seen:** UAVs a shader reaches by indexing the descriptor heap itself, pixel shader UAV writes, and GPU copies into textures.
* **Cost.** CPU read, retile and two hashes per written texture per submission, under the title lock. A compute retile on the GPU is the next step if a profile shows it.

Tested with our own shader (`tex_fill.hlsl`) on an M2 Pro: a 64x64 RGBA8 2D texture in `LINEAR`, `SW_4KB_S` and `SW_64KB_D_X`. A second test sends tiled bytes up into a host texture and back down for 1D, 2D with 5 and 9 mips, a 2D array with mips, and 3D with and without mips (element sizes 1, 4 and 8, the same three modes plus `SW_64KB_S`): every mip, array slice and depth slice comes back where it was. It shows nothing is dropped, not that the console's layout equations are right (both directions use them). Not tested: a dispatch writing those shapes.

A world run on an M5 Pro (Series S, windowed, reported on [#151](https://github.com/LakshanSivakumar/Native-Victory-macOS/pull/151)) named 22 textures as copied back (15 2D, 4 3D, 3 1D; the largest 256x256x16, 4 MiB), none too large, with no frame rate change beyond run-to-run noise up to the first world frame. The run then crashed (image+0x55e56f8, also without the switch), so whether boats float is still unknown. That count may be low: a texture skipped on its first completion was not logged then (fixed since).

#### Size queries of writable images read 0 (fixed)

A program that asks a writable image (a UAV) for its size got 0x0. The translator wrote the level of
the query as DXIL's undefined value for a UAV, as DXC does (an `RWTexture` query has no level), and
Metal through the Shader Converter answers that with 0. Sizes the host rebuilds into an image
descriptor's own dwords, when a program reads its width or height from the descriptor's bits instead
of querying them (`Lower::descriptor_data`), came from the same query and read 0 too (height - 1
read as 16383).

Under Wine the same answer from D3DMetal gave the title's sun-shadow min/max program
(`cascadeshadows_processing:CS_ComputeMinMax`, `07f451a14b13fecb`) a 0x0 image, so its tile
classifier lit or shadowed whole 16x16 tiles ([halfwave50/VIctory-macOS-halfwave50
0f3ed38](https://github.com/halfwave50/VIctory-macOS-halfwave50/commit/0f3ed38), opt-in per program
there).

* **`image_get_resinfo`** now queries the level in the program's first address for every image, as
  for shader resource views. Level 0 is the view's own size.
* **guess:** a level past a writable view's single one reads 0 on the console, as it does here.
  Forcing level 0 in every program under Wine changed the TAA disocclusion mask and put speckles on
  skin, which suggests some programs ask about another level. Those keep reading 0, as before.
* **Descriptor fields** of an image come from level 0, writable or not. A buffer's record count
  keeps the undefined level, which a buffer ignores.
* **Tested** (`heap_tests::a_size_query_of_a_writable_image_reads_its_size`, M1 Pro, macOS 27.2,
  MSC 4.0.3): a 16x8 texture UAV declared in a table. Upstream read `[0, 0, 16383]` for the width,
  height and descriptor height - 1, the query fix alone `[16, 8, 16383]`, and both fixes
  `[16, 8, 7]`. **Not tested:** a game run; the shadow edges themselves.

#### Texture formats the host lacked

The title's texture formats are DXGI numbers (`resource::pc_format` only renames the console's depth views 118 to 120). The first warnings ("a title texture format the host does not represent") named ten of them; once those were mapped, the run reached more textures and named fifteen more. All of them but two now map to a Metal format of the same element size (`descriptors::format`, tested against `resource::element` in `host_formats_have_the_title_formats_element_size`):

| DXGI | Format | First texture seen (flags: 1 render target, 2 depth, 4 unordered access) | Metal |
|---|---|---|---|
| 42 | R32_UINT | 1280x720, unordered access (a compute output) | R32Uint |
| 26 | R11G11B10_FLOAT | 1280x720, unordered access (a compute output) | RG11B10Float |
| 67 | R9G9B9E5_SHAREDEXP | 1280x720, read only | RGB9E5Float |
| 16 | R32G32_FLOAT | 160x90, unordered access | RG32Float |
| 62 | R8_UINT | 640x360, render target and unordered access | R8Uint |
| 34, 35 | R16G16_FLOAT, R16G16_UNORM | 256x256, 127x128, render targets | RG16Float, RG16Unorm |
| 24 | R10G10B10A2_UNORM | 32x32 | RGB10A2Unorm |
| 30 | R8G8B8A8_UINT | 1x1 | RGBA8Uint |
| 99, 72, 78 | BC7, BC1, BC3 UNORM_SRGB | 512x512, 32x32, 512x512 | the `_sRGB` BC formats |
| 53 | R16_TYPELESS | 2048x8192 depth (a shadow map) | Depth16Unorm (R16Unorm when not depth) |
| 55 | D16_UNORM | (its depth views) | Depth16Unorm |
| 3, 17, 36, 57 | R32G32B32A32, R32G32, R16G16, R16 UINT | 128x128 to 2560x1440 | the matching Uint formats |
| 11, 13 | R16G16B16A16_UNORM, _SNORM | 128x4, 256x256 | RGBA16Unorm, RGBA16Snorm |
| 49 | R8G8_UNORM | 16x16 | RG8Unorm |
| 50 | R8G8_UINT | 64x64 | RG8Uint |
| 51 | R8G8_SNORM | 1024x1024 | RG8Snorm |
| 54 | R16_FLOAT | 512x512 | R16Float |
| 91 | B8G8R8A8_UNORM_SRGB | 1024x512, sampled (named once a windowed run reached the world load) | BGRA8Unorm_sRGB |
| 95, 96 | BC6H_UF16, BC6H_SF16 | 96: 512x512 (tester log #123) | BC6H_RGBUfloat, BC6H_RGBFloat. Metal's `RGBFloat` is the signed one: 95 used to map to it, so unsigned BC6H textures decoded with the wrong sign (test: `victory-metal` `sample_bc6h_unsigned_and_signed`) |

Integer render targets never blend (Metal refuses a pipeline that blends one), whatever the state says. B5G5R5A1_UNORM (86, a 1024x4096 render target) and B4G4R4A4_UNORM (115, a 2560x1440 render target) map to Metal's packed formats. A GPU test (`victory-metal` `packed_16_bit_channel_order`) pins their bit order. `BGR5A1Unorm` matches DXGI: blue in bits 0-4, green in 5-9, red in 10-14, alpha in bit 15. `ABGR4Unorm` holds alpha in bits 0-3, blue in 4-7, green in 8-11 and red in 12-15, where DXGI has blue, green, red, alpha from the low nibble up. So title bytes of format 115 are rotated left by 4 bits as they are uploaded (`textures::to_host`). The host's own render writes and samples agree with each other, and uploads need the conversion, as does the copy-back of GPU-written textures (above). Before the story loads, no draw or dispatch binds any of these textures (the targets traced are formats 28, 29 and 19), so which shaders write and read them is still to be seen. With the mapping, the presented images up to the loading screen look the same.

A render target view whose host texture cannot be made keeps its attachment slot: `glue::targets` binds nothing for that pass and refuses the draws that follow, instead of moving the later targets into earlier slots.

A render target view's format says how its writes are stored, so a view that names the other format of its texture's sRGB pair (R8G8B8A8 or B8G8R8A8, UNORM against UNORM_SRGB) is drawn and cleared through a host view of the texture in that format (`Title::target_of`). The title draws its boot movie through a plain view (28) of an sRGB texture (29): drawn into the texture itself, the picture was encoded a second time and shown too bright. A described single-sampled 1D or 2D view selects one mip and one array slice. Both the draw binding and the clear use a cached Metal view narrowed to that subresource, so the render pass uses the selected mip's size and distinct attachments do not merge into one pass. Invalid ranges, nonzero planes, and unsupported explicit layered 1D/2D or multisampled views are refused. A null description or a 3D view based at mip 0 and W slice 0 keeps the existing base attachment. Rectangles in a clear remain unsupported.

### Scaled byte vertex fetches

The shared shader translator supports GFX10 `8`, `8_8`, and `8_8_8_8` USCALED/SSCALED
vertex fetches, including partial channel loads. The existing unsigned/signed integer
stage-in formats supply zero/sign-extended values, which the shader converts to float
before rectangle reconstruction. Scaling here is numeric conversion, not normalization:
unsigned 255 becomes 255.0 and signed byte 0x80 becomes -128.0. The GFX10 encodings were
checked with LLVM's `gfx1030` assembler; the conversion also follows the existing
`Numeric::Uscaled`/`Sscaled` buffer decoding in `emit/lower.rs`.

Synthetic tests cover translation in wave32/wave64 and rectangle modes, all 256 byte
values in every loaded channel through Metal readback, preserved registers after partial
loads, and reconstructed rectangles compared with explicit floating-point quads. 16-bit
and packed scaled formats remain unsupported by vertex fetches. No game shaders were
used, so this does not establish which of issue #4's reported title shaders now translate.

## Whole-wave sections the simulation does not follow

The section emitter can reproduce register results. If it encounters a store, atomic, export,
barrier or another instruction it cannot safely treat as a register operation, translation stops
at that instruction's byte offset and names the opcode. This also covers atomics whose result
goes unused.

A whole-wave section (`emit::wave`, decision 0003) is simulated on the console's lanes, and a
value it leaves is lowered by the reduction or prefix it turns out to be. A value without such a
shape was refused as "a whole-wave section step the simulation does not follow". That text is
now followed by a colon and the first step of the section the simulation does not model: "an
EXEC write that is no constant", "a scalar ALU result", "an instruction that is no VALU
operation", "a v_readlane at a lane it does not know", "a VALU result in an SGPR", "a DPP step
with lanes off", "a DPP control it does not model", "an operand or result modifier", "a
v_permlane with computed lane selects" or "a VALU operation it does not model"; or, when it
models every step, "a value that is no reduction or prefix". The first step not modelled is a
hint, not proof: the value refused may not depend on it. `pipeline::pieces_tests::
the_host_refuses_a_whole_wave_section_naming_its_step` (`victory-gpu`) checks the refusal on the
native host's own path: the container made from the title's pieces, the host's options. The
story-load log of issue #110 (Series S) has three compute programs refused with the old message.
The simulator now follows a constant loaded by `s_mov_b32` inside the section. For example, a
signed minimum can use `INT_MAX` from an SGPR as its inactive-lane identity. Whether the three
logged programs use this pattern is unknown.
Both `v_permlane16_b32` and `v_permlanex16_b32` also use those tracked SGPR constants as lane
selectors. A selector with a runtime value still produces the computed-lane-select refusal.

An EXEC write inside a section is followed when it is a constant lane set. Those were `s_mov`s of
a constant only. `s_bfm_b32` and `s_bfm_b64` of two constant counts (`((1 << S0) - 1) << S1`) and
`s_bitreplicate_b64_b32` of a constant (bit `i` to bits `2i` and `2i + 1`) are one too
(`wave::exec_constant`). The title's prefix sums select rows with them:
- wave32 finishes the upper row of sixteen lanes under `s_bfm_b32 exec_lo, 16, 16`, adding lane 15
  through `v_permlanex16_b32`;
- wave64 does the upper row of each half under `s_bitreplicate_b64_b32 exec, 0xff00ff00`, then the
  upper half under `s_bfm_b64 exec, 32, 32`, adding lane 31.

The section stopped at those writes. Either the opcode was named, as with "s_bfm_b32 is not
supported" (a step that is no register operation is named by its opcode), or the values it left
were unknown and a later read was refused instead. An `s_bfm` or `s_bitreplicate` with a
non-constant operand is still refused there.

Over a 12,020-shader corpus (Series S boot to the menus, 27 November):
- 17 compute shaders now translate: 9 were refused as `s_bfm_b32`, 2 as `s_bitreplicate_b64_b32`,
  and 6 as "a lane mask holding lanes elsewhere read as data: written by a scalar instruction of
  the other width" (each has the wave64 section above);
- 1 is refused for another reason ("a lane-mask join of an untracked mask: data, at a join of
  several edges");
- no other output changes.

Tests:
- `emit::wave::tests::s_bfm_gives_exec_its_field_of_constant_counts`;
- `emit::lower::tests::a_wave32_prefix_sum_may_set_its_upper_row_with_s_bfm`;
- `split_wave_tests::a_wave32_prefix_sum_whose_upper_row_s_bfm_selects_runs` (`victory-gpu`, on
  Metal, with all 32 lanes and with lanes 20 and up off before the section);
- `split_wave_tests::a_wave64_prefix_sum_whose_rows_s_bitreplicate_and_s_bfm_select_runs`
  (`victory-gpu`, on Metal, across the two host waves of a console wave).

The simulator keeps the original source lane of a VGPR value through a shuffle. A constant
`v_readlane_b32` of that value reads the original lane with `WaveReadLaneAt`, or through the
group-shared exchange when a console wave spans two host waves. Uniform values and whole-wave
reductions already have the same result in every lane and need no extra lane read.
That also holds when the read comes after the section: the G-buffer's pixel programs
end a whole-wave minimum by restoring EXEC, then read lane 31 of it in a second section or
outside one. That read was refused as "v_readlane_b32 of a constant lane in a pixel program"
(132 draws a frame in the G-buffer's warm log). A section's value shaped as a constant, a
reduction or one lane's value is now that value at any lane (`Lower::section_uniform`).
Test: `emit::gfx_refusal_tests::a_pixel_program_reads_a_lane_of_a_whole_wave_reduction_as_the_reduction`.

A section of `v_add_f16` steps (`Shape::Half`: the low halves summed, the high half that of
one console lane) was lowered only where a console wave is one host wave. Where the host splits
it into two, it was refused as "a 16-bit wave reduction across the host's waves". Now each host
wave's f32 sum is combined across the pair as other float sums are (`halves_reduce`, still one
rounding to f16). The high half's lane is read across them with `wide_read_lane`. A lane that is
elsewhere must give the identity there, but the exchange reads 0 for a lane that wrote nothing, so
it carries `x ^ ident` and is XORed back. Over a 12,020-shader corpus (Series S boot to the menus,
27 November) the 2 compute programs refused for it translate, and no other output changes.
`split_wave_tests::a_whole_wave_half_sum_reads_across_the_host_waves` (`victory-gpu`) runs the
title's form on Metal in groups of two console waves: 96.0 under lane 63's high half, and 60.0
with the identity's high half when lanes 40 and up are off.

A section of `v_mul_f32` steps is now followed: it is the product of the lanes
(`WaveActiveProduct`, identity 1.0), lowered to `waveActiveOp` with the product kind, and the two
host waves of a console wave multiply their results (`split_wave_tests::
a_whole_wave_product_multiplies_both_halves` runs it on Metal). As with the other float
reductions, the host multiplies in its own order, so the last bits may differ from the console's.
Prefix products and integer products (`v_mul_lo_u32`) are still refused. Whether any title shader
needs a product is not known: no run log names the operation.

When a wave64 reduction combines its two 32-lane halves into only one lane of a VGPR (for example
through `v_readlane_b32` and `v_writelane_b32` into lane 63, or when only the upper half combines
with the lower half's `v_readlane_b32` result) and a constant-lane `v_readlane_b32` reads that lane
after the section (including after narrowing `EXEC`), `Section::lane_shapes` and
`Lower::section_lanes` evaluate that lane's reduction at the section site while keeping reads of
incomplete lanes refused (`NO_SHAPE`). The simulator also follows associative SALU combinations of
reduction terms (`s_add_u32`/`s_add_i32`, `s_min`/`s_max`, `s_or_b32`/`s_and_b32`/`s_xor_b32`),
`v_writelane_b32`, and `v_sub_co_u32`/`v_subrev_*` exclusive prefix differences (`#258`; tests:
`emit::wave::tests::wave64_single_lane_and_salu_reductions_lower_while_partial_lanes_stay_refused`
and `split_wave_tests::wave64_lane63_writelane_and_salu_or_reductions_run_on_the_gpu`).

## Groups whose lanes are not a multiple of 64 (wave32 host)

With `Options::wave32_host`, what crosses the two host waves of a console wave goes through the
`xchg` group-shared array (`lower/halves.rs`, `docs/metal-backend.md`). A group of 96 lanes has a
last console wave of 32 lanes: one host wave, no upper half, and nothing wrote that half's slots,
so its ballot's high dword, its reductions and lane reads took whatever an earlier exchange (or
nothing) left there. Now a slot no thread writes is not used: a ballot's missing half is 0, a
reduction is the lower half's alone (the console's lanes past the group are off, so a whole-wave
section adds the `v_cndmask` 0 they hold) and a lane read of a missing lane gives 0. Guess: the
console's `v_readlane` of a lane the group lacks reads whatever that lane's VGPR holds; 0 is ours.
`split_wave_tests::a_group_of_96_lanes_exchanges_nothing_with_the_lanes_it_lacks` (`victory-gpu`)
runs a ballot and a whole-wave sum in a 96-lane group on Metal.

## ExecuteIndirect on the native host

The prologue world load (frame ~2112) is the first place the title uses `ExecuteIndirect`. It creates five command signatures, each with one argument and no count buffer, and passes `max` 1 every time:

| Stride | Argument type | Record | Native host |
|---|---|---|---|
| 12 | 2, `DISPATCH` | `D3D12_DISPATCH_ARGUMENTS`: three group counts | `dispatchThreadgroupsWithIndirectBuffer` |
| 12 | 105, console single-pass dispatch | three group counts (guess, below) | as `DISPATCH` |
| 16 | 0, `DRAW` | `D3D12_DRAW_ARGUMENTS` | `drawPrimitives:indirectBuffer:` |
| 20 | 1, `DRAW_INDEXED` | `D3D12_DRAW_INDEXED_ARGUMENTS` | `drawIndexedPrimitives:...indirectBuffer:` |
| 12 | 10, `DISPATCH_MESH` | `D3D12_DISPATCH_MESH_ARGUMENTS`: three group counts | `drawMeshThreadgroupsWithIndirectBuffer` (a guessed mesh pipeline) |

The words the log prints after a dispatch's type (`[2, 1, 0, 0]`, `[105, 0, 60000, 0]`) are not part of its argument description; only root-constant, view and vertex-buffer arguments use them.

How it runs (`signature::command`, `native::glue::execute_indirect`): each of the `max` records becomes one draw or dispatch command whose counts are the title address `arguments + offset + i * stride`. At submission that address is resolved to its pool buffer like any other (`NativeHost::resolve`), and the record is read on the GPU when the command runs (by Metal, or by the gate below for dispatches and mesh dispatches), so records a culling pass writes earlier in the same submission are honoured, and nothing is copied on the CPU. D3D12's draw, indexed-draw and dispatch records are word for word Metal's `MTLDrawPrimitivesIndirectArguments`, `MTLDrawIndexedPrimitivesIndirectArguments` and `MTLDispatchThreadgroupsIndirectArguments`, and Metal's base vertex and base instance semantics are D3D12's (Metal adds the base instance to `instance_id` and D3D12 does not add it to `SV_InstanceID`; the converted shader corrects for that from the draw arguments it reads, below). The converted vertex shader reads the draw's own arguments from buffer 4 (`IRRuntimeDrawParams`), so an indirect draw binds the record itself there, which is what the converter runtime's indirect `IRRuntimeDrawPrimitives` does; direct draws keep writing their counts with `setVertexBytes`. An indexed record's start index counts from the bound index buffer's address. GPU tests: `victory-metal` `indirect_draws_read_their_records_from_title_memory` and `indirect_dispatch_reads_its_group_counts_from_title_memory`, and `victory-gpu` `an_indirect_dispatch_reads_a_record_written_earlier_on_the_gpu`.

A count buffer runs too. D3D12 runs `min(count, max)` records, the count read on the GPU when the work runs. Each record still becomes one command, which also names the count's title address and its own index `i` (`NativeList::dispatch_indirect_counted`, `draw_indirect_counted`). Before it runs, `victory-metal` encodes a one-thread kernel (`gfx.rs`, `GATE`) that copies the record into a small buffer of its own (Metal-tracked, so what reads it waits for the write), or writes zeros there when `i` is not below the count; the indirect command then reads that buffer. A zeroed record dispatches no groups and draws no vertices. Dispatches are gated in their own compute encoder, between memory barriers; draws in a compute pass encoded just before their render pass, ordered with the rest by the same fence. So a count written earlier in the same submission (a culling pass) is honoured, and the CPU never reads it. Tests: `victory-metal` `a_count_buffer_limits_the_indirect_draws_that_run` and `a_count_buffer_limits_the_indirect_dispatches_that_run`, `victory-gpu` `a_count_written_earlier_on_the_gpu_limits_the_indirect_dispatches`. An indirect command buffer was the other choice; the converted shaders' argument-buffer bindings make encoding them there larger than a gate per record.

The most groups a record may ask for. Metal runs whatever counts an indirect buffer holds. In a world run (Series S, 2026-11-27) the record of one plain dispatch (type 2, compute `0af6bb55602abf70`, 8 x 4 lanes a group) held `[510, 1182027346, 1314076063]` when the GPU read it: the first word written on the GPU earlier in the submission, the other two left over (float bits; the record is in memory the title reuses). That command buffer never ended, Metal never reported on it, and every command buffer behind it on its queue failed: this was the stop that ended world runs after two to four minutes (see "What a GPU time-out is" below). So every indirect dispatch goes through the gate, counted or not, and so does every indirect mesh dispatch; a record that asks for more than 2^22 groups in all (4,194,304) runs nothing. That limit is a guard of ours, not a console rule: it is D3D12's most for `DispatchMesh`, and far above what the title's records ask for (9,192 groups at most in these runs). It is not D3D12's 65,535 groups a dimension for a plain dispatch: other ports of this title report records of 67,440 to 115,200 x 1 x 1 on 2560x1440 targets (eight groups a tile; not measured here), and the D3D12 runtime neither drops nor clamps an oversized dispatch. Guess: on the console the record never holds such counts when the dispatch runs. Here it does because its writer is refused: compute `eee76bf6265f33bb` (16 x 16 lanes a group, 64-lane waves) begins by storing `[0, 1, 1]` (`buffer_store_dwordx3`) in 25 dispatch records 32 bytes apart, the ones the ray-traced passes run from (`0af6bb55602abf70`, `52f01bcb9deb1536`, `67662685792cf604`, `caa50df725bcb2cb`, 8 x 4 lanes a group), and is dropped ("a cross-lane operation between host waves in divergent code"). So those records keep what their memory held: zeros most frames (`[n, 0, 0]`: the pass does not run at all), leftovers when the title reuses the memory. `VICTORY_TRACE_RECORD_WRITERS=<those four hashes>` lists 88 compute programs that hold the records' buffer for unordered access, 13 of them dropped (nine for that reason). Indirect draws that are not mesh dispatches have no limit to check and are not gated unless counted. Tests: `victory-metal` `an_indirect_dispatch_past_the_most_groups_runs_nothing`, `victory-gpu` `an_indirect_mesh_dispatch_past_the_most_groups_draws_nothing`.

The type-105 records are not the ones that held leftovers: the first three of each pipeline held `[0, 0, 0]`, then `[0, 1, 1]` in title memory (`VICTORY_TRACE_DISPATCH_FLAGS=1`), three words as the guess below takes them.

Still dropped, logged once per signature ("indirect execution dropped"), except rectangle lists, logged once as "native host drops work: an indirect rectangle-list draw":

- Signatures with more than one argument: root constants, root views, vertex or index buffers changed per record. Their values are per record in GPU memory, so they cannot be read on the CPU at submission time. None seen yet.
- Mesh dispatches (type 10) under a pipeline that is not a mesh pipeline, and draws of a mesh pipeline ("a mesh dispatch of a vertex pipeline", "a draw of a mesh pipeline"). A guessed pipeline whose vertex program exports primitives is a mesh pipeline (see the guessed-pipelines list); its type-10 records run as `drawMeshThreadgroupsWithIndirectBuffer`, counted ones gated like the rest. Test: `victory-gpu` `a_counted_indirect_mesh_dispatch_runs_only_the_records_below_the_count`.
- Indirect draws of a rectangle list (topology 17): the host draws a rectangle list as instances of four vertices, which needs the record's counts rewritten first. The title's indexed signature (stride 20) is one (run 1 dropped it on this path, the only one an indexed record without a count buffer drops on); so are direct indexed rectangle lists, which the host also drops. First windowed run with this change (2026-11-27, Series S): the draw, dispatch and type-105 signatures ran, and only the mesh signature and the indexed rectangle-list draw dropped.

Guess: type 105 is not a D3D12 argument type. It is taken to be the console's single-pass dispatch, with the same three-word record as `DISPATCH` (its flags live in the signature, the title passes none); its stride, 12, fits that.

## `DispatchX` with thread counts

`DispatchX(x, y, z, flags)` (list slot 0x270) is how the title's ray-tracing builder runs its node writers. The host ignored the flags and ran `x * y * z` groups. With flag 0x20 the counts are threads, not groups: a dispatch for 45 nodes of a 64-lane program ran 45 groups, 2,880 lanes, and wrote 2,835 empty nodes (child pointers -1, infinite boxes) past the acceleration structure, over whatever the title had mapped after it: texture chunks and the CPU structures of drawables. That was the crash on the way into the prologue (Series S, about 141 s in, frame ~1860: image+0x55ceb22, or the fatal-error path at image+0x55e56f8).

What was measured (Series S, headless, 2026-11-27):

- The two writers (compute `28918b7150a7aad5` and `caa78b8ca62bc3d6`, 64 lanes a group) are always dispatched with flags 0x20 and `x` = 45, 3, 51, 27, 38, ... . A third shader (`4ddd6f9702fab2a7`, 32 lanes a group) always gets 0x8020 and 1 x 1 x 1. No other flags value occurs.
- `x` is a node count: in one structure, the second writer's region starts exactly `x * 128` bytes after the first's (128 bytes a node).
- Both writers count their nodes with `s_bcnt1_i32_b64 exec` at entry, so run as whole groups they would each count 64.
- Dropping the two writers' dispatches alone removes the crash (two full 320 s runs); dropping a third builder shader instead does not.

Guess: the flags are the GPU's dispatch initiator bits as AMD's register headers name them, 0x20 `USE_THREAD_DIMENSIONS` and 0x8000 `CS_W32_EN`. The second fits: the one shader dispatched with it is the one built for 32-lane waves. Guess: the console's lanes past the counts exist with EXEC off, they are not missing. The writers force EXEC on for their whole-wave sums and mask the operand with the EXEC they saved, which is what the compiler does for lanes that are off.

How it runs now (`list::dispatch_x`, `native::glue::dispatch_threads`):

1. The pipeline's shader is translated a second time with `Options::thread_counts` (`pipeline::native_threads`), at the first such dispatch, and cached on disk like any translation. Its host pipeline is kept by title pipeline (`Title::thread_pipes`). The first translation let the container go (keeping every compute shader's would hold 121 MB for the three that need it), so the second reads it again from title memory, where the title last held it when it made a pipeline of it, and uses it only if it still hashes to the shader. Guess: the title keeps its shader containers where they were; in the runs so far it did.
2. The host dispatches `ceil(count / group size)` groups in each dimension and passes the three counts in user-data dwords after the root signature's own (`emit::DISPATCH_THREADS`, filled in `NativeHost::translated_args`).
3. In the translation, EXEC on entry is a lane mask of its own instead of "every lane" (`ir::Program::thread_counts`, `structure::masks_in_order`). The lane's bit is `group id * group size + thread id < count`, for each dimension with more than one lane a group (`Lower::launched`). Every host thread still runs, so whole-wave sections and the exchange between the two host waves of a console wave (`lower/halves.rs`) see all 64 lanes; lanes past the counts store nothing and are off in ballots.

Limits:

- If there is no second translation (refused, or the container is gone or changed), the shader is logged as refused ("compute <hash> by thread counts") and the dispatch is dropped and logged once ("a dispatch of thread counts without a host compute pipeline"): run as groups it would write over other memory.
- A compute pipeline made of pieces (device method 0x218) cannot be translated again: the host put its container together, so the title holds it nowhere. None of Grand Theft Auto VI's dispatches of thread counts is of one.
- The D3D12 host (Windows) still runs the counts as groups and logs that once. It needs the same translation.
- `ExecuteIndirect` has no flags argument. A type-105 record's counts are run as groups (the guess above in "ExecuteIndirect on the native host"); whether the console ever takes them as threads is not known.
- Other flag bits are ignored. `VICTORY_TRACE_DISPATCH_FLAGS=1` logs each flags value once, with its unknown bits.

Tests: `split_wave_tests::a_dispatch_of_thread_counts_starts_the_lanes_past_them_off`, `a_dispatch_of_thread_counts_in_32_lane_waves` and `a_whole_wave_sum_under_thread_counts_leaves_out_the_lanes_past_them` run our own programs on Metal (`victory-gpu`); `structure::tests::exec_on_entry_is_a_lane_mask_under_thread_counts` and `lower::tests::a_program_for_thread_counts_reads_the_count_of_each_dimension_with_lanes` need no GPU (`victory-shader`), nor do `pipeline::pieces_tests::a_compute_shader_is_translated_again_for_thread_counts` and `list::tests::dispatch_x_flags_say_when_the_counts_are_threads` (`victory-gpu`).

## G-buffer and upscale pipelines refused (guessed path)

A census of the world's frames found the G-buffer (1280x720, DXGI 29/28/34, six and three targets
with D32S8) 1-11% non-black, with about 700 of its 1,600-1,900 draws a frame dropped. Three causes
are fixed here:

- **Extra parameters beside a `HALF` of their index.** A pixel shader that reads a `TEXCOORD<n>` its
  vertex shader does not write gets the vertex shader translated again with that parameter added
  (`Options::extra_params`, undefined values). The check for one already there matched any element
  of index n, so a `HALF<n>` (the low or high half of parameter n / 2, "16-bit parameters of guessed
  pipelines") hid it, the parameter was never added, and the converter refused the pipeline:
  "Fragment input(s) `user(texcoord8)` ... not written by vertex shader" (five pipelines, about 219
  dropped draws a frame). The check is by name now.
- **Root constants of a guessed pixel program.** The metadata record of root constants names only
  their first user SGPR; graphics stage data, which would give the count, is not decoded for guessed
  programs. Guess: like compute's (`constants_to_the_end`), root constants that are a pixel program's
  last user data take the SGPRs after them that it reads on entry, up to one that only reaches M0
  (the primitive mask, which follows the user SGPRs). Observed: the upscale program (pixel
  `41e09588771fe7a0`, which writes the 1920x1080 DXGI 10 target that tonemapping and present read)
  reads b500 as s9 to s12 and interpolates nothing; s10 to s12 were refused as "a hardware SGPR
  input". When the program reads a primitive mask, the constants reach up to it, also past
  SGPRs the program writes before reading (two G-buffer programs read b500 as s10 to s14, the mask in
  s15, s13 read on entry).
- **Descriptor dwords swapped by XORs.** The title's compiler swaps SGPRs in three XORs (`a ^= b;
  b ^= a; a ^= b`, in 32 and 64 bits, `vcc` as the other register too), descriptor dwords included.
  The tracer follows a value through `s_xor_b32/b64 d, x, y` when `x` is an XOR of `a` and `b` and
  `y` holds what one of them does: `d` is the other (`Tracer::xor_undone`). Four vertex programs
  refused as "a descriptor neither loaded nor built" or "outside the heaps" get past their
  descriptors; three now stop at "a lane mask read as data", one at a lane-mask join.

### The G-buffer's mesh draws dispatch no groups: their culling shaders were refused (#182)

Measured on a warm Series S run (M5 Pro), with `VICTORY_TRACE_CONTENT=<dir>,<s>,29` (the draws into
the G-buffer's sRGB target): of 600 G-buffer draws traced, 578 were `ExecuteIndirect` mesh dispatches
and 22 indexed indirect draws, and every record read, when it was encoded, a group count of
`[0, 1, 1]` (an instance count of 0 for the indexed ones). The G-buffer stayed 0.000 non-black in
every dump, also with `VICTORY_DEBUG_NO_DEPTH=1`: the draws were encoded, but no mesh group ran.

The records (32 bytes each, 32 KiB) are written on the GPU. One compute shader (`e85d0ba6846a5b5e`)
starts each frame's records as `[0, 1, 1]`, and six culling shaders then count the visible groups
into them, in three pairs over the same buffers: `b30937e9de066fa9` with `4e2edbd69fe414f1`,
`5833c0c20c5bec9b` with `f44473ebe75e0047`, `ff826810673a0198` with `898d9c1a5605ad81`. All six
were refused, so their dispatches were dropped and every count stayed 0.

- The first of each pair (an occlusion test: `image_gather4_l` of a depth pyramid) was refused as "a
  gather at a level that may not be whole". The level is whole: `ceil(log2(...)) - 1`, clamped at 0,
  minus 1, clamped at 0, then `v_cmpx_le_i32` and a `v_mov_b32` of another whole level. But those
  writes are guarded, and `Lower::whole` also asked the value each one keeps where it is skipped to be
  whole: here an unrelated product, kept only in lanes that never reach the gathers. `whole` now takes
  the gather's site and looks at a guarded write's kept value only when a lane that runs the gather may
  have skipped the write (`avoids`). Every step it follows is the lane's own, and a lane that reads
  the value at the gather went through the write's block. All three translate. In game they run, and
  the first non-zero group counts appear (`[1, 1, 1]`, `[5, 1, 1]`, `[24, 1, 1]`). Test:
  `sample_tests::gathers_at_a_level_whole_in_the_lanes_that_gather_translate`.
- The second of each pair was refused as "a cross-lane operation between host waves in divergent
  code". Each is a group of 128 threads (two console waves, so not the lockstep fallback) that ORs a
  flag over its wave (a whole-wave section: `s_or_saveexec_b64 -1`, DPP, `v_permlanex16_b32`,
  `v_readlane_b32` of lane 0). It then reads lane 63 with EXEC set to lane 0 alone, ORs the result
  into LDS word 0, and reads it back after the title's own `s_barrier`. Those reads cross the host
  waves, so they need group barriers, which went only in blocks every thread runs once. This block
  is reached by uniform branches, and a culled wave ends early (`s_cbranch_scc1` to `s_endpgm`). But
  the title's barrier in the same block already needs every live wave there together (the console's
  waits for each wave that has not ended, and the host's group barrier for it relies on the same).
  So a block outside any loop that holds one of the title's `s_barrier`s is now an exchange site too
  (`barrier_blocks`). All three translate. Over the same 3,338 dumped containers, four more compute
  shaders translate and no other outcome changes. Test, on the GPU:
  `split_wave_tests::a_whole_wave_or_one_lane_puts_in_lds_reaches_the_group_at_its_barrier` (a
  synthetic program of the culler's shape, the second console wave ending early, flags in both host
  halves). Not a block reached past a branch on an SCC computed from a `v_readfirstlane`
  (`split_scc`), which `wave_blocks` already excludes: the console wave takes such a branch whole,
  so a title may put one before its barrier, but each host wave reads its own first lane, so one
  half could skip the block while the other waits at the exchange's group barrier, a hang (test
  `wave32_tests::a_barrier_block_behind_a_first_lane_branch_is_no_exchange_site`). Once the first
  lane read is the console wave's (#254), the exclusion is only conservative for compute. But the
  culling shaders branch exactly this way: after the title's barrier, every lane reads LDS word 0
  (the group flag) and the branch is on its first lane. A first lane read of a `ds_read_b32`
  therefore splits nothing when its address is a constant, it runs in one known barrier phase n ≥ 1
  (after the title's n-th `s_barrier` on every path: `barrier_phases`, a bit set per block, any
  where a loop holds a barrier), and no LDS write of the program that can run in phase n, on any
  path, can touch the word. Atomics, `ds_append` and `ds_consume` count as writes (the last two at
  no known address; a split wave refuses them first anyway), and one at constant bytes elsewhere is
  fine. A group barrier's k-th instance pairs every live wave's k-th `s_barrier`, whichever
  instruction it is, so a write of another phase is ordered against the read, and a wave that leaves
  the read's path (or passes a barrier instruction of its own) still has its phase-n writes checked.
  A wave that ends early writes nothing more, and the console's barrier does not wait for it;
  whatever it wrote before ending is in some phase and checked the same way (`barrier_reads`; test
  `wave32_tests::a_barrier_block_behind_a_branch_on_an_lds_word_read_after_a_barrier`: accepted with
  a write to another word or to the word in an earlier phase; refused with a write or an atomic to
  the word in the read's phase, before or after the read, or on another wave's path that ends). Over
  the 4,129 refused compute and graphics containers of a boot run (Series S, offline, both with and
  without thread counts), `4e2edbd69fe414f1` and `f44473ebe75e0047` now translate and nothing else
  changes; `898d9c1a5605ad81` stays refused, as two LDS writes at computed addresses can run in its
  read's phase.

`VICTORY_TRACE_TARGETS=1` cannot show this: an indirect draw counts as drawn when it is encoded,
whatever the GPU later reads as its count.

## Heap buffer views not linear in one pool (#258)

A buffer view in a descriptor heap is written as a view of the pool buffer that holds its range
(`NativeHost::mirror`, through `PoolTable::resolve`), which needs the range to be one linear run
of one pool. The title maps its physical pages one 64 KiB page at a time, often in descending pool
order, so a large buffer is seldom one run. Such a view used to be written as the null view, and
the shader read zeros through it (the "a descriptor was not mirrored ... is not in a mapped pool"
lines, about 500 to 3,000 a run).

The culling shaders of the world's G-buffer read their instances that way. With
`VICTORY_TRACE_DISPATCH_INPUTS` (a local diagnostic) on culler `ff826810673a0198`, its root constants
`[32, 17]` name heap entry 17, a buffer of 182,452,224 bytes (174 MB) the title had filled (a third
of its first 64 KiB non-zero). The host nulled it, so the culler read all-zero instances. Heap entry
9936 (52 MB) was nulled too.

Title memory is mapped at its own addresses in this process. A view whose range the title maps
without a gap, but not as one run of a pool, now wraps those addresses (`NativeHost::title_alias`:
the backend's `wrap_title_memory` over whole 16 KiB host pages, the alignment the no-copy wrap
needs). The wrap is an allocation the heap entry names, so it is resident as the pool buffers are,
with residency sets or by per-encoder naming (`heap_entry`). A wrap is reused while the mappings
under it are the same (`PoolTable::signature`, a hash of the mappings the range crosses), made
again when they change. Every wrap a heap entry shows stays in that cache, with 64 others at most,
and the wraps over a released pool's pages are dropped from it (the heap entries that show one keep
it alive themselves). Dropping the whole cache at each pool release, as the first version did, made
a new wrap of the same range each time the entry was mirrored again, while the old ones stayed
alive in heap entries and work in flight: in the 80 s after the world loaded, the 174 MiB view was
wrapped 5 times with the same signature, the 50 MiB one 10 times and a 24 MiB one 17 times (the
log stops at 32), and that run ended in GPU timeouts and an out-of-memory error. A view with a real gap (part of its range not
mapped) stays the null view, logged at info level (`a buffer view over title memory with a gap
stays the null view`, first 64). Test: `tests::a_buffer_view_across_title_pages_out_of_pool_order_reads_the_title_mapping`.
Two title pages, mapped for real at a title address from pool pages 2 then 1, hold 64 words across
the page boundary. A compute shader reads them through a heap view: it now gets them, and was
refused the view before.

What a wrap costs: it shows the title's own physical pages, so it adds no memory to the process
and takes no process address space (Metal gives it a GPU address range of its size). Measured on a
174 MiB view of title pages mapped in descending pool order (a local test): the process footprint
(`phys_footprint`) stayed at 241 MiB through the wrap and two dispatches that read through it.
Metal's `currentAllocatedSize` counts the wrap once more (+174 MiB), as it counts each pool buffer
over the same pages. In game (M5 Pro, Series S), each large wrap is logged with both numbers
before and after (`a large buffer view wraps the title's own pages`, views from 16 MiB, first 32):
the 174 MiB view (entry 17) at +4 s, footprint 769,473,040 bytes before and after, Metal's count
+182,452,224; the 50 MiB view (entry 9936) at +29 s, footprint 4,156,533,456 before and after,
Metal's count +52,428,800.

What a wrap reads once the title changes its pages: the no-copy wrap keeps the physical pages it
was made over. After the title maps other pool pages at those addresses, or unmaps them, the GPU
still reads and writes the old pages through it, with no fault and no zeros (M5 Pro, macOS 27).
So the host follows the title's pages (`NativeHost::follow_title_pages`): at each submission, if the
pool table changed since it last looked (one comparison otherwise), every heap entry that shows a
wrap, or that was left null for a gap, is mirrored again when the signature of the mappings under
it has changed (`NativeHeap::title_views` keeps each one's range and signature). A remapped range
gets a new wrap, a range with a page unmapped becomes the null view, and a gap the title maps later
is wrapped then (logged: `a heap buffer view's title pages changed: mirrored again`, the first 64
of views from 16 MiB and of smaller ones each). An entry the title has since filled with another
record (plain stores) is dropped from the list. It
matters for entries a shader loads from the heap itself, which no bound table brings back to
`remirror_tables` (the cullers' entry 17), and for an entry the title writes once while it keeps
mapping pages into the range (Ketan-Kapse's fork saw that, #258). Here the title does not remap
entry 17: in a 300 s world run its signature was the same at the 1st, 10th, 100th and 1,000th look
(`following the title's pages under heap buffer views`, logged at those looks with the views from
16 MiB; pool-table generation 6,742 to 321,276), and each of the three large views was wrapped once.
The pool table changed about 2,600 times a second in that run, so a look runs at most once per
submission. The title copies one view into many entries (54 copies of the 174 MiB view, 12 of the 50
MiB one at the 1,000th look), so a look walks the pool table once for each distinct range, not
once an entry: a look like that one took 2.7-3.1 ms with the pool table locked, and 60-90 us now
(about 12 looks a second in that run; measured by `memory::signature_timing`, an ignored test run
by hand). Tests: `tests::copies_of_a_wrapped_view_are_followed_with_one_walk`;
`tests::a_wrapped_view_follows_title_pages_remapped_or_unmapped_under_it` maps a view's second
page to another pool page, then unmaps it: the view reads the new page, then zeros. Without the
step it read the old page both times. Limits: work submitted before the change still reads the old
pages; a range partly unmapped reads zeros as a whole, where a reserved resource reads zeros only
at its unmapped tiles (marked shortcut in `title_alias`).

In game, the G-buffer's mesh records stayed zero after the GPU ran the frame with this change
alone, as read from the CPU (a read the reset pass makes meaningless, #258). With #253 and the
view-stride table (#264) as well, the G-buffer's targets are 11 to 15 % covered in a warm world
run, against 0 to 8 % before.

## View addresses on the native host (#306)

A shader that reads a buffer view's title address from its descriptor (dwords 0-1, the start of a
BVH walk), or a texture view's size fields, gets them from the view addresses, the translator's
root SRV `t1` of the memory space (`victory_shader::emit`, "View addresses"; `HostShader::
view_addresses`). The native host bound no such table and refused every such program ("needs view
addresses or write marks"; write marks stay refused, as "needs write marks"). Each resource heap now
has one (`NativeHost::view_addresses`, `NativeHeap::addresses`): four dwords per title entry, a
buffer view's title address (dword 0, and the low 16 bits of dword 1), stride (dword 1 bits 16-29)
and record count (dword 2; see "Buffer strides in the view addresses"), a texture view's size fields
(`descriptor::texture_words`, the D3D12 path's, now shared), zeros for anything else. It is made
the first time a shader reads it, from the heap's records as they are then, and kept current by
every mirrored write (`NativeHost::mirror`). It follows the page table in the argument buffer
(`Binding::view_addresses`). Cost: 16 bytes an entry of each resource heap a shader reads it
through (16 MiB for 2^20 entries), made only for those heaps. shortcut: an entry the title copies
as bytes outside a bound table keeps its old words until it is mirrored, as for the view-stride
table. Measured offline (27 November): the 7 compute programs of the boot-dumped corpus that #304
lets translate but the host refused for view addresses now make their Metal pipelines (not run;
six walk title memory from a view's address, one reads the heap only). In the all-shaders boot set
(4,921 containers) 44 translated programs read the table: 42 compute, 2 pixel, no vertex. Test:
`heap_tests::a_compute_shader_reads_title_memory_at_a_heap_views_address` (a raw SRV's address read
from its descriptor, the word there stored to the output; refused before, and with only the
refusal lifted the converter refused the pipeline: "Resource SRV in space 2 register 1 ... is not
referenced by the root signature").

A shader can need both per-heap tables: the view-stride table (#264) and the view addresses. The
argument buffer then holds, in order:
1. the user-data block;
2. the root views, the stride table among them (root SRV 255 of the root-view space);
3. the page table (`t0` of the memory space);
4. the view addresses (`t1`).

`Binding::layout` gives the converter that order, and `NativeHost::translated_args`,
`memory_args` and `view_address_args` fill the buffer in that same order. Test:
`heap_tests::a_compute_shader_reads_both_the_view_strides_and_the_view_addresses`. Its compute
program stores through a buffer it takes from the heap itself, and loads the word at an SRV's
title address. Swapping the two memory-space tables, in the layout or in the fill, makes it read
a decoy view's word instead.

The table gives every field it doesn't hold as 0. The translator refuses a program that can see a
bit of one of them, rather than let it read 0 (`Lower::view_addresses`, `Lower::seen_bits`).
Refused fields:
- a buffer's dword 3 (format, destination select, range-check mode);
- an image's base address, minimum level and format (dword 0, dword 1 bits 0-29);
- an image's dword 2 bits 30-31, its tiling (dword 3 bits 20-27) and its dwords 4-7.

The message is "a view address field the table gives as 0: " and the field.

The bits a program can see are followed through its readers:
- `s_and_b32` with a constant keeps that constant's bits;
- `s_pack_ll_b32_b16` keeps the low half;
- `s_or_b32` with a constant and `s_lshr_b64` by a constant (of which the dword is the high half)
  are followed into their results, four readers deep, as long as nothing reads their SCC;
- any other reader, or a phi, sees every bit.

Measured offline (27 November, the 4,921-container all-shaders boot set, main 465db23b + #304):
- 44 translated programs read the table: 42 compute, 2 pixel, no vertex.
- 40 of them read a buffer's dword 1. 23 keep its low 16 bits with `s_and_b32`, 12 pack them with
  `s_pack_ll_b32_b16`, and 5 shift the address pair right with `s_lshr_b64` before masking.
- None reads an image's dwords through the table.
- With the check, all 44 still translate:
  - 40 have the same output;
  - 3 no longer load a dword whose readers are never lowered;
  - 1 is changed only in its refusal message.
- 11 programs that were already refused now name the field (10 a buffer's stride or swizzle, 1 its
  record count or format).

Test: `gfx_refusal_tests::view_address_fields_the_table_gives_as_0_are_refused`. It runs one
refusal per field class, plus accepted reads of a buffer's address (whole dword 0, dword 1 masked
to 16 bits, or shifted with `s_lshr_b64` and masked) and of an image's size fields. Without the
check, a read of a buffer's whole dword 1 translated.

### Buffer strides in the view addresses

The console compiler makes a structured buffer view raw before byte accesses. It reads the
stride from dword 1 bits 16-29, multiplies the record count (dword 2) by max(stride, 1), clears
the stride and sets out-of-bounds mode 3 in dword 3. When the table held only a view's address,
every such load was refused ("a buffer's stride or swizzle"): 255 programs of a warm world run, 247
of them compute. The table now holds a buffer view's four dwords as the console's untyped
descriptor has them (`NativeHost::address_words`):
- dword 1 bits 16-29: the stride. That is a structured view's stride or a typed view's element size
  (`descriptors::stride_word`), and 0 for a raw view, a constant buffer or a whole buffer
  (`descriptors::raw_buffer`).
- dword 2: the record count. That is elements with a stride, else bytes.
- dword 3: `VIEW_BUFFER_DWORD3`: identity destination select, format 32_FLOAT and resource
  level 1. Bits 28-29 hold out-of-bounds mode 1 with a stride and mode 3 without
  (`VIEW_BUFFER_OOB_STRUCTURED`, `VIEW_BUFFER_OOB_RAW`).
- The swizzle bits (dword 1 bits 30-31) are 0, as in a title's view.

shortcut: a typed view's dword 3 has the untyped format, not its own.

Measured offline (27 November) on those 255 programs: 74 translate, and 181 stop at later
refusals (lane-mask joins, an image's dwords 5-7, a loaded buffer descriptor's dword 1). Tests:
`heap_tests::a_compute_shader_reads_a_structured_views_stride_and_record_count`, which reads a
structured SRV's dwords 1-3 on Metal, and the accepted reads in
`view_address_fields_the_table_gives_as_0_are_refused`.

shortcut: the kind of view is the load's (eight dwords at a whole descriptor are an image), and the
swizzle reads as the identity.

## Descriptor dwords read as data

The console compiler computes sizes from descriptor bits instead of querying them; those fields
are rebuilt from the host view (`Lower::descriptor_data`). Other dwords have no host value and
were refused as "a descriptor's address or format read as data". That text is now followed by a
colon and the dword: "a buffer's base address (dword 0)", "a buffer's dword 1 (address high
bits, stride, swizzle)", "a buffer's dword 3 (format and type)", "an image's base address (dword
0)", "an image's dwords 5 to 7" or "a dword outside the descriptor". Nothing is lowered: the
native Metal host serves view addresses (see "View addresses on the native host") but no formats, and would have to first.
`pipeline::pieces_tests::the_host_refuses_a_descriptor_dword_read_as_data_naming_it`
(`victory-gpu`) checks the refusal on the native host's own path: the container made from the
title's pieces, the host's options. The story-load log of issue #110 (Series S) has three
compute programs refused with the old message.

Most pixel programs refused with these reasons (and with "a descriptor read only as data") read no
descriptor word at all (27 November, offline corpus). The console compiler reuses a register that
last held a descriptor dword (often a sampler's, loaded from the sampler heap) for a constant it
loads in an `if`; at the join after it, the skip edge brings the stale dword together with a flag
mask set to 0 (`s_mov_b64 s, 0`), and the code that reads the register sits behind
`s_and_b64 vcc, cond, flag; s_and_b64 s, vcc, exec; s_cbranch_scc0`, which those lanes never
enter. `Lower::dead_on_edge` could not show that: a branch on a joined mask (`Atom::Phi`) known 0
or every lane on the path now goes one way (`Lower::joined_mask`), `konst` knows `s_mov_b64` of
an integer inline constant and `s_and` with a zero dword (and its SCC), a `v_cmp_*` into SGPRs
passes the stale value on as scalar arithmetic does, and a result the path fixes (an AND with 0)
holds nothing of it. Corpus: 651 programs gained (493 pixel, 158 vertex), 0 lost, 0 changed; of
the world run's 14,508 pixel draws dropped for these two reasons, 12,156 were of pipelines whose
every stage now translates. Tests (`victory-gpu`, Metal): `stale_tests`. Still refused, rightly
for now: texture size queries in a waterfall loop that test dword 1 against 0 for a null view
(the view-address table holds dword 1 as width bits only, so a texture of width 4k + 1 would
read as null); buffer dwords 2-3, which no host table holds.

Image dwords 5 to 7 (array pitch, metadata address, compression) now read 0 (`descriptor_data`,
shortcut, 28 November), and so does any descriptor dword read as data where it meets other values
at a join (`descriptor_bits`: the reasons above, "a descriptor read only as data" and "a constant
buffer's descriptor read as data"), in a data join and in a lane-mask join. No HLSL reads
descriptor bits; the reads the console compiler makes of them (sizes, raw views of a buffer) sit
on the path of the load, so a dword that reaches a join is a register the compiler reused. The
world's deferred-light programs (wave32, 32 threads, about 120 KB each) park live SGPRs in
descriptor registers and copy them back around each `image_sample`, and load a light record with
`s_buffer_load_dwordx8` into registers the same code uses as an image descriptor on another
path: in `0e117205dde09753` `vcc_hi` is tested (`s_bitcmp0_b32 vcc_hi, 20`) after a join with
dwords left from earlier descriptor loads, on an edge a flag set to 0 keeps from reaching the
test, which `dead_on_edge` does not follow that far. Wrong only if a program did read a
descriptor dword past a join (it would read 0 there).

## The device's own memory ceiling, logged at native host startup (issue #8)

16 GB Macs still run the title out of GPU memory past the menus (`kIOGPUCommandBufferCallbackErrorOutOfMemory`),
even with `--console series-s` (#12) and per-encoder residency (#76, which cut real usage from
12+ GB to about 9.4 GB on an 18 GB Mac). Nobody has set a real threshold for when that is a problem
on a given Mac, because nobody has the numbers from a run that hits it on current `main` yet.

Rather than guess a threshold (an earlier version of this change tried a 90%-of-recommendation
warning; worked out against this M2 Pro 16 GB's own real numbers it was silent exactly on the
Series S default #12 steers 16 GB Macs to — `Metal` here reports `recommendedMaxWorkingSet: 12,124
MiB`, and Series S's 7,776 MiB budget is 64% of that even before any host overhead, so a fixed
10% margin never fires where it matters), `native::glue::with_title`
(`crates/victory-gpu/src/native/glue.rs`), the one place the native Metal device is made (lazily,
on first GPU use), now always logs `MTLDevice.recommendedMaxWorkingSetSize`
(`Metal::recommended_working_set`, `crates/victory-metal/src/metal.rs`) alongside the console
name and its declared `title_memory` (GPU-optimal plus standard), in the existing "native host:
Metal" line. No switch is needed: it is a log line, not a behaviour change ("dropped work is
logged loudly", above). Every tester's log now carries the numbers needed to work out a real
threshold, or to see the host's own overhead (texture copies, heaps) once compared against a
run's actual peak Metal memory.

Tests: `victory_metal::metal::tests::recommended_working_set_is_nonzero_on_real_hardware` (the
real device on whatever Mac runs the test suite; on this M2 Pro it reported 12,124 MiB).

## Reverb and convolution DSP (`chad.dll`), a CPU fallback (`VICTORY_GUESS_DSP=1`)

The title's audio DSP coprocessor convolves blocks of float samples with an impulse response
(`crates/victory-hle/src/modules/dsp.rs`). `XDspActivate`'s `StreamDesc::ir` holds that response,
but in the hardware's own partitioned frequency-domain layout, which is not documented anywhere
we have found, so it is not read. Issue #17: outputs were always zeroed, so reverb and
convolution effects played back silent (dry sound, which does not go through this path, was
unaffected).

Guess, marked in code (`synthetic_ir`, `convolve_merged`), off by default behind
`VICTORY_GUESS_DSP=1` since it is unverified against a real run (without the variable, behaviour
is unchanged: silence, as before): a convolution stream (`StreamDesc::kind == CONVOLUTION`) gets a
fixed, deterministic ~100 ms (4,800 tap at 48 kHz) synthetic decaying-noise impulse, unrelated to
the title's own response, convolved with the input per channel and scaled by the command's
`gains`. `gains` is itself a second guess: it is read as one gain per output channel, as likely to
actually be `[dry, wet]`, in which case a mono stream's second element is never read.

The tail is carried across blocks (`Stream::history`, the last `ir.len() - 1` frames of input,
fed back in as the start of the next block) rather than reset at every boundary, so a continuous
input does not click at the block rate. This is still a CPU-fallback coloration, not the title's
actual reverb: a real room's decay is 300 ms or more, well past what a per-sample scalar
convolution can do without an FFT-based engine. `Stream::scratch` is reused every call instead of
allocating, and the title's input is always copied into it before any write to the output buffer,
in case the two happen to alias.

`VICTORY_TRACE_DSP=1` logs the leading floats of a convolution stream's own `StreamDesc::ir` at
activation (`trace_ir`). It exists to let a future run with the real game settle what format that
data is actually in (time-domain samples, in which case `ir_floats` for a given `ir_seconds` would
land near `ir_seconds * 48000`, or something else) and replace this guess with the real thing.

`VICTORY_DUMP_DSP=<dir>` goes further: at activation it writes a convolution stream's whole own
`StreamDesc::ir` to `<dir>/stream<n>_ir.f32` (the title's bytes, not the synthetic tail; capped at
16 Mi floats, logged as truncated if longer), and each `XDspSubmitCommand` appends the command's
gains and input block to `<dir>/stream<n>_blocks.f32` and the block the stream wrote — silence, or
the `VICTORY_GUESS_DSP` result — to `<dir>/stream<n>_out.f32`. A run with both set captures what
the title sends and hears, to validate the guess against the real effect once the hardware's
partitioned IR layout is known.

Tests (`modules::dsp::tests`): `convolve_merged_is_a_causal_fir_per_channel_scaled_by_gain` and
`convolve_merged_reads_history_as_the_frames_before_the_block` (the convolution math against
hand-computed values), `a_block_split_matches_one_convolution_over_the_whole_signal` (splitting one
signal into two blocks and carrying history gives the same result as convolving it in one go — the
property that stops the tail from clicking at block boundaries), and
`convolution_stream_as_the_title_drives_it` (the exported `XDspSubmitCommand` path, with
`VICTORY_GUESS_DSP` set, no longer returns digital silence for a convolution stream with real
input), and `dump_captures_the_titles_ir_blocks_and_output` (`VICTORY_DUMP_DSP` writes the title's
own IR, and every command's gains, input and output blocks, in order).

## Depth-stencil textures are not watched or uploaded from CPU memory

Depth and depth-stencil textures (`Format::D32Float`, `Format::D32FloatS8`, `Format::D16Unorm`,
including placed or typeless depth resources such as DXGI formats `19`/`20`/`45` and `118..=120`)
are GPU targets whose CPU tile layout in `resource::native_info` describes plane 0 alone (4 or
2 bytes per texel), whereas Metal backs combined depth-stencil formats as `MTLPixelFormat(260)`
(`Depth32Float_Stencil8`, 8 bytes per texel). Watching or uploading CPU memory for those textures
previously caused `victory_metal::gfx: upload_texture skipped: D32FloatS8` warnings in
`transfer_fits` (`row_pitch` smaller than Metal's combined depth-stencil row bytes) and wasted
CPU hashing/detiling work. `Title::realize` and `Title::sync_keys` now exclude `desc.format.is_depth()`
from `watch`, `VICTORY_MAGENTA`, `unloaded`, and CPU detile/upload (test: `victory-gpu`
`native::texture_tests::depth_textures_are_not_watched_or_uploaded_from_cpu_memory`).

## Depth textures compute writes: colour twins (#258)

The tiled light draws test equal against a 1280x720 depth target of DXGI format 118, the console's `D16_UNORM_S8_UINT` (119 and 120 are its `R16_UNORM_X8` and `X16_G8_UINT` reads; the host keeps it as 32-bit float depth with stencil). Compute programs write that material depth through unordered views: `R16_UNORM` views of its depth plane and `R8_UINT` views of its stencil plane (`VICTORY_TRACE_TARGETS=1` lists it among the UAVs of `f0c91b4d2578f5bf`, 160x90 groups, `570f1473f4641b30` and `bad01f53f53248e5`). A colour view of a depth texture is the depth texture itself (`create_texture_view` keeps its own format), so compute wrote the depth texture directly. Metal does not support shader writes to a depth texture, and on an M4 Pro the result is this: a blit copy of the depth plane reads what compute wrote, but a later render pass's depth test does not see it. A full-screen draw tested `LessEqual` against compute-written depths failed at every pixel, where it should have passed at 10 of 16 columns (the test below, run on main).

Now a depth texture gets a colour twin at the first unordered view of its depth plane (`NativeHost::depth_twin`; `VICTORY_DEPTH_TWINS=0` turns it off): `R16Unorm` for a 16-bit depth texture (the shadow maps, DXGI 53 as a depth target), `R32Float` for a 32-bit one. An `R16_UNORM` view of a 32-bit depth plane writes the twin as 32-bit floats. Its one-channel colour views of the depth plane made after that read the twin too; entries mirrored before it are mirrored again. The newer contents move between the two where the use changes (`NativeHost::twin_copies`), as VIctory's `twin.rs` does on the D3D12 path: into the twin before a compute pass, and into the depth texture before a render pass whose depth target it is, or a copy to or from it. Each copy happens once per change, through a buffer of the twin's (`DepthFromDepthStencil`). A depth clear makes the depth texture the newer side without a copy.

Guesses:
* The title writes `k / 65535` through an `R16_UNORM` view of the 32-bit plane, the very float its draws rasterise for that depth. VIctory's D3D12 path converts `k` to `k * 0x37800080`; the host writes the float the shader stores.
* A render pass that only reads the twin (a colour view of a shadow map sampled by a draw) after draws into its depth texture, with no compute pass between, reads the twin's older contents. In the frames we looked at, compute runs between the shadow draws and the lighting.
* Entries are mirrored again only where the host checks them at submission (the bound tables' entries): a shader that indexes the heap may read an entry still showing the depth texture.
* The stencil plane's unordered views (`R8_UINT`) still reach nothing.

Cost: one twin and one buffer of its size for each depth texture compute writes. One world run made 108 twins: the material depth, 3 other depth targets of 1280x720 to 1920x1080, and 104 shadow maps of 512x384 to 1536x1536. Had they all been alive at once, that would be about 97 MB of twins and as much in buffers.

Measured on one Mac (M4 Pro, 48 GB, Series S, warm cache), on a working branch (main of 27 November morning, plus a placed-view alias like the one main has now, and other changes of ours), not on this main:
* Before the twin of the material depth (r162), the prologue's window scene was washed out: in the screen captures from 199 s to 423 s, the mean luminance was 164 to 174 of 255, and no pixel was dark (below 16). With it (r163, r164), the scene was lit at sunset, close to the Wine route's picture of the same scene: a mean of 26 to 29, with 67 to 72% of pixels dark. We did not measure why the scene was washed out without the twin; the light draws then tested against a depth plane compute had not written.
* With twins both ways (r164): 480 s with no GPU error, 1.8 to 1.9 world frames a second.

Tests: `tests::a_draw_tests_against_the_depth_compute_wrote_through_an_unordered_view` (needs the Shader Converter: in one list, a depth clear, a `tex_fill` dispatch that writes x / 255 through an `R32_FLOAT` unordered view of a `D32_FLOAT` texture, then a full-screen draw at depth 0.02 tested `LessEqual`; it passes at columns 6 and up, and on main it passes nowhere), `texture_tests::an_unordered_view_of_a_depth_texture_writes_a_colour_twin`, and `texture_tests::a_depth_twin_and_its_depth_plane_hold_the_same_texels` (DXGI 118 and a 16-bit depth texture: what the twin holds goes into the depth plane and back bit for bit).

## Command buffers of at most N passes

macOS stops a command buffer that runs too long on the GPU (`kIOGPUCommandBufferCallbackErrorTimeout`). With the ray-traced passes running, a world submission of 72 to 80 passes ran 6 to 21 s on an M4 Pro and failed so again and again (13 time-outs in 600 s of one run), although none of its passes alone took more than 1.4 s when each was timed in a command buffer of its own. With `VICTORY_PASSES_PER_BUFFER=<n>`, `submit_passes` (`victory-metal`, `gfx.rs`) commits a submission's passes in command buffers of at most `n` passes. The default is 0, the whole submission in one command buffer as before: the split has only run on that one Mac. 1 is the safest value: with 8, three runs of 490 to 800 s had no time-out, but a later run with more of the world's passes running had one piece of 8 take 6.3 s and time out (11 time-outs in 339 s), while one pass per command buffer had no GPU error in 337 s. A slower GPU, or a heavier pass, may need fewer passes a piece than a faster one; the cost of 1 is more command buffers.

The earlier pieces are committed as they fill: each gets the submission's waits again (values already reached, since the host keeps unsatisfied waits on the CPU) and the heaps' residency sets again (`Metal::next_piece`): a residency set keeps its allocations resident only for the command buffers it is given to. A first version that attached them to the first piece alone faulted the GPU (`kIOGPUCommandBufferCallbackErrorPageFault`) in the world's later passes, the tiled cascade shadows and the final gather, 90 to 135 s into every run; it did not with every submission waited for on the CPU, and two runs of 307 and 339 s with the sets attached per piece had no fault; encoders stay in order across pieces through the order fence where it is in play, and through Metal's tracking otherwise.

What a submission owes if the GPU fails it (`Owed`: its bottom-of-pipe value writes, its fence signals, the queue's order event) is split among its pieces. Each piece owes the value writes of its own passes and ends by raising the submission's piece event to its number; the next piece waits for that value. The last piece owes the fence signals and the order event too. A piece the GPU fails pays its own share on the CPU, once (the writes, then the event), so the next piece still runs, the later pieces' writes and signals come from the GPU, and nothing is paid twice. Each piece is in flight on its own (`recover_lost` pays one Metal never reports on, which also lets the next piece go). Guess: a failed piece does not raise the piece event itself; if Metal did run that signal, the next piece could start before the CPU has written the failed piece's values. Tests: `owed_tests::a_long_submission_is_cut_into_command_buffers_of_at_most_n_passes`, `owed_tests::a_failed_earlier_piece_pays_its_own_writes_and_the_later_pieces_still_run` (the test drops the first piece's command buffer, as the GPU would fail it: no real GPU failure is made), `heap_set_tests::the_next_piece_of_a_split_submission_attaches_the_heaps_again`.

### What a GPU time-out is

Measured on an M5 Pro with 48 GB (macOS 27.0, Series S, headless, main `905e7aad`, 2026-11-27), where five world runs of five stopped between 110 and 190 s with `kIOGPUCommandBufferCallbackErrorTimeout`, at 48 to 50 % free memory, no swap and 22 GB wired:

- Each time-out is a wait, not work that ran too long. The kernel logs `(IOGPUFamily) Cmd queue <private> sleep port_name <n> value: <v> timed out!` 5 s after the stall (`/usr/bin/log show --predicate 'process == "kernel" AND senderImagePath CONTAINS "IOGPU"'`), and `<v>` is the value an `encodeWaitForEvent` of ours waits for: the queue's order event (with the heaps' residency sets every command buffer waits for the one before it on its queue), or with `VICTORY_PASSES_PER_BUFFER` the piece event of the piece before.
- So the command buffer Metal fails is the one behind the one that is stuck. The stuck one never ends and is never reported on (`recover_lost` pays for it later); every 5 s the next wait behind it times out.
- One pass per command buffer does not help when a pass itself never ends: three runs with `VICTORY_PASSES_PER_BUFFER=1` stopped the same way, the piece after the stuck pass timing out. It names the pass, with `VICTORY_TRACE_GPU_FAILS=1`.
- What was stuck, in the run that read the records back, was an indirect dispatch of 510 x 1,182,027,346 x 1,314,076,063 groups ("ExecuteIndirect on the native host", "The most groups a record may ask for"). With a limit on dispatches alone, four runs of 170 to 300 s had no GPU error; a fifth stopped at 126 s on a command buffer whose seven indirect dispatches all had sane counts, cause not found (mesh dispatches were not gated in that run).
- After such a run the kernel can restart the GPU once the process is gone (`GPURestartSignaled ... type=2`), WindowServer with it, and keep the process's GPU memory (19 to 21 GB "In use system memory" with no game) until a reboot.

A counted append (load a count, `buffer_atomic_cmpswap` it for one more, again until the swap returns what was loaded), the one loop in the pass that was stuck in the fifth run (compute `87893da2440de5c3`), is not the cause as far as our own programs of its shape show: `victory-gpu` `heap_tests::a_swap_agrees_with_a_load_of_the_same_dword_inside_a_view_and_past_its_end`, `lanes_that_swap_one_count_at_once_all_append` and `lanes_of_split_waves_that_swap_counts_at_once_all_append` (256 lanes in 64-lane waves on the host's 32-lane waves) all end with every lane in.

## Zero-count dispatches, draws and buffer blits

D3D12 treats zero-sized dispatches (`Dispatch`, `DispatchX` with thread counts), draws with zero
vertices/indices/instances/mesh groups, zero-width/height viewports, and zero-length buffer copies
or fills as no-ops, whereas Metal's validation layer asserts on `dispatchThreadgroups` with a zero
grid dimension, `fillBuffer`/`copyFromBuffer` with a zero byte length, and `drawPrimitives`/
`drawIndexedPrimitives` with `instanceCount == 0`. `NativeList` now skips recording empty direct
commands (after checking the bound pipeline), and `victory-metal` skips encoding zero-sized direct
dispatches, blits, draws and non-positive viewports (tests: `victory-metal`
`zero_sized_dispatch_and_blit_do_not_fail_validation` and `zero_count_draws_do_not_fail_validation`,
`victory-gpu` `native::tests::zero_count_commands_are_not_recorded_and_still_check_pipeline`).

## Described pipeline rectangle-list draws

A described pipeline's rectangle list (`topology = 17`, a draw or a non-indexed indirect one) is
drawn as a guessed pipeline's is: four-vertex `TriangleStrip` instances, one a rectangle,
`draw_instanced(4, vertices / 3, 0, first / 3)` (`glue::described_rect`, cached per pipeline,
targets and vertex strides in `Title::rect_pipes`).

* The vertex shader is translated with `Options::rect_list`. Its input signature gets each
  element again for corners 1 and 2, semantic index `index | k << 28` on new rows
  (`emit::rect_corner_inputs`), and its body runs once for each corner (`lower/rect.rs`, as a
  guessed program reading `v0` does). Corner k reads corner k's copies (`Lower::rect_input`).
  The fourth corner's outputs are the two other corners' less the right-angle one's (the corner
  opposite the longest `SV_Position.xy` side), so only an affine vertex shader is exact.
* The description's input layout is made per instance for the three corners
  (`Graphics::for_rects`): corner k's elements at the element's offset plus k vertices, each
  stream stepping three vertices. A slot's vertex stride is the one bound, or where its elements
  end when none is bound.
* Seen 27 Nov 2026: Grand Theft Auto VI's HDR scene had a half-screen orange triangle in some
  frames, from a described pipeline (`vertex 5ec7617b`, `pixel 813a7ead`) drawing a rectangle list
  of POSITION, COLOR and TEXCOORD vertices into the scene target (reading the depth texture).
  Drawn as triangles, each full-target rectangle became its diagonal half.
* Indexed rectangle lists of either kind are still not drawn as rectangles; a described one is
  drawn as triangles, and so is one whose vertex shader does not translate so (logged once:
  "a described vertex shader not translated for rectangle lists").

Tests: `victory-shader`
`emit::gfx_refusal_tests::a_described_rect_list_vertex_program_reads_each_corners_inputs`,
`victory-gpu` `native::state::tests::a_rectangle_list_state_reads_three_corners_an_instance`, and
for the triangle fallback
`native::tests::texture_tests::described_rectangle_list_draw_uses_supplied_triangles`.

## Texture copies on the native Metal host

`CopyResource` between textures and `CopyTextureRegion` (subresource-to-subresource, buffer
footprint-to-subresource, and subresource-to-buffer footprint) now record `BlitOp::CopyTexture`,
`BlitOp::CopyBufferToTexture`, and `BlitOp::CopyTextureToBuffer` on `Pass::Blit` and encode them
into `MTLBlitCommandEncoder` in order with adjacent GPU work (format casting via
`newTextureViewWithPixelFormat` where formats differ within a castable family; depth/stencil and
`B4G4R4A4_UNORM` buffer transfers still fall back to `dropped_op` rather than tripping Metal
validation). Previously all texture copy calls incremented `texture_copies` (`DROPPED_KINDS[3]`)
and left destination textures empty (tests: `victory-metal`
`texture_blit_ops_copy_between_textures_and_buffers` and `victory-gpu`
`native::texture_tests::copy_resource_and_copy_texture_region_run_on_the_gpu`).

What the world needs on top of that (`fix/texture-copies-port`):

* **Formats no view joins** but D3D12 copies between go through a scratch buffer of the host's
  own, texture to buffer to texture: plain texels against compressed blocks of the same byte size
  (one texel a 4x4 block; the world copies `R32G32B32A32_UINT` into BC3 and `R32G32_UINT` into
  BC1, a GPU encoder's output), and a depth plane against a plain 4-byte format
  (`victory_metal::gfx::direct_copy` decides; the region given is in source texels, the
  destination's is scaled). The checks sit in `glue::copy_formats` and `glue::dest_size` too, so a
  copy the host cannot do is refused at record time and counted, not skipped silently by the
  encoder.
* **Depth and stencil to and from buffer rows.** A `D32FloatS8` texture moves its 4-byte depth
  plane (`MTLBlitOptionDepthFromDepthStencil`); `D32Float` and `D16Unorm` copy as they are. A
  footprint naming both planes in one element (`D32_FLOAT_S8X24`) is still refused; the rows of
  a depth plane come as `R32_TYPELESS` and the like. Rows and texture must have texels of one
  size (`glue::rows_match`).
* **`ResolveSubresource`.** Host textures have one sample, so a resolve copies what both
  subresources hold (the console clips a larger source to the destination, as the D3D12 host
  does).
* **The destination is the GPU's** (`Title::gpu_written`), as a render target's is: the CPU-side
  sync (`sync_keys`, fed by the scanner and the hot and blank probes) never fills it from title
  memory again, which holds none of what the copy wrote. A sampled texture not uploaded yet still
  takes its first upload at that submission, before the copy runs, so a region copy keeps the
  rest. Guess, marked in `textures.rs`: a title does not refill a copy's destination from the CPU
  later; if one does, that write is not seen.
* **Counters.** "native draws so far" has `copies_recorded` next to `texture_copies` (refused);
  with `VICTORY_TRACE_TARGETS=1` the frame line has `copies=` (since the last present) and each
  kind is logged once ("a texture copy (first of its kind)"). A refused copy warns once per
  operation and pair of title formats ("a texture copy refused (first of this kind)").

Tests (`victory-gpu`, `native::texture_tests`, each failing before): `copy_resource_copies_a_texture_on_the_gpu`
(and its destination keeps the copy when its title memory changes),
`copy_texture_region_moves_plain_texels_into_compressed_blocks`,
`resolve_subresource_copies_what_both_hold`, `depth_stencil_rows_go_in_and_out_as_the_depth_plane`.

## Output stores follow their values (#181)

The export lowering (`Lower::export`) makes every channel's value first and then emits the
`storeOutput`, `storeVertexOutput` or `storePrimitiveOutput` calls of the export back to back. It
used to store each channel as soon as its value was made, so a float cast or a 16-bit conversion
of the next channel sat between two stores of one output. Apple's Metal Shader Converter gathers
the component stores of one output into a vector at the first of them, and the library it wrote
then used that later value before its definition: `metal-as` refuses such a module ("Instruction
does not dominate all uses"), and Metal's compiler failed on mesh programs whose outputs a pixel
stage reads. Only the order of instructions inside the export's block changes, so the DXIL of
most programs with exports differs byte for byte from before while what they compute does not.
Tests: `emit::export_tests` (the order, for pixel, vertex and mesh programs) and
`native::mesh_tests::a_converted_mesh_program_with_run_time_parameters_is_valid_ir` (a converted
mesh program listed with `metal-objdump` and assembled again with `metal-as`; skipped without the
converter or the Metal toolchain).
## Descriptor-heap 64-bit offsets, carries, joins, and SALU-shifted waterfall loops

Programs that index a 32-bit `Table`, `ResourceHeap`, or `SamplerHeap` root parameter with an
unmasked 32-bit index or across control-flow branches form a 64-bit SMEM or global-load address
whose high dword carries the upper bits of the shifted/sign-extended 32-bit byte offset (`s_lshr_b32`,
`s_bfe_u32`, `s_ashr_i32`, `s_lshl_b64` / `v_lshlrev_b64` high dword, `v_add_co_ci_u32`, `v_mad_u64_u32`),
masks the 64-bit pair (`s_and_b64`, 40/48-bit `s_bfe_u32` / `v_bfe_u32`), or joins `0` and
`s_addc_u32` across phis and selects (`#258`). `Tracer::high_half_with`, `Tracer::compute_sums`,
`Tracer::compute_joined`, and `infer_roots` now trace those high halves by matching the exact SSA
offset term and shift or multiplier against the low half's single dynamic addend
(`Tracer::extract_low_offset` / `Tracer::offset_high_half`, rejecting mismatched shifts, bit-fields,
multipliers, and shared-only user parameters), precompute phi joins and additive heap bases in
linear `O(V + E)` time via iterative Tarjan SCCs (with `Tracer::edge_taken` memoized per CFG edge),
and recognize waterfall loops whose `v_readfirstlane_b32` index is shifted in SALU inside the loop
(`Tracer::first_lane_in_offset`).
Tests: `victory-shader` `resource::tests::heap_addresses_with_shifted_offsets_carries_and_joins_trace_cleanly`,
`resource::tests::deep_diamond_phi_and_add_chains_track_in_linear_time`, and
`emit::gfx_refusal_tests::guessed_wave64_pixel_salu_shifted_heap_waterfall_translates`.

## Vertex-ID rectangle lists (#360)

When `Options::rect_list` translates a guessed vertex program whose `Reg::V(0)` (`SV_VertexID`)
input is read outside the ignored vertex-index operand of a `VERTEX_TABLE` stage-in fetch
(`Lower::rect_vertex_id`, `emit/lower/rect.rs`), the program has no fetched position from which
`Lower::vertex_fetch` alone can extrapolate the fourth corner, and strip instances past the first
would otherwise see `SV_VertexID = 0..3` again. `Lower::run_rect_by_id` lowers the program body
three times in one invocation — with `v0 = 3 * SV_InstanceID + 0..2` and `v3` (`SV_InstanceID`)
fixed at `0` as on the console's non-instanced rectangle list — finds the right-angle corner
opposite the longest `SV_Position.xy` side, extrapolates the missing fourth corner `p + q - r`
across every exported float channel (keeping bitwise-equal shared corner values exact), and emits
the `storeOutput` calls back to back after all values are computed. Programs with unordered-access
writes are refused rather than replaying side effects.

Tests: `victory-shader` `emit::export_tests::a_vertex_id_rect_list_export_makes_every_value_before_its_first_store`,
`victory-gpu` `native::tests::mesh_tests::a_rectangle_of_a_vertex_id_program_covers_half_its_rectangle`
(64 of 64 texels) and `vertex_id_rectangle_lists_draw_half_tiles_multiple_rects_and_interpolated_params`
(32 of 64 half-tile, multi-rectangle direct and indirect draws, and interpolated `TEXCOORD` across
the fourth corner).


## Mip-aware texture probes and pool-coalesced partial reads (#179)

In GFX10 `Addr2` layouts (`Layout::tiled` and `Layout::linear`), mips are packed smallest-first:
`[0..64 KiB)` holds the mip tail (with bytes `0..1024` left as zero padding for 2D surfaces),
followed by coarse-to-fine standard mips, with mip 0 occupying the top ~75% of the range. When
world streaming maps physical pages for coarse mips while fine mips stay unmapped (`NO_ACCESS`):

* `probe_nonzero` previously checked only `(size / 2) & !4095` (inside mip 0), missing coarse-mip
  writes into blank textures until the scanner's round; `probe_read` sampled `0..256` (zero tail
  padding) and seven offsets mostly inside mip 1 and mip 0, so coarse-mip updates left
  `sampled_sees_writes = false` and forced immediate whole-texture re-reads (`every = 1`, up to
  `WHOLE_BUDGET = 8 MiB` per submission). `probe_nonzero` now also checks the coarsest and
  first-tail mip centers on multi-mip layouts, and `probe_read` adds one `256`-byte center sample
  per mip level (`Layout::offset` with XOR and pipe-bank XOR applied), while resetting `skipped`
  whenever a whole read runs.
* `PoolTable::read_mapped_with` coalesces adjacent mapped title-address pages across pools and
  out-of-order 64 KiB physical pages into one `safe_read` per contiguous resident span and zeroes
  unmapped fine-mip gaps without `mach_vm_region` traps (`read_texture_chunk` also coalesces
  adjacent readable VM regions for non-pool ranges).
* `detile_changed` (`Tracked::uploaded_mips`) skips staging allocations, CPU detiling, and
  `replaceRegion` uploads for never-uploaded blank textures and all-zero non-tail mips (guess:
  fresh `MTLStorageMode::Shared` textures start zeroed; Metal does not document it), and sampled textures enter `WATCH` once their
  first submission read populates `seen` rather than being hashed twice at realization.

Tests (`victory-gpu`): `native::memory::tests::read_mapped_with_coalesces_adjacent_pages_across_pools_and_zeroes_gaps`,
`native::textures::tests::mip_aware_probes_catch_tail_and_coarse_mips_in_reverse_mip_layout`,
`native::textures::tests::read_texture_chunk_coalesces_adjacent_readable_regions`, and
`native::tests::texture_tests::streamed_mips_across_scattered_pool_pages_upload_each_step_without_whole_rereads`.


## World geometry drawn as shards (fixed)

The world's G-buffer base colour (1280x720, sRGB) is right for most meshes, but some are long
triangles fanning from a point, and thin streaks run to the horizon. Offline findings (27 November,
traced run of 3 world frames, `VICTORY_TRACE_PASSES`, Series S, M5 Pro):

- The pass trace dumps targets once a present, so it cannot say which draw made the shards. The
  G-buffer's 869 draws of those frames are 840 `ExecuteIndirect` mesh draws (one record each) and 29
  vertex draws, from 41 vertex and pixel program pairs; every dispatch of the window ran, and 135
  G-buffer mesh draws were dropped for want of a pipeline (`b94321581f232286` 126 times): missing
  meshes, not shards.
- The mesh programs are NGG groups of 128 lanes (two console waves). They read meshlet headers and
  16-bit local indices from heap views, quantised positions (24, 24 and 16 bits) from a buffer built
  around a root address, and pack small meshlets several to a group. Their vertex and primitive
  limits (128) are the host's. Nothing in the common program (`35d5616385a0a8f8`) was found wrong.
- Candidates, unproven: heap buffer views with a gap stay the null view and read zeros (40 logged in
  that run, 13 KB to 130 KB, at title addresses of streamed data); a 64 KiB title page not wholly in
  one mapping is a null page in the page table, which every vertex and mesh program now reads root
  views through ("Bring-up diagnostics", `VICTORY_TRACE_ROOT_SPAN`); and one access is one lookup,
  so an access's later dwords past a page boundary are read from the pool, not the title's next page.
- Next run: `VICTORY_DEBUG_PS=id` with the pass trace names the pixel program, so the draws and the
  vertex program, of each shard.

Found (28 November): the shards came from many pixel programs, so from the shared mesh path. A
capture of two mesh programs' exports (`VICTORY_DEBUG_MESH`) showed each 128-lane group's vertices
in blocks of 32 lanes, each block coherent but placed by another transform (one block behind the
camera), and triangles joining vertices across a block edge: the shards. The programs take their
group's meshlet so: lane 0 alone (`s_mov_b64 exec, 1`) shifts the group id out of v5 (wave 0) or
reads it from LDS after a barrier (wave 1), then with EXEC whole again `v_readfirstlane_b32` gives it
to every lane. The console's first lane is lane 0 of the 64-lane wave; a 32-lane host wave read its
own first lane, and in lanes 32-63 and 96-127 that lane never got the id, so they drew another
meshlet's data. Mesh programs now read the console wave's lane 0 through group-shared memory for a
`v_readfirstlane_b32` with EXEC whole in a block every thread runs once (`Lower::mesh_first_lane`;
GPU test `a_mesh_program_reads_the_console_first_lane_from_the_upper_host_wave`). The next traced
frame's base colour shows the scene (a room, people, a window on a yard); thin streaks remained
over the yard. Those were the second console wave's vertex and primitive counts: lane 0 alone
(EXEC = lane 0 again) reads them from LDS after a barrier and `v_readfirstlane_b32` passes them on,
in a block only that wave runs, where no group barrier can go. A word of LDS read at a constant
address in one barrier phase that no write of that phase touches reads the same in every lane, so
there each lane reads the word itself (`lds_words`, `FirstLane::Lds`; GPU test
`a_mesh_program_passes_on_an_lds_word_lane_zero_read`). With both, the traced G-buffer (pixel
programs' `id` colours) shows the yard whole: a guard tower, a building, people at tables, no
streaks. Other mesh `v_readfirstlane_b32`s in divergent code (about 30 000 sites in 7 500 programs,
mostly reads of values every lane holds alike) still read the host wave's first lane.
