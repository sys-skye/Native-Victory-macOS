//! `ComputeBackend` on Metal: `MTLCommandQueue`s, `MTLSharedEvent` fences, DXIL through the Metal
//! Shader Converter, dispatch with a top-level argument buffer at slot 2 over zero-copy memory.

use crate::backend::{ComputeBackend, RootLayout, Submission};
use crate::msc::Msc;
use crate::resources::GraphicsBackend;
use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

type Dev = Retained<ProtocolObject<dyn MTLDevice>>;

pub struct Metal {
    pub(crate) dev: Dev,
    /// Opened at creation when installed; shader compilation reports the error if absent.
    pub(crate) msc: Result<std::sync::Arc<Msc>, String>,
    /// Heap residency by `MTLResidencySet` (macOS 15+); off = `useResource` per encoder.
    pub(crate) residency_sets: bool,
    /// The title's pool buffers in one set on every queue too (`VICTORY_METAL_RESIDENCY_SETS=1`);
    /// off = named by the encoders that use them.
    pub(crate) pool_sets: bool,
    /// The set of the title's pool buffers, made when the first one is.
    pool_set: std::cell::RefCell<Option<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
    /// What the encoders did to make buffers resident, for the tests and the frame line.
    pub(crate) counts: std::cell::Cell<ResidencyCounts>,
    /// Compiled pipelines kept across runs, opened at the first pipeline (when the cache folder is set).
    pub(crate) archive: std::cell::OnceCell<Option<std::sync::Arc<crate::archive::Archive>>>,
    /// Compute pipelines being made on threads of their own (`create_compute_pipeline_within`),
    /// by their DXIL and layout.
    compiling: Compiling<Result<ComputeState, String>>,
    /// As `compiling`, the render and mesh pipelines (`create_render_pipeline_within`).
    pub(crate) compiling_render: Compiling<Result<(crate::gfx::RenderState, Option<MTLSize>), String>>,
    /// `VICTORY_TRACE_GPU_FAILS`: label command buffers and encoders, report failures with them.
    pub(crate) trace_fails: bool,
    /// The most passes of a submission in one command buffer (`VICTORY_PASSES_PER_BUFFER`;
    /// default 0: no limit).
    pub(crate) passes_per_buffer: usize,
    /// `VICTORY_METAL_REQUEST_RESIDENCY=1`: each heap's set asks to stay resident between the
    /// command buffers that use it (`requestResidency` after each commit), instead of being made
    /// resident again for each one.
    pub(crate) request_residency: bool,
    /// Command buffers `submit_passes` committed.
    pub(crate) committed: std::cell::Cell<u64>,
    /// Pipeline state pointer to name, kept only when tracing.
    pub(crate) trace_enc: std::cell::RefCell<Vec<String>>,
    /// Tracing: the gate buffers of the indirect dispatches encoded since the last commit.
    pub(crate) trace_args: std::cell::RefCell<Vec<(String, usize, usize)>>,
    pub(crate) pipe_names: std::cell::RefCell<std::collections::HashMap<usize, String>>,
    /// The kernel that gates `ExecuteIndirect` records by a count (`gfx.rs`), made at first use.
    pub(crate) gate:
        std::cell::OnceCell<Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String>>,
    /// Every sampler made, by descriptor: a heap entry holds only the sampler's ID, which Metal
    /// hands to a new sampler once the old one is freed. Few distinct descriptors exist.
    #[allow(clippy::type_complexity)]
    pub(crate) samplers: std::cell::RefCell<
        Vec<(crate::resources::SamplerDesc, Retained<ProtocolObject<dyn MTLSamplerState>>)>,
    >,
}

/// The most passes of a submission in one command buffer unless `VICTORY_PASSES_PER_BUFFER` says
/// otherwise: no limit (0), the whole submission in one, until other Macs than the one it was
/// measured on (an M4 Pro) have run the split. 1 is the safest value there; see
/// docs/native-host.md.
const PASSES_PER_BUFFER: usize = 0;

type ComputeState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
/// Pipelines a `pipeline-compile` thread is making or has made, by what they are made of.
pub(crate) type Compiling<T> =
    std::cell::RefCell<std::collections::HashMap<[u8; 32], std::sync::mpsc::Receiver<crate::archive::Sendable<T>>>>;

/// Seconds since the first `Metal::new` (about process start), for the failure trace.
pub(crate) fn since_start() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

/// `VICTORY_TRACE_RESIDENCY=1`: times residency-set calls and command buffer commits. Each call over
/// 50 ms is logged at once; once a second a summary gives count, total and longest per kind, with
/// the bytes and allocations of the last set seen. Lines up the host's calls with wired memory (#79).
pub(crate) fn residency_timed<R>(
    what: &'static str,
    set: Option<&ProtocolObject<dyn MTLResidencySet>>,
    f: impl FnOnce() -> R,
) -> R {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    #[allow(clippy::type_complexity)]
    static SUM: Mutex<(f64, BTreeMap<&'static str, (u64, f64, f64)>, u64, usize)> = Mutex::new((0.0, BTreeMap::new(), 0, 0));
    if !*ON.get_or_init(|| std::env::var("VICTORY_TRACE_RESIDENCY").is_ok_and(|v| v.trim() == "1")) {
        return f();
    }
    let t = std::time::Instant::now();
    let r = f();
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let size = set.map(|s| (s.allocatedSize(), s.allocationCount()));
    if ms > 50.0 {
        tracing::warn!(what, ms, t = since_start(), set_mb = size.map(|s| s.0 >> 20), allocations = size.map(|s| s.1), "residency: slow call");
    }
    let mut g = SUM.lock().unwrap();
    let e = g.1.entry(what).or_default();
    e.0 += 1;
    e.1 += ms;
    e.2 = e.2.max(ms);
    if let Some((b, n)) = size {
        (g.2, g.3) = (b, n);
    }
    let now = since_start();
    if now - g.0 >= 1.0 {
        let kinds = g.1.iter().map(|(k, (n, tot, max))| format!("{k} {n}x {tot:.0}/{max:.0}ms")).collect::<Vec<_>>().join(", ");
        tracing::info!(t = now, set_mb = g.2 >> 20, allocations = g.3, "residency: last {:.1} s: {kinds}", now - g.0);
        g.0 = now;
        g.1.clear();
    }
    r
}

/// Counters of residency work.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct ResidencyCounts {
    /// `useResource` calls on title buffers (compute, render and blit encoders).
    pub use_resource: u64,
    /// Resources named to encoders for descriptor heaps without sets.
    pub heap_resources: u64,
    /// The most entries naming a resource, and the most distinct allocations they name, seen in
    /// one heap named to an encoder.
    pub heap_entries: u64,
    pub heap_allocations: u64,
    /// The most bytes one heap's set held when it was committed (the heap buffer and everything
    /// its entries name: resident while a command buffer that uses the heap runs).
    pub heap_set_bytes: u64,
    /// Buffers added to the pool set.
    pub added: u64,
    /// Queues the pool set was added to.
    pub queues: u64,
    /// `requestResidency` calls on heaps' sets (`VICTORY_METAL_REQUEST_RESIDENCY`).
    pub requests: u64,
}

impl Metal {
    /// Whether residency sets are in use (macOS 15 and later, unless switched off).
    pub fn residency_sets(&self) -> bool {
        self.residency_sets
    }

    pub fn residency_counts(&self) -> ResidencyCounts {
        self.counts.get()
    }

    pub(crate) fn count(&self, f: impl FnOnce(&mut ResidencyCounts)) {
        let mut c = self.counts.get();
        f(&mut c);
        self.counts.set(c);
    }
}

/// Library file names are unique per process, not per backend: Metal keeps libraries by URL.
static LIBRARIES: AtomicU64 = AtomicU64::new(0);

pub struct Queue(
    pub(crate) Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Whether the host orders its work: residency sets are in use (the heaps' own, and the
    /// pool's once it was added to the queue).
    pub(crate) std::cell::Cell<bool>,
    /// With sets, encoders no longer name what they reach through a heap or by address, and
    /// Metal's hazard tracking cannot see those accesses: command buffers wait for the one before
    /// through this event (and encoders within one through a fence), counted by `.3`.
    /// A shared event, so that a command buffer the GPU fails can still be paid on the CPU.
    pub(crate) Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub(crate) std::cell::Cell<u64>,
);
pub struct Fence(pub(crate) Retained<ProtocolObject<dyn MTLSharedEvent>>);
pub struct Pipeline(
    pub(crate) Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// The shader's static samplers, which its argument buffer points at.
    pub(crate) Option<crate::gfx::StaticSamplers>,
);

/// A wrapped range of title memory.
pub struct Buffer {
    pub(crate) buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) base: *const u8,
    /// The residency set it is in (so that dropping it leaves the set).
    pub(crate) resident: std::cell::RefCell<Option<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(set) = self.resident.get_mut().take() {
            set.removeAllocation(ProtocolObject::from_ref(&*self.buf));
            residency_timed("pool_commit", Some(&set), || set.commit());
        }
    }
}

impl Buffer {
    pub(crate) fn is_resident(&self) -> bool {
        self.resident.borrow().is_some()
    }
}

/// `MTLResidencySet` exists (macOS 15 and later).
fn sets_supported(dev: &Dev) -> bool {
    dev.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:))
}

/// The Mac's memory in bytes (`hw.memsize`), 0 if it cannot be read.
fn physical_memory() -> u64 {
    let (mut bytes, mut len) = (0u64, size_of::<u64>());
    // SAFETY: `hw.memsize` is a 64-bit value, written into `bytes`.
    let read = unsafe {
        libc::sysctlbyname(c"hw.memsize".as_ptr(), (&raw mut bytes).cast(), &mut len, std::ptr::null_mut(), 0)
    };
    if read == 0 { bytes } else { 0 }
}

/// `(the heaps' residency sets, the pool's set on the queues)` for `VICTORY_METAL_RESIDENCY_SETS`
/// (`asked`) on a Mac with `memory` bytes.
///
/// What a descriptor heap names is resident through the heap's own set, given to each command
/// buffer that uses the heap, so residency follows the frame. Naming it to every encoder with
/// `useResource` instead (`0`) cost the kernel about 155 ms per drawing render pass in GTA VI's
/// world, whose heap names 37,000 allocations. The title's pool buffers stay named by the
/// encoders that use them: in a set on the queue (`1`) all of title memory (about 6.7 GB) is
/// resident for the whole process, which on an 18 GB Mac preceded
/// `kIOGPUCommandBufferCallbackErrorTimeout` and `...OutOfMemory`.
///
/// Guess: the heaps' sets are safe from 32 GiB. The world's heap set holds 17 to 18 GB, more
/// than a 16 or 18 GB Mac has, and none has run it: until one has, a smaller Mac keeps
/// `useResource` unless `heap` asks for the sets.
fn residency_mode(asked: Option<&str>, supported: bool, memory: u64) -> (bool, bool) {
    match asked {
        Some("0") => (false, false),
        Some("heap") => (supported, false),
        Some(_) => (supported, supported),
        None => (supported && memory >= 32 << 30, false),
    }
}

impl Queue {
    /// The value the queue's order event has reached and the number of command buffers encoded
    /// with it: equal when every one of them has finished (0 and 0 when the host does not order).
    pub fn order_state(&self) -> (u64, u64) {
        (self.2.signaledValue(), self.3.get())
    }
}

impl Metal {
    /// The system GPU and the installed converter (`VICTORY_MSC_DYLIB` overrides its path).
    /// Succeeds even if the converter is not installed so non-shader Metal operations still work;
    /// shader pipeline creation returns the load error instead of panicking.
    pub fn new() -> Result<Metal, String> {
        let dev = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let asked = std::env::var("VICTORY_METAL_RESIDENCY_SETS").ok();
        let memory = physical_memory();
        let (residency_sets, pool_sets) = residency_mode(asked.as_deref(), sets_supported(&dev), memory);
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let mode = match (residency_sets, pool_sets) {
                (false, _) => "useResource per encoder",
                (true, false) => "a set per heap",
                (true, true) => "a set per heap and the pool's on the queues",
            };
            tracing::info!(
                mode,
                memory_gb = memory >> 30,
                switch = asked.as_deref().unwrap_or("unset"),
                request_residency = std::env::var("VICTORY_METAL_REQUEST_RESIDENCY").is_ok_and(|v| v.trim() == "1"),
                "metal: residency of what descriptor heaps name (VICTORY_METAL_RESIDENCY_SETS: 0, heap or 1)"
            );
        });
        // Pipelines put off past a frame's budget compile side by side (`within`); Metal's
        // default is two at a time. macOS 13.3.
        if dev.respondsToSelector(objc2::sel!(setShouldMaximizeConcurrentCompilation:)) {
            dev.setShouldMaximizeConcurrentCompilation(true);
        }
        Ok(Metal {
            dev,
            msc: Msc::try_open_default().map(std::sync::Arc::new),
            residency_sets,
            pool_sets,
            pool_set: Default::default(),
            counts: Default::default(),
            archive: Default::default(),
            compiling: Default::default(),
            compiling_render: Default::default(),
            trace_fails: {
                since_start();
                std::env::var_os("VICTORY_TRACE_GPU_FAILS").is_some()
            },
            passes_per_buffer: std::env::var("VICTORY_PASSES_PER_BUFFER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(PASSES_PER_BUFFER),
            request_residency: std::env::var("VICTORY_METAL_REQUEST_RESIDENCY").is_ok_and(|v| v.trim() == "1"),
            committed: Default::default(),
            pipe_names: Default::default(),
            trace_enc: Default::default(),
            trace_args: Default::default(),
            gate: Default::default(),
            samplers: Default::default(),
        })
    }

    pub(crate) fn msc(&self) -> Result<&Msc, String> {
        self.msc.as_deref().map_err(Clone::clone)
    }

    /// What makes compute pipeline states: all of it may go to another thread.
    pub(crate) fn compiler(&self) -> Result<Compiler, String> {
        Ok(Compiler { dev: self.dev.clone(), msc: self.msc.clone()?, archive: self.archive().cloned() })
    }

    /// The pipeline archive, if the cache is on.
    pub(crate) fn archive(&self) -> Option<&std::sync::Arc<crate::archive::Archive>> {
        self.archive.get_or_init(|| crate::archive::Archive::open(&self.dev)).as_ref()
    }

    /// At most `n` passes of a submission in one command buffer (0: no limit).
    pub fn with_passes_per_buffer(mut self, n: usize) -> Metal {
        self.passes_per_buffer = n;
        self
    }

    /// Command buffers committed for submissions so far.
    pub fn command_buffers(&self) -> u64 {
        self.committed.get()
    }

    /// All residency through `MTLResidencySet` (`true`: the heaps' sets and the pool's on the
    /// queues, where the system has sets) or `useResource` in every encoder (`false`).
    pub fn with_residency_sets(mut self, on: bool) -> Metal {
        self.residency_sets = on && sets_supported(&self.dev);
        self.pool_sets = self.residency_sets;
        self
    }

    /// Heaps' sets ask to stay resident (`VICTORY_METAL_REQUEST_RESIDENCY`).
    pub fn with_requested_residency(mut self, on: bool) -> Metal {
        self.request_residency = on;
        self
    }

    /// The heaps' sets alone, whatever the Mac's memory: what [`Metal::new`] chooses on a large one.
    pub fn with_heap_sets(mut self) -> Metal {
        self.residency_sets = sets_supported(&self.dev);
        self.pool_sets = false;
        self
    }

    /// Metal's own estimate of how much GPU memory this device can keep
    /// resident at once, in bytes (`recommendedMaxWorkingSetSize`).
    pub fn recommended_working_set(&self) -> u64 {
        self.dev.recommendedMaxWorkingSetSize()
    }
}

/// A metallib image to a library. shortcut: it goes through a file (no dispatch_data here); one
/// name per library, because Metal keeps libraries by URL. Use `newLibraryWithData` once dispatch2
/// is a dependency.
pub(crate) fn load_library(dev: &Dev, lib: &[u8]) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    {
        let n = LIBRARIES.fetch_add(1, Ordering::Relaxed);
        let file =
            std::env::temp_dir().join(format!("victory-metal-{}-{n}.metallib", std::process::id()));
        std::fs::write(&file, lib).map_err(|e| e.to_string())?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(file.to_str().unwrap()));
        let l = dev.newLibraryWithURL_error(&url);
        let _ = std::fs::remove_file(&file);
        l.map_err(|e| e.localizedDescription().to_string())
    }
}

type Job = Box<dyn FnOnce() + Send>;

/// Pipelines waiting for a `pipeline-compile` thread. The one asked for last goes first: it is
/// the one a frame may be waiting for, behind those earlier frames put off.
static JOBS: (std::sync::Mutex<Vec<Job>>, std::sync::Condvar) = (std::sync::Mutex::new(Vec::new()), std::sync::Condvar::new());

/// Runs `job` on one of the `pipeline-compile` threads (as many as cores, started at the first).
fn compile_later(job: Job) {
    static THREADS: std::sync::Once = std::sync::Once::new();
    THREADS.call_once(|| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            let _ = std::thread::Builder::new().name("pipeline-compile".into()).spawn(|| {
                loop {
                    let mut jobs = JOBS.0.lock().unwrap_or_else(|e| e.into_inner());
                    let job = loop {
                        match jobs.pop() {
                            Some(job) => break job,
                            None => jobs = JOBS.1.wait(jobs).unwrap_or_else(|e| e.into_inner()),
                        }
                    };
                    drop(jobs);
                    objc2::rc::autoreleasepool(|_| job());
                }
            });
        }
    });
    JOBS.0.lock().unwrap_or_else(|e| e.into_inner()).push(job);
    JOBS.1.notify_one();
}

/// What the job `start` gives makes, on a `pipeline-compile` thread, if it comes within `wait`.
/// `None` while it is still being made: the same `key` later picks it up, without waiting again
/// (its draws come many times a frame, and frames after the first would stall on it in turn).
pub(crate) fn within<T: 'static, J: FnOnce() -> T + Send + 'static>(
    compiling: &Compiling<T>,
    key: [u8; 32],
    wait: std::time::Duration,
    start: impl FnOnce() -> Result<J, String>,
) -> Result<Option<T>, String> {
    use std::sync::mpsc::RecvTimeoutError;
    let started = compiling.borrow_mut().remove(&key);
    let (rx, wait) = match started {
        Some(rx) => (rx, std::time::Duration::ZERO),
        None => {
            let job = start()?;
            let (tx, rx) = std::sync::mpsc::channel();
            compile_later(Box::new(move || {
                let _ = tx.send(crate::archive::Sendable(job()));
            }));
            (rx, wait)
        }
    };
    match rx.recv_timeout(wait) {
        Ok(made) => Ok(Some(made.0)),
        Err(RecvTimeoutError::Timeout) => {
            compiling.borrow_mut().insert(key, rx);
            Ok(None)
        }
        Err(RecvTimeoutError::Disconnected) => Err("the pipeline's compile thread ended without a result".into()),
    }
}

fn compile_key(dxil: &[u8], layout: Option<&RootLayout>) -> [u8; 32] {
    *blake3::Hasher::new().update(dxil).update(format!("{layout:?}").as_bytes()).finalize().as_bytes()
}

pub(crate) struct Compiler {
    pub(crate) dev: Dev,
    pub(crate) msc: std::sync::Arc<Msc>,
    pub(crate) archive: Option<std::sync::Arc<crate::archive::Archive>>,
}

// SAFETY: Metal's devices may be used from any thread; the converter and the archive are `Sync`.
unsafe impl Send for Compiler {}

impl Compiler {
    /// DXIL to a compute pipeline state: converted, loaded, then from the archive or compiled.
    fn compute(&self, dxil: &[u8], layout: Option<&RootLayout>) -> Result<ComputeState, String> {
        // A pipeline that crashed Metal's compiler on an earlier run would only crash it again.
        let crash_key = crate::disk::key(&[dxil]);
        if crate::disk::crashed(crash_key.as_bytes()) {
            return Err("Metal's compiler crashed on it on an earlier run".into());
        }
        let (_, lib) = self.msc.convert_with(dxil, None, layout, false).map_err(|e| e.to_string())?;
        let l = load_library(&self.dev, &lib)?;
        let name = l
            .functionNames()
            .iter()
            .next()
            .ok_or("converted library is empty")?;
        let f = l.newFunctionWithName(&name).ok_or("no function")?;
        let d = MTLComputePipelineDescriptor::new();
        d.setComputeFunction(Some(&f));
        match &self.archive {
            Some(a) => a.compute(&self.dev, &d, &crash_key),
            None => crate::archive::compile_compute(&self.dev, &d, &crash_key),
        }
    }
}

impl ComputeBackend for Metal {
    type Queue = Queue;
    type Fence = Fence;
    type Buffer = Buffer;
    type Pipeline = Pipeline;
    type Heap = crate::gfx::Heap;

    fn create_queue(&self) -> Result<Queue, String> {
        let q = self.dev.newCommandQueue().ok_or("newCommandQueue failed")?;
        let order = crate::gfx::track_event(self.dev.newSharedEvent().ok_or("newSharedEvent failed")?);
        // With the heaps' sets alone the host orders the queue's work from the start; with the
        // pool's set too, from when that set joins the queue (`make_resident`).
        let ordered = self.residency_sets && !self.pool_sets;
        Ok(Queue(q, std::cell::Cell::new(ordered), order, std::cell::Cell::new(0)))
    }

    fn create_fence(&self, initial: u64) -> Result<Fence, String> {
        let e = crate::gfx::track_event(self.dev.newSharedEvent().ok_or("newSharedEvent failed")?);
        e.setSignaledValue(initial);
        Ok(Fence(e))
    }

    fn allocated_bytes(&self) -> u64 {
        self.dev.currentAllocatedSize() as u64
    }

    fn wrap_title_memory(&self, ptr: *mut u8, len: usize) -> Result<Buffer, String> {
        let p = NonNull::new(ptr.cast()).ok_or("null title memory")?;
        // SAFETY: the caller keeps `ptr..ptr+len` mapped while the buffer and its work live.
        let buf = unsafe {
            self.dev
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    p,
                    len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or("newBufferWithBytesNoCopy refused the range (alignment or size)")?;
        Ok(Buffer {
            buf,
            base: ptr,
            resident: Default::default(),
        })
    }

    fn host_buffer(&self, len: usize) -> Result<(Buffer, *mut u8), String> {
        let buf = self
            .dev
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or("no host buffer")?;
        let base = buf.contents().as_ptr().cast::<u8>();
        let buffer = Buffer {
            buf,
            base,
            resident: Default::default(),
        };
        Ok((buffer, base))
    }

    fn make_resident(&self, queue: &Queue, buffers: &[&Buffer]) -> Result<(), String> {
        if !self.pool_sets {
            return Ok(());
        }
        let set = {
            let mut slot = self.pool_set.borrow_mut();
            match &*slot {
                Some(s) => s.clone(),
                None => {
                    let s = crate::gfx::track_residency_set(
                        self.dev
                            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
                            .map_err(|e| e.localizedDescription().to_string())?,
                    );
                    *slot = Some(s.clone());
                    s
                }
            }
        };
        let mut joined = false;
        for b in buffers.iter().filter(|b| !b.is_resident()) {
            set.addAllocation(ProtocolObject::from_ref(&*b.buf));
            *b.resident.borrow_mut() = Some(set.clone());
            self.count(|c| c.added += 1);
            joined = true;
        }
        if joined {
            set.commit();
        }
        if !queue.1.replace(true) {
            queue.0.addResidencySet(&set);
            self.count(|c| c.queues += 1);
        }
        Ok(())
    }

    fn gpu_address(&self, b: &Buffer, title: *const u8) -> u64 {
        b.buf.gpuAddress() + (title as usize - b.base as usize) as u64
    }

    fn name_pipeline(&self, p: &Pipeline, name: &str) {
        if self.trace_fails {
            let key = Retained::as_ptr(&p.0) as *const u8 as usize;
            self.pipe_names.borrow_mut().insert(key, name.to_string());
        }
    }

    fn create_compute_pipeline(&self, dxil: &[u8]) -> Result<Pipeline, String> {
        Ok(Pipeline(self.compiler()?.compute(dxil, None)?, self.static_samplers(None)?))
    }

    fn create_compute_pipeline_with(
        &self,
        dxil: &[u8],
        layout: &RootLayout,
    ) -> Result<Pipeline, String> {
        Ok(Pipeline(self.compiler()?.compute(dxil, Some(layout))?, self.static_samplers(Some(layout))?))
    }

    /// Converted and compiled on a `pipeline-compile` thread, so pipelines put off past `wait`
    /// are made side by side while the frame goes on.
    fn create_compute_pipeline_within(
        &self,
        dxil: &[u8],
        layout: Option<&RootLayout>,
        wait: std::time::Duration,
    ) -> Result<Option<Pipeline>, String> {
        let made = within(&self.compiling, compile_key(dxil, layout), wait, || {
            let (compiler, dxil, layout) = (self.compiler()?, dxil.to_vec(), layout.cloned());
            Ok(move || compiler.compute(&dxil, layout.as_ref()))
        })?;
        made.map(|state| Ok(Pipeline(state?, self.static_samplers(layout)?))).transpose()
    }

    fn submit(&self, queue: &Queue, work: &Submission<'_, Self>) -> Result<(), String> {
        use crate::resources::{Pass, PassSubmission};
        self.submit_passes(
            queue,
            &PassSubmission {
                waits: work.waits,
                passes: &[Pass::Compute(work.dispatches)],
                signals: work.signals,
            },
        )
        .map_err(|e| e.to_string())
    }

    fn signal(&self, fence: &Fence, value: u64) {
        fence.0.setSignaledValue(value);
    }

    fn completed(&self, fence: &Fence) -> u64 {
        fence.0.signaledValue()
    }

    fn wait(&self, fence: &Fence, value: u64, timeout_ms: u64) -> bool {
        fence.0.waitUntilSignaledValue_timeoutMS(value, timeout_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Dispatch, RootArgs};
    use crate::mem::{Region, page_size};

    /// `dst[i] = src[i] * k.x + k.y` for `numthreads(64)`, root signature
    /// `RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)` (the `--demo` shader, compiled by dxc;
    /// our own HLSL, not a title shader).
    const AFFINE: &[u8] = include_bytes!("../tests/affine.dxil");
    const N: usize = 4096;

    fn backend() -> Option<Metal> {
        if !std::path::Path::new(crate::msc::DEFAULT_DYLIB).exists()
            && std::env::var_os("VICTORY_MSC_DYLIB").is_none()
        {
            eprintln!("skipped: Metal Shader Converter not installed");
            return None;
        }
        Some(Metal::new().unwrap())
    }

    /// Three arrays in one zero-copy buffer: a, b, c.
    struct Mem {
        region: Region,
        buf: Buffer,
        part: usize,
    }

    impl Mem {
        fn new(m: &Metal) -> Mem {
            let part = (N * 4).next_multiple_of(page_size());
            let region = Region::new(3 * part, None).unwrap();
            let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
            Mem { region, buf, part }
        }
        fn at(&self, i: usize) -> *mut u32 {
            // SAFETY: i < 3, inside the region.
            unsafe { self.region.ptr.add(i * self.part).cast() }
        }
        fn gpu(&self, m: &Metal, i: usize) -> u64 {
            m.gpu_address(&self.buf, self.at(i).cast())
        }
        fn set(&self, i: usize, f: impl Fn(usize) -> u32) {
            // SAFETY: the GPU is idle or not yet submitted when tests call this.
            (0..N).for_each(|j| unsafe { self.at(i).add(j).write_volatile(f(j)) });
        }
        fn get(&self, i: usize, j: usize) -> u32 {
            // SAFETY: tests read after the fence wait.
            unsafe { self.at(i).add(j).read_volatile() }
        }
    }

    /// `out = in * mul + add` over a root signature of constants, SRV, UAV.
    fn step<'a>(
        m: &Metal,
        p: &'a Pipeline,
        mem: &'a Mem,
        (src, dst): (usize, usize),
        (mul, add): (u32, u32),
    ) -> Dispatch<'a, Metal> {
        Dispatch {
            pipeline: p,
            root: RootArgs::default()
                .constants(&[mul, add])
                .address(mem.gpu(m, src))
                .address(mem.gpu(m, dst)),
            resources: vec![&mem.buf],
            heaps: Default::default(),
            groups: [N / 64, 1, 1],
            threads_per_group: [64, 1, 1],
            indirect: None,
            count: None,
        }
    }

    /// The D3D12 pattern of the title: queue A dispatches and signals fence N; queue B waits for N
    /// and dispatches over the same memory; the CPU waits. B is submitted first, so it is the fence
    /// and not the submission order that makes the result right.
    #[test]
    fn queue_b_waits_for_queue_a_over_the_same_buffer() {
        let Some(m) = backend() else { return };
        let _flight = crate::gfx::FLIGHTS_IN_TESTS.read().unwrap_or_else(|e| e.into_inner());
        let (qa, qb) = (m.create_queue().unwrap(), m.create_queue().unwrap());
        let fence = m.create_fence(0).unwrap();
        let done = m.create_fence(0).unwrap();
        let p = m.create_compute_pipeline(AFFINE).unwrap();
        let mem = Mem::new(&m);
        mem.set(0, |i| i as u32);

        // B: c = b * 3, after fence reaches 7; signals `done` 1.
        let b_work = [step(&m, &p, &mem, (1, 2), (3, 0))];
        m.submit(
            &qb,
            &Submission {
                waits: &[(&fence, 7)],
                dispatches: &b_work,
                signals: &[(&done, 1)],
            },
        )
        .unwrap();
        assert!(!m.wait(&done, 1, 50), "B ran before A signalled");
        assert_eq!(m.completed(&fence), 0);

        // A: b = a * 2 + 1; signals fence 7.
        let a_work = [step(&m, &p, &mem, (0, 1), (2, 1))];
        m.submit(
            &qa,
            &Submission {
                waits: &[],
                dispatches: &a_work,
                signals: &[(&fence, 7)],
            },
        )
        .unwrap();
        assert!(m.wait(&done, 1, 5000), "timed out");
        assert!(m.completed(&fence) >= 7);
        let ok = (0..N).all(|i| mem.get(2, i) == (i as u32 * 2 + 1) * 3);
        assert!(ok, "c[i] must be (a[i]*2+1)*3");
    }

    /// `ID3D12Fence::Signal` from the CPU releases a queue that waits (a title Signal / `hold.rs`).
    #[test]
    fn cpu_signal_releases_a_waiting_queue() {
        let Some(m) = backend() else { return };
        let _flight = crate::gfx::FLIGHTS_IN_TESTS.read().unwrap_or_else(|e| e.into_inner());
        let q = m.create_queue().unwrap();
        let (gate, done) = (m.create_fence(0).unwrap(), m.create_fence(0).unwrap());
        let p = m.create_compute_pipeline(AFFINE).unwrap();
        let mem = Mem::new(&m);
        mem.set(0, |i| i as u32);
        let w = [step(&m, &p, &mem, (0, 1), (5, 7))];
        m.submit(
            &q,
            &Submission {
                waits: &[(&gate, 1)],
                dispatches: &w,
                signals: &[(&done, 1)],
            },
        )
        .unwrap();
        assert!(!m.wait(&done, 1, 50));
        m.signal(&gate, 1);
        assert!(m.wait(&done, 1, 5000));
        assert!((0..N).all(|i| mem.get(1, i) == i as u32 * 5 + 7));
    }

    /// Many submissions in flight over several queues, chained by fence values, one CPU wait: a
    /// frame's worth of the pattern above (35 submissions, alternating queues, `x = x * 1 + 1`).
    #[test]
    fn many_submissions_in_flight_one_wait() {
        let Some(m) = backend() else { return };
        let _flight = crate::gfx::FLIGHTS_IN_TESTS.read().unwrap_or_else(|e| e.into_inner());
        const SUBS: u64 = 64;
        let queues: Vec<_> = (0..4).map(|_| m.create_queue().unwrap()).collect();
        let chain = m.create_fence(0).unwrap();
        let p = m.create_compute_pipeline(AFFINE).unwrap();
        let mem = Mem::new(&m);
        mem.set(0, |_| 0);
        // Ping-pong between arrays 0 and 1: each step adds 1 to what the previous step wrote.
        let works: Vec<_> = (0..SUBS)
            .map(|s| {
                let (src, dst) = if s % 2 == 0 { (0, 1) } else { (1, 0) };
                [step(&m, &p, &mem, (src, dst), (1, 1))]
            })
            .collect();
        let t = std::time::Instant::now();
        // In order (a queue runs its command buffers in order, so reversing would deadlock): only the
        // fence makes queue k+1 wait for queue k.
        for s in 0..SUBS {
            let waits = [(&chain, s)];
            let signals = [(&chain, s + 1)];
            m.submit(
                &queues[s as usize % queues.len()],
                &Submission {
                    waits: &waits,
                    dispatches: &works[s as usize],
                    signals: &signals,
                },
            )
            .unwrap();
        }
        assert!(
            m.wait(&chain, SUBS, 10_000),
            "timed out at {}",
            m.completed(&chain)
        );
        eprintln!(
            "{SUBS} dependent submissions over 4 queues: {:?}",
            t.elapsed()
        );
        // SUBS is even: the last write is array 0.
        assert!((0..N).all(|i| mem.get(0, i) == SUBS as u32));
    }

    /// The heaps' sets are the default only from 32 GiB (the world's set holds 17 to 18 GB), `heap`
    /// forces them on any Mac, `0` names everything per encoder and `1` adds the pool's queue set.
    #[test]
    fn the_default_residency_mode_follows_the_macs_memory() {
        const GIB: u64 = 1 << 30;
        assert_eq!(residency_mode(None, true, 18 * GIB), (false, false));
        assert_eq!(residency_mode(None, true, 32 * GIB), (true, false));
        assert_eq!(residency_mode(None, false, 64 * GIB), (false, false));
        assert_eq!(residency_mode(Some("heap"), true, 16 * GIB), (true, false));
        assert_eq!(residency_mode(Some("heap"), false, 16 * GIB), (false, false));
        assert_eq!(residency_mode(Some("0"), true, 64 * GIB), (false, false));
        assert_eq!(residency_mode(Some("1"), true, 16 * GIB), (true, true));
        assert!(physical_memory() >= GIB);
    }

    #[test]
    fn missing_msc_returns_error_instead_of_panicking() {
        let mut m = Metal::new().expect("Metal device available on macOS");
        m.msc = crate::msc::Msc::open("/nonexistent/libmetalirconverter.dylib").map(std::sync::Arc::new);
        // Non-shader operations still succeed without MSC.
        let q = m.create_queue().unwrap();
        let f = m.create_fence(0).unwrap();
        m.signal(&f, 1);
        assert_eq!(m.completed(&f), 1);
        drop(q);
        // Shader conversion returns an Err instead of panicking.
        assert!(m.create_compute_pipeline(AFFINE).is_err());
    }

    #[test]
    fn recommended_working_set_is_nonzero_on_real_hardware() {
        let m = Metal::new().expect("Metal device available on macOS");
        assert!(m.recommended_working_set() > 0);
    }

    /// A pipeline still compiling when the wait ends is not an error: the same call later picks
    /// it up, and nothing is compiled twice.
    #[test]
    fn a_pipeline_still_compiling_is_picked_up_later() {
        let Some(m) = backend() else { return };
        let none = std::time::Duration::ZERO;
        let (tx, rx) = std::sync::mpsc::channel();
        m.compiling.borrow_mut().insert(compile_key(AFFINE, None), rx);
        assert!(m.create_compute_pipeline_within(AFFINE, None, none).unwrap().is_none());
        // Not waited for again, whatever the frame has left.
        let asked = std::time::Instant::now();
        assert!(m.create_compute_pipeline_within(AFFINE, None, std::time::Duration::from_secs(30)).unwrap().is_none());
        assert!(asked.elapsed() < std::time::Duration::from_secs(10));
        tx.send(crate::archive::Sendable(m.compiler().unwrap().compute(AFFINE, None))).unwrap();
        assert!(m.create_compute_pipeline_within(AFFINE, None, none).unwrap().is_some());
        assert!(m.compiling.borrow().is_empty());
        // One nothing has started is compiled on its own thread and waited for.
        let other = include_bytes!("../tests/fixtures/heap_rw.dxil");
        assert!(m.create_compute_pipeline_within(other, None, std::time::Duration::MAX).unwrap().is_some());
        assert!(m.compiling.borrow().is_empty());
    }
}
