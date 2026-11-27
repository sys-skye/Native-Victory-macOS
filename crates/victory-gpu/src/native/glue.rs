//! The one native host the title-facing objects call on macOS: `queue.rs`, `fence.rs`, `list.rs`,
//! `descriptor.rs` and `resource.rs` forward to the functions here (the same call sites that go to
//! `crate::host` on Windows).
//!
//! Threads. The title calls from many threads; the backend's objects and the `Rc`s inside
//! [`Title`] are not thread safe, so everything that touches them runs under one lock (`TITLE`),
//! always taken before a list's own lock, and objects holding such `Rc`s (lists) are dropped under
//! it. The GPU never waits on the lock: submissions are asynchronous, and a timer thread moves held
//! work and fence values along.

use super::memory::{PoolObserver, PoolTable};
use super::state::Graphics;
use super::title::{Guessed, Pipe, TargetFormats, Title, TitleList, Wait};
use crate::object::This;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use victory_backend::resources::HeapKind;
use victory_metal::metal::Metal;

pub type MetalTitle = Title<Metal>;
pub type MetalList = TitleList<Metal>;

struct Shared(MetalTitle);
// SAFETY: every access to the title's `Rc`s and Objective-C objects is under `TITLE`.
unsafe impl Send for Shared {}

static TITLE: Mutex<Option<Shared>> = Mutex::new(None);
static FAILED: AtomicBool = AtomicBool::new(false);
static POOLS: OnceLock<Arc<PoolTable>> = OnceLock::new();

/// The table of the title's memory pools. Feed it through [`pool_observer`].
pub fn pools() -> &'static Arc<PoolTable> {
    POOLS.get_or_init(Default::default)
}

/// What `victory_hle`'s physical-memory observer forwards its four calls to
/// (`pool_created`, `pool_released`, `mapped`, `unmapped`): a plain handle on the pool table the
/// native host reads, safe to call from any title thread, before or after the GPU exists.
pub fn pool_observer() -> Arc<dyn PoolObserver> {
    pools().clone()
}

/// Called with the event handle (as the title gave it) when a fence a title waits on with
/// `SetEventOnCompletion` is reached. The runtime sets it; without one, events are not signalled.
static EVENT_HOOK: OnceLock<fn(usize)> = OnceLock::new();

pub fn set_event_hook(hook: fn(usize)) {
    let _ = EVENT_HOOK.set(hook);
}

pub(crate) fn signal_event(event: usize) {
    if let Some(hook) = EVENT_HOOK.get() {
        hook(event);
    }
}

fn lock() -> std::sync::MutexGuard<'static, Option<Shared>> {
    TITLE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs `f` on the native host, made on first use; `None` if there is no Metal device (the call
/// is then ignored, as when there is no host GPU on Windows).
pub(crate) fn with_title<R>(f: impl FnOnce(&mut MetalTitle) -> R) -> Option<R> {
    let mut guard = lock();
    if guard.is_none() {
        if FAILED.load(Ordering::Relaxed) {
            return None;
        }
        match Metal::new() {
            Ok(metal) => {
                let console = crate::console::current();
                let (gpu_optimal, standard) = console.title_memory();
                tracing::info!(
                    console = console.name(),
                    title_memory_mib = (gpu_optimal + standard) / (1 << 20),
                    recommended_working_set_mib = metal.recommended_working_set() / (1 << 20),
                    "native host: Metal"
                );
                *guard = Some(Shared(Title::new(metal, pools().clone())));
                start_timer();
            }
            Err(e) => {
                FAILED.store(true, Ordering::Relaxed);
                tracing::warn!(error = %e, "no native host GPU");
                return None;
            }
        }
    }
    guard.as_mut().map(|t| f(&mut t.0))
}

/// Moves held work along and raises the title's copies of fence values as the GPU signals them
/// (titles read a fence's value from memory), without a call of the title's to drive it.
fn start_timer() {
    let _ = std::thread::Builder::new()
        .name("victory native host".into())
        .spawn(|| {
            loop {
                std::thread::sleep(std::time::Duration::from_millis(1));
                if let Some(t) = lock().as_mut() {
                    refresh(&mut t.0);
                }
            }
        });
}

fn refresh(t: &mut MetalTitle) {
    t.pump();
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_none_or(|l| l.elapsed().as_secs() >= 5) {
            let stuck = t.stuck(std::time::Duration::from_secs(2));
            if !stuck.is_empty() {
                *last = Some(std::time::Instant::now());
                tracing::warn!("work held for a wait that does not hold: {stuck:?}");
                // What a held queue or a fence waits for may be owed by a command buffer the GPU
                // lost (out of memory): those are paid from the CPU so the title goes on.
                let lost = victory_metal::gfx::recover_lost();
                if lost > 0 {
                    tracing::warn!("recovered {lost} command buffers the GPU never reported on");
                }
                // Nothing is submitted while every queue is held, so the per-submission report
                // of command buffers the GPU has not finished would stay silent.
                if env_set!("VICTORY_TRACE_GPU_FAILS") {
                    let flying = victory_metal::gfx::report_in_flight();
                    let order: Vec<String> = t
                        .backend_queues()
                        .map(|(k, q)| (k, q.order_state()))
                        .filter(|(_, (done, made))| done != made)
                        .map(|(k, (done, made))| format!("queue {k:#x} at {done} of {made}"))
                        .collect();
                    tracing::warn!(
                        "command buffers in flight while work is held: {flying}; order events behind: {order:?}; commands dropped at unmapped memory so far: {}",
                        t.host.unmapped()
                    );
                }
            }
        }
    }
    for (fence, value) in t.completed() {
        crate::fence::raise(fence as This, value);
    }
}

/// Runs `f` on the native host and a title list's recording.
pub(crate) fn with_list<R>(
    this: This,
    f: impl FnOnce(&mut MetalTitle, &mut MetalList) -> R,
) -> Option<R> {
    if this.is_null() {
        return None;
    }
    with_title(|t| {
        let list = crate::list::native(this);
        let mut list = list.lock().unwrap_or_else(|e| e.into_inner());
        f(t, &mut list)
    })
}

// --- queues and fences ---

/// `ExecuteCommandLists`.
pub(crate) fn submit(queue: This, lists: &[This]) {
    if env_set!("VICTORY_TRACE_WRITES") {
        tracing::warn!("submit to queue {queue:p}: lists {lists:?}");
    }
    // One submission at a time, in the order they came: the lock below is let go while the
    // textures are read, and the next submission must not overtake this one meanwhile.
    static ORDER: Mutex<()> = Mutex::new(());
    let _order = ORDER.lock().unwrap_or_else(|e| e.into_inner());
    // Tests share one title and count the uploads of their own `sync_dirty` calls, which another
    // test's submission with textures read but not yet uploaded would upset: there the lock is
    // kept from the probes to the uploads (`Title::submit`).
    let unlocked = !cfg!(test);
    if unlocked {
        let jobs = with_title(|t| t.sync_jobs()).unwrap_or_default();
        for batch in super::textures::batches(jobs) {
            // Without the lock: the title's other threads go on recording.
            let prepared = super::textures::prepare(&batch);
            with_title(|t| t.sync_apply(prepared));
        }
    }
    with_title(|t| {
        if t.ensure_queue(queue as usize).is_err() {
            return;
        }
        let mut segments = Vec::new();
        for &l in lists.iter().filter(|l| !l.is_null()) {
            let list = crate::list::native(l);
            segments.extend(list.lock().unwrap_or_else(|e| e.into_inner()).segments());
        }
        if unlocked {
            t.submit_synced(queue as usize, segments);
        } else {
            t.submit(queue as usize, segments);
        }
        // `VICTORY_TRACE_DESCRIPTORS=sites`: the heaps' pages are closed again for the next
        // interval, so the title's first store into each is seen.
        if super::descriptor_trace::sites() {
            victory_platform::watch::watch(&t.host.heap_ranges());
        }
    });
}

pub(crate) fn queue_signal(queue: This, fence: This, value: u64) {
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        t.signal_queue(queue as usize, fence as usize, initial, value);
    });
}

pub(crate) fn queue_wait(queue: This, fence: This, value: u64) {
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        t.wait_queue(queue as usize, fence as usize, initial, value);
    });
}

pub(crate) fn queue_released(queue: This) {
    with_title(|t| t.forget_queue(queue as usize));
}

/// `ID3D12Fence::Signal`.
pub(crate) fn fence_signal(fence: This, value: u64) {
    let initial = crate::fence::initial(fence);
    with_title(|t| t.signal_cpu(fence as usize, initial, value));
}

/// The value fence `fence` has completed, queues pumped first.
pub(crate) fn fence_poll(fence: This) -> u64 {
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        t.pump();
        t.fence_value(fence as usize, initial)
    })
    .unwrap_or(initial)
}

/// `GetCompletedValue` of a fence that has no GPU behind it yet: nothing to ask.
pub(crate) fn fence_released(fence: This) {
    with_title(|t| t.forget_fence(fence as usize));
}

/// Blocks until fence `fence` reaches `value` (the title's `SetEventOnCompletion` with no event):
/// held work keeps moving while it waits. False after `timeout`.
pub(crate) fn fence_wait(fence: This, value: u64, timeout: std::time::Duration) -> bool {
    let initial = crate::fence::initial(fence);
    let start = std::time::Instant::now();
    loop {
        match with_title(|t| t.try_wait(fence as usize, initial, value)) {
            Some(true) => return true,
            None => return false,
            Some(false) => {}
        }
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
}

// --- present ---

/// Gives the host the app's drawable layer (a `CAMetalLayer`, by address; it must outlive the
/// host): frames the title presents are drawn into it from then on. Without one, `PresentX` shows
/// nothing.
pub fn set_layer(layer: *mut std::ffi::c_void) -> Result<(), String> {
    with_title(|t| t.set_surface(layer)).unwrap_or_else(|| Err("no native host GPU".into()))
}

/// A layer drawing to the host's device, `width` by `height`, for an app that has none (or a test):
/// hand [`victory_metal::present::Layer::as_ptr`] to [`set_layer`] and keep the layer alive.
pub fn create_layer(width: usize, height: usize) -> Result<victory_metal::present::Layer, String> {
    with_title(|t| t.host.backend.create_layer(width, height)).ok_or_else(|| "no native host GPU".into())
}

/// `VICTORY_TRACE_TARGETS=1`: per frame, what drew into or cleared which title texture, and what
/// was presented (logged at each present).
pub(crate) fn tracing_targets() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_set!("VICTORY_TRACE_TARGETS"))
}

/// Draws and clears by title texture since the last present, and dropped texture operations.
static TRACE: Mutex<Option<std::collections::HashMap<usize, [u32; 3]>>> = Mutex::new(None);
/// Draws dropped since the last present, by reason (with `VICTORY_TRACE_TARGETS`).
static DROP_REASONS: Mutex<Vec<(&'static str, u32)>> = Mutex::new(Vec::new());
/// Draws dropped since the last present for want of a host pipeline, by shaders and targets.
static DROP_PIPES: Mutex<Vec<(String, u32)>> = Mutex::new(Vec::new());
static DROPPED_OPS: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn trace_bump(key: usize, which: usize) {
    if tracing_targets() {
        TRACE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().entry(key).or_default()[which] += 1;
    }
}

/// `VICTORY_DUMP_TITLE=<dir>,<seconds>,<seconds>...`: at the first present after each time, the
/// title memory behind every 8-bit colour target drawn to since the last present is read whole,
/// the fraction of non-zero bytes logged, and the memory detiled into a scratch texture and
/// written as `<dir>/m<time>_<n>.ppm`: what the title (CPU, DMA) put there, whatever the GPU did.
#[inline(never)]
fn dump_title_memory(stats: &std::collections::HashMap<usize, [u32; 3]>) {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    static DONE: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    let start = START.get_or_init(std::time::Instant::now);
    let Ok(v) = std::env::var("VICTORY_DUMP_TITLE") else { return };
    let mut parts = v.split(',');
    let dir = parts.next().unwrap_or(".").to_owned();
    let now = start.elapsed().as_secs();
    let Some(time) = parts
        .filter_map(|p| p.parse::<u64>().ok())
        .find(|&t| now >= t && !DONE.lock().unwrap_or_else(|e| e.into_inner()).contains(&t))
    else {
        return;
    };
    DONE.lock().unwrap_or_else(|e| e.into_inner()).push(time);
    with_title(|t| {
        for (n, key) in stats.keys().enumerate() {
            let Some(info) = crate::resource::texture_info(*key as This) else { continue };
            if !matches!(info.format, 28 | 87 | 91 | 29) {
                continue;
            }
            let tiled = super::textures::read_memory(info.memory, info.size);
            let nz = tiled.iter().filter(|&&b| b != 0).count();
            tracing::info!(time, n, key = format!("{key:#x}"), at = format!("{:#x}", info.memory as usize), size = info.size, nonzero = nz, fraction = nz as f64 / tiled.len().max(1) as f64, "title memory of a colour target");
            let Some(desc) = info.desc() else { continue };
            let Ok(tex) = t.host.texture(&desc) else { continue };
            if super::textures::upload(&t.host.backend, &tex, &info, &tiled).is_err() {
                continue;
            }
            let (w, h) = (info.width as usize, info.height as usize);
            let mut px = vec![0u8; w * h * 4];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut px, w * 4, w * h * 4);
            let bgra = matches!(info.format, 87 | 91);
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("m{time}_{n}.ppm")), out);
        }
    });
}

/// `VICTORY_DUMP_TARGETS=<dir>,<seconds>`: once, at the first present after that many seconds,
/// every 8-bit colour target drawn to since the last present is written as `<dir>/t<n>.ppm`; HDR
/// and 10-bit targets (DXGI 2, 10, 24, 26) are tone-mapped into `..._f<format>.ppm`, their range logged.
fn dump_targets(stats: &std::collections::HashMap<usize, [u32; 3]>) {
    use std::sync::atomic::AtomicBool;
    static DONE: AtomicBool = AtomicBool::new(false);
    static LAST: Mutex<Option<u64>> = Mutex::new(None);
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    let Ok(v) = std::env::var("VICTORY_DUMP_TARGETS") else { return };
    // `<dir>,<seconds>[,<every>]`: from that second on, once or every that many seconds.
    let mut parts = v.split(',');
    let dir = parts.next().unwrap_or(".").to_owned();
    let secs = parts.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or(25);
    let every = parts.next().and_then(|s| s.parse::<u64>().ok());
    let now = start.elapsed().as_secs();
    let due = match (every, *LAST.lock().unwrap_or_else(|e| e.into_inner())) {
        (Some(e), Some(last)) => now >= last + e,
        _ => true,
    };
    if now < secs || !due || (every.is_none() && DONE.swap(true, Ordering::Relaxed)) {
        return;
    }
    *LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some(now);
    with_title(|t| {
        for (n, key) in stats.keys().enumerate() {
            let Some(info) = crate::resource::texture_info(*key as This) else { continue };
            let (w, h) = (info.width as usize, info.height as usize);
            if matches!(info.format, 2 | 10 | 24 | 26) {
                // HDR and 10-bit targets: tone-mapped to 8 bits, with their range logged.
                let Some(tex) = t.texture_of(*key) else { continue };
                let bpp = match info.format { 2 => 16, 10 => 8, _ => 4 };
                let mut raw = vec![0u8; w * h * bpp];
                victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * bpp, w * h * bpp);
                let rgb: Vec<[f32; 3]> = raw.chunks_exact(bpp).map(|p| decode_hdr(info.format, p)).collect();
                let finite = rgb.iter().flatten().filter(|v| v.is_finite());
                let (mut max, mut sum, mut count, mut lit) = (0f32, 0f64, 0usize, 0usize);
                for v in finite {
                    max = max.max(*v);
                    sum += f64::from(*v);
                    count += 1;
                    lit += usize::from(*v > 1e-4);
                }
                let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                out.extend(rgb.iter().flat_map(|c| c.map(|v| if v.is_finite() && v > 0.0 { ((v / (1.0 + v)).sqrt() * 255.0) as u8 } else { 0 })));
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}_f{}.ppm", info.memory as usize, info.format)), out);
                tracing::info!(n, key = format!("{key:#x}"), w, h, format = info.format, max, mean = sum / count.max(1) as f64, lit = lit as f64 / (rgb.len() * 3).max(1) as f64, "target dumped");
                continue;
            }
            // Depth and one-channel targets (local diagnostic): grey, scaled to their finite range.
            {
                use victory_backend::resources::Format as F;
                if let Some(fmt) = t.format_of(*key).filter(|f| matches!(f, F::D32Float | F::D32FloatS8 | F::R32Float | F::D16Unorm | F::R16Unorm | F::R16Float | F::R8Unorm)) {
                    let Some(tex) = t.texture_of(*key) else { continue };
                    let (vals, w) = if matches!(fmt, F::D32Float | F::D32FloatS8) {
                        let Some(v) = t.host.backend.read_depth(&tex) else { continue };
                        let tw = v.len() / h.max(1);
                        (v, tw)
                    } else {
                        let bpp = match fmt { F::R32Float => 4, F::R8Unorm => 1, _ => 2 };
                        let mut raw = vec![0u8; w * h * bpp];
                        victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * bpp, w * h * bpp);
                        let v: Vec<f32> = match fmt {
                            F::R32Float => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                            F::R8Unorm => raw.iter().map(|&b| f32::from(b) / 255.0).collect(),
                            F::R16Float => raw.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                            _ => raw.chunks_exact(2).map(|c| f32::from(u16::from_le_bytes([c[0], c[1]])) / 65535.0).collect(),
                        };
                        (v, w)
                    };
                    let finite: Vec<f32> = vals.iter().copied().filter(|v| v.is_finite()).collect();
                    let (lo, hi) = finite.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                    let span = (hi - lo).max(f32::MIN_POSITIVE);
                    let distinct = finite.iter().map(|v| v.to_bits()).collect::<std::collections::HashSet<_>>().len();
                    let hh = vals.len() / w.max(1);
                    let mut out = format!("P6\n{w} {hh}\n255\n").into_bytes();
                    out.extend(vals.iter().flat_map(|&v| {
                        let g = if v.is_finite() { ((v - lo) / span * 255.0) as u8 } else { 0 };
                        [g, g, g]
                    }));
                    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}_g.ppm", info.memory as usize)), out);
                    tracing::info!(n, key = format!("{key:#x}"), w, h = hh, ?fmt, lo, hi, distinct, "target dumped");
                    continue;
                }
            }
            if !matches!(info.format, 28 | 87 | 91 | 29) {
                continue;
            }
            let Some(tex) = t.texture_of(*key) else { continue };
            let mut px = vec![0u8; w * h * 4];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut px, w * 4, w * h * 4);
            let bgra = matches!(info.format, 87 | 91);
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}.ppm", info.memory as usize)), out);
            // The alpha channel as grey.
            let mut alpha = format!("P6\n{w} {h}\n255\n").into_bytes();
            alpha.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[3], p[3], p[3]]));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}_alpha.ppm", info.memory as usize)), alpha);
            tracing::info!(n, key = format!("{key:#x}"), w, h, "target dumped");
        }
    });
}

/// One texel of an HDR or 10-bit colour target (DXGI 2, 10, 24, 26) as linear RGB.
fn decode_hdr(format: u32, p: &[u8]) -> [f32; 3] {
    let word = |i: usize| u32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]);
    // An unsigned small float: `mbits` mantissa bits over a 5-bit exponent.
    let small = |v: u32, mbits: u32| {
        let (e, m) = ((v >> mbits) & 0x1f, v & ((1 << mbits) - 1));
        if e == 0 { m as f32 * 2f32.powi(-14 - mbits as i32) } else { (1.0 + m as f32 / (1 << mbits) as f32) * 2f32.powi(e as i32 - 15) }
    };
    match format {
        2 => [0, 4, 8].map(|i| f32::from_bits(word(i))),
        10 => [0, 2, 4].map(|i| {
            let h = u32::from(u16::from_le_bytes([p[i], p[i + 1]]));
            let v = small(h & 0x7fff, 10);
            if h >> 15 == 1 { -v } else if (h >> 10) & 0x1f == 0x1f { f32::NAN } else { v }
        }),
        24 => [0, 10, 20].map(|s| ((word(0) >> s) & 0x3ff) as f32 / 1023.0),
        _ => {
            let v = word(0);
            [small(v & 0x7ff, 6), small((v >> 11) & 0x7ff, 6), small(v >> 22, 5)]
        }
    }
}

/// A texture operation the native host does not do (copies, resolves), counted by the trace.
pub(crate) fn dropped_op(what: &str, dst: This, src: This) {
    crate::list::DROPPED_KINDS[3].fetch_add(1, Ordering::Relaxed);
    warn_once("a texture copy or resolve");
    // Once per operation and title formats (0: a buffer or not a texture), so a refused kind of
    // copy shows what it was.
    static SEEN: Mutex<Vec<(String, u32, u32)>> = Mutex::new(Vec::new());
    let format = |r: This| crate::resource::texture_info(r).map_or(0, |i| i.format);
    let key = (what.to_owned(), format(src), format(dst));
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&key) {
        tracing::warn!(what, src_format = key.1, dst_format = key.2, "a texture copy refused (first of this kind)");
        seen.push(key);
    }
    drop(seen);
    if tracing_targets() {
        DROPPED_OPS.lock().unwrap_or_else(|e| e.into_inner()).push(format!("{what} {src:?}->{dst:?}"));
    }
}

/// `VICTORY_TRACE_RECORD_WRITERS`: one compute program's dispatches since the last present, run
/// or dropped: the buffers its tables hold for unordered access (title memory start and size),
/// and the textures they hold as shader resources and for unordered access.
#[derive(Default)]
struct RecordDispatches {
    buffers: std::collections::BTreeSet<(u64, u64)>,
    reads: std::collections::BTreeSet<usize>,
    writes: std::collections::BTreeSet<usize>,
}

/// `VICTORY_TRACE_RECORD_WRITERS`: the compute dispatches since the last present by program and
/// outcome, and each indirect mesh execution's first record address with its records and how
/// many had a group count of 0 in title memory as the execution was recorded.
#[derive(Default)]
struct RecordTrace {
    dispatches: std::collections::BTreeMap<String, RecordDispatches>,
    executions: std::collections::BTreeMap<u64, [u64; 2]>,
}

static RECORD_TRACE: Mutex<Option<RecordTrace>> = Mutex::new(None);

fn tracing_record_writers() -> bool {
    env_set!("VICTORY_TRACE_RECORD_WRITERS")
}

/// The records and zero records of the executions whose first record is inside one of `buffers`.
fn records_in(buffers: &std::collections::BTreeSet<(u64, u64)>, executions: &std::collections::BTreeMap<u64, [u64; 2]>) -> [u64; 2] {
    executions
        .iter()
        .filter(|(at, _)| buffers.iter().any(|&(start, size)| (start..start.saturating_add(size)).contains(at)))
        .fold([0; 2], |a, (_, n)| [a[0] + n[0], a[1] + n[1]])
}

#[cfg(test)]
#[test]
fn records_are_counted_for_the_buffers_that_hold_their_first_record() {
    let executions = [(0x1000, [4, 4]), (0x1ff4, [2, 1]), (0x2000, [8, 0])].into_iter().collect();
    assert_eq!(records_in(&[(0x1000, 0x1000)].into_iter().collect(), &executions), [6, 5]);
    assert_eq!(records_in(&[(0x1000, 0x1000), (0x1800, 0x1000)].into_iter().collect(), &executions), [14, 5]);
    assert_eq!(records_in(&[(0x3000, 0x10), (u64::MAX, 2)].into_iter().collect(), &executions), [0, 0]);
}

fn trace_record_dispatch(t: &MetalTitle, l: &MetalList, outcome: &str) {
    if !tracing_record_writers() && trace_into_at().is_none() {
        return;
    }
    let (buffers, reads, writes) = t.host.table_views(l.list.compute_args());
    trace_into(writes.iter().copied(), || format!("{:?} {outcome} (unordered access)", stage_hashes(l.pipeline)));
    if !tracing_record_writers() {
        return;
    }
    let mut trace = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner());
    let d = trace.get_or_insert_default().dispatches.entry(format!("{:?} {outcome}", stage_hashes(l.pipeline))).or_default();
    d.buffers.extend(buffers);
    d.reads.extend(reads);
    d.writes.extend(writes);
}

/// `VICTORY_TRACE_RECORD_WRITERS`, at each present: the frame's indirect mesh records and how
/// many were 0 as recorded (the GPU may still write them: a culling pass that ran fills them
/// later, one that was dropped never does); each compute program whose tables hold a record
/// buffer for unordered access, with the records in those buffers and the textures it reads;
/// and for each such texture, the compute programs that hold it for unordered access.
#[inline(never)]
fn trace_record_writers() {
    let Some(trace) = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    let total = trace.executions.values().fold([0u64; 2], |a, n| [a[0] + n[0], a[1] + n[1]]);
    if total[0] == 0 {
        return;
    }
    let texture = |r: usize| {
        let i = crate::resource::texture_info(r as This);
        let (f, w, h) = i.map_or((0, 0, 0), |i| (i.format, i.width, i.height));
        format!("{r:#x} fmt {f} {w}x{h}")
    };
    tracing::info!(records = total[0], zero = total[1], "mesh records");
    let mut read = std::collections::BTreeSet::new();
    for (name, d) in &trace.dispatches {
        let n = records_in(&d.buffers, &trace.executions);
        if n[0] == 0 {
            continue;
        }
        read.extend(&d.reads);
        let reads: Vec<String> = d.reads.iter().map(|&r| texture(r)).collect();
        tracing::info!(records = n[0], zero = n[1], holder = name, ?reads, "mesh record buffer holder");
    }
    for r in read {
        let writers: Vec<&String> = trace.dispatches.iter().filter(|(_, d)| d.writes.contains(&r)).map(|(n, _)| n).collect();
        tracing::info!(texture = texture(r), ?writers, "texture read by a mesh record buffer holder");
    }
}

#[inline(never)]
fn trace_present(resource: This) {
        let stats = TRACE.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
        let ops = std::mem::take(&mut *DROPPED_OPS.lock().unwrap_or_else(|e| e.into_inner()));
        let reasons = std::mem::take(&mut *DROP_REASONS.lock().unwrap_or_else(|e| e.into_inner()));
        let mut pipes = std::mem::take(&mut *DROP_PIPES.lock().unwrap_or_else(|e| e.into_inner()));
        pipes.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for (what, n) in &pipes {
            tracing::info!(n, what, "draws dropped (no host pipeline, or an indirect execution)");
        }
        for (what, n) in std::mem::take(&mut *INTO.lock().unwrap_or_else(|e| e.into_inner())) {
            tracing::info!(n, what, "writes into the traced texture");
        }
        let described: Vec<String> = with_title(|t| {
            stats
                .iter()
                .map(|(k, [d, c, x])| {
                    let i = crate::resource::texture_info(*k as This);
                    let (f, w, h, a) = i.map_or((0, 0, 0, 0), |i| (i.format, i.width, i.height, i.memory as usize));
                    let mark = if *k == resource as usize { " PRESENTED" } else { "" };
                    let _ = &t;
                    format!("{k:#x} fmt {f} {w}x{h} at {a:#x} draws {d} clears {c} dropped {x}{mark}")
                })
                .collect()
        })
        .unwrap_or_default();
        dump_targets(&stats);
        dump_title_memory(&stats);
        let pi = crate::resource::texture_info(resource);
        tracing::info!(
            present = ?resource,
            at = pi.as_ref().map_or(0, |i| i.memory as usize),
            fmt = pi.as_ref().map_or(0, |i| i.format),
            targets = ?described,
            dropped_ops = ?ops.iter().take(8).collect::<Vec<_>>(),
            dropped = ops.len(),
            copies = COPIES[1].swap(0, Ordering::Relaxed),
            ?reasons,
            "frame targets"
        );
}

/// `VICTORY_TRACE_PRESENT=<every>`: the title resources `PresentX` has shown.
static PRESENTED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// When the title first presented (or a present trace first asked).
fn present_start() -> std::time::Instant {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    *START.get_or_init(std::time::Instant::now)
}

/// `VICTORY_TRACE_PRESENT=<every>[,<from seconds>]`.
fn present_every() -> Option<(u32, u64)> {
    static EVERY: OnceLock<Option<(u32, u64)>> = OnceLock::new();
    *EVERY.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_PRESENT").ok()?;
        let mut parts = v.split(',');
        let every = parts.next()?.trim().parse().ok()?;
        let from = parts.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        Some((every, from))
    })
}

/// `VICTORY_TRACE_PRESENT=<every>[,<from seconds>]`: a draw into a texture the title has
/// presented (its present pass), from `<from seconds>` after the first present on (0 when left
/// out), the first and then every `<every>`th, at most 12: its pipeline's shaders, root
/// parameters and binding, which textures its tables hold and how much of each is non-zero, and
/// the first bytes of its constant buffers, all as the draw is recorded.
fn trace_present_draw(t: &MetalTitle, l: &MetalList) {
    use std::sync::atomic::AtomicU32;
    static N: AtomicU32 = AtomicU32::new(0);
    let Some((every, from)) = present_every() else { return };
    let every = every.max(1);
    if present_start().elapsed().as_secs() < from {
        return;
    }
    {
        let shown = PRESENTED.lock().unwrap_or_else(|e| e.into_inner());
        if !l.target_keys.iter().any(|k| shown.contains(k)) {
            return;
        }
    }
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n % every != 0 || n / every >= 12 {
        return;
    }
    let shaders = stage_hashes(l.pipeline);
    let targets: Vec<String> = l.target_keys.iter().map(|k| format!("{k:#x}")).collect();
    tracing::info!(n, pipeline = format!("{:#x}", l.pipeline), ?shaders, ?targets, "present draw");
    trace_draw_args(t, l);
}

/// A draw's static samplers, root parameters and binding, which textures its tables hold and how
/// much of each is non-zero, and the first bytes of its constant buffers.
fn trace_draw_args(t: &MetalTitle, l: &MetalList) {
    let (root, binding) = l.list.graphics_args();
    if let Some(b) = binding {
        // Filter, address u v w, mip LOD bias, anisotropy, comparison, border, min and max LOD.
        let ps = b.ps.as_ref().map(|p| &p.static_samplers);
        tracing::info!(vs = ?b.static_samplers, ?ps, "present draw static samplers");
    }
    t.host.trace_args(root, binding);
}

/// `PresentX` on title queue `queue`: title texture `resource` over the layer.
pub(crate) fn present(queue: This, resource: This) {
    if present_every().is_some() {
        present_start();
        let mut shown = PRESENTED.lock().unwrap_or_else(|e| e.into_inner());
        if !shown.contains(&(resource as usize)) {
            shown.push(resource as usize);
        }
    }
    // Diagnostic: VICTORY_PRESENT_DELAY_MS slows every present, to tell a trigger that follows
    // frames from one that follows time.
    static DELAY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *DELAY.get_or_init(|| {
        std::env::var("VICTORY_PRESENT_DELAY_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    });
    if ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    if tracing_record_writers() {
        trace_record_writers();
    }
    if tracing_targets() {
        trace_present(resource);
    }
    let shown = with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        t.present(queue as usize, resource as usize)
    });
    match shown {
        Some(Err(e)) => tracing::warn!(error = %e, "a frame was not presented"),
        Some(Ok(false)) => {
            NO_TEXTURE.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

/// Frames presented to the layer so far.
pub fn frames_presented() -> u64 {
    with_title(|t| t.surface().map_or(0, |s| s.frames())).unwrap_or(0)
}

/// The pixels (BGRA) of the last frame presented, once the GPU has drawn it; for tests.
pub fn last_frame() -> Option<(usize, usize, Vec<u8>)> {
    with_title(|t| t.surface().and_then(|s| s.read_last())).flatten()
}

// --- resources and descriptors ---

/// A buffer resource object `res` of `len` bytes at title address `addr`.
pub(crate) fn buffer_created(res: This, addr: u64, len: u64) {
    with_title(|t| t.host.register_buffer(res as usize, addr, len));
}

/// A pipeline object's last reference went: the host pipelines made for it are forgotten, as the
/// title makes thousands of variants (device slot 0x1b8) a minute and a new one often gets a freed
/// one's address (which drew text with the old variant's blend state: glyphs as boxes).
pub(crate) fn pipeline_released(pipeline: This) {
    let key = pipeline as usize;
    with_title(|t| {
        t.pipelines.remove(&key);
        t.console_pipes.retain(|(k, _), _| *k != key);
        t.thread_pipes.remove(&key);
    });
}

pub(crate) fn resource_released(res: This) {
    with_title(|t| {
        t.host.release_resource(res as usize);
        t.texture_released(res as usize);
    });
}

/// A title texture object: its host texture is made, and filled from title memory when work is
/// next submitted.
pub(crate) fn texture_created(res: This) {
    if let Some(info) = crate::resource::texture_info(res) {
        with_title(|t| t.texture_created(res as usize, info));
    }
}

/// `OMSetRenderTargets`: the host textures of the views' resources, in the views' formats.
pub(crate) fn targets(this: This, color: &[u64], depth: u64) {
    with_list(this, |t, l| {
        let views: Vec<_> = color.iter().map(|&h| crate::descriptor::rtv_record(h)).collect();
        let depth_key = crate::descriptor::dsv_record(depth).map(|r| r.resource as usize);
        let depth = depth_key.and_then(|k| t.texture_of(k));
        l.target_keys = views.iter().flatten().map(|r| r.resource as usize).chain(depth_key).collect();
        // A view the host cannot make keeps its slot: binding the rest would move every later
        // target down one, and `SV_TargetN` would write a surface this pass does not own.
        let made: Option<Vec<_>> = views
            .iter()
            .map(|slot| slot.and_then(|r| t.target_of(r)))
            .collect();
        let Some(made) = made else {
            l.targets_ready = false;
            l.targets = TargetFormats::default();
            l.target_keys.clear();
            l.list.set_render_targets(&[], None);
            warn_once("a render target the host cannot make; the passes that use it are refused");
            return;
        };
        l.targets_ready = true;
        l.targets = TargetFormats {
            color: made.iter().map(|(_, format)| *format).collect(),
            depth: depth_key.and_then(|k| t.format_of(k)),
        };
        let color: Vec<_> = made.into_iter().map(|(texture, _)| texture).collect();
        // Debugging: no depth attachment at all.
        let drop_depth = env_set!("VICTORY_DEBUG_NO_DEPTH_ATTACH");
        if drop_depth {
            l.targets.depth = None;
        }
        l.list.set_render_targets(&color, depth.as_ref().filter(|_| !drop_depth));
    });
}

/// `ClearRenderTargetView` against the attachment named by the descriptor.
pub(crate) fn clear_rtv(this: This, handle: u64, color: [f32; 4], rects: &[crate::list::Rect]) {
    with_list(this, |t, l| {
        if let Some(record) = crate::descriptor::rtv_record(handle)
            && let Some((texture, _)) = t.target_of(record)
        {
            trace_bump(record.resource as usize, 1);
            if rects.is_empty() {
                l.list.clear_render_target(&texture, color);
            } else {
                l.list.clear_render_target_rects(&texture, color, rects);
            }
        }
    });
}

#[inline(never)]
fn trace_clear_dsv(flags: u32, depth: f32) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 6 || (n % 40 == 0 && n < 3000) {
        tracing::info!(flags, depth, n, "ClearDepthStencilView");
    }
}

/// `ClearDepthStencilView`: depth (flag 1) and stencil (flag 2) are independent.
pub(crate) fn clear_dsv(this: This, handle: u64, flags: u32, depth: f32, stencil: u8, count: u32) {
    if tracing_targets() {
        trace_clear_dsv(flags, depth);
    }
    // Neither rectangle bounds nor attachment mip/layer selection are represented by
    // NativeList's clear command. Do not widen an unsupported clear to the base texture.
    if count != 0 || flags & 3 == 0 {
        return;
    }
    let Some(record) = crate::descriptor::dsv_record(handle).filter(|r| r.native_dsv_base_view()) else {
        return;
    };
    if !crate::resource::native_dsv_clear_supported(record.resource) {
        return;
    }
    with_list(this, |t, l| {
        if let Some(tex) = t.texture_of(record.resource as usize) {
            l.list.clear_depth_stencil(
                &tex,
                (flags & 1 != 0).then_some(depth),
                (flags & 2 != 0).then_some(stencil),
            );
        }
    });
}

/// Shader-visible descriptor heaps by address range: start, end, sampler heap, count. The host
/// heap is made when a list first names the heap, from what the title wrote by then.
static HEAPS: Mutex<Vec<(u64, u64, bool, usize)>> = Mutex::new(Vec::new());

pub(crate) fn heap_created(start: u64, samplers: bool, count: usize) {
    let step = if samplers { 16 } else { 32 };
    HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((start, start + step * count as u64, samplers, count));
}

pub(crate) fn heap_released(start: u64) {
    HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|h| h.0 != start);
    with_title(|t| t.host.drop_heap(start));
}

/// The host heap the descriptor at `handle` is in (made now if it was not) and its index.
fn heap_of(t: &mut MetalTitle, handle: u64) -> Option<(Rc<super::list::NativeHeap<Metal>>, usize)> {
    if let Some(found) = t.host.heap_at(handle) {
        return Some(found);
    }
    let (start, _, samplers, count) = HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .copied()
        .find(|h| (h.0..h.1).contains(&handle))?;
    let kind = if samplers {
        HeapKind::Sampler
    } else {
        HeapKind::Resource
    };
    let step = if samplers { 16 } else { 32 };
    match t.host.heap(kind, start, count) {
        Ok(heap) => {
            // What the title wrote before the heap was made.
            for i in 0..count as u64 {
                let _ = t.host.mirror(start + i * step);
            }
            t.host.heap_at(handle).or(Some((heap, 0)))
        }
        Err(e) => {
            tracing::warn!(error = %e, "no host descriptor heap");
            None
        }
    }
}

/// The title wrote a view at `handle` (the call `descriptor::mirror` is on Windows).
pub(crate) fn mirror(handle: u64) {
    with_title(|t| {
        match crate::descriptor::view(handle) {
            // A texture the title placed in its own memory has no resource object: the host
            // makes one the first time a view names it.
            Some(crate::descriptor::View::Placed(_, res, d)) if res[0] > 1 => {
                let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                let key = super::textures::placed_key(address, &res);
                if !t.host.knows(key)
                    && let Some(info) = crate::resource::placed_info(address, &res)
                    && let Some((pool, _)) = t.host.memory().resolve(address, 1)
                {
                    t.texture_placed(address, key, pool.id);
                    t.texture_created(key, info);
                    t.realize(key);
                }
            }
            // A view of a texture object that no list has used yet.
            Some(
                crate::descriptor::View::Shader(r, _) | crate::descriptor::View::Unordered(r, ..),
            ) => {
                t.realize(r);
            }
            _ => {}
        }
        if let Err(e) = t.host.mirror(handle) {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 3000 {
                tracing::info!(error = %e, handle = format!("{handle:#x}"), "a descriptor was not mirrored");
            }
        }
    });
}

// --- list recording ---

pub(crate) fn set_pipeline(this: This, pipeline: This) {
    with_list(this, |_, l| l.pipeline = pipeline as usize);
}

/// The host pipeline of title pipeline `key` for a draw into `targets`, made at first use. A
/// console pipeline's state names no target formats, so it is made again for each set of targets.
fn pipe(t: &mut MetalTitle, key: usize, targets: &TargetFormats) -> Pipe<Metal> {
    if let Some(p) = t.pipelines.get(&key) {
        return p.clone();
    }
    let console_key = (key, targets.clone());
    if let Some(p) = t.console_pipes.get(&console_key) {
        return p.clone();
    }
    // A cold first world frame binds ~1,300 pipelines; translating them all here holds the frame
    // past the title's watchdog. Once the frame has spent its budget, a pipeline whose shaders are
    // not translated yet is put off: its work is dropped this time (not cached, so a later draw
    // makes it), and its shaders go to the front of the prewarm pool.
    let frame = crate::frame::current();
    let left = frame_left();
    if left.is_zero()
        && let Some((stages, _)) = crate::pipeline::native_stages(key as This)
        && stages.iter().any(|(_, s)| !s.native_ready())
    {
        for (_, s) in stages.iter().filter(|(_, s)| !s.native_ready()) {
            crate::pipeline::prewarm_next(s);
        }
        let n = PUT_OFF.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::info!(frame, put_off = n, "pipelines put off for their translations so far");
        }
        return Pipe::Refused;
    }
    // A pipeline Metal is still compiling when the budget ends is put off the same way.
    t.compile_wait = compile_wait(t);
    let put_off = t.shaders.put_off;
    let (made, console) = make_pipe(t, key, targets);
    if t.shaders.put_off != put_off {
        return made;
    }
    if console {
        t.console_pipes.insert(console_key, made.clone());
    } else {
        t.pipelines.insert(key, made.clone());
    }
    made
}

/// What the current frame has left of its budget for new pipelines (zero: spent).
fn frame_left() -> std::time::Duration {
    BUDGET.lock().unwrap_or_else(|e| e.into_inner()).left(crate::frame::current(), std::time::Instant::now(), crate::prewarm::FIRST_BIND_BUDGET)
}

/// How long a new pipeline may hold the current frame while Metal makes it: what the frame has
/// left of its budget; past it, the few ms a pipeline the disk cache and the archive hold takes
/// (so a warm frame that outlasts its budget puts none off), until one is not made in them; then
/// nothing. A cold world frame asks for some 800 past its budget: 20 ms each would be 16 s.
fn compile_wait(t: &MetalTitle) -> std::time::Duration {
    const ARCHIVED: std::time::Duration = std::time::Duration::from_millis(20);
    /// The frame last past its budget, and the pipelines put off before that.
    static PAST: Mutex<(u64, u64)> = Mutex::new((0, 0));
    let left = frame_left();
    if !left.is_zero() {
        return left;
    }
    let frame = crate::frame::current();
    let mut past = PAST.lock().unwrap_or_else(|e| e.into_inner());
    if past.0 != frame {
        *past = (frame, t.shaders.put_off);
    }
    if past.1 == t.shaders.put_off { ARCHIVED } else { left }
}

/// When this frame made its first new pipeline, and pipelines put off since ([`pipe`]).
static BUDGET: Mutex<crate::prewarm::FrameBudget> = Mutex::new(crate::prewarm::FrameBudget::new());
static PUT_OFF: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Runs a translation `what` names on a thread of its own, once at a time per name, for a
/// pipeline put off past the frame's budget ([`guess`]).
fn put_off(what: String, translate: impl FnOnce() + Send + 'static) {
    static RUNNING: Mutex<Option<std::collections::HashSet<String>>> = Mutex::new(None);
    if !RUNNING.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert(what.clone()) {
        return;
    }
    let n = PUT_OFF.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::info!(put_off = n, "pipelines put off for their translations so far");
    }
    let _ = std::thread::Builder::new().name("shader-put-off".into()).spawn(move || {
        crate::prewarm::lower_priority();
        translate();
        if let Some(r) = RUNNING.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            r.remove(&what);
        }
    });
}

/// Work dropped because its pipeline has no host pipeline.
static DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A draw into `l`'s targets that the host drops, counted per target for the trace.
fn trace_drop(l: &MetalList) {
    l.target_keys.iter().for_each(|&k| trace_bump(k, 2));
    trace_into(l.target_keys.iter().copied(), || format!("{:?} dropped", stage_hashes(l.pipeline)));
}

/// A draw went on to the host list: counted for each of its targets.
fn trace_drawn(t: &MetalTitle, l: &MetalList) {
    l.target_keys.iter().for_each(|&k| trace_bump(k, 0));
    let what = std::cell::Cell::new(None);
    trace_into(l.target_keys.iter().copied(), || {
        let w = format!("{:?} drawn", stage_hashes(l.pipeline));
        what.set(Some(w.clone()));
        w
    });
    let Some(what) = what.take() else { return };
    // The first 3 draws of each kind from `<from seconds>` on: what they read.
    static SHOWN: Mutex<Vec<(String, u32)>> = Mutex::new(Vec::new());
    if present_start().elapsed().as_secs() < trace_into_at().map_or(0, |(_, from)| from) {
        return;
    }
    {
        let mut shown = SHOWN.lock().unwrap_or_else(|e| e.into_inner());
        let i = shown.iter().position(|(w, _)| *w == what).unwrap_or_else(|| {
            shown.push((what.clone(), 0));
            shown.len() - 1
        });
        if shown[i].1 >= 3 {
            return;
        }
        shown[i].1 += 1;
    }
    tracing::info!(what, "traced texture draw");
    trace_draw_args(t, l);
}

/// `VICTORY_TRACE_INTO=<title address, hex>[,<from seconds>]`: the title texture memory whose
/// writers to log, and from when to trace what its draws read.
fn trace_into_at() -> Option<(u64, u64)> {
    static AT: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *AT.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_INTO").ok()?;
        let mut parts = v.split(',');
        let at = u64::from_str_radix(parts.next()?.trim().trim_start_matches("0x"), 16).ok()?;
        let from = parts.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        present_start();
        Some((at, from))
    })
}

/// Writes since the last present into the `VICTORY_TRACE_INTO` texture, by what wrote them.
static INTO: Mutex<Vec<(String, u32)>> = Mutex::new(Vec::new());

/// With `VICTORY_TRACE_INTO`, counts `what` (made only then) when one of `keys` is the texture at
/// that address.
fn trace_into(mut keys: impl Iterator<Item = usize>, what: impl FnOnce() -> String) {
    let Some((at, _)) = trace_into_at() else { return };
    if !keys.any(|k| crate::resource::texture_info(k as This).is_some_and(|i| i.memory as u64 == at)) {
        return;
    }
    let what = what();
    let mut into = INTO.lock().unwrap_or_else(|e| e.into_inner());
    match into.iter_mut().find(|(w, _)| *w == what) {
        Some((_, n)) => *n += 1,
        None => into.push((what, 1)),
    }
}

/// A dropped draw or execution, by what it was (its shaders and targets), for the frame trace.
pub(crate) fn trace_drop_pipe(key: String) {
    let mut d = DROP_PIPES.lock().unwrap_or_else(|e| e.into_inner());
    match d.iter_mut().find(|(k, _)| *k == key) {
        Some((_, n)) => *n += 1,
        None => d.push((key, 1)),
    }
}

/// Work dropped since the last present, by reason, for the `VICTORY_TRACE_TARGETS` frame line.
pub(crate) fn trace_reason(why: &'static str) {
    if tracing_targets() {
        let mut r = DROP_REASONS.lock().unwrap_or_else(|e| e.into_inner());
        match r.iter_mut().find(|(w, _)| *w == why) {
            Some((_, n)) => *n += 1,
            None => r.push((why, 1)),
        }
    }
}

/// Counts a draw or dispatch the host drops, and warns the first time for each `why`.
fn dropped(why: &'static str) {
    DROPPED.fetch_add(1, Ordering::Relaxed);
    warn_once(why);
    trace_reason(why);
}

/// A warning logged once per distinct `what` for the run.
pub(crate) fn warn_once(what: &'static str) {
    static SEEN: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&what) {
        seen.push(what);
        tracing::warn!("native host drops work: {what} (first time; counted in \"native draws so far\")");
    }
}

/// Draws the title recorded, and those that went on to the host list.
static RECORDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ENCODED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Presents whose resource had no host texture.
static NO_TEXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(draws recorded, draws handed to the host list, draws dropped for want of a host pipeline,
/// presents that found no host texture)`, for the frame log.
pub(crate) fn draw_counts() -> (u64, u64, u64, u64) {
    (
        RECORDED.load(Ordering::Relaxed),
        ENCODED.load(Ordering::Relaxed),
        DROPPED.load(Ordering::Relaxed),
        NO_TEXTURE.load(Ordering::Relaxed),
    )
}

/// The frame log's line of native draw counters, and the commands dropped by kind (kept out of
/// `PresentX`, whose frame it would otherwise enlarge).
#[inline(never)]
pub(crate) fn log_draw_counts(frame: u64) {
    let (recorded, encoded, no_pipeline, no_texture) = draw_counts();
    let (mesh, mesh_indirect, _, copies, _) = crate::list::dropped_kinds();
    let copies_recorded = COPIES[0].load(Ordering::Relaxed);
    let indirect = crate::list::indirect_dropped();
    let (r, heap_cleared, [dropped_unset_root, dropped_unmapped, dropped_failed]) =
        with_title(|t| (t.host.backend.residency_counts(), t.host.cleared(), [t.host.unset_roots(), t.host.unmapped(), t.host.failed()])).unwrap_or_default();
    tracing::info!(frame, recorded, encoded, no_pipeline, no_texture, mesh, mesh_indirect, indirect, texture_copies = copies, copies_recorded, use_resource = r.use_resource, heap_resources = r.heap_resources, heap_entries = r.heap_entries, heap_allocations = r.heap_allocations, heap_set_mb = r.heap_set_bytes >> 20, heap_cleared, dropped_unset_root, dropped_unmapped, dropped_failed, "native draws so far");
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *TRACE.get_or_init(|| std::env::var_os("VICTORY_TRACE_SHADER_CACHE").is_some_and(|v| !v.is_empty() && v != "0")) {
        let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
        let (libraries_cached, libraries_converted) = victory_metal::disk::counts();
        tracing::info!(frame, shaders_cached, shaders_translated, libraries_cached, libraries_converted, "native shader cache so far");
    }
    super::descriptor_trace::log(frame);
}

/// A title pipeline object's host pipeline: its shaders translated for the host, then converted.
/// The second result says the pipeline depends on the draw's targets.
fn make_pipe(t: &mut MetalTitle, key: usize, targets: &TargetFormats) -> (Pipe<Metal>, bool) {
    let Some((stages, graphics)) = crate::pipeline::native_stages(key as This) else {
        return (Pipe::Refused, false);
    };
    let graphics = graphics.map(|g| {
        if g.names_targets() {
            (g, false)
        } else {
            (super::console_state::graphics_state_for(&g, targets.color.clone(), targets.depth), true)
        }
    });
    let console = graphics.as_ref().is_some_and(|g| g.1);
    if console && env_set!("VICTORY_TRACE_BIND") {
        tracing::info!(pipeline = key, ?targets, state = ?graphics.as_ref().map(|g| &g.0), "console pipeline state at its draw");
    }
    let graphics = graphics.map(|g| g.0);
    (make_host_pipe(t, stages, graphics), console)
}

/// The host pipeline of compute pipeline `key` for dispatches of thread counts, made at first
/// use: its shader translated for them (`Options::thread_counts`).
fn thread_pipe(t: &mut MetalTitle, key: usize) -> Pipe<Metal> {
    if let Some(p) = t.thread_pipes.get(&key) {
        return p.clone();
    }
    if frame_left().is_zero()
        && let Some(s) = crate::pipeline::compute_shader(key as This)
        && !s.threads_ready()
    {
        put_off(format!("{} threads", s.hash), move || drop(s.native_threads_shader()));
        return Pipe::Refused;
    }
    let (hash, shader) = crate::pipeline::native_threads(key as This);
    let name = format!("compute {hash} by thread counts");
    t.compile_wait = compile_wait(t);
    let put_off = t.shaders.put_off;
    let made = match shader {
        Ok(cs) => t.compute_pipe(&cs, &name),
        Err(e) => {
            t.refused_translation(&name, e);
            Pipe::Refused
        }
    };
    if t.shaders.put_off == put_off {
        t.thread_pipes.insert(key, made.clone());
    }
    made
}

fn make_host_pipe(
    t: &mut MetalTitle,
    stages: Vec<(&'static str, Arc<crate::pipeline::Shader>)>,
    graphics: Option<super::state::Graphics>,
) -> Pipe<Metal> {
    if stages.is_empty() && graphics.as_ref().is_some_and(|g| g.is_guessed()) {
        return Pipe::Skipped;
    }
    let name = stages
        .iter()
        .map(|(stage, s)| format!("{stage} {}", s.hash))
        .collect::<Vec<_>>()
        .join(" ");
    let mut hosts = Vec::new();
    for (stage, shader) in &stages {
        match shader.native() {
            Ok(h) => hosts.push((*stage, h.clone())),
            Err(e) => {
                t.refused_translation(&name, e.to_string());
                return Pipe::Refused;
            }
        }
    }
    let find = |stage: &str| hosts.iter().find(|h| h.0 == stage).map(|h| &*h.1);
    match graphics {
        None => match find("compute") {
            Some(cs) => t.compute_pipe(cs, &name),
            None => Pipe::Refused,
        },
        Some(g) if g.is_guessed() => match hosts.iter().find(|h| h.0 == "vertex") {
            Some((_, vs)) => Pipe::Guess(Rc::new(Guessed {
                vs_source: stages.iter().find(|s| s.0 == "vertex").expect("a vertex stage").1.clone(),
                ps_source: stages.iter().find(|s| s.0 == "pixel").map(|s| s.1.clone()),
                vs: vs.clone(),
                ps: hosts.iter().find(|h| h.0 == "pixel").map(|h| h.1.clone()),
                name,
                state: g.clone(),
                made: Default::default(),
            })),
            None => Pipe::Refused,
        },
        Some(g) => match find("vertex") {
            Some(vs) => t.render_pipe(vs, find("pixel"), &g, &name),
            None => {
                t.refused_translation(&name, "a graphics pipeline without a vertex shader".into());
                Pipe::Refused
            }
        },
    }
}

/// The shader path's counters: pipelines made and cached, refusals by reason.
pub fn shader_stats() -> Option<super::shaders::ShaderStats> {
    with_title(|t| t.shaders.clone())
}

/// Makes title pipeline `key` the compute pipeline of a pre-translated DXIL container with the
/// given group size (what a translated shader's pipeline is made of); for diagnostics and tests.
pub fn install_compute_pipeline(key: usize, dxil: &[u8], threads: [usize; 3]) -> Result<(), String> {
    with_title(|t| {
        let pso = t.host.compute_pipeline(dxil, threads)?;
        t.pipelines.insert(key, Pipe::Compute(pso));
        Ok(())
    })
    .unwrap_or_else(|| Err("no native host GPU".into()))
}

#[cfg(test)]
pub(crate) fn install_test_render_pipeline(
    key: usize,
    pso: Rc<<Metal as victory_backend::resources::GraphicsBackend>::RenderPipeline>,
) {
    with_title(|t| {
        t.pipelines.insert(key, Pipe::Render(pso, None));
    });
}

pub(crate) fn dispatch(this: This, x: u32, y: u32, z: u32) {
    with_list(this, |t, l| match pipe(t, l.pipeline, &TargetFormats::default()) {
        Pipe::Compute(pso) => {
            trace_record_dispatch(t, l, "ran");
            l.list.set_compute_pipeline(&pso);
            if let Err(e) = l.list.dispatch(x, y, z) {
                tracing::warn!(error = %e, "dispatch");
            }
        }
        _ => dropped_dispatch(t, l, &format!("groups [{x}, {y}, {z}]")),
    });
}

/// A dispatch dropped for want of a host compute pipeline; with `VICTORY_TRACE_TARGETS`, counted
/// by its shader, `how` (group counts, or indirect) and the textures its tables hold as
/// unordered-access views (what it may have written) for the frame line.
fn dropped_dispatch(t: &MetalTitle, l: &MetalList, how: &str) {
    dropped("a dispatch without a host compute pipeline");
    trace_record_dispatch(t, l, "dropped");
    if tracing_targets() {
        let uavs: Vec<String> = t
            .host
            .table_uav_textures(l.list.compute_args())
            .into_iter()
            .map(|r| {
                let i = crate::resource::texture_info(r as This);
                let (f, w, h) = i.map_or((0, 0, 0), |i| (i.format, i.width, i.height));
                format!("{r:#x} fmt {f} {w}x{h}")
            })
            .collect();
        trace_drop_pipe(format!("{:?} {how} uavs {uavs:?}", stage_hashes(l.pipeline)));
    }
}

/// The stages of title pipeline `pipeline` with their shader hashes (for traces).
fn stage_hashes(pipeline: usize) -> Vec<String> {
    crate::pipeline::native_stages(pipeline as This)
        .map(|(stages, _)| stages.iter().map(|(s, sh)| format!("{s} {}", sh.hash)).collect())
        .unwrap_or_default()
}

/// `DispatchX` with thread counts (`crate::list::dispatch_x`). Without the translation for
/// them it is dropped: run as group counts, its lanes past the counts would write over memory
/// that is not theirs.
pub(crate) fn dispatch_threads(this: This, threads: [u32; 3]) {
    with_list(this, |t, l| match thread_pipe(t, l.pipeline) {
        Pipe::Compute(pso) => {
            l.list.set_compute_pipeline(&pso);
            if let Err(e) = l.list.dispatch_threads(threads) {
                tracing::warn!(error = %e, "dispatch of thread counts");
            }
        }
        _ => dropped("a dispatch of thread counts without a host compute pipeline"),
    });
}

/// `IASetPrimitiveTopology`.
pub(crate) fn topology(this: This, topology: u32) {
    with_list(this, |_, l| l.topology = topology);
}

/// `RSSetViewports`: the first one.
pub(crate) fn viewports(this: This, first: Option<[f32; 6]>) {
    if tracing_targets() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(?first, "viewport set");
        }
    }
    with_list(this, |_, l| {
        l.list.set_viewport(first.map(|v| victory_backend::resources::Viewport {
            x: v[0],
            y: v[1],
            width: v[2],
            height: v[3],
            min_depth: v[4],
            max_depth: v[5],
        }))
    });
}

/// `RSSetScissorRects`: the first one (left, top, right, bottom).
pub(crate) fn scissors(this: This, first: Option<[i32; 4]>) {
    with_list(this, |_, l| l.list.set_scissor(first));
}

pub(crate) fn blend_factor(this: This, factor: Option<[f32; 4]>) {
    with_list(this, |_, l| l.list.set_blend_factor(factor.unwrap_or([1.0; 4])));
}

pub(crate) fn stencil_ref(this: This, reference: u32) {
    with_list(this, |_, l| l.list.set_stencil_ref(reference));
}

/// `IASetIndexBuffer`: the buffer's address and whether its indices are 32 bits.
pub(crate) fn index_buffer(this: This, view: Option<(u64, bool)>) {
    with_list(this, |_, l| match view {
        Some((address, wide)) => l.list.set_index_buffer(address, wide),
        None => l.list.clear_index_buffer(),
    });
}

/// Title ABI for `D3D12_VERTEX_BUFFER_VIEW`.
#[repr(C)]
pub(crate) struct VertexBufferView {
    address: u64,
    size: u32,
    stride: u32,
}

pub(crate) fn vertex_buffers(this: This, start: u32, count: u32, views: &[VertexBufferView]) {
    with_list(this, |_, l| {
        for i in 0..count as usize {
            let slot = start as usize + i;
            let (address, stride) = match views.get(i) {
                Some(v) if v.address != 0 && v.size != 0 => (v.address, v.stride),
                _ => (0, 0),
            };
            l.list.set_vertex_buffer(slot, address);
            if l.strides.len() <= slot {
                l.strides.resize(slot + 1, 0);
            }
            l.strides[slot] = stride;
        }
    });
}

/// The host pipeline of a guessed pipeline for the targets and vertex buffers the list has bound.
fn guess(t: &mut MetalTitle, l: &MetalList, g: &Guessed<Metal>, rect: bool) -> Pipe<Metal> {
    // Unsigned-integer colour targets, which take the pixel shader translated with integer outputs.
    let uint = l.targets.color.iter().take(8).enumerate().filter(|(_, f)| f.is_integer()).fold(0u8, |m, (n, _)| m | 1 << n);
    let key = format!("{:?}{rect}{uint}", l.strides);
    if let Some(p) = g.made.borrow().get(&key) {
        return p.clone();
    }
    // As in `pipe`: past the frame's budget, a translation not made yet goes to a worker and the
    // draw is dropped this time (nothing is cached; the worker's result is, on disk or in the
    // shader).
    let spent = frame_left().is_zero();
    if spent && rect && !g.vs_source.rect_ready() {
        let s = g.vs_source.clone();
        put_off(format!("{} rect", s.hash), move || drop(s.native_rect()));
        return Pipe::Refused;
    }
    let rect_vs = if rect {
        match g.vs_source.native_rect() {
            Ok(h) => Some(h.clone()),
            Err(e) => {
                t.refused_translation(&g.name, format!("rectangle list: {e}"));
                return Pipe::Refused;
            }
        }
    } else {
        None
    };
    // Parameters the pixel shader reads and the vertex shader does not write: written (undefined)
    // too, as the converter refuses a pipeline whose fragment inputs the vertex stage lacks.
    let base_vs = rect_vs.as_deref().unwrap_or(&g.vs);
    let extra = g.ps.as_ref().map_or(0, |ps| {
        super::state::texcoords(&ps.dxil, victory_shader::FourCC::ISG1)
            & !super::state::texcoords(&base_vs.dxil, victory_shader::FourCC::OSG1)
    });
    let widened = if extra != 0 {
        let mut copy = [0u8; 32];
        for &(orig, new) in g.ps.as_ref().map_or(&[][..], |ps| &ps.splits[..]) {
            if orig < 32 && new < 32 {
                copy[new as usize] = orig as u8 + 1;
            }
        }
        if spent && !g.vs_source.extra_cached(rect, extra, copy) {
            let s = g.vs_source.clone();
            put_off(format!("{} {rect} {extra} {copy:?}", s.hash), move || drop(s.translate_extra(rect, extra, copy)));
            return Pipe::Refused;
        }
        match g.vs_source.translate_extra(rect, extra, copy) {
            Ok(h) => Some(h),
            Err(e) => {
                t.refused_translation(&g.name, format!("extra outputs: {e}"));
                return Pipe::Refused;
            }
        }
    } else {
        None
    };
    let vs = widened.as_deref().unwrap_or(base_vs);
    let int_ps = match (&g.ps_source, uint) {
        (Some(s), 1..) if g.ps.is_some() => {
            if spent && !s.uint_cached(uint) {
                let s = s.clone();
                put_off(format!("{} uint {uint}", s.hash), move || drop(s.translate_uint(uint)));
                return Pipe::Refused;
            }
            match s.translate_uint(uint) {
                Ok(h) => Some(h),
                Err(e) => {
                    t.refused_translation(&g.name, format!("unsigned-integer targets: {e}"));
                    return Pipe::Refused;
                }
            }
        }
        _ => None,
    };
    let ps = int_ps.as_deref().or(g.ps.as_deref());
    let reads = super::state::vertex_inputs(&vs.dxil).unwrap_or_default();
    let state = g.state.for_draw(l.targets.color.clone(), l.targets.depth, l.strides.clone(), &reads, rect);
    t.compile_wait = compile_wait(t);
    let put_off = t.shaders.put_off;
    let made = t.render_pipe(vs, ps, &state, &g.name);
    // Put off, it is asked for again at every draw until Metal has made it: logged once made.
    if t.shaders.put_off == put_off {
        tracing::info!(name = %g.name, ?reads, targets = ?l.targets, strides = ?l.strides, "guessed pipeline");
        if tracing_targets() {
            tracing::info!(name = %g.name, ?state, depth_test = ?state.depth_test(), stencil = ?state.stencil(), cull = ?state.cull(), color = ?state.color_state(), "guessed pipeline state");
        }
        g.made.borrow_mut().insert(key, made.clone());
    }
    made
}

/// Sets the draw pipeline, topology and bindings. Returns rectangle mode for a drawable pipeline.
/// `mesh`: the call is a mesh dispatch, which only a mesh pipeline runs (and a draw only a vertex
/// pipeline).
fn graphics(t: &mut MetalTitle, l: &mut MetalList, mesh: bool) -> Option<bool> {
    if !l.targets_ready {
        dropped("a draw of a pass whose render targets the host cannot make");
        trace_drop(l);
        return None;
    }
    // A rectangle list (topology 17) of a guessed pipeline is drawn as instances of four
    // vertices, a triangle strip, each instance a rectangle (`Options::rect_list`). A described
    // pipeline has no fourth-corner shader translation, so it falls back to its supplied vertices
    // drawn as triangles rather than issuing 4-vertex instanced strips.
    let rect = l.topology == 17;
    let targets = l.targets.clone();
    let (made, rect) = match pipe(t, l.pipeline, &targets) {
        Pipe::Guess(g) => (guess(t, l, &g, rect), rect),
        p if rect => {
            if env_set!("VICTORY_TRACE_RECTS") {
                tracing::info!(pipeline = ?l.pipeline, kind = ?std::mem::discriminant(&p), "rect draw of a pipeline that is not guessed");
            }
            warn_once("a rectangle-list draw of a described pipeline, drawn without its fourth corner");
            (p, false)
        }
        Pipe::Skipped => {
            warn_once("a draw of a guessed pipeline without a vertex stage (a mesh pipeline)");
            trace_reason("a mesh pipeline");
            trace_drop(l);
            return None;
        }
        p => (p, false),
    };
    let Pipe::Render(pso, binding) = made else {
        dropped("a draw without a host render pipeline (refused or not translated)");
        trace_drop(l);
        if tracing_targets() {
            trace_drop_pipe(format!("{:?} into {:?}", stage_hashes(l.pipeline), l.targets));
        }
        return None;
    };
    if pso.mesh_threads().is_some() != mesh {
        dropped(if mesh { "a mesh dispatch of a vertex pipeline" } else { "a draw of a mesh pipeline" });
        trace_drop(l);
        return None;
    }
    let topology = if rect {
        Some(victory_backend::resources::Topology::TriangleStrip)
    } else if l.topology == 17 || mesh {
        // Described rectangle lists draw each 3-corner rectangle as a triangle; a mesh shader
        // declares its own topology and ignores the list's.
        Some(victory_backend::resources::Topology::Triangles)
    } else {
        super::state::topology(l.topology)
    };
    let Some(topology) = topology else {
        static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if l.topology < 64 && SEEN.fetch_or(1 << l.topology, Ordering::Relaxed) >> l.topology & 1 == 0 {
            tracing::warn!(topology = l.topology, "a draw dropped for its primitive topology");
        }
        dropped("a draw of a primitive topology the host does not draw");
        trace_drop(l);
        return None;
    };
    l.list.set_graphics_pipeline(&pso, topology);
    l.list.set_graphics_binding(binding);
    Some(rect)
}

pub(crate) fn draw(this: This, vertices: u32, instances: u32, first: u32, first_instance: u32) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    if tracing_targets() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        let late = START.get_or_init(std::time::Instant::now).elapsed().as_secs() >= 26;
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 8 || (late && n % 7 == 0 && n < 4000) {
            with_list(this, |_, l| {
                let mut head = [0u8; 72];
                let at = l.list.vertex_buffer_address(0);
                let ok = at.is_some_and(|a| super::safe_read(a as *const u8, &mut head));
                tracing::info!(vertices, instances, first, topology = l.topology, vb0 = ?at, ok, head = ?head, strides = ?l.strides, formats = ?l.targets, "draw");
            });
        }
    }
    // Debugging: only indexed draws (the content), not the boot passes.
    if env_set!("VICTORY_DEBUG_ONLY_INDEXED") {
        return;
    }
    with_list(this, |t, l| {
        if let Some(rect) = graphics(t, l, false) {
            if rect && env_set!("VICTORY_TRACE_RECTS") {
                static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
                let mut seen = SEEN.lock().unwrap();
                if seen.iter().filter(|&&p| p == l.pipeline as usize).count() < 3 {
                    seen.push(l.pipeline as usize);
                    let stride = l.strides.first().copied().unwrap_or(0) as usize;
                    let mut head = vec![0u8; (stride * 3).clamp(12, 192)];
                    let at = l.list.vertex_buffer_address(0).map(|a| a + (first as u64) * stride as u64);
                    let ok = at.is_some_and(|a| super::safe_read(a as *const u8, &mut head));
                    let floats: Vec<f32> = head.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                    tracing::info!(pipeline = ?l.pipeline, vertices, first, strides = ?l.strides, ok, ?floats, "rect draw");
                }
            }
            let call = if rect {
                l.list.draw_instanced(4, vertices / 3, 0, first / 3)
            } else {
                l.list.draw_instanced(vertices, instances, first, first_instance)
            };
            match call {
                Ok(()) => {
                    ENCODED.fetch_add(1, Ordering::Relaxed);
                    trace_present_draw(t, l);
                    trace_drawn(t, l);
                }
                Err(e) => tracing::warn!(error = %e, "draw"),
            }
        }
    });
}

pub(crate) fn draw_indexed(this: This, indices: u32, instances: u32, first: u32, base: i32, first_instance: u32) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    with_list(this, |t, l| {
        if let Some(rect) = graphics(t, l, false) {
            if rect {
                // shortcut: indexed rectangle lists are not drawn.
                dropped("an indexed rectangle-list draw");
                trace_drop(l);
                return;
            }
            match l.list.draw_indexed_instanced(indices, instances, first, base, first_instance) {
                Ok(()) => {
                    ENCODED.fetch_add(1, Ordering::Relaxed);
                    trace_present_draw(t, l);
                    trace_drawn(t, l);
                }
                Err(e) => tracing::warn!(error = %e, "indexed draw"),
            }
        }
    });
}

/// `ExecuteIndirect` of a single-command signature: `max` draws or dispatches, record `i` at title
/// address `records + i * stride`, each read on the GPU; with a count buffer at title address
/// `count`, record `i` runs only if `i` is below the count, also read on the GPU (D3D12 runs
/// min(count, max) records). False when the host cannot run it; the caller logs
/// that. Rectangle lists drop here.
pub(crate) fn execute_indirect(
    this: This,
    command: crate::signature::Command,
    stride: u32,
    max: u32,
    records: u64,
    count: Option<u64>,
) -> bool {
    use crate::signature::Command;
    let at = |i: u32| records + u64::from(i) * u64::from(stride);
    match command {
        Command::DispatchMesh => {
            RECORDED.fetch_add(u64::from(max), Ordering::Relaxed);
            with_list(this, |t, l| {
                if graphics(t, l, true).is_none() {
                    return;
                }
                if tracing_record_writers() {
                    let zero = (0..max)
                        .filter(|&i| super::textures::read_memory(at(i) as *const u8, 4).iter().all(|&b| b == 0))
                        .count() as u64;
                    let mut trace = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner());
                    let n = trace.get_or_insert_default().executions.entry(at(0)).or_default();
                    n[0] += u64::from(max);
                    n[1] += zero;
                }
                for i in 0..max {
                    match l.list.dispatch_mesh_indirect(at(i), count.map(|c| (c, i))) {
                        Ok(()) => {
                            ENCODED.fetch_add(1, Ordering::Relaxed);
                            trace_drawn(t, l);
                        }
                        Err(e) => tracing::warn!(error = %e, "indirect mesh dispatch"),
                    }
                }
                trace_present_draw(t, l);
            });
            true
        }
        Command::Dispatch => {
            with_list(this, |t, l| match pipe(t, l.pipeline, &TargetFormats::default()) {
                Pipe::Compute(pso) => {
                    trace_record_dispatch(t, l, "ran");
                    l.list.set_compute_pipeline(&pso);
                    for i in 0..max {
                        let r = match count {
                            Some(c) => l.list.dispatch_indirect_counted(at(i), c, i),
                            None => l.list.dispatch_indirect(at(i)),
                        };
                        if let Err(e) = r {
                            tracing::warn!(error = %e, "indirect dispatch");
                        }
                    }
                }
                _ => dropped_dispatch(t, l, "indirect"),
            });
            true
        }
        Command::Draw | Command::DrawIndexed => {
            RECORDED.fetch_add(u64::from(max), Ordering::Relaxed);
            with_list(this, |t, l| match graphics(t, l, false) {
                // shortcut: a rectangle list's records count vertices of three-corner rectangles;
                // drawing them needs the counts rewritten on the GPU first.
                Some(true) => {
                    dropped("an indirect rectangle-list draw");
                    trace_drop(l);
                    true
                }
                Some(false) => {
                    for i in 0..max {
                        let indexed = command == Command::DrawIndexed;
                        let r = match count {
                            Some(c) => l.list.draw_indirect_counted(at(i), indexed, c, i),
                            None => l.list.draw_indirect(at(i), indexed),
                        };
                        match r {
                            Ok(()) => {
                                ENCODED.fetch_add(1, Ordering::Relaxed);
                                trace_drawn(t, l);
                            }
                            Err(e) => tracing::warn!(error = %e, "indirect draw"),
                        }
                    }
                    true
                }
                None => true,
            })
            .unwrap_or(true)
        }
    }
}

pub(crate) fn root_constants(this: This, compute: bool, slot: u32, offset: u32, words: &[u32]) {
    with_list(this, |_, l| {
        l.list
            .set_root_constants_at(compute, slot as usize, offset as usize, words)
    });
}

pub(crate) fn root_view(this: This, compute: bool, slot: u32, address: u64) {
    with_list(this, |_, l| {
        if compute {
            l.list.set_compute_root_view(slot as usize, address)
        } else {
            l.list.set_graphics_root_view(slot as usize, address)
        }
    });
}

pub(crate) fn root_table(this: This, compute: bool, slot: u32, handle: u64) {
    with_list(this, |t, l| {
        let Some((heap, index)) = heap_of(t, handle) else {
            return;
        };
        if compute {
            l.list.set_compute_root_table(slot as usize, &heap, index)
        } else {
            l.list.set_graphics_root_table(slot as usize, &heap, index)
        }
    });
}

/// `SetDescriptorHeaps`: the title's heap objects.
pub(crate) fn set_heaps(this: This, heaps: &[This]) {
    with_list(this, |t, l| {
        let (mut views, mut samplers) = (None, None);
        for &h in heaps {
            let Some((start, sampler, _)) = crate::descriptor::span(h) else {
                continue;
            };
            let Some((heap, _)) = heap_of(t, start) else {
                continue;
            };
            if sampler {
                samplers = Some(heap);
            } else {
                views = Some(heap);
            }
        }
        l.list.set_heaps(views.as_ref(), samplers.as_ref());
    });
}

/// `CopyBufferRegion` between title buffers.
pub(crate) fn copy_buffer(this: This, dst: (This, u64), src: (This, u64), len: u64) {
    let (Some((to, to_size)), Some((from, from_size))) = (
        crate::resource::buffer_bytes(dst.0),
        crate::resource::buffer_bytes(src.0),
    ) else {
        return;
    };
    if dst.1.saturating_add(len) > to_size || src.1.saturating_add(len) > from_size {
        tracing::warn!(len, "copy outside its buffer");
        return;
    }
    // SAFETY: inside the buffers, checked above.
    let (to, from) = unsafe { (to.add(dst.1 as usize), from.add(src.1 as usize)) };
    copy_memory(this, to as u64, from as u64, len);
}

/// `CopyMemoryToMemoryX` and buffer copies: title addresses.
pub(crate) fn copy_memory(this: This, dst: u64, src: u64, len: u64) {
    with_list(this, |_, l| {
        l.list.copy_buffer_region(dst, src, len as usize);
    });
}

fn subresource_mip_slice(info: &super::textures::TextureInfo, subresource: u32) -> Option<(usize, usize)> {
    let mips = info.layout.mips.len().max(1);
    let slices = if info.dimension == victory_formats::texture::Dimension::Tex3d {
        1
    } else {
        info.layout.slices.max(1) as usize
    };
    let sub = subresource as usize;
    (sub < mips * slices).then(|| (sub % mips, sub / mips))
}

fn mip_dims(info: &super::textures::TextureInfo, mip: usize) -> [usize; 3] {
    [
        (info.width as usize >> mip).max(1),
        (info.height.max(1) as usize >> mip).max(1),
        info.layout.subresource_depth(mip as u32).max(1) as usize,
    ]
}

fn copy_box(
    info: &super::textures::TextureInfo,
    mip: usize,
    source_box: Option<[u32; 6]>,
) -> Option<([usize; 3], [usize; 3])> {
    let dims = mip_dims(info, mip);
    let Some([l, t, f, r, b, back]) = source_box else {
        return Some(([0, 0, 0], dims));
    };
    if r < l || b < t || back < f {
        return None;
    }
    let origin = [l as usize, t as usize, f as usize];
    let size = [(r - l) as usize, (b - t) as usize, (back - f) as usize];
    (0..3).all(|i| origin[i].saturating_add(size[i]) <= dims[i]).then_some((origin, size))
}

fn footprint_span(
    res: This,
    fp: &crate::resource::PlacedFootprint,
    at: [u32; 3],
    size: [u32; 3],
) -> Option<(u64, usize, usize, usize)> {
    // 115 (B4G4R4A4_UNORM) needs CPU nibble rotation; depth formats need plane-separated copies.
    if fp.format == 115 {
        return None;
    }
    let fmt = super::descriptors::format(fp.format)?;
    // A depth and stencil footprint names both planes in one element; the host copies the depth
    // plane alone (as a 4-byte format: `R32_TYPELESS` and the like).
    if fmt == victory_backend::resources::Format::D32FloatS8 {
        return None;
    }
    let (_, block) = crate::resource::element(fp.format)?;
    let (base, buf_size) = crate::resource::buffer_bytes(res)?;
    let (start, end) = fp.span(at, size)?;
    let byte_end = fp.offset.checked_add(end)?;
    if byte_end > buf_size {
        return None;
    }
    let addr = (base as u64).checked_add(fp.offset + start)?;
    let len = usize::try_from(end - start).ok()?;
    let row_pitch = fp.row_pitch as usize;
    let slice_pitch = row_pitch.checked_mul((fp.height as usize).div_ceil(block as usize))?;
    Some((addr, len, row_pitch, slice_pitch))
}

/// Texture copies recorded so far, and since the last present (the `VICTORY_TRACE_TARGETS` frame
/// line).
static COPIES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// A texture copy was recorded: counted, logged once per kind with `VICTORY_TRACE_TARGETS`, and its
/// destination texture (if any) is the GPU's from now on ([`Title::gpu_written`]).
fn copied(t: &mut MetalTitle, what: &str, src: This, dst: Option<This>) {
    COPIES[0].fetch_add(1, Ordering::Relaxed);
    COPIES[1].fetch_add(1, Ordering::Relaxed);
    if let Some(d) = dst {
        t.gpu_written(d as usize);
        trace_bump(d as usize, 0);
        trace_into(std::iter::once(d as usize), || format!("a copy ({what})"));
    }
    if tracing_targets() {
        static KINDS: Mutex<Vec<(String, Option<victory_backend::resources::Format>, Option<victory_backend::resources::Format>)>> = Mutex::new(Vec::new());
        let kind = (what.to_owned(), t.format_of(src as usize), dst.and_then(|d| t.format_of(d as usize)));
        let mut kinds = KINDS.lock().unwrap_or_else(|e| e.into_inner());
        if !kinds.contains(&kind) {
            tracing::info!(what, src = ?kind.1, dst = ?kind.2, "a texture copy (first of its kind)");
            kinds.push(kind);
        }
    }
}

/// Whether texels of `src` copy into `dst` on the host: one blit (the same format, or plain
/// formats of one size through a view), or through a buffer, formats of one block size where one
/// is block-compressed or depth (plain texels into a GPU encoder's compressed blocks, a depth
/// plane into a 4-byte float), as D3D12 copies them.
fn copy_formats(src: victory_backend::resources::Format, dst: victory_backend::resources::Format) -> bool {
    use victory_backend::resources::Format;
    let bytes = |f: Format| if f == Format::D32FloatS8 { 4 } else { f.block().1 };
    let special = |f: Format| f.block().0 > 1 || f.is_depth();
    victory_metal::gfx::direct_copy(src, dst) || ((special(src) || special(dst)) && bytes(src) == bytes(dst))
}

/// `size` source texels in the destination's texels: a plain texel is a whole compressed block
/// (cut at the destination's edge, from `at`), and a block one texel.
fn dest_size(src: victory_backend::resources::Format, dst: victory_backend::resources::Format, size: [usize; 3], at: [usize; 3], dims: [usize; 3]) -> [usize; 3] {
    let (sw, dw) = (src.block().0, dst.block().0);
    std::array::from_fn(|i| match i {
        2 => size[2],
        _ if dw > sw => (size[i] * dw).min(dims[i].saturating_sub(at[i])),
        _ if sw > dw => size[i].div_ceil(sw),
        _ => size[i],
    })
}

/// Whether a footprint's rows hold the texels of a texture (a depth and stencil texture's: its
/// 4-byte depth plane).
fn rows_match(fp: &crate::resource::PlacedFootprint, info: &super::textures::TextureInfo) -> bool {
    match (super::descriptors::format(fp.format), info.desc()) {
        (Some(f), Some(d)) => copy_formats(f, d.format),
        _ => false,
    }
}

/// `ResolveSubresource`: host textures have one sample, so a resolve copies what both
/// subresources hold (the console clips a larger source to the destination, as the D3D12 host
/// does); the format argument names a view the copy makes when it needs one.
pub(crate) fn resolve(this: This, dst: This, dst_sub: u32, src: This, src_sub: u32) -> bool {
    let (Some(si), Some(di)) = (crate::resource::texture_info(src), crate::resource::texture_info(dst)) else {
        return false;
    };
    let (Some((sm, _)), Some((dm, _))) = (subresource_mip_slice(&si, src_sub), subresource_mip_slice(&di, dst_sub)) else {
        return false;
    };
    let (a, b) = (mip_dims(&si, sm), mip_dims(&di, dm));
    let region = [0, 0, 0, a[0].min(b[0]) as u32, a[1].min(b[1]) as u32, a[2].min(b[2]) as u32];
    copy_texture_region(this, crate::list::CopyLocation::sub(dst, dst_sub), [0; 3], crate::list::CopyLocation::sub(src, src_sub), Some(region))
}

/// `CopyResource` between two title textures. Returns `false` if the host cannot copy them.
pub(crate) fn copy_texture_resource(this: This, dst: This, src: This) -> bool {
    let (Some(src_info), Some(dst_info)) = (
        crate::resource::texture_info(src),
        crate::resource::texture_info(dst),
    ) else {
        return false;
    };
    let mips = src_info.layout.mips.len().max(1).min(dst_info.layout.mips.len().max(1));
    let src_slices = if src_info.dimension == victory_formats::texture::Dimension::Tex3d {
        1
    } else {
        src_info.layout.slices.max(1) as usize
    };
    let dst_slices = if dst_info.dimension == victory_formats::texture::Dimension::Tex3d {
        1
    } else {
        dst_info.layout.slices.max(1) as usize
    };
    let slices = src_slices.min(dst_slices);
    with_list(this, |t, l| {
        let (Some(src_tex), Some(dst_tex)) = (t.texture_of(src as usize), t.texture_of(dst as usize)) else {
            return false;
        };
        let (Some(sf), Some(df)) = (t.format_of(src as usize), t.format_of(dst as usize)) else {
            return false;
        };
        if !copy_formats(sf, df) {
            return false;
        }
        for slice in 0..slices {
            for mip in 0..mips {
                let s = mip_dims(&src_info, mip);
                let d = mip_dims(&dst_info, mip);
                let size = [s[0].min(d[0]), s[1].min(d[1]), s[2].min(d[2])];
                l.list.copy_texture_region(
                    &src_tex,
                    mip,
                    slice,
                    [0, 0, 0],
                    size,
                    &dst_tex,
                    mip,
                    slice,
                    [0, 0, 0],
                );
            }
        }
        copied(t, "CopyResource", src, Some(dst));
        true
    })
    .unwrap_or(false)
}

/// `CopyTextureRegion` on macOS: texture-to-texture and placed footprint <-> texture.
pub(crate) fn copy_texture_region(
    this: This,
    dst: crate::list::CopyLocation,
    origin: [u32; 3],
    src: crate::list::CopyLocation,
    source_box: Option<[u32; 6]>,
) -> bool {
    const SUBRESOURCE: u32 = 0;
    const FOOTPRINT: u32 = 1;
    let dst_origin = [origin[0] as usize, origin[1] as usize, origin[2] as usize];
    match (dst.kind, src.kind) {
        (SUBRESOURCE, SUBRESOURCE) => {
            let (Some(src_info), Some(dst_info)) = (
                crate::resource::texture_info(src.resource),
                crate::resource::texture_info(dst.resource),
            ) else {
                return false;
            };
            let Some((src_mip, src_slice)) = subresource_mip_slice(&src_info, src.footprint.offset as u32) else {
                return false;
            };
            let Some((dst_mip, dst_slice)) = subresource_mip_slice(&dst_info, dst.footprint.offset as u32) else {
                return false;
            };
            let Some((src_origin, size)) = copy_box(&src_info, src_mip, source_box) else {
                return false;
            };
            let (Some(sf), Some(df)) = (src_info.desc().map(|d| d.format), dst_info.desc().map(|d| d.format)) else {
                return false;
            };
            if !copy_formats(sf, df) {
                return false;
            }
            let dst_dims = mip_dims(&dst_info, dst_mip);
            let to = dest_size(sf, df, size, dst_origin, dst_dims);
            if (0..3).any(|i| dst_origin[i].saturating_add(to[i]) > dst_dims[i]) {
                return false;
            }
            if size.contains(&0) {
                return true;
            }
            with_list(this, |t, l| {
                let (Some(src_tex), Some(dst_tex)) =
                    (t.texture_of(src.resource as usize), t.texture_of(dst.resource as usize))
                else {
                    return false;
                };
                l.list.copy_texture_region(
                    &src_tex,
                    src_mip,
                    src_slice,
                    src_origin,
                    size,
                    &dst_tex,
                    dst_mip,
                    dst_slice,
                    dst_origin,
                );
                copied(t, "CopyTextureRegion", src.resource, Some(dst.resource));
                true
            })
            .unwrap_or(false)
        }
        (SUBRESOURCE, FOOTPRINT) => {
            let Some(dst_info) = crate::resource::texture_info(dst.resource).filter(|i| rows_match(&src.footprint, i)) else {
                return false;
            };
            let Some((dst_mip, dst_slice)) = subresource_mip_slice(&dst_info, dst.footprint.offset as u32) else {
                return false;
            };
            let (src_at, size_u32) = match source_box {
                Some([l, t, f, r, b, back]) if r >= l && b >= t && back >= f => {
                    ([l, t, f], [r - l, b - t, back - f])
                }
                Some(_) => return false,
                None => (
                    [0, 0, 0],
                    [src.footprint.width, src.footprint.height, src.footprint.depth],
                ),
            };
            let size = size_u32.map(|v| v as usize);
            if size.contains(&0) {
                return true;
            }
            let dst_dims = mip_dims(&dst_info, dst_mip);
            if (0..3).any(|i| dst_origin[i].saturating_add(size[i]) > dst_dims[i]) {
                return false;
            }
            let Some((src_addr, len, row_pitch, slice_pitch)) =
                footprint_span(src.resource, &src.footprint, src_at, size_u32)
            else {
                return false;
            };
            with_list(this, |t, l| {
                let Some(dst_tex) = t.texture_of(dst.resource as usize) else {
                    return false;
                };
                l.list.copy_buffer_to_texture(
                    src_addr,
                    len,
                    row_pitch,
                    slice_pitch,
                    size,
                    &dst_tex,
                    dst_mip,
                    dst_slice,
                    dst_origin,
                );
                copied(t, "CopyTextureRegion", src.resource, Some(dst.resource));
                true
            })
            .unwrap_or(false)
        }
        (FOOTPRINT, SUBRESOURCE) => {
            let Some(src_info) = crate::resource::texture_info(src.resource).filter(|i| rows_match(&dst.footprint, i)) else {
                return false;
            };
            let Some((src_mip, src_slice)) = subresource_mip_slice(&src_info, src.footprint.offset as u32) else {
                return false;
            };
            let Some((src_origin, size)) = copy_box(&src_info, src_mip, source_box) else {
                return false;
            };
            if size.contains(&0) {
                return true;
            }
            let size_u32 = size.map(|v| v as u32);
            let Some((dst_addr, len, row_pitch, slice_pitch)) =
                footprint_span(dst.resource, &dst.footprint, origin, size_u32)
            else {
                return false;
            };
            with_list(this, |t, l| {
                let Some(src_tex) = t.texture_of(src.resource as usize) else {
                    return false;
                };
                l.list.copy_texture_to_buffer(
                    &src_tex,
                    src_mip,
                    src_slice,
                    src_origin,
                    size,
                    dst_addr,
                    len,
                    row_pitch,
                    slice_pitch,
                );
                copied(t, "CopyTextureRegion", src.resource, None);
                true
            })
            .unwrap_or(false)
        }
        _ => false,
    }
}

/// A bottom-of-pipe value write of `value`, `bytes` long.
pub(crate) fn write_value(this: This, address: u64, value: u64, bytes: usize) -> bool {
    if env_set!("VICTORY_TRACE_WRITES") {
        let inside = pools().resolve(address, bytes as u64).is_some();
        tracing::warn!("list write {value:#x} ({bytes} bytes) at {address:#x}, in a pool: {inside}, list {this:p}");
    }
    with_list(this, |_, l| l.list.write_value(address, value, bytes)).is_some()
}

/// A wait for a memory value, recorded at the current position of the list.
pub(crate) fn wait_value(this: This, w: Wait) {
    if env_set!("VICTORY_TRACE_WRITES") {
        tracing::warn!("list wait at {:#x} cmp {} ref {:#x}, list {this:p}", w.address, w.comparison, w.reference);
    }
    with_list(this, |_, l| l.wait(w));
}

/// `Reset`: forgets what was recorded (the host objects it named go with it, under the lock).
pub(crate) fn reset_list(this: This) {
    with_list(this, |_, l| l.reset());
}

/// Drops a list object, whose `Rc`s must be dropped under the lock.
pub(crate) fn locked<R>(f: impl FnOnce() -> R) -> R {
    let mut f = Some(f);
    match with_title(|_| (f.take().expect("once"))()) {
        Some(r) => r,
        None => (f.take().expect("not run"))(),
    }
}

/// Counters of the host for tests and the frame line.
pub fn stats() -> Option<super::title::Stats> {
    with_title(|t| t.stats.clone())
}

/// An IEEE half's value (target dumps).
fn half_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = (u32::from(h >> 15), u32::from(h >> 10 & 0x1F), u32::from(h & 0x3FF));
    let v = match exp {
        0 => frac as f32 * 2f32.powi(-24),
        31 => if frac == 0 { f32::INFINITY } else { f32::NAN },
        e => f32::from_bits(((e + 112) << 23) | (frac << 13)),
    };
    if sign == 1 { -v } else { v }
}
