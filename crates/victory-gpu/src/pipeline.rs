//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! Grand Theft Auto VI creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    /// The container's length, as logged once its bytes are gone.
    size: usize,
    /// The container, until `translated` has made its translation (or its
    /// refusal) and let it go: a run holds one per distinct shader its title
    /// passes, and nothing reads it after. Off Windows nothing translates,
    /// so only tests keep it.
    #[cfg(any(windows, test, target_os = "macos"))]
    container: Mutex<Vec<u8>>,
    #[cfg(windows)]
    translated: OnceLock<Option<Translated>>,
    /// Its translation for the native host (container let go once made).
    #[cfg(target_os = "macos")]
    native: std::sync::OnceLock<Result<Arc<victory_shader::emit::HostShader>, String>>,
    /// Its translation for rectangle lists (guessed vertex shaders; made at the first such draw).
    #[cfg(target_os = "macos")]
    rect: std::sync::OnceLock<Result<Arc<victory_shader::emit::HostShader>, String>>,
    /// Its translation for dispatches of thread counts (compute shaders; made at the first one).
    #[cfg(target_os = "macos")]
    threads: std::sync::OnceLock<Result<Arc<victory_shader::emit::HostShader>, String>>,
    /// Where the title last held the container when it made a pipeline of it (0: nowhere, the
    /// host put it together), to read it again for [`native_threads`].
    #[cfg(target_os = "macos")]
    at: std::sync::atomic::AtomicUsize,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(*full.as_bytes())
        .or_insert_with(|| {
            let s = Arc::new(Shader::new(stage, bytes));
            // Tests that read containers back would race the translation, which lets them go.
            #[cfg(target_os = "macos")]
            if !cfg!(test) || stage == "prewarm-test" {
                prewarm(&s);
            }
            s
        })
        .clone()
}

/// Translate `shader` ahead on a host worker, so the title's thread finds it done when it first
/// binds it: ~1,300 first binds at the first world frame would otherwise translate inline, tens
/// of seconds in all, past the title's own watchdog. `native()` is a `OnceLock`, so whoever gets
/// there first does the work. Shaders earlier runs bound go first ([`crate::prewarm`]); the
/// workers are all cores but two, below the title's threads. `VICTORY_NO_SHADER_PREWARM=1` turns
/// this off.
#[cfg(target_os = "macos")]
static QUEUE: crate::prewarm::Queue<Arc<Shader>> = crate::prewarm::Queue::new();

/// Puts `shader` at the front of the prewarm pool's work (a pipeline that was put off waits for it).
#[cfg(target_os = "macos")]
pub(crate) fn prewarm_next(shader: &Arc<Shader>) {
    QUEUE.push(shader.clone(), true);
}

#[cfg(target_os = "macos")]
fn prewarm(shader: &Arc<Shader>) {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    static POOL: OnceLock<bool> = OnceLock::new();
    static QUEUED: AtomicUsize = AtomicUsize::new(0);
    static DONE: AtomicUsize = AtomicUsize::new(0);
    static DXIL: AtomicUsize = AtomicUsize::new(0);
    let on = *POOL.get_or_init(|| {
        if std::env::var_os("VICTORY_NO_SHADER_PREWARM").is_some_and(|v| !v.is_empty() && v != "0") {
            return false;
        }
        let cores = std::thread::available_parallelism().map_or(8, usize::from);
        let n = crate::prewarm::pool_size(cores);
        for i in 0..n {
            let _ = std::thread::Builder::new().name(format!("shader-prewarm-{i}")).spawn(|| {
                crate::prewarm::lower_priority();
                loop {
                    let s = QUEUE.pop();
                    let held = s.native().map_or(0, |h| h.dxil.len());
                    let dxil_mb = (DXIL.fetch_add(held, Relaxed) + held) >> 20;
                    let n = DONE.fetch_add(1, Relaxed) + 1;
                    if n.is_power_of_two() {
                        tracing::info!("shaders prewarmed n={n} pending={} dxil_mb={dxil_mb}", QUEUED.load(Relaxed).saturating_sub(n));
                    }
                }
            });
        }
        let first = with_first_binds(|l| l.len());
        tracing::info!(threads = n, first_binds = first, "shader prewarm pool started");
        true
    });
    if on {
        QUEUED.fetch_add(1, Relaxed);
        QUEUE.push(shader.clone(), with_first_binds(|l| l.contains(&shader.hash)));
    }
}

/// The list of first binds ([`crate::prewarm::FirstBinds`]), read from the cache folder on first use.
#[cfg(target_os = "macos")]
fn with_first_binds<R>(f: impl FnOnce(&mut crate::prewarm::FirstBinds) -> R) -> R {
    static LIST: Mutex<Option<crate::prewarm::FirstBinds>> = Mutex::new(None);
    let mut list = LIST.lock().unwrap_or_else(|e| e.into_inner());
    f(list.get_or_insert_with(|| {
        let text = crate::shader_cache::first_binds_file().and_then(|p| std::fs::read_to_string(p).ok());
        crate::prewarm::FirstBinds::parse(text.as_deref().unwrap_or(""))
    }))
}

impl Shader {
    fn new(stage: &'static str, bytes: &[u8]) -> Shader {
        Shader {
            stage,
            hash: blake3::hash(bytes).to_hex()[..16].to_owned(),
            size: bytes.len(),
            #[cfg(any(windows, test, target_os = "macos"))]
            container: Mutex::new(bytes.to_vec()),
            #[cfg(windows)]
            translated: OnceLock::new(),
            #[cfg(target_os = "macos")]
            native: std::sync::OnceLock::new(),
            #[cfg(target_os = "macos")]
            rect: std::sync::OnceLock::new(),
            #[cfg(target_os = "macos")]
            threads: std::sync::OnceLock::new(),
            #[cfg(target_os = "macos")]
            at: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// The translation for the native host, made on first use; the reason if there is none.
    #[cfg(target_os = "macos")]
    pub(crate) fn native(&self) -> Result<&Arc<victory_shader::emit::HostShader>, &str> {
        self.native
            .get_or_init(|| {
                let mut container = self.container.lock().unwrap_or_else(|e| e.into_inner());
                let made = crate::native::shaders::translate(&container).map(Arc::new);
                if (made.is_err() || std::env::var_os("VICTORY_DUMP_ALL_SHADERS").is_some())
                    && let Some(dir) = std::env::var_os("VICTORY_DUMP_REFUSED")
                {
                    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{}_{}.bin", self.stage, self.hash)), &*container);
                }
                // A vertex shader keeps its container: a rectangle-list draw translates it again;
                // so does a pixel shader, for a draw into unsigned-integer targets.
                if !matches!(self.stage, "vertex" | "pixel") {
                    *container = Vec::new();
                }
                made
            })
            .as_ref()
            .map_err(String::as_str)
    }
}

#[cfg(target_os = "macos")]
impl Shader {
    /// Whether the translation for the native host is made (or refused) already.
    pub(crate) fn native_ready(&self) -> bool {
        self.native.get().is_some()
    }

    /// Whether the rectangle-list translation is made (or refused) already.
    pub(crate) fn rect_ready(&self) -> bool {
        self.rect.get().is_some()
    }

    /// Whether the thread-count dispatch translation is made (or refused) already.
    pub(crate) fn threads_ready(&self) -> bool {
        self.threads.get().is_some()
    }

    /// The translation for dispatches of thread counts (`Options::thread_counts`), made on first use.
    pub(crate) fn native_threads_shader(&self) -> Result<Arc<victory_shader::emit::HostShader>, String> {
        self.threads
            .get_or_init(|| {
                self.native()?;
                let mut container = vec![0; self.size];
                let at = self.at.load(std::sync::atomic::Ordering::Relaxed);
                if at == 0 || !crate::queue::read_bytes(at, &mut container) {
                    return Err("its container is no longer in title memory".into());
                }
                if blake3::hash(&container).to_hex()[..16] != *self.hash {
                    return Err("its container changed since the pipeline was made".into());
                }
                crate::native::shaders::translate_threads(&container).map(Arc::new)
            })
            .clone()
    }

    /// A pixel shader's translation for draws into the unsigned-integer targets `uint` (bit n:
    /// colour target n).
    pub(crate) fn translate_uint(&self, uint: u8) -> Result<Arc<victory_shader::emit::HostShader>, String> {
        let container = self.container.lock().unwrap_or_else(|e| e.into_inner());
        crate::native::shaders::translate_uint(&container, uint).map(Arc::new)
    }

    /// Whether [`Self::translate_uint`] would find its translation on disk.
    pub(crate) fn uint_cached(&self, uint: u8) -> bool {
        let container = self.container.lock().unwrap_or_else(|e| e.into_inner());
        crate::native::shaders::uint_cached(&container, uint)
    }

    /// Whether [`Self::translate_extra`] would find its translation on disk.
    pub(crate) fn extra_cached(&self, rect: bool, extra: u32, copy: [u8; 32]) -> bool {
        let container = self.container.lock().unwrap_or_else(|e| e.into_inner());
        crate::native::shaders::extra_cached(&container, rect, extra, copy)
    }

    /// A translation that also writes the `TEXCOORD` parameters in `extra` (not cached).
    pub(crate) fn translate_extra(&self, rect: bool, extra: u32, copy: [u8; 32]) -> Result<Arc<victory_shader::emit::HostShader>, String> {
        let container = self.container.lock().unwrap_or_else(|e| e.into_inner());
        let made = crate::native::shaders::translate_extra(&container, rect, extra, copy).map(Arc::new);
        if made.is_err()
            && let Some(dir) = std::env::var_os("VICTORY_DUMP_REFUSED")
        {
            let copy: String = copy.iter().map(|b| format!("{b:02x}")).collect();
            let name = format!("extra_{}_{}_{extra:08x}_{copy}.bin", self.hash, u8::from(rect));
            let _ = std::fs::write(std::path::Path::new(&dir).join(name), &*container);
        }
        made
    }

    /// The translation for rectangle-list draws, made on first use.
    pub(crate) fn native_rect(&self) -> Result<&Arc<victory_shader::emit::HostShader>, &str> {
        self.rect
            .get_or_init(|| {
                let container = self.container.lock().unwrap_or_else(|e| e.into_inner());
                crate::native::shaders::translate_rect(&container).map(Arc::new)
            })
            .as_ref()
            .map_err(String::as_str)
    }
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    pub(crate) used: Vec<victory_shader::UsedResource>,
}

#[cfg(windows)]
impl Shader {
    /// The translation, made on first use (or taken from the cache); a
    /// shader that does not translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        self.translated
            .get_or_init(|| {
                let (started, cpu) = (std::time::Instant::now(), thread_cpu());
                let (done, cached) = {
                    let mut container = self.container.lock().unwrap_or_else(|e| e.into_inner());
                    let done = translate(&container);
                    // After the translation, not before: if it panics, the
                    // bytes are still there for the next try.
                    *container = Vec::new();
                    done
                };
                let ms = started.elapsed().as_millis() as u64;
                let (user_ms, kernel_ms) = thread_cpu_since(cpu);
                match &done {
                    Ok(t) => tracing::info!(
                        stage = self.stage,
                        hash = %self.hash,
                        user_data = t.host.user_data.len(),
                        samplers = t.host.samplers.len(),
                        cached,
                        ms,
                        user_ms,
                        kernel_ms,
                        "shader translated"
                    ),
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        cached,
                        ms,
                        user_ms,
                        kernel_ms,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok()
            })
            .as_ref()
    }
}

/// This thread's processor time so far, user and kernel, in 100 ns units.
#[cfg(windows)]
fn thread_cpu() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let mut t = [FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 }; 4];
    let [created, exited, kernel, user] = &mut t;
    // SAFETY: four times of our own thread into our own variables.
    unsafe { GetThreadTimes(GetCurrentThread(), created, exited, kernel, user) };
    let v = |f: &FILETIME| u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime);
    (v(user), v(kernel))
}

/// This thread's processor time since `then` ([`thread_cpu`]): user and
/// kernel milliseconds. Against the wall time, it tells work from waiting
/// (for a processor, a lock, the disk).
#[cfg(windows)]
fn thread_cpu_since(then: (u64, u64)) -> (u64, u64) {
    let now = thread_cpu();
    ((now.0 - then.0) / 10_000, (now.1 - then.1) / 10_000)
}

/// The translation, and whether it came from the cache.
#[cfg(windows)]
fn translate(container: &[u8]) -> (Result<Translated, String>, bool) {
    let (host, cached) = crate::shader_cache::host_shader(container);
    (host.and_then(|host| with_resources(container, host)), cached)
}

#[cfg(windows)]
fn with_resources(
    container: &[u8],
    host: victory_shader::emit::HostShader,
) -> Result<Translated, String> {
    use victory_shader::{Container, FourCC, HwStage};
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let used = match c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) {
        Some(part) => victory_shader::native_programs(part.data)
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|p| {
                matches!(
                    p.stage,
                    HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
                )
            })
            .map(|p| p.used)
            .unwrap_or_default(),
        None => Vec::new(),
    };
    Ok(Translated { host, used })
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
    /// Its graphics state, for the native host (streams only).
    #[cfg(target_os = "macos")]
    graphics: Option<Box<crate::native::state::Graphics>>,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// A title pipeline's stages (name, shader) and graphics state, for the native host.
#[cfg(target_os = "macos")]
pub(crate) fn native_stages(
    this: This,
) -> Option<(Vec<(&'static str, Arc<Shader>)>, Option<crate::native::state::Graphics>)> {
    if this.is_null() {
        return None;
    }
    let p = pipeline(this);
    let stages = p.shaders.iter().map(|s| (s.stage, s.clone())).collect();
    Some((stages, p.graphics.as_deref().cloned()))
}

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    #[cfg(target_os = "macos")]
    if std::env::var_os("VICTORY_TRACE_BIND").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static BINDS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
        let n = BINDS[usize::from(p.graphics.is_some())].fetch_add(1, Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::info!(with_state = p.graphics.is_some(), binds = n, "pipeline binds so far");
        }
    }
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    #[cfg(target_os = "macos")]
    if std::env::var_os("VICTORY_TRACE_BIND").is_some() {
        tracing::info!(pipeline = ?this, frame, with_state = p.graphics.is_some(), shaders = p.shaders.len(), "bound for the first time");
    }
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.size, hash = %shader.hash, "first bind");
    }
    // For the next run's prewarm: these first.
    #[cfg(target_os = "macos")]
    if let Some(path) = crate::shader_cache::first_binds_file() {
        let due = with_first_binds(|l| {
            for shader in &p.shaders {
                l.add(&shader.hash);
            }
            l.due()
        });
        if let Some(text) = due {
            crate::prewarm::write(path, &text);
        }
    }
}

/// The host pipeline of a title pipeline; `None` if one of its shaders does
/// not translate, the host refuses it, or it is still being made (its first
/// use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    make::ready(&pipeline(this).host)
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    let hashes = HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        parse_skip_hashes(&value)
    });
    if hashes.is_empty() {
        return None;
    }
    p.shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && skipped_shader(pipeline(this)).is_some()
}

/// The shaders of `this`, as host pipelines are named (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn shaders_of(this: This) -> String {
    if this.is_null() {
        return "no pipeline".into();
    }
    format!("{:?}", pipeline(this).shaders)
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        let p = pipeline(this);
        skipped_shader(p)
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| make::missing(&p.host))
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops
        .get_or_insert_with(HashMap::new)
        .entry(why)
        .or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    // SAFETY: the title passes `size` bytes of shader code.
    let shader = code_of(code).map(|bytes| shader(stage, unsafe { &*bytes }))?;
    #[cfg(target_os = "macos")]
    shader.at.store(code.code as usize, std::sync::atomic::Ordering::Relaxed);
    Some(shader)
}

/// The compute stage's shared [`Shader`] on title compute pipeline `this`, if any.
#[cfg(target_os = "macos")]
pub(crate) fn compute_shader(this: This) -> Option<Arc<Shader>> {
    if this.is_null() {
        return None;
    }
    let [shader] = &pipeline(this).shaders[..] else {
        return None;
    };
    Some(shader.clone())
}

/// Compute pipeline `this`'s shader hash (as logged) and its translation for dispatches of thread
/// counts (`Options::thread_counts`), made at the first one. The first translation let the
/// container go, so it is read again where the title held it, and used only if it still hashes
/// to the shader; if it is gone or changed, there is no translation.
// shortcut: a container the host put together (a compute pipeline of pieces, 0x218) has none;
// keep the pieces' addresses if one is ever dispatched with thread counts.
#[cfg(target_os = "macos")]
pub(crate) fn native_threads(this: This) -> (String, Result<Arc<victory_shader::emit::HostShader>, String>) {
    if this.is_null() {
        return (String::new(), Err("no pipeline".into()));
    }
    let Some(shader) = compute_shader(this) else {
        return (String::new(), Err("not a compute pipeline".into()));
    };
    let made = shader.native_threads_shader();
    (shader.hash.clone(), made)
}

/// The bytes a stage's bytecode names, if any.
fn code_of(code: &Bytecode) -> Option<*const [u8]> {
    (!code.code.is_null() && code.size != 0)
        .then(|| std::ptr::slice_from_raw_parts(code.code, code.size))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    #[cfg(target_os = "macos")] graphics: Option<Box<crate::native::state::Graphics>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(shaders.clone(), graphics));
    #[cfg(windows)]
    make::early(&host);
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        #[cfg(target_os = "macos")]
        graphics,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// A new pipeline with `base`'s shaders and `graphics` (device method 0x1b8, see
/// `native::console_state`).
#[cfg(target_os = "macos")]
pub(crate) fn derive(
    base: This,
    graphics: impl FnOnce(Option<&crate::native::state::Graphics>) -> crate::native::state::Graphics,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if base.is_null() {
        return E_INVALIDARG;
    }
    let from = pipeline(base);
    let shaders = from.shaders.clone();
    let graphics = graphics(from.graphics.as_deref());
    hand_out(shaders, Some(Box::new(graphics)), iid, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        #[cfg(target_os = "macos")]
        None,
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        #[cfg(target_os = "macos")]
        None,
        iid,
        out,
    )
}

/// Device methods 0x1c8 and 0x218: a pipeline state from a console description whose layout is
/// not known yet (`(desc, iid, out)`, the pipeline state's IID; title+0x2477578 and +0x24800d5).
/// With graphics inference enabled on macOS, read the shader pieces of the first slot.
/// shortcut: outside the inferred path, the pipeline has no shaders until the description is decoded.
pub(crate) extern "system" fn create_unread_1c8(
    _: This,
    desc: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // `VICTORY_GUESS_GRAPHICS=1`: a pipeline from the pieces of the first slot, with the
    // inferred state of `victory_shader::emit::graphics` and `native::state::Graphics::guessed`.
    #[cfg(target_os = "macos")]
    if !desc.is_null() && std::env::var_os("VICTORY_GUESS_GRAPHICS").is_some() {
        // SAFETY: the title passes a description of this layout.
        match unsafe { graphics_containers(desc.add(8)) } {
            Ok(stages) => {
                // Pair 0 of the slot is its register image (pointer, size).
                // SAFETY: the title passes a description of this layout, and the image's bytes.
                let image: Vec<u32> = unsafe {
                    let (ptr, size) = (desc.add(8).cast::<usize>().read_unaligned(), desc.add(16).cast::<usize>().read_unaligned());
                    if ptr == 0 || size > 0x1000 {
                        Vec::new()
                    } else {
                        let bytes = std::slice::from_raw_parts(ptr as *const u8, size);
                        if let Some(dir) = std::env::var_os("VICTORY_DUMP_DESC_DIR") {
                            let names: Vec<&str> = stages.iter().map(|s| &s.hash[..8]).collect();
                            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("regs_{}_{ptr:x}.bin", names.join("_"))), bytes);
                        }
                        bytes.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect()
                    }
                };
                return hand_out(
                    stages,
                    Some(Box::new(crate::native::state::Graphics::guess_from(&image))),
                    iid,
                    out,
                );
            }
            Err(reason) => {
                static SHOWN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SHOWN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
                    // SAFETY: a debugging read of the title's description.
                    let words: Vec<String> = (1..17).map(|i| format!("{:x}", unsafe { desc.cast::<u64>().add(i).read_unaligned() })).collect();
                    tracing::info!(?words, reason, "a graphics description whose shader pieces could not be inferred");
                }
            }
        }
    }
    create_unread_at(0x1c8, desc, iid, out)
}

/// The vertex and pixel shader containers of the slot at `slot` of a graphics description (pairs
/// 4/5 and 2/3: code and metadata) as guessed pipelines have them. An empty stage list is a
/// slot without a vertex piece.
///
/// # Safety
/// `slot` points at a slot of eight (pointer, size) pairs.
#[cfg(target_os = "macos")]
unsafe fn graphics_containers(slot: *const u8) -> Result<Vec<Arc<Shader>>, &'static str> {
    use victory_shader::HwStage;
    let entry = |pair: usize| {
        // SAFETY: the caller supplies all eight pairs.
        unsafe {
            let ptr = slot.add(16 * pair).cast::<usize>().read_unaligned();
            let size = slot.add(16 * pair + 8).cast::<usize>().read_unaligned();
            (ptr, size)
        }
    };
    let piece = |pair: usize| -> Option<&[u8]> {
        let (ptr, size) = entry(pair);
        // SAFETY: as the caller says; the pieces are the title's, of the size it gives.
        unsafe {
            (ptr != 0 && size != 0 && size <= 1 << 22)
                .then(|| std::slice::from_raw_parts(ptr as *const u8, size))
        }
    };
    let make = |stage, name, code, meta| match victory_shader::emit::graphics::container(stage, code, meta) {
        Ok(c) => Ok(shader(name, &c)),
        Err(e) => {
            tracing::warn!(stage = name, error = %e, "a guessed graphics piece is no program");
            Err("a graphics piece whose shader could not be inferred")
        }
    };
    // The vertex shader is pair 4/5, or pair 6/7 when that is empty (observed; the pairs vary).
    let (code, meta) = match (piece(4), piece(5)) {
        (Some(code), Some(meta)) => (code, meta),
        _ => match (piece(6), piece(7)) {
            (Some(code), Some(meta)) => (code, meta),
            // shortcut: no vertex producer means the slot is skipped, decode stage registers
            // before attempting a mesh pipeline here.
            _ if (4..8).all(|pair| entry(pair) == (0, 0)) => return Ok(Vec::new()),
            _ => return Err("missing or invalid vertex code or metadata"),
        },
    };
    let vs = make(HwStage::Vs, "vertex", code, meta)?;
    let mut shaders = vec![vs];
    match (piece(2), piece(3)) {
        (Some(code), Some(meta)) => shaders.push(make(HwStage::Ps, "pixel", code, meta)?),
        _ if entry(2) == (0, 0) && entry(3) == (0, 0) => {}
        _ => return Err("missing or invalid pixel code or metadata"),
    }
    Ok(shaders)
}

/// Device method 0x218: a compute pipeline. The description is a root signature pointer and one
/// 0x60-byte record, the pieces an `FSPX` part is made of (observed at run time, as the console
/// runtime takes them): pointer and size of the pipeline registers (kind 0, 0x1c bytes), of the
/// hash (kind 1, 0x14), of the GFX10 code (kind 6) and of the metadata (kind 7); more fields
/// follow that are not read. The root signature's blob goes into the part too
/// ([`victory_shader::native::FSPX_ROOT_SIGNATURE`]): it sizes the program's root constants.
pub(crate) extern "system" fn create_compute_pieces(
    _: This,
    desc: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: the title passes a description of this layout.
        let root = crate::root::blob_of(unsafe { desc.cast::<This>().read_unaligned() });
        // SAFETY: as above.
        if let Some(container) = unsafe { compute_container(root.as_deref(), desc.add(8)) } {
            return hand_out(
                vec![shader("compute", &container)],
                #[cfg(windows)]
                None,
                None,
                iid,
                out,
            );
        }
    }
    create_unread_at(0x218, desc, iid, out)
}

/// A container with the `ACCC` part of the pieces at `record` (see [`create_compute_pieces`]),
/// and of the root signature blob `root` if there is one.
///
/// # Safety
/// `record` points at four (pointer, size) pairs.
#[cfg(target_os = "macos")]
unsafe fn compute_container(root: Option<&[u8]>, record: *const u8) -> Option<Vec<u8>> {
    const KINDS: [u32; 4] = [0, 1, 6, 7];
    let mut pieces: Vec<(u32, &[u8])> = Vec::with_capacity(KINDS.len() + 1);
    for (i, kind) in KINDS.into_iter().enumerate() {
        // SAFETY: as the caller says.
        let (ptr, size) = unsafe {
            (
                record.add(16 * i).cast::<usize>().read_unaligned(),
                record.add(16 * i + 8).cast::<usize>().read_unaligned(),
            )
        };
        if ptr == 0 || size == 0 || size > 1 << 22 {
            return None;
        }
        // SAFETY: the title's piece of `size` bytes.
        pieces.push((kind, unsafe {
            std::slice::from_raw_parts(ptr as *const u8, size)
        }));
    }
    pieces.extend(root.map(|r| (victory_shader::native::FSPX_ROOT_SIGNATURE, r)));
    let mut part = vec![0u8; 0x10 + 16 * pieces.len()];
    for (i, (kind, bytes)) in pieces.iter().enumerate() {
        let at = part.len().next_multiple_of(16);
        part.resize(at, 0);
        let entry: Vec<u8> = [*kind, at as u32, bytes.len() as u32, 0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        part[0x10 + 16 * i..0x20 + 16 * i].copy_from_slice(&entry);
        part.extend_from_slice(bytes);
    }
    let header: Vec<u8> = [u32::from_le_bytes(*b"FSPX"), 1, pieces.len() as u32, part.len() as u32]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    part[..16].copy_from_slice(&header);
    Some(victory_shader::dxbc::build(&[(victory_shader::FourCC::ACCC, &part)]))
}

/// Debugging (`VICTORY_DUMP_DESC=1`): the first descriptions of each slot, with the bytes behind
/// every (pointer, length) pair in them, to stderr.
fn dump_description(slot: usize, desc: *const u8) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTS: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _one_at_a_time = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let n = COUNTS[usize::from(slot == 0x218)].fetch_add(1, Ordering::Relaxed);
    if n >= 50 {
        return;
    }
    // SAFETY: a debugging read of the title's description and of what it points at.
    unsafe {
        let words: Vec<u64> = (0..192).map(|i| desc.cast::<u64>().add(i).read_unaligned()).collect();
        eprintln!("DESC {slot:#x} #{n}");
        for (i, w) in words.chunks(4).enumerate() {
            eprintln!("  +{:#05x} {:016x} {:016x} {:016x} {:016x}", i * 32, w[0], w[1], w[2], w[3]);
        }
        if std::env::var_os("VICTORY_DUMP_STATE").is_some() && n < 6 {
            // The block of host pointers at the end of each stage slot: what each points at.
            for w in [0x110usize, 0x118, 0x128, 0x130] {
                let p = words[(8 + w) / 8];
                let q = p & 0xffff_ffff;
                for (tag, p) in [("full", p), ("low32", q)] {
                    if (0x1_0000_0000..0x7fff_ffff_ffff).contains(&p) || (tag == "low32" && p > 0x10000) {
                        let bytes = std::slice::from_raw_parts(p as *const u8, 0x100);
                        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                        eprintln!("  STATEPTR slot0+{w:#x} {tag} {p:#x} {hex}");
                    }
                }
            }
        }
        for i in 0..191 {
            let (p, len) = (words[i], words[i + 1]);
            if (0x1_0000_0000..0x7fff_ffff_ffff).contains(&p) && len > 0 && len <= 0x4000 {
                let bytes = std::slice::from_raw_parts(p as *const u8, (len as usize).min(if n < 3 { 0x1000 } else { 48 }));
                if let Some(dir) = std::env::var_os("VICTORY_DUMP_DESC_DIR")
                    && n < 12
                {
                    let all = std::slice::from_raw_parts(p as *const u8, len as usize);
                    let _ = std::fs::write(
                        std::path::Path::new(&dir).join(format!("{slot:x}_{n}_{:03x}.bin", i * 8)),
                        all,
                    );
                }
                let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                eprintln!("  BLOB +{:#x} ptr={p:#x} len={len:#x} {hex}", i * 8);
            }
        }
    }
}

fn create_unread_at(
    slot: usize,
    _desc: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if std::env::var_os("VICTORY_GUESS_GRAPHICS").is_some() {
        tracing::info!(slot, "a pipeline created without shaders");
    }
    if std::env::var_os("VICTORY_DUMP_DESC").is_some() && !_desc.is_null() {
        dump_description(slot, _desc);
    }
    hand_out(
        Vec::new(),
        #[cfg(windows)]
        None,
        #[cfg(target_os = "macos")]
        None,
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes a whole stream.
    let stream = unsafe { std::slice::from_raw_parts(desc.stream, desc.size) };
    if std::env::var_os("VICTORY_TRACE_STREAM").is_some() {
        trace_stream(stream);
    }
    #[allow(unused_variables)]
    let (shaders, graphics) = read_stream(stream, copy);
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        #[cfg(target_os = "macos")]
        Some(Box::new(graphics)),
        iid,
        out,
    )
}

/// Debugging (`VICTORY_TRACE_STREAM=1`): each distinct sequence of subobject types, with the
/// first stream that had it in full.
fn trace_stream(stream: &[u8]) {
    static SEEN: std::sync::Mutex<Vec<Vec<u32>>> = std::sync::Mutex::new(Vec::new());
    let mut kinds = Vec::new();
    let mut at = 0;
    while at + 8 <= stream.len() {
        let kind = u32::from_le_bytes(stream[at..at + 4].try_into().expect("4 bytes"));
        kinds.push(kind);
        match subobject(kind) {
            Some((size, _)) => at += size,
            None => break,
        }
    }
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() < 40 && !seen.contains(&kinds) {
        let hex: String = stream.iter().take(0x300).map(|b| format!("{b:02x}")).collect();
        eprintln!("STREAM kinds={kinds:?} size={:#x} {hex}", stream.len());
        seen.push(kinds);
    }
}

#[cfg(windows)]
type Graphics = host::GraphicsState;
#[cfg(target_os = "macos")]
type Graphics = crate::native::state::Graphics;
#[cfg(not(any(windows, target_os = "macos")))]
type Graphics = ();

/// A stream's shaders (each made by `shader` from its stage and bytecode)
/// and graphics state. Its pointers (shader code, input elements) must be
/// valid: the title's own, or the pre-build's.
fn read_stream(
    stream: &[u8],
    shader: impl Fn(&'static str, &Bytecode) -> Option<Arc<Shader>>,
) -> (Vec<Arc<Shader>>, Graphics) {
    let mut shaders = Vec::new();
    #[allow(unused_mut, clippy::let_unit_value)]
    let mut graphics = Graphics::default();
    let mut at = 0;
    while at + 8 <= stream.len() {
        let kind = u32::from_le_bytes(stream[at..at + 4].try_into().expect("4 bytes"));
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if at + size > stream.len() {
            break;
        }
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8.
            let code = unsafe { &*stream.as_ptr().add(at + 8).cast::<Bytecode>() };
            shaders.extend(shader(stage, code));
        }
        #[cfg(any(windows, target_os = "macos"))]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, stream.as_ptr().add(at))
        };
        at += size;
    }
    (shaders, graphics)
}

/// A compute pipeline whose shader is `code` (a container, or any bytes).
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn test_compute(code: &[u8]) -> This {
    let head = ComputeHead {
        root: std::ptr::null(),
        stage: Bytecode {
            code: code.as_ptr(),
            size: code.len(),
        },
    };
    let mut out = std::ptr::null();
    assert_eq!(
        create_compute(std::ptr::null(), &head, &PIPELINE_IIDS[0], &mut out),
        crate::object::S_OK
    );
    out
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    #[cfg(target_os = "macos")]
    if !this.is_null() && object::refs(this) == 1 {
        crate::native::glue::pipeline_released(this);
    }
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table::new(f)
};

#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
}

/// The pre-build: a title pipeline made ahead of a run into the pipeline
/// library (`use_pipeline_cache`), by the code a run makes it with, so the
/// run finds it there under the same name.
#[cfg(windows)]
pub struct Prebuild {
    shaders: Vec<Arc<Shader>>,
    graphics: Option<Box<host::GraphicsState>>,
}

// SAFETY: as `make::Pending`: the graphics state's input elements point
// into its own names, and it is only read.
#[cfg(windows)]
unsafe impl Send for Prebuild {}

#[cfg(windows)]
impl Prebuild {
    /// The pipeline a creation stream describes, as the title passes it to
    /// the device.
    ///
    /// # Safety
    /// The stream's pointers point at live shader code and input elements.
    pub unsafe fn graphics(stream: &[u8]) -> Prebuild {
        // Shaders of their own: a pre-build must not keep every container.
        let own = |stage, code: &Bytecode| {
            // SAFETY: the caller's pointers.
            code_of(code).map(|b| Arc::new(Shader::new(stage, unsafe { &*b })))
        };
        let (shaders, graphics) = read_stream(stream, own);
        Prebuild {
            shaders,
            graphics: Some(Box::new(graphics)),
        }
    }

    /// The compute pipeline of a container.
    pub fn compute(container: &[u8]) -> Prebuild {
        Prebuild {
            shaders: vec![Arc::new(Shader::new("compute", container))],
            graphics: None,
        }
    }

    /// What a run lists the pipeline by (its early list).
    pub fn key(&self) -> String {
        make::key(&self.shaders, self.graphics.as_deref())
    }

    /// Its shaders and state as a diagnostics build logs them.
    pub fn describe(&self) -> String {
        let state = self.graphics.as_ref().map(|g| g.describe()).unwrap_or_default();
        format!("{:?}\t{state}", self.shaders)
    }

    /// Makes the host pipeline, or finds it in the library; why not if it
    /// gets none.
    pub fn make(&self) -> Result<(), String> {
        let key = self.key();
        library::keep_early_list();
        // Without the run's log lines per shader and pipeline: tens of
        // thousands of them would bury the pre-build's own report.
        let quiet = tracing::Dispatch::none();
        let made = tracing::dispatcher::with_default(&quiet, || {
            host::HostPipeline::new(&self.shaders, self.graphics.as_deref())
        });
        #[cfg(feature = "diagnostics")]
        make::log_built(&key);
        if made.is_some() {
            // shortcut: a pipeline the library refused to store is listed
            // too; a run then makes it at first use.
            library::note_prebuilt(key);
        }
        library::save(false);
        match made {
            Some(_) => Ok(()),
            None => Err(match self.shaders.iter().find(|s| s.translated().is_none()) {
                Some(s) => format!("{} shader does not translate", s.stage),
                None => "the host refuses it".into(),
            }),
        }
    }
}

/// Host pipelines made, and loaded from the library, since the start.
#[cfg(windows)]
pub fn pipeline_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (library::MADE.load(Relaxed), library::LOADED.load(Relaxed))
}

/// Whether a pre-build has put the title pipeline `key` in the library
/// (a second pre-build skips it without translating or loading anything).
#[cfg(windows)]
pub fn prebuilt(key: &str) -> bool {
    library::prebuilt(key)
}

/// Whether an earlier run drew or dispatched with the title pipeline `key`.
#[cfg(windows)]
pub fn wanted_early(key: &str) -> bool {
    library::wanted_early(key)
}

/// A descriptor changed after loading, on the native host's own path.
#[cfg(all(test, target_os = "macos"))]
mod changed_descriptor_tests {
    use super::*;

    /// The container made from a compute description's pieces (wave32, 32 threads, 5 user SGPRs,
    /// a raw UAV of the table at user data 4), translated with the host's options, is refused
    /// naming the dword the code changed after loading it and how: the text the refusal log line
    /// shows.
    #[test]
    fn the_host_refuses_a_descriptor_changed_after_loading_naming_the_dword() {
        let words = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let program = |change: &[u32]| {
            let mut code = vec![
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF408_0202,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            ];
            code.extend_from_slice(change);
            code.extend([
                0x7E02_0280, // v_mov_b32 v1, 0
                0xE070_0000,
                0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
                0xBF81_0000, // s_endpgm
            ]);
            let code = words(&code);
            // Thread-group size 32, 1, 1; wave32; 16 VGPRs; 5 user SGPRs.
            let registers = words(&[0, 0, 0, 32 | 1 << 16, 1 | 1 << 4 << 16, 1, 5 << 1]);
            let mut uav = [0u8; 32];
            uav[0] = 0x41;
            uav[4..6].copy_from_slice(&[3, 3]);
            uav[16] = 1;
            uav[20..23].copy_from_slice(&[4, 4, 0]);
            let mut metadata = words(&[0x403, 0, code.len() as u32 / 4, 0, 0]);
            metadata.extend([0x40, 4, 1, 0]);
            metadata.extend(uav);
            metadata.extend([0x43].into_iter().chain([0; 31]));
            let size = metadata.len() as u32 - 0x14;
            metadata[4..8].copy_from_slice(&size.to_le_bytes());
            let pieces = [registers, vec![0; 0x14], code, metadata];
            let mut record = Vec::new();
            for p in &pieces {
                record.extend((p.as_ptr() as u64).to_le_bytes());
                record.extend((p.len() as u64).to_le_bytes());
            }
            // SAFETY: four pairs over live vectors.
            unsafe { compute_container(None, record.as_ptr()) }.expect("a container")
        };
        let prefix = "a descriptor changed after loading: ";
        for (change, why) in [
            (&[0xBE8A_03C0][..], "dword 2 is a constant"), // s_mov_b32 s10, 64
            // s_add_u32 s8, s8, 16; s_addc_u32 s9, s9, 0
            (&[0x8008_9008, 0x8209_8009], "dword 0 added to (s_add, s_sub...)"),
        ] {
            let err = crate::native::shaders::translate(&program(change)).err().expect("refused");
            assert!(err.ends_with(&format!("{prefix}{why} is not supported")), "{err}");
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    /// A registered shader is translated without anyone calling `native()`.
    #[cfg(target_os = "macos")]
    #[test]
    fn registered_shaders_translate_ahead() {
        let s = shader("prewarm-test", b"not a container, prewarm test");
        let end = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while s.native.get().is_none() && std::time::Instant::now() < end {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(s.native.get().is_some_and(|r| r.is_err()));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn guessed_graphics_skips_vertexless_slots_and_lowers_depth_only() {
        let code = |words: &[u32]| -> Vec<u8> {
            words.iter().flat_map(|w| w.to_le_bytes()).collect()
        };
        let pixel = code(&[
            0x7E00_02F2, // v_mov_b32 v0, 1.0
            0xF800_180F,
            0, // exp mrt0 v0, v0, v0, v0 done vm
            0xBF81_0000, // s_endpgm
        ]);
        let vertex = code(&[
            0x7E00_02F2, // v_mov_b32 v0, 1.0
            0xF800_08CF,
            0, // exp pos0 v0, v0, v0, v0 done
            0xBF81_0000, // s_endpgm
        ]);
        let meta = [0u8; 0x10];
        let piece = |bytes: &[u8]| [bytes.as_ptr() as usize, bytes.len()];
        let read = |pairs: &[[usize; 2]; 8]| {
            // SAFETY: pairs covers the slot's eight entries and the live slices above back them.
            unsafe { graphics_containers(pairs.as_ptr().cast()) }
        };
        let mut pairs = [[0; 2]; 8];
        pairs[2] = piece(&pixel);
        pairs[3] = piece(&meta);
        assert!(read(&pairs).unwrap().is_empty());
        pairs[4] = piece(&vertex);
        assert_eq!(read(&pairs).err(), Some("missing or invalid vertex code or metadata"));
        pairs[2] = [0; 2];
        pairs[3] = [0; 2];
        pairs[5] = piece(&meta);
        let depth_only = read(&pairs).unwrap();
        assert_eq!(depth_only.len(), 1);
        assert_eq!(depth_only[0].native().unwrap().stage, victory_shader::HwStage::Vs);
    }

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.lock().unwrap().clone()))
            .collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    #[cfg(windows)]
    #[test]
    fn translation_lets_the_container_go() {
        // Not a container: it does not translate, and goes all the same.
        let s = Shader::new("compute", &[1, 2, 3]);
        assert!(s.translated().is_none());
        assert!(s.container.lock().unwrap().is_empty());
        assert_eq!(s.size, 3);
    }

    /// The stream layout Grand Theft Auto VI builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }

    /// The pre-build reads a stream as the title's creation does: the same
    /// shaders and state, so the same key and the same host pipeline name.
    #[cfg(windows)]
    #[test]
    fn a_prebuild_is_the_pipeline_the_title_creates() {
        let (vs, ps) = ([7u8; 9], [5u8; 6]);
        let name = c"TEXCOORD";
        // One input element: name, index 2, format 16, slot 1, offset 8.
        let mut element = [0u8; 32];
        element[..8].copy_from_slice(&(name.as_ptr() as u64).to_le_bytes());
        for (i, w) in [2u32, 16, 1, 8, 0, 0].iter().enumerate() {
            element[8 + 4 * i..12 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }
        let mut stream = vec![0u8; 0x2b0];
        let put = |s: &mut Vec<u8>, at: usize, v: u64| {
            s[at..at + 8].copy_from_slice(&v.to_le_bytes())
        };
        let kinds = [(0x10, 1), (0x28, 2), (0x40, 3), (0x58, 4), (0x70, 5), (0x88, 8)];
        for (at, kind) in kinds.into_iter().chain([(0x210, 21), (0x250, 17)]) {
            put(&mut stream, at, kind);
        }
        put(&mut stream, 0x18, vs.as_ptr() as u64);
        put(&mut stream, 0x20, vs.len() as u64);
        put(&mut stream, 0x30, ps.as_ptr() as u64);
        put(&mut stream, 0x38, ps.len() as u64);
        put(&mut stream, 0x8c, 1 << 32); // independent blend
        put(&mut stream, 0x1d8, 9 | 0xffff_ffff << 32);
        put(&mut stream, 0x1e0, 10 | 3 << 32); // solid fill
        put(&mut stream, 0x254, 1);
        put(&mut stream, 0x260, 12);
        put(&mut stream, 0x268, element.as_ptr() as u64);
        put(&mut stream, 0x270, 1);
        put(&mut stream, 0x278, 14 | 4 << 32);
        put(&mut stream, 0x280, 15 | 87 << 32);
        put(&mut stream, 0x2a0, 1 << 32);
        put(&mut stream, 0x2a8, 16 | 40 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        assert_eq!(create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p), S_OK);
        // SAFETY: the stream points at live code and elements.
        let pre = unsafe { Prebuild::graphics(&stream) };
        let title = pipeline(p);
        assert_eq!(pre.key(), title.host.key);
        let state = title.host.graphics.as_deref().unwrap().describe();
        assert!(state.contains(r#"\"TEXCOORD\" 2 16 1 8 0 0"#), "{state}");
        assert_eq!(pre.describe(), format!("{:?}\t{state}", title.shaders));
        // Another state is another pipeline.
        put(&mut stream, 0x1e0, 10 | 2 << 32);
        // SAFETY: as above.
        assert_ne!(unsafe { Prebuild::graphics(&stream) }.key(), title.host.key);
    }
}

/// The container the host makes from a compute description's pieces: wave32 code of 32 threads
/// with `user_sgprs` user SGPRs and `records` (32-byte `0x41` records), as the title passes them.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn test_compute_container(code: &[u32], records: &[[u8; 32]], user_sgprs: u32) -> Vec<u8> {
    let words = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    // Thread-group size 32, 1, 1; wave32; 16 VGPRs.
    let registers = words(&[0, 0, 0, 32 | 1 << 16, 1 | 1 << 4 << 16, 1, user_sgprs << 1]);
    let code = words(code);
    let mut metadata = words(&[0x403, 0, code.len() as u32 / 4, 0, 0]);
    metadata.extend([0x40, 4, records.len() as u8, 0]);
    records.iter().for_each(|r| metadata.extend(r));
    metadata.extend([0x43].into_iter().chain([0; 31]));
    let size = metadata.len() as u32 - 0x14;
    metadata[4..8].copy_from_slice(&size.to_le_bytes());
    let pieces = [registers, vec![0; 0x14], code, metadata];
    let mut record = Vec::new();
    for p in &pieces {
        record.extend((p.as_ptr() as u64).to_le_bytes());
        record.extend((p.len() as u64).to_le_bytes());
    }
    // SAFETY: four pairs over live vectors.
    unsafe { compute_container(None, record.as_ptr()) }.expect("a container")
}

#[cfg(all(test, target_os = "macos"))]
mod pieces_tests {
    use super::*;

    fn container_of(code: &[u32], records: &[[u8; 32]]) -> Vec<u8> {
        test_compute_container(code, records, 5)
    }

    /// On the native host's path (the container made from the title's pieces, translated with
    /// the host's options: two host waves of 32 lanes, title memory zero-copy), a whole-wave
    /// section value without a shape is refused naming the first step the simulation does not
    /// model; that text is what the refusal log line shows.
    #[test]
    fn the_host_refuses_a_whole_wave_section_naming_its_step() {
        // A raw UAV of the table at user data 4 (`s4`).
        let mut uav = [0u8; 32];
        uav[0] = 0x41;
        uav[4..6].copy_from_slice(&[3, 3]);
        uav[16] = 1;
        uav[20..23].copy_from_slice(&[4, 4, 0]);
        let program = |step: &[u32]| {
            let mut code = vec![
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF408_0202,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
                0x7E02_0280, // v_mov_b32 v1, 0
                0xBE82_3DC1, // s_or_saveexec_b32 s2, -1
            ];
            code.extend_from_slice(step);
            code.extend([
                0xBEFE_0302, // s_mov_b32 exec_lo, s2
                0xE070_0000,
                0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
                0xBF81_0000, // s_endpgm
            ]);
            container_of(&code, &[uav])
        };
        let prefix = "a whole-wave section step the simulation does not follow: ";
        for (step, why) in [
            (&[0x0802_0100][..], "a VALU operation it does not model"), // v_sub_f32 v1, v0, v0
            // v_mov_b32_dpp v1, v0 row_shr:1: each lane its neighbour's value.
            (&[0x7E02_02FA, 0xFF01_1100], "a value that is no reduction or prefix"),
        ] {
            let err = crate::native::shaders::translate(&program(step)).err().expect("refused");
            assert!(err.ends_with(&format!("{prefix}{why} is not supported")), "{err}");
        }
    }

    /// Four (pointer, size) pairs of a console compute description's record, as the title lays
    /// them out: registers, hash, code, metadata.
    #[test]
    fn a_record_becomes_an_fspx_part_in_a_container() {
        let pieces: [Vec<u8>; 4] = [vec![1; 0x1c], vec![2; 0x14], vec![3; 0x40], vec![4; 0x60]];
        let mut record = Vec::new();
        for p in &pieces {
            record.extend((p.as_ptr() as u64).to_le_bytes());
            record.extend((p.len() as u64).to_le_bytes());
        }
        let root = [5u8; 0x24];
        // SAFETY: four pairs over live vectors.
        let container =
            unsafe { compute_container(Some(&root), record.as_ptr()) }.expect("a container");
        let parsed = victory_shader::Container::parse(&container).unwrap();
        let part = parsed.part(victory_shader::FourCC::ACCC).expect("an ACCC part").data;
        let word = |at: usize| u32::from_le_bytes(part[at..at + 4].try_into().unwrap()) as usize;
        assert_eq!(&part[..4], b"FSPX");
        assert_eq!((word(4), word(8), word(12)), (1, 5, part.len()));
        let rts0 = victory_shader::native::FSPX_ROOT_SIGNATURE;
        let all = pieces.iter().map(Vec::as_slice).chain([&root[..]]);
        for (i, (kind, piece)) in [0u32, 1, 6, 7, rts0].into_iter().zip(all).enumerate() {
            let at = 0x10 + 16 * i;
            assert_eq!(word(at), kind as usize);
            let (offset, size) = (word(at + 4), word(at + 8));
            assert_eq!(&part[offset..offset + size], piece);
        }
        // Without a root signature, the four pieces alone.
        // SAFETY: as above.
        let bare = unsafe { compute_container(None, record.as_ptr()) }.expect("a container");
        let parsed = victory_shader::Container::parse(&bare).unwrap();
        let part = parsed.part(victory_shader::FourCC::ACCC).expect("an ACCC part").data;
        assert_eq!(u32::from_le_bytes(part[8..12].try_into().unwrap()), 4);
        // A missing piece is no container.
        record[16..24].copy_from_slice(&0u64.to_le_bytes());
        // SAFETY: as above.
        assert!(unsafe { compute_container(Some(&root), record.as_ptr()) }.is_none());
    }

    /// On the native host's path (the container made from the title's pieces, translated with
    /// the host's options), a buffer descriptor's dword that is no size, read as data, is refused
    /// naming the dword; that text is what the refusal log line shows.
    #[test]
    fn the_host_refuses_a_descriptor_dword_read_as_data_naming_it() {
        // A table at user data 4 (`s4`): a raw UAV at dword 0, the buffer read at dword 8.
        let table = |kind: u8, offset: u8| {
            let mut r = [0u8; 32];
            r[0] = 0x41;
            r[4..6].copy_from_slice(&[kind, 3]);
            r[8] = offset;
            r[16] = 1;
            r[20..23].copy_from_slice(&[4, 4, 0]);
            r
        };
        let program = |read: u32| {
            let code = [
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF408_0202,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
                0xF408_0302,
                0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                0xE030_1000,
                0x8003_0100, // buffer_load_dword v1, v0, s[12:15], 0 offen
                read,        // v_mov_b32 v2, s<12 + n>
                0xBF8C_3F70, // s_waitcnt vmcnt(0)
                0x4A02_0501, // v_add_nc_u32 v1, v1, v2
                0xE070_0000,
                0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
                0xBF81_0000, // s_endpgm
            ];
            container_of(&code, &[table(3, 0), table(0, 8)])
        };
        let prefix = "a descriptor's address or format read as data: ";
        for (read, why) in [
            (0x7E04_020C, "a buffer's base address (dword 0)"),
            (0x7E04_020D, "a buffer's dword 1 (address high bits, stride, swizzle)"),
            (0x7E04_020F, "a buffer's dword 3 (format and type)"),
        ] {
            let err = crate::native::shaders::translate(&program(read)).err().expect("refused");
            assert!(err.ends_with(&format!("{prefix}{why} is not supported")), "{err}");
        }
    }

    /// A dispatch of thread counts needs a second translation, after the first let the container go:
    /// it is read again where the title held it, and only if it still is the shader.
    #[test]
    fn a_compute_shader_is_translated_again_for_thread_counts() {
        let counts = |h: &victory_shader::emit::HostShader| {
            h.user_data.iter().filter(|u| u.slot == victory_shader::emit::DISPATCH_THREADS).count()
        };
        let release = |p| crate::object::slot::<extern "system" fn(This) -> u32>(p, 0x10)(p);
        // `v_mov_b32 v1, <n>; s_endpgm`, a shader of its own each time.
        let container = |n: u32| container_of(&[0x7E02_0280 + n, 0xBF81_0000], &[]);

        let mut held = container(1);
        let p = test_compute(&held);
        let shader = pipeline(p).shaders[0].clone();
        let plain = shader.native().expect("translates").clone();
        assert!(shader.container.lock().unwrap().is_empty());
        let (hash, again) = native_threads(p);
        assert_eq!(hash, shader.hash);
        assert_eq!((counts(&plain), counts(&again.expect("translated again"))), (0, 1), "the count in x");
        release(p);

        // The title wrote over it since: not the shader any more.
        held = container(2);
        let p = test_compute(&held);
        *held.last_mut().unwrap() ^= 1;
        assert_eq!(native_threads(p).1.err().as_deref(), Some("its container changed since the pipeline was made"));
        release(p);

        // The title let it go (an address nothing is mapped at).
        held = container(3);
        let p = test_compute(&held);
        pipeline(p).shaders[0].at.store(0x10, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(native_threads(p).1.err().as_deref(), Some("its container is no longer in title memory"));
        release(p);
    }
}
