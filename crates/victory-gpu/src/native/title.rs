//! What the title-facing objects (`queue.rs`, `fence.rs`, `list.rs`) need from the native host,
//! generic over the backend: queues and fences by the title's object address, title lists cut at
//! memory-value waits, and `hold.rs`'s per-queue FIFO of held work in front of
//! [`NativeHost::execute`]. The macOS glue (`glue.rs`) instantiates it once with Metal.
//!
//! Ordering. Work on one title queue is one backend queue, so submissions run in the order they
//! were made. A list segment that waits for a memory value another queue's list writes stays on
//! the CPU side until the value is there (`hold.rs`; the GPU's own writes into title memory are
//! visible to the CPU as they land, there is no copy); everything else goes to the GPU at once and
//! is never waited for. Fence signals that need no CPU decision are encoded on the GPU
//! (`MTLSharedEvent`); behind held work they are held too, so that they stay in order. Fence
//! waits stay on the CPU until the backend reports completion. A committed producer can be
//! delayed or fail asynchronously; submission is not proof that an event will be signalled.

use super::host::NativeHost;
use super::list::{Binding, ComputePso, NativeList};
use super::memory::PoolTable;
use crate::hold::{Held, Item, ValueWait, World};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use victory_backend::resources::{Format, GraphicsBackend, PresentBackend};

/// A wait for a memory value (`Wait32BitValueX`, `Wait64BitValueX`): the value at `address`
/// (`bytes` 4 or 8, masked) compared with `reference` by a D3D12 comparison function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wait {
    pub address: u64,
    pub bytes: u8,
    pub comparison: u32,
    pub reference: u64,
    pub mask: u64,
}

impl From<Wait> for ValueWait {
    fn from(w: Wait) -> ValueWait {
        ValueWait {
            address: w.address,
            bytes: w.bytes,
            comparison: w.comparison,
            reference: w.reference,
            mask: w.mask,
        }
    }
}

/// A title list: the recording [`NativeList`] plus the segments cut off it at waits recorded after
/// commands, each with the waits at its start.
pub struct TitleList<B: GraphicsBackend> {
    done: Vec<(NativeList<B>, Vec<Wait>)>,
    /// The list the calls record into, and the waits at its start.
    pub list: NativeList<B>,
    waits: Vec<Wait>,
    /// Waits recorded: the ones that cut a segment.
    pub cuts: u32,
    /// The title pipeline object `SetPipelineState` named last (0: none), and the primitive
    /// topology set (the console's `D3D_PRIMITIVE_TOPOLOGY`).
    pub pipeline: usize,
    pub topology: u32,
    /// The formats of the targets `OMSetRenderTargets` bound last.
    pub targets: TargetFormats,
    /// Whether every target `OMSetRenderTargets` bound last has a host texture. When one has
    /// none the binding is refused (a partial one would renumber the slots) and draws are
    /// dropped until the title binds a set the host can make.
    pub targets_ready: bool,
    /// The strides of the bound vertex buffers by slot, which a guessed pipeline is made from at
    /// its draw, and the title resources bound as targets (for the target trace).
    pub strides: Vec<u32>,
    pub target_keys: Vec<usize>,
}

/// The formats of a draw's bound colour and depth targets.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TargetFormats {
    pub color: Vec<Format>,
    pub depth: Option<Format>,
}

impl<B: GraphicsBackend> Default for TitleList<B> {
    fn default() -> Self {
        TitleList {
            done: Vec::new(),
            list: NativeList::default(),
            waits: Vec::new(),
            cuts: 0,
            pipeline: 0,
            topology: 0,
            targets: TargetFormats::default(),
            targets_ready: true,
            strides: Vec::new(),
            target_keys: Vec::new(),
        }
    }
}

impl<B: GraphicsBackend> TitleList<B> {
    /// `Reset`.
    pub fn reset(&mut self) {
        *self = TitleList::default();
    }

    /// A wait recorded at the current position: at the start of the segment if it has no commands
    /// yet, else the segment ends here and the wait starts the next one (state carries over).
    pub fn wait(&mut self, w: Wait) {
        if !self.list.is_empty() {
            let segment = self.list.cut();
            self.done
                .push((segment, std::mem::replace(&mut self.waits, vec![w])));
            self.cuts += 1;
        } else {
            self.waits.push(w);
        }
    }

    /// The segments a submission runs, in order (copies: the title may reset and re-record).
    pub fn segments(&self) -> Vec<(NativeList<B>, Vec<Wait>)> {
        self.done
            .iter()
            .map(|(l, w)| (l.snapshot(), w.clone()))
            .chain([(self.list.snapshot(), self.waits.clone())])
            .collect()
    }
}

/// Counters for the frame line and the tests.
#[derive(Default, Debug, Clone)]
pub struct Stats {
    /// Segments queued by `ExecuteCommandLists`.
    pub segments: u64,
    /// Groups of segments handed to the GPU (a piece is what `hold` runs together).
    pub pieces: u64,
    /// Pieces the host refused (an address outside the mapped pools, ...): logged, their fences
    /// are signalled all the same.
    pub refused: u64,
    /// Segments that waited for a memory value before they ran.
    pub waited: u64,
    /// Title textures detiled (again) because their memory changed.
    pub detiles: u64,
    /// Textures whose layout the host cannot detile (counted once per content).
    pub undetilable: u64,
}

/// What a title pipeline object became on the host.
pub enum Pipe<B: GraphicsBackend> {
    Compute(Rc<ComputePso<B>>),
    /// With the argument layout of its translated shaders (none: the title's root as it is).
    Render(Rc<B::RenderPipeline>, Option<Rc<Binding>>),
    /// A guessed graphics pipeline (`VICTORY_GUESS_GRAPHICS`): made, per target formats and
    /// vertex strides, at its draws.
    Guess(Rc<Guessed<B>>),
    /// A guessed graphics slot without a vertex producer.
    #[cfg(target_os = "macos")]
    Skipped,
    /// The host refused it (the reason is logged and counted once); work using it is dropped.
    Refused,
}

impl<B: GraphicsBackend> Clone for Pipe<B> {
    fn clone(&self) -> Self {
        match self {
            Pipe::Compute(p) => Pipe::Compute(p.clone()),
            Pipe::Render(p, b) => Pipe::Render(p.clone(), b.clone()),
            Pipe::Guess(g) => Pipe::Guess(g.clone()),
            #[cfg(target_os = "macos")]
            Pipe::Skipped => Pipe::Skipped,
            Pipe::Refused => Pipe::Refused,
        }
    }
}

/// The shaders of a guessed pipeline and the host pipelines made from them so far.
pub struct Guessed<B: GraphicsBackend> {
    pub vs: Arc<victory_shader::emit::HostShader>,
    pub ps: Option<Arc<victory_shader::emit::HostShader>>,
    pub name: String,
    /// The pipeline's state with the targets of its draw ([`Graphics::for_draw`] completes it).
    pub state: super::state::Graphics,
    /// The vertex shader's container, translated again for rectangle lists at the first such draw.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) vs_source: Arc<crate::pipeline::Shader>,
    /// The pixel shader's container, translated again for draws into unsigned-integer targets.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) ps_source: Option<Arc<crate::pipeline::Shader>>,
    /// By the formats and strides they were made for.
    pub made: std::cell::RefCell<HashMap<String, Pipe<B>>>,
}

/// An operation that must keep its place between list submissions in the held FIFO.
enum QueuedAction<B: PresentBackend> {
    Signal(Rc<B::Queue>, Rc<B::Fence>, u64),
    Present(Rc<B::Queue>, Rc<B::Surface>, Rc<B::Texture>),
}

pub struct Title<B: PresentBackend> {
    pub host: NativeHost<B>,
    /// Host pipelines by title pipeline object, made at first use.
    pub pipelines: HashMap<usize, Pipe<B>>,
    /// Host pipelines of the console's pipelines (whose state names no target formats) by title
    /// pipeline object and the targets of the draw ([`super::console_state`]).
    pub console_pipes: HashMap<(usize, TargetFormats), Pipe<B>>,
    /// Host pipelines of compute pipelines dispatched with thread counts (`DispatchX`), whose
    /// shader is translated for that, by title pipeline object.
    pub thread_pipes: HashMap<usize, Pipe<B>>,
    /// What the shader path did, and refused.
    pub shaders: super::shaders::ShaderStats,
    /// How long a new pipeline may hold the frame while Metal compiles it (`glue::compile_wait`);
    /// past it the pipeline is put off (`ShaderStats::put_off`).
    pub compile_wait: std::time::Duration,
    pub(super) libraries: super::shaders::Libraries<B>,
    pub(super) textures: super::textures::Textures<B>,
    /// GPU-written textures waiting to be copied back into title memory.
    pub(super) back: super::textures::Back<B>,
    /// Sampled textures realized but not yet uploaded once.
    pub(super) unloaded: std::collections::HashSet<usize>,
    /// Keys of textures currently in the fast per-submission probe set (`hot > 0`).
    pub(super) hot_keys: Vec<usize>,
    /// Round-robin cursor into `hot_keys` when the time budget ends before all are probed.
    pub(super) hot_cursor: usize,
    /// Keys of textures whose last upload was blank, probed after `hot_keys` until written.
    pub(super) blank_keys: Vec<usize>,
    pub(super) blank_cursor: usize,
    /// Reusable scratch buffer for hot-texture probes (avoids per-submission allocations).
    pub(super) probe_buf: Vec<u8>,
    /// Where `Present` shows frames, if the app gave a layer.
    pub(super) surface: Option<Rc<B::Surface>>,
    /// Title textures no view has named yet.
    pub(super) pending: HashMap<usize, super::textures::TextureInfo>,
    /// Textures placed in title memory (no resource object), by address: their key and pool.
    pub(super) placed: HashMap<u64, (usize, usize)>,
    queues: HashMap<usize, Rc<B::Queue>>,
    fences: HashMap<usize, Rc<B::Fence>>,
    held: Held<Rc<NativeList<B>>>,
    /// Operations held behind work, indexed by the token in a FIFO Signal item.
    actions: HashMap<usize, QueuedAction<B>>,
    next_action: usize,
    /// The highest value each fence was told to reach, and when that changed (diagnostics).
    requested: HashMap<usize, (u64, std::time::Instant)>,
    /// Unique wait tokens retain fence instances independently of title address reuse.
    waiting: HashMap<usize, Rc<B::Fence>>,
    pub stats: Stats,
}

impl<B: PresentBackend> Title<B> {
    pub fn new(backend: B, memory: Arc<PoolTable>) -> Self {
        let host = NativeHost::with_memory(backend, memory);
        host.collect_written(std::env::var("VICTORY_TEXTURE_BACK").as_deref() == Ok("1"));
        Title {
            host,
            pipelines: HashMap::new(),
            console_pipes: HashMap::new(),
            thread_pipes: HashMap::new(),
            shaders: Default::default(),
            compile_wait: std::time::Duration::MAX,
            libraries: Default::default(),
            textures: Default::default(),
            back: Default::default(),
            unloaded: std::collections::HashSet::new(),
            hot_keys: Vec::new(),
            hot_cursor: 0,
            blank_keys: Vec::new(),
            blank_cursor: 0,
            probe_buf: Vec::new(),
            surface: None,
            pending: HashMap::new(),
            placed: HashMap::new(),
            queues: HashMap::new(),
            fences: HashMap::new(),
            requested: HashMap::new(),
            waiting: HashMap::new(),
            held: Held::new(),
            actions: HashMap::new(),
            next_action: 0,
            stats: Stats::default(),
        }
    }

    fn queue(&mut self, key: usize) -> Result<Rc<B::Queue>, String> {
        if let Some(q) = self.queues.get(&key) {
            return Ok(q.clone());
        }
        let q = Rc::new(self.host.backend.create_queue()?);
        self.queues.insert(key, q.clone());
        Ok(q)
    }

    /// The backend fence of title fence `key`, made with `initial` the first time.
    pub fn fence(&mut self, key: usize, initial: u64) -> Result<Rc<B::Fence>, String> {
        if let Some(f) = self.fences.get(&key) {
            return Ok(f.clone());
        }
        let f = Rc::new(self.host.backend.create_fence(initial)?);
        self.fences.insert(key, f.clone());
        Ok(f)
    }

    /// The title released fence `key`.
    pub fn forget_fence(&mut self, key: usize) {
        self.fences.remove(&key);
        self.requested.remove(&key);
    }

    /// The title released queue `key`; held work stays held (the title does not do it).
    pub fn forget_queue(&mut self, key: usize) {
        self.queues.remove(&key);
    }

    /// Every title fence with a backend fence and the value it has completed: what the title's own
    /// copy of the value (read from memory) is raised to.
    pub fn completed(&mut self) -> Vec<(usize, u64)> {
        let values = self
            .fences
            .iter()
            .map(|(k, f)| (*k, self.host.backend.completed(f)))
            .collect();
        // After the values are read: the work they cover is done, so what it wrote is in title
        // memory before the title sees them.
        self.copy_back();
        values
    }

    /// Fence `key`'s completed value (made with `initial` if it has no backend fence yet).
    pub fn fence_value(&mut self, key: usize, initial: u64) -> u64 {
        let value = match self.fence(key, initial) {
            Ok(f) => self.host.backend.completed(&f).max(initial),
            Err(_) => initial,
        };
        // As in `completed`: read first, then copy back.
        self.copy_back();
        value
    }

    /// `ID3D12Fence::Signal`: the value is reached now; queues waiting for it go on.
    pub fn signal_cpu(&mut self, key: usize, initial: u64, value: u64) {
        if let Ok(f) = self.fence(key, initial) {
            self.host.backend.signal(&f, value);
        }
        self.pump();
    }

    /// Whether fence `key` reached `value`; pumps held work first, so that a wait for a fence
    /// signalled behind a memory-value wait can make progress when the caller polls.
    pub fn try_wait(&mut self, key: usize, initial: u64, value: u64) -> bool {
        self.pump();
        self.fence_value(key, initial) >= value
    }

    /// `ExecuteCommandLists` on title queue `queue`: the lists' segments join what the queue holds
    /// and whatever may run goes to the GPU.
    pub fn submit(&mut self, queue: usize, segments: Vec<(NativeList<B>, Vec<Wait>)>) {
        self.sync_dirty();
        self.submit_synced(queue, segments);
    }

    /// [`Self::submit`] once the textures are in step with title memory ([`Self::sync_dirty`]).
    pub fn submit_synced(&mut self, queue: usize, segments: Vec<(NativeList<B>, Vec<Wait>)>) {
        for (list, waits) in segments {
            self.stats.segments += 1;
            let waits: Vec<ValueWait> = waits.into_iter().map(Into::into).collect();
            self.held.push(queue, Item::Segment(Rc::new(list), waits));
        }
        self.pump();
    }

    /// `ID3D12CommandQueue::Signal`: behind work the queue still holds, else on the GPU at once
    /// (after the work already submitted to its queue).
    pub fn signal_queue(&mut self, queue: usize, fence: usize, initial: u64, value: u64) {
        let r = self.requested.entry(fence).or_insert((0, std::time::Instant::now()));
        if value > r.0 {
            *r = (value, std::time::Instant::now());
        }
        if self.held.holds_work(queue) {
            let n = self.next_action;
            self.next_action += 1;
            let (Ok(q), Ok(f)) = (self.queue(queue), self.fence(fence, initial)) else { return };
            self.actions.insert(n, QueuedAction::Signal(q, f, value));
            self.held.push(queue, Item::Signal(n, 0));
        } else {
            self.signal_gpu(queue, fence, initial, value);
        }
        self.pump();
    }

    fn signal_gpu(&mut self, queue: usize, fence: usize, initial: u64, value: u64) {
        let (Ok(q), Ok(f)) = (self.queue(queue), self.fence(fence, initial)) else {
            return;
        };
        if let Err(e) = self.host.execute(&q, &[], &[], &[(&*f, value)]) {
            tracing::warn!(error = %e, "a fence signal was not queued");
        }
    }

    /// Queue waits remain on the CPU until completion, including committed producers.
    pub fn wait_queue(&mut self, queue: usize, fence: usize, initial: u64, value: u64) {
        let Ok(f) = self.fence(fence, initial) else { return };
        if self.held.holds_work(queue) || self.host.backend.completed(&f) < value {
            let token = self.next_action;
            self.next_action += 1;
            self.waiting.insert(token, f);
            self.held.push(queue, Item::FenceWait(token, value));
        }
        self.pump();
    }

    /// Runs the heads of held queues that may run, until none can.
    pub fn pump(&mut self) {
        if self.held.is_empty() {
            return;
        }
        let Title {
            host,
            queues,
            waiting,
            held,
            actions,
            stats,
            back,
            ..
        } = self;
        let memory = host.memory();
        let read = |address: u64, bytes: u8| read_title(&memory, address, bytes);
        let fence = |f: usize| {
            waiting
                .get(&f)
                .map_or(0, |fence| host.backend.completed(fence))
        };
        let mut run = |queue: usize, lists: Vec<Rc<NativeList<B>>>| {
            stats.pieces += 1;
            let Some(q) = queues.get(&queue).cloned() else {
                return;
            };
            let refs: Vec<&NativeList<B>> = lists.iter().map(|l| &**l).collect();
            let ours = back.next(host, queue);
            let signals: Vec<(&B::Fence, u64)> = ours.iter().map(|(f, v)| (&**f, *v)).collect();
            if let Err(e) = host.execute(&q, &refs, &[], &signals) {
                stats.refused += 1;
                tracing::warn!(error = %e, "a piece of title work was refused");
            }
            back.submitted(host, ours);
        };
        let mut signal = |n: usize, _: u64| {
            match actions.get(&n) {
                Some(QueuedAction::Signal(q, f, value)) => {
                    if let Err(e) = host.execute(q, &[], &[], &[(&**f, *value)]) {
                        tracing::warn!(error = %e, "a held fence signal was not queued");
                    }
                }
                Some(QueuedAction::Present(queue, surface, texture)) => {
                    match host.backend.try_present(queue, surface, texture) {
                        Ok(false) => return false,
                        Ok(true) => {}
                        Err(e) => tracing::warn!(error = %e, "a held present was not queued"),
                    }
                }
                None => {}
            }
            actions.remove(&n);
            true
        };
        let done = held.pump(&mut World {
            read: &read,
            fence: &fence,
            run: &mut run,
            signal: &mut signal,
        });
        stats.waited += done.waited.len() as u64;
        for token in done.retired_waits {
            waiting.remove(&token);
        }
    }

    /// The backend queue of each title queue made so far.
    pub fn backend_queues(&self) -> impl Iterator<Item = (usize, &B::Queue)> {
        self.queues.iter().map(|(k, q)| (*k, &**q))
    }

    /// Queues whose head has been held longer than `limit`: a line each, with the waits that do
    /// not hold (address, comparison, reference, mask, current value).
    pub fn stuck(&self, limit: std::time::Duration) -> Vec<String> {
        let mut out: Vec<String> = self
            .requested
            .iter()
            .filter_map(|(k, &(want, since))| {
                let f = self.fences.get(k)?;
                let done = self.host.backend.completed(f);
                (done < want && since.elapsed() >= limit)
                    .then(|| format!("fence {k:#x}: signal of {want} queued {:.1?} ago, GPU at {done}", since.elapsed()))
            })
            .collect();
        let memory = self.host.memory();
        let read = |address: u64, bytes: u8| read_title(&memory, address, bytes);
        out.extend(self
            .held
            .blocked(&read)
            .into_iter()
            .filter(|(.., since, _)| *since >= limit)
            .map(|(q, items, since, unmet)| {
                let waits: Vec<String> = unmet
                    .iter()
                    .map(|w| {
                        format!(
                            "{:#x} cmp {} ref {:#x} mask {:#x} now {:?}",
                            w.address,
                            w.comparison,
                            w.reference,
                            w.mask,
                            read(w.address, w.bytes)
                        )
                    })
                    .collect();
                format!("queue {q:#x}: {items} held, head for {since:.1?}, waits {waits:?}")
            }));
        out
    }

    /// Gives the host the app's drawable layer (`CAMetalLayer` on Metal, by address): frames the
    /// title presents are shown there from now on.
    pub fn set_surface(&mut self, layer: *mut std::ffi::c_void) -> Result<(), String> {
        self.surface = Some(Rc::new(self.host.backend.create_surface(layer)?));
        Ok(())
    }

    /// `PresentX`: show title texture `key` after all earlier work on `queue`, including held
    /// work. `Ok(true)` means accepted (possibly deferred); deferred errors are logged by pump.
    /// `Ok(false)` when there is no surface or the title has no such texture.
    pub fn present(&mut self, queue: usize, key: usize) -> Result<bool, String> {
        self.pump();
        if self.surface.is_none() {
            return Ok(false);
        }
        let Some(tex) = self.texture_of(key) else {
            return Ok(false);
        };
        let q = self.queue(queue)?;
        if let Some(surface) = &self.surface {
            let n = self.next_action;
            self.next_action += 1;
            self.actions.insert(n, QueuedAction::Present(q, surface.clone(), tex));
            self.held.push(queue, Item::Signal(n, 0));
        }
        self.pump();
        Ok(true)
    }

    /// The surface presented to (to read the frame counter, say).
    pub fn surface(&self) -> Option<&B::Surface> {
        self.surface.as_deref()
    }

    /// Whether any queue holds work (a timer should test it again soon).
    pub fn holds(&self) -> bool {
        !self.held.is_empty()
    }

    /// Makes sure title queue `key` has a backend queue before work is held for it.
    pub fn ensure_queue(&mut self, key: usize) -> Result<(), String> {
        self.queue(key).map(|_| ())
    }
}

/// The value a wait tests: from the pool when the address lies in one, else read from the
/// process's own memory (which may fault-free fail: the read is checked by the kernel).
fn read_title(memory: &PoolTable, address: u64, bytes: u8) -> Option<u64> {
    let n = usize::from(bytes.min(8));
    let at = match memory.resolve(address, n as u64) {
        Some((_, p)) => p,
        None => address as *const u8,
    };
    let mut v = [0u8; 8];
    crate::native::safe_read(at, &mut v[..n]).then(|| u64::from_le_bytes(v))
}

#[cfg(all(test, target_os = "macos"))]
mod scheduling_tests {
    use super::*;
    use victory_backend::backend::ComputeBackend;
    use victory_metal::metal::Metal;

    #[test]
    fn signalling_a_fence_releases_all_its_queued_wait_tokens() {
        let Ok(metal) = Metal::new() else {
            eprintln!("skipping fence test: no Metal device");
            return;
        };
        let mut title = Title::new(metal, Arc::new(PoolTable::default()));
        for _ in 0..1024 {
            title.wait_queue(1, 10, 0, 1);
        }
        title.wait_queue(2, 20, 0, 1);
        title.pump();
        title.pump();
        assert_eq!(title.waiting.len(), 1025);

        title.signal_cpu(10, 0, 1);
        assert!(!title.held.holds_work(1));
        assert!(title.held.holds_work(2));
        assert_eq!(title.waiting.len(), 1);
        title.signal_cpu(20, 0, 1);
        assert!(title.waiting.is_empty());
    }

    #[test]
    fn a_wait_retains_its_fence_across_release_and_address_reuse() {
        let Ok(metal) = Metal::new() else { return };
        let mut title = Title::new(metal, Arc::new(PoolTable::default()));
        title.ensure_queue(1).unwrap();
        let old = title.fence(10, 0).unwrap();
        let weak = Rc::downgrade(&old);
        title.wait_queue(1, 10, 0, 5);
        title.forget_fence(10);
        drop(old);
        assert!(weak.upgrade().is_some(), "queued wait owns the backend fence");
        title.signal_cpu(10, 0, 5); // A new fence at the same title address.
        assert!(title.held.holds_work(1), "the replacement must not satisfy the old wait");
        title.host.backend.signal(&weak.upgrade().unwrap(), 5);
        title.pump();
        assert!(!title.held.holds_work(1));
        assert!(weak.upgrade().is_none(), "completed waits release their retained fences");
    }

    #[test]
    fn a_committed_producer_does_not_put_an_unsatisfied_wait_on_metal() {
        let Ok(metal) = Metal::new() else { return };
        let mut title = Title::new(metal, Arc::new(PoolTable::default()));
        let producer = title.queue(1).unwrap();
        title.ensure_queue(2).unwrap();
        let gate = title.fence(10, 0).unwrap();
        // Deliberately stall a backend producer. A title wait must stay CPU-side even after
        // its signal commits; the same rule applies if this producer subsequently fails.
        title.host.execute(&producer, &[], &[(&*gate, 1)], &[]).unwrap();
        title.signal_queue(1, 20, 0, 7);
        title.wait_queue(2, 20, 0, 7);
        assert!(title.held.holds_work(2));
        title.host.backend.signal(&gate, 1);
        let start = std::time::Instant::now();
        while title.held.holds_work(2) {
            assert!(start.elapsed() < std::time::Duration::from_secs(3));
            title.pump();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}
