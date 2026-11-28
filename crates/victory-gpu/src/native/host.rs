//! The native host: what `host.rs` is for D3D12, over any [`GraphicsBackend`].
//!
//! One thread drives it (backends hold Objective-C objects). Queues, fences, pipelines, descriptor
//! heaps and textures are made here; title lists are [`NativeList`]s; [`NativeHost::execute`] turns
//! lists into passes and submits them without waiting. Title memory is wrapped lazily, one buffer
//! per pool (see [`super::memory`]); nothing is copied to or from it.

use super::addr::{self, Mapping, Pages, Pairs};
use super::descriptors::{Decoded, Res, decode, typed_buffer};
use super::list::{Binding, Cmd, ComputePso, NativeHeap, NativeList, Root, Targets};
use super::memory::PoolTable;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use victory_backend::backend::{Dispatch, Heaps, RootArgs};
use victory_backend::resources::*;

#[cfg(test)]
thread_local! {
    /// Walks of the pool table for a wrap's signature ([`NativeHost::alias_signature`]), for tests.
    pub(crate) static SIGNATURES: Cell<u64> = const { Cell::new(0) };
}

/// Buffers over ranges of the title's own addresses, by `(start, end)`: the signature of the
/// mappings they were made over, when they were last used, and the buffer.
type TitleAliases<Buf> = RefCell<HashMap<(u64, u64), (u64, u64, Rc<Buf>)>>;

/// Buffers in the host's own memory by `(start, key)`: their end and their wrap once made.
type OwnBuffers<Buf> = RefCell<std::collections::BTreeMap<(u64, usize), (u64, Option<Rc<Buf>>)>>;

pub struct NativeHost<B: GraphicsBackend> {
    pub backend: B,
    memory: Arc<PoolTable>,
    /// A buffer over each pool, made the first time the GPU needs it.
    buffers: RefCell<HashMap<usize, Rc<B::Buffer>>>,
    /// Buffers over the title's own addresses, by page-aligned range ([`Self::title_alias`]).
    aliases: TitleAliases<B::Buffer>,
    /// The pool table's mappings as [`Self::mapping_list`] last read them, and their generation.
    maps: RefCell<Option<(u64, Rc<[Mapping]>)>>,
    /// Pools released whose buffers went, not yet taken by [`Self::released_pools`].
    released: RefCell<Vec<usize>>,
    /// The pool-table generation [`Self::follow_title_pages`] last looked at.
    followed: Cell<u64>,
    resources: RefCell<HashMap<usize, Res<Rc<B::Texture>>>>,
    /// Buffers in the host's own memory (committed, or placed in a heap), which no pool holds
    /// ([`Self::own_buffer`]).
    own: OwnBuffers<B::Buffer>,
    /// Counts changes to [`Self::resources`], so unrealized table entries retry once registered.
    resources_epoch: std::cell::Cell<u64>,
    /// Textures made without their top levels (`VICTORY_TEXTURE_DROP_TOP_MIP`): key to
    /// (levels dropped, levels the host texture has).
    dropped: RefCell<HashMap<usize, (usize, usize)>>,
    heaps: RefCell<Vec<Rc<NativeHeap<B>>>>,
    contract: RefCell<Contract<B>>,
    /// Draws and dispatches dropped because they read a root parameter the title never set.
    unset_roots: std::cell::Cell<u64>,
    /// Draws and clears dropped because none of their targets has a host texture.
    no_targets: std::cell::Cell<u64>,
    /// Commands dropped because they name title memory outside every mapped pool.
    unmapped: std::cell::Cell<u64>,
    /// Heap entries cleared because the texture they showed was released.
    cleared: std::cell::Cell<u64>,
    /// Commands dropped because the host could not prepare them for another reason.
    failed: std::cell::Cell<u64>,
    /// Passes handed to the backend.
    passes: std::cell::Cell<u64>,
    /// Buffer heap entries promoted lazily to `texture_buffer` views so far.
    typed_views: std::cell::Cell<u64>,
    /// With texture copy-back on: the keys dispatches had bound for unordered access, not yet
    /// taken by [`Self::take_written`].
    written: RefCell<Option<std::collections::HashSet<usize>>>,
    /// The same keys always, for the title side to stop filling them from title memory
    /// ([`Self::take_gpu_written`]).
    gpu_written: RefCell<std::collections::HashSet<usize>>,
    /// Colour twins of depth textures compute writes, by the depth texture's address
    /// ([`Self::depth_twin`]).
    twins: RefCell<HashMap<usize, Rc<DepthTwin<B>>>>,
    /// Textures given up under the texture budget (`Title::keep_to_budget`) and not made again.
    evicted: RefCell<std::collections::HashSet<usize>>,
    /// Evicted textures an entry's record named since the last [`Self::take_wanted`].
    wanted: RefCell<Vec<usize>>,
    /// Textures a heap entry showed through an unordered access view: what a shader may write,
    /// so the host texture may hold what title memory does not and is never given up.
    unordered: RefCell<std::collections::HashSet<usize>>,
}

/// A colour texture standing in for a depth texture that compute writes ([`NativeHost::depth_twin`]).
/// Metal does not support shader writes to a depth texture, so the title's unordered views of its depth plane,
/// and its one-channel colour reads made after the first of those, get this twin. The newer
/// contents move between the two by copies where the use changes ([`NativeHost::twin_copies`]):
/// into the twin before a compute pass, into the depth texture before a render pass whose depth
/// target it is. The lighting's material depth is written by compute and tested by draws; the
/// shadow maps are drawn into and also read by compute.
pub(crate) struct DepthTwin<B: GraphicsBackend> {
    depth: std::rc::Weak<B::Texture>,
    texture: Rc<B::Texture>,
    /// A buffer the copies go through: Metal copies a depth plane to and from buffers only.
    stage: Pages<B>,
    width: usize,
    height: usize,
    /// Bytes of a texel of the depth plane, and of the twin: 2 for 16-bit depth, else 4.
    texel: usize,
    /// Whether the twin holds the newer contents (a compute pass may have written it since the
    /// depth texture was last drawn into or cleared); else the depth texture does.
    twin_newer: Cell<bool>,
}

/// Warns that a dropped command was a fill: a bottom-of-pipe value write, which another queue's
/// list (or this queue's next segment) may wait for with no other way to go on.
fn lost_fill<B: GraphicsBackend>(c: &Cmd<B>, why: &str) {
    // Once per page: a title writes the same unmapped words every frame.
    static PAGES: std::sync::Mutex<std::collections::BTreeSet<u64>> = std::sync::Mutex::new(std::collections::BTreeSet::new());
    if let Cmd::Fill { dst, len, value } = c {
        let mut pages = PAGES.lock().unwrap_or_else(|e| e.into_inner());
        if pages.len() < 50 && pages.insert(dst >> 12) {
            tracing::warn!("native host drops a value write: byte {value:#x} x{len} at {dst:#x}: {why}");
        }
    }
}

/// How the error of a root parameter read with nothing set in it ends ([`NativeHost::execute`]
/// drops that draw or dispatch alone).
const NEVER_SET: &str = "was never set";

/// How the error of title memory outside every mapped pool ends ([`NativeHost::execute`] drops
/// that command alone).
const NOT_MAPPED: &str = "is not in a mapped pool";

/// The host's side of the pool-offset contract ([`super::addr`]).
struct Contract<B: GraphicsBackend> {
    pairs: Pairs,
    /// The memory the null view (pair 0) has, with size 0.
    null: Option<Rc<Pages<B>>>,
    /// A resource heap of the host's own, for shaders that need the pairs and the title bound none.
    heap: Option<Rc<NativeHeap<B>>>,
}

impl<B: GraphicsBackend> Default for Contract<B> {
    fn default() -> Self {
        Contract {
            pairs: Pairs::default(),
            null: None,
            heap: None,
        }
    }
}

/// What one command resolved to: title addresses are made into buffers here, kept alive until the
/// submission has been handed over.
struct Prep<B: GraphicsBackend> {
    root: RootArgs,
    /// The pixel stage's own arguments.
    ps_root: Option<RootArgs>,
    bufs: Vec<Rc<B::Buffer>>,
    vertex: Vec<(usize, Rc<B::Buffer>, *const u8)>,
    index: Option<(Rc<B::Buffer>, *const u8)>,
    blit: Vec<(Rc<B::Buffer>, *const u8)>,
    /// The `ExecuteIndirect` argument record.
    indirect: Option<(Rc<B::Buffer>, *const u8)>,
    /// Its count buffer word and the record's index.
    count: Option<(Rc<B::Buffer>, *const u8, u32)>,
    /// The resource heap the work needs in place of the one the title bound (none bound).
    heap: Option<Rc<NativeHeap<B>>>,
}

enum Group {
    Compute(usize, usize),
    Blit(usize, usize),
    Render(usize, usize),
    Clear(usize),
}

type HeapPair<B> = (Option<Rc<NativeHeap<B>>>, Option<Rc<NativeHeap<B>>>);

fn heaps<B: GraphicsBackend>(h: &HeapPair<B>) -> Heaps<'_, B> {
    Heaps {
        resource: h.0.as_ref().map(|h| &h.heap),
        sampler: h.1.as_ref().map(|h| &h.heap),
    }
}

/// The title's record in descriptor slot `handle` (tag, view index, generation).
fn record_at(handle: u64) -> [u8; 12] {
    let mut r = [0u8; 12];
    super::safe_read(handle as *const u8, &mut r);
    r
}

impl<B: GraphicsBackend> NativeHost<B> {
    pub fn new(backend: B) -> Self {
        Self::with_memory(backend, Arc::new(PoolTable::default()))
    }

    /// A host over a pool table the pool observer already feeds.
    pub fn with_memory(backend: B, memory: Arc<PoolTable>) -> Self {
        NativeHost {
            backend,
            memory,
            buffers: RefCell::default(),
            aliases: RefCell::default(),
            released: RefCell::default(),
            followed: Cell::new(0),
            maps: RefCell::default(),
            resources: RefCell::default(),
            own: RefCell::default(),
            resources_epoch: std::cell::Cell::new(0),
            dropped: RefCell::default(),
            heaps: RefCell::default(),
            contract: RefCell::default(),
            unset_roots: std::cell::Cell::new(0),
            no_targets: std::cell::Cell::new(0),
            unmapped: std::cell::Cell::new(0),
            cleared: std::cell::Cell::new(0),
            failed: std::cell::Cell::new(0),
            passes: std::cell::Cell::new(0),
            typed_views: std::cell::Cell::new(0),
            written: Default::default(),
            gpu_written: Default::default(),
            twins: RefCell::default(),
            evicted: Default::default(),
            wanted: Default::default(),
            unordered: Default::default(),
        }
    }

    /// Draws and clears dropped so far because none of their targets has a host texture.
    pub fn no_targets(&self) -> u64 {
        self.no_targets.get()
    }

    /// Draws and dispatches dropped so far because they read a root parameter never set.
    pub fn unset_roots(&self) -> u64 {
        self.unset_roots.get()
    }

    /// Commands dropped so far because they name title memory outside every mapped pool.
    pub fn unmapped(&self) -> u64 {
        self.unmapped.get()
    }

    /// Commands dropped so far because the host could not prepare them (not an unset root
    /// parameter, not unmapped memory).
    pub fn failed(&self) -> u64 {
        self.failed.get()
    }

    /// Passes handed to the backend so far (compute, render and blit; a clear is a render pass).
    pub fn passes(&self) -> u64 {
        self.passes.get()
    }

    /// Buffer heap entries promoted lazily to `texture_buffer` views so far.
    pub fn typed_views(&self) -> u64 {
        self.typed_views.get()
    }

    /// Resource heaps the title made (the host's own contract heap apart).
    pub fn resource_heaps(&self) -> usize {
        self.heaps.borrow().iter().filter(|h| h.kind == HeapKind::Resource).count()
    }

    /// Buffer heap entries across all resource heaps that currently hold a live `texture_buffer`
    /// view (overwriting an entry with `write_buffer` / `write_texture` or `clear_entry` drops it).
    pub fn live_typed_views(&self) -> usize {
        self.heaps
            .borrow()
            .iter()
            .map(|h| self.backend.heap_typed_views(&h.heap))
            .sum()
    }

    /// The table the pool observer keeps current; register it as (or call it from) the observer.
    pub fn memory(&self) -> Arc<PoolTable> {
        self.memory.clone()
    }

    /// The buffer and pool-mapping address of `[addr, addr + len)` of title memory.
    /// Outside every pool, a buffer in the host's own memory that holds the whole range.
    pub fn resolve(&self, addr: u64, len: u64) -> Result<(Rc<B::Buffer>, *const u8), String> {
        crate::native::freed::check(addr, len, "resolve");
        self.forget_released();
        match self.memory.resolve(addr, len) {
            Some((pool, ptr)) => Ok((self.pool_buffer(pool.id, pool.base, pool.len)?, ptr)),
            None => self
                .own_buffer(addr, len)
                .ok_or_else(|| format!("title address {addr:#x}+{len:#x} {NOT_MAPPED}")),
        }
    }

    /// The wrap of the host-memory buffer ([`Self::register_own_buffer`]) holding all of
    /// `[addr, addr + len)`, made at first use and dropped when the buffer is released. It is kept
    /// out of the pool table, so the page table, the view pairs, the pool generation and the
    /// texture placement that walk it never take host memory for title memory.
    // shortcut: only the buffer starting nearest below `addr` is tried, so of buffers placed
    // over one another in a heap an outer one is missed (the command stays refused); look
    // further back if a title roots views in such a buffer.
    fn own_buffer(&self, addr: u64, len: u64) -> Option<(Rc<B::Buffer>, *const u8)> {
        let end = addr.checked_add(len.max(1))?;
        let mut own = self.own.borrow_mut();
        let (&(start, _), (stop, wrap)) = own.range_mut(..=(addr, usize::MAX)).next_back()?;
        if end > *stop {
            return None;
        }
        if wrap.is_none() {
            let (base, top) = Self::alias_range(start, *stop - start)?;
            *wrap = Some(Rc::new(self.backend.wrap_title_memory(base as *mut u8, (top - base) as usize).ok()?));
        }
        Some((wrap.clone()?, addr as *const u8))
    }

    /// Drops what the host keeps of pools released since the last call: their buffers, and their
    /// view pairs (free for other pools from now on). [`Self::released_pools`] lists them.
    fn forget_released(&self) {
        let gone = self.memory.take_released();
        if gone.is_empty() {
            return;
        }
        // A wrap over pages of a released pool no longer matches the mappings there: later
        // mirrors make new ones. The others stay, so a heap entry mirrored again reuses its wrap
        // (dropping them all made a new 174 MB wrap at each release, while the old ones stayed
        // alive in the heap entries and in-flight work: Metal's count rose by gigabytes).
        self.aliases.borrow_mut().retain(|&(base, end), (sig, _, _)| self.memory.signature(base, end - base) == Some(*sig));
        let mut buffers = self.buffers.borrow_mut();
        let mut c = self.contract.borrow_mut();
        for &id in &gone {
            buffers.remove(&id);
            c.pairs.release(id);
        }
        self.released.borrow_mut().extend(gone);
    }

    /// Pools released since the last call (for what the title side keeps of them).
    pub(crate) fn released_pools(&self) -> Vec<usize> {
        self.forget_released();
        std::mem::take(&mut self.released.borrow_mut())
    }

    /// A buffer over the title's own addresses `[addr, addr + len)`, for a buffer view whose range
    /// is not one linear run of a pool but which the title maps without a gap: title memory is
    /// mapped at its own addresses in this process, so the view wraps those (whole 16 KiB host
    /// pages, the alignment the no-copy wrap needs, also under Rosetta's 4 KiB pages; the title's
    /// own pages are 64 KiB). The G-buffer's culling programs read
    /// their instances through a heap view of 174 MB whose pages the title mapped from scattered
    /// pool pages (#258); the null view gave them zeros. A wrap is reused while the mappings under
    /// it are the same ([`PoolTable::signature`]), made again when they change; every wrap a heap
    /// entry shows is kept and 64 others at most, and those over a released pool's pages are
    /// dropped. A range with a gap stays the null view. A wrap keeps
    /// the physical pages it was made over, so the heap entries that show one are mirrored again
    /// when the title's mappings under them change ([`Self::follow_title_pages`]).
    // shortcut: a range partly unmapped is the null view as a whole (zeros), where a reserved
    // resource reads zeros only at its unmapped tiles (D3D12 tier 2); give such a range a wrap of
    // its own with zero pages in the holes if a title reads a view it maps sparsely.
    fn title_alias(&self, addr: u64, len: u64) -> Option<(Rc<B::Buffer>, *const u8, u64)> {
        // Wraps kept for reuse: the heap entries that show one keep it alive themselves.
        const KEPT: usize = 64;
        static USE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let (base, end) = Self::alias_range(addr, len)?;
        let Some(sig) = self.memory.signature(base, end - base) else {
            // A real gap: the title maps nothing at part of the view. It stays the null view.
            static GAPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if GAPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 64 {
                tracing::info!(addr = format!("{addr:#x}"), len, "a buffer view over title memory with a gap stays the null view");
            }
            return None;
        };
        let now = USE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut aliases = self.aliases.borrow_mut();
        if let Some((s, used, b)) = aliases.get_mut(&(base, end))
            && *s == sig
        {
            *used = now;
            return Some((b.clone(), addr as *const u8, sig));
        }
        // What a large wrap costs (#267 review): the process footprint and Metal's own count.
        #[cfg(target_os = "macos")]
        let footprint = victory_platform::mem::footprint;
        #[cfg(not(target_os = "macos"))]
        let footprint = || 0usize;
        let measure = || (footprint(), self.backend.allocated_bytes());
        let before = (len >= 16 << 20).then(measure);
        let buf = Rc::new(self.backend.wrap_title_memory(base as *mut u8, (end - base) as usize).ok()?);
        if let Some(before) = before {
            static BIG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if BIG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
                let after = measure();
                tracing::info!(addr = format!("{addr:#x}"), len, signature = format!("{sig:#x}"), footprint_before = before.0, footprint_after = after.0, metal_before = before.1, metal_after = after.1, "a large buffer view wraps the title's own pages");
            }
        }
        if aliases.len() >= KEPT {
            // Only a wrap no heap entry shows goes: one that is shown would be made again, and the
            // entry's copy would stay alive beside it.
            let shown: std::collections::HashSet<(u64, u64, u64)> = self
                .heaps
                .borrow()
                .iter()
                .flat_map(|h| {
                    h.title_views
                        .borrow()
                        .values()
                        .filter_map(|&(a, l, s)| Some((Self::alias_range(a, l)?, s?)))
                        .map(|((b, e), s)| (b, e, s))
                        .collect::<Vec<_>>()
                })
                .collect();
            if let Some(&oldest) = aliases
                .iter()
                .filter(|&(&(b, e), &(s, _, _))| !shown.contains(&(b, e, s)))
                .min_by_key(|(_, (_, used, _))| *used)
                .map(|(k, _)| k)
            {
                aliases.remove(&oldest);
            }
        }
        aliases.insert((base, end), (sig, now, buf.clone()));
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(addr = format!("{addr:#x}"), len, "a buffer view not linear in its pool reads the title's own mapping (first time)");
        });
        Some((buf, addr as *const u8, sig))
    }

    /// The whole 16 KiB host pages a wrap of `[addr, addr + len)` covers (the alignment the
    /// no-copy wrap needs, also under Rosetta's 4 KiB pages), as `(start, end)`.
    fn alias_range(addr: u64, len: u64) -> Option<(u64, u64)> {
        const PAGE: u64 = 1 << 14;
        if addr == 0 || len == 0 {
            return None;
        }
        let end = addr.checked_add(len)?.checked_add(PAGE - 1)? & !(PAGE - 1);
        Some((addr & !(PAGE - 1), end))
    }

    /// The signature of the title's mappings under a wrap of `[addr, addr + len)`; `None` where
    /// part of it is not mapped.
    fn alias_signature(&self, addr: u64, len: u64) -> Option<u64> {
        let (base, end) = Self::alias_range(addr, len)?;
        #[cfg(test)]
        SIGNATURES.with(|n| n.set(n.get() + 1));
        self.memory.signature(base, end - base)
    }

    /// Mirrors again, once per pool-table change and before the work that reads them, the heap
    /// buffer views over title memory no pool run holds whose pages changed since they were
    /// written: a wrap whose mappings the title changed or removed, and a null view whose gap the
    /// title has mapped since. A shader can load such an entry from the heap itself (the
    /// G-buffer culling programs' 174 MB view, #258), so no bound table brings it back to
    /// [`Self::remirror_tables`], and the title writes the entry once, early, while it keeps
    /// mapping pages into the range. Costs one comparison while the pool table is unchanged.
    fn follow_title_pages(&self) {
        let generation = self.memory.generation();
        if self.followed.replace(generation) == generation {
            return;
        }
        let heaps: Vec<Rc<NativeHeap<B>>> = self.heaps.borrow().clone();
        // The 1st, 10th, 100th... look: the large views followed and their signatures now.
        static LOOKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let looks = LOOKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if matches!(looks, 1 | 10 | 100 | 1_000 | 10_000 | 100_000 | 1_000_000) {
            let large: Vec<String> = heaps
                .iter()
                .flat_map(|h| h.title_views.borrow().iter().filter(|v| v.1.1 >= 16 << 20).map(|(i, &(a, l, _))| format!("{i}: {a:#x}+{l:#x} {:?}", self.alias_signature(a, l).map(|s| format!("{s:#x}")))).collect::<Vec<_>>())
                .collect();
            tracing::info!(looks, generation, ?large, "following the title's pages under heap buffer views");
        }
        // The title copies one view into many entries (54 of the 174 MiB one in a world run): each
        // range's walk is made once a look (about 3 ms a look before, the pool table locked).
        let mut walked: HashMap<(u64, u64), Option<u64>> = HashMap::new();
        for heap in heaps {
            let moved: Vec<_> = heap
                .title_views
                .borrow()
                .iter()
                .filter_map(|(&index, &(addr, len, was))| {
                    let now = match Self::alias_range(addr, len) {
                        Some(range) => *walked.entry(range).or_insert_with(|| self.alias_signature(addr, len)),
                        None => None,
                    };
                    (now != was).then_some((index, addr, len, was, now))
                })
                .collect();
            for (index, addr, len, was, now) in moved {
                let handle = heap.start + index as u64 * Self::step(heap.kind);
                let result = self.mirror(handle);
                if let Ok(false) = result {
                    // No longer a view the host can write (the title put another record there):
                    // nothing to follow until it is one again.
                    heap.title_views.borrow_mut().remove(&index);
                }
                // Large views apart, so the few that matter are not lost among small ones.
                static LARGE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                static SMALL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let logged = if len >= 16 << 20 { &LARGE } else { &SMALL };
                if logged.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 64 {
                    let sig = |s: Option<u64>| s.map(|s| format!("{s:#x}"));
                    tracing::info!(index, addr = format!("{addr:#x}"), len, was = ?sig(was), now = ?sig(now), ?result, "a heap buffer view's title pages changed: mirrored again");
                }
            }
        }
    }

    /// The buffer over pool `id`, made the first time the GPU needs it.
    fn pool_buffer(&self, id: usize, base: usize, len: usize) -> Result<Rc<B::Buffer>, String> {
        let mut buffers = self.buffers.borrow_mut();
        Ok(match buffers.entry(id) {
            std::collections::hash_map::Entry::Occupied(e) => e.get().clone(),
            std::collections::hash_map::Entry::Vacant(e) => e
                .insert(Rc::new(self.backend.wrap_title_memory(base as *mut u8, len)?))
                .clone(),
        })
    }

    pub fn compute_pipeline(
        &self,
        dxil: &[u8],
        threads: [usize; 3],
    ) -> Result<Rc<ComputePso<B>>, String> {
        Ok(Rc::new(ComputePso {
            pso: self.backend.create_compute_pipeline(dxil)?,
            threads,
            binding: None,
        }))
    }

    pub fn render_pipeline(
        &self,
        desc: &RenderPipelineDesc<'_>,
    ) -> Result<Rc<B::RenderPipeline>, String> {
        self.backend.create_render_pipeline(desc).map(Rc::new)
    }

    pub fn texture(&self, desc: &TextureDesc) -> Result<Rc<B::Texture>, String> {
        self.backend.create_texture(desc).map(Rc::new)
    }

    /// The colour twin of depth texture `depth` (title texture `resource`) that a view of format
    /// `format` shows instead of it, and the format of that view; made at the first unordered view
    /// of the depth plane. `None` for views that stay on the depth texture: depth formats, formats
    /// other than the one-channel 16- and 32-bit ones a depth plane takes, any view before the
    /// first unordered one, and every view with `VICTORY_DEPTH_TWINS=0`.
    ///
    /// The lighting's material depth (1280x720, DXGI 118, the console's `D16_UNORM_S8_UINT`) is
    /// written by compute through `R16_UNORM` views of its depth plane and `R8_UINT` views of its
    /// stencil plane, then tested equal by the tiled light draws. A view of the depth texture
    /// itself let compute write it, but Metal does not support shader writes to a depth texture:
    /// a blit reads them, a later render pass's depth test does not (M4 Pro; the test
    /// `a_draw_tests_against_the_depth_compute_wrote_through_an_unordered_view`).
    /// The host keeps that format as 32-bit float depth with stencil, so an `R16_UNORM` view of
    /// it writes the twin as 32-bit floats. Guess: the title writes `k / 65535` there, the very
    /// float its draws rasterise (VIctory's D3D12 path converts `k` to `k * 0x37800080`). A
    /// 16-bit depth texture (`D16Unorm`: the shadow maps) has a 16-bit twin, copied bit for bit.
    fn depth_twin(&self, resource: usize, depth: &Rc<B::Texture>, format: Format, unordered: bool) -> Option<(Rc<B::Texture>, Format)> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_DEPTH_TWINS").is_none_or(|v| v != "0")) {
            return None;
        }
        if !matches!(format, Format::R32Float | Format::R32Uint | Format::R16Unorm | Format::R16Uint) {
            return None;
        }
        let view_on = |texel: usize| match (texel, format) {
            (2, Format::R16Unorm | Format::R16Uint) | (4, Format::R32Float | Format::R32Uint) => Some(format),
            (4, Format::R16Unorm) => Some(Format::R32Float),
            _ => None,
        };
        let at = Rc::as_ptr(depth) as usize;
        let known = self
            .twins
            .borrow()
            .get(&at)
            .filter(|t| t.depth.upgrade().is_some_and(|d| Rc::ptr_eq(&d, depth)))
            .map(|t| (t.texture.clone(), t.texel));
        if let Some((texture, texel)) = known {
            return view_on(texel).map(|v| (texture, v));
        }
        if !unordered {
            return None;
        }
        let info = texture_info_of(resource)?;
        let desc = info.desc()?;
        if !desc.format.is_depth() || desc.dim != TexDim::D2 {
            return None;
        }
        let (texel, twin_format) = if desc.format == Format::D16Unorm { (2, Format::R16Unorm) } else { (4, Format::R32Float) };
        let view = view_on(texel)?;
        let texture = self
            .texture(&TextureDesc {
                dim: TexDim::D2,
                format: twin_format,
                width: desc.width,
                height: desc.height,
                depth_or_layers: 1,
                mips: 1,
                render_target: false,
                unordered_access: true,
            })
            .ok()?;
        let stage = Pages::new(&self.backend, desc.width * desc.height * texel).ok()?;
        tracing::info!(format = info.format, w = desc.width, h = desc.height, "a depth texture compute writes gets a colour twin");
        let twin = DepthTwin {
            depth: Rc::downgrade(depth),
            texture: texture.clone(),
            stage,
            width: desc.width,
            height: desc.height,
            texel,
            twin_newer: Cell::new(false),
        };
        self.twins.borrow_mut().insert(at, Rc::new(twin));
        // Entries mirrored before still show the depth texture: they are mirrored again, so that
        // colour reads of the depth plane read the twin.
        for heap in self.heaps.borrow().iter() {
            for index in heap.take_entries(resource) {
                heap.mirrored.borrow_mut().remove(&index);
            }
        }
        Some((texture, view))
    }

    /// The colour twin of depth texture `depth`, if it has one.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn twin_of(&self, depth: &Rc<B::Texture>) -> Option<Rc<B::Texture>> {
        let twins = self.twins.borrow();
        let twin = twins.get(&(Rc::as_ptr(depth) as usize))?;
        twin.depth.upgrade().is_some_and(|d| Rc::ptr_eq(&d, depth)).then(|| twin.texture.clone())
    }

    /// The copies between depth textures and their twins that a pass needs before it runs, as
    /// (twin, its depth texture, whether into the depth texture); notes which side holds the
    /// newer contents after the pass ([`DepthTwin::twin_newer`]). Before a compute pass
    /// (`compute`), every twin whose depth texture is newer takes its contents. Before a pass
    /// that draws into, clears or copies to or from one of the textures `depths` names, a newer
    /// twin goes into it first (not when the pass writes it whole: a depth clear), and the
    /// depth texture is the newer side from then on. Guess: a render pass that only reads a twin
    /// (a shadow map sampled through a colour view) follows a compute pass, not the draws into
    /// its depth texture alone; else it reads the twin's older contents.
    #[allow(clippy::type_complexity)]
    fn twin_copies(&self, compute: bool, depths: &[(&Rc<B::Texture>, bool)]) -> Vec<(Rc<DepthTwin<B>>, Rc<B::Texture>, bool)> {
        let mut out = Vec::new();
        let mut twins = self.twins.borrow_mut();
        if twins.is_empty() {
            return out;
        }
        twins.retain(|_, t| t.depth.strong_count() > 0);
        if compute {
            for t in twins.values() {
                if !t.twin_newer.get()
                    && let Some(d) = t.depth.upgrade()
                {
                    t.twin_newer.set(true);
                    out.push((t.clone(), d, false));
                }
            }
        }
        for &(d, whole) in depths {
            if let Some(t) = twins.get(&(Rc::as_ptr(d) as usize))
                && t.depth.upgrade().is_some_and(|x| Rc::ptr_eq(&x, d))
            {
                if t.twin_newer.get() && !whole {
                    out.push((t.clone(), d.clone(), true));
                }
                t.twin_newer.set(false);
            }
        }
        out
    }

    /// The depth textures the commands `cmds` draw into, clear, or copy to or from, for
    /// [`Self::twin_copies`]: with whether the command writes the depth plane whole.
    fn twin_uses<'a>(&self, cmds: &[&'a Cmd<B>]) -> Vec<(&'a Rc<B::Texture>, bool)> {
        if self.twins.borrow().is_empty() {
            return vec![];
        }
        let mut uses = vec![];
        for &c in cmds {
            match c {
                Cmd::Draw { targets: Targets { depth: Some(d), .. }, .. } => uses.push((d, false)),
                Cmd::Clear { depth: Some((d, z, _)), .. } => uses.push((d, z.is_some())),
                Cmd::CopyTexture { src, dst, .. } => uses.extend([(src, false), (dst, false)]),
                Cmd::CopyTextureToBuffer { src, .. } => uses.push((src, false)),
                Cmd::CopyBufferToTexture { dst, .. } => uses.push((dst, false)),
                _ => {}
            }
        }
        uses
    }

    /// [`Self::twin_copies`] for one depth texture, before a compute pass (`compute`) or a render
    /// pass into it: whether each copy it would make goes into the depth texture (`true`) or
    /// into the twin.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn twin_copies_for_test(&self, compute: bool, depth: &Rc<B::Texture>) -> Vec<bool> {
        let depths: &[(&Rc<B::Texture>, bool)] = if compute { &[] } else { &[(depth, false)] };
        self.twin_copies(compute, depths).into_iter().filter(|c| Rc::ptr_eq(&c.1, depth)).map(|c| c.2).collect()
    }

    /// The blits that make `copies` ([`Self::twin_copies`]): each through the twin's buffer, since
    /// Metal copies a depth plane only to and from buffers (`DepthFromDepthStencil`).
    #[allow(clippy::type_complexity)]
    fn twin_blits<'a>(copies: &'a [(Rc<DepthTwin<B>>, Rc<B::Texture>, bool)]) -> Vec<BlitOp<'a, B>> {
        let mut ops = Vec::new();
        for (twin, depth, to_depth) in copies {
            let (w, h, b) = (twin.width, twin.height, twin.texel);
            let stage = (&*twin.stage.buf, twin.stage.ptr().cast_const());
            let (src, dst): (&B::Texture, &B::Texture) = if *to_depth { (&twin.texture, depth) } else { (depth, &twin.texture) };
            ops.push(BlitOp::CopyTextureToBuffer {
                src,
                src_mip: 0,
                src_slice: 0,
                src_origin: [0; 3],
                size: [w, h, 1],
                dst: stage,
                len: w * h * b,
                row_pitch: w * b,
                slice_pitch: w * h * b,
            });
            ops.push(BlitOp::CopyBufferToTexture {
                src: stage,
                len: w * h * b,
                row_pitch: w * b,
                slice_pitch: w * h * b,
                size: [w, h, 1],
                dst,
                dst_mip: 0,
                dst_slice: 0,
                dst_origin: [0; 3],
            });
        }
        ops
    }

    /// The copies [`Self::twin_copies`] gives, run on the GPU now and waited for.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn run_twin_copies_for_test(&self, compute: bool, depth: &Rc<B::Texture>) -> usize {
        let depths: &[(&Rc<B::Texture>, bool)] = if compute { &[] } else { &[(depth, false)] };
        let copies: Vec<_> = self.twin_copies(compute, depths).into_iter().filter(|c| Rc::ptr_eq(&c.1, depth)).collect();
        let q = self.backend.create_queue().unwrap();
        let f = self.backend.create_fence(0).unwrap();
        let passes = [Pass::Blit(Self::twin_blits(&copies))];
        self.backend.submit_passes(&q, &PassSubmission { waits: &[], passes: &passes, signals: &[(&f, 1)] }).unwrap();
        assert!(self.backend.wait(&f, 1, 10_000), "GPU did not finish");
        copies.len()
    }

    /// Names a buffer resource object `key` (what `CreateShaderResourceView` is passed) at title memory.
    pub fn register_buffer(&self, key: usize, addr: u64, len: u64) {
        self.resources
            .borrow_mut()
            .insert(key, Res::Buffer(addr, len));
        self.resources_epoch.set(self.resources_epoch.get() + 1);
    }

    /// Besides [`Self::register_buffer`]: the buffer's memory is the host's own, not a pool's, and
    /// root views, copies and buffer views into it reach it ([`Self::resolve`]) until it is released.
    pub fn register_own_buffer(&self, key: usize, addr: u64, len: u64) {
        self.register_buffer(key, addr, len);
        self.own.borrow_mut().insert((addr, key), (addr.saturating_add(len), None));
    }

    /// Records that texture `key` was made without its top `dropped` levels, `host_mips` left,
    /// so views of it map the title's levels onto the host's.
    pub fn set_dropped_levels(&self, key: usize, dropped: usize, host_mips: usize) {
        let mut map = self.dropped.borrow_mut();
        if dropped == 0 {
            map.remove(&key);
        } else {
            map.insert(key, (dropped, host_mips));
        }
    }

    pub fn register_texture(&self, key: usize, texture: &Rc<B::Texture>) {
        self.evicted.borrow_mut().remove(&key);
        self.resources
            .borrow_mut()
            .insert(key, Res::Texture(texture.clone()));
        self.resources_epoch.set(self.resources_epoch.get() + 1);
        // The entries cleared when the texture last at this address was released are checked
        // again: the record the title writes for the new one may equal the old one's.
        for heap in self.heaps.borrow().iter() {
            for index in heap.take_gone(key) {
                heap.mirrored.borrow_mut().remove(&index);
                heap.pending_mirror.borrow_mut().remove(&index);
            }
        }
    }

    /// A shader-visible heap over the title's descriptor memory at `start` (zeroed memory the title
    /// writes views into; 32 bytes an entry, 16 for samplers).
    pub fn heap(
        &self,
        kind: HeapKind,
        start: u64,
        count: usize,
    ) -> Result<Rc<NativeHeap<B>>, String> {
        // A resource heap is longer than the title made it by the entries that hold the pool-offset
        // contract's view pairs.
        let entries = if kind == HeapKind::Resource {
            if !addr::fits(count) {
                tracing::warn!(
                    count,
                    "a resource heap too long for title memory views: shaders that find memory by address will not run with it"
                );
            }
            Self::resource_entries(count)
        } else {
            count
        };
        let heap = Rc::new(NativeHeap {
            heap: self.backend.create_heap(kind, entries)?,
            kind,
            start,
            count,
            pairs: Default::default(),
            table: Default::default(),
            strides: Default::default(),
            addresses: Default::default(),
            mirrored: Default::default(),
            pending_mirror: Default::default(),
            title_views: Default::default(),
            images: Default::default(),
            used: Default::default(),
            typed_pending: Default::default(),
        });
        self.heaps.borrow_mut().push(heap.clone());
        Ok(heap)
    }

    /// The entries of a resource heap of `count` title entries: the title's, the view pairs of
    /// the pool-offset contract, and then up to the translator's clamp of heap indices
    /// (`HEAP_VIEWS`: a translated shader reads no entry past it, and reads the host's null
    /// buffer views just below it, see [`Self::write_null_buffers`]). A heap too long for the
    /// pairs ([`addr::fits`]) stays as long as it is.
    ///
    /// shortcut: every such heap is `HEAP_VIEWS` long (48 MiB of Metal entries, and its stride
    /// table 8 MiB, both resident from its first dispatch) because a translated shader's clamp
    /// and its null view index are constants, not the heap's length: sizing heap and table to
    /// `count` needs the shader to read a per-heap bound (a root constant, say) instead.
    fn resource_entries(count: usize) -> usize {
        let entries = count + addr::RESERVED;
        if addr::fits(count) {
            entries.max(victory_shader::emit::HEAP_VIEWS as usize)
        } else {
            entries
        }
    }

    /// Writes the null raw buffer views a translated shader reads in place of a heap entry that
    /// holds no buffer view (`victory_shader::emit`, "Buffer views at computed heap indices"):
    /// the shader view at `NULL_BUFFER_VIEW` and the unordered one just below it. Written with
    /// the heap's view-stride table ([`Self::view_strides`]), which is what sends a shader there.
    fn write_null_buffers(&self, heap: &NativeHeap<B>) -> Result<(), String> {
        if !addr::fits(heap.count) {
            return Err(format!(
                "a resource heap of {} entries has no room for the null buffer views",
                heap.count
            ));
        }
        let null = self.null_view()?;
        let at = victory_shader::emit::NULL_BUFFER_VIEW as usize;
        for index in [at - 1, at] {
            self.backend.write_buffer(&heap.heap, index, &null.buf, null.ptr().cast_const(), 0);
        }
        Ok(())
    }

    /// Each heap's descriptor records, `[start, end)` in the title's memory (its own view, which
    /// the device's view calls and the title's stores write).
    #[cfg(target_os = "macos")]
    pub(crate) fn heap_ranges(&self) -> Vec<(usize, usize)> {
        self.heaps
            .borrow()
            .iter()
            .map(|h| {
                let end = h.start + h.count as u64 * Self::step(h.kind);
                (h.start as usize, end as usize)
            })
            .collect()
    }

    /// The heap `handle` (a descriptor's address, a root table's) lies in, and the entry's index.
    pub fn heap_at(&self, handle: u64) -> Option<(Rc<NativeHeap<B>>, usize)> {
        self.heaps.borrow().iter().find_map(|h| {
            let index = (handle.checked_sub(h.start)? / Self::step(h.kind)) as usize;
            (index < h.count).then(|| (h.clone(), index))
        })
    }

    /// The title released the heap whose descriptors start at `start`.
    pub fn drop_heap(&self, start: u64) {
        self.heaps.borrow_mut().retain(|h| h.start != start);
    }

    /// The title released resource object `key`: later views naming it decode to nothing.
    pub fn forget_resource(&self, key: usize) {
        self.forget_marks(key);
        if self.resources.borrow_mut().remove(&key).is_some() {
            self.resources_epoch.set(self.resources_epoch.get() + 1);
        }
    }

    /// The title released texture `key` for good (its object's last reference, or the pool it was
    /// placed in): besides [`Self::forget_resource`], the heap entries that show it are cleared.
    /// Left as they were, they kept the host texture alive and resident until the title happened
    /// to write another view there, so the memory the GPU holds wired grew with play time.
    /// A cleared entry is written again when the title writes another record there, or makes a
    /// texture at this address again ([`Self::register_texture`]). Its old record is not: it may
    /// name the next object at this address, of a format the view cannot be made over.
    pub fn release_resource(&self, key: usize) {
        self.forget_marks(key);
        let removed = self.resources.borrow_mut().remove(&key);
        if removed.is_some() {
            self.resources_epoch.set(self.resources_epoch.get() + 1);
        }
        if let Some(Res::Buffer(addr, _)) = removed {
            self.own.borrow_mut().remove(&(addr, key));
        }
        let texture = matches!(removed, Some(Res::Texture(_)));
        if !texture {
            return;
        }
        for heap in self.heaps.borrow().iter() {
            for index in heap.take_entries(key) {
                heap.typed_pending.borrow_mut().set(index, false, false);
                self.backend.clear_entry(&heap.heap, index);
                self.cleared.set(self.cleared.get() + 1);
            }
        }
    }

    /// What the host noted of texture `key`, which is gone for good (the next object at its
    /// address starts afresh).
    fn forget_marks(&self, key: usize) {
        self.evicted.borrow_mut().remove(&key);
        self.unordered.borrow_mut().remove(&key);
    }

    /// Gives up the host texture of `key` under the texture budget: as [`Self::release_resource`]
    /// (no heap entry shows it, so none is cleared), but an entry that names it again asks for it
    /// ([`Self::take_wanted`]).
    pub(crate) fn evict_texture(&self, key: usize) {
        self.release_resource(key);
        self.evicted.borrow_mut().insert(key);
    }

    /// Whether `key` was given up under the texture budget and not made again since.
    pub(crate) fn is_evicted(&self, key: usize) -> bool {
        self.evicted.borrow().contains(&key)
    }

    /// The evicted textures entries named since the last call (`Title` makes them again).
    pub(crate) fn take_wanted(&self) -> Vec<usize> {
        std::mem::take(&mut self.wanted.borrow_mut())
    }

    /// Whether an entry ever showed texture `key` for unordered access.
    pub(crate) fn ever_unordered(&self, key: usize) -> bool {
        self.unordered.borrow().contains(&key)
    }

    /// Whether a heap entry shows texture `key` (shaders index heaps themselves, so a texture an
    /// entry shows counts as used), or ever showed it for unordered access.
    pub(crate) fn texture_in_use(&self, key: usize) -> bool {
        self.unordered.borrow().contains(&key) || self.heaps.borrow().iter().any(|h| h.shows_texture(key))
    }

    /// Heap entries cleared so far because the texture they showed was released.
    pub fn cleared(&self) -> u64 {
        self.cleared.get()
    }

    /// What the descriptor heaps keep resident, each allocation counted once over all of them,
    /// split by whether a bound table named it within the last `idle_frames` frames
    /// (`VICTORY_TRACE_HEAP_SET`): `[all, idle, never]`, each `(textures, texture bytes, buffers,
    /// buffer bytes)`.
    /// shortcut: entries a shader reaches by indexing the heap itself are never "named" here, so
    /// `idle` and `never` are upper bounds.
    pub fn heap_set_report(&self, idle_frames: u32) -> Option<[(usize, u64, usize, u64); 3]> {
        let now = (crate::frame::current() as u32).saturating_add(1);
        let before = now.saturating_sub(idle_frames).max(1);
        let mut merged = std::collections::HashMap::<usize, HeapAllocation>::new();
        let mut any = false;
        for h in self.heaps.borrow().iter() {
            let Some(all) = self.backend.heap_allocations(&h.heap, &h.used.borrow()) else { continue };
            any = true;
            for a in all {
                merged.entry(a.id).and_modify(|m| m.latest = m.latest.max(a.latest)).or_insert(a);
            }
        }
        any.then(|| {
            let mut r = [(0, 0, 0, 0); 3];
            for a in merged.values() {
                let kinds = [true, a.latest != 0 && a.latest < before, a.latest == 0];
                for (k, hit) in kinds.into_iter().enumerate() {
                    if hit {
                        let e = &mut r[k];
                        if a.texture {
                            e.0 += 1;
                            e.1 += a.bytes;
                        } else {
                            e.2 += 1;
                            e.3 += a.bytes;
                        }
                    }
                }
            }
            r
        })
    }

    /// Whether resource object `key` is registered (as a buffer or a texture).
    pub fn knows(&self, key: usize) -> bool {
        self.resources.borrow().contains_key(&key)
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn texture_at(&self, key: usize) -> Option<Rc<B::Texture>> {
        match self.resources.borrow().get(&key) {
            Some(Res::Texture(t)) => Some(t.clone()),
            _ => None,
        }
    }

    fn step(kind: HeapKind) -> u64 {
        u64::from(crate::descriptor::increment(u32::from(kind == HeapKind::Sampler)).unwrap_or(32))
    }

    /// The title wrote a view at descriptor `handle` (the call `descriptor::mirror` makes on
    /// D3D12): decode its record and write the backend's entry; returns whether an entry was
    /// written. A buffer view of memory in no pool gets the null view (no bytes) and the error:
    /// left as it was, the entry would still show the buffer an earlier view named there, which
    /// the title may have given to something else since, and the shader would read and write it.
    /// The entry is not marked mirrored, so the real view is written once the memory is mapped.
    /// Undecodable records and image or sampler views the host cannot make still leave the entry
    /// as it was: a buffer in a texture or sampler entry is not something to hand the GPU.
    pub fn mirror(&self, handle: u64) -> Result<bool, String> {
        let found = self.heaps.borrow().iter().find_map(|h| {
            let at = handle.checked_sub(h.start)?;
            let index = (at / Self::step(h.kind)) as usize;
            (index < h.count).then(|| (h.clone(), index))
        });
        let Some((heap, index)) = found else {
            return Ok(false);
        };
        let view = crate::descriptor::view(handle);
        if heap.kind == HeapKind::Resource
            && let Some(strides) = &*heap.strides.borrow()
        {
            strides.write_at(index, &[self.stride_word(view.as_ref())]);
        }
        if heap.kind == HeapKind::Resource
            && let Some(addresses) = &*heap.addresses.borrow()
        {
            addresses.write_at(4 * index, &self.address_words(view.as_ref()));
        }
        let Some(view) = view else {
            // Emptied (by the title's stores): not mirrored any more, so a submission does not
            // look at it again until the title writes a record there.
            heap.mirrored.borrow_mut().remove(&index);
            return Ok(false);
        };
        let resources = self.resources.borrow();
        let Some(decoded) = decode(&view, &resources) else {
            return Ok(false);
        };
        if !matches!(decoded, Decoded::Buffer { .. }) {
            heap.title_views.borrow_mut().remove(&index);
        }
        match decoded {
            // A buffer view of address 0 (what D3D12 writes for a null view): the null view, and
            // mirrored like any other entry, whatever its size. Nothing is ever mapped at 0 here
            // (title addresses are this process's, and `apps/victory/build.rs` keeps page 0 as
            // `__PAGEZERO`), and no buffer object is at 0 (a resource at address 0 is refused), so
            // neither a pool nor a wrap of the title's pages can ever hold it. Taken for memory in
            // no pool, it was tried again after every change to the pool table, which world
            // streaming makes all the time. Not a view whose title pages are followed either.
            Decoded::Buffer { addr: 0, .. } => {
                heap.title_views.borrow_mut().remove(&index);
                let null = self.null_view()?;
                self.backend.write_buffer(&heap.heap, index, &null.buf, null.ptr().cast_const(), 0);
                // No typed view over the null view: one pending from the entry's last record goes.
                heap.typed_pending.borrow_mut().set(index, false, false);
                heap.shows(index, None);
            }
            Decoded::Buffer { addr, size } => {
                // Read before resolving: a pool mapped by a title thread after a failed resolve
                // must bump past these, so the entry is retried.
                let (pool_gen, res_gen) = (self.memory.generation(), self.resources_epoch.get());
                let alias = || self.title_alias(addr, u64::from(size)).map(|(b, p, sig)| (b, p, Some(sig)));
                let found = self.resolve(addr, u64::from(size)).map(|(b, p)| (b, p, None));
                let (buf, ptr) = match found.map_or_else(|e| alias().ok_or(e), Ok) {
                    Ok((buf, ptr, wrapped)) => {
                        let mut views = heap.title_views.borrow_mut();
                        match wrapped {
                            Some(sig) => {
                                views.insert(index, (addr, u64::from(size), Some(sig)));
                            }
                            None => {
                                views.remove(&index);
                            }
                        }
                        (buf, ptr)
                    }
                    Err(e) => {
                        // Tried again once the pages under it change: a gap the title maps
                        // later, say ([`Self::follow_title_pages`]).
                        if addr != 0 && size != 0 {
                            let sig = self.alias_signature(addr, u64::from(size));
                            heap.title_views.borrow_mut().insert(index, (addr, u64::from(size), sig));
                        }
                        let null = self.null_view()?;
                        self.backend.write_buffer(
                            &heap.heap,
                            index,
                            &null.buf,
                            null.ptr().cast_const(),
                            0,
                        );
                        heap.typed_pending.borrow_mut().set(index, false, false);
                        // The null view has replaced whatever the cached record used to show.
                        heap.mirrored.borrow_mut().remove(&index);
                        heap.pending_mirror.borrow_mut().insert(
                            index,
                            (
                                record_at(handle),
                                pool_gen,
                                res_gen,
                            ),
                        );
                        heap.shows(index, None);
                        return Err(e);
                    }
                };
                self.backend
                    .write_buffer(&heap.heap, index, &buf, ptr, size);
                let (is_typed, is_explicit) = match typed_buffer(&view, &resources) {
                    Some((_, _, explicit)) => (true, explicit),
                    None => (false, false),
                };
                heap.typed_pending
                    .borrow_mut()
                    .set(index, is_typed, is_explicit);
                heap.shows(index, None);
            }
            Decoded::Image {
                resource,
                mips,
                slices,
                format,
            } => {
                let Some(Res::Texture(tex)) = resources.get(&resource) else {
                    // An evicted texture: made again at the next submission, which writes this
                    // entry then (registering bumps the epoch its retry waits for). Until then the
                    // entry shows nothing rather than what it showed before.
                    if self.evicted.borrow().contains(&resource) {
                        self.wanted.borrow_mut().push(resource);
                        self.backend.clear_entry(&heap.heap, index);
                        heap.shows(index, None);
                    }
                    return Ok(false);
                };
                // A colour view of a depth texture that compute writes shows its twin. Twins are
                // made at unordered views of texture objects only (a placed view's key is no
                // object to read the description of); a placed view shows an existing one.
                let unordered = matches!(view, crate::descriptor::View::Unordered(..));
                let twin = format.and_then(|f| self.depth_twin(resource, tex, f, unordered));
                let (tex, format) = match &twin {
                    Some((t, f)) => (t, Some(*f)),
                    None => (tex, format),
                };
                if matches!(view, crate::descriptor::View::Unordered(..) | crate::descriptor::View::Placed(true, ..)) {
                    self.unordered.borrow_mut().insert(resource);
                }
                match (mips, format) {
                    (Some((first, count)), Some(format)) if count > 0 => {
                        let (dropped, host_mips) = self.dropped.borrow().get(&resource).copied().unwrap_or((0, 0));
                        let mips = super::textures::host_mip_range(first as usize, count as usize, dropped, host_mips);
                        let v = self.backend.create_texture_view(
                            tex,
                            &ViewDesc {
                                format,
                                mips,
                                slices: slices.map_or((0, usize::MAX), |(first, count)| {
                                    (first as usize, count as usize)
                                }),
                            },
                        )?;
                        self.backend.write_texture(&heap.heap, index, &v, 0.0);
                    }
                    _ => self.backend.write_texture(&heap.heap, index, tex, 0.0),
                }
                heap.typed_pending.borrow_mut().set(index, false, false);
                heap.shows(index, Some(resource));
            }
            Decoded::Sampler { desc, lod_bias } => {
                let s = self.backend.create_sampler(&desc)?;
                self.backend.write_sampler(&heap.heap, index, &s, lod_bias);
                heap.typed_pending.borrow_mut().set(index, false, false);
                heap.shows(index, None);
            }
        }
        heap.pending_mirror.borrow_mut().remove(&index);
        heap.mirrored.borrow_mut().insert(index, record_at(handle));
        Ok(true)
    }

    /// Minimum entries of each bound table checked at submission when a shader has no
    /// metadata for that table (or declares fewer entries).
    const TABLE_CHECK: usize = 16;

    /// Turns the collection of GPU-written texture keys on or off.
    pub(super) fn collect_written(&self, on: bool) {
        *self.written.borrow_mut() = on.then(Default::default);
    }

    pub(super) fn collecting_written(&self) -> bool {
        self.written.borrow().is_some()
    }

    /// The keys collected since the last call.
    pub(super) fn take_written(&self) -> Vec<usize> {
        self.written.borrow_mut().as_mut().map_or_else(Vec::new, |w| w.drain().collect())
    }

    /// The textures dispatches prepared since the last call bound for unordered access, copy-back
    /// or not ([`Self::note_written`]).
    #[cfg(target_os = "macos")]
    pub(super) fn take_gpu_written(&self) -> Vec<usize> {
        self.gpu_written.borrow_mut().drain().collect()
    }

    /// Notes the textures the tables `root` binds hold unordered access views of: what a prepared
    /// dispatch may write. More than it does write (the shader may not use every entry).
    ///
    /// shortcut: a shader that indexes the descriptor heap itself, and pixel shader writes, are
    /// not seen; add them when a title texture needs it.
    fn note_written(&self, root: &[Option<Root<B>>], binding: Option<&Binding>) {
        let mut written = Vec::new();
        for (slot, r) in root.iter().enumerate() {
            let Some(Root::Table(h, at)) = r else { continue };
            if h.kind != HeapKind::Resource {
                continue;
            }
            let end = (*at + table_check_count(slot, binding, Self::TABLE_CHECK)).min(h.count);
            for index in *at..end {
                match crate::descriptor::view(h.start + index as u64 * Self::step(h.kind)) {
                    Some(crate::descriptor::View::Unordered(r, ..)) => written.push(r),
                    Some(crate::descriptor::View::Placed(true, res, d)) if res[0] > 1 => {
                        let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                        written.push(super::textures::placed_key(address, &res));
                    }
                    _ => {}
                }
            }
        }
        if let Some(back) = self.written.borrow_mut().as_mut() {
            back.extend(written.iter().copied());
        }
        self.gpu_written.borrow_mut().extend(written);
    }

    /// The entries of the table bound at root `slot` from entry `at` that a submission checks
    /// ([`Self::remirror_tables`]): as many as the shader's table ranges reach, at least
    /// [`Self::TABLE_CHECK`], within the heap.
    fn table_span(h: &NativeHeap<B>, slot: usize, at: usize, binding: Option<&Binding>) -> std::ops::Range<usize> {
        at..(at + table_check_count(slot, binding, Self::TABLE_CHECK)).min(h.count)
    }

    /// The resources the records of the tables `lists`' draws and dispatches bind name now: the
    /// entries [`Self::remirror_tables`] mirrors when the lists are encoded, so a texture named
    /// there must be made before (the texture budget).
    pub(crate) fn resources_named<'a>(
        &self,
        lists: impl Iterator<Item = &'a NativeList<B>>,
    ) -> std::collections::HashSet<usize>
    where
        B: 'a,
    {
        use crate::descriptor::View;
        let (mut tables, mut named) = (std::collections::HashSet::new(), std::collections::HashSet::new());
        for cmd in lists.flat_map(|l| &l.cmds) {
            let (root, binding) = match cmd {
                Cmd::Dispatch { root, pso, .. } => (root, pso.binding.as_ref()),
                Cmd::Draw { root, binding, .. } => (root, binding.as_deref()),
                _ => continue,
            };
            for (slot, r) in root.iter().enumerate() {
                let Some(Root::Table(h, at)) = r else { continue };
                let span = Self::table_span(h, slot, *at, binding);
                if h.kind != HeapKind::Resource || !tables.insert((h.start, span.clone())) {
                    continue;
                }
                let step = Self::step(h.kind) as usize;
                let mut records = vec![0u8; span.len() * step];
                if !super::safe_read((h.start + (span.start * step) as u64) as *const u8, &mut records) {
                    continue;
                }
                for (i, record) in records.chunks(step).enumerate() {
                    if record[..12] == [0; 12] {
                        continue;
                    }
                    match crate::descriptor::view(h.start + ((span.start + i) * step) as u64) {
                        Some(View::Shader(r, _) | View::Unordered(r, ..)) => named.insert(r),
                        Some(View::Placed(_, res, d)) => {
                            named.insert(super::textures::placed_key(u64::from(d[10]) | u64::from(d[11]) << 32, &res))
                        }
                        _ => continue,
                    };
                }
            }
        }
        named
    }

    /// Whether any texture is given up under the texture budget and not made again.
    pub(crate) fn any_evicted(&self) -> bool {
        !self.evicted.borrow().is_empty()
    }

    /// Mirrors again the entries of each table `root` binds whose title record changed since
    /// they were last mirrored. The title copies records into its shader-visible heaps with plain
    /// stores (the content draws' constant buffers, from frame ~700), which no device call reports,
    /// so their host entries stayed empty and the shaders read zeros.
    fn remirror_tables(&self, root: &[Option<Root<B>>], binding: Option<&Binding>) {
        let pool_gen = self.memory.generation();
        let res_gen = self.resources_epoch.get();
        for (slot, r) in root.iter().enumerate() {
            let Some(Root::Table(h, at)) = r else { continue };
            let (mut checked, mut changed) = (0, 0);
            let step = Self::step(h.kind) as usize;
            let end = Self::table_span(h, slot, *at, binding).end;
            if heap_set_trace().is_some() {
                let frame = (crate::frame::current() as u32).saturating_add(1);
                let mut used = h.used.borrow_mut();
                if used.len() < end {
                    used.resize(h.count.max(end), 0);
                }
                used[*at..end].fill(frame);
            }
            // The whole span in one read (a system call each); entry by entry only if part of it
            // cannot be read.
            let mut span = vec![0u8; end.saturating_sub(*at) * step];
            let whole = super::safe_read((h.start + (*at * step) as u64) as *const u8, &mut span);
            for index in *at..end {
                let handle = h.start + (index * step) as u64;
                let record = if whole {
                    span[(index - *at) * step..][..12].try_into().expect("12 bytes")
                } else {
                    record_at(handle)
                };
                checked += 1;
                let seen = h.mirrored.borrow().get(&index).copied();
                // An empty record is mirrored only if a view was mirrored there before: the title
                // zeroed it with stores, and its view-stride dword must go to 0.
                if seen == Some(record)
                    || (record == [0; 12] && seen.is_none())
                    || h.pending_mirror.borrow().get(&index) == Some(&(record, pool_gen, res_gen))
                {
                    continue;
                }
                changed += 1;
                // shortcut: views of resources the host has not realized yet are not made here
                // (`glue::mirror` does that for device writes); such an entry stays as it was.
                match self.mirror(handle) {
                    Ok(true) => {}
                    Ok(false) => {
                        h.pending_mirror
                            .borrow_mut()
                            .insert(index, (record, pool_gen, res_gen));
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, handle = format!("{handle:#x}"), "a table entry was not mirrored at submission");
                    }
                }
            }
            super::descriptor_trace::record(h.start, checked, changed);
        }
    }

    /// Promotes unmaterialized buffer entries to `texture_buffer` views lazily when the bound
    /// shader uses DXIL `TypedBuffer` operations: only the bound `Root::Table` spans, plus
    /// explicit (`stride == 0`) typed buffer views in `heap` when the shader also indexes
    /// `ResourceDescriptorHeap` directly.
    fn materialize_typed_buffers(
        &self,
        root: &[Option<Root<B>>],
        binding: Option<&Binding>,
        heap: Option<&Rc<NativeHeap<B>>>,
    ) {
        let any_typed = binding
            .is_none_or(|b| b.typed_buffers || b.ps.as_ref().is_some_and(|p| p.typed_buffers));
        if !any_typed {
            return;
        }
        let promote = |h: &NativeHeap<B>, index: usize| {
            let handle = h.start + index as u64 * Self::step(h.kind);
            let Some(view) = crate::descriptor::view(handle) else { return };
            let resources = self.resources.borrow();
            let (Some(Decoded::Buffer { addr, size }), Some((format, writable, _))) =
                (decode(&view, &resources), typed_buffer(&view, &resources))
            else {
                return;
            };
            drop(resources);
            let resolved = self
                .resolve(addr, u64::from(size))
                .ok()
                .or_else(|| self.title_alias(addr, u64::from(size)).map(|(b, p, _)| (b, p)));
            if let Some((buf, ptr)) = resolved
                && self.backend.write_typed_buffer(
                    &h.heap,
                    index,
                    &buf,
                    ptr,
                    size,
                    format,
                    writable,
                )
            {
                self.typed_views.set(self.typed_views.get() + 1);
            }
        };
        let mut indices = Vec::new();
        for (slot, r) in root.iter().enumerate() {
            let Some(Root::Table(h, at)) = r else { continue };
            if h.kind != HeapKind::Resource {
                continue;
            }
            let check = table_check_count(slot, binding, Self::TABLE_CHECK);
            let end = (*at + check).min(h.count);
            indices.clear();
            h.typed_pending.borrow_mut().drain_range(*at, end, &mut indices);
            for &index in &indices {
                promote(h, index);
            }
        }
        let heap_typed = binding.is_some_and(|b| {
            b.heap_typed_buffers || b.ps.as_ref().is_some_and(|p| p.heap_typed_buffers)
        });
        if heap_typed
            && let Some(h) = heap
            && h.typed_pending.borrow().has_explicit()
        {
            indices.clear();
            h.typed_pending.borrow_mut().drain_explicit(&mut indices);
            for &index in &indices {
                promote(h, index);
            }
        }
    }

    /// The arguments of a translated title shader: its user-data block, then its root views.
    /// `threads`: the dispatch's thread counts, for a shader that reads them.
    fn translated_args(
        &self,
        root: &[Option<Root<B>>],
        b: &Binding,
        heap: Option<&Rc<NativeHeap<B>>>,
        threads: [u32; 3],
        p: &mut Prep<B>,
    ) -> Result<(), String> {
        use victory_shader::emit::{DISPATCH_THREADS, ROOT_SRV_VIEW, ROOT_UAV_VIEW, ROOT_VIEW_OFFSET};
        use victory_shader::RootType;
        let set = |slot: u8| {
            root.get(usize::from(slot))
                .and_then(|r| r.as_ref())
                .ok_or_else(|| format!("root parameter {slot} {NEVER_SET}"))
        };
        let mut args = std::mem::take(&mut p.root);
        if b.dwords > 0 {
            let mut values = vec![0u32; b.dwords];
            for u in &b.user_data {
                if u.slot == DISPATCH_THREADS {
                    values[u.index as usize] = threads.get(u.dword as usize).copied().unwrap_or(u32::MAX);
                    continue;
                }
                values[u.index as usize] = match (u.root, set(u.slot)?) {
                    (RootType::Constants, Root::Constants(c)) => {
                        c.get(u.dword as usize).copied().unwrap_or(0)
                    }
                    // The byte offset of the table's first entry in its heap.
                    (RootType::Table, Root::Table(h, at)) => (*at as u64 * Self::step(h.kind)) as u32,
                    (RootType::Descriptor, Root::View(a)) if u.dword < 2 => {
                        (a >> (32 * u.dword)) as u32
                    }
                    // A view of the pool holding the root view, and the address's offset in it.
                    (RootType::Descriptor, Root::View(a))
                        if matches!(u.dword, ROOT_SRV_VIEW | ROOT_UAV_VIEW | ROOT_VIEW_OFFSET) =>
                    {
                        let heap = match &p.heap {
                            Some(h) => h.clone(),
                            None => match heap {
                                Some(h) => h.clone(),
                                None => self.host_heap()?,
                            },
                        };
                        p.heap = Some(heap.clone());
                        let (view, offset) = self.root_view(&heap, *a);
                        match u.dword {
                            ROOT_SRV_VIEW => view,
                            ROOT_UAV_VIEW => view + 1,
                            _ => offset,
                        }
                    }
                    (kind, _) => {
                        return Err(format!(
                            "root parameter {} is not what the shader reads it as ({kind:?}, dword {})",
                            u.slot, u.dword
                        ));
                    }
                };
            }
            args = args.constants(&values);
        }
        for &(slot, _) in &b.views {
            // The translator's own table, not a title root parameter: the strides of the views
            // in the heap the shader indexes.
            if slot == victory_shader::emit::VIEW_STRIDES {
                let heap = match (&p.heap, heap) {
                    (Some(h), _) | (None, Some(h)) => h.clone(),
                    (None, None) => {
                        let h = self.host_heap()?;
                        p.heap = Some(h.clone());
                        h
                    }
                };
                let pages = self.view_strides(&heap)?;
                let gpu = self.backend.gpu_address(&pages.buf, pages.ptr());
                p.bufs.push(pages.buf.clone());
                args = args.address(gpu);
                continue;
            }
            let Root::View(a) = set(slot)? else {
                return Err(format!("root parameter {slot} is not a root view"));
            };
            let (buf, ptr) = self.resolve(*a, 1)?;
            let gpu = self.backend.gpu_address(&buf, ptr);
            p.bufs.push(buf);
            args = args.address(gpu);
        }
        p.root = args;
        Ok(())
    }

    /// A heap entry's dword of the view-stride table ([`super::descriptors::stride_word`]).
    fn stride_word(&self, view: Option<&crate::descriptor::View>) -> u32 {
        view.map_or(0, |v| super::descriptors::stride_word(v, &self.resources.borrow()))
    }

    /// The view-stride table of `heap` (the translator's root SRV `t255`, which a shader that
    /// takes a buffer view from the resource heap itself looks up before it reads the entry).
    /// One dword for every index a shader can compute (`HEAP_VIEWS`): the buffer view's element
    /// size at the title's entries, 0 elsewhere (the shader then reads the null buffer view).
    /// Made at first use from the heap as the title's memory has it now and kept current by
    /// [`Self::mirror`] (also for a bound table's record the title zeroed with stores,
    /// [`Self::remirror_tables`]). shortcut: an entry the title copied or zeroed as bytes outside
    /// a bound table keeps its old dword until it is mirrored.
    fn view_strides(&self, heap: &NativeHeap<B>) -> Result<Rc<Pages<B>>, String> {
        if let Some(made) = &*heap.strides.borrow() {
            return Ok(made.clone());
        }
        self.write_null_buffers(heap)?;
        let slots = heap.count.max(victory_shader::emit::HEAP_VIEWS as usize);
        let pages = Rc::new(Pages::new(&self.backend, slots * 4)?);
        let step = Self::step(heap.kind);
        for index in 0..heap.count {
            let view = crate::descriptor::view(heap.start + index as u64 * step);
            if view.is_some() {
                pages.write_at(index, &[self.stride_word(view.as_ref())]);
            }
        }
        *heap.strides.borrow_mut() = Some(pages.clone());
        Ok(pages)
    }

    /// The page table of the pool-offset contract for `heap` and the pool table as it is now:
    /// rebuilt, into a buffer of its own, when a pool or a mapping changed since it was made.
    fn memory_table(&self, heap: &NativeHeap<B>) -> Result<Rc<Pages<B>>, String> {
        if !addr::fits(heap.count) {
            return Err(format!(
                "a resource heap of {} entries has no room for title memory views",
                heap.count
            ));
        }
        if let Some((g, t)) = &*heap.table.borrow()
            && *g == self.memory.generation()
        {
            return Ok(t.clone());
        }
        let (generation, maps) = self.mapping_list();
        let mut c = self.contract.borrow_mut();
        let words = addr::build(&maps, &mut c.pairs, heap.count as u32);
        if std::env::var_os("VICTORY_TRACE_TABLE_HOLES").is_some() {
            static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
            let (pages, bytes) = addr::split_pages(&maps);
            if LAST.swap(bytes, std::sync::atomic::Ordering::Relaxed) != bytes {
                tracing::info!(pages, bytes, mappings = maps.len(), "title pages mapped in part or in pieces read as zeros through the page table");
            }
        }
        let table = Rc::new(Pages::new(&self.backend, words.len() * 4)?);
        table.write(&words);
        // The table it replaces stays with the submissions that read it ([`Pages`]).
        *heap.table.borrow_mut() = Some((generation, table.clone()));
        Ok(table)
    }

    /// The memory of the null view (a page, shown with size 0), made at first use. Borrows the
    /// contract only for the call.
    fn null_view(&self) -> Result<Rc<Pages<B>>, String> {
        let mut c = self.contract.borrow_mut();
        if c.null.is_none() {
            c.null = Some(Rc::new(Pages::new(&self.backend, 1)?));
        }
        Ok(c.null.clone().expect("made above"))
    }

    /// Writes the view pairs the table names into `heap`'s reserved entries (those it lacks).
    fn write_pairs(&self, heap: &NativeHeap<B>) -> Result<(), String> {
        let null = self.null_view()?;
        let c = self.contract.borrow();
        let (epoch, have) = heap.pairs.get();
        let have = if epoch == c.pairs.epoch { have } else { 0 };
        let want = c.pairs.keys.len() + 1;
        for n in have..want {
            let index = addr::pair_index(heap.count as u32, n as u32) as usize;
            // Pair 0, and a pair whose pool was released: the null view.
            let (buffer, ptr, size) = match n.checked_sub(1).and_then(|k| c.pairs.keys[k]) {
                None => (null.buf.clone(), null.ptr().cast_const(), 0),
                Some(key) => {
                    let buffer = self.pool_buffer(key.pool, key.base, key.len as usize)?;
                    let (from, size) = Pairs::span(&key);
                    (buffer, (key.base + from as usize) as *const u8, size)
                }
            };
            // Both of the pair name the same bytes (the converter's descriptors are the same).
            for i in [index, index + 1] {
                self.backend.write_buffer(&heap.heap, i, &buffer, ptr, size);
            }
        }
        heap.pairs.set((c.pairs.epoch, want));
        Ok(())
    }

    /// The pair view and the offset in it of root view address `a` (see [`addr::root_view`]).
    fn root_view(&self, heap: &NativeHeap<B>, a: u64) -> (u32, u32) {
        let mapping = self.memory.mapping_at(a);
        addr::mapped_root_view(mapping.as_ref(), &mut self.contract.borrow_mut().pairs, heap.count as u32, a)
    }

    /// The pool table in title-address order, and its generation: made again only
    /// when the pool table changed.
    fn mapping_list(&self) -> (u64, Rc<[Mapping]>) {
        if let Some((g, maps)) = &*self.maps.borrow()
            && *g == self.memory.generation()
        {
            return (*g, maps.clone());
        }
        let (generation, mappings) = self.memory.mappings();
        let maps: Rc<[Mapping]> = mappings
            .iter()
            .map(|(at, len, pool, offset)| Mapping {
                at: *at as u64,
                len: *len as u64,
                pool: pool.id,
                base: pool.base,
                pool_len: pool.len as u64,
                offset: *offset as u64,
            })
            .collect();
        *self.maps.borrow_mut() = Some((generation, maps.clone()));
        (generation, maps)
    }

    /// What a shader that finds title memory through the page table is given besides its user data:
    /// the table (the last root parameter) and, in the heap it reads, the view pairs.
    fn memory_args(
        &self,
        heap: Option<&Rc<NativeHeap<B>>>,
        p: &mut Prep<B>,
    ) -> Result<(), String> {
        let heap = match heap {
            Some(h) => h.clone(),
            None => self.host_heap()?,
        };
        let table = self.memory_table(&heap)?;
        self.write_pairs(&heap)?;
        let gpu = self.backend.gpu_address(&table.buf, table.ptr());
        p.bufs.push(table.buf.clone());
        p.root = std::mem::take(&mut p.root).address(gpu);
        p.heap = Some(heap);
        Ok(())
    }

    /// What a shader that reads views' title addresses or texture sizes from descriptors is given
    /// after the page table: the view addresses of the heap it reads.
    fn view_address_args(&self, heap: Option<&Rc<NativeHeap<B>>>, p: &mut Prep<B>) -> Result<(), String> {
        let heap = match (&p.heap, heap) {
            (Some(h), _) | (None, Some(h)) => h.clone(),
            (None, None) => self.host_heap()?,
        };
        let table = self.view_addresses(&heap)?;
        let gpu = self.backend.gpu_address(&table.buf, table.ptr());
        p.bufs.push(table.buf.clone());
        p.root = std::mem::take(&mut p.root).address(gpu);
        p.heap = Some(heap);
        Ok(())
    }

    /// The four dwords of a heap entry in the view addresses: a buffer view's title address
    /// (dword 0, and dword 1 bits 0-15), stride (dword 1 bits 16-29: a structured or typed view's
    /// element size, 0 for a raw view or a constant buffer) and record count (dword 2: elements
    /// with a stride, else bytes) and dword 3 (the console's untyped one,
    /// [`victory_shader::emit::VIEW_BUFFER_DWORD3`], with the view's out-of-bounds mode), as the
    /// console's buffer descriptor has them; a texture view's
    /// size fields ([`crate::descriptor::texture_words`]); zeros for anything else.
    fn address_words(&self, view: Option<&crate::descriptor::View>) -> [u32; 4] {
        let Some(view) = view else { return [0; 4] };
        if let Some(words) = crate::descriptor::texture_words(view) {
            return words;
        }
        let resources = self.resources.borrow();
        match decode(view, &resources) {
            Some(Decoded::Buffer { addr, size }) => {
                // A raw view's element is 4 in the stride table; its descriptor has no stride.
                let stride = match super::descriptors::raw_buffer(view) {
                    true => 0,
                    false => {
                        super::descriptors::stride_word(view, &resources)
                            & victory_shader::emit::VIEW_STRIDE_BITS
                    }
                };
                let records = size.checked_div(stride).unwrap_or(size);
                let high = (addr >> 32) as u32 & 0xFFFF | (stride & 0x3FFF) << 16;
                use victory_shader::emit::{VIEW_BUFFER_DWORD3, VIEW_BUFFER_OOB_RAW, VIEW_BUFFER_OOB_STRUCTURED};
                let oob = if stride > 0 { VIEW_BUFFER_OOB_STRUCTURED } else { VIEW_BUFFER_OOB_RAW };
                [addr as u32, high, records, VIEW_BUFFER_DWORD3 | oob]
            }
            _ => [0; 4],
        }
    }

    /// The view addresses of `heap` (the translator's root SRV `t1` of the memory space,
    /// `victory_shader::emit`, "View addresses"): per title entry, four dwords of the console
    /// descriptor as far as the host knows them ([`Self::address_words`]), so a shader that walks
    /// a BVH from its SRV's address, or sizes a texture from its descriptor, reads them. 16 bytes
    /// an entry. Made at first use from the heap as the title's memory has it now and kept
    /// current by [`Self::mirror`]. shortcut: an entry the title copied as bytes outside a bound
    /// table keeps its old words until it is mirrored.
    fn view_addresses(&self, heap: &NativeHeap<B>) -> Result<Rc<Pages<B>>, String> {
        if let Some(made) = &*heap.addresses.borrow() {
            return Ok(made.clone());
        }
        let pages = Rc::new(Pages::new(&self.backend, heap.count.max(1) * 16)?);
        let step = Self::step(heap.kind);
        for index in 0..heap.count {
            let view = crate::descriptor::view(heap.start + index as u64 * step);
            if view.is_some() {
                pages.write_at(4 * index, &self.address_words(view.as_ref()));
            }
        }
        *heap.addresses.borrow_mut() = Some(pages.clone());
        Ok(pages)
    }

    /// A resource heap of the host's own, for work that needs the view pairs and has no heap.
    fn host_heap(&self) -> Result<Rc<NativeHeap<B>>, String> {
        if let Some(h) = &self.contract.borrow().heap {
            return Ok(h.clone());
        }
        let heap = Rc::new(NativeHeap {
            heap: self.backend.create_heap(HeapKind::Resource, Self::resource_entries(0))?,
            kind: HeapKind::Resource,
            start: 0,
            count: 0,
            pairs: Default::default(),
            table: Default::default(),
            strides: Default::default(),
            addresses: Default::default(),
            mirrored: Default::default(),
            pending_mirror: Default::default(),
            title_views: Default::default(),
            images: Default::default(),
            used: Default::default(),
            typed_pending: Default::default(),
        });
        self.contract.borrow_mut().heap = Some(heap.clone());
        Ok(heap)
    }

    fn root_args(&self, root: &[Option<Root<B>>], p: &mut Prep<B>) -> Result<(), String> {
        for (i, r) in root.iter().enumerate() {
            let args = std::mem::take(&mut p.root);
            p.root = match r
                .as_ref()
                .ok_or_else(|| format!("root parameter {i} {NEVER_SET}"))?
            {
                Root::Constants(w) => args.constants(w),
                Root::View(a) => {
                    let (buf, ptr) = self.resolve(*a, 1)?;
                    let gpu = self.backend.gpu_address(&buf, ptr);
                    p.bufs.push(buf);
                    args.address(gpu)
                }
                Root::Table(h, at) => args.address(self.backend.heap_address(&h.heap, *at)),
            };
        }
        Ok(())
    }

    /// `VICTORY_TRACE_TARGETS=1`: the root parameters of the first draws that have vertex buffers,
    /// and the title memory they name (descriptor words, then the first 64 bytes at the address a
    /// constant buffer descriptor holds).
    fn trace_draw(&self, root: &[Option<Root<B>>], binding: Option<&Binding>, dispatch: bool) {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
        if !env_set!("VICTORY_TRACE_TARGETS") || N[usize::from(dispatch)].fetch_add(1, Ordering::Relaxed) >= 6 {
            return;
        }
        self.trace_args(root, binding);
    }

    /// `VICTORY_TRACE_ROOT_SPAN[=<bytes>]`: each root SRV or UAV a translated shader binds whose
    /// title range stays linear in its pool for less than `<bytes>` (default 1 MiB) from its address:
    /// what the shader reaches past that is not what the title has there when it reads the root
    /// view through a pair view (`through = "pair"`), linear in the pool. Once per pipeline, root
    /// parameter and mapping; at most 2000 lines of each kind.
    /// Programs reading title memory (`through = "table"`) look each access up in the page table.
    fn trace_root_span(&self, root: &[Option<Root<B>>], binding: Option<&Binding>, groups: &[usize; 3]) {
        use victory_shader::emit::{ROOT_SRV_VIEW, ROOT_UAV_VIEW};
        static LIMIT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
        let Some(limit) = *LIMIT.get_or_init(|| {
            let v = std::env::var("VICTORY_TRACE_ROOT_SPAN").ok()?;
            Some(v.parse().ok().filter(|&n| n > 1).unwrap_or(1 << 20))
        }) else {
            return;
        };
        type Seen = [std::collections::HashSet<(Rc<str>, u8, u64)>; 2];
        thread_local! {
            static SORTED: RefCell<(u64, Vec<usize>)> = RefCell::default();
            static SEEN: RefCell<Seen> = RefCell::default();
        }
        let (generation, maps) = self.mapping_list();
        for b in binding.into_iter().chain(binding.and_then(|b| b.ps.as_deref())) {
            for u in &b.user_data {
                let (through, k) = match u.dword {
                    ROOT_SRV_VIEW | ROOT_UAV_VIEW => ("pair", 0),
                    0 if b.memory && u.root == victory_shader::RootType::Descriptor => ("table", 1),
                    _ => continue,
                };
                let Some(Some(Root::View(a))) = root.get(usize::from(u.slot)) else { continue };
                let run = SORTED.with_borrow_mut(|(g, sorted)| {
                    if *g != generation || sorted.len() != maps.len() {
                        *sorted = (0..maps.len()).collect();
                        sorted.sort_by_key(|&i| maps[i].at);
                        *g = generation;
                    }
                    addr::linear_run(&maps, sorted, *a, limit)
                });
                let Some((run, m, next)) = run.filter(|r| r.0 < limit) else { continue };
                if !SEEN.with_borrow_mut(|s| s[k].len() < 2000 && s[k].insert((b.name.clone(), u.slot, m.at))) {
                    continue;
                }
                // Whose bytes the pool offsets past the run are: what a pair view reaches there.
                let pool_end = m.offset + (a + run - m.at);
                let linear = maps.iter().find(|n| n.pool == m.pool && pool_end >= n.offset && pool_end - n.offset < n.len);
                tracing::info!(
                    pipeline = &*b.name,
                    slot = u.slot,
                    uav = u.dword == ROOT_UAV_VIEW,
                    through,
                    ?groups,
                    address = format!("{a:#x}"),
                    run = format!("{run:#x}"),
                    mapping = format!("{:#x}+{:#x} pool {} at {:#x}", m.at, m.len, m.pool, m.offset),
                    title_next = ?next.map(|n| format!("pool {} at {:#x}", n.pool, n.offset)),
                    pool_next_is = ?linear.map(|n| format!("{:#x}", n.at + (pool_end - n.offset))),
                    "a root view's range leaves its pool's linear run"
                );
            }
        }
    }

    /// The root parameters and binding [`Self::trace_draw`] logs, for any one draw.
    pub(super) fn trace_args(&self, root: &[Option<Root<B>>], binding: Option<&Binding>) {
        let read = |at: u64, n: usize| {
            let mut b = vec![0u8; n];
            let ok = super::safe_read(at as *const u8, &mut b);
            let words: Vec<String> = b.chunks(4).map(|w| format!("{:08x}", u32::from_le_bytes(w.try_into().unwrap()))).collect();
            format!("{}{}", if ok { "" } else { "UNREADABLE " }, words.join(" "))
        };
        for (i, r) in root.iter().enumerate() {
            match r {
                Some(Root::Constants(w)) => tracing::info!(slot = i, ?w, "draw root constants"),
                Some(Root::View(a)) => tracing::info!(slot = i, addr = format!("{a:#x}"), mem = read(*a, 64), "draw root view"),
                Some(Root::Table(h, at)) => {
                    let base = h.start + *at as u64 * Self::step(h.kind);
                    let d = read(base, 32 * 4);
                    tracing::info!(slot = i, heap_start = format!("{:#x}", h.start), index = at, kind = ?h.kind, descriptors = d, "draw root table");
                    // `VICTORY_TRACE_TABLE_ENTRIES=<n>`: how many entries of each table (4 if unset).
                    static ENTRIES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                    let n = *ENTRIES.get_or_init(|| {
                        std::env::var("VICTORY_TRACE_TABLE_ENTRIES").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
                    });
                    for k in 0..n {
                        let handle = base + k * Self::step(h.kind);
                        match crate::descriptor::view(handle) {
                            Some(crate::descriptor::View::Constant(a, n)) => tracing::info!(entry = k, addr = format!("{a:#x}"), size = n, mem = read(a, 64), "table constant buffer"),
                            Some(crate::descriptor::View::Shader(r, d)) => {
                                #[cfg(target_os = "macos")]
                                let info = crate::resource::texture_info(r as crate::object::This);
                                #[cfg(not(target_os = "macos"))]
                                let info: Option<super::textures::TextureInfo> = None;
                                let (nz, total) = info.as_ref().map_or((0, 0), |i| {
                                    let m = super::textures::read_memory(i.memory, i.size.min(1 << 22));
                                    (m.iter().filter(|&&b| b != 0).count(), m.len())
                                });
                                tracing::info!(entry = k, resource = format!("{r:#x}"), desc = ?d, texture = ?info.as_ref().map(|i| (i.format, i.width, i.height, i.memory as usize, i.size)), nonzero_bytes = nz, of = total, "table texture");
                            }
                            Some(crate::descriptor::View::Unordered(r, counter, d)) => {
                                let info = texture_info_of(r);
                                let buffer = self.resources.borrow().get(&r).map(|b| match b {
                                    Res::Buffer(a, n) => (*a, *n),
                                    _ => (0, 0),
                                });
                                tracing::info!(entry = k, resource = format!("{r:#x}"), counter, desc = ?d, texture = ?info.as_ref().map(|i| (i.format, i.width, i.height, i.memory as usize, i.size)), buffer = ?buffer, "table unordered view");
                            }
                            Some(crate::descriptor::View::Placed(_, res, d)) => {
                                let at = u64::from(d[10]) | u64::from(d[11]) << 32;
                                let mem = if res[0] == 1 { read(at, 128) } else { String::new() };
                                tracing::info!(entry = k, ?res, ?d, mem, "table placed view")
                            }
                            Some(v) => tracing::info!(entry = k, ?v, "table view"),
                            None => {}
                        }
                    }
                }
                None => {}
            }
        }
        if let Some(b) = binding {
            tracing::info!(dwords = b.dwords, user_data = ?b.user_data, views = ?b.views, ps = ?b.ps.as_ref().map(|p| p.user_data.clone()), "draw binding");
        }
    }

    /// The textures the first 32 entries from each of `root`'s tables name as unordered-access
    /// views (for traces of work whose program is not known: a table may be shorter).
    #[cfg(target_os = "macos")]
    pub(super) fn table_uav_textures(&self, root: &[Option<Root<B>>]) -> Vec<usize> {
        let handles = root.iter().flatten().filter_map(|r| match r {
            Root::Table(h, at) if h.kind == HeapKind::Resource => Some((h.start, *at as u64)),
            _ => None,
        });
        let mut out: Vec<usize> = handles
            .flat_map(|(start, at)| (0..32).map(move |k| start + (at + k) * Self::step(HeapKind::Resource)))
            .filter_map(|handle| match crate::descriptor::view(handle) {
                Some(crate::descriptor::View::Unordered(r, _, _)) if texture_info_of(r).is_some() => Some(r),
                _ => None,
            })
            .collect();
        out.dedup();
        out
    }

    /// Pass trace: the textures `root`'s tables name: shader-resource ones, unordered-access ones, and
    /// how many of the entries hold no view at all. `limits`: entries per table slot (else 24).
    #[cfg(target_os = "macos")]
    pub(super) fn pass_views(&self, root: &[Option<Root<B>>], limits: &[(u8, usize)]) -> (Vec<String>, Vec<String>, usize, Vec<usize>) {
        use crate::descriptor::View;
        let (mut reads, mut writes, mut none, mut keys) = (Vec::new(), Vec::new(), 0usize, Vec::new());
        let d = |r: usize, i: Option<super::textures::TextureInfo>| match i {
            Some(i) => format!("{r:#x}:f{}:{}x{}@{:#x}", i.format, i.width, i.height, i.memory as usize),
            None => format!("{r:#x}:buf"),
        };
        for (slot, r) in root.iter().enumerate() {
            let Some(Root::Table(h, at)) = r else { continue };
            if h.kind != HeapKind::Resource {
                continue;
            }
            let n = limits.iter().filter(|(s, _)| *s as usize == slot).map(|(_, n)| *n).max().unwrap_or(if limits.is_empty() { 24 } else { 0 });
            // Not past the heap's end; entries the shader names beyond it are counted.
            let past = n.saturating_sub(h.count.saturating_sub(*at));
            if past > 0 {
                reads.push(format!("{slot}.past-heap={past}@{at}/{}", h.count));
            }
            for k in 0..n.min(h.count.saturating_sub(*at)) as u64 {
                match crate::descriptor::view(h.start + (*at as u64 + k) * Self::step(HeapKind::Resource)) {
                    Some(View::Shader(r, desc)) => match texture_info_of(r) {
                        Some(i) => { keys.push(r); reads.push(format!("{slot}.{k}={}", d(r, Some(i)))); }
                        // Named too, so a table entry the host cannot draw from shows in the trace.
                        None => {
                            let buffer = matches!(self.resources.borrow().get(&r), Some(Res::Buffer(..)));
                            reads.push(format!("{slot}.{k}={}", untextured_srv(r, buffer, desc)));
                        }
                    },
                    Some(View::Unordered(r, _, _)) => {
                        if texture_info_of(r).is_some() { keys.push(r); }
                        writes.push(format!("{slot}.{k}={}", d(r, texture_info_of(r))));
                    }
                    Some(View::Placed(u, res, v)) if res[0] > 1 => {
                        let address = u64::from(v[10]) | u64::from(v[11]) << 32;
                        let s = format!("{slot}.{k}=placed:{}", d(address as usize, placed_info_of(address, &res)));
                        if u { writes.push(s) } else { reads.push(s) }
                    }
                    // A constant buffer view: its address, size and first dwords now (up to 512 bytes).
                    Some(View::Constant(a, size)) => {
                        let mut w = vec![0u8; (size as usize).clamp(4, 512) & !3];
                        let ok = a != 0 && super::safe_read(a as *const u8, &mut w);
                        let words: Vec<String> = w.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes(c.try_into().unwrap()))).collect();
                        reads.push(format!("{slot}.{k}=cbv:{a:#x}+{size}:{}", if ok { words.join(",") } else { "unreadable".into() }));
                    }
                    Some(v @ View::Placed(u, res, _)) => {
                        // Dimension 1 is a buffer, which the host reads as memory: its range and first dwords (up to 512 bytes).
                        let s = match res[0] {
                            1 => match super::descriptors::decode(&v, &self.resources.borrow()) {
                                Some(super::descriptors::Decoded::Buffer { addr, size }) => {
                                    let mut w = vec![0u8; (size as usize).clamp(4, 512) & !3];
                                    let ok = addr != 0 && super::safe_read(addr as *const u8, &mut w);
                                    let words: Vec<String> = w.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes(c.try_into().unwrap()))).collect();
                                    format!("{slot}.{k}=placed:buf:{addr:#x}+{size}:desc{:?}:{}", if let View::Placed(_, _, d) = &v { &d[..10] } else { &[][..] }, if ok { words.join(",") } else { "unreadable".into() })
                                }
                                _ => format!("{slot}.{k}=placed:buf"),
                            },
                            n => format!("{slot}.{k}=placed-dim{n}"),
                        };
                        if u { writes.push(s) } else { reads.push(s) }
                    }
                    Some(_) => {}
                    None => none += 1,
                }
            }
        }
        (reads, writes, none, keys)
    }

    /// Pass trace: the table entries of a noted draw or dispatch as they are when it is submitted,
    /// against its `PASS` line (made when it was recorded). A `PASSSUB` line lists the entries that
    /// read differently now: the title rewrote them between recording and submitting.
    #[cfg(target_os = "macos")]
    fn pass_at_submit(&self, exec: u64, notes: &[(usize, String)], li: usize, ci: usize, c: &Cmd<B>) {
        // Only work that ran: a dropped draw's note reads its tables to other limits.
        let Some((_, note)) = notes.iter().find(|(at, n)| *at == ci && n.contains("out=\"ran")) else { return };
        let (root, limits): (&[Option<Root<B>>], Vec<(u8, usize)>) = match c {
            Cmd::Draw { root, binding, .. } => {
                let mut limits: Vec<(u8, usize)> = Vec::new();
                // As the `PASS` note read them (`glue::pass_note`).
                if let Some(b) = binding {
                    limits.extend(b.table_entries.iter().copied());
                    if let Some(ps) = &b.ps {
                        limits.extend(ps.table_entries.iter().copied());
                    }
                    if limits.is_empty() {
                        limits.push((255, 0));
                    }
                }
                (root, limits)
            }
            Cmd::Dispatch { root, .. } => (root, Vec::new()),
            _ => return,
        };
        let (reads, writes, _, _) = self.pass_views(root, &limits);
        let changed = changed_entries(note, reads.iter().chain(&writes));
        if !changed.is_empty() {
            eprintln!("PASSSUB exec={exec} li={li} ci={ci} changed={changed:?}");
        }
    }

    /// For `VICTORY_TRACE_RECORD_WRITERS`: what the first 64 entries from each of `root`'s tables
    /// name: the title memory (start, size) of buffers viewed for unordered access, the textures
    /// viewed as shader resources, and the textures viewed for unordered access. A table may be
    /// shorter, so a neighbour's views can be counted in; views of memory without a resource
    /// object are left out.
    #[cfg(target_os = "macos")]
    #[allow(clippy::type_complexity)]
    pub(super) fn table_views(&self, root: &[Option<Root<B>>]) -> (Vec<(u64, u64)>, Vec<usize>, Vec<usize>) {
        use crate::descriptor::View;
        let (mut buffers, mut reads, mut writes) = (Vec::new(), Vec::new(), Vec::new());
        for r in root.iter().flatten() {
            let Root::Table(h, at) = r else { continue };
            if h.kind != HeapKind::Resource {
                continue;
            }
            for k in 0..64u64 {
                match crate::descriptor::view(h.start + (*at as u64 + k) * Self::step(HeapKind::Resource)) {
                    Some(View::Shader(r, _)) if texture_info_of(r).is_some() => reads.push(r),
                    Some(View::Unordered(r, _, _)) if texture_info_of(r).is_some() => writes.push(r),
                    Some(View::Unordered(r, _, _)) => {
                        if let Some((p, n)) = crate::resource::buffer_bytes(r as crate::object::This) {
                            buffers.push((p as u64, n));
                        }
                    }
                    _ => {}
                }
            }
        }
        (buffers, reads, writes)
    }

    /// `VICTORY_TRACE_CONTENT=<dir>`: for the first draws into a depth target, every texture and
    /// constant buffer their descriptor tables hold: its view, the fraction of its title memory
    /// that is non-zero, and (8-bit colour textures) the title memory detiled and the host
    /// texture, written as `<dir>/c<n>_<entry>_{title,host}.ppm`.
    #[inline(never)]
    fn trace_content(&self, root: &[Option<Root<B>>], vertex: &[(usize, u64)], call: &victory_backend::resources::DrawCall, binding: Option<&Binding>, targets: &Targets<B>) {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        // `<dir>[,<seconds>[,<format>]]`: from that many seconds after the first draw on; with a
        // format, only the draws into render targets of that title format, boot passes included.
        static OPTIONS: std::sync::OnceLock<Option<(std::path::PathBuf, u64, Option<u32>)>> = std::sync::OnceLock::new();
        let Some((dir, after, only)) = OPTIONS.get_or_init(|| {
            let v = std::env::var_os("VICTORY_TRACE_CONTENT")?.to_string_lossy().into_owned();
            let mut parts = v.split(',');
            let dir = std::path::PathBuf::from(parts.next().unwrap_or_default());
            let after: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            Some((dir, after, parts.next().and_then(|s| s.parse().ok())))
        }) else {
            return;
        };
        let (after, only) = (*after, *only);
        let start = START.get_or_init(std::time::Instant::now);
        if start.elapsed().as_secs() < after {
            return;
        }
        let target = targets.color.first().and_then(|c| {
            self.resources.borrow().iter().find_map(|(k, r)| matches!(r, Res::Texture(t) if Rc::ptr_eq(t, c)).then_some(*k))
        });
        if let Some(f) = only
            && target.and_then(texture_info_of).map(|i| i.format) != Some(f)
        {
            return;
        }
        // The boot passes (a 3840-wide texture sampled, nothing else) are skipped.
        let boot = root.iter().any(|r| {
            let Some(Root::Table(h, at)) = r else { return false };
            let base = h.start + *at as u64 * Self::step(h.kind);
            (0..8u64).any(|k| match crate::descriptor::view(base + k * Self::step(h.kind)) {
                Some(crate::descriptor::View::Shader(r, _)) => texture_info_of(r).is_some_and(|i| i.width >= 3840),
                _ => false,
            })
        });
        // ... and the fullscreen triangle's (one 96-byte constant buffer and no texture).
        let plain = !root.iter().any(|r| {
            let Some(Root::Table(h, at)) = r else { return false };
            let base = h.start + *at as u64 * Self::step(h.kind);
            (0..8u64).any(|k| match crate::descriptor::view(base + k * Self::step(h.kind)) {
                Some(crate::descriptor::View::Shader(..) | crate::descriptor::View::Placed(..)) => true,
                Some(crate::descriptor::View::Constant(_, n)) => n != 96,
                _ => false,
            })
        });
        if only.is_none() && (boot || plain) {
            return;
        }
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n >= 600 {
            return;
        }
        let ppm = |name: String, w: usize, h: usize, px: &[u8], bgra: bool| {
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
            let _ = std::fs::write(std::path::Path::new(&dir).join(name), out);
        };
        let kinds: Vec<String> = root
            .iter()
            .map(|r| match r {
                Some(Root::Constants(w)) => format!("constants{w:?}"),
                Some(Root::View(a)) => format!("view{a:#x}"),
                Some(Root::Table(_, at)) => format!("table@{at}"),
                None => "unset".into(),
            })
            .collect();
        let head = vertex.first().map(|&(_, a)| {
            let mut b = [0u8; 32];
            super::safe_read(a as *const u8, &mut b);
            b
        });
        let target = target.map(|k| format!("{k:#x}"));
        tracing::info!(draw = n, ?target, ?kinds, ?vertex, ?call, ?head, vs = ?binding.map(|b| (b.dwords, b.user_data.clone(), b.views.clone())), ps = ?binding.and_then(|b| b.ps.as_ref()).map(|p| (p.dwords, p.user_data.clone())), "content: draw");
        for (slot, r) in root.iter().enumerate() {
            let Some(Root::Table(h, at)) = r else { continue };
            let base = h.start + *at as u64 * Self::step(h.kind);
            for k in 0..8u64 {
                let handle = base + k * Self::step(h.kind);
                let (key, info, what) = match crate::descriptor::view(handle) {
                    Some(crate::descriptor::View::Constant(a, size)) => {
                        let mut b = vec![0u8; (size as usize).min(1 << 16)];
                        let ok = super::safe_read(a as *const u8, &mut b);
                        let nz = b.iter().filter(|&&x| x != 0).count();
                        let floats: Vec<f32> = b.as_chunks::<4>().0.iter().take(24).map(|w| f32::from_le_bytes(*w)).collect();
                        tracing::info!(draw = n, slot, entry = k, addr = format!("{a:#x}"), size, readable = ok, nonzero = nz, ?floats, "content: constant buffer");
                        continue;
                    }
                    Some(crate::descriptor::View::Shader(r, d)) => (r, texture_info_of(r), format!("srv {d:?}")),
                    Some(crate::descriptor::View::Placed(_, res, d)) if res[0] > 1 => {
                        let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                        (super::textures::placed_key(address, &res), placed_info_of(address, &res), format!("placed {res:?}"))
                    }
                    _ => continue,
                };
                let Some(info) = info else {
                    tracing::info!(draw = n, slot, entry = k, what, "content: view without a texture");
                    continue;
                };
                let tiled = super::textures::read_memory(info.memory, info.size.min(1 << 26));
                let nz = tiled.iter().filter(|&&b| b != 0).count();
                tracing::info!(draw = n, slot, entry = k, key = format!("{key:#x}"), format = info.format, w = info.width, h = info.height, layers = info.depth_or_layers, flags = info.flags, at = format!("{:#x}", info.memory as usize), size = info.size, nonzero = nz, fraction = nz as f64 / tiled.len().max(1) as f64, what, "content: texture");
                // A single-channel 8-bit texture (A8, R8) is written as grey: the title's memory once,
                // the host texture at every draw (the last one standing).
                if matches!(info.format, 65 | 61) {
                    let (w, hh) = (info.width as usize, info.height as usize);
                    let grey = |px: &[u8]| px.iter().flat_map(|&v| [v, v, v, 255]).collect::<Vec<u8>>();
                    let mut px = vec![0u8; w * hh];
                    if !SEEN.lock().unwrap_or_else(|e| e.into_inner()).contains(&key) {
                        SEEN.lock().unwrap_or_else(|e| e.into_inner()).push(key);
                        if let Some(desc) = info.desc()
                            && let Ok(tmp) = self.backend.create_texture(&desc)
                            && super::textures::upload(&self.backend, &tmp, &info, &tiled).is_ok()
                        {
                            self.backend.read_texture(&tmp, 0, 0, &mut px, w, w * hh);
                            ppm(format!("{key:x}_title_{n}.ppm"), w, hh, &grey(&px), false);
                        }
                    }
                    if let Some(Res::Texture(t)) = self.resources.borrow().get(&key) {
                        self.backend.read_texture(t, 0, 0, &mut px, w, w * hh);
                        ppm(format!("{key:x}_host.ppm"), w, hh, &grey(&px), false);
                    }
                    continue;
                }
                if !matches!(info.format, 28 | 87 | 91 | 29) || SEEN.lock().unwrap_or_else(|e| e.into_inner()).contains(&key) {
                    continue;
                }
                SEEN.lock().unwrap_or_else(|e| e.into_inner()).push(key);
                let (w, hh) = (info.width as usize, info.height as usize);
                let bgra = matches!(info.format, 87 | 91);
                let mut px = vec![0u8; w * hh * 4];
                if let Some(desc) = info.desc()
                    && let Ok(tmp) = self.backend.create_texture(&desc)
                    && super::textures::upload(&self.backend, &tmp, &info, &tiled).is_ok()
                {
                    self.backend.read_texture(&tmp, 0, 0, &mut px, w * 4, w * hh * 4);
                    ppm(format!("c{n}_{k}_title.ppm"), w, hh, &px, bgra);
                }
                if let Some(Res::Texture(t)) = self.resources.borrow().get(&key) {
                    self.backend.read_texture(t, 0, 0, &mut px, w * 4, w * hh * 4);
                    ppm(format!("c{n}_{k}_host.ppm"), w, hh, &px, bgra);
                }
            }
        }
    }

    fn prep(&self, cmd: &Cmd<B>) -> Result<Prep<B>, String> {
        let mut p = Prep {
            root: RootArgs::default(),
            ps_root: None,
            bufs: vec![],
            vertex: vec![],
            index: None,
            blit: vec![],
            indirect: None,
            count: None,
            heap: None,
        };
        self.forget_released();
        match cmd {
            Cmd::Dispatch {
                root,
                pso,
                heaps,
                groups,
                threads,
                indirect,
                count,
            } => {
                if let Some(a) = indirect {
                    p.indirect = Some(self.resolve(*a, 12)?);
                }
                if let Some((a, index)) = count {
                    let (b, at) = self.resolve(*a, 4)?;
                    p.count = Some((b, at, *index));
                }
                tracing::debug!(?groups, "dispatch");
                if env_set!("VICTORY_TRACE_TARGETS") {
                    static G: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if G.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6 {
                        tracing::info!(?groups, threads = ?pso.threads, "dispatch");
                    }
                }
                self.trace_draw(root, pso.binding.as_ref(), true);
                self.trace_root_span(root, pso.binding.as_ref(), groups);
                self.remirror_tables(root, pso.binding.as_ref());
                self.materialize_typed_buffers(root, pso.binding.as_ref(), heaps.0.as_ref());
                match &pso.binding {
                Some(b) => {
                    self.translated_args(root, b, heaps.0.as_ref(), *threads, &mut p)?;
                    if b.memory {
                        self.memory_args(heaps.0.as_ref(), &mut p)?;
                    } else if let Some(h) = p.heap.clone() {
                        self.write_pairs(&h)?;
                    }
                    if b.view_addresses {
                        self.view_address_args(heaps.0.as_ref(), &mut p)?;
                    }
                }
                None => self.root_args(root, &mut p)?,
            }},
            Cmd::Draw {
                root,
                binding,
                heaps,
                vertex,
                index,
                call,
                targets,
                indirect,
                count,
                ..
            } => {
                if let Some(a) = indirect {
                    p.indirect = Some(self.resolve(*a, match call { DrawCall::Indexed { .. } => 20, DrawCall::Mesh { .. } => 12, DrawCall::Arrays { .. } | DrawCall::Rects => 16 })?);
                }
                if let Some((a, index)) = count {
                    let (b, at) = self.resolve(*a, 4)?;
                    p.count = Some((b, at, *index));
                }
                if !vertex.is_empty() {
                    self.trace_draw(root, binding.as_deref(), false);
                }
                self.remirror_tables(root, binding.as_deref());
                self.materialize_typed_buffers(root, binding.as_deref(), heaps.0.as_ref());
                self.trace_root_span(root, binding.as_deref(), &[0; 3]);
                self.trace_content(root, vertex, call, binding.as_deref(), targets);
                match binding {
                    Some(b) => {
                        self.translated_args(root, b, heaps.0.as_ref(), [u32::MAX; 3], &mut p)?;
                        if b.memory {
                            self.memory_args(heaps.0.as_ref(), &mut p)?;
                        }
                        if b.view_addresses {
                            self.view_address_args(heaps.0.as_ref(), &mut p)?;
                        }
                        if let Some(ps) = &b.ps {
                            let vs_args = std::mem::take(&mut p.root);
                            self.translated_args(root, ps, heaps.0.as_ref(), [u32::MAX; 3], &mut p)?;
                            if ps.memory {
                                self.memory_args(heaps.0.as_ref(), &mut p)?;
                            }
                            if ps.view_addresses {
                                self.view_address_args(heaps.0.as_ref(), &mut p)?;
                            }
                            p.ps_root = Some(std::mem::replace(&mut p.root, vs_args));
                        }
                        if !b.memory && !b.ps.as_ref().is_some_and(|ps| ps.memory)
                            && let Some(h) = p.heap.clone()
                        {
                            self.write_pairs(&h)?;
                        }
                    }
                    None => self.root_args(root, &mut p)?,
                }
                for &(slot, address) in vertex {
                    let (buffer, ptr) = self.resolve(address, 1)?;
                    p.vertex.push((slot, buffer, ptr));
                }
                if let Some((a, _)) = index {
                    p.index = Some(self.resolve(*a, 1)?);
                }
            }
            Cmd::Copy { dst, src, len } => {
                p.blit.push(self.resolve(*src, *len as u64)?);
                p.blit.push(self.resolve(*dst, *len as u64)?);
            }
            Cmd::Fill { dst, len, .. } => p.blit.push(self.resolve(*dst, *len as u64)?),
            Cmd::CopyBufferToTexture { src, len, .. } => {
                p.blit.push(self.resolve(*src, *len as u64)?);
            }
            Cmd::CopyTextureToBuffer { dst, len, .. } => {
                p.blit.push(self.resolve(*dst, *len as u64)?);
            }
            Cmd::CopyTexture { .. } | Cmd::Clear { .. } => {}
        }
        Ok(p)
    }

    /// `ExecuteCommandLists` plus the queue's waits and signals: resolve every title address, group
    /// the commands into compute, render and blit passes in order, and queue them. Never waits.
    pub fn execute(
        &self,
        queue: &B::Queue,
        lists: &[&NativeList<B>],
        waits: &[(&B::Fence, u64)],
        signals: &[(&B::Fence, u64)],
    ) -> Result<(), String> {
        let _counted = super::descriptor_trace::Submission;
        self.follow_title_pages();
        // A draw or dispatch that reads a root parameter the title never set is dropped by itself:
        // refusing the whole piece for it also dropped the piece's value writes, and a queue that
        // waits for one of those then holds everything behind it for good. (What the console
        // gives a shader in a root slot the title set nothing in is not known. Slot 255 is not
        // one: it is the translator's view-stride table, which `translated_args` binds.) Any
        // other error drops its command alone too.
        // A dropped command still ends the pass it would have ended: the commands either side of it
        // that it kept apart stay in two passes (`cut`), so the backend still orders their
        // accesses to pool buffers. `VICTORY_TRACE_TARGETS=1` warns at the first drop of each kind.
        let same = |a: &Cmd<B>, b: &Cmd<B>| match (a, b) {
            (Cmd::Dispatch { .. }, Cmd::Dispatch { .. }) => true,
            (
                Cmd::Copy { .. }
                | Cmd::Fill { .. }
                | Cmd::CopyTexture { .. }
                | Cmd::CopyBufferToTexture { .. }
                | Cmd::CopyTextureToBuffer { .. },
                Cmd::Copy { .. }
                | Cmd::Fill { .. }
                | Cmd::CopyTexture { .. }
                | Cmd::CopyBufferToTexture { .. }
                | Cmd::CopyTextureToBuffer { .. },
            ) => true,
            (Cmd::Draw { targets: x, .. }, Cmd::Draw { targets: y, .. }) => x.same(y),
            _ => false,
        };
        let trace = env_set!("VICTORY_TRACE_TARGETS");
        let mut cmds: Vec<&Cmd<B>> = Vec::new();
        let mut preps = Vec::new();
        // Whether a pass ends before the kept command of that index, and the commands dropped since
        // the last kept one.
        let mut cut: Vec<bool> = Vec::new();
        let mut dropped: Vec<&Cmd<B>> = Vec::new();
        // The pass trace (`glue::pass_note`): its lines in queue order. Only lists recorded while
        // its window was open hold notes.
        let noted = lists.iter().any(|l| !l.notes.is_empty());
        let exec = if noted {
            static EXEC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            EXEC.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        } else {
            0
        };
        if noted {
            for (li, l) in lists.iter().enumerate() {
                for (ci, note) in &l.notes {
                    eprintln!("PASS exec={exec} q={queue:p} li={li} ci={ci} {note}");
                }
            }
        }
        for (li, ci, c) in lists
            .iter()
            .enumerate()
            .flat_map(|(li, l)| l.cmds.iter().enumerate().map(move |(ci, c)| (li, ci, c)))
        {
            // A draw or clear left with no target (the title's targets have formats the host does
            // not represent, so none was realized) cannot open a render pass: Metal gives no
            // encoder for a pass without attachments, and that refused the whole piece.
            let no_target = match c {
                Cmd::Draw { targets, .. } => targets.color.is_empty() && targets.depth.is_none(),
                Cmd::Clear { color, depth, .. } => color.is_none() && depth.is_none(),
                _ => false,
            };
            if no_target {
                self.no_targets.set(self.no_targets.get() + 1);
                static ONCE: std::sync::Once = std::sync::Once::new();
                if trace {
                    ONCE.call_once(|| {
                        tracing::warn!("native host drops work: a draw or clear with no render target the host represents (first time)");
                    });
                }
                dropped.push(c);
                continue;
            }
            // Reported from this one preparation: preparing again for the trace would mirror tables
            // and write view pairs twice.
            let prepared = self.prep(c);
            #[cfg(target_os = "macos")]
            if noted {
                self.pass_at_submit(exec, &lists[li].notes, li, ci, c);
            }
            if noted && let Err(e) = &prepared {
                eprintln!("PASSFAIL exec={exec} li={li} ci={ci} err={e}");
            }
            match prepared {
                Ok(p) => {
                    // Only a dispatch that will run writes its views.
                    if let Cmd::Dispatch { root, pso, .. } = c {
                        self.note_written(root, pso.binding.as_ref());
                    }
                    cut.push(dropped.drain(..).any(|d| !same(d, c)));
                    cmds.push(c);
                    preps.push(p);
                }
                Err(e)
                    if e.ends_with(NEVER_SET)
                        && matches!(c, Cmd::Draw { .. } | Cmd::Dispatch { .. }) =>
                {
                    self.unset_roots.set(self.unset_roots.get() + 1);
                    // Logged without a switch (the first of each root parameter): such work was
                    // dropped for a whole run unseen (every shader that reads root parameter 255,
                    // the view-stride table, before the host had one).
                    static SEEN: std::sync::Mutex<std::collections::BTreeSet<String>> = std::sync::Mutex::new(std::collections::BTreeSet::new());
                    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                    if seen.len() < 16 && seen.insert(e.clone()) {
                        tracing::warn!(error = %e, "native host drops work: a draw or dispatch that reads a root parameter the title never set (first time for this parameter; counted as dropped_unset_root in \"native draws so far\")");
                    }
                    drop(seen);
                    dropped.push(c);
                }
                // The GPU cannot reach that memory, so the command is not encoded (no GPU fault),
                // but the rest of the piece still runs: refusing it all dropped its valid
                // dispatches, copies and value writes too, and the title read their buffers stale.
                Err(e) if e.ends_with(NOT_MAPPED) => {
                    self.unmapped.set(self.unmapped.get() + 1);
                    // Logged without a switch, as the refused piece was: each one at debug level.
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!(error = %e, "native host drops work: a command names title memory outside every mapped pool (first time; each at debug level)");
                    });
                    tracing::debug!(error = %e, "native host drops a command at unmapped title memory");
                    lost_fill(c, &e);
                    dropped.push(c);
                }
                // Anything else the host cannot give one command (arguments that are not what the
                // shader reads, a heap without room for the view pairs, a buffer not made) drops
                // that command alone too: refusing the piece also dropped its value writes.
                Err(e) => {
                    self.failed.set(self.failed.get() + 1);
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!(error = %e, "native host drops work: a command the host could not prepare (first time; each at debug level)");
                    });
                    tracing::debug!(error = %e, "native host drops a command it could not prepare");
                    lost_fill(c, &e);
                    dropped.push(c);
                }
            }
        }

        let mut groups = vec![];
        let mut i = 0;
        while i < cmds.len() {
            let mut j = i + 1;
            while j < cmds.len() && !cut[j] && same(cmds[i], cmds[j]) {
                j += 1;
            }
            groups.push(match cmds[i] {
                Cmd::Dispatch { .. } => Group::Compute(i, j),
                Cmd::Copy { .. }
                | Cmd::Fill { .. }
                | Cmd::CopyTexture { .. }
                | Cmd::CopyBufferToTexture { .. }
                | Cmd::CopyTextureToBuffer { .. } => Group::Blit(i, j),
                Cmd::Draw { .. } => Group::Render(i, j),
                Cmd::Clear { .. } => {
                    j = i + 1;
                    Group::Clear(i)
                }
            });
            i = j;
        }

        // Every pool buffer the work reaches is resident for the queue's work from now on (once).
        let mut buffers: Vec<&B::Buffer> = Vec::new();
        for p in &preps {
            buffers.extend(p.bufs.iter().map(|b| &**b));
            buffers.extend(p.vertex.iter().map(|(_, buffer, _)| &**buffer));
            buffers.extend(p.index.iter().map(|b| &*b.0));
            buffers.extend(p.blit.iter().map(|b| &*b.0));
            buffers.extend(p.indirect.iter().map(|b| &*b.0));
            buffers.extend(p.count.iter().map(|b| &*b.0));
        }
        if let Err(e) = self.backend.make_resident(queue, &buffers) {
            tracing::warn!(error = %e, "pool buffers were not made resident");
        }

        let compute: Vec<Vec<Dispatch<'_, B>>> = groups
            .iter()
            .map(|g| match g {
                Group::Compute(a, b) => (*a..*b)
                    .map(|k| {
                        let Cmd::Dispatch {
                            pso,
                            heaps: h,
                            groups,
                            ..
                        } = cmds[k]
                        else {
                            unreachable!()
                        };
                        Dispatch {
                            pipeline: &pso.pso,
                            root: preps[k].root.clone(),
                            resources: preps[k].bufs.iter().map(|b| &**b).collect(),
                            heaps: match &preps[k].heap {
                                Some(m) => Heaps {
                                    resource: Some(&m.heap),
                                    sampler: h.1.as_ref().map(|s| &s.heap),
                                },
                                None => heaps(h),
                            },
                            groups: *groups,
                            threads_per_group: pso.threads,
                            indirect: preps[k].indirect.as_ref().map(|(b, p)| (&**b, *p)),
                            count: preps[k].count.as_ref().map(|(b, p, i)| (&**b, *p, *i)),
                        }
                    })
                    .collect(),
                _ => vec![],
            })
            .collect();

        // Copies between depth textures and their twins, before the passes that need them
        // ([`Self::twin_copies`]).
        #[allow(clippy::type_complexity)]
        let twin_copies: Vec<Vec<(Rc<DepthTwin<B>>, Rc<B::Texture>, bool)>> = groups
            .iter()
            .map(|g| match *g {
                Group::Compute(..) => self.twin_copies(true, &[]),
                // A render pass has one depth target, its first draw's.
                Group::Render(a, _) | Group::Clear(a) => self.twin_copies(false, &self.twin_uses(&cmds[a..a + 1])),
                Group::Blit(a, b) => self.twin_copies(false, &self.twin_uses(&cmds[a..b])),
            })
            .collect();
        let mut passes: Vec<Pass<'_, B>> = vec![];
        for (n, g) in groups.iter().enumerate() {
            if !twin_copies[n].is_empty() {
                passes.push(Pass::Blit(Self::twin_blits(&twin_copies[n])));
            }
            passes.push(match g {
                Group::Compute(..) => Pass::Compute(&compute[n]),
                Group::Blit(a, b) => Pass::Blit(
                    (*a..*b)
                        .map(|k| match cmds[k] {
                            Cmd::Copy { len, .. } => {
                                let p = &preps[k].blit;
                                BlitOp::Copy {
                                    src: (&*p[0].0, p[0].1),
                                    dst: (&*p[1].0, p[1].1),
                                    len: *len,
                                }
                            }
                            Cmd::Fill { len, value, .. } => {
                                let p = &preps[k].blit[0];
                                BlitOp::Fill {
                                    dst: (&*p.0, p.1),
                                    len: *len,
                                    value: *value,
                                }
                            }
                            Cmd::CopyTexture {
                                src,
                                src_mip,
                                src_slice,
                                src_origin,
                                size,
                                dst,
                                dst_mip,
                                dst_slice,
                                dst_origin,
                            } => BlitOp::CopyTexture {
                                src: &**src,
                                src_mip: *src_mip,
                                src_slice: *src_slice,
                                src_origin: *src_origin,
                                size: *size,
                                dst: &**dst,
                                dst_mip: *dst_mip,
                                dst_slice: *dst_slice,
                                dst_origin: *dst_origin,
                            },
                            Cmd::CopyBufferToTexture {
                                len,
                                row_pitch,
                                slice_pitch,
                                size,
                                dst,
                                dst_mip,
                                dst_slice,
                                dst_origin,
                                ..
                            } => {
                                let p = &preps[k].blit[0];
                                BlitOp::CopyBufferToTexture {
                                    src: (&*p.0, p.1),
                                    len: *len,
                                    row_pitch: *row_pitch,
                                    slice_pitch: *slice_pitch,
                                    size: *size,
                                    dst: &**dst,
                                    dst_mip: *dst_mip,
                                    dst_slice: *dst_slice,
                                    dst_origin: *dst_origin,
                                }
                            }
                            Cmd::CopyTextureToBuffer {
                                src,
                                src_mip,
                                src_slice,
                                src_origin,
                                size,
                                len,
                                row_pitch,
                                slice_pitch,
                                ..
                            } => {
                                let p = &preps[k].blit[0];
                                BlitOp::CopyTextureToBuffer {
                                    src: &**src,
                                    src_mip: *src_mip,
                                    src_slice: *src_slice,
                                    src_origin: *src_origin,
                                    size: *size,
                                    dst: (&*p.0, p.1),
                                    len: *len,
                                    row_pitch: *row_pitch,
                                    slice_pitch: *slice_pitch,
                                }
                            }
                            _ => unreachable!(),
                        })
                        .collect(),
                ),
                Group::Clear(k) => match cmds[*k] {
                    Cmd::Clear {
                        color: Some((target, color)),
                        rects: Some(rects),
                        ..
                    } => Pass::ClearColorRects {
                        target,
                        color: *color,
                        rects,
                    },
                    Cmd::Clear { color, depth, .. } => Pass::Render(RenderPass {
                        color: color
                            .iter()
                            .map(|(t, c)| ColorTarget {
                                texture: &**t,
                                clear: Some(*c),
                            })
                            .collect(),
                        depth: depth.as_ref().map(|(t, d, s)| DepthTarget {
                            texture: &**t,
                            clear: *d,
                            clear_stencil: *s,
                        }),
                        draws: vec![],
                    }),
                    _ => unreachable!(),
                },
                Group::Render(a, b) => {
                    let Cmd::Draw {
                        targets: Targets { color, depth },
                        ..
                    } = cmds[*a]
                    else {
                        unreachable!()
                    };
                    Pass::Render(RenderPass {
                        color: color
                            .iter()
                            .map(|t| ColorTarget {
                                texture: &**t,
                                clear: None,
                            })
                            .collect(),
                        depth: depth.as_ref().map(|t| DepthTarget {
                            texture: &**t,
                            clear: None,
                            clear_stencil: None,
                        }),
                        draws: (*a..*b)
                            .map(|k| {
                                let Cmd::Draw {
                                    pso,
                                    topology,
                                    heaps: h,
                                    state,
                                    call,
                                    ..
                                } = cmds[k]
                                else {
                                    unreachable!()
                                };
                                Draw {
                                    pipeline: &**pso,
                                    topology: *topology,
                                    state: *state,
                                    root: preps[k].root.clone(),
                                    fragment_root: preps[k].ps_root.clone(),
                                    heaps: match &preps[k].heap {
                                        Some(m) => Heaps {
                                            resource: Some(&m.heap),
                                            sampler: h.1.as_ref().map(|s| &s.heap),
                                        },
                                        None => heaps(h),
                                    },
                                    resources: preps[k].bufs.iter().map(|b| &**b).collect(),
                                    vertex_buffers: preps[k]
                                        .vertex
                                        .iter()
                                        .map(|(slot, b, p)| (*slot, &**b, *p))
                                        .collect(),
                                    index_buffer: preps[k].index.as_ref().map(|(b, p)| (&**b, *p)),
                                    call: *call,
                                    indirect: preps[k].indirect.as_ref().map(|(b, p)| (&**b, *p)),
                                    count: preps[k].count.as_ref().map(|(b, p, i)| (&**b, *p, *i)),
                                }
                            })
                            .collect(),
                    })
                }
            });
        }
        self.passes.set(self.passes.get() + passes.len() as u64);
        let submitted = self.backend.submit_passes(
            queue,
            &PassSubmission {
                waits,
                passes: &passes,
                signals,
            },
        );
        // A refused submission must not lose its bottom-of-pipe value writes: another queue waits
        // on one of them with no other way to go on, and a lost write is the title's
        // `Unsignaled CrossQueueBarrier`, which kills it (#80). Only the lost passes' fills are
        // made here on the CPU: a committed submission's kept passes still run on the GPU (paying
        // their fills too could let a waiting queue go on before earlier GPU passes have run, and
        // the GPU's later fill would overwrite a newer value the title wrote), and a submission
        // that never committed lost every pass. Pass `n` is group `n` below.
        if let Err(f) = &submitted {
            let lost: Vec<usize> = if f.committed {
                f.refused.iter().copied().filter(|&n| n < groups.len()).collect()
            } else {
                (0..groups.len()).collect()
            };
            let mut paid = 0usize;
            for n in lost {
                let Group::Blit(a, b) = &groups[n] else { continue };
                for k in *a..*b {
                    if let Cmd::Fill { len, value, .. } = cmds[k] {
                        let (_, ptr) = &preps[k].blit[0];
                        // SAFETY: the range lies inside a mapped pool's title memory: it resolved.
                        unsafe { std::ptr::write_bytes(*ptr as *mut u8, *value, *len) };
                        paid += 1;
                    }
                }
            }
            if paid > 0 {
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!("native host pays a lost pass's value writes on the CPU (first time; each after at debug level)");
                });
                tracing::debug!("native host paid {paid} value writes of lost passes on the CPU: {f}");
            }
        }
        submitted.map_err(|e| e.to_string())
    }
}

/// The title texture a view names (macOS only; the target trace's).
#[cfg(target_os = "macos")]
fn texture_info_of(r: usize) -> Option<super::textures::TextureInfo> {
    crate::resource::texture_info(r as crate::object::This)
}

#[cfg(not(target_os = "macos"))]
fn texture_info_of(_: usize) -> Option<super::textures::TextureInfo> {
    None
}

/// The pass trace's name for a shader-resource view the host has no texture for: a buffer the host
/// knows (`<key>:buf`), a null resource with its view description's format and dimension
/// (`null:f<format>:dim<dimension>`, read as zeros), or anything else (`<key>:untracked`).
/// The entries (`<slot>.<index>=<view>`) of `now` that a `PASS` note does not hold as they are.
#[cfg(any(test, target_os = "macos"))]
fn changed_entries<'a>(note: &str, now: impl Iterator<Item = &'a String>) -> Vec<&'a String> {
    now.filter(|e| !note.contains(&format!("\"{e}\""))).collect()
}

#[test]
fn an_entry_rewritten_after_recording_is_listed_at_submit() {
    let note = r#"kind=draw rd=["1.29=0x10:f54:240x135@0xa0", "1.30=null:f34:dim4"] wr=[]"#;
    let now = ["1.29=0x10:f54:240x135@0xa0".to_owned(), "1.30=0x20:f34:1280x720@0xb0".to_owned()];
    assert_eq!(changed_entries(note, now.iter()), [&now[1]]);
    // A prefix of an entry is not that entry.
    assert_eq!(changed_entries(r#"rd=["1.3=0x2"]"#, ["1.3=0x".to_owned()].iter()).len(), 1);
}

#[cfg(any(test, target_os = "macos"))]
fn untextured_srv(r: usize, buffer: bool, desc: Option<[u32; 10]>) -> String {
    match (r, desc) {
        (0, Some(d)) => format!("null:f{}:dim{}", d[0], d[1]),
        (0, None) => "null".into(),
        _ if buffer => format!("{r:#x}:buf"),
        _ => format!("{r:#x}:untracked"),
    }
}

#[test]
fn the_pass_trace_names_views_without_a_texture_by_kind() {
    assert_eq!(untextured_srv(0, false, Some([34, 4, 0, 0, 0, 0, 0, 0, 0, 0])), "null:f34:dim4");
    assert_eq!(untextured_srv(0, false, None), "null");
    assert_eq!(untextured_srv(0x1000, true, None), "0x1000:buf");
    assert_eq!(untextured_srv(0x1000, false, None), "0x1000:untracked");
}

#[cfg(target_os = "macos")]
fn placed_info_of(address: u64, res: &[u32; 14]) -> Option<super::textures::TextureInfo> {
    crate::resource::placed_info(address, res)
}

#[cfg(not(target_os = "macos"))]
fn placed_info_of(_: u64, _: &[u32; 14]) -> Option<super::textures::TextureInfo> {
    None
}

/// Entries of a bound table at root `slot` to check for plain-store updates at submission:
/// at least `floor`, extended to cover any higher table entry declared or read by the bound
/// shader(s).
/// `VICTORY_TRACE_HEAP_SET=<frames>`: report what the heaps keep resident that no bound table has
/// named for that many frames (diagnostics; [`NativeHost::heap_set_report`]).
pub(crate) fn heap_set_trace() -> Option<u32> {
    static N: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_TRACE_HEAP_SET").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n > 0))
}

fn table_check_count(slot: usize, binding: Option<&Binding>, floor: usize) -> usize {
    let mut count = floor;
    let Ok(slot) = u8::try_from(slot) else {
        return count;
    };
    let mut scan = |b: &Binding| {
        for &(s, n) in &b.table_entries {
            if s == slot {
                count = count.max(n);
            }
        }
    };
    if let Some(b) = binding {
        scan(b);
        if let Some(ps) = &b.ps {
            scan(ps);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_check_count_uses_shader_metadata_above_default() {
        assert_eq!(table_check_count(0, None, 16), 16);
        let vs = Binding {
            table_entries: vec![(0, 4), (2, 24)],
            ps: Some(Rc::new(Binding {
                table_entries: vec![(2, 30), (3, 19)],
                ..Binding::default()
            })),
            ..Binding::default()
        };
        assert_eq!(table_check_count(0, Some(&vs), 16), 16);
        assert_eq!(table_check_count(1, Some(&vs), 16), 16);
        assert_eq!(table_check_count(2, Some(&vs), 16), 30);
        assert_eq!(table_check_count(3, Some(&vs), 16), 19);
    }

    /// The mappings are read from the pool table once per generation, not at every root view.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_mapping_list_is_read_again_only_when_the_pool_table_changes() {
        use super::super::memory::{PoolInfo, PoolObserver};
        let Ok(metal) = victory_metal::metal::Metal::new() else { return };
        let h = NativeHost::new(metal);
        let pool = PoolInfo { id: 1, base: 0x1000_0000, len: 0x10_0000, memory_entry: None };
        h.memory().mapped(0x7000_0000, 0x1_0000, &pool, 0);
        let (g, a) = h.mapping_list();
        let (_, b) = h.mapping_list();
        assert!(Rc::ptr_eq(&a, &b));
        h.memory().mapped(0x7001_0000, 0x1_0000, &pool, 0x1_0000);
        let (g2, c) = h.mapping_list();
        assert!(g2 != g && c.len() == 2);
    }

    #[test]
    fn root_view_looks_up_pool_table_without_rebuilding_mapping_list() {
        use super::super::memory::{PoolInfo, PoolObserver};
        let h = NativeHost::new(super::super::vertex_tests::Capture::default());
        let heap = h.heap(HeapKind::Resource, 0x1000, 8).unwrap();
        let pool = PoolInfo { id: 1, base: 0x1000_0000, len: 0x10_0000, memory_entry: None };
        h.memory().mapped(0x7000_0000, 0x1_0000, &pool, 0x2000);

        let (view, offset) = h.root_view(&heap, 0x7000_0180);
        assert_eq!((view, offset), (addr::pair_index(8, 1), 0x2180));
        assert!(h.maps.borrow().is_none(), "root_view should not snapshot all mappings");

        // Streaming maps another tile and bumps generation; root_view still avoids mapping_list().
        h.memory().mapped(0x7002_0000, 0x1_0000, &pool, 0x4000);
        assert_eq!(h.root_view(&heap, 0x7002_0040), (addr::pair_index(8, 1), 0x4040));
        assert_eq!(h.root_view(&heap, 0x7001_5000), (addr::pair_index(8, 0), 0));
        assert!(h.maps.borrow().is_none());
    }
}
