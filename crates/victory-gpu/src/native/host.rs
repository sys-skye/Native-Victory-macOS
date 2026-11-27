//! The native host: what `host.rs` is for D3D12, over any [`GraphicsBackend`].
//!
//! One thread drives it (backends hold Objective-C objects). Queues, fences, pipelines, descriptor
//! heaps and textures are made here; title lists are [`NativeList`]s; [`NativeHost::execute`] turns
//! lists into passes and submits them without waiting. Title memory is wrapped lazily, one buffer
//! per pool (see [`super::memory`]); nothing is copied to or from it.

use super::addr::{self, Mapping, Pages, Pairs};
use super::descriptors::{Decoded, Res, decode};
use super::list::{Binding, Cmd, ComputePso, NativeHeap, NativeList, Root, Targets};
use super::memory::PoolTable;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use victory_backend::backend::{Dispatch, Heaps, RootArgs};
use victory_backend::resources::*;

/// Buffers over ranges of the title's own addresses, by `(start, end)`: the signature of the
/// mappings they were made over, when they were last used, and the buffer.
type TitleAliases<Buf> = RefCell<HashMap<(u64, u64), (u64, u64, Rc<Buf>)>>;

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
    resources: RefCell<HashMap<usize, Res<Rc<B::Texture>>>>,
    /// Counts changes to [`Self::resources`], so unrealized table entries retry once registered.
    resources_epoch: std::cell::Cell<u64>,
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
    /// With texture copy-back on: the keys dispatches had bound for unordered access, not yet
    /// taken by [`Self::take_written`].
    written: RefCell<Option<std::collections::HashSet<usize>>>,
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
            maps: RefCell::default(),
            resources: RefCell::default(),
            resources_epoch: std::cell::Cell::new(0),
            heaps: RefCell::default(),
            contract: RefCell::default(),
            unset_roots: std::cell::Cell::new(0),
            no_targets: std::cell::Cell::new(0),
            unmapped: std::cell::Cell::new(0),
            cleared: std::cell::Cell::new(0),
            failed: std::cell::Cell::new(0),
            passes: std::cell::Cell::new(0),
            written: Default::default(),
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

    /// The table the pool observer keeps current; register it as (or call it from) the observer.
    pub fn memory(&self) -> Arc<PoolTable> {
        self.memory.clone()
    }

    /// The buffer and pool-mapping address of `[addr, addr + len)` of title memory.
    pub fn resolve(&self, addr: u64, len: u64) -> Result<(Rc<B::Buffer>, *const u8), String> {
        self.forget_released();
        let (pool, ptr) = self
            .memory
            .resolve(addr, len)
            .ok_or_else(|| format!("title address {addr:#x}+{len:#x} {NOT_MAPPED}"))?;
        Ok((self.pool_buffer(pool.id, pool.base, pool.len)?, ptr))
    }

    /// Drops what the host keeps of pools released since the last call: their buffers, and their
    /// view pairs (free for other pools from now on). [`Self::released_pools`] lists them.
    fn forget_released(&self) {
        let gone = self.memory.take_released();
        if gone.is_empty() {
            return;
        }
        // An alias may show pages of a released pool: later mirrors make new ones.
        self.aliases.borrow_mut().clear();
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
    /// it are the same ([`PoolTable::signature`]), made again when they change, the last 64 kept,
    /// all dropped when a pool is released. A range with a gap stays the null view.
    // shortcut: a wrap is made when an entry is mirrored. A later remap of those title pages to
    // other pool pages is seen only when the entry is mirrored again (as for pool views); drop the
    // wraps a pool-table change touches if a title is found to remap a viewed range in place.
    fn title_alias(&self, addr: u64, len: u64) -> Option<(Rc<B::Buffer>, *const u8)> {
        // Host pages of Apple silicon; Rosetta's 4 KiB pages divide them.
        const PAGE: u64 = 1 << 14;
        // Wraps kept for reuse: the heap entries that show one keep it alive themselves.
        const KEPT: usize = 64;
        static USE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if addr == 0 || len == 0 {
            return None;
        }
        let base = addr & !(PAGE - 1);
        let end = addr.checked_add(len)?.checked_add(PAGE - 1)? & !(PAGE - 1);
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
            return Some((b.clone(), addr as *const u8));
        }
        let buf = Rc::new(self.backend.wrap_title_memory(base as *mut u8, (end - base) as usize).ok()?);
        if aliases.len() >= KEPT
            && let Some(&oldest) = aliases.iter().min_by_key(|(_, (_, used, _))| *used).map(|(k, _)| k)
        {
            aliases.remove(&oldest);
        }
        aliases.insert((base, end), (sig, now, buf.clone()));
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(addr = format!("{addr:#x}"), len, "a buffer view not linear in its pool reads the title's own mapping (first time)");
        });
        Some((buf, addr as *const u8))
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

    /// Names a buffer resource object `key` (what `CreateShaderResourceView` is passed) at title memory.
    pub fn register_buffer(&self, key: usize, addr: u64, len: u64) {
        self.resources
            .borrow_mut()
            .insert(key, Res::Buffer(addr, len));
        self.resources_epoch.set(self.resources_epoch.get() + 1);
    }

    pub fn register_texture(&self, key: usize, texture: &Rc<B::Texture>) {
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
            mirrored: Default::default(),
            pending_mirror: Default::default(),
            images: Default::default(),
        });
        self.heaps.borrow_mut().push(heap.clone());
        Ok(heap)
    }

    /// The entries of a resource heap of `count` title entries: the title's, the view pairs of
    /// the pool-offset contract, and then up to the translator's clamp of heap indices
    /// (`HEAP_VIEWS`: a translated shader reads no entry past it, and reads the host's null
    /// buffer views just below it, see [`Self::write_null_buffers`]). A heap too long for the
    /// pairs ([`addr::fits`]) stays as long as it is.
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
        let removed = self.resources.borrow_mut().remove(&key);
        if removed.is_some() {
            self.resources_epoch.set(self.resources_epoch.get() + 1);
        }
        let texture = matches!(removed, Some(Res::Texture(_)));
        if !texture {
            return;
        }
        for heap in self.heaps.borrow().iter() {
            for index in heap.take_entries(key) {
                self.backend.clear_entry(&heap.heap, index);
                self.cleared.set(self.cleared.get() + 1);
            }
        }
    }

    /// Heap entries cleared so far because the texture they showed was released.
    pub fn cleared(&self) -> u64 {
        self.cleared.get()
    }

    /// Whether resource object `key` is registered (as a buffer or a texture).
    pub fn knows(&self, key: usize) -> bool {
        self.resources.borrow().contains_key(&key)
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
        let Some(view) = view else {
            return Ok(false);
        };
        let resources = self.resources.borrow();
        let Some(decoded) = decode(&view, &resources) else {
            return Ok(false);
        };
        match decoded {
            Decoded::Buffer { addr, size } => {
                // Read before resolving: a pool mapped by a title thread after a failed resolve
                // must bump past these, so the entry is retried.
                let (pool_gen, res_gen) = (self.memory.generation(), self.resources_epoch.get());
                let alias = || self.title_alias(addr, u64::from(size));
                let (buf, ptr) = match self.resolve(addr, u64::from(size)).map_or_else(|e| alias().ok_or(e), Ok) {
                    Ok(found) => found,
                    Err(e) => {
                        let null = self.null_view()?;
                        self.backend.write_buffer(
                            &heap.heap,
                            index,
                            &null.buf,
                            null.ptr().cast_const(),
                            0,
                        );
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
                heap.shows(index, None);
            }
            Decoded::Image {
                resource,
                mips,
                slices,
                format,
            } => {
                let Some(Res::Texture(tex)) = resources.get(&resource) else {
                    return Ok(false);
                };
                match (mips, format) {
                    (Some((first, count)), Some(format)) if count > 0 => {
                        let v = self.backend.create_texture_view(
                            tex,
                            &ViewDesc {
                                format,
                                mips: (first as usize, count as usize),
                                slices: slices.map_or((0, usize::MAX), |(first, count)| {
                                    (first as usize, count as usize)
                                }),
                            },
                        )?;
                        self.backend.write_texture(&heap.heap, index, &v, 0.0);
                    }
                    _ => self.backend.write_texture(&heap.heap, index, tex, 0.0),
                }
                heap.shows(index, Some(resource));
            }
            Decoded::Sampler { desc, lod_bias } => {
                let s = self.backend.create_sampler(&desc)?;
                self.backend.write_sampler(&heap.heap, index, &s, lod_bias);
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

    /// Notes the textures the tables `root` binds hold unordered access views of: what a dispatch
    /// may write. More than it does write (the shader may not use every entry).
    ///
    /// shortcut: a shader that indexes the descriptor heap itself, and pixel shader writes, are
    /// not seen; add them when a title texture needs it.
    fn note_written(&self, root: &[Option<Root<B>>], binding: Option<&Binding>) {
        let mut written = self.written.borrow_mut();
        let Some(written) = written.as_mut() else { return };
        for (slot, r) in root.iter().enumerate() {
            let Some(Root::Table(h, at)) = r else { continue };
            if h.kind != HeapKind::Resource {
                continue;
            }
            let end = (*at + table_check_count(slot, binding, Self::TABLE_CHECK)).min(h.count);
            for index in *at..end {
                match crate::descriptor::view(h.start + index as u64 * Self::step(h.kind)) {
                    Some(crate::descriptor::View::Unordered(r, ..)) => {
                        written.insert(r);
                    }
                    Some(crate::descriptor::View::Placed(true, res, d)) if res[0] > 1 => {
                        let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                        written.insert(super::textures::placed_key(address, &res));
                    }
                    _ => {}
                }
            }
        }
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
            let check = table_check_count(slot, binding, Self::TABLE_CHECK);
            let (mut checked, mut changed) = (0, 0);
            let step = Self::step(h.kind) as usize;
            let end = (*at + check).min(h.count);
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
                if record == [0; 12]
                    || h.mirrored.borrow().get(&index) == Some(&record)
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
    /// [`Self::mirror`]. shortcut: an entry the title copied as bytes outside a bound table
    /// keeps its old dword until it is mirrored.
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
            mirrored: Default::default(),
            pending_mirror: Default::default(),
            images: Default::default(),
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
                    for k in 0..4u64 {
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
                self.note_written(root, pso.binding.as_ref());
                match &pso.binding {
                Some(b) => {
                    self.translated_args(root, b, heaps.0.as_ref(), *threads, &mut p)?;
                    if b.memory {
                        self.memory_args(heaps.0.as_ref(), &mut p)?;
                    } else if let Some(h) = p.heap.clone() {
                        self.write_pairs(&h)?;
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
                    p.indirect = Some(self.resolve(*a, match call { DrawCall::Indexed { .. } => 20, DrawCall::Mesh { .. } => 12, DrawCall::Arrays { .. } => 16 })?);
                }
                if let Some((a, index)) = count {
                    let (b, at) = self.resolve(*a, 4)?;
                    p.count = Some((b, at, *index));
                }
                if !vertex.is_empty() {
                    self.trace_draw(root, binding.as_deref(), false);
                }
                self.remirror_tables(root, binding.as_deref());
                self.trace_root_span(root, binding.as_deref(), &[0; 3]);
                self.trace_content(root, vertex, call, binding.as_deref(), targets);
                match binding {
                    Some(b) => {
                        self.translated_args(root, b, heaps.0.as_ref(), [u32::MAX; 3], &mut p)?;
                        if b.memory {
                            self.memory_args(heaps.0.as_ref(), &mut p)?;
                        }
                        if let Some(ps) = &b.ps {
                            let vs_args = std::mem::take(&mut p.root);
                            self.translated_args(root, ps, heaps.0.as_ref(), [u32::MAX; 3], &mut p)?;
                            if ps.memory {
                                self.memory_args(heaps.0.as_ref(), &mut p)?;
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
        for c in lists.iter().flat_map(|l| l.cmds.iter()) {
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
            match self.prep(c) {
                Ok(p) => {
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

        let mut passes: Vec<Pass<'_, B>> = vec![];
        for (n, g) in groups.iter().enumerate() {
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
        self.backend.submit_passes(
            queue,
            &PassSubmission {
                waits,
                passes: &passes,
                signals,
            },
        )
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
