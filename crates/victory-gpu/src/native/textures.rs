//! Title textures on the native host: the console's tiled layouts (`victory_formats::texture`)
//! detiled from title memory into backend textures.
//!
//! A title texture lives in title memory in the console's swizzled layout; the backend's texture is
//! Apple's own. Each texture the host samples is detiled into a linear staging buffer and uploaded,
//! mip by mip and slice by slice, when the memory's content changed since the last time (a hash of
//! the memory, taken when work is submitted). Render targets and depth targets are the GPU's: they
//! start empty and are not read from memory. With `VICTORY_TEXTURE_BACK=1`, textures compute writes
//! are retiled back into title memory ([`Back`]).
//!
//! shortcut: the detile runs on the CPU (`Layout::detile`, the code the D3D12 host uses too) and
//! a thread hashes the memory of every sampled texture round after round (see `Watch`); a compute detile that reads the
//! pool buffer directly and a per-page change test (what `watch.rs` gives the D3D12 host) are the
//! next step. A change detiles the whole texture.
//!
//! A submission's textures are read, hashed and detiled on a few threads without the host's lock
//! ([`prepare`]); only the probes that pick them ([`Title::sync_jobs`]) and the uploads
//! ([`Title::sync_apply`]) hold it. Under the lock that work stopped every thread of a title that
//! streams (hundreds of MiB of textures a second), for most of each second.

use super::descriptors::format;
use super::host::NativeHost;
use super::memory::PoolTable;
use super::list::NativeList;
use super::title::Title;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;
use victory_backend::resources::*;
use victory_formats::texture::{Dimension, Layout};

/// `D3D12_RESOURCE_FLAGS`.
const RENDER_TARGET: u32 = 1;
const DEPTH_STENCIL: u32 = 2;
const UNORDERED_ACCESS: u32 = 4;

/// A title texture as the title made it: its description and where its memory is.
#[derive(Clone)]
pub struct TextureInfo {
    pub dimension: Dimension,
    /// DXGI format number, the console's (before `pc_format`).
    pub format: u32,
    pub width: u32,
    pub height: u32,
    /// Array length, or depth for 3D.
    pub depth_or_layers: u32,
    pub flags: u32,
    /// The layout of plane 0 (depth, for depth and stencil).
    pub layout: Layout,
    pub memory: *const u8,
    pub size: usize,
}

impl TextureInfo {
    pub fn render_target(&self) -> bool {
        self.flags & (RENDER_TARGET | DEPTH_STENCIL) != 0
    }

    /// The backend's description of it; `None` for formats or shapes the host does not know.
    pub fn desc(&self) -> Option<TextureDesc> {
        let depth = self.flags & DEPTH_STENCIL != 0;
        let mut format = format(self.format)?;
        // A typeless 32-bit float is a depth target when the texture is one.
        if depth && format == Format::R32Float {
            format = Format::D32Float;
        }
        if depth && format == Format::R16Unorm {
            format = Format::D16Unorm;
        }
        let (dim, d) = match self.dimension {
            Dimension::Tex3d => (TexDim::D3, self.depth_or_layers as usize),
            _ if self.depth_or_layers > 1 => (TexDim::D2Array, self.depth_or_layers as usize),
            _ => (TexDim::D2, 1),
        };
        Some(TextureDesc {
            dim,
            format,
            width: self.width as usize,
            height: self.height.max(1) as usize,
            depth_or_layers: d,
            mips: self.layout.mips.len().max(1),
            render_target: self.render_target(),
            unordered_access: self.flags & UNORDERED_ACCESS != 0,
        })
    }

    /// Top mip levels left off the host texture (0 or 1; see [`drop_top_mip`]).
    pub fn dropped_levels(&self) -> usize {
        dropped_levels(
            drop_top_mip(),
            self.render_target() || self.flags & UNORDERED_ACCESS != 0,
            self.dimension == Dimension::Tex3d,
            self.layout.mips.len(),
            self.width.max(self.height),
        )
    }

    /// [`Self::desc`] without the dropped top levels: what the host texture is made as.
    pub fn host_desc(&self) -> Option<TextureDesc> {
        let mut desc = self.desc()?;
        let n = self.dropped_levels();
        desc.width = (desc.width >> n).max(1);
        desc.height = (desc.height >> n).max(1);
        desc.mips -= n;
        Some(desc)
    }

}

/// Largest side (texels) from which a sampled texture loses its top level under [`drop_top_mip`].
const DROP_TOP_MIP_MIN_SIDE: u32 = 1024;

/// `VICTORY_TEXTURE_DROP_TOP_MIP=1`: large sampled textures are made on the host without their
/// top mip (a quarter of the memory of that level, about 75% of the texture's): for hosts whose
/// unified memory cannot hold the title's full set (an iPad's ~9 GiB before it swaps).
/// Sampling with implicit LOD then uses the next level, a little blurrier up close. Shortcut, not
/// console behaviour: explicit-LOD samples come out one level blurrier and integer texel loads
/// address the smaller texture; render targets, UAV and 3D textures are never touched.
pub fn drop_top_mip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TEXTURE_DROP_TOP_MIP").is_ok_and(|v| v == "1"))
}

/// Levels to leave off: one for a large, sampled-only 2D texture that has a level below its top.
fn dropped_levels(enabled: bool, written_by_gpu: bool, volume: bool, mips: usize, largest_side: u32) -> usize {
    let eligible = enabled && !written_by_gpu && !volume && mips >= 2 && largest_side >= DROP_TOP_MIP_MIN_SIDE;
    usize::from(eligible)
}

/// A view's title mip range `(first, count)` on a host texture made without its top `dropped`
/// levels (`host_mips` left): the same levels where they exist, else the nearest host level.
pub fn host_mip_range(first: usize, count: usize, dropped: usize, host_mips: usize) -> (usize, usize) {
    if dropped == 0 || host_mips == 0 {
        return (first, count);
    }
    let end = (first + count.max(1)).saturating_sub(dropped).clamp(1, host_mips);
    let start = first.saturating_sub(dropped).min(end - 1);
    (start, end - start)
}

/// Uploads and change probes must see the same bytes, including across holes in title memory.
/// False if they could not be read at once (a hole: zeros there). When `known_partial` says an
/// earlier whole read of this texture met an unmapped or protected range, skips the initial
/// whole-range kernel read that would copy the readable prefix and then fail at the hole, and
/// returns `true` again once every page is readable.
fn read_texture_bytes_known(at: *const u8, out: &mut [u8], known_partial: bool) -> bool {
    #[cfg(target_os = "macos")]
    return read_texture_chunk(
        at,
        out,
        known_partial,
        super::safe_read,
        victory_platform::mem::next_region,
    );
    #[cfg(not(target_os = "macos"))]
    {
        let _ = known_partial;
        let read = super::safe_read(at, out);
        if !read {
            out.fill(0);
        }
        read
    }
}

#[cfg(target_os = "macos")]
fn read_texture_chunk(
    at: *const u8,
    out: &mut [u8],
    known_partial: bool,
    mut read: impl FnMut(*const u8, &mut [u8]) -> bool,
    mut next_region: impl FnMut(usize) -> Option<victory_platform::mem::Region>,
) -> bool {
    use victory_platform::mem::{Protect, page_size};

    if out.is_empty() {
        return true;
    }
    let base = at.addr();
    let page = page_size();
    let single_page = out.len() <= page - base % page;
    if (!known_partial || single_page) && read(at, out) {
        return true;
    }
    if single_page {
        out.fill(0);
        return false;
    }
    let Some(end) = base.checked_add(out.len()) else {
        out.fill(0);
        return false;
    };
    let mut flush_readable = |start: usize, stop: usize, out: &mut [u8], whole: &mut bool| {
        if start >= stop {
            return;
        }
        if !read(start as *const u8, &mut out[start - base..stop - base]) {
            *whole = false;
            // The mapping may have changed after the query. Salvage this span page by page.
            let mut pos = start;
            while pos < stop {
                let n = (page - pos % page).min(stop - pos);
                let part = &mut out[pos - base..pos - base + n];
                if !read(pos as *const u8, part) {
                    part.fill(0);
                }
                pos += n;
            }
        }
    };
    let mut cur = base;
    let mut whole = true;
    let mut run_start: Option<usize> = None;
    while cur < end {
        let Some(region) = next_region(cur) else {
            break;
        };
        let start = cur.max(region.base);
        let stop = end.min(region.base.saturating_add(region.len));
        if start >= stop {
            break;
        }
        if cur < start {
            if let Some(rs) = run_start.take() {
                flush_readable(rs, cur, out, &mut whole);
            }
            out[cur - base..start - base].fill(0);
            whole = false;
        }
        if region.protect.0 & Protect::READ.0 == 0 {
            if let Some(rs) = run_start.take() {
                flush_readable(rs, start, out, &mut whole);
            }
            out[start - base..stop - base].fill(0);
            whole = false;
        } else if run_start.is_none() {
            run_start = Some(start);
        }
        cur = stop;
    }
    if let Some(rs) = run_start.take() {
        flush_readable(rs, cur, out, &mut whole);
    }
    if cur < end {
        out[cur - base..].fill(0);
        whole = false;
    }
    whole
}

/// Reads `[at, at + out.len())` of title memory: coalescing mapped spans through [`PoolTable`]
/// first when a pool backs the range (one fault-checked `safe_read` per contiguous mapped span and
/// zero syscalls over unmapped fine-mip holes), else through [`read_texture_bytes_known`].
fn read_bytes(memory: Option<&PoolTable>, at: usize, out: &mut [u8], known_partial: bool) -> bool {
    if let Some(whole) = memory.and_then(|m| {
        m.read_mapped_with(at, out, |span_at, dst| {
            if super::safe_read(span_at as *const u8, dst) {
                true
            } else {
                let _ = read_texture_bytes_known(span_at as *const u8, dst, true);
                false
            }
        })
    }) {
        return whole;
    }
    read_texture_bytes_known(at as *const u8, out, known_partial)
}

/// Reads `len` bytes of title memory (zeros where it cannot be read: a page the title never mapped).
pub fn read_memory(at: *const u8, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let _ = read_into(None, at as usize, &mut out, false);
    out
}

/// Fills `out` from title memory at `at`, with zeros where it cannot be read; whether some of it
/// could not be read at once (`VICTORY_TRACE_TEXTURE_SYNC=1` counts those textures).
fn read_into(memory: Option<&PoolTable>, at: usize, out: &mut [u8], known_partial: bool) -> bool {
    !read_bytes(memory, at, out, known_partial)
}

/// Host texels as title memory holds them: the reverse of [`to_host`].
fn from_host(format: u32, texels: &mut [u8]) {
    if format == 115 {
        for t in texels.as_chunks_mut::<2>().0 {
            *t = u16::from_le_bytes(*t).rotate_right(4).to_le_bytes();
        }
    }
}

/// Title texels as the host format holds them: DXGI `B4G4R4A4_UNORM` (115) has blue in the low
/// nibble and alpha in the top one, Metal's `ABGR4Unorm` alpha low and red top. The host's own
/// writes and reads agree with each other, so only bytes from title memory are converted.
pub fn to_host(format: u32, texels: &mut [u8]) {
    if format == 115 {
        for t in texels.as_chunks_mut::<2>().0 {
            *t = u16::from_le_bytes(*t).rotate_left(4).to_le_bytes();
        }
    }
}

/// One mip of one slice, detiled: what `ResourceBackend::upload_texture` takes.
struct Level {
    mip: usize,
    slice: usize,
    texels: Vec<u8>,
    row: usize,
    slice_pitch: usize,
}

fn is_all_zero(bytes: &[u8]) -> bool {
    let (chunks, tail) = bytes.as_chunks::<64>();
    chunks.iter().all(|&c| c == [0u8; 64]) && tail.iter().all(|&b| b == 0)
}

/// Detiles the mips of `tiled` needed to bring a host texture up to date. Mips whose bit in
/// `uploaded_mips` (`mip < 32`) is still clear sit at zero in the fresh host texture, so non-tail
/// mips whose tiled bytes are all zero (unmapped fine mips of a streamed texture) skip staging
/// allocation, CPU detile, and `upload_texture`.
/// `skip`: top levels the host texture was made without ([`TextureInfo::dropped_levels`]); title
/// level `m` becomes host level `m - skip` and the skipped ones are not detiled.
fn detile_changed(
    layout: &Layout,
    dimension: Dimension,
    format: u32,
    tiled: &[u8],
    uploaded_mips: u32,
    skip: usize,
) -> (Vec<Level>, u32, Result<(), String>) {
    if !layout.addressable() {
        return (
            Vec::new(),
            0,
            Err(format!(
                "detile mip 0 slice 0: {:?}",
                victory_formats::texture::Error::Unsupported
            )),
        );
    }
    let slices = if dimension == Dimension::Tex3d { 1 } else { layout.slices };
    let n = layout.desc.element_bytes as usize;
    let slice_size = layout.slice_size as usize;
    let mut levels = Vec::new();
    let mut detiled_mips = 0u32;
    for mip in skip as u32..layout.mips.len() as u32 {
        let bit = if mip < 32 { 1u32 << mip } else { 0 };
        if bit != 0 && uploaded_mips & bit == 0 && mip < layout.first_tail_mip {
            let mip_start = layout.mips[mip as usize].offset as usize;
            let mip_end = if mip > 0 {
                layout.mips[mip as usize - 1].offset as usize
            } else {
                slice_size
            };
            let zero_across_slices = mip_start <= mip_end
                && (0..layout.slices as usize).all(|s| {
                    let base = s * slice_size;
                    tiled
                        .get(base + mip_start..base + mip_end)
                        .is_some_and(is_all_zero)
                });
            if zero_across_slices {
                continue;
            }
        }
        let m = &layout.mips[mip as usize];
        let row = m.width as usize * n;
        let slice_pitch = row * m.height as usize;
        let depth = layout.subresource_depth(mip).max(1) as usize;
        for slice in 0..slices {
            let mut texels = vec![0u8; slice_pitch * depth];
            if let Err(e) = layout.detile(tiled, mip, slice, &mut texels, row, slice_pitch) {
                return (levels, detiled_mips, Err(format!("detile mip {mip} slice {slice}: {e:?}")));
            }
            to_host(format, &mut texels);
            levels.push(Level { mip: mip as usize - skip, slice: slice as usize, texels, row, slice_pitch });
        }
        detiled_mips |= bit;
    }
    (levels, detiled_mips, Ok(()))
}

/// Detiles every mip of every slice of `tiled` (a texture's whole memory): the levels detiled,
/// and the error that stopped it short if one did.
/// `skip` as for [`detile_changed`].
fn detile(layout: &Layout, dimension: Dimension, format: u32, tiled: &[u8], skip: usize) -> (Vec<Level>, Result<(), String>) {
    let (levels, _, result) = detile_changed(layout, dimension, format, tiled, u32::MAX, skip);
    (levels, result)
}

fn upload_levels<B: ResourceBackend>(backend: &B, tex: &B::Texture, levels: &[Level]) {
    for l in levels {
        backend.upload_texture(tex, l.mip, l.slice, &l.texels, l.row, l.slice_pitch);
    }
}

/// Detiles every mip of every slice of `tiled` (the texture's whole memory) into `tex`.
pub fn upload<B: ResourceBackend>(
    backend: &B,
    tex: &B::Texture,
    info: &TextureInfo,
    tiled: &[u8],
) -> Result<(), String> {
    upload_from(backend, tex, info, tiled, 0)
}

/// [`upload`] into a host texture made without the title's top `skip` levels.
pub fn upload_from<B: ResourceBackend>(
    backend: &B,
    tex: &B::Texture,
    info: &TextureInfo,
    tiled: &[u8],
    skip: usize,
) -> Result<(), String> {
    let (levels, result) = detile(&info.layout, info.dimension, info.format, tiled, skip);
    upload_levels(backend, tex, &levels);
    result
}

/// The reverse of [`upload`]: every mip of every slice of `tex`, retiled into `tiled` (the
/// texture's whole memory; bytes between subresources stay as they are).
fn download<B: ResourceBackend>(
    backend: &B,
    tex: &B::Texture,
    info: &TextureInfo,
    tiled: &mut [u8],
) -> Result<(), String> {
    let layout = &info.layout;
    let slices = if info.dimension == Dimension::Tex3d {
        1
    } else {
        layout.slices
    };
    let n = layout.desc.element_bytes as usize;
    for mip in 0..layout.mips.len() as u32 {
        let m = &layout.mips[mip as usize];
        let row = m.width as usize * n;
        let slice_pitch = row * m.height as usize;
        let depth = layout.subresource_depth(mip).max(1) as usize;
        let mut linear = vec![0u8; slice_pitch * depth];
        for slice in 0..slices {
            backend.read_texture(tex, mip as usize, slice as usize, &mut linear, row, slice_pitch);
            from_host(info.format, &mut linear);
            layout
                .tile(tiled, mip, slice, &linear, row, slice_pitch)
                .map_err(|e| format!("tile mip {mip} slice {slice}: {e:?}"))?;
        }
    }
    Ok(())
}

/// The largest texture copied back (the ocean maps are small; a guess until a world run's
/// "copied back" lines show what is written).
const BACK_LIMIT: usize = 4 << 20;

/// Textures compute work wrote, to be copied back into title memory once it completes
/// (`VICTORY_TEXTURE_BACK=1`). Host textures are the GPU's own, so without this the title's CPU
/// reads what was there before (VIctory-macOS issue #46: boats sink).
///
/// shortcut: copied back before a fence value is shown to the title; a title that instead polls a
/// label the GPU writes into memory may read a frame late.
pub(super) struct Back<B: PresentBackend> {
    /// A fence of ours per title queue, and the last value queued on it.
    fences: HashMap<usize, (Rc<B::Fence>, u64)>,
    /// Submissions not yet copied back: reached when the fence has the value.
    pending: Vec<(Rc<B::Fence>, u64, Vec<usize>)>,
    /// Textures already logged.
    logged: std::collections::HashSet<usize>,
}

impl<B: PresentBackend> Default for Back<B> {
    fn default() -> Self {
        Back {
            fences: HashMap::new(),
            pending: Vec::new(),
            logged: Default::default(),
        }
    }
}

impl<B: PresentBackend> Back<B> {
    /// The signal the next work of title queue `key` carries (`None` with copy-back off): in the
    /// work's own command buffer, so it does not hang on the order command buffers complete in.
    pub(super) fn next(&mut self, host: &NativeHost<B>, key: usize) -> Option<(Rc<B::Fence>, u64)> {
        if !host.collecting_written() {
            return None;
        }
        let (fence, value) = match self.fences.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => match host.backend.create_fence(0) {
                Ok(f) => e.insert((Rc::new(f), 0)),
                Err(error) => {
                    tracing::warn!(%error, "no fence to copy textures back by");
                    return None;
                }
            },
        };
        *value += 1;
        Some((fence.clone(), *value))
    }

    /// That work went to the GPU: if it wrote textures, they are copied back once `signal` is
    /// reached.
    pub(super) fn submitted(&mut self, host: &NativeHost<B>, signal: Option<(Rc<B::Fence>, u64)>) {
        let written = host.take_written();
        if let Some((fence, value)) = signal
            && !written.is_empty()
        {
            self.pending.push((fence, value, written));
        }
    }
}

/// Textures whose memory the scanner thread watches, and those it found changed.
///
/// Hashing every sampled texture's memory on every submission (150 ms at 170 MiB, under the
/// title lock the title's other threads wait on) made a title that streams textures slow to a
/// crawl as they accumulated. A thread of its own hashes them in rounds, off the lock, and a
/// submission only looks at the textures it marked.
/// shortcut: a change is seen one round later (a round is as long as the hashing takes, about
/// 100 ms for 150 MiB), so a texture written just before it is sampled may show its old content
/// for a few frames; a per-page write test would make that exact.
#[derive(Default)]
struct Watch {
    /// Title texture key to (address, bytes, whether it is a render target: its first sight is
    /// its baseline, not a change; which texture of that key: [`Tracked::serial`]).
    entries: HashMap<usize, (usize, usize, bool, u64)>,
    /// The pool table of the title host that realized each watched texture, so pool-backed scans
    /// read in userspace rather than trapping into Mach VM per page.
    pools: HashMap<usize, Arc<PoolTable>>,
    dirty: std::collections::HashSet<usize>,
    /// The hash of each texture's memory as last read whole, by the scanner or by a submission.
    /// What a submission read, the host texture holds: the scanner does not report it again (it
    /// used to report every new texture and every write a probe had caught already, and each
    /// report is the whole texture read once more).
    seen: HashMap<usize, blake3::Hash>,
    /// Textures whose last whole read met an unmapped or protected range (fine mips not mapped yet).
    partial: std::collections::HashSet<usize>,
    /// Bumped whenever a submission begins or applies a whole read of a watched texture, so an
    /// in-flight scanner round that started earlier discards its older (possibly torn) hash.
    epoch: HashMap<usize, u64>,
}

impl Watch {
    /// Texture `key` is gone. Its hash goes with it: the key (an object's address, a placement)
    /// names another texture later, and a render target compared with the old one's hash had
    /// its memory uploaded over what was drawn into it.
    fn released(&mut self, key: usize) {
        self.entries.remove(&key);
        self.pools.remove(&key);
        self.dirty.remove(&key);
        self.seen.remove(&key);
        self.partial.remove(&key);
        self.epoch.remove(&key);
    }
}

type Watched = std::sync::Mutex<Option<Watch>>;

static WATCH: Watched = std::sync::Mutex::new(None);

fn watch<R>(f: impl FnOnce(&mut Watch) -> R) -> R {
    watch_in(&WATCH, f)
}

fn watch_in<R>(watched: &Watched, f: impl FnOnce(&mut Watch) -> R) -> R {
    f(watched.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// One round of the scanner: hashes the memory of each texture `watched` lists, without holding
/// it, and marks those that changed since they were last read whole.
fn scan_round(watched: &Watched, buf: &mut [u8]) {
    let list: Vec<(usize, (usize, usize, bool, u64))> = watch_in(watched, |w| {
        let Watch { entries, pools, seen, partial, epoch, .. } = w;
        pools.retain(|k, _| entries.contains_key(k));
        seen.retain(|k, _| entries.contains_key(k));
        partial.retain(|k| entries.contains_key(k));
        epoch.retain(|k, _| entries.contains_key(k));
        entries.iter().map(|(&k, &v)| (k, v)).collect()
    });
    for &(key, entry) in &list {
        let (at, size, target, _) = entry;
        // What it is compared with; `None` (skipped) if it was released since the list.
        let known = |w: &Watch| {
            (w.entries.get(&key) == Some(&entry))
                .then(|| (w.seen.get(&key).copied(), w.epoch.get(&key).copied().unwrap_or(0)))
        };
        let Some((state @ (before, _), was_partial, pool)) = watch_in(watched, |w| {
            known(w).map(|s| (s, w.partial.contains(&key), w.pools.get(&key).cloned()))
        }) else {
            continue;
        };
        let mut h = blake3::Hasher::new();
        let mut unreadable = false;
        for off in (0..size).step_by(buf.len()) {
            let n = buf.len().min(size - off);
            unreadable |= !read_bytes(pool.as_deref(), at + off, &mut buf[..n], was_partial);
            h.update(&buf[..n]);
        }
        let h = h.finalize();
        watch_in(watched, |w| {
            // Released, or begun/applied by a submission while it was hashed here: this reading is
            // the older one, and the next round compares with the submission's.
            if known(w) != Some(state) {
                return;
            }
            if unreadable {
                w.partial.insert(key);
            } else {
                w.partial.remove(&key);
            }
            w.seen.insert(key, h);
            if before != Some(h) && !(target && before.is_none()) {
                w.dirty.insert(key);
            }
        });
    }
}

fn start_scanner() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("victory texture scan".into()).spawn(|| {
            let mut buf = vec![0u8; 1 << 20];
            loop {
                scan_round(&WATCH, &mut buf);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
    });
}

/// A texture the host keeps for a title texture.
pub struct Tracked<B: ResourceBackend> {
    pub texture: Rc<B::Texture>,
    info: TextureInfo,
    /// Which texture this is: its key (an object's address, a placement) names another one later.
    serial: u64,
    /// The hash of the memory last uploaded; `None` for targets and before the first upload.
    last: Option<blake3::Hash>,
    /// Uploads of its memory so far.
    pub uploads: u64,
    /// Bitmask of mip levels (`mip < 32`) uploaded with non-zero texels (or `u32::MAX` for targets
    /// and UAV textures that GPU passes may write). Clear bits sit at zero in the fresh host
    /// texture, so all-zero fine mips of streamed textures skip detile and `upload_texture`.
    uploaded_mips: u32,
    /// Submission serial (`Title::sync_serial`) through which its memory is probed at submission
    /// rather than left to the scanner (0: cold), and the hash of its samples when it was last
    /// read whole (a probe that reads others reports a write).
    hot_until: u64,
    probe: Option<blake3::Hash>,
    /// Whether its last write showed in the sampled probe (a movie's planes, a streamed texture:
    /// written whole): then it is read whole only every `WHOLE_EVERY` probes (backing off across
    /// consecutive unchanged whole checks), and probes skipped since. Set at each upload
    /// (`sync_apply`), and returned to periodic checks once a follow-up whole read finds no write.
    sampled_sees_writes: bool,
    unchanged_whole: u8,
    skipped: u32,
    /// Whether its last whole read met an unmapped or protected range (fine mips not mapped yet).
    partial: bool,
    /// Its last upload was all zeros: it is probed (`Title::blank_keys`) until written, however
    /// long that takes (a movie's planes are made blank about 40 s before the movie decodes into
    /// them; once left to the scanner, their first frames showed green for a second or two).
    blank: bool,
    /// Render-target views already made, kept so the same attachment keeps its identity.
    #[cfg(target_os = "macos")]
    target_views: Vec<(ViewDesc, Rc<B::Texture>)>,
    /// The GPU wrote it (a copy, draw, clear or dispatch): its host texture holds what title
    /// memory does not, and is never filled from that memory again.
    gpu_written: bool,
    /// The submission ([`Budget::serial`]) it was last named or shown at: the texture budget gives
    /// up the least recently used first.
    used: u64,
}

impl<B: ResourceBackend> Tracked<B> {
    /// Whether the host texture is no longer filled from title memory: the GPU wrote it
    /// ([`Title::gpu_written`]). One never uploaded and not a render target takes its first upload.
    fn kept_from_memory(&self) -> bool {
        self.gpu_written && (self.uploads > 0 || self.info.render_target())
    }
}

/// The other format of an sRGB pair.
#[cfg(target_os = "macos")]
fn srgb_twin(f: Format) -> Option<Format> {
    Some(match f {
        Format::Rgba8Unorm => Format::Rgba8UnormSrgb,
        Format::Rgba8UnormSrgb => Format::Rgba8Unorm,
        Format::Bgra8Unorm => Format::Bgra8UnormSrgb,
        Format::Bgra8UnormSrgb => Format::Bgra8Unorm,
        _ => return None,
    })
}

/// Submissions a texture whose memory changed stays hot without a further change.
pub(super) const HOT: u32 = 600;
/// Time one submission spends probing hot textures, and again blank ones. Boot movies and UI
/// glyph caches have a few hot textures and a few hundred blank ones (the movie's planes among
/// them, blank until the title decodes into them), all of which a submission must reach: a count
/// limit left a plane's first frames unseen (solid green). When world streaming realizes thousands
/// of blank textures the time bounds the probing and the cursor resumes where the last submission
/// stopped, while the scanner thread catches the rest off the host lock.
///
/// shortcut: past a few thousand blank textures a write is noticed a few submissions late (unless
/// the scanner catches it first); at least `MIN_SYNC_HOT` of each are probed whatever the time.
const PROBE_BUDGET: std::time::Duration = std::time::Duration::from_millis(2);
/// Hot textures a submission probes at least, however long the probes take.
const MIN_SYNC_HOT: usize = 32;
/// Pages of a large hot texture's memory a probe reads, spread over it.
const PROBE_PAGES: usize = 8;
/// Bytes sampled per page when probing a large hot texture.
const PROBE_SAMPLE: usize = 256;
/// Hot textures up to this size whose last write its samples missed are read whole at every probe
/// (a glyph cache gets a few glyphs at a time). Those written so that the samples showed it (a
/// streamed texture, written whole after its blank upload) are read whole now and then
/// (`WHOLE_EVERY`): in the world they are thousands, and read whole at every probe they were 2,000
/// textures and 500 MiB a second, most of what a submission read.
///
/// shortcut: an atlas whose first glyphs reach a sampled page gets its next ones up to
/// `WHOLE_EVERY` probes late, once (then its samples have missed a write); per-sample hashes
/// would tell a partial write from a whole one.
const PROBE_WHOLE: usize = 4 << 20;

/// A hot texture whose writes show in its samples (or whose follow-up whole check after a missed
/// write found nothing further) is read whole once in this many probes, doubling after each
/// further unchanged whole check up to `WHOLE_EVERY << WHOLE_BACKOFF_MAX`, to catch a write that
/// misses the samples (a glyph added to an atlas a movie-like rewrite went through). Reading the
/// movie's chroma planes whole at every submission cost 15 ms a submission, halving the movie's
/// frame rate.
pub(super) const WHOLE_EVERY: u32 = 8;
/// Maximum doubling steps (`WHOLE_EVERY << 5` = 256 probes) for repeated unchanged whole checks
/// while the background scanner continues to watch every texture (a guess).
const WHOLE_BACKOFF_MAX: u32 = 5;
/// Bytes of hot textures one submission reads whole to check them, once `MIN_SYNC_HOT` were
/// probed: the reading is off the host's lock but not free (a guess at what a submission can
/// spare: about 5 ms of one thread under Rosetta).
const WHOLE_BUDGET: usize = 8 << 20;
/// Of those, bytes of the ones read whole only now and then (`WHOLE_EVERY`: their samples showed
/// their last write). In the world they are nearly all of the hot textures and the title submits
/// some 30 times a frame; one that is due and not read stays due (a guess, as above: 1 ms).
const WHOLE_SPARE: usize = 1 << 20;

/// Byte offset of a `PROBE_SAMPLE`-aligned window inside mip `mip` of slice 0 (at its center
/// element so letterboxed images and mip-tail micro-tiles land on real texels rather than zero
/// padding).
fn mip_sample_offset(layout: &Layout, size: usize, mip: u32) -> Option<usize> {
    let m = layout.mips.get(mip as usize)?;
    let byte = layout
        .offset(mip, m.width / 2, m.height / 2, 0)
        .unwrap_or_else(|_| m.offset + m.tail.map_or(0, |t| u64::from(t.offset)));
    let off = (byte as usize) & !(PROBE_SAMPLE - 1);
    (off < size).then_some(off)
}

/// Hash of the bytes a probe of `size` bytes at `at` reads ([`probe_read`]).
fn probe_hash(
    memory: Option<&PoolTable>,
    at: usize,
    size: usize,
    layout: Option<&Layout>,
    whole: bool,
    known_partial: bool,
    scratch: &mut Vec<u8>,
) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    probe_read(memory, at, size, layout, whole, known_partial, scratch, |b| {
        hasher.update(b);
    });
    hasher.finalize()
}

/// Whether `PROBE_SAMPLE` bytes in the middle (or in the coarsest mips of a mipmapped texture
/// whose fine mips are still unmapped) of the `size` bytes at `at` are not all zero: a blank
/// texture's probe (a few hundred blank textures are probed every submission).
///
/// shortcut: a first write that misses both the middle and the coarse mip samples (a glyph in a
/// blank single-mip atlas) waits for the scanner; textures written whole or streamed coarse-first
/// are seen at once.
fn probe_nonzero(at: usize, size: usize, layout: &Layout, scratch: &mut Vec<u8>) -> bool {
    if size == 0 {
        return false;
    }
    let mut offsets = [0usize; 4];
    let mut count = 0usize;
    let mut push = |off: usize| {
        if off < size && !offsets[..count].contains(&off) && count < offsets.len() {
            offsets[count] = off;
            count += 1;
        }
    };
    push((size / 2) & !4095);
    if layout.mips.len() > 1 {
        let last = layout.mips.len() as u32 - 1;
        if let Some(off) = mip_sample_offset(layout, size, layout.first_tail_mip.min(last)) {
            push(off);
        }
        if let Some(off) = mip_sample_offset(layout, size, last) {
            push(off);
        }
        if layout.first_tail_mip > 0
            && let Some(off) = mip_sample_offset(layout, size, layout.first_tail_mip - 1)
        {
            push(off);
        }
    }
    for &off in &offsets[..count] {
        let n = PROBE_SAMPLE.min(size - off);
        if scratch.len() < n {
            scratch.resize(n, 0);
        }
        if super::safe_read((at + off) as *const u8, &mut scratch[..n])
            && scratch[..n].iter().any(|&b| b != 0)
        {
            return true;
        }
    }
    false
}

/// Hands `seen` the bytes a probe of `size` bytes at `at` reads: all of them if `whole`, else
/// `PROBE_PAGES` samples of `PROBE_SAMPLE` bytes spread across the texture plus one sample per mip
/// level when `layout` has a mip chain, reusing `scratch`.
///
/// shortcut: when sampled, a write that misses every sampled window goes unseen until a periodic
/// whole probe or the scanner's next round; a texture the title rewrites whole (a movie's planes)
/// or streams mip by mip (reverse-mip GFX10 layout, where mip 0 sits at the top of the buffer) is
/// caught at once without forcing immediate unchanged whole re-reads.
#[allow(clippy::too_many_arguments)]
fn probe_read(
    memory: Option<&PoolTable>,
    at: usize,
    size: usize,
    layout: Option<&Layout>,
    whole: bool,
    known_partial: bool,
    scratch: &mut Vec<u8>,
    mut seen: impl FnMut(&[u8]),
) {
    if whole || size <= PROBE_PAGES * PROBE_SAMPLE {
        const CHUNK: usize = 1 << 20;
        let cap = size.min(CHUNK);
        if scratch.len() < cap {
            scratch.resize(cap, 0);
        }
        for off in (0..size).step_by(CHUNK) {
            let n = CHUNK.min(size - off);
            read_bytes(memory, at + off, &mut scratch[..n], known_partial);
            seen(&scratch[..n]);
        }
        return;
    }
    const PAGE: usize = 4096;
    let pages = size.div_ceil(PAGE);
    let mut offsets = [0usize; PROBE_PAGES + 24];
    let mut count = 0usize;
    let mut push = |off: usize| {
        if off < size && !offsets[..count].contains(&off) && count < offsets.len() {
            offsets[count] = off;
            count += 1;
        }
    };
    for k in 0..PROBE_PAGES {
        push(k * pages / PROBE_PAGES * PAGE);
    }
    if let Some(layout) = layout.filter(|l| l.mips.len() > 1) {
        for mip in 0..layout.mips.len() as u32 {
            if let Some(off) = mip_sample_offset(layout, size, mip) {
                push(off);
            }
        }
    }
    if scratch.len() < PROBE_SAMPLE {
        scratch.resize(PROBE_SAMPLE, 0);
    }
    for &off in &offsets[..count] {
        let n = PROBE_SAMPLE.min(size - off);
        if !super::safe_read((at + off) as *const u8, &mut scratch[..n]) {
            scratch[..n].fill(0);
        }
        seen(&scratch[..n]);
    }
}

/// For the trace: hot textures a probe read whole, and their bytes.
static WHOLE_PROBES: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];
/// For the trace: placements that took the place of another at their address, and those of them
/// that had been at that address before.
static REPLACED: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Probes `keys` round-robin from `cursor`, each at most once. An expired deadline, or
/// `WHOLE_BUDGET` of whole reads, stops the round after at least `MIN_SYNC_HOT` probes (no deadline:
/// neither does). Those whose memory changed, and the hot ones to be read whole to tell, go to
/// `dirty`. Hot textures (`hot_serial = Some(sync_serial)`) that stay unchanged for `HOT`
/// submissions cool down and leave without spending a probe once expired; blank ones (`None`) stay
/// until they are written.
#[allow(clippy::too_many_arguments)]
fn probe_round<B: ResourceBackend>(
    memory: &PoolTable,
    textures: &mut Textures<B>,
    keys: &mut Vec<usize>,
    cursor: &mut usize,
    until: Option<std::time::Instant>,
    hot_serial: Option<u64>,
    scratch: &mut Vec<u8>,
    dirty: &mut std::collections::HashSet<usize>,
) {
    if keys.is_empty() {
        *cursor = 0;
        return;
    }
    // Partition invariant:
    // - Before wrap (`wrapped == false`, `start <= idx <= keys.len()`):
    //   `keys[0..start]` is unvisited, `keys[start..idx]` was visited and kept, and
    //   `keys[idx..keys.len()]` is unvisited (`swap_remove(idx)` pulls an unvisited tail element).
    // - After wrap (`wrapped == true`, `0 <= idx <= start <= keys.len()`):
    //   `keys[0..idx]` was visited after wrap, `keys[idx..start]` is unvisited, and
    //   `keys[start..keys.len()]` was visited before wrap. Decrementing `start`, swapping `idx`
    //   with `start` (the last unvisited slot), and `swap_remove(start)` (pulling an already-
    //   visited tail element into `start`) keeps both visited slices disjoint from `keys[idx..start]`.
    let remove = |keys: &mut Vec<usize>, idx: usize, start: &mut usize, wrapped: bool| {
        if wrapped {
            *start -= 1;
            keys.swap(idx, *start);
            keys.swap_remove(*start);
        } else {
            keys.swap_remove(idx);
        }
    };
    let mut start = *cursor % keys.len();
    let mut idx = start;
    let mut wrapped = false;
    let mut probed = 0;
    // Bytes of the textures to be read whole, and of those among them read so only now and then.
    let mut whole = 0;
    let mut spare = 0;
    loop {
        if !wrapped && idx >= keys.len() {
            idx = 0;
            wrapped = true;
        }
        if wrapped && idx >= start {
            break;
        }
        if probed >= MIN_SYNC_HOT && until.is_some_and(|until| std::time::Instant::now() >= until || whole >= WHOLE_BUDGET) {
            break;
        }
        let key = keys[idx];
        let Some(t) = textures.get_mut(&key) else {
            remove(keys, idx, &mut start, wrapped);
            continue;
        };
        if let Some(sync_serial) = hot_serial {
            if t.hot_until == 0 || t.hot_until < sync_serial {
                t.hot_until = 0;
                remove(keys, idx, &mut start, wrapped);
                continue;
            }
        } else if !t.blank {
            remove(keys, idx, &mut start, wrapped);
            continue;
        }
        probed += 1;
        let at = t.info.memory as usize;
        let leave = if let Some(sync_serial) = hot_serial {
            let sampled = probe_hash(
                Some(memory),
                at,
                t.info.size,
                Some(&t.info.layout),
                false,
                t.partial,
                scratch,
            );
            // Its samples at the last upload (`sync_apply` moves them on).
            let before = t.probe;
            let changed = before != Some(sampled);
            if !changed && t.uploads >= 2 && t.info.size <= PROBE_WHOLE {
                let every = if t.sampled_sees_writes {
                    WHOLE_EVERY << u32::from(t.unchanged_whole).min(WHOLE_BACKOFF_MAX)
                } else {
                    1
                };
                t.skipped = t.skipped.saturating_add(1);
                if t.skipped >= every && !(t.sampled_sees_writes && spare >= WHOLE_SPARE) {
                    t.skipped = 0;
                    if t.sampled_sees_writes {
                        spare += t.info.size;
                    }
                    // Read whole with the textures the submission reads anyway, without the host's
                    // lock ([`prepare`]), and uploaded if that is not what the host texture holds
                    // (a write the samples missed: `sync_apply`). Hashed here, the hot textures
                    // of a title that streams were hundreds of MiB a second under the lock.
                    dirty.insert(key);
                    whole += t.info.size;
                    WHOLE_PROBES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    WHOLE_PROBES[1].fetch_add(t.info.size as u64, std::sync::atomic::Ordering::Relaxed);
                }
            }
            if changed {
                t.skipped = 0;
                dirty.insert(key);
            } else if t.hot_until <= sync_serial {
                t.hot_until = 0;
            }
            t.hot_until == 0
        } else {
            // Written at last.
            let written = probe_nonzero(at, t.info.size, &t.info.layout, scratch);
            if written {
                dirty.insert(key);
                t.blank = false;
            }
            written
        };
        if leave {
            remove(keys, idx, &mut start, wrapped);
        } else {
            idx += 1;
        }
    }
    *cursor = idx;
}

/// The key of a texture a view names by address and description (no resource object): a number no
/// object address has (the top bit is set).
pub fn placed_key(address: u64, description: &[u32; 14]) -> usize {
    let mut h = blake3::Hasher::new();
    h.update(&address.to_le_bytes());
    for (i, w) in description.iter().enumerate() {
        // Words 1 and 13 are padding (stack bytes in the title's records).
        h.update(&(if i == 1 || i == 13 { 0 } else { *w }).to_le_bytes());
    }
    let b = h.finalize();
    (u64::from_le_bytes(b.as_bytes()[..8].try_into().expect("8 bytes")) | 1 << 63) as usize
}

impl<B: PresentBackend> Title<B> {
    /// Turns copying GPU-written textures back into title memory on or off.
    pub fn set_texture_back(&mut self, on: bool) {
        self.host.collect_written(on);
    }

    /// Copies back what completed submissions wrote.
    pub(super) fn copy_back(&mut self) {
        let mut done = Vec::new();
        let backend = &self.host.backend;
        self.back.pending.retain(|(fence, value, keys)| {
            let reached = backend.completed(fence) >= *value;
            if reached {
                done.extend(keys.iter().copied());
            }
            !reached
        });
        done.sort_unstable();
        done.dedup();
        for key in done {
            self.write_back(key);
        }
    }

    /// Retiles the host texture of `key` into its title memory.
    fn write_back(&mut self, key: usize) {
        let pools = self.host.memory();
        // Not a texture (a buffer view), or released since: nothing to copy.
        let Some(t) = self.textures.get_mut(&key) else { return };
        let info = &t.info;
        // Render and depth targets are left out: nothing says the title reads one.
        if info.flags & UNORDERED_ACCESS == 0 || info.render_target() {
            return;
        }
        let described = (info.format, info.width, info.height, info.depth_or_layers, info.size);
        if info.size > BACK_LIMIT {
            if self.back.logged.insert(key) {
                tracing::info!(texture = key, ?described, "a GPU-written texture is too large to copy back");
            }
            return;
        }
        let mut tiled = vec![0u8; info.size];
        let _ = read_into(Some(&pools), info.memory as usize, &mut tiled, t.partial);
        // The title wrote the memory since the host texture was filled from it: its bytes stay,
        // and the next submission uploads them.
        if t.last != Some(blake3::hash(&tiled)) {
            return;
        }
        if let Err(error) = download(&self.host.backend, &t.texture, info, &mut tiled) {
            if self.back.logged.insert(key) {
                tracing::warn!(texture = key, ?described, %error, "a GPU-written texture was not copied back");
            }
            return;
        }
        let hash = blake3::hash(&tiled);
        if t.last == Some(hash) {
            return;
        }
        // Through the pool's own mapping; `None` when the title unmapped the memory meanwhile.
        let Ok((_buffer, at)) = self.host.resolve(info.memory as u64, info.size as u64) else { return };
        // SAFETY: `resolve` gives `size` mapped, writable bytes of the pool; the buffer is kept.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), at.cast_mut(), tiled.len()) };
        // What the host texture holds, so the scanner's sighting of this write uploads nothing.
        t.last = Some(hash);
        if self.back.logged.insert(key) {
            tracing::info!(texture = key, ?described, layout = ?info.layout.desc, "a GPU-written texture is copied back into title memory");
        }
    }

    /// A title texture exists. Its host texture is made when a view, a render target or a clear
    /// first names it (most of a title's textures are never used), and filled from title memory
    /// when work next goes to the GPU.
    pub fn texture_created(&mut self, key: usize, info: TextureInfo) {
        self.pending.insert(key, info);
    }

    /// Makes the host texture of title texture `key` if it has none yet and registers it for
    /// views. `false` when there is none (not a title texture, or one the host cannot represent).
    pub fn realize(&mut self, key: usize) -> bool {
        if self.textures.contains_key(&key) || self.copies_of.contains_key(&key) {
            return true;
        }
        let Some(info) = self.pending.remove(&key) else {
            return false;
        };
        let Some(desc) = info.host_desc() else {
            // Loud once per format: views of it get no texture and their draws sample nothing.
            static SEEN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(&info.format) {
                seen.push(info.format);
                tracing::warn!(format = info.format, w = info.width, h = info.height, flags = info.flags, "a title texture format the host does not represent (first of this format)");
            }
            return false;
        };
        if self.host.is_evicted(key) {
            BUDGET_TRACE[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(owner) = self.identical_copy(&info) {
            let tex = self.textures[&owner].texture.clone();
            self.host.register_texture(key, &tex);
            self.copies_of.insert(key, (owner, info));
            SHARED_TRACE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
        let tex = match self.host.texture(&desc) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, ?desc, "no host texture");
                return false;
            }
        };
        self.at_memory.entry(info.memory as usize).or_default().push(key);
        self.host.register_texture(key, &tex);
        self.host.set_dropped_levels(key, info.dropped_levels(), desc.mips);
        self.budget.live += info.size as u64;
        static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Depth-stencil textures are GPU-written depth targets whose CPU tile layout (e.g.
        // 4-byte or 2-byte plane 0) does not match Metal's combined depth-stencil texel layout,
        // so watching or uploading CPU memory cannot produce valid replaceRegion transfers.
        if !desc.format.is_depth() {
            start_scanner();
            if info.render_target() {
                // Render targets are not read at first submission (`unloaded`), so the scanner
                // takes their baseline directly; what the title (CPU, DMA) writes into their
                // memory afterwards is uploaded when it changes.
                let pools = self.host.memory();
                watch(|w| {
                    w.entries.insert(key, (info.memory as usize, info.size, true, serial));
                    w.pools.insert(key, pools);
                });
                // Debugging (VICTORY_MAGENTA=1): render targets start magenta, so a frame that
                // stays so was never drawn into, and a black one was drawn black.
                if env_set!("VICTORY_MAGENTA") {
                    let m = &info.layout.mips[0];
                    let n = info.layout.desc.element_bytes as usize;
                    let row = m.width as usize * n;
                    let pixel = [0xffu8, 0, 0xff, 0xff];
                    let linear: Vec<u8> = (0..row * m.height as usize).map(|i| pixel[i % 4]).collect();
                    self.host.backend.upload_texture(&tex, 0, 0, &linear, row, row * m.height as usize);
                }
            } else {
                // Sampled textures enter `WATCH` once their first submission read populates `seen`
                // (`sync_apply`), avoiding a duplicate whole read by the background scanner.
                self.unloaded.insert(key);
            }
        }
        let uploaded_mips = if info.render_target() || info.flags & UNORDERED_ACCESS != 0 {
            u32::MAX
        } else {
            0
        };
        self.textures.insert(
            key,
            Tracked {
                texture: tex,
                info,
                serial,
                last: None,
                uploads: 0,
                uploaded_mips,
                hot_until: 0,
                probe: None,
                sampled_sees_writes: false,
                unchanged_whole: 0,
                skipped: 0,
                partial: false,
                blank: false,
                #[cfg(target_os = "macos")]
                target_views: Vec::new(),
                gpu_written: false,
                used: self.budget.serial,
            },
        );
        true
    }

    /// A live copy that title texture `info` would duplicate: one over the same memory with the
    /// same layout, dimensions and host format, filled from that memory the same way (B4G4R4A4 is
    /// converted on upload), not a render target or depth (the GPU's, not memory's) and not
    /// GPU-written. Views of either then sample the same texels from one texture.
    fn identical_copy(&self, info: &TextureInfo) -> Option<usize> {
        let desc = info.desc()?;
        if info.render_target() || desc.format.is_depth() {
            return None;
        }
        self.at_memory.get(&(info.memory as usize))?.iter().copied().find(|k| {
            let o = &self.textures[k].info;
            !self.textures[k].gpu_written
                && !o.render_target()
                && o.size == info.size
                && o.layout == info.layout
                && o.dimension == info.dimension
                && (o.width, o.height, o.depth_or_layers) == (info.width, info.height, info.depth_or_layers)
                && o.flags & UNORDERED_ACCESS == info.flags & UNORDERED_ACCESS
                && (o.format == 115) == (info.format == 115)
                && o.desc().is_some_and(|d| d.format == desc.format)
        })
    }

    /// The key whose copy title texture `key` uses: its own, or the identical copy it shares.
    fn owner_of(&self, key: usize) -> usize {
        self.copies_of.get(&key).map_or(key, |&(owner, _)| owner)
    }

    /// The format of the host texture of title texture `key`.
    pub fn format_of(&self, key: usize) -> Option<Format> {
        let key = self.owner_of(key);
        self.textures.get(&key).and_then(|t| t.info.desc()).map(|d| d.format)
    }

    /// The host texture of title texture `key`.
    pub fn texture_of(&mut self, key: usize) -> Option<Rc<B::Texture>> {
        self.realize(key);
        let key = self.owner_of(key);
        let serial = self.budget.serial;
        self.textures.get_mut(&key).map(|t| {
            t.used = serial;
            t.texture.clone()
        })
    }

    /// Resolves a render-target view to its attachment and its effective format. The base
    /// attachment can use the texture itself. A narrowed view also supplies the mip's extent.
    #[cfg(target_os = "macos")]
    pub(crate) fn target_of(
        &mut self,
        record: crate::descriptor::RtvRecord,
    ) -> Option<(Rc<B::Texture>, Format)> {
        let key = record.resource as usize;
        self.realize(key);
        let key = self.owner_of(key);
        let t = self.textures.get_mut(&key)?;
        let desc = t.info.desc()?;
        let (mip, slice) = record.native_rtv_subresource(
            t.info.dimension,
            t.info.depth_or_layers,
            desc.mips,
            t.info.layout.desc.samples,
        )?;
        let base = (mip, slice) == (0, 0);
        let own = desc.format;
        let twin = format(record.format()).filter(|&f| srgb_twin(own) == Some(f));
        // A failed format reinterpretation can fall back to the resource format, but must
        // keep the same mip and slice.
        for format in twin.into_iter().chain([own]) {
            if base && format == own {
                return Some((t.texture.clone(), own));
            }
            let view = ViewDesc {
                format,
                mips: if base { (0, usize::MAX) } else { (mip, 1) },
                slices: if base { (0, usize::MAX) } else { (slice, 1) },
            };
            if let Some((_, texture)) = t.target_views.iter().find(|(cached, _)| *cached == view) {
                return Some((texture.clone(), format));
            }
            match self.host.backend.create_texture_view(&t.texture, &view) {
                Ok(texture) => {
                    let texture = Rc::new(texture);
                    t.target_views.push((view, texture.clone()));
                    return Some((texture, format));
                }
                Err(e) => tracing::warn!(error = %e, ?view, "no host view for a render target view"),
            }
        }
        None
    }

    /// The GPU writes title texture `key` (made already): a copy, draw or clear recorded into it,
    /// or a prepared dispatch's unordered-access view of it. From now on the host texture is the
    /// GPU's and the CPU-side sync no longer fills it from title memory (which titles change under
    /// their render targets: uploaded, that was block noise over what the GPU drew).
    /// A sampled texture not uploaded yet still takes its first upload, at the submission and so
    /// before the write runs (a copy of a region keeps the rest).
    /// shortcut: what the title writes into its memory afterwards is not seen; when a title is
    /// found to refill a GPU-written texture from the CPU, upload a change again.
    pub fn gpu_written(&mut self, key: usize) {
        let key = self.owner_of(key);
        // A placed view sharing an object's texture ([`Self::alias_placed`]) writes the object's.
        let key = self.shared.iter().find(|(_, (_, placed))| *placed == key).map_or(key, |(&object, _)| object);
        if let Some(t) = self.textures.get_mut(&key) {
            t.gpu_written = true;
            (t.hot_until, t.blank) = (0, false);
            if t.kept_from_memory() {
                watch(|w| w.released(key));
            }
        }
    }

    /// How many times title texture `key`'s memory was detiled into its host texture.
    pub fn uploads_of(&self, key: usize) -> u64 {
        self.textures.get(&self.owner_of(key)).map_or(0, |t| t.uploads)
    }

    pub fn texture_released(&mut self, key: usize) {
        self.copies_of.remove(&key);
        self.covered.remove(&key);
        if let Some(t) = self.textures.remove(&key) {
            self.budget.live -= t.info.size as u64;
            let at = t.info.memory as usize;
            if let Some(keys) = self.at_memory.get_mut(&at) {
                keys.retain(|&k| k != key);
                if keys.is_empty() {
                    self.at_memory.remove(&at);
                }
            }
            // The copy stays with those that shared it: the first takes it over (their entries
            // show it, so it is neither made nor uploaded again), the others share its.
            let mut sharing: Vec<usize> = self.copies_of.iter().filter(|(_, (o, _))| *o == key).map(|(&k, _)| k).collect();
            sharing.sort_unstable();
            if let Some((&first, rest)) = sharing.split_first() {
                let (_, info) = self.copies_of.remove(&first).expect("listed");
                for k in rest {
                    self.copies_of.get_mut(k).expect("listed").0 = first;
                }
                if self.unloaded.contains(&key) {
                    self.unloaded.insert(first);
                }
                self.budget.live += info.size as u64;
                self.at_memory.entry(at).or_default().push(first);
                watch(|w| w.entries.insert(first, (at, info.size, false, t.serial)));
                let mut t = t;
                t.info = info;
                self.textures.insert(first, t);
            }
        }
        // A placed view that shares this texture ([`Self::alias_placed`]) goes with it.
        if let Some((address, placed)) = self.shared.remove(&key)
            && self.placed.get(&address).is_some_and(|&(k, _)| k == placed)
        {
            self.placed.remove(&address);
            self.host.release_resource(placed);
        }
        self.pending.remove(&key);
        self.unloaded.remove(&key);
        self.hot_keys.retain(|&k| k != key);
        self.blank_keys.retain(|&k| k != key);
        watch(|w| w.released(key));
    }

    /// A view placed texture `key` at title `address` (in pool `pool`). The title has no object to
    /// release for it, so the texture another description placed at that address before goes
    /// now (guess: the title is done with it once it places another there), and every texture
    /// placed in a pool goes when the pool is released ([`Self::sync_dirty`]).
    pub fn texture_placed(&mut self, address: u64, key: usize, pool: usize, size: usize) {
        // A view of a copy given up before comes back at its own address: not a new placement.
        if self.placed.get(&address).is_none_or(|&(k, _)| k != key) {
            self.give_up_overlapped(address, key, size as u64);
        }
        self.placed_reach = self.placed_reach.max(size as u64);
        if let Some((old, _)) = self.placed.insert(address, (key, pool))
            && old != key
        {
            if env_set!("VICTORY_TRACE_TEXTURE_SYNC") {
                // Whether two descriptions keep taking one address from each other.
                static WERE: std::sync::Mutex<Option<std::collections::HashSet<(u64, usize)>>> = std::sync::Mutex::new(None);
                let mut were = WERE.lock().unwrap_or_else(|e| e.into_inner());
                let were = were.get_or_insert_default();
                were.insert((address, old));
                REPLACED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                REPLACED[1].fetch_add(u64::from(were.contains(&(address, key))), std::sync::atomic::Ordering::Relaxed);
            }
            self.host.forget_resource(old);
            self.texture_released(old);
        }
    }

    /// Placed copies over `[address, address + size)` that start elsewhere, which placed texture
    /// `key` now covers, are marked covered: given up once nothing uses them
    /// ([`Self::give_up_covered`]). Guess: the title is done with a placed view once it places
    /// another over its memory (it does so when it streams something else in). Never marked:
    /// copies that hold what title memory does not (render targets, depth, GPU-written, ever
    /// viewed for unordered access).
    fn give_up_overlapped(&mut self, address: u64, key: usize, size: u64) {
        let end = address.saturating_add(size);
        let over: Vec<usize> = self
            .placed
            .range(address.saturating_sub(self.placed_reach)..end)
            .filter(|&(&a, &(k, _))| a != address && k != key)
            .map(|(_, &(k, _))| k)
            .filter(|k| {
                self.textures.get(k).is_some_and(|t| {
                    let o = t.info.memory as u64;
                    o < end
                        && o + t.info.size as u64 > address
                        && self.holds_only_memory(*k, t)
                })
            })
            .collect();
        self.covered.extend(over);
    }

    /// At each sync: covered copies ([`Self::give_up_overlapped`]) that nothing uses any more are
    /// given up as the budget gives one up ([`Self::evictable`]: no heap entry shows one, so no
    /// work in flight reads it through an entry, and no list holds one), and none a submission in
    /// flight names. One still shown waits until the title writes another record over that entry.
    /// One named again later is made again from title memory.
    fn give_up_covered(&mut self) {
        let keys: Vec<usize> = self.covered.iter().copied().collect();
        for k in keys {
            let Some(t) = self.textures.get(&k) else {
                self.covered.remove(&k);
                continue;
            };
            if self.budget.named.contains_key(&k) || !self.evictable(k, t) {
                continue;
            }
            let info = t.info.clone();
            OVERLAP_TRACE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            OVERLAP_TRACE[1].fetch_add(info.size as u64, std::sync::atomic::Ordering::Relaxed);
            self.host.evict_texture(k);
            self.texture_released(k);
            self.pending.insert(k, info);
        }
    }

    /// A placed R9G9B9E5 texture `key` laid out exactly as a live R32_UINT resource object at the
    /// same memory (the title reads as floats what compute wrote as integers): `key` names the
    /// object's host texture, which its views cast, instead of a copy only title memory fills.
    /// `false` (nothing done) for any other placed texture.
    pub fn alias_placed(&mut self, address: u64, key: usize, info: &TextureInfo) -> bool {
        let over = |o: &TextureInfo| {
            o.format == 42 && o.memory == info.memory && o.size == info.size && o.layout == info.layout
        };
        // Object keys are addresses; placed keys have the top bit set ([`placed_key`]).
        let object = |k: usize| k >> 63 == 0;
        if info.format != 67 {
            return false;
        }
        let Some(obj) = self
            .textures
            .iter()
            .map(|(&k, t)| (k, &t.info))
            .chain(self.pending.iter().map(|(&k, i)| (k, i)))
            .find(|&(k, i)| object(k) && over(i))
            .map(|(k, _)| k)
        else {
            return false;
        };
        let Some(texture) = self.texture_of(obj) else { return false };
        self.host.register_texture(key, &texture);
        self.shared.insert(obj, (address, key));
        static ALIASED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = ALIASED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n <= 1000 {
            tracing::info!(placed = format!("{key:#x}"), object = format!("{obj:#x}"), aliased = n, "a placed R9G9B9E5 view shares an R32_UINT texture");
        }
        true
    }

    /// Releases the textures placed in pools the title released since the last call.
    fn release_placed(&mut self) {
        let gone = self.host.released_pools();
        if gone.is_empty() {
            return;
        }
        let dead: Vec<u64> = self
            .placed
            .iter()
            .filter(|(_, (_, pool))| gone.contains(pool))
            .map(|(&a, _)| a)
            .collect();
        for a in dead {
            let (key, _) = self.placed.remove(&a).expect("listed above");
            self.host.release_resource(key);
            self.texture_released(key);
        }
    }

    /// Brings the textures the host samples up to date with title memory: those whose memory
    /// changed since their last upload are detiled again.
    pub fn sync_textures(&mut self) {
        let keys: Vec<usize> = self
            .textures
            .iter()
            .filter(|(_, t)| !t.info.render_target())
            .map(|(&k, _)| k)
            .collect();
        let jobs = self.jobs(keys);
        self.sync(jobs);
    }

    /// What a submission does: the textures not uploaded yet, and those the scanner or hot probe
    /// found changed since. (`glue::submit` takes the same steps, the middle one without the
    /// host's lock.)
    pub fn sync_dirty(&mut self) {
        let jobs = self.sync_jobs();
        self.sync(jobs);
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn sync_dirty_without_probe_deadline(&mut self) {
        let jobs = self.sync_jobs_within(None);
        self.sync(jobs);
    }

    /// Reads `jobs` and uploads what changed.
    pub(super) fn sync(&mut self, jobs: Vec<Job>) {
        for batch in batches(jobs) {
            let prepared = prepare(&batch);
            self.sync_apply(prepared);
        }
    }

    /// The textures a submission must read: those not uploaded yet, and those the scanner or
    /// the probes of hot and blank textures found changed.
    pub(super) fn sync_jobs(&mut self) -> Vec<Job> {
        self.sync_jobs_within(Some(PROBE_BUDGET))
    }

    /// [`Self::sync_jobs`] with `budget` of time for each round of probes (none: every hot and
    /// blank texture is probed).
    pub(super) fn sync_jobs_within(&mut self, budget: Option<std::time::Duration>) -> Vec<Job> {
        self.sync_serial = self.sync_serial.wrapping_add(1);
        self.release_placed();
        for key in self.host.take_wanted() {
            self.realize(key);
        }
        self.keep_to_budget();
        self.give_up_covered();
        let pools = self.host.memory();
        let mut dirty = watch(|w| std::mem::take(&mut w.dirty));
        // Hot textures (a movie's planes, a glyph cache) are probed synchronously, round-robin within
        // a time budget so world streaming (thousands of blank textures) never stalls submissions;
        // the ones that changed lately first, then the blank ones.
        let started = std::time::Instant::now();
        let until = budget.map(|budget| started + budget);
        probe_round(&pools, &mut self.textures, &mut self.hot_keys, &mut self.hot_cursor, until, Some(self.sync_serial), &mut self.probe_buf, &mut dirty);
        // The blank ones get a budget of their own: a busy movie must not starve its next planes.
        let hot = std::time::Instant::now();
        let until = budget.map(|budget| hot + budget);
        probe_round(&pools, &mut self.textures, &mut self.blank_keys, &mut self.blank_cursor, until, None, &mut self.probe_buf, &mut dirty);
        if env_set!("VICTORY_TRACE_TEXTURE_SYNC") {
            SyncCost { probe: [hot - started, hot.elapsed()], hot: self.hot_keys.len() as u64, ..Default::default() }.log();
        }
        let unloaded = std::mem::take(&mut self.unloaded);
        self.jobs(keys_to_sync(&unloaded, &dirty))
    }

    pub(super) fn jobs(&self, keys: Vec<usize>) -> Vec<Job> {
        let pools = self.host.memory();
        let job = |key| {
            let t = self.textures.get(&key)?;
            // Depth-stencil memory is not laid out as the host's texels are (`realize`).
            if t.info.desc().is_some_and(|d| d.format.is_depth()) || t.kept_from_memory() {
                return None;
            }
            Some(Job {
                key,
                serial: t.serial,
                pools: pools.clone(),
                memory: t.info.memory as usize,
                size: t.info.size,
                format: t.info.format,
                dimension: t.info.dimension,
                layout: t.info.layout.clone(),
                last: t.last,
                uploaded_mips: t.uploaded_mips,
                partial: t.partial,
                skip: t.info.dropped_levels(),
            })
        };
        let out: Vec<Job> = keys.into_iter().filter_map(job).collect();
        if !out.is_empty() {
            watch(|w| {
                for j in &out {
                    if w.entries.contains_key(&j.key) {
                        *w.epoch.entry(j.key).or_default() += 1;
                    }
                }
            });
        }
        out
    }

    /// Uploads what [`prepare`] detiled, into the textures that are still the ones it read.
    pub(super) fn sync_apply(&mut self, prepared: Prepared) {
        let Title {
            host,
            textures,
            sync_serial,
            hot_keys,
            blank_keys,
            stats,
            ..
        } = self;
        let started = std::time::Instant::now();
        let mut cost = SyncCost { batches: 1, prepare: prepared.took, ..Default::default() };
        let mut read = Vec::new();
        for r in prepared.ready {
            cost.textures += 1;
            cost.bytes += r.bytes as u64;
            cost.read += r.cost[0];
            cost.hash += r.cost[1];
            cost.detile += r.cost[2];
            cost.unreadable += u64::from(r.unreadable);
            // Released while it was read (its key may name another texture by now).
            // Or a copy into it was recorded while it was read: the GPU's from now on.
            let Some(t) = textures.get_mut(&r.key).filter(|t| t.serial == r.serial && !t.kept_from_memory()) else {
                continue;
            };
            let key = r.key;
            t.partial = r.unreadable;
            t.skipped = 0;
            let entry = (t.info.memory as usize, t.info.size, t.info.render_target(), t.serial);
            cost.first += u64::from(t.last.is_none());
            // What a probe of this memory reads: the next one reports a write since, not this
            // content again (a texture's first probe used to, and that was a whole read wasted).
            let sampled = t.probe.replace(r.probe);
            // `last` again: a copy back into title memory may have moved it meanwhile.
            let Some((levels, detiled_mips, detiled)) = r.levels.filter(|_| t.last != Some(r.hash)) else {
                // Nothing changed since the last upload: if its previous write missed the samples,
                // this follow-up whole read has checked for an immediate next write, and further
                // checks return to `WHOLE_EVERY` within `WHOLE_SPARE` with backoff.
                if t.sampled_sees_writes {
                    t.unchanged_whole = t.unchanged_whole.saturating_add(1);
                } else {
                    t.sampled_sees_writes = true;
                }
                if !t.kept_from_memory() {
                    read.push((key, entry, r.hash, r.unreadable));
                }
                cost.unchanged += 1;
                continue;
            };
            t.unchanged_whole = 0;
            if let Some(sampled) = sampled {
                // Whoever found this write: one its samples do not show has it read whole at
                // every probe while it is hot, one they show only now and then.
                t.sampled_sees_writes = sampled != r.probe;
            }
            cost.blank += u64::from(r.blank);
            if !t.info.render_target() {
                if r.blank && !t.blank {
                    blank_keys.push(key);
                } else if !r.blank && t.uploads > 0 {
                    // Written again after its first upload: hot for the next `HOT` submissions.
                    if t.hot_until == 0 {
                        hot_keys.push(key);
                    }
                    t.hot_until = sync_serial.wrapping_add(u64::from(HOT));
                }
                t.blank = r.blank;
            }
            // Pass trace: uploads from title memory into textures the GPU writes (targets, UAVs).
            if super::pass_on() && (t.info.render_target() || t.info.flags & UNORDERED_ACCESS != 0) {
                eprintln!("PASSUPLOAD key={key:#x} fmt={} {}x{} at={:#x} rt={} uav={} uploads={} blank={} unreadable={}", t.info.format, t.info.width, t.info.height, t.info.memory as usize, t.info.render_target(), t.info.flags & UNORDERED_ACCESS != 0, t.uploads, r.blank, r.unreadable);
            }
            upload_levels(&host.backend, &t.texture, &levels);
            t.last = Some(r.hash);
            match detiled {
                Ok(()) => {
                    t.uploads += 1;
                    stats.detiles += 1;
                    if !t.info.render_target() && t.info.flags & UNORDERED_ACCESS == 0 {
                        t.uploaded_mips = if r.blank { 0 } else { t.uploaded_mips | detiled_mips };
                    }
                }
                Err(e) => {
                    stats.undetilable += 1;
                    tracing::warn!(texture = key, error = %e, "a title texture was not detiled");
                }
            }
            if !t.kept_from_memory() {
                read.push((key, entry, r.hash, r.unreadable));
            }
        }
        let pools = host.memory();
        watch(|w| {
            for (key, entry, hash, unreadable) in read {
                let (_, _, target, serial) = entry;
                let watch_entry = if target {
                    w.entries.get(&key).is_some_and(|e| e.3 == serial)
                } else {
                    w.entries.entry(key).or_insert(entry).3 == serial
                };
                if watch_entry {
                    w.pools.entry(key).or_insert_with(|| pools.clone());
                    *w.epoch.entry(key).or_default() += 1;
                    if unreadable {
                        w.partial.insert(key);
                    } else {
                        w.partial.remove(&key);
                    }
                    // What the scanner reported while this was read is this same content.
                    if w.seen.insert(key, hash) == Some(hash) {
                        w.dirty.remove(&key);
                    }
                }
            }
        });
        if env_set!("VICTORY_TRACE_TEXTURE_SYNC") {
            cost.upload = started.elapsed();
            cost.log();
        }
    }
}

/// A texture whose memory a submission reads: what the reading needs, taken under the host's lock
/// so that [`prepare`] runs without it.
pub(super) struct Job {
    pub(super) key: usize,
    serial: u64,
    pools: Arc<PoolTable>,
    memory: usize,
    size: usize,
    format: u32,
    dimension: Dimension,
    layout: Layout,
    /// The hash of what the host texture holds: memory that still hashes to it is not detiled.
    last: Option<blake3::Hash>,
    uploaded_mips: u32,
    /// Whether an earlier whole read met an unmapped or protected range.
    partial: bool,
    /// Top levels the host texture was made without ([`TextureInfo::dropped_levels`]).
    skip: usize,
}

/// What a [`Job`]'s memory held when [`prepare`] read it.
struct Ready {
    key: usize,
    serial: u64,
    bytes: usize,
    hash: blake3::Hash,
    /// What a sampled probe of that memory hashes to ([`probe_hash`]).
    probe: blake3::Hash,
    /// Some of it could not be read (and counts as zeros).
    unreadable: bool,
    /// All zeros.
    blank: bool,
    /// Detiled levels, bitmask of mips detiled, and how that went; `None` when the host texture
    /// holds this content already.
    levels: Option<(Vec<Level>, u32, Result<(), String>)>,
    /// Time reading, hashing and detiling.
    cost: [std::time::Duration; 3],
}

/// What [`prepare`] made of a batch of jobs, and how long it took.
pub(super) struct Prepared {
    ready: Vec<Ready>,
    took: std::time::Duration,
}

/// Threads a submission's textures are read on (the submitting thread is one of them).
const SYNC_THREADS: usize = 6;
/// Texture memory read at a time: its detiled copy is held until it is uploaded.
const SYNC_BATCH: usize = 256 << 20;

/// `jobs` in runs of at most `SYNC_BATCH` bytes of texture memory (a larger texture alone).
pub(super) fn batches(jobs: Vec<Job>) -> Vec<Vec<Job>> {
    let mut out: Vec<Vec<Job>> = Vec::new();
    let mut bytes = 0;
    for job in jobs {
        if out.is_empty() || bytes + job.size > SYNC_BATCH {
            out.push(Vec::new());
            bytes = 0;
        }
        bytes += job.size;
        out.last_mut().expect("pushed above").push(job);
    }
    out
}

impl Job {
    /// Reads its memory into `tiled` (kept from job to job: a fresh buffer each time maps and
    /// faults in new pages) and detiles it unless it is what the host texture holds.
    fn read(&self, tiled: &mut Vec<u8>) -> Ready {
        let started = std::time::Instant::now();
        // Sampled before the whole read: a write between the two is then a change to the next
        // probe, not part of what it compares with.
        let probe = probe_hash(
            Some(&self.pools),
            self.memory,
            self.size,
            Some(&self.layout),
            false,
            self.partial,
            tiled,
        );
        if tiled.len() < self.size {
            tiled.resize(self.size, 0);
        }
        let tiled = &mut tiled[..self.size];
        let unreadable = read_into(Some(&self.pools), self.memory, tiled, self.partial);
        let read = started.elapsed();
        let hash = blake3::hash(tiled);
        let hashed = started.elapsed();
        let changed = self.last != Some(hash);
        let addressable = self.layout.addressable();
        let blank = changed && addressable && is_all_zero(tiled);
        let levels = changed.then(|| {
            if blank && self.uploaded_mips == 0 {
                // Guess: fresh host textures start zeroed (`MTLStorageMode::Shared`; Metal does not
                // document it, see docs/native-host.md), so a never-uploaded
                // all-zero texture needs neither staging allocation, CPU detile, nor `replaceRegion`.
                (Vec::new(), 0, Ok(()))
            } else {
                detile_changed(&self.layout, self.dimension, self.format, tiled, self.uploaded_mips, self.skip)
            }
        });
        Ready {
            key: self.key,
            serial: self.serial,
            bytes: self.size,
            hash,
            probe,
            unreadable,
            blank,
            levels,
            cost: [read, hashed - read, started.elapsed() - hashed],
        }
    }
}

/// Reads, hashes and detiles the textures of `jobs`, in their order. This is the bulk of a texture
/// sync and needs nothing of the host, so it runs without the host's lock, on a few threads.
pub(super) fn prepare(jobs: &[Job]) -> Prepared {
    let started = std::time::Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::Mutex::new(Vec::with_capacity(jobs.len()));
    let work = || {
        let mut tiled = Vec::new();
        loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(job) = jobs.get(i) else { break };
            let ready = job.read(&mut tiled);
            done.lock().unwrap_or_else(|e| e.into_inner()).push((i, ready));
        }
    };
    std::thread::scope(|s| {
        for _ in 1..SYNC_THREADS.min(jobs.len()) {
            // Without the thread, the others (this one at least) read its share.
            let _ = std::thread::Builder::new().name("victory texture read".into()).spawn_scoped(s, work);
        }
        work();
    });
    let mut done = done.into_inner().unwrap_or_else(|e| e.into_inner());
    done.sort_by_key(|&(i, _)| i);
    Prepared { ready: done.into_iter().map(|(_, r)| r).collect(), took: started.elapsed() }
}

/// What syncing textures cost: logged with `VICTORY_TRACE_TEXTURE_SYNC=1`, summed over each second.
#[derive(Default)]
struct SyncCost {
    batches: u64,
    textures: u64,
    /// Never uploaded before (a new texture, or a render target's memory written for the first time).
    first: u64,
    /// Read and hashed, and what the host texture holds already.
    unchanged: u64,
    /// All zeros (detiled and uploaded all the same).
    blank: u64,
    /// With memory that could not be read.
    unreadable: u64,
    bytes: u64,
    /// Hot textures at a submission (the last of the second is logged).
    hot: u64,
    /// Under the host's lock: probing hot textures and blank ones, and uploading.
    probe: [std::time::Duration; 2],
    upload: std::time::Duration,
    /// What the reading took the submission, without the host's lock.
    prepare: std::time::Duration,
    /// On the reading threads, summed: reading title memory, hashing it, detiling it.
    read: std::time::Duration,
    hash: std::time::Duration,
    detile: std::time::Duration,
}

impl SyncCost {
    fn log(self) {
        static SUM: std::sync::Mutex<Option<(std::time::Instant, [u64; 14])>> = std::sync::Mutex::new(None);
        let mut sum = SUM.lock().unwrap_or_else(|e| e.into_inner());
        let (since, s) = sum.get_or_insert_with(|| (std::time::Instant::now(), [0; 14]));
        let us = |d: std::time::Duration| d.as_micros() as u64;
        let add = [
            self.batches, self.textures, self.first, self.unchanged, self.blank, self.unreadable, self.bytes,
            us(self.probe[0]), us(self.probe[1]), us(self.upload), us(self.prepare), us(self.read), us(self.hash), us(self.detile),
        ];
        s.iter_mut().zip(add).for_each(|(s, a)| *s += a);
        static HOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if self.hot > 0 {
            HOT.store(self.hot, std::sync::atomic::Ordering::Relaxed);
        }
        if since.elapsed().as_secs() >= 1 {
            let take = |a: &std::sync::atomic::AtomicU64| a.swap(0, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(
                "texture sync in the last {:.1} s: {} batches, {} textures ({} for the first time, {} unchanged, {} blank, {} partly unreadable), {} MiB; under the lock: probes of {} hot textures {} ms ({} read whole, {} MiB), of blank ones {} ms, uploads {} ms; off it: {} ms (on its threads: read {} ms, hash {} ms, detile {} ms); placements replaced {} ({} by one that was there before)",
                since.elapsed().as_secs_f32(), s[0], s[1], s[2], s[3], s[4], s[5], s[6] >> 20,
                take(&HOT), s[7] / 1000, take(&WHOLE_PROBES[0]), take(&WHOLE_PROBES[1]) >> 20, s[8] / 1000, s[9] / 1000,
                s[10] / 1000, s[11] / 1000, s[12] / 1000, s[13] / 1000,
                take(&REPLACED[0]), take(&REPLACED[1])
            );
            *sum = None;
        }
    }
}

/// The texture budget: how many bytes of sampled host texture copies the host keeps before it
/// gives up the least recently used of those it may ([`Title::keep_to_budget`]).
#[derive(Default)]
pub struct Budget {
    /// Bytes allowed (0: no budget).
    pub limit: u64,
    /// Bytes of every host texture copy made from a title texture (its size in title memory).
    pub live: u64,
    /// Submissions so far: the clock [`Tracked::used`] is read on.
    serial: u64,
    /// The next submission a check over budget looks for textures to give up (a check that found
    /// too few waits [`RECHECK`] submissions: what entries show changes as the title rewrites them).
    next_check: u64,
    /// When the trace last logged.
    logged: Option<std::time::Instant>,
    /// Resources the tables of submissions being made (and of their held work) name, with how many
    /// of those submissions name each: never given up until the submission is handed to its queue
    /// ([`Title::keep_named`], [`Title::let_named_go`]).
    named: HashMap<usize, u32>,
}

impl Budget {
    /// Whether the trace logs at this submission's check (once a second).
    fn trace_due(&self) -> bool {
        env_set!("VICTORY_TRACE_TEXTURE_SYNC") && self.logged.is_none_or(|t| t.elapsed().as_secs() >= 1)
    }
}

/// Submissions between checks while over budget with too little to give up.
pub(super) const RECHECK: u64 = 64;

/// The budget when Metal's working set leaves the title's memory too little (Series X on 16 GB).
/// Guess: about what a world frame samples; a smaller one would give up and remake textures each
/// frame.
pub const MIN_TEXTURE_BUDGET: u64 = 1536 << 20;

/// Bytes of host texture copies to allow: `VICTORY_METAL_TEXTURE_BUDGET_MB` MiB. Off (0) unless
/// that is set to a number, until there are runs on more machines.
pub fn texture_budget(var: Option<&str>) -> u64 {
    var.and_then(|v| v.trim().parse::<u64>().ok()).map_or(0, |mb| mb << 20)
}

/// The budget to try on this Mac, logged at startup: what Metal's recommended working set leaves
/// after the title's memory, at least [`MIN_TEXTURE_BUDGET`].
pub fn suggested_texture_budget(working_set: u64, title_memory: u64) -> u64 {
    working_set.saturating_sub(title_memory).max(MIN_TEXTURE_BUDGET)
}

/// Which of `candidates` (key, bytes, last use) to give up to free `over` bytes: the least recently
/// used first (the lower key first among equals, so the choice does not depend on map order).
pub(super) fn choose_evictions(mut candidates: Vec<(usize, u64, u64)>, over: u64) -> Vec<usize> {
    candidates.sort_unstable_by_key(|&(key, _, used)| (used, key));
    let mut freed = 0;
    candidates
        .into_iter()
        .take_while(|&(_, bytes, _)| {
            let more = freed < over;
            freed += bytes;
            more
        })
        .map(|(key, ..)| key)
        .collect()
}

/// Bytes of title memory under `ranges` (start, length), each byte once: copies over one memory
/// (aliasing objects, placed descriptions at other starts) are counted once here.
pub(super) fn distinct_bytes(mut ranges: Vec<(usize, usize)>) -> u64 {
    ranges.sort_unstable();
    let (mut total, mut end) = (0u64, 0usize);
    for (start, len) in ranges {
        let stop = start.saturating_add(len);
        if stop > end {
            total += (stop - start.max(end)) as u64;
            end = stop;
        }
    }
    total
}

/// For the trace: textures given up under the budget, their bytes, and evicted ones made again.
static BUDGET_TRACE: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// For the trace: placed copies given up as another placement covered them, and their bytes.
static OVERLAP_TRACE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
/// For the trace: title textures that took an identical live copy's texture instead of their own.
static SHARED_TRACE: [std::sync::atomic::AtomicU64; 1] = [const { std::sync::atomic::AtomicU64::new(0) }; 1];

impl<B: PresentBackend> Title<B> {
    /// Sets the texture budget in bytes (0: none).
    pub fn set_texture_budget(&mut self, bytes: u64) {
        self.budget.limit = bytes;
    }

    /// Whether `t` (title texture `key`) may be given up: a sampled texture filled only from title
    /// memory (not a render target, depth or GPU-written texture, nor one ever viewed for unordered
    /// access), that no heap entry shows and no recorded or submitted work holds (the host's two
    /// references are all there are: `Title::textures` and the host's resources).
    fn evictable(&self, key: usize, t: &Tracked<B>) -> bool {
        self.holds_only_memory(key, t) && Rc::strong_count(&t.texture) <= 2 && !self.host.texture_in_use(key)
    }

    /// Whether copy `t` (title texture `key`) holds only what title memory does: a sampled texture
    /// (not a render target or depth), not GPU-written, never viewed for unordered access.
    fn holds_only_memory(&self, key: usize, t: &Tracked<B>) -> bool {
        !t.info.render_target()
            && !t.gpu_written
            && t.info.desc().is_some_and(|d| !d.format.is_depth())
            && !self.host.ever_unordered(key)
    }

    /// For the trace: bytes of title memory under the copies, each byte once, and bytes of copies
    /// outside the title's pools. Copies in a pool are placed by their pool offset, so pages the
    /// title maps at two addresses count once.
    pub(super) fn memory_under_copies(&self) -> (u64, u64) {
        let pools = self.host.memory();
        let (mut ranges, mut outside) = (Vec::new(), 0);
        for t in self.textures.values() {
            match pools.resolve(t.info.memory as u64, 1) {
                Some((_, at)) => ranges.push((at as usize, t.info.size)),
                None => {
                    ranges.push((t.info.memory as usize, t.info.size));
                    outside += t.info.size as u64;
                }
            }
        }
        (distinct_bytes(ranges), outside)
    }

    /// Before the sync of a submission of `lists` (with the work held already): a texture the
    /// records of the tables they bind name is made again if it was given up, and uploaded at
    /// that sync; none is given up then. The host mirrors those records when it encodes the
    /// lists, and an entry naming a texture not made yet shows nothing: the draw would sample
    /// transparent black. The names stay out of the budget's reach until the caller hands them to
    /// [`Self::let_named_go`], once the submission is with its queue, so another submission's sync
    /// in between cannot give them up. Nothing is read under the budget with nothing given up
    /// (`glue::submit` runs one submission at a time, so none goes over budget in between), unless
    /// the trace logs at this submission, so its figures match a run with the budget on.
    #[must_use = "the names stay kept until handed to `let_named_go`"]
    /// Whether [`Self::keep_named`] has anything to do: over budget, a texture given up, or the
    /// trace due. Checked before the lists' snapshots are taken for it.
    pub(super) fn names_wanted(&self) -> bool {
        let over = self.budget.limit > 0 && self.budget.live > self.budget.limit;
        over || self.host.any_evicted() || self.budget.trace_due()
    }

    pub(super) fn keep_named<'a>(&mut self, lists: impl Iterator<Item = &'a NativeList<B>>) -> HashSet<usize>
    where
        B: 'a,
    {
        if !self.names_wanted() {
            return HashSet::new();
        }
        // shortcut: a record changed after this read (before held work is encoded) still gets the
        // cleared entry and `take_wanted`; reading again at encoding needs the title's state there.
        let held: Vec<Rc<NativeList<B>>> = self.held.segments().cloned().collect();
        let mut all: Vec<&NativeList<B>> = lists.collect();
        all.extend(held.iter().map(|l| &**l));
        let named = self.host.resources_named(all.into_iter());
        for &key in &named {
            if self.host.is_evicted(key) {
                self.realize(key);
            }
        }
        for &key in &named {
            *self.budget.named.entry(key).or_default() += 1;
        }
        named
    }

    /// The submission that [`Self::keep_named`] returned `named` for is with its queue: what only
    /// it named may be given up again.
    pub(super) fn let_named_go(&mut self, named: HashSet<usize>) {
        for key in named {
            if let Some(n) = self.budget.named.get_mut(&key) {
                *n -= 1;
                if *n == 0 {
                    self.budget.named.remove(&key);
                }
            }
        }
    }

    /// Past the budget, gives up the least recently used textures that may be given up
    /// ([`Self::evictable`]) until under it. One a view names again is made and uploaded again.
    /// Textures an entry shows, or the submission's tables name ([`Self::keep_named`]), are
    /// stamped used now.
    pub(super) fn keep_to_budget(&mut self) {
        self.budget.serial += 1;
        let serial = self.budget.serial;
        let trace = self.budget.trace_due();
        let over = self.budget.live.saturating_sub(self.budget.limit);
        let check = self.budget.limit > 0 && over > 0 && serial >= self.budget.next_check;
        if !check && !trace {
            return;
        }
        let (mut candidates, mut shown) = (Vec::new(), 0);
        let keys: Vec<usize> = self.textures.keys().copied().collect();
        for key in keys {
            let t = &self.textures[&key];
            let named = self.budget.named.contains_key(&key);
            if !named && self.evictable(key, t) {
                candidates.push((key, t.info.size as u64, t.used));
            } else if named || self.host.texture_in_use(key) {
                shown += t.info.size as u64;
                self.textures.get_mut(&key).expect("listed").used = serial;
            }
        }
        let evictable: u64 = candidates.iter().map(|c| c.1).sum();
        if check {
            for key in choose_evictions(candidates, over) {
                let t = self.textures.get(&key).expect("a candidate");
                let bytes = t.info.size as u64;
                let info = t.info.clone();
                self.host.evict_texture(key);
                self.texture_released(key);
                // Made again when a view names it next ([`Self::realize`]).
                self.pending.insert(key, info);
                BUDGET_TRACE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                BUDGET_TRACE[1].fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
            }
            if self.budget.live > self.budget.limit {
                self.budget.next_check = serial + RECHECK;
            }
        }
        if trace {
            self.budget.logged = Some(std::time::Instant::now());
            let take = |a: &std::sync::atomic::AtomicU64| a.swap(0, std::sync::atomic::Ordering::Relaxed);
            let (distinct, outside) = self.memory_under_copies();
            tracing::info!(
                "host texture copies: {} MiB, budget {} MiB; {} MiB shown by heap entries, named by the submission's tables or viewed for unordered access, {} MiB could be given up; given up in the last second {} ({} MiB), made again {}; {} MiB of distinct title memory under the copies, {} MiB of copies outside the title's pools; in the last second {} textures shared an identical copy ({} share one now), {} placed copies another placement covered were given up ({} MiB), {} covered still in use",
                self.budget.live >> 20, self.budget.limit >> 20, shown >> 20, evictable >> 20,
                take(&BUDGET_TRACE[0]), take(&BUDGET_TRACE[1]) >> 20, take(&BUDGET_TRACE[2]),
                distinct >> 20, outside >> 20,
                take(&SHARED_TRACE[0]), self.copies_of.len(), take(&OVERLAP_TRACE[0]), take(&OVERLAP_TRACE[1]) >> 20, self.covered.len(),
            );
        }
    }
}

fn keys_to_sync(unloaded: &HashSet<usize>, dirty: &HashSet<usize>) -> Vec<usize> {
    unloaded.iter().chain(dirty.difference(unloaded)).copied().collect()
}

/// The textures of a title host, by title resource key.
pub type Textures<B> = HashMap<usize, Tracked<B>>;

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn sync_keys_include_overlaps_once_and_keep_unloaded_first() {
        let unloaded = (0..20_000).collect();
        let dirty = (10_000..30_000).collect();
        let mut keys = keys_to_sync(&unloaded, &dirty);

        let (first, rest) = keys.split_at_mut(unloaded.len());
        first.sort_unstable();
        rest.sort_unstable();
        assert!(first.iter().copied().eq(0..20_000));
        assert!(rest.iter().copied().eq(20_000..30_000));
    }

    fn watched(entries: impl Into<HashMap<usize, (usize, usize, bool, u64)>>) -> Watched {
        Watched::new(Some(Watch { entries: entries.into(), ..Default::default() }))
    }

    fn dirty_this_round(watched: &Watched) -> std::collections::HashSet<usize> {
        scan_round(watched, &mut [0; 2]);
        watch_in(watched, |w| std::mem::take(&mut w.dirty))
    }

    fn seen(watched: &Watched) -> HashMap<usize, blake3::Hash> {
        watch_in(watched, |w| w.seen.clone())
    }

    #[test]
    fn scanner_forgets_removed_keys_and_reports_them_when_they_reappear() {
        let bytes = [1u8, 2, 3];
        let entry = (bytes.as_ptr() as usize, bytes.len(), false, 0);
        let w = watched([(1, entry), (2, entry)]);

        assert_eq!(dirty_this_round(&w), [1, 2].into());
        assert!(dirty_this_round(&w).is_empty());

        watch_in(&w, |w| w.entries.remove(&1));
        assert!(dirty_this_round(&w).is_empty());
        assert_eq!(seen(&w), HashMap::from([(2, blake3::hash(&bytes))]));

        watch_in(&w, |w| w.entries.insert(1, entry));
        assert_eq!(dirty_this_round(&w), [1].into());
    }

    #[test]
    fn scanner_uses_the_first_render_target_hash_as_its_baseline() {
        let mut bytes = [3u8];
        let w = watched([(1, (bytes.as_ptr() as usize, bytes.len(), true, 0))]);
        assert!(dirty_this_round(&w).is_empty());
        assert_eq!(seen(&w)[&1], blake3::hash(&bytes));

        bytes[0] = 4;
        assert_eq!(dirty_this_round(&w), [1].into());
        assert_eq!(seen(&w)[&1], blake3::hash(&bytes));
    }

    /// What a submission read whole is what the scanner compares with: it reports the write after
    /// it, not that content once more.
    #[test]
    fn scanner_compares_with_what_a_submission_read() {
        let mut bytes = [3u8];
        let w = watched([(1, (bytes.as_ptr() as usize, bytes.len(), false, 0))]);
        assert_eq!(dirty_this_round(&w), [1].into());
        bytes[0] = 4;
        // A submission read it meanwhile (`sync_apply`).
        watch_in(&w, |w| w.seen.insert(1, blake3::hash(&bytes)));
        assert!(dirty_this_round(&w).is_empty());
        bytes[0] = 5;
        assert_eq!(dirty_this_round(&w), [1].into());
    }

    /// A render target released and made again under the same key (an object's address is used
    /// again) starts over: its first sight is its baseline. Compared with the old texture's hash
    /// it counted as written, and its memory was uploaded over what the GPU had drawn into it.
    #[test]
    fn scanner_starts_over_for_a_render_target_made_again_under_a_released_key() {
        let mut bytes = [3u8];
        let entry = |serial| (bytes.as_ptr() as usize, bytes.len(), true, serial);
        let (old, new) = (entry(0), entry(1));
        let w = watched([(1, old)]);
        assert!(dirty_this_round(&w).is_empty());
        assert_eq!(seen(&w)[&1], blake3::hash(&bytes));

        // Released and made again between two rounds, its memory holding something else by then.
        watch_in(&w, |w| {
            w.released(1);
            w.entries.insert(1, new);
        });
        bytes[0] = 6;
        assert!(dirty_this_round(&w).is_empty(), "the new texture's first sight is not a change");
        assert_eq!(seen(&w)[&1], blake3::hash(&bytes));
        bytes[0] = 7;
        assert_eq!(dirty_this_round(&w), [1].into(), "a write into it after that is");
    }

    /// Every mip, array slice and depth slice goes up and comes back where it was: 1D, 2D and 3D,
    /// element sizes 1 to 8, the shapes a world run names (#46).
    #[test]
    fn download_undoes_upload_for_every_mip_and_slice() {
        use victory_formats::texture::{Desc, Swizzle};
        let Ok(metal) = victory_metal::metal::Metal::new() else { return };
        use Dimension::{Tex1d, Tex2d, Tex3d};
        // (dimension, DXGI format, element bytes, width, height, depth or layers, mips)
        let shapes = [
            (Tex1d, 28, 4, 256, 1, 1, 1),
            (Tex2d, 61, 1, 128, 64, 1, 5),
            (Tex2d, 10, 8, 256, 256, 1, 9),
            (Tex2d, 41, 4, 64, 64, 4, 6),
            (Tex3d, 41, 4, 256, 256, 16, 1),
            (Tex3d, 17, 8, 64, 64, 64, 1),
            (Tex3d, 28, 4, 32, 32, 32, 6),
        ];
        let mut ran = 0;
        for (dimension, format, element_bytes, width, height, depth, mips) in shapes {
            for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S, Swizzle::SW_64KB_D_X] {
                let what = format!("{dimension:?} {format} {width}x{height}x{depth} mips {mips} {swizzle:?}");
                let desc = Desc { dimension, width, height, depth, mips, element_bytes, block: 1, samples: 1, swizzle };
                // Not every mode exists for every shape.
                let Ok(layout) = Layout::new(&desc) else { continue };
                let mut want = vec![0u8; layout.size as usize];
                let slices = if dimension == Tex3d { 1 } else { layout.slices };
                let tiled = (0..layout.mips.len() as u32).all(|mip| {
                    let m = &layout.mips[mip as usize];
                    let row = (m.width * element_bytes) as usize;
                    let pitch = row * m.height as usize;
                    (0..slices).all(|slice| {
                        let n = pitch * layout.subresource_depth(mip).max(1) as usize;
                        let linear: Vec<u8> = (0..n).map(|i| (i % 251) as u8 ^ (mip * 37 + slice * 11) as u8).collect();
                        layout.tile(&mut want, mip, slice, &linear, row, pitch).is_ok()
                    })
                });
                if !tiled {
                    continue;
                }
                let info = TextureInfo {
                    dimension,
                    format,
                    width,
                    height,
                    depth_or_layers: depth,
                    flags: UNORDERED_ACCESS,
                    size: want.len(),
                    layout,
                    memory: std::ptr::null(),
                };
                let tex = metal.create_texture(&info.desc().unwrap()).unwrap();
                upload(&metal, &tex, &info, &want).unwrap();
                let mut got = vec![0u8; want.len()];
                download(&metal, &tex, &info, &mut got).unwrap();
                let at = got.iter().zip(&want).position(|(a, b)| a != b);
                assert!(at.is_none(), "{what}: differs first at byte {at:?}");
                eprintln!("round trip: {what}");
                ran += 1;
            }
        }
        // Linear at least, for each shape.
        assert!(ran >= shapes.len(), "{ran}");
    }

    #[test]
    fn from_host_undoes_to_host() {
        let title: Vec<u8> = (0..=255).collect();
        for format in [28, 115] {
            let mut texels = title.clone();
            to_host(format, &mut texels);
            from_host(format, &mut texels);
            assert_eq!(texels, title, "{format}");
        }
    }

    #[test]
    fn an_inaccessible_megabyte_takes_one_failed_read_and_one_region_query() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, next_region, protect};

        let len = 1 << 20;
        let memory = Region::new(len, None).unwrap();
        // SAFETY: the test owns this entire mapping until the reader finishes.
        unsafe { protect(memory.ptr, len, Protect::NO_ACCESS) }.unwrap();
        let mut out = vec![0xff; len];
        let (mut reads, mut queries) = (0, 0);
        read_texture_chunk(
            memory.ptr,
            &mut out,
            false,
            |at, out| {
                reads += 1;
                crate::native::safe_read(at, out)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(out.iter().all(|&b| b == 0));
        assert_eq!((reads, queries), (1, 1));
    }

    #[test]
    fn a_region_that_changes_after_the_query_still_preserves_readable_pages() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, next_region, page_size, protect};

        let page = page_size();
        let memory = Region::new(4 * page, None).unwrap();
        // SAFETY: the test owns all four pages and writes them before changing protection.
        unsafe {
            memory.ptr.write_bytes(0x72, 4 * page);
            protect(memory.ptr, page, Protect::NO_ACCESS).unwrap();
        }
        let mut out = vec![0xff; 4 * page - 2];
        read_texture_chunk(
            memory.ptr.wrapping_add(1),
            &mut out,
            false,
            crate::native::safe_read,
            |at| {
                let region = next_region(at);
                if at == memory.ptr.addr() + page {
                    // SAFETY: the third page is ours. Revoke access just after querying the span.
                    unsafe { protect(memory.ptr.add(2 * page), page, Protect::NO_ACCESS) }.unwrap();
                }
                region
            },
        );
        for (offset, &byte) in out.iter().enumerate() {
            let readable = matches!((offset + 1) / page, 1 | 3);
            assert_eq!(byte, if readable { 0x72 } else { 0 });
        }
    }

    #[test]
    fn a_texture_read_preserves_texels_on_both_sides_of_an_unreadable_page() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, page_size, protect};

        let page = page_size();
        let memory = Region::new(3 * page, None).unwrap();
        // SAFETY: all three pages belong to this test. Protect only the middle one after filling it.
        unsafe {
            memory.ptr.write_bytes(0x31, 2 * page);
            memory.ptr.add(2 * page).write_bytes(0x72, page);
            protect(memory.ptr.add(page), page, Protect::NO_ACCESS).unwrap();
        }
        let start = memory.ptr.wrapping_add(page - 7);
        let mut expected = vec![0; 7 + page + 11];
        expected[..7].fill(0x31);
        expected[7 + page..].fill(0x72);

        let bytes = read_memory(start, expected.len());
        assert_eq!(&bytes[..7], &expected[..7]);
        assert!(bytes[7..7 + page].iter().all(|&b| b == 0));
        assert_eq!(&bytes[7 + page..], &expected[7 + page..]);
        let mut scratch = vec![0xff; expected.len()];
        assert_eq!(
            probe_hash(None, start as usize, expected.len(), None, true, false, &mut scratch),
            blake3::hash(&expected)
        );
    }

    #[test]
    fn probe_hash_detects_writes_in_sampled_and_whole_modes() {
        let mut mem = vec![0u8; 16 * 4096];
        let mut scratch = Vec::new();
        let h0_sampled = probe_hash(None, mem.as_ptr() as usize, mem.len(), None, false, false, &mut scratch);
        let h0_whole = probe_hash(None, mem.as_ptr() as usize, mem.len(), None, true, false, &mut scratch);
        // Flip a byte in the first sampled window.
        mem[42] = 0xAB;
        let h1_sampled = probe_hash(None, mem.as_ptr() as usize, mem.len(), None, false, false, &mut scratch);
        let h1_whole = probe_hash(None, mem.as_ptr() as usize, mem.len(), None, true, false, &mut scratch);
        assert_ne!(h0_sampled, h1_sampled);
        assert_ne!(h0_whole, h1_whole);
        // Flip a byte outside sampled windows: whole mode catches it.
        mem[42] = 0;
        mem[1000] = 0xCD;
        assert_ne!(
            h0_whole,
            probe_hash(None, mem.as_ptr() as usize, mem.len(), None, true, false, &mut scratch)
        );
    }

    #[test]
    fn mip_aware_probes_catch_tail_and_coarse_mips_in_reverse_mip_layout() {
        use victory_formats::texture::{Desc, PipeConfig, Swizzle};

        // 1024x1024 BC7 (16 B per 4x4 block), 11 mips, 64 KiB XOR-swizzled with non-zero pipe-bank XOR:
        // mips 4..10 sit in the 64 KiB mip tail (past byte 0x400, leaving 0..256 permanently zero),
        // mips 3..1 sit in [64 KiB .. 0.25 * size), and mip 0 occupies the top 75% of the buffer.
        let desc = Desc {
            dimension: Dimension::Tex2d,
            width: 1024,
            height: 1024,
            depth: 1,
            mips: 11,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S_X,
        };
        let layout = Layout::new(&desc)
            .unwrap()
            .with_pipe_bank_xor(0x1b, PipeConfig::DEFAULT);
        let mut tiled = vec![0u8; layout.size as usize];
        let mut scratch = Vec::new();
        assert!(!probe_nonzero(tiled.as_ptr() as usize, tiled.len(), &layout, &mut scratch));

        // Tile only the smallest 1x1 mip (`mip = 10`) into the 64 KiB tail while mip 0 stays zero:
        // `probe_nonzero` must catch it on the next submission despite `(size / 2) & !4095` and
        // `0..256` both being zero.
        layout.tile(&mut tiled, 10, 0, &[0x5au8; 16], 16, 16).unwrap();
        assert_eq!(&tiled[..256], &[0u8; 256], "tail padding at 0..256 stays zero");
        assert!(
            probe_nonzero(tiled.as_ptr() as usize, tiled.len(), &layout, &mut scratch),
            "blank probe must detect writes into the coarsest tail mip while mip 0 is zero"
        );

        // Now step from smallest mip to mip 0 and confirm every single mip write changes the
        // sampled `probe_hash` (keeping `sampled_sees_writes == true`).
        tiled.fill(0);
        let mut prev = probe_hash(
            None,
            tiled.as_ptr() as usize,
            tiled.len(),
            Some(&layout),
            false,
            true,
            &mut scratch,
        );
        for mip in (0..desc.mips).rev() {
            let m = &layout.mips[mip as usize];
            let row = m.width as usize * 16;
            let linear = vec![0x30u8.wrapping_add(mip as u8 * 13); row * m.height as usize];
            layout.tile(&mut tiled, mip, 0, &linear, row, linear.len()).unwrap();
            let next = probe_hash(
                None,
                tiled.as_ptr() as usize,
                tiled.len(),
                Some(&layout),
                false,
                true,
                &mut scratch,
            );
            assert_ne!(next, prev, "sampled probe missed streaming write of mip {mip}");
            prev = next;
        }
    }

    #[test]
    fn read_texture_chunk_coalesces_adjacent_readable_regions() {
        use victory_platform::mem::{Protect, Region};

        let chunk = 64 << 10;
        let backing = vec![0x4du8; 8 * chunk];
        let base = backing.as_ptr().addr();
        let mut out = vec![0xffu8; 8 * chunk];
        let mut reads = 0usize;
        // Simulate 4 separate 64 KiB `mach_vm_map` regions (all readable) followed by one 256 KiB
        // `NO_ACCESS` reservation tail.
        let whole = read_texture_chunk(
            backing.as_ptr(),
            &mut out,
            true,
            |at, buf| {
                reads += 1;
                let off = at.addr() - base;
                buf.copy_from_slice(&backing[off..off + buf.len()]);
                true
            },
            |cur| {
                let idx = (cur - base) / chunk;
                if idx < 4 {
                    Some(Region {
                        base: base + idx * chunk,
                        len: chunk,
                        protect: Protect::READ_WRITE,
                    })
                } else if idx < 8 {
                    Some(Region {
                        base: base + 4 * chunk,
                        len: 4 * chunk,
                        protect: Protect::NO_ACCESS,
                    })
                } else {
                    None
                }
            },
        );
        assert!(!whole);
        assert_eq!(reads, 1, "4 adjacent readable 64 KiB regions must coalesce into 1 read");
        assert!(out[..4 * chunk].iter().all(|&b| b == 0x4d));
        assert!(out[4 * chunk..].iter().all(|&b| b == 0));
    }

    #[test]
    fn a_known_partial_chunk_skips_the_whole_read_until_every_page_is_readable() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, next_region, page_size, protect};

        let page = page_size();
        let memory = Region::new(4 * page, None).unwrap();
        // SAFETY: the test owns all four pages; pages 0..2 hold coarse-mip data and 2..4 are unmapped/no-access.
        unsafe {
            memory.ptr.write_bytes(0x5a, 2 * page);
            protect(memory.ptr.add(2 * page), 2 * page, Protect::NO_ACCESS).unwrap();
        }

        let mut out = vec![0xff; 4 * page];
        let (mut reads, mut queries) = (0, 0);
        let whole = read_texture_chunk(
            memory.ptr,
            &mut out,
            false,
            |at, buf| {
                reads += 1;
                crate::native::safe_read(at, buf)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(!whole);
        assert_eq!((reads, queries), (2, 2));
        assert!(out[..2 * page].iter().all(|&b| b == 0x5a));
        assert!(out[2 * page..].iter().all(|&b| b == 0));

        // Once marked partial, the doomed whole-span read is skipped and only readable spans are read.
        out.fill(0xff);
        (reads, queries) = (0, 0);
        let whole = read_texture_chunk(
            memory.ptr,
            &mut out,
            true,
            |at, buf| {
                reads += 1;
                crate::native::safe_read(at, buf)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(!whole);
        assert_eq!((reads, queries), (1, 2));
        assert!(out[..2 * page].iter().all(|&b| b == 0x5a));
        assert!(out[2 * page..].iter().all(|&b| b == 0));

        // SAFETY: restore access to pages 2..4 (fine mips mapped) and fill them.
        unsafe {
            protect(memory.ptr.add(2 * page), 2 * page, Protect::READ_WRITE).unwrap();
            memory.ptr.add(2 * page).write_bytes(0xa5, 2 * page);
        }
        out.fill(0xff);
        (reads, queries) = (0, 0);
        let whole = read_texture_chunk(
            memory.ptr,
            &mut out,
            true,
            |at, buf| {
                reads += 1;
                crate::native::safe_read(at, buf)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(whole, "once every page is readable again, the partial flag clears");
        assert_eq!((reads, queries), (1, 1));
        assert!(out[..2 * page].iter().all(|&b| b == 0x5a));
        assert!(out[2 * page..].iter().all(|&b| b == 0xa5));
    }

    #[test]
    fn probe_round_wraparound_removals_visit_each_live_entry_once() {
        use crate::native::vertex_tests::Capture;
        use victory_formats::texture::{Desc, Swizzle};

        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 4,
            height: 4,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .unwrap();
        let mem = [0x42u8; 256];
        let pools = PoolTable::default();
        let mut scratch = Vec::new();
        let probe = probe_hash(
            Some(&pools),
            mem.as_ptr() as usize,
            mem.len(),
            Some(&layout),
            false,
            false,
            &mut scratch,
        );
        let sync_serial = 10u64;

        let make_tracked = |hot_until: u64| Tracked::<Capture> {
            texture: Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 4, 4)),
            used: 0,
            info: TextureInfo {
                dimension: Dimension::Tex2d,
                format: 28,
                width: 4,
                height: 4,
                depth_or_layers: 1,
                flags: 0,
                layout: layout.clone(),
                memory: mem.as_ptr(),
                size: mem.len(),
            },
            serial: 1,
            last: Some(blake3::hash(&mem)),
            uploads: 2,
            uploaded_mips: u32::MAX,
            hot_until,
            probe: Some(probe),
            sampled_sees_writes: true,
            unchanged_whole: 1,
            skipped: 0,
            partial: false,
            blank: false,
            target_views: Vec::new(),
            gpu_written: false,
        };

        // 1. Explicit trace removing entries after wraparound (`cursor = 4` on 6 entries `[0..6]`,
        //    so `[4, 5]` are visited before wrap and `[0, 2]` are removed after wrap while `[1, 3]`
        //    survive).
        let mut textures: Textures<Capture> = [
            (0, make_tracked(5)),  // expired before probe (removed after wrap)
            (1, make_tracked(20)), // survives after wrap
            (2, make_tracked(10)), // probed once, then cools (`hot_until == sync_serial`) after wrap
            (3, make_tracked(20)), // survives after wrap
            (4, make_tracked(20)), // visited and kept before wrap
            (5, make_tracked(20)), // visited and kept before wrap
        ]
        .into_iter()
        .collect();
        let mut keys = vec![0, 1, 2, 3, 4, 5];
        let mut cursor = 4usize;
        let mut dirty = HashSet::new();
        probe_round(
            &pools,
            &mut textures,
            &mut keys,
            &mut cursor,
            None,
            Some(sync_serial),
            &mut scratch,
            &mut dirty,
        );
        assert!(dirty.is_empty());
        assert_eq!(textures[&0].skipped, 0, "expired entry evicted without probing");
        for k in [1, 2, 3, 4, 5] {
            assert_eq!(textures[&k].skipped, 1, "live entry {k} must be visited once, not skipped or visited twice");
        }
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 3, 4, 5]);

        // 2. Exhaustive check across all 6 initial cursor positions and all 3^6 = 729 combinations
        //    of (already expired = 5, cools after probe = 10, stays hot = 20): 4,374 schedules.
        const N: usize = 6;
        for total_states in 0..3usize.pow(N as u32) {
            let mut state = [0u64; N];
            let mut rem = total_states;
            for s in &mut state {
                *s = match rem % 3 {
                    0 => 5,
                    1 => 10,
                    _ => 20,
                };
                rem /= 3;
            }
            for start_cursor in 0..N {
                let mut textures: Textures<Capture> = (0..N)
                    .map(|k| (k, make_tracked(state[k])))
                    .collect();
                let mut keys: Vec<usize> = (0..N).collect();
                let mut cursor = start_cursor;
                let mut dirty = HashSet::new();
                probe_round(
                    &pools,
                    &mut textures,
                    &mut keys,
                    &mut cursor,
                    None,
                    Some(sync_serial),
                    &mut scratch,
                    &mut dirty,
                );
                let mut want_survivors = Vec::new();
                for k in 0..N {
                    let expected_visits = u32::from(state[k] >= sync_serial);
                    assert_eq!(
                        textures[&k].skipped, expected_visits,
                        "state={state:?} start={start_cursor} key={k}"
                    );
                    if state[k] > sync_serial {
                        want_survivors.push(k);
                    } else {
                        assert_eq!(textures[&k].hot_until, 0);
                    }
                }
                keys.sort_unstable();
                assert_eq!(keys, want_survivors, "state={state:?} start={start_cursor}");
            }
        }
    }

    #[test]
    #[ignore = "benchmark"]
    fn bench_hot_texture_serial_cooling_under_probe_budget() {
        use crate::native::vertex_tests::Capture;
        use std::time::{Duration, Instant};
        use victory_formats::texture::{Desc, Swizzle};

        const N: usize = 5_000;
        const SUBMISSIONS: u64 = 800;
        const SIZE: usize = 64 << 10;
        let backing = vec![0x5au8; N * SIZE];
        let pools = PoolTable::default();
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 128,
            height: 128,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .unwrap();
        let mut scratch = Vec::new();
        let mut textures: Textures<Capture> = (0..N)
            .map(|i| {
                let ptr = unsafe { backing.as_ptr().add(i * SIZE) };
                let probe = probe_hash(
                    Some(&pools),
                    ptr as usize,
                    SIZE,
                    Some(&layout),
                    false,
                    false,
                    &mut scratch,
                );
                (
                    i,
                    Tracked {
                        texture: Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 128, 128)),
                        used: 0,
                        info: TextureInfo {
                            dimension: Dimension::Tex2d,
                            format: 28,
                            width: 128,
                            height: 128,
                            depth_or_layers: 1,
                            flags: 0,
                            layout: layout.clone(),
                            memory: ptr,
                            size: SIZE,
                        },
                        serial: 1,
                        last: None,
                        uploads: 1,
                        uploaded_mips: u32::MAX,
                        hot_until: u64::from(HOT),
                        probe: Some(probe),
                        sampled_sees_writes: true,
                        unchanged_whole: 1,
                        skipped: 0,
                        partial: false,
                        blank: false,
                        target_views: Vec::new(),
                        gpu_written: false,
                    },
                )
            })
            .collect();
        let mut keys: Vec<usize> = (0..N).collect();
        let mut cursor = 0usize;
        let mut dirty = HashSet::new();
        let mut early_ms = 0.0;
        let mut post_cool_ms = 0.0;
        for s in 1..=SUBMISSIONS {
            let t0 = Instant::now();
            probe_round(
                &pools,
                &mut textures,
                &mut keys,
                &mut cursor,
                Some(t0 + PROBE_BUDGET),
                Some(s),
                &mut scratch,
                &mut dirty,
            );
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if s <= u64::from(HOT) {
                early_ms += ms;
            } else {
                post_cool_ms += ms;
            }
            let _ = Duration::ZERO;
        }
        eprintln!(
            "BENCH #252 ({N} hot 64 KiB textures, {SUBMISSIONS} submissions, 2 ms budget):\n\
             - submissions 1..={HOT}: avg {:.3} ms/sub (budget-bound at 2.0 ms)\n\
             - submission {}: hot_keys remaining = {} (vs {N} under per-visit decrement until ~{} submissions)\n\
             - submissions {}..={SUBMISSIONS}: avg {:.4} ms/sub",
            early_ms / f64::from(HOT),
            SUBMISSIONS,
            keys.len(),
            N * HOT as usize / 180,
            HOT + 1,
            post_cool_ms / (SUBMISSIONS - u64::from(HOT)) as f64,
        );
        assert_eq!(keys.len(), 0);
    }
}

#[cfg(test)]
mod drop_top_mip_tests {
    use super::{dropped_levels, host_mip_range};

    #[test]
    fn only_large_sampled_2d_textures_with_a_lower_level_lose_their_top() {
        assert_eq!(dropped_levels(true, false, false, 11, 2048), 1);
        assert_eq!(dropped_levels(true, false, false, 2, 1024), 1);
        assert_eq!(dropped_levels(false, false, false, 11, 2048), 0, "off unless asked for");
        assert_eq!(dropped_levels(true, true, false, 11, 2048), 0, "render targets and UAVs keep it");
        assert_eq!(dropped_levels(true, false, true, 11, 2048), 0, "3D textures keep it");
        assert_eq!(dropped_levels(true, false, false, 1, 4096), 0, "no lower level to fall back on");
        assert_eq!(dropped_levels(true, false, false, 10, 1023), 0, "small textures keep it");
    }

    #[test]
    fn views_map_title_levels_onto_the_host_texture() {
        // Nothing dropped: unchanged.
        assert_eq!(host_mip_range(2, 3, 0, 0), (2, 3));
        // Whole chain (11 title levels, 10 on the host).
        assert_eq!(host_mip_range(0, 11, 1, 10), (0, 10));
        // A view of the dropped top level alone sees the host's top.
        assert_eq!(host_mip_range(0, 1, 1, 10), (0, 1));
        // Lower levels keep their content, one index down.
        assert_eq!(host_mip_range(3, 2, 1, 10), (2, 2));
        // A count past the end is clamped to the host's levels.
        assert_eq!(host_mip_range(5, 99, 1, 10), (4, 6));
    }
}
