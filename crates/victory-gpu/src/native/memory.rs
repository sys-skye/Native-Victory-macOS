//! Title memory for the native host: which physical-page pool a title address shows.
//!
//! On the native host a GPU buffer wraps a pool's own mapping (one `MTLBuffer` per pool, zero copy),
//! so title writes and GPU writes are the same bytes. There is no shadow copy, no write guard, no
//! copy-back: `shadow.rs`, `watch.rs` and the page bitmaps of `victory_shader::emit` are Windows
//! only and never reached from here. What is left is a table from title pages to
//! `(pool, offset)`, which the pool observer of `victory_hle::modules::xmem_physical` (branch
//! `native/host-mem`, merged by `native/xmem-rest`) keeps current. The types below have the
//! observer's shape, so the host implements `PoolObserver` for [`PoolTable`] with a one-line
//! forward per method; this crate does not depend on `victory-hle`.

use std::collections::BTreeMap;
use std::sync::Mutex;

/// `xmem_physical::PoolInfo`.
#[derive(Clone, Debug)]
pub struct PoolInfo {
    pub id: usize,
    /// The pool's own read-write mapping, 16 KiB aligned, never moved until `pool_released`.
    pub base: usize,
    pub len: usize,
    pub memory_entry: Option<u32>,
}

/// `xmem_physical::Observer`.
pub trait PoolObserver: Send + Sync {
    fn pool_created(&self, pool: &PoolInfo);
    fn pool_released(&self, pool: &PoolInfo);
    fn mapped(&self, at: usize, len: usize, pool: &PoolInfo, offset: usize);
    fn unmapped(&self, at: usize, len: usize);
}

#[derive(Clone, Copy)]
struct Map {
    len: usize,
    pool: usize,
    offset: usize,
}

#[derive(Default)]
struct Inner {
    pools: BTreeMap<usize, PoolInfo>,
    /// Title address of a mapping to what it shows; mappings never overlap.
    maps: BTreeMap<usize, Map>,
    /// Pools released since the last [`PoolTable::take_released`]: their GPU buffers must go.
    released: Vec<usize>,
    /// Counts every change to pools and mappings ([`PoolTable::generation`]).
    generation: u64,
}

impl Inner {
    /// Removes `[at, at + len)` from the mappings, keeping what is left of any it cuts.
    fn cut(&mut self, at: usize, len: usize) {
        self.generation += 1;
        let end = at + len;
        // Only the immediate predecessor can overlap the start of the cut.
        let mut preceding = self
            .maps
            .range(..at)
            .next_back()
            .filter(|&(&a, m)| a + m.len > at)
            .map(|(&a, _)| a);
        while let Some(a) = preceding
            .take()
            .or_else(|| self.maps.range(at..end).next().map(|(&a, _)| a))
        {
            let m = self.maps.remove(&a).expect("overlapping mapping");
            if a < at {
                self.maps.insert(a, Map { len: at - a, ..m });
            }
            if a + m.len > end {
                let cut = end - a;
                self.maps.insert(
                    end,
                    Map {
                        len: m.len - cut,
                        offset: m.offset + cut,
                        ..m
                    },
                );
                break;
            }
        }
    }
}

/// The title page to pool table. Thread safe: the observer is called from title threads.
#[derive(Default)]
pub struct PoolTable {
    inner: Mutex<Inner>,
}

impl PoolTable {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The pool `[addr, addr + len)` lies in, and the pool-mapping address of `addr`. The range may
    /// run over several mappings when each follows the one before it both in title memory and in
    /// the same pool (the pages of one allocation are reported one mapping at a time, and titles
    /// build buffer views across them). A range over a gap, over two pools, or over mappings that
    /// are not next to each other in the pool is refused: one pointer cannot show it.
    pub fn resolve(&self, addr: u64, len: u64) -> Option<(PoolInfo, *const u8)> {
        let (addr, len) = (
            usize::try_from(addr).ok()?,
            usize::try_from(len.max(1)).ok()?,
        );
        let g = self.lock();
        let (&at, m) = g.maps.range(..=addr).next_back()?;
        if addr >= at + m.len {
            return None;
        }
        let end = addr.checked_add(len)?;
        let (mut title_end, mut pool_end) = (at + m.len, m.offset + m.len);
        while end > title_end {
            let next = g.maps.get(&title_end)?;
            // An empty mapping continues nothing: without this the walk would not advance.
            if next.len == 0 || next.pool != m.pool || next.offset != pool_end {
                return None;
            }
            title_end += next.len;
            pool_end += next.len;
        }
        let pool = g.pools.get(&m.pool)?;
        Some((
            pool.clone(),
            (pool.base + m.offset + (addr - at)) as *const u8,
        ))
    }

    /// What the title maps at `[addr, addr + len)`, as a number that changes when any mapping in
    /// it does: a hash of the mappings it crosses (title address, length, pool, offset), which
    /// must follow one another in title addresses. `None` where part of it is not mapped.
    pub fn signature(&self, addr: u64, len: u64) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let (addr, len) = (usize::try_from(addr).ok()?, usize::try_from(len.max(1)).ok()?);
        let end = addr.checked_add(len)?;
        let g = self.lock();
        let (&at, m) = g.maps.range(..=addr).next_back()?;
        if addr >= at + m.len {
            return None;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (at, m.len, m.pool, m.offset).hash(&mut h);
        let mut title_end = at + m.len;
        while end > title_end {
            let next = g.maps.get(&title_end).filter(|n| n.len > 0)?;
            (title_end, next.len, next.pool, next.offset).hash(&mut h);
            title_end += next.len;
        }
        Some(h.finish())
    }

    /// A number that changes whenever a pool or a mapping does: what a table built from
    /// [`PoolTable::mappings`] is current for.
    pub fn generation(&self) -> u64 {
        self.lock().generation
    }

    /// The mapping containing title address `addr`, if any, without copying the table.
    pub(super) fn mapping_at(&self, addr: u64) -> Option<super::addr::Mapping> {
        let addr_usize = usize::try_from(addr).ok()?;
        let g = self.lock();
        let (&at, m) = g.maps.range(..=addr_usize).next_back()?;
        if addr_usize - at >= m.len {
            return None;
        }
        let pool = g.pools.get(&m.pool)?;
        Some(super::addr::Mapping {
            at: at as u64,
            len: m.len as u64,
            pool: pool.id,
            base: pool.base,
            pool_len: pool.len as u64,
            offset: m.offset as u64,
        })
    }

    /// Every mapping, by title address: `(at, len, pool, offset in the pool)`.
    pub fn mappings(&self) -> (u64, Vec<(usize, usize, PoolInfo, usize)>) {
        let g = self.lock();
        let maps = g
            .maps
            .iter()
            .filter_map(|(&at, m)| Some((at, m.len, g.pools.get(&m.pool)?.clone(), m.offset)))
            .collect();
        (g.generation, maps)
    }

    /// Pools released since the last call; the host drops their buffers.
    pub fn take_released(&self) -> Vec<usize> {
        std::mem::take(&mut self.lock().released)
    }

    /// Reads `[addr, addr + out.len())` across the title-address spans mapped in this table,
    /// coalescing adjacent mappings (even across different pools or out-of-order physical pages)
    /// into one `read_span` call each and zeroing unmapped gaps without calling `read_span` or
    /// querying VM regions. Returns `None` when no mapping overlaps `[addr, addr + out.len())`,
    /// `Some(true)` when the entire range was mapped and read cleanly, and `Some(false)` when part
    /// of the range was unmapped or `read_span` reported a partial/failed span.
    pub fn read_mapped_with(
        &self,
        addr: usize,
        out: &mut [u8],
        mut read_span: impl FnMut(usize, &mut [u8]) -> bool,
    ) -> Option<bool> {
        if out.is_empty() {
            return Some(true);
        }
        let end = addr.checked_add(out.len())?;
        let mut inline = [(0usize, 0usize); 8];
        let mut inline_len = 0usize;
        let mut extra: Vec<(usize, usize)> = Vec::new();
        {
            let g = self.lock();
            let before = g
                .maps
                .range(..addr)
                .next_back()
                .filter(|&(&a, m)| a.saturating_add(m.len) > addr);
            for (&a, m) in before.into_iter().chain(g.maps.range(addr..end)) {
                if m.len == 0 || !g.pools.contains_key(&m.pool) {
                    continue;
                }
                let start = addr.max(a);
                let stop = end.min(a.saturating_add(m.len));
                if start >= stop {
                    continue;
                }
                let merged = if !extra.is_empty() {
                    extra.last_mut()
                } else if inline_len > 0 {
                    Some(&mut inline[inline_len - 1])
                } else {
                    None
                };
                if let Some(last) = merged.filter(|last| last.1 == start) {
                    last.1 = stop;
                } else if inline_len < inline.len() {
                    inline[inline_len] = (start, stop);
                    inline_len += 1;
                } else {
                    extra.push((start, stop));
                }
            }
            if inline_len == 0 {
                return None;
            }
        }
        let mut cur = addr;
        let mut whole = true;
        for &(start, stop) in inline[..inline_len].iter().chain(extra.iter()) {
            if cur < start {
                out[cur - addr..start - addr].fill(0);
                whole = false;
            }
            if !read_span(start, &mut out[start - addr..stop - addr]) {
                whole = false;
            }
            cur = stop;
        }
        if cur < end {
            out[cur - addr..].fill(0);
            whole = false;
        }
        Some(whole)
    }
}

impl PoolObserver for PoolTable {
    fn pool_created(&self, pool: &PoolInfo) {
        let mut g = self.lock();
        g.generation += 1;
        g.pools.insert(pool.id, pool.clone());
    }

    fn pool_released(&self, pool: &PoolInfo) {
        let mut g = self.lock();
        g.generation += 1;
        g.pools.remove(&pool.id);
        g.maps.retain(|_, m| m.pool != pool.id);
        g.released.push(pool.id);
    }

    fn mapped(&self, at: usize, len: usize, pool: &PoolInfo, offset: usize) {
        let mut g = self.lock();
        g.pools.entry(pool.id).or_insert_with(|| pool.clone());
        g.cut(at, len);
        g.maps.insert(
            at,
            Map {
                len,
                pool: pool.id,
                offset,
            },
        );
    }

    fn unmapped(&self, at: usize, len: usize) {
        self.lock().cut(at, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(id: usize) -> PoolInfo {
        PoolInfo {
            id,
            base: 0x1000_0000 * id,
            len: 0x100_0000,
            memory_entry: None,
        }
    }

    #[test]
    fn maps_cut_and_resolve() {
        let t = PoolTable::default();
        let p = pool(1);
        t.pool_created(&p);
        t.mapped(0x7000_0000, 0x40_0000, &p, 0x1_0000);
        let (info, ptr) = t.resolve(0x7000_0100, 16).unwrap();
        assert_eq!((info.id, ptr as usize), (1, p.base + 0x1_0100));
        // Unmapping the middle leaves both ends, still showing the right offsets.
        t.unmapped(0x7010_0000, 0x10_0000);
        assert!(t.resolve(0x7010_0000, 1).is_none());
        let (_, ptr) = t.resolve(0x7020_0000, 4).unwrap();
        assert_eq!(ptr as usize, p.base + 0x1_0000 + 0x20_0000);
        // A range across the hole is refused; a remap over it replaces what it showed.
        assert!(t.resolve(0x700F_FFF0, 0x20).is_none());
        let q = pool(2);
        t.mapped(0x7000_0000, 0x1000, &q, 0);
        assert_eq!(t.resolve(0x7000_0010, 1).unwrap().0.id, 2);
        assert_eq!(t.resolve(0x7000_1000, 1).unwrap().0.id, 1);
        t.pool_released(&p);
        assert!(t.resolve(0x7000_1000, 1).is_none());
        assert_eq!(t.take_released(), vec![1]);
    }

    #[test]
    fn a_range_over_adjacent_mappings_resolves_when_the_pool_continues() {
        let t = PoolTable::default();
        let (p, q) = (pool(1), pool(2));
        t.pool_created(&p);
        t.pool_created(&q);
        // Three 64 KiB pages of one allocation, reported one at a time, in pool order.
        for i in 0..3 {
            t.mapped(0x7000_0000 + i * 0x1_0000, 0x1_0000, &p, 0x4_0000 + i * 0x1_0000);
        }
        // Over one boundary, and over all three.
        let (info, ptr) = t.resolve(0x7000_fff0, 0x20).unwrap();
        assert_eq!((info.id, ptr as usize), (1, p.base + 0x4_fff0));
        let (_, ptr) = t.resolve(0x7000_8000, 0x2_0000).unwrap();
        assert_eq!(ptr as usize, p.base + 0x4_8000);
        // Up to the last byte, not past it.
        assert!(t.resolve(0x7000_0000, 0x3_0000).is_some());
        assert!(t.resolve(0x7000_0000, 0x3_0001).is_none());
        // The next title page shows the same pool, but not the bytes that follow in it.
        t.mapped(0x7003_0000, 0x1_0000, &p, 0x9_0000);
        assert!(t.resolve(0x7002_fff0, 0x20).is_none());
        assert!(t.resolve(0x7003_0000, 0x10).is_some());
        // The next title page shows another pool at the offset that would follow.
        t.mapped(0x7003_0000, 0x1_0000, &q, 0x7_0000);
        assert!(t.resolve(0x7002_fff0, 0x20).is_none());
        // A length that wraps is refused, not followed.
        assert!(t.resolve(0x7000_0000, u64::MAX).is_none());
    }

    #[test]
    fn an_empty_mapping_ends_the_walk() {
        let t = PoolTable::default();
        let p = pool(1);
        t.pool_created(&p);
        t.mapped(0x7000_0000, 0x1_0000, &p, 0x4_0000);
        // Empty, at the next address, with the pool and offset that would continue the first.
        t.mapped(0x7001_0000, 0, &p, 0x5_0000);
        assert!(t.resolve(0x7000_fff0, 0x20).is_none());
        // What lies wholly in the first mapping still resolves.
        assert_eq!(t.resolve(0x7000_fff0, 0x10).unwrap().1 as usize, p.base + 0x4_fff0);
        // An address in the empty mapping is in no mapping.
        t.unmapped(0x7000_0000, 0x1_0000);
        assert!(t.resolve(0x7001_0000, 1).is_none());
    }

    #[test]
    fn cut_spans_multiple_mappings_and_keeps_their_edges() {
        let t = PoolTable::default();
        let p = pool(1);
        for (base, offset) in [(0x1000, 0), (0x3000, 0x4000), (0x6000, 0x8000)] {
            t.mapped(base, 0x2000, &p, offset);
        }
        let before = t.generation();
        t.unmapped(0x2000, 0x5000);
        assert_eq!(t.generation(), before + 1);
        assert_eq!(t.resolve(0x1fff, 1).unwrap().1 as usize, p.base + 0xfff);
        for addr in [0x2000, 0x3000, 0x6000, 0x6fff] {
            assert!(t.resolve(addr, 1).is_none());
        }
        assert_eq!(t.resolve(0x7000, 1).unwrap().1 as usize, p.base + 0x9000);
    }

    #[test]
    fn mapping_at_finds_single_address_without_snapshot() {
        let t = PoolTable::default();
        let (p, q) = (pool(1), pool(2));
        t.mapped(0x2000, 0x1000, &p, 0x400);
        t.mapped(0x5000, 0x2000, &q, 0x800);
        assert!(t.mapping_at(0x1fff).is_none());
        let first = t.mapping_at(0x2000).unwrap();
        assert_eq!((first.at, first.len, first.pool, first.offset), (0x2000, 0x1000, 1, 0x400));
        assert!(t.mapping_at(0x2fff).is_some());
        assert!(t.mapping_at(0x3000).is_none());
        let second = t.mapping_at(0x6fff).unwrap();
        assert_eq!((second.at, second.len, second.pool, second.offset), (0x5000, 0x2000, 2, 0x800));
        assert!(t.mapping_at(0x7000).is_none());
    }

    #[test]
    fn read_mapped_with_coalesces_adjacent_pages_across_pools_and_zeroes_gaps() {
        let t = PoolTable::default();
        let (pa, pb) = (pool(1), pool(2));
        t.pool_created(&pa);
        t.pool_created(&pb);

        // No mapping yet: returns None so callers can fall back to non-pool memory reads.
        let mut out = vec![0xffu8; 6 * 0x1_0000];
        assert_eq!(
            t.read_mapped_with(0x9000_0000, &mut out, |_, _| panic!("should not be called")),
            None
        );
        assert!(out.iter().all(|&b| b == 0xff));

        // Map scattered pages in descending pool order across two pools, leaving page 1 and pages
        // 4..6 unmapped (mip-streamed texture with an unmapped hole and unmapped fine mips).
        t.mapped(0x9000_0000, 0x1_0000, &pa, 5 * 0x1_0000);
        t.mapped(0x9002_0000, 0x1_0000, &pb, 2 * 0x1_0000);
        t.mapped(0x9003_0000, 0x1_0000, &pa, 0x1_0000);

        let mut spans_seen = Vec::new();
        assert_eq!(
            t.read_mapped_with(0x9000_0000, &mut out, |at, dst| {
                spans_seen.push((at, dst.len()));
                dst.fill(((at - 0x9000_0000) >> 16) as u8 + 0x21);
                true
            }),
            Some(false)
        );
        // Pages 2 and 3 come from different pools (`pb` then `pa`) at out-of-order pool offsets,
        // yet coalesce into one contiguous title-address span `(0x9002_0000, 0x2_0000)`.
        assert_eq!(
            spans_seen,
            vec![(0x9000_0000, 0x1_0000), (0x9002_0000, 0x2_0000)]
        );
        assert!(out[..0x1_0000].iter().all(|&b| b == 0x21));
        assert!(out[0x1_0000..0x2_0000].iter().all(|&b| b == 0));
        assert!(out[0x2_0000..0x4_0000].iter().all(|&b| b == 0x23));
        assert!(out[0x4_0000..].iter().all(|&b| b == 0));

        // Mapping the remaining pages coalesces all 6 pages into a single whole read (`Some(true)`).
        t.mapped(0x9001_0000, 0x1_0000, &pa, 7 * 0x1_0000);
        t.mapped(0x9004_0000, 0x2_0000, &pb, 0);
        out.fill(0xff);
        spans_seen.clear();
        assert_eq!(
            t.read_mapped_with(0x9000_0000, &mut out, |at, dst| {
                spans_seen.push((at, dst.len()));
                dst.fill(0x5a);
                true
            }),
            Some(true)
        );
        assert_eq!(spans_seen, vec![(0x9000_0000, 6 * 0x1_0000)]);
        assert!(out.iter().all(|&b| b == 0x5a));
    }
}

/// A regression measure, run by hand (`cargo test --release -p victory-gpu --lib
/// signature_walks_of_a_follow_pass -- --ignored --nocapture`): `PoolTable::signature` over the
/// ranges `NativeHost::follow_title_pages` checks at each look, in a pool table laid out like the
/// world's (synthetic addresses and sizes, no title data). M5 Pro, 27 November: 36-78 us for the
/// 174 MiB view; a look over the entries a world run had (one range per entry) 2.7-3.1 ms; the
/// same look walking each distinct range once, as `follow_title_pages` does, 60-90 us.
#[cfg(test)]
mod signature_timing {
    use super::*;

    const PAGE: usize = 1 << 16;

    /// A table of `others` scattered one-page mappings, plus the world's three large views as
    /// one-page mappings from pools in descending page order (the 174 MiB one over three runs
    /// of pool pages apart).
    fn world(others: usize) -> (PoolTable, [(u64, u64); 3]) {
        let t = PoolTable::default();
        let pool = |id: usize, len: usize| PoolInfo { id, base: 0x1000_0000_0000 + id * (1 << 34), len, memory_entry: None };
        let views = [(0x2000_0000u64, 182_452_224u64), (0xd_0000_0000, 52_428_800), (0xd_4000_0000, 25_165_824)];
        for (k, &(at, len)) in views.iter().enumerate() {
            let pages = (len as usize).div_ceil(PAGE);
            let p = pool(1 + k, 4 * pages * PAGE);
            t.pool_created(&p);
            for n in 0..pages {
                // Three runs of descending pool pages, apart from one another.
                let run = n * 3 / pages;
                let offset = (run * 2 * pages + (pages - n)) * PAGE;
                t.mapped(at as usize + n * PAGE, PAGE, &p, offset);
            }
        }
        let p = pool(9, others * PAGE);
        t.pool_created(&p);
        for n in 0..others {
            t.mapped(0x8_0000_0000 + 2 * n * PAGE, PAGE, &p, n * PAGE);
        }
        (t, views)
    }

    fn time(what: &str, n: u32, mut f: impl FnMut()) -> f64 {
        f();
        let start = std::time::Instant::now();
        for _ in 0..n {
            f();
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(n);
        println!("{what}: {us:.1} us");
        us
    }

    #[test]
    #[ignore]
    fn signature_walks_of_a_follow_pass() {
        for others in [1_000, 10_000, 100_000] {
            let (t, [big, mid, small]) = world(others);
            println!("-- {others} other mappings");
            let sig = |(a, l): (u64, u64)| std::hint::black_box(t.signature(a, l));
            time("174 MiB view", 200, || { sig(big); });
            time("50 MiB view", 200, || { sig(mid); });
            time("24 MiB view", 200, || { sig(small); });
            time("a range in no pool (host memory)", 2_000, || { sig((0x7f00_0000_0000, 21_045_760)); });
            // The entries a look at 1,000 checked in a world run: 54 of the 174 MiB range, 12 of
            // the 50 MiB one, 59 of the 24 MiB one, 571 outside every pool.
            time("a look, one walk per entry (54 + 12 + 59 + 571 entries)", 20, || {
                for _ in 0..54 { sig(big); }
                for _ in 0..12 { sig(mid); }
                for _ in 0..59 { sig(small); }
                for _ in 0..571 { sig((0x7f00_0000_0000, 21_045_760)); }
            });
            time("the same look, each distinct range walked once", 20, || {
                let mut walked = std::collections::HashMap::new();
                let entries = [(big, 54), (mid, 12), (small, 59), ((0x7f00_0000_0000, 21_045_760), 571)];
                for (range, copies) in entries {
                    for _ in 0..copies {
                        std::hint::black_box(*walked.entry(range).or_insert_with(|| sig(range)));
                    }
                }
            });
        }
    }
}
