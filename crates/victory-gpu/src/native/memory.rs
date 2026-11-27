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
}
