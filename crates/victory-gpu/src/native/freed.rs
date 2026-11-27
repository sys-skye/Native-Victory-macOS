//! Title memory the title's allocator returned, watched while
//! `VICTORY_TRACE_FREED` is on: `XMemAllocDefault` notes every range it
//! hands out, `XMemFreeDefault` the ones it takes back (the header holds
//! the size), and `Host::resolve` — the way every host and GPU use of
//! title memory begins — checks the range it is about to map and names
//! any overlap with freed memory (#339: a host copy or a GPU write into
//! a range the title has freed is one candidate for the corrupted pool
//! links).
//!
//! What the table cannot see: the title can hook the allocator
//! (`XMemAlloc`'s hooks), and pools it sub-allocates inside a live
//! `XMemAlloc` block free and reuse memory without any call here; this
//! names the `XMem`-granularity class only.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// The freed spans `(start, length, serial)`, kept by start address.
static FREED: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());
/// The serial of the next freed span, so one log line names its span.
static SERIAL: AtomicU64 = AtomicU64::new(0);

/// Whether `VICTORY_TRACE_FREED` names uses of freed title memory.
fn tracing_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_TRACE_FREED"))
}

/// The allocator took `[addr, addr + len)` back.
pub fn record_free(addr: u64, len: u64) {
    if len == 0 || !tracing_on() {
        return;
    }
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed) + 1;
    freed(&mut FREED.lock().unwrap_or_else(|e| e.into_inner()), addr, len, serial);
}

/// Adds the freed span `[addr, addr + len)` to `spans`, kept by start address.
fn freed(spans: &mut Vec<(u64, u64, u64)>, addr: u64, len: u64, serial: u64) {
    let at = spans.partition_point(|&(start, _, _)| start < addr);
    spans.insert(at, (addr, len, serial));
}

/// The allocator handed out `[addr, addr + len)`: none of it is freed.
/// A part-reuse leaves the freed ends beside it.
pub fn record_alloc(addr: u64, len: u64) {
    if len == 0 || !tracing_on() {
        return;
    }
    alloc(&mut FREED.lock().unwrap_or_else(|e| e.into_inner()), addr, len);
}

/// Takes `[addr, addr + len)` out of the freed `spans`, leaving the freed ends beside it.
fn alloc(spans: &mut Vec<(u64, u64, u64)>, addr: u64, len: u64) {
    let end = addr + len;
    let mut at = 0;
    while at < spans.len() {
        let (start, length, serial) = spans[at];
        if start >= end || start + length <= addr {
            at += 1;
            continue;
        }
        spans.remove(at);
        if start < addr {
            spans.insert(at, (start, addr - start, serial));
            at += 1;
        }
        if start + length > end {
            let left = start + length - end;
            spans.insert(at, (end, left, serial));
            at += 1;
        }
    }
}

/// Names every freed span that `[addr, addr + len)` overlaps, with `what`
/// (the host use that reached for it).
pub fn check(addr: u64, len: u64, what: &str) {
    if len == 0 || !tracing_on() {
        return;
    }
    let spans = FREED.lock().unwrap_or_else(|e| e.into_inner());
    for (start, length, serial) in overlapping(&spans, addr, len) {
        tracing::warn!(
            serial,
            freed = format_args!("{start:#x}+{length:#x}"),
            asked = format_args!("{addr:#x}+{len:#x}"),
            "{what}: title memory the allocator has freed"
        );
    }
}

/// The spans of `spans` that overlap `[addr, addr + len)`.
fn overlapping(spans: &[(u64, u64, u64)], addr: u64, len: u64) -> Vec<(u64, u64, u64)> {
    spans
        .iter()
        .filter(|&&(start, length, _)| start < addr + len && start + length > addr)
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::overlapping;

    /// The real table only notes while `VICTORY_TRACE_FREED` is on; the tests drive its
    /// functions on a list of their own.
    fn freed(list: &[(u64, u64, u64)], addr: u64, len: u64, serial: u64) -> Vec<(u64, u64, u64)> {
        let mut spans = list.to_vec();
        super::freed(&mut spans, addr, len, serial);
        spans
    }

    fn alloc(list: &[(u64, u64, u64)], addr: u64, len: u64) -> Vec<(u64, u64, u64)> {
        let mut spans = list.to_vec();
        super::alloc(&mut spans, addr, len);
        spans
    }

    #[test]
    fn a_freed_span_overlaps_until_it_is_handed_out_again() {
        let spans = freed(&[], 0x1000, 0x100, 1);
        assert_eq!(overlapping(&spans, 0x1080, 0x10), [(0x1000, 0x100, 1)]);
        assert_eq!(overlapping(&spans, 0x10ff, 0x10), [(0x1000, 0x100, 1)]);
        assert_eq!(overlapping(&spans, 0x0ff0, 0x10), []);
        // The allocator hands the range out again: nothing overlaps it.
        assert_eq!(alloc(&spans, 0x1000, 0x100), []);
    }

    #[test]
    fn a_partial_reuse_leaves_the_freed_ends() {
        let spans = freed(&[], 0x1000, 0x100, 1);
        // The first half is reused: the freed tail keeps its serial.
        let reused = alloc(&spans, 0x1000, 0x80);
        assert_eq!(overlapping(&reused, 0x1080, 0x10), [(0x1080, 0x80, 1)]);
        assert_eq!(overlapping(&reused, 0x1000, 0x10), []);
        // A reuse in the middle splits the span in two.
        let reused = alloc(&spans, 0x1040, 0x20);
        assert_eq!(overlapping(&reused, 0x1030, 0x10), [(0x1000, 0x40, 1)]);
        assert_eq!(overlapping(&reused, 0x1060, 0x10), [(0x1060, 0xa0, 1)]);
        assert_eq!(overlapping(&reused, 0x1040, 0x20), []);
    }

    #[test]
    fn neighbouring_spans_do_not_overlap_a_gap() {
        let spans = [((0x1000u64), 0x10u64, 1u64), (0x1020u64, 0x10u64, 2u64)];
        assert_eq!(overlapping(&spans, 0x1010, 0x10), []);
        assert_eq!(overlapping(&spans, 0x1028, 0x8), [(0x1020, 0x10, 2)]);
        assert_eq!(overlapping(&spans, 0x0ff0, 0x20), [(0x1000, 0x10, 1)]);
    }
}
