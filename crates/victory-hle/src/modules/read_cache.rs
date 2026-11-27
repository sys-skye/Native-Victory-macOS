//! `VICTORY_READ_CACHE=<dir>:<prefix>`: keeps the blocks DirectStorage reads
//! from files under `<prefix>` (after resolving links) in `<dir>`, across
//! runs. For game data on a slow network mount: the second boot reads what
//! the first one fetched from local disk.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

const BLOCK: u64 = 64 << 10;
/// Most blocks fetched in one read (a run of missing ones).
const RUN: u64 = 64;
// The budget counts this run's additions only; prune <dir> by hand if the disk fills.
const LIMIT: u64 = 12 << 30;
static ADDED: AtomicU64 = AtomicU64::new(0);

pub struct BlockCache {
    data: File,
    map: File,
    have: Vec<AtomicU8>,
    len: u64,
}

/// A cache for `host` if the variable is set and `host` is under its prefix.
pub fn open(host: &Path) -> Option<BlockCache> {
    let var = std::env::var("VICTORY_READ_CACHE").ok()?;
    let (dir, prefix) = var.split_once(':')?;
    let real = std::fs::canonicalize(host).ok()?;
    if !real.starts_with(prefix) {
        return None;
    }
    let meta = real.metadata().ok()?;
    let len = meta.len();
    // The mtime is in the key so a patched file of the same size is not served stale.
    let mtime = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    let name = format!("{}-{len}-{mtime}", real.file_name()?.to_string_lossy());
    std::fs::create_dir_all(dir).ok()?;
    let open = |ext: &str| {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(Path::new(dir).join(format!("{name}.{ext}")))
    };
    let (data, map) = (open("data").ok()?, open("map").ok()?);
    let blocks = len.div_ceil(BLOCK);
    data.set_len(len).ok()?; // sparse until blocks land
    map.set_len(blocks).ok()?;
    let mut bytes = vec![0; blocks as usize];
    map.read_exact_at(&mut bytes, 0).ok()?;
    let cached = bytes.iter().filter(|&&b| b != 0).count();
    tracing::info!(file = %real.display(), cached_blocks = cached, blocks, "read cache");
    Some(BlockCache {
        data,
        map,
        have: bytes.into_iter().map(AtomicU8::new).collect(),
        len,
    })
}

impl BlockCache {
    fn has(&self, block: u64) -> bool {
        self.have[block as usize].load(Ordering::Acquire) != 0
    }

    /// `pread` from the cache; missing blocks (with the missing ones after
    /// them that `buf` also covers) come from `source` and are kept.
    pub fn read_at(&self, source: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
        if offset >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let first = offset / BLOCK;
        if self.has(first) {
            let end = ((first + 1) * BLOCK).min(self.len);
            let n = buf.len().min((end - offset) as usize);
            return self.data.read_at(&mut buf[..n], offset);
        }
        if ADDED.load(Ordering::Relaxed) >= LIMIT {
            return source.read_at(buf, offset);
        }
        let want_end = (offset + buf.len() as u64).min(self.len);
        let mut last = first;
        while last + 1 < first + RUN && (last + 1) * BLOCK < want_end && !self.has(last + 1) {
            last += 1;
        }
        let start = first * BLOCK;
        let end = ((last + 1) * BLOCK).min(self.len);
        let mut blocks = vec![0; (end - start) as usize];
        source.read_exact_at(&mut blocks, start)?;
        self.data.write_all_at(&blocks, start)?;
        // Data reaches the disk before the map marks it present, so a crash
        // can lose blocks but never mark one present with nothing behind it.
        self.data.sync_data()?;
        self.map.write_all_at(&vec![1; (last - first + 1) as usize], first)?;
        for b in first..=last {
            self.have[b as usize].store(1, Ordering::Release);
        }
        ADDED.fetch_add(end - start, Ordering::Relaxed);
        let n = buf.len().min((end - offset) as usize);
        buf[..n].copy_from_slice(&blocks[(offset - start) as usize..][..n]);
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn reads_match_the_source_cold_and_warm() {
        let dir = std::env::temp_dir().join(format!("rc-test-{}", std::process::id()));
        let src_dir = dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let path = src_dir.join("f.bin");
        let bytes: Vec<u8> = (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let real = std::fs::canonicalize(&src_dir).unwrap();
        // SAFETY: the test sets it before any reader.
        unsafe {
            std::env::set_var("VICTORY_READ_CACHE", format!("{}:{}", dir.join("c").display(), real.display()))
        };
        let source = std::fs::File::open(&path).unwrap();
        for _pass in 0..2 {
            let cache = super::open(&path).unwrap();
            for (off, len) in [(0u64, 10usize), (70_000, 200_000), (299_990, 50), (65_536, 65_536)] {
                let mut buf = vec![0; len];
                let mut done = 0;
                while done < len {
                    let n = cache.read_at(&source, &mut buf[done..], off + done as u64).unwrap();
                    if n == 0 {
                        break;
                    }
                    done += n;
                }
                let end = (off as usize + len).min(bytes.len());
                assert_eq!(&buf[..done], &bytes[off as usize..end]);
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
