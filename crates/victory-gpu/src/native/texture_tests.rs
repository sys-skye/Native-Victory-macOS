//! Title textures on the native host: textures laid out in the console's tiled layouts in title
//! memory, detiled into host textures; textures the title places in memory itself; render targets
//! cleared through the title's views. Our own synthetic texels. Skipped without Metal.

use super::Pool;
use crate::native::glue::{self, with_list, with_title};
use crate::native::list::NativeList;
use crate::native::title::Title;
use crate::native::memory::{PoolInfo, PoolObserver};
use crate::object::{This, slot};
use victory_backend::resources::*;
use victory_formats::texture::Swizzle;
use victory_metal::mem::{Alias, page_size};

fn pool(id: usize, pages: usize) -> Option<Pool> {
    // Needs a device; the title is made on first use.
    with_title(|_| ())?;
    let mem = Alias::new(pages * page_size().max(16384)).unwrap();
    // Textures earlier tests left over memory now freed and mapped here again: a new texture laid
    // out as one of them would share its copy (`Title::realize`). The title releases its own.
    let span = mem.alias as usize..mem.alias as usize + mem.len;
    with_title(|t| loop {
        let stale: Vec<usize> =
            t.at_memory.iter().filter(|(a, _)| span.contains(a)).flat_map(|(_, k)| k.clone()).collect();
        if stale.is_empty() {
            break;
        }
        for k in stale {
            t.host.release_resource(k);
            t.texture_released(k);
        }
    });
    let info = PoolInfo {
        id,
        base: mem.primary as usize,
        len: mem.len,
        memory_entry: None,
    };
    let observer = glue::pool_observer();
    observer.pool_created(&info);
    observer.mapped(mem.alias as usize, mem.len, &info, 0);
    Some(Pool { mem })
}

/// What `x`, `y` hold in mip `mip` of the test texture.
fn texel(mip: u32, x: u32, y: u32) -> [u8; 4] {
    [x as u8, y as u8, mip as u8 * 40, 255]
}

/// Tiles a texture's mips into `pool` memory at `at`, texel by `texel`.
fn tile(pool: &Pool, at: usize, layout: &victory_formats::texture::Layout) {
    let mut bytes = vec![0u8; layout.size as usize];
    for mip in 0..layout.mips.len() as u32 {
        let m = &layout.mips[mip as usize];
        let linear: Vec<u8> = (0..m.height)
            .flat_map(|y| (0..m.width).flat_map(move |x| texel(mip, x, y)))
            .collect();
        layout
            .tile(&mut bytes, mip, 0, &linear, m.width as usize * 4, linear.len())
            .unwrap();
    }
    pool.write(at, &bytes);
}

fn read_mip(key: usize, mip: usize, w: usize, h: usize) -> Vec<u8> {
    with_title(|t| {
        let tex = t.texture_of(key).expect("a host texture");
        let mut out = vec![0u8; w * h * 4];
        t.host.backend.read_texture(&tex, mip, 0, &mut out, w * 4, 0);
        out
    })
    .unwrap()
}

/// A tiled texture with a mip chain is detiled mip by mip into its host texture when work is
/// submitted, again only when its memory changed.
#[test]
fn a_tiled_title_texture_is_detiled_into_the_host_texture() {
    let Some(pool) = pool(9101, 8) else { return };
    // The XOR modes (`_X`, `_T`) too: their equations come from the pipe configuration, not a
    // fixed pattern, and a texture of theirs is no longer left empty.
    for swizzle in [
        Swizzle::LINEAR,
        Swizzle::SW_256B_S,
        Swizzle::SW_4KB_S,
        Swizzle::SW_64KB_D,
        Swizzle::SW_4KB_S_X,
        Swizzle::SW_64KB_D_X,
        Swizzle::SW_64KB_S_T,
    ] {
        let at = 0;
        let res = crate::resource::test_texture_at(pool.va(at), (64, 32), 4, 28, swizzle.0, 0);
        let info = crate::resource::texture_info(res).unwrap();
        tile(&pool, at, &info.layout);
        let key = res as usize;
        let uploads = |key| with_title(|t| t.uploads_of(key)).unwrap();
        with_title(|t| t.sync_textures());
        // Nothing named the texture yet: no host texture, nothing detiled.
        assert_eq!(uploads(key), 0);
        let mips: Vec<(usize, usize)> = (0..4).map(|m| (64 >> m, 32 >> m)).collect();
        // Naming it makes it; the next sync fills it.
        with_title(|t| {
            assert!(t.realize(key));
            t.sync_textures();
        });
        for (mip, &(w, h)) in mips.iter().enumerate() {
            let out = read_mip(key, mip, w, h);
            for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w / 2, h / 2)] {
                let at = (y * w + x) * 4;
                assert_eq!(
                    out[at..at + 4],
                    texel(mip as u32, x as u32, y as u32),
                    "{swizzle:?} mip {mip} at {x},{y}"
                );
            }
        }
        assert_eq!(uploads(key), 1, "uploaded once");
        // Unchanged memory is not detiled again; changed memory is.
        with_title(|t| t.sync_textures());
        assert_eq!(uploads(key), 1);
        pool.write(at, &[0xFFu8; 16]);
        with_title(|t| t.sync_textures());
        assert_eq!(uploads(key), 2);
    }
}

/// `VICTORY_TEXTURE_DROP_TOP_MIP`: a host texture made without the top level receives the title's
/// level 1 as its level 0, and so on down the chain.
#[test]
fn a_texture_made_without_its_top_level_gets_the_lower_levels() {
    let Some(pool) = pool(9301, 8) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (64, 32), 4, 28, Swizzle::SW_64KB_D.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    tile(&pool, 0, &info.layout);
    let tiled = super::super::textures::read_memory(info.memory, info.size);
    let mut desc = info.desc().unwrap();
    (desc.width, desc.height, desc.mips) = (32, 16, 3);
    with_title(|t| {
        let tex = t.host.backend.create_texture(&desc).unwrap();
        super::super::textures::upload_from(&t.host.backend, &tex, &info, &tiled, 1).unwrap();
        for host_mip in 0..3usize {
            let (w, h) = (32 >> host_mip, 16 >> host_mip);
            let mut out = vec![0u8; w * h * 4];
            t.host.backend.read_texture(&tex, host_mip, 0, &mut out, w * 4, 0);
            for (x, y) in [(0, 0), (w - 1, h - 1), (w / 2, h / 2)] {
                let at = (y * w + x) * 4;
                assert_eq!(out[at..at + 4], texel(host_mip as u32 + 1, x as u32, y as u32), "host mip {host_mip} at {x},{y}");
            }
        }
    })
    .unwrap();
}

/// Held by the tests that count on a texture staying hot: the many submissions of
/// `a_texture_blank_for_thousands_of_submissions_is_still_probed` and
/// `hot_textures_cool_by_submissions_even_when_probe_round_is_capped` cool every other test's
/// textures.
static AGING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A texture uploaded blank and written whole later (a movie's planes) is uploaded again at the
/// next submission, not a scanner round later (a round is seconds long once the title streams).
#[test]
fn a_texture_uploaded_blank_is_probed_at_each_submission() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    // 256 KiB: 64 pages, more than a sampling probe reads.
    let Some(pool) = pool(9105, (256 << 10) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (256, 256), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    // One title lock for the whole test, and no probe deadline: see
    // `a_texture_rewritten_whole_is_read_whole_only_now_and_then`.
    with_title(|t| {
        assert!(t.realize(key));
        t.sync_dirty_without_probe_deadline();
        assert_eq!(t.uploads_of(key), 1, "the blank first upload");
        pool.write(0, &vec![0xABu8; info.size]);
        t.sync_dirty_without_probe_deadline();
        assert_eq!(t.uploads_of(key), 2, "uploaded again at the next submission");
        // Its samples showed that write, as they show a streamed texture's: it is not read whole
        // at every submission after it (thousands of those were most of what a world submission
        // read).
        let reads = (0..crate::native::textures::WHOLE_EVERY)
            .filter(|_| {
                let jobs = t.sync_jobs();
                let read = jobs.iter().any(|j| j.key == key);
                t.sync(jobs);
                read
            })
            .count();
        assert!(reads <= 1, "read whole {reads} times in a whole-read period");
        // A glyph's worth of bytes in page 3 (a page a sampling probe would skip).
        pool.write(3 * 4096 + 100, &[1u8; 16]);
        // (Twice the period: other tests' textures share what a submission spares for these reads.)
        let n = (1..=2 * crate::native::textures::WHOLE_EVERY).find(|_| {
            t.sync_dirty_without_probe_deadline();
            t.uploads_of(key) == 3
        });
        assert!(n.is_some(), "a small write is seen within a whole-read period or two");
        pool.write(5 * 4096 + 100, &[1u8; 16]);
        t.sync_dirty_without_probe_deadline();
        assert_eq!(t.uploads_of(key), 4, "and, its samples having missed one, the next at the next submission");
    });
}

/// A texture uploaded blank stays hot however long it stays blank: the boot movie's planes are
/// made about 40 s (thousands of submissions) before the movie decodes into them, and once cold
/// their first frames waited seconds for the scanner (green).
#[test]
fn a_texture_blank_for_thousands_of_submissions_is_still_probed() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pool) = pool(9123, (256 << 10) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (256, 256), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    // One title lock for the whole test, and no probe deadline for the submission after the write:
    // other tests leave hundreds of blank textures, and a round the deadline cuts short under load
    // probes only `MIN_SYNC_HOT` of them, so a few submissions may not reach this one.
    with_title(|t| {
        assert!(t.realize(key));
        for _ in 0..2000 {
            t.sync_dirty();
        }
        assert_eq!(t.uploads_of(key), 1, "the blank first upload");
        pool.write(0, &vec![0xABu8; info.size]);
        t.sync_dirty_without_probe_deadline();
        assert_eq!(t.uploads_of(key), 2, "still probed: the first frame is uploaded at the next submission");
    });
}

/// A texture rewritten whole (a movie's chroma plane) is not read whole at every submission, yet
/// a small write its samples miss is still seen within `WHOLE_EVERY` submissions, and from then
/// on at the next one (a glyph atlas).
#[test]
fn a_texture_rewritten_whole_is_read_whole_only_now_and_then() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pool) = pool(9124, (256 << 10) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (256, 256), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    // The title is shared, so the test holds its lock throughout: another test's submission in
    // between probes and uploads this texture too, and the follow-up whole read after a missed
    // write may then already be spent when the next write comes ("at the next" fails). And no
    // probe deadline: other tests' textures stay hot after them (the 200 of
    // `a_hot_texture_is_probed_at_each_submission_among_hundreds`), and a round the deadline cuts
    // short under load may not reach this one ("at once" fails).
    with_title(|t| {
        assert!(t.realize(key));
        let mut sync = || {
            t.sync_dirty_without_probe_deadline();
            t.uploads_of(key)
        };
        assert_eq!(sync(), 1, "the blank first upload");
        for (i, n) in [(1u8, 2), (2, 3), (3, 4)] {
            pool.write(0, &vec![i; info.size]);
            assert_eq!(sync(), n, "a whole rewrite is seen by the samples at once");
            sync();
        }
        // A glyph's worth of bytes in page 3 (a page the samples skip).
        pool.write(3 * 4096 + 100, &[9u8; 16]);
        // (Twice the period: other tests' textures share what a submission spares for these reads.)
        let n = (1..=2 * crate::native::textures::WHOLE_EVERY).find(|_| sync() == 5);
        assert!(n.is_some(), "a write the samples miss is seen within a whole-read period or two");
        pool.write(5 * 4096 + 100, &[9u8; 16]);
        assert_eq!(sync(), 6, "and the next one at the next submission");
    });
}

/// A texture whose last write missed the sampled pages (a streamed mip tail, or a single glyph) is
/// checked whole at the next probe for a burst, then returns to periodic checks with backoff once
/// a whole read finds nothing changed instead of reading whole at every probe until it cools.
#[test]
fn a_missed_write_not_followed_by_another_backs_off_whole_reads() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pool) = pool(9131, (256 << 10) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (256, 256), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    with_title(|t| {
        assert!(t.realize(key));
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 1);
    });
    pool.write(0, &vec![0xABu8; info.size]);
    with_title(|t| {
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 2);
    });
    // Write in page 3, which the 8 sampled pages skip.
    pool.write(3 * 4096 + 100, &[7u8; 16]);
    with_title(|t| {
        let found = (1..=2 * crate::native::textures::WHOLE_EVERY).any(|_| {
            t.sync_dirty();
            t.uploads_of(key) == 3
        });
        assert!(found, "missed-sample write uploaded");
    });
    // Over 64 idle submissions: 1 immediate follow-up check, then +8, +16, +32 (at most 4 reads,
    // instead of all 64).
    with_title(|t| {
        let reads = (0..64)
            .filter(|_| {
                let jobs = t.sync_jobs();
                let read = jobs.iter().any(|j| j.key == key);
                t.sync(jobs);
                read
            })
            .count();
        assert!(reads <= 4, "unchanged texture read whole {reads} times across 64 idle submissions");
    });
}

/// A movie plane uploaded blank among hundreds of other hot textures (a title streaming) is still
/// probed at the next submission when the probe deadline allows it: a count limit on the probes
/// left its first frames green.
/// (Every texture is written, so which one a limited probe reaches first does not matter.)
#[test]
fn a_hot_texture_is_probed_at_each_submission_among_hundreds() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    const N: usize = 200;
    const STRIDE: usize = 64 << 10;
    let Some(pool) = pool(9121, N * STRIDE / page_size() + 1) else { return };
    let keys: Vec<usize> = (0..N)
        .map(|i| crate::resource::test_texture_at(pool.va(i * STRIDE), (64, 64), 1, 28, Swizzle::LINEAR.0, 0) as usize)
        .collect();
    let size = crate::resource::texture_info(keys[0] as _).unwrap().size;
    with_title(|t| {
        for &k in &keys {
            assert!(t.realize(k));
        }
        t.sync_dirty();
        assert!(keys.iter().all(|&k| t.uploads_of(k) == 1), "every blank upload");
        let bytes = vec![0xABu8; size];
        for i in 0..N {
            pool.write(i * STRIDE, &bytes);
        }
        t.sync_dirty_without_probe_deadline();
        // The scanner can trigger uploads too. The probe removes written textures from blank_keys.
        let probed = keys.iter().filter(|&k| !t.blank_keys.contains(k)).count();
        assert_eq!(probed, N, "all {N} blank textures probed in one submission");
        let seen = keys.iter().filter(|&&k| t.uploads_of(k) == 2).count();
        assert_eq!(seen, N, "all {N} hot textures written, seen at the next submission");
    });
}

/// Hundreds of hot textures cool after `HOT` submissions even when each submission's probe budget
/// can visit only `MIN_SYNC_HOT` of them, while a texture written again inside the window stays hot.
#[test]
fn hot_textures_cool_by_submissions_even_when_probe_round_is_capped() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    const N: usize = 128;
    const STRIDE: usize = 64 << 10;
    let Some(pool) = pool(9132, N * STRIDE / page_size() + 1) else { return };
    let keys: Vec<usize> = (0..N)
        .map(|i| crate::resource::test_texture_at(pool.va(i * STRIDE), (64, 64), 1, 28, Swizzle::LINEAR.0, 0) as usize)
        .collect();
    let size = crate::resource::texture_info(keys[0] as _).unwrap().size;
    let refreshed = keys[N - 1];
    with_title(|t| {
        for &k in &keys {
            assert!(t.realize(k));
        }
        t.sync_dirty_without_probe_deadline();
        assert!(keys.iter().all(|&k| t.uploads_of(k) == 1));
        let first = vec![0x33u8; size];
        for i in 0..N {
            pool.write(i * STRIDE, &first);
        }
        t.sync_dirty_without_probe_deadline();
        assert!(keys.iter().all(|&k| t.uploads_of(k) == 2 && t.hot_keys.contains(&k)));

        let hot = crate::native::textures::HOT as usize;
        for step in 0..hot {
            if step == hot / 2 {
                pool.write((N - 1) * STRIDE, &vec![0x77u8; size]);
                t.sync_dirty_without_probe_deadline();
                assert_eq!(t.uploads_of(refreshed), 3);
                continue;
            }
            let jobs = t.sync_jobs_within(Some(std::time::Duration::ZERO));
            t.sync(jobs);
        }
        // One more submission sweeps every expired texture. Without a deadline: the shared title
        // can hold other tests' hot textures that take a capped round's probes before ours.
        t.sync_dirty_without_probe_deadline();

        let still_hot = keys[..N - 1].iter().filter(|&&k| t.hot_keys.contains(&k)).count();
        assert_eq!(still_hot, 0, "all unchanged textures cooled after HOT submissions");
        assert!(t.hot_keys.contains(&refreshed), "texture written inside the window stays hot");

        pool.write((N - 1) * STRIDE, &vec![0xCCu8; size]);
        t.sync_dirty_without_probe_deadline();
        assert_eq!(t.uploads_of(refreshed), 4, "still-hot texture is probed on the next submission");
    });
}

/// A submission's textures are read and detiled on several threads: each host texture gets its
/// own memory's texels, once.
#[test]
fn textures_read_on_several_threads_get_their_own_texels() {
    const N: usize = 48;
    const STRIDE: usize = 64 << 10;
    let Some(pool) = pool(9125, N * STRIDE / page_size() + 1) else { return };
    let keys: Vec<usize> = (0..N)
        .map(|i| {
            let res = crate::resource::test_texture_at(pool.va(i * STRIDE), (64, 64), 1, 28, Swizzle::SW_4KB_S.0, 0);
            let layout = crate::resource::texture_info(res).unwrap().layout;
            let mut bytes = vec![0u8; layout.size as usize];
            let texels = vec![i as u8 + 1; 64 * 64 * 4];
            layout.tile(&mut bytes, 0, 0, &texels, 64 * 4, texels.len()).unwrap();
            pool.write(i * STRIDE, &bytes);
            res as usize
        })
        .collect();
    with_title(|t| {
        for &k in &keys {
            assert!(t.realize(k));
        }
        t.sync_dirty();
        assert!(keys.iter().all(|&k| t.uploads_of(k) == 1), "each uploaded once");
    });
    for (i, &k) in keys.iter().enumerate() {
        assert!(read_mip(k, 0, 64, 64).iter().all(|&b| b == i as u8 + 1), "texture {i} holds its own texels");
    }
}

/// A texture released while a submission read its memory (without the host's lock): what was read
/// is not uploaded into the texture that has its key by then.
#[test]
fn memory_read_for_a_texture_released_meanwhile_is_dropped() {
    let Some(pool) = pool(9126, 8) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (64, 64), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    pool.write(0, &vec![7u8; info.size]);
    with_title(|t| {
        assert!(t.realize(key));
        let jobs = t.sync_jobs();
        assert!(jobs.iter().any(|j| j.key == key));
        let prepared = crate::native::textures::prepare(&jobs);
        // Released, and another made at its key, while the memory was read.
        t.texture_released(key);
        t.texture_created(key, info.clone());
        assert!(t.realize(key));
        t.sync_apply(prepared);
        assert_eq!(t.uploads_of(key), 0, "the new texture gets nothing read for the old one");
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 1, "and is filled at the next submission");
    });
}

/// A GPU copy into a texture recorded while a submission read its memory (without the host's
/// lock): the texture is the GPU's by then, and what was read is not uploaded over it.
#[test]
fn memory_read_for_a_texture_a_copy_writes_meanwhile_is_dropped() {
    let Some(pool) = pool(9128, 8) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (64, 64), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    pool.write(0, &vec![7u8; info.size]);
    with_title(|t| {
        assert!(t.realize(key));
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 1);
        pool.write(0, &vec![8u8; info.size]);
        let jobs = t.jobs(vec![key]);
        assert_eq!(jobs.len(), 1);
        let prepared = crate::native::textures::prepare(&jobs);
        t.gpu_written(key);
        t.sync_apply(prepared);
        assert_eq!(t.uploads_of(key), 1, "the copy's destination is not filled from title memory");
        assert!(t.jobs(vec![key]).is_empty(), "nor read for it again");
    });
}

/// A texture just uploaded is not read whole again until its memory changes: neither its first
/// probe as a hot texture nor the scanner's next round reports what the submission read.
#[test]
fn a_texture_just_uploaded_is_not_read_again_until_it_changes() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    // 8 MiB: more than a probe reads whole.
    let Some(pool) = pool(9127, (8 << 20) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (2048, 1024), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    // A submission: whether it reads the texture, and the texture's uploads after it.
    let submit = || {
        with_title(|t| {
            let jobs = t.sync_jobs();
            let read = jobs.iter().any(|j| j.key == key);
            t.sync(jobs);
            (read, t.uploads_of(key))
        })
        .unwrap()
    };
    // Written under the lock, so that no other test's submission reads half of it.
    let write = |bytes: &[u8]| with_title(|_| pool.write(0, bytes)).unwrap();
    with_title(|t| assert!(t.realize(key)));
    assert_eq!(submit().1, 1, "the blank first upload");
    write(&vec![0xABu8; info.size]);
    // Within a few submissions (other tests' blank textures share the probe's time budget), this
    // test's or another's.
    assert!((1..=4).any(|_| submit().1 == 2), "written, so uploaded again");
    for _ in 0..20 {
        // Time for a round of the scanner.
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(submit(), (false, 2), "read again though nothing was written");
    }
    write(&[1u8; 16]);
    assert!((1..=4).any(|_| submit().1 == 3), "a write is seen");
}

/// A block-compressed texture: 16 bytes a 4x4 block, in the console's layout, detiled as blocks.
#[test]
fn a_block_compressed_title_texture_is_detiled_as_blocks() {
    let Some(pool) = pool(9102, 4) else { return };
    // BC7 (98): a 32x16 texture is 8x4 blocks.
    let res = crate::resource::test_texture_at(pool.va(0), (32, 16), 1, 98, Swizzle::SW_64KB_D.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    assert_eq!(info.layout.desc.element_bytes, 16);
    let m = &info.layout.mips[0];
    let blocks: Vec<u8> = (0..m.width * m.height * 16).map(|i| (i * 7) as u8).collect();
    let mut bytes = vec![0u8; info.layout.size as usize];
    info.layout
        .tile(&mut bytes, 0, 0, &blocks, m.width as usize * 16, blocks.len())
        .unwrap();
    pool.write(0, &bytes);
    let key = res as usize;
    with_title(|t| {
        assert!(t.realize(key));
        t.sync_textures();
        let tex = t.texture_of(key).unwrap();
        let mut out = vec![0u8; blocks.len()];
        t.host.backend.read_texture(&tex, 0, 0, &mut out, m.width as usize * 16, 0);
        assert_eq!(out, blocks);
    });
}

/// A texture a view names by memory address and description, without a resource object (the
/// console's `CreateShaderResourceViewPlaced`): the host makes it when the view is written.
#[test]
fn a_placed_texture_view_makes_and_fills_its_texture() {
    let Some(pool) = pool(9103, 4) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (32, 32), 1, 28, Swizzle::SW_64KB_S.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    tile(&pool, 0, &info.layout);
    let words = crate::resource::texture_desc(res).unwrap();
    let heap = crate::descriptor::test_heap(false, 4);
    let start = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let mut view = [0u32; 16];
    (view[0], view[1], view[5]) = (28, 4, 1);
    (view[10], view[11]) = (pool.va(0) as u32, (pool.va(0) >> 32) as u32);
    crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, start);
    // The heap is made (and the view written into it) when a list names it.
    let key = crate::native::textures::placed_key(pool.va(0), &words);
    with_title(|t| {
        assert!(t.host.knows(key), "made when the view was written");
        t.sync_textures();
    });
    let out = read_mip(key, 0, 32, 32);
    assert_eq!(out[(5 * 32 + 9) * 4..][..4], texel(0, 9, 5));
}

/// A placed texture has no object the title releases: it goes when another description is placed
/// at its address, and when its pool is released.
#[test]
fn a_placed_texture_goes_when_replaced_and_with_its_pool() {
    let Some(pool) = pool(9107, 4) else { return };
    let heap = crate::descriptor::test_heap(false, 4);
    let start = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let place = |size: u32| {
        let res = crate::resource::test_texture_at(pool.va(0), (size.into(), size), 1, 28, Swizzle::SW_64KB_S.0, 0);
        let words = crate::resource::texture_desc(res).unwrap();
        let mut view = [0u32; 16];
        (view[0], view[1], view[5]) = (28, 4, 1);
        (view[10], view[11]) = (pool.va(0) as u32, (pool.va(0) >> 32) as u32);
        crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, start);
        crate::native::textures::placed_key(pool.va(0), &words)
    };
    let first = place(32);
    assert!(with_title(|t| t.host.knows(first)).unwrap());
    let second = place(16);
    assert_ne!(first, second);
    with_title(|t| {
        assert!(t.host.knows(second));
        assert!(!t.host.knows(first), "replaced at its address");
    });
    let info = PoolInfo {
        id: 9107,
        base: pool.mem.primary as usize,
        len: pool.mem.len,
        memory_entry: None,
    };
    glue::pool_observer().pool_released(&info);
    with_title(|t| {
        t.sync_dirty();
        assert!(!t.host.knows(second), "released with its pool");
        assert!(t.texture_of(second).is_none());
    });
}

/// A render target the title clears through its view: the host texture is the target, never
/// filled from memory.
#[test]
fn a_render_target_is_cleared_through_the_titles_view() {
    let Some(pool) = pool(9104, 4) else { return };
    let rt = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::SW_64KB_R_X.0, 1);
    pool.write(0, &[0x7Fu8; 64]);
    // An RTV in descriptor memory the title owns.
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
    let list = crate::list::test_list(false, std::ptr::null());
    let (q, f) = (crate::queue::test_queue(false), {
        let mut f = std::ptr::null();
        crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
        f
    });
    // OMSetRenderTargets(1, &handle, false, null), then ClearRenderTargetView(handle, color).
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        1,
        &handle,
        0,
        std::ptr::null(),
    );
    let color = [0.0f32, 1.0, 0.0, 1.0];
    slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(list, 0x1a0)(
        list,
        handle,
        &color,
        0,
        std::ptr::null(),
    );
    assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    assert_eq!(
        slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(
            f,
            1,
            std::ptr::null_mut()
        ),
        0
    );
    let out = read_mip(rt as usize, 0, 16, 16);
    assert!(out.chunks(4).all(|p| p == [0, 255, 0, 255]), "{:?}", &out[..8]);
}

/// A render target view names the format its writes are stored in. Through a plain view of an
/// sRGB texture a value is stored as it is (a movie's picture, which is display values already);
/// through an sRGB view of a plain texture it is encoded.
#[test]
fn a_render_target_view_stores_in_the_format_it_names() {
    let Some(pool) = pool(9120, 16) else { return };
    // The texture's format, the view's (0: none described), and the byte half grey is stored as.
    for (i, (own, named, stored)) in [(29, 28, 128), (28, 29, 188), (29, 0, 188), (28, 28, 128)]
        .into_iter()
        .enumerate()
    {
        let at = i * 0x10000;
        let rt = crate::resource::test_texture_at(
            pool.va(at),
            (16, 16),
            1,
            own,
            Swizzle::SW_64KB_R_X.0,
            1,
        );
        let heap = crate::descriptor::test_heap(false, 4);
        let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
        // A 2D view of mip 0.
        let desc = [named, 4, 0, 0, 0, 0];
        let desc = if named == 0 {
            std::ptr::null()
        } else {
            &raw const desc
        };
        crate::descriptor::create_rtv(std::ptr::null(), rt, desc, handle);
        let list = crate::list::test_list(false, std::ptr::null());
        let (q, f) = (crate::queue::test_queue(false), {
            let mut f = std::ptr::null();
            crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
            f
        });
        slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
            list,
            1,
            &handle,
            0,
            std::ptr::null(),
        );
        let color = [0.5f32, 0.5, 0.5, 1.0];
        slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(list, 0x1a0)(
            list,
            handle,
            &color,
            0,
            std::ptr::null(),
        );
        assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
        slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
        slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
        assert_eq!(
            slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(
                f,
                1,
                std::ptr::null_mut()
            ),
            0
        );
        let out = read_mip(rt as usize, 0, 16, 16);
        assert!(
            out.chunks(4)
                .all(|p| p[0].abs_diff(stored) <= 1 && p[3] == 255),
            "texture {own} view {named}: {:?}",
            &out[..4]
        );
    }
}

/// A layout the host cannot address (`SW_4KB_R_X`, which has no equations) is counted and left empty, not
/// an error that stops the submission.
#[test]
fn an_undetilable_layout_is_counted_not_fatal() {
    let Some(pool) = pool(9105, 4) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (64, 64), 1, 28, Swizzle::SW_4KB_R_X.0, 0);
    let before = with_title(|t| t.stats.undetilable).unwrap();
    with_title(|t| {
        assert!(t.realize(res as usize));
        t.sync_textures();
        t.sync_textures();
    });
    assert_eq!(with_title(|t| t.stats.undetilable).unwrap(), before + 1, "once per content");
}

/// `PresentX`: the first resource of the largest plane is drawn over the app's layer (scaled,
/// whatever its channel order), after the queue's earlier work. Offscreen: the layer is not in a
/// window; its last drawable is read back.
#[test]
fn present_draws_the_back_buffer_over_the_apps_layer() {
    check_present_order(false);
    check_present_order(true);
}

fn check_present_order(held: bool) {
    let Some(pool) = pool(if held { 9116 } else { 9106 }, 8) else { return };
    let rt = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::SW_64KB_S.0, 1);
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
    let layer = glue::create_layer(32, 32).expect("a layer");
    glue::set_layer(layer.as_ptr()).expect("a surface");
    let (q, f) = (crate::queue::test_queue(false), {
        let mut f = std::ptr::null();
        crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
        f
    });
    let mut gate = std::ptr::null();
    crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut gate);
    if held {
        glue::queue_wait(q, gate, 1);
    }
    let list = crate::list::test_list(false, std::ptr::null());
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        1,
        &handle,
        0,
        std::ptr::null(),
    );
    // Red in the title's RGBA.
    let color = [1.0f32, 0.0, 0.0, 1.0];
    slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(list, 0x1a0)(
        list,
        handle,
        &color,
        0,
        std::ptr::null(),
    );
    slot::<extern "system" fn(This) -> i32>(list, 0x48)(list);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    // One plane: the frame token, one resource, its address.
    #[repr(C)]
    struct Plane {
        token: u64,
        count: u32,
        pad: u32,
        resources: *const This,
    }
    let plane = Plane {
        token: 1,
        count: 1,
        pad: 0,
        resources: &rt,
    };
    let before = glue::frames_presented();
    let present = slot::<extern "system" fn(This, u32, *const Plane, *const u8) -> i32>(q, 0xf0);
    let frames = if held { 6 } else { 1 };
    for _ in 0..frames {
        assert_eq!(present(q, 1, &plane, std::ptr::null()), 0);
    }
    // A signal after the present must also stay behind it.
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    if held {
        // Present must remain behind the unsignalled wait and the red clear.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(glue::frames_presented(), before, "present bypassed the fence wait");
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 0);
        // Signal the backend directly so only the timer worker drains the deferred presents.
        glue::with_title(|t| {
            use victory_backend::backend::ComputeBackend;
            let f = t.fence(gate as usize, 0).unwrap();
            t.host.backend.signal(&f, 1);
        });
    }
    let start = std::time::Instant::now();
    while glue::frames_presented() < before + frames {
        assert!(start.elapsed() < std::time::Duration::from_secs(5), "worker failed to drain presents");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(glue::frames_presented(), before + frames);
    // The frame is drawn when the fence is reached.
    assert_eq!(
        slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(f, 1, std::ptr::null_mut()),
        0
    );
    let (w, h, bgra) = glue::last_frame().expect("a frame");
    assert_eq!((w, h), (32, 32));
    assert!(
        bgra.chunks(4).all(|p| p == [0, 0, 255, 255]),
        "red as BGRA: {:?}",
        &bgra[..8]
    );
    drop(layer);
}

/// A colour target the host cannot make must keep its slot: binding the later ones would put
/// `SV_Target1` on slot 0 and write a surface this pass does not own. Nothing is bound instead,
/// and the draws that follow are refused.
#[test]
fn a_target_the_host_cannot_make_does_not_shift_the_ones_after_it() {
    let Some(pool) = pool(9201, 4) else { return };
    // 12 (R16G16B16A16_UINT) has no host texture; 28 is R8G8B8A8_UNORM.
    let gone = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 12, Swizzle::SW_64KB_R_X.0, 1);
    let good = crate::resource::test_texture_at(pool.va(0x2000), (16, 16), 1, 28, Swizzle::SW_64KB_R_X.0, 1);
    // Two view handles, 64 bytes apart, as a title's own descriptor memory holds them.
    let mut slots = vec![0u64; 16];
    let first = slots.as_mut_ptr() as u64;
    let second = first + 64;
    crate::descriptor::create_rtv(std::ptr::null(), gone, std::ptr::null(), first);
    crate::descriptor::create_rtv(std::ptr::null(), good, std::ptr::null(), second);
    let list = crate::list::test_list(false, std::ptr::null());
    let handles = [first, second];
    // OMSetRenderTargets(2, &handles, false, null).
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        handles.len() as u32,
        handles.as_ptr(),
        0,
        std::ptr::null(),
    );
    let bound = with_list(list, |_, l| l.targets.color.len()).unwrap();
    assert_eq!(bound, 0, "a target without a host texture must not renumber the ones after it");
    // A set the host can make is taken again.
    let only = [second];
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        only.len() as u32,
        only.as_ptr(),
        0,
        std::ptr::null(),
    );
    let bound = with_list(list, |_, l| l.targets.color.len()).unwrap();
    assert_eq!(bound, 1, "a makeable target binds again after a refused pass");
    drop(slots);
}

/// A compute dispatch writes a tiled title texture through a UAV; once its fence value shows, the
/// title's own memory holds the texels, in the texture's layout (the ocean maps' path, #46).
#[test]
fn a_texture_compute_writes_is_copied_back_into_title_memory() {
    // Skips without the Shader Converter.
    if super::host().is_none() {
        return;
    }
    let dxil = include_bytes!("../../../victory-metal/tests/fixtures/tex_fill.dxil");
    const W: u32 = 64;
    // A title host of its own: no timer thread, and a pool no other test shares.
    let mut t = Title::new(victory_metal::metal::Metal::new().unwrap(), Default::default());
    let mem = Alias::new((64 << 10).max(page_size())).unwrap();
    let info = PoolInfo { id: 9131, base: mem.primary as usize, len: mem.len, memory_entry: None };
    t.host.memory().pool_created(&info);
    t.host.memory().mapped(mem.alias as usize, mem.len, &info, 0);
    let pool = Pool { mem };
    t.set_texture_back(true);
    for (n, swizzle) in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_D_X].into_iter().enumerate() {
        // Flags 4: unordered access. DXGI 28: RGBA8.
        let res = crate::resource::test_texture_at(pool.va(0), (u64::from(W), W), 1, 28, swizzle.0, 4);
        let info = crate::resource::texture_info(res).unwrap();
        let key = res as usize;
        pool.write(0, &vec![0u8; info.size]);
        // The title's descriptor memory: one UAV (format, 2D, mip 0).
        let slots = vec![0u32; 8 * 8];
        let handle = slots.as_ptr() as u64;
        let uav = [28u32, 4, 0, 0, 0, 0, 0, 0, 0, 0];
        crate::descriptor::create_uav(std::ptr::null(), res, std::ptr::null(), &uav, handle);
        let (queue, fence) = (1 + n, 11 + n);
        t.texture_created(key, info.clone());
        assert!(t.realize(key), "{swizzle:?}");
        let heap = t.host.heap(HeapKind::Resource, handle, 8).unwrap();
        let pso = t.host.compute_pipeline(dxil, [8, 8, 1]).unwrap();
        let mut l = NativeList::default();
        l.set_heaps(Some(&heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_table(0, &heap, 0);
        l.dispatch(W / 8, W / 8, 1).unwrap();
        t.ensure_queue(queue).unwrap();
        t.submit(queue, vec![(l, vec![])]);
        t.signal_queue(queue, fence, 0, 1);
        let start = std::time::Instant::now();
        while !t.try_wait(fence, 0, 1) {
            assert!(start.elapsed().as_secs() < 10, "the dispatch did not complete");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // The value shows, so the memory is there already.
        let mut out = vec![0u8; info.size];
        assert!(crate::native::safe_read(pool.va(0) as *const u8, &mut out));
        let linear: Vec<u8> = (0..W)
            .flat_map(|y| (0..W).flat_map(move |x| [x as u8, y as u8, 64, 255]))
            .collect();
        let mut want = vec![0u8; info.size];
        info.layout
            .tile(&mut want, 0, 0, &linear, W as usize * 4, linear.len())
            .unwrap();
        assert!(out == want, "{swizzle:?}: title memory is not the retiled texels");
        // Our own write is not mistaken for the title's: nothing is uploaded again.
        let uploads = t.uploads_of(key);
        t.sync_textures();
        assert_eq!(t.uploads_of(key), uploads, "{swizzle:?}");
        // The title writes the memory while the same work runs: its bytes stay.
        let mut l = NativeList::default();
        let heap = t.host.heap_at(handle).unwrap().0;
        l.set_heaps(Some(&heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_table(0, &heap, 0);
        l.dispatch(W / 8, W / 8, 1).unwrap();
        t.submit(queue, vec![(l, vec![])]);
        t.signal_queue(queue, fence, 0, 2);
        pool.write(0, &vec![0x5Au8; info.size]);
        let start = std::time::Instant::now();
        while !t.try_wait(fence, 0, 2) {
            assert!(start.elapsed().as_secs() < 10, "the dispatch did not complete");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(crate::native::safe_read(pool.va(0) as *const u8, &mut out));
        assert!(out.iter().all(|&b| b == 0x5A), "{swizzle:?}: the title's own write was overwritten");
        t.host.drop_heap(handle);
        t.host.forget_resource(key);
        t.texture_released(key);
    }
}

/// Depth and depth-stencil textures are GPU targets whose CPU tile layout (e.g. a 4-byte or
/// 2-byte depth plane) does not match Metal's combined texel layout, so they must not be watched
/// or uploaded from CPU memory even when placed/typeless (not marked as a render target).
#[test]
fn depth_textures_are_not_watched_or_uploaded_from_cpu_memory() {
    let Some(pool) = pool(9202, (512 << 10) / page_size() + 1) else { return };
    // DXGI formats: 20 (D32_FLOAT_S8X24_UINT), 40 (D32_FLOAT), 55 (D16_UNORM),
    // and typeless 19 (R32G8X24_TYPELESS) with flags = 0 (not marked ALLOW_RENDER_TARGET).
    for (i, format) in [20u32, 40, 55, 19].into_iter().enumerate() {
        let at = i * 0x20000;
        let res = crate::resource::test_texture_at(
            pool.va(at),
            (64, 64),
            1,
            format,
            Swizzle::SW_64KB_R_X.0,
            0,
        );
        let key = res as usize;
        with_title(|t| {
            assert!(t.realize(key), "format {format} should realize");
            t.sync_textures();
            assert_eq!(t.uploads_of(key), 0, "depth format {format} must not upload on sync_textures");
        });
        pool.write(at, &[0xA5u8; 64]);
        with_title(|t| {
            t.sync_dirty();
            assert_eq!(t.uploads_of(key), 0, "depth format {format} must not upload on sync_dirty");
        });
    }
}

/// A described RTV binds and clears its selected subresource in the named format.
#[test]
fn a_render_target_view_selects_its_mip_and_slice() {
    use std::rc::Rc;
    use victory_backend::backend::ComputeBackend;

    let pool = pool(9138, 16).expect("Metal device and generated texture pool");
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let other_heap = crate::descriptor::test_heap(false, 4);
    let next_handle = slot::<extern "system" fn(This) -> u64>(other_heap, 0x48)(other_heap);
    let grey = [0.5, 0.5, 0.5, 1.0];

    for (case, own_format, layers, array_view, mip, slice) in [
        ("2D mip", 28, 1, false, 1, 0),
        ("array slice", 28, 2, true, 0, 1),
        ("array mip and slice", 28, 2, true, 1, 1),
        ("plain view of an sRGB array mip and slice", 29, 2, true, 1, 1),
        ("nonarray view of the first slice", 28, 2, false, 1, 0),
    ] {
        let rt = crate::resource::test_texture_array_at(pool.va(0), (8, 8), 2, layers, own_format, Swizzle::LINEAR.0, 1);
        let key = rt as usize;
        let subresources: Vec<_> = (0..usize::from(layers))
            .flat_map(|layer| (0..2).map(move |level| {
                (level, layer, 8 >> level, [17 + (level * 40 + layer * 80) as u8, 31, 47, 255])
            }))
            .collect();
        with_title(|title| {
            let texture = title.texture_of(key).expect("realized texture");
            for &(level, layer, n, sentinel) in &subresources {
                title.host.backend.upload_texture(&texture, level, layer, &sentinel.repeat(n * n), n * 4, 0);
            }
        }).expect("Metal title");
        let desc = if array_view {
            [28, 5, mip, slice, 1, 0]
        } else {
            [28, 4, mip, 0, 0, 0]
        };
        crate::descriptor::create_rtv(std::ptr::null(), rt, &desc, handle);
        crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), next_handle);
        let list = crate::list::test_list(false, std::ptr::null());
        glue::targets(list, &[handle], 0);
        with_list(list, |title, l| {
            let (expected, format) = title.target_of(crate::descriptor::rtv_record(handle).unwrap()).unwrap();
            let (base, base_format) = title.target_of(crate::descriptor::rtv_record(next_handle).unwrap()).unwrap();
            assert!(l.targets_ready);
            assert_eq!(l.targets.color, [Format::Rgba8Unorm]);
            assert_eq!(format, Format::Rgba8Unorm);
            let [attached] = l.list.color_targets() else { panic!("one attachment was expected") };
            assert!(Rc::ptr_eq(attached, &expected), "binding reuses the cached attachment");
            assert!(Rc::ptr_eq(&base, &title.texture_of(key).unwrap()), "default keeps base attachment");
            assert!(!Rc::ptr_eq(&base, attached));
            assert_eq!(base_format, if own_format == 29 { Format::Rgba8UnormSrgb } else { Format::Rgba8Unorm });
            if array_view {
                for (other_mip, other_slice) in [(1 - mip, slice), (mip, 1 - slice)] {
                    let other_desc = [28, 5, other_mip, other_slice, 1, 0];
                    crate::descriptor::create_rtv(std::ptr::null(), rt, &other_desc, next_handle);
                    let (other, _) = title.target_of(crate::descriptor::rtv_record(next_handle).unwrap()).unwrap();
                    assert!(!Rc::ptr_eq(attached, &other), "distinct subresources must keep distinct identities");
                }
            }
        }).expect("native command list");

        let clear = slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(list, 0x1a0);
        clear(list, handle, &grey, 0, std::ptr::null());
        with_list(list, |_, l| {
            let [super::super::list::Cmd::Clear { color: Some((cleared, _)), .. }] = l.list.cmds.as_slice() else {
                panic!("one clear was expected");
            };
            assert!(Rc::ptr_eq(&l.list.color_targets()[0], cleared), "draw binding and clear resolve the same view");
        }).expect("native command list");
        if layers == 2 {
            for invalid in [
                [28, 5, 2, 0, 1, 0], // Out-of-range mip.
                [28, 5, 0, 2, 1, 0], // Out-of-range slice.
                [28, 5, 0, 0, 0, 0], // Empty array view.
                [28, 5, 0, 0, 2, 0], // Unsupported layered view.
                [28, 5, 0, 0, 1, 1], // Unsupported plane.
                [28, 2, 0, 0, 0, 0], // 1D view of a 2D resource.
                [28, 6, 0, 0, 0, 0], // Multisampled view of a single-sampled resource.
            ] {
                crate::descriptor::create_rtv(std::ptr::null(), rt, &invalid, next_handle);
                glue::targets(list, &[next_handle, handle], 0);
                clear(list, next_handle, &[1.0, 0.0, 0.0, 1.0], 0, std::ptr::null());
                with_list(list, |_, l| {
                    assert!(!l.targets_ready, "{case} accepted {invalid:?}");
                    assert!(l.list.color_targets().is_empty(), "valid attachment must keep its slot");
                    assert_eq!(l.list.cmds.len(), 1, "an invalid clear must record no command");
                }).expect("native command list");
            }
        }

        assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
        let queue = crate::queue::test_queue(false);
        let mut fence = std::ptr::null();
        assert_eq!(crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence), 0);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(slot::<extern "system" fn(This, This, u64) -> i32>(queue, 0x70)(queue, fence, 1), 0);
        with_title(|title| {
            let fence = title.fence(fence as usize, 0).expect("native fence");
            assert!(title.host.backend.wait(&fence, 1, 2000), "native clear did not finish");
            let texture = title.texture_of(key).expect("realized texture");
            let mut pixels = 0;
            for &(level, layer, n, sentinel) in &subresources {
                let expected = if (level, layer) == (mip as usize, slice as usize) { [128, 128, 128, 255] } else { sentinel };
                let mut data = vec![0u8; n * n * 4];
                title.host.backend.read_texture(&texture, level, layer, &mut data, n * 4, 0);
                assert!(data.as_chunks::<4>().0.iter().all(|p| *p == expected),
                        "{case} mip {level} slice {layer}: expected {expected:?}, got {:?}", &data[..4]);
                pixels += n * n;
            }
            println!("{case}: cleared mip {mip} slice {slice}, verified {pixels} texels");
        }).expect("Metal title");
    }
}

/// A rectangle-list draw (`D3D_PRIMITIVE_TOPOLOGY_RECTLIST`, 17) on a described (non-guessed)
/// pipeline must draw the supplied vertices as `Topology::Triangles` instead of issuing
/// 4-vertex instanced strips designed for `Options::rect_list`.
#[test]
fn described_rectangle_list_draw_uses_supplied_triangles() {
    let Some(pool) = pool(9204, 4) else { return };
    let rt = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::SW_64KB_R_X.0, 1);
    let pso = with_title(|t| {
        let vs = super::fixture("tri_vs");
        let ps = super::fixture("tri_ps");
        t.host
            .render_pipeline(&RenderPipelineDesc {
                vs: &vs,
                ps: Some(&ps),
                vs_layout: None,
                ps_layout: None,
                attributes: &[
                    VertexAttr { format: VertexFormat::Float2, slot: 0, offset: 0 },
                    VertexAttr { format: VertexFormat::Float2, slot: 0, offset: 8 },
                    VertexAttr { format: VertexFormat::Float4, slot: 0, offset: 16 },
                ],
                streams: &[VertexStream { slot: 0, stride: 32, per_instance: false }],
                color: &[Format::Rgba8Unorm],
                depth: None,
                depth_test: None,
                cull: Cull::None,
                ..Default::default()
            })
            .unwrap()
    })
    .unwrap();
    let pipe_key = 0xDEAD_BEE0usize;
    glue::install_test_render_pipeline(pipe_key, pso);

    let list = crate::list::test_list(false, std::ptr::null());
    glue::set_pipeline(list, pipe_key as This);
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        1,
        &handle,
        0,
        std::ptr::null(),
    );
    glue::topology(list, 17);
    glue::draw(list, 6, 1, 3, 0);

    let recorded = with_list(list, |_, l| match l.list.cmds.last() {
        Some(crate::native::list::Cmd::Draw {
            topology,
            call:
                DrawCall::Arrays {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                },
            ..
        }) => Some((*topology, *vertex_count, *instance_count, *first_vertex, *first_instance)),
        _ => None,
    })
    .flatten()
    .expect("recorded draw");
    assert_eq!(recorded, (Topology::Triangles, 6, 1, 3, 0));
}

/// `DispatchMeshX` with an argument address, called through the list's method table: it records a
/// mesh draw whose group counts are the record at that address, read on the GPU. A record of one
/// group colours the target; a record of zero groups draws nothing.
#[test]
fn an_indirect_dispatch_mesh_x_draws_the_groups_its_record_holds() {
    use super::mesh_tests::{RED, TRIANGLE, translate};
    use victory_backend::backend::ComputeBackend;
    use victory_shader::native::HwStage;
    let Some(pool) = pool(9205, 4) else { return };
    let (ms, ps) = (translate(HwStage::Vs, TRIANGLE), translate(HwStage::Ps, RED));
    let state = crate::native::state::Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let pso = match with_title(|t| t.render_pipe(&ms, Some(&ps), &state, "mesh")).unwrap() {
        crate::native::title::Pipe::Render(pso, _) => pso,
        _ => panic!("no mesh pipeline"),
    };
    let pipe_key = 0xDEAD_BEE8usize;
    glue::install_test_render_pipeline(pipe_key, pso);
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let records = 0x8000;
    pool.write(records, &[1u32, 1, 1, 0, 0, 0]);
    for (i, want) in [(0usize, [255u8, 128, 128, 255]), (1, [0; 4])] {
        let rt = crate::resource::test_texture_at(pool.va(0x2000 * i), (16, 16), 1, 28, Swizzle::SW_64KB_R_X.0, 1);
        crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
        let list = crate::list::test_list(false, std::ptr::null());
        glue::set_pipeline(list, pipe_key as This);
        slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(list, 1, &handle, 0, std::ptr::null());
        let record = pool.va(records + 12 * i);
        // DispatchMeshX(x, flags, indirect arguments, payload): x is unused with an address.
        slot::<extern "system" fn(This, u32, u64, u64, u64)>(list, 0x310)(list, 7, 0, record, 0);
        let recorded = with_list(list, |_, l| match l.list.cmds.last() {
            Some(crate::native::list::Cmd::Draw { call: DrawCall::Mesh { .. }, indirect, .. }) => *indirect,
            _ => None,
        })
        .flatten();
        assert_eq!(recorded, Some(record), "record {i}: an indirect mesh draw of the record");
        assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
        let queue = crate::queue::test_queue(false);
        let mut fence = std::ptr::null();
        assert_eq!(crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence), 0);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(slot::<extern "system" fn(This, This, u64) -> i32>(queue, 0x70)(queue, fence, 1), 0);
        with_title(|t| {
            let fence = t.fence(fence as usize, 0).expect("native fence");
            assert!(t.host.backend.wait(&fence, 1, 2000), "the mesh draw did not finish");
            let texture = t.texture_of(rt as usize).expect("realized target");
            let mut data = vec![0u8; 16 * 16 * 4];
            t.host.backend.read_texture(&texture, 0, 0, &mut data, 16 * 4, 0);
            assert!(data.as_chunks::<4>().0.iter().all(|p| *p == want), "record {i}: {:?}", &data[..8]);
        })
        .unwrap();
    }
}

/// A list with a mesh pipeline (one triangle covering the target, red) bound and a 16x16 RGBA8
/// target at `pool` offset `at`, after `call` records its mesh dispatch through the method table;
/// the target's pixels once the list ran.
fn mesh_dispatch_into(pool: &Pool, at: usize, pipe_key: usize, call: impl FnOnce(This)) -> Vec<[u8; 4]> {
    use super::mesh_tests::{RED, TRIANGLE, translate};
    use victory_shader::native::HwStage;
    let (ms, ps) = (translate(HwStage::Vs, TRIANGLE), translate(HwStage::Ps, RED));
    let state = crate::native::state::Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let pso = match with_title(|t| t.render_pipe(&ms, Some(&ps), &state, "mesh")).unwrap() {
        crate::native::title::Pipe::Render(pso, _) => pso,
        _ => panic!("no mesh pipeline"),
    };
    glue::install_test_render_pipeline(pipe_key, pso);
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let rt = crate::resource::test_texture_at(pool.va(at), (16, 16), 1, 28, Swizzle::SW_64KB_R_X.0, 1);
    crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
    let list = crate::list::test_list(false, std::ptr::null());
    glue::set_pipeline(list, pipe_key as This);
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(list, 1, &handle, 0, std::ptr::null());
    call(list);
    run_title_list(list);
    read_mip(rt as usize, 0, 16, 16).as_chunks::<4>().0.to_vec()
}

/// `DispatchMeshX` without an argument address runs `x` groups of the bound mesh pipeline (#353):
/// one group colours the target, zero groups leave it as it was. It was dropped before.
#[test]
fn a_direct_dispatch_mesh_x_draws_its_groups() {
    let Some(pool) = pool(9217, 4) else { return };
    let red = [255u8, 128, 128, 255];
    for (i, (x, want)) in [(1u32, red), (0, [0; 4])].into_iter().enumerate() {
        let before = crate::list::DROPPED_KINDS[0].load(std::sync::atomic::Ordering::Relaxed);
        // DispatchMeshX(x, flags, indirect arguments, payload), no arguments.
        let got = mesh_dispatch_into(&pool, 0x2000 * i, 0xDEAD_BEF1, |list| {
            slot::<extern "system" fn(This, u32, u64, u64, u64)>(list, 0x310)(list, x, 0, 0, 0)
        });
        assert!(got.iter().all(|p| *p == want), "x {x}: {:?}", &got[..2]);
        assert_eq!(crate::list::DROPPED_KINDS[0].load(std::sync::atomic::Ordering::Relaxed), before, "x {x}: not dropped");
    }
}

/// `DispatchMesh(x, y, z)` (method 0x3f8) runs those groups of the bound mesh pipeline (#353).
#[test]
fn a_direct_dispatch_mesh_draws_its_groups() {
    let Some(pool) = pool(9218, 4) else { return };
    let got = mesh_dispatch_into(&pool, 0, 0xDEAD_BEF2, |list| {
        slot::<extern "system" fn(This, u32, u32, u32)>(list, 0x3f8)(list, 1, 1, 1)
    });
    assert!(got.iter().all(|p| *p == [255, 128, 128, 255]), "{:?}", &got[..2]);
}

/// A render target a recorded draw wrote is the GPU's: when the title then changes the memory
/// under it, the next submission does not upload that memory over what was drawn.
#[test]
fn a_drawn_target_is_not_filled_from_title_memory_again() {
    use super::mesh_tests::{RED, TRIANGLE, translate};
    use victory_backend::backend::ComputeBackend;
    use victory_shader::native::HwStage;
    let Some(pool) = pool(9206, 4) else { return };
    let (ms, ps) = (translate(HwStage::Vs, TRIANGLE), translate(HwStage::Ps, RED));
    let state = crate::native::state::Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let pso = match with_title(|t| t.render_pipe(&ms, Some(&ps), &state, "mesh")).unwrap() {
        crate::native::title::Pipe::Render(pso, _) => pso,
        _ => panic!("no mesh pipeline"),
    };
    let pipe_key = 0xDEAD_BEE9usize;
    glue::install_test_render_pipeline(pipe_key, pso);
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let records = 0x8000;
    pool.write(records, &[1u32, 1, 1]);
    let rt = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::SW_64KB_R_X.0, 1);
    crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
    let list = crate::list::test_list(false, std::ptr::null());
    glue::set_pipeline(list, pipe_key as This);
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(list, 1, &handle, 0, std::ptr::null());
    slot::<extern "system" fn(This, u32, u64, u64, u64)>(list, 0x310)(list, 7, 0, pool.va(records), 0);
    run_title_list(list);
    let red = [255u8, 128, 128, 255];
    let drawn = || read_mip(rt as usize, 0, 16, 16).as_chunks::<4>().0.iter().all(|p| *p == red);
    assert!(drawn(), "the draw coloured the target");
    // The title changes the memory under the target, and the scanner reports it at the next
    // submission.
    let uploads = with_title(|t| t.uploads_of(rt as usize)).unwrap();
    with_title(|_| pool.write(0, &[0x55u8; 16 * 16 * 4])).unwrap();
    run_title_list(crate::list::test_list(false, std::ptr::null()));
    with_title(|t| {
        let jobs = t.jobs(vec![rt as usize]);
        t.sync(jobs);
    });
    assert_eq!(with_title(|t| t.uploads_of(rt as usize)).unwrap(), uploads, "not uploaded again");
    assert!(drawn(), "what was drawn stays");
}

/// `CopyResource` between textures and `CopyTextureRegion` (subresource-to-subresource and
/// footprint-to/from-subresource) execute on Metal without dropping to `dropped_op`.
#[test]
fn copy_resource_and_copy_texture_region_run_on_the_gpu() {
    let Some(pool) = pool(9203, (512 << 10) / page_size() + 1) else { return };
    let tex_a = crate::resource::test_texture_at(
        pool.va(0),
        (16, 16),
        1,
        28,
        Swizzle::SW_64KB_R_X.0,
        1,
    );
    let tex_b = crate::resource::test_texture_at(
        pool.va(0x10000),
        (16, 16),
        1,
        28,
        Swizzle::SW_64KB_R_X.0,
        1,
    );
    let tex_c = crate::resource::test_texture_at(
        pool.va(0x20000),
        (16, 16),
        1,
        28,
        Swizzle::SW_64KB_R_X.0,
        1,
    );
    let up_buf = crate::resource::test_buffer_at(pool.va(0x30000), 0x4000);
    let down_buf = crate::resource::test_buffer_at(pool.va(0x40000), 0x4000);

    // Stage 8x8 linear texels at row pitch 256 into `up_buf`.
    let pitch = 256usize;
    let mut staged = vec![0u8; 7 * pitch + 8 * 4];
    for y in 0..8usize {
        for x in 0..8usize {
            let at = y * pitch + x * 4;
            staged[at..at + 4].copy_from_slice(&[x as u8 * 11, y as u8 * 23, 77, 255]);
        }
    }
    pool.write(0x30000, &staged);

    let list = crate::list::test_list(false, std::ptr::null());
    let q = crate::queue::test_queue(false);
    let mut f = std::ptr::null();
    crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);

    // 1. CopyTextureRegion (footprint -> tex_a at (4, 4, 0) from 4x4 box starting at (2, 3))
    let src_fp = crate::list::CopyLocation::placed(
        up_buf,
        crate::resource::PlacedFootprint::new(0, 28, 8, 8, pitch as u32),
    );
    let dst_a = crate::list::CopyLocation::whole(tex_a);
    let box_a = [2u32, 3, 0, 6, 7, 1];
    slot::<extern "system" fn(This, *const crate::list::CopyLocation, u32, u32, u32, *const crate::list::CopyLocation, *const [u32; 6])>(
        list, 0x80,
    )(list, &dst_a, 4, 4, 0, &src_fp, &box_a);

    // 2. CopyResource (tex_a -> tex_b whole texture)
    slot::<extern "system" fn(This, This, This)>(list, 0x88)(list, tex_b, tex_a);

    // 3. CopyTextureRegion (tex_b subresource (4, 4)..(8, 8) -> tex_c at (1, 2))
    let src_b = crate::list::CopyLocation::whole(tex_b);
    let dst_c = crate::list::CopyLocation::whole(tex_c);
    let box_b = [4u32, 4, 0, 8, 8, 1];
    slot::<extern "system" fn(This, *const crate::list::CopyLocation, u32, u32, u32, *const crate::list::CopyLocation, *const [u32; 6])>(
        list, 0x80,
    )(list, &dst_c, 1, 2, 0, &src_b, &box_b);

    // 4. CopyTextureRegion (tex_c subresource (1, 2)..(5, 6) -> down_buf footprint at (0, 0))
    let src_c = crate::list::CopyLocation::whole(tex_c);
    let dst_fp = crate::list::CopyLocation::placed(
        down_buf,
        crate::resource::PlacedFootprint::new(0, 28, 4, 4, pitch as u32),
    );
    let box_c = [1u32, 2, 0, 5, 6, 1];
    slot::<extern "system" fn(This, *const crate::list::CopyLocation, u32, u32, u32, *const crate::list::CopyLocation, *const [u32; 6])>(
        list, 0x80,
    )(list, &dst_fp, 0, 0, 0, &src_c, &box_c);

    assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
    let before_drops = crate::list::DROPPED_KINDS[3].load(std::sync::atomic::Ordering::Relaxed);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    assert_eq!(
        slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(
            f,
            1,
            std::ptr::null_mut()
        ),
        0
    );
    let after_drops = crate::list::DROPPED_KINDS[3].load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(after_drops, before_drops, "texture copies must not be counted as dropped");

    let mut readback = vec![0u8; 3 * pitch + 16];
    assert!(crate::native::safe_read(pool.va(0x40000) as *const u8, &mut readback));
    for row in 0..4usize {
        for col in 0..4usize {
            let at = row * pitch + col * 4;
            assert_eq!(
                &readback[at..at + 4],
                &[(col as u8 + 2) * 11, (row as u8 + 3) * 23, 77, 255],
                "readback texel ({col}, {row})"
            );
        }
    }
}

#[test]
fn a_rectangular_color_clear_preserves_texels_outside_the_selected_regions() {
    for (storage_format, view_format, color, pixel) in [
        (28, 28, [1.0, 0.0, 0.0, 1.0], [255, 0, 0, 255]),
        (28, 29, [0.5, 0.5, 0.5, 1.0], [188, 188, 188, 255]),
        (30, 30, [17.0, 65.0, 201.0, 255.0], [17, 65, 201, 255]),
    ] {
        check_rectangular_color_clear(storage_format, view_format, color, pixel);
    }
}

fn check_rectangular_color_clear(
    storage_format: u32,
    view_format: u32,
    color: [f32; 4],
    cleared_pixel: [u8; 4],
) {
    use crate::list::Rect;
    use victory_backend::backend::ComputeBackend;

    let pool = pool(9139, 16).expect("Metal device and generated texture pool");
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let resource = crate::resource::test_texture_array_at(
        pool.va(0), (8, 6), 2, 2, storage_format, Swizzle::LINEAR.0, 1,
    );
    let key = resource as usize;
    let desc = [view_format, 5, 1, 1, 1, 0];
    crate::descriptor::create_rtv(std::ptr::null(), resource, &desc, handle);

    let sentinel = |mip, slice| [17 + (mip * 40 + slice * 80) as u8, 31, 47, 255];
    let queue = crate::queue::test_queue(false);
    let mut fence = std::ptr::null();
    assert_eq!(crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence), 0);
    let cases: [(&str, &[[i32; 4]], &[usize]); 6] = [
        ("interior", &[[1, 1, 3, 2]], &[5, 6]),
        ("bounded", &[[-2, 1, 2, 2]], &[4, 5]),
        ("empty and inverted", &[[1, 1, 1, 2], [3, 1, 1, 2]], &[]),
        ("wholly outside", &[[4, 0, 5, 3]], &[]),
        ("overlapping and disjoint", &[[0, 0, 2, 1], [1, 0, 3, 1], [2, 2, 8, 3]], &[0, 1, 2, 10, 11]),
        ("zero count", &[], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ];

    for (case_index, (name, rects, required)) in cases.into_iter().enumerate() {
        with_title(|title| {
            let texture = title.texture_of(key).expect("realized texture");
            for slice in 0..2 {
                for mip in 0..2 {
                    let (w, h) = (8 >> mip, 6 >> mip);
                    let pixel = sentinel(mip, slice);
                    title.host.backend.upload_texture(&texture, mip, slice, &pixel.repeat(w * h), w * 4, 0);
                }
            }
        }).expect("Metal title");

        let list = crate::list::test_list(false, std::ptr::null());
        let clear = slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const Rect)>(list, 0x1a0);
        let unused = Rect { left: 0, top: 0, right: 0, bottom: 0 };
        let mut caller_rects: Vec<_> = rects.iter().map(|&[left, top, right, bottom]| {
            Rect { left, top, right, bottom }
        }).collect();
        let ptr = if caller_rects.is_empty() { &unused as *const _ } else { caller_rects.as_ptr() };
        clear(list, handle, &color, caller_rects.len() as u32, ptr);
        // The list must own the original coordinates at record time.
        caller_rects.fill(unused);
        std::hint::black_box(&caller_rects);

        assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
        let value = case_index as u64 + 1;
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(slot::<extern "system" fn(This, This, u64) -> i32>(queue, 0x70)(queue, fence, value), 0);
        with_title(|title| {
            let fence = title.fence(fence as usize, 0).expect("native fence");
            assert!(title.host.backend.wait(&fence, value, 2000), "native clear did not finish");
            let texture = title.texture_of(key).expect("realized texture");
            for slice in 0..2 {
                for mip in 0..2 {
                    let (w, h) = (8 >> mip, 6 >> mip);
                    let mut data = vec![0; w * h * 4];
                    title.host.backend.read_texture(&texture, mip, slice, &mut data, w * 4, 0);
                    for (index, pixel) in data.as_chunks::<4>().0.iter().enumerate() {
                        let expected = if (mip, slice) == (1, 1) && required.contains(&index) {
                            cleared_pixel
                        } else {
                            sentinel(mip, slice)
                        };
                        assert_eq!(*pixel, expected, "DXGI {view_format}, {name}, mip {mip}, slice {slice}, pixel {index}");
                    }
                }
            }
        }).expect("Metal title");
        println!("DXGI {view_format}, {name}: {} changed / {} preserved in view, adjacent subresources unchanged", required.len(), 12 - required.len());
    }
}

/// Closes title list `list`, submits it and waits for the GPU to run it.
fn run_title_list(list: This) {
    let (q, f) = (crate::queue::test_queue(false), {
        let mut f = std::ptr::null();
        crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
        f
    });
    assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    assert_eq!(slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(f, 1, std::ptr::null_mut()), 0);
}

type CopyRegion = extern "system" fn(This, *const crate::list::CopyLocation, u32, u32, u32, *const crate::list::CopyLocation, *const [u32; 6]);

/// `CopyResource` of two title textures is a GPU copy, and the destination stays the GPU's: a
/// later sync does not fill it from its title memory.
#[test]
fn copy_resource_copies_a_texture_on_the_gpu() {
    let Some(pool) = pool(9211, 8) else { return };
    let src = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::LINEAR.0, 0);
    let dst = crate::resource::test_texture_at(pool.va(65536), (16, 16), 1, 28, Swizzle::LINEAR.0, 0);
    tile(&pool, 0, &crate::resource::texture_info(src).unwrap().layout);
    let list = crate::list::test_list(false, std::ptr::null());
    slot::<extern "system" fn(This, This, This)>(list, 0x88)(list, dst, src);
    run_title_list(list);
    let want: Vec<u8> = (0..16).flat_map(|y| (0..16).flat_map(move |x| texel(0, x, y))).collect();
    assert!(read_mip(dst as usize, 0, 16, 16) == want, "the copy did not reach the destination");
    // Its title memory changes (it never held the copy): the host texture keeps the copy.
    pool.write(65536, &[0x11u8; 1024]);
    with_title(|t| t.sync_textures());
    assert!(read_mip(dst as usize, 0, 16, 16) == want, "the destination was filled again from title memory");
}

/// Plain texels of a compressed block's size copy into compressed blocks (a GPU encoder's
/// output), which no view joins: through a buffer, one texel a block.
#[test]
fn copy_texture_region_moves_plain_texels_into_compressed_blocks() {
    let Some(pool) = pool(9213, 8) else { return };
    // DXGI 3 (R32G32B32A32_UINT) 2x2 into DXGI 77 (BC3) 8x8.
    let src = crate::resource::test_texture_at(pool.va(0), (2, 2), 1, 3, Swizzle::LINEAR.0, 0);
    let dst = crate::resource::test_texture_at(pool.va(65536), (8, 8), 1, 77, Swizzle::LINEAR.0, 0);
    let blocks: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(7).wrapping_add(3)).collect();
    let info = crate::resource::texture_info(src).unwrap();
    let mut tiled = vec![0u8; info.layout.size as usize];
    info.layout.tile(&mut tiled, 0, 0, &blocks, 32, 64).unwrap();
    pool.write(0, &tiled);
    let list = crate::list::test_list(false, std::ptr::null());
    slot::<CopyRegion>(list, 0x80)(list, &crate::list::CopyLocation::whole(dst), 0, 0, 0, &crate::list::CopyLocation::whole(src), std::ptr::null());
    run_title_list(list);
    let out = with_title(|t| {
        let tex = t.texture_of(dst as usize).expect("a host texture");
        let mut out = vec![0u8; 64];
        t.host.backend.read_texture(&tex, 0, 0, &mut out, 32, 64);
        out
    })
    .unwrap();
    assert_eq!(out, blocks);
}

/// `ResolveSubresource` of single-sample textures: a copy of what both subresources hold (a
/// larger source is clipped to the destination).
#[test]
fn resolve_subresource_copies_what_both_hold() {
    let Some(pool) = pool(9214, 8) else { return };
    let src = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::LINEAR.0, 0);
    let dst = crate::resource::test_texture_at(pool.va(65536), (8, 8), 1, 28, Swizzle::LINEAR.0, 0);
    tile(&pool, 0, &crate::resource::texture_info(src).unwrap().layout);
    let list = crate::list::test_list(false, std::ptr::null());
    slot::<extern "system" fn(This, This, u32, This, u32, u32)>(list, 0x98)(list, dst, 0, src, 0, 28);
    run_title_list(list);
    let want: Vec<u8> = (0..8).flat_map(|y| (0..8).flat_map(move |x| texel(0, x, y))).collect();
    assert!(read_mip(dst as usize, 0, 8, 8) == want, "the resolve did not reach the destination");
}

/// A depth and stencil texture to and from buffer rows moves its 4-byte depth plane: rows in, then
/// a region of them out again into other rows.
#[test]
fn depth_stencil_rows_go_in_and_out_as_the_depth_plane() {
    use crate::list::CopyLocation;
    use crate::resource::PlacedFootprint;
    let Some(pool) = pool(9215, 8) else { return };
    // DXGI 19 (D32_FLOAT_S8X24_UINT), a depth target; rows as DXGI 41 (R32_FLOAT).
    let depth = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 19, Swizzle::LINEAR.0, 2);
    let buf = crate::resource::test_buffer_at(pool.va(65536), 65536);
    let values: Vec<f32> = (0..16 * 16).map(|i| i as f32 / 256.0).collect();
    for y in 0..16 {
        pool.write(65536 + y * 256, &values[y * 16..y * 16 + 16]);
    }
    let list = crate::list::test_list(false, std::ptr::null());
    let copy = slot::<CopyRegion>(list, 0x80);
    let rows = CopyLocation::placed(buf, PlacedFootprint::new(0, 41, 16, 16, 256));
    copy(list, &CopyLocation::whole(depth), 0, 0, 0, &rows, std::ptr::null());
    let out = CopyLocation::placed(buf, PlacedFootprint::new(8192, 41, 4, 2, 256));
    copy(list, &out, 0, 0, 0, &CopyLocation::whole(depth), &[3, 5, 0, 7, 7, 1]);
    run_title_list(list);
    for r in 0..2 {
        let got: Vec<f32> = pool.read(65536 + 8192 + r * 256, 4);
        let at = (5 + r) * 16 + 3;
        assert_eq!(got, values[at..at + 4], "row {r}");
    }
}

/// A placed R9G9B9E5 view laid out exactly as a live R32_UINT object at its address (a texture
/// compute writes as integers and the title samples as floats) names the object's host texture;
/// one that differs in size gets its own, as before; the shared one goes with the object.
#[test]
fn a_placed_rgb9e5_view_over_an_r32_uint_object_shares_its_texture() {
    let Some(pool) = pool(9109, 4) else { return };
    // Flags 4: unordered access. DXGI 42: R32_UINT; 67: R9G9B9E5.
    let object = crate::resource::test_texture_at(pool.va(0), (32, 32), 1, 42, Swizzle::SW_64KB_S.0, 4);
    let info = crate::resource::texture_info(object).unwrap();
    with_title(|t| {
        t.texture_created(object as usize, info);
        assert!(t.realize(object as usize));
    });
    let heap = crate::descriptor::test_heap(false, 4);
    let start = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let place = |size: u32| {
        let res = crate::resource::test_texture_at(pool.va(0), (size.into(), size), 1, 67, Swizzle::SW_64KB_S.0, 4);
        let words = crate::resource::texture_desc(res).unwrap();
        let mut view = [0u32; 16];
        (view[0], view[1], view[5]) = (67, 4, 1);
        (view[10], view[11]) = (pool.va(0) as u32, (pool.va(0) >> 32) as u32);
        crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, start);
        crate::native::textures::placed_key(pool.va(0), &words)
    };
    let key = place(32);
    with_title(|t| {
        let shared = t.host.texture_at(key).expect("registered");
        assert!(std::rc::Rc::ptr_eq(&shared, &t.texture_of(object as usize).unwrap()), "the object's texture");
        assert!(t.texture_of(key).is_none(), "no copy of its own");
    });
    let other = place(16);
    with_title(|t| {
        let own = t.texture_of(other).expect("a copy of its own");
        assert!(!std::rc::Rc::ptr_eq(&own, &t.texture_of(object as usize).unwrap()));
    });
    let key = place(32);
    with_title(|t| assert!(t.host.knows(key)));
    // A GPU write through the shared view makes the object's texture the GPU's: title memory
    // is not uploaded over it again.
    let uploads = with_title(|t| {
        let jobs = t.jobs(vec![object as usize]);
        t.sync(jobs);
        t.gpu_written(key);
        t.uploads_of(object as usize)
    })
    .unwrap();
    with_title(|_| pool.write(0, &[0x55u8; 32 * 32 * 4])).unwrap();
    run_title_list(crate::list::test_list(false, std::ptr::null()));
    with_title(|t| {
        let jobs = t.jobs(vec![object as usize]);
        t.sync(jobs);
        assert_eq!(t.uploads_of(object as usize), uploads, "the object is not refilled");
    });
    glue::resource_released(object);
    with_title(|t| assert!(!t.host.knows(key), "released with the object"));
}

/// Compute writes a depth texture through unordered views of its depth plane (the lighting's
/// material depth), which Metal shaders cannot write: the view shows a colour twin. A colour read
/// made before the first unordered view stays on the depth texture; one made after reads the twin.
/// The newer contents move where the use changes: into the twin before compute, into the depth
/// texture before a render pass into it, once per change.
#[test]
fn an_unordered_view_of_a_depth_texture_writes_a_colour_twin() {
    let Some(pool) = pool(9216, 8) else { return };
    // DXGI 19 (D32_FLOAT_S8X24_UINT); flags 2 | 4: a depth target compute also writes.
    let depth = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 19, Swizzle::LINEAR.0, 2 | 4);
    let heap = crate::descriptor::test_heap(false, 4);
    let start = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    // R32_FLOAT, 2D, mip 0.
    let srv = [41u32, 4, 5768, 0, 0, 1, 0, 0, 0, 0];
    let uav = [41u32, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    crate::descriptor::create_srv(std::ptr::null(), depth, &srv, start + 64);
    with_title(|t| {
        let tex = t.texture_of(depth as usize).expect("the depth texture");
        t.host.heap(HeapKind::Resource, start, 4).unwrap();
        assert_eq!(t.host.mirror(start + 64), Ok(true));
        assert!(t.host.twin_of(&tex).is_none(), "a colour read alone makes no twin");
    });
    crate::descriptor::create_uav(std::ptr::null(), depth, std::ptr::null(), &uav, start);
    with_title(|t| {
        let tex = t.texture_of(depth as usize).unwrap();
        assert_eq!(t.host.mirror(start), Ok(true));
        let twin = t.host.twin_of(&tex).expect("an unordered view of a depth texture makes a colour twin");
        assert!(!std::rc::Rc::ptr_eq(&twin, &tex));
        assert_eq!(t.host.twin_copies_for_test(false, &tex), Vec::<bool>::new(), "the depth texture is newer");
        assert_eq!(t.host.twin_copies_for_test(true, &tex), [false], "into the twin before compute");
        assert_eq!(t.host.twin_copies_for_test(true, &tex), Vec::<bool>::new(), "the twin is newer already");
        assert_eq!(t.host.twin_copies_for_test(false, &tex), [true], "into the depth texture before it is drawn into");
        assert_eq!(t.host.twin_copies_for_test(false, &tex), Vec::<bool>::new(), "the depth texture is newer now");
    });
    // A colour read written after the twin exists reads the twin.
    crate::descriptor::create_srv(std::ptr::null(), depth, &srv, start + 32);
    with_title(|t| assert_eq!(t.host.mirror(start + 32), Ok(true)));
    glue::resource_released(depth);
}

/// The console's 16-bit depth with stencil (DXGI 118, the lighting's material depth), which the
/// host keeps as 32-bit float depth with stencil, is written through `R16_UNORM` views of its
/// depth plane: they get a 32-bit float twin. A 16-bit depth texture (the shadow maps: DXGI 53
/// as a depth target) gets a 16-bit twin. What the twin holds goes into the depth plane and back
/// unchanged, bit for bit.
#[test]
fn a_depth_twin_and_its_depth_plane_hold_the_same_texels() {
    let Some(pool) = pool(9218, 16) else { return };
    let heap = crate::descriptor::test_heap(false, 4);
    let start = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    with_title(|t| {
        t.host.heap(HeapKind::Resource, start, 4).unwrap();
    });
    // (DXGI format, bytes a texel of the twin); R16_UNORM views (56), 2D, mip 0.
    for (n, (format, texel)) in [(118u32, 4usize), (53, 2)].into_iter().enumerate() {
        let depth = crate::resource::test_texture_at(pool.va(n * 65536), (16, 16), 1, format, Swizzle::LINEAR.0, 2 | 4);
        crate::descriptor::create_uav(std::ptr::null(), depth, std::ptr::null(), &[56u32, 4, 0, 0, 0, 0, 0, 0, 0, 0], start + 32 * n as u64);
        with_title(|t| {
            let tex = t.texture_of(depth as usize).expect("the depth texture");
            assert_eq!(t.host.mirror(start + 32 * n as u64), Ok(true), "{format}");
            let twin = t.host.twin_of(&tex).expect("an R16_UNORM view of the depth plane makes the twin");
            // The twin as compute leaves it, after it took the depth texture's contents.
            assert_eq!(t.host.run_twin_copies_for_test(true, &tex), 1, "{format}: into the twin before compute");
            let written: Vec<u8> = match texel {
                4 => (0..256u32).flat_map(|i| (i as f32 / 65535.0).to_le_bytes()).collect(),
                _ => (0..256u32).flat_map(|i| (i as u16 * 251).to_le_bytes()).collect(),
            };
            t.host.backend.upload_texture(&twin, 0, 0, &written, 16 * texel, written.len());
            assert_eq!(t.host.run_twin_copies_for_test(false, &tex), 1, "{format}: into the depth plane before a draw");
            // The twin is wiped, then takes the depth plane back before the next compute.
            t.host.backend.upload_texture(&twin, 0, 0, &vec![0u8; written.len()], 16 * texel, written.len());
            assert_eq!(t.host.run_twin_copies_for_test(true, &tex), 1, "{format}: into the twin again");
            let mut out = vec![0u8; written.len()];
            t.host.backend.read_texture(&twin, 0, 0, &mut out, 16 * texel, 0);
            assert!(out == written, "{format}: the depth plane did not hold what the twin had");
        });
        glue::resource_released(depth);
    }
}


/// A mip-streamed 64 KiB XOR-tiled texture placed in an unmapped reservation (`NO_ACCESS`) and
/// populated step by step across scattered 64 KiB physical pages (tail mips first, then mid mips,
/// then mip 0): each residency step is caught on the next submission by the mip-aware probes
/// without forcing immediate unchanged whole-texture re-reads while fine mips remain unmapped.
#[test]
fn streamed_mips_across_scattered_pool_pages_upload_each_step_without_whole_rereads() {
    use victory_platform::mem::{Protect, protect};

    let Ok(metal) = victory_metal::metal::Metal::new() else { return };
    let mut t = Title::new(metal, Default::default());
    const TILE: usize = 64 << 10;
    const POOL_TILES: usize = 32;
    let mem = Alias::new(POOL_TILES * TILE).unwrap();
    let pool_info = PoolInfo {
        id: 9290,
        base: mem.primary as usize,
        len: mem.len,
        memory_entry: None,
    };
    t.host.memory().pool_created(&pool_info);

    // 512x512 RGBA8, 10 mips, SW_64KB_D_X: 1,441,792 bytes (22 x 64 KiB pages):
    //   [0 .. 64 KiB)       = tail mips 3..9
    //   [64 .. 128 KiB)     = mip 2 (128x128)
    //   [128 .. 384 KiB)    = mip 1 (256x256, 4 pages)
    //   [384 .. 1408 KiB)   = mip 0 (512x512, 16 pages)
    let proto = crate::resource::test_texture_at(mem.alias as u64, (512, 512), 10, 28, Swizzle::SW_64KB_D_X.0, 0);
    let mut info = crate::resource::texture_info(proto).unwrap();
    let total_tiles = info.size / TILE;
    assert_eq!(total_tiles, 22);

    let px = |mip: u32, x: u32, y: u32| [x as u8, y as u8, (mip as u8).wrapping_mul(23).wrapping_add(17), 255];
    let mut tiled_full = vec![0u8; info.size];
    for mip in 0..10u32 {
        let m = &info.layout.mips[mip as usize];
        let linear: Vec<u8> = (0..m.height)
            .flat_map(|y| (0..m.width).flat_map(move |x| px(mip, x, y)))
            .collect();
        info.layout
            .tile(&mut tiled_full, mip, 0, &linear, m.width as usize * 4, linear.len())
            .unwrap();
    }

    let mut va = 0u64;
    let task = unsafe { super::mach_task_self() };
    assert_eq!(unsafe { super::mach_vm_allocate(task, &mut va, info.size as u64, 1) }, 0);
    // SAFETY: `va..va + info.size` is our freshly allocated reservation.
    unsafe { protect(va as *mut u8, info.size, Protect::NO_ACCESS) }.unwrap();
    info.memory = va as *const u8;
    let key = 0x7E57_0179usize;

    let map_tile = |t: &Title<victory_metal::metal::Metal>, title_idx: usize, pool_idx: usize| {
        let pool_off = pool_idx * TILE;
        let title_off = title_idx * TILE;
        // SAFETY: `pool_off..+TILE` is inside `mem.primary` and `va + title_off..+TILE` is inside `va`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                tiled_full[title_off..title_off + TILE].as_ptr(),
                mem.primary.add(pool_off),
                TILE,
            );
            let (mut at, mut cur, mut max) = (va + title_off as u64, 0, 0);
            let r = super::mach_vm_remap(
                task,
                &mut at,
                TILE as u64,
                0,
                0x4000,
                task,
                mem.primary as u64 + pool_off as u64,
                0,
                &mut cur,
                &mut max,
                0,
            );
            assert_eq!(r, 0, "mach_vm_remap tile {title_idx} -> {pool_idx}");
        }
        t.host.memory().mapped(va as usize + title_off, TILE, &pool_info, pool_off);
    };

    let check_mip = |t: &mut Title<victory_metal::metal::Metal>, mip: u32, expect_zero: bool| {
        let (w, h) = ((512usize >> mip).max(1), (512usize >> mip).max(1));
        let tex = t.texture_of(key).unwrap();
        let mut out = vec![0xffu8; w * h * 4];
        t.host.backend.read_texture(&tex, mip as usize, 0, &mut out, w * 4, 0);
        if expect_zero {
            assert!(out.iter().all(|&b| b == 0), "unmapped mip {mip} must stay zero");
        } else {
            for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w / 2, h / 2)] {
                let p = (y * w + x) * 4;
                assert_eq!(&out[p..p + 4], &px(mip, x as u32, y as u32), "mip {mip} ({x},{y})");
            }
        }
    };

    // Hold the shared title lock across all steps so no parallel test drains the process-wide
    // `WATCH.dirty` set while this test's watched texture is live (`5ba3cce6`).
    with_title(|_shared| {
        // Step 0: Completely unmapped reservation (`blank` first upload).
        t.texture_created(key, info);
        assert!(t.realize(key));
        t.sync_dirty_without_probe_deadline();
        assert_eq!(t.uploads_of(key), 1, "initial blank upload");
        check_mip(&mut t, 0, true);
        check_mip(&mut t, 4, true);

        // Step 1: Stream tail mips 3..9 (`[0..64 KiB)`) into scattered pool page 29 while mips 0..2
        // remain `NO_ACCESS`. `probe_nonzero` must catch it on the very next submission.
        map_tile(&t, 0, 29);
        t.sync_dirty_without_probe_deadline();
        assert_eq!(t.uploads_of(key), 2, "tail mips uploaded on next submission");
        for mip in 3..10 {
            check_mip(&mut t, mip, false);
        }
        for mip in 0..3 {
            check_mip(&mut t, mip, true);
        }

        // Step 2: Stream mip 2 (`[64..128 KiB)`) into scattered pool page 7 while mips 0..1 stay
        // `NO_ACCESS`. Mip-aware `probe_read` must catch it immediately AND keep `sampled_sees_writes`
        // true so the next `WHOLE_EVERY - 1` idle submissions do 0 whole re-reads.
        map_tile(&t, 1, 7);
        t.sync_dirty_without_probe_deadline();
        assert_eq!(t.uploads_of(key), 3, "mip 2 uploaded on next submission");
        check_mip(&mut t, 2, false);
        check_mip(&mut t, 1, true);
        check_mip(&mut t, 0, true);
        for sub in 0..crate::native::textures::WHOLE_EVERY - 1 {
            let jobs = t.sync_jobs_within(None);
            assert!(
                !jobs.iter().any(|j| j.key == key),
                "unchanged partial texture re-read whole at idle submission {sub}"
            );
            t.sync(jobs);
        }

        // Step 3: Stream mip 1 (`[128..384 KiB)`) across 4 scattered out-of-order pool pages.
        for (k, &pool_idx) in [21usize, 3, 18, 11].iter().enumerate() {
            map_tile(&t, 2 + k, pool_idx);
        }
        t.sync_dirty_without_probe_deadline();
        assert_eq!(t.uploads_of(key), 4, "mip 1 uploaded on next submission");
        check_mip(&mut t, 1, false);
        check_mip(&mut t, 0, true);
        for sub in 0..crate::native::textures::WHOLE_EVERY - 1 {
            let jobs = t.sync_jobs_within(None);
            assert!(!jobs.iter().any(|j| j.key == key), "idle submission {sub} after mip 1");
            t.sync(jobs);
        }

        // Step 4: Stream mip 0 (`[384..1408 KiB)`) across 16 scattered pool pages.
        let mip0_pages = [30usize, 0, 27, 4, 25, 1, 23, 6, 20, 8, 17, 9, 15, 10, 14, 12];
        for (k, &pool_idx) in mip0_pages.iter().enumerate() {
            map_tile(&t, 6 + k, pool_idx);
        }
        t.sync_dirty_without_probe_deadline();
        assert_eq!(t.uploads_of(key), 5, "mip 0 uploaded on next submission");
        check_mip(&mut t, 0, false);

        t.texture_released(key);
    });
    t.host.memory().pool_released(&pool_info);
    // SAFETY: Release our test reservation before dropping `mem`.
    assert_eq!(unsafe { super::mach_vm_deallocate(task, va, (total_tiles * TILE) as u64) }, 0);
}

#[test]
#[ignore = "benchmark"]
fn bench_streamed_texture_sync_pool_reads_and_mip_probes() {
    use std::time::Instant;
    use victory_platform::mem::{Protect, next_region, page_size, protect};

    let Ok(metal) = victory_metal::metal::Metal::new() else { return };
    let mut t = Title::new(metal, Default::default());
    const N: usize = 32;
    const TILE: usize = 64 << 10;
    const TILES_PER_TEX: usize = 22; // 512x512 RGBA8, 10 mips, SW_64KB_D_X = 1.375 MiB
    const RESIDENT_TILES: usize = 6; // mips 1..9 (384 KiB) mapped; mip 0 (1 MiB) NO_ACCESS

    let mem = Alias::new(N * RESIDENT_TILES * TILE).unwrap();
    let pool_info = PoolInfo {
        id: 9299,
        base: mem.primary as usize,
        len: mem.len,
        memory_entry: None,
    };
    t.host.memory().pool_created(&pool_info);

    let proto = crate::resource::test_texture_at(mem.alias as u64, (512, 512), 10, 28, Swizzle::SW_64KB_D_X.0, 0);
    let base_info = crate::resource::texture_info(proto).unwrap();
    assert_eq!(base_info.size, TILES_PER_TEX * TILE);

    let mut tiled_template = vec![0u8; base_info.size];
    for mip in 1..10u32 {
        let m = &base_info.layout.mips[mip as usize];
        let linear = vec![0x40u8 + mip as u8 * 9; (m.width * m.height * 4) as usize];
        base_info
            .layout
            .tile(&mut tiled_template, mip, 0, &linear, m.width as usize * 4, linear.len())
            .unwrap();
    }

    let task = unsafe { super::mach_task_self() };
    let mut va = 0u64;
    let total_va = N * TILES_PER_TEX * TILE;
    assert_eq!(unsafe { super::mach_vm_allocate(task, &mut va, total_va as u64, 1) }, 0);
    unsafe { protect(va as *mut u8, total_va, Protect::NO_ACCESS) }.unwrap();

    // Realize all 32 textures blank first.
    let keys: Vec<usize> = (0..N)
        .map(|i| {
            let key = 0xBEEF_0000 + i;
            let mut info = base_info.clone();
            info.memory = (va as usize + i * TILES_PER_TEX * TILE) as *const u8;
            t.texture_created(key, info);
            assert!(t.realize(key));
            key
        })
        .collect();

    let t_blank = Instant::now();
    t.sync_dirty();
    let blank_ms = t_blank.elapsed().as_secs_f64() * 1000.0;

    // Map 6 scattered 64 KiB physical pages per texture (reverse pool order so adjacent pages are
    // separate `vm_map_entry` regions in XNU) while leaving mip 0 (`1 MiB`) `NO_ACCESS`.
    for i in 0..N {
        let tex_va = va as usize + i * TILES_PER_TEX * TILE;
        for r in 0..RESIDENT_TILES {
            let pool_slot = (N * RESIDENT_TILES - 1) - (i * RESIDENT_TILES + r);
            let pool_off = pool_slot * TILE;
            let src_off = r * TILE;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tiled_template[src_off..src_off + TILE].as_ptr(),
                    mem.primary.add(pool_off),
                    TILE,
                );
                // Ensure each texture has a unique content hash.
                *mem.primary.add(pool_off + 0x800) ^= (i as u8).wrapping_add(1);
                let (mut at, mut cur, mut max) = ((tex_va + src_off) as u64, 0, 0);
                assert_eq!(
                    super::mach_vm_remap(
                        task,
                        &mut at,
                        TILE as u64,
                        0,
                        0x4000,
                        task,
                        mem.primary as u64 + pool_off as u64,
                        0,
                        &mut cur,
                        &mut max,
                        0,
                    ),
                    0
                );
            }
            t.host.memory().mapped(tex_va + src_off, TILE, &pool_info, pool_off);
        }
    }

    // 1. Measure old page-by-page VM region walk vs `PoolTable::read_mapped_with` across 10 rounds
    //    (320 partial texture reads = 440 MiB logical).
    let mut buf = vec![0u8; base_info.size];
    let page = page_size();
    let mut old_syscalls = 0usize;
    let t_old_read = Instant::now();
    for _ in 0..10 {
        for i in 0..N {
            let base = va as usize + i * TILES_PER_TEX * TILE;
            let end = base + base_info.size;
            let mut cur = base;
            while cur < end {
                old_syscalls += 1;
                let Some(region) = next_region(cur) else { break };
                let start = cur.max(region.base);
                let stop = end.min(region.base.saturating_add(region.len));
                if start >= stop {
                    break;
                }
                if region.protect.0 & Protect::READ.0 == 0 {
                    buf[start - base..stop - base].fill(0);
                } else {
                    old_syscalls += 1;
                    let _ = crate::native::safe_read(start as *const u8, &mut buf[start - base..stop - base]);
                }
                cur = stop;
            }
            let _ = page;
            std::hint::black_box(&buf[0]);
        }
    }
    let old_read_ms = t_old_read.elapsed().as_secs_f64() * 1000.0;

    let pools = t.host.memory();
    let mut new_syscalls = 0usize;
    let t_new_read = Instant::now();
    for _ in 0..10 {
        for i in 0..N {
            let base = va as usize + i * TILES_PER_TEX * TILE;
            let _ = pools.read_mapped_with(base, &mut buf, |span_at, dst| {
                new_syscalls += 1;
                crate::native::safe_read(span_at as *const u8, dst)
            });
            std::hint::black_box(&buf[0]);
        }
    }
    let new_read_ms = t_new_read.elapsed().as_secs_f64() * 1000.0;

    // 2. Sync all 32 partial textures (probes + prepare + upload, skipping unmapped 1 MiB mip 0),
    //    then measure whole-read volume across 30 idle submissions (~1 world frame).
    let t_stream_sync = Instant::now();
    t.sync_dirty_without_probe_deadline();
    let stream_sync_ms = t_stream_sync.elapsed().as_secs_f64() * 1000.0;
    assert!(keys.iter().all(|&k| t.uploads_of(k) == 2));

    let mut idle_whole_jobs = 0usize;
    let mut idle_whole_bytes = 0usize;
    let t_idle = Instant::now();
    for _ in 0..30 {
        let jobs = t.sync_jobs();
        idle_whole_jobs += jobs.len();
        idle_whole_bytes += jobs.iter().map(|_| base_info.size).sum::<usize>();
        t.sync(jobs);
    }
    let idle_ms = t_idle.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "BENCH #179 ({N}x 1.375 MiB partial streamed textures, 6 scattered 64K pages + 1 MiB unmapped mip 0):\n\
         - initial blank sync ({N} textures): {blank_ms:.2} ms\n\
         - 320 partial reads (440 MiB): page-by-page VM walk {old_read_ms:.2} ms ({old_syscalls} kernel calls) -> PoolTable coalesced {new_read_ms:.2} ms ({new_syscalls} kernel calls, {:.2}x speedup)\n\
         - streamed mips 1..9 sync ({N} textures, mip 0 skipped): {stream_sync_ms:.2} ms\n\
         - 30 idle submissions (~1 world frame): {idle_whole_jobs} whole checks ({:.2} MiB, {idle_ms:.2} ms total; vs ~240 MiB before mip-aware probes)",
        old_read_ms / new_read_ms.max(1e-6),
        idle_whole_bytes as f64 / (1024.0 * 1024.0),
    );

    for k in keys {
        t.texture_released(k);
    }
    t.host.memory().pool_released(&pool_info);
    assert_eq!(unsafe { super::mach_vm_deallocate(task, va, total_va as u64) }, 0);
}


/// Over the texture budget, the least recently used go first, only until the excess is freed.
#[test]
fn the_least_recently_used_textures_go_first_until_the_excess_is_freed() {
    use crate::native::textures::choose_evictions;
    // (key, bytes, last used)
    let c = vec![(1, 100, 5), (2, 100, 1), (3, 100, 3), (4, 100, 3)];
    assert_eq!(choose_evictions(c.clone(), 0), Vec::<usize>::new());
    assert_eq!(choose_evictions(c.clone(), 1), vec![2]);
    assert_eq!(choose_evictions(c.clone(), 150), vec![2, 3], "equal use: the lower key first");
    assert_eq!(choose_evictions(c.clone(), 300), vec![2, 3, 4]);
    assert_eq!(choose_evictions(c, 10_000), vec![2, 3, 4, 1]);
}

/// The budget is off unless `VICTORY_METAL_TEXTURE_BUDGET_MB` sets it; the figure suggested for a
/// Mac is what Metal's working set leaves after the title's memory, with a floor.
#[test]
fn the_texture_budget_is_off_unless_set() {
    use crate::native::textures::{MIN_TEXTURE_BUDGET, suggested_texture_budget, texture_budget};
    const MIB: u64 = 1 << 20;
    assert_eq!(texture_budget(None), 0, "off by default");
    assert_eq!(texture_budget(Some("2048")), 2048 * MIB);
    assert_eq!(texture_budget(Some("0")), 0, "0: no budget");
    assert_eq!(texture_budget(Some("lots")), 0, "not a number: no budget");
    // An M4 with 16 GB, Series S (#8).
    assert_eq!(suggested_texture_budget(12124 * MIB, 7776 * MIB), 4348 * MIB);
    // Series X on the same Mac: the title alone is past the working set.
    assert_eq!(suggested_texture_budget(12124 * MIB, 13824 * MIB), MIN_TEXTURE_BUDGET);
}

/// A title host of its own over a pool of its own, and a 16x16 RGBA8 texture of `flags` at `at`
/// in it, filled with `byte` and made.
fn budget_title() -> Option<(Title<victory_metal::metal::Metal>, Pool)> {
    with_title(|_| ())?;
    let t = Title::new(victory_metal::metal::Metal::new().unwrap(), Default::default());
    let mem = Alias::new((256 << 10).max(page_size())).unwrap();
    let info = PoolInfo { id: 9141, base: mem.primary as usize, len: mem.len, memory_entry: None };
    t.host.memory().pool_created(&info);
    t.host.memory().mapped(mem.alias as usize, mem.len, &info, 0);
    Some((t, Pool { mem }))
}

fn budget_texture(t: &mut Title<victory_metal::metal::Metal>, pool: &Pool, at: usize, flags: u32, byte: u8) -> usize {
    let res = crate::resource::test_texture_at(pool.va(at), (16, 16), 1, 28, Swizzle::LINEAR.0, flags);
    let info = crate::resource::texture_info(res).unwrap();
    pool.write(at, &vec![byte; info.size]);
    let key = res as usize;
    t.texture_created(key, info);
    assert!(t.realize(key));
    key
}

/// Over budget, only textures no heap entry shows, no recorded or submitted work holds, and that
/// are filled only from title memory are given up; one named again is made and filled again.
#[test]
fn over_budget_only_textures_nothing_uses_are_given_up_and_come_back() {
    let Some((mut t, pool)) = budget_title() else { return };
    let idle = budget_texture(&mut t, &pool, 0, 0, 0x11);
    let shown = budget_texture(&mut t, &pool, 0x4000, 0, 0x22);
    let held = budget_texture(&mut t, &pool, 0x8000, 0, 0x33);
    // D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ALLOW_UNORDERED_ACCESS.
    let target = budget_texture(&mut t, &pool, 0xC000, 1, 0x44);
    let unordered = budget_texture(&mut t, &pool, 0x10000, 4, 0x55);
    t.sync_textures();
    // A heap entry shows `shown`; one showed `unordered` through an unordered access view.
    let records = [0u32; 8 * 2];
    let start = records.as_ptr() as u64;
    let _heap = t.host.heap(HeapKind::Resource, start, 2).unwrap();
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null(), shown as _, &srv, start);
    assert!(t.host.mirror(start).unwrap());
    let uav = [28u32, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    crate::descriptor::create_uav(std::ptr::null(), unordered as _, std::ptr::null(), &uav, start + 32);
    assert!(t.host.mirror(start + 32).unwrap());
    crate::descriptor::create_srv(std::ptr::null(), shown as _, &srv, start + 32);
    assert!(t.host.mirror(start + 32).unwrap(), "the unordered view is gone from the heap");
    // A recorded list copies from `held` (work the GPU has not run yet).
    let mut list = NativeList::<victory_metal::metal::Metal>::default();
    let src = t.texture_of(held).unwrap();
    let dst = t.texture_of(target).unwrap();
    list.copy_texture_region(&src, 0, 0, [0; 3], [16, 16, 1], &dst, 0, 0, [0; 3]);
    drop((src, dst));

    t.set_texture_budget(1);
    t.keep_to_budget();
    assert!(!t.host.knows(idle), "the one texture nothing uses is given up");
    for key in [shown, held, target, unordered] {
        assert!(t.host.knows(key), "{key:#x} is in use");
    }
    drop(list);
    // Still over budget after that check, so the next one waits a while.
    t.keep_to_budget();
    assert!(t.host.knows(held), "not checked again at once");
    for _ in 0..crate::native::textures::RECHECK {
        t.keep_to_budget();
    }
    assert!(!t.host.knows(held), "once the work that held it is gone");

    // Named again (a view, a copy): made again and filled from title memory at the next sync.
    assert!(t.realize(idle));
    t.sync_textures();
    let tex = t.texture_of(idle).unwrap();
    let mut out = vec![0u8; 16 * 16 * 4];
    t.host.backend.read_texture(&tex, 0, 0, &mut out, 16 * 4, 0);
    assert!(out.iter().all(|&b| b == 0x11));
}

/// The fallback when a record naming an evicted texture is mirrored outside the scan of a
/// submission's tables ([`Title::keep_named`]): the entry asks for the texture, the next submission
/// makes it again and the entry is written then. Until then the entry shows nothing (a draw would
/// sample transparent black), not what it showed before.
#[test]
fn an_entry_copied_with_plain_stores_gets_an_evicted_texture_back() {
    let Some((mut t, pool)) = budget_title() else { return };
    let gone = budget_texture(&mut t, &pool, 0, 0, 0x66);
    let other = budget_texture(&mut t, &pool, 0x4000, 0, 0x77);
    let records = [0u32; 8];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 1).unwrap();
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null(), other as _, &srv, start);
    assert!(t.host.mirror(start).unwrap());
    t.set_texture_budget(1);
    t.keep_to_budget();
    assert!(!t.host.knows(gone) && t.host.knows(other));

    // The title's record now names the evicted texture.
    crate::descriptor::create_srv(std::ptr::null(), gone as _, &srv, start);
    assert!(!t.host.mirror(start).unwrap(), "not made yet");
    assert!(!heap.shows_texture(other), "the entry no longer shows the texture it showed");
    t.set_texture_budget(0);
    t.sync_dirty();
    assert!(t.host.knows(gone), "made again at the next submission");
    assert!(t.host.mirror(start).unwrap());
    assert!(heap.shows_texture(gone));
}

/// Draws a triangle covering the centre of a 16x16 target, red vertex colour times texture `key`
/// sampled through entry 0 of a heap whose record the title stored with plain stores (no device
/// call, so the host has not mirrored it), in one submission of `t`; the centre pixel.
fn draw_sampling(t: &mut Title<victory_metal::metal::Metal>, pool: &Pool, key: usize, queue: usize) -> [u8; 4] {
    let (l, color, records) = sampling_list(t, pool, key);
    let fence = queue + 100;
    t.ensure_queue(queue).unwrap();
    t.submit(queue, vec![(l, vec![])]);
    t.signal_queue(queue, fence, 0, 1);
    let start = std::time::Instant::now();
    while !t.try_wait(fence, 0, 1) {
        assert!(start.elapsed().as_secs() < 10, "the draw did not complete");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let mut out = vec![0u8; SAMPLED * SAMPLED * 4];
    t.host.backend.read_texture(&color, 0, 0, &mut out, SAMPLED * 4, 0);
    drop(records);
    out[(8 * SAMPLED + 8) * 4..][..4].try_into().unwrap()
}

const SAMPLED: usize = 16;

/// The list [`draw_sampling`] submits, its target, and the heaps' records (kept while it is used).
#[allow(clippy::type_complexity)]
fn sampling_list(
    t: &mut Title<victory_metal::metal::Metal>,
    pool: &Pool,
    key: usize,
) -> (NativeList<victory_metal::metal::Metal>, std::rc::Rc<<victory_metal::metal::Metal as ResourceBackend>::Texture>, (Vec<u32>, Vec<u32>)) {
    const W: usize = SAMPLED;
    let v = |x: f32, y: f32| [x, y, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0];
    pool.write(0x20000, &[v(-0.5, -0.5), v(0.5, -0.5), v(0.0, 0.5)]);
    let attrs = [
        VertexAttr { format: VertexFormat::Float2, slot: 0, offset: 0 },
        VertexAttr { format: VertexFormat::Float2, slot: 0, offset: 8 },
        VertexAttr { format: VertexFormat::Float4, slot: 0, offset: 16 },
    ];
    let pso = t
        .host
        .render_pipeline(&RenderPipelineDesc {
            vs: &super::fixture("tri_vs"),
            ps: Some(&super::fixture("tri_ps")),
            vs_layout: None,
            ps_layout: None,
            attributes: &attrs,
            streams: &[VertexStream { slot: 0, stride: 32, per_instance: false }],
            color: &[Format::Rgba8Unorm],
            depth: None,
            depth_test: None,
            cull: Cull::None,
            ..Default::default()
        })
        .unwrap();
    let mut rt = TextureDesc::d2(Format::Rgba8Unorm, W, W);
    rt.render_target = true;
    let color = t.host.texture(&rt).unwrap();
    let (views, smp) = (vec![0u32; 8], vec![0u32; 4]);
    let (vstart, sstart) = (views.as_ptr() as u64, smp.as_ptr() as u64);
    let vheap = t.host.heap(HeapKind::Resource, vstart, 1).unwrap();
    let sheap = t.host.heap(HeapKind::Sampler, sstart, 1).unwrap();
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null(), key as _, &srv, vstart);
    let mut sw = [0u32; 27];
    (sw[1], sw[2], sw[6], sw[7], sw[8]) = (1, 1, 3, 3, 3);
    crate::descriptor::create_sampler_x(std::ptr::null_mut(), &sw, sstart);
    assert!(t.host.mirror(sstart).unwrap());

    let mut l = NativeList::default();
    l.set_render_targets(std::slice::from_ref(&color), None);
    l.clear_render_target(&color, [0.0, 0.0, 1.0, 1.0]);
    l.set_graphics_pipeline(&pso, Topology::Triangles);
    l.set_heaps(Some(&vheap), Some(&sheap));
    l.set_graphics_root_constants(0, &[0.5f32.to_bits(), 0, 0, 0]);
    l.set_graphics_root_table(1, &vheap, 0);
    l.set_graphics_root_table(2, &sheap, 0);
    l.set_vertex_buffer(0, pool.va(0x20000));
    l.draw_instanced(3, 1, 0, 0).unwrap();
    (l, color, (views, smp))
}

/// Over the texture budget, a draw samples a texture given up before through an entry the title
/// stored with plain stores in the table it binds: the texture is made again for that draw, in
/// that submission (it sampled transparent black, the entry cleared, until the next one).
#[test]
fn a_texture_given_up_is_back_for_the_draw_whose_table_names_it() {
    let Some((mut t, pool)) = budget_title() else { return };
    let gone = budget_texture(&mut t, &pool, 0, 0, 0x66);
    t.set_texture_budget(1);
    t.keep_to_budget();
    assert!(!t.host.knows(gone));
    assert_eq!(draw_sampling(&mut t, &pool, gone, 1), [0x66, 0, 0, 0x66]);
    assert!(t.host.knows(gone));
}

/// Over the texture budget, a texture no entry the host mirrored shows, which the table of a draw
/// in the submission names through a record the title stored with plain stores, is not given up
/// at that submission's sync, under the draw.
#[test]
fn a_texture_the_submission_names_is_not_given_up_under_its_draw() {
    let Some((mut t, pool)) = budget_title() else { return };
    let live = budget_texture(&mut t, &pool, 0, 0, 0x66);
    t.set_texture_budget(1);
    assert_eq!(draw_sampling(&mut t, &pool, live, 2), [0x66, 0, 0, 0x66]);
    assert!(t.host.knows(live));
}

/// Names a submission's tables read stay out of the budget's reach until that submission is with
/// its queue, even when another submission's sync comes in between (`glue::submit` lets the title
/// lock go between the two).
#[test]
fn a_texture_named_by_a_submission_in_flight_is_not_given_up_by_another() {
    let Some((mut t, pool)) = budget_title() else { return };
    let named = budget_texture(&mut t, &pool, 0, 0, 0x66);
    let idle = budget_texture(&mut t, &pool, 0x4000, 0, 0x77);
    t.set_texture_budget(1);
    let (a, _color, records) = sampling_list(&mut t, &pool, named);
    let first = t.keep_named(std::iter::once(&a));
    assert!(first.contains(&named));
    // Another queue's submission, with nothing bound, synced in between.
    let b = NativeList::<victory_metal::metal::Metal>::default();
    let second = t.keep_named(std::iter::once(&b));
    let _ = t.sync_jobs();
    assert!(!t.host.knows(idle), "over budget: the idle one goes");
    assert!(t.host.knows(named), "the first submission still names it");
    t.let_named_go(second);
    for _ in 0..crate::native::textures::RECHECK {
        t.keep_to_budget();
    }
    assert!(t.host.knows(named), "still in flight");
    t.let_named_go(first);
    for _ in 0..crate::native::textures::RECHECK {
        t.keep_to_budget();
    }
    assert!(!t.host.knows(named), "given up once nothing in flight names it");
    drop((a, records));
}

/// The budget counts a copy by its size in title memory, once per title texture: views of one
/// object, in other formats or of other mips, name the object, and share its one host texture.
#[test]
fn views_of_one_texture_share_one_copy_counted_once() {
    use crate::descriptor::View;
    let Some((mut t, pool)) = budget_title() else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (32, 32), 6, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    t.texture_created(key, info.clone());
    // Two views of it: RGBA8 UNORM, all mips; RGBA8 UNORM sRGB (29), mip 2 only.
    let heap = crate::descriptor::test_heap(false, 4);
    let start = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[2], srv[5]) = (28, 4, 0x1688, u32::MAX);
    crate::descriptor::create_srv(std::ptr::null(), res, &srv, start);
    (srv[0], srv[4], srv[5]) = (29, 2, 1);
    crate::descriptor::create_srv(std::ptr::null(), res, &srv, start + 32);
    for handle in [start, start + 32] {
        let Some(View::Shader(r, _)) = crate::descriptor::view(handle) else { panic!("a view") };
        assert_eq!(r, key, "the view names the object");
        assert!(t.realize(r));
    }
    let a = t.texture_of(key).unwrap();
    assert!(std::rc::Rc::ptr_eq(&a, &t.texture_of(key).unwrap()), "one host texture");
    assert_eq!(t.budget.live, info.size as u64, "counted once, all mips");
}

/// Views of memory the title placed itself are keyed by address and resource description, not
/// by the view's own description: views of one placed texture share one copy.
#[test]
fn placed_views_of_one_description_share_one_key() {
    use crate::native::textures::placed_key;
    let res = crate::resource::test_texture_at(0x10_0000_0000, (32, 32), 6, 28, Swizzle::SW_64KB_S.0, 0);
    let words = crate::resource::texture_desc(res).unwrap();
    let key = placed_key(0x10_0000_0000, &words);
    // The key reads no view description at all; padding words do not change it either.
    let mut padded = words;
    (padded[1], padded[13]) = (0xdead, 0xbeef);
    assert_eq!(placed_key(0x10_0000_0000, &padded), key);
    let mut other = words;
    other[8] = 29; // another format
    assert_ne!(placed_key(0x10_0000_0000, &other), key, "another description: another copy");
    assert_ne!(placed_key(0x10_0001_0000, &words), key, "another address: another copy");
}

/// Copies are counted, and held, once per title texture, not once per byte of title memory: two
/// objects over the same memory (aliasing placed resources) and a placed description starting
/// inside another's memory each get a host texture of their own, so the copies add up to more
/// than the title memory under them. The trace's distinct figure counts that memory once.
#[test]
fn textures_over_one_memory_each_hold_a_copy_of_their_own() {
    use crate::native::textures::{distinct_bytes, placed_key};
    let Some((mut t, pool)) = budget_title() else { return };
    let a = crate::resource::test_texture_at(pool.va(0), (64, 64), 1, 28, Swizzle::LINEAR.0, 0);
    let b = crate::resource::test_texture_at(pool.va(0), (64, 64), 1, 71, Swizzle::LINEAR.0, 0);
    let (ia, ib) = (crate::resource::texture_info(a).unwrap(), crate::resource::texture_info(b).unwrap());
    // A description the title places at 4 KiB into the same memory (no object; a view names it).
    let c = crate::resource::test_texture_at(pool.va(0x1000), (32, 32), 1, 28, Swizzle::LINEAR.0, 0);
    let words = crate::resource::texture_desc(c).unwrap();
    let ic = crate::resource::placed_info(pool.va(0x1000), &words).unwrap();
    let kc = placed_key(pool.va(0x1000), &words);
    t.texture_placed(pool.va(0x1000), kc, 9141, 0x1000);
    for (key, info) in [(a as usize, ia.clone()), (b as usize, ib.clone()), (kc, ic.clone())] {
        t.texture_created(key, info);
        assert!(t.realize(key));
    }
    let (ta, tb, tc) = (t.texture_of(a as usize).unwrap(), t.texture_of(b as usize).unwrap(), t.texture_of(kc).unwrap());
    assert!(!std::rc::Rc::ptr_eq(&ta, &tb) && !std::rc::Rc::ptr_eq(&ta, &tc), "three host textures");
    let counted = (ia.size + ib.size + ic.size) as u64;
    assert_eq!(t.budget.live, counted);
    let under = distinct_bytes(vec![(ia.memory as usize, ia.size), (ib.memory as usize, ib.size), (ic.memory as usize, ic.size)]);
    assert_eq!(under, ia.size as u64, "all three lie in the first one's 16 KiB");
    assert!(counted > under);
    assert_eq!(t.memory_under_copies(), (under, 0), "what the trace logs");
    // The same pool pages mapped at a second title address: a texture there is over the same
    // bytes, and counts once too.
    let info = PoolInfo { id: 9141, base: pool.mem.primary as usize, len: pool.mem.len, memory_entry: None };
    t.host.memory().mapped(pool.mem.primary as usize, pool.mem.len, &info, 0);
    let d = crate::resource::test_texture_at(pool.mem.primary as u64, (64, 64), 1, 28, Swizzle::LINEAR.0, 0);
    let id = crate::resource::texture_info(d).unwrap();
    t.texture_created(d as usize, id.clone());
    assert!(t.realize(d as usize));
    assert_eq!(t.budget.live, counted + id.size as u64);
    assert_eq!(t.memory_under_copies(), (under, 0), "one memory at two addresses");
}

/// A copy over memory in no title pool is logged as outside them.
#[test]
fn a_copy_outside_the_titles_pools_is_logged_so() {
    let Some((mut t, _pool)) = budget_title() else { return };
    let own = vec![0u8; 16 * 16 * 4];
    let res = crate::resource::test_texture_at(own.as_ptr() as u64, (16, 16), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    t.texture_created(res as usize, info.clone());
    assert!(t.realize(res as usize));
    assert_eq!(t.memory_under_copies(), (info.size as u64, info.size as u64));
}

/// The trace's distinct title memory under copies: each byte once, gaps not counted.
#[test]
fn distinct_bytes_counts_overlaps_once() {
    use crate::native::textures::distinct_bytes;
    assert_eq!(distinct_bytes(vec![]), 0);
    assert_eq!(distinct_bytes(vec![(0, 100), (0, 100)]), 100);
    assert_eq!(distinct_bytes(vec![(50, 100), (0, 100)]), 150);
    assert_eq!(distinct_bytes(vec![(0, 100), (10, 20), (200, 10)]), 110);
    assert_eq!(distinct_bytes(vec![(0, 100), (100, 1)]), 101);
}

/// A 16x16 (`size`) RGBA8 texture the title places at `at` in the budget pool (no object; a view
/// names it), filled with `byte` and made.
fn budget_placed(t: &mut Title<victory_metal::metal::Metal>, pool: &Pool, at: usize, size: u32, byte: u8) -> usize {
    let res = crate::resource::test_texture_at(pool.va(at), (size.into(), size), 1, 28, Swizzle::LINEAR.0, 0);
    let words = crate::resource::texture_desc(res).unwrap();
    let info = crate::resource::placed_info(pool.va(at), &words).unwrap();
    pool.write(at, &vec![byte; info.size]);
    let key = crate::native::textures::placed_key(pool.va(at), &words);
    t.texture_placed(pool.va(at), key, 9141, info.size);
    t.texture_created(key, info);
    assert!(t.realize(key));
    key
}

/// A title texture laid out exactly as a live copy over the same memory (here a placed view of
/// an object's memory in the object's description) uses that copy's texture, counted once; one
/// in another host format gets its own. A GPU write through the sharing key makes the shared
/// copy the GPU's.
#[test]
fn an_identical_texture_over_the_same_memory_shares_its_copy() {
    let Some((mut t, pool)) = budget_title() else { return };
    let object = budget_texture(&mut t, &pool, 0, 0, 0x11);
    let size = t.budget.live;
    let placed = budget_placed(&mut t, &pool, 0, 16, 0x11);
    let own = t.texture_of(object).unwrap();
    assert!(std::rc::Rc::ptr_eq(&own, &t.texture_of(placed).unwrap()), "one texture");
    assert!(std::rc::Rc::ptr_eq(&own, &t.host.texture_at(placed).expect("registered")));
    assert_eq!(t.budget.live, size, "counted once");
    // B8G8R8A8 (87) over the same bytes: another host format, its own copy.
    let res = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 87, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    t.texture_created(res as usize, info);
    assert!(t.realize(res as usize));
    assert!(!std::rc::Rc::ptr_eq(&own, &t.texture_of(res as usize).unwrap()));
    assert_eq!(t.budget.live, 2 * size);
    // Uploaded once; a GPU write through the placed key keeps the object's copy from memory.
    t.sync_textures();
    assert_eq!(t.uploads_of(object), 1);
    t.gpu_written(placed);
    pool.write(0, &[0x22u8; 64]);
    t.sync_textures();
    assert_eq!(t.uploads_of(object), 1, "the GPU's now");
}

/// When the copy a texture shared goes with its title texture, the texture that shared it takes the
/// copy over: still registered, the same texture, not uploaded again, counted once; a draw reads it.
#[test]
fn a_texture_sharing_a_released_copy_takes_it_over() {
    let Some((mut t, pool)) = budget_title() else { return };
    let object = budget_texture(&mut t, &pool, 0, 0, 0x66);
    let size = t.budget.live;
    let placed = budget_placed(&mut t, &pool, 0, 16, 0x66);
    let copy = t.texture_of(object).unwrap();
    t.sync_textures();
    assert_eq!(t.uploads_of(placed), 1);
    t.host.release_resource(object);
    t.texture_released(object);
    assert!(t.host.knows(placed) && !t.host.is_evicted(placed));
    assert!(std::rc::Rc::ptr_eq(&copy, &t.texture_of(placed).unwrap()), "the same texture");
    assert_eq!(t.budget.live, size);
    t.sync_textures();
    assert_eq!(t.uploads_of(placed), 1, "not uploaded again");
    assert_eq!(draw_sampling(&mut t, &pool, placed, 3), [0x66, 0, 0, 0x66]);
}

/// Has the one entry of a host heap over `views` show title texture `key` (a mirrored view).
fn show(t: &mut Title<victory_metal::metal::Metal>, views: &[u32], key: usize) {
    let at = views.as_ptr() as u64;
    if t.host.heap_at(at).is_none() {
        t.host.heap(HeapKind::Resource, at, 1).unwrap();
    }
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null(), key as _, &srv, at);
    assert!(t.host.mirror(at).unwrap());
}

/// A placed copy another placement covers at another start is given up at the next sync once
/// nothing uses it; one a heap entry still shows stays until the title writes another record over
/// that entry (work in flight may read it through the entry). One named again comes back for its
/// draw, with its texels. One not overlapped stays, as does one that holds what title memory does
/// not (GPU-written). A view of a copy given up, written again at its own address, is not a new
/// placement and covers nothing.
#[test]
fn a_placed_copy_another_placement_covers_is_given_up_once_unused() {
    let Some((mut t, pool)) = budget_title() else { return };
    let old = budget_placed(&mut t, &pool, 0, 32, 0x44);
    let shown = budget_placed(&mut t, &pool, 0x4000, 32, 0x33);
    let apart = budget_placed(&mut t, &pool, 0x8000, 16, 0x55);
    let written = budget_placed(&mut t, &pool, 0xA000, 32, 0x77);
    t.gpu_written(written);
    let views = vec![0u32; 8];
    show(&mut t, &views, shown);
    let new = budget_placed(&mut t, &pool, 0x800, 16, 0x44);
    let _ = budget_placed(&mut t, &pool, 0x4800, 16, 0x33);
    let _ = budget_placed(&mut t, &pool, 0xA800, 16, 0x77);
    assert!(t.host.knows(old), "not at the placement itself");
    let _ = t.sync_jobs();
    assert!(!t.host.knows(old) && t.host.is_evicted(old), "covered and unused: given up");
    assert!(t.host.knows(shown), "an entry still shows it");
    assert!(t.host.knows(new) && t.host.knows(apart));
    assert!(t.host.knows(written), "the GPU's copy stays");
    assert!(!t.covered.contains(&written), "and is not even marked");
    show(&mut t, &views, apart);
    let _ = t.sync_jobs();
    assert!(!t.host.knows(shown), "given up once its entry shows another");
    // The old view written again: back, and the new one stays.
    let res = crate::resource::test_texture_at(pool.va(0), (32, 32), 1, 28, Swizzle::LINEAR.0, 0);
    let words = crate::resource::texture_desc(res).unwrap();
    let info = crate::resource::placed_info(pool.va(0), &words).unwrap();
    t.texture_placed(pool.va(0), old, 9141, info.size);
    let _ = t.sync_jobs();
    assert!(t.host.knows(new), "a copy coming back covers nothing anew");
    assert_eq!(draw_sampling(&mut t, &pool, old, 4), [0x44, 0, 0, 0x44]);
    drop(views);
}
