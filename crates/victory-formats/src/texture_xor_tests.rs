//! The XOR modes' addresses (`_X` and `_T`): equations from addrlib's tables, checked against the
//! modes without XOR, against a reference computed from the header's text, and for bijection and
//! round trips.

use super::*;

include!("texture_xor_golden.rs");

fn desc(w: u32, h: u32, mips: u32, bpe: u32, swizzle: Swizzle) -> Desc {
    Desc {
        dimension: Dimension::Tex2d,
        width: w,
        height: h,
        depth: 1,
        mips,
        element_bytes: bpe,
        block: 1,
        samples: 1,
        swizzle,
    }
}

/// Every XOR mode with equations.
const XOR_MODES: [Swizzle; 8] = [
    Swizzle::SW_64KB_S_T,
    Swizzle::SW_64KB_D_T,
    Swizzle::SW_4KB_S_X,
    Swizzle::SW_4KB_D_X,
    Swizzle::SW_64KB_Z_X,
    Swizzle::SW_64KB_S_X,
    Swizzle::SW_64KB_D_X,
    Swizzle::SW_64KB_R_X,
];

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

/// The equations extracted from addrlib's tables give the addresses of the modes without XOR that
/// this module computes from its own micro-tile patterns, for every element size, in 2D and thick
/// 3D: the two came from different data, so this checks the extraction and the coordinates.
#[test]
fn equations_reproduce_the_modes_without_xor() {
    let cases: [(Swizzle, bool, &[[u16; 4]; 75]); 8] = [
        (Swizzle::SW_256B_S, false, &xor_tables::SW_256B_S),
        (Swizzle::SW_256B_D, false, &xor_tables::SW_256B_D),
        (Swizzle::SW_4KB_S, false, &xor_tables::SW_4KB_S),
        (Swizzle::SW_4KB_D, false, &xor_tables::SW_4KB_D),
        (Swizzle::SW_64KB_S, false, &xor_tables::SW_64KB_S),
        (Swizzle::SW_64KB_D, false, &xor_tables::SW_64KB_D),
        (Swizzle::SW_4KB_S, true, &xor_tables::SW_4KB_S3),
        (Swizzle::SW_64KB_S, true, &xor_tables::SW_64KB_S3),
    ];
    let mut seed = 1;
    for (swizzle, three_d, table) in cases {
        for e in 0..5u32 {
            let bpe = 1 << e;
            let shape = |w, h| Desc {
                dimension: if three_d {
                    Dimension::Tex3d
                } else {
                    Dimension::Tex2d
                },
                depth: if three_d { 40 } else { 1 },
                ..desc(w, h, 1, bpe, swizzle)
            };
            let [bw, bh, _] = Layout::new(&shape(1, 1)).unwrap().block;
            let d = shape(2 * bw + 1, 2 * bh + 1);
            let l = Layout::new(&d).unwrap();
            let row = table[PipeConfig::DEFAULT.row().unwrap() * 5 + e as usize];
            let xor = Xor::new(xor_bits(row, swizzle.block_log2().unwrap()), 0);
            let m = &l.mips[0];
            for _ in 0..4000 {
                let (x, y) = (lcg(&mut seed) % m.width, lcg(&mut seed) % m.height);
                let z = lcg(&mut seed) % d.depth;
                let (slice, _) = l.split(z);
                let at = u64::from(slice) * l.slice_size + l.xor_address(&xor, m, x, y, z);
                let want = l.offset(0, x, y, z).unwrap();
                assert_eq!(at, want, "{swizzle:?} {bpe} B 3D {three_d} ({x},{y},{z})");
            }
        }
    }
}

/// Addresses of the XOR modes against a reference that evaluates addrlib's tables as text
/// (`tools/swizzle/xor_reference.py`, which shares no code with the table generator).
#[test]
fn addresses_match_the_reference() {
    for &(mode, three_d, e, packers_log2, pipes_log2, x, y, z, address, slice) in REFERENCE {
        let config = PipeConfig {
            pipes_log2,
            packers_log2,
        };
        let shape = |w, h| Desc {
            dimension: if three_d {
                Dimension::Tex3d
            } else {
                Dimension::Tex2d
            },
            depth: 8,
            ..desc(w, h, 1, 1 << e, Swizzle(mode))
        };
        let [bw, bh, _] = Layout::new_for(&shape(1, 1), config).unwrap().block;
        let l = Layout::new_for(&shape(3 * bw, 2 * bh), config).unwrap();
        assert!(l.addressable(), "{mode} {e}");
        let at = l.offset(0, x, y, z).unwrap();
        assert_eq!(
            at,
            u64::from(slice) * l.slice_size + address,
            "mode {mode} 3D {three_d} e {e} {config:?} ({x},{y},{z})"
        );
    }
}

/// Tiles and detiles every mip of every slice of `d` through `l`, and checks that each element has
/// its own address inside the texture.
fn round_trips(l: &Layout, d: &Desc) {
    let mut seen = vec![false; l.size as usize];
    let mut tiled = vec![0u8; l.size as usize];
    let n = d.element_bytes as usize;
    let slices = if d.dimension == Dimension::Tex3d {
        1
    } else {
        d.depth
    };
    for slice in 0..slices {
        for mip in 0..d.mips {
            let m = l.mips[mip as usize];
            let active = m.width as usize * n;
            let row = active + 13;
            let plane = row * m.height as usize + 11;
            let linear: Vec<u8> = (0..plane * m.depth as usize)
                .map(|i| (i * 7 + mip as usize * 13 + slice as usize) as u8)
                .collect();
            l.tile(&mut tiled, mip, slice, &linear, row, plane).unwrap();
            let mut back = vec![0xcc; linear.len()];
            l.detile(&tiled, mip, slice, &mut back, row, plane).unwrap();
            for z in 0..m.depth {
                let base = z as usize * plane;
                for y in 0..m.height {
                    let start = base + y as usize * row;
                    assert_eq!(back[start..start + active], linear[start..start + active]);
                    assert!(back[start + active..start + row].iter().all(|&b| b == 0xcc));
                    for x in 0..m.width {
                        let at = l.offset(mip, x, y, slice + z).unwrap() as usize;
                        let source = start + x as usize * n;
                        assert_eq!(tiled[at..at + n], linear[source..source + n]);
                        assert!(at + n <= seen.len(), "{d:?} mip {mip} outside");
                        assert!(at.is_multiple_of(n));
                        assert!(!seen[at], "{d:?} mip {mip} ({x},{y},{z}) twice");
                        seen[at] = true;
                    }
                }
            }
        }
    }
}

/// Addresses are a bijection and copies round trip in every XOR mode, with mips (and their tails),
/// array slices and 3D depth, odd sizes and block compression, for two pipe configurations.
#[test]
fn modes_are_bijections_and_copies_round_trip() {
    let mut cases = vec![];
    for (i, swizzle) in XOR_MODES.into_iter().enumerate() {
        cases.push(desc(
            100 + 37 * i as u32,
            60 + 11 * i as u32,
            7,
            1 << (i % 5),
            swizzle,
        ));
        cases.push(Desc {
            depth: 3,
            ..desc(70, 33, 5, 4, swizzle)
        });
        cases.push(Desc {
            block: 4,
            ..desc(109, 71, 6, 16, swizzle)
        });
    }
    for swizzle in [
        Swizzle::SW_64KB_S_X,
        Swizzle::SW_4KB_S_X,
        Swizzle::SW_64KB_Z_X,
        Swizzle::SW_64KB_S_T,
    ] {
        cases.push(Desc {
            dimension: Dimension::Tex3d,
            depth: 13,
            ..desc(40, 20, 4, 4, swizzle)
        });
    }
    let other = PipeConfig {
        pipes_log2: 4,
        packers_log2: 3,
    };
    for config in [PipeConfig::DEFAULT, other] {
        for d in &cases {
            let l = Layout::new_for(d, config).unwrap();
            assert!(l.addressable(), "{d:?}");
            round_trips(&l, d);
        }
    }
}

/// The surface's pipe-bank XOR flips pipe and bank bits of the address inside a block, and nothing
/// else; the addresses stay a bijection.
#[test]
fn the_pipe_bank_xor_flips_the_pipe_and_bank_bits() {
    let config = PipeConfig::DEFAULT;
    let d = desc(300, 200, 3, 4, Swizzle::SW_64KB_D_X);
    let plain = Layout::new_for(&d, config).unwrap();
    // 8 pipes: bits 8-10 are the pipe, 11-12 the column (never XORed), 13-15 the bank.
    let flipped = plain.clone().with_pipe_bank_xor(0b1_1111_1111, config);
    assert_ne!(plain, flipped);
    let mut seed = 5;
    for _ in 0..2000 {
        let (x, y) = (lcg(&mut seed) % 300, lcg(&mut seed) % 200);
        let (a, b) = (
            plain.offset(0, x, y, 0).unwrap(),
            flipped.offset(0, x, y, 0).unwrap(),
        );
        assert_eq!(a ^ b, 0xE700, "({x},{y})");
    }
    round_trips(&flipped, &d);
    // Modes without XOR ignore it.
    let s = Layout::new(&desc(64, 64, 1, 4, Swizzle::SW_4KB_S)).unwrap();
    assert_eq!(s.clone().with_pipe_bank_xor(0xFF, config), s);
}

/// What has no equations stays unaddressable, and says so.
#[test]
fn modes_without_tables_and_msaa_are_not_addressable() {
    let r = Layout::new(&desc(64, 64, 1, 4, Swizzle::SW_4KB_R_X)).unwrap();
    assert!(!r.addressable());
    assert_eq!(r.offset(0, 0, 0, 0), Err(Error::Unsupported));
    let msaa = Layout::new(&Desc {
        samples: 4,
        ..desc(64, 64, 1, 4, Swizzle::SW_64KB_R_X)
    })
    .unwrap();
    assert!(!msaa.addressable());
    // A pipe configuration without tables.
    let bad = PipeConfig {
        pipes_log2: 1,
        packers_log2: 4,
    };
    let l = Layout::new_for(&desc(64, 64, 1, 4, Swizzle::SW_64KB_D_X), bad).unwrap();
    assert!(!l.addressable());
}

/// `Layout::walk` visits each active element once and pairs its linear byte offset with the exact
/// per-element `Layout::xor_address` across every addressable XOR mode (`_X` and `_T`),
/// power-of-two element byte width (`1, 2, 4, 8, 16`), 2D array and 3D depth layout, mip chain
/// (including tail mips), pipe configuration, and pipe-bank XOR.
#[test]
fn walk_matches_per_element_xor_address_across_modes() {
    fn check_walk(l: &Layout, d: &Desc) {
        let xor = l.xor.as_deref().expect("XOR layout");
        let n = d.element_bytes as usize;
        let slices = if d.dimension == Dimension::Tex3d {
            1
        } else {
            d.depth
        };
        for slice in 0..slices {
            for mip in 0..d.mips {
                let m = l.mips[mip as usize];
                let row = m.width as usize * n + 13;
                let plane = row * m.height as usize + 11;
                let linear_len = plane * m.depth as usize;
                let mut got = vec![None::<usize>; linear_len];
                let mut total_elems = 0usize;
                l.walk(
                    l.size as usize,
                    mip,
                    slice,
                    linear_len,
                    row,
                    plane,
                    |tiled_off, linear_off, len| {
                        assert!(len >= n && len.is_multiple_of(n));
                        for step in (0..len).step_by(n) {
                            let slot = &mut got[linear_off + step];
                            assert!(
                                slot.is_none(),
                                "{d:?} mip {mip} slice {slice} linear byte {} visited twice",
                                linear_off + step
                            );
                            *slot = Some(tiled_off + step);
                            total_elems += 1;
                        }
                    },
                )
                .unwrap();
                assert_eq!(
                    total_elems,
                    m.width as usize * m.height as usize * m.depth as usize
                );
                for z in 0..m.depth {
                    let (slab, _) = l.split(slice + z);
                    let slab_base = u64::from(slab) * l.slice_size + m.offset;
                    for y in 0..m.height {
                        for x in 0..m.width {
                            let lin =
                                z as usize * plane + y as usize * row + x as usize * n;
                            let want =
                                (slab_base + l.xor_address(xor, &m, x, y, slice + z)) as usize;
                            assert_eq!(
                                got[lin],
                                Some(want),
                                "{d:?} mip {mip} slice {slice} ({x},{y},{z})"
                            );
                        }
                    }
                }
            }
        }
    }

    let other = PipeConfig {
        pipes_log2: 4,
        packers_log2: 3,
    };
    for config in [PipeConfig::DEFAULT, other] {
        for flip in [0u32, 0x1AB] {
            for swizzle in XOR_MODES {
                for e in 0..5u32 {
                    let d = Desc {
                        depth: 2,
                        ..desc(67, 35, 6, 1 << e, swizzle)
                    };
                    let l = Layout::new_for(&d, config)
                        .unwrap()
                        .with_pipe_bank_xor(flip, config);
                    check_walk(&l, &d);
                }
            }
            for swizzle in [
                Swizzle::SW_64KB_S_T,
                Swizzle::SW_4KB_S_X,
                Swizzle::SW_64KB_Z_X,
                Swizzle::SW_64KB_S_X,
                Swizzle::SW_64KB_D_X,
                Swizzle::SW_64KB_R_X,
            ] {
                for e in [0u32, 2, 4] {
                    let d = Desc {
                        dimension: Dimension::Tex3d,
                        depth: 9,
                        ..desc(35, 19, 4, 1 << e, swizzle)
                    };
                    let l = Layout::new_for(&d, config)
                        .unwrap()
                        .with_pipe_bank_xor(flip, config);
                    check_walk(&l, &d);
                }
            }
        }
    }
    // Non-power-of-two element byte widths (`3, 6, 12`) are rejected by `Layout::new_for`.
    for swizzle in XOR_MODES {
        for bpe in [3u32, 6, 12] {
            assert_eq!(
                Layout::new(&desc(4, 4, 1, bpe, swizzle)),
                Err(Error::Invalid)
            );
        }
    }
}

/// Partial trailing micro-runs (widths not divisible by the 16-byte micro-run length) with a
/// non-zero pipe-bank XOR round-trip across every element byte width (`1, 2, 4, 8, 16`), including
/// mip tails.
#[test]
fn partial_xor_runs_with_pipe_bank_xor_round_trip_every_element_size() {
    let config = PipeConfig::DEFAULT;
    for e in 0..5u32 {
        let bpe = 1 << e;
        for swizzle in [Swizzle::SW_64KB_S_X, Swizzle::SW_64KB_D_X, Swizzle::SW_4KB_S_X] {
            let d = desc(67, 35, 6, bpe, swizzle);
            let l = Layout::new_for(&d, config)
                .unwrap()
                .with_pipe_bank_xor(0x1AB, config);
            round_trips(&l, &d);
        }
    }
}

#[test]
#[ignore = "benchmark"]
fn xor_detile_throughput() {
    for bpe in [4u32, 16] {
        let d = desc(2048, 2048, 1, bpe, Swizzle::SW_64KB_S_X);
        let l = Layout::new(&d)
            .unwrap()
            .with_pipe_bank_xor(0x155, PipeConfig::DEFAULT);
        let tiled = vec![0xA5u8; l.size as usize];
        let row = 2048 * bpe as usize;
        let slice_pitch = row * 2048;
        let mut linear = vec![0u8; slice_pitch];
        let iters = 20u32;
        let started = std::time::Instant::now();
        for _ in 0..iters {
            l.detile(&tiled, 0, 0, &mut linear, row, slice_pitch).unwrap();
            std::hint::black_box(&linear);
        }
        let per_ms = started.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
        eprintln!("2048x2048 SW_64KB_S_X {bpe}B detile: {per_ms:.2} ms");
    }
}
