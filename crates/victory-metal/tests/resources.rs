//! Descriptor heaps, textures and samplers on Metal: compute shaders (our own HLSL, compiled by dxc
//! and converted by the Metal Shader Converter) reach buffers and textures only through descriptor
//! tables in a heap. Each test runs with `MTLResidencySet` residency and with `useResource` fallback.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use victory_metal::backend::{ComputeBackend, Dispatch, RootArgs};
use victory_metal::mem::page_size;
use victory_metal::metal::Metal;
use victory_metal::resources::*;

const N: usize = 4096;

#[test]
fn srv_and_uav_through_a_descriptor_table() {
    for m in backends() {
        let p = m.create_compute_pipeline(&fixture("heap_rw")).unwrap();
        // Two 16 KiB buffers = 32 KiB total; pages = ceil(32 KiB / page_size()).
        let pages = (2usize * 16 * 1024).div_ceil(page_size());
        let a = Arena::new(&m, pages);
        let half = a.region.len / 2;
        a.write(0, &(0..N as u32).collect::<Vec<_>>());
        let heap = m.create_heap(HeapKind::Resource, 16).unwrap();
        // Entries 5 and 6 are the table the root signature points at: SRV t0, UAV u0.
        m.write_buffer(&heap, 5, &a.buf, a.at(0), (N * 4) as u32);
        m.write_buffer(&heap, 6, &a.buf, a.at(half), (N * 4) as u32);
        let w = [Dispatch {
            pipeline: &p,
            root: RootArgs::default()
                .constants(&[5, 7])
                .address(m.heap_address(&heap, 5)),
            resources: vec![],
            heaps: victory_metal::backend::Heaps {
                resource: Some(&heap),
                sampler: None,
            },
            groups: [N / 64, 1, 1],
            threads_per_group: [64, 1, 1],
            indirect: None,
            count: None,
        }];
        run(&m, &w);
        let out: Vec<u32> = a.read(half, N);
        assert!(
            out.iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7),
            "dst[i] must be src[i]*5+7"
        );
        // Re-pointing an entry takes effect: swap SRV and UAV roles' memory by rewriting entry 5.
        a.write(0, &vec![1u32; N]);
        m.write_buffer(&heap, 5, &a.buf, a.at(0), (N * 4) as u32);
        run(&m, &w);
        assert!(a.read::<u32>(half, N).iter().all(|v| *v == 12));
    }
}

/// A 2D/array/3D texture sampled at the cell centres of a `g` grid; one float4 per cell.
struct Grid<'a> {
    m: &'a Metal,
    dxil: &'a [u8],
    g: [u32; 2],
    z: usize,
    filter: Filter,
}

impl Grid<'_> {
    fn run(&self, tex: &<Metal as ResourceBackend>::Texture) -> Vec<[f32; 4]> {
        let m = self.m;
        let p = m.create_compute_pipeline(self.dxil).unwrap();
        let cells = self.g[0] as usize * self.g[1] as usize * self.z;
        let a = Arena::new(
            m,
            (cells * 16)
                .div_ceil(victory_metal::mem::page_size())
                .max(1),
        );
        a.write(0, &vec![0xdead_beefu32; cells * 4]);
        let res = m.create_heap(HeapKind::Resource, 4).unwrap();
        let smp = m.create_heap(HeapKind::Sampler, 2).unwrap();
        m.write_texture(&res, 0, tex, 0.0);
        m.write_buffer(&res, 1, &a.buf, a.at(0), (cells * 16) as u32);
        let s = m
            .create_sampler(&SamplerDesc {
                min: self.filter,
                mag: self.filter,
                mip: Filter::Nearest,
                address: [Address::Clamp; 3],
                border_color: BorderColor::TransparentBlack,
                max_anisotropy: 1,
            })
            .unwrap();
        m.write_sampler(&smp, 0, &s, 0.0);
        let w = [Dispatch {
            pipeline: &p,
            root: RootArgs::default()
                .constants(&self.g)
                .address(m.heap_address(&res, 0))
                .address(m.heap_address(&smp, 0)),
            resources: vec![],
            heaps: victory_metal::backend::Heaps {
                resource: Some(&res),
                sampler: Some(&smp),
            },
            groups: [
                self.g[0].div_ceil(4) as usize,
                self.g[1].div_ceil(4) as usize,
                self.z,
            ],
            threads_per_group: [4, 4, 1],
            indirect: None,
            count: None,
        }];
        run(m, &w);
        let f: Vec<f32> = a.read(0, cells * 4);
        f.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
    }
}

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
}

fn d2(f: Format, w: usize, h: usize) -> TextureDesc {
    TextureDesc::d2(f, w, h)
}

#[test]
fn sample_rgba8_2d() {
    for m in backends() {
        let t = m.create_texture(&d2(Format::Rgba8Unorm, 4, 4)).unwrap();
        let px: Vec<u8> = (0..16)
            .flat_map(|i| [(i % 4) as u8 * 60, (i / 4) as u8 * 60, 128, 255])
            .collect();
        m.upload_texture(&t, 0, 0, &px, 16, 0);
        let g = Grid {
            m: &m,
            dxil: &fixture("sample2d"),
            g: [4, 4],
            z: 1,
            filter: Filter::Nearest,
        };
        let out = g.run(&t);
        for (i, c) in out.iter().enumerate() {
            let want = [
                (i % 4) as f32 * 60.0 / 255.0,
                (i / 4) as f32 * 60.0 / 255.0,
                128.0 / 255.0,
                1.0,
            ];
            assert!(
                (0..4).all(|k| near(c[k], want[k])),
                "texel {i}: {c:?} != {want:?}"
            );
        }
    }
}

#[test]
fn sample_rgba16f_and_r32f() {
    let Some(m) = backend() else { return };
    // Exact half floats: 0 = 0x0000, 0.25 = 0x3400, 0.5 = 0x3800, 1 = 0x3C00.
    let h = [0x0000u16, 0x3400, 0x3800, 0x3c00];
    let t = m.create_texture(&d2(Format::Rgba16Float, 2, 2)).unwrap();
    let px: Vec<u16> = (0..4)
        .flat_map(|i| [h[i], h[(i + 1) % 4], h[(i + 2) % 4], 0x3c00])
        .collect();
    m.upload_texture(&t, 0, 0, bytes(&px), 16, 0);
    let g = Grid {
        m: &m,
        dxil: &fixture("sample2d"),
        g: [2, 2],
        z: 1,
        filter: Filter::Nearest,
    };
    let v = [0.0, 0.25, 0.5, 1.0];
    for (i, c) in g.run(&t).iter().enumerate() {
        assert_eq!(*c, [v[i], v[(i + 1) % 4], v[(i + 2) % 4], 1.0], "texel {i}");
    }

    let t = m.create_texture(&d2(Format::R32Float, 2, 2)).unwrap();
    m.upload_texture(&t, 0, 0, bytes(&[1.5f32, -2.0, 3.25, 4.0]), 8, 0);
    let out = g.run(&t);
    assert_eq!(
        out.iter().map(|c| c[0]).collect::<Vec<_>>(),
        [1.5, -2.0, 3.25, 4.0]
    );
}

fn bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices of ints and floats.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

#[test]
fn sample_bc1_and_create_every_bc_format() {
    let Some(m) = backend() else { return };
    // 8x8 BC1 = 2x2 blocks: red, green, blue, white (colour0 = RGB565, colour1 = 0, all indices 0).
    let blk = |c: u16| {
        let mut b = c.to_le_bytes().to_vec();
        b.extend([0u8; 6]);
        b
    };
    let data: Vec<u8> = [0xf800u16, 0x07e0, 0x001f, 0xffff]
        .iter()
        .flat_map(|c| blk(*c))
        .collect();
    let t = m.create_texture(&d2(Format::Bc1, 8, 8)).unwrap();
    m.upload_texture(&t, 0, 0, &data, Format::Bc1.row_bytes(8), 0);
    let g = Grid {
        m: &m,
        dxil: &fixture("sample2d"),
        g: [8, 8],
        z: 1,
        filter: Filter::Nearest,
    };
    let out = g.run(&t);
    let want = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    for (i, c) in out.iter().enumerate() {
        let w = want[(i / 8 / 4) * 2 + (i % 8) / 4];
        assert!(
            (0..3).all(|k| near(c[k], w[k])) && near(c[3], 1.0),
            "texel {i}: {c:?} != {w:?}"
        );
    }
    for f in [
        Format::Bc1,
        Format::Bc2,
        Format::Bc3,
        Format::Bc4,
        Format::Bc5,
        Format::Bc6h,
        Format::Bc6hSigned,
        Format::Bc7,
    ] {
        let t = m
            .create_texture(&d2(f, 16, 16))
            .unwrap_or_else(|e| panic!("{f:?}: {e}"));
        m.upload_texture(
            &t,
            0,
            0,
            &vec![0u8; f.row_bytes(16) * 4],
            f.row_bytes(16),
            0,
        );
    }
}

/// One BC6H block, mode 11 (mode bits `00011`: one region, 10-bit endpoints, not transformed),
/// endpoint 0 = (0x100, 0x300, 0), every index 0, so every texel is endpoint 0. Decoded as
/// Microsoft's "BC6H Format" page and the reference decoder (DirectXTex `BC6HBC7.cpp`,
/// `Unquantize` and `FinishUnquantize`) say:
/// - unsigned (`BC6H_UF16`): `(x << 16 | 0x8000) >> 10`, then `* 31 >> 6`, as half bits:
///   0x100 -> 0x1F0F = 1807 / 2^18, 0x300 -> 0x5D0F = 323.75;
/// - signed (`BC6H_SF16`): endpoints are 10-bit two's complement, so 0x300 is -256;
///   `(|x| << 15 | 0x4000) >> 9`, then `* 31 >> 5`, sign kept: 0x100 -> 0x3E1F = 1567 / 1024,
///   0x300 -> 0xBE1F = -1567 / 1024.
///
/// Metal's `BC6H_RGBFloat` is the signed format and `BC6H_RGBUfloat` the unsigned one; the host
/// used the signed one for DXGI 95.
#[test]
fn sample_bc6h_unsigned_and_signed() {
    let Some(m) = backend() else { return };
    let block = (0b00011u128 | 0x100 << 5 | 0x300 << 15).to_le_bytes();
    let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * b.abs().max(1.0);
    for (format, want) in [
        (Format::Bc6h, [1807.0 / 262_144.0, 323.75, 0.0]),
        (Format::Bc6hSigned, [1567.0 / 1024.0, -1567.0 / 1024.0, 0.0]),
    ] {
        let t = m.create_texture(&d2(format, 4, 4)).unwrap();
        m.upload_texture(&t, 0, 0, &block, format.row_bytes(4), 0);
        let g = Grid {
            m: &m,
            dxil: &fixture("sample2d"),
            g: [4, 4],
            z: 1,
            filter: Filter::Nearest,
        };
        for (i, c) in g.run(&t).iter().enumerate() {
            assert!(
                (0..3).all(|k| close(c[k], want[k])) && near(c[3], 1.0),
                "{format:?} texel {i}: {c:?} != {want:?}"
            );
        }
    }
}

#[test]
fn sample_2d_array_and_3d() {
    let Some(m) = backend() else { return };
    let mut d = d2(Format::Rgba8Unorm, 4, 4);
    d.dim = TexDim::D2Array;
    d.depth_or_layers = 3;
    let t = m.create_texture(&d).unwrap();
    for s in 0..3u8 {
        let px: Vec<u8> = (0..16)
            .flat_map(|i| [s * 100, (i % 4) as u8 * 60, (i / 4) as u8 * 60, 255])
            .collect();
        m.upload_texture(&t, 0, s as usize, &px, 16, 0);
    }
    let g = Grid {
        m: &m,
        dxil: &fixture("sample2darray"),
        g: [4, 4],
        z: 3,
        filter: Filter::Nearest,
    };
    for (i, c) in g.run(&t).iter().enumerate() {
        let (s, x, y) = (i / 16, i % 4, (i / 4) % 4);
        let want = [
            s as f32 * 100.0 / 255.0,
            x as f32 * 60.0 / 255.0,
            y as f32 * 60.0 / 255.0,
        ];
        assert!(
            (0..3).all(|k| near(c[k], want[k])),
            "slice {s} texel {x},{y}: {c:?}"
        );
    }

    let mut d = d2(Format::Rgba8Unorm, 4, 4);
    d.dim = TexDim::D3;
    d.depth_or_layers = 4;
    let t = m.create_texture(&d).unwrap();
    let px: Vec<u8> = (0..64)
        .flat_map(|i| {
            [
                (i % 4) as u8 * 60,
                ((i / 4) % 4) as u8 * 60,
                (i / 16) as u8 * 60,
                255,
            ]
        })
        .collect();
    m.upload_texture(&t, 0, 0, &px, 16, 64);
    let g = Grid {
        m: &m,
        dxil: &fixture("sample3d"),
        g: [4, 4],
        z: 4,
        filter: Filter::Nearest,
    };
    for (i, c) in g.run(&t).iter().enumerate() {
        let (x, y, z) = (i % 4, (i / 4) % 4, i / 16);
        let want = [
            x as f32 * 60.0 / 255.0,
            y as f32 * 60.0 / 255.0,
            z as f32 * 60.0 / 255.0,
        ];
        assert!(
            (0..3).all(|k| near(c[k], want[k])),
            "voxel {x},{y},{z}: {c:?}"
        );
    }
}

#[test]
fn texture_views_and_sampler_filters() {
    let Some(m) = backend() else { return };
    // A 4x4 texture with 3 mips: mip 0 all 0, mip 1 (2x2) all 200, mip 2 all 50. A view of mip 1 only.
    let mut d = d2(Format::Rgba8Unorm, 4, 4);
    d.mips = 3;
    let t = m.create_texture(&d).unwrap();
    for (mip, (w, v)) in [(4usize, 0u8), (2, 200), (1, 50)].into_iter().enumerate() {
        m.upload_texture(&t, mip, 0, &vec![v; w * w * 4], w * 4, 0);
    }
    let view = m
        .create_texture_view(
            &t,
            &ViewDesc {
                format: Format::Rgba8Unorm,
                mips: (1, 1),
                slices: (0, 1),
            },
        )
        .unwrap();
    let g = Grid {
        m: &m,
        dxil: &fixture("sample2d"),
        g: [2, 2],
        z: 1,
        filter: Filter::Nearest,
    };
    assert!(
        g.run(&view).iter().all(|c| near(c[0], 200.0 / 255.0)),
        "view of mip 1"
    );

    // The same bytes read as sRGB: 128 decodes to ~0.2158 instead of 0.502.
    let t = m.create_texture(&d2(Format::Rgba8Unorm, 2, 2)).unwrap();
    m.upload_texture(&t, 0, 0, &[128u8; 16], 8, 0);
    let srgb = m
        .create_texture_view(
            &t,
            &ViewDesc {
                format: Format::Rgba8UnormSrgb,
                mips: (0, 1),
                slices: (0, 1),
            },
        )
        .unwrap();
    assert!(g.run(&t).iter().all(|c| near(c[0], 0.502)));
    assert!(g.run(&srgb).iter().all(|c| near(c[0], 0.2158)));

    // A format Metal cannot cast the texture to is an error, not Metal's assertion (a title view
    // left over a resource object the title made anew); one of the same texel size is a view.
    let view = |format| m.create_texture_view(&t, &ViewDesc { format, mips: (0, 1), slices: (0, 1) });
    for format in [Format::R8Unorm, Format::Bc1, Format::Rgba16Float, Format::D32Float] {
        assert!(view(format).is_err(), "{format:?}");
    }
    for format in [Format::R32Uint, Format::Rgb10a2Unorm, Format::Bgra8Unorm] {
        assert!(view(format).is_ok(), "{format:?}");
    }
    let bc = m.create_texture(&d2(Format::Bc7, 4, 4)).unwrap();
    let view = |format| m.create_texture_view(&bc, &ViewDesc { format, mips: (0, 1), slices: (0, 1) });
    assert!(view(Format::Bc7Srgb).is_ok() && view(Format::Bc3).is_err() && view(Format::Rgba32Float).is_err());

    // Filters: a 2x2 texture, red 0 | 255 (both rows), sampled on a 4x4 grid. Nearest only ever
    // returns the two texel values; linear blends between the texel centres.
    let t = m.create_texture(&d2(Format::Rgba8Unorm, 2, 2)).unwrap();
    let px = [0u8, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255, 255, 0, 0, 255];
    m.upload_texture(&t, 0, 0, &px, 8, 0);
    let run = |filter| {
        Grid {
            m: &m,
            dxil: &fixture("sample2d"),
            g: [4, 4],
            z: 1,
            filter,
        }
        .run(&t)
    };
    assert!(
        run(Filter::Nearest)
            .iter()
            .all(|c| c[0] == 0.0 || c[0] == 1.0)
    );
    let lin = run(Filter::Linear);
    assert!(
        lin.iter().any(|c| c[0] > 0.1 && c[0] < 0.9),
        "linear must blend: {lin:?}"
    );
}

#[test]
fn texture_over_zero_copy_memory() {
    let Some(m) = backend() else { return };
    let a = Arena::new(&m, 2);
    let align = m.linear_alignment(Format::Rgba8Unorm);
    let pitch = 16usize.next_multiple_of(align);
    // The texture starts one alignment unit into the arena, like a title resource at an offset.
    let base = align;
    let px = |v: u8| -> Vec<u8> {
        (0..4)
            .flat_map(|r| {
                (0..pitch).map(move |c| {
                    if c < 16 {
                        v + r as u8 + (c / 4) as u8
                    } else {
                        0
                    }
                })
            })
            .collect()
    };
    a.write(base, &px(10));
    let t = m
        .create_texture_over_memory(&a.buf, a.at(base), &d2(Format::Rgba8Unorm, 4, 4), pitch)
        .unwrap();
    let g = Grid {
        m: &m,
        dxil: &fixture("sample2d"),
        g: [4, 4],
        z: 1,
        filter: Filter::Nearest,
    };
    let check = |v: u8| {
        for (i, c) in g.run(&t).iter().enumerate() {
            let want = (v as usize + i / 4 + i % 4) as f32 / 255.0;
            assert!(near(c[0], want), "texel {i}: {} != {want}", c[0]);
        }
    };
    check(10);
    // A CPU write into the title memory after the texture exists is seen by the next dispatch.
    a.write(base, &px(60));
    check(60);
}

/// The packed 16-bit formats' bit order: `BGR5A1Unorm` is DXGI's `B5G5R5A1_UNORM` (blue in the low
/// bits, alpha the top one); `ABGR4Unorm` has alpha in the low nibble and red in the top one, where
/// DXGI's `B4G4R4A4_UNORM` has blue low and alpha top.
#[test]
fn packed_16_bit_channel_order() {
    let Some(m) = backend() else { return };
    let rgba = [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    for (f, px, want) in [
        (Format::Bgr5a1Unorm, [0x001fu16, 0x03e0, 0x7c00, 0x8000], rgba),
        (Format::Abgr4Unorm, [0x00f0, 0x0f00, 0xf000, 0x000f], rgba),
    ] {
        let t = m.create_texture(&d2(f, 2, 2)).unwrap();
        m.upload_texture(&t, 0, 0, bytes(&px), 4, 0);
        let g = Grid { m: &m, dxil: &fixture("sample2d"), g: [2, 2], z: 1, filter: Filter::Nearest };
        assert_eq!(g.run(&t), want, "{f:?}");
    }
}

/// Overwriting heap entries (including sharing one texture across multiple slots) detaches
/// replaced allocations from the heap's residency set before dropping them, and dropping the
/// heap detaches all remaining allocations cleanly.
#[test]
fn heap_overwrite_detaches_replaced_residency_allocations() {
    for m in backends() {
        let heap = m.create_heap(HeapKind::Resource, 4).unwrap();
        let t1 = m.create_texture(&d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        let t2 = m.create_texture(&d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        // Bind t1 in two slots, then replace one (t1 still resident via slot 1), then the other.
        m.write_texture(&heap, 0, &t1, 0.0);
        m.write_texture(&heap, 1, &t1, 0.0);
        m.write_texture(&heap, 0, &t2, 0.0);
        m.write_texture(&heap, 1, &t2, 0.0);
        drop(t1);
        // Overwrite with a new texture and drop both heap and remaining textures.
        let t3 = m.create_texture(&d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        m.write_texture(&heap, 0, &t3, 0.0);
        drop(t2);
        drop(heap);
        drop(t3);
    }
}

/// The indirect tests' arena: the source at 0, the records after it, the destination at half.
/// In bytes, not pages: 4 KiB pages (x86_64 under Rosetta) would leave 3 pages too small.
const INDIRECT_ARENA: usize = 48 * 1024;

#[test]
fn indirect_dispatch_reads_its_group_counts_from_title_memory() {
    for m in backends() {
        let p = m.create_compute_pipeline(&fixture("heap_rw")).unwrap();
        let a = Arena::new(&m, INDIRECT_ARENA.div_ceil(page_size()));
        let (dst, record) = (a.region.len / 2, N * 4);
        a.write(0, &(0..N as u32).collect::<Vec<_>>());
        // `D3D12_DISPATCH_ARGUMENTS`: half the groups.
        a.write(record, &[(N / 128) as u32, 1, 1]);
        let heap = m.create_heap(HeapKind::Resource, 16).unwrap();
        m.write_buffer(&heap, 5, &a.buf, a.at(0), (N * 4) as u32);
        m.write_buffer(&heap, 6, &a.buf, a.at(dst), (N * 4) as u32);
        let w = [Dispatch {
            pipeline: &p,
            root: RootArgs::default()
                .constants(&[5, 7])
                .address(m.heap_address(&heap, 5)),
            resources: vec![],
            heaps: victory_metal::backend::Heaps {
                resource: Some(&heap),
                sampler: None,
            },
            groups: [0, 0, 0],
            threads_per_group: [64, 1, 1],
            indirect: Some((&a.buf, a.at(record) as *const u8)),
            count: None,
        }];
        run(&m, &w);
        let out: Vec<u32> = a.read(dst, N);
        assert!(out[..N / 2].iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7));
        assert!(out[N / 2..].iter().all(|v| *v == 0), "groups past the record's count do not run");
    }
}

/// A record of over 2^24 groups (a count written wrong) runs no groups instead of keeping the GPU
/// busy until Metal's watchdog fails the queue. A record within the limit in the same encoder
/// still runs, as does one past D3D12's 65535 a dimension (the title dispatches 100000 in x).
#[test]
fn an_indirect_dispatch_past_the_group_limit_runs_nothing() {
    for m in backends() {
        let p = m.create_compute_pipeline(&fixture("heap_rw")).unwrap();
        let a = Arena::new(&m, INDIRECT_ARENA.div_ceil(page_size()));
        let (dst, records) = (a.region.len / 2, N * 4);
        a.write(0, &(0..N as u32).collect::<Vec<_>>());
        // Half the groups, then 2^24 + 1 groups (the heap view stops each past the first N).
        a.write(records, &[(N / 128) as u32, 1, 1, 4097, 4096, 1]);
        let heap = m.create_heap(HeapKind::Resource, 16).unwrap();
        m.write_buffer(&heap, 5, &a.buf, a.at(0), (N * 4) as u32);
        m.write_buffer(&heap, 6, &a.buf, a.at(dst), (N * 4) as u32);
        let w: Vec<_> = (0..2usize)
            .map(|i| Dispatch {
                pipeline: &p,
                root: RootArgs::default()
                    .constants(&[5, 7])
                    .address(m.heap_address(&heap, 5)),
                resources: vec![],
                heaps: victory_metal::backend::Heaps {
                    resource: Some(&heap),
                    sampler: None,
                },
                groups: [0, 0, 0],
                threads_per_group: [64, 1, 1],
                indirect: Some((&a.buf, a.at(records + 12 * i) as *const u8)),
                count: None,
            })
            .collect();
        run(&m, &w);
        let out: Vec<u32> = a.read(dst, N);
        assert!(out[..N / 2].iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7));
        assert!(out[N / 2..].iter().all(|v| *v == 0), "the record past the limit ran groups");
    }
}

/// D3D12's limit is 65535 groups a dimension, but the title dispatches over 100000 groups in x
/// indirectly, which the console and Metal run: those run.
#[test]
fn an_indirect_dispatch_past_65535_groups_in_x_runs() {
    for m in backends() {
        // One thread a group, so every thread has an element of its own (Metal does not stop
        // writes past a view, which would land in other tests' memory).
        const G: usize = 70_000;
        let p = m.create_compute_pipeline(&fixture("heap_rw")).unwrap();
        let a = Arena::new(&m, (2 * G * 4 + 4096).div_ceil(page_size()));
        let (src, dst, record) = (0, G * 4, 2 * G * 4);
        a.write(src, &(0..G as u32).collect::<Vec<_>>());
        a.write(record, &[G as u32, 1, 1]);
        let heap = m.create_heap(HeapKind::Resource, 16).unwrap();
        m.write_buffer(&heap, 5, &a.buf, a.at(src), (G * 4) as u32);
        m.write_buffer(&heap, 6, &a.buf, a.at(dst), (G * 4) as u32);
        let w = [Dispatch {
            pipeline: &p,
            root: RootArgs::default()
                .constants(&[5, 7])
                .address(m.heap_address(&heap, 5)),
            resources: vec![],
            heaps: victory_metal::backend::Heaps {
                resource: Some(&heap),
                sampler: None,
            },
            groups: [0, 0, 0],
            threads_per_group: [1, 1, 1],
            indirect: Some((&a.buf, a.at(record) as *const u8)),
            count: None,
        }];
        run(&m, &w);
        let out: Vec<u32> = a.read(dst, G);
        assert!(out.iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7));
    }
}

#[test]
fn a_count_buffer_limits_the_indirect_dispatches_that_run() {
    for m in backends() {
        let p = m.create_compute_pipeline(&fixture("heap_rw")).unwrap();
        // Three records of 2, 4 and 8 groups of 64: what ran shows in how much was written.
        for (count, written) in [(2u32, 256), (0, 0), (5, 512)] {
            let a = Arena::new(&m, INDIRECT_ARENA.div_ceil(page_size()));
            let (dst, records, counter) = (a.region.len / 2, N * 4, N * 4 + 64);
            a.write(0, &(0..N as u32).collect::<Vec<_>>());
            a.write(records, &[2u32, 1, 1, 4, 1, 1, 8, 1, 1]);
            a.write(counter, &[count]);
            let heap = m.create_heap(HeapKind::Resource, 16).unwrap();
            m.write_buffer(&heap, 5, &a.buf, a.at(0), (N * 4) as u32);
            m.write_buffer(&heap, 6, &a.buf, a.at(dst), (N * 4) as u32);
            let w: Vec<_> = (0..3u32)
                .map(|i| Dispatch {
                    pipeline: &p,
                    root: RootArgs::default()
                        .constants(&[5, 7])
                        .address(m.heap_address(&heap, 5)),
                    resources: vec![],
                    heaps: victory_metal::backend::Heaps {
                        resource: Some(&heap),
                        sampler: None,
                    },
                    groups: [0, 0, 0],
                    threads_per_group: [64, 1, 1],
                    indirect: Some((&a.buf, a.at(records + 12 * i as usize) as *const u8)),
                    count: Some((&a.buf, a.at(counter) as *const u8, i)),
                })
                .collect();
            run(&m, &w);
            let out: Vec<u32> = a.read(dst, N);
            let want: Vec<u32> = (0..N as u32)
                .map(|i| if (i as usize) < written { i * 5 + 7 } else { 0 })
                .collect();
            assert_eq!(out, want, "count {count}");
        }
    }
}

#[test]
fn zero_sized_dispatch_and_blit_do_not_fail_validation() {
    for m in backends() {
        let p = m.create_compute_pipeline(&fixture("heap_rw")).unwrap();
        let a = Arena::new(&m, 1);
        let heap = m.create_heap(HeapKind::Resource, 16).unwrap();
        m.write_buffer(&heap, 5, &a.buf, a.at(0), 64);
        m.write_buffer(&heap, 6, &a.buf, a.at(64), 64);
        let q = m.create_queue().unwrap();
        let fence = m.create_fence(0).unwrap();
        let dispatches = [Dispatch {
            pipeline: &p,
            root: RootArgs::default()
                .constants(&[5, 7])
                .address(m.heap_address(&heap, 5)),
            resources: vec![],
            heaps: victory_metal::backend::Heaps {
                resource: Some(&heap),
                sampler: None,
            },
            groups: [0, 1, 1],
            threads_per_group: [64, 1, 1],
            indirect: None,
            count: None,
        }];
        let blits = vec![
            BlitOp::Fill { dst: (&a.buf, a.at(0) as *const u8), len: 0, value: 0xaa },
            BlitOp::Copy { src: (&a.buf, a.at(0) as *const u8), dst: (&a.buf, a.at(64) as *const u8), len: 0 },
        ];
        let passes = [Pass::Compute(&dispatches), Pass::Blit(blits)];
        let signals = [(&fence, 1)];
        m.submit_passes(&q, &PassSubmission { waits: &[], passes: &passes, signals: &signals }).unwrap();
        assert!(m.wait(&fence, 1, 5000));
    }
}

#[test]
fn texture_blit_ops_copy_between_textures_and_buffers() {
    let Ok(m) = Metal::new() else { return };
    let q = m.create_queue().unwrap();
    let f = m.create_fence(0).unwrap();
    let a = Arena::new(&m, 0x20000);

    let (w, h) = (8usize, 8usize);
    let pitch = 256usize;
    let src_data: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| [x as u8 * 17, y as u8 * 29, 99, 255]))
        .collect();
    let mut staged = vec![0u8; (h - 1) * pitch + w * 4];
    for y in 0..h {
        staged[y * pitch..y * pitch + w * 4].copy_from_slice(&src_data[y * w * 4..(y + 1) * w * 4]);
    }
    a.write(0, &staged);

    let desc = TextureDesc {
        dim: TexDim::D2,
        format: Format::Rgba8Unorm,
        width: w,
        height: h,
        depth_or_layers: 1,
        mips: 1,
        render_target: false,
        unordered_access: false,
    };
    let tex_a = m.create_texture(&desc).unwrap();
    let mut srgb_desc = desc.clone();
    srgb_desc.format = Format::Rgba8UnormSrgb;
    let tex_b = m.create_texture(&srgb_desc).unwrap();
    let dst_off = 0x10000;

    let passes = [Pass::Blit(vec![
        BlitOp::CopyBufferToTexture {
            src: (&a.buf, a.at(0) as *const u8),
            len: staged.len(),
            row_pitch: pitch,
            slice_pitch: pitch * h,
            size: [w, h, 1],
            dst: &tex_a,
            dst_mip: 0,
            dst_slice: 0,
            dst_origin: [0, 0, 0],
        },
        BlitOp::CopyTexture {
            src: &tex_a,
            src_mip: 0,
            src_slice: 0,
            src_origin: [2, 1, 0],
            size: [4, 4, 1],
            dst: &tex_b,
            dst_mip: 0,
            dst_slice: 0,
            dst_origin: [1, 2, 0],
        },
        BlitOp::CopyTextureToBuffer {
            src: &tex_b,
            src_mip: 0,
            src_slice: 0,
            src_origin: [1, 2, 0],
            size: [4, 4, 1],
            dst: (&a.buf, a.at(dst_off) as *const u8),
            len: 3 * pitch + 16,
            row_pitch: pitch,
            slice_pitch: pitch * 4,
        },
    ])];
    m.submit_passes(&q, &PassSubmission { waits: &[], passes: &passes, signals: &[(&f, 1)] })
        .unwrap();
    assert!(m.wait(&f, 1, 5000));

    let out: Vec<u8> = a.read(dst_off, 3 * pitch + 16);
    for row in 0..4 {
        for col in 0..4 {
            let at = row * pitch + col * 4;
            assert_eq!(
                &out[at..at + 4],
                &[(col as u8 + 2) * 17, (row as u8 + 1) * 29, 99, 255],
                "texel ({col}, {row})"
            );
        }
    }
}
