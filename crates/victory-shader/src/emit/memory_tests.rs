//! Title memory at addresses the program computes (decision 0005, the
//! unified-memory item), run on the host GPU and WARP with the test host
//! serving the contract (`emit` "Title memory"): a synthetic GFX10 program
//! takes a 64-bit pointer from root constants, reads through it with a
//! global and a scalar load, follows a pointer it loaded with another
//! scalar load, stores through a pointer it computes (64-bit multiply-add)
//! and does float min/max atomics, reads through the address it takes
//! from a buffer view's descriptor and an image descriptor's size fields
//! (the host's view addresses), reads
//! pointers into memory no copy holds (zeros, marked missed) and stores and
//! loads through a root UAV (title memory in such a program).

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
/// Inline constant 0, 64.
const ZERO: u16 = 128;
const SIXTY_FOUR: u16 = 192;
/// The title address of the memory the program reads: above 4 GiB.
const TITLE: u64 = 0x12_3450_0000;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dword*` (`op` 0, 1, 2: one, two, four dwords) at
/// `s[base:base+1] + offset`.
const fn s_load(op: u32, sdata: u8, sbase: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | op << 18 | (sdata as u32) << 6 | (sbase as u32 >> 1),
        125 << 25 | offset,
    ]
}

/// VOP1 `v_mov_b32 vdst, src0`.
const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | (src0 as u32 & 0x1FF)
}

/// GLOBAL `op` with address `s[saddr:saddr+1] + vaddr + offset`.
const fn global(op: u32, vdst: u8, vaddr: u8, data: u8, saddr: u8, offset: u32) -> [u32; 2] {
    [
        0x37 << 26 | op << 18 | 2 << 14 | (offset & 0xFFF),
        vaddr as u32 | (data as u32) << 8 | (saddr as u32) << 16 | (vdst as u32) << 24,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

fn program() -> HostShader {
    let mut w = vec![s_mov_b32(5, 0x80)]; // s[4:5]: the table at s4
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result UAV
    w.push(v_mov_b32(1, ZERO));
    w.extend(global(12, 2, 1, 0, 0, 0)); // global_load_dword v2, v1, s[0:1]
    w.extend(s_load(1, 12, 0, 8)); // s[12:13] = the pointer at +8
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 14, 12, 0)); // s14 = what it points at
    w.push(v_mov_b32(3, SIXTY_FOUR));
    w.push(v_mov_b32(4, 2)); // v4 = s2
    // v_mad_u64_u32 v[8:9], s[20:21], v3, 1, s[0:1]: the pointer plus 0x40,
    // stored through as a VGPR pair.
    w.extend([0xD576_1408, 0x0001_0303]);
    w.extend(global(28, 0, 8, 4, 0x7D, 0)); // global_store_dword v[8:9], v4, off
    w.push(v_mov_b32(6, 3)); // v6 = s3 (a float)
    w.extend(global(63, 0, 3, 6, 0, 0x40)); // global_atomic_fmin at +0x80
    w.extend(global(64, 0, 3, 6, 0, 0x44)); // global_atomic_fmax at +0x84
    // The address of the view at heap index 1 from its descriptor (as the
    // tracers take their BVH's), read through.
    w.extend(s_load(1, 16, 4, 0x20)); // s[16:17] = descriptor 1, dwords 0-1
    w.push(S_WAITCNT_ALL);
    // Only the address's high bits of dword 1: its stride and swizzle are no address (#311).
    w.extend([0x8711_FF11, 0x0000_FFFF]); // s_and_b32 s17, s17, 0xffff
    w.extend(s_load(0, 18, 16, 0x100)); // s18 = the dword at +0x100
    w.push(S_WAITCNT_ALL);
    // Pointers into a page no copy holds and into a 256 MiB region with no
    // block, read through: zeros, and both marked missed.
    w.extend(s_load(1, 22, 0, 0x10)); // s[22:23]
    w.extend(s_load(1, 24, 0, 0x18)); // s[24:25]
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 26, 22, 0));
    w.extend(s_load(0, 27, 24, 0));
    w.push(S_WAITCNT_ALL);
    w.push(v_mov_b32(5, 14));
    w.push(v_mov_b32(7, 18));
    w.extend(buffer_store_dword(2, 8, 0));
    w.extend(buffer_store_dword(5, 8, 4));
    w.extend(buffer_store_dword(7, 8, 8));
    w.push(v_mov_b32(5, 26));
    w.push(v_mov_b32(7, 27));
    w.extend(buffer_store_dword(5, 8, 12));
    w.extend(buffer_store_dword(7, 8, 16));
    // A root UAV (s[6:7]), title memory in such a program: a store and a
    // load at its address.
    w.extend(global(28, 0, 1, 4, 6, 0x20)); // global_store_dword v1, v4, s[6:7]
    w.extend(global(12, 9, 1, 0, 6, 0x24)); // global_load_dword v9, v1, s[6:7]
    w.push(S_WAITCNT_ALL);
    w.extend(buffer_store_dword(9, 8, 20));
    // The image descriptor at heap index 2 loaded whole and read only for
    // its size fields (dwords 1-3).
    w.extend(s_load(3, 28, 4, 0x40)); // s[28:35]
    w.push(S_WAITCNT_ALL);
    // Only the size fields, as a title reads them: the table gives the rest of these dwords as 0.
    w.extend([0x871D_FF1D, 0xC000_0000]); // s_and_b32 s29, s29, 0xc0000000 (width, low bits)
    w.extend([0x871E_FF1E, 0x3FFF_FFFF]); // s_and_b32 s30, s30, 0x3fffffff
    w.extend([0x871F_FF1F, 0xF00F_FFFF]); // s_and_b32 s31, s31, !tiling
    for (k, at) in [(29, 24), (30, 28), (31, 32)] {
        w.push(v_mov_b32(5, k));
        w.extend(buffer_store_dword(5, 8, at));
    }
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 10,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let memory = res.scalar.iter().filter(|s| s.1.param == resource::MEMORY).count();
    assert_eq!(memory, 11, "every access through the pointer is title memory");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        rect_list: false,
        copy_params: [0; 32],
        view_marks: false,
    })
    .expect("lower");
    assert!(l.memory && l.memory_writes && l.view_addresses);
    assert!(l.root_uavs.is_empty(), "the root UAV is read as title memory");
    let shader = HostShader {
        splits: Vec::new(),
        table_entries: l.table_entries,
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    shader
}

/// The page table for one copy of title memory at [`TITLE`] whose views
/// are at heap index `view` (null pair at `null`).
fn page_table(view: u32, null: u32) -> Vec<u32> {
    let mut t = vec![0u32; 4096];
    let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
    t[(hi << 4 | lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        t.extend([null, 0]);
    }
    let entry = 4096 + 2 * (4096 + (lo >> 16 & 0xFFF) as usize);
    t[entry..entry + 2].copy_from_slice(&[view, lo]);
    t
}

#[test]
fn title_memory_is_read_and_written_at_computed_addresses() {
    let shader = program();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut memory = vec![0u32; 256];
    memory[0] = 0x1111_1111;
    let pointer = TITLE + 0x100;
    memory[2] = pointer as u32;
    memory[3] = (pointer >> 32) as u32;
    // The next page (no copy) and another 256 MiB region (no block).
    let (next, far) = (TITLE + 0x1_0010, 0x55_0000_0040u64);
    memory[4..8].copy_from_slice(&[next as u32, (next >> 32) as u32, far as u32, (far >> 32) as u32]);
    memory[0x40] = 0x2222_2222;
    // The root UAV's address, and what it loads there.
    let root_uav = TITLE + 0x200;
    memory[(0x200 + 0x24) / 4] = 0x3333_3333;
    memory[0x80 / 4] = 1.0f32.to_bits();
    memory[0x84 / 4] = (-1.0f32).to_bits();
    let value = 0xCAFE_F00D;
    // Heap index 1 (the memory's view) has the title address, index 2 a
    // texture's size fields.
    let mut view_addresses = vec![0u32; 4 * 5];
    view_addresses[4..6].copy_from_slice(&[TITLE as u32, (TITLE >> 32) as u32]);
    let image = [0xC000_0000, 0x0957_C0F9, 0x9004_2FAC];
    view_addresses[9..12].copy_from_slice(&image);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        // fmin with 1.0 and fmax with -1.0 in memory, against a negative
        // and a positive operand.
        for operand in [-2.0f32, 0.5] {
            let mut ud = vec![0u32; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                let constants = [TITLE as u32, (TITLE >> 32) as u32, value, operand.to_bits()];
                ud[u.index as usize] = match u.root {
                    RootType::Constants => constants[u.dword as usize],
                    RootType::Descriptor => (root_uav >> (32 * u.dword)) as u32,
                    _ => 0,
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (1, Root::View { srv: true, data: page_table(1, 3) }),
                            (2, Root::Uav(2 * 4096 / 32)),
                            (3, Root::View { srv: true, data: view_addresses.clone() }),
                        ],
                        views: &[
                            View::RawUav(9),
                            View::Memory(&memory),
                            View::Pair,
                            View::NullPair,
                            View::Pair,
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(
                out[0],
                [0x1111_1111, 0x2222_2222, 0x2222_2222, 0, 0, 0x3333_3333, image[0], image[1], image[2]],
                "{label}: reads"
            );
            assert_eq!(out[1][0x40 / 4], value, "{label}: the store");
            assert_eq!(out[1][(0x200 + 0x20) / 4], value, "{label}: the root UAV store");
            let floats = [f32::from_bits(out[1][0x20]), f32::from_bits(out[1][0x21])];
            let want = [operand.min(1.0), operand.max(-1.0)];
            assert_eq!(floats, want, "{label}: float min and max");
            // The far region by its level-1 index, the page written, the
            // page missed after it; nothing else.
            let entry = 4096 + (TITLE as u32 >> 16 & 0xFFF) as usize;
            let marked: Vec<usize> = (0..out[2].len() * 32)
                .filter(|&b| out[2][b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(marked, [(far >> 28) as usize, entry, entry + 1], "{label}: pages");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A root SRV in a program that does not use title memory is read through
/// the host's view of its host copy (`emit`, "Root SRVs and root UAVs"): the
/// view's index and the root address's offset in it come as user data, and
/// the shader adds the offset. Lanes read 64 bytes apart from the root
/// address 16 bytes into a 1 KiB copy: those inside it read the copy, those
/// past its end read 0 (a host root descriptor read whatever lay there).
#[test]
fn root_views_read_their_copy_and_zeros_past_it() {
    const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
    }
    /// MUBUF `op` (offen) of `vdata` at `vaddr` through `s[srsrc:srsrc+3]`.
    const fn mubuf(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [0x38 << 26 | op << 18 | 1 << 12, 0x80 << 24 | (srsrc as u32 >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32]
    }
    const V_LSHLREV_B32: u32 = 0x1A;
    let mut w = vec![
        s_mov_b32(5, ZERO as u8), // s[4:5]: the table at s4
        s_mov_b32(0, 6),          // s[0:3]: the root SRV's buffer descriptor
        0x8801_FF07,
        0x0001_0000, // s_or_b32 s1, s7, stride 1
        0xBE82_03C1, // s_mov_b32 s2, -1
        0xBE83_03FF,
        0x1101_4FAC, // s_mov_b32 s3, (the console's buffer word 3)
    ];
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result UAV
    w.push(vop2(V_LSHLREV_B32, 1, 128 + 6, 0)); // v1 = lane << 6
    w.extend(mubuf(0x0C, 2, 1, 0)); // buffer_load_dword v2, v1, s[0:3] offen
    w.push(vop2(V_LSHLREV_B32, 3, 128 + 2, 0)); // v3 = lane << 2
    w.push(S_WAITCNT_ALL);
    w.extend(mubuf(0x1C, 2, 3, 8)); // buffer_store_dword v2, v3, s[8:11] offen
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 1,
                dwords: 4,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [32, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        rect_list: false,
        copy_params: [0; 32],
        view_marks: false,
    })
    .expect("lower");
    assert!(l.root_srvs.is_empty() && !l.memory, "no root descriptor, no title memory");
    let marks: Vec<(u32, u32)> = l
        .user_data
        .iter()
        .filter(|u| u.root == RootType::Descriptor)
        .map(|u| (u.index, u.dword))
        .collect();
    assert_eq!(marks, [(6, super::ROOT_SRV_VIEW), (7, super::ROOT_VIEW_OFFSET)]);
    let shader = HostShader {
        splits: Vec::new(),
        table_entries: l.table_entries,
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let copy: Vec<u32> = (0..256).map(|k| 0x1000 + k).collect();
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords() as usize];
        // The table at heap index 0, the copy's view at 1, the root address
        // 16 bytes into it.
        (ud[4], ud[6], ud[7]) = (0, 1, 16);
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::RawUav(32),
                        View::Buffer {
                            data: &copy,
                            stride: super::host::Stride::Raw,
                        },
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let want: Vec<u32> = (0..32).map(|lane| if lane < 16 { 0x1000 + 4 + 16 * lane } else { 0 }).collect();
        assert_eq!(out[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Writes through unordered buffer views mark their pages (`emit`, "Write
/// marks"): a dword store and a 16-byte store through view 0 (base bit 100,
/// its first byte 16 bytes before a 64 KiB page ends, so the second store
/// crosses into the next page), a store through view 1 (not marked) and an
/// atomic add through view 2 (base bit 200). The stores land; exactly bits
/// 100, 101 and 200 are set.
#[test]
fn writes_through_views_mark_their_pages() {
    /// MUBUF `op` of `vdata` at `byte_off` through `s[srsrc:srsrc+3]`.
    const fn mubuf(op: u32, vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
        [
            0x38 << 26 | op << 18 | (byte_off as u32 & 0xFFF),
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
        ]
    }
    const BUFFER_STORE_DWORDX4: u32 = 0x1E;
    const BUFFER_ATOMIC_ADD: u32 = 0x32;
    let mut w = vec![s_mov_b32(5, ZERO as u8)];
    for (k, sdata) in [8u8, 12, 16].into_iter().enumerate() {
        w.extend(s_load(2, sdata, 4, 32 * k as u32));
    }
    w.push(S_WAITCNT_ALL);
    w.extend((0..4).map(|v| v_mov_b32(v, 0)));
    w.extend(buffer_store_dword(0, 8, 0));
    w.extend(mubuf(BUFFER_STORE_DWORDX4, 0, 8, 12));
    w.extend(buffer_store_dword(0, 12, 4));
    w.extend(mubuf(BUFFER_ATOMIC_ADD, 0, 16, 8));
    w.push(S_ENDPGM);
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let uav = |k: u32| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 8 * k,
        lower_bound: k,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![param(RootType::Constants, 0, 4, 0), param(RootType::Table, 4, 1, 1)],
        used: (0..3).map(uav).collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        rect_list: false,
        copy_params: [0; 32],
        view_marks: true,
    })
    .expect("lower");
    assert!(l.view_marks && !l.memory, "marks without the page table");
    let shader = HostShader {
        splits: Vec::new(),
        table_entries: l.table_entries,
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let value = 0x1234_5678;
    let marks = vec![100, 0xFFF0, super::VIEW_MARK_NONE, 0, 200, 0];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords() as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = if u.root == RootType::Constants { value } else { 0 };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (1, Root::View { srv: true, data: marks.clone() }),
                        (2, Root::Uav(8)),
                    ],
                    views: &[View::RawUav(8), View::RawUav(8), View::RawUav(8)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(out[0], [value, 0, 0, value, value, value, value, 0], "{label}: view 0");
        assert_eq!(out[1][1], value, "{label}: view 1");
        assert_eq!(out[2][2], value, "{label}: the atomic");
        let marked: Vec<usize> = (0..out[3].len() * 32)
            .filter(|&b| out[3][b / 32] >> (b % 32) & 1 != 0)
            .collect();
        assert_eq!(marked, [100, 101, 200], "{label}: pages");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Reads of texture views the program takes from the resource heap itself
/// mark their heap index (`emit`, "Read marks"): an `image_load` of the view
/// at heap byte offset 32 (index 1) sets exactly the bit one past
/// [`super::VIEW_READS`]. The same read through a table marks nothing (a
/// table read makes its slot on the host).
#[test]
fn heap_direct_texture_reads_mark_their_heap_index() {
    let lower_with = |root: RootType, used: Vec<UsedResource>, offset: u32| {
        let mut w = vec![s_mov_b32(5, ZERO as u8)];
        w.extend(s_load(3, 12, 4, offset));
        w.push(S_WAITCNT_ALL);
        w.extend((0..2).map(|v| v_mov_b32(v, ZERO)));
        // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        w.extend([0xF000_1F08, 0x0003_0200]);
        w.push(S_WAITCNT_ALL);
        w.push(S_ENDPGM);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 0,
            }],
            used,
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        let io = io::raise(&p, &native).expect("io::raise");
        lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: true,
            atomic_locks: false,
            bias_as_gradients: false,
            narrow_waves: false,
            wave32_host: false,
            zero_copy_memory: false,
            rect_list: false,
            copy_params: [0; 32],
        })
        .expect("lower")
    };
    let table = lower_with(
        RootType::Table,
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 4,
            slot: 0,
        }],
        0,
    );
    assert!(!table.view_marks, "a table read marks nothing");
    let l = lower_with(RootType::ResourceHeap, Vec::new(), 32);
    assert!(l.view_marks && l.heap, "a heap-direct read is marked");
    let shader = HostShader {
        splits: Vec::new(),
        table_entries: l.table_entries,
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let words = (super::VIEW_READS + 64) / 32;
    let texel = [[0.5f32; 4]];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        // The heap's base is 0 in the user data (`emit`, "Binding").
        let ud = vec![0u32; shader.user_data_dwords() as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (1, Root::View { srv: true, data: vec![super::VIEW_MARK_NONE; 4] }),
                        (2, Root::Uav(words)),
                    ],
                    views: &[View::Texture(1, 1, &texel), View::Texture(1, 1, &texel)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let bits = &out[0];
        let marked: Vec<usize> = (0..bits.len() * 32).filter(|&b| bits[b / 32] >> (b % 32) & 1 != 0).collect();
        assert_eq!(marked, [super::VIEW_READS as usize + 1], "{label}: the slot read");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn computed_buffer_subword_loads_preserve_values_and_bounds() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // Two separately backed pages across a 4 GiB boundary. A halfword at
    // PAGE + 0xffff must use both page-table entries, carrying into address_hi.
    const PAGE: u64 = 0x12_FFFF_0000;
    let first = vec![0xFF80_7F01u32; 0x1_0000 / 4];
    let second = vec![0x017F_FF80u32; 0x1_0000 / 4];
    let data: Vec<u8> = first.iter().chain(&second).flat_map(|w| w.to_le_bytes()).collect();
    let mut table = vec![0u32; 4096];
    for _ in 0..3 * 4096 {
        table.extend([5, 0]); // null pair
    }
    for (page, block, view) in [(PAGE, 1, 1), (PAGE + 0x1_0000, 2, 3)] {
        table[(page >> 28) as usize] = block;
        let entry = 4096 + 2 * (block as usize * 4096 + (page as usize >> 16 & 0xFFF));
        table[entry..entry + 2].copy_from_slice(&[view, page as u32]);
    }
    // Explicit acceptance masks (bit = OOB_SELECT), reviewed against RDNA2
    // table 35, rather than repeating the emitter's bounds expressions.
    // Mode 0 checks the start offset: a halfword starting at stride-1 is
    // accepted. Mode 3 checks the payload end and rejects that halfword.
    // (base byte, stride, records, index, vector offset, scalar offset,
    //  byte acceptance mask, halfword acceptance mask)
    let cases = [
        (0, 4, 4, 0, 0, 0, 0b1111, 0b1111),
        (0, 4, 4, 0, 1, 0, 0b1111, 0b1111),
        (0, 4, 4, 0, 2, 0, 0b1111, 0b1111),
        (0, 4, 4, 0, 3, 0, 0b1111, 0b0111),
        (0, 4, 4, 0, 4, 0, 0b0110, 0b0110),
        (0, 4, 4, 1, 0, 0, 0b0111, 0b0111),
        (0, 4, 4, 4, 0, 0, 0b0100, 0b0100),
        (0, 4, 0, 0, 0, 0, 0b0000, 0b0000),
        (0, 4, 1, 0, 0, 0, 0b1111, 0b0111),
        // Scalar offsets are excluded, as documented by LLVM's buffer intrinsics.
        (0, 4, 4, 0, 0, 3, 0b1111, 0b1111),
        (0xFFFC, 4, 4, 0, 3, 0, 0b1111, 0b0111),
        (0xFFFC, 4, 8, 1, 0, 0, 0b1111, 0b1111),
        (0x1_FFFC, 4, 4, 0, 3, 0, 0b1111, 0b0111),
    ];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for op in 8..=11 {
            for bounds in 0..4 {
                let shader = super::gfx_refusal_tests::computed_subword_test_shader(op, bounds);
                if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
                    v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
                }
                let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
                for (base, stride, records, index, offset, scalar, byte_modes, half_modes) in cases {
                    let address = PAGE + u64::from(base);
                    let constants = [address as u32, (address >> 32) as u32 | stride << 16,
                        records, index, offset, scalar];
                    let mut ud = vec![0; shader.user_data_dwords() as usize];
                    for u in &shader.user_data {
                        if u.root == RootType::Constants {
                            ud[u.index as usize] = constants[u.dword as usize];
                        }
                    }
                    let out = host.dispatch(&root, &pso, &Dispatch {
                        roots: vec![(0, Root::Constants(ud)),
                            (1, Root::View { srv: true, data: table.clone() })],
                        views: &[View::RawUav(1), View::Memory(&first), View::Pair,
                            View::Memory(&second), View::Pair, View::NullPair, View::Pair],
                        samplers: &[], groups: [1, 1, 1],
                    }).unwrap_or_else(|e| panic!("{label}: {e}"));
                    let byte_offset = index * stride + offset;
                    let modes = if op < 10 { byte_modes } else { half_modes };
                    let valid = modes & (1 << bounds) != 0;
                    let mut expected = 0u32;
                    if valid {
                        let at = (base + scalar + byte_offset) as usize;
                        let low = data.get(at).copied().unwrap_or(0);
                        let high = data.get(at + 1).copied().unwrap_or(0);
                        expected = match op {
                            8 => u32::from(low),
                            9 => i32::from(low as i8) as u32,
                            10 => u32::from(u16::from_le_bytes([low, high])),
                            11 => i32::from(i16::from_le_bytes([low, high])) as u32,
                            _ => unreachable!(),
                        };
                    }
                    assert_eq!(out[0][0], expected,
                        "{label}: op={op} bounds={bounds} base={base:#x} index={index} offset={offset} scalar={scalar}");
                }
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
