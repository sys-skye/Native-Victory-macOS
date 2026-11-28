//! The pool-offset contract on Metal ([`super::addr`]): synthetic GFX10 programs of our own,
//! assembled with `llvm-mc -arch=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize64`, translated with the
//! host's options, converted by Apple's Metal Shader Converter with the host's root layout and run on
//! the GPU. They read and write title memory at addresses they compute (a global load, a scalar load
//! through a pointer loaded from memory, global stores through an SGPR base and through a VGPR pair,
//! an atomic), at title addresses no CPU pointer is at: the pool is shown to the title at addresses
//! the process never mapped, and the results are read from the pool's own mapping. Skipped without
//! the Metal Shader Converter.

use crate::native::list::NativeList;
use crate::native::memory::{PoolInfo, PoolObserver};
use crate::native::shaders;
use crate::native::state::Graphics;
use crate::native::title::{Pipe, Title};
use std::sync::Arc;
use victory_backend::resources::{Format, HeapKind, ResourceBackend, TextureDesc, Topology};
use victory_metal::mem::Alias;
use victory_metal::metal::Metal;
use victory_shader::emit::{HostShader, emit_with};
use victory_shader::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use victory_shader::{Container, io, ir, resource, structure};

/// Every lane: `dst[tid] = src[tid] + **(src + 0x1000)`; the same again at `dst + 1024` through a
/// 64-bit VGPR address; `dst[+2040] += 1`. `s[0:1]` is `src`, `s[2:3]` is `dst`.
const COMPUTED: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
    0xDC30_8000,
    0x0200_0001, // global_load_dword v2, v1, s[0:1]
    0xF404_0100,
    0xFA00_1000, // s_load_dwordx2 s[4:5], s[0:1], 0x1000
    0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
    0xF400_0182,
    0xFA00_0000, // s_load_dword s6, s[4:5], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x4A04_0406, // v_add_nc_u32_e32 v2, s6, v2
    0xDC70_8000,
    0x0002_0201, // global_store_dword v1, v2, s[2:3]
    0x7E0C_0202, // v_mov_b32_e32 v6, s2
    0x7E0E_0203, // v_mov_b32_e32 v7, s3
    0xD70F_6A06,
    0x0002_0306, // v_add_co_u32 v6, vcc, v6, v1
    0x500E_0E80, // v_add_co_ci_u32_e32 v7, vcc, 0, v7, vcc
    0xDC70_8400,
    0x007D_0206, // global_store_dword v[6:7], v2, off offset:1024
    0x7E10_0280, // v_mov_b32_e32 v8, 0
    0x7E12_0281, // v_mov_b32_e32 v9, 1
    0xDCC8_87F8,
    0x0002_0908, // global_atomic_add v8, v9, s[2:3] offset:2040
    0xBF81_0000, // s_endpgm
];

/// `s_absdiff_i32 s4, s2, s3` and its SCC, stored by every lane to `s[0:1]` and 256 bytes on.
const ABSDIFF: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
    0x9604_0302, // s_absdiff_i32 s4, s2, s3
    0x8505_8081, // s_cselect_b32 s5, 1, 0
    0x7E04_0204, // v_mov_b32_e32 v2, s4
    0x7E06_0205, // v_mov_b32_e32 v3, s5
    0xDC70_8000,
    0x0000_0201, // global_store_dword v1, v2, s[0:1]
    0xDC70_8100,
    0x0000_0301, // global_store_dword v1, v3, s[0:1] offset:256
    0xBF81_0000, // s_endpgm
];

/// `dst[tid] = src[tid] + 0x55` stored to `s[2:3]`, and the same value stored to `s[4:5]`, where
/// the pointers may point at memory no pool shows.
const MISSES: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
    0xDC30_8000,
    0x0200_0001, // global_load_dword v2, v1, s[0:1]
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x4A04_04FF,
    0x0000_0055, // v_add_nc_u32_e32 v2, 0x55, v2
    0xDC70_8000,
    0x0002_0201, // global_store_dword v1, v2, s[2:3]
    0xDC70_8000,
    0x0004_0201, // global_store_dword v1, v2, s[4:5]
    0xBF81_0000, // s_endpgm
];

/// A wave64 lane mask joined with data at a uniform loop's header: lanes below 40, then on each
/// back edge that number shifted right by 4 and passed through VGPRs (data). Every lane stores its
/// bit of the last pass's number to `s[0:1]`: lane < 36, the low dword's for lanes below 32 and
/// the high dword's above.
const MASK_DATA_JOIN: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
    0x7D88_00A8, // v_cmp_gt_u32_e32 vcc, 40, v0
    0x8786_7E6A, // s_and_b64 s[6:7], vcc, exec
    0xBE88_0380, // s_mov_b32 s8, 0
    0xD501_0004,
    0x0019_0280, // header: v_cndmask_b32_e64 v4, 0, 1, s[6:7]
    0x9086_8406, // s_lshr_b64 s[6:7], s[6:7], 4
    0x7E0A_0206, // v_mov_b32_e32 v5, s6
    0x7E0C_0207, // v_mov_b32_e32 v6, s7
    0x7E0C_0505, // v_readfirstlane_b32 s6, v5
    0x7E0E_0506, // v_readfirstlane_b32 s7, v6
    0x8008_8108, // s_add_u32 s8, s8, 1
    0xBF0A_8208, // s_cmp_lt_u32 s8, 2
    0xBF85_FFF6, // s_cbranch_scc1 header
    0xDC70_8000,
    0x0000_0401, // global_store_dword v1, v4, s[0:1]
    0xBF81_0000, // s_endpgm
];

/// A 64-lane compute shader whose user SGPRs `s0..s<sgprs>` are root constants, translated the way
/// the host translates, from machine code of our own in a container of no parts.
fn shader(words: &[u32], sgprs: u32) -> HostShader {
    translate(words, sgprs, shaders::options())
}

fn translate(words: &[u32], sgprs: u32, options: victory_shader::emit::Options) -> HostShader {
    translate_io(words, sgprs, 0, 64, options)
}

/// [`translate`] with more of `COMPUTE_PGM_RSRC2` set (`rsrc2`: LDS, system SGPRs) and `threads` in
/// a group.
pub(super) fn translate_io(
    words: &[u32],
    sgprs: u32,
    rsrc2: u32,
    threads: u32,
    options: victory_shader::emit::Options,
) -> HostShader {
    translate_wave(words, sgprs, rsrc2, threads, true, options)
}

/// [`translate_io`] of a program of 64-lane waves (`wave64`) or 32, structured for dispatches of
/// thread counts when `options` ask for them (as `translate_with` does).
pub(super) fn translate_wave(
    words: &[u32],
    sgprs: u32,
    rsrc2: u32,
    threads: u32,
    wave64: bool,
    options: victory_shader::emit::Options,
) -> HostShader {
    try_translate_wave(words, sgprs, rsrc2, threads, wave64, options).expect("emit")
}

/// [`translate_wave`], or what the translation refuses.
pub(super) fn try_translate_wave(
    words: &[u32],
    sgprs: u32,
    rsrc2: u32,
    threads: u32,
    wave64: bool,
    options: victory_shader::emit::Options,
) -> Result<HostShader, victory_shader::ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: sgprs << 1 | rsrc2,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let mut p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    p.thread_counts = options.thread_counts;
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    // A container of no parts: `DXBC`, a digest, version 1.0, its size, no parts.
    let mut empty = b"DXBC".to_vec();
    empty.extend([0; 16]);
    empty.extend([1, 0, 0, 0]);
    empty.extend(32u32.to_le_bytes());
    empty.extend(0u32.to_le_bytes());
    let c = Container::parse(&empty).expect("container");
    emit_with(&c, &native, &p, &st, &io, &res, options)
}

pub(super) fn title() -> Option<Title<Metal>> {
    if !std::path::Path::new(victory_metal::msc::DEFAULT_DYLIB).exists()
        && std::env::var_os("VICTORY_MSC_DYLIB").is_none()
    {
        eprintln!("skipped: Metal Shader Converter not installed");
        return None;
    }
    Some(Title::new(
        Metal::new().unwrap(),
        Arc::new(crate::native::memory::PoolTable::default()),
    ))
}

/// Title addresses nothing in this process is mapped at.
pub(super) const VA_A: u64 = 0x12_3450_0000;
const VA_B: u64 = 0x7000_0000;
const POOL: usize = 0x10_0000;
pub(super) const CANARY: u32 = 0xEEEE_EEEE;

/// A pool of 1 MiB filled with a canary, shown at `VA_A` (512 KiB from pool offset 0x2_0000) and
/// `VA_B` (128 KiB from the pool's start).
pub(super) struct Fixture {
    mem: Alias,
    info: PoolInfo,
}

impl Fixture {
    pub(super) fn new(t: &Title<Metal>) -> Fixture {
        Fixture::with_id(t, 1)
    }

    fn with_id(t: &Title<Metal>, id: usize) -> Fixture {
        let mem = Alias::new(POOL).unwrap();
        let info = PoolInfo {
            id,
            base: mem.primary as usize,
            len: POOL,
            memory_entry: None,
        };
        let f = Fixture { mem, info };
        f.put(0, &[CANARY; POOL / 4]);
        let m = t.host.memory();
        m.pool_created(&f.info);
        m.mapped(VA_A as usize, 0x8_0000, &f.info, 0x2_0000);
        m.mapped(VA_B as usize, 0x2_0000, &f.info, 0);
        f
    }

    pub(super) fn put<T: Copy>(&self, at: usize, v: &[T]) {
        // SAFETY: inside the pool's own mapping; the GPU is idle.
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), self.mem.primary.add(at).cast(), v.len()) };
    }

    pub(super) fn get(&self, at: usize, n: usize) -> Vec<u32> {
        // SAFETY: inside the pool's own mapping; read after the fence wait.
        unsafe { std::slice::from_raw_parts(self.mem.primary.add(at).cast::<u32>(), n).to_vec() }
    }
}

pub(super) fn dispatch(t: &mut Title<Metal>, shader: &HostShader, root: &[u32], heap: bool) {
    let Pipe::Compute(pso) = t.compute_pipe(shader, "addr") else {
        panic!("not made: {:?}", t.shaders)
    };
    assert_eq!(pso.threads[1..], [1, 1]);
    let mut l = NativeList::default();
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, root);
    if heap {
        // The title's own resource heap, 8 views of 32 bytes of its descriptor memory.
        let slots = vec![0u32; 8 * 8];
        let h = t.host.heap(HeapKind::Resource, slots.as_ptr() as u64, 8).unwrap();
        l.set_heaps(Some(&h), None);
    }
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
}

pub(super) fn halves(v: u64) -> [u32; 2] {
    [v as u32, (v >> 32) as u32]
}

/// Expectations of [`COMPUTED`] with `src` at pool offset `src_at`, `dst` at `dst_at`.
fn check_computed(f: &Fixture, src_at: usize, dst_at: usize) {
    let first = f.get(dst_at, 64);
    let second = f.get(dst_at + 1024, 64);
    for i in 0..64u32 {
        let want = i * 3 + 1 + 0xBEEF;
        assert_eq!(first[i as usize], want, "dst[{i}]");
        assert_eq!(second[i as usize], want, "dst2[{i}]");
    }
    assert_eq!(f.get(dst_at + 2040, 1), [64], "the atomic");
    // Nothing else was written: the source, the pointer's target and the stretch between.
    assert_eq!(f.get(src_at, 64), (0..64).map(|i| i * 3 + 1).collect::<Vec<u32>>());
    assert_eq!(f.get(dst_at + 256, 192), [CANARY; 192]);
    assert_eq!(f.get(dst_at + 1280, 190), [CANARY; 190]);
    assert_eq!(f.get(dst_at + 2044, 1), [CANARY]);
}

fn prepare_computed(f: &Fixture, src_at: usize, dst_at: usize) {
    f.put(src_at, &(0..64u32).map(|i| i * 3 + 1).collect::<Vec<_>>());
    // At src + 0x1000, a pointer to VA_B + 0x100, where there is 0xBEEF.
    f.put(src_at + 0x1000, &[VA_B + 0x100]);
    f.put(0x100, &[0xBEEFu32]);
    f.put(dst_at + 2040, &[0u32]);
}

#[test]
fn computed_addresses_read_and_write_the_pool_at_its_title_addresses() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = shader(COMPUTED, 4);
    // The translation as the host asks for it: the page table and nothing to mark.
    assert!(cs.memory && !cs.memory_writes && !cs.view_addresses && !cs.heap);
    prepare_computed(&f, 0x2_0000, 0x6_0000);
    let [a, b] = halves(VA_A);
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs, &[a, b, c, d], false);
    check_computed(&f, 0x2_0000, 0x6_0000);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

/// `s_absdiff_i32` on the GPU: |S0 - S1| of the 32-bit (wrapping) difference read as signed,
/// SCC whether it is not 0 (RDNA2 ISA, SOP2).
#[test]
fn absdiff_is_the_magnitude_of_the_signed_difference_on_the_gpu() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = shader(ABSDIFF, 4);
    let [lo, hi] = halves(VA_A);
    for (a, b, want) in [
        (5i32, 9i32, 4u32),
        (9, 5, 4),
        (-3, 4, 7),
        (7, 7, 0),
        // The difference wraps: i32::MIN - 1 is i32::MAX; 1 - i32::MIN is 0x8000_0001, negative,
        // whose magnitude is i32::MAX again.
        (i32::MIN, 1, i32::MAX as u32),
        (1, i32::MIN, i32::MAX as u32),
        (i32::MIN, 0, 0x8000_0000),
    ] {
        dispatch(&mut t, &cs, &[lo, hi, a as u32, b as u32], false);
        assert_eq!(f.get(0x2_0000, 64), vec![want; 64], "|{a} - {b}|");
        assert_eq!(f.get(0x2_0100, 64), vec![u32::from(want != 0); 64], "SCC of |{a} - {b}|");
    }
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

/// `s_sext_i32_i8` and `s_sext_i32_i16` of s2 on the GPU: the low byte and the low half
/// sign-extended to 32 bits (RDNA2 ISA, SOP1), stored to dst and dst + 256.
#[test]
fn scalar_sign_extensions_on_the_gpu() {
    const SEXT: &[u32] = &[
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0xBE84_1902, // s_sext_i32_i8 s4, s2
        0xBE85_1A02, // s_sext_i32_i16 s5, s2
        0x7E04_0204, // v_mov_b32_e32 v2, s4
        0x7E06_0205, // v_mov_b32_e32 v3, s5
        0xDC70_8000, 0x0000_0201, // global_store_dword v1, v2, s[0:1]
        0xDC70_8100, 0x0000_0301, // global_store_dword v1, v3, s[0:1] offset:256
        0xBF81_0000, // s_endpgm
    ];
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = shader(SEXT, 3);
    let [lo, hi] = halves(VA_A);
    for (x, i8, i16) in [
        (0x1234_80F0u32, 0xFFFF_FFF0u32, 0xFFFF_80F0u32),
        (0xFFFF_7F70, 0x70, 0x7F70),
        (0x0000_0080, 0xFFFF_FF80, 0x80),
    ] {
        dispatch(&mut t, &cs, &[lo, hi, x], false);
        assert_eq!(f.get(0x2_0000, 64), vec![i8; 64], "s_sext_i32_i8 {x:#x}");
        assert_eq!(f.get(0x2_0100, 64), vec![i16; 64], "s_sext_i32_i16 {x:#x}");
    }
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

/// A VALU compare with DPP on the GPU: `v_cmp_gt_f32 vcc, v2 quad_perm:[2,3,0,1], v2` of
/// v2 = the lane index compares the value two lanes over in the quad with the lane's own, so
/// lanes 0 and 1 of each quad hold 1 and lanes 2 and 3 hold 0.
#[test]
fn a_compare_with_dpp_reads_the_quad_neighbour_on_the_gpu() {
    const DPP_CMP: &[u32] = &[
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7C08_04FA, 0xFF0C_4E02, // v_cmp_gt_f32 vcc, v2, v2 quad_perm:[2,3,0,1] bound_ctrl:1 fi:1
        0xD501_0002, 0x01A9_0280, // v_cndmask_b32 v2, 0, 1, vcc
        0xDC70_8000, 0x0000_0201, // global_store_dword v1, v2, s[0:1]
        0xBF81_0000, // s_endpgm
    ];
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = shader(DPP_CMP, 2);
    let [lo, hi] = halves(VA_A);
    dispatch(&mut t, &cs, &[lo, hi], false);
    let want: Vec<u32> = (0..64).map(|i| u32::from(i % 4 < 2)).collect();
    assert_eq!(f.get(0x2_0000, 64), want);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

/// A per-lane `buffer_load_dword` through a buffer descriptor the program builds around its
/// constant table (`s_getpc_b64` plus an offset, 16 bytes in dword 2), as the console compiler
/// does for spline tables: lane i reads dword i & 7 of the table, and the four past its 16 bytes
/// read 0. Before, the load was refused and its value dropped, so a program using it ran on
/// without the table (a colour grading LUT of one value).
#[test]
fn a_buffer_load_through_a_descriptor_around_the_code_table_reads_it_on_the_gpu() {
    const CODE_TABLE: &[u32] = &[
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0xBE84_1F00, // s_getpc_b64 s[4:5] (8)
        0x8004_FF04, 56, // s_add_u32 s4, s4, 56: the table at 64
        0x8205_8005, // s_addc_u32 s5, s5, 0
        0xBE86_0390, // s_mov_b32 s6, 16
        0xBE87_03FF, 0x3101_6FAC, // s_mov_b32 s7, 0x31016fac (raw, OOB_SELECT 3)
        0x3604_0087, // v_and_b32_e32 v2, 7, v0
        0x3404_0482, // v_lshlrev_b32_e32 v2, 2, v2
        0xE030_1000, 0x8001_0302, // buffer_load_dword v3, v2, s[4:7], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xDC70_8000, 0x0000_0301, // global_store_dword v1, v3, s[0:1]
        0xBF81_0000, // s_endpgm
        11, 22, 33, 44, // the table
    ];
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = shader(CODE_TABLE, 2);
    let [lo, hi] = halves(VA_A);
    dispatch(&mut t, &cs, &[lo, hi], false);
    let want: Vec<u32> = (0..64).map(|i| [11, 22, 33, 44, 0, 0, 0, 0][i % 8]).collect();
    assert_eq!(f.get(0x2_0000, 64), want);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

#[test]
fn compares_with_16bit_dpp_vcmpx_and_class_run_on_the_gpu() {
    // Input words at `VA_A + 0x1000` (`s[2:3]`): each lane loads its own pre-built dword into `v2`
    // (with junk in the upper 16 bits to verify `half_src` masks low 16 bits), and writes its
    // result to `VA_A` (`s[0:1]`). Expected outputs are fixed constants, independent of lowering.
    //
    // 1. `v_cmp_gt_f16 vcc, v2, v2 quad_perm:[2,3,0,1]`:
    //    Quad inputs `[1.0h, 2.0h, 4.0h, 0.5h]` -> permuted `[4.0h, 0.5h, 1.0h, 2.0h]`:
    //    `4.0 > 1.0` (1), `0.5 > 2.0` (0), `1.0 > 4.0` (0), `2.0 > 0.5` (1) -> `[1, 0, 0, 1]`.
    const DPP_CMP_F16: &[u32] = &[
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0xDC30_8000,
        0x0202_0001, // global_load_dword v2, v1, s[2:3]
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7D98_04FA,
        0xFF0C_4E02, // v_cmp_gt_f16 vcc, v2, v2 quad_perm:[2,3,0,1] bound_ctrl:1 fi:1
        0xD501_0002,
        0x01A9_0280, // v_cndmask_b32 v2, 0, 1, vcc
        0xDC70_8000,
        0x0000_0201, // global_store_dword v1, v2, s[0:1]
        0xBF81_0000, // s_endpgm
    ];
    // 2. `v_cmp_lt_i16 vcc, v2, v2 row_shr:1 bound_ctrl:1`:
    //    Alternating `[-5i16 (0xFFFB), +7i16 (0x0007)]`:
    //    Row-start lanes `0, 16, 32, 48` compare `0 < -5` (0); other even lanes compare `7 < -5`
    //    (0); odd lanes compare `-5 < 7` (1) -> `[0, 1]` repeated across all 64 lanes.
    const DPP_CMP_I16: &[u32] = &[
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0xDC30_8000,
        0x0202_0001, // global_load_dword v2, v1, s[2:3]
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7D12_04FA,
        0xFF08_1102, // v_cmp_lt_i16 vcc, v2, v2 row_shr:1 bound_ctrl:1
        0xD501_0002,
        0x01A9_0280, // v_cndmask_b32 v2, 0, 1, vcc
        0xDC70_8000,
        0x0000_0201, // global_store_dword v1, v2, s[0:1]
        0xBF81_0000, // s_endpgm
    ];
    // 3. `v_cmp_class_f16 vcc, v2, v3 quad_perm:[3,2,1,0]`:
    //    Quad inputs `[1.0h (0x3C00), +inf (0x7C00), -inf (0xFC00), -0.0h (0x8000)]`, class mask
    //    `0x204` (`+inf` bit 9 | `-inf` bit 2). Reversed quad `[-0.0h, -inf, +inf, 1.0h]` matches
    //    `+-inf` in lanes 1 and 2 of each quad -> `[0, 1, 1, 0]`.
    const DPP_CLASS_F16: &[u32] = &[
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0xDC30_8000,
        0x0202_0001, // global_load_dword v2, v1, s[2:3]
        0x7E06_02FF,
        0x0000_0204, // v_mov_b32 v3, 0x204
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7D1E_06FA,
        0xFF0C_1B02, // v_cmp_class_f16 vcc, v2, v3 quad_perm:[3,2,1,0] bound_ctrl:1 fi:1
        0xD501_0002,
        0x01A9_0280, // v_cndmask_b32 v2, 0, 1, vcc
        0xDC70_8000,
        0x0000_0201, // global_store_dword v1, v2, s[0:1]
        0xBF81_0000, // s_endpgm
    ];
    // 4. `v_cmpx_eq_u16 v2, v2 quad_perm:[1,0,3,2]`:
    //    Quad inputs `[11u16, 11u16, 22u16, 33u16]` -> swapped `[11, 11, 33, 22]`: only lanes 0
    //    and 1 of each quad stay active in `EXEC` and store `77`; lanes 2 and 3 keep `CANARY`.
    const DPP_CMPX_U16: &[u32] = &[
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0xDC30_8000,
        0x0202_0001, // global_load_dword v2, v1, s[2:3]
        0x7E06_02FF,
        0x0000_004D, // v_mov_b32 v3, 77
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xBE84_047E, // s_mov_b64 s[4:5], exec
        0x7D74_04FA,
        0xFF0C_B102, // v_cmpx_eq_u16 v2, v2 quad_perm:[1,0,3,2] bound_ctrl:1 fi:1
        0xDC70_8000,
        0x0000_0301, // global_store_dword v1, v3, s[0:1]
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xBF81_0000, // s_endpgm
    ];

    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let [out_lo, out_hi] = halves(VA_A);
    let [in_lo, in_hi] = halves(VA_A + 0x1000);
    let args = [out_lo, out_hi, in_lo, in_hi];

    f.put(0x2_0000, &[CANARY; 64]);
    let in_f16: Vec<u32> = [0xCAFE_3C00u32, 0xBEEF_4000, 0x1234_4400, 0x9999_3800].repeat(16);
    f.put(0x2_1000, &in_f16);
    let cs_f16 = shader(DPP_CMP_F16, 4);
    dispatch(&mut t, &cs_f16, &args, false);
    assert_eq!(f.get(0x2_0000, 64), [1u32, 0, 0, 1].repeat(16));

    f.put(0x2_0000, &[CANARY; 64]);
    let in_i16: Vec<u32> = [0xDEAD_FFFBu32, 0x0001_0007].repeat(32);
    f.put(0x2_1000, &in_i16);
    let cs_i16 = shader(DPP_CMP_I16, 4);
    dispatch(&mut t, &cs_i16, &args, false);
    assert_eq!(f.get(0x2_0000, 64), [0u32, 1].repeat(32));

    f.put(0x2_0000, &[CANARY; 64]);
    let in_class: Vec<u32> = [0x7FFF_3C00u32, 0x0000_7C00, 0x1111_FC00, 0xFFFF_8000].repeat(16);
    f.put(0x2_1000, &in_class);
    let cs_class = shader(DPP_CLASS_F16, 4);
    dispatch(&mut t, &cs_class, &args, false);
    assert_eq!(f.get(0x2_0000, 64), [0u32, 1, 1, 0].repeat(16));

    f.put(0x2_0000, &[CANARY; 64]);
    let in_cmpx: Vec<u32> = [0xAAAA_000Bu32, 0xBBBB_000B, 0xCCCC_0016, 0xDDDD_0021].repeat(16);
    f.put(0x2_1000, &in_cmpx);
    let cs_cmpx = shader(DPP_CMPX_U16, 4);
    dispatch(&mut t, &cs_cmpx, &args, false);
    assert_eq!(f.get(0x2_0000, 64), [77u32, 77, CANARY, CANARY].repeat(16));

    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

#[test]
fn guessed_fragment_buffer_reads_the_pool_and_checks_record_bounds() {
    let vertex = SCREEN_TRIANGLE;
    let pixel = [
        0x8004_9000, // s_add_u32 s4, s0, 16
        0x8205_8080, // s_addc_u32 s5, 0, 0
        0x8805_FF05, 0x0010_0000, // s_or_b32 s5, s5, 0x100000
        0xBE86_0382, // s_mov_b32 s6, 2
        0xBE87_03FF, 0x0001_4FAC, // s_mov_b32 s7, 0x14fac
        0x7E00_0281, // v_mov_b32 v0, 1
        0xE030_2004, 0x8001_0200, // buffer_load_dword v2, v0, s[4:7], 0 idxen offset:4
        0xE030_2010, 0x8001_0300, // buffer_load_dword v3, v0, s[4:7], 0 idxen offset:16
        0x7E00_0282, // v_mov_b32 v0, 2
        0xE030_2000, 0x8001_0400, // buffer_load_dword v4, v0, s[4:7], 0 idxen
        0xF420_0202, 0xFA00_0004, // s_buffer_load_dword s8, s[4:7], 4
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x7E0A_0208, // v_mov_b32 v5, s8
        0xF800_180F, 0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
        0xBF81_0000, // s_endpgm
    ];
    let translate = |stage, words: &[u32], meta: &[u8]| {
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let c = victory_shader::emit::graphics::container(stage, &bytes, meta).unwrap();
        shaders::translate(&c).unwrap()
    };
    let vs = translate(HwStage::Vs, &vertex, &[0; 0x10]);
    let mut meta = [0; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // Root constants in slot 0.
    let ps = translate(HwStage::Ps, &pixel, &meta);
    use victory_metal::msc::{DEFAULT_DYLIB, Msc};
    if !std::path::Path::new(DEFAULT_DYLIB).exists() && std::env::var_os("VICTORY_MSC_DYLIB").is_none() {
        eprintln!("skipped: Metal Shader Converter not installed");
        return;
    }
    let layout = crate::native::list::Binding::of(&ps).layout().unwrap();
    Msc::open_default().convert_with(&ps.dxil, None, Some(&layout), false).unwrap();
    let Ok(metal) = Metal::new() else {
        eprintln!("skipped the draw: no Metal device");
        return;
    };
    let mut t = Title::new(metal, Arc::new(crate::native::memory::PoolTable::default()));
    let f = Fixture::new(&t);
    f.put(20, &[1.0f32]); // base + 4
    f.put(36, &[0.5f32]); // base + 1 * stride + 4
    f.put(48, &[1.0f32]); // backing memory at the address of both out-of-bounds reads

    let state = Graphics::guess_base()
        .for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &state, "computed fragment") else {
        panic!("{:?}", t.shaders);
    };
    let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 8, 8);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    let mut list = NativeList::default();
    list.set_render_targets(std::slice::from_ref(&color), None);
    list.set_graphics_pipeline(&pso, Topology::Triangles);
    list.set_graphics_binding(binding);
    list.set_graphics_root_constants(0, &[VA_B as u32]);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    super::run(&t.host, &[&list]);
    let mut pixels = [0; 8 * 8 * 4];
    t.host.backend.read_texture(&color, 0, 0, &mut pixels, 8 * 4, 0);
    assert_eq!(&pixels[(4 * 8 + 4) * 4..][..4], [128, 0, 0, 255]);
}

/// A guessed vertex shader of the triangle over the whole target.
const SCREEN_TRIANGLE: [u32; 13] = [
    0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
    0xD501_0001, 0x01A9_FEF3, 0x4040_0000, // v_cndmask_b32 v1, -1.0, 3.0, vcc
    0x7D84_0082, // v_cmp_eq_u32 vcc, 2, v0
    0xD501_0002, 0x01A9_FEF3, 0x4040_0000, // v_cndmask_b32 v2, -1.0, 3.0, vcc
    0x7E06_0280, // v_mov_b32 v3, 0
    0x7E08_02F2, // v_mov_b32 v4, 1.0
    0xF800_08CF, 0x0403_0201, // exp pos0 v1, v2, v3, v4 done
    0xBF81_0000, // s_endpgm
];

/// A pixel program's wave maximum, finished by the compiler only in lanes 32..63, read at lane
/// 63 inside a later whole-wave section (and shifted there by a scalar step) and after it:
/// (39 >> 1, 39) out as unorm bytes.
#[test]
fn a_pixel_wave_maximum_read_at_its_last_lane_is_drawn() {
    let pixel = [
        0x7E00_02A7, // v_mov_b32_e32 v0, 39
        0xBE88_25C1, // s_or_saveexec_b64 s[8:9], -1
        0xD501_0001, 0x0022_0080, // v_cndmask_b32_e64 v1, 0, v0, s[8:9]
        0x2802_02FA, 0xFF04_B101, // v_max_u32_dpp v1, v1, v1 quad_perm:[1,0,3,2] fi:1
        0x2802_02FA, 0xFF04_4E01, // v_max_u32_dpp v1, v1, v1 quad_perm:[2,3,0,1] fi:1
        0x2802_02FA, 0xFF05_4101, // v_max_u32_dpp v1, v1, v1 row_half_mirror fi:1
        0x2802_02FA, 0xFF05_4001, // v_max_u32_dpp v1, v1, v1 row_mirror fi:1
        0xD778_0002, 0x0201_0101, // v_permlanex16_b32 v2, v1, 0, 0
        0x2802_0501, // v_max_u32_e32 v1, v1, v2
        0xD760_000A, 0x0201_0101, // v_readlane_b32 s10, v1, 0
        0x2802_020A, // v_max_u32_e32 v1, s10, v1
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
        0xD760_000E, 0x0201_7F01, // v_readlane_b32 s14, v1, 63
        0x900F_810E, // s_lshr_b32 s15, s14, 1
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0xD760_0010, 0x0201_7F01, // v_readlane_b32 s16, v1, 63
        0x7E04_0C0F, // v_cvt_f32_u32_e32 v2, s15
        0x7E06_0C10, // v_cvt_f32_u32_e32 v3, s16
        0x1004_04FF, 0x3B80_8081, // v_mul_f32_e32 v2, 1/255, v2
        0x1006_06FF, 0x3B80_8081, // v_mul_f32_e32 v3, 1/255, v3
        0x7E08_0280, // v_mov_b32_e32 v4, 0
        0x7E0A_02F2, // v_mov_b32_e32 v5, 1.0
        0xF800_180F, 0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
        0xBF81_0000, // s_endpgm
    ];
    let translate = |stage, words: &[u32]| {
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let c = victory_shader::emit::graphics::container(stage, &bytes, &[0; 0x10]).unwrap();
        shaders::translate(&c).unwrap()
    };
    let (vs, ps) = (translate(HwStage::Vs, &SCREEN_TRIANGLE), translate(HwStage::Ps, &pixel));
    use victory_metal::msc::{DEFAULT_DYLIB, Msc};
    if !std::path::Path::new(DEFAULT_DYLIB).exists() && std::env::var_os("VICTORY_MSC_DYLIB").is_none() {
        eprintln!("skipped: Metal Shader Converter not installed");
        return;
    }
    let Ok(metal) = Metal::new() else {
        eprintln!("skipped the draw: no Metal device");
        return;
    };
    let mut t = Title::new(metal, Arc::new(crate::native::memory::PoolTable::default()));
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &state, "wave maximum") else {
        panic!("{:?}", t.shaders);
    };
    let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 8, 8);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    let mut list = NativeList::default();
    list.set_render_targets(std::slice::from_ref(&color), None);
    list.set_graphics_pipeline(&pso, Topology::Triangles);
    list.set_graphics_binding(binding);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    super::run(&t.host, &[&list]);
    let mut pixels = [0; 8 * 8 * 4];
    t.host.backend.read_texture(&color, 0, 0, &mut pixels, 8 * 4, 0);
    assert!(pixels.chunks(4).all(|p| p == [19, 39, 0, 255]), "{pixels:?}");
}

/// A guessed vertex shader reads its positions through a buffer descriptor built around the address
/// in its root constants: the page table is bound for the vertex stage. Read as zeros (no table),
/// the triangle would have no area and nothing would be drawn.
#[test]
fn guessed_vertex_buffer_reads_the_pool() {
    let vertex = [
        0x8004_9000, // s_add_u32 s4, s0, 16
        0x8205_8080, // s_addc_u32 s5, 0, 0
        0x8805_FF05, 0x0008_0000, // s_or_b32 s5, s5, 0x80000 (stride 8)
        0xBE86_0383, // s_mov_b32 s6, 3
        0xBE87_03FF, 0x0001_4FAC, // s_mov_b32 s7, 0x14fac
        0xE034_2000, 0x8001_0100, // buffer_load_dwordx2 v[1:2], v0, s[4:7], 0 idxen
        0x7E06_0280, // v_mov_b32 v3, 0
        0x7E08_02F2, // v_mov_b32 v4, 1.0
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_08CF, 0x0403_0201, // exp pos0 v1, v2, v3, v4 done
        0xBF81_0000, // s_endpgm
    ];
    let pixel = [
        0x7E00_02F2, // v_mov_b32 v0, 1.0
        0x7E02_02F0, // v_mov_b32 v1, 0.5
        0xF800_180F, 0x0001_0100, // exp mrt0 v0, v1, v1, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    let translate = |stage, words: &[u32], meta: &[u8]| {
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let c = victory_shader::emit::graphics::container(stage, &bytes, meta).unwrap();
        shaders::translate(&c).unwrap()
    };
    let mut meta = [0; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // Root constants in slot 0.
    let vs = translate(HwStage::Vs, &vertex, &meta);
    assert!(vs.memory && !vs.memory_writes);
    let ps = translate(HwStage::Ps, &pixel, &[0; 0x10]);
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    // VA_B + 16: a triangle over the whole target.
    f.put(16, &[-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0]);

    let state = Graphics::guess_base()
        .for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &state, "computed vertex") else {
        panic!("{:?}", t.shaders);
    };
    let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 8, 8);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    let mut list = NativeList::default();
    list.set_render_targets(std::slice::from_ref(&color), None);
    list.set_graphics_pipeline(&pso, Topology::Triangles);
    list.set_graphics_binding(binding);
    list.set_graphics_root_constants(0, &[VA_B as u32]);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    super::run(&t.host, &[&list]);
    let mut pixels = [0; 8 * 8 * 4];
    t.host.backend.read_texture(&color, 0, 0, &mut pixels, 8 * 4, 0);
    for p in pixels.chunks(4) {
        assert_eq!(p, [255, 128, 128, 255]);
    }
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

/// A vertex shader fetching its positions through a root SRV (Grand Theft Auto VI's vertex fetch)
/// reads them at the title's addresses across two title pages that are not next to each other in
/// the pool: the first vertex at the end of one 64 KiB page, the others at the start of the next,
/// which the title mapped elsewhere in the pool (pages mapped in descending pool order). A view of
/// the pool from the root address on read the pool's own next bytes there (the canary): the
/// triangle had no area and nothing was drawn.
#[test]
fn a_vertex_root_view_follows_the_title_across_pages_apart_in_the_pool() {
    let vertex = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0x8805_FF01, 0x0008_0000, // s_or_b32 s5, s1, 0x80000 (stride 8)
        0xBE86_03C1, // s_mov_b32 s6, -1 (no record limit, as the title builds them)
        0xBE87_03FF, 0x0001_4FAC, // s_mov_b32 s7, 0x14fac
        0xE034_2000, 0x8001_0100, // buffer_load_dwordx2 v[1:2], v0, s[4:7], 0 idxen
        0x7E06_0280, // v_mov_b32 v3, 0
        0x7E08_02F2, // v_mov_b32 v4, 1.0
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_08CF, 0x0403_0201, // exp pos0 v1, v2, v3, v4 done
        0xBF81_0000, // s_endpgm
    ];
    let pixel = [
        0x7E00_02F2, // v_mov_b32 v0, 1.0
        0x7E02_02F0, // v_mov_b32 v1, 0.5
        0xF800_180F, 0x0001_0100, // exp mrt0 v0, v1, v1, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    let translate = |stage, words: &[u32], meta: &[u8]| {
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let c = victory_shader::emit::graphics::container(stage, &bytes, meta).unwrap();
        shaders::translate(&c).unwrap()
    };
    let mut meta = [0; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x15] = 1; // A root SRV (kind 0, root descriptor) in slot 0, user data 0-1.
    let vs = translate(HwStage::Vs, &vertex, &meta);
    let ps = translate(HwStage::Ps, &pixel, &[0; 0x10]);
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    // Two title pages, at pool offsets 0x4_0000 and 0x8_0000: the pool after the first is canary.
    const VA_C: u64 = 0x7100_0000;
    t.host.memory().mapped(VA_C as usize, 0x1_0000, &f.info, 0x4_0000);
    t.host.memory().mapped(VA_C as usize + 0x1_0000, 0x1_0000, &f.info, 0x8_0000);
    // A triangle over the whole target: its first vertex in the first page, the others in the next.
    f.put(0x4_FFF8, &[-1.0f32, -1.0]);
    f.put(0x8_0000, &[3.0f32, -1.0, -1.0, 3.0]);

    let state = Graphics::guess_base()
        .for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &state, "root view vertex") else {
        panic!("{:?}", t.shaders);
    };
    let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 8, 8);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    let mut list = NativeList::default();
    list.set_render_targets(std::slice::from_ref(&color), None);
    list.set_graphics_pipeline(&pso, Topology::Triangles);
    list.set_graphics_binding(binding);
    list.set_graphics_root_view(0, VA_C + 0xFFF8);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    super::run(&t.host, &[&list]);
    let mut pixels = [0; 8 * 8 * 4];
    t.host.backend.read_texture(&color, 0, 0, &mut pixels, 8 * 4, 0);
    for p in pixels.chunks(4) {
        assert_eq!(p, [255, 128, 128, 255]);
    }
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

/// The table follows the title's maps: the same dispatch, the same title addresses, after the title
/// mapped the pages somewhere else.
#[test]
fn the_table_follows_remaps_and_a_title_heap_serves_as_well() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = shader(COMPUTED, 4);
    prepare_computed(&f, 0x2_0000, 0x6_0000);
    let [a, b] = halves(VA_A);
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs, &[a, b, c, d], true);
    check_computed(&f, 0x2_0000, 0x6_0000);

    // VA_A now shows the pool from 0x4_0000: source at 0x4_0000, destination at 0x8_0000.
    t.host.memory().mapped(VA_A as usize, 0x8_0000, &f.info, 0x4_0000);
    f.put(0, &[CANARY; POOL / 4]);
    prepare_computed(&f, 0x4_0000, 0x8_0000);
    dispatch(&mut t, &cs, &[a, b, c, d], true);
    check_computed(&f, 0x4_0000, 0x8_0000);
    // The old destination was left alone.
    assert_eq!(f.get(0x6_0000, 64), [CANARY; 64]);
    assert_eq!(t.shaders.pipelines, 1);
}

/// Work the GPU has not run yet reads the page table it was submitted with, however often the pool
/// table changes meanwhile: a dispatch waits for a fence while the title moves a view twelve times,
/// with a dispatch (and so a new table) each time. The host freed a table eight rebuilds after it
/// was replaced, so the waiting dispatch read freed memory, or a later table's words in it, and
/// wrote wherever those sent it (Grand Theft Auto VI's first world frame: its BVH builder's empty
/// nodes over streamed resources, with a compute queue held for seconds).
#[test]
fn a_page_table_lasts_until_the_gpu_ran_the_work_submitted_with_it() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = shader(COMPUTED, 4);
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "addr") else {
        panic!("not made: {:?}", t.shaders)
    };
    prepare_computed(&f, 0x2_0000, 0x6_0000);
    let [a, b] = halves(VA_A);
    let [c, d] = halves(VA_A + 0x4_0000);
    let mut waiting = NativeList::default();
    waiting.set_compute_pipeline(&pso);
    waiting.set_compute_root_constants(0, &[a, b, c, d]);
    waiting.dispatch(1, 1, 1).unwrap();
    use victory_metal::backend::ComputeBackend;
    let backend = &t.host.backend;
    let (q, gate, done) = (backend.create_queue().unwrap(), backend.create_fence(0).unwrap(), backend.create_fence(0).unwrap());
    t.host.execute(&q, &[&waiting], &[(&gate, 1)], &[(&done, 1)]).unwrap();

    // Dispatches of memory no pool shows (nothing written), each after VA_A moved in the pool.
    let [e, g] = halves(0x55_0000_0000);
    for k in 0..12 {
        let offset = if k % 2 == 0 { 0x4_0000 } else { 0 };
        t.host.memory().mapped(VA_A as usize, 0x8_0000, &f.info, offset);
        dispatch(&mut t, &cs, &[e, g, e, g], false);
    }
    t.host.backend.signal(&gate, 1);
    assert!(t.host.backend.wait(&done, 1, 10_000), "GPU did not finish");
    check_computed(&f, 0x2_0000, 0x6_0000);
}

/// A released pool's view pair goes to the next pool, and a heap that wrote the old pair (the
/// host's own) writes it again: the same title addresses reach the new pool.
#[test]
fn a_released_pools_pair_serves_the_next_pool() {
    let Some(mut t) = title() else { return };
    let cs = shader(COMPUTED, 4);
    let [a, b] = halves(VA_A);
    let [c, d] = halves(VA_A + 0x4_0000);
    let f = Fixture::new(&t);
    prepare_computed(&f, 0x2_0000, 0x6_0000);
    dispatch(&mut t, &cs, &[a, b, c, d], false);
    check_computed(&f, 0x2_0000, 0x6_0000);
    t.host.memory().pool_released(&f.info);
    drop(f);
    let g = Fixture::with_id(&t, 2);
    prepare_computed(&g, 0x2_0000, 0x6_0000);
    dispatch(&mut t, &cs, &[a, b, c, d], false);
    check_computed(&g, 0x2_0000, 0x6_0000);
}

/// Memory no pool shows reads as zero and its writes are dropped, in a page past a mapping, in a
/// 256 MiB region with no mapping at all and in the pool's own range below it; none of it lands in
/// the pool where a naive offset from the mapping would put it.
#[test]
fn memory_no_pool_shows_reads_zero_and_drops_writes() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = shader(MISSES, 6);
    assert!(cs.memory && !cs.memory_writes);
    // dst is mapped; src and the store target are the page after the mapping (pool 0xA_0000 if the
    // mapping went on) and a region of its own.
    for (src, wild) in [(VA_A + 0x8_0000, VA_A + 0x9_0000), (0x55_0000_0040, 0x7F_FFFF_0000)] {
        f.put(0, &[CANARY; POOL / 4]);
        let [a, b] = halves(src);
        let [c, d] = halves(VA_A + 0x4_0000);
        let [e, g] = halves(wild);
        dispatch(&mut t, &cs, &[a, b, c, d, e, g], false);
        assert_eq!(f.get(0x6_0000, 64), [0x55; 64]);
        // Everything but the 64 words stored is as it was.
        assert_eq!(f.get(0x6_0100, 64), [CANARY; 64]);
        assert_eq!(f.get(0xA_0000, 64), [CANARY; 64]);
        assert_eq!(f.get(0xB_0000, 64), [CANARY; 64]);
        assert_eq!(f.get(0, 64), [CANARY; 64]);
    }
}

/// Shaders the translator marks pages in (translated without the host's option) are refused, not
/// run with a bitmap nobody binds.
#[test]
fn a_shader_that_marks_pages_is_refused() {
    let Some(mut t) = title() else { return };
    let mut cs = shader(COMPUTED, 4);
    cs.memory_writes = true;
    assert!(matches!(t.compute_pipe(&cs, "marking"), Pipe::Refused));
    assert!(t.shaders.refused.iter().any(|(why, _)| why.contains("marks the pages")));
}

/// `zero_copy_memory` is what removes the bitmap and its atomics; D3D12's translation has both.
#[test]
fn only_the_zero_copy_option_drops_the_page_marks() {
    let d3d12 = translate(COMPUTED, 4, victory_shader::emit::Options::default());
    assert!(d3d12.memory && d3d12.memory_writes);
    let host = shader(COMPUTED, 4);
    assert!(host.memory && !host.memory_writes);
    assert_ne!(d3d12.dxil, host.dxil);
}

#[test]
fn a_wave64_mask_joined_with_data_takes_each_lanes_bit_on_the_gpu() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = shader(MASK_DATA_JOIN, 2);
    dispatch(&mut t, &cs, &halves(VA_A), false);
    let want: Vec<u32> = (0..64).map(|lane| u32::from(lane < 36)).collect();
    assert_eq!(f.get(0x2_0000, 64), want);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

/// LDS by M0 and by lane (RDNA2 ISA 12.13): 20 of 32 lanes `ds_append` twice to the dword at
/// M0 = 8 (one atomic add of 20 each, the old value to every lane), then all 32 `ds_consume` it;
/// `ds_write_addtid_b32` with M0 = 64 and offset 4 puts each lane's id at 68 + 4 × lane, read
/// back by address and by `ds_read_addtid_b32`. Stored to `s[0:1]` + 0, 512, 768, 256 and 384.
const LDS_BY_M0: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
    0x7E04_0280, // v_mov_b32_e32 v2, 0
    0x7E06_02C1, // v_mov_b32_e32 v3, -1
    0x7E0C_02C1, // v_mov_b32_e32 v6, -1
    0xD834_0008, 0x0000_0202, // ds_write_b32 v2, v2 offset:8
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7D88_0094, // v_cmp_gt_u32_e32 vcc_lo, 20, v0
    0xBE84_3C6A, // s_and_saveexec_b32 s4, vcc_lo
    0xBEFC_0388, // s_mov_b32 m0, 8
    0xD8F8_0000, 0x0300_0000, // ds_append v3
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xD8F8_0000, 0x0600_0000, // ds_append v6
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x887E_047E, // s_or_b32 exec_lo, exec_lo, s4
    0xDC70_8000, 0x0000_0301, // global_store_dword v1, v3, s[0:1]
    0xDC70_8200, 0x0000_0601, // global_store_dword v1, v6, s[0:1] offset:512
    0xD8F4_0000, 0x0900_0000, // ds_consume v9
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xDC70_8300, 0x0000_0901, // global_store_dword v1, v9, s[0:1] offset:768
    0xBEFC_03C0, // s_mov_b32 m0, 64
    0xDAC0_0004, 0x0000_0000, // ds_write_addtid_b32 v0 offset:4
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xD8D8_0044, 0x0700_0001, // ds_read_b32 v7, v1 offset:68
    0xDAC4_0004, 0x0500_0000, // ds_read_addtid_b32 v5 offset:4
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xDC70_8100, 0x0000_0701, // global_store_dword v1, v7, s[0:1] offset:256
    0xDC70_8180, 0x0000_0501, // global_store_dword v1, v5, s[0:1] offset:384
    0xBF81_0000, // s_endpgm
];

/// `ds_write_addtid_b32` and `ds_read_addtid_b32` in a wave64 group of 64 (two host waves on
/// Metal): each lane's dword is at 68 + 4 × its console lane, read back by address too. Stored to
/// `s[0:1]` + 1024 and 1280.
const ADDTID_WAVE64: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
    0xBEFC_03C0, // s_mov_b32 m0, 64
    0xDAC0_0004, 0x0000_0000, // ds_write_addtid_b32 v0 offset:4
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xD8D8_0044, 0x0700_0001, // ds_read_b32 v7, v1 offset:68
    0xDAC4_0004, 0x0500_0000, // ds_read_addtid_b32 v5 offset:4
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xDC70_8400, 0x0000_0701, // global_store_dword v1, v7, s[0:1] offset:1024
    0xDC70_8500, 0x0000_0501, // global_store_dword v1, v5, s[0:1] offset:1280
    0xBF81_0000, // s_endpgm
];

/// [`LDS_BY_M0`]'s appends in a wave64 group of two console waves (four host waves on Metal): lanes
/// 0..39 of each wave (both halves) `ds_append` twice to the dword at M0 = 8, one atomic add of 40
/// each with the old value to the wave's lanes on, then every lane `ds_consume`s it past a
/// barrier. Stored to `s[0:1]` + 0, 512 and 1024.
const APPEND_WAVE64: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
    0x7E04_0280, // v_mov_b32_e32 v2, 0
    0x7E06_02C1, // v_mov_b32_e32 v3, -1
    0x7E0C_02C1, // v_mov_b32_e32 v6, -1
    0xD834_0008, 0x0000_0202, // ds_write_b32 v2, v2 offset:8
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF8A_0000, // s_barrier
    0x3610_00BF, // v_and_b32_e32 v8, 63, v0
    0x7D88_10A8, // v_cmp_gt_u32_e32 vcc, 40, v8
    0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
    0xBEFC_0388, // s_mov_b32 m0, 8
    0xD8F8_0000, 0x0300_0000, // ds_append v3
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xD8F8_0000, 0x0600_0000, // ds_append v6
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x88FE_047E, // s_or_b64 exec, exec, s[4:5]
    0xBF8A_0000, // s_barrier
    0xD8F4_0000, 0x0900_0000, // ds_consume v9
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xDC70_8000, 0x0000_0301, // global_store_dword v1, v3, s[0:1]
    0xDC70_8200, 0x0000_0601, // global_store_dword v1, v6, s[0:1] offset:512
    0xDC70_8400, 0x0000_0901, // global_store_dword v1, v9, s[0:1] offset:1024
    0xBF81_0000, // s_endpgm
];

#[test]
fn lds_append_and_consume_of_a_console_wave_split_in_two_add_its_lanes_once() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    run_with_lds(&mut t, APPEND_WAVE64, true, 128);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    let (first, second, taken) =
        (f.get(0x2_0000, 128), f.get(0x2_0000 + 512, 128), f.get(0x2_0000 + 1024, 128));
    // Each wave's lanes on share one old value per append; the waves' appends interleave.
    let mut olds = Vec::new();
    for w in 0..2 {
        let wave = 64 * w..64 * w + 64;
        let (x, y) = (first[64 * w], second[64 * w]);
        assert!(x < y, "wave {w}: {x} then {y}");
        olds.extend([x, y]);
        for i in wave.clone() {
            let (want_x, want_y) = if i % 64 < 40 { (x, y) } else { (u32::MAX, u32::MAX) };
            assert_eq!((first[i], second[i]), (want_x, want_y), "lane {i}");
        }
        assert!(taken[wave.clone()].iter().all(|&z| z == taken[64 * w]), "wave {w}");
    }
    olds.sort_unstable();
    assert_eq!(olds, [0, 40, 80, 120]);
    let mut consumed = [taken[0], taken[64]];
    consumed.sort_unstable();
    assert_eq!(consumed, [96, 160], "each wave takes 64 once");
}

/// `words` as one compute group of `threads` with 512 bytes of LDS and `s[0:1]` a root constant
/// pair, translated with the host's options, run once with `s[0:1]` = `VA_A`.
fn run_with_lds(t: &mut Title<Metal>, words: &[u32], wave64: bool, threads: u32) {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo { rsrc2: 1 << 15 | 2 << 1, threads: [threads, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let mut empty = b"DXBC".to_vec();
    empty.extend([0; 16]);
    empty.extend([1, 0, 0, 0]);
    empty.extend(32u32.to_le_bytes());
    empty.extend(0u32.to_le_bytes());
    let c = Container::parse(&empty).expect("container");
    let cs = emit_with(&c, &native, &p, &st, &io, &res, shaders::options()).expect("emit");
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "lds") else {
        panic!("not made: {:?}", t.shaders)
    };
    let mut l = NativeList::default();
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &halves(VA_A));
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
}

#[test]
fn lds_append_consume_and_addtid_follow_m0_on_the_gpu() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    run_with_lds(&mut t, LDS_BY_M0, false, 32);
    let first: Vec<u32> = (0..32).map(|i| if i < 20 { 0 } else { u32::MAX }).collect();
    assert_eq!(f.get(0x2_0000, 32), first, "the first ds_append");
    let second: Vec<u32> = (0..32).map(|i| if i < 20 { 20 } else { u32::MAX }).collect();
    assert_eq!(f.get(0x2_0000 + 512, 32), second, "the second ds_append");
    assert_eq!(f.get(0x2_0000 + 768, 32), vec![40; 32], "ds_consume");
    let ids: Vec<u32> = (0..32).collect();
    assert_eq!(f.get(0x2_0000 + 256, 32), ids, "ds_write_addtid_b32");
    assert_eq!(f.get(0x2_0000 + 384, 32), ids, "ds_read_addtid_b32");
    run_with_lds(&mut t, ADDTID_WAVE64, true, 64);
    let ids: Vec<u32> = (0..64).collect();
    assert_eq!(f.get(0x2_0000 + 1024, 64), ids, "wave64 ds_write_addtid_b32");
    assert_eq!(f.get(0x2_0000 + 1280, 64), ids, "wave64 ds_read_addtid_b32");
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

/// A structured buffer view (16-byte elements) whose descriptor the program loads and then clears
/// the stride of (`s_and_b32` of dword 1 with 0xc000ffff, a title's form): every lane loads by
/// index and offset, so with stride 0 it reads dword `lane` (index × 0 + lane × 4, RDNA2 ISA
/// 8.1.5), and stores it to a raw view.
const STRIDE_CLEARED: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202, 0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xF408_0302, 0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x8709_FF09, 0xC000_FFFF, // s_and_b32 s9, s9, 0xc000ffff
    0xE030_3000, 0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000, 0x8003_0201, // buffer_store_dword v2, v1, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

#[test]
fn a_descriptor_with_its_stride_cleared_reads_the_view_raw_on_the_gpu() {
    use victory_shader::native::{ResourceType, UsedResource};
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let src: Vec<u32> = (0..64).map(|i| i * 7 + 3).collect();
    f.put(0x2_1000, &src);
    f.put(0x2_2000, &[0u32; 64]);
    // The title's heap: a structured SRV of 16-byte elements, then a raw UAV.
    let records = vec![0u32; 8 * 2];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 2).unwrap();
    let (rs, rd) = (0x57D_0001usize, 0x57D_0002usize);
    t.host.register_buffer(rs, VA_A + 0x1000, 4096);
    t.host.register_buffer(rd, VA_A + 0x2000, 4096);
    let mut srv = [0u32; 10];
    (srv[1], srv[6], srv[7]) = (1, 16, 16);
    let mut uav = [0u32; 10];
    (uav[1], uav[4], uav[8]) = (1, 64, 1);
    crate::descriptor::create_srv(std::ptr::null(), rs as _, &srv, start);
    crate::descriptor::create_uav(std::ptr::null(), rd as _, std::ptr::null(), &uav, start + 32);
    assert!(t.host.mirror(start).unwrap() && t.host.mirror(start + 32).unwrap());
    let used = |kind, table_offset_dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 4,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: vec![used(ResourceType::Srv, 0, 16), used(ResourceType::Uav, 8, 0)],
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [32, 1, 1] }),
        code: STRIDE_CLEARED.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let mut empty = b"DXBC".to_vec();
    empty.extend([0; 16]);
    empty.extend([1, 0, 0, 0]);
    empty.extend(32u32.to_le_bytes());
    empty.extend(0u32.to_le_bytes());
    let c = Container::parse(&empty).expect("container");
    let cs = emit_with(&c, &native, &p, &st, &io, &res, shaders::options()).expect("emit");
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "stride") else {
        panic!("not made: {:?}", t.shaders)
    };
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(f.get(0x2_2000, 32), src[..32]);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}
