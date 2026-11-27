//! Hand-assembled GFX10 unit tests covering all 6 graphics shader refusal
//! classes (both positive sequences that must lower — across `wave32_host`
//! false/true where relevant — and genuinely unsupported variants that must
//! remain refused). Encodings assembled with `llvm-mc -mcpu=gfx1030` (plus
//! `-mattr=+wavefrontsize64` for wave64).

use super::container::SigElement;
use super::lower::{self, Lowered, Parts};
use super::{Options, graphics, translate_with};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, PsInput, ResourceType, RootParam, RootType, StageIo, Usage,
    UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

fn bytes_of(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// `v_pk_fmac_f16` (VOP2) with a constant first source translates: an inline float constant is
/// its f16 in the low half and 0 in the high one, an integer one its 32-bit value, a literal its
/// two halves (llvm-mc's encodings for gfx1030). Values: `valu_tests`.
#[test]
fn packed_fmac_takes_constant_operands() {
    use crate::native::tests::{fspx, fspx_resource};
    let program = |fmac: &[u32]| {
        let mut code = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        ];
        code.extend_from_slice(fmac);
        code.extend([
            0xE070_0000,
            0x8002_0000, // buffer_store_dword v0, off, s[8:11], 0
            0xBF81_0000, // s_endpgm
        ]);
        let table = [fspx_resource(3, 3, 0, 0, 4, 0)];
        let part = fspx(&bytes_of(&code), 5 << 1, [1, 1, 1], true, &table);
        crate::dxbc::build(&[(FourCC::ACCC, &part)])
    };
    for fmac in [
        &[0x7800_00F2][..],          // v_pk_fmac_f16 v0, 1.0, v0
        &[0x7800_00C1],              // v_pk_fmac_f16 v0, -1, v0
        &[0x7800_00FF, 0x4000_3C00], // v_pk_fmac_f16 v0, 0x40003c00, v0
    ] {
        translate_with(&program(fmac), Options::default()).unwrap();
    }
}

/// Joins of values with no DXIL form are all found by one lowering pass, however many there are:
/// a pass for each made pixel shaders with some hundreds of them take 40 s instead of 0.2 s.
/// Here a branch's arm copies a descriptor's dword (nothing the host has) to 32 SGPRs, each
/// joined after it with what it held on entry and copied to a VGPR nothing reads.
#[test]
fn joins_without_a_value_are_found_in_one_pass() {
    use crate::native::tests::{fspx, fspx_resource};
    let program = |joins: u32| {
        let mut code = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xBE8C_0380, // s_mov_b32 s12, 0
            0xBF06_800C, // s_cmp_eq_u32 s12, 0
            0xBF85_0000 | joins, // s_cbranch_scc1 past the arm
        ];
        code.extend((0..joins).map(|k| 0xBE80_0308 | (20 + k) << 16)); // s_mov_b32 s(20 + k), s8
        code.extend((0..joins).map(|k| 0x7E00_0200 | (1 + k) << 17 | (20 + k))); // v_mov_b32 v(1 + k), s(20 + k)
        code.extend([
            0xE070_0000,
            0x8002_0000, // buffer_store_dword v0, off, s[8:11], 0
            0xBF81_0000, // s_endpgm
        ]);
        let table = [fspx_resource(3, 3, 0, 0, 4, 0)];
        let part = fspx(&bytes_of(&code), 5 << 1, [1, 1, 1], true, &table);
        crate::dxbc::build(&[(FourCC::ACCC, &part)])
    };
    for joins in [1, 32] {
        translate_with(&program(joins), Options::default()).unwrap();
        assert_eq!(lower::PASSES.get(), 2, "{joins} joins");
    }
}

#[test]
fn guessed_pixel_computed_buffer_loads_use_title_memory() {
    let mut meta = [0u8; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // CBV, root constants in slot 0, live-in s0
    let code = [
        0x8004_9000, // s_add_u32 s4, s0, 16
        0x8205_8080, // s_addc_u32 s5, 0, 0
        0x8705_FF05,
        0x0000_FFFF, // s_and_b32 s5, s5, 0xffff
        0x8F06_9090, // s_lshl_b32 s6, 16, 16
        0x8805_0605, // s_or_b32 s5, s5, s6
        0xBE86_0384, // s_mov_b32 s6, 4
        0xBE87_03FF,
        0x0001_4FAC, // s_mov_b32 s7, 0x00014fac
        0xE034_3004,
        0x8001_0200, // buffer_load_dwordx2 v[2:3], v[0:1], s[4:7], 0 idxen offen offset:4
        0xF420_0202,
        0xFA00_000C, // s_buffer_load_dword s8, s[4:7], 12
        0x7E08_0208, // v_mov_b32 v4, s8
        0xF800_180F,
        0x0404_0302, // exp mrt0 v2, v3, v4, v4 done vm
        0xBF81_0000,
    ];
    let container = |code: &[u32]| graphics::container(HwStage::Ps, &bytes_of(code), &meta).unwrap();
    let options = Options { wave32_host: true, zero_copy_memory: true, ..Options::default() };
    for flags in [0x0001_4FAC, 0x3001_4FAC] {
        let mut code = code;
        code[8] = flags;
        let shader = translate_with(&container(&code), options).unwrap();
        assert!(shader.memory);
        assert!(!shader.memory_writes && !shader.heap);
        assert_eq!(shader.user_data.len(), 1);
        assert_eq!(shader.user_data[0].slot, 0);
    }
    let mut dynamic_count = code;
    dynamic_count[6] = 0xBE86_0300; // s_mov_b32 s6, s0
    assert!(translate_with(&container(&dynamic_count), options).unwrap().memory);
    assert!(translate_with(&container(&code), Options::default()).is_err());

    let mut swizzled = code;
    swizzled[5] = 0x8805_FF05; // s_or_b32 s5, s5, 0x80000000
    let mut swizzled = swizzled.to_vec();
    swizzled.insert(6, 0x8000_0000);
    let err = translate_with(&container(&swizzled), options).err().unwrap();
    assert!(err.to_string().contains("computed buffer descriptor with swizzling"), "{err}");
}

/// A guessed vertex shader's buffer descriptor built around a computed address reads title
/// memory through the page table too: the Metal host binds the table for the vertex stage.
#[test]
fn guessed_vertex_computed_buffer_loads_use_title_memory() {
    let mut meta = [0u8; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // CBV, root constants in slot 0, live-in s0
    let code = [
        0x8004_9000, // s_add_u32 s4, s0, 16
        0x8205_8080, // s_addc_u32 s5, 0, 0
        0x8705_FF05,
        0x0000_FFFF, // s_and_b32 s5, s5, 0xffff
        0x8F06_9090, // s_lshl_b32 s6, 16, 16
        0x8805_0605, // s_or_b32 s5, s5, s6
        0xBE86_0384, // s_mov_b32 s6, 4
        0xBE87_03FF,
        0x0001_4FAC, // s_mov_b32 s7, 0x00014fac
        0xE034_3004,
        0x8001_0200, // buffer_load_dwordx2 v[2:3], v[0:1], s[4:7], 0 idxen offen offset:4
        0xF420_0202,
        0xFA00_000C, // s_buffer_load_dword s8, s[4:7], 12
        0x7E08_0208, // v_mov_b32 v4, s8
        0xF800_08CF,
        0x0404_0302, // exp pos0 v2, v3, v4, v4 done
        0xBF81_0000,
    ];
    let container = graphics::container(HwStage::Vs, &bytes_of(&code), &meta).unwrap();
    let options = Options { wave32_host: true, zero_copy_memory: true, ..Options::default() };
    let shader = translate_with(&container, options).unwrap();
    assert!(shader.memory);
    assert!(!shader.memory_writes && !shader.heap);
    assert!(translate_with(&container, Options::default()).is_err());
}

#[test]
fn guessed_pixel_computed_subword_loads_use_title_memory() {
    let mut meta = [0u8; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2;
    let code = [
        0x8004_9000, // s_add_u32 s4, s0, 16
        0x8205_8080, // s_addc_u32 s5, 0, 0
        0x8705_FF05, 0x0000_FFFF, // s_and_b32 s5, s5, 0xffff
        0x8F06_9090, // s_lshl_b32 s6, 16, 16
        0x8805_0605, // s_or_b32 s5, s5, s6
        0xBE86_0384, // s_mov_b32 s6, 4
        0xBE87_03FF, 0x3001_4FAC, // s_mov_b32 s7, flags
        0xE020_3000, 0x8001_0200, // buffer_load_ubyte v2, v[0:1], s[4:7], 0 idxen offen
        0xF800_180F, 0x0202_0202, // exp mrt0 v2, v2, v2, v2 done vm
        0xBF81_0000,
    ];
    let translate = |words: &[u32], wave32_host| {
        let container = graphics::container(HwStage::Ps, &bytes_of(words), &meta).unwrap();
        translate_with(&container, Options { wave32_host, zero_copy_memory: true, ..Options::default() })
    };
    for wave32_host in [false, true] {
        for op in 8..=11 {
            for bounds in 0..4 {
                for offset in [0, 1, 2, 3, 15, 16, 4095] {
                    let mut words = code;
                    words[8] = 0x0001_4FAC | bounds << 28;
                    words[9] = 0xE000_3000 | op << 18 | offset;
                    let shader = translate(&words, wave32_host).unwrap();
                    assert!(shader.memory && !shader.memory_writes && !shader.heap);
                }
            }
        }
    }
    // Typed loads, stores and atomics must not gain support accidentally.
    for op in [0, 24, 26, 28, 50] {
        let mut words = code;
        words[9] = 0xE000_3000 | op << 18;
        assert!(translate(&words, true).is_err(), "opcode {op}");
    }
    for flags in [0x3081_4FAC, 0x7001_4FAC] {
        let mut words = code;
        words[8] = flags; // ADD_TID or non-buffer type
        assert!(translate(&words, true).is_err());
    }
    let mut swizzled = code.to_vec();
    swizzled[5] = 0x8805_FF05;
    swizzled.insert(6, 0x8000_0000);
    assert!(translate(&swizzled, true).is_err());
}

/// Exercise the graphics descriptor classification with a compute entry point
/// so the Windows test host can dispatch it and read back exact integer results.
/// This does not enable computed descriptors for production compute shaders.
pub(super) fn computed_subword_test_shader(op: u32, bounds: u32) -> super::HostShader {
    use crate::native::{ComputeIo, ResourceType, UsedResource};
    let words = [
        0xBE88_0300, // s_mov_b32 s8, s0 (base low)
        0x8709_FF01, 0x3FFF_FFFF, // s_and_b32 s9, s1, 0x3fffffff (base high + stride)
        0xBE8A_0302, // s_mov_b32 s10, s2 (record count)
        0xBE8B_03FF, 0x0001_4FAC | bounds << 28,
        0x7E00_0203, // v_mov_b32 v0, s3 (index)
        0x7E02_0204, // v_mov_b32 v1, s4 (vector offset)
        0xE000_3000 | op << 18, 0x0502_0200, // buffer_load_* v2, v[0:1], s[8:11], s5 idxen offen
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0303, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], 0
        0xBF8C_0000, // s_waitcnt
        0xE070_0000, 0x8003_0200, // buffer_store_dword v2, off, s[12:15], 0
        0xBF81_0000,
    ];
    let native = NativeProgram {
        stage: HwStage::Cs, wave64: false, vgprs: 4,
        layout: vec![root_const(0, 6), RootParam {
            root: RootType::Table, table_offset_dwords: 0, sgprs: 1,
            base_register: 0, user_data: 6, slot: 1,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 0,
            lower_bound: 0, stride: 0, dwords: 4, user_data: 6, slot: 1,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2: 7 << 1, threads: [1, 1, 1] }),
        code: bytes_of(&words),
    };
    let p = ir::build(&native.code, false, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track_graphics(&p, &native.layout, HwStage::Ps, Some(&st), true).unwrap();
    assert!(res.descriptors.iter().any(|d| matches!(d.source, resource::Source::BuiltMemory { .. })));
    let io = io::raise(&p, &native).unwrap();
    let l = lower::lower(&Parts {
        native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 0,
        atomic_locks: false, bias_as_gradients: false, narrow_waves: false,
        wave32_host: false, zero_copy_memory: true, view_marks: false,
        rect_list: false, copy_params: [0; 32],
    }).unwrap();
    super::HostShader {
        stage: HwStage::Cs, dxil: l.dxil, splits: l.splits, table_entries: l.table_entries,
        user_data: l.user_data, samplers: l.samplers, static_samplers: vec![],
        root_cbvs: l.root_cbvs, root_srvs: l.root_srvs, root_uavs: l.root_uavs,
        memory: l.memory, memory_writes: l.memory_writes, view_addresses: l.view_addresses,
        heap: l.heap, heap_writes: l.heap_writes, view_marks: l.view_marks,
    }
}

#[test]
fn computed_subword_execution_fixtures_lower() {
    for op in 8..=11 {
        for bounds in 0..4 {
            assert!(computed_subword_test_shader(op, bounds).memory);
        }
    }
}

fn sig_el(
    name: &str,
    index: u32,
    kind: u8,
    comp: u8,
    interp: u8,
    row: Option<u32>,
    cols: u8,
) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp,
        interp,
        row,
        col: 0,
        cols,
        usage: 0,
        half: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_ps(
    words: &[u32],
    wave64: bool,
    layout: Vec<RootParam>,
    ps_inputs: Vec<PsInput>,
    input_addr: u32,
    db_shader_control: u32,
    inputs: Vec<SigElement>,
    wave32_host: bool,
) -> Result<Lowered, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64,
        vgprs: 16,
        layout,
        used: vec![],
        io: StageIo::Ps {
            inputs: ps_inputs,
            input_addr,
            db_shader_control,
            patches: vec![],
        },
        code: bytes_of(words),
    };
    let p = ir::build(&native.code, wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs,
        outputs: vec![sig_el("SV_Target", 0, 16, 9, 0, Some(0), 4)],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host,
        zero_copy_memory: false,
        view_marks: false,
        rect_list: false,
        copy_params: [0; 32],
    })
}

fn lower_vs(
    words: &[u32],
    wave64: bool,
    layout: Vec<RootParam>,
    rect_list: bool,
) -> Result<Lowered, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Vs,
        wave64,
        vgprs: 16,
        layout,
        used: vec![],
        io: StageIo::Vs {
            inputs: vec![],
            outputs: vec![],
        },
        code: bytes_of(words),
    };
    let p = ir::build(&native.code, wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![sig_el("SV_Position", 0, 3, 9, 0, Some(0), 4)],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        view_marks: false,
        rect_list,
        copy_params: [0; 32],
    })
}

fn parsed_isg1(dxil: &[u8]) -> Vec<SigElement> {
    let c = Container::parse(dxil).unwrap();
    let part = c.part(FourCC::ISG1).expect("ISG1");
    super::signature(part.data).unwrap()
}

fn root_const(user_data: u8, sgprs: u32) -> RootParam {
    RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot: 0,
    }
}

fn vertex_table_layout() -> Vec<RootParam> {
    vec![RootParam {
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        sgprs: 2,
        base_register: 0,
        user_data: 0,
        slot: graphics::VERTEX_TABLE,
    }]
}

// =========================================================================
// Refusal #1: "opaque lane mask" / "a constant lane mask" in MaskData::Host
// =========================================================================

#[test]
fn refusal1_mask_scc_lowers_in_pixel_shader() {
    // v_cmp_lt_u32_e32 vcc, 5, v0
    // s_and_b64 s[2:3], vcc, exec
    // s_cselect_b32 s4, 7, 9
    // v_mov_b32_e32 v0, s4
    // exp mrt0 v0, v0, v0, v0 done vm
    // s_endpgm
    let code = [
        0x7D82_0085,
        0x8782_7E6A,
        0x8504_8987,
        0x7E00_0204,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    for wave32_host in [false, true] {
        let out = lower_ps(&code, true, vec![], vec![], 0, 0, vec![], wave32_host)
            .unwrap_or_else(|e| panic!("wave32_host={wave32_host}: {e}"));
        assert!(out.wave_ops);
    }
}

#[test]
fn refusal1_wave_is_first_lane_idioms_lower_in_pixel_shader() {
    // Wave64: 1 << s_ff1_i32_b64(exec)
    let w64_lshl = [
        0xBE80_147E, // s_ff1_i32_b64 s0, exec
        0x8F82_0081, // s_lshl_b64 s[2:3], 1, s0
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000, // exp mrt0 v0, v0, v0, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    // Wave64: s_bitset1_b64(0, s_ff1_i32_b64(exec))
    let w64_bitset1 = [
        0xBE82_0480, // s_mov_b64 s[2:3], 0
        0xBE80_147E, // s_ff1_i32_b64 s0, exec
        0xBE82_1E00, // s_bitset1_b64 s[2:3], s0
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave64: 1 << (63 - s_flbit_i32_b64(exec))
    let w64_flbit = [
        0xBE80_167E, // s_flbit_i32_b64 s0, exec
        0x8180_00BF, // s_sub_i32 s0, 63, s0
        0x8F82_0081, // s_lshl_b64 s[2:3], 1, s0
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: 1 << s_ff1_i32_b32(exec_lo)
    let w32_lshl = [
        0xBE80_137E, // s_ff1_i32_b32 s0, exec_lo
        0x8F02_0081, // s_lshl_b32 s2, 1, s0
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: 1 << (31 - s_flbit_i32_b32(exec_lo))
    let w32_flbit = [
        0xBE80_157E, // s_flbit_i32_b32 s0, exec_lo
        0x8180_009F, // s_sub_i32 s0, 31, s0
        0x8F02_0081, // s_lshl_b32 s2, 1, s0
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    for (label, code, wave64) in [
        ("w64_lshl", &w64_lshl[..], true),
        ("w64_bitset1", &w64_bitset1[..], true),
        ("w64_flbit", &w64_flbit[..], true),
        ("w32_lshl", &w32_lshl[..], false),
        ("w32_flbit", &w32_flbit[..], false),
    ] {
        for wave32_host in [false, true] {
            let out = lower_ps(code, wave64, vec![], vec![], 0, 0, vec![], wave32_host)
                .unwrap_or_else(|e| panic!("{label} wave32_host={wave32_host}: {e}"));
            assert!(out.wave_ops, "{label}");
        }
    }
}

#[test]
fn refusal1_mask_bit_tricks_lower_in_pixel_shader() {
    // Wave64: mask & -mask
    let w64_neg = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0x8082_6A80, // s_sub_u32 s2, 0, vcc_lo
        0x8283_6B80, // s_subb_u32 s3, 0, vcc_hi
        0x8782_026A, // s_and_b64 s[2:3], vcc, s[2:3]
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave64: mask & (mask - 1)
    let w64_dec = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0x8082_816A, // s_sub_u32 s2, vcc_lo, 1
        0x8283_806B, // s_subb_u32 s3, vcc_hi, 0
        0x8782_026A, // s_and_b64 s[2:3], vcc, s[2:3]
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave64: s_bitset0_b64(mask, s_ff1_i32_b64(mask))
    let w64_bitset0 = [
        0xBE82_047E, // s_mov_b64 s[2:3], exec
        0xBE80_1402, // s_ff1_i32_b64 s0, s[2:3]
        0xBE82_1C00, // s_bitset0_b64 s[2:3], s0
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: mask & -mask
    let w32_neg = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc_lo, 5, v0
        0x8182_6A80, // s_sub_i32 s2, 0, vcc_lo
        0x8702_026A, // s_and_b32 s2, vcc_lo, s2
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: mask & (mask - 1)
    let w32_dec = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc_lo, 5, v0
        0x8182_816A, // s_sub_i32 s2, vcc_lo, 1
        0x8702_026A, // s_and_b32 s2, vcc_lo, s2
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    for (label, code, wave64) in [
        ("w64_neg", &w64_neg[..], true),
        ("w64_dec", &w64_dec[..], true),
        ("w64_bitset0", &w64_bitset0[..], true),
        ("w32_neg", &w32_neg[..], false),
        ("w32_dec", &w32_dec[..], false),
    ] {
        for wave32_host in [false, true] {
            lower_ps(code, wave64, vec![], vec![], 0, 0, vec![], wave32_host)
                .unwrap_or_else(|e| panic!("{label} wave32_host={wave32_host}: {e}"));
        }
    }
}

#[test]
fn refusal1_uniform_and_quad_periodic_masks_lower_in_pixel_shader() {
    // Wave32: s_cselect_b32 -1, 0
    let w32_cselect = [
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x8502_80C1, // s_cselect_b32 s2, -1, 0
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: s_ashr_i32 x, 31
    let w32_ashr = [
        0x9102_9F00, // s_ashr_i32 s2, s0, 31
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: s_sub_i32 0, (s0 & 1)
    let w32_sub_flag = [
        0x8701_8100, // s_and_b32 s1, s0, 1
        0x8182_0180, // s_sub_i32 s2, 0, s1
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave64: s_cselect_b64 -1, 0
    let w64_cselect = [
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x8582_80C1, // s_cselect_b64 s[2:3], -1, 0
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave64 quad-periodic constant 0x5555_5555_5555_5555
    let w64_quad = [
        0xBE82_03FF,
        0x5555_5555, // s_mov_b32 s2, 0x55555555
        0xBE83_03FF,
        0x5555_5555, // s_mov_b32 s3, 0x55555555
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32 quad-periodic constant 0x5555_5555
    let w32_quad = [
        0xBE84_3CFF,
        0x5555_5555, // s_and_saveexec_b32 s4, 0x55555555
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    for (label, code, wave64) in [
        ("w32_cselect", &w32_cselect[..], false),
        ("w32_ashr", &w32_ashr[..], false),
        ("w32_sub_flag", &w32_sub_flag[..], false),
        ("w64_cselect", &w64_cselect[..], true),
        ("w64_quad", &w64_quad[..], true),
        ("w32_quad", &w32_quad[..], false),
    ] {
        for wave32_host in [false, true] {
            lower_ps(
                code,
                wave64,
                vec![root_const(0, 1)],
                vec![],
                0,
                0,
                vec![],
                wave32_host,
            )
            .unwrap_or_else(|e| panic!("{label} wave32_host={wave32_host}: {e}"));
        }
    }
}

#[test]
fn refusal1_split_exec_and_guessed_wave64_inference() {
    // Split 32-bit moves of EXEC in wave64 PS
    let split_exec = [
        0xBE82_037E, // s_mov_b32 s2, exec_lo
        0xBE83_037F, // s_mov_b32 s3, exec_hi
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0302, // s_mov_b32 exec_lo, s2
        0xBEFF_0303, // s_mov_b32 exec_hi, s3
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    lower_ps(&split_exec, true, vec![], vec![], 0, 0, vec![], true).unwrap();

    // Guessed wave64 PS with v_cmp_lt_u32_e32 (which formats implicit vcc as
    // `vcc_lo` in text(false)): must be inferred as wave64 and lower cleanly.
    let guessed_w64 = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0402, // s_mov_b64 exec, s[2:3]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    let code = bytes_of(&guessed_w64);
    let container = graphics::container(HwStage::Ps, &code, &[0u8; 0x10]).unwrap();
    let translated = translate_with(
        &container,
        Options {
            wave32_host: true,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(!translated.dxil.is_empty());
}

#[test]
fn refusal1_console_layout_dependent_masks_remain_refused() {
    // 1. Opaque mask from root constants s[0:1]
    let from_root = [
        0xBE84_2400, // s_and_saveexec_b64 s[4:5], s[0:1]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // 2. Ballot shifted by non-zero constant: s_lshl_b64 s[2:3], vcc, 1
    let shifted_ballot = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0x8F82_816A, // s_lshl_b64 s[2:3], vcc, 1
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281,
        0xBEFE_0404,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // 3. Constant single-lane mask: 1 << 5
    let const_bit = [
        0x8F82_8581, // s_lshl_b64 s[2:3], 1, 5
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281,
        0xBEFE_0404,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // 4. Non-quad-periodic constant: 0xFFFF_FFFF_0000_0000 (wave64)
    let non_quad_w64 = [
        0xBE82_0380, // s_mov_b32 s2, 0
        0xBE83_03C1, // s_mov_b32 s3, -1
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281,
        0xBEFE_0404,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // 5. Non-quad-periodic constant: 0x0000_FFFF (wave32)
    let non_quad_w32 = [
        0xBE84_3CFF,
        0x0000_FFFF, // s_and_saveexec_b32 s4, 0xffff
        0x7E00_0281,
        0xBEFE_0304,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // 6. Multiply of a mask: s_mul_i32 s2, vcc_lo, 3 (wave32)
    let mul_mask = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc_lo, 5, v0
        0x9302_836A, // s_mul_i32 s2, vcc_lo, 3
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281,
        0xBEFE_0304,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    for (label, code, wave64, expect) in [
        ("from_root", &from_root[..], true, "opaque lane mask"),
        ("shifted_ballot", &shifted_ballot[..], true, "opaque lane mask"),
        ("const_bit", &const_bit[..], true, "opaque lane mask"),
        ("non_quad_w64", &non_quad_w64[..], true, "lane mask"),
        ("non_quad_w32", &non_quad_w32[..], false, "constant lane mask"),
        ("mul_mask", &mul_mask[..], false, "opaque lane mask"),
    ] {
        let err = lower_ps(
            code,
            wave64,
            vec![root_const(0, 2)],
            vec![],
            0,
            0,
            vec![],
            true,
        )
        .err()
        .unwrap_or_else(|| panic!("{label}: expected refusal"));
        assert!(
            err.to_string().contains(expect),
            "{label}: expected `{expect}`, got `{err}`"
        );
    }
}

// =========================================================================
// Refusal #2: "a pixel system input the signature lacks"
// =========================================================================

/// A buffer descriptor's dword that is no size, read as data, is refused naming it: the host
/// has no value for the base address, the high address bits and stride, or the format.
#[test]
fn descriptor_dwords_read_as_data_say_which() {
    use crate::native::tests::{fspx, fspx_resource};
    let program = |read: u32| {
        let code = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE030_1000,
            0x8003_0100, // buffer_load_dword v1, v0, s[12:15], 0 offen
            read,        // v_mov_b32 v2, s<12 + n>
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x4A02_0501, // v_add_nc_u32 v1, v1, v2
            0xE070_0000,
            0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF81_0000, // s_endpgm
        ];
        // A raw UAV at table dword 0, the buffer read at dword 8.
        let resources = [fspx_resource(3, 3, 0, 0, 4, 0), fspx_resource(0, 3, 8, 0, 4, 0)];
        let part = fspx(&bytes_of(&code), 5 << 1, [32, 1, 1], true, &resources);
        crate::dxbc::build(&[(FourCC::ACCC, &part)])
    };
    let prefix = "a descriptor's address or format read as data: ";
    for (read, why) in [
        (0x7E04_020C, "a buffer's base address (dword 0)"),
        (0x7E04_020D, "a buffer's dword 1 (address high bits, stride, swizzle)"),
        (0x7E04_020F, "a buffer's dword 3 (format and type)"),
    ] {
        let err = translate_with(&program(read), Options::default()).err();
        assert!(
            matches!(err, Some(ShaderError::Unsupported { what, .. })
                if what.strip_prefix(prefix) == Some(why)),
            "{err:?}"
        );
    }
}

#[test]
fn refusal2_pixel_system_inputs_lower_when_declared() {
    // Reads v0..v6 = Position.xyzw, FrontFace, Ancillary, SampleCoverage
    let code = [
        0x4A00_0300, // v_add_nc_u32_e32 v0, v0, v1
        0x4A00_0500, // v_add_nc_u32_e32 v0, v0, v2
        0x4A00_0700, // v_add_nc_u32_e32 v0, v0, v3
        0x4A00_0900, // v_add_nc_u32_e32 v0, v0, v4
        0x4A00_0B00, // v_add_nc_u32_e32 v0, v0, v5
        0x4A00_0D00, // v_add_nc_u32_e32 v0, v0, v6
        0xF800_180F,
        0x0000_0000, // exp mrt0 v0, v0, v0, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    let inputs = vec![
        sig_el("SV_Position", 0, 3, 9, 4, Some(0), 4),
        sig_el("SV_IsFrontFace", 0, 13, 5, 1, Some(1), 1),
        sig_el("SV_SampleIndex", 0, 12, 5, 0, None, 1),
        sig_el("SV_Coverage", 0, 14, 5, 0, None, 1),
    ];
    let out = lower_ps(&code, true, vec![], vec![], 0x7F00, 0, inputs, false).unwrap();
    let isg1 = parsed_isg1(&out.dxil);
    assert!(isg1.iter().any(|e| e.kind == 3));
    assert!(isg1.iter().any(|e| e.kind == 13));
    assert!(isg1.iter().any(|e| e.kind == 12));
    assert!(isg1.iter().any(|e| e.kind == 14));

    // Guessed PS (`FSPG`) infers all 4 system values automatically.
    let container = graphics::container(HwStage::Ps, &bytes_of(&code), &[0u8; 0x10]).unwrap();
    let translated = translate_with(&container, Options::default()).unwrap();
    let guessed_isg1 = parsed_isg1(&translated.dxil);
    for kind in [3u8, 12, 13, 14] {
        assert!(
            guessed_isg1.iter().any(|e| e.kind == kind),
            "guessed ISG1 missing system value kind {kind}"
        );
    }
}

#[test]
fn signed_flbit_i64_lowers_in_pixel_shader() {
    let code = [
        0xBF06_8002, // s_cmp_eq_u32 s2, 0
        0xBE84_1800, // s_flbit_i32_i64 s4, s[0:1]
        0x7E00_0204, // v_mov_b32 v0, s4
        0x7E02_02FD, // v_mov_b32 v1, src_scc
        0xF800_180F, 0x0100_0100, // exp mrt0 v0, v1, v0, v1 done vm
        0xBF81_0000,
    ];
    lower_ps(&code, false, vec![root_const(0, 3)], vec![], 0, 0, vec![], false).unwrap();
}

/// `s_absdiff_i32` translates (values: `victory-gpu` `addr_tests`).
#[test]
fn absdiff_translates() {
    use crate::native::tests::{fspx, fspx_resource};
    let code = [
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0x9606_0504, // s_absdiff_i32 s6, s4, s5
        0x7E02_0206, // v_mov_b32 v1, s6
        0xE070_0000,
        0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
        0xBF81_0000, // s_endpgm
    ];
    let resources = [fspx_resource(3, 3, 0, 0, 4, 0)];
    let part = fspx(&bytes_of(&code), 5 << 1, [32, 1, 1], true, &resources);
    translate_with(&crate::dxbc::build(&[(FourCC::ACCC, &part)]), Options::default()).unwrap();
}

/// The DS instructions whose LDS address is M0's (and the lane's) translate (values:
/// `victory-gpu` `addr_tests`), except an append or consume where the host splits the console's
/// wave (wave64 on 32-lane host waves), which needs one atomic for both halves.
#[test]
fn lds_instructions_addressed_by_m0_translate() {
    use crate::native::tests::{fspx, fspx_resource};
    let program = |ds: [u32; 2], wave32: bool, threads: u16| {
        let code = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xBEFC_0380, // s_mov_b32 m0, 0
            ds[0],
            ds[1],
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE070_0000,
            0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF81_0000, // s_endpgm
        ];
        // Five user SGPRs and 512 bytes of LDS.
        let resources = [fspx_resource(3, 3, 0, 0, 4, 0)];
        let rsrc2 = 1 << 15 | 5 << 1;
        let part = fspx(&bytes_of(&code), rsrc2, [threads, 1, 1], wave32, &resources);
        crate::dxbc::build(&[(FourCC::ACCC, &part)])
    };
    let append = [0xD8F8_0000, 0x0100_0000]; // ds_append v1
    for ds in [
        append,
        [0xD8F4_0004, 0x0100_0000], // ds_consume v1 offset:4
        [0xDAC4_0000, 0x0100_0000], // ds_read_addtid_b32 v1
    ] {
        translate_with(&program(ds, true, 32), Options::default()).unwrap();
    }
    let host32 = Options { wave32_host: true, ..Options::default() };
    translate_with(&program(append, false, 32), host32).unwrap();
    let err = translate_with(&program(append, false, 64), host32).err();
    assert!(
        matches!(err, Some(ShaderError::Unsupported { what, .. })
            if what == "ds_append or ds_consume in a console wave the host splits"),
        "{err:?}"
    );
}

/// Some of the title's compute programs append to a counter in LDS without setting M0 (the counter
/// at 0 + offset, zeroed and read there by address): M0 read before it is written is LDS address 0,
/// not "a hardware SGPR input".
#[test]
fn an_append_with_m0_never_set_addresses_lds_from_0() {
    use crate::native::tests::{fspx, fspx_resource};
    let code = [
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xD8F8_0004, 0x0100_0000, // ds_append v1 offset:4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_0000,
        0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
        0xBF81_0000, // s_endpgm
    ];
    let resources = [fspx_resource(3, 3, 0, 0, 4, 0)];
    let part = fspx(&bytes_of(&code), 1 << 15 | 5 << 1, [32, 1, 1], true, &resources);
    translate_with(&crate::dxbc::build(&[(FourCC::ACCC, &part)]), Options::default()).unwrap();
}

/// A loaded buffer descriptor whose stride the program clears (`s_and_b32` of dword 1 with
/// 0xc000ffff, a title's form) is the loaded view read raw, the index adding nothing (RDNA2 ISA
/// 8.1.5): its dword loads translate, by index or by offset. A mask that clears only part of the
/// stride (which may then not be 0), one that also clears base address bits, and a typed load stay
/// refused.
#[test]
fn a_descriptor_with_its_stride_cleared_reads_the_loaded_view_raw() {
    use crate::native::tests::{fspx, fspx_resource};
    let program = |mask: u32, load: u32| {
        let code = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x870D_FF0D,
            mask, // s_and_b32 s13, s13, mask
            load,
            0x8003_0100, // buffer_load_* v1, v0, s[12:15], 0 (idxen or offen)
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xE070_0000,
            0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF81_0000, // s_endpgm
        ];
        // A raw UAV at table dword 0, the buffer read at dword 8.
        let resources = [fspx_resource(3, 3, 0, 0, 4, 0), fspx_resource(0, 3, 8, 0, 4, 0)];
        let part = fspx(&bytes_of(&code), 5 << 1, [32, 1, 1], true, &resources);
        translate_with(&crate::dxbc::build(&[(FourCC::ACCC, &part)]), Options::default())
    };
    let (idxen, offen, typed) = (0xE030_2000, 0xE030_1000, 0xE000_2000);
    program(0xC000_FFFF, idxen).unwrap();
    program(0xC000_FFFF, offen).unwrap();
    let refused = |r: Result<_, ShaderError>| match r {
        Err(ShaderError::Unsupported { what, .. }) => what,
        Ok(_) => panic!("translated"),
        Err(e) => panic!("{e}"),
    };
    // A stride of 16 (bit 20) survives 0xc03fffff: the index would still count.
    let part = refused(program(0xC03F_FFFF, idxen));
    assert!(part.contains("dword 1 masked or merged"), "{part}");
    let base = refused(program(0xC000_FFFE, idxen));
    assert!(base.contains("dword 1 masked or merged"), "{base}");
    let format = refused(program(0xC000_FFFF, typed));
    assert!(format.contains("dword 1 masked or merged"), "{format}");
}

/// A buffer descriptor made raw (stride cleared, records times max(stride, 1), bounds in bytes:
/// a title's form before typed loads too) read by a typed load that names its own format
/// (`tbuffer_load_format_*`, by offset) is the loaded view read by bytes. A
/// `buffer_load_format_*`, which reads the view's elements by its format, stays refused.
#[test]
fn a_descriptor_made_raw_serves_a_typed_load_of_its_own_format() {
    use crate::native::tests::{fspx, fspx_resource};
    let program = |load: u32| {
        let code = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x9390_FF0D,
            0x000E_0010, // s_bfe_u32 s16, s13, 0x000e0010
            0x8490_8110, // s_max_u32 s16, s16, 1
            0x870D_FF0D,
            0xC000_FFFF, // s_and_b32 s13, s13, 0xc000ffff
            0x870F_FF0F,
            0xCFFF_FFFF, // s_and_b32 s15, s15, 0xcfffffff
            0x930E_100E, // s_mul_i32 s14, s14, s16
            0x880F_FF0F,
            0x3000_0000, // s_or_b32 s15, s15, 0x30000000
            load,
            0x8003_0100, // v1 from v0 through s[12:15], offen
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xE070_0000,
            0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF81_0000, // s_endpgm
        ];
        let resources = [fspx_resource(3, 3, 0, 0, 4, 0), fspx_resource(0, 3, 8, 0, 4, 0)];
        let part = fspx(&bytes_of(&code), 5 << 1, [32, 1, 1], true, &resources);
        translate_with(&crate::dxbc::build(&[(FourCC::ACCC, &part)]), Options::default())
    };
    // tbuffer_load_format_x v1, v0, s[12:15], 0 offen format:[BUF_FMT_16_FLOAT]
    program(0xE868_1000).unwrap();
    // The same descriptor's dword loads translated before.
    program(0xE030_1000).unwrap();
    // buffer_load_format_x v1, v0, s[12:15], 0 offen
    match program(0xE000_1000) {
        Err(ShaderError::Unsupported { what, .. }) => {
            assert!(what.contains("dword 1 masked or merged"), "{what}")
        }
        r => panic!("{:?}", r.err()),
    }
}

#[test]
fn refusal2_missing_pixel_system_inputs_remain_refused() {
    let read_v0 = [
        0xF800_180F,
        0x0000_0000, // exp mrt0 v0, v0, v0, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    for (label, addr) in [
        ("Position", 1u32 << 8),
        ("FrontFace", 1u32 << 12),
        ("Ancillary", 1u32 << 13),
        ("SampleCoverage", 1u32 << 14),
    ] {
        let err = lower_ps(&read_v0, true, vec![], vec![], addr, 0, vec![], false)
            .err()
            .unwrap_or_else(|| panic!("{label}: expected refusal"));
        assert!(
            err.to_string()
                .contains("a pixel system input the signature lacks"),
            "{label}: got `{err}`"
        );
    }
}

// =========================================================================
// Refusal #3: "a hardware SGPR input"
// =========================================================================

#[test]
fn refusal3_interp_m0_sgpr_and_pops_lower() {
    let ps_in = vec![PsInput {
        usage: Usage::Generic,
        row: 0,
        mask: 1,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    }];
    let sig_in = vec![sig_el("TEXCOORD", 0, 0, 9, 0, Some(0), 4)];

    // 1. Direct `s_mov_b32 m0, s0` (s0 is below default `initialized_sgprs - 2`)
    //    plus `s_sendmsg(MSG_ORDERED_PS_DONE)` which implicitly reads M0 in IR.
    let direct_m0 = [
        0xBEFC_0300, // s_mov_b32 m0, s0
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_180F,
        0x0404_0404, // exp mrt0 v4, v4, v4, v4 done vm
        0xBF81_0000,
    ];
    let out = lower_ps(
        &direct_m0,
        true,
        vec![],
        ps_in.clone(),
        1 << 1,
        0,
        sig_in.clone(),
        false,
    )
    .unwrap();
    assert!(!out.rovs);

    // 2. Spilled through `v_writelane_b32` / `v_readlane_b32` into `m0`.
    let spilled_m0 = [
        0xD761_0007,
        0x0001_0000, // v_writelane_b32 v7, s0, 0
        0xD760_0001,
        0x0001_0107, // v_readlane_b32 s1, v7, 0
        0xBEFC_0301, // s_mov_b32 m0, s1
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0xF800_180F,
        0x0404_0404,
        0xBF81_0000,
    ];
    lower_ps(&spilled_m0, true, vec![], ps_in, 1 << 1, 0, sig_in, false).unwrap();

    // 3. Guessed PS with POPS (`src_pops_exiting_wave_id` + `MSG_ORDERED_PS_DONE`).
    let guessed_pops = [
        0xBF06_EF09, // s_cmp_eq_u32 s9, src_pops_exiting_wave_id
        0x8502_8281, // s_cselect_b32 s2, 1, 2
        0x7E00_0202, // v_mov_b32_e32 v0, s2
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    let container =
        graphics::container(HwStage::Ps, &bytes_of(&guessed_pops), &[0u8; 0x10]).unwrap();
    translate_with(&container, Options::default()).unwrap();
}

#[test]
fn refusal3_non_interp_hardware_sgpr_remains_refused() {
    // 1. Live-in s0 read by VALU and exported.
    let alu_sgpr = [
        0x7E00_0200, // v_mov_b32_e32 v0, s0
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    let err = lower_ps(&alu_sgpr, true, vec![], vec![], 0, 0, vec![], false)
        .err()
        .expect("expected refusal");
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");

    // 2. Non-guessed PS with db_shader_control == 0 reading collision id s9.
    let pops_no_ctrl = [
        0x7E00_0209, // v_mov_b32_e32 v0, s9
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    let err = lower_ps(&pops_no_ctrl, true, vec![], vec![], 0, 0, vec![], false)
        .err()
        .expect("expected refusal");
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");

    // 3. Live-in s0 moved to m0 and read by both v_interp AND v_movrels_b32.
    let ps_in = vec![PsInput {
        usage: Usage::Generic,
        row: 0,
        mask: 1,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    }];
    let sig_in = vec![sig_el("TEXCOORD", 0, 0, 9, 0, Some(0), 4)];
    let m0_movrels = [
        0xBEFC_0300, // s_mov_b32 m0, s0
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0x7E0A_8700, // v_movrels_b32_e32 v5, v0
        0x0608_0B04, // v_add_f32_e32 v4, v4, v5
        0xF800_180F,
        0x0404_0404,
        0xBF81_0000,
    ];
    let err = lower_ps(&m0_movrels, true, vec![], ps_in, 1 << 1, 0, sig_in, false)
        .err()
        .expect("expected refusal");
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");
}

#[test]
fn guessed_pixel_masked_prim_mask_anchors_the_pops_input() {
    let code = [
        0x8705_FF06,
        0x0000_FFFF, // s_and_b32 s5, s6, 0xffff
        0xBEFC_0305, // s_mov_b32 m0, s5
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0xBF06_EF07, // s_cmp_eq_u32 s7, src_pops_exiting_wave_id
        0x8512_8081, // s_cselect_b32 s18, 1, 0
        0x7E0A_0212, // v_mov_b32 v5, s18
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_180F,
        0x0404_0504, // exp mrt0 v4, v5, v4, v4 done vm
        0xBF81_0000,
    ];
    let lower = |code: &[u32]| {
        let container = graphics::container(HwStage::Ps, &bytes_of(code), &[0; 0x10]).unwrap();
        translate_with(&container, Options::default())
    };
    lower(&code).unwrap();

    let mut mixed = code;
    mixed[7] = 0x7E0A_0206; // v_mov_b32 v5, s6: the mask also escapes into the export.
    let err = lower(&mixed).err().unwrap();
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");

    let mut scc_use = code;
    scc_use.swap(5, 6); // The select now consumes the SCC result of the mask operation.
    let err = lower(&scc_use).err().unwrap();
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");
}

#[test]
fn guessed_pixel_extracted_prim_mask_anchors_the_pops_input() {
    // Bit-field extraction is another way to prepare M0 for interpolation.
    // The collision ID is s7 because the primitive mask is live-in s6.
    let code = [
        0x9385_FF06,
        0x0010_0000, // s_bfe_u32 s5, s6, 0x00100000 (offset 0, width 16)
        0xBEFC_0305, // s_mov_b32 m0, s5
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0xBF06_EF07, // s_cmp_eq_u32 s7, src_pops_exiting_wave_id
        0x8512_8081, // s_cselect_b32 s18, 1, 0
        0x7E0A_0212, // v_mov_b32 v5, s18
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_180F,
        0x0404_0504, // exp mrt0 v4, v5, v4, v4 done vm
        0xBF81_0000,
    ];
    for wave32_host in [false, true] {
        let lower = |code: &[u32]| {
            let container = graphics::container(HwStage::Ps, &bytes_of(code), &[0; 0x10]).unwrap();
            translate_with(&container, Options { wave32_host, ..Options::default() })
        };
        lower(&code).unwrap();

        let mut mixed = code;
        mixed[7] = 0x7E0A_0206; // The live-in mask also escapes into an export.
        let err = lower(&mixed).expect_err("exported mask must remain refused");
        assert!(err.to_string().contains("a hardware SGPR input"), "{err}");

        let mut extracted = code;
        extracted[7] = 0x7E0A_0205; // The extracted mask escapes into an export.
        let err = lower(&extracted).expect_err("exported extraction must remain refused");
        assert!(err.to_string().contains("a hardware SGPR input"), "{err}");

        let mut scc_use = code;
        scc_use.swap(5, 6); // The select consumes the extraction's SCC result.
        let err = lower(&scc_use).expect_err("observable SCC must remain refused");
        assert!(err.to_string().contains("a hardware SGPR input"), "{err}");
    }
}

// =========================================================================
// Refusal #4: "one input interpolated two ways"
// =========================================================================

#[test]
fn refusal4_dual_interpolated_generic_input_splits_onto_fresh_row() {
    // Interpolates attr0.x with PerspectiveCenter (v[0:1]), PerspectiveCentroid
    // (v[2:3]), AND flat provoking vertex (v_interp_mov_f32 p0).
    let code = [
        0xBEFC_0300, // s_mov_b32 m0, s0
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0xC814_0002, // v_interp_p1_f32 v5, v2, attr0.x
        0xC815_0003, // v_interp_p2_f32 v5, v3, attr0.x
        0xC81A_0002, // v_interp_mov_f32 v6, p0, attr0.x
        0x0608_0B04, // v_add_f32_e32 v4, v4, v5
        0x0608_0D04, // v_add_f32_e32 v4, v4, v6
        0xF800_180F,
        0x0404_0404, // exp mrt0 v4, v4, v4, v4 done vm
        0xBF81_0000,
    ];
    let ps_in = vec![PsInput {
        usage: Usage::Generic,
        row: 0,
        mask: 1,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    }];
    let sig_in = vec![sig_el("TEXCOORD", 0, 0, 9, 0, Some(0), 4)];
    let out = lower_ps(
        &code,
        true,
        vec![],
        ps_in,
        (1 << 1) | (1 << 2),
        0,
        sig_in,
        false,
    )
    .unwrap();
    let isg1 = parsed_isg1(&out.dxil);
    assert_eq!(isg1.len(), 3, "expected 3 split TEXCOORD elements: {isg1:?}");
    let mut rows: Vec<u32> = isg1.iter().filter_map(|e| e.row).collect();
    rows.sort_unstable();
    rows.dedup();
    assert_eq!(rows.len(), 3, "each mode must live on its own row");
    let mut indices: Vec<u32> = isg1.iter().map(|e| e.index).collect();
    indices.sort_unstable();
    indices.dedup();
    assert_eq!(indices, vec![0, 1, 2], "semantic indices must be unique");

    // Also via guessed graphics (`FSPG`) inferring multiple barycentric pairs.
    let container = graphics::container(HwStage::Ps, &bytes_of(&code), &[0u8; 0x10]).unwrap();
    translate_with(&container, Options::default()).unwrap();
}

#[test]
fn refusal4_non_generic_and_non_provoking_interp_remain_refused() {
    let ps_in = vec![PsInput {
        usage: Usage::Generic,
        row: 0,
        mask: 1,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    }];
    // 1. Non-provoking vertex (p10) stays refused.
    let non_provoking = [
        0xBEFC_0300, // s_mov_b32 m0, s0
        0xC81A_0000, // v_interp_mov_f32 v6, p10, attr0.x
        0xF800_180F,
        0x0606_0606,
        0xBF81_0000,
    ];
    let err = lower_ps(
        &non_provoking,
        true,
        vec![],
        ps_in.clone(),
        1 << 1,
        0,
        vec![sig_el("TEXCOORD", 0, 0, 9, 0, Some(0), 4)],
        false,
    )
    .err()
    .expect("expected refusal");
    assert!(
        err.to_string().contains("non-provoking vertex"),
        "got `{err}`"
    );

    // 2. Non-generic input (kind != 0) interpolated two ways stays refused.
    let dual = [
        0xBEFC_0300,
        0xC810_0000,
        0xC811_0001,
        0xC814_0002,
        0xC815_0003,
        0x0608_0B04,
        0xF800_180F,
        0x0404_0404,
        0xBF81_0000,
    ];
    let err = lower_ps(
        &dual,
        true,
        vec![],
        ps_in,
        (1 << 1) | (1 << 2),
        0,
        vec![sig_el("SV_ClipDistance", 0, 6, 9, 0, Some(0), 4)],
        false,
    )
    .err()
    .expect("expected refusal");
    assert!(
        err.to_string().contains("one input interpolated two ways"),
        "got `{err}`"
    );
}

// =========================================================================
// Former refusal #5: "an address in an SGPR that is no root parameter".
// Such an address now tracks as title memory and plain loads through it
// lower; what still refuses is reading the SGPR itself when nothing declares
// it ("a hardware SGPR input") or building a descriptor around it.
// =========================================================================

#[test]
fn refusal5_guessed_vs_inferred_heap_lowers_while_nonguessed_refuses() {
    // s_mov_b32 s3, 0
    // s_load_dwordx4 s[4:7], s[2:3], 0x0
    // s_waitcnt lgkmcnt(0)
    // buffer_load_dword v1, v0, s[4:7], 0 idxen
    // s_waitcnt vmcnt(0)
    // exp pos0 v1, v1, v1, v1 done
    // s_endpgm
    let guessed_heap_vs = [
        0xBE83_0380,
        0xF408_0101,
        0xFA00_0000,
        0xBF8C_C07F,
        0xE030_2000,
        0x8001_0100,
        0xBF8C_3F70,
        0xF800_08CF,
        0x0101_0101,
        0xBF81_0000,
    ];
    let container =
        graphics::container(HwStage::Vs, &bytes_of(&guessed_heap_vs), &[0u8; 0x10]).unwrap();
    translate_with(&container, Options::default()).unwrap();

    // Non-guessed VS (`layout = []`) reading from undeclared s[2:3] MUST still refuse:
    // the address tracks as title memory, but lowering then reads s[2:3] itself.
    let undeclared_load = [
        0xF400_0101,
        0xFA00_0000, // s_load_dword s4, s[2:3], 0x0
        0xBF8C_C07F,
        0x7E02_0204, // v_mov_b32_e32 v1, s4
        0xF800_08CF,
        0x0101_0101,
        0xBF81_0000,
    ];
    let err = lower_vs(&undeclared_load, true, vec![], false)
        .err()
        .expect("expected refusal");
    assert!(
        err.to_string().contains("a hardware SGPR input"),
        "got `{err}`"
    );
}

#[test]
fn guessed_vertex_rebases_a_header_relative_table_alongside_vertex_fetch() {
    let mut meta = [0u8; 0x30];
    meta[0] = 2;
    meta[12..15].copy_from_slice(&[0x40, 4, 1]);
    meta[0x10] = 0x41;
    meta[0x14..0x16].copy_from_slice(&[2, 3]); // CBV in a table
    meta[0x18] = 8; // descriptor at byte 32 of the table
    meta[0x24..0x27].copy_from_slice(&[4, 0, 3]); // four dwords, relative s0, title slot 3
    let code = [
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 32
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        0xF420_0406,
        0xFA00_0000, // s_buffer_load_dword s16, s[12:15], 0
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x7E04_0210, // v_mov_b32 v2, s16
        0xF800_08CF,
        0x0202_0201, // exp pos0 v1, v2, v2, v2 done
        0xBF81_0000,
    ];
    let lower = |code: &[u32], meta: &[u8]| {
        let container = graphics::container(HwStage::Vs, &bytes_of(code), meta).unwrap();
        translate_with(&container, Options::default())
    };
    let shader = lower(&code, &meta).unwrap();
    assert!(!shader.heap);
    assert_eq!(shader.user_data.len(), 1);
    assert_eq!(shader.user_data[0].slot, 3);
    assert_eq!(shader.user_data[0].root, RootType::Table);
    assert_eq!(shader.table_entries, [(3, 2)]);
    assert!(parsed_isg1(&shader.dxil).iter().any(|e| e.name == "VBU1"));

    // Reading a descriptor past the 16-entry table window (entry 20 = byte 640 = 0x280,
    // table_offset_dwords = 160) reports table_entries = [(3, 21)] for host re-mirroring.
    let mut wide_meta = meta;
    wide_meta[0x18] = 160;
    let mut wide_code = code;
    wide_code[4] = 0xFA00_0280; // s_load_dwordx4 s[12:15], s[4:5], 640
    let wide_shader = lower(&wide_code, &wide_meta).unwrap();
    assert_eq!(wide_shader.table_entries, [(3, 21)]);

    meta[0x25] = 4; // The original absolute interpretation also supports an explicit s4 record.
    assert_eq!(lower(&code, &meta).unwrap().user_data, shader.user_data);

    let unbound = [
        0xF400_0103,
        0xFA00_0000, // s_load_dword s4, s[6:7], 0
        0x7E02_0204, // v_mov_b32 v1, s4
        0xF800_08CF,
        0x0101_0101, // exp pos0 v1, v1, v1, v1 done
        0xBF81_0000,
    ];
    meta[0x25] = 0;
    // An address in the undeclared s[6:7] pair is title memory: inference
    // declares the pair as root constants and the load lowers through the
    // memory space.
    let unbound_shader = lower(&unbound, &meta).unwrap();
    assert!(
        unbound_shader
            .user_data
            .iter()
            .any(|u| u.index == 6 && u.root == RootType::Constants)
            && unbound_shader
                .user_data
                .iter()
                .any(|u| u.index == 7 && u.root == RootType::Constants),
        "{:?}",
        unbound_shader.user_data
    );
}

#[test]
fn guessed_vertex_rebases_a_header_relative_root_view() {
    let mut meta = [0u8; 0x30];
    meta[0] = 2;
    meta[12..15].copy_from_slice(&[0x40, 4, 1]);
    meta[0x10] = 0x41;
    meta[0x14..0x16].copy_from_slice(&[2, 1]); // Root CBV
    meta[0x26] = 3; // Title slot 3
    let code = bytes_of(&[
        0xBE88_0404, // s_mov_b64 s[8:9], s[4:5]
        0xBE8A_03C1, // s_mov_b32 s10, -1
        0xBE8B_03FF,
        0x3001_4FAC, // s_mov_b32 s11, 0x30014fac
        0xF420_0304,
        0xFA00_0000, // s_buffer_load_dword s12, s[8:11], 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E02_020C, // v_mov_b32 v1, s12
        0xF800_08CF,
        0x0101_0101, // exp pos0 v1, v1, v1, v1 done
        0xBF81_0000,
    ]);
    let lower = |meta: &[u8]| {
        let container = graphics::container(HwStage::Vs, &code, meta).unwrap();
        translate_with(&container, Options::default())
    };
    assert_eq!(lower(&meta).unwrap().root_cbvs, [3]);
    meta[13] = 6;
    let err = lower(&meta).err().unwrap();
    assert!(
        err.to_string()
            .contains("a buffer descriptor built around a computed address"),
        "{err}"
    );
}

// =========================================================================
// Refusal #6: "a vertex fetch format the host's vertex layout lacks"
// =========================================================================

/// Encodes `tbuffer_load_format_{x,xy,xyz,xyzw} v[1..], v0, s[4:7], 0 format:<fmt> idxen`.
fn tbuffer_fetch_vs(fmt: u8, count: u32) -> [u32; 10] {
    let w0 = 0xE800_2000 | (u32::from(fmt) << 19) | ((count - 1) << 16);
    let w1 = 0x8001_0100;
    let exp_v = match count {
        1 => 0x0101_0101,
        2 => 0x0202_0201,
        3 => 0x0303_0201,
        _ => 0x0403_0201,
    };
    [
        0xF408_0100,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0x0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        w0,
        w1,          // tbuffer_load_format_* v[1..], v0, s[4:7], 0 format:<fmt> idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_08CF,
        exp_v,       // exp pos0 ... done
        0xBF81_0000, // s_endpgm
        0xBF80_0000, // s_nop 0 (padding)
    ]
}

#[test]
fn refusal6_all_42_vertex_fetch_formats_and_partial_loads_lower() {
    // (GFX10 unified format number, load count, expected semantic name, expected comp, expected cols)
    let cases: &[(u8, u32, &str, u8, u8)] = &[
        // 8-bit 1/2/4 channels
        (1, 1, "VBN1", 9, 1),
        (2, 1, "VBQ1", 9, 1),
        (5, 1, "VBB1", 5, 1),
        (6, 1, "VBC1", 4, 1),
        (14, 2, "VBN2", 9, 2),
        (15, 2, "VBQ2", 9, 2),
        (18, 2, "VBB2", 5, 2),
        (19, 2, "VBC2", 4, 2),
        (56, 4, "VBN4", 9, 4),
        (57, 4, "VBQ4", 9, 4),
        (60, 4, "VBB4", 5, 4),
        (61, 4, "VBC4", 4, 4),
        // 16-bit 1/2/4 channels
        (7, 1, "VBM1", 9, 1),
        (8, 1, "VBS1", 9, 1),
        (11, 1, "VBW1", 5, 1),
        (12, 1, "VBK1", 4, 1),
        (13, 1, "VBH1", 9, 1),
        (23, 2, "VBM2", 9, 2),
        (24, 2, "VBS2", 9, 2),
        (27, 2, "VBW2", 5, 2),
        (28, 2, "VBK2", 4, 2),
        (29, 2, "VBH2", 9, 2),
        (65, 4, "VBM4", 9, 4),
        (66, 4, "VBS4", 9, 4),
        (69, 4, "VBW4", 5, 4),
        (70, 4, "VBK4", 4, 4),
        (71, 4, "VBH4", 9, 4),
        // 32-bit 1/2/3/4 channels
        (20, 1, "VBU1", 5, 1),
        (21, 1, "VBI1", 4, 1),
        (22, 1, "VBF1", 9, 1),
        (62, 2, "VBU2", 5, 2),
        (63, 2, "VBI2", 4, 2),
        (64, 2, "VBF2", 9, 2),
        (72, 3, "VBU3", 5, 3),
        (73, 3, "VBI3", 4, 3),
        (74, 3, "VBF3", 9, 3),
        (75, 4, "VBU4", 5, 4),
        (76, 4, "VBI4", 4, 4),
        (77, 4, "VBF4", 9, 4),
        // Packed formats
        (36, 3, "VBG3", 9, 3), // [11, 11, 10] Float
        (50, 4, "VBP4", 9, 4), // [10, 10, 10, 2] Unorm
        (51, 4, "VBT4", 9, 4), // [10, 10, 10, 2] Snorm
        // Partial channel loads (w.len() > count)
        (71, 3, "VBH4", 9, 4), // xyz of [16,16,16,16] Float
        (50, 3, "VBP4", 9, 4), // xyz of [10,10,10,2] Unorm
        (56, 3, "VBN4", 9, 4), // xyz of [8,8,8,8] Unorm
    ];
    for &(fmt, count, expect_name, expect_comp, expect_cols) in cases {
        let code = tbuffer_fetch_vs(fmt, count);
        for rect_list in [false, true] {
            let out = lower_vs(&code, true, vertex_table_layout(), rect_list)
                .unwrap_or_else(|e| panic!("fmt={fmt} count={count} rect={rect_list}: {e}"));
            let isg1 = parsed_isg1(&out.dxil);
            let el = isg1
                .iter()
                .find(|e| e.name == expect_name)
                .unwrap_or_else(|| {
                    panic!("fmt={fmt} count={count}: missing `{expect_name}` in {isg1:?}")
                });
            assert_eq!(el.comp, expect_comp, "{expect_name}");
            assert_eq!(el.cols, expect_cols, "{expect_name}");
        }
    }
}

#[test]
fn refusal6_unsupported_vertex_fetch_formats_remain_refused() {
    let bad_tbuffer: &[(u8, u32, &str)] = &[
        (58, 4, "8_8_8_8 Uscaled"),
        (59, 4, "8_8_8_8 Sscaled"),
        (44, 4, "2_10_10_10 Unorm (low 2-bit)"),
        (43, 3, "10_11_11 Float (reversed)"),
        (54, 4, "10_10_10_2 Uint"),
        (55, 4, "10_10_10_2 Sint"),
        (29, 4, "xyzw load from 2-channel 16_16 Float (count > w.len())"),
    ];
    for &(fmt, count, label) in bad_tbuffer {
        let code = tbuffer_fetch_vs(fmt, count);
        let err = lower_vs(&code, true, vertex_table_layout(), false)
            .err()
            .unwrap_or_else(|| panic!("{label}: expected refusal"));
        assert!(
            err.to_string()
                .contains("a vertex fetch format the host's vertex layout lacks"),
            "{label}: got `{err}`"
        );
    }

    // Untyped `buffer_load_format_xyzw` (format == None) must remain refused.
    let untyped_format = [
        0xF408_0100,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0x0
        0xBF8C_C07F,
        0xE00C_2000,
        0x8001_0100, // buffer_load_format_xyzw v[1:4], v0, s[4:7], 0 idxen
        0xBF8C_3F70,
        0xF800_08CF,
        0x0403_0201,
        0xBF81_0000,
    ];
    let err = lower_vs(&untyped_format, true, vertex_table_layout(), false)
        .err()
        .expect("expected refusal");
    assert!(
        err.to_string()
            .contains("a vertex fetch format the host's vertex layout lacks"),
        "untyped buffer_load_format: got `{err}`"
    );
}

// =========================================================================
// Compute: "a typed buffer store of some channels or at an offset"
// =========================================================================

/// A wave32 compute program whose table in user data 0 holds a typed buffer
/// UAV: each lane stores registers from v1 on (copies of its lane number) to
/// element `lane` with the typed store `store` (two words).
fn typed_store_cs(store: [u32; 2]) -> Result<Lowered, ShaderError> {
    typed_store_cs_after(&[], store)
}

/// [`typed_store_cs`] with the words `setup` run just before the store.
fn typed_store_cs_after(setup: &[u32], store: [u32; 2]) -> Result<Lowered, ShaderError> {
    store_cs(4, setup, store)
}

/// [`typed_store_cs_after`] with a descriptor of `dwords` (4, or 8 for an
/// image) in s[8:].
fn store_cs(dwords: u8, setup: &[u32], store: [u32; 2]) -> Result<Lowered, ShaderError> {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        // s_load_dwordx4 s[8:11] (x8 s[8:15]), s[4:5], 0x0
        if dwords == 8 {
            0xF40C_0202
        } else {
            0xF408_0202
        },
        0xFA00_0000,
        0x7E02_0300, // v_mov_b32 v1, v0
        0x7E04_0300, // v_mov_b32 v2, v0
        0x7E06_0300, // v_mov_b32 v3, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    words.extend_from_slice(setup);
    words.extend([store[0], store[1], 0xBF81_0000]); // then s_endpgm
    let table = RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![table],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2: 3 << 1, threads: [32, 1, 1] }),
        code: bytes_of(&words),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        view_marks: false,
        rect_list: false,
        copy_params: [0; 32],
    })
}

#[test]
fn signed_i16_add_and_sub_lower_with_and_without_clamp() {
    for (opcode, opsel) in [(0x30d, 0b0001), (0x30e, 0b1010)] {
        for clamp in [false, true] {
            let setup = [
                0x7E02_02FF, 0xA1B2_C3D4, // v1: old destination
                0x7E04_02FF, 0x7FFF_8000, // v2: 32767 high, -32768 low
                0x7E06_02FF, 0x0001_0001, // v3: 1 in either half
                0xD400_0001 | opcode << 16 | u32::from(clamp) << 15 | opsel << 11,
                0x0002_0702, // v1 = op(v2, v3)
            ];
            store_cs(4, &setup, [0xE070_0000, 0x8002_0100]).unwrap();
        }
    }
}

/// Typed stores of one, two or three channels write the whole element (0 in
/// the channels they do not name: `stride_tests` runs one), so they lower as
/// the four-channel store does.
#[test]
fn typed_buffer_stores_of_some_channels_lower() {
    let stores: &[(u32, &str)] = &[
        (0xE010_2000, "buffer_store_format_x v1, v0, s[8:11], 0 idxen"),
        (0xE014_2000, "buffer_store_format_xy v[1:2], v0, s[8:11], 0 idxen"),
        (0xE018_2000, "buffer_store_format_xyz v[1:3], v0, s[8:11], 0 idxen"),
        (0xE01C_2000, "buffer_store_format_xyzw v[1:4], v0, s[8:11], 0 idxen"),
        (0xE210_2000, "buffer_store_format_d16_x v1, v0, s[8:11], 0 idxen"),
        (0xE214_2000, "buffer_store_format_d16_xy v1, v0, s[8:11], 0 idxen"),
        (0xE218_2000, "buffer_store_format_d16_xyz v[1:2], v0, s[8:11], 0 idxen"),
        (0xE21C_2000, "buffer_store_format_d16_xyzw v[1:2], v0, s[8:11], 0 idxen"),
    ];
    for &(w0, asm) in stores {
        let l = typed_store_cs([w0, 0x8002_0100]).unwrap_or_else(|e| panic!("{asm}: {e}"));
        assert!(!l.dxil.is_empty(), "{asm}");
    }
}

/// A typed view has no bytes inside an element to address: a typed store at
/// an offset stays refused.
#[test]
fn typed_buffer_stores_at_an_offset_remain_refused() {
    let stores: &[([u32; 2], &str)] = &[
        ([0xE010_2004, 0x8002_0100], "buffer_store_format_x v1, v0, s[8:11], 0 idxen offset:4"),
        ([0xE010_3000, 0x8002_0102], "buffer_store_format_x v1, v[2:3], s[8:11], 0 idxen offen"),
        ([0xE01C_2004, 0x8002_0100], "buffer_store_format_xyzw v[1:4], v0, s[8:11], 0 idxen offset:4"),
    ];
    for &(store, asm) in stores {
        let err = typed_store_cs(store).err().unwrap_or_else(|| panic!("{asm}: expected refusal"));
        assert!(err.to_string().contains("a typed buffer store at an offset"), "{asm}: got `{err}`");
    }
}

/// A scalar offset register that provably holds 0 (a move of the constant) is
/// no offset: the store lowers, whatever its channel count.
#[test]
fn typed_buffer_stores_at_a_zero_sgpr_offset_lower() {
    let stores: &[(u32, &str)] = &[
        (0xE01C_2000, "buffer_store_format_xyzw v[1:4], v0, s[8:11], s2 idxen"),
        (0xE010_2000, "buffer_store_format_x v1, v0, s[8:11], s2 idxen"),
    ];
    for &(w0, asm) in stores {
        // s_mov_b32 s2, 0
        let l = typed_store_cs_after(&[0xBE82_0380], [w0, 0x0202_0100])
            .unwrap_or_else(|e| panic!("{asm}: {e}"));
        assert!(!l.dxil.is_empty(), "{asm}");
    }
}

/// A scalar offset register holding another constant, or a value that is not
/// known, is an offset: refused, not dropped.
#[test]
fn typed_buffer_stores_at_a_nonzero_or_unknown_sgpr_offset_remain_refused() {
    let setups: &[(u32, &str)] = &[
        (0xBE82_0384, "s_mov_b32 s2, 4"),
        (0xBE82_0300, "s_mov_b32 s2, s0"),
    ];
    for &(setup, asm) in setups {
        // buffer_store_format_xyzw v[1:4], v0, s[8:11], s2 idxen
        let err = typed_store_cs_after(&[setup], [0xE01C_2000, 0x0202_0100])
            .err()
            .unwrap_or_else(|| panic!("{asm}: expected refusal"));
        assert!(err.to_string().contains("a typed buffer store at an offset"), "{asm}: got `{err}`");
    }
}

// =========================================================================
// "v_cvt_pknorm_u16_f32 is not supported"
// =========================================================================

#[test]
fn normalized_16_bit_packing_lowers_in_pixel_shader() {
    // v_mov_b32_e32 v1, 0.5
    // v_mov_b32_e32 v2, 1.0
    // <pack> v0, <src0>, <src1>
    // exp mrt0 v0, v0, v0, v0 done vm
    // s_endpgm
    let code = |pack: [u32; 2]| {
        [
            0x7E02_02F0,
            0x7E04_02F2,
            pack[0],
            pack[1],
            0xF800_180F,
            0x0000_0000,
            0xBF81_0000,
        ]
    };
    for (what, pack) in [
        (
            "v_cvt_pknorm_u16_f32 v0, v1, v2",
            [0xD769_0000, 0x0202_0501],
        ),
        (
            "v_cvt_pknorm_i16_f32 v0, v1, v2",
            [0xD768_0000, 0x0202_0501],
        ),
        (
            "v_cvt_pknorm_u16_f32 v0, |v1|, -v2",
            [0xD769_0100, 0x4202_0501],
        ),
        (
            "v_cvt_pknorm_u16_f32 v0, 0.5, v2",
            [0xD769_0000, 0x0202_04F0],
        ),
    ] {
        for wave32_host in [false, true] {
            lower_ps(&code(pack), true, vec![], vec![], 0, 0, vec![], wave32_host)
                .unwrap_or_else(|e| panic!("{what}, wave32_host={wave32_host}: {e}"));
        }
    }
    // v_cvt_pknorm_u16_f16 v0, v1, v2: the f16 forms stay refused.
    let f16 = code([0xD713_0000, 0x0202_0501]);
    let err = lower_ps(&f16, true, vec![], vec![], 0, 0, vec![], false)
        .err()
        .expect("expected refusal");
    assert!(
        err.to_string()
            .contains("v_cvt_pknorm_u16_f16 is not supported"),
        "got `{err}`"
    );
}

// =========================================================================
// "s_pack_ll_b32_b16 / s_bfm_b64 / s_bitreplicate_b64_b32 / v_mad_u32_u16
// is not supported"
// =========================================================================

#[test]
fn scalar_packs_masks_and_16_bit_mads_lower_in_pixel_shader() {
    // s_mov_b32 s0, 0x12345678
    // s_mov_b32 s1, 0x9abcdef0
    // <op>: a scalar result in s2 (s[2:3]), or the mad's in v0
    // v_mov_b32_e32 v0, s2         (scalar ops)
    // v_mov_b32_e32 v1, s2 (s3)    (scalar ops)
    // exp mrt0 v0, v1, v0, v1 done vm
    // s_endpgm
    let code = |op: &[u32], wide: bool| {
        let mut w = vec![0xBE80_03FF, 0x1234_5678, 0xBE81_03FF, 0x9ABC_DEF0];
        w.extend_from_slice(op);
        if op.len() == 1 {
            w.extend([0x7E00_0202, if wide { 0x7E02_0203 } else { 0x7E02_0202 }]);
        }
        w.extend([0xF800_180F, 0x0100_0100, 0xBF81_0000]);
        w
    };
    // v_mov_b32_e32 v1, s0; v_mov_b32_e32 v2, s1; then the mad.
    let mad = |word0: u32, word1: u32| [0x7E02_0200, 0x7E04_0201, word0, word1];
    for (what, op) in [
        ("s_pack_ll_b32_b16 s2, s0, s1", code(&[0x9902_0100], false)),
        ("s_pack_lh_b32_b16 s2, s0, s1", code(&[0x9982_0100], false)),
        ("s_pack_hh_b32_b16 s2, s0, s1", code(&[0x9A02_0100], false)),
        ("s_bfm_b64 s[2:3], s0, s1", code(&[0x9282_0100], true)),
        ("s_bfm_b64 s[2:3], 40, 3", code(&[0x9282_83A8], true)),
        (
            "s_bitreplicate_b64_b32 s[2:3], s0",
            code(&[0xBE82_3B00], true),
        ),
        (
            "v_mad_u32_u16 v0, v1, v2, v1",
            code(&mad(0xD773_0000, 0x0406_0501), false),
        ),
        (
            "v_mad_u32_u16 v0, v1, v2, v1 op_sel:[1,0,0,0]",
            code(&mad(0xD773_0800, 0x0406_0501), false),
        ),
        (
            "v_mad_u32_u16 v0, v1, v2, v1 op_sel:[0,1,0,0]",
            code(&mad(0xD773_1000, 0x0406_0501), false),
        ),
        (
            "v_mad_u32_u16 v0, v1, v2, v1 op_sel:[1,1,0,0]",
            code(&mad(0xD773_1800, 0x0406_0501), false),
        ),
        (
            "v_mad_u32_u16 v0, s0, 7, v1",
            code(&mad(0xD773_0000, 0x0405_0E00), false),
        ),
    ] {
        for wave32_host in [false, true] {
            lower_ps(&op, true, vec![], vec![], 0, 0, vec![], wave32_host)
                .unwrap_or_else(|e| panic!("{what}, wave32_host={wave32_host}: {e}"));
        }
    }
    // v_mad_u32_u16 v0, v1, v2, v1 op_sel:[0,0,1,0]: op_sel on the 32-bit
    // addend has no meaning in the manual and stays refused.
    let opsel2 = code(&mad(0xD773_2000, 0x0406_0501), false);
    let err = lower_ps(&opsel2, true, vec![], vec![], 0, 0, vec![], false)
        .err()
        .expect("expected refusal");
    assert!(
        err.to_string()
            .contains("op_sel or modifiers on a 32-bit operand"),
        "got `{err}`"
    );
}

/// What "a computed buffer descriptor used as a view" names: only an image
/// instruction asks for a view of four computed dwords, as its sampler (told
/// apart here) or as a 128-bit image descriptor (`r128`, instruction bit 15).
/// The refusal is reported where the missing result is first needed.
#[test]
fn guessed_pixel_computed_samplers_and_short_images_remain_refused() {
    let mut meta = [0u8; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // CBV, root constants in slot 0, live-in s0
    let head = [
        0xBE8C_030A, // s_mov_b32 s12, s10
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF40C_0406,
        0xFA00_0000, // s_load_dwordx8 s[16:23], s[12:13], 0x0
        0xBE98_03FF,
        0x0001_2345, // s_mov_b32 s24, 0x12345
        0xBE99_03FF,
        0x00FF_F000, // s_mov_b32 s25, 0xfff000
        0xBE9A_0300, // s_mov_b32 s26, s0
        0xBE9B_0380, // s_mov_b32 s27, 0
        0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
        0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
        0xC80C_0100, // v_interp_p1_f32 v3, v0, attr0.y
        0xC80D_0101, // v_interp_p2_f32 v3, v1, attr0.y
    ];
    // image_sample_lz v[4:7], v[2:3], s[16:23], s[24:27] dmask:0xf dim:2D
    let sample = [0xF09C_0F08, 0x00C4_0402];
    let tail = [
        0xF800_180F,
        0x0706_0504, // exp mrt0 v4, v5, v6, v7 done vm
        0xBF81_0000,
    ];
    let options = Options { wave32_host: true, zero_copy_memory: true, ..Options::default() };
    let run = |mid: &[u32]| {
        let code: Vec<u32> = head.iter().chain(mid).chain(&tail).copied().collect();
        let c = graphics::container(HwStage::Ps, &bytes_of(&code), &meta).unwrap();
        translate_with(&c, options)
    };
    // Sampler dword 2 from a root constant: refused at the export, 8 bytes on.
    let at = 4 * head.len() + 8;
    let err = run(&sample).err().unwrap();
    assert_eq!(err, ShaderError::Unsupported { offset: at, what: "a sampler built from computed words" });
    // The same sampler with dword 2 fixed is an immediate sampler.
    let fixed = [0xBE9A_0380, sample[0], sample[1]]; // s_mov_b32 s26, 0
    assert_eq!(run(&fixed).unwrap().samplers, [[0x0001_2345, 0x00FF_F000, 0, 0]]);
    // A 128-bit image descriptor of loaded and built dwords.
    let short = [
        0xBE93_03FF,
        0x0001_2345, // s_mov_b32 s19, 0x12345
        fixed[0],
        sample[0] | 1 << 15, // ... s[16:19], s[24:27] ... r128
        sample[1],
    ];
    let err = run(&short).err().unwrap();
    assert_eq!(
        err,
        ShaderError::Unsupported { offset: at + 12, what: "a computed buffer descriptor used as a view" }
    );
}

// =========================================================================
// Compute: "an image store of some channels"
// =========================================================================

/// Image stores of some channels write the whole texel (0 in the channels
/// dmask leaves out: `valu_tests` runs some), the data holding the dmask's
/// channels in turn, so they lower as the four-channel store does.
#[test]
fn image_stores_of_some_channels_lower() {
    // v_mov_b32 v5, v0; v_mov_b32 v6, 0: the texel (lane, 0).
    let at = [0x7E0A_0300, 0x7E0C_0280];
    let stores: &[(u32, u32, &str)] = &[
        (
            0xF020_1108,
            0x0002_0105,
            "image_store v1, v[5:6], s[8:15] dmask:0x1 dim:2D unorm",
        ),
        (
            0xF020_1308,
            0x0002_0105,
            "image_store v[1:2], v[5:6], s[8:15] dmask:0x3 dim:2D unorm",
        ),
        (
            0xF020_1508,
            0x0002_0105,
            "image_store v[1:2], v[5:6], s[8:15] dmask:0x5 dim:2D unorm",
        ),
        (
            0xF020_1E08,
            0x0002_0105,
            "image_store v[1:3], v[5:6], s[8:15] dmask:0xe dim:2D unorm",
        ),
        (
            0xF020_1F08,
            0x0002_0105,
            "image_store v[1:4], v[5:6], s[8:15] dmask:0xf dim:2D unorm",
        ),
        (
            0xF020_1508,
            0x8002_0105,
            "image_store v1, v[5:6], s[8:15] dmask:0x5 dim:2D unorm d16",
        ),
    ];
    for &(w0, w1, asm) in stores {
        let l = store_cs(8, &at, [w0, w1]).unwrap_or_else(|e| panic!("{asm}: {e}"));
        assert!(!l.dxil.is_empty(), "{asm}");
    }
    // dmask 0: what the hardware stores is not documented.
    let err = store_cs(8, &at, [0xF020_1008, 0x0002_0105])
        .err()
        .expect("expected refusal");
    assert!(
        err.to_string().contains("an image store of no channel"),
        "got `{err}`"
    );
}

// =========================================================================
// Compute: "a buffer descriptor built around a computed address"
// =========================================================================

/// A wave32 compute program: s0 is a root constant holding a title address,
/// s2 a table with a raw UAV. Each lane builds a descriptor over `s0 + 16`
/// in s[4:7], loads dword `lane` from it by byte offset and stores it to
/// dword `lane` of the UAV. `computed` is what the translation of a
/// computed descriptor asks for (title memory, as the native host does).
fn computed_descriptor_cs(computed: bool) -> Result<Lowered, ShaderError> {
    let words = [
        0x8004_9000u32, // s_add_u32 s4, s0, 16
        0x8205_8080,    // s_addc_u32 s5, 0, 0
        0x8705_FF05,
        0x0000_FFFF, // s_and_b32 s5, s5, 0xffff
        0xBE86_0384, // s_mov_b32 s6, 4
        0xBE87_03FF,
        0x0001_4FAC, // s_mov_b32 s7, 0x00014fac
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_1000,
        0x8001_0201, // buffer_load_dword v2, v1, s[4:7], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let param = |root, sgprs, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![param(RootType::Constants, 2, 0, 0), param(RootType::Table, 1, 2, 1)],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 2,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2: 3 << 1, threads: [32, 1, 1] }),
        code: bytes_of(&words),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track_graphics(&p, &native.layout, native.stage, Some(&st), computed)?;
    let io = io::raise(&p, &native)?;
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: computed,
        view_marks: false,
        rect_list: false,
        copy_params: [0; 32],
    })
}

/// A compute buffer load through a descriptor built around a computed
/// address reads title memory, as the guessed pixel path does.
#[test]
fn compute_computed_buffer_loads_use_title_memory() {
    let err = computed_descriptor_cs(false).err().expect("refused without title memory");
    assert!(
        err.to_string().contains("a buffer descriptor built around a computed address"),
        "got `{err}`"
    );
    let l = computed_descriptor_cs(true).unwrap_or_else(|e| panic!("{e}"));
    assert!(!l.dxil.is_empty());
}

// =========================================================================
// Compute: "a structured buffer of unknown stride"
// =========================================================================

/// A wave32 compute program: s0 a table with an SRV, s1 a table with a raw
/// UAV. Each lane reads dword 1 of element `lane` of the SRV by index and
/// stores it to dword `lane` of the UAV. `stride` is what the metadata says
/// of the SRV (FSPX records say 0).
fn indexed_srv_cs(stride: u32) -> Result<Lowered, ShaderError> {
    let words = [
        0xBE84_0300u32, // s_mov_b32 s4, s0
        0xBE85_0380,    // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0403,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_2004,
        0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen offset:4
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let table = |user_data, slot| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    };
    let used = |kind, user_data, slot, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![table(0, 0), table(1, 1)],
        used: vec![used(ResourceType::Srv, 0, 0, stride), used(ResourceType::Uav, 1, 1, 0)],
        io: StageIo::Cs(ComputeIo { rsrc2: 2 << 1, threads: [32, 1, 1] }),
        code: bytes_of(&words),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        view_marks: false,
        rect_list: false,
        copy_params: [0; 32],
    })
}

/// A table SRV the metadata gives no stride is read by element at the
/// stride of its view in the host's view-stride table, as a heap-indexed
/// one is; one with a stride keeps it.
#[test]
fn indexed_table_buffers_without_a_stride_use_the_view_stride_table() {
    for stride in [0, 16] {
        let l = indexed_srv_cs(stride).unwrap_or_else(|e| panic!("stride {stride}: {e}"));
        assert!(!l.dxil.is_empty());
    }
}

/// A compute program adds dword 2 of 4-dword root constants (s6) to a 1-dword constant (s2) and
/// stores the sum through the table two slots on (s8): each read names its parameter and dword.
/// Without the root signature the 4-dword block is guessed one dword (no slot 2), so s6 is
/// refused.
#[test]
fn compute_root_constants_reach_the_user_data() {
    use crate::native::tests::{fspx_resource, fspx_with, rts0};
    let code = bytes_of(&[
        0xBE89_0380, // s_mov_b32 s9, 0
        0xF408_0304,
        0xFA00_0000, // s_load_dwordx4 s[12:15], s[8:9], 0x0
        0x7E00_0206, // v_mov_b32 v0, s6
        0x4A00_0002, // v_add_nc_u32 v0, s2, v0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x8003_0001, // buffer_store_dword v0, v1, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let records = [
        fspx_resource(2, 0, 0, 0, 2, 0),
        fspx_resource(2, 0, 0, 1, 4, 1),
        fspx_resource(3, 3, 0, 0, 8, 3),
    ];
    let container = |root: Option<&[u8]>| {
        let part = fspx_with(&code, 9 << 1, [64, 1, 1], true, &records, root);
        crate::dxbc::build(&[(FourCC::ACCC, &part)])
    };
    let root = rts0(&[(1, 1), (1, 4), (0, 0), (0, 0)]);
    let shader = translate_with(&container(Some(&root)), Options::default()).unwrap();
    let reads: Vec<_> = shader
        .user_data
        .iter()
        .map(|u| (u.index, u.slot, u.root, u.dword))
        .collect();
    for read in [
        (2, 0, RootType::Constants, 0),
        (6, 1, RootType::Constants, 2),
        (8, 3, RootType::Table, 0),
    ] {
        assert!(reads.contains(&read), "{reads:?}");
    }
    let err = translate_with(&container(None), Options::default()).unwrap_err();
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");
}

fn whole_wave_minimum(
    setup: &[u32],
    identity: u32,
    selectors: u32,
) -> Result<super::HostShader, ShaderError> {
    use crate::native::tests::{fspx, fspx_resource};
    let mut code = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7D82_0085, // v_cmp_lt_u32 vcc_lo, 5, v0
        0xBE8C_3C6A, // s_and_saveexec_b32 s12, vcc_lo
        0xBE82_3DC1, // s_or_saveexec_b32 s2, -1
    ];
    code.extend_from_slice(setup);
    code.extend([0xD501_0001, 0x000A_0000 | identity]); // v_cndmask_b32 v1, identity, v0, s2
    if identity == 255 {
        code.push(i32::MAX as u32);
    }
    // Each row of 16 reduces in four butterflies.
    for ctrl in [0xB1, 0x4E, 0x141, 0x140] {
        code.extend([0x2202_02FA, 0xFF00_0001 | ctrl << 8]); // v_min_i32_dpp v1, v1, v1
    }
    code.extend([
        0xD778_0002,
        selectors,   // v_permlanex16_b32 v2, v1, src1, src2
        0x2202_0501, // v_min_i32 v1, v1, v2
        0xBEFE_0302, // s_mov_b32 exec_lo, s2
        0xBEFE_030C, // s_mov_b32 exec_lo, s12
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8002_0103, // buffer_store_dword v1, v3, s[8:11], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let resources = [fspx_resource(3, 3, 0, 0, 4, 0)];
    let part = fspx(&bytes_of(&code), 5 << 1, [32, 1, 1], true, &resources);
    let container = crate::dxbc::build(&[(FourCC::ACCC, &part)]);
    translate_with(&container, Options::default())
}

/// Reduce the active lanes with signed min. Each inactive lane contributes INT_MAX,
/// supplied either directly to v_cndmask or through an SGPR in the whole-wave section.
#[test]
fn a_whole_wave_minimum_accepts_a_scalar_literal_identity() {
    let minimum = |setup: &[u32], identity| whole_wave_minimum(setup, identity, 0x0305_8301);
    let direct = minimum(&[], 255).expect("literal identity");
    let moved = minimum(&[0xBE86_03FF, 0x7FFF_FFFF], 6).expect("s_mov_b32 s6, INT_MAX");
    assert_eq!(moved.dxil, direct.dxil);

    let exec_off = [
        0xBEFE_0380, // s_mov_b32 exec_lo, 0
        0xBE86_03FF, 0x7FFF_FFFF, // s_mov_b32 s6, INT_MAX
        0xBEFE_03C1, // s_mov_b32 exec_lo, -1
    ];
    let moved = minimum(&exec_off, 6).expect("scalar move with EXEC off");
    assert_eq!(moved.dxil, direct.dxil);

    // Overwriting INT_MAX with -1 no longer supplies the signed-min identity.
    let wrong = minimum(&[0xBE86_03FF, 0x7FFF_FFFF, 0xBE86_03C1], 6);
    assert!(matches!(
        wrong,
        Err(ShaderError::Unsupported { what: super::wave::NO_SHAPE, .. })
    ));
    // A scalar register read is not a literal.
    let dynamic = minimum(&[0xBE86_0302], 6); // s_mov_b32 s6, s2 (saved EXEC)
    assert!(matches!(dynamic, Err(ShaderError::Unsupported { what, .. })
        if what.ends_with("a scalar ALU result")));
}

#[test]
fn a_whole_wave_minimum_requires_constant_sgpr_permlane_selectors() {
    let setup = [
        0xBE86_03C1, // s_mov_b32 s6, -1
        0xBE87_03FF,
        0x7654_3210, // s_mov_b32 s7, 0x76543210
    ];

    // s2 holds the saved EXEC mask from the lane comparison.
    let err = whole_wave_minimum(&setup, 255, 0x001C_0501) // v_permlanex16_b32 v2, v1, s2, s7
        .unwrap_err();
    assert!(
        matches!(
            err,
            ShaderError::Unsupported {
                what: "a whole-wave section step the simulation does not follow: a v_permlane with computed lane selects",
                ..
            }
        ),
        "{err}"
    );

    let direct = whole_wave_minimum(&[], 255, 0x0305_8301) // v_permlanex16_b32 v2, v1, -1, -1
        .expect("inline selectors lower");
    let moved = whole_wave_minimum(&setup, 255, 0x001C_0D01) // v_permlanex16_b32 v2, v1, s6, s7
        .expect("constant SGPR selectors lower");
    assert_eq!(moved.dxil, direct.dxil);
}

/// A value a whole-wave section leaves without a shape is refused naming the
/// first step the simulation does not model, or, when it models every step,
/// saying the value is no reduction or prefix.
#[test]
fn whole_wave_sections_name_the_step_they_do_not_follow() {
    use crate::native::tests::{fspx, fspx_resource};
    let program = |step: &[u32]| {
        let mut code = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0x7E02_0280, // v_mov_b32 v1, 0
            0xBE82_3DC1, // s_or_saveexec_b32 s2, -1
        ];
        code.extend_from_slice(step);
        code.extend([
            0xBEFE_0302, // s_mov_b32 exec_lo, s2
            0xE070_0000,
            0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF81_0000, // s_endpgm
        ]);
        let resources = [fspx_resource(3, 3, 0, 0, 4, 0)];
        let part = fspx(&bytes_of(&code), 5 << 1, [32, 1, 1], true, &resources);
        crate::dxbc::build(&[(FourCC::ACCC, &part)])
    };
    let prefix = "a whole-wave section step the simulation does not follow: ";
    for (step, why) in [
        (&[0x0802_0100][..], "a VALU operation it does not model"), // v_sub_f32 v1, v0, v0
        // v_mov_b32_dpp v1, v0 row_shr:1: each lane its neighbour's value.
        (&[0x7E02_02FA, 0xFF01_1100], "a value that is no reduction or prefix"),
    ] {
        let err = translate_with(&program(step), Options::default()).err();
        assert!(
            matches!(err, Some(ShaderError::Unsupported { what, .. })
                if what.strip_prefix(prefix) == Some(why)),
            "{err:?}"
        );
    }
}

/// "a cross-lane operation outside compute" names the operation and the stage, so a tester log
/// shows which form the title's pixel programs use (127 of 639 refusals in the M5 Ultra log of
/// #147, all graphics). A quad permutation still lowers in a pixel program, not in a vertex one.
#[test]
fn a_cross_lane_operation_outside_compute_says_which_and_where() {
    const MRT0: [u32; 2] = [0xF800_180F, 0x0101_0101]; // exp mrt0 v1, v1, v1, v1 done vm
    const POS0: [u32; 2] = [0xF800_08CF, 0x0101_0101]; // exp pos0 v1, v1, v1, v1 done
    const END: u32 = 0xBF81_0000; // s_endpgm
    let row_shr = [0x7E02_02FA, 0xFF01_1100]; // v_mov_b32_dpp v1, v0 row_shr:1
    let quad = [0x7E02_02FA, 0xFF00_B100]; // v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2]
    // ds_bpermute_b32 v1, v0, v0; s_waitcnt lgkmcnt(0)
    let bpermute = [0xDACC_0000, 0x0100_0000, 0xBF8C_C07F];
    // v_readlane_b32 s4, v0, 3; v_mov_b32 v1, s4
    let readlane = [0xD760_0004, 0x0201_0700, 0x7E02_0204];
    let with = |op: &[u32], exp: [u32; 2]| [op, &exp[..], &[END]].concat();
    let ps = |code: &[u32]| lower_ps(code, false, vec![], vec![], 0, 0, vec![], false);

    ps(&with(&quad, MRT0)).unwrap_or_else(|e| panic!("quad_perm: {e}"));
    for (op, want) in [
        (
            &row_shr[..],
            "DPP row_shl, row_shr or row_ror in a pixel program",
        ),
        (&bpermute[..], "ds_bpermute_b32 in a pixel program"),
        (
            &readlane[..],
            "v_readlane_b32 of a constant lane in a pixel program",
        ),
    ] {
        let err = ps(&with(op, MRT0)).err().expect(want).to_string();
        assert!(
            err.contains(&format!("a cross-lane operation outside compute: {want}")),
            "{err}"
        );
    }
    for (op, want) in [
        (&bpermute[..], "ds_bpermute_b32 in a vertex program"),
        (&quad[..], "DPP quad_perm in a vertex program"),
    ] {
        let vs = lower_vs(&with(op, POS0), false, vec![], false);
        let err = vs.err().expect(want).to_string();
        assert!(
            err.contains(&format!("a cross-lane operation outside compute: {want}")),
            "{err}"
        );
    }
}

/// A pixel program reads lane 31 of a whole-wave minimum in a second section: every lane of
/// the first section's result holds the minimum, so the lane read is the reduction itself, with
/// no read of a console lane (`WaveReadLaneAt`) in the host's wave. Outside a section too.
#[test]
fn a_pixel_program_reads_a_lane_of_a_whole_wave_reduction_as_the_reduction() {
    let mut code = vec![
        0xBE82_3DC1, // s_or_saveexec_b32 s2, -1
        0xD501_0001, 0x000A_00C1, // v_cndmask_b32 v1, -1, v0, s2
    ];
    // Each row of 16 reduces in four butterflies.
    for ctrl in [0xB1, 0x4E, 0x141, 0x140] {
        code.extend([0x2602_02FA, 0xFF00_0001 | ctrl << 8]); // v_min_u32_dpp v1, v1, v1
    }
    code.extend([
        0xD778_0002, 0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x2602_0501, // v_min_u32 v1, v1, v2
        0xBEFE_0302, // s_mov_b32 exec_lo, s2
        0xBE83_3DC1, // s_or_saveexec_b32 s3, -1
        0xD760_0004, 0x0001_3F01, // v_readlane_b32 s4, v1, 31
        0xBEFE_0303, // s_mov_b32 exec_lo, s3
        0x7E02_0204, // v_mov_b32 v1, s4
        0xF800_180F, 0x0101_0101, // exp mrt0 v1, v1, v1, v1 done vm
        0xBF81_0000, // s_endpgm
    ]);
    // The same read outside a section, under the program's EXEC.
    let mut bare = code.clone();
    bare.retain(|&w| w != 0xBE83_3DC1 && w != 0xBEFE_0303);
    for code in [code, bare] {
        let l = lower_ps(&code, false, vec![], vec![], 0, 0, vec![], false).expect("lowers");
        assert!(!l.wave_active_calls.is_empty());
        assert!(l.wave_read_lane_calls.is_empty(), "{:?}", l.wave_read_lane_calls);
    }
}

/// A scalar step amid a whole-wave section that reads neither EXEC nor VCC (`wave::passthrough`)
/// computes what it does outside one: a shift of the lane read of an earlier reduction, as the
/// title's pixel programs place it between the DPP steps of the next, lowers as itself. A shift
/// of EXEC itself stays refused, naming the instruction.
#[test]
fn a_scalar_step_amid_a_whole_wave_section_lowers_as_itself() {
    let program = |shift: u32| {
        let mut code = vec![
            0xBE82_3DC1, // s_or_saveexec_b32 s2, -1
            0xD501_0001, 0x000A_00C1, // v_cndmask_b32 v1, -1, v0, s2
        ];
        for ctrl in [0xB1, 0x4E, 0x141, 0x140] {
            code.extend([0x2602_02FA, 0xFF00_0001 | ctrl << 8]); // v_min_u32_dpp v1, v1, v1
        }
        code.extend([
            0xD778_0002, 0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
            0x2602_0501, // v_min_u32 v1, v1, v2
            0xBEFE_0302, // s_mov_b32 exec_lo, s2
            0xBE83_3DC1, // s_or_saveexec_b32 s3, -1
            0xD760_0004, 0x0001_3F01, // v_readlane_b32 s4, v1, 31
            shift,       // s_lshr_b32 s5, s4 (or exec_lo), 5
            0xBEFE_0303, // s_mov_b32 exec_lo, s3
            0x7E02_0205, // v_mov_b32 v1, s5
            0xF800_180F, 0x0101_0101, // exp mrt0 v1, v1, v1, v1 done vm
            0xBF81_0000, // s_endpgm
        ]);
        code
    };
    let ps = |code: &[u32]| lower_ps(code, false, vec![], vec![], 0, 0, vec![], false);
    let l = ps(&program(0x9005_8504)).unwrap_or_else(|e| panic!("s4: {e}"));
    assert!(!l.wave_active_calls.is_empty());
    let err = ps(&program(0x9005_857E)).err().expect("a shift of EXEC").to_string();
    assert!(err.contains("s_lshr_b32"), "{err}");
}

/// A wave64 pixel program's loop carries a mask that is SCC's all-or-nothing on entry and
/// `s_mov_b64 -1` on the way round. The structure cannot join it (it is not a function of fixed
/// lane bits), and the host's lanes are not the console's, but data of every lane or none is that
/// bit in any lane. Any other constant stays refused.
#[test]
fn a_pixel_program_joins_a_loop_mask_of_all_lanes_or_none_on_the_hosts_lanes() {
    let program = |back: u32| {
        [
            0x7E08_0500, // v_readfirstlane_b32 s4, v0
            0xBF06_8004, // s_cmp_eq_u32 s4, 0
            0x8586_80C1, // s_cselect_b64 s[6:7], -1, 0
            0xBE88_0383, // s_mov_b32 s8, 3
            0x7E02_0280, // v_mov_b32 v1, 0
            // loop:
            0x4A04_0281, // v_add_nc_u32 v2, 1, v1
            0xD501_0001, 0x001A_0501, // v_cndmask_b32 v1, v1, v2, s[6:7]
            0xBE86_0400 | back, // s_mov_b64 s[6:7], back
            0x8088_8108, // s_sub_u32 s8, s8, 1
            0xBF07_8008, // s_cmp_lg_u32 s8, 0
            0xBF85_FFF9, // s_cbranch_scc1 loop
            0xF800_180F, 0x0101_0101, // exp mrt0 v1, v1, v1, v1 done vm
            0xBF81_0000, // s_endpgm
        ]
    };
    let ps = |code: &[u32]| lower_ps(code, true, vec![], vec![], 0, 0, vec![], false);
    ps(&program(0xC1)).unwrap_or_else(|e| panic!("-1: {e}"));
    ps(&program(0x80)).unwrap_or_else(|e| panic!("0: {e}"));
    let err = ps(&program(0x81)).err().expect("1 is some lanes").to_string();
    assert!(err.contains("data, on the host's lanes"), "{err}");
}

/// A register holding data is reused as a mask in an if's then part, and the else part, a block
/// further down, first clears its own lanes' bits of it (`s_andn2` by EXEC). No lane that skipped
/// the then part reads the data, so the join needs no bit from that edge. With `s_and` by EXEC
/// the else lanes read it, and that stays refused.
#[test]
fn a_pixel_program_joins_data_an_else_part_clears_with_its_exec() {
    let program = |clear: u32| {
        [
            0x7E00_0500, // v_readfirstlane_b32 s0, v0
            0xD404_0001, 0x0002_0300, // v_cmp_gt_f32 s1, v0, v1
            0xBE82_3C01, // s_and_saveexec_b32 s2, s1
            0xBF88_0001, // s_cbranch_execz else
            0xBE80_03C1, // s_mov_b32 s0, -1
            // else:
            0x8A7E_7E02, // s_andn2_b32 exec_lo, s2, exec_lo
            0xBF88_0004, // s_cbranch_execz end
            0xD401_0003, 0x0002_0500, // v_cmp_lt_f32 s3, v0, v2
            clear,       // s_andn2_b32 (or s_and_b32) s0, s0, exec_lo
            0x8800_0300, // s_or_b32 s0, s0, s3
            // end:
            0xBEFE_0302, // s_mov_b32 exec_lo, s2
            0xD501_0001, 0x0002_0300, // v_cndmask_b32 v1, v0, v1, s0
            0xF800_180F, 0x0101_0101, // exp mrt0 v1, v1, v1, v1 done vm
            0xBF81_0000, // s_endpgm
        ]
    };
    let ps = |code: &[u32]| lower_ps(code, false, vec![], vec![], 0, 0, vec![], false);
    ps(&program(0x8A00_7E00)).unwrap_or_else(|e| panic!("s_andn2: {e}"));
    let err = ps(&program(0x8700_7E00)).err().expect("s_and keeps the data").to_string();
    assert!(err.contains("data, on the host's lanes"), "{err}");
}

/// A WaveActiveProduct butterfly, followed by an optional step before restoring EXEC.
fn whole_wave_product(step: &[u32]) -> Result<super::HostShader, ShaderError> {
    use crate::native::tests::{fspx, fspx_resource};
    let mut code = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBE82_3DC1, // s_or_saveexec_b32 s2, -1
        0xD501_0001,
        0x000A_00F2, // v_cndmask_b32_e64 v1, 1.0, v0, s2
        0x1002_02FA,
        0xFF00_B101, // v_mul_f32_dpp v1, v1, v1 quad_perm:[1,0,3,2]
        0x1002_02FA,
        0xFF00_4E01, // v_mul_f32_dpp v1, v1, v1 quad_perm:[2,3,0,1]
        0x1002_02FA,
        0xFF01_4101, // v_mul_f32_dpp v1, v1, v1 row_half_mirror
        0x1002_02FA,
        0xFF01_4001, // v_mul_f32_dpp v1, v1, v1 row_mirror
        0xD778_0002,
        0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x1002_0501, // v_mul_f32_e32 v1, v1, v2
    ];
    code.extend_from_slice(step);
    code.extend([
        0xBEFE_0302, // s_mov_b32 exec_lo, s2
        0xE070_0000,
        0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
        0xBF81_0000, // s_endpgm
    ]);
    let resources = [fspx_resource(3, 3, 0, 0, 4, 0)];
    let part = fspx(&bytes_of(&code), 5 << 1, [32, 1, 1], true, &resources);
    let container = crate::dxbc::build(&[(FourCC::ACCC, &part)]);
    translate_with(&container, Options::default())
}

#[test]
fn whole_wave_float_products_lower() {
    let product = whole_wave_product(&[]).expect("pure wave product");
    let dead_move = whole_wave_product(&[0x7E04_0280]).expect("v_mov_b32 v2, 0");
    assert_eq!(product.dxil, dead_move.dxil);
}

#[test]
fn whole_wave_sections_refuse_unmodeled_effects_at_the_instruction() {
    for (name, step) in [
        ("buffer_store_dword", &[0xE070_0004, 0x8002_0100][..]),
        ("buffer_atomic_add", &[0xE0C8_4004, 0x8002_0000]), // glc, unused result in v0
        ("global_store_dword", &[0xDC70_8000, 0x007D_0100]),
        ("image_store", &[0xF020_9F08, 0x0002_0100]),
        ("exp", &[0xF800_1890, 0x0000_0000]), // exp null, done vm
        ("ds_write_b32", &[0xD834_0000, 0x0000_0001]),
        ("s_barrier", &[0xBF8A_0000]),
        ("s_setreg_imm32_b32", &[0xBA80_0881, 3]), // change FP rounding mode
    ] {
        let err = whole_wave_product(step).err();
        assert!(
            matches!(err, Some(ShaderError::Unsupported { offset: 68, what }) if what == name),
            "{name}: {err:?}"
        );
    }
}

/// A wave64 compute program of `threads` lanes with five root-constant SGPRs (`s[2:3]` an
/// address), through `emit_with` as `translate_with` runs it on a 32-lane host.
fn emit_wave64_cs(words: &[u32], threads: u32, lds: bool) -> Result<super::HostShader, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![root_const(0, 5)],
        used: vec![],
        io: StageIo::Cs(ComputeIo { rsrc2: u32::from(lds) << 15 | 5 << 1, threads: [threads, 1, 1] }),
        code: bytes_of(words),
    };
    let p = ir::build(&native.code, true, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    let mut empty = b"DXBC".to_vec();
    empty.extend([0; 16]);
    empty.extend([1, 0, 0, 0]);
    empty.extend(32u32.to_le_bytes());
    empty.extend(0u32.to_le_bytes());
    let c = Container::parse(&empty)?;
    super::emit_with(&c, &native, &p, &st, &io, &res, Options { wave32_host: true, ..Options::default() })
}

/// Lanes below 40 read lane 35 and `exec_hi` as data behind an EXEC branch (`lockstep.rs`): the
/// halves no longer meet at the exchange on the fast path, so the lockstep one translates it.
pub(crate) const LOCKSTEP_READLANE: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x4A06_00FF,
    0x0000_03E8, // v_add_nc_u32 v3, 1000, v0
    0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
    0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
    0xBF88_0008, // s_cbranch_execz 8
    0xD760_0006,
    0x0001_4703, // v_readlane_b32 s6, v3, 35
    0x7E04_0206, // v_mov_b32 v2, s6
    0xDC70_8000,
    0x0002_0201, // global_store_dword v1, v2, s[2:3]
    0x7E04_027F, // v_mov_b32 v2, exec_hi
    0xDC70_8100,
    0x0002_0201, // global_store_dword v1, v2, s[2:3] offset:256
    0x88FE_047E, // s_or_b64 exec, exec, s[4:5]
    0xBF81_0000, // s_endpgm
];

#[test]
fn a_lane_read_behind_an_exec_branch_translates_in_lockstep() {
    if let Err(e) = emit_wave64_cs(LOCKSTEP_READLANE, 64, false) {
        panic!("{e}");
    }
}

#[test]
fn divergent_readlane_in_a_two_wave_group_is_refused() {
    // Only local IDs 0..40 enter the branch. They should all read 1035 from lane 35;
    // the second console wave skips the branch and cannot join a group-wide barrier.
    let err = emit_wave64_cs(LOCKSTEP_READLANE, 128, false).err();
    assert!(
        matches!(err, Some(ShaderError::Unsupported { what: super::lower::DIVERGENT_EXCHANGE, .. })),
        "expected a divergent exchange refusal, got {err:?}",
    );
}

/// A guessed wave64 pixel shader whose waterfall loop reads an unshifted descriptor index with
/// `v_readfirstlane_b32`, shifts it in 64-bit SALU (`s_lshl_b32` + `s_lshr_b32` + `s_addc_u32`)
/// onto an undeclared SM 6.6 `ResourceHeap` SGPR, and loads from it translates on `wave32_host`
/// without refusing on `"a descriptor outside the heaps"`, `"an address into a descriptor heap
/// with an untracked high half"`, or `"a cross-lane operation outside compute"`.
#[test]
fn guessed_wave64_pixel_salu_shifted_heap_waterfall_translates() {
    let code = [
        0xBEFC_0380u32, // s_mov_b32 m0, 0
        0xC80A_0002,    // v_interp_mov_f32 v2, p0, attr0.x
        0xBE96_047E,    // s_mov_b64 s[22:23], exec
        0xBE98_047E,    // s_mov_b64 s[24:25], exec
        0x7E28_0502,    // loop: v_readfirstlane_b32 s20, v2
        0x7DA4_0414,    // v_cmpx_eq_u32 exec, s20, v2
        0x8F0E_8514,    // s_lshl_b32 s14, s20, 5
        0x900F_9B14,    // s_lshr_b32 s15, s20, 27
        0x800C_0E0A,    // s_add_u32 s12, s10, s14
        0x820D_0F80,    // s_addc_u32 s13, 0, s15
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_0000,
        0x8004_0300, // buffer_load_dword v3, off, s[16:19], 0
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xBE98_3A18, // s_andn2_wrexec_b64 s[24:25], s[24:25]
        0xBF89_FFF2, // s_cbranch_execnz -> loop
        0xBEFE_0416, // s_mov_b64 exec, s[22:23]
        0xF800_180F,
        0x0303_0303, // exp mrt0 v3, v3, v3, v3 done vm
        0xBF81_0000, // s_endpgm
    ];
    let c = graphics::container(HwStage::Ps, &bytes_of(&code), &[0u8; 0x10]).unwrap();
    let options = Options {
        wave32_host: true,
        zero_copy_memory: true,
        ..Options::default()
    };
    let shader = translate_with(&c, options).unwrap();
    assert!(shader.heap);

    // Control-flow joined heap address (`s4 = s10 + 0x20, s5 = 0` on one arm and
    // `s4 = s10 + s9, s5 = s_addc_u32(0, 0)` on the other arm) also translates cleanly.
    let mut meta = [0u8; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // Constants in s9 (user_data 9, slot 0)
    meta[0x25] = 9;
    let joined_code = [
        0xBE8C_030Au32, // s_mov_b32 s12, s10
        0xBE8D_0380,    // s_mov_b32 s13, 0
        0xF408_0606,
        0xFA00_0000, // s_load_dwordx4 s[24:27], s[12:13], 0x0 (establishes ResourceHeap s10)
        0xBF06_8009, // s_cmp_eq_u32 s9, 0
        0xBF84_0003, // s_cbranch_scc0 -> else (+3 dwords)
        0x8004_A00A, // s_add_u32 s4, s10, 32
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBF82_0002, // s_branch -> join (+2 dwords)
        0x8004_090A, // else: s_add_u32 s4, s10, s9
        0x8205_8080, // s_addc_u32 s5, 0, 0
        0xF408_0402,
        0xFA00_0000, // join: s_load_dwordx4 s[16:19], s[4:5], 0x0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_0000,
        0x8004_0200, // buffer_load_dword v2, off, s[16:19], 0
        0xE030_0000,
        0x8006_0300, // buffer_load_dword v3, off, s[24:27], 0
        0x0604_0702, // v_add_f32 v2, v2, v3
        0xF800_180F,
        0x0202_0202, // exp mrt0 v2, v2, v2, v2 done vm
        0xBF81_0000, // s_endpgm
    ];
    let cj = graphics::container(HwStage::Ps, &bytes_of(&joined_code), &meta).unwrap();
    let sj = translate_with(&cj, options).unwrap();
    assert!(sj.heap);
}
