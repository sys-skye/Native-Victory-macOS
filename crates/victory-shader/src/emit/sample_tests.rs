//! Samples and gathers with texel offsets (`_o`), comparisons (`_c`) and
//! levels, hand-assembled with `llvm-mc -mcpu=gfx1030`: which DXIL operation
//! each becomes and which address VGPR feeds each argument. The address
//! order is RDNA2 ISA 8.2.5's: offset, bias, z-compare, derivatives, then
//! the coordinates and the level or clamp.

use super::container::SigElement;
use super::lower::{self, Lowered, Parts};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// Descriptors from the tables at s4 (texture s[8:15], raw UAV s[20:23])
/// and s6 (sampler s[16:19]); then v1 the offset word (x -3, y 5), v2 2.0,
/// v3 0.25, v4 0.75, v5 0.5, v6 1/16, v7 1/32, v8 1/64.
const PREFIX: [u32; 23] = [
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0202,
    0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
    0xF408_0502,
    0xFA00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_02FF,
    0x0000_053D, // v_mov_b32 v1, 0x53d
    0x7E04_02F4, // v_mov_b32 v2, 2.0
    0x7E06_02FF,
    0x3E80_0000, // v_mov_b32 v3, 0.25
    0x7E08_02FF,
    0x3F40_0000, // v_mov_b32 v4, 0.75
    0x7E0A_02F0, // v_mov_b32 v5, 0.5
    0x7E0C_02FF,
    0x3D80_0000, // v_mov_b32 v6, 1/16
    0x7E0E_02FF,
    0x3D00_0000, // v_mov_b32 v7, 1/32
    0x7E10_02FF,
    0x3C80_0000, // v_mov_b32 v8, 1/64
];

const S_ENDPGM: u32 = 0xBF81_0000;

/// Constant arguments of a call: the float's bits, or an integer.
const U: Option<i64> = None;
const fn k(x: u32) -> Option<i64> {
    Some(x as i64)
}
const fn i(x: i32) -> Option<i64> {
    Some(x as i64)
}
const TWO: Option<i64> = k(0x4000_0000);
const QUARTER: Option<i64> = k(0x3E80_0000);
const THREE_QUARTERS: Option<i64> = k(0x3F40_0000);
const HALF: Option<i64> = k(0x3F00_0000);

/// `image` (with `before`, instructions after the prefix) in a wave32
/// compute shader that stores v10, or a pixel shader exporting v[10:13].
fn lower(stage: HwStage, before: &[u32], image: [u32; 2]) -> Result<Lowered, ShaderError> {
    let mut words = PREFIX.to_vec();
    words.extend(before);
    words.extend(image);
    words.extend(if stage == HwStage::Ps {
        [0xF800_180F, 0x0D0C_0B0A] // exp mrt0 v10, v11, v12, v13 done vm
    } else {
        [0xE070_0000, 0x8005_0A00] // buffer_store_dword v10, off, s[20:23], 0
    });
    words.push(S_ENDPGM);
    let table = |user_data, slot| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    };
    let used = |user_data, kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data / 2 - 1,
    };
    let native = NativeProgram {
        stage,
        wave64: false,
        vgprs: 32,
        layout: vec![table(4, 1), table(6, 2)],
        used: vec![
            used(4, ResourceType::Srv, 0, 8),
            used(4, ResourceType::Uav, 8, 4),
            used(6, ResourceType::Sampler, 0, 4),
        ],
        io: if stage == HwStage::Ps {
            StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            }
        } else {
            StageIo::Cs(ComputeIo {
                rsrc2: 7 << 1,
                threads: [32, 1, 1],
            })
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, false, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    let target = SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: if stage == HwStage::Ps {
            vec![target]
        } else {
            vec![]
        },
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: true,
        zero_copy_memory: false,
        rect_list: false,
        copy_params: [0; 32],
        view_marks: false,
    })
}

/// The one texture call `image` lowers to: its name, and its arguments
/// after the opcode, handle and sampler.
fn call(stage: HwStage, image: [u32; 2]) -> (String, Vec<Option<i64>>) {
    let lowered = lower(stage, &[], image).unwrap_or_else(|e| panic!("{image:08x?}: {e}"));
    let [(name, args)] = &lowered.texture_calls[..] else {
        panic!("{image:08x?}: {:?}", lowered.texture_calls);
    };
    (name.clone(), args[3..].to_vec())
}

fn refusal(stage: HwStage, before: &[u32], image: [u32; 2]) -> String {
    match lower(stage, before, image) {
        Ok(_) => panic!("{image:08x?} lowered"),
        Err(e) => e.to_string(),
    }
}

const CS: HwStage = HwStage::Cs;

#[test]
fn offset_samples_at_a_level_read_the_offset_first() {
    // image_sample_lz_o v[10:13], v[1:3], s[8:15], s[16:19] dmask:0xf dim:2D
    let (name, args) = call(CS, [0xF0DC_0F08, 0x0082_0A01]);
    assert_eq!(name, "dx.op.sampleLevel.f32");
    // Coordinates (v2, v3), offsets (-3, 5), level 0.
    assert_eq!(args, [TWO, QUARTER, U, U, i(-3), i(5), U, k(0)]);

    // image_sample_l_o v[10:13], v[1:4]: the level (v4) after the coordinates.
    let (name, args) = call(CS, [0xF0D0_0F08, 0x0082_0A01]);
    assert_eq!(name, "dx.op.sampleLevel.f32");
    assert_eq!(args, [TWO, QUARTER, U, U, i(-3), i(5), U, THREE_QUARTERS]);
}

#[test]
fn an_offset_sample_with_bias_reads_the_bias_after_the_offset() {
    // image_sample_b_o v[10:13], v[1:4], s[8:15], s[16:19] dmask:0xf dim:2D
    let (name, args) = call(CS, [0xF0D4_0F08, 0x0082_0A01]);
    assert_eq!(name, "dx.op.sampleBias.f32");
    // Coordinates (v3, v4), offsets, bias v2, no clamp.
    assert_eq!(
        args,
        [QUARTER, THREE_QUARTERS, U, U, i(-3), i(5), U, TWO, U]
    );
}

#[test]
fn offset_comparisons_read_the_reference_after_the_offset() {
    // image_sample_c_lz_o v10, v[1:4], s[8:15], s[16:19] dmask:0x1 dim:2D
    let (name, args) = call(CS, [0xF0FC_0108, 0x0082_0A01]);
    assert_eq!(name, "dx.op.sampleCmpLevelZero.f32");
    assert_eq!(args, [QUARTER, THREE_QUARTERS, U, U, i(-3), i(5), U, TWO]);

    // image_sample_c_o, the same addresses: implicit derivatives, so pixel
    // shaders only.
    let sample_c_o = [0xF0E0_0108, 0x0082_0A01];
    let (name, args) = call(HwStage::Ps, sample_c_o);
    assert_eq!(name, "dx.op.sampleCmp.f32");
    assert_eq!(
        args,
        [QUARTER, THREE_QUARTERS, U, U, i(-3), i(5), U, TWO, U]
    );
    assert!(refusal(CS, &[], sample_c_o).contains("image_sample_c_o"));

    // image_sample_c_lz v10, v[2:4] without an offset: unchanged.
    let (name, args) = call(CS, [0xF0BC_0108, 0x0082_0A02]);
    assert_eq!(name, "dx.op.sampleCmpLevelZero.f32");
    assert_eq!(args, [QUARTER, THREE_QUARTERS, U, U, k(0), k(0), U, TWO]);
}

#[test]
fn offset_samples_with_gradients_read_them_before_the_coordinates() {
    // image_sample_d_o v[10:13], v[1:8], s[8:15], s[16:19] dmask:0xf dim:2D:
    // offset, dx/dh v2, dy/dh v3, dx/dv v4, dy/dv v5, coordinates v6, v7.
    let (name, args) = call(CS, [0xF0C8_0F08, 0x0082_0A01]);
    assert_eq!(name, "dx.op.sampleGrad.f32");
    #[rustfmt::skip]
    let want = [
        k(0x3D80_0000), k(0x3D00_0000), U, U,
        i(-3), i(5), U,
        TWO, QUARTER, U,
        THREE_QUARTERS, HALF, U,
        U,
    ];
    assert_eq!(args, want);
}

#[test]
fn gradients_stay_32_bit_with_16_bit_coordinates() {
    // image_sample_d v[10:13], v[2:6], s[8:15], s[16:19] dmask:0xf dim:2D a16: a16 packs the
    // coordinates (v6), not the gradients: dx/dh v2, dy/dh v3, dx/dv v4, dy/dv v5, as LLVM sizes
    // the address (#320). They used to be read as f16 halves of v2 and v3.
    let (name, args) = call(CS, [0xF088_0F08, 0x4082_0A02]);
    assert_eq!(name, "dx.op.sampleGrad.f32");
    assert_eq!(args[7..13], [TWO, QUARTER, U, THREE_QUARTERS, HALF, U]);

    // image_sample_d v[10:13], v[2:4], ... dim:1D a16: dx/dh v2, dx/dv v3, then the coordinate
    // in v4's low half (was refused as "16-bit gradients of a 1D or 3D image").
    let (name, args) = call(CS, [0xF088_0F00, 0x4082_0A02]);
    assert_eq!(name, "dx.op.sampleGrad.f32");
    assert_eq!(args[7..13], [TWO, U, U, QUARTER, U, U]);

    // Arrays: `dim` 4 (1D array) is `kind` 6 with two coordinates and `dim` 5 (2D array) `kind`
    // 7 with three; the slice is no spatial coordinate, so there are 2 and 4 gradients, as
    // `image_addr_dwords` sizes them (one and two a direction): v[2:4] and v[2:7] with a16.
    let (name, args) = call(CS, [0xF088_0F20, 0x4082_0A02]);
    assert_eq!(name, "dx.op.sampleGrad.f32");
    assert_eq!(args[7..13], [TWO, U, U, QUARTER, U, U]);
    let (name, args) = call(CS, [0xF088_0F28, 0x4082_0A02]);
    assert_eq!(name, "dx.op.sampleGrad.f32");
    assert_eq!(args[7..13], [TWO, QUARTER, U, THREE_QUARTERS, HALF, U]);
}

#[test]
fn comparison_gathers_at_level_0_compare_red() {
    // image_gather4_c_lz v[10:13], v[2:4], s[8:15], s[16:19] dmask:0x1 dim:2D
    let (name, args) = call(CS, [0xF13C_0108, 0x0082_0A02]);
    assert_eq!(name, "dx.op.textureGatherCmp.f32");
    // Coordinates (v3, v4), offsets 0, red, reference v2.
    assert_eq!(args, [QUARTER, THREE_QUARTERS, U, U, k(0), k(0), k(0), TWO]);

    // image_gather4_c_lz_o v[10:13], v[1:4]: the offsets are extracted from
    // v1 at run time (gathers take computed ones).
    let (name, args) = call(CS, [0xF17C_0108, 0x0082_0A01]);
    assert_eq!(name, "dx.op.textureGatherCmp.f32");
    assert_eq!(args[..4], [QUARTER, THREE_QUARTERS, U, U]);
    assert_eq!(args[6..], [k(0), TWO]);
}

/// The culling shaders' sampler (black border, levels 0-12, linear filters) as an immediate in
/// s[16:19], over the one the prefix loads: a gather at a level takes only an immediate one.
const IMMEDIATE_SAMPLER: [u32; 6] = [
    0xB010_01B6, // s_movk_i32 s16, 0x1b6
    0xBE91_03FF,
    0x00C0_0000, // s_mov_b32 s17, 0xc00000
    0xBE92_03FF,
    0x1850_0000, // s_mov_b32 s18, 0x18500000
    0xBE93_0380, // s_mov_b32 s19, 0
];

#[test]
fn gathers_at_a_level_whole_in_the_lanes_that_gather_translate() {
    // image_gather4_l v[10:13], v[3:5], s[8:15], s[16:19] dmask:0x1 dim:2D: the level is v5,
    // 0.5 from the prefix, made 2.0 by a write only lanes below 16 run, and only they gather
    // (the G-buffer culling shaders' `v_cmpx` before their level, #182).
    let gather = [0xF110_0108, 0x0082_0A03];
    let mut before = IMMEDIATE_SAMPLER.to_vec();
    before.extend([
        0x7DA8_0090, // v_cmpx_gt_u32 exec_lo, 16, v0
        0x7E0A_02F4, // v_mov_b32 v5, 2.0
    ]);
    if let Err(e) = lower(CS, &before, gather) {
        panic!("a level whole in every lane that gathers was refused: {e}");
    }
    // Every lane gathers: those from 16 up at 0.5, so which level is not known.
    before.push(0xBEFE_03C1); // s_mov_b32 exec_lo, -1
    let refused = refusal(CS, &before, gather);
    assert!(refused.contains("a gather at a level that may not be whole"), "{refused}");
}

#[test]
fn what_dxil_cannot_express_stays_refused() {
    // A comparison gather of green: DXIL's compares red.
    let green = refusal(CS, &[], [0xF13C_0208, 0x0082_0A02]);
    assert!(
        green.contains("comparison gather of another channel"),
        "{green}"
    );
    // A cube sample with offsets.
    let cube = refusal(CS, &[], [0xF0DC_0F18, 0x0082_0A01]);
    assert!(cube.contains("cube sample with texel offsets"), "{cube}");
    // An offset past DXIL's -8..7: v_mov_b32 v1, 10.
    let far = refusal(CS, &[0x7E02_028A], [0xF0DC_0F08, 0x0082_0A01]);
    assert!(far.contains("sample offset past -8..7"), "{far}");
    // A comparison at an explicit level (DXIL 1.7's SampleCmpLevel) and a
    // comparison gather at the computed level (DXIL gathers from level 0).
    let at_level = refusal(CS, &[], [0xF0B0_0108, 0x0082_0A01]);
    assert!(at_level.contains("image_sample_c_l"), "{at_level}");
    let gather_c = refusal(CS, &[], [0xF120_0108, 0x0082_0A02]);
    assert!(gather_c.contains("image_gather4_c"), "{gather_c}");
}
