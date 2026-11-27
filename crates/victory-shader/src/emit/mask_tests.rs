//! Lane masks read as numbers in compute programs: synthetic wave64 GFX10
//! sequences (assembled with LLVM's `llvm-mc -mcpu=gfx1030
//! -mattr=+wavefrontsize64`), each followed by stores of the VGPRs it
//! writes so that a value the translation cannot make is refused, lowered
//! for the console's wave size and for the 32-lane host
//! (`Options::wave32_host`). A mask is a ballot (or, held where its lanes
//! are not those that run, the scalar operation's own result on the
//! console's lanes); SCC of an operation on masks read as data is
//! "some lane in the result".

use super::lower::{self, Lowered, Parts};
use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};
use crate::{io, ir, resource, structure};

/// `v_mov_b32 v1, 0; v_mov_b32 v2, 0; v_mov_b32 v5, 0; v_mov_b32 v8, 0;
/// v_mov_b32 v9, 0`: the registers a case writes and stores start defined.
const PRELUDE: &[u32] = &[
    0x7E02_0280,
    0x7E04_0280,
    0x7E0A_0280,
    0x7E10_0280,
    0x7E12_0280,
];

/// `global_store_dword v[8:9], v1, off` (and v2, v5), `s_endpgm`.
const EPILOGUE: &[u32] = &[
    0xDC70_8000,
    0x007D_0108,
    0xDC70_8000,
    0x007D_0208,
    0xDC70_8000,
    0x007D_0508,
    0xBF81_0000,
];

/// Lowers PRELUDE, `body` and EPILOGUE as a wave64 compute program of
/// `lanes` threads.
fn lowered(body: &[u32], lanes: u32, wave32_host: bool) -> Result<Lowered, crate::ShaderError> {
    let words: Vec<u32> = [PRELUDE, body, EPILOGUE].concat();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [lanes, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
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
        wave32_host,
        zero_copy_memory: false,
        rect_list: false,
        copy_params: [0; 32],
        view_marks: false,
    })
}

const BCNT: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE82_106A, // s_bcnt1_i32_b64 s2, vcc
    0x7E0A_0202, // v_mov_b32_e32 v5, s2
];
const SCC_AFTER_AND: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8782_7E6A, // s_and_b64 s[2:3], vcc, exec
    0x8505_8987, // s_cselect_b32 s5, 7, 9
    0x7E02_0205, // v_mov_b32_e32 v1, s5
];
const MASK_TO_VGPR: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8782_7E6A, // s_and_b64 s[2:3], vcc, exec
    0x7E02_0202, // v_mov_b32_e32 v1, s2
    0x7E04_0203, // v_mov_b32_e32 v2, s3
];
const CMP_MASK_CSELECT: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBF12_806A, // s_cmp_eq_u64 vcc, 0
    0x8505_8987, // s_cselect_b32 s5, 7, 9
    0x7E02_0205, // v_mov_b32_e32 v1, s5
];
const LSHL_EXEC: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE82_047E, // s_mov_b64 s[2:3], exec
    0x8FFE_816A, // s_lshl_b64 exec, vcc, 1
    0x7E02_0283, // v_mov_b32_e32 v1, 3
    0xBEFE_0402, // s_mov_b64 exec, s[2:3]
];
const FF1_READLANE: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE84_146A, // s_ff1_i32_b64 s4, vcc
    0xD760_0005, // v_readlane_b32 s5, v1, s4
    0x0000_0901,
    0x7E04_0205, // v_mov_b32_e32 v2, s5
];
const AND_ARITH: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8704_FF6A, // s_and_b32 s4, vcc_lo, 0xff
    0x0000_00FF,
    0x8005_8104, // s_add_u32 s5, s4, 1
    0x7E02_0205, // v_mov_b32_e32 v1, s5
];
const ARITH_THEN_SELECT: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE82_046A, // s_mov_b64 s[2:3], vcc
    0x8002_8102, // s_add_u32 s2, s2, 1
    0xD501_0001, // v_cndmask_b32_e64 v1, 0, 1, s[2:3]
    0x0009_0280,
];
const LANE_SELECT_MASK: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE84_146A, // s_ff1_i32_b64 s4, vcc
    0x8F82_0481, // s_lshl_b64 s[2:3], 1, s4
    0xD501_0001, // v_cndmask_b32_e64 v1, 0, 1, s[2:3]
    0x0009_0280,
];
const WRITELANE_MASK: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8782_7E6A, // s_and_b64 s[2:3], vcc, exec
    0xD761_0001, // v_writelane_b32 v1, s2, 3
    0x0001_0602,
];
const BITSET: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE82_046A, // s_mov_b64 s[2:3], vcc
    0xBE82_1D85, // s_bitset1_b32 s2, 5
    0xD501_0001, // v_cndmask_b32_e64 v1, 0, 1, s[2:3]
    0x0009_0280,
];
const CMP_LO_BRANCH: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBF07_806A, // s_cmp_lg_u32 vcc_lo, 0
    0xBF85_0001, // s_cbranch_scc1 1
    0x7E02_0283, // v_mov_b32_e32 v1, 3
];
const LSHR_DATA: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x9084_886A, // s_lshr_b64 s[4:5], vcc, 8
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const SAVEEXEC_SCC_DATA: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
    0x8505_8987, // s_cselect_b32 s5, 7, 9
    0xBEFE_0402, // s_mov_b64 exec, s[2:3]
    0x7E02_0205, // v_mov_b32_e32 v1, s5
];
const ADDC_SCC: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8782_7E6A, // s_and_b64 s[2:3], vcc, exec
    0x8205_8005, // s_addc_u32 s5, s5, 0
    0x7E02_0205, // v_mov_b32_e32 v1, s5
];
const NOT_MASK_DATA: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE82_086A, // s_not_b64 s[2:3], vcc
    0x7E02_0202, // v_mov_b32_e32 v1, s2
];
const BFE_MASK: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x9384_FF6A, // s_bfe_u32 s4, vcc_lo, 0x80008
    0x0008_0008,
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const S_MUL_EXEC: &[u32] = &[
    0xBE82_047E, // s_mov_b64 s[2:3], exec
    0x9304_8302, // s_mul_i32 s4, s2, 3
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const POPCOUNT_EXEC_BRANCH: &[u32] = &[
    0xBE84_107E, // s_bcnt1_i32_b64 s4, exec
    0xBF06_C004, // s_cmp_eq_u32 s4, 64
    0x8505_8081, // s_cselect_b32 s5, 1, 0
    0x7E02_0205, // v_mov_b32_e32 v1, s5
];
const MASK_XOR_MASK_DATA: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xD4C4_0006, // v_cmp_gt_u32_e64 s[6:7], 9, v0
    0x0002_0089,
    0x8982_066A, // s_xor_b64 s[2:3], vcc, s[6:7]
    0xBF13_8002, // s_cmp_lg_u64 s[2:3], 0
    0x8505_8987, // s_cselect_b32 s5, 7, 9
    0x7E02_0205, // v_mov_b32_e32 v1, s5
];
const MASK_MIN_MAX: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8782_7E6A, // s_and_b64 s[2:3], vcc, exec
    0x8384_8502, // s_min_u32 s4, s2, 5
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const MASK_TO_VGPR_READFIRST: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xD501_0001, // v_cndmask_b32_e64 v1, 0, -1, vcc
    0x01A9_8280,
    0x7E08_0501, // v_readfirstlane_b32 s4, v1
];
const BCNT0: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE84_0E6A, // s_bcnt0_i32_b64 s4, vcc
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const FLBIT: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE84_166A, // s_flbit_i32_b64 s4, vcc
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const FF0: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE84_126A, // s_ff0_i32_b64 s4, vcc
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const BITCMP: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBF0D_876A, // s_bitcmp1_b32 vcc_lo, 7
    0x8504_8281, // s_cselect_b32 s4, 1, 2
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const BITCMP64: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBF0F_A86A, // s_bitcmp1_b64 vcc, 40
    0x8504_8281, // s_cselect_b32 s4, 1, 2
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const ASHR: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x9184_886A, // s_ashr_i64 s[4:5], vcc, 8
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const BFE64: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x9484_FF6A, // s_bfe_u64 s[4:5], vcc, 0x80008
    0x0008_0008,
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const BREV: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE84_0C6A, // s_brev_b64 s[4:5], vcc
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const MASK_EQ_EXEC_SCC: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBF12_7E6A, // s_cmp_eq_u64 vcc, exec
    0x8504_8281, // s_cselect_b32 s4, 1, 2
    0x7E02_0204, // v_mov_b32_e32 v1, s4
];
const MASK_COPY_DATA: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8784_7E6A, // s_and_b64 s[4:5], vcc, exec
    0xBE86_0304, // s_mov_b32 s6, s4
    0xBE87_0305, // s_mov_b32 s7, s5
    0x7E02_0206, // v_mov_b32_e32 v1, s6
    0x7E04_0207, // v_mov_b32_e32 v2, s7
];
const CSELECT_MASK_DATA: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8784_7E6A, // s_and_b64 s[4:5], vcc, exec
    0x8586_8004, // s_cselect_b64 s[6:7], s[4:5], 0
    0x7E02_0206, // v_mov_b32_e32 v1, s6
];
const SUB_MASK: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8784_7E6A, // s_and_b64 s[4:5], vcc, exec
    0x8086_8104, // s_sub_u32 s6, s4, 1
    0x8287_8005, // s_subb_u32 s7, s5, 0
    0x8786_0406, // s_and_b64 s[6:7], s[6:7], s[4:5]
    0x7E02_0206, // v_mov_b32_e32 v1, s6
    0x7E04_0207, // v_mov_b32_e32 v2, s7
];
const DIV_BCNT: &[u32] = &[
    0x7D88_008A, // v_cmp_gt_u32_e32 vcc, 10, v0
    0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
    0xBF88_0004, // s_cbranch_execz 4
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE84_106A, // s_bcnt1_i32_b64 s4, vcc
    0x7E02_0204, // v_mov_b32_e32 v1, s4
    0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
];
const DIV_SCC: &[u32] = &[
    0x7D88_008A, // v_cmp_gt_u32_e32 vcc, 10, v0
    0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
    0xBF88_0003, // s_cbranch_execz 3
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8786_7E6A, // s_and_b64 s[6:7], vcc, exec
    0x8504_8281, // s_cselect_b32 s4, 1, 2
    0x7E02_0204, // v_mov_b32_e32 v1, s4
    0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
];
const VCC_TO_VGPR: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x7E02_026A, // v_mov_b32_e32 v1, vcc_lo
    0x7E04_026B, // v_mov_b32_e32 v2, vcc_hi
];
const ANDN2_EXEC_SCC: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8A84_6A7E, // s_andn2_b64 s[4:5], exec, vcc
    0x8506_8281, // s_cselect_b32 s6, 1, 2
    0x7E02_0206, // v_mov_b32_e32 v1, s6
];
const XOR_CNDMASK_SCALAR: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0x8984_7E6A, // s_xor_b64 s[4:5], vcc, exec
    0xBE86_1404, // s_ff1_i32_b64 s6, s[4:5]
    0x8F88_0681, // s_lshl_b64 s[8:9], 1, s6
    0x8A88_086A, // s_andn2_b64 s[8:9], vcc, s[8:9]
    0xD501_0001, // v_cndmask_b32_e64 v1, 0, 1, s[8:9]
    0x0021_0280,
];

/// Every consumer of a mask as a number lowers: on the console's wave, and
/// on the 32-lane host with one or two host waves for the console's wave.
#[test]
fn masks_read_as_data_lower() {
    for (name, body) in [
        ("bcnt", BCNT),
        ("scc_after_and", SCC_AFTER_AND),
        ("mask_to_vgpr", MASK_TO_VGPR),
        ("cmp_mask_cselect", CMP_MASK_CSELECT),
        ("lshl_exec", LSHL_EXEC),
        ("ff1_readlane", FF1_READLANE),
        ("and_arith", AND_ARITH),
        ("arith_then_select", ARITH_THEN_SELECT),
        ("lane_select_mask", LANE_SELECT_MASK),
        ("writelane_mask", WRITELANE_MASK),
        ("bitset", BITSET),
        ("cmp_lo_branch", CMP_LO_BRANCH),
        ("lshr_data", LSHR_DATA),
        ("saveexec_scc_data", SAVEEXEC_SCC_DATA),
        ("addc_scc", ADDC_SCC),
        ("not_mask_data", NOT_MASK_DATA),
        ("bfe_mask", BFE_MASK),
        ("s_mul_exec", S_MUL_EXEC),
        ("popcount_exec_branch", POPCOUNT_EXEC_BRANCH),
        ("mask_xor_mask_data", MASK_XOR_MASK_DATA),
        ("mask_min_max", MASK_MIN_MAX),
        ("mask_to_vgpr_readfirst", MASK_TO_VGPR_READFIRST),
        ("bcnt0", BCNT0),
        ("flbit", FLBIT),
        ("ff0", FF0),
        ("bitcmp", BITCMP),
        ("bitcmp64", BITCMP64),
        ("ashr", ASHR),
        ("brev", BREV),
        ("mask_eq_exec_scc", MASK_EQ_EXEC_SCC),
        ("mask_copy_data", MASK_COPY_DATA),
        ("cselect_mask_data", CSELECT_MASK_DATA),
        ("sub_mask", SUB_MASK),
        ("div_scc", DIV_SCC),
        ("vcc_to_vgpr", VCC_TO_VGPR),
        ("andn2_exec_scc", ANDN2_EXEC_SCC),
        ("xor_cndmask_scalar", XOR_CNDMASK_SCALAR),
    ] {
        for (lanes, host32) in [(64, false), (64, true), (32, true)] {
            if let Err(e) = lowered(body, lanes, host32) {
                panic!("{name} ({lanes} lanes, wave32 host {host32}): {e:?}");
            }
        }
    }
}

/// SCC of `s_and_b64 s[2:3], vcc, exec` read as data (`s_cselect`,
/// `s_addc`, after `s_and_saveexec`) is a ballot of the result's lanes.
#[test]
fn scc_of_a_mask_operation_is_a_ballot() {
    for body in [SCC_AFTER_AND, ADDC_SCC, SAVEEXEC_SCC_DATA, DIV_SCC] {
        for host32 in [false, true] {
            lowered(body, 64, host32).unwrap_or_else(|e| panic!("{body:x?}: {e:?}"));
        }
    }
}

/// A mask shifted into EXEC or built from scalars holds lanes the
/// structurisation cannot see into: the console's lanes are asked for.
#[test]
fn a_shifted_or_built_mask_needs_the_consoles_lanes() {
    for body in [LSHL_EXEC, BITSET, LANE_SELECT_MASK] {
        let l = lowered(body, 64, true).unwrap();
        assert_eq!(l.wave_size, Some(32));
        let l = lowered(body, 64, false).unwrap();
        assert_eq!(l.wave_size, Some(64));
    }
}

/// A mask holding lanes elsewhere read as data is refused saying where it
/// comes from: here a saveexec of the other width (32 bits in a wave64
/// program).
#[test]
fn masks_holding_lanes_elsewhere_say_where_they_come_from() {
    const SAVEEXEC32_DATA: &[u32] = &[
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0xBE82_3C6A, // s_and_saveexec_b32 s2, vcc_lo
        0x7E02_0202, // v_mov_b32_e32 v1, s2
    ];
    let r = lowered(SAVEEXEC32_DATA, 64, false).err();
    let why = "a lane mask holding lanes elsewhere read as data: written by a scalar \
               instruction of the other width";
    assert!(
        matches!(r, Some(crate::ShaderError::Unsupported { what, .. }) if what == why),
        "{r:?}"
    );
}

/// What is not translated stays refused, saying what.
#[test]
fn what_has_no_form_is_refused_clearly() {
    // A 64-bit bit field of a mask.
    let r = lowered(BFE64, 64, false).err();
    assert!(
        matches!(
            r,
            Some(crate::ShaderError::Unsupported {
                what: "s_bfe_u64",
                ..
            })
        ),
        "{r:?}"
    );
    // A ballot in lanes below 10: the second host wave never gets there.
    let r = lowered(DIV_BCNT, 64, true).err();
    assert!(
        matches!(r, Some(crate::ShaderError::Unsupported { .. })),
        "{r:?}"
    );
}

/// A wave64 lane mask joined with data at a uniform loop's header (the
/// wave takes one edge): each lane takes its bit of the pair's number, the
/// low dword for lanes below 32, the high one above, as wave32 does.
#[test]
fn a_wave64_mask_joined_with_data_at_a_uniform_loop_is_its_number() {
    const LOOP: &[u32] = &[
        0x7D88_0098, // v_cmp_gt_u32_e32 vcc, 24, v0
        0x8786_7E6A, // s_and_b64 s[6:7], vcc, exec
        0xBE88_0380, // s_mov_b32 s8, 0
        0xD501_0001,
        0x0019_0280, // header: v_cndmask_b32_e64 v1, 0, 1, s[6:7]
        0x9086_8406, // s_lshr_b64 s[6:7], s[6:7], 4
        0x7E0A_0206, // v_mov_b32_e32 v5, s6
        0x7E0C_0207, // v_mov_b32_e32 v6, s7
        0x7E0C_0505, // v_readfirstlane_b32 s6, v5
        0x7E0E_0506, // v_readfirstlane_b32 s7, v6
        0x8008_8108, // s_add_u32 s8, s8, 1
        0xBF0A_8208, // s_cmp_lt_u32 s8, 2
        0xBF85_FFF6, // s_cbranch_scc1 header
    ];
    for wave32_host in [false, true] {
        if let Err(e) = lowered(LOOP, 64, wave32_host) {
            panic!("wave32_host {wave32_host}: {e}");
        }
    }
}

#[test]
fn wqm_bitset_bfm_brev_quadmask_and_saveexec_masks_lower_as_data() {
    const WQM_MASK_DATA: &[u32] = &[
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0xBE82_0A6A, // s_wqm_b64 s[2:3], vcc
        0x8504_8987, // s_cselect_b32 s4, 7, 9
        0x7E02_0202, // v_mov_b32_e32 v1, s2
        0x7E04_0204, // v_mov_b32_e32 v2, s4
    ];
    const WQM_PLAIN_DATA: &[u32] = &[
        0xBE82_03FF,
        0x0100_0004, // s_mov_b32 s2, 0x01000004
        0xBE83_0380, // s_mov_b32 s3, 0
        0xBE84_0A02, // s_wqm_b64 s[4:5], s[2:3]
        0x7E02_0204, // v_mov_b32_e32 v1, s4
    ];
    const BITSET_EXEC: &[u32] = &[
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0xBE82_047E, // s_mov_b64 s[2:3], exec
        0xBEFE_046A, // s_mov_b64 exec, vcc
        0xBEFE_1EA8, // s_bitset1_b64 exec, 40
        0xBEFE_1C81, // s_bitset0_b64 exec, 1
        0x7E02_0283, // v_mov_b32_e32 v1, 3
        0xBEFE_0402, // s_mov_b64 exec, s[2:3]
    ];
    const BFM_EXEC: &[u32] = &[
        0xBE82_047E, // s_mov_b64 s[2:3], exec
        0x92FE_8488, // s_bfm_b64 exec, 8, 4
        0x7E02_0283, // v_mov_b32_e32 v1, 3
        0xBEFE_0402, // s_mov_b64 exec, s[2:3]
    ];
    const BREV_EXEC: &[u32] = &[
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0xBE82_047E, // s_mov_b64 s[2:3], exec
        0xBEFE_0C6A, // s_brev_b64 exec, vcc
        0x7E02_0283, // v_mov_b32_e32 v1, 3
        0xBEFE_0402, // s_mov_b64 exec, s[2:3]
    ];
    const QUADMASK_EXEC: &[u32] = &[
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0xBE82_047E, // s_mov_b64 s[2:3], exec
        0xBEFE_2D6A, // s_quadmask_b64 exec, vcc
        0x7E02_0283, // v_mov_b32_e32 v1, 3
        0xBEFE_0402, // s_mov_b64 exec, s[2:3]
    ];
    const SAVE_AND_WREXEC_DATA: &[u32] = &[
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0xBE84_386A, // s_orn1_saveexec_b64 s[4:5], vcc
        0xBE86_396A, // s_andn1_wrexec_b64 s[6:7], vcc
        0xBE88_3A6A, // s_andn2_wrexec_b64 s[8:9], vcc
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0x7E02_0204, // v_mov_b32_e32 v1, s4
        0x7E04_0206, // v_mov_b32_e32 v2, s6
        0x7E0A_0208, // v_mov_b32_e32 v5, s8
    ];

    for (name, body) in [
        ("wqm_mask_data", WQM_MASK_DATA),
        ("wqm_plain_data", WQM_PLAIN_DATA),
        ("bitset_exec", BITSET_EXEC),
        ("bfm_exec", BFM_EXEC),
        ("brev_exec", BREV_EXEC),
        ("quadmask_exec", QUADMASK_EXEC),
        ("save_and_wrexec_data", SAVE_AND_WREXEC_DATA),
    ] {
        for (lanes, host32) in [(64, false), (64, true), (32, true)] {
            if let Err(e) = lowered(body, lanes, host32) {
                panic!("{name} ({lanes} lanes, wave32 host {host32}): {e:?}");
            }
        }
    }
}

#[test]
fn pixel_and_other_width_mask_number_negatives_stay_refused() {
    let pixel = |body: &[u32]| -> Result<Lowered, crate::ShaderError> {
        let mut words = vec![
            0x7E02_0280, // v_mov_b32_e32 v1, 0
            0xD765_0002,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v2, -1, 0
        ];
        words.extend_from_slice(body);
        words.extend([
            0xF800_180F,
            0x0101_0101, // exp mrt0 v1, v1, v1, v1 done vm
            0xBF81_0000, // s_endpgm
        ]);
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: true,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, true, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower::lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![super::container::SigElement {
                name: "SV_Target".into(),
                index: 0,
                kind: 16,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 0xF,
                half: false,
            }],
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
    };

    for (label, body, expected) in [
        (
            "pixel s_bfm_b64 in mask_number",
            &[
                0x7D88_0485, // v_cmp_gt_u32_e32 vcc, 5, v2
                0xBE88_106A, // s_bcnt1_i32_b64 s8, vcc
                0xBE86_0380, // s_mov_b32 s6, 0
                0xBE8A_047E, // s_mov_b64 s[10:11], exec
                0x7DA8_0485, // v_cmpx_gt_u32_e32 5, v2
                0xBF88_0002, // s_cbranch_execz +2 (narrows block.reach)
                0x92FE_8408, // s_bfm_b64 exec, s8, 4 (non-active EXEC mask)
                0xBE86_107E, // s_bcnt1_i32_b64 s6, exec (reads via mask_number)
                0xBEFE_040A, // s_mov_b64 exec, s[10:11]
                0x7E02_0206, // v_mov_b32_e32 v1, s6
            ][..],
            "shifted on the host's lanes",
        ),
        (
            "pixel s_bfm_b64 written to EXEC",
            &[
                0x7D88_0485, // v_cmp_gt_u32_e32 vcc, 5, v2
                0xBE88_106A, // s_bcnt1_i32_b64 s8, vcc
                0xBE82_047E, // s_mov_b64 s[2:3], exec
                0x92FE_8408, // s_bfm_b64 exec, s8, 4
                0x7E02_02F2, // v_mov_b32_e32 v1, 1.0
                0xBEFE_0402, // s_mov_b64 exec, s[2:3]
            ][..],
            "a lane mask read as data",
        ),
        (
            "pixel s_brev_b64 in mask_number",
            &[
                0x7D88_0485, // v_cmp_gt_u32_e32 vcc, 5, v2
                0xBE86_0380, // s_mov_b32 s6, 0
                0xBE8A_047E, // s_mov_b64 s[10:11], exec
                0x7DA8_0482, // v_cmpx_gt_u32_e32 2, v2
                0xBF88_0002, // s_cbranch_execz +2
                0xBEFE_0C6A, // s_brev_b64 exec, vcc
                0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
                0xBEFE_040A, // s_mov_b64 exec, s[10:11]
                0x7E02_0206, // v_mov_b32_e32 v1, s6
            ][..],
            "shifted on the host's lanes",
        ),
        (
            "pixel s_quadmask_b64 in mask_number",
            &[
                0x7D88_0485, // v_cmp_gt_u32_e32 vcc, 5, v2
                0xBE86_0380, // s_mov_b32 s6, 0
                0xBE8A_047E, // s_mov_b64 s[10:11], exec
                0x7DA8_0482, // v_cmpx_gt_u32_e32 2, v2
                0xBF88_0002, // s_cbranch_execz +2
                0xBEFE_2D6A, // s_quadmask_b64 exec, vcc
                0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
                0xBEFE_040A, // s_mov_b64 exec, s[10:11]
                0x7E02_0206, // v_mov_b32_e32 v1, s6
            ][..],
            "shifted on the host's lanes",
        ),
        (
            "pixel s_wqm_b64 read as data in mask_dword",
            &[
                0x7D88_0485, // v_cmp_gt_u32_e32 vcc, 5, v2
                0xBEFE_0A6A, // s_wqm_b64 exec, vcc (Atom::Quad, in st.active)
                0xBE86_107E, // s_bcnt1_i32_b64 s6, exec (routed to mask_number, not all-1s)
                0x7E02_0206, // v_mov_b32_e32 v1, s6
            ][..],
            "shifted on the host's lanes",
        ),
        (
            "pixel s_bitset0_b64 by non-host-lane constant",
            &[
                0x7D88_0485, // v_cmp_gt_u32_e32 vcc, 5, v2
                0xBE86_0380, // s_mov_b32 s6, 0
                0xBE8A_047E, // s_mov_b64 s[10:11], exec
                0x7DA8_0482, // v_cmpx_gt_u32_e32 2, v2
                0xBF88_0002, // s_cbranch_execz +2
                0xBEFE_1C83, // s_bitset0_b64 exec, 3
                0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
                0xBEFE_040A, // s_mov_b64 exec, s[10:11]
                0x7E02_0206, // v_mov_b32_e32 v1, s6
            ][..],
            "shifted on the host's lanes",
        ),
        (
            "pixel s_andn2_wrexec_b64 in mask_number",
            &[
                0x7D88_0485, // v_cmp_gt_u32_e32 vcc, 5, v2 (lanes 0..4)
                0xBE8A_047E, // s_mov_b64 s[10:11], exec
                0x7DA8_0482, // v_cmpx_gt_u32_e32 2, v2 (old_exec = lanes 0..1)
                0xBE84_3A6A, // s_andn2_wrexec_b64 s[4:5], vcc (s[4:5] = lanes 2..4, outside old_exec)
                0xBE86_1004, // s_bcnt1_i32_b64 s6, s[4:5]
                0xBEFE_040A, // s_mov_b64 exec, s[10:11]
                0x7E02_0206, // v_mov_b32_e32 v1, s6
            ][..],
            "saveexec",
        ),
        (
            "pixel Atom::Opaque outer s_and_b64 wrapping inner s_brev_b64",
            &[
                0xBE82_047E, // s_mov_b64 s[2:3], exec
                0x7D88_0485, // v_cmp_gt_u32_e32 vcc, 5, v2
                0xBE84_0C6A, // s_brev_b64 s[4:5], vcc
                0x87FE_047E, // s_and_b64 exec, exec, s[4:5]
                0x7E02_02F2, // v_mov_b32_e32 v1, 1.0
                0xBEFE_0402, // s_mov_b64 exec, s[2:3]
            ][..],
            "lane mask",
        ),
        (
            "pixel Atom::Opaque outer s_and_b64 wrapping inner s_lshr_b64",
            &[
                0xBE82_047E, // s_mov_b64 s[2:3], exec
                0x7D88_0485, // v_cmp_gt_u32_e32 vcc, 5, v2
                0x9084_816A, // s_lshr_b64 s[4:5], vcc, 1
                0x87FE_047E, // s_and_b64 exec, exec, s[4:5]
                0x7E02_02F2, // v_mov_b32_e32 v1, 1.0
                0xBEFE_0402, // s_mov_b64 exec, s[2:3]
            ][..],
            "lane mask",
        ),
    ] {
        let err = pixel(body).err();
        assert!(
            matches!(&err, Some(crate::ShaderError::Unsupported { what, .. }) if what.contains(expected)),
            "{label}: expected `{expected}`, got {err:?}"
        );
    }

    // `saveexec` of the other width (`s_orn1_saveexec_b32` in wave64, and `s_orn1_saveexec_b64`
    // in wave32) stays refused in both `MaskData::Host` and `MaskData::Console`.
    let err_pixel_b32_in_w64 = pixel(&[
        0x7D88_0485, // v_cmp_gt_u32_e32 vcc, 5, v2
        0xBE84_446A, // s_orn1_saveexec_b32 s4, vcc_lo
        0xBE86_0F04, // s_bcnt1_i32_b32 s6, s4
        0x7E02_0206, // v_mov_b32_e32 v1, s6
    ])
    .err();
    assert!(
        matches!(&err_pixel_b32_in_w64, Some(crate::ShaderError::Unsupported { what, .. }) if what.contains("lane mask")),
        "{err_pixel_b32_in_w64:?}"
    );
    let err_cs_b32_in_w64 = lowered(
        &[
            0x7D88_0085, // v_cmp_gt_u32_e32 vcc, 5, v0
            0xBE84_446A, // s_orn1_saveexec_b32 s4, vcc_lo
            0x7E02_0204, // v_mov_b32_e32 v1, s4
        ],
        64,
        true,
    )
    .err();
    assert!(
        matches!(&err_cs_b32_in_w64, Some(crate::ShaderError::Unsupported { what, .. }) if what.contains("other width")),
        "{err_cs_b32_in_w64:?}"
    );
    let w32_words: Vec<u32> = [
        PRELUDE,
        &[
            0xBEEB_0380, // s_mov_b32 vcc_hi, 0
            0xBEFF_0380, // s_mov_b32 exec_hi, 0
            0x7D82_0085, // v_cmp_lt_u32_e32 vcc_lo, 5, v0
            0xBE84_386A, // s_orn1_saveexec_b64 s[4:5], vcc
            0x7E02_0204, // v_mov_b32_e32 v1, s4
        ],
        EPILOGUE,
    ]
    .concat();
    let native_w32 = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: w32_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p_w32 = ir::build(&native_w32.code, false, 256).unwrap();
    let st_w32 = structure::structure(&p_w32).unwrap();
    let res_w32 = resource::track(&p_w32, &native_w32.layout, HwStage::Cs, Some(&st_w32)).unwrap();
    let io_w32 = io::raise(&p_w32, &native_w32).unwrap();
    let err_b64_in_w32 = lower::lower(&Parts {
        native: &native_w32,
        p: &p_w32,
        st: &st_w32,
        io: &io_w32,
        res: &res_w32,
        inputs: vec![],
        outputs: vec![],
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
    .err();
    assert!(
        matches!(&err_b64_in_w32, Some(crate::ShaderError::Unsupported { what, .. }) if what.contains("other width")),
        "{err_b64_in_w32:?}"
    );
}

/// Two nested loops of that shape, each with an SGPR for the lanes still looping and one for
/// the lanes that broke out.
const NESTED: &[u32] = &[
    0x7DA8_00A8, // v_cmpx_gt_u32 40, v0 (lanes 40 and up leave first)
    0xBE8A_047E, // s_mov_b64 s[10:11], exec (the outer loop's lanes still looping)
    0xBE8E_0480, // s_mov_b64 s[14:15], 0 (the outer loop's lanes that broke out)
    0x7E08_0300, // v_mov_b32 v4, v0
    0x7DA4_0880, // O: v_cmpx_eq_u32 0, v4
    0x8A8A_7E0A, // s_andn2_b64 s[10:11], s[10:11], exec
    0xBF84_001E, // s_cbranch_scc0 OEND
    0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    0xBE84_047E, // s_mov_b64 s[4:5], exec (the inner loop's lanes still looping)
    0xBE86_047E, // s_mov_b64 s[6:7], exec (EXEC to restore)
    0xBE88_0480, // s_mov_b64 s[8:9], 0 (the inner loop's lanes that broke out)
    0x7E06_0304, // v_mov_b32 v3, v4
    0x7DA4_0680, // L: v_cmpx_eq_u32 0, v3 (the lanes whose count ran out leave)
    0x8A84_7E04, // s_andn2_b64 s[4:5], s[4:5], exec (SCC: does any lane loop on?)
    0xBF84_000A, // s_cbranch_scc0 END
    0xBEFE_0404, // s_mov_b64 exec, s[4:5]
    0x4A02_0281, // v_add_nc_u32 v1, 1, v1
    0x4A06_06C1, // v_add_nc_u32 v3, -1, v3
    0xD4C2_000C, // v_cmp_eq_u32 s[12:13], 5, v3
    0x0202_0685,
    0xBE8C_080C, // s_not_b64 s[12:13], s[12:13] (the lanes not at 5 break out)
    0x87FE_7E0C, // s_and_b64 exec, s[12:13], exec
    0x8888_7E08, // s_or_b64 s[8:9], s[8:9], exec
    0xBE84_3A04, // s_andn2_wrexec_b64 s[4:5], s[4:5]
    0xBF85_FFF3, // s_cbranch_scc1 L
    0xBEFE_0406, // END: s_mov_b64 exec, s[6:7]
    0x87FE_7E08, // s_and_b64 exec, s[8:9], exec
    0x4A04_0487, // v_add_nc_u32 v2, 7, v2 (in the lanes that broke out)
    0xBEFE_0406, // s_mov_b64 exec, s[6:7]
    0x4A08_08C1, // v_add_nc_u32 v4, -1, v4
    0xD4C2_0010, // v_cmp_eq_u32 s[16:17], 9, v4
    0x0202_0889,
    0xBE90_0810, // s_not_b64 s[16:17], s[16:17]
    0x87FE_7E10, // s_and_b64 exec, s[16:17], exec
    0x888E_7E0E, // s_or_b64 s[14:15], s[14:15], exec
    0xBE8A_3A0A, // s_andn2_wrexec_b64 s[10:11], s[10:11]
    0xBF85_FFDF, // s_cbranch_scc1 O
    0xBEFE_04C1, // OEND: s_mov_b64 exec, -1
];

/// A structurized loop keeps the lanes still looping in an SGPR beside EXEC: a copy of EXEC
/// before the loop, `s_andn2_b64 s, s, exec` for the exit test at the head (its SCC: does any
/// lane loop on?), and `s_andn2_wrexec_b64 s, s` at the back edge, which writes the same result
/// to both. At the head that SGPR is EXEC itself, so a ballot there gives it. Where the arrivals
/// at the head were unprovable (an outer loop with lanes that broke out), it was refused as "a
/// lane mask holding lanes elsewhere read as data: a program input": a world lighting compute
/// program of #258.
#[test]
fn a_loop_register_that_is_exec_on_every_edge_is_exec_at_the_head() {
    // The inner loop's register is EXEC where its head joins it, a mask a ballot there gives.
    let words: Vec<u32> = [PRELUDE, NESTED, EPILOGUE].concat();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let joins: Vec<_> = p.blocks.iter().flat_map(|b| &b.phis).filter(|ph| ph.reg == ir::Reg::S(4)).collect();
    assert!(!joins.is_empty());
    assert!(joins.iter().all(|ph| st.active.contains(&ph.value)), "s4 at its loop's head is not EXEC");
    for host32 in [false, true] {
        let r = lowered(NESTED, 64, host32);
        assert!(r.is_ok(), "{host32}: {:?}", r.err());
    }
}

/// The same loops with the inner back edge's `s_andn2_wrexec_b64 s[4:5], s[4:5]` made
/// `s_andn2_b64 s[4:5], s[4:5], exec`: there `s4` is no longer the EXEC that edge brings, so its
/// join at the head is not EXEC's mask.
#[test]
fn a_loop_register_that_is_not_exec_on_one_edge_is_not_exec_at_the_head() {
    let body: Vec<u32> = NESTED.iter().map(|&w| if w == 0xBE84_3A04 { 0x8A84_7E04 } else { w }).collect();
    assert_ne!(body, NESTED);
    let words: Vec<u32> = [PRELUDE, &body, EPILOGUE].concat();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let joins: Vec<_> = p.blocks.iter().flat_map(|b| &b.phis).filter(|ph| ph.reg == ir::Reg::S(4)).collect();
    assert!(!joins.is_empty());
    assert!(joins.iter().all(|ph| !st.active.contains(&ph.value)), "s4 taken as EXEC at its loop's head");
}
