//! Wave64 programs on a 32-lane host (`Options::wave32_host`,
//! `lower/halves.rs`): synthetic GFX10 programs, written for these tests
//! (assembled with LLVM's `llvm-mc -mcpu=gfx1030 -mattr=+wavefrontsize64`),
//! lowered with and without the option. What each lowering must contain is
//! checked in the DXIL it gives: the wave size, the group barriers and the
//! group-shared exchange array; `a_wave64_program_matches_a_reference_on_metal`
//! in `victory-metal` runs the output.

use super::lower::{self, Lowered, Parts};
use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `s_cbranch_execz` and `s_cbranch_scc1` over the next `words` words.
const fn skip_execz(words: u32) -> u32 {
    0xBF88_0000 | words
}
const fn skip_scc1(words: u32) -> u32 {
    0xBF85_0000 | words
}

/// `v_cmp_lt_u32 vcc, 5, v0; s_bcnt1_i32_b64 s2, vcc`: the population count
/// of a ballot.
const BALLOT: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE82_106A, // s_bcnt1_i32_b64 s2, vcc
    S_ENDPGM,
];

/// `v_readlane_b32 s4, v1, 40`: lane 40, a lane of the upper half.
const READLANE: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
    0xD760_0004,
    0x0001_5101, // v_readlane_b32 s4, v1, 40
    S_ENDPGM,
];

/// Adding 32 to the dword index wraps back to this lane in its half.
const BPERMUTE: &[u32] = &[
    0x4A02_00A0, // v_add_nc_u32_e32 v1, 32, v0
    0x3402_0282, // v_lshlrev_b32_e32 v1, 2, v1
    0xDACC_0000,
    0x0200_0001, // ds_bpermute_b32 v2, v1, v0
    S_ENDPGM,
];

/// What stays inside a host wave: a DPP row shift, lane numbers, a lane
/// write, a swizzle.
const IN_HALF: &[u32] = &[
    0x7E02_0300, // v_mov_b32_e32 v1, v0
    0x4A02_02FA,
    0xFF09_1101, // v_add_nc_u32_dpp v1, v1, v1 row_shr:1 bound_ctrl
    0xD765_0002,
    0x0001_00C1, // v_mbcnt_lo_u32_b32 v2, -1, 0
    0xD766_0002,
    0x0002_04C1, // v_mbcnt_hi_u32_b32 v2, -1, v2
    0xD761_0001,
    0x0001_5003, // v_writelane_b32 v1, s3, 40
    0xD8D4_041F,
    0x0300_0001, // ds_swizzle_b32 v3, v1 offset:0x041F (lane ^ 1)
    S_ENDPGM,
];

/// A readlane in the lanes below 10 only.
const DIVERGENT_READLANE: &[u32] = &[
    0x7D88_008A, // v_cmp_gt_u32_e32 vcc, 10, v0
    0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
    skip_execz(2),
    0xD760_0004,
    0x0001_5101, // v_readlane_b32 s4, v1, 40
    0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
    0x7E0A_0204, // v_mov_b32_e32 v5, s4 (the result is used)
    S_ENDPGM,
];

/// A readlane behind a branch on a scalar (the same in every lane).
const UNIFORM_BRANCH_READLANE: &[u32] = &[
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE82_106A, // s_bcnt1_i32_b64 s2, vcc (SCC: some lane)
    skip_scc1(3),
    0xD760_0004,
    0x0001_5101, // v_readlane_b32 s4, v1, 40
    0x7E0A_0204, // v_mov_b32_e32 v5, s4
    S_ENDPGM,
];

/// A ballot in the lanes below 10 only.
const DIVERGENT_BALLOT: &[u32] = &[
    0x7D88_008A, // v_cmp_gt_u32_e32 vcc, 10, v0
    0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
    skip_execz(2),
    0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
    0xBE84_106A, // s_bcnt1_i32_b64 s4, vcc
    0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
    0x7E0A_0204, // v_mov_b32_e32 v5, s4 (the result is used)
    S_ENDPGM,
];

/// Lowers `words` as a wave64 compute program of `lanes` threads (512 bytes
/// of LDS if `lds`).
fn lowered(
    words: &[u32],
    lanes: u32,
    lds: bool,
    wave32_host: bool,
) -> Result<Lowered, crate::ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: u32::from(lds) << 15 | 2 << 1,
            threads: [lanes, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let l = lower::lower(&Parts {
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
    })?;
    // VICTORY_DUMP_DXIL=<dir>: keep the containers, to run Apple's converter on.
    if let (Some(dir), true) = (std::env::var_os("VICTORY_DUMP_DXIL"), wave32_host) {
        let hash = words.iter().fold(0u32, |h, w| h.rotate_left(5) ^ w);
        let name = format!("{hash:08x}_{lanes}.dxil");
        std::fs::write(std::path::Path::new(&dir).join(name), &l.dxil).unwrap();
    }
    Ok(l)
}

fn host32(words: &[u32], lanes: u32) -> Lowered {
    lowered(words, lanes, false, true).expect("lowers")
}

#[test]
fn without_the_option_wave64_programs_still_ask_for_64_lanes() {
    for words in [BALLOT, READLANE, BPERMUTE] {
        let l = lowered(words, 64, false, false).unwrap();
        assert_eq!(l.wave_size, Some(64));
        assert_eq!(l.barriers, 0);
        assert!(!l.exchanges);
    }
}

#[test]
fn a_ballot_of_64_lanes_is_two_host_waves_joined() {
    let l = host32(BALLOT, 64);
    assert_eq!(l.wave_size, Some(32));
    // Write, barrier, read, barrier.
    assert_eq!(l.barriers, 2);
    assert!(l.exchanges);
}

#[test]
fn a_lane_read_across_the_halves_goes_through_group_shared_memory() {
    let l = host32(READLANE, 64);
    assert_eq!(l.wave_size, Some(32));
    assert!(l.exchanges);
    assert_eq!(l.barriers, 2);
}

#[test]
fn bpermute_reads_within_each_host_wave() {
    // Each host wave must read its own lane 0 at byte address 124 + 4.
    // Store the result so the lane read is live in the emitted DXIL.
    let words = [
        0x7E02_02FF, 124, // v_mov_b32 v1, 124
        0xDACC_0004, 0x0200_0001, // ds_bpermute_b32 v2, v1, v0 offset:4
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xD834_0000, 0x0000_0203, // ds_write_b32 v3, v2
        S_ENDPGM,
    ];
    let l = lowered(&words, 64, true, true).unwrap();
    assert_eq!(l.wave_size, Some(32));
    assert_eq!(l.barriers, 0);
    assert!(!l.exchanges);
    assert_eq!(
        l.wave_read_lane_calls,
        [("dx.op.waveReadLaneAt.i32".into(), vec![Some(117), None, Some(0)])]
    );
}

#[test]
fn operations_inside_32_lanes_stay_native() {
    let l = host32(IN_HALF, 64);
    assert_eq!(l.wave_size, Some(32));
    assert_eq!(l.barriers, 0);
    assert!(!l.exchanges);
}

#[test]
fn a_group_of_32_lanes_or_fewer_needs_no_exchange() {
    for words in [BALLOT, READLANE, BPERMUTE, IN_HALF] {
        let l = host32(words, 32);
        assert_eq!(l.wave_size, Some(32));
        assert_eq!(l.barriers, 0);
        assert!(!l.exchanges);
    }
}

#[test]
fn exchanges_wait_for_every_thread_to_arrive() {
    // In the lanes below 10: the second host wave never gets there. The
    // value is left unknown (the program is refused where something uses
    // it), and no barrier is emitted.
    for words in [DIVERGENT_READLANE, DIVERGENT_BALLOT] {
        let l = host32(words, 64);
        assert_eq!((l.barriers, l.exchanges), (0, false), "{words:x?}");
    }
    // Behind a scalar branch every lane takes the same way, both halves.
    assert_eq!(host32(UNIFORM_BRANCH_READLANE, 64).barriers, 4);
    // Not in a group of several waves: the scalar may differ between them.
    assert_eq!(host32(UNIFORM_BRANCH_READLANE, 128).barriers, 2);
}

#[test]
fn lds_a_wave_shares_unsynced_in_divergent_code_is_refused() {
    // Write and read both under the branch: no barrier can go there.
    const WRITE_READ_DIVERGENT: &[u32] = &[
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0xBE80_047E, // s_mov_b64 s[0:1], exec
        0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
        skip_execz(4),
        0xD834_0000,
        0x0000_0001, // ds_write_b32 v1, v0
        0xD8D8_0004,
        0x0200_0001, // ds_read_b32 v2, v1 offset:4
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
        S_ENDPGM,
    ];
    let r = lowered(WRITE_READ_DIVERGENT, 64, true, true);
    assert!(matches!(r, Err(crate::ShaderError::Unsupported { .. })));
    // Where a barrier fits it is added, and no 64-lane wave is asked for.
    const WRITE_THEN_READ: &[u32] = &[
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0xD834_0000,
        0x0000_0001, // ds_write_b32 v1, v0
        0xD8D8_0004,
        0x0200_0001, // ds_read_b32 v2, v1 offset:4
        S_ENDPGM,
    ];
    let l = lowered(WRITE_THEN_READ, 64, true, true).unwrap();
    assert_eq!(l.wave_size, None);
    assert_eq!(l.barriers, 1);
}

#[test]
fn a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves() {
    // `v_readfirstlane` reads each host wave's own first lane (0 and 32 here), so a branch on
    // what it gives can split the console wave: the halves would not meet at a barrier.
    const FIRST: [u32; 3] = [
        0x7E04_0500, // v_readfirstlane_b32 s2, v0
        0xBF06_8002, // s_cmp_eq_u32 s2, 0
        0xBF84_0003, // s_cbranch_scc0 3
    ];
    let mut readlane = FIRST.to_vec();
    readlane.extend([
        0xD760_0004,
        0x0001_5101, // v_readlane_b32 s4, v1, 40
        0x7E0A_0204, // v_mov_b32_e32 v5, s4
        S_ENDPGM,
    ]);
    let l = host32(&readlane, 64);
    assert_eq!((l.barriers, l.exchanges), (0, false));
    // LDS written and read behind such a branch (over 6 words): no barrier fits.
    let mut lds = FIRST.to_vec();
    lds[2] = 0xBF84_0006; // s_cbranch_scc0 6
    lds.extend([
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0xD834_0000,
        0x0000_0001, // ds_write_b32 v1, v0
        0xD8D8_0004,
        0x0200_0001, // ds_read_b32 v2, v1 offset:4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        S_ENDPGM,
    ]);
    let r = lowered(&lds, 64, true, true);
    assert!(matches!(r, Err(crate::ShaderError::Unsupported { .. })));
}

#[test]
fn a_barrier_block_behind_a_first_lane_branch_is_no_exchange_site() {
    // Two console waves (no lockstep): a branch on SCC skips a block holding the title's
    // `s_barrier` and a lane read across the halves (an exchange).
    let program = |scc: u32| {
        vec![
            0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
            0x7E10_0280, // v_mov_b32_e32 v8, 0
            0x7E12_0280, // v_mov_b32_e32 v9, 0
            scc,         // s2: a first lane read, or a constant
            0xBF06_8002, // s_cmp_eq_u32 s2, 0
            0xBF84_0006, // s_cbranch_scc0 6
            0xBF8A_0000, // s_barrier
            0xD760_0004,
            0x0001_5101, // v_readlane_b32 s4, v1, 40
            0x7E0A_0204, // v_mov_b32_e32 v5, s4
            0xDC70_8000,
            0x007D_0508, // global_store_dword v[8:9], v5, off (a dead read is not made)
            S_ENDPGM,
        ]
    };
    // A constant SCC: every half takes the branch alike, the block is an exchange site.
    let l = host32(&program(0xBE82_0380), 128); // s_mov_b32 s2, 0
    assert!(l.exchanges);
    // `v_readfirstlane_b32 s2, v0`: 0 in the lower half, 32 in the upper, which would skip the
    // block while the lower half waits at the exchange's barrier.
    let r = lowered(&program(0x7E04_0500), 128, false, true);
    assert!(
        matches!(r, Err(crate::ShaderError::Unsupported { what, .. }) if what == lower::DIVERGENT_EXCHANGE)
    );
}

/// A sum over the 64 lanes the way a compiler expands `WaveActiveSum`: all
/// lanes on, butterflies inside each row of 16, then the rows' sums by lane.
const REDUCE: &[u32] = &[
    0xBE82_25C1, // s_or_saveexec_b64 s[2:3], -1
    0xD501_0001,
    0x000A_0080, // v_cndmask_b32_e64 v1, 0, v0, s[2:3]
    0x4A02_02FA,
    0xFF00_B101, // v_add_nc_u32_dpp v1, v1, v1 quad_perm:[1,0,3,2]
    0x4A02_02FA,
    0xFF00_4E01, // v_add_nc_u32_dpp v1, v1, v1 quad_perm:[2,3,0,1]
    0x4A02_02FA,
    0xFF01_4101, // v_add_nc_u32_dpp v1, v1, v1 row_half_mirror
    0x4A02_02FA,
    0xFF01_4001, // v_add_nc_u32_dpp v1, v1, v1 row_mirror
    0xD778_0002,
    0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
    0x4A02_0501, // v_add_nc_u32_e32 v1, v1, v2
    0xD760_0004,
    0x0001_0101, // v_readlane_b32 s4, v1, 0
    0xD760_0005,
    0x0001_4101, // v_readlane_b32 s5, v1, 32
    0xD525_0003,
    0x0000_0A04, // v_add_nc_u32_e64 v3, s4, s5
    0xBEFE_0402, // s_mov_b64 exec, s[2:3]
    0x7E0A_0303, // v_mov_b32_e32 v5, v3
    S_ENDPGM,
];

#[test]
fn a_whole_wave_sum_combines_the_two_halves() {
    let l = host32(REDUCE, 64);
    assert_eq!(l.wave_size, Some(32));
    // Whatever the section leaves is one reduction: write, barrier, read, barrier.
    assert_eq!(l.barriers, 2);
}

/// [`REDUCE`] with `v_mul_f32` and 1.0 in the lanes off: a product, which the
/// two host waves of a console wave exchange as they do a sum.
#[test]
fn a_whole_wave_product_combines_the_two_halves() {
    let mut words = REDUCE.to_vec();
    words[2] = 0x000A_00F2; // v_cndmask_b32_e64 v1, 1.0, v0, s[2:3]
    for w in &mut words {
        match *w {
            0x4A02_02FA => *w = 0x1002_02FA, // v_mul_f32_dpp v1, v1, v1
            0x4A02_0501 => *w = 0x1002_0501, // v_mul_f32_e32 v1, v1, v2
            0xD525_0003 => *w = 0xD508_0003, // v_mul_f32_e64 v3, s4, s5
            _ => {}
        }
    }
    let l = host32(&words, 64);
    assert_eq!(l.wave_size, Some(32));
    assert_eq!(l.barriers, 2);
}

#[test]
fn whole_wave_float_prefix_products() {
    for inclusive in [false, true] {
        // Form an inclusive scan within each 16-lane row in v1.
        let mut words = vec![
            0xBE82_25C1, // s_or_saveexec_b64 s[2:3], -1
            0xD501_0001,
            0x000A_00F2, // v_cndmask_b32 v1, 1.0, v0, s[2:3]
        ];
        for shift in [1, 2, 4, 8] {
            words.extend([0x1002_02FA, 0xFF01_1001 | (shift << 8)]); // v_mul_f32 row_shr
        }
        if inclusive {
            words.push(0x7E04_0301); // v_mov_b32 v2, v1
        } else {
            // Shift the row prefix one lane, leaving 1.0 in its first lane.
            words.extend([0x7E04_02F2, 0x7E04_02FA, 0xFF01_1101]);
        }
        // Multiply each completed row's product into all subsequent rows.
        for lane in [15, 31, 47] {
            let mask = u64::MAX << (lane + 1);
            words.extend([
                0xD760_0004,
                ((128 + lane) << 9) | 257, // v_readlane_b32 s4, v1, lane
                0xBEFE_03FF,
                mask as u32, // s_mov_b32 exec_lo, mask.lo
                0xBEFF_03FF,
                (mask >> 32) as u32, // s_mov_b32 exec_hi, mask.hi
                0x1004_0404,         // v_mul_f32 v2, s4, v2
                0xBEFE_04C1,         // s_mov_b64 exec, -1
            ]);
        }
        words.extend([
            0xBEFE_0402, // s_mov_b64 exec, s[2:3]
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
            0xD834_0000,
            0x0000_0203, // ds_write_b32 v3, v2
            S_ENDPGM,
        ]);

        for host32 in [false, true] {
            let l = lowered(&words, 64, true, host32).unwrap();
            assert_eq!(
                l.wave_prefix_calls,
                [(
                    "dx.op.wavePrefixOp.f32".into(),
                    vec![Some(121), None, Some(1), Some(0)]
                )]
            );
            if host32 {
                assert_eq!(
                    l.wave_active_calls,
                    [(
                        "dx.op.waveActiveOp.f32".into(),
                        vec![Some(119), None, Some(1), Some(0)]
                    )]
                );
            } else {
                assert!(l.wave_active_calls.is_empty());
            }
            // Inclusive folds in this lane; the split wave folds in the lower half.
            assert_eq!(
                l.float_multiplies,
                usize::from(inclusive) + usize::from(host32)
            );
            assert_eq!(l.barriers, if host32 { 2 } else { 0 });
        }
    }
}

#[test]
fn permlane16_and_permlanex16_lower_within_host32_half() {
    // v_permlane16_b32 v2, v0, 0, 0 fi:1 (op 0x377, opsel=1 -> 0xD777_0802)
    // v_permlanex16_b32 v3, v2, -1, -1 fi:1 (op 0x378, opsel=1 -> 0xD778_0803)
    const PROG: &[u32] = &[
        0xD777_0802,
        0x0201_0100, // v_permlane16_b32 v2, v0, 0, 0 fi:1
        0xD778_0803,
        0x0305_8302, // v_permlanex16_b32 v3, v2, -1, -1 fi:1
        0x7E0A_0303, // v_mov_b32_e32 v5, v3
        S_ENDPGM,
    ];
    let l = host32(PROG, 64);
    assert_eq!(l.wave_size, Some(32));
    // Both stay within the 32-lane half, so no cross-half LDS exchange barriers are needed.
    assert_eq!(l.barriers, 0);
}
