//! A console wave of 64 lanes split into two 32-lane host waves (`lower/halves.rs` in
//! `victory_shader`), run on the GPU: synthetic GFX10 programs of our own (assembled with
//! `llvm-mc -arch=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize64`) whose lanes exchange values across
//! the halves after part of the wave has left. `s[2:3]` is `dst`, `s4` a flag. Skipped without the
//! Metal Shader Converter.

use super::addr_tests::{
    CANARY, Fixture, VA_A, dispatch, halves, title, translate_io, translate_wave, try_translate_wave,
};
use crate::native::list::NativeList;
use crate::native::shaders;
use crate::native::title::Pipe;

const S_ENDPGM: u32 = 0xBF81_0000;
/// `v_cmp_gt_u32 vcc, 40, v0; s_and_saveexec_b64 s[4:5], vcc`: lanes 40 and up leave.
const LANES_BELOW_40: [u32; 2] = [0x7D88_00A8, 0xBE84_246A];
const fn skip_execz(words: u32) -> u32 {
    0xBF88_0000 | words
}
const V_LSHLREV_V1_2_V0: u32 = 0x3402_0082;
const GLOBAL_STORE_V2_AT_V1: [u32; 2] = [0xDC70_8000, 0x0002_0201];
const DS_WRITE_V0_AT_V1: [u32; 2] = [0xD834_0000, 0x0000_0001];
const DS_READ_V2_AT_V3: [u32; 2] = [0xD8D8_0000, 0x0200_0003];
const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F;

/// Pool offset of `dst`, at `VA_A + 0x4_0000`.
const DST: usize = 0x6_0000;

/// What the fast path refuses `words` for (five user SGPRs, five wave64 waves a group, more
/// than lockstep takes).
fn refusal(words: &[u32]) -> Option<String> {
    try_translate_wave(words, 5, 0, 320, true, shaders::options()).err().map(|e| e.to_string())
}

/// [`refusal`] of a program of 32-lane waves, one a group.
fn refusal32(words: &[u32]) -> Option<String> {
    try_translate_wave(words, 5, 0, 32, false, shaders::options()).err().map(|e| e.to_string())
}

/// [`run`] of a program of 32-lane waves, one a group.
fn run32(words: &[u32], flag: u32) -> Vec<u32> {
    let Some(mut t) = title() else { return Vec::new() };
    let f = Fixture::new(&t);
    let cs = translate_wave(words, 5, 0, 32, false, shaders::options());
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs, &[0, 0, c, d, flag], false);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    f.get(DST, 32)
}

fn run(words: &[u32], lds: u32, flag: u32) -> Vec<u32> {
    run_io(words, lds << 15, 64, flag)
}

/// [`run`] of `threads` in a group, with `rsrc2` bits of system SGPRs and LDS.
fn run_io(words: &[u32], rsrc2: u32, threads: u32, flag: u32) -> Vec<u32> {
    let Some(mut t) = title() else { return Vec::new() };
    let f = Fixture::new(&t);
    let cs = translate_io(words, 5, rsrc2, threads, shaders::options());
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs, &[0, 0, c, d, flag], false);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    f.get(DST, threads as usize)
}

#[test]
fn bpermute_reads_an_active_source_in_the_same_half() {
    // Only lanes 0..39 run. Byte address 124 + offset 4 wraps to local lane 0
    // (console lanes 0 and 32). Offset 36 picks local lane 8 (8 on, 40 off).
    for (offset, lo, hi) in [(4, 1, 33), (36, 9, 0)] {
        let mut w = vec![
            0x7E06_02FF, 124, // v_mov_b32 v3, 124
            0x4A08_0081, // v_add_nc_u32 v4, 1, v0 (distinct nonzero data)
            V_LSHLREV_V1_2_V0,
        ];
        w.extend(LANES_BELOW_40);
        w.extend([0xDACC_0000 | offset, 0x0200_0403, S_WAITCNT_LGKM]); // ds_bpermute_b32 v2, v3, v4
        w.extend(GLOBAL_STORE_V2_AT_V1);
        w.push(S_ENDPGM);
        let got = run(&w, 0, 0);
        if got.is_empty() {
            return;
        }
        assert_eq!(got[..32], [lo; 32]);
        assert_eq!(got[32..40], [hi; 8]);
        assert_eq!(got[40..], [CANARY; 24]);
    }
}

#[test]
fn a_ballot_across_the_halves_after_lanes_left_for_the_end() {
    // Lanes 0..39 stay; each stores the count of lanes above 5 among them (34: 26 in the
    // lower half, 8 in the upper). The lanes that left store nothing and count as off.
    let mut w = LANES_BELOW_40.to_vec();
    w.extend([
        skip_execz(6),
        0x7D82_0085, // v_cmp_lt_u32 vcc, 5, v0
        0xBE86_106A, // s_bcnt1_i32_b64 s6, vcc
        V_LSHLREV_V1_2_V0,
        0x7E04_0206, // v_mov_b32 v2, s6
    ]);
    w.extend(GLOBAL_STORE_V2_AT_V1);
    w.push(S_ENDPGM);
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..40], [34; 40]);
    assert_eq!(got[40..], [CANARY; 24]);
}

#[test]
fn lds_shared_across_the_halves_after_lanes_left_for_the_end() {
    // Lanes 0..39 write their index to LDS and read lane 39 - i's, with no barrier (the console
    // runs the wave in lockstep).
    let mut w = LANES_BELOW_40.to_vec();
    w.extend([skip_execz(11), V_LSHLREV_V1_2_V0]);
    w.extend(DS_WRITE_V0_AT_V1);
    w.extend([
        0x4C06_00A7, // v_sub_nc_u32 v3, 39, v0
        0x3406_0682, // v_lshlrev_b32 v3, 2, v3
        S_WAITCNT_LGKM,
    ]);
    w.extend(DS_READ_V2_AT_V3);
    w.push(S_WAITCNT_LGKM);
    w.extend(GLOBAL_STORE_V2_AT_V1);
    w.push(S_ENDPGM);
    let got = run(&w, 1, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..40], (0..40).rev().collect::<Vec<u32>>());
    assert_eq!(got[40..], [CANARY; 24]);
}

#[test]
fn lds_shared_across_the_halves_behind_a_uniform_branch() {
    // Every lane writes its index to LDS; when `s4` is not 0, each reads lane 63 - i's, with no
    // barrier. The branch is on SCC, so both host waves take it or skip it together.
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_WRITE_V0_AT_V1);
    w.extend([
        0x4C06_00BF, // v_sub_nc_u32 v3, 63, v0
        0x3406_0682, // v_lshlrev_b32 v3, 2, v3
        0xBF07_8004, // s_cmp_lg_u32 s4, 0
        0xBF84_0006, // s_cbranch_scc0 6
        S_WAITCNT_LGKM,
    ]);
    w.extend(DS_READ_V2_AT_V3);
    w.push(S_WAITCNT_LGKM);
    w.extend(GLOBAL_STORE_V2_AT_V1);
    w.push(S_ENDPGM);
    let got = run(&w, 1, 1);
    if got.is_empty() {
        return;
    }
    assert_eq!(got, (0..64).rev().collect::<Vec<u32>>());
    assert_eq!(run(&w, 1, 0), [CANARY; 64]);
}

#[test]
fn the_thread_group_info_sgpr_numbers_the_console_waves() {
    // Two console waves of 64, four host waves. `s5`, the info SGPR (bit 10), holds the wave's
    // index from bit 6 and the group's wave count; index * 64 plus the lane's count of lanes below
    // it is the thread's index, so each slot is written once.
    let w = [
        0x8706_FF05,
        0x0000_0FC0, // s_and_b32 s6, s5, 0xfc0
        0xD765_0001,
        0x0200_0CC1, // v_mbcnt_lo_u32_b32 v1, -1, s6
        0xD766_0001,
        0x0202_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x3404_0282, // v_lshlrev_b32 v2, 2, v1
        0x7E06_0205, // v_mov_b32 v3, s5
        0xDC70_8000,
        0x0002_0302, // global_store_dword v2, v3, s[2:3]
        S_ENDPGM,
    ];
    let got = run_io(&w, 1 << 10, 128, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got, (0..128).map(|i| i / 64 << 6 | 2).collect::<Vec<u32>>());
}

#[test]
fn a_wave32_programs_group_info_numbers_its_waves_from_bit_20() {
    // Two console waves of 32 (GTA VI's temporal resolve numbers its threads so): bits 20-24 of
    // the info SGPR times 32 plus the lane's count of lanes below it is the thread's index, so
    // each slot is written once. Bits 6-11 hold the index too, bits 0-5 the wave count.
    let w = [
        0x9386_FF05,
        0x0005_0014, // s_bfe_u32 s6, s5, 0x50014 (bits 20-24)
        0x8F06_8506, // s_lshl_b32 s6, s6, 5
        0xD765_0001,
        0x0200_0CC1, // v_mbcnt_lo_u32_b32 v1, -1, s6
        0x3404_0282, // v_lshlrev_b32 v2, 2, v1
        0x7E06_0205, // v_mov_b32 v3, s5
        0xDC70_8000,
        0x0002_0302, // global_store_dword v2, v3, s[2:3]
        S_ENDPGM,
    ];
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = translate_wave(&w, 5, 1 << 10, 64, false, shaders::options());
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs, &[0, 0, c, d, 0], false);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    let got = f.get(DST, 64);
    assert_eq!(got, (0..64).map(|i| i / 32 << 20 | i / 32 << 6 | 2).collect::<Vec<u32>>());
}

/// `v_ldexp_f32` (a pixel shader of GTA VI's world was refused for it): `s0` scaled by two to
/// the power `s1`, a signed integer, in range or past it.
#[test]
fn ldexp_scales_by_an_integer_power_of_two() {
    let cases: &[(f32, i32, f32)] = &[
        (1.5, 3, 12.0),
        (3.0, -2, 0.75),
        (1.0, 127, 2f32.powi(127)),
        (2f32.powi(-126), 200, 2f32.powi(74)),
        (2f32.powi(100), -200, 2f32.powi(-100)),
        (1.0, 1000, f32::INFINITY),
        (-1.0, 1000, f32::NEG_INFINITY),
        (1.0, -1000, 0.0),
    ];
    for &(a, e, want) in cases {
        let Some(mut t) = title() else { return };
        let f = Fixture::new(&t);
        let mut w = vec![
            0x7E04_0201, // v_mov_b32 v2, s1
            0xD762_0002, // v_ldexp_f32 v2, s0, v2
            0x0002_0400,
            V_LSHLREV_V1_2_V0,
        ];
        w.extend(GLOBAL_STORE_V2_AT_V1);
        w.push(S_ENDPGM);
        let cs = translate_io(&w, 5, 0, 64, shaders::options());
        let [c, d] = halves(VA_A + 0x4_0000);
        dispatch(&mut t, &cs, &[a.to_bits(), e as u32, c, d, 0], false);
        assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
        let got = f.get(DST, 1)[0];
        assert_eq!(got, want.to_bits(), "ldexp({a}, {e}) = {}, want {want}", f32::from_bits(got));
    }
}

/// `v_mov_b32 v1, -1; v_readlane_b32 s7, v1, 0`: a lane read, which fills every thread's exchange
/// slot with -1.
const FILL_XCHG: [u32; 3] = [0x7E02_02C1, 0xD760_0007, 0x0201_0101];

#[test]
fn a_group_of_96_lanes_exchanges_nothing_with_the_lanes_it_lacks() {
    // Two console waves, the second of 32 lanes: one host wave, no upper half to exchange with.
    // Each lane stores the count of lanes above 5 in its wave's ballot, plus the -1 of the fill:
    // the second wave's high dword is no lane's, 0, not the slot the fill left.
    let mut w = FILL_XCHG.to_vec();
    w.extend([
        0x7D82_0085, // v_cmp_lt_u32 vcc, 5, v0
        0xBE86_106A, // s_bcnt1_i32_b64 s6, vcc
        0x8006_0706, // s_add_u32 s6, s6, s7
        V_LSHLREV_V1_2_V0,
        0x7E04_0206, // v_mov_b32 v2, s6
    ]);
    w.extend(GLOBAL_STORE_V2_AT_V1);
    w.push(S_ENDPGM);
    let got = run_io(&w, 0, 96, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..64], [57; 64]);
    assert_eq!(got[64..], [31; 32]);

    // A whole-wave sum of the lanes' indices (the lanes the wave lacks add the 0 of the
    // v_cndmask), plus the fill's -1: the missing upper half adds nothing.
    let mut w = FILL_XCHG.to_vec();
    w.extend([
        0xBE88_25C1, // s_or_saveexec_b64 s[8:9], -1
        0xD501_0001, 0x0022_0080, // v_cndmask_b32_e64 v1, 0, v0, s[8:9]
        0x4A02_02FA, 0xFF00_B101, // v_add_nc_u32_dpp v1, v1, v1 quad_perm:[1,0,3,2]
        0x4A02_02FA, 0xFF00_4E01, // v_add_nc_u32_dpp v1, v1, v1 quad_perm:[2,3,0,1]
        0x4A02_02FA, 0xFF01_4101, // v_add_nc_u32_dpp v1, v1, v1 row_half_mirror
        0x4A02_02FA, 0xFF01_4001, // v_add_nc_u32_dpp v1, v1, v1 row_mirror
        0xD778_0002, 0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x4A02_0501, // v_add_nc_u32_e32 v1, v1, v2
        0xD760_0004, 0x0201_0101, // v_readlane_b32 s4, v1, 0
        0xD760_0005, 0x0201_4101, // v_readlane_b32 s5, v1, 32
        0xD525_0003, 0x0200_0A04, // v_add_nc_u32_e64 v3, s4, s5
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        0x4A06_0607, // v_add_nc_u32_e32 v3, s7, v3
        V_LSHLREV_V1_2_V0,
        0xDC70_8000, 0x0002_0301, // global_store_dword v1, v3, s[2:3]
        S_ENDPGM,
    ]);
    let got = run_io(&w, 0, 96, 0);
    assert_eq!(got[..64], [(0..64).sum::<u32>() - 1; 64]);
    assert_eq!(got[64..], [(64..96).sum::<u32>() - 1; 32]);
}

/// The shape of the G-buffer culling shaders' group flag (#182), as a synthetic program written
/// for this test, in groups of two console waves (128 lanes, `s5` the info SGPR, LDS on): LDS
/// word 0 cleared and a barrier, then (with `end_wave_1`) the second console wave ends, as a
/// culled one does. Each wave ORs its lanes' flags in a whole-wave section (row rotations, then
/// `v_permlanex16_b32`, then lanes 31 and 32 into every lane), lane 0 alone ORs lane 63's total
/// into LDS word 0, and after the barrier every lane stores that word. Thread 5 (the lower host
/// half) flags 2 and thread 40 (the upper) 4, so the total needs both halves; thread 70 flags 8.
fn group_flag(end_wave_1: bool) -> Vec<u32> {
    let mut w = vec![
        0x7E12_0280, // v_mov_b32 v9, 0
        0xD834_0000, 0x0000_0909, // ds_write_b32 v9, v9
        S_WAITCNT_LGKM,
        0xBF8A_0000, // s_barrier
    ];
    if end_wave_1 {
        w.extend([
            0x9010_8605, // s_lshr_b32 s16, s5, 6
            0x8710_BF10, // s_and_b32 s16, s16, 63 (the wave's number in the group)
            0xBF07_8010, // s_cmp_lg_u32 s16, 0
            0xBF85_0031, // s_cbranch_scc1 (the s_endpgm)
        ]);
    }
    w.extend([
        0x7E08_0280, // v_mov_b32 v4, 0
        0x7D84_0085, // v_cmp_eq_u32 vcc, 5, v0
        0xD501_0004, 0x01A9_0504, // v_cndmask_b32_e64 v4, v4, 2, vcc
        0x7D84_00A8, // v_cmp_eq_u32 vcc, 40, v0
        0xD501_0004, 0x01A9_0904, // v_cndmask_b32_e64 v4, v4, 4, vcc
        0x7D84_00FF, 70, // v_cmp_eq_u32 vcc, 70, v0
        0xD501_0004, 0x01A9_1104, // v_cndmask_b32_e64 v4, v4, 8, vcc
        // The whole-wave OR.
        0xBE98_25C1, // s_or_saveexec_b64 s[24:25], -1
        0xD501_000C, 0x0062_0880, // v_cndmask_b32_e64 v12, 0, v4, s[24:25]
        0x3818_18FA, 0xFF01_210C, // v_or_b32_dpp v12, v12, v12 row_ror:1
        0x3818_18FA, 0xFF01_220C, // v_or_b32_dpp v12, v12, v12 row_ror:2
        0x3818_18FA, 0xFF01_240C, // v_or_b32_dpp v12, v12, v12 row_ror:4
        0x3818_18FA, 0xFF01_280C, // v_or_b32_dpp v12, v12, v12 row_ror:8
        0xD778_000D, 0x0305_830C, // v_permlanex16_b32 v13, v12, -1, -1
        0x3818_190D, // v_or_b32_e32 v12, v13, v12
        0xD760_0009, 0x0201_3F0C, // v_readlane_b32 s9, v12, 31
        0x3818_1809, // v_or_b32_e32 v12, s9, v12
        0xD760_000B, 0x0201_410C, // v_readlane_b32 s11, v12, 32
        0x3818_180B, // v_or_b32_e32 v12, s11, v12
        0xBEFE_0418, // s_mov_b64 exec, s[24:25]
        // Lane 0 alone puts lane 63's total in LDS word 0.
        0xBEFE_0481, // s_mov_b64 exec, 1
        0xD760_000A, 0x0201_7F0C, // v_readlane_b32 s10, v12, 63
        0x7E1C_0280, // v_mov_b32 v14, 0
        0x7E1E_020A, // v_mov_b32 v15, s10
        0xD828_0000, 0x0000_0F0E, // ds_or_b32 v14, v15
        0xBEFE_0418, // s_mov_b64 exec, s[24:25]
        S_WAITCNT_LGKM,
        0xBF8A_0000, // s_barrier
        // Every lane stores LDS word 0.
        0x7E20_0280, // v_mov_b32 v16, 0
        0xD8D8_0000, 0x1000_0010, // ds_read_b32 v16, v16
        S_WAITCNT_LGKM,
        V_LSHLREV_V1_2_V0,
        0xDC70_8000, 0x0002_1001, // global_store_dword v1, v16, s[2:3]
        S_ENDPGM,
    ]);
    run_io(&w, 1 << 10 | 1 << 15, 128, 0)
}

#[test]
fn a_whole_wave_or_one_lane_puts_in_lds_reaches_the_group_at_its_barrier() {
    // Without the barrier in the block, the exchanges were refused there: the block is reached
    // by uniform branches, and a culled wave may already have ended.
    let got = group_flag(true);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..64], [6; 64]);
    assert_eq!(got[64..], [CANARY; 64]);
    // Both console waves there: each ORs its own flags in.
    assert_eq!(group_flag(false), [14; 128]);
}

/// A whole-wave product (`WaveActiveProduct`): lanes 0..34 hold 2.0, the rest 1.0, so both
/// halves contribute and every lane must see 2^35.
#[test]
fn a_whole_wave_product_multiplies_both_halves() {
    let mut w = vec![
        0x7D88_00A3, // v_cmp_gt_u32 vcc, 35, v0
        0xD501_0004,
        0x01A9_E8F2, // v_cndmask_b32_e64 v4, 1.0, 2.0, vcc
        0xBE88_25C1, // s_or_saveexec_b64 s[8:9], -1
        0xD501_0001,
        0x0022_08F2, // v_cndmask_b32_e64 v1, 1.0, v4, s[8:9]
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
        0xD760_0004,
        0x0201_0101, // v_readlane_b32 s4, v1, 0
        0xD760_0005,
        0x0201_4101, // v_readlane_b32 s5, v1, 32
        0xD508_0003,
        0x0200_0A04, // v_mul_f32_e64 v3, s4, s5
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        V_LSHLREV_V1_2_V0,
        0xDC70_8000,
        0x0002_0301, // global_store_dword v1, v3, s[2:3]
    ];
    w.push(S_ENDPGM);
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got, [2f32.powi(35).to_bits(); 64]);
}

/// A dispatch of thread counts (`DispatchX` with thread dimensions, `Options::thread_counts`) of
/// `count` threads in x, in groups of `per` lanes of 64-lane waves (`wave64`) or 32: the first
/// `slots` dwords of `dst`. `s5` is the group id.
fn run_threads(words: &[u32], wave64: bool, per: u32, count: u32, slots: usize) -> Vec<u32> {
    let Some(mut t) = title() else { return Vec::new() };
    let f = Fixture::new(&t);
    let options = victory_shader::emit::Options { thread_counts: true, ..shaders::options() };
    let cs = translate_wave(words, 5, 1 << 7, per, wave64, options);
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "threads") else {
        panic!("not made: {:?}", t.shaders)
    };
    let mut l = NativeList::default();
    l.set_compute_pipeline(&pso);
    let [c, d] = halves(VA_A + 0x4_0000);
    l.set_compute_root_constants(0, &[0, 0, c, d, 0]);
    l.dispatch_threads([count, 1, 1]).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    f.get(DST, slots)
}

/// Each lane EXEC has on stores the count of EXEC's lanes at `dst[group * lanes + lane]`:
/// `s_bcnt1_i32_b64 s6, exec` (`_b32`, `exec_lo` in 32-lane waves), `s_lshl_b32 s7, s5, 6` (5).
fn count_of_exec(wave64: bool) -> Vec<u32> {
    let (count, shift) = if wave64 { (0xBE86_107E, 0x8F07_8605) } else { (0xBE86_0F7E, 0x8F07_8505) };
    let mut w = vec![
        count,
        0x7E04_0206, // v_mov_b32 v2, s6
        shift,
        0x4A02_0007, // v_add_nc_u32 v1, s7, v0
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
    ];
    w.extend(GLOBAL_STORE_V2_AT_V1);
    w.push(S_ENDPGM);
    w
}

#[test]
fn a_dispatch_of_thread_counts_starts_the_lanes_past_them_off() {
    // 100 threads in groups of 64: the second group's lanes from 36 on are not launched. They
    // store nothing and the group's count of EXEC is 36 (the ray-tracing node writers of Grand
    // Theft Auto VI count their nodes so; run as 100 groups they wrote 6,400).
    let w = count_of_exec(true);
    let got = run_threads(&w, true, 64, 100, 192);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..64], [64; 64]);
    assert_eq!(got[64..100], [36; 36]);
    assert_eq!(got[100..], [CANARY; 92]);
    // One host wave's worth: the upper half of the console wave has no lane on.
    let got = run_threads(&w, true, 64, 21, 64);
    assert_eq!(got[..21], [21; 21]);
    assert_eq!(got[21..], [CANARY; 43]);
    // A count that fills its groups leaves every lane on.
    let got = run_threads(&w, true, 64, 128, 192);
    assert_eq!(got[..128], [64; 128]);
    assert_eq!(got[128..], [CANARY; 64]);
}

#[test]
fn a_dispatch_of_thread_counts_in_32_lane_waves() {
    // The title's third such shader runs 32-lane waves, one thread of one group.
    let w = count_of_exec(false);
    let got = run_threads(&w, false, 32, 1, 64);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[0], 1);
    assert_eq!(got[1..], [CANARY; 63]);
    let got = run_threads(&w, false, 32, 40, 64);
    assert_eq!(got[..32], [32; 32]);
    assert_eq!(got[32..40], [8; 8]);
    assert_eq!(got[40..], [CANARY; 24]);
}

#[test]
fn a_whole_wave_sum_under_thread_counts_leaves_out_the_lanes_past_them() {
    // The sum of the lanes' indices with every lane forced on, the lanes EXEC had off adding the
    // 0 of the v_cndmask: the lanes past the count are such lanes, as on the console, where they
    // run the section too.
    let w = [
        0xBE88_25C1, // s_or_saveexec_b64 s[8:9], -1
        0xD501_0001, 0x0022_0080, // v_cndmask_b32_e64 v1, 0, v0, s[8:9]
        0x4A02_02FA, 0xFF00_B101, // v_add_nc_u32_dpp v1, v1, v1 quad_perm:[1,0,3,2]
        0x4A02_02FA, 0xFF00_4E01, // v_add_nc_u32_dpp v1, v1, v1 quad_perm:[2,3,0,1]
        0x4A02_02FA, 0xFF01_4101, // v_add_nc_u32_dpp v1, v1, v1 row_half_mirror
        0x4A02_02FA, 0xFF01_4001, // v_add_nc_u32_dpp v1, v1, v1 row_mirror
        0xD778_0002, 0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x4A02_0501, // v_add_nc_u32_e32 v1, v1, v2
        0xD760_0004, 0x0201_0101, // v_readlane_b32 s4, v1, 0
        0xD760_0005, 0x0201_4101, // v_readlane_b32 s5, v1, 32
        0xD525_0003, 0x0200_0A04, // v_add_nc_u32_e64 v3, s4, s5
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        V_LSHLREV_V1_2_V0,
        0xDC70_8000, 0x0002_0301, // global_store_dword v1, v3, s[2:3]
        S_ENDPGM,
    ];
    let got = run_threads(&w, true, 64, 45, 64);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..45], [(0..45).sum::<u32>(); 45]);
    assert_eq!(got[45..], [CANARY; 19]);
}

#[test]
fn a_mask_moved_to_every_lane_before_a_branch_to_a_join_holds_their_lanes() {
    // A join of a lane mask where lanes arrive from two places (per lane, as the translation
    // follows the console's EXEC regions): lanes 0..39 set `s[8:9]` to -1 and branch to it, lanes
    // 40..63 skip that region and compare themselves against 50. Each reads its own path's bit:
    // 1 below 50, 0 from 50 (was refused: "data, at a join of several edges").
    let w = [
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        0x8A86_7E04, // s_andn2_b64 s[6:7], s[4:5], exec
        skip_execz(2),
        0xBE88_04C1, // s_mov_b64 s[8:9], -1
        0xBF82_0003, // s_branch to the join
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xD4C4_0008, 0x0202_00B2, // v_cmp_gt_u32 s[8:9], 50, v0
        0xBEFE_0404, // join: s_mov_b64 exec, s[4:5]
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s[8:9]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..50], [1; 50]);
    assert_eq!(got[50..], [0; 14]);
}

#[test]
fn a_mask_anded_with_an_exec_its_skipping_lanes_are_not_in_is_theirs_at_zero() {
    // `s[8:9]` holds an address (data) until lanes 0..39 make it the lanes below 10 or outside
    // EXEC (`s_orn2`, which sets lanes 40..63); lanes 40..63 skip that and join holding the
    // address, but the join's only read is an AND with an EXEC they are not in, so each reads a 0
    // there (was refused: "data, ...").
    let w = [
        0xBE88_0402, // s_mov_b64 s[8:9], s[2:3]
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        skip_execz(2),
        0x7D88_008A, // v_cmp_gt_u32 vcc, 10, v0
        0x8B88_7E6A, // s_orn2_b64 s[8:9], vcc, exec: lanes 40..63 set too
        0x8788_7E08, // join: s_and_b64 s[8:9], s[8:9], exec
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s[8:9]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..10], [1; 10]);
    assert_eq!(got[10..], [0; 54]);
}

#[test]
fn data_joined_into_a_mask_seen_only_anded_off_at_a_later_join_is_not_needed() {
    // `s[8:9]` holds a mask (the lanes below 50) that lanes 0..39 replace with loaded data; the
    // first join (data for lanes 0..39) is read only by a second join, where lanes 20..63 skip in holding it and
    // the only read is an AND with an EXEC they are not in. No lane sees the first join's bits
    // (the title's 1280x720 compute pass; was refused: "data, ...").
    let w = [
        0xD4C4_0008, 0x0202_00B2, // v_cmp_gt_u32 s[8:9], 50, v0
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        skip_execz(3),
        0xF404_0201, 0xFA00_0000, // s_load_dwordx2 s[8:9], s[2:3], 0
        S_WAITCNT_LGKM,
        0xBEFE_0404, // first join: s_mov_b64 exec, s[4:5]
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        skip_execz(2),
        0xD4C4_0008, 0x0202_0085, // v_cmp_gt_u32 s[8:9], 5, v0
        0x8788_7E08, // second join: s_and_b64 s[8:9], s[8:9], exec
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s[8:9]
        V_LSHLREV_V1_2_V0,
        GLOBAL_STORE_V2_AT_V1[0], GLOBAL_STORE_V2_AT_V1[1],
        S_ENDPGM,
    ];
    assert_eq!(refusal(&w), None);
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..5], [1; 5]);
    assert_eq!(got[5..], [0; 59]);
}

#[test]
fn data_joined_into_a_mask_whose_lanes_write_it_again_in_an_else_is_not_needed() {
    // `s[8:9]` holds the lanes below 50, then loaded data for lanes 0..59 (a first join of data
    // and a mask). An if/else on the lanes below 40 follows: the then part makes a new mask, and
    // the lanes that skip it (40..63, holding the first join) run the else part, which writes
    // the register again before any read. They never take the edge that brings the first join
    // to the end, so no lane sees it (the title's 1280x720 compute pass; was refused: "data,
    // ...").
    let w = [
        0xD4C4_0008, 0x0202_00B2, // v_cmp_gt_u32 s[8:9], 50, v0
        0x7D88_00BC, // v_cmp_gt_u32 vcc, 60, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        skip_execz(3),
        0xF404_0201, 0xFA00_0000, // s_load_dwordx2 s[8:9], s[2:3], 0
        S_WAITCNT_LGKM,
        0xBEFE_0404, // first join: s_mov_b64 exec, s[4:5]
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        skip_execz(2),
        0xD4C4_0008, 0x0202_008A, // then: v_cmp_gt_u32 s[8:9], 10, v0
        0x8AFE_7E06, // s_andn2_b64 exec, s[6:7], exec
        skip_execz(2),
        0xD4C4_0008, 0x0202_00B2, // else: v_cmp_gt_u32 s[8:9], 50, v0
        0xBEFE_0406, // end: s_mov_b64 exec, s[6:7]
        0x8788_7E08, // s_and_b64 s[8:9], s[8:9], exec
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s[8:9]
        V_LSHLREV_V1_2_V0,
        GLOBAL_STORE_V2_AT_V1[0],
        GLOBAL_STORE_V2_AT_V1[1],
        S_ENDPGM,
    ];
    assert_eq!(refusal(&w), None);
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..10], [1; 10]);
    assert_eq!(got[10..40], [0; 30]);
    assert_eq!(got[40..50], [1; 10]);
    assert_eq!(got[50..], [0; 14]);
}

#[test]
fn a_mask_selected_by_scc_joined_with_data_nanded_with_exec_is_each_lanes_bit() {
    // `s[8:9]` holds loaded data; lanes 0..39 make it -1 or 0 by SCC (here -1). At the join
    // the only read is an `s_nand` with EXEC: lanes 40..63 skipped in with the data and are not
    // in that EXEC, so their bits are 1 whatever it holds; lanes 0..39 read SCC's bit, giving
    // 0 (a small compute pass of the title's that holds the HDR scene colour; was refused:
    // "data, ...").
    let w = [
        0xF404_0201, 0xFA00_0000, // s_load_dwordx2 s[8:9], s[2:3], 0
        S_WAITCNT_LGKM,
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        skip_execz(2),
        0xBF07_8004, // s_cmp_lg_u32 s4, 0
        0x8588_80C1, // s_cselect_b64 s[8:9], -1, 0
        0x8C88_7E08, // join: s_nand_b64 s[8:9], s[8:9], exec
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s[8:9]
        V_LSHLREV_V1_2_V0,
        GLOBAL_STORE_V2_AT_V1[0],
        GLOBAL_STORE_V2_AT_V1[1],
        S_ENDPGM,
    ];
    assert_eq!(refusal(&w), None);
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..40], [0; 40]);
    assert_eq!(got[40..], [1; 24]);
}

#[test]
fn a_mask_join_read_by_two_ands_with_execs_its_skipping_lanes_are_not_in_is_theirs_at_zero() {
    // `s[8:9]` holds the lanes below 50, then loaded data for lanes 0..39 (a join of data and a
    // mask). Two ifs follow, on the lanes below 20 and below 30, each making a new mask; the
    // lanes that skip each join holding the older value, read there only ANDed with the EXEC
    // of the lanes that ran the if: no lane sees the data.
    let w = [
        0xD4C4_0008, 0x0002_00B2, // v_cmp_gt_u32 s[8:9], 50, v0
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        0xBF88_0003, // s_cbranch_execz j1
        0xF404_0201, 0xFA00_0000, // s_load_dwordx2 s[8:9], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        // j1:
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        0xBF88_0002, // s_cbranch_execz j2
        0xD4C4_0008, 0x0002_0085, // v_cmp_gt_u32 s[8:9], 5, v0
        // j2:
        0x878A_7E08, // s_and_b64 s[10:11], s[8:9], exec
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0x7D88_009E, // v_cmp_gt_u32 vcc, 30, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        0xBF88_0002, // s_cbranch_execz j3
        0xD4C4_0008, 0x0002_0083, // v_cmp_gt_u32 s[8:9], 3, v0
        // j3:
        0x8788_7E08, // s_and_b64 s[8:9], s[8:9], exec
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xD501_0002, 0x0029_0280, // v_cndmask_b32 v2, 0, 1, s[10:11]
        0xD501_0003, 0x0021_0480, // v_cndmask_b32 v3, 0, 2, s[8:9]
        0x3804_0702, // v_or_b32 v2, v2, v3
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    assert_eq!(refusal(&w), None);
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..3], [3; 3]);
    assert_eq!(got[3..5], [1; 2]);
    assert_eq!(got[5..], [0; 59]);
}

#[test]
fn a_mask_a_uniform_loop_carries_round_anded_off_is_not_needed() {
    // A uniform loop of three: the lanes below 40 run an if on the lanes below 20 that makes
    // the lanes below 10, read ANDed with the if's EXEC, then load data into the register,
    // which the loop carries round to the next if's join. The lanes that skip the inner if
    // read it ANDed off, so no lane sees the data: the lanes below 10 count 3, the rest 0.
    let w = [
        0xD4C4_0008, 0x0002_00B2, // v_cmp_gt_u32 s[8:9], 50, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        0xBE8E_0380, // s_mov_b32 s14, 0
        // loop:
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_000D, // s_cbranch_execz latch
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0002, // s_cbranch_execz ij
        0xD4C4_0008, 0x0002_008A, // v_cmp_gt_u32 s[8:9], 10, v0
        // ij:
        0x8788_7E08, // s_and_b64 s[8:9], s[8:9], exec
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0xD501_0003, 0x0021_0280, // v_cndmask_b32 v3, 0, 1, s[8:9]
        0x4A04_0702, // v_add_nc_u32 v2, v2, v3
        0xF404_0201, 0xFA00_0000, // s_load_dwordx2 s[8:9], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        // latch:
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
        0x800E_810E, // s_add_u32 s14, s14, 1
        0xBF0A_830E, // s_cmp_lt_u32 s14, 3
        0xBF85_FFEC, // s_cbranch_scc1 loop
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    assert_eq!(refusal(&w), None);
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..10], [3; 10]);
    assert_eq!(got[10..], [0; 54]);
}

#[test]
fn data_joined_with_a_mask_nanded_with_exec_is_set_for_lanes_outside_it() {
    // `s[8:9]` holds an address; lanes 0..39 make it the lanes below 50. The join's only read
    // is an `s_nand` with EXEC (lanes 0..39): lanes 40..63 hold the address there but are not in
    // that EXEC, so their bits are 1 whatever it holds; lanes 0..39 get NOT of their bit, 0.
    let w = [
        0xD4C4_000C, 0x0002_00B2, // v_cmp_gt_u32 s[12:13], 50, v0
        0xBE88_0402, // s_mov_b64 s[8:9], s[2:3]
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        0xBF88_0001, // s_cbranch_execz j
        0xBE88_040C, // s_mov_b64 s[8:9], s[12:13]
        // j:
        0x8C88_7E08, // s_nand_b64 s[8:9], s[8:9], exec
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s[8:9]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    assert_eq!(refusal(&w), None);
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..40], [0; 40]);
    assert_eq!(got[40..], [1; 24]);
}

#[test]
fn a_mask_selected_by_scc_is_scc_for_every_lane() {
    // `s[8:9]` holds an address; lanes 0..39 make it -1 or 0 by SCC (`s4` is 0: -1). The join
    // ANDs it with their EXEC: lanes 0..39 read SCC, 1; the others are not in that EXEC.
    let w = [
        0xBE88_0402, // s_mov_b64 s[8:9], s[2:3]
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        0xBF88_0002, // s_cbranch_execz j
        0xBF06_8004, // s_cmp_eq_u32 s4, 0
        0x8588_80C1, // s_cselect_b64 s[8:9], -1, 0
        // j:
        0x8788_7E08, // s_and_b64 s[8:9], s[8:9], exec
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s[8:9]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    assert_eq!(refusal(&w), None);
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..40], [1; 40]);
    assert_eq!(got[40..], [0; 24]);
}

#[test]
fn a_mask_of_the_lanes_leaving_a_loop_by_a_uniform_exit_holds_them() {
    // A loop whose lanes leave as `v3` (`v0` plus 16 a time) reaches 64, ORing each leaver into
    // `s[8:9]` (moved to 0 before it), and that the wave leaves early when `s14` reaches `s4`
    // (2): at the uniform exit the mask holds the lanes from 32 (the title's shadow pass; was
    // refused: "data, at a join of several edges").
    let w = [
        0xBE88_0480, // s_mov_b64 s[8:9], 0
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0xBE8E_0380, // s_mov_b32 s14, 0
        0x7E06_0300, // v_mov_b32 v3, v0
        // loop:
        0xBE8C_047E, // s_mov_b64 s[12:13], exec
        0xBF06_040E, // s_cmp_eq_u32 s14, s4
        0xBF85_0006, // s_cbranch_scc1 out
        0x800E_810E, // s_add_u32 s14, s14, 1
        0x4A06_0690, // v_add_nc_u32 v3, 16, v3
        0x7DA6_06C0, // v_cmpx_le_u32 64, v3
        0x8888_7E08, // s_or_b64 s[8:9], s[8:9], exec
        0xBE8C_3A0C, // s_andn2_wrexec_b64 s[12:13], s[12:13]
        0xBF85_FFF7, // s_cbranch_scc1 loop
        // out:
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s[8:9]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    assert_eq!(refusal(&w), None);
    let got = run(&w, 0, 2);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..32], [0; 32]);
    assert_eq!(got[32..], [1; 32]);
}

#[test]
fn a_mask_join_read_by_an_or_with_exec_is_not_taken_as_anded_off() {
    // As the two-AND case, but the first read is an `s_or` with EXEC, whose bits for lanes
    // outside EXEC are the data's own: still refused.
    let w = [
        0xD4C4_0008, 0x0002_00B2, // v_cmp_gt_u32 s[8:9], 50, v0
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        0xBF88_0003, // s_cbranch_execz j1
        0xF404_0201, 0xFA00_0000, // s_load_dwordx2 s[8:9], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        // j1:
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        0xBF88_0002, // s_cbranch_execz j2
        0xD4C4_0008, 0x0002_0085, // v_cmp_gt_u32 s[8:9], 5, v0
        // j2:
        0x888A_7E08, // s_or_b64 s[10:11], s[8:9], exec
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0x7D88_009E, // v_cmp_gt_u32 vcc, 30, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        0xBF88_0002, // s_cbranch_execz j3
        0xD4C4_0008, 0x0002_0083, // v_cmp_gt_u32 s[8:9], 3, v0
        // j3:
        0x8788_7E08, // s_and_b64 s[8:9], s[8:9], exec
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xD501_0002, 0x0029_0280, // v_cndmask_b32 v2, 0, 1, s[10:11]
        0xD501_0003, 0x0021_0480, // v_cndmask_b32 v3, 0, 2, s[8:9]
        0x3804_0702, // v_or_b32 v2, v2, v3
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    let why = refusal(&w).expect("refused");
    assert!(why.contains("a lane-mask join"), "{why}");
}

#[test]
fn a_mask_join_read_by_an_xor_with_exec_is_not_taken_as_anded_off() {
    // As the NAND case, but an `s_xor` with EXEC: lanes outside EXEC keep the address's bits,
    // so it is still refused.
    let w = [
        0xD4C4_000C, 0x0002_00B2, // v_cmp_gt_u32 s[12:13], 50, v0
        0xBE88_0402, // s_mov_b64 s[8:9], s[2:3]
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        0xBF88_0001, // s_cbranch_execz j
        0xBE88_040C, // s_mov_b64 s[8:9], s[12:13]
        // j:
        0x8988_7E08, // s_xor_b64 s[8:9], s[8:9], exec
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s[8:9]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    let why = refusal(&w).expect("refused");
    assert!(why.contains("a lane-mask join"), "{why}");
}

#[test]
fn a_loop_carried_mask_read_by_an_or_with_exec_is_not_taken_as_anded_off() {
    // As the uniform loop case, but the inner join's read is an `s_or` with EXEC: still refused.
    let w = [
        0xD4C4_0008, 0x0002_00B2, // v_cmp_gt_u32 s[8:9], 50, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        0xBE8E_0380, // s_mov_b32 s14, 0
        // loop:
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_000D, // s_cbranch_execz latch
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0002, // s_cbranch_execz ij
        0xD4C4_0008, 0x0002_008A, // v_cmp_gt_u32 s[8:9], 10, v0
        // ij:
        0x8888_7E08, // s_or_b64 s[8:9], s[8:9], exec
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0xD501_0003, 0x0021_0280, // v_cndmask_b32 v3, 0, 1, s[8:9]
        0x4A04_0702, // v_add_nc_u32 v2, v2, v3
        0xF404_0201, 0xFA00_0000, // s_load_dwordx2 s[8:9], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        // latch:
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
        0x800E_810E, // s_add_u32 s14, s14, 1
        0xBF0A_830E, // s_cmp_lt_u32 s14, 3
        0xBF85_FFEC, // s_cbranch_scc1 loop
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    let why = refusal(&w).expect("refused");
    assert!(why.contains("a lane-mask join"), "{why}");
}

#[test]
fn a_select_of_an_address_by_scc_is_not_taken_as_a_mask() {
    // As the SCC case, but the select is of the address or 0, not -1 or 0: still refused.
    let w = [
        0xBE88_0402, // s_mov_b64 s[8:9], s[2:3]
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        0xBF88_0002, // s_cbranch_execz j
        0xBF06_8004, // s_cmp_eq_u32 s4, 0
        0x8588_8002, // s_cselect_b64 s[8:9], s[2:3], 0
        // j:
        0x8788_7E08, // s_and_b64 s[8:9], s[8:9], exec
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s[8:9]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    let why = refusal(&w).expect("refused");
    assert!(why.contains("a lane-mask join"), "{why}");
}

/// [`run_io`] of a wave64 group of 64, `dwords` read back from `dst`.
fn run_lockstep(words: &[u32], lds: u32, dwords: usize) -> Vec<u32> {
    let Some(mut t) = title() else { return Vec::new() };
    let f = Fixture::new(&t);
    let cs = translate_io(words, 5, lds << 15, 64, shaders::options());
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs, &[0, 0, c, d, 0], false);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    f.get(DST, dwords)
}

#[test]
fn a_lane_read_behind_an_exec_branch_runs_in_lockstep() {
    // Lanes 0..39 read lane 35's `v3` (1035) and `exec_hi` as data (lanes 32..39: 0xff) behind
    // `s_cbranch_execz`: refused on the fast path (the halves split), run in lockstep.
    let w = [
        V_LSHLREV_V1_2_V0,
        0x4A06_00FF,
        0x0000_03E8, // v_add_nc_u32 v3, 1000, v0
        LANES_BELOW_40[0],
        LANES_BELOW_40[1],
        skip_execz(8),
        0xD760_0006,
        0x0001_4703, // v_readlane_b32 s6, v3, 35
        0x7E04_0206, // v_mov_b32 v2, s6
        GLOBAL_STORE_V2_AT_V1[0],
        GLOBAL_STORE_V2_AT_V1[1],
        0x7E04_027F, // v_mov_b32 v2, exec_hi
        0xDC70_8100,
        0x0002_0201, // global_store_dword v1, v2, s[2:3] offset:256
        0x88FE_047E, // s_or_b64 exec, exec, s[4:5]
        S_ENDPGM,
    ];
    let got = run_lockstep(&w, 0, 128);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..40], [1035; 40]);
    assert_eq!(got[40..64], [CANARY; 24]);
    assert_eq!(got[64..104], [0xff; 40]);
    assert_eq!(got[104..], [CANARY; 24]);
}

#[test]
fn a_first_lane_read_in_lockstep_is_the_console_waves_first_lane() {
    // Lanes 10..63 read the first lane EXEC leaves on (10: `v3` 1010), not each host wave's own
    // (32 in the upper one); with the lane read behind the branch the program runs in lockstep.
    let w = [
        V_LSHLREV_V1_2_V0,
        0x4A06_00FF,
        0x0000_03E8, // v_add_nc_u32 v3, 1000, v0
        0x7D82_0089, // v_cmp_lt_u32 vcc, 9, v0
        LANES_BELOW_40[1],
        skip_execz(9),
        0x7E0C_0503, // v_readfirstlane_b32 s6, v3
        0xD760_0007,
        0x0001_4703, // v_readlane_b32 s7, v3, 35
        0x7E04_0206, // v_mov_b32 v2, s6
        GLOBAL_STORE_V2_AT_V1[0],
        GLOBAL_STORE_V2_AT_V1[1],
        0x7E04_0207, // v_mov_b32 v2, s7
        0xDC70_8100,
        0x0002_0201, // global_store_dword v1, v2, s[2:3] offset:256
        0x88FE_047E, // s_or_b64 exec, exec, s[4:5]
        S_ENDPGM,
    ];
    let got = run_lockstep(&w, 0, 128);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..10], [CANARY; 10]);
    assert_eq!(got[10..64], [1010; 54]);
    assert_eq!(got[74..], [1035; 54]);
}

#[test]
fn a_per_lane_loop_shares_lds_across_the_halves_in_lockstep() {
    // Lane i loops `i & 3` times; each trip it adds the value lane `i ^ 32` last wrote to LDS to
    // its own and writes that back, with no barrier (the console runs the wave in lockstep).
    let w = [
        V_LSHLREV_V1_2_V0,
        0x3A06_02FF,
        0x0000_0080, // v_xor_b32 v3, 0x80, v1: lane i ^ 32's slot
        0x3608_0083, // v_and_b32 v4, 3, v0
        0x7E0A_0300, // v_mov_b32 v5, v0
        0xD834_0000,
        0x0000_0501, // ds_write_b32 v1, v5
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0x7D8A_0880, // loop: v_cmp_ne_u32 vcc, 0, v4
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        skip_execz(8),
        DS_READ_V2_AT_V3[0],
        DS_READ_V2_AT_V3[1],
        S_WAITCNT_LGKM,
        0x4A0A_0505, // v_add_nc_u32 v5, v5, v2
        0xD834_0000,
        0x0000_0501, // ds_write_b32 v1, v5
        0x4A08_08C1, // v_add_nc_u32 v4, -1, v4
        0xBF82_FFF5, // s_branch loop
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xDC70_8000,
        0x0002_0501, // global_store_dword v1, v5, s[2:3]
        S_ENDPGM,
    ];
    let got = run_lockstep(&w, 1, 64);
    if got.is_empty() {
        return;
    }
    // The console: each trip, every lane still looping reads, then each writes.
    let (mut acc, mut lds): (Vec<u32>, Vec<u32>) = ((0..64).collect(), (0..64).collect());
    for trip in 0..3 {
        let on: Vec<usize> = (0..64).filter(|&i| i & 3 > trip).collect();
        let read: Vec<u32> = on.iter().map(|&i| lds[i ^ 32]).collect();
        for (&i, r) in on.iter().zip(read) {
            acc[i] += r;
            lds[i] = acc[i];
        }
    }
    assert_eq!(got, acc);
}

/// Trips of thread `i` of a group of two wave64 waves in the tests below: the upper console
/// wave loops two more times than the lower.
fn trips(i: usize) -> usize {
    2 * (i >> 6) + (i & 1) + 1
}

#[test]
fn a_lane_read_in_per_lane_loops_of_two_console_waves_meets_its_own_halves() {
    // A group of two console waves. Lane i of wave w loops `trips(i)` times, adding lane 35's
    // `v3` (1000 + 64w + 35) each trip: refused on the fast path (the halves split), run in
    // lockstep, where wave 1 makes two more lane reads than wave 0, so no group barrier fits.
    let w = [
        V_LSHLREV_V1_2_V0,
        0x4A06_00FF,
        0x0000_03E8, // v_add_nc_u32 v3, 1000, v0
        0x2C08_0086, // v_lshrrev_b32 v4, 6, v0
        0x3408_0881, // v_lshlrev_b32 v4, 1, v4
        0x360C_0081, // v_and_b32 v6, 1, v0
        0x4A08_0D04, // v_add_nc_u32 v4, v4, v6
        0x4A08_0881, // v_add_nc_u32 v4, 1, v4
        0x7E0A_0280, // v_mov_b32 v5, 0
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0x7D8A_0880, // loop: v_cmp_ne_u32 vcc, 0, v4
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        skip_execz(5),
        0xD760_0008,
        0x0001_4703, // v_readlane_b32 s8, v3, 35
        0x4A0A_0A08, // v_add_nc_u32 v5, s8, v5
        0x4A08_08C1, // v_add_nc_u32 v4, -1, v4
        0xBF82_FFF8, // s_branch loop
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xDC70_8000,
        0x0002_0501, // global_store_dword v1, v5, s[2:3]
        S_ENDPGM,
    ];
    let got = run_io(&w, 0, 128, 0);
    if got.is_empty() {
        return;
    }
    let want: Vec<u32> = (0..128).map(|i| (trips(i) * (1035 + 64 * (i >> 6))) as u32).collect();
    assert_eq!(got, want);
}

#[test]
fn lds_shared_across_the_halves_in_per_lane_loops_of_two_console_waves() {
    // `a_per_lane_loop_shares_lds_across_the_halves_in_lockstep` in a group of two console
    // waves, lane i looping `i & 3` times and wave 1 two more, also adding lane 35's sum each
    // trip (a lane read: refused on the fast path): each wave's halves meet around their LDS
    // writes and lane reads, a different number of times in each wave.
    let w = [
        V_LSHLREV_V1_2_V0,
        0x3A06_02FF,
        0x0000_0080, // v_xor_b32 v3, 0x80, v1: lane i ^ 32's slot
        0x3608_0083, // v_and_b32 v4, 3, v0
        0x2C0C_0086, // v_lshrrev_b32 v6, 6, v0
        0x340C_0C81, // v_lshlrev_b32 v6, 1, v6
        0x4A08_0D04, // v_add_nc_u32 v4, v4, v6
        0x7E0A_0300, // v_mov_b32 v5, v0
        0xD834_0000,
        0x0000_0501, // ds_write_b32 v1, v5
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0x7D8A_0880, // loop: v_cmp_ne_u32 vcc, 0, v4
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        skip_execz(11),
        0xD760_0008,
        0x0001_4705, // v_readlane_b32 s8, v5, 35
        DS_READ_V2_AT_V3[0],
        DS_READ_V2_AT_V3[1],
        S_WAITCNT_LGKM,
        0x4A0A_0505, // v_add_nc_u32 v5, v5, v2
        0x4A0A_0A08, // v_add_nc_u32 v5, s8, v5
        0xD834_0000,
        0x0000_0501, // ds_write_b32 v1, v5
        0x4A08_08C1, // v_add_nc_u32 v4, -1, v4
        0xBF82_FFF2, // s_branch loop
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xDC70_8000,
        0x0002_0501, // global_store_dword v1, v5, s[2:3]
        S_ENDPGM,
    ];
    let got = run_io(&w, 1 << 15, 128, 0);
    if got.is_empty() {
        return;
    }
    // The console, each wave on its own: each trip, the wave reads lane 35's sum, every lane
    // still looping reads LDS, then each writes.
    let (mut acc, mut lds): (Vec<u32>, Vec<u32>) = ((0..128).collect(), (0..128).collect());
    let n = |i: usize| (i & 3) + 2 * (i >> 6);
    for trip in 0..5 {
        for wave in 0..2 {
            let on: Vec<usize> = (64 * wave..64 * wave + 64).filter(|&i| n(i) > trip).collect();
            let lane35 = acc[64 * wave + 35];
            let read: Vec<u32> = on.iter().map(|&i| lds[i ^ 32]).collect();
            for (&i, r) in on.iter().zip(read) {
                acc[i] += r + lane35;
                lds[i] = acc[i];
            }
        }
    }
    assert_eq!(got, acc);
}

#[test]
fn a_whole_wave_prefix_sum_behind_an_exec_branch_runs_in_lockstep() {
    // In lanes 0..49, an inclusive prefix sum of the lane indices over the whole wave (lanes off
    // give 0): DPP row shifts, the other row's lane 15, then lane 31 added in the upper half.
    let w = [
        V_LSHLREV_V1_2_V0,
        0x7D88_00B2, // v_cmp_gt_u32 vcc, 50, v0
        LANES_BELOW_40[1],
        skip_execz(23),
        0xBE88_25C1, // s_or_saveexec_b64 s[8:9], -1
        0xD501_0003,
        0x0022_0080, // v_cndmask_b32_e64 v3, 0, v0, s[8:9]
        0x4A06_06FA,
        0xFF09_1103, // v_add_nc_u32_dpp v3, v3, v3 row_shr:1 bound_ctrl:1
        0x4A06_06FA,
        0xFF09_1203, // v_add_nc_u32_dpp v3, v3, v3 row_shr:2 bound_ctrl:1
        0x4A06_06FA,
        0xFF09_1403, // v_add_nc_u32_dpp v3, v3, v3 row_shr:4 bound_ctrl:1
        0x4A06_06FA,
        0xFF09_1803, // v_add_nc_u32_dpp v3, v3, v3 row_shr:8 bound_ctrl:1
        0xD778_0002,
        0x0305_8303, // v_permlanex16_b32 v2, v3, -1, -1
        0x4A06_06FA,
        0xAF00_E402, // v_add_nc_u32_dpp v3, v2, v3 row_mask:0xa
        0xD760_000A,
        0x0201_3F03, // v_readlane_b32 s10, v3, 31
        0x7E04_020A, // v_mov_b32 v2, s10
        0x4A06_06FA,
        0xCF00_E402, // v_add_nc_u32_dpp v3, v2, v3 row_mask:0xc
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        0xDC70_8000,
        0x0002_0301, // global_store_dword v1, v3, s[2:3]
        0x88FE_047E, // s_or_b64 exec, exec, s[4:5]
        S_ENDPGM,
    ];
    let got = run_lockstep(&w, 0, 64);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..50], (0..50).map(|i| i * (i + 1) / 2).collect::<Vec<u32>>());
    assert_eq!(got[50..], [CANARY; 14]);
}

#[test]
fn whole_wave_code_that_is_no_reduction_runs_literally_in_lockstep() {
    // Behind an EXEC branch, with every lane on: a DPP row shift (lane i takes lane i - 1 of its
    // row of 16, 0 at a row's start) and lane 33's result read back, which no reduction or prefix
    // describes. Lockstep runs the instructions as they are.
    let w = [
        V_LSHLREV_V1_2_V0,
        0x7D88_00B2, // v_cmp_gt_u32 vcc, 50, v0
        LANES_BELOW_40[1],
        skip_execz(9),
        0xBE88_25C1, // s_or_saveexec_b64 s[8:9], -1
        0x7E06_02FA,
        0xFF09_1100, // v_mov_b32_dpp v3, v0 row_shr:1 bound_ctrl:1
        0xD760_000A,
        0x0201_4303, // v_readlane_b32 s10, v3, 33
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        0x4A06_060A, // v_add_nc_u32 v3, s10, v3
        0xDC70_8000,
        0x0002_0301, // global_store_dword v1, v3, s[2:3]
        0x88FE_047E, // s_or_b64 exec, exec, s[4:5]
        S_ENDPGM,
    ];
    let got = run_lockstep(&w, 0, 64);
    if got.is_empty() {
        return;
    }
    let want: Vec<u32> = (0..50).map(|i| if i % 16 == 0 { 32 } else { i - 1 + 32 }).collect();
    assert_eq!(got[..50], want);
    assert_eq!(got[50..], [CANARY; 14]);
}

#[test]
fn a_mask_a_32_lane_wave_starts_as_a_pairs_constant_holds_the_lanes_ored_in() {
    // The loop above in 32-lane waves, its mask made 0 by `s_mov_b64 s[8:9], 0` or
    // `s_bfm_b64 s[8:9], 32, 32` (as the G-buffer's 32-lane compute programs do): lanes leave
    // as `v0` plus 8 a time reaches 32, and the wave when `s14` reaches `s4` (2), so the mask
    // holds the lanes from 16 (was refused as data at a lane-mask join).
    // s_mov_b64 s[8:9], 0; s_bfm_b64 s[8:9], 32, 32
    for zero in [0xBE88_0480, 0x9288_A0A0] {
        let w = [
            zero,
            0xBE8A_037E, // s_mov_b32 s10, exec_lo
            0xBE8E_0380, // s_mov_b32 s14, 0
            0x7E06_0300, // v_mov_b32 v3, v0
            // loop:
            0xBE8C_037E, // s_mov_b32 s12, exec_lo
            0xBF06_040E, // s_cmp_eq_u32 s14, s4
            0xBF85_0006, // s_cbranch_scc1 out
            0x800E_810E, // s_add_u32 s14, s14, 1
            0x4A06_0688, // v_add_nc_u32 v3, 8, v3
            0x7DA6_06A0, // v_cmpx_le_u32 32, v3
            0x8808_7E08, // s_or_b32 s8, s8, exec_lo
            0xBE8C_470C, // s_andn2_wrexec_b32 s12, s12
            0xBF85_FFF7, // s_cbranch_scc1 loop
            // out:
            0xBEFE_030A, // s_mov_b32 exec_lo, s10
            0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s8
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
            S_ENDPGM,
        ];
        assert_eq!(refusal32(&w), None, "{zero:#x}");
        let got = run32(&w, 2);
        if got.is_empty() {
            return;
        }
        assert_eq!(got[..16], [0; 16]);
        assert_eq!(got[16..], [1; 16]);
    }
}

#[test]
fn a_mask_a_divergent_loop_copies_away_and_back_is_the_mask_it_entered_with() {
    // VCC holds the lanes below 20 into a loop the lanes leave as `v0` plus 8 a time reaches 32;
    // inside, the lanes of VCC count their passes and copy VCC to `s20` and back (the G-buffer's
    // 32-lane compute programs spill so). Every edge brings VCC or a copy of it, so it is the
    // mask it entered with (was refused as data at a lane-mask join).
    let w = [
        0x7D88_0094, // v_cmp_gt_u32 vcc_lo, 20, v0
        0xBE8A_037E, // s_mov_b32 s10, exec_lo
        0x7E06_0300, // v_mov_b32 v3, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        // loop:
        0xBE90_3C6A, // s_and_saveexec_b32 s16, vcc_lo
        0xBF88_0003, // s_cbranch_execz skip
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0xBEEA_0314, // s_mov_b32 vcc_lo, s20
        // skip:
        0xBEFE_0310, // s_mov_b32 exec_lo, s16
        0x4A06_0688, // v_add_nc_u32 v3, 8, v3
        0x7DA8_06A0, // v_cmpx_gt_u32 32, v3
        0xBF89_FFF7, // s_cbranch_execnz loop
        0xBEFE_030A, // s_mov_b32 exec_lo, s10
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    assert_eq!(refusal32(&w), None);
    let got = run32(&w, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..8], [4; 8]);
    assert_eq!(got[8..16], [3; 8]);
    assert_eq!(got[16..20], [2; 4]);
    assert_eq!(got[20..], [0; 12]);
}

#[test]
fn a_mask_a_divergent_loop_replaces_is_not_taken_as_the_one_it_entered_with() {
    // As the copy case, but the copy back is of another mask (the lanes below 10): after the
    // first pass VCC holds those, so lanes 10 to 19 count one pass. Refused, or that.
    let w = [
        0xD4C4_0015, 0x0002_008A, // v_cmp_gt_u32 s21, 10, v0
        0x7D88_0094, // v_cmp_gt_u32 vcc_lo, 20, v0
        0xBE8A_037E, // s_mov_b32 s10, exec_lo
        0x7E06_0300, // v_mov_b32 v3, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        // loop:
        0xBE90_3C6A, // s_and_saveexec_b32 s16, vcc_lo
        0xBF88_0003, // s_cbranch_execz skip
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0xBEEA_0315, // s_mov_b32 vcc_lo, s21
        // skip:
        0xBEFE_0310, // s_mov_b32 exec_lo, s16
        0x4A06_0688, // v_add_nc_u32 v3, 8, v3
        0x7DA8_06A0, // v_cmpx_gt_u32 32, v3
        0xBF89_FFF7, // s_cbranch_execnz loop
        0xBEFE_030A, // s_mov_b32 exec_lo, s10
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    if refusal32(&w).is_some() {
        return;
    }
    let got = run32(&w, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..8], [4; 8]);
    assert_eq!(got[8..10], [3; 2]);
    assert_eq!(got[10..20], [1; 10]);
    assert_eq!(got[20..], [0; 12]);
}

#[test]
fn a_mask_a_divergent_loop_swaps_away_and_back_by_xors_is_the_mask_it_entered_with() {
    // As the copy case, but VCC is swapped with data (`s4`) by three XORs and back by three
    // more (the NGG vertex programs swap SGPRs so): the swap's last XOR is a copy, so VCC at
    // the loop head is still the mask it entered with.
    let w = [
        0x7D88_0094, // v_cmp_gt_u32 vcc_lo, 20, v0
        0xBE8A_037E, // s_mov_b32 s10, exec_lo
        0x7E06_0300, // v_mov_b32 v3, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        // loop:
        0xBE90_3C6A, // s_and_saveexec_b32 s16, vcc_lo
        0xBF88_0008, // s_cbranch_execz skip
        0xBE94_0304, // s_mov_b32 s20, s4
        0x8914_6A14, // s_xor_b32 s20, s20, vcc_lo
        0x896A_6A14, // s_xor_b32 vcc_lo, s20, vcc_lo
        0x8914_6A14, // s_xor_b32 s20, s20, vcc_lo
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0x896A_146A, // s_xor_b32 vcc_lo, vcc_lo, s20
        0x8914_146A, // s_xor_b32 s20, vcc_lo, s20
        0x896A_146A, // s_xor_b32 vcc_lo, vcc_lo, s20
        // skip:
        0xBEFE_0310, // s_mov_b32 exec_lo, s16
        0x4A06_0688, // v_add_nc_u32 v3, 8, v3
        0x7DA8_06A0, // v_cmpx_gt_u32 32, v3
        0xBF89_FFF2, // s_cbranch_execnz loop
        0xBEFE_030A, // s_mov_b32 exec_lo, s10
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    assert_eq!(refusal32(&w), None);
    let got = run32(&w, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..8], [4; 8]);
    assert_eq!(got[8..16], [3; 8]);
    assert_eq!(got[16..20], [2; 4]);
    assert_eq!(got[20..], [0; 12]);
}

/// A 32-lane program that swaps a mask into `s8` by three `s_xor_b64` of SGPR pairs (the tile
/// classifiers swap their loop masks out of a waterfall loop so) on lanes 0..19, and joins it
/// with the address the lanes that skip bring in `s8`; the join is ANDed with EXEC before EXEC
/// is restored, so only the swapped edge's lanes need a bit. `last` is the third XOR.
fn swapped_mask(last: u32) -> [u32; 17] {
    [
        0xBE88_0302, // s_mov_b32 s8, s2
        0xBE89_0303, // s_mov_b32 s9, s3
        0x7D88_0094, // v_cmp_gt_u32 vcc_lo, 20, v0
        0xBE84_3C6A, // s_and_saveexec_b32 s4, vcc_lo
        skip_execz(5),
        0xD4C4_0000, 0x0002_008A, // v_cmp_gt_u32 s0, 10, v0
        0x8988_0008, // s_xor_b64 s[8:9], s[8:9], s[0:1]
        0x8980_0008, // s_xor_b64 s[0:1], s[8:9], s[0:1]
        last,
        0x8708_7E08, // join: s_and_b32 s8, s8, exec_lo
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xD501_0002, 0x0021_0280, // v_cndmask_b32 v2, 0, 1, s8
        V_LSHLREV_V1_2_V0,
        GLOBAL_STORE_V2_AT_V1[0], GLOBAL_STORE_V2_AT_V1[1],
    ]
}

#[test]
fn a_mask_swapped_into_a_pair_by_64_bit_xors_in_a_32_lane_wave_is_that_mask_at_a_join() {
    // The swap's last XOR gives the mask back bit for bit, so the join holds lanes 0..9 on the
    // edge that swapped it (was refused: "data, at a join of several edges").
    let mut w = swapped_mask(0x8988_0008).to_vec(); // s_xor_b64 s[8:9], s[8:9], s[0:1]
    w.push(S_ENDPGM);
    assert_eq!(refusal32(&w), None);
    // Look-alike: a third XOR with another pair is no swap; `s8` is the mask XORed with data.
    // Refused in a group of two waves (a group of one runs in lockstep, as the console does).
    let mut other = swapped_mask(0x8988_0208).to_vec(); // s_xor_b64 s[8:9], s[8:9], s[2:3]
    other.push(S_ENDPGM);
    let why = try_translate_wave(&other, 5, 0, 64, false, shaders::options()).err().map(|e| e.to_string());
    assert!(why.as_ref().is_some_and(|r| r.contains("at a join of several edges")), "{why:?}");
    let got = run32(&w, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..10], [1; 10]);
    assert_eq!(got[10..], [0; 22]);
}

#[test]
fn mask_number_bfm_brev_bitset_wqm_saveexec_and_wrexec_run_on_the_gpu() {
    // 1. `s_bfm_b64` with a dynamic width/offset (`((1 << 10) - 1) << 30 = 0xFF_C000_0000`,
    //    lanes `30..39` across both 32-bit halves) written to `EXEC` and read back as data.
    let bfm = [
        0xBE86_047Eu32, // s_mov_b64 s[6:7], exec
        0x7E10_028A,    // v_mov_b32 v8, 10
        0x7E10_0508,    // v_readfirstlane_b32 s8, v8 (width = 10, dynamic)
        0x92FE_9E08,    // s_bfm_b64 exec, s8, 30 (offset = 30 -> lanes 30..39)
        0xBE8A_047E,    // s_mov_b64 s[10:11], exec
        0x7E04_0287,    // v_mov_b32 v2, 7
        V_LSHLREV_V1_2_V0,
        0xDC70_8000,
        0x0002_0201, // global_store_dword v1, v2, s[2:3]
        0xBEFE_0481, // s_mov_b64 exec, 1
        0x7E06_020A, // v_mov_b32 v3, s10
        0x7E08_020B, // v_mov_b32 v4, s11
        0xDC74_8100,
        0x0002_0301, // global_store_dwordx2 v1, v[3:4], s[2:3] offset:256
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        S_ENDPGM,
    ];
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs_bfm = translate_io(&bfm, 5, 0, 64, shaders::options());
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs_bfm, &[0, 0, c, d, 0], false);
    let got_bfm = f.get(DST, 66);
    for (i, &v) in got_bfm[..64].iter().enumerate() {
        assert_eq!(v, if (30..40).contains(&i) { 7 } else { CANARY }, "bfm lane {i}");
    }
    assert_eq!(&got_bfm[64..66], &[0xC000_0000, 0x0000_00FF], "bfm mask words");

    // 2. `s_brev_b64`: `v_cmp_gt_u32 vcc, 5, v0` sets bits `0..4` (`0x1F`), so `s_brev_b64` under a
    //    narrowed `EXEC` (forcing `mask_number`) reverses into bits `59..63` (`lo = 0, hi = 0xF800_0000`).
    let brev = [
        0x7D88_0085u32, // v_cmp_gt_u32 vcc, 5, v0 (bits 0..4)
        0xBE86_047E,    // s_mov_b64 s[6:7], exec
        0xBEFE_0481,    // s_mov_b64 exec, 1 (narrow EXEC so vcc holds lanes outside EXEC)
        0xBE88_0C6A,    // s_brev_b64 s[8:9], vcc
        0x7E06_0208,    // v_mov_b32 v3, s8
        0x7E08_0209,    // v_mov_b32 v4, s9
        V_LSHLREV_V1_2_V0,
        0xDC74_8000,
        0x0002_0301, // global_store_dwordx2 v1, v[3:4], s[2:3]
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        S_ENDPGM,
    ];
    let cs_brev = translate_io(&brev, 5, 0, 64, shaders::options());
    dispatch(&mut t, &cs_brev, &[0, 0, c, d, 0], false);
    assert_eq!(f.get(DST, 2), [0x0000_0000, 0xF800_0000], "s_brev_b64 words");

    // 3. `s_bitset1_b64` and `s_bitset0_b64`: start with lanes `0..2` (`0x7`), clear bit 1 (`0x5`),
    //    and set bit 35 (`hi = 0x8`).
    let bitset = [
        0x7D88_0083u32, // v_cmp_gt_u32 vcc, 3, v0 (bits 0..2 = 0x7)
        0xBE86_047E,    // s_mov_b64 s[6:7], exec
        0xBE88_046A,    // s_mov_b64 s[8:9], vcc
        0xBEFE_0481,    // s_mov_b64 exec, 1
        0xBE88_1C81,    // s_bitset0_b64 s[8:9], 1
        0xBE88_1EA3,    // s_bitset1_b64 s[8:9], 35
        0x7E06_0208,    // v_mov_b32 v3, s8
        0x7E08_0209,    // v_mov_b32 v4, s9
        V_LSHLREV_V1_2_V0,
        0xDC74_8000,
        0x0002_0301, // global_store_dwordx2 v1, v[3:4], s[2:3]
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        S_ENDPGM,
    ];
    let cs_bitset = translate_io(&bitset, 5, 0, 64, shaders::options());
    dispatch(&mut t, &cs_bitset, &[0, 0, c, d, 0], false);
    assert_eq!(f.get(DST, 2), [0x0000_0005, 0x0000_0008], "s_bitset0/1_b64 words");

    // 4. `s_wqm_b64`: only lane 37 (`quad 9`, lanes `36..39`, `hi = 0x0000_00F0`) is set in `vcc`.
    //    Verifies `mask_dword` routes `Atom::Quad` through `mask_number` instead of returning all-1s.
    let wqm = [
        0x7D84_00A5u32, // v_cmp_eq_u32 vcc, 37, v0
        0xBE88_0A6A,    // s_wqm_b64 s[8:9], vcc
        0xBE86_047E,    // s_mov_b64 s[6:7], exec
        0xBEFE_0481,    // s_mov_b64 exec, 1
        0x7E06_0208,    // v_mov_b32 v3, s8
        0x7E08_0209,    // v_mov_b32 v4, s9
        V_LSHLREV_V1_2_V0,
        0xDC74_8000,
        0x0002_0301, // global_store_dwordx2 v1, v[3:4], s[2:3]
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        S_ENDPGM,
    ];
    let cs_wqm = translate_io(&wqm, 5, 0, 64, shaders::options());
    dispatch(&mut t, &cs_wqm, &[0, 0, c, d, 0], false);
    assert_eq!(f.get(DST, 2), [0x0000_0000, 0x0000_00F0], "s_wqm_b64 quad 9 words");

    // 5. `s_orn1_saveexec_b64` and `s_andn1_wrexec_b64`:
    //    - Start with `EXEC = 0x0000_FF00` (lanes `8..15`), `vcc = v0 >= 12` (`lo = 0xFFFF_F000, hi = u32::MAX`).
    //    - `s_orn1_saveexec_b64 s[8:9], vcc`: `s[8:9]` saves old `EXEC = (0x0000_FF00, 0)`, and new
    //      `EXEC = !vcc | old_exec = 0x0000_0FFF | 0x0000_FF00 = (0x0000_FFFF, 0)` (lanes `0..15`).
    //    - Next set `s[10:11] = v0 < 4` (`(0x0000_000F, 0)`) and run `s_andn1_wrexec_b64 s[12:13], s[10:11]`:
    //      both `s[12:13]` and `EXEC` become `!s[10:11] & EXEC = (0x0000_FFF0, 0)` (lanes `4..15`).
    let save_wrexec = [
        0xBE86_047Eu32, // s_mov_b64 s[6:7], exec
        0x7D86_008C,    // v_cmp_le_u32 vcc, 12, v0 (v0 >= 12 -> lo = 0xffff_f000, hi = 0xffff_ffff)
        0xD4C4_000A,
        0x0002_0084, // v_cmp_gt_u32 s[10:11], 4, v0 (v0 < 4 -> lo = 0x0000_000f, hi = 0)
        0x92FE_8888, // s_bfm_b64 exec, 8, 8 (EXEC = 0x0000_ff00)
        0xBE88_386A, // s_orn1_saveexec_b64 s[8:9], vcc
        0xBE8C_390A, // s_andn1_wrexec_b64 s[12:13], s[10:11]
        0x7E04_0299, // v_mov_b32 v2, 25 (active lanes 4..15 store 25)
        V_LSHLREV_V1_2_V0,
        0xDC70_8000,
        0x0002_0201, // global_store_dword v1, v2, s[2:3]
        0xBEFE_0481, // s_mov_b64 exec, 1
        0x7E06_0208, // v_mov_b32 v3, s8
        0x7E08_0209, // v_mov_b32 v4, s9
        0x7E0A_020C, // v_mov_b32 v5, s12
        0x7E0C_020D, // v_mov_b32 v6, s13
        0xDC78_8100,
        0x0002_0301, // global_store_dwordx4 v1, v[3:6], s[2:3] offset:256
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        S_ENDPGM,
    ];
    f.put(DST, &[CANARY; 68]);
    let cs_sw = translate_io(&save_wrexec, 5, 0, 64, shaders::options());
    dispatch(&mut t, &cs_sw, &[0, 0, c, d, 0], false);
    let got_sw = f.get(DST, 68);
    for (i, &v) in got_sw[..64].iter().enumerate() {
        assert_eq!(v, if (4..16).contains(&i) { 25 } else { CANARY }, "save/wrexec lane {i}");
    }
    assert_eq!(
        &got_sw[64..68],
        &[0x0000_FF00, 0, 0x0000_FFF0, 0],
        "saved s[8:9] and wrexec s[12:13]"
    );
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

#[test]
fn wave64_lane63_writelane_and_salu_or_reductions_run_on_the_gpu() {
    // 1. Sum `v0 + 1` across each 64-lane console wave with lower-half `v_readlane_b32 s6, v1, 0`
    //    written into lane 63 only (`v_writelane_b32 v2, s6, 63; v_add_nc_u32 v1, v2, v1`), then
    //    narrow `EXEC` to lane 0 (`s_mov_b64 exec, 1`) and read lane 63 (`v_readlane_b32 s7, v1, 63`).
    //    Tested for 1-wave (64-lane) and 2-wave (128-lane) groups.
    let lane63_sum = [
        0x4A08_0081u32, // v_add_nc_u32 v4, 1, v0
        0xBE88_25C1,    // s_or_saveexec_b64 s[8:9], -1
        0xD501_0001,
        0x0022_0880, // v_cndmask_b32_e64 v1, 0, v4, s[8:9]
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
        0x7E04_0280, // v_mov_b32 v2, 0
        0xD760_0006,
        0x0001_0101, // v_readlane_b32 s6, v1, 0
        0xD761_0002,
        0x0001_7E06, // v_writelane_b32 v2, s6, 63
        0x4A02_0302, // v_add_nc_u32_e32 v1, v2, v1
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        0xBEFE_0481, // s_mov_b64 exec, 1
        0xD760_0007,
        0x0001_7F01, // v_readlane_b32 s7, v1, 63
        0x7E04_0207, // v_mov_b32 v2, s7
        V_LSHLREV_V1_2_V0,
        0xDC70_8000,
        0x0002_0201, // global_store_dword v1, v2, s[2:3]
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        S_ENDPGM,
    ];
    let got64 = run_io(&lane63_sum, 0, 64, 0);
    if got64.is_empty() {
        return;
    }
    assert_eq!(got64[0], 2080, "sum(1..=64)");
    assert_eq!(&got64[1..64], &[CANARY; 63]);

    let got128 = run_io(&lane63_sum, 0, 128, 0);
    assert_eq!(got128[0], 2080, "wave 0 sum(1..=64)");
    assert_eq!(&got128[1..64], &[CANARY; 63]);
    assert_eq!(got128[64], 6176, "wave 1 sum(65..=128)");
    assert_eq!(&got128[65..128], &[CANARY; 63]);

    // 2. Divergent `EXEC` (`v0 % 4 == 0` active) bitwise-OR reduction combining `v_readlane_b32` of
    //    lanes 0 and 32 via `s_or_b32 s6, s4, s5` inside the whole-wave section:
    //    Active lanes in wave 0 (`0, 4, ..., 60`) set bits `0..7` in the lower half and `8..15` in
    //    the upper half (`0x0000_FFFF`), while wave 1 (`64, 68, ..., 124`) sets bits `16..23` and
    //    `24..31` (`0xFFFF_0000`).
    let salu_or = [
        0x360A_0083u32, // v_and_b32 v5, 3, v0
        0x7DA4_0A80,    // v_cmpx_eq_u32 0, v5 (only lanes `v0 % 4 == 0` stay active)
        0x2C08_0082,    // v_lshrrev_b32 v4, 2, v0 (`0..15` in wave 0, `16..31` in wave 1)
        0x7E0A_0281,    // v_mov_b32 v5, 1
        0x3408_0B04,    // v_lshlrev_b32 v4, v4, v5 (`1 << (v0 / 4)`)
        0xBE88_25C1,    // s_or_saveexec_b64 s[8:9], -1
        0xD501_0001,
        0x0022_0880, // v_cndmask_b32_e64 v1, 0, v4, s[8:9]
        0x3802_02FA,
        0xFF00_B101, // v_or_b32_dpp v1, v1, v1 quad_perm:[1,0,3,2]
        0x3802_02FA,
        0xFF00_4E01, // v_or_b32_dpp v1, v1, v1 quad_perm:[2,3,0,1]
        0x3802_02FA,
        0xFF01_4101, // v_or_b32_dpp v1, v1, v1 row_half_mirror
        0x3802_02FA,
        0xFF01_4001, // v_or_b32_dpp v1, v1, v1 row_mirror
        0xD778_0002,
        0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x3802_0501, // v_or_b32_e32 v1, v1, v2
        0xD760_0006,
        0x0001_0101, // v_readlane_b32 s6, v1, 0
        0xD760_0007,
        0x0001_4101, // v_readlane_b32 s7, v1, 32
        0x880A_0706, // s_or_b32 s10, s6, s7
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        0x7E04_020A, // v_mov_b32 v2, s10
        V_LSHLREV_V1_2_V0,
        0xDC70_8000,
        0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    let or64 = run_io(&salu_or, 0, 64, 0);
    for (i, &got) in or64.iter().enumerate() {
        let want = if i % 4 == 0 { 0x0000_FFFF } else { CANARY };
        assert_eq!(got, want, "64-lane salu_or lane {i}");
    }
    let or128 = run_io(&salu_or, 0, 128, 0);
    for (i, &got) in or128.iter().enumerate() {
        let want = if i % 4 != 0 {
            CANARY
        } else if i < 64 {
            0x0000_FFFF
        } else {
            0xFFFF_0000
        };
        assert_eq!(got, want, "128-lane salu_or lane {i}");
    }

    // 3. Single-wave (64-lane) compute programs that previously failed `emit_fast` fell back to
    //    `lockstep::retry`; verify `emit_fast` produces the exact same GPU output as `lockstep`
    //    (forced by appending a `v_readlane_b32` behind an `EXEC` branch that `emit_fast` refuses).
    for (fast_out, src) in [(&got64, &lane63_sum[..]), (&or64, &salu_or[..])] {
        let mut lockstep_words = src[..src.len() - 1].to_vec();
        lockstep_words.extend([
            0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
            0xBE90_246A, // s_and_saveexec_b64 s[16:17], vcc
            0xBF88_0005, // s_cbranch_execz +5
            0xD760_0012,
            0x0001_4700, // v_readlane_b32 s18, v0, 35 (behind EXEC branch -> forces lockstep)
            0x7E0C_0212, // v_mov_b32 v6, s18
            0xDC70_8200,
            0x0002_0601, // global_store_dword v1, v6, s[2:3] offset:512
            0x88FE_107E, // s_or_b64 exec, exec, s[16:17]
            S_ENDPGM,
        ]);
        assert!(
            refusal(&lockstep_words).is_some(),
            "readlane behind EXEC branch must refuse emit_fast"
        );
        assert_eq!(&run_io(&lockstep_words, 0, 64, 0), fast_out);
    }
}

#[test]
fn a_mask_joined_with_data_on_an_exec_skips_edge_is_its_lanes_in_lockstep() {
    // Lanes below the flag enter a region that `s_cbranch_execz` skips when none do; `s[10:11]` is
    // a mask in it (the lanes below 8) and data on the skip's edge (5, and a mask's high dword),
    // read at the join only ANDed with EXEC. A lane read makes it run in lockstep, where the skip
    // is `s_cmp_eq_u64 exec, 0; s_cbranch_scc1` and its edge brings an empty EXEC: the data is not
    // needed (was refused: "data whose high dword is not known").
    let w = [
        0xD4C4_000A, 0x0202_0090, // v_cmp_gt_u32 s[10:11], 16, v0
        0xBE8A_0385, // s_mov_b32 s10, 5
        V_LSHLREV_V1_2_V0,
        0x4A06_00FF, 0x0000_03E8, // v_add_nc_u32 v3, 1000, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D88_0004, // v_cmp_gt_u32 vcc, s4, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        skip_execz(5),
        0xD760_0008, 0x0201_4703, // v_readlane_b32 s8, v3, 35
        0x7E04_0208, // v_mov_b32 v2, s8
        0xD4C4_000A, 0x0202_0088, // v_cmp_gt_u32 s[10:11], 8, v0
        0x878C_7E0A, // join: s_and_b64 s[12:13], s[10:11], exec
        0x88FE_067E, // s_or_b64 exec, exec, s[6:7]
        0x87FE_7E0C, // s_and_b64 exec, s[12:13], exec
        0x4A04_0487, // v_add_nc_u32 v2, 7, v2
        0x88FE_067E, // s_or_b64 exec, exec, s[6:7]
        GLOBAL_STORE_V2_AT_V1[0], GLOBAL_STORE_V2_AT_V1[1],
        S_ENDPGM,
    ];
    let got = run(&w, 0, 40);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..8], [1042; 8]);
    assert_eq!(got[8..40], [1035; 32]);
    assert_eq!(got[40..], [0; 24]);
    // No lane enters: the skip's edge is the one taken.
    assert_eq!(run(&w, 0, 0), [0; 64]);
}

#[test]
fn a_mask_relayed_by_a_jump_and_read_only_with_exec_is_its_lanes_in_lockstep() {
    // As above, with a second EXEC skip (on the flag, in s4) inside the region: the join of
    // `s[10:11]` after it is only relayed by the fall-through into the block where it is ANDed
    // with EXEC. Lanes below 40 pass the outer skip, those below the flag the inner one (and read
    // lane 35); of those, the lanes below 8 are in the mask.
    let w = [
        0xD4C4_000A, 0x0202_0090, // v_cmp_gt_u32 s[10:11], 16, v0
        0xBE8A_0385, // s_mov_b32 s10, 5
        V_LSHLREV_V1_2_V0,
        0x4A06_00FF, 0x0000_03E8, // v_add_nc_u32 v3, 1000, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        skip_execz(8), // the outer skip, to the AND
        0x7DA8_0004, // v_cmpx_gt_u32 s4, v0
        skip_execz(5), // the inner skip, to the relay block
        0xD760_0008, 0x0201_4703, // v_readlane_b32 s8, v3, 35
        0x7E04_0208, // v_mov_b32 v2, s8
        0xD4C4_000A, 0x0202_0088, // v_cmp_gt_u32 s[10:11], 8, v0
        0xBE89_0380, // relay block: s_mov_b32 s9, 0
        0x878C_7E0A, // s_and_b64 s[12:13], s[10:11], exec
        0x88FE_067E, // s_or_b64 exec, exec, s[6:7]
        0x87FE_7E0C, // s_and_b64 exec, s[12:13], exec
        0x4A04_0487, // v_add_nc_u32 v2, 7, v2
        0x88FE_067E, // s_or_b64 exec, exec, s[6:7]
        GLOBAL_STORE_V2_AT_V1[0], GLOBAL_STORE_V2_AT_V1[1],
        S_ENDPGM,
    ];
    let got = run(&w, 0, 40);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..8], [1042; 8]);
    assert_eq!(got[8..40], [1035; 32]);
    assert_eq!(got[40..], [0; 24]);
    // Only lanes below 4 stay in EXEC past the `v_cmpx`; the others add nothing at the join.
    let got = run(&w, 0, 4);
    assert_eq!(got[..4], [1042; 4]);
    assert_eq!(got[4..], [0; 60]);
    // No lane enters at all.
    assert_eq!(run(&w, 0, 0), [0; 64]);
}

#[test]
fn a_loop_register_that_is_exec_on_every_edge_counts_the_lanes_still_looping() {
    // Two nested loops keeping the lanes still looping in an SGPR beside EXEC (a copy of EXEC
    // before the loop, `s_andn2_b64 s, s, exec` at the head, `s_andn2_wrexec_b64 s, s` at the back
    // edge). At the inner head `s[4:5]` is EXEC: each lane adds its bit count there to v6. Lanes
    // 1..39 run the inner loop once (39 looping), lane 6 twice (then alone), lane 10 runs the
    // outer loop twice (alone the second time); lane 0 and lanes 40 and up never get there.
    // Stored: v6 << 16 | v1 (inner iterations) << 8 | v2 (7 per inner loop left).
    let w = [
        0x7E02_0280, // v_mov_b32 v1, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7E0C_0280, // v_mov_b32 v6, 0
        0x7DA8_00A8, // v_cmpx_gt_u32 40, v0
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0xBE8E_0480, // s_mov_b64 s[14:15], 0
        0x7E08_0300, // v_mov_b32 v4, v0
        0x7DA4_0880, // O: v_cmpx_eq_u32 0, v4
        0x8A8A_7E0A, // s_andn2_b64 s[10:11], s[10:11], exec
        0xBF84_0020, // s_cbranch_scc0 OEND
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
        0xBE84_047E, // s_mov_b64 s[4:5], exec
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0xBE88_0480, // s_mov_b64 s[8:9], 0
        0x7E06_0304, // v_mov_b32 v3, v4
        0xBE92_1004, // L: s_bcnt1_i32_b64 s18, s[4:5]
        0x4A0C_0C12, // v_add_nc_u32 v6, s18, v6
        0x7DA4_0680, // v_cmpx_eq_u32 0, v3
        0x8A84_7E04, // s_andn2_b64 s[4:5], s[4:5], exec
        0xBF84_000A, // s_cbranch_scc0 END
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0x4A02_0281, // v_add_nc_u32 v1, 1, v1
        0x4A06_06C1, // v_add_nc_u32 v3, -1, v3
        0xD4C2_000C, 0x0202_0685, // v_cmp_eq_u32 s[12:13], 5, v3
        0xBE8C_080C, // s_not_b64 s[12:13], s[12:13]
        0x87FE_7E0C, // s_and_b64 exec, s[12:13], exec
        0x8888_7E08, // s_or_b64 s[8:9], s[8:9], exec
        0xBE84_3A04, // s_andn2_wrexec_b64 s[4:5], s[4:5]
        0xBF85_FFF1, // s_cbranch_scc1 L
        0xBEFE_0406, // END: s_mov_b64 exec, s[6:7]
        0x87FE_7E08, // s_and_b64 exec, s[8:9], exec
        0x4A04_0487, // v_add_nc_u32 v2, 7, v2
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0x4A08_08C1, // v_add_nc_u32 v4, -1, v4
        0xD4C2_0010, 0x0202_0889, // v_cmp_eq_u32 s[16:17], 9, v4
        0xBE90_0810, // s_not_b64 s[16:17], s[16:17]
        0x87FE_7E10, // s_and_b64 exec, s[16:17], exec
        0x888E_7E0E, // s_or_b64 s[14:15], s[14:15], exec
        0xBE8A_3A0A, // s_andn2_wrexec_b64 s[10:11], s[10:11]
        0xBF85_FFDD, // s_cbranch_scc1 O
        0xBEFE_04C1, // OEND: s_mov_b64 exec, -1
        0x3402_0288, // v_lshlrev_b32 v1, 8, v1
        0x340C_0C90, // v_lshlrev_b32 v6, 16, v6
        0x4A04_0501, // v_add_nc_u32 v2, v1, v2
        0x4A04_0506, // v_add_nc_u32 v2, v6, v2
        V_LSHLREV_V1_2_V0,
        GLOBAL_STORE_V2_AT_V1[0], GLOBAL_STORE_V2_AT_V1[1],
        S_ENDPGM,
    ];
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    let lane = |n: u32| match n {
        0 | 40.. => 0,
        6 => 40 << 16 | 2 << 8 | 7,
        10 => 40 << 16 | 2 << 8 | 14,
        _ => 39 << 16 | 1 << 8 | 7,
    };
    assert_eq!(got, (0..64).map(lane).collect::<Vec<_>>());
}

/// A wave maximum the compiler finishes only in the upper half (lanes 32..63 take lane 0's half
/// maximum), read at lane 63 by a later whole-wave section that shifts it with a scalar step
/// inside the section, and again after it. Lanes 0..39 are on: every read is 39, the shift 19.
fn max_read_at_lane(lane_after: u32) -> Vec<u32> {
    vec![
        0x7D88_00A8, // v_cmp_gt_u32_e32 vcc, 40, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
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
        0xD760_0010, 0x0200_0101 | (128 + lane_after) << 9, // v_readlane_b32 s16, v1, lane_after
        0x8F10_8810, // s_lshl_b32 s16, s16, 8
        0x880F_100F, // s_or_b32 s15, s15, s16
        0x7E06_020F, // v_mov_b32_e32 v3, s15
        V_LSHLREV_V1_2_V0,
        GLOBAL_STORE_V2_AT_V1[0],
        0x0002_0301, // global_store_dword v1, v3, s[2:3]
        S_ENDPGM,
    ]
}

#[test]
fn a_wave_maximum_read_at_its_last_lane_after_its_section_is_the_maximum() {
    let w = max_read_at_lane(63);
    assert_eq!(refusal(&w), None);
    let got = run(&w, 0, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..40], [19 | 39 << 8; 40]);
    assert_eq!(got[40..64], [CANARY; 24]);
}

/// Lane 0 holds only the lower half's maximum: no wave operation gives it, so it stays refused.
#[test]
fn a_lane_holding_half_a_wave_maximum_is_refused() {
    let why = refusal(&max_read_at_lane(0));
    assert!(why.as_deref().is_some_and(|w| w.contains("no reduction or prefix")), "{why:?}");
}

/// A wave32 prefix sum as the title's compute programs end it: an inclusive sum in each row of 16
/// lanes (`row_shr` 1, 2, 4, 8), then the upper row alone, under `s_bfm_b32 exec_lo, 16, 16`,
/// adds lane 15 through `v_permlanex16_b32`. The `s_bfm` is a constant lane set, as an `s_mov` of
/// 0xffff0000 is; it was refused as "s_bfm_b32 is not supported". Lane `i` stores the sum of
/// `1..=i + 1` over the lanes that were on before the section, the others nothing.
#[test]
fn a_wave32_prefix_sum_whose_upper_row_s_bfm_selects_runs() {
    let section = [
        0x4A08_0081u32, // v_add_nc_u32 v4, 1, v0
        0xBE88_3DC1,    // s_or_saveexec_b32 s8, -1
        0xD501_0001,
        0x0022_0880, // v_cndmask_b32_e64 v1, 0, v4, s8
        0x4A02_02FA,
        0xFF01_1101, // v_add_nc_u32_dpp v1, v1, v1 row_shr:1
        0x4A02_02FA,
        0xFF01_1201, // v_add_nc_u32_dpp v1, v1, v1 row_shr:2
        0x4A02_02FA,
        0xFF01_1401, // v_add_nc_u32_dpp v1, v1, v1 row_shr:4
        0x4A02_02FA,
        0xFF01_1801, // v_add_nc_u32_dpp v1, v1, v1 row_shr:8
        0x927E_9090, // s_bfm_b32 exec_lo, 16, 16
        0xD778_0002,
        0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x4A02_0501, // v_add_nc_u32_e32 v1, v1, v2
        0xBEFE_0308, // s_mov_b32 exec_lo, s8
        0x7E04_0301, // v_mov_b32 v2, v1
        V_LSHLREV_V1_2_V0,
        GLOBAL_STORE_V2_AT_V1[0],
        GLOBAL_STORE_V2_AT_V1[1],
        S_ENDPGM,
    ];
    let sum = |i: u32| (i + 1) * (i + 2) / 2;
    let got = run32(&section, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got, (0..32).map(sum).collect::<Vec<_>>());
    // Lanes 20 and up off before the section: their terms are the identity and they store nothing.
    let mut below_20 = vec![
        0x7D88_0094, // v_cmp_gt_u32 vcc_lo, 20, v0
        0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
    ];
    below_20.extend(section);
    let got = run32(&below_20, 0);
    let want: Vec<u32> = (0..32).map(|i| if i < 20 { sum(i) } else { CANARY }).collect();
    assert_eq!(got, want);
}

/// The wave64 form of the title's prefix sums: an inclusive sum in each row of 16 lanes, then the
/// upper row of each half (`s_bitreplicate_b64_b32 exec, 0xff00ff00`) adds lane 15 of its half
/// through `v_permlanex16_b32`, then the upper half (`s_bfm_b64 exec, 32, 32`) adds lane 31. Lane
/// `i` stores the sum of `1..=i + 1`, across the two host waves of the console wave.
#[test]
fn a_wave64_prefix_sum_whose_rows_s_bitreplicate_and_s_bfm_select_runs() {
    let words = [
        0x4A08_0081u32, // v_add_nc_u32 v4, 1, v0
        0xBE88_25C1,    // s_or_saveexec_b64 s[8:9], -1
        0xD501_0001,
        0x0022_0880, // v_cndmask_b32_e64 v1, 0, v4, s[8:9]
        0x4A02_02FA,
        0xFF01_1101, // v_add_nc_u32_dpp v1, v1, v1 row_shr:1
        0x4A02_02FA,
        0xFF01_1201, // v_add_nc_u32_dpp v1, v1, v1 row_shr:2
        0x4A02_02FA,
        0xFF01_1401, // v_add_nc_u32_dpp v1, v1, v1 row_shr:4
        0x4A02_02FA,
        0xFF01_1801, // v_add_nc_u32_dpp v1, v1, v1 row_shr:8
        0xBEFE_3BFF,
        0xFF00_FF00, // s_bitreplicate_b64_b32 exec, 0xff00ff00
        0xD778_0002,
        0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x4A02_0501, // v_add_nc_u32_e32 v1, v1, v2
        0x92FE_A0A0, // s_bfm_b64 exec, 32, 32
        0xD760_0009,
        0x0001_3F01, // v_readlane_b32 s9, v1, 31
        0x4A02_0209, // v_add_nc_u32_e32 v1, s9, v1
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        0x7E04_0301, // v_mov_b32 v2, v1
        V_LSHLREV_V1_2_V0,
        GLOBAL_STORE_V2_AT_V1[0],
        GLOBAL_STORE_V2_AT_V1[1],
        S_ENDPGM,
    ];
    let got = run_io(&words, 0, 64, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got, (0..64u32).map(|i| (i + 1) * (i + 2) / 2).collect::<Vec<_>>());
}

/// `v_sub_nc_u32` and `v_add_nc_u32` with the clamp bit saturate as unsigned numbers (`usubsat`
/// and `uaddsat` in LLVM's AMDGPU backend): lane `i` stores `i - 5` clamped at 0, and
/// `0xfffffff0 + i` clamped at 0xffffffff. They were refused as "integer output modifiers".
#[test]
fn clamped_unsigned_sums_and_differences_saturate() {
    let program = |op: [u32; 2], v3: [u32; 2]| {
        vec![
            v3[0],
            v3[1], // v_mov_b32 v3, <constant>
            op[0],
            op[1], // v_sub_nc_u32 / v_add_nc_u32 v2, v0, v3 clamp
            V_LSHLREV_V1_2_V0,
            GLOBAL_STORE_V2_AT_V1[0],
            GLOBAL_STORE_V2_AT_V1[1],
            S_ENDPGM,
        ]
    };
    const V_NOP: u32 = 0x7E00_0000;
    let sub = program([0xD526_8002, 0x0002_0700], [0x7E06_0285, V_NOP]); // v3 = 5
    let got = run32(&sub, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got, (0..32u32).map(|i| i.saturating_sub(5)).collect::<Vec<_>>());
    let add = program([0xD525_8002, 0x0002_0700], [0x7E06_02FF, 0xFFFF_FFF0]); // v3 = 0xfffffff0
    let got = run32(&add, 0);
    assert_eq!(got, (0..32u32).map(|i| 0xFFFF_FFF0u32.saturating_add(i)).collect::<Vec<_>>());
}

/// A whole-wave f16 sum of the title's shape (`v_add_f16` steps keeping the high half: row
/// butterflies, `v_permlanex16_b32` of lane 0, then lane 0 added into every lane), read at lane
/// 63, in groups of two console waves, so the fast path splits each into two host waves. Each
/// lane's term is `i & 3` as an f16 with 0x1234 in its high half. Lane 63 of each wave holds the
/// wave's total in its low half (96.0; 60.0 for the first wave with lanes 40 and up off before the
/// section, where the second has none on) and lane 63's high half (0x1234, or the identity's 0
/// when lane 63 is off). It was refused as "a 16-bit wave reduction across the host's waves".
#[test]
fn a_whole_wave_half_sum_reads_across_the_host_waves() {
    let section = [
        0x3608_0083u32, // v_and_b32 v4, 3, v0
        0x7E08_A104,    // v_cvt_f16_u16 v4, v4
        0x3808_08FF,
        0x1234_0000, // v_or_b32 v4, 0x12340000, v4
        0xBE96_25C1, // s_or_saveexec_b64 s[22:23], -1
        0xD501_000C,
        0x005A_0880, // v_cndmask_b32_e64 v12, 0, v4, s[22:23]
        0x6418_18FA,
        0xFF00_B10C, // v_add_f16_dpp v12, v12, v12 quad_perm:[1,0,3,2]
        0x6418_18FA,
        0xFF00_4E0C, // v_add_f16_dpp v12, v12, v12 quad_perm:[2,3,0,1]
        0x6418_18FA,
        0xFF01_410C, // v_add_f16_dpp v12, v12, v12 row_half_mirror
        0x6418_18FA,
        0xFF01_400C, // v_add_f16_dpp v12, v12, v12 row_mirror
        0xD778_000D,
        0x0201_010C, // v_permlanex16_b32 v13, v12, 0, 0
        0x6418_1B0C, // v_add_f16 v12, v12, v13
        0xD760_000D,
        0x0001_010C, // v_readlane_b32 s13, v12, 0
        0x6418_180D, // v_add_f16 v12, s13, v12
        0xBEFE_0416, // s_mov_b64 exec, s[22:23]
        0xD760_000D,
        0x0001_7F0C, // v_readlane_b32 s13, v12, 63
        0x7E04_020D, // v_mov_b32 v2, s13
        V_LSHLREV_V1_2_V0,
        GLOBAL_STORE_V2_AT_V1[0],
        GLOBAL_STORE_V2_AT_V1[1],
        S_ENDPGM,
    ];
    assert_eq!(refusal(&section), None);
    let got = run_io(&section, 0, 128, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got, vec![0x1234_5600; 128], "96.0 under lane 63's high half, in each wave");
    let mut below_40 = LANES_BELOW_40.to_vec();
    below_40.extend(section);
    let got = run_io(&below_40, 0, 128, 0);
    let want: Vec<u32> = (0..128).map(|i| if i < 40 { 0x0000_5380 } else { CANARY }).collect();
    assert_eq!(got, want, "60.0, and lane 63 (off) holds the identity");
}

/// Floats at the edges (NaN, infinity, zero, f16 overflow) come out as RDNA2 gives them: the
/// clamp modifier makes NaN 0, `v_max`/`v_min` of NaN take the number, `v_med3` with NaN takes the
/// least number, 0 x infinity is NaN but 0 for the legacy multiply, and an f16 conversion
/// overflows to infinity, or to the largest f16 toward zero (`v_cvt_pkrtz`). `s4` is a NaN the
/// compiler cannot fold (each result is stored at `dst` + 4 x its index; a 16-bit result keeps the
/// high half the register held before).
#[test]
fn float_edge_cases_come_out_as_the_hardware_gives_them() {
    let w = [
        0x7E14_0280, // v_mov_b32 v10, 0
        0x7E02_0204, // v_mov_b32 v1, s4
        0xD51B_0004, 0x0200_08FF, 0x7F80_0000, // v_and_b32 v4, 0x7f800000, s4
        0xD516_0005, 0x0200_089F, // v_lshrrev_b32 v5, 31, s4
        0xD508_8002, 0x0201_E501, // v_mul_f32_e64 v2, v1, 1.0 clamp
        0xDC70_8000, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:0
        0xD510_0002, 0x0201_E101, // v_max_f32 v2, v1, 0.5
        0xDC70_8004, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:4
        0xD50F_0002, 0x0201_E101, // v_min_f32 v2, v1, 0.5
        0xDC70_8008, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:8
        0xD557_0002, 0x03C9_0101, // v_med3_f32 v2, v1, 0, 1.0
        0xDC70_800C, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:12
        0x7E04_5505, // v_rcp_f32 v2, v5
        0xDC70_8010, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:16
        0x7E04_5D05, // v_rsq_f32 v2, v5
        0xDC70_8014, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:20
        0x7E04_66F3, // v_sqrt_f32 v2, -1.0
        0xDC70_8018, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:24
        0x7E04_4F05, // v_log_f32 v2, v5
        0xDC70_801C, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:28
        0xD5A5_0002, 0x2201_0104, // v_exp_f32_e64 v2, -v4
        0xDC70_8020, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:32
        0xD508_8002, 0x0201_E504, // v_mul_f32_e64 v2, v4, 1.0 clamp
        0xDC70_8024, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:36
        0xD503_8002, 0x0201_0101, // v_add_f32_e64 v2, v1, 0 clamp
        0xDC70_8028, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:40
        0xD54B_8002, 0x0201_E501, // v_fma_f32 v2, v1, 1.0, 0 clamp
        0xDC70_802C, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:44
        0xD510_8002, 0x0201_0101, // v_max_f32_e64 v2, v1, 0 clamp
        0xDC70_8030, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:48
        0xD5AA_8002, 0x0201_0101, // v_rcp_f32_e64 v2, v1 clamp
        0xDC70_8034, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:52
        0x5E04_0301, // v_cvt_pkrtz_f16_f32 v2, v1, v1
        0xDC70_8038, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:56
        0xCC21_8002, 0x0201_E501, // v_fma_mixlo_f16 v2, v1, 1.0, 0 clamp
        0xDC70_803C, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:60
        0x100C_0AFF, 0x4974_2400, // v_mul_f32 v6, 0x49742400, v5
        0x060C_0CFF, 0x4974_2400, // v_add_f32 v6, 0x49742400, v6
        0x5E04_0906, // v_cvt_pkrtz_f16_f32 v2, v6, v4
        0xDC70_8040, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:64
        0x7E04_1506, // v_cvt_f16_f32 v2, v6
        0xDC70_8044, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:68
        0xCC21_0002, 0x0201_E506, // v_fma_mixlo_f16 v2, v6, 1.0, 0
        0xDC70_8048, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:72
        0x1004_0B04, // v_mul_f32 v2, v4, v5
        0xDC70_804C, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:76
        0x0E04_0B04, // v_mul_legacy_f32 v2, v4, v5
        0xDC70_8050, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:80
        0x7E04_5504, // v_rcp_f32 v2, v4
        0xDC70_8054, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:84
        0x7E04_5CF3, // v_rsq_f32 v2, -1.0
        0x1004_0D02, // v_mul_f32 v2, v2, v6
        0xDC70_8058, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:88
        0x7E04_6705, // v_sqrt_f32 v2, v5
        0xDC70_805C, 0x0002_020A, // global_store_dword v10, v2, s[2:3] offset:92
        S_ENDPGM,
    ];
    let got = run32(&w, 0x7FC0_0000);
    if got.is_empty() {
        return;
    }
    let want: [u32; 24] = [
        0, 0x3F00_0000, 0x3F00_0000, 0, // clamp(NaN), max(NaN, .5), min(NaN, .5), med3(NaN, 0, 1)
        0x7F80_0000, 0x7F80_0000, 0x7FC0_0000, 0xFF80_0000, // rcp 0, rsq 0, sqrt -1, log 0
        0, 0x3F80_0000, 0, 0, // exp -inf, clamp(inf), clamp(NaN + 0), clamp(fma NaN)
        0, 0, 0x7E00_7E00, 0x7E00_0000, // clamp(max NaN, 0), clamp(rcp NaN), pkrtz NaN, mixlo clamp NaN
        0x7C00_7BFF, 0x7C00_7C00, 0x7C00_7C00, 0x7FC0_0000, // pkrtz 1e6 inf, cvt 1e6, mixlo 1e6, inf x 0
        0, 0, 0x7FC0_0000, 0, // legacy inf x 0, rcp inf, rsq -1 x 1e6, sqrt 0
    ];
    assert_eq!(got[..24], want, "{:x?}", &got[..24]);
}

#[test]
fn a_mask_copied_back_in_a_lockstep_loop_holds_its_lanes_on_every_pass() {
    // As `a_mask_a_divergent_loop_replaces_is_not_taken_as_the_one_it_entered_with`, with the
    // lane read behind the branch that makes a wave64 group of one wave run in lockstep. The
    // latch's VCC is data (the mask of the lanes below 10, or the loop's own on the skip edge)
    // that the header's mask join reads as its number: it was taken as never read and joined as
    // 0, so no lane counted after the first pass.
    let w64 = [
        0xD4C4_0016, 0x0002_008A, // v_cmp_gt_u32 s[22:23], 10, v0
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0x7E06_0300, // v_mov_b32 v3, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        // loop:
        0xBE90_246A, // s_and_saveexec_b64 s[16:17], vcc
        0xBF88_0005, // s_cbranch_execz skip
        0xBE94_046A, // s_mov_b64 s[20:21], vcc
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0xD760_0012, 0x0001_4700, // v_readlane_b32 s18, v0, 35
        0xBEEA_0416, // s_mov_b64 vcc, s[22:23]
        // skip:
        0xBEFE_0410, // s_mov_b64 exec, s[16:17]
        0x4A06_0688, // v_add_nc_u32 v3, 8, v3
        0x7DA8_06C0, // v_cmpx_gt_u32 64, v3
        0xBF89_FFF5, // s_cbranch_execnz loop
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    // The fast path refuses it (a mask joined with data at a join of several edges).
    assert!(refusal(&w64).is_some());
    let got = run_lockstep(&w64, 0, 64);
    if !got.is_empty() {
        assert_eq!(got[..8], [8; 8]);
        assert_eq!(got[8..10], [7; 2]);
        assert_eq!(got[10..20], [1; 10]);
        assert_eq!(got[20..], [0; 44]);
    }
    // The wave32 program in a group of one wave: lockstep, ballots on the one host wave.
    let w32 = [
        0xD4C4_0015, 0x0002_008A, // v_cmp_gt_u32 s21, 10, v0
        0x7D88_0094, // v_cmp_gt_u32 vcc_lo, 20, v0
        0xBE8A_037E, // s_mov_b32 s10, exec_lo
        0x7E06_0300, // v_mov_b32 v3, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        // loop:
        0xBE90_3C6A, // s_and_saveexec_b32 s16, vcc_lo
        0xBF88_0002, // s_cbranch_execz skip
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0xBEEA_0315, // s_mov_b32 vcc_lo, s21
        // skip:
        0xBEFE_0310, // s_mov_b32 exec_lo, s16
        0x4A06_0688, // v_add_nc_u32 v3, 8, v3
        0x7DA8_06A0, // v_cmpx_gt_u32 32, v3
        0xBF89_FFF8, // s_cbranch_execnz loop
        0xBEFE_030A, // s_mov_b32 exec_lo, s10
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    let got = run32(&w32, 0);
    if !got.is_empty() {
        assert_eq!(got[..8], [4; 8]);
        assert_eq!(got[8..10], [3; 2]);
        assert_eq!(got[10..20], [1; 10]);
        assert_eq!(got[20..], [0; 12]);
    }
}

#[test]
fn a_mask_xored_into_a_pairs_high_register_by_a_64_bit_xor_reads_as_its_number() {
    // A wave32 program XORs register pairs as 64-bit values (the deferred-light programs swap
    // and chain masks so): the lanes below 8 in `s9` XORed with the lanes below 24 in `s1`, behind
    // a branch on the lanes below 20, read there as data. The result holds lanes outside that
    // EXEC, so its number comes from the XOR's high dwords (was refused, "written by a scalar
    // instruction of the other width").
    let w = [
        0x7E04_0280, // v_mov_b32 v2, 0
        0xD4C4_0001, 0x0002_0098, // v_cmp_gt_u32 s1, 24, v0
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE88_0380, // s_mov_b32 s8, 0
        0xD4C4_0009, 0x0002_0088, // v_cmp_gt_u32 s9, 8, v0
        0x7D88_0094, // v_cmp_gt_u32 vcc_lo, 20, v0
        0xBE84_3C6A, // s_and_saveexec_b32 s4, vcc_lo
        skip_execz(2),
        0x8988_0008, // s_xor_b64 s[8:9], s[8:9], s[0:1]
        0x7E04_0209, // v_mov_b32 v2, s9
        0xBEFE_0304, // join: s_mov_b32 exec_lo, s4
        V_LSHLREV_V1_2_V0,
        GLOBAL_STORE_V2_AT_V1[0], GLOBAL_STORE_V2_AT_V1[1],
        S_ENDPGM,
    ];
    assert_eq!(refusal32(&w), None);
    let got = run32(&w, 0);
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..20], [0x00FF_FF00; 20]);
    assert_eq!(got[20..], [0; 12]);
}
