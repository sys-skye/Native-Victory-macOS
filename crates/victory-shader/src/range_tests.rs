//! Pins for the register-range rules of [`crate::range::build`], on
//! hand-assembled programs: which values bound the index (M0) of a
//! `v_movreld_b32`, shown as the VGPRs the move writes (16 VGPRs, base 2).

use crate::ir::{Program, Reg};
use crate::range;

const S_ENDPGM: u32 = 0xBF81_0000;
const V: u32 = 256;

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

const fn vopc(op: u32, src1: u32, src0: u32) -> u32 {
    0x7C00_0000 | op << 17 | src1 << 9 | src0
}

/// `s_mov_b32 m0, s0`.
const M0_S0: u32 = 0xBEFC_0300;

fn build_words(words: &[u32]) -> Program {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let (p, st) = range::build(&code, false, 16).unwrap();
    st.unwrap();
    p
}

/// The VGPRs the program's `v_movreld_b32` writes.
fn movreld_writes(p: &Program) -> Vec<u8> {
    p.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .find(|i| i.name() == "v_movreld_b32")
        .unwrap()
        .writes
        .iter()
        .filter_map(|(r, _)| match r {
            Reg::V(v) => Some(*v),
            _ => None,
        })
        .collect()
}

const UNBOUNDED: [u8; 14] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// `v_readfirstlane_b32 s0, v<n>; s_mov_b32 m0, s0; v_movreld_b32 v2, v7`,
/// preceded by `prelude`.
fn indexed(prelude: &[u32], n: u32) -> Vec<u8> {
    let mut words = prelude.to_vec();
    words.extend([vop1(0x02, 0, V + n), M0_S0, vop1(0x42, 2, V + 7), S_ENDPGM]);
    movreld_writes(&build_words(&words))
}

#[test]
fn a_program_input_is_unbounded() {
    assert_eq!(indexed(&[], 0), UNBOUNDED);
}

#[test]
fn an_and_with_a_constant_bounds_the_index() {
    // v1 = v0 & 3: the index is at most 3, so the move writes v2-v5.
    assert_eq!(indexed(&[vop2(0x1B, 1, 131, 0)], 1), [2, 3, 4, 5]);
}

#[test]
fn a_copy_of_a_bounded_value_stays_bounded() {
    // v2 = v1 after v1 = v0 & 3.
    assert_eq!(
        indexed(&[vop2(0x1B, 1, 131, 0), vop1(0x01, 2, V + 1)], 2),
        [2, 3, 4, 5]
    );
}

#[test]
fn an_unsigned_min_with_a_constant_bounds_the_index() {
    // v1 = min(v0, 4): at most 4, the move writes v2-v6.
    assert_eq!(indexed(&[vop2(0x13, 1, 132, 0)], 1), [2, 3, 4, 5, 6]);
}

#[test]
fn a_sum_of_bounded_values_bounds_the_index() {
    // v3 = (v0 & 3) + (v0 & 1): at most 3 + 1 = 4 without wrapping.
    assert_eq!(
        indexed(
            &[
                vop2(0x1B, 1, 131, 0),       // v_and_b32 v1, v0, 3
                vop2(0x1B, 2, 129, 0),       // v_and_b32 v2, v0, 1
                vop2(0x25, 3, V + 1, V + 2), // v_add_nc_u32 v3, v1, v2
            ],
            3
        ),
        [2, 3, 4, 5, 6]
    );
}

#[test]
fn a_load_is_unbounded() {
    // v1 = global_load_dword v1, v0, null: whatever memory holds.
    assert_eq!(indexed(&[0xDC30_8000, 0x017D_0001], 1), UNBOUNDED);
}

#[test]
fn a_guarded_write_is_bounded_only_if_the_kept_value_is_too() {
    // Under an EXEC guard the lanes outside it keep v1's old (unbounded) value.
    let guarded = [
        vopc(0xC4, 0, 130),    // v_cmp_gt_u32 vcc, 2, v0
        0xBE82_3C6A,           // s_and_saveexec_b32 s2, vcc_lo
        0xBF88_0002,           // s_cbranch_execz +2
        vop2(0x1B, 1, 131, 0), // v_and_b32 v1, v0, 3
    ];
    assert_eq!(indexed(&guarded, 1), UNBOUNDED);
}

#[test]
fn a_signed_compare_does_not_bound_a_loop_counter() {
    // for (v7 = 0; 4 > v7 (signed); v7++) v[2 + v7] = v0: only unsigned
    // compares bound, so the move's index is unbounded here.
    let p = build_words(&[
        vop1(0x01, 7, 128),   // v_mov_b32 v7, 0
        vop1(0x02, 0, V + 7), // v_readfirstlane_b32 s0, v7
        M0_S0,
        vop1(0x42, 2, V),          // v_movreld_b32 v2, v0
        vop2(0x25, 7, 129, V + 7), // v_add_nc_u32 v7, 1, v7
        vopc(0x84, 7, 132),        // v_cmp_gt_i32 vcc, 4, v7 (signed)
        0xBF87_FFFA,               // s_cbranch_vccnz to the readfirstlane
        S_ENDPGM,
    ]);
    assert_eq!(movreld_writes(&p), UNBOUNDED);
    // The unsigned form of the same loop does bound it: v2-v5.
    let p = build_words(&[
        vop1(0x01, 7, 128),
        vop1(0x02, 0, V + 7),
        M0_S0,
        vop1(0x42, 2, V),
        vop2(0x25, 7, 129, V + 7),
        vopc(0xC4, 7, 132), // v_cmp_gt_u32 vcc, 4, v7
        0xBF87_FFFA,
        S_ENDPGM,
    ]);
    assert_eq!(movreld_writes(&p), [2, 3, 4, 5]);
}
