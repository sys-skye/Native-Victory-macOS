//! A register holding a stale value (here one made from SCC on entry, which the host has no value for) joined with data at a block
//! entry, where the path that brings the stale value also sets a flag mask to 0 that the reading
//! branch ANDs in (`lower.rs`, `dead_on_edge`): the console compiler's form behind most pixel
//! refusals "a descriptor read only as data" and "a descriptor's address or format read as data".
//! Synthetic GFX10 programs of our own (`llvm-mc -arch=amdgcn -mcpu=gfx1030
//! -mattr=+wavefrontsize64`), run on the GPU. `s[2:3]` is `dst`, `s4` a flag. Skipped without
//! the Metal Shader Converter.

use super::addr_tests::{
    CANARY, Fixture, VA_A, dispatch, halves, title, translate_io, try_translate_wave,
};
use crate::native::shaders;

/// Pool offset of `dst`, at `VA_A + 0x4_0000`.
const DST: usize = 0x6_0000;

/// `s20` first holds a value made from SCC on entry. When `s4` is not 0, `s20 = 7` and the flag
/// mask `s[6:7]` is every lane; else `s20` still holds that value and the flag is 0. Then
/// `s20 == 7`, ANDed with the flag, decides whether every lane stores `s20`.
fn program(and_flag: bool) -> Vec<u32> {
    let mut w = vec![
        0x8514_8081, // s_cselect_b32 s20, 1, 0 (SCC on entry: no value the host has)
        0xBF07_8004, // s_cmp_lg_u32 s4, 0
        0xBF84_0003, // s_cbranch_scc0 ELSE
        0xB014_0007, // s_movk_i32 s20, 7
        0x8586_80C1, // s_cselect_b64 s[6:7], -1, 0
        0xBF82_0001, // s_branch JOIN
        0xBE86_0480, // ELSE: s_mov_b64 s[6:7], 0
        0xD4C2_006A, // JOIN: v_cmp_eq_u32_e64 vcc, s20, 7
        0x0201_0E14,
    ];
    if and_flag {
        w.push(0x87EA_066A); // s_and_b64 vcc, vcc, s[6:7]
    }
    w.extend([
        0x8788_7E6A, // s_and_b64 s[8:9], vcc, exec
        0xBF84_0004, // s_cbranch_scc0 END
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7E04_0214, // v_mov_b32 v2, s20
        0xDC70_8000, // global_store_dword v1, v2, s[2:3]
        0x0002_0201,
        0xBF81_0000, // END: s_endpgm
    ]);
    w
}

fn run(words: &[u32], flag: u32) -> Vec<u32> {
    let Some(mut t) = title() else {
        return Vec::new();
    };
    let f = Fixture::new(&t);
    let cs = translate_io(words, 5, 0, 64, shaders::options());
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs, &[0, 0, c, d, flag], false);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    f.get(DST, 64)
}

#[test]
fn a_stale_register_behind_a_flag_mask_of_0_is_never_read() {
    let w = program(true);
    let got = run(&w, 1);
    if got.is_empty() {
        return;
    }
    assert_eq!(got, [7; 64]);
    assert_eq!(run(&w, 0), [CANARY; 64]);
}

/// Without the flag in the branch, the stale value decides it: still refused, naming it.
#[test]
fn a_stale_register_that_decides_a_branch_stays_refused() {
    let why = try_translate_wave(&program(false), 5, 0, 64, true, shaders::options())
        .err()
        .map(|e| e.to_string());
    assert!(
        why.as_deref().is_some_and(|w| w.contains("SCC on entry")),
        "{why:?}"
    );
}
