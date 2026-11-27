//! #321 on Metal: on RDNA2 (GFX10.3) VOP3 0x140 and VOP2 0x006 are `v_fma_legacy_f32` and
//! `v_fmac_legacy_f32`, one rounding with DX9's zero rule, where GFX10.1 had `v_mad_legacy_f32`
//! and `v_mac_legacy_f32` (`llvm-mc -mcpu=gfx1030` and `gfx1010`). Skipped without the Metal
//! Shader Converter.

use super::addr_tests::{Fixture, VA_A, dispatch, halves, title, translate_io};
use crate::native::shaders;

/// Pool offset of `dst`, at `VA_A + 0x4_0000`.
const DST: usize = 0x6_0000;

#[test]
fn the_legacy_multiply_adds_round_once_and_keep_dx9_zeros() {
    // (1 + 2^-12)^2 - 1 is 2^-11 + 2^-24 exactly (0x3a000400); rounding the product first loses
    // the 2^-24 (0x3a000000). Each result goes to its own 64-dword row of `dst`.
    let w = [
        0x7E04_02FF, 0x3F80_0800, // v_mov_b32 v2, 0x3f800800
        0x7E06_02FF, 0x3F80_0800, // v_mov_b32 v3, 0x3f800800
        0x7E08_02F3, // v_mov_b32 v4, -1.0
        0xD54B_0005, 0x0412_0702, // v_fma_f32 v5, v2, v3, v4
        0xD540_0006, 0x0412_0702, // v_fma_legacy_f32 v6, v2, v3, v4
        0x7E0E_0304, // v_mov_b32 v7, v4
        0x0C0E_0702, // v_fmac_legacy_f32 v7, v2, v3
        0x7E12_02FF, 0x7F80_0000, // v_mov_b32 v9, 0x7f800000
        0x7E14_0280, // v_mov_b32 v10, 0
        0xD540_000B, 0x03CA_130A, // v_fma_legacy_f32 v11, v10, v9, 1.0
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xDC70_8000, 0x0002_0501, // global_store_dword v1, v5, s[2:3]
        0xDC70_8100, 0x0002_0601, // global_store_dword v1, v6, s[2:3] offset:256
        0xDC70_8200, 0x0002_0701, // global_store_dword v1, v7, s[2:3] offset:512
        0xDC70_8300, 0x0002_0B01, // global_store_dword v1, v11, s[2:3] offset:768
        0xBF81_0000, // s_endpgm
    ];
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = translate_io(&w, 5, 0, 64, shaders::options());
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs, &[0, 0, c, d, 0], false);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    let got = f.get(DST, 4 * 64);
    // The plain fused multiply-add, which the legacy ones must match.
    assert_eq!(got[..64], [0x3A00_0400; 64]);
    assert_eq!(got[64..128], [0x3A00_0400; 64], "v_fma_legacy_f32");
    assert_eq!(got[128..192], [0x3A00_0400; 64], "v_fmac_legacy_f32");
    // 0 * infinity is +0 under DX9's rule: 1.0.
    assert_eq!(got[192..], [0x3F80_0000; 64], "v_fma_legacy_f32 of 0 and infinity");
}
