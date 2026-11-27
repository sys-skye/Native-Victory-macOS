//! Pins for the operand-field decode rules of `fields.rs`, on hand-assembled
//! words: the source codes, per-opcode dword counts, the SDWA/DPP forms, and
//! the SMEM/MUBUF/MIMG/DS/VINTRP/EXP field layouts.

use super::{Cache, Dpp, Fields, Instruction, NULL, Operand, Sdwa, instruction};

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn inst(words: &[u32]) -> Instruction {
    instruction(&bytes(words), 0).expect("decodes")
}

fn fields(words: &[u32], wave64: bool) -> Fields {
    inst(words).fields(wave64)
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}

const fn sop2(op: u32, sdst: u32, src1: u32, src0: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}

const fn sopp(op: u32, imm: u16) -> u32 {
    0xBF80_0000 | op << 16 | imm as u32
}

const fn sopk(op: u32, sdst: u32, imm: u16) -> u32 {
    0xB000_0000 | op << 23 | sdst << 16 | imm as u32
}

const fn vop1(op: u32, dst: u32, src0: u32) -> u32 {
    0x7E00_0000 | dst << 17 | op << 9 | src0
}

const fn vop2(op: u32, dst: u32, src1: u32, src0: u32) -> u32 {
    op << 25 | dst << 17 | src1 << 9 | src0
}

const fn vopc(op: u32, src1: u32, src0: u32) -> u32 {
    0x7C00_0000 | op << 17 | src1 << 9 | src0
}

const fn vop3(op: u32, vdst: u32, w0lo: u32, w1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | vdst | w0lo, w1]
}

const fn vop3p(op: u32, vdst: u32, w0lo: u32, w1: u32) -> [u32; 2] {
    [0xCC00_0000 | op << 16 | vdst | w0lo, w1]
}

const fn smem(op: u32, sdata: u32, base_field: u32, w0bits: u32, w1: u32) -> [u32; 2] {
    [
        0xF400_0000 | op << 18 | w0bits | sdata << 6 | base_field,
        w1,
    ]
}

const fn mubuf(op: u32, w0lo: u32, w1: u32) -> [u32; 2] {
    [0xE000_0000 | op << 18 | w0lo, w1]
}

const fn mimg(op: u32, w0lo: u32, w1: u32) -> [u32; 2] {
    [0xF000_0000 | op << 18 | w0lo, w1]
}

const fn ds(op: u32, gds: bool, off1: u32, off0: u32, w1: u32) -> [u32; 2] {
    [
        0xD800_0000 | op << 18 | (gds as u32) << 17 | off1 << 8 | off0,
        w1,
    ]
}

const fn vintrp(op: u32, dst: u32, index: u32, chan: u32, vsrc: u32) -> u32 {
    0xC800_0000 | op << 16 | dst << 18 | index << 10 | chan << 8 | vsrc
}

const fn exp(target: u32, en: u32, compr: bool, done: bool, vm: bool, w1: u32) -> [u32; 2] {
    [
        0xF800_0000
            | (vm as u32) << 12
            | (done as u32) << 11
            | (compr as u32) << 10
            | target << 4
            | en,
        w1,
    ]
}

const S: fn(u8, u8) -> Operand = |code, dwords| Operand::Scalar { code, dwords };
const V: fn(u8, u8) -> Operand = |reg, dwords| Operand::Vector { reg, dwords };

#[test]
fn scalar_source_codes() {
    // 0-105 s0-s105, 106/107 vcc, 108-123 ttmp, 124 m0, 125 null, 126/127 exec.
    for (code, want) in [
        (0u32, S(0, 1)),
        (105u32, S(105, 1)),
        (106u32, S(106, 1)),
        (108u32, S(108, 1)),
        (124u32, S(124, 1)),
        (u32::from(NULL), S(NULL, 1)),
        (126u32, S(126, 1)),
        (127u32, S(127, 1)),
    ] {
        let Fields::Sop { src, .. } = fields(&[sop1(3, 6, code)], false) else {
            panic!("not a Sop")
        };
        assert_eq!(src, [want], "source code {code}");
    }
}

#[test]
fn inline_constants_and_the_literal() {
    // 128-192: inline 0-64, 193-208: -1 to -16, 240-248: float constants,
    // 251-254: vccz, execz, scc, lds_direct; 255 is the following literal.
    for (code, want) in [
        (128u32, Operand::Const(128)),
        (133u32, Operand::Const(133)),
        (192u32, Operand::Const(192)),
        (193u32, Operand::Const(193)),
        (208u32, Operand::Const(208)),
        (240u32, Operand::Const(240)),
        (248u32, Operand::Const(248)),
        (251u32, Operand::Const(251)),
        (252u32, Operand::Const(252)),
        (253u32, Operand::Const(253)),
        (254u32, Operand::Const(254)),
    ] {
        let Fields::Sop { src, .. } = fields(&[sop1(3, 6, code)], false) else {
            panic!("not a Sop")
        };
        assert_eq!(src, [want], "source code {code}");
    }
    let Fields::Sop { src, .. } = fields(&[sop1(3, 0, 0xFF), 0x3F80_0000], false) else {
        panic!("not a Sop")
    };
    assert_eq!(src, [Operand::Literal(0x3F80_0000)]);
}

#[test]
fn scalar_dwords_follow_the_type_suffix() {
    let Fields::Sop { dst, src, imm } = fields(&[sop1(4, 6, 8)], false) else {
        panic!("not a Sop")
    };
    assert_eq!(dst, Some(S(6, 2))); // s_mov_b64
    assert_eq!(src, [S(8, 2)]);
    assert_eq!(imm, None);
    let Fields::Sop { dst, src, .. } = fields(&[sop2(0x0E, 2, 4, 3)], false) else {
        panic!("not a Sop")
    };
    assert_eq!(dst, Some(S(2, 1))); // s_and_b32
    assert_eq!(src, [S(3, 1), S(4, 1)]);
    let Fields::Sop { dst, src, .. } = fields(&[sop2(0x0F, 2, 6, 4)], false) else {
        panic!("not a Sop")
    };
    assert_eq!(dst, Some(S(2, 2))); // s_and_b64
    assert_eq!(src, [S(4, 2), S(6, 2)]);
}

#[test]
fn sopp_and_sopk_immediates() {
    assert_eq!(
        fields(&[sopp(0, 0x7F04)], false),
        Fields::Sopp { imm: 0x7F04 }
    );
    // s_waitcnt_vmcnt 7: no destination, the register as the source, the count as the immediate.
    assert_eq!(
        fields(&[sopk(0x18, 0, 7)], false),
        Fields::Sop {
            dst: None,
            src: vec![S(0, 1)],
            imm: Some(7)
        }
    );
}

#[test]
fn vop2_sources_and_literal_placement() {
    let Fields::Valu(v) = fields(&[vop2(3, 5, 2, 257)], false) else {
        panic!("not a Valu")
    };
    assert_eq!(v.dst, Some(V(5, 1))); // v_add_f32 v5, v1, v2
    assert_eq!(v.src, [V(1, 1), V(2, 1)]);
    // The literal of the madmk form sits between the sources, of madak last.
    let Fields::Valu(v) = fields(&[vop2(0x20, 5, 2, 257), 0x3F80_0000], false) else {
        panic!("not a Valu")
    };
    assert_eq!(v.src, [V(1, 1), Operand::Literal(0x3F80_0000), V(2, 1)]);
    let Fields::Valu(v) = fields(&[vop2(0x21, 5, 2, 257), 0x3F80_0000], false) else {
        panic!("not a Valu")
    };
    assert_eq!(v.src, [V(1, 1), V(2, 1), Operand::Literal(0x3F80_0000)]);
    // v_cndmask_b32 appends the implicit VCC as the last source.
    let Fields::Valu(v) = fields(&[vop2(1, 3, 6, 128)], false) else {
        panic!("not a Valu")
    };
    assert_eq!(v.src, [Operand::Const(128), V(6, 1), S(106, 1)]);
    let Fields::Valu(v) = fields(&[vop1(1, 8, 128)], false) else {
        panic!("not a Valu")
    };
    assert_eq!(v.dst, Some(V(8, 1))); // v_mov_b32 v8, 0
    assert_eq!(v.src, [Operand::Const(128)]);
}

#[test]
fn wave64_sizes_the_lane_masks() {
    // A compare writes VCC (v_cmpx_*: EXEC), one dword on wave32, two on wave64.
    for wave64 in [false, true] {
        let lanes = if wave64 { 2 } else { 1 };
        let Fields::Valu(v) = fields(&[vopc(0xC4, 0, 136)], wave64) else {
            panic!("not a Valu")
        };
        assert_eq!(v.dst, Some(S(106, lanes)), "v_cmp_gt_u32 wave64={wave64}");
        assert_eq!(v.src, [Operand::Const(136), V(0, 1)]);
        let Fields::Valu(v) = fields(&[vopc(0xD4, 0, 136)], wave64) else {
            panic!("not a Valu")
        };
        assert_eq!(v.dst, Some(S(126, lanes)), "v_cmpx_gt_u32 wave64={wave64}");
        let Fields::Valu(v) = fields(&[vop2(1, 3, 6, 128)], wave64) else {
            panic!("not a Valu")
        };
        assert_eq!(
            v.src[2],
            S(106, lanes),
            "v_cndmask_b32's VCC wave64={wave64}"
        );
    }
}

#[test]
fn the_sdwa_source_form() {
    let ext = (1 << 8)
        | (2 << 11)
        | (3 << 16)
        | (1 << 19)
        | (1 << 20)
        | (1 << 21)
        | (1 << 13)
        | (2 << 14);
    let Fields::Valu(v) = fields(&[vop2(3, 5, 2, 0xF9), ext], false) else {
        panic!("not a Valu")
    };
    assert_eq!(v.src, [V(0, 1), V(2, 1)]);
    assert_eq!(v.neg, 1);
    assert_eq!(v.abs, 1);
    assert!(v.clamp);
    assert_eq!(v.omod, 2);
    assert_eq!(
        v.sdwa,
        Some(Sdwa {
            dst_sel: 1,
            dst_unused: 2,
            src_sel: [3, 0],
            sext: 1
        })
    );
}

#[test]
fn the_dpp_source_forms() {
    // DPP16: the 9-bit ctrl, the row and bank masks, bound_ctrl and fi.
    let ext = 3 | (0x1B << 8) | (1 << 18) | (1 << 19) | (0xF << 24) | (0xF << 28) | (1 << 20);
    let Fields::Valu(v) = fields(&[vop2(3, 5, 2, 0xFA), ext], false) else {
        panic!("not a Valu")
    };
    assert_eq!(v.src, [V(3, 1), V(2, 1)]);
    assert_eq!(v.neg, 1);
    assert_eq!(
        v.dpp,
        Some(Dpp::Row {
            ctrl: 0x1B,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: true
        })
    );
    // DPP8: the eight 3-bit selects; 0xEA is the fetch-inactive form.
    for (src0, fi) in [(0xE9, false), (0xEA, true)] {
        let Fields::Valu(v) = fields(&[vop2(3, 5, 2, src0), 3 | (0xB4 << 8)], false) else {
            panic!("not a Valu")
        };
        assert_eq!(v.src, [V(3, 1), V(2, 1)]);
        assert_eq!(v.dpp, Some(Dpp::Lanes { sel: 0xB4, fi }));
    }
}

#[test]
fn vop3_source_counts_and_carry() {
    let [w0, w1] = vop3(0x14B, 0, 0, 257 | 258 << 9 | 259 << 18);
    let Fields::Valu(v) = fields(&[w0, w1], false) else {
        panic!("not a Valu")
    };
    assert_eq!(v.dst, Some(V(0, 1))); // v_fma_f32: three sources
    assert_eq!(v.src, [V(1, 1), V(2, 1), V(3, 1)]);
    // A two-source VOP3 opcode: v_readlane_b32 also writes an SGPR.
    let [w0, w1] = vop3(0x360, 2, 0, 256 | 259 << 9);
    let Fields::Valu(v) = fields(&[w0, w1], false) else {
        panic!("not a Valu")
    };
    assert_eq!(v.dst, Some(S(2, 1)));
    assert_eq!(v.src, [V(0, 1), V(3, 1)]);
    // A VOP3B opcode carries its result out to an SGPR in bits 14:8 of w0.
    for wave64 in [false, true] {
        let lanes = if wave64 { 2 } else { 1 };
        let [w0, w1] = vop3(0x30F, 0, 0, 257 | 258 << 9);
        let Fields::Valu(v) = fields(&[w0, w1], wave64) else {
            panic!("not a Valu")
        };
        assert_eq!(v.dst, Some(V(0, 1)));
        assert_eq!(v.src, [V(1, 1), V(2, 1)]);
        assert_eq!(v.carry, Some(S(0, lanes)));
    }
}

#[test]
fn vop3p_halves_fields() {
    let [w0, w1] = vop3p(
        0x021,
        4,
        2 << 8 | 1 << 11 | 1 << 13 | 1 << 15,
        257 | 258 << 9 | 259 << 18 | 1 << 27 | 2 << 29,
    );
    let Fields::Valu(v) = fields(&[w0, w1], false) else {
        panic!("not a Valu")
    };
    assert_eq!(v.dst, Some(V(4, 1))); // v_fma_mixlo_f16
    assert_eq!(v.src, [V(1, 1), V(2, 1), V(3, 1)]);
    assert_eq!(v.neg, 2);
    assert_eq!(v.neg_hi, 2);
    assert_eq!(v.opsel, 5);
    assert_eq!(v.opsel_hi, 1);
    assert!(v.clamp);
}

#[test]
fn smem_fields() {
    // s_buffer_load_dword: a 4-dword resource base, a sign-extended 12-bit
    // offset, and the cache bits.
    let [w0, w1] = smem(0x008, 0, 1, 1 << 16, 0xFFFF_FFFF | 0x7F << 25);
    let Fields::Smem {
        data,
        base,
        offset,
        soffset,
        cache,
    } = fields(&[w0, w1], false)
    else {
        panic!("not a Smem")
    };
    assert_eq!(data, Some(S(0, 1)));
    assert_eq!(base, S(2, 4));
    assert_eq!(offset, -1);
    assert_eq!(soffset, S(127, 1));
    assert_eq!(
        cache,
        Cache {
            glc: true,
            slc: false,
            dlc: false
        }
    );
    // s_memtime reads two dwords.
    let [w0, w1] = smem(0x024, 6, 0, 0, 0);
    let Fields::Smem { data, base, .. } = fields(&[w0, w1], false) else {
        panic!("not a Smem")
    };
    assert_eq!(data, Some(S(6, 2)));
    assert_eq!(base, S(0, 2));
}

#[test]
fn buffer_fields() {
    // buffer_load_dword v0, v[1:2], s[4:7], s4 offen idxen offset 0x10 glc:
    // offen+idxen makes the address two dwords, the resource four.
    let [w0, w1] = mubuf(
        0x00C,
        0x10 | 1 << 12 | 1 << 13 | 1 << 14,
        1 | 1 << 16 | 4 << 24,
    );
    let Fields::Buffer {
        data,
        addr,
        rsrc,
        soffset,
        offset,
        offen,
        idxen,
        format,
        lds,
        tfe,
        cache,
    } = fields(&[w0, w1], false)
    else {
        panic!("not a Buffer")
    };
    assert_eq!(data, Some(V(0, 1)));
    assert_eq!(addr, V(1, 2));
    assert_eq!(rsrc, S(4, 4));
    assert_eq!(soffset, S(4, 1));
    assert_eq!(offset, 0x10);
    assert!(offen && idxen);
    assert_eq!(format, None);
    assert!(!lds && !tfe);
    assert_eq!(
        cache,
        Cache {
            glc: true,
            slc: false,
            dlc: false
        }
    );
}

#[test]
fn image_fields() {
    // image_sample 2D dmask 0xf: four data dwords, one 2-dword address range,
    // an 8-dword resource, and a sampler (only the sample forms have one).
    let [w0, w1] = mimg(
        0x020,
        1 << 3 | 0xF << 8 | 1 << 12 | 1 << 13,
        1 | 2 << 16 | 3 << 21 | 1 << 22,
    );
    let Fields::Image {
        data,
        addr,
        rsrc,
        sampler,
        dmask,
        dim,
        unorm,
        r128,
        a16,
        d16,
        tfe,
        lwe,
        cache,
    } = fields(&[w0, w1], false)
    else {
        panic!("not an Image")
    };
    assert_eq!(data, V(0, 4));
    assert_eq!(addr, [V(1, 2)]);
    assert_eq!(rsrc, S(8, 8));
    assert_eq!(sampler, Some(S(12, 4)));
    assert_eq!(dmask, 0xF);
    assert_eq!(dim, 1);
    assert!(unorm && !r128 && !a16 && !d16 && !tfe && !lwe);
    assert_eq!(
        cache,
        Cache {
            glc: true,
            slc: false,
            dlc: false
        }
    );
    // image_store takes no sampler; d16 halves the data dwords (2 -> 1).
    let [w0, w1] = mimg(0x008, 1 << 3 | 0x3 << 8, 1 | 5 << 8 | 2 << 16 | 1 << 31);
    let Fields::Image {
        data, sampler, d16, ..
    } = fields(&[w0, w1], false)
    else {
        panic!("not an Image")
    };
    assert_eq!(data, V(5, 1));
    assert_eq!(sampler, None);
    assert!(d16);
}

#[test]
fn ds_fields() {
    // ds_read_b32 v7, v1 offset0=0x12 offset1=0x34: a destination, an address, no data.
    let [w0, w1] = ds(0x036, false, 0x34, 0x12, 1 | 7 << 24);
    let Fields::Ds {
        dst,
        addr,
        data,
        offset0,
        offset1,
        gds,
    } = fields(&[w0, w1], false)
    else {
        panic!("not a Ds")
    };
    assert_eq!(dst, Some(V(7, 1)));
    assert_eq!(addr, Some(V(1, 1)));
    assert!(data.is_empty());
    assert_eq!((offset0, offset1, gds), (0x12, 0x34, false));
    // ds_write_b32 v1, v2 offset 0x40 gds: data, no destination.
    let [w0, w1] = ds(0x00D, true, 0, 0x40, 1 | 2 << 8);
    let Fields::Ds {
        dst,
        addr,
        data,
        gds,
        ..
    } = fields(&[w0, w1], false)
    else {
        panic!("not a Ds")
    };
    assert_eq!(dst, None);
    assert_eq!(addr, Some(V(1, 1)));
    assert_eq!(data, [V(2, 1)]);
    assert!(gds);
    // The GWS group and ds_append take no address; ds_append still returns a dword.
    let [w0, w1] = ds(0x01A, true, 0, 0, 3 << 24);
    let Fields::Ds { dst, addr, .. } = fields(&[w0, w1], false) else {
        panic!("not a Ds")
    };
    assert_eq!((dst, addr), (None, None));
    let [w0, w1] = ds(0x03E, true, 0, 0, 0);
    let Fields::Ds { dst, addr, .. } = fields(&[w0, w1], false) else {
        panic!("not a Ds")
    };
    assert_eq!(dst, Some(V(0, 1)));
    assert_eq!(addr, None);
}

#[test]
fn vintrp_fields() {
    let Fields::Vintrp { dst, src, attr } = fields(&[vintrp(0, 0, 5, 2, 1)], false) else {
        panic!("not a Vintrp")
    };
    assert_eq!(dst, V(0, 1));
    assert_eq!(src, V(1, 1));
    assert_eq!(
        attr,
        Operand::Attr {
            index: 5,
            chan: 2,
            high: false
        }
    );
    // v_interp_mov_f32's source is the parameter as a constant, not a VGPR.
    let Fields::Vintrp { src, .. } = fields(&[vintrp(2, 0, 0, 0, 2)], false) else {
        panic!("not a Vintrp")
    };
    assert_eq!(src, Operand::Const(2));
}

#[test]
fn exp_fields() {
    // The enable bits pick the channels; compr packs two channels per dword.
    let [w0, w1] = exp(0, 0b1010, false, true, false, 1 << 8);
    let Fields::Exp {
        target,
        src,
        compr,
        done,
        vm,
    } = fields(&[w0, w1], false)
    else {
        panic!("not an Exp")
    };
    assert_eq!(target, 0);
    assert_eq!(src, [None, Some(V(1, 1)), None, Some(V(0, 1))]);
    assert!(!compr && done && !vm);
    let [w0, w1] = exp(0, 0b0101, true, true, true, 1 << 16);
    let Fields::Exp {
        src,
        compr,
        done,
        vm,
        ..
    } = fields(&[w0, w1], false)
    else {
        panic!("not an Exp")
    };
    assert_eq!(src, [Some(V(0, 1)), None, Some(V(0, 1)), None]);
    assert!(compr && done && vm);
}
