//! Whole-wave sections (decision 0003): code between `s_or_saveexec -1`
//! (or `s_nand_saveexec 0`) and the restore of the saved mask runs in every lane of the console's
//! wave, the inactive ones included, which is how the compiler expands
//! HLSL wave reductions and prefix sums (an identity put into the inactive
//! lanes with `v_cndmask`, then DPP, `v_permlanex16` and `v_readlane`
//! steps). A host lane that is elsewhere cannot run that code, so each
//! section is simulated on the console's lanes with symbolic values, and a
//! value it leaves is lowered by what it turned out to be, such as a lane
//! read or a reduction or prefix over the lanes' (masked) inputs. No
//! instruction sequence is matched. Any unmodelled step leaves the values
//! it writes unknown.

use std::collections::HashMap;

use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, VCC, Value};

/// An associative operation and its identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    AddU,
    AddF,
    MulF,
    MinU,
    MaxU,
    MinI,
    MaxI,
    MinF,
    MaxF,
    Or,
    And,
    Xor,
    /// `v_add_f16`: on the low halves (see [`Sym::Half`]).
    AddH,
}

impl Op {
    fn of(name: &str) -> Option<Op> {
        Some(match name {
            // The low dwords of a 64-bit sum (the carry is not followed).
            "v_add_nc_u32" | "v_add_co_u32" => Op::AddU,
            "v_add_f32" => Op::AddF,
            "v_mul_f32" => Op::MulF,
            "v_min_u32" => Op::MinU,
            "v_max_u32" => Op::MaxU,
            "v_min_i32" => Op::MinI,
            "v_max_i32" => Op::MaxI,
            "v_min_f32" => Op::MinF,
            "v_max_f32" => Op::MaxF,
            "v_or_b32" => Op::Or,
            "v_and_b32" => Op::And,
            "v_xor_b32" => Op::Xor,
            _ => return None,
        })
    }

    fn salu_of(name: &str) -> Option<Op> {
        Some(match name {
            "s_add_u32" | "s_add_i32" => Op::AddU,
            "s_min_u32" => Op::MinU,
            "s_max_u32" => Op::MaxU,
            "s_min_i32" => Op::MinI,
            "s_max_i32" => Op::MaxI,
            "s_or_b32" => Op::Or,
            "s_and_b32" => Op::And,
            "s_xor_b32" => Op::Xor,
            _ => return None,
        })
    }

    /// Whether `k` leaves the other operand unchanged.
    fn identity(self, k: u32) -> bool {
        match self {
            Op::AddU | Op::Or | Op::Xor | Op::MaxU => k == 0,
            Op::AddF => k == 0 || k == 0x8000_0000,
            Op::MulF => k == 0x3F80_0000,
            Op::MinU | Op::And => k == u32::MAX,
            Op::MinI => k == 0x7FFF_FFFF,
            Op::MaxI => k == 0x8000_0000,
            Op::MinF => k == 0x7F80_0000,
            Op::MaxF => k == 0xFF80_0000,
            Op::AddH => k & 0x7FFF == 0,
        }
    }

    /// Counting a lane twice changes the result.
    fn counts(self) -> bool {
        matches!(self, Op::AddU | Op::AddF | Op::MulF | Op::Xor | Op::AddH)
    }
}

/// A lane's value inside a section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sym {
    Unknown,
    K(u32),
    /// A VGPR value from before the section and the source lane that held it.
    Lane(Value, u32),
    /// A uniform value from before the section.
    Uni(Value),
    /// `op` over `lanes` of `mask ? src : ident` (`op` none: one lane).
    Red {
        op: Option<Op>,
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
    },
    /// A 16-bit operation's result: the low half is `op` over `lanes` of
    /// the low halves of `mask ? src : ident`; the high half is that of
    /// lane `hi`'s `mask ? src : ident` (16-bit results keep the
    /// register's high half).
    Half {
        op: Op,
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
        hi: u32,
    },
}

/// What a value a section leaves holds in every lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    K(u32),
    Value(Value),
    /// Every lane reads the same source lane's VGPR value.
    ReadLane(Value, u32),
    /// The lane's own `mask ? src : ident`.
    Masked(Value, Value, u32),
    /// `op` over every lane.
    Active(Op, Value, Value, u32),
    /// `op` over the lanes below (`inclusive`: and including) each lane.
    Prefix(Op, Value, Value, u32, bool),
    /// [`Sym::Half`] over every lane: `op`, src, mask, identity, the lane
    /// giving the high half.
    Half(Op, Value, Value, u32, u32),
}

/// The section starting at instruction `start` of a block: its restore
/// index and the shapes of the values it writes.
pub(crate) struct Section {
    pub end: usize,
    pub shapes: HashMap<Value, Shape>,
    /// Per-lane shapes `(v, lane)` for VGPR values whose lanes do not all agree (for example a
    /// wave64 reduction whose second half combines into lane 63 only, read by `v_readlane_b32`).
    pub lane_shapes: HashMap<(Value, u32), Shape>,
    /// The first step the simulation does not model, if any: why a value
    /// the section leaves may have no shape.
    pub why: Option<&'static str>,
}

/// A refusal of a value a section leaves: the first step not modelled.
macro_rules! unfollowed {
    ($what:literal) => {
        concat!("a whole-wave section step the simulation does not follow: ", $what)
    };
}
const EXEC_WRITE: &str = unfollowed!("an EXEC write that is no constant");
const SCALAR: &str = unfollowed!("a scalar ALU result");
const NOT_VALU: &str = unfollowed!("an instruction that is no VALU operation");
const READLANE: &str = unfollowed!("a v_readlane at a lane it does not know");
const VALU_SGPR: &str = unfollowed!("a VALU result in an SGPR");
const DPP_LANES_OFF: &str = unfollowed!("a DPP step with lanes off");
const DPP: &str = unfollowed!("a DPP control it does not model");
const MODIFIERS: &str = unfollowed!("an operand or result modifier");
const PERMLANE: &str = unfollowed!("a v_permlane with computed lane selects");
const VALU: &str = unfollowed!("a VALU operation it does not model");
/// Every step modelled, but the value is no single reduction or prefix.
pub(crate) const NO_SHAPE: &str = unfollowed!("a value that is no reduction or prefix");

/// Section lowering can replace register results, but cannot replay effects on
/// lanes the host is not executing. Admit register operations conservatively.
pub(super) fn register_only(inst: &Inst) -> bool {
    match &inst.fields {
        Fields::Valu(v) => v.dst.is_some(),
        Fields::Sop { .. } => {
            matches!(inst.name(), "s_mov_b32" | "s_mov_b64" | "s_and_b32" | "s_and_b64")
                || Op::salu_of(inst.name()).is_some()
                // A constant lane set for EXEC (`simulate` follows it); other `s_bfm`s and
                // `s_bitreplicate`s are data.
                || ((inst.name().starts_with("s_bfm_") || inst.name() == "s_bitreplicate_b64_b32")
                    && inst.write(Reg::S(EXEC)).is_some()
                    && exec_constant(inst).is_some())
        }
        _ => false,
    }
}

/// A scalar ALU step of a whole-wave section that neither reads nor writes EXEC or VCC, so it
/// computes the same as outside the section: lowered as itself there (`Lower::section_inst`),
/// from what the section's earlier steps leave, its result a value of its own to the
/// simulation. The console compiler interleaves such steps with a reduction, as the next
/// reduction's operand: `v_readlane_b32 s64, v110, 31` of one, then `s_lshr_b32 s68, s64, 5`
/// amid the DPP steps of the next (the title's pixel programs).
pub(super) fn passthrough(inst: &Inst) -> bool {
    let lane_mask = |r: &Reg| matches!(*r, Reg::S(n) if n == EXEC || n == EXEC + 1 || n == VCC || n == VCC + 1);
    matches!(inst.fields, Fields::Sop { .. })
        && matches!(
            inst.name(),
            // Not the adds, minimums, maximums and bitwise ops: the simulation combines
            // reduction terms through those (`s_or_b32` of two halves' lane reads).
            "s_lshr_b32" | "s_lshl_b32" | "s_ashr_i32" | "s_bfm_b32" | "s_bfe_u32" | "s_bfe_i32" | "s_sub_u32" | "s_sub_i32" | "s_mul_i32"
        )
        && !inst.reads.iter().any(|(r, _)| lane_mask(r))
        && !inst.writes.iter().any(|(r, _)| lane_mask(r))
}

/// A scalar ALU step inside a section that only writes SGPRs (or SCC) and does
/// not touch EXEC: it runs once for the wave whatever the lanes, so it is
/// lowered as anywhere else. (`s_setreg` writes no register: still refused.)
pub(super) fn scalar_alone(inst: &Inst) -> bool {
    matches!(inst.fields, Fields::Sop { .. })
        && !register_only(inst)
        && !inst.writes.is_empty()
        && [EXEC, EXEC + 1]
            .iter()
            .all(|&r| inst.read(Reg::S(r)).is_none() && inst.write(Reg::S(r)).is_none())
}

/// Every whole-wave section of `block`, by start index.
pub(crate) fn sections(p: &Program, block: usize) -> HashMap<usize, Section> {
    let insts = &p.blocks[block].insts;
    let wave: u32 = if p.wave64 { 64 } else { 32 };
    let mut out = HashMap::new();
    let mut from = 0;
    while let Some((start, end, saved)) = next_section(insts, from) {
        out.insert(start, simulate(&insts[start + 1..end], end, wave, saved));
        from = end + 1;
    }
    out
}

/// Whether any block has a whole-wave section, including one with unmodelled steps.
pub(super) fn has_sections(p: &Program) -> bool {
    p.blocks.iter().any(|b| next_section(&b.insts, 0).is_some())
}

/// The next save and restore indices and the saved EXEC value.
fn next_section(insts: &[Inst], from: usize) -> Option<(usize, usize, Value)> {
    for (i, inst) in insts.iter().enumerate().skip(from) {
        // EXEC becomes every lane: `s_or_saveexec -1` or `s_nand_saveexec 0`.
        let all = match inst.name() {
            "s_or_saveexec_b64" | "s_or_saveexec_b32" => Some(Operand::Const(193)),
            "s_nand_saveexec_b64" | "s_nand_saveexec_b32" => Some(Operand::Const(128)),
            _ => None,
        };
        let start = all.is_some_and(
            |k| matches!(&inst.fields, Fields::Sop { src, .. } if src.first() == Some(&k)),
        );
        let saved = match &inst.fields {
            Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                ..
            } if start => inst.write(Reg::S(*code)),
            _ => None,
        };
        let Some(saved) = saved else {
            continue;
        };
        let end = (i + 1..insts.len()).find(|&j| {
            insts[j].write(Reg::S(EXEC)).is_some()
                && insts[j].read(Reg::S(saved_code(inst))) == Some(saved)
        })?;
        return Some((i, end, saved));
    }
    None
}

fn saved_code(inst: &Inst) -> u8 {
    match &inst.fields {
        Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            ..
        } => *code,
        _ => NULL,
    }
}

/// The lane `l` reads with DPP, whether it reads 0 instead, and whether it
/// writes; `None` for patterns not modelled.
fn dpp_lane(d: Dpp, l: u32) -> Option<(u32, bool, bool)> {
    match d {
        Dpp::Lanes { sel, .. } => Some(((l & !7) | (sel >> (3 * (l & 7)) & 7), false, true)),
        Dpp::Row {
            ctrl,
            row_mask,
            bank_mask,
            bound_ctrl,
            ..
        } => {
            let (row, pos, n) = (l & !15, l & 15, u32::from(ctrl & 15));
            let enabled = row_mask >> (l >> 4 & 3) & 1 != 0 && bank_mask >> (l >> 2 & 3) & 1 != 0;
            let (src, inside) = match ctrl {
                0..=0xFF => ((l & !3) | (u32::from(ctrl) >> (2 * (l & 3)) & 3), true),
                0x101..=0x10F => (l + n, pos + n < 16),
                0x111..=0x11F => (l.wrapping_sub(n), pos >= n),
                0x121..=0x12F => (row | (pos.wrapping_sub(n) & 15), true),
                0x140 => (l ^ 15, true),
                0x141 => (l ^ 7, true),
                0x150..=0x15F => (row | n, true),
                0x160..=0x16F => (l ^ n, true),
                _ => return None,
            };
            match (inside, bound_ctrl) {
                (true, _) => Some((src, false, enabled)),
                (false, true) => Some((l, true, enabled)),
                (false, false) => Some((l, false, false)),
            }
        }
    }
}

fn combine(op: Op, a: Sym, b: Sym) -> Sym {
    match (a, b) {
        (Sym::K(k), x) | (x, Sym::K(k)) if op.identity(k) => x,
        (
            Sym::Red {
                op: oa,
                src,
                mask,
                ident,
                lanes: la,
            },
            Sym::Red {
                op: ob,
                src: sb,
                mask: mb,
                ident: ib,
                lanes: lb,
            },
        ) if (src, mask, ident) == (sb, mb, ib)
            && op.identity(ident)
            && oa.is_none_or(|o| o == op)
            && ob.is_none_or(|o| o == op)
            && (!op.counts() || la & lb == 0) =>
        {
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes: la | lb,
            }
        }
        _ => Sym::Unknown,
    }
}

/// A lane of 16-bit `op` on the low halves of `a` and `b`, the high half
/// kept from the destination's `old` value.
fn half(op: Op, a: Sym, b: Option<Sym>, old: Sym) -> Sym {
    // A low half: the identity, or `op` over lanes of one masked source.
    let low = |s: Sym| match s {
        Sym::K(k) if op.identity(k) => Some(None),
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if op.identity(ident) => Some(Some((src, mask, ident, lanes))),
        Sym::Half {
            op: o,
            src,
            mask,
            ident,
            lanes,
            ..
        } if o == op => Some(Some((src, mask, ident, lanes))),
        _ => None,
    };
    let (src, mask, ident, hi) = match old {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if lanes.is_power_of_two() => (src, mask, ident, lanes.trailing_zeros()),
        Sym::Half {
            src,
            mask,
            ident,
            hi,
            ..
        } => (src, mask, ident, hi),
        _ => return Sym::Unknown,
    };
    let (Some(x), Some(y)) = (low(a), b.and_then(low)) else {
        return Sym::Unknown;
    };
    let mut lanes = 0;
    for (s, m, i, l) in [x, y].into_iter().flatten() {
        if (s, m, i) != (src, mask, ident) || lanes & l != 0 {
            return Sym::Unknown;
        }
        lanes |= l;
    }
    if lanes == 0 {
        return Sym::Unknown;
    }
    Sym::Half {
        op,
        src,
        mask,
        ident,
        lanes,
        hi,
    }
}

#[cfg(test)]
std::thread_local! {
    pub(super) static SIMULATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn simulate(insts: &[Inst], end: usize, wave: u32, saved: Value) -> Section {
    #[cfg(test)]
    SIMULATIONS.set(SIMULATIONS.get() + 1);
    let mut why = None;
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut exec = all;
    let mut vec: HashMap<Value, Vec<Sym>> = HashMap::new();
    let mut uni: HashMap<Value, Sym> = HashMap::new();
    // Masks the section derives from the saved EXEC (`s_and` with it).
    let mut masked = vec![saved];
    for inst in insts {
        let name = inst.name();
        if inst.write(Reg::S(EXEC)).is_some() {
            // Constant lane sets (prefix sums); anything else ends what
            // the simulation can follow.
            let Some(k) = exec_constant(inst) else {
                let (shapes, lane_shapes) = shapes(vec, uni, wave);
                return Section { end, shapes, lane_shapes, why: why.or(Some(EXEC_WRITE)) };
            };
            exec = (if matches!(name, "s_mov_b64" | "s_bfm_b64" | "s_bitreplicate_b64_b32") {
                k
            } else {
                (exec & !0xFFFF_FFFF) | (k & 0xFFFF_FFFF)
            }) & all;
            continue;
        }
        if inst.write(Reg::S(EXEC + 1)).is_some() {
            let Some(k) = exec_constant(inst) else {
                let (shapes, lane_shapes) = shapes(vec, uni, wave);
                return Section { end, shapes, lane_shapes, why: why.or(Some(EXEC_WRITE)) };
            };
            exec = ((exec & 0xFFFF_FFFF) | ((k & 0xFFFF_FFFF) << 32)) & all;
            continue;
        }
        if passthrough(inst) {
            // Lowered as itself: its results are values of their own, uniform as all SGPRs.
            for &(_, w) in &inst.writes {
                uni.insert(w, Sym::Uni(w));
            }
            continue;
        }
        if let Fields::Sop { src, .. } = &inst.fields {
            // A reduction can load its inactive-lane identity into an SGPR.
            if name == "s_mov_b32"
                && let Some(k) = constant(inst)
                && let [(Reg::S(_), dst)] = inst.writes.as_slice()
            {
                uni.insert(*dst, Sym::K(k as u32));
                continue;
            }
            if name.starts_with("s_and_b")
                && src.iter().any(|o| matches!(*o, Operand::Scalar { code, .. } if inst.read(Reg::S(code)).is_some_and(|v| masked.contains(&v))))
                && let Some(&(_, w)) = inst.writes.iter().find(|w| matches!(w.0, Reg::S(_)))
            {
                masked.push(w);
            } else if let Some(op) = Op::salu_of(name) {
                let sym_of = |o: &Operand| match *o {
                    Operand::Scalar { code, .. } if code != NULL => {
                        inst.read(Reg::S(code)).and_then(|x| uni.get(&x).copied())
                    }
                    Operand::Const(c) => crate::resource::inline(c).map(Sym::K),
                    Operand::Literal(x) => Some(Sym::K(x)),
                    _ => None,
                };
                let combined = match (src.first().and_then(sym_of), src.get(1).and_then(sym_of)) {
                    (Some(a), Some(b)) => combine(op, a, b),
                    _ => Sym::Unknown,
                };
                if combined != Sym::Unknown
                    && let Some(&(_, w)) = inst.writes.iter().find(|w| matches!(w.0, Reg::S(_)))
                {
                    uni.insert(w, combined);
                } else {
                    why.get_or_insert(SCALAR);
                }
            } else {
                why.get_or_insert(SCALAR);
            }
            for &(_, w) in &inst.writes {
                uni.entry(w).or_insert(Sym::Unknown);
            }
            continue;
        }
        let Fields::Valu(v) = &inst.fields else {
            why.get_or_insert(NOT_VALU);
            for &(_, w) in &inst.writes {
                vec.insert(w, vec![Sym::Unknown; wave as usize]);
            }
            continue;
        };
        let lanes = |op: &Operand,
                     vec: &HashMap<Value, Vec<Sym>>,
                     uni: &HashMap<Value, Sym>|
         -> Vec<Sym> {
            let one = |s: Sym| vec![s; wave as usize];
            match *op {
                Operand::Vector { reg, .. } => match inst.read(Reg::V(reg)) {
                    Some(x) => vec
                        .get(&x)
                        .cloned()
                        .unwrap_or_else(|| (0..wave).map(|l| Sym::Lane(x, l)).collect()),
                    None => one(Sym::Unknown),
                },
                Operand::Scalar { code, .. } if code != NULL => match inst.read(Reg::S(code)) {
                    Some(x) => one(uni.get(&x).copied().unwrap_or(Sym::Uni(x))),
                    None => one(Sym::Unknown),
                },
                Operand::Const(c) => one(crate::resource::inline(c).map_or(Sym::Unknown, Sym::K)),
                Operand::Literal(x) => one(Sym::K(x)),
                _ => one(Sym::Unknown),
            }
        };
        let modifiers = v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 || v.sdwa.is_some();
        if name == "v_readlane_b32" {
            let (Some(Operand::Scalar { code, .. }), Some(src), Some(lane)) =
                (v.dst, v.src.first(), v.src.get(1))
            else {
                continue;
            };
            let l = match *lane {
                Operand::Const(c) => crate::resource::inline(c),
                Operand::Literal(x) => Some(x),
                _ => None,
            };
            let x = lanes(src, &vec, &uni);
            if let Some(w) = inst.write(Reg::S(code)) {
                let s = l.and_then(|l| x.get(l as usize).copied());
                if s.is_none() {
                    why.get_or_insert(READLANE);
                }
                uni.insert(w, s.unwrap_or(Sym::Unknown));
            }
            continue;
        }
        let Some(Operand::Vector { reg: dreg, .. }) = v.dst else {
            why.get_or_insert(VALU_SGPR);
            for &(_, w) in &inst.writes {
                uni.insert(w, Sym::Unknown);
            }
            continue;
        };
        let Some(dst) = inst.write(Reg::V(dreg)) else {
            continue;
        };
        let old = lanes(
            &Operand::Vector {
                reg: dreg,
                dwords: 1,
            },
            &vec,
            &uni,
        );
        // src0 as each lane fetches it (DPP), with the lanes that write.
        let src0 = v.src.first().map(|o| lanes(o, &vec, &uni));
        let mut writes = exec;
        let fetched: Option<Vec<Sym>> = match (v.dpp, &src0) {
            (Some(d), Some(s)) => {
                if exec != all {
                    why.get_or_insert(DPP_LANES_OFF);
                    vec.insert(dst, vec![Sym::Unknown; wave as usize]);
                    continue;
                }
                let mut f = Vec::with_capacity(wave as usize);
                let mut ok = true;
                for l in 0..wave {
                    match dpp_lane(d, l) {
                        Some((src, zero, on)) if src < wave => {
                            f.push(if zero { Sym::K(0) } else { s[src as usize] });
                            if !on {
                                writes &= !(1u64 << l);
                            }
                        }
                        _ => ok = false,
                    }
                }
                if !ok {
                    why.get_or_insert(DPP);
                }
                ok.then_some(f)
            }
            (None, Some(s)) => Some(s.clone()),
            _ => None,
        };
        let src = |k: usize| v.src.get(k).map(|o| lanes(o, &vec, &uni));
        let result: Vec<Sym> = match (name, fetched) {
            _ if modifiers => {
                why.get_or_insert(MODIFIERS);
                vec![Sym::Unknown; wave as usize]
            }
            ("v_mov_b32", Some(a)) => a,
            ("v_bfrev_b32", Some(a)) => a
                .into_iter()
                .map(|s| match s {
                    Sym::K(k) => Sym::K(k.reverse_bits()),
                    _ => Sym::Unknown,
                })
                .collect(),
            ("v_cndmask_b32", Some(a)) => {
                let mask = match v.src.get(2) {
                    Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                    _ => None,
                };
                let b = src(1).unwrap_or_default();
                a.into_iter()
                    .enumerate()
                    .map(|(l, a)| match (a, b.get(l).copied(), mask) {
                        // This reduction term uses the same lane for the value and mask.
                        (Sym::K(ident), Some(Sym::Lane(x, source)), Some(m))
                            if source as usize == l && masked.contains(&m) =>
                        {
                            Sym::Red {
                                op: None,
                                src: x,
                                mask: m,
                                ident,
                                lanes: 1 << l,
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_permlane16_b32" | "v_permlanex16_b32", Some(a)) => {
                // Lanes of each row read their own (`v_permlane16`) or the other
                // (`v_permlanex16`) row of their half; lane i's pick is nibble i
                // of {src2, src1}.
                let cross = inst.name() == "v_permlanex16_b32";
                let sel = |k: usize| match v.src.get(k) {
                    Some(&Operand::Const(c)) => crate::resource::inline(c),
                    Some(&Operand::Literal(x)) => Some(x),
                    Some(&Operand::Scalar { code, .. }) if code != NULL => {
                        match uni.get(&inst.read(Reg::S(code))?) {
                            Some(&Sym::K(x)) => Some(x),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                match (sel(1), sel(2)) {
                    (Some(s1), Some(s2)) => {
                        let s = u64::from(s1) | u64::from(s2) << 32;
                        (0..wave)
                            .map(|l| {
                                let row = if cross { (l & !15) ^ 16 } else { l & !15 };
                                let pick = (s >> (4 * (l & 15)) & 15) as u32;
                                a[(row | pick) as usize]
                            })
                            .collect()
                    }
                    _ => {
                        why.get_or_insert(PERMLANE);
                        vec![Sym::Unknown; wave as usize]
                    }
                }
            }
            ("v_writelane_b32", Some(a)) => {
                writes = all;
                let lane = match v.src.get(1) {
                    Some(&Operand::Const(c)) => crate::resource::inline(c),
                    Some(&Operand::Literal(x)) => Some(x),
                    Some(&Operand::Scalar { code, .. }) if code != NULL => {
                        inst.read(Reg::S(code)).and_then(|v| match uni.get(&v) {
                            Some(&Sym::K(k)) => Some(k),
                            _ => None,
                        })
                    }
                    _ => None,
                };
                match lane.filter(|&l| l < wave) {
                    Some(target) => (0..wave as usize)
                        .map(|l| if l == target as usize { a[l] } else { old[l] })
                        .collect(),
                    None => {
                        why.get_or_insert(VALU);
                        vec![Sym::Unknown; wave as usize]
                    }
                }
            }
            ("v_sub_nc_u32" | "v_sub_co_u32" | "v_subrev_nc_u32" | "v_subrev_co_u32", Some(a)) => {
                // A prefix sum less the lane's own term: the exclusive one.
                let b = src(1).unwrap_or_default();
                let rev = name.starts_with("v_subrev_");
                (0..wave as usize)
                    .map(|l| {
                        let (lhs, rhs) = if rev {
                            (b.get(l).copied(), Some(a[l]))
                        } else {
                            (Some(a[l]), b.get(l).copied())
                        };
                        match (lhs, rhs) {
                            (
                                Some(Sym::Red {
                                    op: None | Some(Op::AddU),
                                    src,
                                    mask,
                                    ident: 0,
                                    lanes,
                                }),
                                Some(Sym::Red {
                                    op: None | Some(Op::AddU),
                                    src: s2,
                                    mask: m2,
                                    ident: 0,
                                    lanes: l2,
                                }),
                            ) if (src, mask) == (s2, m2) && lanes & l2 == l2 => {
                                let rest = lanes & !l2;
                                if rest == 0 {
                                    Sym::K(0)
                                } else {
                                    Sym::Red {
                                        op: Some(Op::AddU),
                                        src,
                                        mask,
                                        ident: 0,
                                        lanes: rest,
                                    }
                                }
                            }
                            _ => Sym::Unknown,
                        }
                    })
                    .collect()
            }
            ("v_add_f16", Some(a)) if v.opsel == 0 => {
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| half(Op::AddH, a[l], b.get(l).copied(), old[l]))
                    .collect()
            }
            (n, Some(a)) => match (Op::of(n), src(1)) {
                (Some(op), Some(b)) => (0..wave as usize)
                    .map(|l| combine(op, a[l], b[l]))
                    .collect(),
                _ => {
                    why.get_or_insert(VALU);
                    vec![Sym::Unknown; wave as usize]
                }
            },
            _ => {
                why.get_or_insert(VALU);
                vec![Sym::Unknown; wave as usize]
            }
        };
        let merged = (0..wave as usize)
            .map(|l| {
                if writes >> l & 1 != 0 {
                    result[l]
                } else {
                    old[l]
                }
            })
            .collect();
        vec.insert(dst, merged);
        for &(r, w) in &inst.writes {
            if matches!(r, Reg::S(_)) {
                uni.insert(w, Sym::Unknown);
            }
        }
    }
    let (shapes, lane_shapes) = shapes(vec, uni, wave);
    Section { end, shapes, lane_shapes, why }
}

/// The constant lane set an EXEC write gives: a constant `s_mov`, an `s_bfm` of two constant
/// counts (`((1 << S0) - 1) << S1`, RDNA2 ISA SOP2 opcodes 36 and 37, with counts of five and six
/// bits), or an `s_bitreplicate_b64_b32` of a constant (bit `i` to bits `2i` and `2i + 1`, SOP1
/// opcode 59). A wave32 prefix sum selects its upper sixteen lanes with `s_bfm_b32 exec_lo, 16,
/// 16`; a wave64 one the upper row of each half with `s_bitreplicate_b64_b32 exec, 0xff00ff00`,
/// then the upper half with `s_bfm_b64 exec, 32, 32`.
fn exec_constant(inst: &Inst) -> Option<u64> {
    let name = inst.name();
    if name.starts_with("s_mov_b") {
        return constant(inst);
    }
    if name == "s_bitreplicate_b64_b32" {
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        let x = match *src.first()? {
            Operand::Const(c @ 128..=208) => crate::resource::inline(c)?,
            Operand::Literal(x) => x,
            _ => return None,
        };
        return Some((0..32).filter(|i| x >> i & 1 != 0).fold(0, |m, i| m | 3 << (2 * i)));
    }
    let bits = match name {
        "s_bfm_b32" => 31,
        "s_bfm_b64" => 63,
        _ => return None,
    };
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    let count = |o: Option<&Operand>| match *o? {
        Operand::Const(c @ 128..=208) => crate::resource::inline(c).map(|x| u64::from(x) & bits),
        Operand::Literal(x) => Some(u64::from(x) & bits),
        _ => None,
    };
    let (width, offset) = (count(src.first())?, count(src.get(1))?);
    let mask = if bits == 31 { u64::from(u32::MAX) } else { u64::MAX };
    Some((((1u64 << width) - 1) << offset) & mask)
}

fn constant(inst: &Inst) -> Option<u64> {
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    if let ("s_bfm_b32" | "s_bfm_b64", [w, at]) = (inst.name(), src.as_slice()) {
        // A field of constant width and place (`s_bfm_b32 exec_lo, 16, 16`: the upper half's
        // lanes of a wave32 prefix sum).
        let bits = if inst.name() == "s_bfm_b32" { 31 } else { 63 };
        let field = |o: &Operand| match *o {
            Operand::Const(c @ 128..=192) => Some(u32::from(c - 128) & bits),
            Operand::Literal(x) => Some(x & bits),
            _ => None,
        };
        let (w, at) = (field(w)?, field(at)?);
        let m = ((1u64 << w) - 1) << at;
        return Some(if bits == 31 { m & 0xFFFF_FFFF } else { m });
    }
    if !inst.name().starts_with("s_mov_b") {
        return None;
    }
    let wide = inst.name() == "s_mov_b64";
    match *src.first()? {
        Operand::Scalar { code: NULL, .. } => Some(0),
        Operand::Const(c @ 128..=208) => {
            let x = crate::resource::inline(c)?;
            Some(if wide {
                i64::from(x as i32) as u64
            } else {
                u64::from(x)
            })
        }
        Operand::Const(c) if !wide => crate::resource::inline(c).map(u64::from),
        Operand::Literal(x) => Some(u64::from(x)),
        _ => None,
    }
}

/// The shape of each value where every lane agrees on one, plus per-lane
/// uniform shapes `(v, lane)` for VGPR values where only specific lanes hold
/// the finished reduction.
fn shapes(
    vec: HashMap<Value, Vec<Sym>>,
    uni: HashMap<Value, Sym>,
    wave: u32,
) -> (HashMap<Value, Shape>, HashMap<(Value, u32), Shape>) {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut out = HashMap::new();
    let mut lane_out = HashMap::new();
    for (v, s) in uni {
        if let Some(sh) = uniform_shape(s, all) {
            out.insert(v, sh);
        }
    }
    for (v, lanes) in vec {
        if let Some(sh) = per_lane(&lanes, all) {
            out.insert(v, sh);
        } else {
            for (l, &s) in lanes.iter().enumerate() {
                if let Some(sh) = uniform_shape(s, all) {
                    lane_out.insert((v, l as u32), sh);
                }
            }
        }
    }
    (out, lane_out)
}

fn uniform_shape(s: Sym, all: u64) -> Option<Shape> {
    match s {
        Sym::K(k) => Some(Shape::K(k)),
        Sym::Uni(x) => Some(Shape::Value(x)),
        Sym::Lane(x, source) => Some(Shape::ReadLane(x, source)),
        Sym::Red {
            op: Some(op),
            src,
            mask,
            ident,
            lanes,
        } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
        Sym::Half {
            op,
            src,
            mask,
            ident,
            lanes,
            hi,
        } if lanes == all => Some(Shape::Half(op, src, mask, ident, hi)),
        _ => None,
    }
}

fn per_lane(lanes: &[Sym], all: u64) -> Option<Shape> {
    let first = lanes[0];
    if let Sym::Lane(x, 0) = first
        && lanes
            .iter()
            .enumerate()
            .all(|(l, &s)| s == Sym::Lane(x, l as u32))
    {
        return Some(Shape::Value(x));
    }
    let own = |l: usize, s: Sym| match s {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if lanes == 1 << l => Some((src, mask, ident)),
        _ => None,
    };
    if let Some(m) = own(0, first)
        && lanes.iter().enumerate().all(|(l, &s)| own(l, s) == Some(m))
    {
        return Some(Shape::Masked(m.0, m.1, m.2));
    }
    if lanes.iter().all(|&s| s == first) {
        return uniform_shape(first, all);
    }
    // Prefixes: lane l holds lanes 0..l (exclusive) or 0..=l (inclusive).
    let mut meta: Option<(Value, Value, u32, bool)> = None;
    let mut op_found: Option<Op> = None;
    for (l, &s) in lanes.iter().enumerate() {
        let below = (1u64 << l) - 1;
        let (op, src, mask, ident, set) = match s {
            Sym::Red {
                op,
                src,
                mask,
                ident,
                lanes,
            } => (op, src, mask, ident, lanes),
            // Lane 0 of an exclusive prefix holds the identity.
            Sym::K(_) if l == 0 => continue,
            _ => return None,
        };
        let inclusive = set == below | (1u64 << l);
        if !inclusive && set != below {
            return None;
        }
        if l == 0 && !inclusive {
            return None;
        }
        if op.is_none() && set.count_ones() != 1 {
            return None;
        }
        let lane_meta = (src, mask, ident, inclusive);
        match meta {
            None => meta = Some(lane_meta),
            Some(prev) if prev == lane_meta => {}
            _ => return None,
        }
        if let Some(o) = op {
            match op_found {
                None => op_found = Some(o),
                Some(prev) if prev == o => {}
                _ => return None,
            }
        }
    }
    let (src, mask, ident, inclusive) = meta?;
    let op = op_found?;
    if !op.identity(ident) {
        return None;
    }
    match lanes[0] {
        Sym::K(k) if !inclusive && k == ident => {}
        Sym::Red { .. } if inclusive => {}
        _ => return None,
    }
    Some(Shape::Prefix(op, src, mask, ident, inclusive))
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    const fn sop1(sdst: u8, op: u8, ssrc0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | ssrc0 as u32
    }

    const fn vop1(vdst: u8, op: u8, src0: u16) -> u32 {
        (0x3F << 25) | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn vop2(vdst: u8, op: u8, src0: u16, vsrc1: u8) -> u32 {
        (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn dpp16(vsrc0: u8, ctrl: u16, row_mask: u8, bank_mask: u8, bound_ctrl: bool) -> u32 {
        (row_mask as u32) << 28
            | (bank_mask as u32) << 24
            | (bound_ctrl as u32) << 19
            | (ctrl as u32) << 8
            | vsrc0 as u32
    }

    const fn vop3(vdst: u8, op: u16, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0xD400_0000 | (op as u32) << 16 | vdst as u32,
            (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
        ]
    }

    fn build(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words
            .iter()
            .copied()
            .chain(std::iter::once(S_ENDPGM))
            .flat_map(u32::to_le_bytes)
            .collect();
        crate::ir::build(&code, wave64, 256).unwrap()
    }

    #[test]
    fn dpp_lanes_follow_the_isa() {
        // row_shr:1 with bound_ctrl: lane 16 reads 0, lane 17 lane 16.
        let shr = Dpp::Row {
            ctrl: 0x111,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(shr, 16), Some((16, true, true)));
        assert_eq!(dpp_lane(shr, 17), Some((16, false, true)));
        // row_mirror: lane 3 reads lane 12.
        let mirror = Dpp::Row {
            ctrl: 0x140,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(mirror, 3), Some((12, false, true)));
        // dpp8 [1,0,3,2,5,4,7,6] swaps neighbours.
        let swap = Dpp::Lanes {
            sel: 1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21,
            fi: false,
        };
        assert_eq!(dpp_lane(swap, 8), Some((9, false, true)));
    }

    #[test]
    fn prefixes_are_recognised() {
        let (x, m) = (Value(1), Value(2));
        let red = |lanes: u64| Sym::Red {
            op: Some(Op::AddU),
            src: x,
            mask: m,
            ident: 0,
            lanes,
        };
        let inclusive: Vec<Sym> = (0..64).map(|l| red(u64::MAX >> (63 - l))).collect();
        assert_eq!(
            per_lane(&inclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, true))
        );
        let mut exclusive: Vec<Sym> = (0..64).map(|l| red((1u64 << l) - 1)).collect();
        exclusive[0] = Sym::K(0);
        assert_eq!(
            per_lane(&exclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, false))
        );
        assert_eq!(
            per_lane(&vec![red(u64::MAX); 64], u64::MAX),
            Some(Shape::Active(Op::AddU, x, m, 0))
        );
    }

    #[test]
    fn constant_decodes_width_and_inline_kinds() {
        let b64 = 0x04;
        let b32 = 0x03;
        let check = |words: &[u32], wave64: bool| -> Option<u64> {
            let p = build(words, wave64);
            constant(&p.blocks[0].insts[0])
        };

        assert_eq!(
            check(&[sop1(EXEC, b64, 194)], true),
            Some(0xFFFF_FFFF_FFFF_FFFE)
        );
        assert_eq!(
            check(&[sop1(EXEC, b64, 196)], true),
            Some(0xFFFF_FFFF_FFFF_FFFC)
        );
        assert_eq!(
            check(&[sop1(EXEC, b64, 208)], true),
            Some(0xFFFF_FFFF_FFFF_FFF0)
        );
        assert_eq!(check(&[sop1(EXEC, b64, 193)], true), Some(u64::MAX));
        assert_eq!(check(&[sop1(EXEC, b64, NULL)], true), Some(0));
        assert_eq!(check(&[sop1(EXEC, b64, 243)], true), None);
        assert_eq!(
            check(&[sop1(EXEC, b64, 255), 0x1234_5678], true),
            Some(0x1234_5678)
        );

        assert_eq!(check(&[sop1(EXEC, b32, 194)], false), Some(0xFFFF_FFFE));
        assert_eq!(check(&[sop1(EXEC, b32, 243)], false), Some(0xBF80_0000));
        assert_eq!(check(&[sop1(EXEC, b32, NULL)], false), Some(0));
        assert_eq!(
            check(&[sop1(EXEC, b32, 255), 0x1234_5678], false),
            Some(0x1234_5678)
        );
        assert_eq!(check(&[sop1(EXEC, b32, 0)], false), None);

        // s_bfm: a field of constant width and place.
        assert_eq!(check(&[0x927E_9090], false), Some(0xFFFF_0000)); // s_bfm_b32 exec_lo, 16, 16
        assert_eq!(check(&[0x92FE_8488], true), Some(0xFF0)); // s_bfm_b64 exec, 8, 4
        assert_eq!(check(&[0x927E_9F9F], false), Some(0x8000_0000)); // s_bfm_b32 exec_lo, 31, 31
        assert_eq!(check(&[0x927E_0090], false), None); // s_bfm_b32 exec_lo, 16, s0
    }

    #[test]
    fn exec_minus_two_masks_lane_zero_and_enforces_dpp_guard() {
        for wave64 in [false, true] {
            let mut w = Vec::new();
            let save_op = if wave64 { 0x25 } else { 0x3D };
            let mov_exec = if wave64 { 0x04 } else { 0x03 };
            let push_exec = |w: &mut Vec<u32>, k: u8| w.push(sop1(EXEC, mov_exec, k));
            let push_exec_lit = |w: &mut Vec<u32>, lit: u32| {
                w.push(sop1(EXEC, mov_exec, 255));
                w.push(lit);
            };

            w.push(sop1(0, save_op, 193));
            w.extend(vop3(1, 0x101, 128, 256, 0));
            for shift in [1u16, 2, 4, 8] {
                w.push(vop2(1, 0x25, 0xFA, 1));
                w.push(dpp16(1, 0x110 + shift, 0xF, 0xF, false));
            }
            w.extend(vop3(2, 0x360, 257, 128 + 15, 0));
            push_exec_lit(&mut w, 0xFFFF_0000);
            w.push(vop2(1, 0x25, 2, 1));

            if wave64 {
                w.extend(vop3(2, 0x360, 257, 128 + 31, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0x0000_FFFF);
                w.push(vop2(1, 0x25, 2, 1));

                w.extend(vop3(2, 0x360, 257, 128 + 47, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0xFFFF_0000);
                w.push(vop2(1, 0x25, 2, 1));
            }

            push_exec(&mut w, 193);
            w.push(vop1(3, 0x01, 257));
            w.push(vop1(3, 0x01, 0xFA));
            w.push(dpp16(1, 0x111, 0xF, 0xF, false));

            w.extend(vop3(2, 0x360, 257, 128 + 15, 0));
            push_exec_lit(&mut w, 0x0001_0000);
            w.push(vop1(3, 0x01, 2));

            if wave64 {
                w.extend(vop3(2, 0x360, 257, 128 + 31, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 129));
                w.push(vop1(3, 0x01, 2));

                w.extend(vop3(2, 0x360, 257, 128 + 47, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0x0001_0000);
                w.push(vop1(3, 0x01, 2));
            }

            push_exec(&mut w, 193);
            w.push(vop1(2, 0x01, 128));
            w.push(vop1(5, 0x01, 135));
            push_exec(&mut w, 194);
            w.push(vop1(2, 0x01, 259));
            w.push(vop1(4, 0x01, 0xFA));
            w.push(dpp16(5, 0xE4, 0xF, 0xF, false));
            w.push(sop1(EXEC, mov_exec, 0));

            let p = build(&w, wave64);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(0)).unwrap();
            let v0_in = p
                .inputs
                .iter()
                .find_map(|&(r, v)| (r == Reg::V(0)).then_some(v))
                .unwrap();
            let v2_val = insts[insts.len() - 4].write(Reg::V(2)).unwrap();
            let v4_val = insts[insts.len() - 3].write(Reg::V(4)).unwrap();
            let sec = &sections(&p, 0)[&0];
            assert_eq!(
                sec.shapes.get(&v2_val),
                Some(&Shape::Prefix(Op::AddU, v0_in, saved, 0, false))
            );
            assert_eq!(sec.shapes.get(&v4_val), None);
        }
    }

    #[test]
    fn s_bfm_gives_exec_its_field_of_constant_counts() {
        let bfm = |op: u32, s0: u32, s1: u32| 0x8000_0000 | op << 23 | (EXEC as u32) << 16 | s1 << 8 | s0;
        let k = |w: &[u32], wave64| exec_constant(&build(w, wave64).blocks[0].insts[0]);
        // Counts are inline constants (128 + n) or literals; S0 is the width, S1 the offset.
        assert_eq!(k(&[bfm(0x24, 128 + 16, 128 + 16)], false), Some(0xFFFF_0000));
        assert_eq!(k(&[bfm(0x24, 128 + 8, 128 + 4)], false), Some(0x0000_0FF0));
        // Five bits of each count in the 32-bit form: 33 & 31 = 1.
        assert_eq!(k(&[bfm(0x24, 255, 128), 33], false), Some(1));
        assert_eq!(k(&[bfm(0x25, 128 + 32, 128 + 32)], true), Some(0xFFFF_FFFF_0000_0000));
        // A count in an SGPR is not a constant.
        assert_eq!(k(&[bfm(0x24, 2, 128 + 16)], false), None);
        // s_bitreplicate_b64_b32 exec, 0xff00ff00 (SOP1 opcode 59): the upper row of each half.
        let rep = |s0: u8| 0xBE80_0000 | (EXEC as u32) << 16 | 59 << 8 | s0 as u32;
        assert_eq!(k(&[rep(255), 0xFF00_FF00], true), Some(0xFFFF_0000_FFFF_0000));
        assert_eq!(k(&[rep(128 + 5)], true), Some(0b11_0011));
        assert_eq!(k(&[rep(2)], true), None);
    }

    #[test]
    fn wave32_mov_b64_all_lanes_keeps_exec_masked_for_dpp() {
        let w = [
            sop1(0, 0x3D, 193),
            vop1(1, 0x01, 135),
            sop1(EXEC, 0x04, 193),
            vop1(2, 0x01, 0xFA),
            dpp16(1, 0xE4, 0xF, 0xF, false),
            sop1(EXEC + 1, 0x03, 193),
            vop1(3, 0x01, 0xFA),
            dpp16(1, 0xE4, 0xF, 0xF, false),
            sop1(EXEC, 0x03, 0),
        ];
        let p = build(&w, false);
        let insts = &p.blocks[0].insts;
        let v2_val = insts[3].write(Reg::V(2)).unwrap();
        let v3_val = insts[5].write(Reg::V(3)).unwrap();
        let sec = &sections(&p, 0)[&0];
        assert_eq!(sec.shapes.get(&v2_val), Some(&Shape::K(7)));
        assert_eq!(sec.shapes.get(&v3_val), Some(&Shape::K(7)));
    }

    #[test]
    fn per_lane_prefix_soundness() {
        let (v0, v_other, m0, m_other) = (Value(1), Value(2), Value(3), Value(4));
        let red = |op: Option<Op>, src: Value, mask: Value, ident: u32, lanes: u64| Sym::Red {
            op,
            src,
            mask,
            ident,
            lanes,
        };

        for (wave, all) in [(32usize, u64::from(u32::MAX)), (64usize, u64::MAX)] {
            let below_mask = |l: usize| if l == 0 { 0 } else { (1u64 << l) - 1 };

            let mut valid_inc: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 0, below_mask(l) | (1u64 << l)))
                .collect();
            valid_inc[0] = red(None, v0, m0, 0, 1);
            assert_eq!(
                per_lane(&valid_inc, all),
                Some(Shape::Prefix(Op::AddU, v0, m0, 0, true))
            );

            let mut valid_exc: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 0, below_mask(l)))
                .collect();
            valid_exc[0] = Sym::K(0);
            valid_exc[1] = red(None, v0, m0, 0, 1);
            assert_eq!(
                per_lane(&valid_exc, all),
                Some(Shape::Prefix(Op::AddU, v0, m0, 0, false))
            );

            // Lane 0 holding `-0.0` (`0x8000_0000`) when `ident` is `+0.0` (`0`) must not match.
            let mut neg_zero_lane0: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddF), v0, m0, 0, below_mask(l)))
                .collect();
            neg_zero_lane0[0] = Sym::K(0x8000_0000);
            neg_zero_lane0[1] = red(None, v0, m0, 0, 1);
            assert_eq!(per_lane(&neg_zero_lane0, all), None);

            let mut mixed_other_src = valid_exc.clone();
            mixed_other_src[0] = red(None, v_other, m0, 0, 1);
            assert_eq!(per_lane(&mixed_other_src, all), None);

            let mut mixed_same_src = valid_exc.clone();
            mixed_same_src[0] = red(None, v0, m0, 0, 1);
            assert_eq!(per_lane(&mixed_same_src, all), None);

            let mut exc_bad_lane1_src = valid_exc.clone();
            exc_bad_lane1_src[1] = red(None, v_other, m0, 0, 1);
            assert_eq!(per_lane(&exc_bad_lane1_src, all), None);

            let mut exc_bad_lane1_set = valid_exc.clone();
            exc_bad_lane1_set[1] = red(None, v0, m0, 0, 0b11);
            assert_eq!(per_lane(&exc_bad_lane1_set, all), None);

            for bad0 in [
                red(None, v0, m0, 1, 1),
                red(None, v_other, m0, 0, 1),
                red(None, v0, m_other, 0, 1),
                red(None, v0, m0, 0, 0),
                Sym::K(0),
            ] {
                let mut bad_inc = valid_inc.clone();
                bad_inc[0] = bad0;
                assert_eq!(per_lane(&bad_inc, all), None);
            }

            let mut bad_ident: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 1, below_mask(l) | (1u64 << l)))
                .collect();
            bad_ident[0] = red(None, v0, m0, 1, 1);
            assert_eq!(per_lane(&bad_ident, all), None);
        }
    }

    #[test]
    fn wave64_single_lane_and_salu_reductions_lower_while_partial_lanes_stay_refused() {
        use crate::emit::lower::{self, Parts};
        use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};

        let lower_cs = |body: &[u32], lanes: u32| {
            let mut words = vec![
                vop1(2, 1, 128), // v_mov_b32 v2, 0
                vop1(3, 1, 128), // v_mov_b32 v3, 0
                vop1(8, 1, 128), // v_mov_b32 v8, 0
                vop1(9, 1, 128), // v_mov_b32 v9, 0
            ];
            words.extend_from_slice(body);
            words.extend([
                0xDC70_8000,
                0x007D_0308, // global_store_dword v[8:9], v3, off
                S_ENDPGM,
            ]);
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
            let p = crate::ir::build(&native.code, true, 256).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
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
                wave32_host: true,
                zero_copy_memory: false,
                rect_list: false,
                copy_params: [0; 32],
                view_marks: false,
            })
        };

        let mut half_or = vec![sop1(0, 0x25, 193)]; // s_or_saveexec_b64 s[0:1], -1
        half_or.extend(vop3(1, 0x101, 128, 256, 0)); // v_cndmask_b32 v1, 0, v0, s[0:1]
        for ctrl in [0xB1, 0x4E, 0x141, 0x140] {
            half_or.push(vop2(1, 0x1C, 0xFA, 1)); // v_or_b32_dpp v1, v1, v1
            half_or.push(dpp16(1, ctrl, 0xF, 0xF, false));
        }
        half_or.extend([0xD778_0002, 0x0305_8301]); // v_permlanex16_b32 v2, v1, -1, -1
        half_or.push(vop2(1, 0x1C, 256 + 1, 2)); // v_or_b32 v1, v1, v2

        // Variant 1: combine lower half into lane 63 only (`v_writelane_b32`), then read lane 63
        // outside the section after narrowing `EXEC` (`s_mov_b64 exec, 1`) in 64- and 128-lane groups.
        let mut lane63 = half_or.clone();
        lane63.push(vop1(2, 1, 128)); // v_mov_b32 v2, 0
        lane63.extend(vop3(4, 0x360, 256 + 1, 128, 0)); // v_readlane_b32 s4, v1, 0
        lane63.extend(vop3(2, 0x361, 4, 128 + 63, 0)); // v_writelane_b32 v2, s4, 63
        lane63.push(vop2(1, 0x1C, 256 + 2, 1)); // v_or_b32 v1, v2, v1
        lane63.extend([
            0xBEFE_0400, // s_mov_b64 exec, s[0:1]
            0xBEFE_0481, // s_mov_b64 exec, 1
        ]);
        lane63.extend(vop3(5, 0x360, 256 + 1, 128 + 63, 0)); // v_readlane_b32 s5, v1, 63
        lane63.push(vop1(3, 1, 5)); // v_mov_b32 v3, s5
        lane63.push(0xBEFE_0400); // s_mov_b64 exec, s[0:1]
        for group in [64, 128] {
            lower_cs(&lane63, group).expect("lane 63 holds all 64 lanes and lowers at the section");
        }

        // Negative 1: lane 0 only holds lanes 0..31, so reading lane 0 stays refused.
        let mut lane0_partial = half_or.clone();
        lane0_partial.push(vop1(2, 1, 128));
        lane0_partial.extend(vop3(4, 0x360, 256 + 1, 128, 0));
        lane0_partial.extend(vop3(2, 0x361, 4, 128 + 63, 0));
        lane0_partial.push(vop2(1, 0x1C, 256 + 2, 1));
        lane0_partial.extend([0xBEFE_0400, 0xBEFE_0481]);
        lane0_partial.extend(vop3(5, 0x360, 256 + 1, 128, 0)); // v_readlane_b32 s5, v1, 0
        lane0_partial.push(vop1(3, 1, 5));
        lane0_partial.push(0xBEFE_0400);
        assert!(matches!(
            lower_cs(&lane0_partial, 128).err(),
            Some(crate::ShaderError::Unsupported { what: NO_SHAPE, .. })
        ));

        // Negative 2: reading a non-constant lane (`s2` live-in) after the section stays refused.
        let mut read_nonconst = half_or.clone();
        read_nonconst.push(vop1(2, 1, 128));
        read_nonconst.extend(vop3(4, 0x360, 256 + 1, 128, 0));
        read_nonconst.extend(vop3(2, 0x361, 4, 128 + 63, 0));
        read_nonconst.push(vop2(1, 0x1C, 256 + 2, 1));
        read_nonconst.push(0xBEFE_0400); // s_mov_b64 exec, s[0:1]
        read_nonconst.extend(vop3(5, 0x360, 256 + 1, 2, 0)); // v_readlane_b32 s5, v1, s2
        read_nonconst.push(vop1(3, 1, 5));
        assert!(matches!(
            lower_cs(&read_nonconst, 128).err(),
            Some(crate::ShaderError::Unsupported { what: NO_SHAPE, .. })
        ));

        // Negative 3: `v_writelane_b32` with a non-constant lane (`s2`) stays refused (`VALU`).
        let mut write_nonconst = half_or.clone();
        write_nonconst.push(vop1(2, 1, 128));
        write_nonconst.extend(vop3(4, 0x360, 256 + 1, 128, 0));
        write_nonconst.extend(vop3(2, 0x361, 4, 2, 0)); // v_writelane_b32 v2, s4, s2
        write_nonconst.push(vop2(1, 0x1C, 256 + 2, 1));
        write_nonconst.extend([0xBEFE_0400, 0xBEFE_0481]);
        write_nonconst.extend(vop3(5, 0x360, 256 + 1, 128 + 63, 0));
        write_nonconst.push(vop1(3, 1, 5));
        write_nonconst.push(0xBEFE_0400);
        assert!(matches!(
            lower_cs(&write_nonconst, 128).err(),
            Some(crate::ShaderError::Unsupported { what: VALU, .. })
        ));

        // Variant 2: combine the two 32-lane halves in SALU (`s_or_b32`) inside the section.
        let mut salu_or = half_or;
        salu_or.extend(vop3(4, 0x360, 256 + 1, 128, 0)); // v_readlane_b32 s4, v1, 0
        salu_or.extend(vop3(5, 0x360, 256 + 1, 128 + 32, 0)); // v_readlane_b32 s5, v1, 32
        salu_or.extend([
            0x8806_0504,   // s_or_b32 s6, s4, s5
            0xBEFE_0400,   // s_mov_b64 exec, s[0:1]
            vop1(3, 1, 6), // v_mov_b32 v3, s6
        ]);
        for group in [64, 128] {
            lower_cs(&salu_or, group).expect("SALU-combined wave64 reduction lowers");
        }

        // Negative 4: `s_add_u32` of the same partial value twice (`s4 + s4`, overlapping lanes
        // `0..31`) stays `Sym::Unknown` (`SCALAR`).
        let mut half_sum = vec![sop1(0, 0x25, 193)]; // s_or_saveexec_b64 s[0:1], -1
        half_sum.extend(vop3(1, 0x101, 128, 256, 0)); // v_cndmask_b32 v1, 0, v0, s[0:1]
        for ctrl in [0xB1, 0x4E, 0x141, 0x140] {
            half_sum.push(vop2(1, 0x25, 0xFA, 1)); // v_add_nc_u32_dpp v1, v1, v1
            half_sum.push(dpp16(1, ctrl, 0xF, 0xF, false));
        }
        half_sum.extend([0xD778_0002, 0x0305_8301]); // v_permlanex16_b32 v2, v1, -1, -1
        half_sum.push(vop2(1, 0x25, 256 + 1, 2)); // v_add_nc_u32 v1, v1, v2
        let mut dup_salu = half_sum;
        dup_salu.extend(vop3(4, 0x360, 256 + 1, 128, 0)); // v_readlane_b32 s4, v1, 0 (lanes 0..31)
        dup_salu.extend([
            0x8006_0404,   // s_add_u32 s6, s4, s4 (duplicate lanes)
            0xBEFE_0400,   // s_mov_b64 exec, s[0:1]
            vop1(3, 1, 6), // v_mov_b32 v3, s6
        ]);
        assert!(matches!(
            lower_cs(&dup_salu, 128).err(),
            Some(crate::ShaderError::Unsupported { what: SCALAR, .. })
        ));

        // Exclusive prefix difference (`v_sub_co_u32` and `v_subrev_nc_u32`):
        // Build an inclusive prefix sum across all 64 lanes in `v1` from each lane's masked term `v4`.
        let mut incl = vec![sop1(0, 0x25, 193)]; // s_or_saveexec_b64 s[0:1], -1
        incl.extend(vop3(4, 0x101, 128, 256, 0)); // v_cndmask_b32 v4, 0, v0, s[0:1]
        incl.push(vop1(1, 1, 256 + 4)); // v_mov_b32 v1, v4
        for shift in [1u16, 2, 4, 8] {
            incl.push(vop2(1, 0x25, 0xFA, 1)); // v_add_nc_u32_dpp v1, v1, v1 row_shr:shift
            incl.push(dpp16(1, 0x110 + shift, 0xF, 0xF, false));
        }
        incl.extend(vop3(2, 0x360, 256 + 1, 128 + 15, 0)); // v_readlane_b32 s2, v1, 15
        incl.extend([sop1(EXEC, 0x04, 255), 0xFFFF_0000]); // s_mov_b64 exec, 0xffff0000
        incl.push(vop2(1, 0x25, 2, 1)); // v_add_nc_u32 v1, s2, v1
        incl.extend(vop3(2, 0x360, 256 + 1, 128 + 31, 0)); // v_readlane_b32 s2, v1, 31
        incl.extend([sop1(EXEC, 0x03, 128), sop1(EXEC + 1, 0x03, 255), 0x0000_FFFF]);
        incl.push(vop2(1, 0x25, 2, 1));
        incl.extend(vop3(2, 0x360, 256 + 1, 128 + 47, 0)); // v_readlane_b32 s2, v1, 47
        incl.extend([sop1(EXEC, 0x03, 128), sop1(EXEC + 1, 0x03, 255), 0xFFFF_0000]);
        incl.push(vop2(1, 0x25, 2, 1));
        incl.push(sop1(EXEC, 0x04, 193)); // s_mov_b64 exec, -1

        // Positive `v_subrev_nc_u32 v3, v4, v1` (`v1 - v4` = exclusive prefix).
        let mut subrev_ok = incl.clone();
        subrev_ok.push(vop2(3, 0x27, 256 + 4, 1)); // v_subrev_nc_u32 v3, v4, v1
        subrev_ok.push(0xBEFE_0400);
        lower_cs(&subrev_ok, 128).expect("v_subrev_nc_u32 (v1 - v4) lowers as exclusive prefix");

        // Negative 5: reversed `v_subrev_nc_u32 v3, v1, v4` (`v4 - v1`) stays refused (`NO_SHAPE`).
        let mut subrev_bad = incl.clone();
        subrev_bad.push(vop2(3, 0x27, 256 + 1, 4)); // v_subrev_nc_u32 v3, v1, v4
        subrev_bad.push(0xBEFE_0400);
        assert!(matches!(
            lower_cs(&subrev_bad, 128).err(),
            Some(crate::ShaderError::Unsupported { what: NO_SHAPE, .. })
        ));

        // Positive `v_sub_co_u32 v3, s[6:7], v1, v4` when carry-out `s[6:7]` is unread outside.
        let mut sub_co_ok = incl.clone();
        sub_co_ok.extend([
            0xD710_0603,
            0x0002_0901, // v_sub_co_u32 v3, s[6:7], v1, v4
            0xBEFE_0400, // s_mov_b64 exec, s[0:1]
        ]);
        lower_cs(&sub_co_ok, 128).expect("v_sub_co_u32 exclusive prefix lowers when carry is unread");

        // Negative 6: reading the carry-out `s6` of `v_sub_co_u32` after the section stays refused.
        let mut sub_co_carry_read = incl;
        sub_co_carry_read.extend([
            0xD710_0603,
            0x0002_0901,   // v_sub_co_u32 v3, s[6:7], v1, v4
            0xBEFE_0400,   // s_mov_b64 exec, s[0:1]
            vop1(3, 1, 6), // v_mov_b32 v3, s6 (reads carry-out outside section)
        ]);
        assert!(matches!(
            lower_cs(&sub_co_carry_read, 128).err(),
            Some(crate::ShaderError::Unsupported { what: NO_SHAPE, .. })
        ));
    }
}
