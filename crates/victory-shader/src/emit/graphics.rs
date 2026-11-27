//! Graphics programs of the console's pipeline descriptions (device slot 0x1c8), inferred.
//!
//! A description gives a stage's GFX10 code and its metadata (the compute metadata's records
//! after a 0x10-byte header, [`crate::native::metadata_resources`]), but not the stage data an
//! `FSSM` record carries: the register image is undecoded. So what that data says is inferred
//! from the code, and a container is made that holds the code, the metadata, and the two
//! signatures the translator reads from a container (`ISG1`, `OSG1`), made from the same
//! inference. Every inference here is a guess (docs/native-host.md, "Guessed graphics
//! pipelines"), used only when asked for (`VICTORY_GUESS_GRAPHICS=1`):
//!
//! * a vertex shader fetches its own vertices (its VGPR inputs are the ids), so it has no
//!   attributes; its outputs are the `exp param<n>` it makes, each a `TEXCOORD<n>` of register n,
//!   with the channels it enables, and `SV_Position` after them;
//! * a pixel shader's attribute n is a `TEXCOORD<n>` of register n (so the interpolant of
//!   vertex parameter n), with the channels it reads; the VGPRs it reads on entry are, in order,
//!   barycentric pairs (when it interpolates), then position, front face, ancillary, coverage,
//!   with matching system-value elements declared in `ISG1`; its colour exports `exp mrt<n>` are
//!   targets of register n, in float (a compressed export is two FP16 values per dword);
//! * the wave size is inferred from wave64/wave32-only scalar mask instructions and high-half
//!   mask registers (`exec_hi`/`vcc_hi`), defaulting to 64 when neither signal is present.

use super::container::{SigElement, signature};
use crate::ShaderError::{self, Malformed};
use crate::dxbc::{self, FourCC, u32_at};
use crate::gfx10::{Fields, Instruction, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::native::{
    HwStage, NativeProgram, PsInput, RootParam, RootType, StageIo, Usage, VsOutput,
    expand_constants, metadata_resources,
};

/// The part a graphics piece pair is carried in: `"FSPG"`, u32 stage (0 vertex, 1 pixel), u32
/// code size, u32 metadata size, then the code and the metadata.
pub const MAGIC: &[u8] = b"FSPG";

pub(super) fn is_guessed(c: &dxbc::Container<'_>) -> bool {
    c.part(FourCC::DXIL)
        .or(c.part(FourCC::ACCC))
        .is_some_and(|part| part.data.starts_with(MAGIC))
}

/// The root slot a vertex shader's vertex buffer table (s0/s1) is given: no title root parameter.
pub const VERTEX_TABLE: u8 = 253;

/// Guessed pipelines' cross-stage convention for 16-bit parameters (docs/native-host.md, "16-bit
/// parameters of guessed pipelines"). Neither guessed stage knows which parameter dwords the
/// other packs as two f16 halves, so a vertex or mesh program exports every parameter channel
/// three times: as itself (`TEXCOORD<n>`), and its low and high halves each widened to f32
/// (`HALF<2n>`, `HALF<2n + 1>`), all interpolated. A pixel program's f16 interpolation of
/// attribute n reads the half it names from the `HALF` element; its 32-bit reads stay
/// `TEXCOORD<n>`. Interpolating each half as an f32 is what the console's f16 interpolation of
/// that half computes, up to the f16 rounding the pixel side applies.
pub(crate) const HALF: &str = "HALF";

/// Signature elements (of up to four channels) a guessed vertex or mesh program may export, the
/// halves included: Metal takes at most 124 scalar inputs to a fragment function (31 rows).
const MAX_ROWS: u32 = 31;

/// Where the metadata's records start (the header is 0x10 bytes).
const RECORDS: usize = 0x10;

/// A container with the part of `stage`'s `code` and `meta`, and the signatures inferred from the
/// code.
pub fn container(stage: HwStage, code: &[u8], meta: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let (program, sigs) = infer(stage, code, meta)?;
    let mut part = MAGIC.to_vec();
    for w in [u32::from(program.stage == HwStage::Ps), code.len() as u32, meta.len() as u32] {
        part.extend(w.to_le_bytes());
    }
    part.extend(code);
    part.extend(meta);
    let mut parts: Vec<(FourCC, Vec<u8>)> = vec![(FourCC::ACCC, part)];
    parts.push((FourCC::ISG1, signature(&sigs.0, true)));
    parts.push((FourCC::OSG1, signature(&sigs.1, false)));
    let parts: Vec<(FourCC, &[u8])> = parts.iter().map(|(f, d)| (*f, &d[..])).collect();
    Ok(dxbc::build(&parts))
}

/// The program of an `FSPG` part.
pub fn program(part: &[u8]) -> Result<NativeProgram, ShaderError> {
    let (stage, code_len, meta_len) =
        (u32_at(part, 4)?, u32_at(part, 8)? as usize, u32_at(part, 12)? as usize);
    let code = part.get(16..16 + code_len).ok_or(Malformed("FSPG code out of bounds"))?;
    let meta = part
        .get(16 + code_len..16 + code_len + meta_len)
        .ok_or(Malformed("FSPG metadata out of bounds"))?;
    let stage = if stage == 0 { HwStage::Vs } else { HwStage::Ps };
    Ok(infer(stage, code, meta)?.0)
}

type Signatures = (Vec<SigElement>, Vec<SigElement>);

fn element(name: &str, index: u32, kind: u8, row: u32, mask: u8) -> SigElement {
    let lo = mask.trailing_zeros().min(3) as u8;
    let hi = 8 - mask.leading_zeros() as u8;
    SigElement {
        name: name.into(),
        index,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: lo,
        cols: hi.saturating_sub(lo).max(1),
        usage: 0,
        half: false,
    }
}

fn infer(
    stage: HwStage,
    code: &[u8],
    meta: &[u8],
) -> Result<(NativeProgram, Signatures), ShaderError> {
    if !matches!(stage, HwStage::Vs | HwStage::Ps) {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "a graphics stage but vertex and pixel",
        });
    }
    let list = crate::gfx10::decode(code)?;
    let wave64 = infer_wave64(&list);
    // A vertex stage that exports primitives is an NGG program, run as a mesh shader.
    let stage = if stage == HwStage::Vs
        && list.iter().any(|i| matches!(i.fields(wave64), Fields::Exp { target: EXP_PRIM, .. }))
    {
        HwStage::Ms
    } else {
        stage
    };
    let (mut layout, mut used) = metadata_resources(meta, RECORDS, |_| Ok(1))?;
    // The registers the program names: its VGPR count, and what it reads on entry.
    let p = crate::ir::build(code, wave64, 256)?;
    if stage == HwStage::Vs
        && matches!(u32_at(meta, 0), Ok(2))
        && let Some(&[0x40, first]) = meta.get(12..14)
    {
        // shortcut: try header-relative 0x41 indices only when at least one lies below
        // the stated start, until the register image identifies the user-data layout.
        if first > 0
            && layout.iter().any(|r| r.user_data < first)
            && layout.iter().all(|r| u32::from(first) + u32::from(r.user_data) + r.sgprs <= 32)
        {
            let mut relative = layout.clone();
            for r in &mut relative {
                r.user_data += first;
            }
            if crate::resource::vertex_roots_match_rebased(&p, &layout, &relative)? {
                layout = relative;
                for r in &mut used {
                    if r.root != RootType::ImmediateSampler {
                        r.user_data = r.user_data.checked_add(first)
                            .ok_or(Malformed("FSPG user-data index out of bounds"))?;
                    }
                }
            }
        }
    }
    expand_constants(&mut layout);
    let occupied = |layout: &[RootParam], reg: u8| {
        let r = u32::from(reg);
        layout
            .iter()
            .any(|p| r >= u32::from(p.user_data) && r < u32::from(p.user_data) + p.sgprs)
    };
    // shortcut: SM 6.6 ResourceHeap / SamplerHeap SGPRs are inferred from 32-bit descriptor loads
    // (`hi == 0`) feeding resource/sampler instructions; upgrade when the register image or root
    // signature blob is decoded.
    let inferred_roots = crate::resource::infer_roots(&p, &layout, stage)?;
    // The tracer names SGPRs; a mesh program's user data starts at s8 (`HwStage::user_sgpr`).
    let user_data_of = |sgpr: u8| sgpr.checked_sub(stage.user_sgpr(0));
    for (user_data, root) in inferred_roots.heaps {
        let Some(user_data) = user_data_of(user_data) else { continue };
        if !occupied(&layout, user_data) {
            layout.push(RootParam {
                root,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data,
                slot: if root == RootType::SamplerHeap { 254 } else { 252 },
            });
        }
    }
    // Guess: the heaps' SGPRs follow the last root parameter's, so root constants the metadata
    // leaves at one SGPR, with no parameter after them but a heap, reach up to the heap (the
    // title's mesh programs read b500 as two dwords, the heap SGPR after them).
    for i in 0..layout.len() {
        let (start, end) = (layout[i].user_data, u32::from(layout[i].user_data) + layout[i].sgprs);
        let next = layout.iter().map(|p| p.user_data).filter(|&u| u > start).min();
        let heap = |u: u8| {
            layout.iter().any(|p| {
                p.user_data == u && matches!(p.root, RootType::ResourceHeap | RootType::SamplerHeap)
            })
        };
        if layout[i].root == RootType::Constants
            && let Some(next) = next.filter(|&u| u32::from(u) > end && heap(u))
        {
            layout[i].sgprs = u32::from(next - start);
        }
    }
    // Guess: root constants that are a pixel program's last user data take the SGPRs after them
    // (the metadata names only their first), as compute's reach its user SGPR count
    // (`constants_to_the_end`): up to the primitive mask, which follows the user SGPRs and only
    // reaches M0, when the program reads one (observed: b500 in s10 to s14, the mask in s15, s13
    // read on entry); else as far as it reads them on entry (the upscale program's b500 in s9 to
    // s12, with no interpolation).
    if stage == HwStage::Ps
        && let Some(last) = layout.iter_mut().max_by_key(|r| r.user_data)
        && last.root == RootType::Constants
    {
        let uses = crate::io::PrimMaskUses::new(&p);
        let live = |s: u32| p.inputs.iter().find(|i| i.0 == Reg::S(s as u8)).map(|i| i.1);
        let mask = |s: u32| live(s).is_some_and(|v| uses.is_prim_mask_only(v, true));
        let start = u32::from(last.user_data) + last.sgprs;
        let end = match (start..32).find(|&s| mask(s)) {
            Some(m) => m,
            None => (start..32).find(|&s| live(s).is_none()).unwrap_or(32),
        };
        last.sgprs = end - u32::from(last.user_data);
    }
    // A vertex shader's own vertex fetch: usually s0/s1 address the table of vertex buffer
    // descriptors it loads from (observed: `s_load_dwordx4 sN, s[0:1]` before its `buffer_load`),
    // unless metadata already occupies s0..s1 or an inferred vertex table sits at another pair.
    if stage == HwStage::Vs {
        let vt = inferred_roots
            .vertex_table
            .filter(|&n| !occupied(&layout, n) && !occupied(&layout, n + 1))
            .or_else(|| (!occupied(&layout, 0) && !occupied(&layout, 1)).then_some(0));
        if let Some(user_data) = vt {
            layout.push(RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data,
                slot: VERTEX_TABLE,
            });
        }
    }
    // An address in a live-in SGPR pair no parameter fills is title memory
    // (guessed: the console's user data carries raw addresses). Declare the
    // pair as root constants so lowering can read it from the user-data
    // block instead of refusing the SGPR read.
    // shortcut: the slot is the first free one; the register image does not
    // yet say which root parameter holds the pair.
    if let Ok(res) = crate::resource::track(&p, &layout, stage, None) {
        for n in res.memory_inputs.into_iter().filter_map(user_data_of) {
            if occupied(&layout, n) || occupied(&layout, n + 1) {
                continue;
            }
            let Some(slot) = (0..252u16).find(|s| !layout.iter().any(|p| p.slot == *s as u8))
            else {
                continue;
            };
            layout.push(RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: n,
                slot: slot as u8,
            });
        }
    }
    // Root constants run on over the user SGPRs right after them that the program reads on entry
    // and no other parameter fills, as compute's do (`constants_to_the_end`): a guessed block is
    // one dword unless the next root slot bounds it, and the title's mesh-style programs read a
    // second word from the SGPR after it (an index they mask and scale into a descriptor
    // offset), which was "a hardware SGPR input".
    let live: std::collections::BTreeSet<u8> = p
        .inputs
        .iter()
        .filter_map(|&(r, _)| match r {
            Reg::S(n) => user_data_of(n),
            _ => None,
        })
        .collect();
    // A pixel program's take the rule above, which stops at its primitive mask.
    for i in 0..layout.len() {
        if stage == HwStage::Ps || layout[i].root != RootType::Constants {
            continue;
        }
        while layout[i].sgprs < 32 {
            let Ok(next) = u8::try_from(u32::from(layout[i].user_data) + layout[i].sgprs) else {
                break;
            };
            if !live.contains(&next) || occupied(&layout, next) {
                break;
            }
            layout[i].sgprs += 1;
        }
    }
    let top = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .flat_map(|i| i.reads.iter().chain(&i.writes))
        .filter_map(|r| match r.0 {
            Reg::V(n) => Some(u16::from(n) + 1),
            _ => None,
        })
        .max()
        .unwrap_or(1);
    let vgprs = top.next_multiple_of(4);
    // shortcut: live-in VGPR count is inferred from entry VGPRs explicitly read as instruction
    // sources (before any write), ignoring IR old-value ties of writes; upgrade when
    // `SPI_PS_INPUT_ADDR` is read from the register image.
    let live_in = explicit_entry_vgprs(&p)
        .into_iter()
        .map(|n| u16::from(n) + 1)
        .max()
        .unwrap_or(0);

    let (mut params, mut targets) = (Vec::<(u8, u8)>::new(), Vec::<(u8, u8)>::new());
    let mut attrs: Vec<(u8, u8, bool)> = Vec::new(); // (attribute, 32-bit channels, flat)
    let mut halves: Vec<(u8, u8, u8)> = Vec::new(); // (attribute, low, high halves' channels)
    let mut depth = false;
    let mut interpolates = false;
    for i in &list {
        let flat = i.name() == Some("v_interp_mov_f32");
        let f16 = i.name().is_some_and(|n| n.starts_with("v_interp") && n.ends_with("_f16"));
        let mut attr = |a: &Operand| {
            if let Operand::Attr { index, chan, high } = *a {
                let half = f16 || high;
                let bits = if half { 0 } else { 1 << chan };
                match attrs.iter_mut().find(|e| e.0 == index) {
                    Some(e) => {
                        e.1 |= bits;
                        e.2 &= flat;
                    }
                    None => attrs.push((index, bits, flat)),
                }
                if half {
                    let e = match halves.iter().position(|e| e.0 == index) {
                        Some(n) => &mut halves[n],
                        None => {
                            halves.push((index, 0, 0));
                            halves.last_mut().expect("pushed")
                        }
                    };
                    if high {
                        e.2 |= 1 << chan;
                    } else {
                        e.1 |= 1 << chan;
                    }
                }
                interpolates |= !flat;
            }
        };
        match i.fields(wave64) {
            Fields::Vintrp { attr: a, .. } => attr(&a),
            Fields::Valu(v) => v.src.iter().for_each(attr),
            Fields::Exp { target, src, .. } => {
                let mask = (0..4).filter(|&c| src[c].is_some()).fold(0u8, |m, c| m | 1 << c);
                match target {
                    0..=7 => targets.push((target, mask)),
                    8 => depth |= mask & 1 != 0,
                    32..=63 => params.push((target - 32, mask)),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let merge = |v: &mut Vec<(u8, u8)>| {
        let mut out: Vec<(u8, u8)> = Vec::new();
        for &(n, m) in v.iter() {
            match out.iter_mut().find(|e| e.0 == n) {
                Some(e) => e.1 |= m,
                None => out.push((n, m)),
            }
        }
        out.sort_unstable();
        *v = out;
    };
    merge(&mut params);
    merge(&mut targets);
    attrs.sort_unstable();

    let (io, sigs) = if stage != HwStage::Ps {
        let rows = params.iter().map(|p| u32::from(p.0) + 1).max().unwrap_or(0);
        let packing = packed_params(&p);
        let of = |n: u8| packing.get(&n).copied().unwrap_or(Packing::Plain);
        // Four channels whatever the code exports: Metal links a fragment input only to an
        // output of the same type, and the pixel side reads channels the code does not show.
        // A parameter only packing instructions make has its `TEXCOORD` last, if there is room.
        let mut outs: Vec<SigElement> = params
            .iter()
            .filter(|p| of(p.0) != Packing::Halves)
            .map(|&(n, _)| element("TEXCOORD", u32::from(n), 0, u32::from(n), 0xf))
            .collect();
        outs.push(element("SV_Position", 0, 3, rows, 0xf));
        let outputs = params
            .iter()
            .map(|&(n, _)| VsOutput {
                usage: Usage::Generic,
                row: n,
                param: n,
                fp16: false,
                pack2: false,
            })
            .collect();
        let io = if stage == HwStage::Ms {
            StageIo::Ms(mesh_io(&p, &list, wave64, outputs)?)
        } else {
            StageIo::Vs { inputs: Vec::new(), outputs }
        };
        // The halves (`HALF`) of each parameter that may hold them, as many as fit, by how
        // likely: a mesh stage's outputs take 16 bytes an element for each vertex, at most
        // 32 KiB in all (the converter's limit).
        let max = match &io {
            StageIo::Ms(ms) => (32 * 1024 / (16 * ms.max_vertices.max(1))).min(MAX_ROWS),
            _ => MAX_ROWS,
        } as usize;
        let mut next = rows + 1;
        let mut raw = Vec::new();
        for kind in [Packing::Halves, Packing::Mixed, Packing::Unknown] {
            for &(n, _) in params.iter().filter(|p| of(p.0) == kind) {
                if outs.len() + 2 <= max {
                    outs.push(element(HALF, 2 * u32::from(n), 0, next, 0xf));
                    outs.push(element(HALF, 2 * u32::from(n) + 1, 0, next + 1, 0xf));
                    next += 2;
                }
                if kind == Packing::Halves {
                    raw.push(n);
                }
            }
        }
        // The dwords of the parameters of halves too, as room allows (a pixel program may read
        // them as 32 bits, flat): after every parameter's halves.
        for n in raw {
            let halves = outs.iter().any(|o| o.name == HALF && o.index == 2 * u32::from(n));
            if !halves || outs.len() < max {
                outs.push(element("TEXCOORD", u32::from(n), 0, u32::from(n), 0xf));
            }
        }
        (io, (Vec::new(), outs))
    } else {
        let mut ins: Vec<SigElement> = attrs
            .iter()
            // Four channels, as the vertex side declares them; none for an attribute read only
            // as halves (whose vertex side may export no `TEXCOORD`).
            .filter(|a| a.1 != 0)
            .map(|&(n, _, _)| element("TEXCOORD", u32::from(n), 0, u32::from(n), 0xf))
            .collect();
        let mut next_row = attrs.iter().map(|a| u32::from(a.0) + 1).max().unwrap_or(0);
        halves.sort_unstable();
        for &(n, lo, hi) in &halves {
            for (index, m) in [(2 * u32::from(n), lo), (2 * u32::from(n) + 1, hi)] {
                if m != 0 {
                    // Four channels, as the vertex side declares them: Metal links an input
                    // only to an output of the same type.
                    ins.push(element(HALF, index, 0, next_row, 0xf));
                    next_row += 1;
                }
            }
        }
        let mut outs: Vec<SigElement> = targets
            .iter()
            .map(|&(n, m)| element("SV_Target", u32::from(n), 16, u32::from(n), m))
            .collect();
        if depth {
            outs.push(element("SV_Depth", 0, 17, 0, 1));
        }
        let inputs = attrs
            .iter()
            .map(|&(n, m, flat)| PsInput {
                usage: Usage::Generic,
                row: n,
                mask: m,
                attr: n,
                flat,
                fp16: false,
                pack2: false,
                row2: 0,
            })
            .collect();
        // Entry VGPRs, as `SPI_PS_INPUT_ADDR` enables them: barycentric pairs first (in common
        // order: PerspectiveCenter, PerspectiveCentroid, LinearCenter, LinearCentroid,
        // PerspectiveSample, LinearSample), then system values from position x (bit 8) on.
        // shortcut: without the register image, exact barycentric modes and sparse system-value
        // bits (e.g. FrontFace without Position.zw) cannot be distinguished; upgrade when
        // `SPI_PS_INPUT_ADDR` is decoded from the 0x1c8 register image.
        const BARY_BITS: [u32; 6] = [1, 2, 5, 6, 0, 4];
        let mut bary_pairs = barycentric_entry_vgprs(&p)
            .into_iter()
            .map(|r| usize::from(r) / 2 + 1)
            .max()
            .unwrap_or(0)
            .min(BARY_BITS.len());
        if interpolates && bary_pairs == 0 {
            bary_pairs = 1;
        }
        let mut addr = 0u32;
        let mut left = i32::from(live_in);
        for &bit in &BARY_BITS[..bary_pairs] {
            addr |= 1 << bit;
            left -= 2;
        }
        for bit in 8..=14 {
            if left > 0 {
                addr |= 1 << bit;
                left -= 1;
            }
        }
        if addr & (0xF << 8) != 0 {
            ins.push(SigElement {
                name: "SV_Position".into(),
                index: 0,
                kind: 3,
                comp: 9,
                interp: 4,
                row: Some(next_row),
                col: 0,
                cols: 4,
                usage: 0,
                half: false,
            });
            next_row += 1;
        }
        if addr & (1 << 12) != 0 {
            ins.push(SigElement {
                name: "SV_IsFrontFace".into(),
                index: 0,
                kind: 13,
                comp: 5,
                interp: 1,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
        }
        if addr & (1 << 13) != 0 {
            ins.push(SigElement {
                name: "SV_SampleIndex".into(),
                index: 0,
                kind: 12,
                comp: 5,
                interp: 0,
                row: None,
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
        }
        if addr & (1 << 14) != 0 {
            ins.push(SigElement {
                name: "SV_Coverage".into(),
                index: 0,
                kind: 14,
                comp: 5,
                interp: 0,
                row: None,
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
        }
        let db_shader_control = if uses_pops(&list) { 1 << 16 } else { 0 };
        (
            StageIo::Ps { inputs, input_addr: addr, db_shader_control, patches: Vec::new() },
            (ins, outs),
        )
    };
    Ok((NativeProgram { stage, wave64, vgprs, layout, used, io, code: code.to_vec() }, sigs))
}

/// What a guessed parameter's dwords hold, from the instructions that make them (its channels'
/// values followed back through moves, `v_cndmask_b32` and joins to their sources).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Packing {
    /// 32-bit values: every source is a 32-bit operation (`*_f32` but a 16-bit conversion,
    /// `*_u32`, `*_i32`), an inline constant or a literal that is a normal f32 with a low half
    /// of 0.
    Plain,
    /// Two f16 halves a dword: every source is a packing or 16-bit instruction (`v_fma_mix*`,
    /// `v_cvt_pk*`, `v_cvt_f16_*`, `v_pack_b32_f16`, `v_pk_*`, `*_f16`), an inline constant or
    /// a literal that is no normal f32 (as halves such as 0x3c00 or 0xffffbc00 read).
    Halves,
    /// Some sources are of the second kind, others not.
    Mixed,
    /// Neither: loads, SGPRs, other literals and instructions, with no 16-bit source. Program
    /// inputs count as no source (ids, or what lanes that export nothing on a path hold).
    Unknown,
}

/// Each parameter's (`exp param<n>`) [`Packing`]. A guess (docs/native-host.md, "16-bit
/// parameters of guessed pipelines") that keeps the halves to the parameters that can need them.
fn packed_params(p: &Program) -> std::collections::HashMap<u8, Packing> {
    use std::collections::{HashMap, HashSet};
    const PLAIN: u8 = 1;
    const HALVES: u8 = 2;
    const OTHER: u8 = 4;
    let mut insts: HashMap<Value, &Inst> = HashMap::new();
    let mut phis: HashMap<Value, &[Value]> = HashMap::new();
    for b in &p.blocks {
        for ph in &b.phis {
            phis.insert(ph.value, &ph.args);
        }
        for i in &b.insts {
            for &(_, w) in &i.writes {
                insts.insert(w, i);
            }
        }
    }
    // The kinds of source a value comes from.
    let sources = |v: Value| {
        let (mut todo, mut seen, mut kinds) = (vec![v], HashSet::new(), 0u8);
        while let Some(v) = todo.pop() {
            if !seen.insert(v) {
                continue;
            }
            if let Some(args) = phis.get(&v) {
                todo.extend(args.iter().copied());
                continue;
            }
            let Some(i) = insts.get(&v) else { continue };
            let name = i.name();
            // A write to part of the dword (`sdwa.dst_sel`) that preserves the other bits leaves
            // what the parameter held there, and that old value is as much a source of this one.
            // A half packed into a dword is written one word at a time, and the last write before
            // an export is often a zero to the half the lane does not export (`v_cvt_f16_f32 v3,
            // v9`, then `v_mov_b32 v3, 0 dst_sel:WORD_1`), which would otherwise hide the half.
            if let Fields::Valu(f) = &i.fields
                && let Some(sd) = f.sdwa
                && sd.dst_sel < 6
                && sd.dst_unused == 2
                && let Some(Operand::Vector { reg, .. }) = f.dst
                && let Some(old) = i.read(Reg::V(reg))
            {
                todo.push(old);
            }
            let srcs = match (&i.fields, name) {
                (Fields::Valu(f), "v_mov_b32") => &f.src[..f.src.len().min(1)],
                (Fields::Valu(f), "v_cndmask_b32") => &f.src[..f.src.len().min(2)],
                (Fields::Valu(_), _)
                    if name.starts_with("v_cvt_pk")
                        || name.starts_with("v_cvt_f16_")
                        || name.starts_with("v_fma_mix")
                        || name.starts_with("v_pk_")
                        || name == "v_pack_b32_f16"
                        || name.ends_with("_f16") =>
                {
                    kinds |= HALVES;
                    continue;
                }
                (Fields::Valu(_), _)
                    if name.ends_with("_f32") || name.ends_with("_u32") || name.ends_with("_i32") =>
                {
                    kinds |= PLAIN;
                    continue;
                }
                _ => {
                    kinds |= OTHER;
                    continue;
                }
            };
            for s in srcs {
                match *s {
                    Operand::Const(_) => {}
                    Operand::Literal(0) => {}
                    Operand::Literal(x) if matches!(x >> 23 & 0xff, 0 | 0xff) => kinds |= HALVES,
                    Operand::Literal(x) if x & 0xffff == 0 => kinds |= PLAIN,
                    Operand::Literal(_) => kinds |= OTHER,
                    Operand::Vector { reg, .. } => match i.read(Reg::V(reg)) {
                        Some(v) => todo.push(v),
                        None => kinds |= OTHER,
                    },
                    _ => kinds |= OTHER,
                }
            }
        }
        kinds
    };
    let mut kinds: HashMap<u8, u8> = HashMap::new();
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        if let Fields::Exp { target: t @ 32..=63, src, .. } = &i.fields {
            let k = kinds.entry(t - 32).or_insert(0);
            for s in src.iter().flatten() {
                *k |= match *s {
                    Operand::Vector { reg, .. } => i.read(Reg::V(reg)).map_or(OTHER, sources),
                    _ => OTHER,
                };
            }
        }
    }
    kinds
        .into_iter()
        .map(|(n, k)| {
            let packing = match k {
                0 | PLAIN => Packing::Plain,
                HALVES => Packing::Halves,
                _ if k & HALVES != 0 => Packing::Mixed,
                _ => Packing::Unknown,
            };
            (n, packing)
        })
        .collect()
}

/// `exp prim`'s target.
const EXP_PRIM: u8 = 20;

/// What a mesh-style (NGG) vertex program's stage data would say, inferred from its code. All of
/// it is a guess (docs/native-host.md, "Mesh-style vertex programs"): the register image that
/// holds the group size is not decoded.
/// * The group id arrives in v5 times the group stride, and the program shifts it right
///   (`v_lshrrev_b32 vN, k, v5`): the group is `1 << k` lanes, which is also its maximum
///   vertex and primitive count (the title's programs count both in 7 bits plus one, k = 7).
/// * Triangles, every output per vertex, no clip distances.
/// * LDS: up to the highest `ds_*` offset plus 16 bytes.
// shortcut: LDS addresses are taken as constant 0 plus the offset (all the title's NGG programs
// share three dwords at offsets 0, 4 and 8); a computed address beyond would be out of bounds.
fn mesh_io(
    p: &Program,
    list: &[Instruction],
    wave64: bool,
    outputs: Vec<VsOutput>,
) -> Result<crate::native::MeshIo, ShaderError> {
    let group = p.inputs.iter().find(|i| i.0 == Reg::V(5)).map(|i| i.1);
    let shift = p.blocks.iter().flat_map(|b| &b.insts).find_map(|i| match &i.fields {
        Fields::Valu(v) if i.name() == "v_lshrrev_b32" && group.is_some_and(|g| i.reads.contains(&(Reg::V(5), g))) => {
            match v.src.first() {
                Some(&Operand::Const(c @ 128..=135)) => Some(u32::from(c - 128)),
                _ => None,
            }
        }
        _ => None,
    });
    let Some(shift) = shift else {
        return Err(ShaderError::Unsupported { offset: 0, what: "a mesh-style vertex program whose group stride is not a shift" });
    };
    let threads = 1 << shift;
    let lds_bytes = list
        .iter()
        .filter_map(|i| match i.fields(wave64) {
            Fields::Ds { offset0, offset1, gds: false, .. } => Some((u32::from(offset1) << 8 | u32::from(offset0)) + 16),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    Ok(crate::native::MeshIo {
        outputs,
        clip_cntl: 0,
        threads,
        max_vertices: threads,
        max_primitives: threads,
        topology: 2,
        lds_bytes,
    })
}

/// Infers whether `list` runs as wave64 (else wave32) from ISA signals rather than formatted text
/// (formatting with `wave64 = false` prints every implicit-VCC VALU op as `vcc_lo`).
/// Default is wave64 when neither wave64-only nor wave32-only signals are present.
fn infer_wave64(list: &[Instruction]) -> bool {
    let mut strong_w64 = false;
    let mut strong_w32 = false;
    let mut weak_w64 = false;
    let mut weak_w32 = false;
    for i in list {
        let name = i.name().unwrap_or("");
        let fields = i.fields(false);
        if name.ends_with("_saveexec_b64")
            || name.ends_with("_wrexec_b64")
            || matches!(name, "s_wqm_b64" | "s_quadmask_b64")
        {
            strong_w64 = true;
        }
        if name.ends_with("_saveexec_b32")
            || name.ends_with("_wrexec_b32")
            || matches!(name, "s_wqm_b32" | "s_quadmask_b32")
        {
            strong_w32 = true;
        }
        let (hi_exec, hi_vcc) = touches_hi_mask(&fields);
        if hi_exec {
            strong_w64 = true;
        }
        if hi_vcc {
            weak_w64 = true;
        }
        if let Fields::Sop { dst, src, .. } = &fields {
            let is_64 = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
            let is_32 = name.ends_with("_b32") || name.ends_with("_u32") || name.ends_with("_i32");
            for op in dst.iter().chain(src) {
                if let Operand::Scalar { code, .. } = *op {
                    if is_64 && code == 126 {
                        strong_w64 = true;
                    }
                    // A wave32 program may use the VCC pair as a spare 64-bit SGPR pair
                    // (`s_mov_b64 vcc, s[82:83]` beside `s_and_saveexec_b32`), so only weak.
                    if is_64 && code == 106 {
                        weak_w64 = true;
                    }
                    if is_32 && code == 126 {
                        weak_w32 = true;
                    }
                }
            }
        }
    }
    strong_w64 || (!strong_w32 && (weak_w64 || !weak_w32))
}

/// Whether `fields` (decoded with `wave64 = false`) explicitly names `exec_hi` (127) or `vcc_hi`
/// (107).
fn touches_hi_mask(fields: &Fields) -> (bool, bool) {
    let mut hi_exec = false;
    let mut hi_vcc = false;
    let mut check = |op: &Operand| {
        if let Operand::Scalar { code, .. } = *op {
            if code == 127 {
                hi_exec = true;
            } else if code == 107 {
                hi_vcc = true;
            }
        }
    };
    match fields {
        Fields::Sop { dst, src, .. } => dst.iter().chain(src).for_each(check),
        Fields::Smem { data, base, soffset, .. } => {
            data.iter().chain([base, soffset]).for_each(check);
        }
        Fields::Valu(v) => v.dst.iter().chain(&v.carry).chain(&v.src).for_each(check),
        Fields::Buffer { rsrc, soffset, .. } => {
            check(rsrc);
            check(soffset);
        }
        Fields::Image { rsrc, sampler, .. } => {
            check(rsrc);
            sampler.iter().for_each(check);
        }
        Fields::Flat { saddr, .. } => saddr.iter().for_each(check),
        Fields::Sopp { .. } | Fields::Vintrp { .. } | Fields::Ds { .. } | Fields::Exp { .. } => {}
    }
    (hi_exec, hi_vcc)
}

/// Whether a pixel shader uses Primitive Ordered Pixel Shading (POPS / ROVs).
fn uses_pops(list: &[Instruction]) -> bool {
    let is_pops_const = |op: &Operand| matches!(*op, Operand::Const(239));
    list.iter().any(|i| {
        let name = i.name().unwrap_or("");
        if name == "s_endpgm_ordered_ps_done" {
            return true;
        }
        match i.fields(false) {
            Fields::Sopp { imm } => {
                matches!(name, "s_sendmsg" | "s_sendmsghalt") && imm & 0xF == 7
            }
            Fields::Sop { src, .. } => src.iter().any(is_pops_const),
            Fields::Valu(v) => v.src.iter().any(is_pops_const),
            Fields::Smem { soffset, .. } | Fields::Buffer { soffset, .. } => {
                is_pops_const(&soffset)
            }
            _ => false,
        }
    })
}

/// Entry VGPRs that are explicitly read as source operands before being overwritten (ignoring
/// writes whose only read of the VGPR is the IR's inactive-lane old-value tie).
fn explicit_entry_vgprs(p: &Program) -> Vec<u8> {
    let entry: std::collections::HashMap<Value, u8> = p
        .inputs
        .iter()
        .filter_map(|&(reg, v)| match reg {
            Reg::V(n) => Some((v, n)),
            _ => None,
        })
        .collect();
    let mut found = Vec::new();
    let mark = |found: &mut Vec<u8>, inst: &Inst, reg: u8| {
        if let Some(v) = inst.read(Reg::V(reg))
            && let Some(&n) = entry.get(&v)
            && !found.contains(&n)
        {
            found.push(n);
        }
    };
    let mark_op = |found: &mut Vec<u8>, inst: &Inst, op: &Operand| {
        if let Operand::Vector { reg, dwords } = *op {
            for k in 0..dwords {
                if let Some(r) = reg.checked_add(k) {
                    mark(found, inst, r);
                }
            }
        }
    };
    for block in &p.blocks {
        for inst in &block.insts {
            let name = inst.name();
            match &inst.fields {
                Fields::Valu(v) => {
                    let src_indexed = matches!(
                        name,
                        "v_movrels_b32"
                            | "v_movrelsd_b32"
                            | "v_movrelsd_2_b32"
                            | "v_swaprel_b32"
                    );
                    for (idx, s) in v.src.iter().enumerate() {
                        if idx == 0
                            && src_indexed
                            && let Operand::Vector { reg, .. } = *s
                        {
                            for r in u16::from(reg)..p.vgprs {
                                mark(&mut found, inst, r as u8);
                            }
                        } else {
                            mark_op(&mut found, inst, s);
                        }
                    }
                    let reads_dst = name.contains("fmac")
                        || name.contains("_mac_")
                        || name.contains("dot2c")
                        || name.contains("dot4c")
                        || name.contains("dot8c")
                        || matches!(name, "v_swap_b32" | "v_swaprel_b32" | "v_writelane_b32");
                    if reads_dst && let Some(d) = &v.dst {
                        if name == "v_swaprel_b32"
                            && let Operand::Vector { reg, .. } = *d
                        {
                            for r in u16::from(reg)..p.vgprs {
                                mark(&mut found, inst, r as u8);
                            }
                        } else {
                            mark_op(&mut found, inst, d);
                        }
                    }
                }
                Fields::Vintrp { dst, src, .. } => {
                    mark_op(&mut found, inst, src);
                    if name == "v_interp_p2_f32" {
                        mark_op(&mut found, inst, dst);
                    }
                }
                Fields::Ds { addr, data, .. } => {
                    for o in addr.iter().chain(data) {
                        mark_op(&mut found, inst, o);
                    }
                }
                Fields::Buffer { data, addr, offen, idxen, .. } => {
                    if *offen || *idxen {
                        mark_op(&mut found, inst, addr);
                    }
                    if (name.contains("_store_") || name.contains("_atomic_"))
                        && let Some(d) = data
                    {
                        mark_op(&mut found, inst, d);
                    }
                }
                Fields::Image { data, addr, .. } => {
                    for a in addr {
                        mark_op(&mut found, inst, a);
                    }
                    if name.contains("_store") || name.contains("_atomic_") {
                        mark_op(&mut found, inst, data);
                    }
                }
                Fields::Flat { addr, data, .. } => {
                    for o in [addr, data].into_iter().flatten() {
                        mark_op(&mut found, inst, o);
                    }
                }
                Fields::Exp { src, .. } => {
                    for s in src.iter().flatten() {
                        mark_op(&mut found, inst, s);
                    }
                }
                Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. } => {}
            }
        }
    }
    found
}

/// Entry VGPR indices read as barycentrics (`i`/`j`) by `v_interp_p1*` / `v_interp_p2*`,
/// following plain `v_mov_b32` copies and phis.
fn barycentric_entry_vgprs(p: &Program) -> Vec<u8> {
    let entry: std::collections::HashMap<Value, u8> = p
        .inputs
        .iter()
        .filter_map(|&(reg, v)| match reg {
            Reg::V(n) => Some((v, n)),
            _ => None,
        })
        .collect();
    let defs: std::collections::HashMap<Value, &Inst> = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .flat_map(|i| i.writes.iter().map(move |w| (w.1, i)))
        .collect();
    let phis: std::collections::HashMap<Value, &[Value]> = p
        .blocks
        .iter()
        .flat_map(|b| &b.phis)
        .map(|phi| (phi.value, phi.args.as_slice()))
        .collect();

    let mut out = Vec::new();
    let mut trace = |start: Value| {
        let mut stack = vec![start];
        let mut seen = Vec::new();
        while let Some(v) = stack.pop() {
            if seen.contains(&v) {
                continue;
            }
            seen.push(v);
            if let Some(&n) = entry.get(&v) {
                if !out.contains(&n) {
                    out.push(n);
                }
            } else if let Some(&args) = phis.get(&v) {
                stack.extend_from_slice(args);
            } else if let Some(&inst) = defs.get(&v)
                && inst.name() == "v_mov_b32"
                && let Fields::Valu(f) = &inst.fields
                && f.abs == 0
                && f.neg == 0
                && f.sdwa.is_none()
                && f.dpp.is_none()
                && let Some(&Operand::Vector { reg, .. }) = f.src.first()
                && let Some(src_v) = inst.read(Reg::V(reg))
            {
                stack.push(src_v);
            }
        }
    };

    for block in &p.blocks {
        for inst in &block.insts {
            let name = inst.name();
            match &inst.fields {
                Fields::Vintrp { src: Operand::Vector { reg, .. }, .. }
                    if matches!(name, "v_interp_p1_f32" | "v_interp_p2_f32") =>
                {
                    if let Some(v) = inst.read(Reg::V(*reg)) {
                        trace(v);
                    }
                }
                Fields::Valu(v) if matches!(name, "v_interp_p1ll_f16" | "v_interp_p2_f16") => {
                    if let Some(&Operand::Vector { reg, .. }) = v.src.get(1)
                        && let Some(val) = inst.read(Reg::V(reg))
                    {
                        trace(val);
                    }
                }
                _ => {}
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENDPGM: u32 = 0xBF81_0000;

    fn words(ws: &[u32]) -> Vec<u8> {
        ws.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn meta_with(records: &[[u8; 32]]) -> Vec<u8> {
        let mut meta = vec![0u8; 0x10];
        meta[12] = 0x40;
        meta[14] = records.len() as u8;
        for r in records {
            meta.extend_from_slice(r);
        }
        let mut hash = [0u8; 32];
        hash[0] = 0x43;
        meta.extend_from_slice(&hash);
        meta
    }

    fn rec(kind: u8, root: u8, user_data: u8, slot: u8) -> [u8; 32] {
        let mut r = [0u8; 32];
        r[0] = 0x41;
        r[4] = kind;
        r[5] = root;
        r[20..23].copy_from_slice(&[4, user_data, slot]);
        r
    }

    /// `v_interp_p1_f32 v2, v0, attr0.x`, `v_interp_p2_f32 v2, v1, attr0.x`, then `exp mrt0` of
    /// v2 in x and `s_endpgm`: a pixel shader of one attribute and one colour.
    fn pixel_code() -> Vec<u8> {
        words(&[0xC808_0000, 0xC809_0001, 0xF800_0801, 0x0000_0002, ENDPGM])
    }

    #[test]
    fn a_pixel_shaders_attributes_and_exports_are_inferred() {
        let meta = [0u8; 0x10];
        let (p, (ins, outs)) = infer(HwStage::Ps, &pixel_code(), &meta).unwrap();
        let StageIo::Ps { inputs, input_addr, .. } = &p.io else {
            panic!("not a pixel program");
        };
        assert_eq!((inputs.len(), inputs[0].attr, inputs[0].mask), (1, 0, 1));
        // Perspective-centre barycentrics only.
        assert_eq!(*input_addr, 2);
        assert_eq!(ins.len(), 1);
        assert_eq!(
            (outs[0].name.as_str(), outs[0].kind, outs[0].row),
            ("SV_Target", 16, Some(0))
        );
        let c = container(HwStage::Ps, &pixel_code(), &meta).unwrap();
        let c = crate::Container::parse(&c).unwrap();
        assert!(c.part(FourCC::ISG1).is_some() && c.part(FourCC::OSG1).is_some());
        let part = c.part(FourCC::ACCC).unwrap();
        assert_eq!(program(part.data).unwrap().stage, HwStage::Ps);
    }

    #[test]
    fn wave64_inference_uses_isa_signals_not_formatted_vcc_lo() {
        let meta = [0u8; 0x10];
        // Implicit-VCC VALU ops (v_cmp_lt_f32_e32 + v_cndmask_b32_e32) format as `vcc_lo` in
        // `text(false)`, yet must default to wave64=true.
        let implicit_vcc = words(&[0x7C02_0300, 0x0204_0903, ENDPGM]);
        assert!(infer(HwStage::Ps, &implicit_vcc, &meta).unwrap().0.wave64);

        // Wave64 saveexec_b64.
        let w64_saveexec = words(&[0xBE80_246A, ENDPGM]);
        assert!(infer(HwStage::Ps, &w64_saveexec, &meta).unwrap().0.wave64);

        // Wave64 split exec_lo + exec_hi moves (exec_hi wins over exec_lo).
        let w64_split = words(&[0xBE80_037E, 0xBE81_037F, ENDPGM]);
        assert!(infer(HwStage::Ps, &w64_split, &meta).unwrap().0.wave64);

        // Wave64 vcc_hi move.
        let w64_vcc_hi = words(&[0xBE80_036B, ENDPGM]);
        assert!(infer(HwStage::Ps, &w64_vcc_hi, &meta).unwrap().0.wave64);

        // Wave32 saveexec_b32 (does not name exec_lo/vcc_lo in text!).
        let w32_saveexec = words(&[0xBE80_3C01, ENDPGM]);
        assert!(!infer(HwStage::Ps, &w32_saveexec, &meta).unwrap().0.wave64);

        // Wave32 SOP touching exec_lo.
        let w32_exec_lo = words(&[0x877E_007E, ENDPGM]);
        assert!(!infer(HwStage::Ps, &w32_exec_lo, &meta).unwrap().0.wave64);

        // Wave32 saveexec_b32 with vcc_hi used as a scratch SGPR: strong wave32 wins.
        let w32_with_vcc_hi = words(&[0xBE80_3C01, 0xBEEB_0302, ENDPGM]);
        assert!(!infer(HwStage::Ps, &w32_with_vcc_hi, &meta).unwrap().0.wave64);

        // s_mov_b64 vcc, s[2:3] alone: wave64 (the default holds).
        let vcc_pair = words(&[0xBEEA_0402, ENDPGM]);
        assert!(infer(HwStage::Ps, &vcc_pair, &meta).unwrap().0.wave64);
        // Beside s_and_saveexec_b32 the VCC pair is a spare SGPR pair: wave32.
        let w32_vcc_pair = words(&[0xBE80_3C01, 0xBEEA_0402, ENDPGM]);
        assert!(!infer(HwStage::Ps, &w32_vcc_pair, &meta).unwrap().0.wave64);
    }

    #[test]
    fn pixel_system_inputs_and_multiple_barycentrics_populate_isg1() {
        let meta = [0u8; 0x10];
        // Two barycentric pairs (v0..v3) + Position (v4..v7) + FrontFace (v8) + Ancillary (v9) +
        // SampleCoverage (v10); v20 is written via v_mov_b32 (not live-in!).
        let code = words(&[
            0xBEFC_0300, // s_mov_b32 m0, s0
            0xC830_0000, // v_interp_p1_f32 v12, v0, attr0.x
            0xC831_0001, // v_interp_p2_f32 v12, v1, attr0.x
            0xC834_0002, // v_interp_p1_f32 v13, v2, attr0.x
            0xC835_0003, // v_interp_p2_f32 v13, v3, attr0.x
            0x0618_090C, // v_add_f32 v12, v12, v4
            0x0618_110C, // v_add_f32 v12, v12, v8
            0x0618_130C, // v_add_f32 v12, v12, v9
            0x0618_150C, // v_add_f32 v12, v12, v10
            0x7E28_02F2, // v_mov_b32 v20, 1.0
            0xF800_180F,
            0x1414_0D0C, // exp mrt0 v12, v13, v20, v20 done vm
            ENDPGM,
        ]);
        let (p, (ins, _)) = infer(HwStage::Ps, &code, &meta).unwrap();
        let StageIo::Ps { input_addr, .. } = p.io else {
            panic!("not PS");
        };
        // Barycentrics bits 1 & 2 (v0..v3), plus bits 8..=14 (v4..v10).
        let want_addr = (1 << 1) | (1 << 2) | (0x7F << 8);
        assert_eq!(input_addr, want_addr);
        // ISG1 declares TEXCOORD0 + SV_Position + SV_IsFrontFace + SV_SampleIndex + SV_Coverage.
        let kinds: Vec<(u8, Option<u32>, u8)> =
            ins.iter().map(|e| (e.kind, e.row, e.cols)).collect();
        assert_eq!(
            kinds,
            [
                (0, Some(0), 4), // TEXCOORD: four channels, as the vertex side declares them
                (3, Some(1), 4),
                (13, Some(2), 1),
                (12, None, 1),
                (14, None, 1),
            ]
        );
    }

    #[test]
    fn entry_vgpr_read_before_interp_dst_overwrite_stays_live_in() {
        let meta = [0u8; 0x10];
        // v2 is read before v_interp_p1_f32 overwrites v2: must still be counted as Position.x.
        let code = words(&[
            0xD503_0004,
            0x0001_E502, // v_add_f32_e64 v4, v2, 1.0
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0404_0402, // exp mrt0 v2, v4, v4, v4 done vm
            ENDPGM,
        ]);
        let (p, (ins, _)) = infer(HwStage::Ps, &code, &meta).unwrap();
        let StageIo::Ps { input_addr, .. } = p.io else {
            panic!("not PS");
        };
        assert_eq!(input_addr, (1 << 1) | (1 << 8));
        assert!(ins.iter().any(|e| e.kind == 3 && e.name == "SV_Position"));
    }

    #[test]
    fn pops_pixel_shader_enables_db_shader_control() {
        let meta = [0u8; 0x10];
        let pops_const = words(&[0x8102_81EF, ENDPGM]); // s_add_i32 s2, src_pops_exiting_wave_id, 1
        let pops_msg = words(&[0xBF90_0007, ENDPGM]); // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        for code in [pops_const, pops_msg] {
            let (p, _) = infer(HwStage::Ps, &code, &meta).unwrap();
            let StageIo::Ps { db_shader_control, .. } = p.io else {
                panic!("not PS");
            };
            assert_eq!(db_shader_control, 1 << 16);
        }
        let (plain, _) = infer(HwStage::Ps, &pixel_code(), &meta).unwrap();
        let StageIo::Ps { db_shader_control, .. } = plain.io else {
            panic!("not PS");
        };
        assert_eq!(db_shader_control, 0);
    }

    #[test]
    fn vs_infers_heaps_avoids_vertex_table_collision_and_expands_constants() {
        // Metadata: Constants at s4 (slot 0), Table at s8 (slot 1) -> Constants expands to 4 SGPRs.
        let meta = meta_with(&[rec(2, 0, 4, 0), rec(0, 3, 8, 1)]);
        let code = words(&[
            0xBE8C_030A, // s_mov_b32 s12, s10
            0xBE8D_0380, // s_mov_b32 s13, 0
            0xF40C_0406,
            0x1200_0000, // s_load_dwordx8 s[16:23], s[12:13], s9
            0xBE8E_030B, // s_mov_b32 s14, s11
            0xBE8F_0380, // s_mov_b32 s15, 0
            0xF408_0607,
            0xFA00_0010, // s_load_dwordx4 s[24:27], s[14:15], 0x10
            0xF09C_0F08,
            0x00C4_0400, // image_sample_lz v[4:7], v[0:1], s[16:23], s[24:27] dmask:0xf dim:2D
            0xF800_08CF,
            0x0706_0504, // exp pos0 v4, v5, v6, v7 done
            ENDPGM,
        ]);
        let (p, _) = infer(HwStage::Vs, &code, &meta).unwrap();
        assert!(p.layout.iter().any(|r| r.root == RootType::Constants && r.user_data == 4 && r.sgprs == 4));
        assert!(p.layout.iter().any(|r| r.root == RootType::ResourceHeap && r.user_data == 10));
        assert!(p.layout.iter().any(|r| r.root == RootType::SamplerHeap && r.user_data == 11));
        assert!(p.layout.iter().any(|r| r.slot == VERTEX_TABLE && r.user_data == 0));

        // When metadata occupies s0, VERTEX_TABLE must not collide with s0, and an indexed vertex
        // fetch from s[2:3] infers VERTEX_TABLE at s[2:3].
        let meta_s0 = meta_with(&[rec(0, 3, 0, 0)]);
        let vs_vt2 = words(&[
            0xF408_0201,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[2:3], 0x0
            0xEA52_2000,
            0x8002_0100, // tbuffer_load_format_xyz v[1:3], v0, s[8:11], 0 format:32_32_32_FLOAT idxen
            0xF800_08CF,
            0x0303_0201, // exp pos0 v1, v2, v3, v3 done
            ENDPGM,
        ]);
        let (p2, _) = infer(HwStage::Vs, &vs_vt2, &meta_s0).unwrap();
        assert!(p2.layout.iter().any(|r| r.root == RootType::Table && r.user_data == 0));
        assert!(p2.layout.iter().any(|r| r.slot == VERTEX_TABLE && r.user_data == 2));
        assert!(!p2.layout.iter().any(|r| r.slot == VERTEX_TABLE && r.user_data == 0));

        // Genuinely unknown SGPR address (non-descriptor s_load_dword from s[6:7]) is NOT promoted
        // to a root parameter; it tracks as a title-memory address.
        let unknown = words(&[
            0xF400_0203,
            0xFA00_0000, // s_load_dword s8, s[6:7], 0x0
            0xF800_08CF,
            0x0000_0000, // exp pos0 v0, v0, v0, v0 done
            ENDPGM,
        ]);
        let (p3, _) = infer(HwStage::Vs, &unknown, &meta_s0).unwrap();
        let ir = crate::ir::build(&p3.code, p3.wave64, p3.vgprs).unwrap();
        let r = crate::resource::track(&ir, &p3.layout, p3.stage, None).unwrap();
        assert!(
            r.scalar
                .iter()
                .any(|&(_, a)| a.param == crate::resource::MEMORY),
            "{r:?}"
        );
    }

    /// A mesh-style vertex program as the title's start: lane 0 alone (EXEC 1) takes the group
    /// id from v5 by a shift of 7, then every lane exports a position and a primitive.
    pub(crate) fn ngg_code() -> Vec<u8> {
        words(&[
            0xBEFE_0481, // s_mov_b64 exec, 1
            0x2C0A_0A87, // v_lshrrev_b32 v5, 7, v5
            0xBEFE_04C1, // s_mov_b64 exec, -1
            0xF800_08CF,
            0x0000_0000, // exp pos0 v0, v0, v0, v0 done
            0xF800_0941,
            0x0000_0001, // exp prim v1 done
            ENDPGM,
        ])
    }

    #[test]
    fn a_vertex_program_exporting_primitives_is_a_mesh_program() {
        let (p, _) = infer(HwStage::Vs, &ngg_code(), &meta_with(&[])).unwrap();
        assert_eq!(p.stage, HwStage::Ms);
        let StageIo::Ms(ms) = &p.io else { panic!("not a mesh program: {:?}", p.io) };
        assert_eq!((ms.threads, ms.max_vertices, ms.max_primitives, ms.topology), (128, 128, 128, 2));
        // Translated as a mesh shader; its EXEC of a constant 1 is lane 0 of each console wave.
        let c = container(HwStage::Vs, &ngg_code(), &meta_with(&[])).unwrap();
        let options = super::super::Options { wave32_host: true, zero_copy_memory: true, ..Default::default() };
        let h = super::super::translate_with(&c, options).unwrap();
        assert_eq!(h.stage, HwStage::Ms);
    }

    /// A guessed pixel program drawn into an unsigned-integer target (`Options::uint_targets`)
    /// declares that colour output a 32-bit unsigned integer, the type Metal pairs with such an
    /// attachment, and stores the bits it exports. A compressed (16-bit) export to it stays
    /// refused: its halves are widened as floats.
    #[test]
    fn a_pixel_programs_output_to_an_unsigned_integer_target_is_an_unsigned_integer() {
        let program = |export: [u32; 2]| {
            let mut code = words(&[0x7E00_02FF, 0x0000_002A]); // v_mov_b32 v0, 42
            code.extend(words(&export));
            code.extend(words(&[ENDPGM]));
            container(HwStage::Ps, &code, &meta_with(&[])).unwrap()
        };
        let plain = program([0xF800_180F, 0x0000_0000]); // exp mrt0 v0, v0, v0, v0 done vm
        let options = |uint_targets| super::super::Options { wave32_host: true, zero_copy_memory: true, uint_targets, ..Default::default() };
        let comp = |o| {
            let h = super::super::translate_with(&plain, o).unwrap();
            let c = super::super::Container::parse(&h.dxil).unwrap();
            let outs = super::super::signature(c.part(super::super::FourCC::OSG1).unwrap().data).unwrap();
            outs.iter().find(|e| e.kind == 16 && e.index == 0).expect("SV_Target0").comp
        };
        assert_eq!(comp(options(0)), 9, "a float by default");
        assert_eq!(comp(options(1)), 5, "an unsigned integer for an unsigned-integer target");
        let packed = program([0xF800_1C03, 0x0000_0100]); // exp mrt0 v0, v0 done compr vm
        let Err(err) = super::super::translate_with(&packed, options(1)) else { panic!("16-bit halves translated") };
        let err = err.to_string();
        assert!(err.contains("a 16-bit export to an integer output"), "{err}");
    }

    /// The vertex buffer table (s0/s1) is no root argument: where root views are title memory
    /// (a vertex program with `zero_copy_memory`), its address is not read from user data, which
    /// no host could fill (the fetch then read from address 0, and nothing was drawn).
    #[test]
    fn the_vertex_table_is_not_read_from_user_data_where_root_views_are_title_memory() {
        let code = words(&[
            0xBE82_0300, // s_mov_b32 s2, s0
            0xBE83_0301, // s_mov_b32 s3, s1
            0xF408_0201,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[2:3], 0x0
            0xEA52_2000,
            0x8002_0100, // tbuffer_load_format_xyz v[1:3], v0, s[8:11], 0 format:32_32_32_FLOAT idxen
            0xF800_08CF,
            0x0303_0201, // exp pos0 v1, v2, v3, v3 done
            ENDPGM,
        ]);
        let (p, _) = infer(HwStage::Vs, &code, &meta_with(&[])).unwrap();
        assert!(p.layout.iter().any(|r| r.slot == VERTEX_TABLE && r.user_data == 0));
        let c = container(HwStage::Vs, &code, &meta_with(&[])).unwrap();
        for zero_copy_memory in [false, true] {
            let options = super::super::Options { wave32_host: true, zero_copy_memory, ..Default::default() };
            let h = super::super::translate_with(&c, options).unwrap();
            assert!(h.user_data.iter().all(|u| u.slot != VERTEX_TABLE), "{zero_copy_memory}: {:?}", h.user_data);
            assert!(!h.memory);
        }
    }

    /// The vertex buffer table's address is only an address: a program that uses s0 as a number
    /// (exports it, as it is, copied, or with an offset added) is refused, with and without
    /// `zero_copy_memory`. No host has a value for it.
    #[test]
    fn the_vertex_tables_address_used_as_a_number_is_refused() {
        let exported = [
            0x7E02_0200, // v_mov_b32 v1, s0
            0xF800_08CF,
            0x0101_0101, // exp pos0 v1, v1, v1, v1 done
            ENDPGM,
        ];
        let copied = [
            0xBE82_0300, // s_mov_b32 s2, s0
            0x7E02_0202, // v_mov_b32 v1, s2
            0xF800_08CF,
            0x0101_0101, // exp pos0 v1, v1, v1, v1 done
            ENDPGM,
        ];
        let offset = [
            0x8002_9000, // s_add_u32 s2, s0, 16
            0x7E02_0202, // v_mov_b32 v1, s2
            0xF800_08CF,
            0x0101_0101, // exp pos0 v1, v1, v1, v1 done
            ENDPGM,
        ];
        for (name, code) in [("exported", &exported[..]), ("copied", &copied[..]), ("offset", &offset[..])] {
            let code = words(code);
            let (p, _) = infer(HwStage::Vs, &code, &meta_with(&[])).unwrap();
            assert!(p.layout.iter().any(|r| r.slot == VERTEX_TABLE && r.user_data == 0), "{name}");
            let c = container(HwStage::Vs, &code, &meta_with(&[])).unwrap();
            for zero_copy_memory in [false, true] {
                let options = super::super::Options { wave32_host: true, zero_copy_memory, ..Default::default() };
                let e = super::super::translate_with(&c, options).err().map(|e| e.to_string());
                assert!(
                    e.as_deref().is_some_and(|e| e.contains("a root descriptor")),
                    "{name}, zero_copy_memory {zero_copy_memory}: {e:?}"
                );
            }
        }
    }

    /// Only parameters that may hold two f16 halves get `HALF` outputs, those that can hold
    /// nothing else first and their `TEXCOORD` last: param0 is a 32-bit float product (plain),
    /// param1 a literal of two f16s (either), param2 a `v_cvt_pkrtz_f16_f32` or 0 (halves).
    #[test]
    fn halves_go_to_parameters_that_may_hold_them() {
        let code = [
            0x7E02_02F2, // v_mov_b32 v1, 1.0
            0x1004_0301, // v_mul_f32 v2, v1, v1
            0x7E06_02FF, 0x3C00_3800, // v_mov_b32 v3, 0x3c003800
            0x5E08_0301, // v_cvt_pkrtz_f16_f32 v4, v1, v1
            0x7D84_0281, // v_cmp_eq_u32 vcc, 1, v1
            0x020A_0880, // v_cndmask_b32 v5, 0, v4, vcc
            0xF800_0201, 0x0000_0002, // exp param0 v2
            0xF800_0211, 0x0000_0003, // exp param1 v3
            0xF800_0221, 0x0000_0005, // exp param2 v5
            0xF800_08CF, 0x0101_0101, // exp pos0 v1, v1, v1, v1 done
            ENDPGM,
        ];
        let (_, (_, outs)) = infer(HwStage::Vs, &words(&code), &meta_with(&[])).unwrap();
        let named = |name: &str| -> Vec<u32> { outs.iter().filter(|o| o.name == name).map(|o| o.index).collect() };
        assert_eq!(named(HALF), [4, 5, 2, 3], "{outs:?}");
        assert_eq!(named("TEXCOORD"), [0, 1, 2], "{outs:?}");
        let k = packed_params(&crate::ir::build(&words(&code), true, 256).unwrap());
        assert_eq!((k[&0], k[&1], k[&2]), (Packing::Plain, Packing::Unknown, Packing::Halves));
    }

    /// A parameter whose last write zeroes one half of its dword and keeps the other (`v_mov_b32
    /// v2, 0 dst_sel:WORD_1` after `v_cvt_f16_f32 v2, v1`) still holds halves: the bits the write
    /// keeps are the half. Judged by the write alone the parameter is plain 0 and the vertex
    /// program exports no `HALF` rows, so the pixel program's `user(half10)` links to nothing (the
    /// world's particle pass, refused by the converter).
    #[test]
    fn a_half_kept_below_a_zeroed_word_still_counts() {
        let code = [
            0x7E02_02F2, // v_mov_b32 v1, 1.0
            0x7E04_1501, // v_cvt_f16_f32 v2, v1
            0x7E04_02F9, 0x0086_1580, // v_mov_b32 v2, 0 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
            0xF800_0201, 0x0000_0002, // exp param0 v2
            0xF800_08CF, 0x0101_0101, // exp pos0 v1, v1, v1, v1 done
            ENDPGM,
        ];
        let k = packed_params(&crate::ir::build(&words(&code), true, 256).unwrap());
        assert_eq!(k[&0], Packing::Halves);
        let (_, (_, outs)) = infer(HwStage::Vs, &words(&code), &meta_with(&[])).unwrap();
        let named = |name: &str| -> Vec<u32> { outs.iter().filter(|o| o.name == name).map(|o| o.index).collect() };
        assert_eq!(named(HALF), [0, 1], "{outs:?}");
    }

    /// A parameter its pixel shader reads and the vertex program does not write is written too
    /// (`Options::extra_params`), even when a `HALF` element of another parameter has its index:
    /// the converter refused the pipeline (`user(texcoord8)` not written by the vertex shader).
    #[test]
    fn an_extra_parameter_is_written_beside_a_half_of_its_index() {
        let code = [
            0x7E02_02F2, // v_mov_b32 v1, 1.0
            0x7E06_02FF, 0x3C00_3800, // v_mov_b32 v3, 0x3c003800
            0xF800_0201, 0x0000_0001, // exp param0 v1
            0xF800_0211, 0x0000_0003, // exp param1 v3 (HALF2, HALF3)
            0xF800_08CF, 0x0101_0101, // exp pos0 v1, v1, v1, v1 done
            ENDPGM,
        ];
        let c = container(HwStage::Vs, &words(&code), &meta_with(&[])).unwrap();
        let options = super::super::Options { extra_params: 1 << 2, ..Default::default() };
        let h = super::super::translate_with(&c, options).unwrap();
        let c = crate::Container::parse(&h.dxil).unwrap();
        let outs = super::super::signature(c.part(FourCC::OSG1).unwrap().data).unwrap();
        let named = |name: &str| -> Vec<u32> { outs.iter().filter(|o| o.name == name).map(|o| o.index).collect() };
        assert_eq!(named(HALF), [2, 3], "{outs:?}");
        assert_eq!(named("TEXCOORD"), [0, 1, 2], "{outs:?}");
    }

    /// Translates `code` as a guessed vertex program with root constants in user data 0 (s8 of a
    /// mesh program).
    fn mesh(code: &[u32]) -> Result<super::super::HostShader, ShaderError> {
        let c = container(HwStage::Vs, &words(code), &meta_with(&[rec(2, 0, 0, 0)])).unwrap();
        let options = super::super::Options { wave32_host: true, zero_copy_memory: true, ..Default::default() };
        super::super::translate_with(&c, options)
    }

    #[test]
    fn a_mesh_mask_join_reads_each_lanes_bit_of_the_scalar_pair() {
        let code = [
            0xBEFE_0481, // s_mov_b64 exec, 1
            0x2C0A_0A87, // v_lshrrev_b32 v5, 7, v5 (128 mesh threads)
            0xBEFE_04C1, // s_mov_b64 exec, -1
            0xD765_0000, 0x0201_00C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
            0x8014_8108, // s_add_u32 s20, s8, 1
            0x8095_8108, // s_sub_u32 s21, s8, 1
            0xBF06_8008, // s_cmp_eq_u32 s8, 0
            0xBF85_0002, // s_cbranch_scc1 join
            0xD4C4_0014, 0x0002_0089, // v_cmp_gt_u32 s[20:21], 9, v0
            0xD501_0001, 0x0051_0280, // join: v_cndmask_b32 v1, 0, 1, s[20:21]
            0xF800_08CF, 0x0101_0101, // exp pos0 v1, v1, v1, v1 done
            0xF800_0941, 0x0000_0001, // exp prim v1 done
            ENDPGM,
        ];
        let shader = mesh(&code).unwrap();
        assert_eq!(shader.stage, HwStage::Ms);

        let mut missing_high = code;
        missing_high[6] = 0xBF80_0000; // s_nop 0: s21 is an unknown hardware input
        let err = mesh(&missing_high).unwrap_err();
        assert!(err.to_string().contains("high dword is not known"), "{err}");

        // Wave32 needs only s20, even when s21 has no value on the data edge.
        let mut wave32 = missing_high;
        wave32[0] = 0xBEFE_0381; // s_mov_b32 exec_lo, 1
        wave32[2] = 0xBEFE_03C1; // s_mov_b32 exec_lo, -1
        mesh(&wave32).unwrap();

        // Each 32-lane host wave can read a different first lane of v0.
        let mut split = code.to_vec();
        split.splice(7..8, [
            0x7E14_0500, // v_readfirstlane_b32 s10, v0
            0xBF06_800A, // s_cmp_eq_u32 s10, 0
        ]);
        let err = mesh(&split).unwrap_err();
        assert!(err.to_string().contains("lane-mask join"), "{err}");
        let mut split_high = code;
        split_high[6] = 0x7E2A_0500; // v_readfirstlane_b32 s21, v0
        let err = mesh(&split_high).unwrap_err();
        assert!(err.to_string().contains("lane-mask join"), "{err}");

        // Both arms jump to the join, but different lanes enter them. The
        // compare before the fork can include lanes that take the data arm.
        let divergent = [
            &code[..5],
            &[
                0xD4C4_000A, 0x0002_0089, // v_cmp_gt_u32 s[10:11], 9, v0
                0x7DA8_0084, // v_cmpx_gt_u32 4, v0
                0xBF88_0003, // s_cbranch_execz mask
                code[5], code[6], // data arm
                0xBF82_0001, // s_branch join
                0xBE94_040A, // mask: s_mov_b64 s[20:21], s[10:11]
                0xBEFE_04C1, // join: s_mov_b64 exec, -1
            ],
            &code[11..],
        ].concat();
        let err = mesh(&divergent).unwrap_err();
        assert!(err.to_string().contains("lane-mask join"), "{err}");
    }

    /// The title's mesh programs read root constants of two dwords (b500, user data 13 and 14 in
    /// their metadata, which gives one) just below the descriptor heap's SGPR (15, inferred).
    /// The constants reach up to the heap, so the second dword is user data, not a hardware SGPR.
    #[test]
    fn root_constants_reach_up_to_an_inferred_heap() {
        let code = [
            0xBEFE_0481, // s_mov_b64 exec, 1
            0x2C0A_0A87, // v_lshrrev_b32 v5, 7, v5
            0xBEFE_04C1, // s_mov_b64 exec, -1
            0xBE8B_0380, // s_mov_b32 s11, 0
            0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
            0x8010_0908, // s_add_u32 s16, s8, s9
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE030_0000, 0x1003_0100, // buffer_load_dword v1, off, s[12:15], s16
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xF800_08CF, 0x0101_0101, // exp pos0 v1, v1, v1, v1 done
            0xF800_0941, 0x0000_0001, // exp prim v1 done
            ENDPGM,
        ];
        let meta = meta_with(&[rec(2, 0, 0, 0)]);
        let (p, _) = infer(HwStage::Vs, &words(&code), &meta).unwrap();
        assert!(p.layout.iter().any(|r| r.root == RootType::ResourceHeap && r.user_data == 2), "{:?}", p.layout);
        assert!(p.layout.iter().any(|r| r.root == RootType::Constants && r.user_data == 0 && r.sgprs == 2), "{:?}", p.layout);
        mesh(&code).unwrap();
    }

    /// The title's upscale pixel program reads its b500 root constants as four dwords, s9 to s12,
    /// from a metadata record that names only s9 (and interpolates nothing). The constants take
    /// the SGPRs after them that the program reads on entry, so s10 to s12 are user data, not
    /// hardware SGPR inputs.
    #[test]
    fn a_pixel_programs_last_root_constants_take_the_sgprs_it_reads_after_them() {
        let code = [
            0x7E00_0209, // v_mov_b32 v0, s9
            0x7E02_020A, // v_mov_b32 v1, s10
            0x7E04_020B, // v_mov_b32 v2, s11
            0x7E06_020C, // v_mov_b32 v3, s12
            0xF800_080F, 0x0302_0100, // exp mrt0 v0, v1, v2, v3 done
            ENDPGM,
        ];
        let meta = meta_with(&[rec(0, 3, 8, 0), rec(2, 0, 9, 1)]);
        let (p, _) = infer(HwStage::Ps, &words(&code), &meta).unwrap();
        assert!(p.layout.iter().any(|r| r.root == RootType::Constants && r.user_data == 9 && r.sgprs == 4), "{:?}", p.layout);
        let c = container(HwStage::Ps, &words(&code), &meta).unwrap();
        super::super::translate_with(&c, super::super::Options::default()).unwrap();
        // An interpolating program's primitive mask, which follows its user SGPRs and only
        // reaches M0, is not taken.
        let interp = [
            0x7E00_0209, // v_mov_b32 v0, s9
            0xBEFC_030A, // s_mov_b32 m0, s10
            0xC808_0000, 0xC809_0001, // v_interp_p1_f32 v2, v0, attr0.x; v_interp_p2_f32 v2, v1, attr0.x
            0xF800_0803, 0x0000_0200, // exp mrt0 v0, v2 done
            ENDPGM,
        ];
        let (p, _) = infer(HwStage::Ps, &words(&interp), &meta).unwrap();
        assert!(p.layout.iter().any(|r| r.root == RootType::Constants && r.user_data == 9 && r.sgprs == 1), "{:?}", p.layout);
        // Constants the program reads past SGPRs it writes first (s10, s11) reach up to its
        // primitive mask (s13): s12 is theirs (two of the title's G-buffer pixel programs).
        let gap = [
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0x7E06_020C, // v_mov_b32 v3, s12
            0xBEFC_030D, // s_mov_b32 m0, s13
            0xC808_0000, 0xC809_0001, // v_interp_p1_f32 v2, v0, attr0.x; v_interp_p2_f32 v2, v1, attr0.x
            0xF800_0803, 0x0000_0203, // exp mrt0 v3, v2 done
            ENDPGM,
        ];
        let (p, _) = infer(HwStage::Ps, &words(&gap), &meta).unwrap();
        assert!(p.layout.iter().any(|r| r.root == RootType::Constants && r.user_data == 9 && r.sgprs == 4), "{:?}", p.layout);
        let c = container(HwStage::Ps, &words(&gap), &meta).unwrap();
        super::super::translate_with(&c, super::super::Options::default()).unwrap();
    }

    /// `s_sext_i32_i16` and `s_sext_i32_i8` (some of the title's mesh programs) translate.
    #[test]
    fn scalar_sign_extensions_translate() {
        mesh(&[
            0xBEFE_0481, // s_mov_b64 exec, 1
            0x2C0A_0A87, // v_lshrrev_b32 v5, 7, v5
            0xBEFE_04C1, // s_mov_b64 exec, -1
            0xBE89_1A08, // s_sext_i32_i16 s9, s8
            0xBE8A_1909, // s_sext_i32_i8 s10, s9
            0x7E02_020A, // v_mov_b32 v1, s10
            0xF800_08CF, 0x0101_0101, // exp pos0 v1, v1, v1, v1 done
            0xF800_0941, 0x0000_0001, // exp prim v1 done
            ENDPGM,
        ])
        .unwrap();
    }

    /// The title's G-buffer pixel programs compare a value with its quad neighbour's
    /// (`v_cmp_gt_f32 vcc, v9, -v9 quad_perm:[2,3,0,1]`): a compare with DPP reads src0 from the
    /// lane the pattern names, as other DPP operations do.
    #[test]
    fn a_pixel_program_compares_with_dpp() {
        let code = [
            0x7E04_02F2, // v_mov_b32 v2, 1.0
            0x7C08_04FA, 0xFF4C_4E02, // v_cmp_gt_f32 vcc, v2, -v2 quad_perm:[2,3,0,1] bound_ctrl:1 fi:1
            0xD501_0003, 0x01A9_E480, // v_cndmask_b32 v3, 0, 1.0, vcc
            0xF800_180F, 0x0303_0303, // exp mrt0 v3, v3, v3, v3 done vm
            ENDPGM,
        ];
        let c = container(HwStage::Ps, &words(&code), &meta_with(&[])).unwrap();
        let options = super::super::Options { wave32_host: true, ..Default::default() };
        super::super::translate_with(&c, options).unwrap();
    }

    #[test]
    fn pixel_compares_with_16bit_dpp_and_64bit_or_partial_mask_refusals() {
        let options = super::super::Options { wave32_host: true, ..Default::default() };
        let run = |cmp: [u32; 2], restore_exec: bool| {
            let mut code = vec![
                0x7E04_02FF,
                0x0000_3C00, // v_mov_b32 v2, 0x3c00 (1.0f16)
                0x7E06_02FF,
                0x0000_0204, // v_mov_b32 v3, 0x204 (+inf | -inf class mask)
            ];
            if restore_exec {
                code.extend([
                    0x7E06_0280, // v_mov_b32 v3, 0
                    0xBE84_037E, // s_mov_b32 s4, exec_lo
                    cmp[0],
                    cmp[1],
                    0x7E06_02F2, // v_mov_b32 v3, 1.0
                    0xBEFE_0304, // s_mov_b32 exec_lo, s4
                    0xF800_180F,
                    0x0303_0303, // exp mrt0 v3, v3, v3, v3 done vm
                    ENDPGM,
                ]);
            } else {
                code.extend([
                    cmp[0],
                    cmp[1],
                    0xD501_0003,
                    0x01A9_E480, // v_cndmask_b32 v3, 0, 1.0, vcc
                    0xF800_180F,
                    0x0303_0303, // exp mrt0 v3, v3, v3, v3 done vm
                    ENDPGM,
                ]);
            }
            let c = container(HwStage::Ps, &words(&code), &meta_with(&[])).unwrap();
            super::super::translate_with(&c, options)
        };
        // Full-mask 16-bit DPP compares (`v_cmp_gt_f16`, `v_cmp_lt_i16`, `v_cmp_class_f16`,
        // `v_cmpx_eq_u16`) lower through `Lower::half_src`.
        run([0x7D98_04FA, 0xFF4C_4E02], false).expect("v_cmp_gt_f16 dpp");
        run([0x7D12_04FA, 0xFF08_1102], false).expect("v_cmp_lt_i16 row_shr:1 bound_ctrl:1");
        run([0x7D1E_06FA, 0xFF0C_1B02], false).expect("v_cmp_class_f16 dpp");
        run([0x7D74_04FA, 0xFF0C_B102], true).expect("v_cmpx_eq_u16 dpp");

        // 64-bit compare with DPP and partial `row_mask`/`bank_mask` compares stay refused.
        let err64 = run([0x7DC4_04FA, 0xFF0C_B102], false).unwrap_err();
        assert!(err64.to_string().contains("a 64-bit compare with DPP"), "{err64}");
        let err_mask = run([0x7D98_04FA, 0x530C_4E02], false).unwrap_err();
        assert!(
            err_mask.to_string().contains("a compare with DPP row masks or bounds"),
            "{err_mask}"
        );
    }

    /// The title's G-buffer mesh programs test a uniform value as a lane mask: in the lanes of
    /// an EXEC region (lane < count), a VALU compare of an SGPR, ANDed with EXEC, its SCC kept in
    /// an SGPR and branched on later. The compare holds the same bit in every lane, so SCC (some
    /// lane set) is that bit, in every lane.
    #[test]
    fn a_mesh_program_keeps_the_scc_of_a_uniform_lane_mask() {
        // EXEC as a v_cmpx sets it, or as the join of two lane conditions.
        let cmpx: &[u32] = &[0x7DA8_0008]; // v_cmpx_gt_u32 s8, v0
        let join: &[u32] = &[
            0xD4C4_001E, 0x0002_0008, // v_cmp_gt_u32 s[30:31], s8, v0
            0xD4C1_0020, 0x0002_0208, // v_cmp_lt_u32 s[32:33], s8, v1
            0x88FE_201E, // s_or_b64 exec, s[30:31], s[32:33]
        ];
        let program = |exec: &[u32], cmp: [u32; 2]| {
            let mut code = vec![
                0xBEFE_0481, // s_mov_b64 exec, 1
                0x2C0A_0A87, // v_lshrrev_b32 v5, 7, v5
                0xBEFE_04C1, // s_mov_b64 exec, -1
            ];
            code.extend(exec);
            code.extend([
                0xBF88_0007, // s_cbranch_execz 7
                cmp[0], cmp[1], // v_cmp_ge_f32 vcc, <src0>, 0
                0x8796_7E6A, // s_and_b64 s[22:23], vcc, exec
                0xBE94_03FD, // s_mov_b32 s20, src_scc
                0xBF01_8014, // s_cmp_lg_i32 s20, 0
                0xBF84_0001, // s_cbranch_scc0 1
                0x7E00_02F2, // v_mov_b32 v0, 1.0
                0xBEFE_04C1, // s_mov_b64 exec, -1
                0xF800_08CF, 0x0000_0000, // exp pos0 v0, v0, v0, v0 done
                0xF800_0941, 0x0000_0001, // exp prim v1 done
                ENDPGM,
            ]);
            code
        };
        for exec in [cmpx, join] {
            mesh(&program(exec, [0xD406_006A, 0x0001_0008])).unwrap(); // src0 s8: uniform
            // A compare of a VGPR differs between lanes: SCC is a ballot the mesh stage cannot
            // take.
            let e = mesh(&program(exec, [0xD406_006A, 0x0001_0100])).unwrap_err(); // src0 v0
            assert!(e.to_string().contains("lane mask"), "{e}");
        }
    }

    /// The NGG vertex programs ahead of the G-buffer test EXEC without such a mask: the compare
    /// ANDed with an `s_cselect` of -1 and 0 by SCC, then `s_andn2_b64 sN, exec, mask`, its SCC
    /// kept and branched on (was refused: "a lane mask read as data"). On EXEC the mask is a
    /// function of uniform atoms, so SCC (some lane of EXEC outside it) is its complement.
    #[test]
    fn a_mesh_program_keeps_the_scc_of_exec_without_a_uniform_lane_mask() {
        let program = |cmp: [u32; 2]| {
            [
                0xBEFE_0481, // s_mov_b64 exec, 1
                0x2C0A_0A87, // v_lshrrev_b32 v5, 7, v5
                0xBEFE_04C1, // s_mov_b64 exec, -1
                0x7DA8_0008, // v_cmpx_gt_u32 s8, v0
                0xBF88_000A, // s_cbranch_execz 10
                cmp[0], cmp[1], // v_cmp_ge_f32 vcc, <src0>, 0
                0xBF0D_8008, // s_bitcmp1_b32 s8, 0
                0x8598_80C1, // s_cselect_b64 s[24:25], -1, 0
                0x8798_186A, // s_and_b64 s[24:25], vcc, s[24:25]
                0x8A96_187E, // s_andn2_b64 s[22:23], exec, s[24:25]
                0xBE94_03FD, // s_mov_b32 s20, src_scc
                0xBF01_8014, // s_cmp_lg_i32 s20, 0
                0xBF84_0001, // s_cbranch_scc0 1
                0x7E00_02F2, // v_mov_b32 v0, 1.0
                0xBEFE_04C1, // s_mov_b64 exec, -1
                0xF800_08CF, 0x0000_0000, // exp pos0 v0, v0, v0, v0 done
                0xF800_0941, 0x0000_0001, // exp prim v1 done
                ENDPGM,
            ]
        };
        mesh(&program([0xD406_006A, 0x0001_0008])).unwrap(); // src0 s8: uniform
        let e = mesh(&program([0xD406_006A, 0x0001_0100])).unwrap_err(); // src0 v0
        assert!(e.to_string().contains("lane mask"), "{e}");
    }

    /// The NGG vertex programs ahead of the G-buffer spill SGPRs to VGPR lanes on two paths
    /// into a join and read the lane back after it (was refused: "a cross-lane operation in a
    /// mesh shader"): the read is a phi of the SGPRs written.
    #[test]
    fn a_mesh_program_reads_back_an_sgpr_spilled_on_two_paths() {
        mesh(&[
            0xBEFE_0481, // s_mov_b64 exec, 1
            0x2C0A_0A87, // v_lshrrev_b32 v5, 7, v5
            0xBEFE_04C1, // s_mov_b64 exec, -1
            0xBE94_03F2, // s_mov_b32 s20, 1.0
            0xBE95_0380, // s_mov_b32 s21, 0
            0xBF0D_9F08, // s_bitcmp1_b32 s8, 31
            0xBF85_0003, // s_cbranch_scc1 3
            0xD761_000E, 0x0001_0614, // v_writelane_b32 v14, s20, 3
            0xBF82_0002, // s_branch 2
            0xD761_000E, 0x0001_0615, // v_writelane_b32 v14, s21, 3
            0xD760_0016, 0x0001_070E, // v_readlane_b32 s22, v14, 3
            0x7E00_0216, // v_mov_b32 v0, s22
            0xF800_08CF, 0x0000_0000, // exp pos0 v0, v0, v0, v0 done
            0xF800_0941, 0x0000_0001, // exp prim v1 done
            ENDPGM,
        ])
        .unwrap();
    }

    /// They also swap a mask and data between two SGPRs by three XORs (was refused: "a lane
    /// mask read as data"): the last XOR gives back the data.
    #[test]
    fn a_mesh_program_swaps_a_mask_and_data_by_three_xors() {
        mesh(&[
            0xBEFE_0481, // s_mov_b64 exec, 1
            0x2C0A_0A87, // v_lshrrev_b32 v5, 7, v5
            0xBEFE_04C1, // s_mov_b64 exec, -1
            0xBE96_0308, // s_mov_b32 s22, s8
            0xBE98_047E, // s_mov_b64 s[24:25], exec
            0x8918_1618, // s_xor_b32 s24, s24, s22
            0x8916_1618, // s_xor_b32 s22, s24, s22
            0x8918_1618, // s_xor_b32 s24, s24, s22
            0x7E00_0218, // v_mov_b32 v0, s24
            0xF800_08CF, 0x0000_0000, // exp pos0 v0, v0, v0, v0 done
            0xF800_0941, 0x0000_0001, // exp prim v1 done
            ENDPGM,
        ])
        .unwrap();
    }
}
