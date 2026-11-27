//! I/O raising (decision 0003): what a program's input registers, its
//! interpolation and its exports mean.
//!
//! Pixel shaders get the VGPRs `SPI_PS_INPUT_ADDR` enables, in a fixed
//! order (barycentrics, then position, front face...). `v_interp_p1` and
//! `v_interp_p2` (and their f16 forms) evaluate an attribute from a pair of
//! barycentrics; `v_interp_mov` reads one vertex's value. After the 32-bit
//! export patch ([`NativeProgram::patched_code`]) exports carry plain
//! 32-bit values: `mrt<n>` is render target n, `mrtz` depth, stencil and
//! sample mask, `null` nothing. A lane whose EXEC bit is clear at the
//! export that ends the program (`done`) is discarded: that export's guard
//! after structurisation.
//!
//! Vertex shaders that run as the hardware VS get the vertex id in v0, the
//! primitive id in v2 and the instance id in v3; the fetch shader loads
//! channel k of each attribute into its VGPR + k. `exp pos0` is the
//! position, `exp param<n>` the output with parameter n.
//!
//! Attributes and outputs are named by their row: a register of the
//! container's signature part.
//!
//! Compute programs get the thread id in the group in v0-v2 and the group
//! id in the SGPRs after the user data, as `COMPUTE_PGM_RSRC2` enables them.
//!
//! Mesh programs run as one NGG stage. The program computes its thread
//! index from the wave index (s3 bits 24-27) and the lane (`v_mbcnt`), and
//! its group id from v5 of the first lane (`v_readfirstlane`: the group
//! times the group size). Lane n exports vertex n (`exp pos`, `exp param`)
//! and primitive n (`exp prim`: 9-bit vertex indices at bits 0, 10, 20,
//! each with an edge flag above it, bit 31 culls); `param` entries with
//! bit 7 are per-primitive outputs. One lane sends the output counts
//! (`s_sendmsg` GS_ALLOC_REQ, m0 = vertices | primitives << 12).

use crate::ShaderError;
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Phi, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage};
use crate::resource::Site;
use crate::structure::{Map, Set};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    VertexId,
    PrimitiveId,
    InstanceId,
    /// A vertex attribute the fetch shader loads.
    Attribute {
        row: u8,
        channel: u8,
    },
    /// `index` 0 is i, 1 is j (2 the pull model's w).
    Barycentric {
        kind: Barycentric,
        index: u8,
    },
    /// Pixel position channel (0-3: x, y, z, w).
    Position(u8),
    FrontFace,
    /// Sample id and other bits.
    Ancillary,
    SampleCoverage,
    FixedPosition,
    LineStipple,
    /// Compute: thread-group id component (an SGPR).
    GroupId(u8),
    /// Compute: thread id within the group, component (a VGPR).
    ThreadIdInGroup(u8),
    /// Compute: the thread-group info SGPR (wave index and count).
    GroupInfo,
    /// Mesh: v5, the group id times the group size (read as uniform).
    MeshGroup,
    /// Mesh: s3, the wave's index in the group in bits 24-27.
    MergedWaveInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Barycentric {
    PerspectiveSample,
    PerspectiveCenter,
    PerspectiveCentroid,
    PerspectivePullModel,
    LinearSample,
    LinearCenter,
    LinearCentroid,
}

/// The pixel input VGPRs in hardware order: one entry per
/// `SPI_PS_INPUT_ADDR` bit.
const PS_VGPRS: [(u8, Option<Barycentric>); 16] = [
    (2, Some(Barycentric::PerspectiveSample)),
    (2, Some(Barycentric::PerspectiveCenter)),
    (2, Some(Barycentric::PerspectiveCentroid)),
    (3, Some(Barycentric::PerspectivePullModel)),
    (2, Some(Barycentric::LinearSample)),
    (2, Some(Barycentric::LinearCenter)),
    (2, Some(Barycentric::LinearCentroid)),
    (1, None), // line stipple
    (1, None), // position x
    (1, None), // y
    (1, None), // z
    (1, None), // w
    (1, None), // front face
    (1, None), // ancillary
    (1, None), // sample coverage
    (1, None), // fixed-point position
];

fn ps_system(bit: usize) -> Input {
    match bit {
        7 => Input::LineStipple,
        8..=11 => Input::Position(bit as u8 - 8),
        12 => Input::FrontFace,
        13 => Input::Ancillary,
        14 => Input::SampleCoverage,
        _ => Input::FixedPosition,
    }
}

/// An attribute channel of the input signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attribute {
    pub usage: Usage,
    pub row: u8,
    pub channel: u8,
    /// The upper f16 half of the channel (packed fp16 inputs).
    pub high: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Interp {
    /// This site's result (`v_interp_p2*`) is the attribute interpolated
    /// with `barycentric`; `first` are the `v_interp_p1*` halves it
    /// completes (one per path where the compiler shares a first half).
    /// Where the compiler reused the i or j register on another path,
    /// `unless` says where it holds something else, which no interpolating
    /// lane may have been (emission proves it with structurisation's lane
    /// conditions).
    Interpolated {
        attribute: Attribute,
        barycentric: Barycentric,
        first: Vec<Site>,
        unless: Vec<Elsewhere>,
    },
    /// One vertex's value (`v_interp_mov_f32`), as encoded: 0 for P10,
    /// 1 for P20, 2 for P0 (the provoking vertex).
    Vertex { attribute: Attribute, vertex: u8 },
}

/// Where an interpolation's i or j register holds another value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Elsewhere {
    /// The edge (predecessor, block) into a join that brings another value.
    Edge(usize, usize),
    /// A vector instruction writing the register, in the lanes that run it.
    Write(Site),
    /// A copy into the register, in the lanes that skip it (old value).
    Skip(Site),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Color(u8),
    /// Depth, stencil, sample mask (channels as `DB_SHADER_CONTROL` enables).
    Depth,
    Null,
    Position(u8),
    Param {
        usage: Usage,
        row: u8,
    },
    /// `param0` of a vertex shader without outputs: the hardware wants one
    /// parameter export regardless.
    Unread,
    /// The primitive export of a merged or mesh stage.
    Primitive,
    /// A mesh program's per-primitive output (a row of `PSG1`).
    PrimitiveParam {
        usage: Usage,
        row: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub site: Site,
    pub target: Target,
    /// Per channel, as the instruction enables them.
    pub values: [Option<Value>; 4],
    /// Two f16 values per dword.
    pub compressed: bool,
    /// The program's last export (for pixels: kills the lanes not in EXEC).
    pub done: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Io {
    /// Input values the hardware loads, with their meaning; other VGPRs
    /// read before written hold undefined values.
    pub inputs: Vec<(Value, Input)>,
    pub interps: Vec<(Site, Interp)>,
    pub exports: Vec<Export>,
    /// Compute and pixel: how many SGPRs the hardware initializes
    /// ([`ComputeIo::initialized_sgprs`](crate::native::ComputeIo::initialized_sgprs);
    /// pixel: the user SGPRs, at least the 8 ABI slots and up to the last
    /// root parameter's, then the primitive mask and the ordered-section
    /// (POPS) collision id, as every pixel program of the title reads them).
    pub initialized_sgprs: Option<u8>,
}

/// Raises the I/O of `p`, built from `native`'s patched code. Fails for
/// stages whose stage data is not read and on constructs not covered yet.
pub fn raise(p: &Program, native: &NativeProgram) -> Result<Io, ShaderError> {
    raise_graphics(p, native, false)
}

pub(crate) fn raise_graphics(
    p: &Program,
    native: &NativeProgram,
    guessed: bool,
) -> Result<Io, ShaderError> {
    let unsupported = |offset, what| ShaderError::Unsupported { offset, what };
    let (loaded, ps_inputs, params) = match (&native.io, native.stage) {
        (
            StageIo::Ps {
                inputs, input_addr, ..
            },
            HwStage::Ps,
        ) => {
            // A pixel shader exports no parameters; `[]` never reaches `Unread`.
            let loaded = ps_vgprs(*input_addr)
                .into_iter()
                .map(|(r, i)| (Reg::V(r), i))
                .collect();
            (loaded, &inputs[..], &[][..])
        }
        (StageIo::Vs { inputs, outputs }, HwStage::Vs) => {
            let mut loaded = vec![
                (Reg::V(0), Input::VertexId),
                (Reg::V(2), Input::PrimitiveId),
                (Reg::V(3), Input::InstanceId),
            ];
            for i in inputs {
                for channel in (0..4).filter(|c| i.mask >> c & 1 != 0) {
                    let input = Input::Attribute {
                        row: i.row,
                        channel,
                    };
                    loaded.push((Reg::V(i.vgpr + channel), input));
                }
            }
            (loaded, &[][..], &outputs[..])
        }
        // Compute: thread ids in v0-v2, group ids and the group info in the
        // SGPRs after the user data.
        (StageIo::Cs(cs), HwStage::Cs) => {
            let (ids, info) = cs.system_sgprs();
            let mut loaded: Vec<(Reg, Input)> = (0..cs.thread_id_vgprs())
                .map(|c| (Reg::V(c), Input::ThreadIdInGroup(c)))
                .collect();
            for (c, s) in ids.into_iter().enumerate() {
                if let Some(s) = s {
                    loaded.push((Reg::S(s), Input::GroupId(c as u8)));
                }
            }
            loaded.extend(info.map(|s| (Reg::S(s), Input::GroupInfo)));
            (loaded, &[][..], &[][..])
        }
        (StageIo::Ms(ms), HwStage::Ms) => {
            let group = p.inputs.iter().find(|i| i.0 == Reg::V(5)).map(|i| i.1);
            // The group input holds the same value in every lane only as the
            // first lane reads it; a VGPR write reads its old value too.
            let per_lane = p.blocks.iter().flat_map(|b| &b.insts).find(|i| {
                group.is_some_and(|g| i.reads.contains(&(Reg::V(5), g)))
                    && i.name() != "v_readfirstlane_b32"
                    && i.write(Reg::V(5)).is_none()
            });
            if let Some(i) = per_lane {
                return Err(unsupported(
                    i.machine.offset,
                    "the mesh group input read per lane",
                ));
            }
            let loaded = vec![
                (Reg::V(5), Input::MeshGroup),
                (Reg::S(3), Input::MergedWaveInfo),
            ];
            (loaded, &[][..], &ms.outputs[..])
        }
        _ => return Err(unsupported(0, "I/O of this stage")),
    };
    let inputs: Vec<(Value, Input)> = p
        .inputs
        .iter()
        .filter_map(|&(reg, v)| loaded.iter().find(|l| l.0 == reg).map(|l| (v, l.1)))
        .collect();
    let input_of = |v: Value| inputs.iter().find(|i| i.0 == v).map(|i| i.1);
    let attribute = |inst: &Inst, attr: &Operand| match *attr {
        Operand::Attr { index, chan, high } => ps_inputs
            .iter()
            .find(|i: &&PsInput| i.attr == index)
            .map(|i| Attribute {
                usage: i.usage,
                row: i.row,
                channel: chan,
                high,
            })
            .ok_or(unsupported(
                inst.machine.offset,
                "an interpolated attribute the stage data does not list",
            )),
        _ => Err(unsupported(inst.machine.offset, "an interpolation operand")),
    };
    // A VGPR value's barycentric kind and index, if it is one as loaded.
    let barycentric = |v: Option<Value>| match v.and_then(input_of) {
        Some(Input::Barycentric { kind, index }) => Some((kind, index)),
        _ => None,
    };

    let mut io = Io {
        inputs: inputs.clone(),
        initialized_sgprs: match &native.io {
            StageIo::Cs(cs) => Some(cs.initialized_sgprs()),
            // A guessed graphics description's vertex shader (`emit::graphics`): the vertex
            // buffer table sits in two SGPRs (usually s0/s1), and the base vertex and instance
            // after them read 0.
            StageIo::Vs { .. } => native
                .layout
                .iter()
                .find(|p| p.slot == crate::emit::graphics::VERTEX_TABLE)
                .map(|p| p.user_data.saturating_add(2)),
            StageIo::Ps { .. } => {
                let user = native
                    .layout
                    .iter()
                    .map(|r| u32::from(r.user_data) + r.sgprs)
                    .max()
                    .unwrap_or(0);
                // Anchor to the primitive-mask SGPR when the code loads one exclusively into M0
                // for interpolation (handles guessed PS whose user data is < 8 or has gaps),
                // otherwise fall back to the 8-SGPR ABI floor.
                let base = ps_prim_mask(p, user, guessed)
                    .map(u32::from)
                    .unwrap_or_else(|| user.clamp(8, 104));
                Some(base.min(104) as u8 + 2)
            }
            // A mesh-style (NGG) program: 8 system SGPRs, then at most 32 user SGPRs (the user
            // count is 5 bits). Above s39 a register holds what the last wave left, so a read the
            // paths only seem to allow reads zero, as for compute.
            StageIo::Ms(_) => Some(40),
            _ => None,
        },
        ..Io::default()
    };
    let defs: std::collections::HashMap<Value, Site> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .insts
                .iter()
                .enumerate()
                .flat_map(move |(i, inst)| inst.writes.iter().map(move |w| (w.1, (b, i))))
        })
        .collect();
    let phis: std::collections::HashMap<Value, (usize, usize)> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .phis
                .iter()
                .enumerate()
                .map(move |(i, phi)| (phi.value, (b, i)))
        })
        .collect();
    let halves = Halves {
        p,
        defs: &defs,
        phis: &phis,
    };
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let at = inst.machine.offset;
            let name = inst.name();
            match &inst.fields {
                Fields::Vintrp { dst, src, attr } => {
                    let attribute = attribute(inst, attr)?;
                    match (name, *src) {
                        ("v_interp_mov_f32", Operand::Const(vertex)) => {
                            io.interps
                                .push((site, Interp::Vertex { attribute, vertex }));
                        }
                        ("v_interp_p1_f32", _) => {}
                        ("v_interp_p2_f32", Operand::Vector { reg, .. }) => {
                            let Operand::Vector { reg: d, .. } = *dst else {
                                return Err(unsupported(at, "an interpolation operand"));
                            };
                            // The first half's result is this one's old dst.
                            let interp = halves
                                .pair(
                                    attr,
                                    inst.read(Reg::V(d)),
                                    inst.read(Reg::V(reg)),
                                    &barycentric,
                                )
                                .map_err(|what| unsupported(at, what))?;
                            io.interps.push((site, interp.with(attribute)));
                        }
                        _ => return Err(unsupported(at, "an interpolation form")),
                    }
                }
                Fields::Valu(v) if name.starts_with("v_interp_") => match name {
                    "v_interp_p1ll_f16" => {}
                    "v_interp_p2_f16" => {
                        let attribute = attribute(inst, &v.src[0])?;
                        let read = |k: usize| match v.src.get(k) {
                            Some(&Operand::Vector { reg, .. }) => inst.read(Reg::V(reg)),
                            _ => None,
                        };
                        let interp = halves
                            .pair(&v.src[0], read(2), read(1), &barycentric)
                            .map_err(|what| unsupported(at, what))?;
                        io.interps.push((site, interp.with(attribute)));
                    }
                    _ => return Err(unsupported(at, "an interpolation form")),
                },
                Fields::Exp {
                    target,
                    src,
                    compr,
                    done,
                    ..
                } => {
                    let target = match *target {
                        t @ 0..=7 => Target::Color(t),
                        8 => Target::Depth,
                        9 => Target::Null,
                        t @ 12..=16 => Target::Position(t - 12),
                        20 => Target::Primitive,
                        32 if params.is_empty() => Target::Unread,
                        t @ 32..=63 => {
                            // Bit 7: a mesh program's per-primitive output.
                            let o = params.iter().find(|o| o.param & 0x7F == t - 32).ok_or(
                                unsupported(at, "a parameter export the stage data does not list"),
                            )?;
                            let (usage, row) = (o.usage, o.row);
                            if o.param & 0x80 != 0 {
                                Target::PrimitiveParam { usage, row }
                            } else {
                                Target::Param { usage, row }
                            }
                        }
                        _ => return Err(unsupported(at, "an export target")),
                    };
                    let values = src.map(|s| match s {
                        Some(Operand::Vector { reg, .. }) => inst.read(Reg::V(reg)),
                        _ => None,
                    });
                    io.exports.push(Export {
                        site,
                        target,
                        values,
                        compressed: *compr,
                        done: *done,
                    });
                }
                _ => {}
            }
        }
    }
    Ok(io)
}

/// Finds a unique live-in SGPR `s<pm>` (`user <= pm <= 104`) that is used solely to feed `m0`
/// for pixel interpolation (`v_interp_*`), following scalar copies, `v_writelane`/`v_readlane`
/// spills, and phis.
fn ps_prim_mask(p: &Program, user: u32, guessed: bool) -> Option<u8> {
    let uses = PrimMaskUses::new(p);
    let mut found = None;
    for &(reg, init_v) in &p.inputs {
        let Reg::S(pm) = reg else { continue };
        if pm > 104 || u32::from(pm) < user {
            continue;
        }
        if uses.is_prim_mask_only(init_v, guessed) {
            if found.is_some() {
                return None;
            }
            found = Some(pm);
        }
    }
    found
}

/// Indexed SSA phi and instruction readers of a [`Program`], built once and shared across entry
/// SGPR queries ([`PrimMaskUses::is_prim_mask_only`]): scanning every block and instruction per
/// popped value was quadratic across spilled SGPRs in large pixel shaders.
pub(crate) struct PrimMaskUses<'a> {
    phis: Map<Value, Vec<&'a Phi>>,
    insts: Map<Value, Vec<&'a Inst>>,
    #[cfg(test)]
    visits: std::cell::Cell<usize>,
}

impl<'a> PrimMaskUses<'a> {
    pub(crate) fn new(p: &'a Program) -> Self {
        let mut phis: Map<Value, Vec<&'a Phi>> = Map::default();
        let mut insts: Map<Value, Vec<&'a Inst>> = Map::default();
        for block in &p.blocks {
            for phi in &block.phis {
                for (i, &arg) in phi.args.iter().enumerate() {
                    if !phi.args[..i].contains(&arg) {
                        phis.entry(arg).or_default().push(phi);
                    }
                }
            }
            for inst in &block.insts {
                for (i, &(_, val)) in inst.reads.iter().enumerate() {
                    if !inst.reads[..i].iter().any(|r| r.1 == val) {
                        insts.entry(val).or_default().push(inst);
                    }
                }
            }
        }
        Self {
            phis,
            insts,
            #[cfg(test)]
            visits: Default::default(),
        }
    }

    fn phis_of(&self, v: Value) -> &[&'a Phi] {
        self.phis.get(&v).map_or(&[], Vec::as_slice)
    }

    fn insts_of(&self, v: Value) -> &[&'a Inst] {
        self.insts.get(&v).map_or(&[], Vec::as_slice)
    }

    /// Whether `init_v` (a live-in SGPR) flows only through copies / `v_writelane`+`v_readlane`
    /// spills into `m0`, and every consumer of those `m0` values is `v_interp_*` (or the benign
    /// `MSG_ORDERED_PS_DONE` sendmsg, which ignores `m0` in hardware).
    // shortcut: guessed shaders may also mask, shift or extract bits on the way to M0, extend the
    // recognised scalar operations if a new compiler pattern still uses the value only for M0.
    pub(crate) fn is_prim_mask_only(&self, init_v: Value, guessed: bool) -> bool {
        let m0 = crate::ir::M0;
        let mut scalar_vals = Set::default();
        scalar_vals.insert(init_v);
        let mut scalar_q = vec![init_v];
        let mut spilled: Set<(Value, u32)> = Set::default();
        let mut spill_q: Vec<(Value, u32)> = Vec::new();
        let mut m0_vals: Set<Value> = Set::default();
        let mut m0_q: Vec<Value> = Vec::new();

        while !scalar_q.is_empty() || !spill_q.is_empty() {
            while let Some(v) = scalar_q.pop() {
                for &phi in self.phis_of(v) {
                    #[cfg(test)]
                    self.visits.set(self.visits.get() + 1);
                    match phi.reg {
                        Reg::S(d) if d < 106 => {
                            if scalar_vals.insert(phi.value) {
                                scalar_q.push(phi.value);
                            }
                        }
                        _ => return false,
                    }
                }
                for &inst in self.insts_of(v) {
                    #[cfg(test)]
                    self.visits.set(self.visits.get() + 1);
                    match (inst.name(), &inst.fields) {
                        (
                            name,
                            Fields::Sop {
                                dst: Some(Operand::Scalar { code: d, .. }),
                                src,
                                ..
                            },
                        ) if (name == "s_mov_b32"
                            && matches!(src.first(), Some(&Operand::Scalar { code: s, .. }) if inst.read(Reg::S(s)) == Some(v)))
                            || (guessed && matches!(name, "s_and_b32" | "s_or_b32" | "s_lshl_b32" | "s_lshr_b32" | "s_bfe_u32")) =>
                        {
                            // The mask bits have no host use, including through SCC.
                            if let Some(scc) = inst.write(Reg::Scc)
                                && (!self.phis_of(scc).is_empty()
                                    || self.insts_of(scc).iter().any(|u| u.read(Reg::Scc) == Some(scc)))
                            {
                                return false;
                            }
                            let Some(dv) = inst.write(Reg::S(*d)) else {
                                return false;
                            };
                            if *d == m0 {
                                if m0_vals.insert(dv) {
                                    m0_q.push(dv);
                                }
                            } else if *d < 106 {
                                if scalar_vals.insert(dv) {
                                    scalar_q.push(dv);
                                }
                            } else {
                                return false;
                            }
                        }
                        ("v_writelane_b32", Fields::Valu(f))
                            if matches!(
                                (f.dst, f.src.as_slice()),
                                (
                                    Some(Operand::Vector { .. }),
                                    [Operand::Scalar { code: s, .. }, _]
                                ) if inst.read(Reg::S(*s)) == Some(v)
                            ) =>
                        {
                            let Some(Operand::Vector { reg, .. }) = f.dst else {
                                return false;
                            };
                            let Some(lane) = crate::resource::lane_of(f.src[1]) else {
                                return false;
                            };
                            let Some(vv) = inst.write(Reg::V(reg)) else {
                                return false;
                            };
                            if spilled.insert((vv, lane)) {
                                spill_q.push((vv, lane));
                            }
                        }
                        _ => return false,
                    }
                }
            }

            while let Some((vv, lane)) = spill_q.pop() {
                for &phi in self.phis_of(vv) {
                    #[cfg(test)]
                    self.visits.set(self.visits.get() + 1);
                    match phi.reg {
                        Reg::V(_) => {
                            if spilled.insert((phi.value, lane)) {
                                spill_q.push((phi.value, lane));
                            }
                        }
                        _ => return false,
                    }
                }
                for &inst in self.insts_of(vv) {
                    #[cfg(test)]
                    self.visits.set(self.visits.get() + 1);
                    match (inst.name(), &inst.fields) {
                        ("v_readlane_b32", Fields::Valu(f))
                            if matches!(
                                (f.dst, f.src.as_slice()),
                                (
                                    Some(Operand::Scalar { .. }),
                                    [Operand::Vector { reg, .. }, _]
                                ) if inst.read(Reg::V(*reg)) == Some(vv)
                            ) =>
                        {
                            let Some(read_lane) = crate::resource::lane_of(f.src[1]) else {
                                return false;
                            };
                            if read_lane == lane {
                                let Some(Operand::Scalar { code: d, .. }) = f.dst else {
                                    return false;
                                };
                                let Some(dv) = inst.write(Reg::S(d)) else {
                                    return false;
                                };
                                if d == m0 {
                                    if m0_vals.insert(dv) {
                                        m0_q.push(dv);
                                    }
                                } else if d < 106 {
                                    if scalar_vals.insert(dv) {
                                        scalar_q.push(dv);
                                    }
                                } else {
                                    return false;
                                }
                            }
                        }
                        ("v_writelane_b32", Fields::Valu(f))
                            if matches!(f.dst, Some(Operand::Vector { reg, .. }) if inst.read(Reg::V(reg)) == Some(vv)) =>
                        {
                            let Some(Operand::Vector { reg, .. }) = f.dst else {
                                return false;
                            };
                            let Some(write_lane) = f.src.get(1).and_then(|o| crate::resource::lane_of(*o)) else {
                                return false;
                            };
                            if write_lane != lane {
                                let Some(nv) = inst.write(Reg::V(reg)) else {
                                    return false;
                                };
                                if spilled.insert((nv, lane)) {
                                    spill_q.push((nv, lane));
                                }
                            }
                        }
                        _ => {
                            // Allow a full overwrite of the spill VGPR where `vv` is only read
                            // as the IR's implicit old-value tie.
                            let tie_only = inst
                                .reads
                                .iter()
                                .filter(|r| r.1 == vv)
                                .all(|(r, _)| inst.write(*r).is_some())
                                && !match &inst.fields {
                                    Fields::Valu(f) => f.src.iter().any(|s| {
                                        matches!(*s, Operand::Vector { reg, dwords }
                                            if (0..dwords).any(|k| inst.read(Reg::V(reg + k)) == Some(vv)))
                                    }),
                                    _ => true,
                                };
                            if !tie_only {
                                return false;
                            }
                        }
                    }
                }
            }
        }

        if m0_vals.is_empty() {
            return false;
        }
        while let Some(mv) = m0_q.pop() {
            for &phi in self.phis_of(mv) {
                #[cfg(test)]
                self.visits.set(self.visits.get() + 1);
                if phi.reg == Reg::S(m0) {
                    if m0_vals.insert(phi.value) {
                        m0_q.push(phi.value);
                    }
                } else {
                    return false;
                }
            }
        }
        let mut saw_interp = false;
        for &mv in &m0_vals {
            for &inst in self.insts_of(mv) {
                #[cfg(test)]
                self.visits.set(self.visits.get() + 1);
                if matches!(inst.fields, Fields::Vintrp { .. }) || inst.name().starts_with("v_interp_")
                {
                    saw_interp = true;
                } else if matches!(
                    inst.fields,
                    Fields::Sopp { imm }
                        if matches!(inst.name(), "s_sendmsg" | "s_sendmsghalt") && imm & 0xF == 7
                ) {
                    // `MSG_ORDERED_PS_DONE` ignores `m0` in hardware (`ir.rs` adds `m0` read for
                    // all `s_sendmsg`).
                } else {
                    return false;
                }
            }
        }
        saw_interp
    }
}

/// The pixel input VGPRs that `addr` enables, with their meaning.
fn ps_vgprs(addr: u32) -> Vec<(u8, Input)> {
    let mut out = Vec::new();
    let mut next = 0u8;
    for (bit, &(count, kind)) in PS_VGPRS.iter().enumerate() {
        if addr >> bit & 1 == 0 {
            continue;
        }
        for index in 0..count {
            let input = match kind {
                Some(kind) => Input::Barycentric { kind, index },
                None => ps_system(bit),
            };
            out.push((next + index, input));
        }
        next += count;
    }
    out
}

/// An interpolation before its attribute is known.
struct Pair {
    barycentric: Barycentric,
    first: Vec<Site>,
    unless: Vec<Elsewhere>,
}

impl Pair {
    fn with(self, attribute: Attribute) -> Interp {
        Interp::Interpolated {
            attribute,
            barycentric: self.barycentric,
            first: self.first,
            unless: self.unless,
        }
    }
}

/// Finds the first halves of interpolations.
struct Halves<'a> {
    p: &'a Program,
    defs: &'a std::collections::HashMap<Value, Site>,
    phis: &'a std::collections::HashMap<Value, (usize, usize)>,
}

impl Halves<'_> {
    /// The interpolation whose second half reads the first half's result
    /// `acc`, the barycentric `j`, and attribute `attr`: every first half
    /// must read the same attribute channel and the same kind's i.
    fn pair(
        &self,
        attr: &Operand,
        acc: Option<Value>,
        j: Option<Value>,
        barycentric: &impl Fn(Option<Value>) -> Option<(Barycentric, u8)>,
    ) -> Result<Pair, &'static str> {
        let mut first = Vec::new();
        self.sites(
            acc.ok_or("an interpolation without its first half")?,
            &mut Vec::new(),
            &mut first,
        )?;
        let mut unless = Vec::new();
        let Some((kind, 1)) = self.joined(j, barycentric, &mut unless) else {
            return Err("an interpolation with computed barycentrics");
        };
        for &(b, i) in &first {
            let inst = &self.p.blocks[b].insts[i];
            let (a, i) = match &inst.fields {
                Fields::Vintrp {
                    src: Operand::Vector { reg, .. },
                    attr,
                    ..
                } if inst.name() == "v_interp_p1_f32" => (*attr, inst.read(Reg::V(*reg))),
                Fields::Valu(v) if inst.name() == "v_interp_p1ll_f16" => match v.src.get(1) {
                    Some(&Operand::Vector { reg, .. }) => (v.src[0], inst.read(Reg::V(reg))),
                    _ => return Err("an interpolation without its first half"),
                },
                _ => return Err("an interpolation without its first half"),
            };
            if a != *attr {
                return Err("interpolation halves of different attributes");
            }
            if self.joined(i, barycentric, &mut unless) != Some((kind, 0)) {
                return Err("an interpolation with computed barycentrics");
            }
        }
        Ok(Pair {
            barycentric: kind,
            first,
            unless,
        })
    }

    /// The barycentric `v` holds as loaded, also through joins and the
    /// writes of lanes that skip them: every barycentric found must be the
    /// same one, and where the register holds something else is added to
    /// `other`.
    fn joined(
        &self,
        v: Option<Value>,
        barycentric: &impl Fn(Option<Value>) -> Option<(Barycentric, u8)>,
        other: &mut Vec<Elsewhere>,
    ) -> Option<(Barycentric, u8)> {
        let (mut found, mut stack, mut seen) = (None, vec![v?], Vec::new());
        while let Some(v) = stack.pop() {
            if seen.contains(&v) {
                continue;
            }
            seen.push(v);
            if let Some(k) = barycentric(Some(v)) {
                if found.is_some_and(|f| f != k) {
                    return None;
                }
                found = Some(k);
            } else if let Some(&(b, i)) = self.phis.get(&v) {
                let block = &self.p.blocks[b];
                for (n, &a) in block.phis[i].args.iter().enumerate() {
                    if barycentric(Some(a)).is_some() || self.phis.contains_key(&a) {
                        stack.push(a);
                    } else {
                        other.push(Elsewhere::Edge(block.preds[n], b));
                    }
                }
            } else {
                // A vector write keeps the old value in the lanes it skips
                // (`v_writelane` ignores EXEC).
                let &(b, i) = self.defs.get(&v)?;
                let inst = &self.p.blocks[b].insts[i];
                let &(reg, _) = inst.writes.iter().find(|w| w.1 == v)?;
                if !matches!(reg, Reg::V(_)) || inst.name() == "v_writelane_b32" {
                    return None;
                }
                // A plain copy of a barycentric: that one where it runs.
                let copied = match (&inst.fields, inst.name()) {
                    (Fields::Valu(c), "v_mov_b32") if c.dpp.is_none() && c.sdwa.is_none() => {
                        match c.src.first() {
                            Some(&Operand::Vector { reg: from, .. }) => inst.read(Reg::V(from)),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                match copied.filter(|&c| barycentric(Some(c)).is_some()) {
                    Some(c) => {
                        other.push(Elsewhere::Skip((b, i)));
                        stack.push(c);
                    }
                    None => {
                        other.push(Elsewhere::Write((b, i)));
                        stack.push(inst.read(reg)?);
                    }
                }
            }
        }
        found
    }

    /// The instructions `v` comes from, through phis (`seen`: phis on the
    /// way; a loop back to one adds nothing).
    fn sites(
        &self,
        v: Value,
        seen: &mut Vec<Value>,
        out: &mut Vec<Site>,
    ) -> Result<(), &'static str> {
        if let Some(&site) = self.defs.get(&v) {
            if !out.contains(&site) {
                out.push(site);
            }
            return Ok(());
        }
        let &(b, i) = self
            .phis
            .get(&v)
            .ok_or("an interpolation without its first half")?;
        if !seen.contains(&v) {
            seen.push(v);
            for &a in &self.p.blocks[b].phis[i].args {
                self.sites(a, seen, out)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::ExportPatch;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn program(words: &[u32]) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::ir::build(&code, false, 256).unwrap()
    }

    fn pixel(inputs: Vec<PsInput>, input_addr: u32) -> NativeProgram {
        NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs,
                input_addr,
                db_shader_control: 0,
                patches: Vec::<ExportPatch>::new(),
            },
            code: vec![],
        }
    }

    const TEXCOORD: PsInput = PsInput {
        usage: Usage::Generic,
        row: 1,
        mask: 3,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 1,
    };

    #[test]
    fn interpolation_pairs_and_color_exports() {
        // Blit's pixel shader: TEXCOORD.xy from the perspective-center i/j
        // in v0/v1, then exported unconverted to MRT0.
        let p = program(&[
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0302_0102, // exp mrt0, v2, v1, v2, v3 done vm
            S_ENDPGM,
        ]);
        let io = raise(&p, &pixel(vec![TEXCOORD], 2)).unwrap();
        let input = |reg| p.inputs.iter().find(|i| i.0 == Reg::V(reg)).unwrap().1;
        let center = |index| Input::Barycentric {
            kind: Barycentric::PerspectiveCenter,
            index,
        };
        assert!(io.inputs.contains(&(input(0), center(0))));
        assert!(io.inputs.contains(&(input(1), center(1))));
        // v2 and v3 are read before written but hold nothing.
        assert!(!io.inputs.iter().any(|i| i.0 == input(3)));
        assert_eq!(
            io.interps,
            [(
                (0, 1),
                Interp::Interpolated {
                    attribute: Attribute {
                        usage: Usage::Generic,
                        row: 1,
                        channel: 0,
                        high: false
                    },
                    barycentric: Barycentric::PerspectiveCenter,
                    first: vec![(0, 0)],
                    unless: vec![],
                }
            )]
        );
        let [e] = &io.exports[..] else {
            panic!("{:?}", io.exports)
        };
        assert_eq!(
            (e.target, e.done, e.compressed),
            (Target::Color(0), true, false)
        );
        assert_eq!(e.values[0], p.blocks[0].insts[1].write(Reg::V(2)));
    }

    #[test]
    fn unpaired_or_unknown_interpolation_fails() {
        // p2 alone: its old dst is no p1's result.
        let lone = program(&[0xC809_0001, S_ENDPGM]); // v_interp_p2_f32 v2, v1, attr0.x
        assert!(raise(&lone, &pixel(vec![TEXCOORD], 2)).is_err());
        // attr1 is not in the stage data.
        let other = program(&[0xC808_0400, S_ENDPGM]); // v_interp_p1_f32 v2, v0, attr1.x
        assert!(raise(&other, &pixel(vec![TEXCOORD], 2)).is_err());
    }

    #[test]
    fn mesh_inputs_and_primitive_export() {
        let mesh = |words: &[u32]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = crate::ir::build(&code, true, 256).unwrap();
            let native = NativeProgram {
                stage: HwStage::Ms,
                wave64: true,
                vgprs: 256,
                layout: vec![],
                used: vec![],
                io: StageIo::Ms(crate::native::MeshIo {
                    outputs: vec![],
                    clip_cntl: 0,
                    threads: 128,
                    max_vertices: 128,
                    max_primitives: 128,
                    topology: 2,
                    lds_bytes: 0,
                }),
                code,
            };
            raise(&p, &native)
        };
        let io = mesh(&[
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0xF800_0941,
            0x0000_0009, // exp prim, v9, off, off, off done
            S_ENDPGM,
        ])
        .unwrap();
        assert!(io.inputs.iter().any(|i| i.1 == Input::MeshGroup));
        assert_eq!(io.exports[0].target, Target::Primitive);
        // The group input is uniform only as the first lane reads it.
        assert!(mesh(&[0x7E02_0305, S_ENDPGM]).is_err()); // v_mov_b32 v1, v5
    }

    #[test]
    fn pixel_vgprs_follow_the_input_address() {
        // Linear center (bit 5), then position x and y (bits 8, 9).
        let v = ps_vgprs(1 << 5 | 1 << 8 | 1 << 9);
        assert_eq!(
            v,
            [
                (
                    0,
                    Input::Barycentric {
                        kind: Barycentric::LinearCenter,
                        index: 0
                    }
                ),
                (
                    1,
                    Input::Barycentric {
                        kind: Barycentric::LinearCenter,
                        index: 1
                    }
                ),
                (2, Input::Position(0)),
                (3, Input::Position(1)),
            ]
        );
    }

    #[test]
    fn pixel_initialized_sgprs_anchors_to_prim_mask() {
        // Direct s0 -> m0 -> v_interp: anchors initialized_sgprs to 0 + 2 = 2 (below 8 ABI floor).
        let direct = program(&[
            0xBEFC_0300, // s_mov_b32 m0, s0
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            S_ENDPGM,
        ]);
        assert_eq!(
            raise(&direct, &pixel(vec![TEXCOORD], 2)).unwrap().initialized_sgprs,
            Some(2)
        );

        // Spilled s4 -> v5 lane 2 -> s12 -> m0 -> v_interp + MSG_ORDERED_PS_DONE: anchors to 4 + 2 = 6.
        let spilled = program(&[
            0xD761_0005,
            0x0001_0404, // v_writelane_b32 v5, s4, 2
            0xD761_0005,
            0x0001_0609, // v_writelane_b32 v5, s9, 3
            0xD760_000C,
            0x0001_0505, // v_readlane_b32 s12, v5, 2
            0xBEFC_030C, // s_mov_b32 m0, s12
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
            S_ENDPGM,
        ]);
        assert_eq!(
            raise(&spilled, &pixel(vec![TEXCOORD], 2)).unwrap().initialized_sgprs,
            Some(6)
        );

        // s4 moved to m0 for v_movrels_b32 (not v_interp): must NOT anchor, keeps 8 + 2 = 10.
        let movrel = program(&[
            0xBEFC_0304, // s_mov_b32 m0, s4
            0x7E04_8700, // v_movrels_b32 v2, v0
            S_ENDPGM,
        ]);
        assert_eq!(
            raise(&movrel, &pixel(vec![], 0)).unwrap().initialized_sgprs,
            Some(10)
        );
    }

    #[test]
    fn prim_mask_uses_visits_spills_and_phis_linearly() {
        let chain = |diamonds: usize| {
            let mut w = vec![
                0xD761_0005u32,
                0x0001_0002, // v_writelane_b32 v5, s2, 0 (non-prim-mask SGPR spilled)
                0xD761_0005,
                0x0001_0203, // v_writelane_b32 v5, s3, 1 (prim-mask SGPR spilled)
            ];
            for _ in 0..diamonds {
                w.push(0x7D88_0084); // v_cmp_gt_u32 vcc_lo, 4, v0
                w.push(0xBE88_3C6A); // s_and_saveexec_b32 s8, vcc_lo
                w.push(0xBF88_0002); // s_cbranch_execz +2
                w.extend([0xD761_0005, 0x0001_0481]); // v_writelane_b32 v5, 1, 2
                w.push(0xBEFE_0308); // s_mov_b32 exec_lo, s8
            }
            w.extend([
                0xD760_000Au32,
                0x0001_0105, // v_readlane_b32 s10, v5, 0
                0x800A_810A, // s_add_u32 s10, s10, 1 (rejects s2)
                0xD760_000B,
                0x0001_0305, // v_readlane_b32 s11, v5, 1
                0xBEFC_030B, // s_mov_b32 m0, s11
                0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
                0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
                S_ENDPGM,
            ]);
            let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            let p = crate::ir::build(&code, false, 16).unwrap();
            let uses = super::PrimMaskUses::new(&p);
            let s2 = p.inputs.iter().find(|(r, _)| *r == Reg::S(2)).unwrap().1;
            let s3 = p.inputs.iter().find(|(r, _)| *r == Reg::S(3)).unwrap().1;
            assert!(!uses.is_prim_mask_only(s2, true));
            uses.visits.set(0);
            assert!(uses.is_prim_mask_only(s3, true));
            uses.visits.get()
        };
        let v64 = chain(64);
        let v128 = chain(128);
        // Doubling the diamond count must only double the visited sites (linear, not quadratic).
        assert!(v128 <= 2 * v64 + 8, "v64={v64}, v128={v128}");
    }

    /// Every `SPI_PS_INPUT_ADDR` system bit maps to its input, as `PS_VGPRS`
    /// documents them; the rest are the fixed-point position.
    #[test]
    fn ps_system_bits_map_as_documented() {
        assert!(matches!(ps_system(7), Input::LineStipple));
        for b in 8..=11 {
            assert!(
                matches!(ps_system(b), Input::Position(p) if p == b as u8 - 8),
                "bit {b}"
            );
        }
        assert!(matches!(ps_system(12), Input::FrontFace));
        assert!(matches!(ps_system(13), Input::Ancillary));
        assert!(matches!(ps_system(14), Input::SampleCoverage));
        for b in [0, 1, 2, 3, 4, 5, 6, 15] {
            assert!(matches!(ps_system(b), Input::FixedPosition), "bit {b}");
        }
    }

    /// VGPRs are assigned in bit order across kinds: the first set bit's
    /// inputs get the lowest VGPRs, whatever their kind, and all 16 bits set
    /// give every entry of the table.
    #[test]
    fn ps_vgprs_numbers_vgprs_in_bit_order_across_kinds() {
        // Line stipple (bit 7), position x (bit 8), front face (bit 12).
        let v = ps_vgprs(1 << 7 | 1 << 8 | 1 << 12);
        assert_eq!(
            v,
            [
                (0, Input::LineStipple),
                (1, Input::Position(0)),
                (2, Input::FrontFace),
            ]
        );
        // All bits: 6 barycentric kinds of 2, pull-model of 3, 9 system bits
        // of 1 each.
        assert_eq!(ps_vgprs(0xFFFF).len(), 6 * 2 + 3 + 9);
        assert_eq!(ps_vgprs(0).len(), 0);
    }
}
