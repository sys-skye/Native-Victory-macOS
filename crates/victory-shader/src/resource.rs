//! Resource tracking (decision 0003): where each descriptor a program uses
//! comes from, in terms of the root signature.
//!
//! Root parameters arrive in user SGPRs ([`RootParam`]): a descriptor table
//! as the 32-bit heap address of its range (the code sets the high half to
//! 0), a root descriptor as a 64-bit buffer address, root constants as
//! themselves. A root signature with dynamic resources (SM 6.6) adds the
//! 32-bit base addresses of both heaps; a descriptor is then loaded from
//! one of them at an index the code computes (from a root constant, say). The code loads table descriptors from heap memory with
//! `s_load_dword*`, and builds a buffer descriptor around a root
//! descriptor's address itself. Each descriptor operand and each scalar
//! memory address is traced through the SSA values back to those
//! parameters, by data flow: copies, phis that agree, and additions.

use crate::ShaderError;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, RootParam, RootType};
use crate::structure::{Bool, LaneExit, Structured};

/// An instruction: block and index in [`crate::ir::Block::insts`].
pub type Site = (usize, usize);

/// A byte address: a root parameter's, plus a constant and at most one
/// value the program computes (an index times a stride, say).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    /// Index into the layout, or [`CODE`].
    pub param: usize,
    /// Added modulo 2^32 (with carry into the high half of a root
    /// descriptor's address).
    pub offset: u32,
    pub dynamic: Option<Value>,
}

/// [`Address::param`] of the program's own code: a constant table the
/// compiler appends after the code and reads through `s_getpc_b64`; the
/// offset is then the byte offset in the code.
pub const CODE: usize = usize::MAX;

/// [`Address::param`] of title memory at an address the program computes
/// from data (a pointer loaded from memory or passed in root constants,
/// arithmetic on such values): lowering takes the address from the
/// instruction's own operands (decision 0005, title memory).
pub const MEMORY: usize = usize::MAX - 1;

/// [`Address::param`] of a load from the root signature's memory: a compute
/// program gets the root parameters past its 16 user SGPRs in memory whose
/// 40-bit address is in user data 0-1, starting with user-data dword 16.
/// The offset is the user-data byte offset (dword 16 at 0x40).
pub const USER_DATA: usize = usize::MAX - 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Dwords `first..` of what the scalar load at `load` reads (see
    /// [`Resources::scalar`] for its address).
    Loaded { load: Site, first: u8 },
    /// A buffer descriptor built around an address: `bits` OR-ed into
    /// dword 1 (stride, swizzle), dwords 2 and 3 as given.
    Built {
        address: Address,
        bits: u32,
        words: [u32; 2],
    },
    /// A guessed pixel shader's buffer descriptor over a computed title address.
    BuiltMemory {
        control: u32,
        /// Whether dword 1 may enable swizzling.
        swizzled: bool,
    },
    /// Fixed in the code (an immediate sampler, say).
    Constant(Vec<u32>),
    /// Chosen by control flow: one source per predecessor of `block`, in
    /// [`crate::ir::Block::preds`] order.
    Phi { block: usize, args: Vec<Source> },
    /// Chosen by SCC (`scc`, the value an `s_cselect` or `s_cmov` of each
    /// dword reads): the source when it is set, then when it is clear.
    Select { scc: Value, args: Vec<Source> },
    /// Dwords `first..` of what the vector load at `load` (`global_load_*`)
    /// reads, each lane at its own address (see [`Resources::scalar`]); the
    /// dwords past that load come from loads of the bytes that follow. A
    /// descriptor that differs between lanes.
    Fetched { load: Site, first: u8 },
    /// Picked per lane by `v_cndmask_b32` reading the lane mask whose low
    /// dword is `mask`: the source where the lane's bit is set, then where it
    /// is clear.
    Lanes { mask: Value, args: Vec<Source> },
    /// Per-lane dwords chosen by control flow (VGPRs joined where lanes
    /// that took different paths meet): one source per predecessor of
    /// `block`, in [`crate::ir::Block::preds`] order, `None` on an edge no
    /// lane that accesses the descriptor takes. A descriptor that differs
    /// between lanes.
    LanePhi {
        block: usize,
        args: Vec<Option<Source>>,
    },
}

impl Source {
    /// The sources this one picks from (itself unless a phi).
    pub fn leaves(&self) -> Vec<&Source> {
        match self {
            Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } => {
                args.iter().flat_map(Source::leaves).collect()
            }
            Source::LanePhi { args, .. } => {
                args.iter().flatten().flat_map(Source::leaves).collect()
            }
            s => vec![s],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Image, buffer or `s_buffer_load` resource.
    Resource,
    Sampler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub site: Site,
    pub role: Role,
    pub source: Source,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resources {
    /// Every scalar memory access by address (`s_load_*`, `s_store_*`), and
    /// every global memory access (`global_*`, the dynamic term per lane).
    pub scalar: Vec<(Site, Address)>,
    pub descriptors: Vec<Descriptor>,
    /// User SGPRs holding root parameters: value on entry (or loaded from
    /// the root signature's memory, [`USER_DATA`]), layout index, dword
    /// within the parameter.
    pub params: Vec<(Value, usize, u32)>,
    /// Live-in SGPRs an address was taken in that no root parameter fills
    /// (each pairs with the next); their addresses track as title memory
    /// ([`MEMORY`]).
    pub memory_inputs: Vec<u8>,
    pub waterfall: Waterfall,
    /// Buffer accesses through a loaded descriptor whose stride the program cleared
    /// (`Tracer::stride_cleared`): the index adds nothing to the address (index × stride, RDNA2
    /// ISA 8.1.5).
    pub unstrided: Vec<Site>,
}

/// Waterfall loops over descriptors that differ between lanes: each pass
/// reads the first active lane's descriptor with `v_readfirstlane_b32`,
/// compares it with every lane's own (`v_cmp_eq`), and runs the access with
/// the lanes that match. Their accesses take each lane's own descriptor
/// ([`Source::Fetched`], [`Source::Lanes`]), so the first lane's dwords are
/// each lane's own too: the compares hold in every lane and one pass serves
/// all lanes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Waterfall {
    /// The compares, true in every lane.
    pub compares: Vec<Site>,
    /// The `v_readfirstlane_b32` results they compare, each the lane's own
    /// source dword.
    pub reads: Vec<Value>,
    /// Descriptor loads at a table offset among `reads` (the console
    /// compiler's form): each lane loads its own entry.
    pub loads: Vec<Site>,
}

/// Traces every descriptor and scalar address of `p`, which runs as
/// `stage`, to the root parameters in `layout`. Fails on the first one it
/// cannot trace. `lanes`, `p` as one lane runs it, tells which paths the
/// lanes that access a per-lane descriptor took and which writes they ran;
/// without it, per-lane descriptors chosen by control flow are refused and
/// per-lane writes are taken to run in every lane.
pub fn track(
    p: &Program,
    layout: &[RootParam],
    stage: HwStage,
    lanes: Option<&Structured>,
) -> Result<Resources, ShaderError> {
    track_graphics(p, layout, stage, lanes, false)
}

fn collect_memory_inputs(t: &Tracer) -> Vec<u8> {
    let mut v: Vec<u8> = t.memory_inputs.borrow().iter().copied().collect();
    v.sort_unstable();
    v
}

pub(crate) fn track_graphics(
    p: &Program,
    layout: &[RootParam],
    stage: HwStage,
    lanes: Option<&Structured>,
    computed_buffers: bool,
) -> Result<Resources, ShaderError> {
    let mut t = Tracer::new(p, layout, stage, lanes)?;
    t.computed_buffers =
        computed_buffers && matches!(stage, HwStage::Vs | HwStage::Ps | HwStage::Cs);
    let mut out = Resources {
        params: p
            .inputs
            .iter()
            .filter_map(|&(reg, v)| {
                let (param, dword) = t.param_of(reg)?;
                Some((v, param, dword))
            })
            .chain(t.spilled.iter().map(|(&v, &(param, dword))| (v, param, dword)))
            .collect(),
        ..Resources::default()
    };
    // Group accesses in the same loop so a shared mask can select several
    // descriptors together.
    let mut first_lane_at = std::collections::BTreeMap::<usize, (Vec<usize>, Vec<Value>)>::new();
    // The console compiler's waterfall reads the first lane's table offset
    // and loads that entry: accesses, offset reads and loads by block.
    let mut offset_at =
        std::collections::BTreeMap::<usize, (Vec<usize>, Vec<Value>, Vec<Site>)>::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let fail = |what| ShaderError::Unsupported {
                offset: inst.machine.offset,
                what,
            };
            let mut first_lane = Vec::new();
            let mut offset = Vec::new();
            let mut descriptor = |op: &Operand, role| -> Result<(), ShaderError> {
                let source = t.descriptor(inst, op, b).map_err(fail)?;
                first_lane.extend(t.first_lane_words(inst, op).unwrap_or_default());
                if let Source::Loaded { load, .. } = source
                    && let Some((_, a)) = out.scalar.iter().find(|s| s.0 == load)
                    && let Some(reads) = a.dynamic.and_then(|d| t.first_lane_in_offset(load.0, d))
                {
                    for v in reads {
                        offset.push((v, load));
                    }
                }
                out.descriptors.push(Descriptor { site, role, source });
                Ok(())
            };
            match &inst.fields {
                Fields::Image { rsrc, .. } if inst.name().starts_with("image_bvh") => {
                    if let Some(source) = t.bvh_descriptor(inst, rsrc).map_err(fail)? {
                        out.descriptors.push(Descriptor {
                            site,
                            role: Role::Resource,
                            source,
                        });
                    }
                }
                Fields::Image { rsrc, sampler, .. } => {
                    descriptor(rsrc, Role::Resource)?;
                    if let Some(s) = sampler {
                        descriptor(s, Role::Sampler)?;
                    }
                }
                // A cache invalidation names a buffer only to scope the
                // address range it flushes; it reads no data through the
                // descriptor. Lowering one is a no-op on the host, so there
                // is no descriptor to keep alive for it.
                Fields::Buffer { rsrc, .. } if !inst.name().ends_with("_inv") => {
                    descriptor(rsrc, Role::Resource)?;
                    if t.stride_cleared(inst, rsrc).is_some() {
                        out.unstrided.push(site);
                    }
                }
                Fields::Buffer { .. } => {}
                Fields::Smem { base, .. } if inst.name().starts_with("s_buffer_") => {
                    descriptor(base, Role::Resource)?;
                }
                Fields::Smem {
                    base,
                    offset,
                    soffset,
                    ..
                } if inst.name().contains("_load_") || inst.name().contains("_store_") => {
                    let address = t.smem_address(inst, base, *offset, soffset).map_err(fail)?;
                    out.scalar.push((site, address));
                }
                Fields::Flat {
                    addr,
                    saddr,
                    offset,
                    ..
                } if inst.name().starts_with("global_") => {
                    let address = t
                        .flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset)
                        .map_err(fail)?;
                    out.scalar.push((site, address));
                }
                _ => {}
            }
            if !first_lane.is_empty() {
                let (accesses, reads) = first_lane_at.entry(b).or_default();
                accesses.push(i);
                reads.extend(first_lane);
            }
            if !offset.is_empty() {
                let (accesses, reads, loads) = offset_at.entry(b).or_default();
                accesses.push(i);
                for (v, load) in offset {
                    reads.push(v);
                    loads.push(load);
                }
            }
        }
    }
    // The proof needs the per-lane form. Without it nothing is emitted, and
    // its failure is the one to report, so the reads stay unproven here.
    if t.lanes.is_none() {
        out.memory_inputs = collect_memory_inputs(&t);
        return Ok(out);
    }
    for (block, (accesses, mut reads)) in first_lane_at {
        reads.sort_unstable();
        reads.dedup();
        let proof =
            t.waterfall(block, &accesses, &reads)
                .map_err(|what| ShaderError::Unsupported {
                    offset: t.inst((block, accesses[0])).machine.offset,
                    what,
                })?;
        out.waterfall.compares.extend(proof.compares);
        out.waterfall.reads.extend(proof.reads);
    }
    // Unproven, an offset read stays the first lane's (one uniform entry),
    // which is what the hardware reads outside a waterfall loop.
    for (block, (accesses, mut reads, loads)) in offset_at {
        reads.sort_unstable();
        reads.dedup();
        if let Ok(proof) = t.waterfall(block, &accesses, &reads) {
            out.waterfall.compares.extend(proof.compares);
            out.waterfall.reads.extend(proof.reads);
            out.waterfall.loads.extend(loads);
        }
    }
    out.memory_inputs = collect_memory_inputs(&t);
    Ok(out)
}

/// A dword after copy propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Term {
    Const(u32),
    Value(Value),
}

/// The single non-constant 32-bit offset added to a heap/root base on the low half of an address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LowOffset {
    /// Canonicalized low-half addend `d` (`lo + extra == base + const + d`).
    raw: Term,
    /// `(canon(src), shift)` when `d` is `src << shift` (`1 <= shift <= 31`).
    shifted: Option<(Term, u32)>,
    /// Canonicalized unordered operand pair `[a, b]` and whether `d` came from a shift by 31
    /// (`1 << 31` is negative in signed 32-bit multiplication).
    product: Option<([Term; 2], bool)>,
    /// Low output `Value` when `d` is `writes[0]` of a 64-bit shift or `v_mad_*64_*32`.
    pair_lo: Option<Value>,
}

/// A 32-bit sum: optional root parameter (as its first dword), constant,
/// optional computed term.
#[derive(Clone, Copy)]
struct Sum {
    base: Option<Value>,
    offset: u32,
    dynamic: Option<Value>,
}

struct Tracer<'a> {
    p: &'a Program,
    layout: &'a [RootParam],
    stage: HwStage,
    computed_buffers: bool,
    /// Defining instruction per value (phis and inputs have none).
    defs: Vec<Option<Site>>,
    /// Block and index of each phi value.
    phis: Vec<Option<(usize, usize)>>,
    /// Register per input value.
    inputs: Vec<Option<Reg>>,
    origin: Vec<Term>,
    lanes: Option<&'a Structured>,
    /// Reverse-postorder index per block.
    rpo: Vec<usize>,
    /// Values loaded from the root signature's memory ([`USER_DATA`]): the
    /// root parameter and dword each one is.
    spilled: std::collections::HashMap<Value, (usize, u32)>,
    /// The first such value of each parameter dword, which the others are
    /// copies of.
    first_spilled: std::collections::HashMap<(usize, u32), Value>,
    /// The loads from that memory by code offset, with their user-data byte
    /// offset.
    spill_at: std::collections::HashMap<usize, Result<u32, &'static str>>,
    /// Live-in SGPRs whose pair an address was taken in with no root
    /// parameter filling them (tracked as title memory).
    memory_inputs: std::cell::RefCell<std::collections::BTreeSet<u8>>,
    /// Memoized [`Self::edge_taken`] per `(pred, block)` CFG edge.
    edge_taken_cache: std::cell::RefCell<std::collections::HashMap<(usize, usize), bool>>,
    /// Precomputed single-value resolution of each phi along taken edges (`O(P + E)`).
    joined: Vec<Option<Term>>,
    /// Precomputed [`Sum`] for every SSA value (`O(V + E)`).
    sums: Vec<Sum>,
}

/// What lanes are known to satisfy: each [`Bool`] where its flag is set,
/// its negation where not.
type Known = Vec<(Bool, bool)>;

/// Whether some lane can satisfy all of `known`. The search gives up (yes)
/// on many large conditions, as the reach of blocks deep in a program is,
/// before it gets to the branch conditions of the path, so those few small
/// ones are tried alone first: if they cannot hold together, neither can
/// all.
fn meet(lanes: &Structured, known: &Known) -> bool {
    let small: Known = known
        .iter()
        .copied()
        .filter(|&(b, _)| few_atoms(&lanes.bools, b))
        .collect();
    (small.len() == known.len() || lanes.bools.meet(&small)) && lanes.bools.meet(known)
}

/// Whether `b` is a small condition: at most 4 atoms within 32 nodes.
fn few_atoms(bools: &crate::structure::Bools, b: Bool) -> bool {
    let (mut atoms, mut stack, mut nodes) = (Vec::new(), vec![b], 0);
    while let Some(x) = stack.pop() {
        let Some((atom, lo, hi)) = bools.node(x) else {
            continue;
        };
        nodes += 1;
        if !atoms.contains(&atom) {
            atoms.push(atom);
        }
        if nodes > 32 || atoms.len() > 4 {
            return false;
        }
        stack.extend([lo, hi]);
    }
    true
}

const MAX_ROUNDS: usize = 64;
const MAX_JOINS: usize = 8;
const MAX_HIGH_STEPS: usize = 512;

/// Iterative Tarjan strongly-connected-components pass over `0..n`, returning SCCs in reverse
/// topological order (every callee SCC appears before any caller SCC that reaches it).
fn tarjan_sccs(n: usize, mut edges_of: impl FnMut(usize, &mut Vec<usize>)) -> Vec<Vec<usize>> {
    let mut index = vec![0u32; n];
    let mut lowlink = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut tarjan_stack: Vec<usize> = Vec::new();
    let mut frames: Vec<(usize, Vec<usize>, usize)> = Vec::new();
    let mut next_idx = 1u32;
    let mut sccs = Vec::new();
    for start in 0..n {
        if index[start] != 0 {
            continue;
        }
        index[start] = next_idx;
        lowlink[start] = next_idx;
        next_idx += 1;
        tarjan_stack.push(start);
        on_stack[start] = true;
        let mut nbrs = Vec::new();
        edges_of(start, &mut nbrs);
        frames.push((start, nbrs, 0));
        while let Some((v, nbrs, pos)) = frames.last_mut() {
            let v = *v;
            if *pos < nbrs.len() {
                let w = nbrs[*pos];
                *pos += 1;
                if w >= n {
                    continue;
                }
                if index[w] == 0 {
                    index[w] = next_idx;
                    lowlink[w] = next_idx;
                    next_idx += 1;
                    tarjan_stack.push(w);
                    on_stack[w] = true;
                    let mut w_nbrs = Vec::new();
                    edges_of(w, &mut w_nbrs);
                    frames.push((w, w_nbrs, 0));
                } else if on_stack[w] {
                    lowlink[v] = lowlink[v].min(index[w]);
                }
            } else {
                frames.pop();
                if let Some((parent, _, _)) = frames.last() {
                    lowlink[*parent] = lowlink[*parent].min(lowlink[v]);
                }
                if lowlink[v] == index[v] {
                    let mut comp = Vec::new();
                    while let Some(u) = tarjan_stack.pop() {
                        on_stack[u] = false;
                        comp.push(u);
                        if u == v {
                            break;
                        }
                    }
                    sccs.push(comp);
                }
            }
        }
    }
    sccs
}

impl<'a> Tracer<'a> {
    fn new(
        p: &'a Program,
        layout: &'a [RootParam],
        stage: HwStage,
        lanes: Option<&'a Structured>,
    ) -> Result<Self, ShaderError> {
        let all = || {
            p.inputs
                .iter()
                .map(|i| i.1)
                .chain(p.blocks.iter().flat_map(|b| {
                    b.phis
                        .iter()
                        .map(|phi| phi.value)
                        .chain(b.insts.iter().flat_map(|i| i.writes.iter().map(|w| w.1)))
                }))
        };
        let n = all().map(|v| v.0 as usize + 1).max().unwrap_or(0);
        let mut defs = vec![None; n];
        let mut phis = vec![None; n];
        let mut inputs = vec![None; n];
        for &(reg, v) in &p.inputs {
            inputs[v.0 as usize] = Some(reg);
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phis[phi.value.0 as usize] = Some((b, i));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs[v.0 as usize] = Some((b, i));
                }
            }
        }
        let mut rpo = vec![usize::MAX; p.blocks.len()];
        for (n, b) in crate::ir::reverse_postorder(&p.blocks)
            .into_iter()
            .enumerate()
        {
            rpo[b] = n;
        }
        let mut t = Tracer {
            p,
            layout,
            stage,
            computed_buffers: false,
            defs,
            phis,
            inputs,
            origin: Vec::new(),
            lanes,
            rpo,
            spilled: Default::default(),
            first_spilled: Default::default(),
            spill_at: Default::default(),
            memory_inputs: Default::default(),
            edge_taken_cache: Default::default(),
            joined: Vec::new(),
            sums: Vec::new(),
        };
        t.origin = t.propagate_copies(n)?;
        // Loads from the root signature's memory are found by their address,
        // which copies reach; then a reload is a copy of the first load.
        for (site, at) in t.spill_loads() {
            t.spill_at.insert(t.inst(site).machine.offset, at);
            let Ok(at) = at else { continue };
            for (k, &(_, v)) in t.inst(site).writes.iter().enumerate() {
                if let Some(pd) = t.spilled_param(at / 4 + k as u32) {
                    t.spilled.insert(v, pd);
                    t.first_spilled.entry(pd).or_insert(v);
                }
            }
        }
        if !t.spilled.is_empty() {
            t.origin = t.propagate_copies(n)?;
        }
        t.joined = t.compute_joined(n);
        t.sums = t.compute_sums(n);
        Ok(t)
    }

    /// The scalar loads from the root signature's memory ([`USER_DATA`])
    /// and the user-data byte offset each reads at (an error for a computed
    /// one). Compute programs only: they get 16 user SGPRs, and one whose
    /// parameters do not fit reads them through the pointer in user data
    /// 0-1 (its high dword masked to the 40-bit address).
    fn spill_loads(&self) -> Vec<(Site, Result<u32, &'static str>)> {
        let input = |r| self.p.inputs.iter().find(|i| i.0 == r).map(|i| i.1);
        let (Some(lo), Some(hi)) = (input(Reg::S(0)), input(Reg::S(1))) else {
            return Vec::new();
        };
        if self.stage != HwStage::Cs
            || !self.layout.iter().any(|p| p.user_data >= 16)
            || self.layout.iter().any(|p| p.user_data < 2 && p.sgprs > 0)
        {
            return Vec::new();
        }
        let high = |t: Term| {
            t == Term::Value(hi)
                || matches!(t, Term::Value(v) if self.sop2(v, &["s_and_b32"]).is_some_and(
                    |(_, [a, b])| matches!((a, b), (x, Term::Const(_)) | (Term::Const(_), x) if x == Term::Value(hi))
                ))
        };
        let mut out = Vec::new();
        for (b, block) in self.p.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                let Fields::Smem {
                    base: Operand::Scalar { code, .. },
                    offset,
                    soffset,
                    ..
                } = &inst.fields
                else {
                    continue;
                };
                if !inst.name().starts_with("s_load_") {
                    continue;
                }
                let word = |k| inst.read(Reg::S(code + k)).map(|v| self.origin(v));
                if word(0) != Some(Term::Value(lo)) || !word(1).is_some_and(high) {
                    continue;
                }
                let at = match self.term(inst, soffset) {
                    Some(Term::Const(c)) => Ok((*offset as u32).wrapping_add(c).wrapping_add(64)),
                    _ => Err("a computed offset into the root signature's memory"),
                };
                out.push(((b, i), at));
            }
        }
        out
    }

    /// The root parameter and dword that user-data dword `index` holds,
    /// when it is in the root signature's memory.
    fn spilled_param(&self, index: u32) -> Option<(usize, u32)> {
        self.layout.iter().enumerate().find_map(|(i, p)| {
            let dword = index.checked_sub(p.user_data.into())?;
            (p.user_data >= 16 && dword < p.sgprs).then_some((i, dword))
        })
    }

    /// The root parameter and dword `v` holds on entry or loaded from the
    /// root signature's memory.
    fn param_value(&self, v: Value) -> Option<(usize, u32)> {
        match self.inputs[v.0 as usize] {
            Some(reg) => self.param_of(reg),
            None => self.spilled.get(&v).copied(),
        }
    }

    /// The value holding dword `dword` of root parameter `param`.
    fn param_word(&self, param: usize, dword: u32) -> Option<Value> {
        if self.layout[param].user_data >= 16 && self.stage == HwStage::Cs {
            return self.first_spilled.get(&(param, dword)).copied();
        }
        let reg = Reg::S(self.stage.user_sgpr(self.layout[param].user_data) + dword as u8);
        self.p.inputs.iter().find(|i| i.0 == reg).map(|i| i.1)
    }

    /// Optimistic copy propagation: a move's result is its source's
    /// origin, a phi's the one origin its arguments agree on.
    ///
    /// A sweep reads some arguments as this sweep made them and others as
    /// the last one did, so a phi whose arguments are all copies of one
    /// value can see them disagree, stand for itself for a sweep, and
    /// return to the copy once they agree again. Where such phis feed each
    /// other across nested loops that repeats without end (the test
    /// `copies_restored_in_nested_loops_settle`). So after [`MAX_ROUNDS`]
    /// sweeps a phi that still changes stands for itself for good, as any
    /// real join does: that names no wrong origin, only fewer copies.
    fn propagate_copies(&self, n: usize) -> Result<Vec<Term>, ShaderError> {
        let mut o: Vec<Option<Term>> = vec![None; n];
        for &(_, v) in &self.p.inputs {
            o[v.0 as usize] = Some(Term::Value(v));
        }
        let order = crate::ir::reverse_postorder(&self.p.blocks);
        // Once pinning, a phi changes at most twice, and a sweep without
        // such a change settles the moves that follow their sources.
        for round in 0..MAX_ROUNDS + 2 * n + 2 {
            let pinning = round >= MAX_ROUNDS;
            let mut changed = false;
            for &b in &order {
                let block = &self.p.blocks[b];
                for phi in &block.phis {
                    let own = Term::Value(phi.value);
                    let old = o[phi.value.0 as usize];
                    if pinning && old == Some(own) {
                        continue;
                    }
                    let mut args = phi
                        .args
                        .iter()
                        .filter_map(|a| o[a.0 as usize])
                        .filter(|&t| t != own);
                    let new = args
                        .next()
                        .map(|first| if args.all(|t| t == first) { first } else { own })
                        .map(|new| match old {
                            Some(t) if pinning && t != new => own,
                            _ => new,
                        });
                    if new.is_some() && new != old {
                        o[phi.value.0 as usize] = new;
                        changed = true;
                    }
                }
                for inst in &block.insts {
                    for (k, &(_, v)) in inst.writes.iter().enumerate() {
                        let reload = self
                            .spilled
                            .get(&v)
                            .map(|pd| self.first_spilled[pd])
                            .filter(|&first| first != v)
                            .map(Moved::Value);
                        let new = match reload
                            .or_else(|| copy_source(inst, k))
                            .or_else(|| self.restored(inst))
                        {
                            Some(Moved::Const(c)) => Some(Term::Const(c)),
                            Some(Moved::Value(s)) => o[s.0 as usize],
                            None => self
                                .identity_op(inst, k, &o)
                                .or_else(|| self.xor_undone(inst, k, &o))
                                .or(Some(Term::Value(v))),
                        };
                        if new.is_some() && new != o[v.0 as usize] {
                            o[v.0 as usize] = new;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                return Ok(o
                    .into_iter()
                    .enumerate()
                    .map(|(v, t)| t.unwrap_or(Term::Value(Value(v as u32))))
                    .collect());
            }
        }
        Err(ShaderError::Unsupported {
            offset: 0,
            what: "copies that do not settle",
        })
    }

    fn inst(&self, (b, i): Site) -> &'a Inst {
        &self.p.blocks[b].insts[i]
    }

    fn def(&self, v: Value) -> Option<&'a Inst> {
        self.defs[v.0 as usize].map(|s| self.inst(s))
    }

    fn origin(&self, v: Value) -> Term {
        self.origin[v.0 as usize]
    }

    /// The origin of a one-dword operand as `inst` reads it.
    fn term(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code))?)),
            Operand::Const(c) => inline(c).map(Term::Const),
            Operand::Literal(x) => Some(Term::Const(x)),
            _ => None,
        }
    }

    /// Layout index and dword of the root parameter `reg` holds on entry.
    fn param_of(&self, reg: Reg) -> Option<(usize, u32)> {
        let Reg::S(code) = reg else { return None };
        self.layout.iter().enumerate().find_map(|(i, p)| {
            let first = self.stage.user_sgpr(p.user_data);
            let dword = u32::from(code).checked_sub(first.into())?;
            (dword < p.sgprs).then_some((i, dword))
        })
    }

    /// The SOP2 operands of `v` if one of `names` wrote it as dword `half`.
    fn sop2_half(&self, v: Value, half: usize, names: &[&str]) -> Option<(&'a Inst, [Term; 2])> {
        let inst = self.def(v)?;
        if !names.contains(&inst.name()) || inst.writes.get(half)?.1 != v {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        let half_term = |op: &Operand| match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code + half as u8))?)),
            Operand::Const(c) if half == 0 => inline(c).map(Term::Const),
            Operand::Const(c @ 128..=208) => inline(c).map(|lo| Term::Const(((lo as i32) >> 31) as u32)),
            Operand::Literal(x) if half == 0 => Some(Term::Const(x)),
            Operand::Literal(x) if x >> 31 == 0 => Some(Term::Const(0)),
            _ => None,
        };
        Some((inst, [half_term(src.first()?)?, half_term(src.get(1)?)?]))
    }

    /// The SOP2 operands of `v` if an `s_add` with one of `names` wrote it.
    fn sop2(&self, v: Value, names: &[&str]) -> Option<(&'a Inst, [Term; 2])> {
        self.sop2_half(v, 0, names)
    }

    fn sum(&self, t: Term) -> Sum {
        match t {
            Term::Const(c) => Sum {
                base: None,
                offset: c,
                dynamic: None,
            },
            Term::Value(v) => self.sums[v.0 as usize],
        }
    }

    /// Precomputes [`Self::joined_on_taken_edges`] across all phis in `O(P + E)` via Tarjan SCCs.
    fn compute_joined(&self, n: usize) -> Vec<Option<Term>> {
        let mut out = vec![None; n];
        let mut phi_vals = Vec::new();
        let mut phi_id = vec![None; n];
        for (vi, p) in self.phis.iter().enumerate() {
            if p.is_some() {
                phi_id[vi] = Some(phi_vals.len());
                phi_vals.push(Value(vi as u32));
            }
        }
        if phi_vals.is_empty() {
            return out;
        }
        let sccs = tarjan_sccs(phi_vals.len(), |k, nbrs| {
            let v = phi_vals[k];
            let (b, i) = self.phis[v.0 as usize].unwrap();
            let block = &self.p.blocks[b];
            let is_vgpr = matches!(block.phis[i].reg, Reg::V(_));
            for (&a, &q) in block.phis[i].args.iter().zip(&block.preds) {
                if !self.edge_taken(q, b) {
                    continue;
                }
                let raw = if is_vgpr { self.vorigin(a) } else { self.origin(a) };
                if let Term::Value(w) = raw
                    && let Some(wk) = phi_id[w.0 as usize]
                {
                    nbrs.push(wk);
                }
            }
        });
        let mut in_comp = vec![false; phi_vals.len()];
        for comp in sccs {
            for &k in &comp {
                in_comp[k] = true;
            }
            let mut first: Option<Term> = None;
            let mut mismatch = false;
            for &k in &comp {
                let u = phi_vals[k];
                let (b, i) = self.phis[u.0 as usize].unwrap();
                let block = &self.p.blocks[b];
                let is_vgpr = matches!(block.phis[i].reg, Reg::V(_));
                for (&a, &q) in block.phis[i].args.iter().zip(&block.preds) {
                    if !self.edge_taken(q, b) {
                        continue;
                    }
                    let raw = if is_vgpr { self.vorigin(a) } else { self.origin(a) };
                    if let Term::Value(w) = raw
                        && let Some(wk) = phi_id[w.0 as usize]
                        && in_comp[wk]
                    {
                        continue;
                    }
                    let resolved = match raw {
                        Term::Value(w) => out[w.0 as usize].unwrap_or(raw),
                        _ => raw,
                    };
                    match first {
                        None => first = Some(resolved),
                        Some(f) if f == resolved => {}
                        Some(_) => {
                            mismatch = true;
                            break;
                        }
                    }
                }
                if mismatch {
                    break;
                }
            }
            for &k in &comp {
                in_comp[k] = false;
            }
            if !mismatch && let Some(val) = first {
                for &k in &comp {
                    out[phi_vals[k].0 as usize] = Some(val);
                }
            }
        }
        out
    }

    /// Precomputes [`Sum`] across all SSA values `0..n` in `O(V + E)` via Tarjan SCCs.
    fn compute_sums(&self, n: usize) -> Vec<Sum> {
        let mut sums: Vec<Sum> = (0..n)
            .map(|vi| Sum {
                base: None,
                offset: 0,
                dynamic: Some(Value(vi as u32)),
            })
            .collect();
        if n == 0 {
            return sums;
        }
        let is_heap_param = |b: Value| {
            self.param_value(b).is_some_and(|(p, d)| {
                d == 0
                    && matches!(
                        self.layout[p].root,
                        RootType::ResourceHeap | RootType::SamplerHeap | RootType::Table
                    )
            })
        };
        let push_term = |t: Term, nbrs: &mut Vec<usize>| {
            if let Term::Value(w) = t {
                nbrs.push(w.0 as usize);
            }
        };
        let sccs = tarjan_sccs(n, |vi, nbrs| {
            let v = Value(vi as u32);
            if let Some(j) = self.joined[vi]
                && j != Term::Value(v)
            {
                push_term(j, nbrs);
            }
            if let Some((b, i)) = self.phis[vi] {
                let block = &self.p.blocks[b];
                let is_vgpr = matches!(block.phis[i].reg, Reg::V(_));
                for (&a, &q) in block.phis[i].args.iter().zip(&block.preds) {
                    if self.edge_taken(q, b) {
                        push_term(if is_vgpr { self.vorigin(a) } else { self.origin(a) }, nbrs);
                    }
                }
                return;
            }
            if self.inputs[vi].is_some() || self.spilled.contains_key(&v) || self.getpc(v).is_some() {
                return;
            }
            if let Some(unshifted) = self
                .sop2(
                    v,
                    &[
                        "s_lshl1_add_u32",
                        "s_lshl2_add_u32",
                        "s_lshl3_add_u32",
                        "s_lshl4_add_u32",
                    ],
                )
                .map(|(_, [_, u])| u)
                .or_else(|| self.vlshl_add(v).map(|(_, _, u)| u))
                .or_else(|| self.vmad64_addend(v, 0))
            {
                push_term(unshifted, nbrs);
                return;
            }
            if let Some(lane) = self.first_lane_source(v) {
                push_term(self.vorigin(lane), nbrs);
                return;
            }
            if let Some([a, b]) = self
                .sop2(v, &["s_add_u32", "s_add_i32"])
                .map(|(_, t)| t)
                .or_else(|| self.vadd(v))
            {
                push_term(a, nbrs);
                push_term(b, nbrs);
            }
        });
        let term_sum = |t: Term, sums: &[Sum]| match t {
            Term::Const(c) => Sum {
                base: None,
                offset: c,
                dynamic: None,
            },
            Term::Value(w) => sums[w.0 as usize],
        };
        let mut in_comp = vec![false; n];
        for comp in sccs {
            let single_acyclic = if comp.len() == 1 {
                let vi = comp[0];
                let v = Value(vi as u32);
                match self.phis[vi] {
                    Some((b, i)) => {
                        let block = &self.p.blocks[b];
                        let is_vgpr = matches!(block.phis[i].reg, Reg::V(_));
                        !block
                            .phis[i]
                            .args
                            .iter()
                            .zip(&block.preds)
                            .any(|(&a, &q)| {
                                self.edge_taken(q, b)
                                    && (if is_vgpr { self.vorigin(a) } else { self.origin(a) })
                                        == Term::Value(v)
                            })
                    }
                    None => true,
                }
            } else {
                false
            };
            if !single_acyclic {
                for &ui in &comp {
                    in_comp[ui] = true;
                }
                let mut ok = comp.iter().all(|&ui| self.phis[ui].is_some());
                let mut base_found: Option<Value> = None;
                let mut ext_count = 0usize;
                if ok {
                    for &ui in &comp {
                        let (b, i) = self.phis[ui].unwrap();
                        let block = &self.p.blocks[b];
                        let is_vgpr = matches!(block.phis[i].reg, Reg::V(_));
                        let mut any_taken = false;
                        for (&a, &q) in block.phis[i].args.iter().zip(&block.preds) {
                            if !self.edge_taken(q, b) {
                                continue;
                            }
                            any_taken = true;
                            let term = if is_vgpr { self.vorigin(a) } else { self.origin(a) };
                            if let Term::Value(w) = term
                                && in_comp[w.0 as usize]
                            {
                                continue;
                            }
                            ext_count += 1;
                            match term_sum(term, &sums).base.filter(|&hb| self.heap_base(hb)) {
                                Some(hb) if base_found.is_none_or(|prev| prev == hb) => {
                                    base_found = Some(hb);
                                }
                                _ => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if !any_taken || !ok {
                            ok = false;
                            break;
                        }
                    }
                }
                for &ui in &comp {
                    in_comp[ui] = false;
                }
                if ok && ext_count > 0 && let Some(hb) = base_found {
                    for &ui in &comp {
                        sums[ui] = Sum {
                            base: Some(hb),
                            offset: 0,
                            dynamic: Some(Value(ui as u32)),
                        };
                    }
                }
                continue;
            }
            let vi = comp[0];
            let v = Value(vi as u32);
            let dynamic = sums[vi];
            if let Some(joined) = self.joined[vi]
                && joined != Term::Value(v)
            {
                let y = term_sum(joined, &sums);
                if y.base.is_some() {
                    sums[vi] = Sum {
                        base: y.base,
                        offset: y.offset,
                        dynamic: if y.dynamic.is_some() { Some(v) } else { None },
                    };
                    continue;
                }
            }
            if let Some((b, i)) = self.phis[vi] {
                let block = &self.p.blocks[b];
                let is_vgpr = matches!(block.phis[i].reg, Reg::V(_));
                let mut count = 0usize;
                let mut base_found: Option<Value> = None;
                let mut ok = true;
                for (&a, &q) in block.phis[i].args.iter().zip(&block.preds) {
                    if !self.edge_taken(q, b) {
                        continue;
                    }
                    count += 1;
                    let term = if is_vgpr { self.vorigin(a) } else { self.origin(a) };
                    match term_sum(term, &sums).base.filter(|&hb| self.heap_base(hb)) {
                        Some(hb) if base_found.is_none_or(|prev| prev == hb) => {
                            base_found = Some(hb);
                        }
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && count > 0 && let Some(hb) = base_found {
                    sums[vi] = Sum {
                        base: Some(hb),
                        offset: 0,
                        dynamic: Some(v),
                    };
                }
                continue;
            }
            if self.inputs[vi].is_some() || self.spilled.contains_key(&v) {
                sums[vi] = match self.param_value(v) {
                    Some((p, 0))
                        if matches!(
                            self.layout[p].root,
                            RootType::Table
                                | RootType::Descriptor
                                | RootType::ResourceHeap
                                | RootType::SamplerHeap
                        ) =>
                    {
                        Sum {
                            base: Some(v),
                            offset: 0,
                            dynamic: None,
                        }
                    }
                    _ => dynamic,
                };
                continue;
            }
            if self.getpc(v).is_some() {
                sums[vi] = Sum {
                    base: Some(v),
                    offset: 0,
                    dynamic: None,
                };
                continue;
            }
            if let Some(unshifted) = self
                .sop2(
                    v,
                    &[
                        "s_lshl1_add_u32",
                        "s_lshl2_add_u32",
                        "s_lshl3_add_u32",
                        "s_lshl4_add_u32",
                    ],
                )
                .map(|(_, [_, u])| u)
                .or_else(|| self.vlshl_add(v).map(|(_, _, u)| u))
                .or_else(|| self.vmad64_addend(v, 0))
            {
                let y = term_sum(unshifted, &sums);
                // Only for heap/table bases, where lowering reads the SMEM/flat base register
                // directly rather than adding `Address::dynamic` on top of the root address.
                if y.dynamic.is_none() && y.base.is_some_and(is_heap_param) {
                    sums[vi] = Sum {
                        base: y.base,
                        offset: y.offset,
                        dynamic: Some(v),
                    };
                }
                continue;
            }
            if let Some(lane) = self.first_lane_source(v) {
                let y = term_sum(self.vorigin(lane), &sums);
                if y.base.is_some_and(is_heap_param) {
                    sums[vi] = Sum {
                        base: y.base,
                        offset: y.offset,
                        dynamic: Some(v),
                    };
                }
                continue;
            }
            let Some([a, b]) = self
                .sop2(v, &["s_add_u32", "s_add_i32"])
                .map(|(_, t)| t)
                .or_else(|| self.vadd(v))
            else {
                continue;
            };
            let (x, y) = (term_sum(a, &sums), term_sum(b, &sums));
            if x.base.is_some() && y.base.is_some() || x.dynamic.is_some() && y.dynamic.is_some() {
                continue;
            }
            sums[vi] = Sum {
                base: x.base.or(y.base),
                offset: x.offset.wrapping_add(y.offset),
                dynamic: x.dynamic.or(y.dynamic),
            };
        }
        sums
    }

    /// The `s_getpc_b64` writing `lo` as its low dword.
    fn getpc(&self, lo: Value) -> Option<&'a Inst> {
        let inst = self.def(lo)?;
        (inst.name() == "s_getpc_b64" && inst.writes.first().is_some_and(|w| w.1 == lo))
            .then_some(inst)
    }

    /// `v_lshl_add_u32 dst, src0, shift, unshifted` writing `v`: `(src0, shift, unshifted)`.
    fn vlshl_add(&self, v: Value) -> Option<(Term, u32, Term)> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        if inst.name() != "v_lshl_add_u32"
            || inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        let Term::Const(shift) = self.vterm(inst, f.src.get(1)?)? else {
            return None;
        };
        Some((
            self.vterm(inst, f.src.first()?)?,
            shift & 31,
            self.vterm(inst, f.src.get(2)?)?,
        ))
    }

    /// `v_mad_u64_u32` or `v_mad_i64_i32` writing `v` as output dword `half` (`0` or `1`):
    /// `(lo_out, [src0, src1], signed, [add_lo, add_hi])`.
    fn vmad64(&self, v: Value, half: usize) -> Option<(Value, [Term; 2], bool, [Term; 2])> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        let signed = match inst.name() {
            "v_mad_u64_u32" => false,
            "v_mad_i64_i32" => true,
            _ => return None,
        };
        if inst.write(Reg::V(reg + half as u8)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        let lo_out = inst.write(Reg::V(reg))?;
        let src0 = self.vterm(inst, f.src.first()?)?;
        let src1 = self.vterm(inst, f.src.get(1)?)?;
        let add_op = f.src.get(2)?;
        let add_word = |h: usize| -> Option<Term> {
            match *add_op {
                Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
                Operand::Scalar { code, .. } => {
                    Some(self.origin(inst.read(Reg::S(code + h as u8))?))
                }
                Operand::Vector { reg: r, .. } => {
                    Some(self.vorigin(inst.read(Reg::V(r + h as u8))?))
                }
                Operand::Const(c) if h == 0 => inline(c).map(Term::Const),
                Operand::Const(c @ 128..=208) => {
                    inline(c).map(|lo| Term::Const(((lo as i32) >> 31) as u32))
                }
                Operand::Literal(x) if h == 0 => Some(Term::Const(x)),
                Operand::Literal(x) if x >> 31 == 0 => Some(Term::Const(0)),
                _ => None,
            }
        };
        Some((lo_out, [src0, src1], signed, [add_word(0)?, add_word(1)?]))
    }

    /// Dword `half` (`0` or `1`) of the 64-bit addend `src2` of `v_mad_*64_*32` writing `v` as
    /// its `half` output dword.
    fn vmad64_addend(&self, v: Value, half: usize) -> Option<Term> {
        self.vmad64(v, half).map(|(_, _, _, add)| add[half])
    }

    /// The operands of a VALU addition (`v_add_co_u32`, `v_add_nc_u32`)
    /// writing `v`, through VGPR copies of scalars.
    fn vadd(&self, v: Value) -> Option<[Term; 2]> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if !matches!(inst.name(), "v_add_co_u32" | "v_add_nc_u32")
            || f.dst.is_none_or(|d| !matches!(d, Operand::Vector { .. }))
            || inst.write(Reg::V(match f.dst? {
                Operand::Vector { reg, .. } => reg,
                _ => return None,
            })) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
        ])
    }

    /// The origin of a VALU operand.
    fn vterm(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(self.vorigin(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// A VGPR value's origin through `v_mov_b32` copies.
    fn vorigin(&self, v: Value) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let Some(t) = f.src.first().and_then(|s| self.vterm(inst, s))
        {
            return t;
        }
        self.origin(v)
    }

    /// A global memory address: an SGPR pair plus a VGPR offset, or a VGPR
    /// pair.
    fn flat_address(
        &self,
        inst: &Inst,
        addr: Option<&Operand>,
        saddr: Option<&Operand>,
        offset: i16,
    ) -> Result<Address, &'static str> {
        let Some(&Operand::Vector { reg, .. }) = addr else {
            return Err("a global address not in VGPRs");
        };
        let off = offset as i32 as u32;
        match saddr {
            Some(&Operand::Scalar { code, .. }) if code != NULL => {
                let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("an address not in SGPRs");
                };
                let v = inst.read(Reg::V(reg)).ok_or("a global address")?;
                let mut extra = self.sum(self.vorigin(v));
                if extra.base.is_some() {
                    return Err("an offset from a root parameter");
                }
                extra.offset = extra.offset.wrapping_add(off);
                self.address(lo, hi, extra)
            }
            _ => {
                let word = |k| inst.read(Reg::V(reg + k)).map(|v| self.vorigin(v));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("a global address");
                };
                self.address(
                    lo,
                    hi,
                    Sum {
                        base: None,
                        offset: off,
                        dynamic: None,
                    },
                )
            }
        }
    }

    /// The address `lo` and `hi` hold, plus `extra` (an instruction's
    /// immediate and offset register); title memory ([`MEMORY`]) when the
    /// program computes it from data, including from a live-in SGPR the
    /// root signature does not fill (the console's user data carries raw
    /// addresses; guessed: such a pair is always a title-memory address).
    /// Addresses into descriptor heaps or the code are traced or refused.
    fn address(&self, lo: Term, hi: Term, extra: Sum) -> Result<Address, &'static str> {
        const DATA: &[&str] = &[
            "an address not from a root parameter",
            "an address loaded from memory",
            "an address from a phi",
            "an address with two computed terms",
            "an address with an untracked high half",
            "an address from a root parameter that is no address",
            "an address in an SGPR that is no root parameter",
            "a constant address",
        ];
        match self.traced_address(lo, hi, extra) {
            // The code's table at an offset the program computes from the
            // counter (made uniform with a first-lane read, say): lowering
            // gives the counter the code offset, so `lo` is the byte offset.
            Err(_) if self.reaches_pc(lo) => match (lo, extra.dynamic) {
                (Term::Value(v), None) => Ok(Address {
                    param: CODE,
                    offset: extra.offset,
                    dynamic: Some(v),
                }),
                _ => Err("an address relative to the program counter"),
            },
            Err(why) if DATA.contains(&why) => Ok(Address {
                param: MEMORY,
                offset: 0,
                dynamic: None,
            }),
            a => a,
        }
    }

    /// Whether the low dword `t` of an address is the program counter's plus
    /// offsets (a table in the code, which is not title memory on the host),
    /// through copies, additions, lane reads and phis.
    fn reaches_pc(&self, t: Term) -> bool {
        const ADDITIVE: &[&str] = &[
            "s_mov_b32",
            "s_add_u32",
            "s_add_i32",
            "s_or_b32",
            "v_mov_b32",
            "v_add_co_u32",
            "v_add_nc_u32",
            "v_add3_u32",
            "v_lshl_add_u32",
            "v_or_b32",
            "v_readfirstlane_b32",
        ];
        let Term::Value(v) = t else { return false };
        let (mut stack, mut seen) = (vec![v], std::collections::HashSet::new());
        while let Some(v) = stack.pop() {
            if !seen.insert(v) || seen.len() > 256 {
                continue;
            }
            if let Some((b, i)) = self.phis[v.0 as usize] {
                stack.extend(&self.p.blocks[b].phis[i].args);
                continue;
            }
            let Some(inst) = self.def(v) else { continue };
            if inst.name() == "s_getpc_b64" {
                return true;
            }
            if ADDITIVE.contains(&inst.name()) {
                // Not through the lane mask a vector instruction runs under.
                let operand =
                    |r: &&(Reg, Value)| !matches!(r.0, Reg::S(c) if c == EXEC || c == EXEC + 1);
                stack.extend(inst.reads.iter().filter(operand).map(|r| r.1));
            }
        }
        false
    }

    /// [`Self::address`] traced to a root parameter or the code.
    fn traced_address(&self, lo: Term, hi: Term, extra: Sum) -> Result<Address, &'static str> {
        let s = self.sum(lo);
        if s.dynamic.is_some() && extra.dynamic.is_some() {
            if s.base.is_some_and(|b| self.heap_base(b)) {
                return Err("an address into a descriptor heap with two computed terms");
            }
            return Err("an address with two computed terms");
        }
        let base = s.base.ok_or_else(|| self.why_no_base(s.dynamic))?;
        if let Some(pc) = self.getpc(base) {
            // A table in the code: the base is the next instruction's offset.
            let high = pc.writes.get(1).map(|w| w.1);
            if !self.high_half(hi, high) && !high.is_some_and(|h| self.carried(hi, h)) {
                return Err("an address with an untracked high half");
            }
            return Ok(Address {
                param: CODE,
                offset: (pc.machine.offset as u32 + 4)
                    .wrapping_add(s.offset)
                    .wrapping_add(extra.offset),
                dynamic: s.dynamic.or(extra.dynamic),
            });
        }
        let (param, _) = self
            .param_value(base)
            .ok_or("an address not from a root parameter")?;
        let high = match self.layout[param].root {
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap => None,
            RootType::Descriptor => Some(
                self.param_word(param, 1)
                    .ok_or("a root descriptor without its high half")?,
            ),
            _ => return Err("an address from a root parameter that is no address"),
        };
        // shortcut: with a computed term, a high half that adds anything to the
        // base's is taken as the carry of that term (offsets below 4 GB).
        let dynamic = s.dynamic.or(extra.dynamic);
        let low_terms = [lo, extra.dynamic.map_or(Term::Const(0), Term::Value)];
        if !self.high_half_with(hi, high, &low_terms, Some(base))
            && !(dynamic.is_some() && high.is_some_and(|h| self.carried(hi, h)))
        {
            return Err(if high.is_none() {
                "an address into a descriptor heap with an untracked high half"
            } else {
                "an address with an untracked high half"
            });
        }
        Ok(Address {
            param,
            offset: s.offset.wrapping_add(extra.offset),
            dynamic: s.dynamic.or(extra.dynamic),
        })
    }

    /// Whether `v` is a root parameter holding a descriptor heap address.
    fn heap_base(&self, v: Value) -> bool {
        self.param_value(v).is_some_and(|(p, _)| {
            matches!(
                self.layout[p].root,
                RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap
            )
        })
    }

    /// `hi` is `high` (0 for a table), possibly plus a carry.
    fn high_half(&self, hi: Term, high: Option<Value>) -> bool {
        self.high_half_with(hi, high, &[], None)
    }

    /// [`Self::high_half`] when `low_terms` (excluding `base`) hold the 32-bit
    /// offset terms added to the low half of the address.
    fn high_half_with(
        &self,
        hi: Term,
        high: Option<Value>,
        low_terms: &[Term],
        base: Option<Value>,
    ) -> bool {
        let mut steps = 0usize;
        let mut memo = std::collections::HashMap::new();
        self.high_half_inner(hi, high, low_terms, base, &mut Vec::new(), &mut steps, &mut memo)
            .0
    }

    /// Returns `(valid, min_cycle_level, used_backedge)`, where `0` is grounded in a base case
    /// and `1..=seen.len()` is the earliest ancestor phi in `seen` reached along a cycle back-edge.
    #[allow(clippy::too_many_arguments)]
    fn high_half_inner(
        &self,
        hi: Term,
        high: Option<Value>,
        low_terms: &[Term],
        base: Option<Value>,
        seen: &mut Vec<Value>,
        steps: &mut usize,
        memo: &mut std::collections::HashMap<(Value, Vec<Term>), bool>,
    ) -> (bool, usize, bool) {
        *steps += 1;
        if *steps > MAX_HIGH_STEPS {
            return (false, usize::MAX, true);
        }
        let hi = self.strip_high_mask(hi);
        let want = high.map_or(Term::Const(0), Term::Value);
        if hi == want {
            return (true, 0, false);
        }
        let Term::Value(v) = hi else {
            return (false, usize::MAX, false);
        };
        // The high half cut to a 40-bit or 48-bit address (`s_bfe_u32` / `v_bfe_u32` from bit 0).
        if let Some((_, [a, Term::Const(0x8_0000 | 0x10_0000)])) = self.sop2(v, &["s_bfe_u32"]) {
            return self.high_half_inner(a, high, low_terms, base, seen, steps, memo);
        }
        if let Some((a, 0, 8 | 16)) = self.vbfe_u32(v) {
            return self.high_half_inner(a, high, low_terms, base, seen, steps, memo);
        }
        if let Some([a, b]) = self
            .sop2(v, &["s_addc_u32"])
            .map(|(_, ops)| ops)
            .or_else(|| self.vadd_ci(v))
        {
            if a == want && self.offset_high_half(b, low_terms, base)
                || b == want && self.offset_high_half(a, low_terms, base)
            {
                return (true, 0, false);
            }
            // Carry added onto an already-valid high half (e.g. a second offset's carry).
            if b == Term::Const(0) {
                return self.high_half_inner(a, high, low_terms, base, seen, steps, memo);
            }
            if a == Term::Const(0) {
                return self.high_half_inner(b, high, low_terms, base, seen, steps, memo);
            }
        }
        if let Some((lo_out, _, _, [_, add_hi])) = self.vmad64(v, 1)
            && self
                .extract_low_offset(low_terms, base)
                .is_some_and(|l| l.pair_lo == Some(lo_out))
        {
            return self.high_half_inner(add_hi, high, low_terms, base, seen, steps, memo);
        }
        if let Some(lane) = self.first_lane_source(v) {
            return self.high_half_inner(
                self.vorigin(lane),
                high,
                low_terms,
                base,
                seen,
                steps,
                memo,
            );
        }
        let key = (v, low_terms.to_vec());
        if let Some(&cached) = memo.get(&key) {
            return if cached {
                (true, 0, false)
            } else {
                (false, usize::MAX, false)
            };
        }
        if let Some(idx) = seen.iter().position(|&s| s == v) {
            return (true, idx + 1, true);
        }
        if seen.len() >= MAX_ROUNDS {
            return (false, usize::MAX, true);
        }
        if let Some((cond, t, f)) = self.picked(v).or_else(|| self.lane_picked(v)) {
            let step_pick = |arm_true: bool| -> Vec<Term> {
                low_terms
                    .iter()
                    .map(|&lt| {
                        let Term::Value(lv) = lt else { return lt };
                        match self.picked(lv).or_else(|| self.lane_picked(lv)) {
                            Some((c, lt_t, lt_f)) if c == cond => {
                                if arm_true { lt_t } else { lt_f }
                            }
                            _ => lt,
                        }
                    })
                    .collect()
            };
            seen.push(v);
            let (ok_t, lv_t, be_t) =
                self.high_half_inner(t, high, &step_pick(true), base, seen, steps, memo);
            let (ok_f, lv_f, be_f) = if ok_t {
                self.high_half_inner(f, high, &step_pick(false), base, seen, steps, memo)
            } else {
                (false, usize::MAX, false)
            };
            seen.pop();
            let used_be = be_t || be_f;
            let res = if ok_t && ok_f {
                (true, lv_t.min(lv_f), used_be)
            } else {
                (false, usize::MAX, used_be)
            };
            if res.0 && res.1 == 0 {
                memo.insert(key, true);
            } else if !res.0 && !res.2 {
                memo.insert(key, false);
            }
            return res;
        }
        if let Some((b, i)) = self.phis[v.0 as usize] {
            let block = &self.p.blocks[b];
            let phi = &block.phis[i];
            let is_vgpr = matches!(phi.reg, Reg::V(_));
            let my_level = seen.len() + 1;
            seen.push(v);
            let mut count = 0usize;
            let mut min_level = usize::MAX;
            let mut used_be = false;
            for (edge, (&arg, &q)) in phi.args.iter().zip(&block.preds).enumerate() {
                if !self.edge_taken(q, b) {
                    continue;
                }
                count += 1;
                let term = if is_vgpr {
                    self.vorigin(arg)
                } else {
                    self.origin(arg)
                };
                let edge_low: Vec<Term> = low_terms
                    .iter()
                    .map(|&lt| {
                        let Term::Value(lv) = lt else { return lt };
                        match self.phis[lv.0 as usize] {
                            Some((lb, li)) if lb == b => {
                                let lphi = &block.phis[li];
                                let la = lphi.args[edge];
                                if matches!(lphi.reg, Reg::V(_)) {
                                    self.vorigin(la)
                                } else {
                                    self.origin(la)
                                }
                            }
                            _ => lt,
                        }
                    })
                    .collect();
                let (ok, lv, be) =
                    self.high_half_inner(term, high, &edge_low, base, seen, steps, memo);
                used_be |= be;
                if !ok {
                    seen.pop();
                    if !used_be {
                        memo.insert(key, false);
                    }
                    return (false, usize::MAX, used_be);
                }
                min_level = min_level.min(lv);
            }
            seen.pop();
            if count > 0 && min_level < my_level {
                if min_level == 0 {
                    memo.insert(key, true);
                }
                return (true, min_level, used_be);
            }
            if count == 0 && !used_be {
                memo.insert(key, false);
            }
            return (false, usize::MAX, true);
        }
        memo.insert(key, false);
        (false, usize::MAX, false)
    }

    /// Strips bit-preserving masks (`s_and_b32`, `s_and_b64`, `v_and_b32`, and zero-offset
    /// `s_bfe_u32` / `v_bfe_u32`) from `hi` when every possible set bit of the inner value is kept.
    fn strip_high_mask(&self, mut hi: Term) -> Term {
        for _ in 0..4 {
            let Term::Value(v) = hi else { break };
            let masked = self
                .sop2_half(v, 0, &["s_and_b32", "s_and_b64"])
                .or_else(|| self.sop2_half(v, 1, &["s_and_b64"]))
                .map(|(_, ops)| ops)
                .or_else(|| self.vand_b32(v));
            if let Some([t, Term::Const(mask)] | [Term::Const(mask), t]) = masked
                && self.possible_bits(t, 4) & !mask == 0
            {
                hi = t;
                continue;
            }
            let bfe = self
                .sop2(v, &["s_bfe_u32"])
                .and_then(|(_, [t, b])| match b {
                    Term::Const(pack) => Some((t, pack & 31, (pack >> 16) & 0x7f)),
                    Term::Value(_) => None,
                })
                .or_else(|| self.vbfe(v, false));
            if let Some((t, 0, width)) = bfe
                && width > 0
            {
                let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
                if self.possible_bits(t, 4) & !mask == 0 {
                    hi = t;
                    continue;
                }
            }
            break;
        }
        hi
    }

    /// Canonicalizes scalar/VGPR copies and `v_readfirstlane_b32` reads without traversing ALU ops.
    fn canon_term(&self, mut t: Term) -> Term {
        for _ in 0..8 {
            let Term::Value(v) = t else { break };
            if let Some(lane) = self.first_lane_source(v) {
                t = self.vorigin(lane);
                continue;
            }
            let next = self.vorigin(v);
            if next == t {
                break;
            }
            t = next;
        }
        t
    }

    fn canon_pair(&self, a: Term, b: Term) -> [Term; 2] {
        let (a, b) = (self.canon_term(a), self.canon_term(b));
        if a <= b { [a, b] } else { [b, a] }
    }

    /// Whether `t` reaches `target` as an unshifted additive base term.
    fn reaches_base(&self, t: Term, target: Value) -> bool {
        let mut stack = vec![t];
        let mut steps = 0usize;
        while let Some(cur) = stack.pop() {
            steps += 1;
            if steps > 32 {
                return false;
            }
            if cur == Term::Value(target) {
                return true;
            }
            let c = self.canon_term(cur);
            if c == Term::Value(target) {
                return true;
            }
            let Term::Value(v) = c else { continue };
            if let Some(j) = self.joined_on_taken_edges(c)
                && j != c
            {
                stack.push(j);
                continue;
            }
            if let Some([a, b]) = self
                .sop2(v, &["s_add_u32", "s_add_i32"])
                .map(|(_, ops)| ops)
                .or_else(|| self.vadd(v))
            {
                stack.extend([a, b]);
            } else if let Some(u) = self
                .sop2(
                    v,
                    &[
                        "s_lshl1_add_u32",
                        "s_lshl2_add_u32",
                        "s_lshl3_add_u32",
                        "s_lshl4_add_u32",
                    ],
                )
                .map(|(_, [_, u])| u)
                .or_else(|| self.vlshl_add(v).map(|(_, _, u)| u))
                .or_else(|| self.vmad64_addend(v, 0))
            {
                stack.push(u);
            }
        }
        false
    }

    /// Extracts the unique non-constant 32-bit dynamic offset added to `base` in `low_terms`,
    /// recording its exact def-use shift, product, or 64-bit pair low output.
    fn extract_low_offset(&self, low_terms: &[Term], base: Option<Value>) -> Option<LowOffset> {
        let mut found: Option<LowOffset> = None;
        let mut push_found = |item: LowOffset| -> bool {
            if found.is_some() {
                return false;
            }
            found = Some(item);
            true
        };
        let base_or_const = |x: Term| {
            matches!(self.canon_term(x), Term::Const(_))
                || base.is_some_and(|hb| self.reaches_base(x, hb))
        };
        let mut stack = low_terms.to_vec();
        let mut steps = 0usize;
        while let Some(raw) = stack.pop() {
            steps += 1;
            if steps > 32 {
                return None;
            }
            if base.is_some_and(|hb| raw == Term::Value(hb)) {
                continue;
            }
            let c = self.canon_term(raw);
            if matches!(c, Term::Const(_)) || base.is_some_and(|hb| c == Term::Value(hb)) {
                continue;
            }
            let Term::Value(v) = c else { continue };
            if let Some(j) = self.joined_on_taken_edges(c)
                && j != c
            {
                stack.push(j);
                continue;
            }
            if let Some([a, b]) = self
                .sop2(v, &["s_add_u32", "s_add_i32"])
                .map(|(_, ops)| ops)
                .or_else(|| self.vadd(v))
                && (base_or_const(a) || base_or_const(b))
            {
                stack.extend([a, b]);
                continue;
            }
            let lshl_add = self
                .sop2(
                    v,
                    &[
                        "s_lshl1_add_u32",
                        "s_lshl2_add_u32",
                        "s_lshl3_add_u32",
                        "s_lshl4_add_u32",
                    ],
                )
                .map(|(inst, [x, u])| {
                    let k = u32::from(inst.name().as_bytes()[6] - b'0');
                    (x, k, u)
                })
                .or_else(|| self.vlshl_add(v));
            if let Some((x, k, u)) = lshl_add
                && base_or_const(u)
            {
                stack.push(u);
                if k == 0 {
                    stack.push(x);
                } else {
                    let src = self.canon_term(x);
                    if !push_found(LowOffset {
                        raw: Term::Value(v),
                        shifted: Some((src, k)),
                        product: Some((self.canon_pair(src, Term::Const(1u32 << k)), k == 31)),
                        pair_lo: None,
                    }) {
                        return None;
                    }
                }
                continue;
            }
            if let Some((lo_out, [a, b], signed, [add_lo, _])) = self.vmad64(v, 0)
                && base_or_const(add_lo)
            {
                stack.push(add_lo);
                let (ca, cb) = (self.canon_term(a), self.canon_term(b));
                let shifted = (!signed)
                    .then(|| match (ca, cb) {
                        (x, Term::Const(m)) | (Term::Const(m), x)
                            if m > 1 && m.is_power_of_two() =>
                        {
                            Some((x, m.trailing_zeros()))
                        }
                        _ => None,
                    })
                    .flatten();
                if !push_found(LowOffset {
                    raw: Term::Value(v),
                    shifted,
                    product: Some((self.canon_pair(ca, cb), false)),
                    pair_lo: Some(lo_out),
                }) {
                    return None;
                }
                continue;
            }
            if !push_found(self.classify_low_leaf(v)) {
                return None;
            }
        }
        found
    }

    fn classify_low_leaf(&self, v: Value) -> LowOffset {
        if let Some((x, k)) = self.ulshl32(v) {
            let src = self.canon_term(x);
            if k == 0 {
                return match src {
                    Term::Value(sv) if sv != v => self.classify_low_leaf(sv),
                    _ => LowOffset {
                        raw: src,
                        shifted: None,
                        product: None,
                        pair_lo: None,
                    },
                };
            }
            return LowOffset {
                raw: Term::Value(v),
                shifted: Some((src, k)),
                product: Some((self.canon_pair(src, Term::Const(1u32 << k)), k == 31)),
                pair_lo: None,
            };
        }
        if let Some((lo_out, src_lo, shift, signed_in)) = self.lshl64(v, 0) {
            let src = self.canon_term(src_lo);
            return LowOffset {
                raw: Term::Value(v),
                shifted: Some((src, shift)),
                product: (!signed_in).then(|| {
                    (
                        self.canon_pair(src, Term::Const(1u32 << shift)),
                        shift == 31,
                    )
                }),
                pair_lo: Some(lo_out),
            };
        }
        if let Some([a, b]) = self.mul_lo32(v) {
            let (ca, cb) = (self.canon_term(a), self.canon_term(b));
            let shifted = match (ca, cb) {
                (x, Term::Const(m)) | (Term::Const(m), x) if m > 1 && m.is_power_of_two() => {
                    Some((x, m.trailing_zeros()))
                }
                _ => None,
            };
            return LowOffset {
                raw: Term::Value(v),
                shifted,
                product: Some((self.canon_pair(ca, cb), false)),
                pair_lo: None,
            };
        }
        LowOffset {
            raw: Term::Value(v),
            shifted: None,
            product: None,
            pair_lo: None,
        }
    }

    /// Whether `t` is 0 or the exact upper 32 bits of the dynamic 32-bit offset in `low_terms`
    /// (excluding `base`), matched by SSA def-use and exact shift/multiplier rather than graph
    /// reachability.
    fn offset_high_half(&self, t: Term, low_terms: &[Term], base: Option<Value>) -> bool {
        let t = self.strip_high_mask(self.canon_term(self.strip_high_mask(t)));
        if t == Term::Const(0) {
            return true;
        }
        let Term::Value(v) = t else { return false };
        let Some(low) = self.extract_low_offset(low_terms, base) else {
            return false;
        };
        if let Some((lo_out, src_lo, shift, signed_in)) = self.lshl64(v, 1) {
            return low.pair_lo == Some(lo_out)
                || (!signed_in && low.shifted == Some((self.canon_term(src_lo), shift)));
        }
        if let Some((lo_out, [a, b], signed, [Term::Const(0), Term::Const(0)])) = self.vmad64(v, 1)
        {
            let ops = self.canon_pair(a, b);
            return low.pair_lo == Some(lo_out)
                || low
                    .product
                    .is_some_and(|(p, sh31)| p == ops && !(signed && sh31));
        }
        if let Some(([a, b], signed)) = self.mul_hi32(v) {
            let ops = self.canon_pair(a, b);
            return low
                .product
                .is_some_and(|(p, sh31)| p == ops && !(signed && sh31));
        }
        if let Some((x, r)) = self.ulshr32(v) {
            return low.shifted == Some((self.canon_term(x), 32 - r));
        }
        if let Some((x, off, width)) = self.ubfe32(v) {
            return off > 0
                && off + width == 32
                && low.shifted == Some((self.canon_term(x), width));
        }
        if let Some((x, r)) = self.ashr32(v) {
            let cx = self.canon_term(x);
            return (r == 31 && low.raw == cx) || low.shifted == Some((cx, 32 - r));
        }
        if let Some((x, off, width)) = self.ibfe32(v) {
            let cx = self.canon_term(x);
            return off > 0
                && off + width == 32
                && ((off == 31 && low.raw == cx) || low.shifted == Some((cx, width)));
        }
        false
    }

    /// `s_lshl_b32(x, shift)` or `v_lshlrev_b32(shift, x)` writing `v`: `(x, shift & 31)`.
    fn ulshl32(&self, v: Value) -> Option<(Term, u32)> {
        if let Some((_, [x, Term::Const(s)])) = self.sop2(v, &["s_lshl_b32"]) {
            return Some((x, s & 31));
        }
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        if inst.name() != "v_lshlrev_b32"
            || inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        let Term::Const(s) = self.vterm(inst, f.src.first()?)? else {
            return None;
        };
        Some((self.vterm(inst, f.src.get(1)?)?, s & 31))
    }

    /// `s_lshr_b32(x, shift)` or `v_lshrrev_b32(shift, x)` (`1..32`) writing `v`.
    fn ulshr32(&self, v: Value) -> Option<(Term, u32)> {
        let (x, r) = if let Some((_, [x, Term::Const(s)])) = self.sop2(v, &["s_lshr_b32"]) {
            (x, s & 31)
        } else {
            let inst = self.def(v)?;
            let Fields::Valu(f) = &inst.fields else {
                return None;
            };
            let Some(Operand::Vector { reg, .. }) = f.dst else {
                return None;
            };
            if inst.name() != "v_lshrrev_b32"
                || inst.write(Reg::V(reg)) != Some(v)
                || f.abs != 0
                || f.neg != 0
                || f.sdwa.is_some()
                || f.dpp.is_some()
            {
                return None;
            }
            let Term::Const(s) = self.vterm(inst, f.src.first()?)? else {
                return None;
            };
            (self.vterm(inst, f.src.get(1)?)?, s & 31)
        };
        (1..32).contains(&r).then_some((x, r))
    }

    /// `s_ashr_i32(x, shift)` or `v_ashrrev_i32(shift, x)` (`1..32`) writing `v`.
    fn ashr32(&self, v: Value) -> Option<(Term, u32)> {
        let (x, r) = if let Some((_, [x, Term::Const(s)])) = self.sop2(v, &["s_ashr_i32"]) {
            (x, s & 31)
        } else {
            let inst = self.def(v)?;
            let Fields::Valu(f) = &inst.fields else {
                return None;
            };
            let Some(Operand::Vector { reg, .. }) = f.dst else {
                return None;
            };
            if inst.name() != "v_ashrrev_i32"
                || inst.write(Reg::V(reg)) != Some(v)
                || f.abs != 0
                || f.neg != 0
                || f.sdwa.is_some()
                || f.dpp.is_some()
            {
                return None;
            }
            let Term::Const(s) = self.vterm(inst, f.src.first()?)? else {
                return None;
            };
            (self.vterm(inst, f.src.get(1)?)?, s & 31)
        };
        (1..32).contains(&r).then_some((x, r))
    }

    fn ubfe32(&self, v: Value) -> Option<(Term, u32, u32)> {
        self.sop2(v, &["s_bfe_u32"])
            .and_then(|(_, [x, b])| match b {
                Term::Const(pack) => Some((x, pack & 31, (pack >> 16) & 0x7f)),
                Term::Value(_) => None,
            })
            .or_else(|| self.vbfe(v, false))
    }

    fn ibfe32(&self, v: Value) -> Option<(Term, u32, u32)> {
        self.sop2(v, &["s_bfe_i32"])
            .and_then(|(_, [x, b])| match b {
                Term::Const(pack) => Some((x, pack & 31, (pack >> 16) & 0x7f)),
                Term::Value(_) => None,
            })
            .or_else(|| self.vbfe(v, true))
    }

    /// 32-bit low multiply (`s_mul_i32`, `s_mulk_i32`, `v_mul_lo_u32`, `v_mul_lo_i32`) writing `v`.
    fn mul_lo32(&self, v: Value) -> Option<[Term; 2]> {
        if let Some((_, ops)) = self.sop2(v, &["s_mul_i32"]) {
            return Some(ops);
        }
        let inst = self.def(v)?;
        match (inst.name(), &inst.fields) {
            ("s_mulk_i32", Fields::Sop { dst: Some(d), imm: Some(imm), .. })
                if inst.writes.first().is_some_and(|w| w.1 == v) =>
            {
                Some([self.term(inst, d)?, Term::Const(i32::from(*imm as i16) as u32)])
            }
            ("v_mul_lo_u32" | "v_mul_lo_i32", Fields::Valu(f))
                if f.abs == 0 && f.neg == 0 && f.sdwa.is_none() && f.dpp.is_none() =>
            {
                let Some(Operand::Vector { reg, .. }) = f.dst else {
                    return None;
                };
                if inst.write(Reg::V(reg)) != Some(v) {
                    return None;
                }
                Some([
                    self.vterm(inst, f.src.first()?)?,
                    self.vterm(inst, f.src.get(1)?)?,
                ])
            }
            _ => None,
        }
    }

    /// 32-bit high multiply (`s_mul_hi_u32`, `s_mul_hi_i32`, `v_mul_hi_u32`, `v_mul_hi_i32`)
    /// writing `v`: `([src0, src1], signed)`.
    fn mul_hi32(&self, v: Value) -> Option<([Term; 2], bool)> {
        if let Some((inst, ops)) = self.sop2(v, &["s_mul_hi_u32", "s_mul_hi_i32"]) {
            return Some((ops, inst.name() == "s_mul_hi_i32"));
        }
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        let signed = match inst.name() {
            "v_mul_hi_u32" => false,
            "v_mul_hi_i32" => true,
            _ => return None,
        };
        if inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        Some((
            [
                self.vterm(inst, f.src.first()?)?,
                self.vterm(inst, f.src.get(1)?)?,
            ],
            signed,
        ))
    }

    /// For `v` written as output dword `half` (`0` or `1`) of `s_lshl_b64` or `v_lshlrev_b64`
    /// shifting a zero- or sign-extended 32-bit value by `1..32`:
    /// `(low_output_dword, low_input_term, shift, signed_input)`.
    fn lshl64(&self, v: Value, half: usize) -> Option<(Value, Term, u32, bool)> {
        let inst = self.def(v)?;
        if inst.writes.get(half)?.1 != v {
            return None;
        }
        let lo_out = inst.writes.first()?.1;
        let (src_lo, src_hi, shift) = match (inst.name(), &inst.fields) {
            ("s_lshl_b64", Fields::Sop { src, .. }) => {
                let Term::Const(shift) = self.term(inst, src.get(1)?)? else {
                    return None;
                };
                let s = shift & 63;
                if !(1..32).contains(&s) {
                    return None;
                }
                let Operand::Scalar { code, .. } = *src.first()? else {
                    return None;
                };
                if code == NULL {
                    return None;
                }
                (
                    self.origin(inst.read(Reg::S(code))?),
                    self.origin(inst.read(Reg::S(code + 1))?),
                    s,
                )
            }
            ("v_lshlrev_b64", Fields::Valu(f))
                if f.abs == 0 && f.neg == 0 && f.sdwa.is_none() && f.dpp.is_none() =>
            {
                let Term::Const(shift) = self.vterm(inst, f.src.first()?)? else {
                    return None;
                };
                let s = shift & 63;
                if !(1..32).contains(&s) {
                    return None;
                }
                let Operand::Vector { reg, .. } = *f.src.get(1)? else {
                    return None;
                };
                (
                    self.vorigin(inst.read(Reg::V(reg))?),
                    self.vorigin(inst.read(Reg::V(reg + 1))?),
                    s,
                )
            }
            _ => return None,
        };
        if src_hi == Term::Const(0) {
            return Some((lo_out, src_lo, shift, false));
        }
        let Term::Value(hi_v) = self.canon_term(src_hi) else {
            return None;
        };
        let signed = self
            .ashr32(hi_v)
            .is_some_and(|(x, r)| r == 31 && self.canon_term(x) == self.canon_term(src_lo));
        signed.then_some((lo_out, src_lo, shift, true))
    }

    /// `v_add_co_ci_u32` operands `[src0, src1]` when `v` is its VGPR destination.
    fn vadd_ci(&self, v: Value) -> Option<[Term; 2]> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        if inst.name() != "v_add_co_ci_u32"
            || inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
        ])
    }

    /// `v_and_b32` operands `[src0, src1]` when `v` is its VGPR destination.
    fn vand_b32(&self, v: Value) -> Option<[Term; 2]> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        if inst.name() != "v_and_b32"
            || inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
        ])
    }

    /// `v_bfe_u32` (`signed == false`) or `v_bfe_i32` (`signed == true`) writing `v` with constant
    /// offset and width.
    fn vbfe(&self, v: Value, signed: bool) -> Option<(Term, u32, u32)> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        let want = if signed { "v_bfe_i32" } else { "v_bfe_u32" };
        if inst.name() != want
            || inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        let (Term::Const(off), Term::Const(width)) = (
            self.vterm(inst, f.src.get(1)?)?,
            self.vterm(inst, f.src.get(2)?)?,
        ) else {
            return None;
        };
        Some((self.vterm(inst, f.src.first()?)?, off & 31, width & 31))
    }

    fn vbfe_u32(&self, v: Value) -> Option<(Term, u32, u32)> {
        self.vbfe(v, false)
    }

    /// What phi `t` holds in every lane that reads it: the one value its
    /// args bring along the edges some lane takes (a register set inside a
    /// region that only a skip no lane takes bypasses, as
    /// `indirect_light_probes:CS_Alpha_Schedule`'s table address).
    fn joined_on_taken_edges(&self, t: Term) -> Option<Term> {
        let Term::Value(v) = t else { return None };
        self.joined.get(v.0 as usize).copied().flatten()
    }

    /// `hi` adds something (with carry) to `high`: `s_addc_u32` or
    /// `v_add_co_ci_u32` with `high` as an operand.
    fn carried(&self, hi: Term, high: Value) -> bool {
        let Term::Value(v) = hi else { return false };
        if v == high {
            return true;
        }
        if let Some((_, [a, b])) = self.sop2(v, &["s_addc_u32"]) {
            return self.carried(a, high) || self.carried(b, high);
        }
        let Some(inst) = self.def(v) else {
            return false;
        };
        let Fields::Valu(f) = &inst.fields else {
            return false;
        };
        inst.name() == "v_add_co_ci_u32"
            && f.src[..2.min(f.src.len())]
                .iter()
                .any(|o| self.vterm(inst, o).is_some_and(|t| self.carried(t, high)))
    }

    fn why_no_base(&self, dynamic: Option<Value>) -> &'static str {
        let Some(v) = dynamic else {
            return "a constant address";
        };
        if self.param_value(v).is_some() {
            return "an address from a root parameter that is no address";
        }
        if let Some(reg) = self.inputs[v.0 as usize] {
            if let Reg::S(n) = reg {
                self.memory_inputs.borrow_mut().insert(n);
            }
            return "an address in an SGPR that is no root parameter";
        }
        if self
            .leaves(v)
            .any(|l| self.def(l).is_some_and(|i| i.name() == "s_getpc_b64"))
        {
            return "an address relative to the program counter";
        }
        match self.def(v).map(|i| i.name()) {
            Some(n) if n.starts_with("s_load_") => "an address loaded from memory",
            None => "an address from a phi",
            _ => "an address not from a root parameter",
        }
    }

    /// The terms of the additions that compute `v`.
    fn leaves(&self, v: Value) -> Box<dyn Iterator<Item = Value> + '_> {
        match self.sop2(v, &["s_add_u32", "s_add_i32"]) {
            Some((_, terms)) => Box::new(terms.into_iter().flat_map(move |t| match t {
                Term::Value(x) => self.leaves(x),
                Term::Const(_) => Box::new(std::iter::empty()),
            })),
            None => Box::new(std::iter::once(v)),
        }
    }

    fn smem_address(
        &self,
        inst: &Inst,
        base: &Operand,
        offset: i32,
        soffset: &Operand,
    ) -> Result<Address, &'static str> {
        let Operand::Scalar { code, .. } = *base else {
            return Err("an address not in SGPRs");
        };
        if let Some(&at) = self.spill_at.get(&inst.machine.offset) {
            return at.map(|offset| Address {
                param: USER_DATA,
                offset,
                dynamic: None,
            });
        }
        let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
        let (Some(lo), Some(hi)) = (word(0), word(1)) else {
            return Err("an address not in SGPRs");
        };
        let mut extra = self.sum(
            self.term(inst, soffset)
                .ok_or("an unknown offset operand")?,
        );
        if extra.base.is_some() {
            return Err("an offset from a root parameter");
        }
        extra.offset = extra.offset.wrapping_add(offset as u32);
        self.address(lo, hi, extra)
    }

    /// The source of the descriptor operand `op` of `inst`, in block `at`.
    fn descriptor(&self, inst: &Inst, op: &Operand, at: usize) -> Result<Source, &'static str> {
        let Operand::Scalar { code, dwords } = *op else {
            return Err("a descriptor not in SGPRs");
        };
        let mut words = (0..dwords)
            .map(|k| inst.read(Reg::S(code + k)).map(|v| self.origin(v)))
            .collect::<Option<Vec<_>>>()
            .ok_or("a descriptor not in SGPRs")?;
        // A byte access through a buffer descriptor made raw reads what the
        // descriptor's view does with a raw host view. So does a typed load
        // that names its own format (`tbuffer_load_format_*`), which reads
        // the raw view by bytes; `buffer_load_format_*` reads the view's
        // elements, which the cleared stride would change.
        let own_format = matches!(inst.fields, Fields::Buffer { format: Some(_), .. });
        if (own_format || !inst.name().contains("format"))
            && let Some(raw) = self.made_raw(&words)
        {
            words = raw.to_vec();
        }
        if let Some(loaded) = self.stride_cleared(inst, op) {
            words = loaded;
        }
        // Only the lanes that reach the access read the descriptor.
        let mut known = Vec::from_iter(self.lanes.map(|l| (l.reach[at], true)));
        self.classify(&words, &mut Vec::new(), &mut known)
    }

    /// The source of a descriptor's words; `joins` are the phi blocks
    /// being resolved, `known` holds in every lane that reads them.
    fn classify(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Source, &'static str> {
        if let Some(source) = self.phi(words, joins, known)? {
            return Ok(source);
        }
        if let Some(source) = self.select(words, joins, known)? {
            return Ok(source);
        }
        let read = |w: &Term| match *w {
            Term::Value(v) => self.first_lane_source(v),
            Term::Const(_) => None,
        };
        if let Some(sources) = words.iter().map(read).collect::<Option<Vec<_>>>() {
            // Read from the first active lane, as a waterfall loop does: the
            // dwords each lane holds.
            let lanes: Vec<Term> = sources.into_iter().map(Term::Value).collect();
            return self.per_lane(&lanes, joins, known);
        }
        if words.iter().any(|w| read(w).is_some()) {
            return Err("a descriptor partly read from the first lane");
        }
        if let Some(consts) = words
            .iter()
            .map(|w| match *w {
                Term::Const(c) => Some(c),
                Term::Value(_) => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            return Ok(Source::Constant(consts));
        }
        if let Some(source) = self.loaded(words) {
            return Ok(source);
        }
        if let [lo, hi, w2, Term::Const(w3)] = *words
            && (self.computed_buffers || matches!(w2, Term::Const(_)))
        {
            let descriptor_hi = hi;
            let (mut hi, mut bits) = (hi, 0);
            while let Term::Value(v) = hi
                && let Some((x, more)) = self.set_bits(v)
            {
                (hi, bits) = (x, bits | more);
            }
            let address = self.address(
                lo,
                hi,
                Sum {
                    base: None,
                    offset: 0,
                    dynamic: None,
                },
            )?;
            if self.computed_buffers && address.param == MEMORY {
                return Ok(Source::BuiltMemory {
                    control: w3,
                    swizzled: self.possible_bits(descriptor_hi, 8) >> 30 != 0,
                });
            }
            if let Term::Const(w2) = w2 {
                return Ok(Source::Built { address, bits, words: [w2, w3] });
            }
        }
        let loaded = |w: &Term| {
            matches!(*w, Term::Value(v)
                if self.def(v).is_some_and(|i| i.name().starts_with("s_load_")))
        };
        Err(if words.iter().any(loaded) {
            self.changed(words)
        } else {
            "a descriptor neither loaded nor built"
        })
    }

    /// Why a descriptor with a loaded dword is no load's: the first dword that is not the load's
    /// next one, and what made it, so a refusal log says which form a title uses.
    fn changed(&self, words: &[Term]) -> &'static str {
        // The load of the first loaded dword, and where in it the descriptor would start.
        let reference = words.iter().enumerate().find_map(|(k, w)| {
            let Term::Value(v) = *w else { return None };
            let inst = self.def(v).filter(|i| i.name().starts_with("s_load_"))?;
            let at = inst.writes.iter().position(|x| x.1 == v)?;
            Some((inst, at.checked_sub(k)))
        });
        let Some((load, Some(start))) = reference else {
            return changed_why(0, Change::OtherLoad);
        };
        let Some((k, word)) = words
            .iter()
            .enumerate()
            .find(|&(k, w)| load.writes.get(start + k).is_none_or(|x| *w != Term::Value(x.1)))
        else {
            return changed_why(0, Change::OtherLoad);
        };
        let kind = match *word {
            Term::Const(_) => Change::Constant,
            Term::Value(v) if self.phis[v.0 as usize].is_some() => Change::Selected,
            Term::Value(v) => match self.def(v).map(Inst::name) {
                None => Change::Input,
                Some(n) if n.starts_with("s_load_") => Change::OtherLoad,
                Some(n) if n.starts_with("s_cselect") || n.starts_with("s_cmov") => {
                    Change::Selected
                }
                Some(n)
                    if ["s_and", "s_or", "s_xor", "s_nand", "s_nor", "s_xnor", "s_bfi", "s_bfm"]
                        .iter()
                        .any(|p| n.starts_with(p))
                        || n.starts_with("s_bitset") =>
                {
                    Change::Masked
                }
                Some(n) if ["s_add", "s_sub"].iter().any(|p| n.starts_with(p)) => Change::Offset,
                Some(n)
                    if ["s_mul", "s_lshl", "s_lshr", "s_ashr", "s_bfe"]
                        .iter()
                        .any(|p| n.starts_with(p)) =>
                {
                    Change::Scaled
                }
                Some(_) => Change::Computed,
            },
        };
        if let (Change::Masked, Term::Value(v)) = (kind, *word)
            && let Some(inst) = self.def(v)
        {
            return self.masked_why(k, inst, v);
        }
        changed_why(k, kind)
    }

    /// The refusal of dword `k` masked or merged by `inst` (writing `v`): the operation and its
    /// constant operands, the half of a 64-bit one that `v` is, so a log tells the exact form.
    fn masked_why(&self, k: usize, inst: &Inst, v: Value) -> &'static str {
        let Fields::Sop { dst, src, .. } = &inst.fields else {
            return changed_why(k, Change::Masked);
        };
        // Which dword of a 64-bit result `v` is.
        let half = match (dst, inst.writes.iter().find(|w| w.1 == v)) {
            (Some(Operand::Scalar { code, .. }), Some(&(Reg::S(r), _))) => r.wrapping_sub(*code),
            _ => 0,
        };
        let constants: Vec<String> = src
            .iter()
            .filter_map(|op| match *op {
                Operand::Scalar { code, .. } if code != NULL => {
                    match self.origin(inst.read(Reg::S(code + half))?) {
                        Term::Const(c) => Some(format!("{c:#x}")),
                        Term::Value(_) => None,
                    }
                }
                _ => match self.term(inst, op)? {
                    Term::Const(c) => Some(format!("{c:#x}")),
                    Term::Value(_) => None,
                },
            })
            .collect();
        let with = if constants.is_empty() {
            "computed values".to_string()
        } else {
            constants.join(" and ")
        };
        interned(format!(
            "a descriptor changed after loading: dword {k} masked or merged ({} with {with})",
            inst.name()
        ))
    }

    /// Bits that may be set in a scalar term.
    // shortcut: prove bounds through short scalar ALU chains, extend the analysis if
    // a buffer descriptor's swizzle bits are cleared through other instructions.
    fn possible_bits(&self, term: Term, depth: u8) -> u32 {
        let v = match term {
            Term::Const(bits) => return bits,
            Term::Value(v) => v,
        };
        if depth == 0 {
            return u32::MAX;
        }
        let bits = |t| self.possible_bits(t, depth - 1);
        if let Some([a, b]) = self.vand_b32(v) {
            return bits(a) & bits(b);
        }
        if let Some([a, b]) = self.vadd_ci(v) {
            let max = bits(a).saturating_add(bits(b)).saturating_add(1);
            return u32::MAX.checked_shr(max.leading_zeros()).unwrap_or(0);
        }
        if let Some((a, off, width)) = self.vbfe_u32(v) {
            let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
            return (bits(a) >> off) & mask;
        }
        if let Some(inst) = self.def(v)
            && inst.name() == "v_lshrrev_b32"
            && let Fields::Valu(f) = &inst.fields
            && let Some(Operand::Vector { reg, .. }) = f.dst
            && inst.write(Reg::V(reg)) == Some(v)
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let (Some(Term::Const(shift)), Some(a)) = (
                f.src.first().and_then(|s| self.vterm(inst, s)),
                f.src.get(1).and_then(|s| self.vterm(inst, s)),
            )
        {
            return bits(a) >> (shift & 31);
        }
        if let Some((_, _, s, false)) = self.lshl64(v, 1) {
            return (1u32 << s) - 1;
        }
        let Some((inst, [a, b])) = self.sop2(
            v,
            &[
                "s_and_b32",
                "s_or_b32",
                "s_lshl_b32",
                "s_lshr_b32",
                "s_bfe_u32",
                "s_add_u32",
                "s_addc_u32",
            ],
        ) else {
            return u32::MAX;
        };
        match inst.name() {
            "s_and_b32" => bits(a) & bits(b),
            "s_or_b32" => bits(a) | bits(b),
            "s_lshl_b32" => match b {
                Term::Const(shift) => bits(a) << (shift & 31),
                _ => u32::MAX,
            },
            "s_lshr_b32" => match b {
                Term::Const(shift) => bits(a) >> (shift & 31),
                _ => u32::MAX,
            },
            "s_bfe_u32" => match b {
                Term::Const(pack) => {
                    let (off, width) = (pack & 31, (pack >> 16) & 0x7f);
                    let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
                    (bits(a) >> off) & mask
                }
                _ => u32::MAX,
            },
            // A carry can set every bit below the most significant possible bit.
            _ => {
                let max = bits(a).saturating_add(bits(b)).saturating_add(u32::from(inst.name() == "s_addc_u32"));
                u32::MAX.checked_shr(max.leading_zeros()).unwrap_or(0)
            }
        }
    }

    /// Words that phis of one block join, traced per predecessor (words
    /// defined before the join hold on every edge).
    fn phi(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        // Each level multiplies the sources by the predecessor count.
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let preds = &self.p.blocks[block].preds;
        // An edge no lane takes (a skip only helper lanes would take, say)
        // brings nothing a lane reads: it takes a taken edge's source, and
        // lowering joins no value there (`never_taken`).
        let taken: Vec<bool> = preds.iter().map(|&q| self.edge_taken(q, block)).collect();
        let edge = |j: usize, joins: &mut Vec<usize>, known: &mut Known| {
            let edge: Vec<Term> = words
                .iter()
                .map(|w| match phi_of(w) {
                    Some((_, i)) => self.origin(phis[i].args[j]),
                    None => *w,
                })
                .collect();
            self.classify(&edge, joins, known)
        };
        let args = match taken.iter().position(|&t| t) {
            Some(first) if taken.contains(&false) => edge(first, joins, known).and_then(|live| {
                (0..preds.len())
                    .map(|j| if taken[j] { edge(j, joins, known) } else { Ok(live.clone()) })
                    .collect::<Result<_, _>>()
            }),
            _ => (0..preds.len()).map(|j| edge(j, joins, known)).collect::<Result<_, _>>(),
        };
        joins.pop();
        Ok(Some(Source::Phi { block, args: args? }))
    }

    /// Whether some lane goes from block `q` to `b` (true without the
    /// lanes' paths): `Lower::never_taken`'s test.
    fn edge_taken(&self, q: usize, b: usize) -> bool {
        let Some(lanes) = self.lanes else {
            return true;
        };
        if let Some(&taken) = self.edge_taken_cache.borrow().get(&(q, b)) {
            return taken;
        }
        let mut edge = vec![(lanes.reach[q], true)];
        if let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = lanes.exits[q]
            && taken != not_taken
        {
            edge.push((when, taken == b));
        }
        // The very test lowering makes, so both drop the same edges.
        let taken = lanes.bools.meet(&edge);
        self.edge_taken_cache.borrow_mut().insert((q, b), taken);
        taken
    }

    /// Words that `s_cselect` (or `s_cmov`) picks by one SCC value, traced
    /// per outcome (other words are the same for both).
    fn select(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let picks: Vec<_> = words
            .iter()
            .map(|w| match *w {
                Term::Value(v) => self.picked(v),
                Term::Const(_) => None,
            })
            .collect();
        let Some(scc) = picks.iter().flatten().map(|p| p.0).next() else {
            return Ok(None);
        };
        if picks.iter().flatten().any(|p| p.0 != scc) {
            return Err("a descriptor chosen by several conditions");
        }
        // Each level doubles the sources; selects count like joins (a
        // marker no block index reaches).
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(usize::MAX);
        let side = |set: bool| -> Vec<Term> {
            words
                .iter()
                .zip(&picks)
                .map(|(w, p)| match p {
                    Some((_, t, f)) => {
                        if set {
                            *t
                        } else {
                            *f
                        }
                    }
                    None => *w,
                })
                .collect()
        };
        let args = [true, false]
            .into_iter()
            .map(|set| self.classify(&side(set), joins, known))
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Select { scc, args: args? }))
    }

    /// For `v` written by `s_cselect_b32`/`_b64` or `s_cmov_b32`/`_b64`:
    /// the SCC it reads and the dword it is when SCC is set, then clear.
    fn picked(&self, v: Value) -> Option<(Value, Term, Term)> {
        let inst = self.def(v)?;
        let Fields::Sop { dst, src, .. } = &inst.fields else {
            return None;
        };
        let k = inst.writes.iter().position(|w| w.1 == v)? as u8;
        let wide = match inst.name() {
            "s_cselect_b32" | "s_cmov_b32" => false,
            "s_cselect_b64" | "s_cmov_b64" => true,
            _ => return None,
        };
        // A dword of a 32- or 64-bit operand: 64-bit inline integers are
        // sign-extended; 64-bit literals and float constants are not
        // followed.
        let dword = |op: &Operand| match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code + k))?)),
            Operand::Const(c) if !wide => inline(c).map(Term::Const),
            Operand::Const(c @ 128..=208) => {
                let low = inline(c)?;
                Some(Term::Const(if k == 0 {
                    low
                } else {
                    ((low as i32) >> 31) as u32
                }))
            }
            Operand::Literal(x) if !wide => Some(Term::Const(x)),
            _ => None,
        };
        let set = dword(src.first()?)?;
        let clear = if inst.name().starts_with("s_cmov") {
            dword(dst.as_ref()?)?
        } else {
            dword(src.get(1)?)?
        };
        Some((inst.read(Reg::Scc)?, set, clear))
    }

    /// The descriptor of `image_bvh*_intersect_ray`: no view, but the BVH's
    /// address over 256 (dword 0 and the low byte of dword 1) and flags (see
    /// `emit`'s BVH lowering). Only the flags it emulates pass: box sorting,
    /// no box growth, no node bound, triangles returning barycentrics. The
    /// address is a [`Source::Built`] around a root parameter's when the code
    /// shifted one into place (`s_lshr_b64 ..., 8`), else `None`: the nodes
    /// are then at a raw GPU address.
    fn bvh_descriptor(&self, inst: &Inst, op: &Operand) -> Result<Option<Source>, &'static str> {
        const WHY: &str = "a BVH descriptor with other flags";
        let Operand::Scalar { code, dwords: 4 } = *op else {
            return Err("a descriptor not in SGPRs");
        };
        let word = |k| {
            inst.read(Reg::S(code + k))
                .map(|v| self.origin(v))
                .ok_or("a descriptor not in SGPRs")
        };
        let (Term::Const(w2), Term::Const(w3)) = (word(2)?, word(3)?) else {
            return Err(WHY);
        };
        // Dword 1 holds address bits 32-39 under the flags: a constant, or
        // the high half of the shift with the flags set.
        let (high, flags) = match word(1)? {
            Term::Const(c) => (None, c & !0xFF),
            Term::Value(v) => match self.set_bits(v) {
                Some((Term::Value(x), bits)) => (Some(x), bits),
                _ => return Err(WHY),
            },
        };
        // Bit 123 (dword 3 bit 27) is a page-size hint.
        if flags != 0x8000_0000 || w2 != u32::MAX || w3 & !(1 << 27) != 0x8100_03FF {
            return Err(WHY);
        }
        let shifted = match word(0)? {
            Term::Value(lo) => self
                .def(lo)
                .filter(|s| s.name() == "s_lshr_b64" && s.writes.first().map(|w| w.1) == Some(lo)),
            Term::Const(_) => None,
        };
        let Some(shift) = shifted else {
            return if high.is_none() { Ok(None) } else { Err(WHY) };
        };
        let Fields::Sop { src, .. } = &shift.fields else {
            return Err(WHY);
        };
        let pair = match (src.first(), src.get(1).and_then(|s| self.term(shift, s))) {
            (Some(&Operand::Scalar { code, .. }), Some(Term::Const(8))) if code != NULL => {
                let half = |k| shift.read(Reg::S(code + k)).map(|v| self.origin(v));
                half(0).zip(half(1))
            }
            _ => None,
        };
        let (Some((lo, hi)), true) = (pair, high == shift.writes.get(1).map(|w| w.1)) else {
            return Err(WHY);
        };
        let at = Sum {
            base: None,
            offset: 0,
            dynamic: None,
        };
        // A computed base (title memory) is read from the operand.
        match self.address(lo, hi, at) {
            Ok(address) if address.param != MEMORY => Ok(Some(Source::Built {
                address,
                bits: flags,
                words: [w2, w3],
            })),
            _ => Ok(None),
        }
    }

    /// `v` as some other dword with constant bits set.
    fn set_bits(&self, v: Value) -> Option<(Term, u32)> {
        if let Some((_, [x, Term::Const(bits)] | [Term::Const(bits), x])) =
            self.sop2(v, &["s_or_b32"])
        {
            return Some((x, bits));
        }
        let inst = self.def(v)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        match (inst.name(), self.term(inst, src.first()?)?) {
            ("s_bitset1_b32", Term::Const(bit)) => {
                Some((self.origin(inst.read(Reg::S(*code))?), 1 << (bit & 31)))
            }
            _ => None,
        }
    }

    /// The words of a buffer descriptor that `words` makes raw, as the
    /// FidelityFX upscaler's compiler does before byte accesses: the stride
    /// (dword 1 bits 16-29) cleared, the record count times max(stride, 1),
    /// out-of-bounds checking by bytes (dword 3 bits 28-29 set). The bounds
    /// are then the view's size in bytes either way.
    fn made_raw(&self, words: &[Term]) -> Option<[Term; 4]> {
        let [w0, Term::Value(w1), Term::Value(w2), Term::Value(w3)] = *words else {
            return None;
        };
        // `x op c` with `c` either side.
        let with = |v: Value, name: &str, c: u32| match self.sop2(v, &[name])?.1 {
            [x, Term::Const(k)] | [Term::Const(k), x] if k == c => Some(x),
            _ => None,
        };
        let d1 = with(w1, "s_and_b32", 0xC000_FFFF)?;
        let Term::Value(m3) = with(w3, "s_or_b32", 0x3000_0000)? else {
            return None;
        };
        let d3 = with(m3, "s_and_b32", 0xCFFF_FFFF)?;
        let [a, b] = self.sop2(w2, &["s_mul_i32"])?.1;
        let max = |t: Term| match t {
            Term::Value(s) => with(s, "s_max_u32", 1),
            Term::Const(_) => None,
        };
        let (d2, field) = match (max(a), max(b)) {
            (_, Some(f)) => (a, f),
            (Some(f), None) => (b, f),
            (None, None) => return None,
        };
        let Term::Value(field) = field else {
            return None;
        };
        // s_bfe_u32: offset 16, width 14.
        let (_, [x, Term::Const(0x000E_0010)]) = self.sop2(field, &["s_bfe_u32"])? else {
            return None;
        };
        (x == d1).then_some([w0, d1, d2, d3])
    }

    /// The loaded words of a buffer descriptor whose dword 1 is the load's ANDed with a constant
    /// that clears the whole stride and nothing else (bits 16-29 cleared; the base address bits
    /// 0-15 and the swizzle bits 30-31 kept), for a buffer access that is not typed. With
    /// stride 0 the address is base + offsets (the index times the stride adds nothing, RDNA2
    /// ISA 8.1.5), which the loaded view's raw access gives once the index is dropped
    /// (`Resources::unstrided`).
    // Guess: the bounds stay the host view's size. That is the console's when the title's V#
    // checks raw bounds in bytes (OOB_SELECT 3, no swizzle: offset + payload > num_records, in
    // bytes with stride 0, ISA tables 33 and 35); select 0 would put every access out of range,
    // and select 1 would check the dropped index against num_records.
    fn stride_cleared(&self, inst: &Inst, op: &Operand) -> Option<Vec<Term>> {
        let Fields::Buffer { format: None, .. } = inst.fields else {
            return None;
        };
        if inst.name().contains("format") {
            return None;
        }
        let Operand::Scalar { code, dwords: 4 } = *op else {
            return None;
        };
        let mut words = (0..4)
            .map(|k| inst.read(Reg::S(code + k)).map(|v| self.origin(v)))
            .collect::<Option<Vec<_>>>()?;
        let Term::Value(w1) = words[1] else {
            return None;
        };
        const STRIDE: u32 = 0x3FFF_0000;
        let loaded = match self.sop2(w1, &["s_and_b32"])?.1 {
            // The whole stride cleared (a part could leave a stride, and the index would count),
            // nothing else.
            [x, Term::Const(c)] | [Term::Const(c), x] if c & STRIDE == 0 && !c & !STRIDE == 0 => x,
            _ => return None,
        };
        words[1] = loaded;
        self.loaded(&words)?;
        Some(words)
    }

    /// For a `v_readfirstlane_b32` result: the VGPR value it reads.
    fn first_lane_source(&self, v: Value) -> Option<Value> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readfirstlane_b32" || f.sdwa.is_some() || f.dpp.is_some() {
            return None;
        }
        let Some(&Operand::Vector { reg, .. }) = f.src.first() else {
            return None;
        };
        inst.read(Reg::V(reg))
    }

    /// The `v_readfirstlane_b32` results the descriptor operand `op` of
    /// `inst` is, when every dword is one.
    fn first_lane_words(&self, inst: &Inst, op: &Operand) -> Option<Vec<Value>> {
        let Operand::Scalar { code, dwords } = *op else {
            return None;
        };
        (0..dwords)
            .map(|k| match self.origin(inst.read(Reg::S(code + k))?) {
                Term::Value(v) => self.first_lane_source(v).map(|_| v),
                Term::Const(_) => None,
            })
            .collect()
    }

    /// `t` through VGPR copies (of VGPRs or of scalars) and copies.
    fn lane_term(&self, mut t: Term) -> Term {
        for _ in 0..MAX_JOINS * 2 {
            let Term::Value(v) = t else { break };
            let next = self.vorigin(v);
            if next == t {
                break;
            }
            t = next;
        }
        t
    }

    /// Whether `t` is a value each lane holds its own of (one in VGPRs).
    fn varies(&self, t: Term) -> bool {
        let Term::Value(v) = t else { return false };
        let reg = match (self.inputs[v.0 as usize], self.phis[v.0 as usize]) {
            (Some(reg), _) => Some(reg),
            (None, Some((b, i))) => Some(self.p.blocks[b].phis[i].reg),
            (None, None) => self
                .def(v)
                .and_then(|i| i.writes.iter().find(|w| w.1 == v))
                .map(|w| w.0),
        };
        matches!(reg, Some(Reg::V(_)))
    }

    /// [`Self::lane_term`], where every lane in `known` runs each VGPR copy
    /// it follows.
    fn lane_value(&self, mut t: Term, known: &Known) -> Result<Term, &'static str> {
        for _ in 0..MAX_JOINS * 2 {
            let Term::Value(v) = t else { break };
            let next = self.vstep(v);
            if next == t {
                break;
            }
            if self.varies(t)
                && let Some(site) = self.defs[v.0 as usize]
            {
                self.runs(site, known)?;
            }
            t = next;
        }
        Ok(t)
    }

    /// One copy of [`Self::vorigin`]: what the `v_mov_b32` writing `v`
    /// reads, or `v`'s origin.
    fn vstep(&self, v: Value) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let Some(t) = f.src.first().and_then(|s| self.raw(inst, s))
        {
            return t;
        }
        self.origin(v)
    }

    /// A VALU operand as `inst` reads it: a VGPR's value itself, other
    /// operands' origins.
    fn raw(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(Term::Value(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// Fails unless every lane in `known` that reaches the vector
    /// instruction at `site` runs it: one with its EXEC bit clear there
    /// keeps what the registers held before.
    fn runs(&self, site: Site, known: &Known) -> Result<(), &'static str> {
        let Some(lanes) = self.lanes else {
            return Ok(());
        };
        let Some(&Some(guard)) = lanes.guards.get(site.0).and_then(|g| g.get(site.1)) else {
            return Ok(());
        };
        let mut skips = known.clone();
        skips.extend([(lanes.reach[site.0], true), (guard, false)]);
        if meet(lanes, &skips) {
            return Err("a per-lane descriptor written where some lanes may keep the old one");
        }
        Ok(())
    }

    /// The source of descriptor dwords each lane holds its own of (a
    /// waterfall loop reads them from the first active lane): the same in
    /// every lane, loaded per lane from memory a heap address points to, or
    /// picked per lane between such sources, by a lane mask or by the path
    /// the lane took. `known` holds in every lane that accesses them.
    fn per_lane(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Source, &'static str> {
        let words = words
            .iter()
            .map(|&w| self.lane_value(w, known))
            .collect::<Result<Vec<_>, _>>()?;
        let words = &words[..];
        let Some(&lane) = words.iter().find(|&&w| self.varies(w)) else {
            return self.classify(words, joins, known);
        };
        if let Some(source) = self.fetched(words, known)? {
            return Ok(source);
        }
        if let Some(source) = self.lane_select(words, joins, known)? {
            return Ok(source);
        }
        if let Some(source) = self.lane_phi(words, joins, known)? {
            return Ok(source);
        }
        let Term::Value(v) = lane else { unreachable!() };
        let name = self.def(v).map(|i| i.name());
        Err(match name {
            _ if self.inputs[v.0 as usize].is_some() => {
                "a descriptor read from the first lane of an input VGPR"
            }
            None => "a descriptor read from the first lane of a phi",
            Some(n) if n.starts_with("ds_read") => {
                "a descriptor read from the first lane of an LDS read"
            }
            Some(n) if n.starts_with("buffer_load") => {
                "a descriptor read from the first lane of a buffer load"
            }
            Some(n) if n.starts_with("global_load") || n.starts_with("flat_load") => {
                "a descriptor read from the first lane of a partial or mixed load"
            }
            Some(_) => "a descriptor read from the first lane of computed VGPRs",
        })
    }

    /// Dwords of `global_load_dword*` that line up: `first..` of one load,
    /// then the bytes that follow (an image descriptor takes two
    /// `global_load_dwordx4`).
    fn fetched(&self, words: &[Term], known: &Known) -> Result<Option<Source>, &'static str> {
        let at = |w: &Term| -> Option<(Site, usize)> {
            let Term::Value(v) = *w else { return None };
            let site = self.defs[v.0 as usize]?;
            let inst = self.inst(site);
            if !inst.name().starts_with("global_load_dword") {
                return None;
            }
            let k = inst
                .writes
                .iter()
                .filter(|w| matches!(w.0, Reg::V(_)))
                .position(|w| w.1 == v)?;
            Some((site, k))
        };
        let Some(sites) = words.iter().map(at).collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        let address = |site: Site| {
            let inst = self.inst(site);
            let Fields::Flat {
                addr,
                saddr,
                offset,
                ..
            } = &inst.fields
            else {
                return Err("a global load");
            };
            self.flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset)
        };
        let (load, first) = sites[0];
        let a0 = address(load)?;
        for (k, &(site, dword)) in sites.iter().enumerate() {
            self.runs(site, known)?;
            let a = address(site)?;
            let want = a0.offset.wrapping_add(4 * (first + k) as u32);
            if (a.param, a.dynamic) != (a0.param, a0.dynamic)
                || a.offset.wrapping_add(4 * dword as u32) != want
            {
                return Err("a descriptor read from the first lane of loads that do not line up");
            }
        }
        Ok(Some(Source::Fetched {
            load,
            first: first as u8,
        }))
    }

    /// Words that `v_cndmask_b32` picks by one lane mask, traced per outcome
    /// (other words are the same for both).
    fn lane_select(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let picks: Vec<_> = words
            .iter()
            .map(|w| match *w {
                Term::Value(v) => self.lane_picked(v),
                Term::Const(_) => None,
            })
            .collect();
        let Some(mask) = picks.iter().flatten().map(|p| p.0).next() else {
            return Ok(None);
        };
        if picks.iter().flatten().any(|p| p.0 != mask) {
            return Err("a descriptor chosen per lane by several masks");
        }
        for (w, p) in words.iter().zip(&picks) {
            if let (Term::Value(v), Some(_)) = (*w, p)
                && let Some(site) = self.defs[v.0 as usize]
            {
                self.runs(site, known)?;
            }
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(usize::MAX);
        // Each side holds where the lane's bit of the mask is set, or clear.
        let bit = self.lanes.and_then(|l| l.masks.get(&mask).copied());
        let side = |set: bool| -> Vec<Term> {
            words
                .iter()
                .zip(&picks)
                .map(|(w, p)| match p {
                    Some((_, t, f)) => {
                        if set {
                            *t
                        } else {
                            *f
                        }
                    }
                    None => *w,
                })
                .collect()
        };
        let args = [true, false]
            .into_iter()
            .map(|set| {
                let n = known.len();
                known.extend(bit.map(|b| (b, set)));
                let arg = self.per_lane(&side(set), joins, known);
                known.truncate(n);
                arg
            })
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Lanes { mask, args: args? }))
    }

    /// For `v` written by a plain `v_cndmask_b32`: the low dword of the
    /// lane mask it reads, and the dword it is where the lane's bit is set
    /// (src1), then clear (src0).
    fn lane_picked(&self, v: Value) -> Option<(Value, Term, Term)> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        if inst.name() != "v_cndmask_b32"
            || inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.clamp
            || f.omod != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        let [clear, set, Operand::Scalar { code, .. }] = f.src[..] else {
            return None;
        };
        let mask = inst.read(Reg::S(code))?;
        Some((mask, self.raw(inst, &set)?, self.raw(inst, &clear)?))
    }

    /// Per-lane words that phis of one block join, as an if/else on a
    /// per-lane condition leaves them: traced on each edge into the block
    /// that lanes in `known` can take (other words hold on every edge).
    /// Lanes take their own paths ([`Structured`]), so each gets the value
    /// of the edge it came by; an edge they cannot take is `None` (a flow
    /// block's phi joins an arm's dwords with what lanes that skipped the
    /// arm held before, which only they read).
    fn lane_phi(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        let preds = &self.p.blocks[block].preds;
        if preds.iter().any(|&q| self.rpo[q] >= self.rpo[block]) {
            // shortcut: what a lane holds after some passes is not traced.
            return Err("a descriptor read from the first lane of a phi at a loop header");
        }
        let Some(lanes) = self.lanes else {
            return Err("a descriptor read from the first lane of a phi, without the lanes' paths");
        };
        // The other words must be there before the join: defined in a
        // block that dominates it, which (both dominating the access) is
        // one the join does not reach.
        for w in words {
            if let Term::Value(v) = *w
                && phi_of(w).is_none()
                && let Some((d, _)) = self.defs[v.0 as usize]
                && (d == block || self.reaches(block, d))
            {
                return Err("a descriptor partly joined by control flow");
            }
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let mut args = Vec::new();
        for (j, &q) in preds.iter().enumerate() {
            let n = known.len();
            known.push((lanes.reach[q], true));
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = lanes.exits[q]
                && taken != not_taken
            {
                known.push((when, taken == block));
            }
            let arg = if meet(lanes, known) {
                let edge: Vec<Term> = words
                    .iter()
                    .map(|w| match phi_of(w) {
                        Some((_, i)) => Term::Value(phis[i].args[j]),
                        None => *w,
                    })
                    .collect();
                self.per_lane(&edge, joins, known).map(Some)
            } else {
                Ok(None)
            };
            known.truncate(n);
            match arg {
                Ok(a) => args.push(a),
                Err(e) => {
                    joins.pop();
                    return Err(e);
                }
            }
        }
        joins.pop();
        Ok(Some(Source::LanePhi { block, args }))
    }

    /// Whether a path leads from block `from` to block `to` (to `from`
    /// itself only through a back edge).
    fn reaches(&self, from: usize, to: usize) -> bool {
        let mut seen = vec![false; self.p.blocks.len()];
        let mut stack = self.p.blocks[from].exit.successors();
        while let Some(x) = stack.pop() {
            if x == to {
                return true;
            }
            if std::mem::replace(&mut seen[x], true) {
                continue;
            }
            stack.extend(self.p.blocks[x].exit.successors());
        }
        false
    }

    /// Prove a single-block waterfall at the descriptor access. In each
    /// iteration the access serves exactly the matching lanes, and the
    /// back edge keeps exactly the other lanes active. Otherwise, why not.
    ///
    // shortcut: only the accesses are proven. Any other per-lane work in the
    // loop is taken to run once for each lane, as the access does; a body
    // whose effect adds up over the passes a lane waits through (a counter
    // before the mask is narrowed) would differ. Compilers put nothing but
    // the reads, the compares and the access in these loops.
    fn waterfall(
        &self,
        b: usize,
        accesses: &[usize],
        reads: &[Value],
    ) -> Result<Waterfall, &'static str> {
        const LOOP: &str = "a descriptor read from the first lane outside a waterfall loop";
        const SPLIT: &str = "a waterfall loop nested in another or split over blocks";
        const SELECT: &str = "a waterfall loop whose compares do not select the access's lanes";
        const RETRY: &str = "a waterfall loop whose retry does not keep the lanes still waiting";
        let access = accesses[0];
        let block = &self.p.blocks[b];
        let lanes = self.lanes.ok_or(LOOP)?;
        let exit = block.insts.last().ok_or(LOOP)?;
        if !matches!(block.exit, Exit::Branch { taken, not_taken } if taken == b && not_taken != b)
            || exit.name() != "s_cbranch_execnz"
        {
            return Err(LOOP);
        }
        let mask = |low: Option<Value>, high: Option<Value>| {
            let high = if self.p.wave64 { Some(high?) } else { None };
            lanes.mask_bits(low?, high)
        };
        let exec = |inst: &Inst| mask(inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC + 1)));
        // The reads belong to this loop's own block: a loop around this one
        // (one per operand, as LLVM nests them for a per-lane image and
        // sampler) is not followed.
        if reads
            .iter()
            .any(|v| self.defs[v.0 as usize].is_none_or(|at| at.0 != b))
        {
            return Err(SPLIT);
        }
        let selected = exec(&block.insts[access]).ok_or(SELECT)?;
        if accesses
            .iter()
            .any(|&i| exec(&block.insts[i]) != Some(selected))
        {
            return Err(SELECT);
        }
        let retry = exec(exit).ok_or(RETRY)?;
        let first = self.defs[reads[0].0 as usize].ok_or(LOOP)?;
        let remaining = exec(self.inst(first)).ok_or(SELECT)?;
        if reads.iter().any(|v| {
            self.defs[v.0 as usize]
                .is_none_or(|at| at.1 >= access || exec(self.inst(at)) != Some(remaining))
        }) {
            return Err(SELECT);
        }
        let reach = (lanes.reach[b], true);
        let meet = |conditions: &[(Bool, bool)]| self::meet(lanes, &conditions.to_vec());
        let mut out = Waterfall::default();
        let mut required = vec![(remaining, true)];
        for (i, inst) in block.insts[..access].iter().enumerate() {
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            let Some(ty) = inst
                .name()
                .strip_prefix("v_cmp_eq_")
                .or(inst.name().strip_prefix("v_cmpx_eq_"))
            else {
                continue;
            };
            let dwords = match ty {
                "u32" | "i32" => 1,
                "u64" | "i64" => 2,
                _ => continue,
            };
            if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() {
                continue;
            }
            // The first lane's dwords on one side, the lane's own on the
            // other.
            let pair = |first: &Operand, own: &Operand| -> Option<Vec<Value>> {
                let (&Operand::Scalar { code, .. }, &Operand::Vector { reg, .. }) = (first, own)
                else {
                    return None;
                };
                (0..dwords)
                    .map(|k| {
                        let Term::Value(x) = self.origin(inst.read(Reg::S(code + k))?) else {
                            return None;
                        };
                        let lane = inst.read(Reg::V(reg + k))?;
                        let (_, read) = self.defs[x.0 as usize]?;
                        let same = reads.contains(&x)
                            && read < i
                            && self.lane_term(Term::Value(self.first_lane_source(x)?))
                                == self.lane_term(Term::Value(lane));
                        same.then_some(x)
                    })
                    .collect()
            };
            let (Some(a), Some(c)) = (f.src.first(), f.src.get(1)) else {
                continue;
            };
            if let Some(xs) = pair(a, c).or_else(|| pair(c, a)) {
                let code = if inst.name().starts_with("v_cmpx_") {
                    EXEC
                } else if let Some(Operand::Scalar { code, .. }) = f.dst {
                    code
                } else {
                    continue;
                };
                let Some(matched) = mask(inst.write(Reg::S(code)), inst.write(Reg::S(code + 1)))
                else {
                    continue;
                };
                if meet(&[reach, (selected, true), (matched, false)]) {
                    continue;
                }
                required.push((matched, true));
                out.compares.push((b, i));
                for x in xs {
                    if !out.reads.contains(&x) {
                        out.reads.push(x);
                    }
                }
            }
        }
        if reads.iter().any(|v| !out.reads.contains(v))
            || meet(&[reach, (selected, true), (remaining, false)])
        {
            return Err(SELECT);
        }
        // The access must take every match.
        required.extend([reach, (selected, false)]);
        if meet(&required) {
            return Err(SELECT);
        }
        // The retry must keep exactly the lanes that were eligible but did
        // not perform the access.
        if meet(&[reach, (retry, true), (remaining, false)])
            || meet(&[reach, (retry, true), (selected, true)])
            || meet(&[reach, (remaining, true), (selected, false), (retry, false)])
        {
            return Err(RETRY);
        }
        Ok(out)
    }

    /// Consecutive dwords of one scalar load.
    fn loaded(&self, words: &[Term]) -> Option<Source> {
        let Term::Value(v0) = words[0] else {
            return None;
        };
        let load = self.defs[v0.0 as usize]?;
        let inst = self.inst(load);
        if !inst.name().starts_with("s_load_") {
            return None;
        }
        let first = inst.writes.iter().position(|w| w.1 == v0)?;
        let same = words.iter().enumerate().all(|(k, w)| {
            inst.writes
                .get(first + k)
                .is_some_and(|x| *w == Term::Value(x.1))
        });
        same.then_some(Source::Loaded {
            load,
            first: first as u8,
        })
    }
}

/// `text` as a refusal reason, which is `&'static str` throughout the tracer: each distinct text
/// is kept once for the program's life.
// shortcut: formatted reasons are leaked, one per distinct text (a handful per title); give
// refusals an owned reason if that ever grows.
fn interned(text: String) -> &'static str {
    static SEEN: std::sync::Mutex<std::collections::BTreeSet<&'static str>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&s) = seen.get(text.as_str()) {
        return s;
    }
    let s: &'static str = Box::leak(text.into_boxed_str());
    seen.insert(s);
    s
}

/// What a dword of a descriptor that is no load's is instead (see `Tracer::changed`).
#[derive(Clone, Copy)]
enum Change {
    Constant,
    OtherLoad,
    Masked,
    Offset,
    Scaled,
    Selected,
    Input,
    Computed,
}

/// "a descriptor changed after loading", then which dword and how.
fn changed_why(dword: usize, kind: Change) -> &'static str {
    macro_rules! how {
        ($d:literal) => {
            match kind {
                Change::Constant => concat!("a descriptor changed after loading: dword ", $d,
                    " is a constant"),
                Change::OtherLoad => concat!("a descriptor changed after loading: dword ", $d,
                    " comes from another load or place in it"),
                Change::Masked => concat!("a descriptor changed after loading: dword ", $d,
                    " masked or merged (s_and, s_or, s_bfi...)"),
                Change::Offset => concat!("a descriptor changed after loading: dword ", $d,
                    " added to (s_add, s_sub...)"),
                Change::Scaled => concat!("a descriptor changed after loading: dword ", $d,
                    " shifted, multiplied or extracted (s_lshl, s_mul, s_bfe...)"),
                Change::Selected => concat!("a descriptor changed after loading: dword ", $d,
                    " chosen by a select or a join"),
                Change::Input => concat!("a descriptor changed after loading: dword ", $d,
                    " is a program input"),
                Change::Computed => concat!("a descriptor changed after loading: dword ", $d,
                    " computed otherwise"),
            }
        };
    }
    match dword {
        0 => how!("0"),
        1 => how!("1"),
        2 => how!("2"),
        3 => how!("3"),
        4 => how!("4"),
        5 => how!("5"),
        6 => how!("6"),
        _ => how!("7"),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Moved {
    Const(u32),
    Value(Value),
}

/// Undeclared root parameters inferred from descriptor loads in a guessed graphics program
/// ([`crate::emit::graphics`]).
pub(crate) struct InferredRoots {
    /// Entry SGPRs used as 32-bit descriptor heap bases (`hi == 0`), as `ResourceHeap` or
    /// `SamplerHeap`.
    pub heaps: Vec<(u8, RootType)>,
    /// For a vertex shader: an even entry SGPR pair used as a 64-bit vertex buffer descriptor
    /// table (`s_load_dwordx4` feeding only indexed `buffer_load`/`tbuffer_load`).
    pub vertex_table: Option<u8>,
}

/// Infers undeclared SM 6.6 heap base SGPRs (`ResourceHeap` / `SamplerHeap`, 32-bit with `hi == 0`)
/// and (for vertex shaders) an undeclared 64-bit vertex buffer table SGPR pair from `p`'s
/// descriptor loads. Only called by [`crate::emit::graphics`] for guessed pipelines; genuinely
/// unknown SGPR addresses (non-descriptor loads or 64-bit pointers) are left uninferred.
pub(crate) fn infer_roots(
    p: &Program,
    layout: &[RootParam],
    stage: HwStage,
) -> Result<InferredRoots, ShaderError> {
    let t = Tracer::new(p, layout, stage, None)?;
    // Per descriptor load site: (used_as_resource, used_as_sampler, only_indexed_buffer).
    let mut loads: std::collections::BTreeMap<Site, (bool, bool, bool)> =
        std::collections::BTreeMap::new();
    let mut record = |inst: &Inst, op: &Operand, role: Role, indexed_buf: bool| {
        let Operand::Scalar { code, dwords } = *op else {
            return;
        };
        let words: Vec<Term> = (0..dwords)
            .filter_map(|k| inst.read(Reg::S(code + k)).map(|v| t.origin(v)))
            .collect();
        let mut sites = Vec::new();
        t.collect_descriptor_loads(&words, &mut Vec::new(), &mut sites);
        for site in sites {
            let entry = loads.entry(site).or_insert((false, false, true));
            match role {
                Role::Resource => entry.0 = true,
                Role::Sampler => entry.1 = true,
            }
            entry.2 &= indexed_buf;
        }
    };
    for block in &p.blocks {
        for inst in &block.insts {
            match &inst.fields {
                Fields::Image { rsrc, sampler, .. } if !inst.name().starts_with("image_bvh") => {
                    record(inst, rsrc, Role::Resource, false);
                    if let Some(s) = sampler {
                        record(inst, s, Role::Sampler, false);
                    }
                }
                Fields::Buffer { rsrc, idxen, .. } if !inst.name().ends_with("_inv") => {
                    record(inst, rsrc, Role::Resource, *idxen);
                }
                Fields::Smem { base, .. } if inst.name().starts_with("s_buffer_") => {
                    record(inst, base, Role::Resource, false);
                }
                _ => {}
            }
        }
    }
    // Per undeclared entry SGPR: (seen_as_resource_heap, seen_as_sampler_heap).
    let mut heap_roles: std::collections::BTreeMap<u8, (bool, bool)> =
        std::collections::BTreeMap::new();
    let mut vt_candidates: Vec<u8> = Vec::new();
    for (site, (res, smp, only_idx_buf)) in loads {
        let inst = t.inst(site);
        match &inst.fields {
            Fields::Smem {
                base: Operand::Scalar { code, .. },
                offset,
                soffset,
                ..
            } if inst.name().starts_with("s_load_dword") => {
                let (Some(lo), Some(hi)) = (
                    inst.read(Reg::S(*code)).map(|v| t.origin(v)),
                    inst.read(Reg::S(code + 1)).map(|v| t.origin(v)),
                ) else {
                    continue;
                };
                let soff = t.term(inst, soffset).unwrap_or(Term::Const(0));
                if let Some(n) = t.undeclared_entry_sgpr(lo)
                    && let base = p.inputs.iter().find(|i| i.0 == Reg::S(n)).map(|i| i.1)
                    && t.high_half_with(hi, None, &[lo, soff], base)
                {
                    let e = heap_roles.entry(n).or_default();
                    e.0 |= res;
                    e.1 |= smp;
                } else if stage == HwStage::Vs
                    && inst.name() == "s_load_dwordx4"
                    && res
                    && !smp
                    && only_idx_buf
                    && *offset >= 0
                    && *offset % 16 == 0
                    && matches!(t.term(inst, soffset), Some(Term::Const(0)))
                    && let (Some(n), Term::Value(hv)) = (t.undeclared_entry_sgpr(lo), hi)
                    && n % 2 == 0
                    && n < 105
                    && t.inputs[hv.0 as usize] == Some(Reg::S(n + 1))
                    && t.param_of(Reg::S(n + 1)).is_none()
                    && !vt_candidates.contains(&n)
                {
                    vt_candidates.push(n);
                }
            }
            Fields::Flat {
                addr: Some(Operand::Vector { reg, .. }),
                saddr,
                ..
            } if inst.name().starts_with("global_load_dword") => {
                let (pair, voff) = match saddr {
                    Some(Operand::Scalar { code, .. }) if *code != NULL => (
                        inst.read(Reg::S(*code))
                            .map(|v| t.origin(v))
                            .zip(inst.read(Reg::S(code + 1)).map(|v| t.origin(v))),
                        inst.read(Reg::V(*reg)).map_or(Term::Const(0), |v| t.vorigin(v)),
                    ),
                    _ => (
                        inst.read(Reg::V(*reg))
                            .map(|v| t.vorigin(v))
                            .zip(inst.read(Reg::V(reg + 1)).map(|v| t.vorigin(v))),
                        Term::Const(0),
                    ),
                };
                if let Some((lo, hi)) = pair
                    && let Some(n) = t.undeclared_entry_sgpr(lo)
                    && let base = p.inputs.iter().find(|i| i.0 == Reg::S(n)).map(|i| i.1)
                    && t.high_half_with(hi, None, &[lo, voff], base)
                {
                    let e = heap_roles.entry(n).or_default();
                    e.0 |= res;
                    e.1 |= smp;
                }
            }
            _ => {}
        }
    }
    let heaps: Vec<(u8, RootType)> = heap_roles
        .into_iter()
        .filter_map(|(n, (res, smp))| match (res, smp) {
            (true, false) => Some((n, RootType::ResourceHeap)),
            (false, true) => Some((n, RootType::SamplerHeap)),
            _ => None,
        })
        .collect();
    let vertex_table = match vt_candidates.as_slice() {
        &[n] if !heaps.iter().any(|&(h, _)| h == n || h == n + 1) => Some(n),
        _ => None,
    };
    Ok(InferredRoots {
        heaps,
        vertex_table,
    })
}

/// Whether shifting a guessed vertex layout explains an otherwise unbound address. The alternate
/// layout must agree with existing root-address uses and cannot move a live-in away from an
/// existing root unless the code identifies that pair as the vertex fetch table.
pub(crate) fn vertex_roots_match_rebased(
    p: &Program,
    absolute: &[RootParam],
    relative: &[RootParam],
) -> Result<bool, ShaderError> {
    let old = Tracer::new(p, absolute, HwStage::Vs, None)?;
    let new = Tracer::new(p, relative, HwStage::Vs, None)?;
    if p.inputs.iter().any(|&(reg, _)| old.param_of(reg).is_some()) {
        let vt = infer_roots(p, relative, HwStage::Vs)?.vertex_table;
        if p.inputs.iter().any(|&(reg, _)| {
            old.param_of(reg) != new.param_of(reg)
                && old.param_of(reg).is_some()
                && !vt.is_some_and(|n| reg == Reg::S(n) || reg == Reg::S(n + 1))
        }) {
            return Ok(false);
        }
    }
    let mut found = false;
    for (b, block) in p.blocks.iter().enumerate() {
        for inst in &block.insts {
            let address = |t: &Tracer<'_>| match &inst.fields {
                Fields::Smem { base, offset, soffset, .. } if inst.name().starts_with("s_load_") => {
                    t.smem_address(inst, base, *offset, soffset).map(Some)
                }
                Fields::Flat { addr, saddr, offset, .. } if inst.name().starts_with("global_") => {
                    t.flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset).map(Some)
                }
                Fields::Smem { base: rsrc, .. } | Fields::Buffer { rsrc, .. }
                    if inst.name().starts_with("s_buffer_")
                        || inst.name().starts_with("buffer_load_")
                        || inst.name().starts_with("tbuffer_load_") =>
                {
                    t.descriptor(inst, rsrc, b).map(|src| match src {
                        Source::Built { address, .. } => Some(address),
                        _ => None,
                    })
                }
                _ => Ok(None),
            };
            match (address(&old), address(&new)) {
                (Ok(Some(a)), after) if a.param < absolute.len() && after != Ok(Some(a)) => {
                    return Ok(false);
                }
                // The old layout leaves the pair undeclared, so its address
                // is title memory; the rebased one binds it to a table or
                // descriptor parameter.
                (Ok(Some(old_a)), Ok(Some(new_a)))
                    if old_a.param == MEMORY
                        && relative.get(new_a.param).is_some_and(|r|
                            matches!(r.root, RootType::Table | RootType::Descriptor)) =>
                {
                    found = true;
                }
                _ => {}
            }
        }
    }
    Ok(found)
}

impl Tracer<'_> {
    /// Collects `s_load_dword*` and `global_load_dword*` sites feeding descriptor `words`.
    fn collect_descriptor_loads(
        &self,
        words: &[Term],
        seen: &mut Vec<Value>,
        out: &mut Vec<Site>,
    ) {
        for &w in words {
            let Term::Value(v) = w else { continue };
            if seen.contains(&v) || seen.len() >= 256 {
                continue;
            }
            seen.push(v);
            if let Some((b, i)) = self.phis[v.0 as usize] {
                let args: Vec<Term> = self.p.blocks[b].phis[i]
                    .args
                    .iter()
                    .map(|&a| self.origin(a))
                    .collect();
                self.collect_descriptor_loads(&args, seen, out);
            } else if let Some((_, a, b)) = self.picked(v) {
                self.collect_descriptor_loads(&[a, b], seen, out);
            } else if let Some(lane_v) = self.first_lane_source(v) {
                let lt = self.lane_term(Term::Value(lane_v));
                self.collect_lane_descriptor_loads(lt, seen, out);
            } else if let Some(site) = self.defs[v.0 as usize] {
                let name = self.inst(site).name();
                if (name.starts_with("s_load_dword") || name.starts_with("global_load_dword"))
                    && !out.contains(&site)
                {
                    out.push(site);
                }
            }
        }
    }

    /// Follows per-lane descriptor values (VGPRs from `global_load_dword*`, `v_cndmask_b32`, phis).
    fn collect_lane_descriptor_loads(
        &self,
        t: Term,
        seen: &mut Vec<Value>,
        out: &mut Vec<Site>,
    ) {
        let Term::Value(v) = self.lane_term(t) else {
            return;
        };
        if seen.contains(&v) || seen.len() >= 256 {
            return;
        }
        seen.push(v);
        if let Some((b, i)) = self.phis[v.0 as usize] {
            for &a in &self.p.blocks[b].phis[i].args {
                self.collect_lane_descriptor_loads(Term::Value(a), seen, out);
            }
        } else if let Some((_, a, b)) = self.lane_picked(v) {
            self.collect_lane_descriptor_loads(a, seen, out);
            self.collect_lane_descriptor_loads(b, seen, out);
        } else if let Some(site) = self.defs[v.0 as usize] {
            let name = self.inst(site).name();
            if (name.starts_with("s_load_dword") || name.starts_with("global_load_dword"))
                && !out.contains(&site)
            {
                out.push(site);
            }
        }
    }

    /// If `t` is an additive expression containing exactly one undeclared entry SGPR (`< 106`,
    /// not in `layout`) and no declared address base or PC, returns that SGPR's code.
    fn undeclared_entry_sgpr(&self, t: Term) -> Option<u8> {
        let mut found: Option<u8> = None;
        let mut stack = vec![t];
        let mut seen = Vec::new();
        while let Some(cur) = stack.pop() {
            let Term::Value(v) = cur else { continue };
            if seen.contains(&v) || seen.len() >= 64 {
                continue;
            }
            seen.push(v);
            if self.inputs[v.0 as usize].is_some() || self.spilled.contains_key(&v) {
                match self.param_value(v) {
                    Some((p, 0))
                        if matches!(
                            self.layout[p].root,
                            RootType::Table
                                | RootType::Descriptor
                                | RootType::ResourceHeap
                                | RootType::SamplerHeap
                        ) =>
                    {
                        return None;
                    }
                    Some(_) => continue,
                    None => {
                        let Some(Reg::S(n)) = self.inputs[v.0 as usize] else {
                            continue;
                        };
                        if n >= 106 || found.is_some() {
                            return None;
                        }
                        found = Some(n);
                    }
                }
                continue;
            }
            if self.getpc(v).is_some() {
                return None;
            }
            if let Some((_, [a, b])) = self.sop2(v, &["s_add_u32", "s_add_i32"]) {
                stack.extend([a, b]);
            } else if let Some(unshifted) = self
                .sop2(
                    v,
                    &[
                        "s_lshl1_add_u32",
                        "s_lshl2_add_u32",
                        "s_lshl3_add_u32",
                        "s_lshl4_add_u32",
                    ],
                )
                .map(|(_, [_, u])| u)
                .or_else(|| self.vlshl_add(v).map(|(_, _, u)| u))
                .or_else(|| self.vmad64_addend(v, 0))
            {
                stack.push(unshifted);
            } else if let Some([a, b]) = self.vadd(v) {
                stack.extend([a, b]);
            } else if let Some(lane) = self.first_lane_source(v) {
                stack.push(self.vorigin(lane));
            } else if let Some(joined) = self.joined_on_taken_edges(cur)
                && joined != cur
            {
                stack.push(joined);
            }
        }
        found
    }

    /// `v_readfirstlane_b32` values in `block` that compute dynamic descriptor offset `d` through
    /// pure scalar ALU operations and values from outside `block`.
    fn first_lane_in_offset(&self, block: usize, d: Value) -> Option<Vec<Value>> {
        let mut reads = Vec::new();
        let mut stack = vec![self.origin(d)];
        let mut seen = Vec::new();
        while let Some(cur) = stack.pop() {
            let Term::Value(v) = cur else { continue };
            if seen.contains(&v) {
                continue;
            }
            if seen.len() >= 64 {
                return None;
            }
            seen.push(v);
            if self.first_lane_source(v).is_some() {
                if !reads.contains(&v) {
                    reads.push(v);
                }
                continue;
            }
            if self.phis[v.0 as usize].is_some_and(|(pb, _)| pb == block) {
                return None;
            }
            let Some((def_b, def_i)) = self.defs[v.0 as usize] else {
                continue;
            };
            if def_b != block {
                continue;
            }
            let inst = self.inst((def_b, def_i));
            if let Some((_, ops)) = self.sop2(
                v,
                &[
                    "s_lshl_b32",
                    "s_lshr_b32",
                    "s_ashr_i32",
                    "s_and_b32",
                    "s_or_b32",
                    "s_xor_b32",
                    "s_bfe_u32",
                    "s_bfe_i32",
                    "s_mul_i32",
                    "s_add_u32",
                    "s_add_i32",
                    "s_sub_u32",
                    "s_sub_i32",
                    "s_lshl1_add_u32",
                    "s_lshl2_add_u32",
                    "s_lshl3_add_u32",
                    "s_lshl4_add_u32",
                ],
            ) {
                stack.extend(ops);
            } else if inst.name() == "s_lshl_b64"
                && inst.writes.first().is_some_and(|w| w.1 == v)
                && let Fields::Sop { src, .. } = &inst.fields
                && let (Some(&Operand::Scalar { code, .. }), Some(shift)) =
                    (src.first(), src.get(1).and_then(|s| self.term(inst, s)))
                && code != NULL
            {
                let lo = self.origin(inst.read(Reg::S(code))?);
                let hi = self.origin(inst.read(Reg::S(code + 1))?);
                stack.extend([lo, hi, shift]);
            } else {
                return None;
            }
        }
        (!reads.is_empty()).then_some(reads)
    }

    /// Bitwise identity SOP2 operations (`x & 0xffffffff`, `x | 0`, `x ^ 0`) on dword `k`.
    fn identity_op(&self, inst: &Inst, k: usize, o: &[Option<Term>]) -> Option<Term> {
        let (pair, neutral) = match inst.name() {
            "s_and_b32" => (false, u32::MAX),
            "s_and_b64" => (true, u32::MAX),
            "s_or_b32" | "s_xor_b32" => (false, 0),
            "s_or_b64" | "s_xor_b64" => (true, 0),
            _ => return None,
        };
        if k > usize::from(pair) {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        let half = |op: &Operand| -> Option<Term> {
            match *op {
                Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
                Operand::Scalar { code, .. } => o[inst.read(Reg::S(code + k as u8))?.0 as usize],
                Operand::Const(c) if k == 0 => inline(c).map(Term::Const),
                Operand::Const(c @ 128..=208) => {
                    inline(c).map(|lo| Term::Const(((lo as i32) >> 31) as u32))
                }
                Operand::Literal(x) if k == 0 => Some(Term::Const(x)),
                Operand::Literal(x) if x >> 31 == 0 => Some(Term::Const(0)),
                _ => None,
            }
        };
        match (half(src.first()?)?, half(src.get(1)?)?) {
            (x, Term::Const(c)) | (Term::Const(c), x) if c == neutral => Some(x),
            _ => None,
        }
    }

    /// For `v_readlane_b32` of a constant lane: the SGPR spilled to that
    /// lane, found by following the VGPR's `v_writelane_b32` writes.
    /// What dword `k` of `s_xor_b32/b64 d, x, y` leaves when `x` is `a ^ b` and `y` holds what `b`
    /// (or `a`) does: the other one. The title's compiler swaps descriptor dwords between SGPRs in
    /// three XORs.
    fn xor_undone(&self, inst: &Inst, k: usize, o: &[Option<Term>]) -> Option<Term> {
        let xor = |inst: &Inst, k: usize| -> Option<[Value; 2]> {
            let pair = match inst.name() {
                "s_xor_b32" => false,
                "s_xor_b64" => true,
                _ => return None,
            };
            let Fields::Sop { src, .. } = &inst.fields else { return None };
            let [Operand::Scalar { code: a, .. }, Operand::Scalar { code: b, .. }] = src[..] else {
                return None;
            };
            if k > usize::from(pair) || a == NULL || b == NULL {
                return None;
            }
            Some([inst.read(Reg::S(a + k as u8))?, inst.read(Reg::S(b + k as u8))?])
        };
        let [x, y] = xor(inst, k)?;
        for (x, y) in [(x, y), (y, x)] {
            let Some(def) = self.def(x) else { continue };
            let Some(j) = def.writes.iter().position(|w| w.1 == x) else { continue };
            let Some([a, b]) = xor(def, j) else { continue };
            let (a, b, y) = (o[a.0 as usize], o[b.0 as usize], o[y.0 as usize]);
            if y.is_some() && y == b {
                return a;
            }
            if y.is_some() && y == a {
                return b;
            }
        }
        None
    }

    fn restored(&self, inst: &Inst) -> Option<Moved> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readlane_b32" {
            return None;
        }
        let [Operand::Vector { reg, .. }, lane] = v.src[..] else {
            return None;
        };
        self.spilled(inst.read(Reg::V(reg))?, lane_of(lane)?, &mut Vec::new())
            .ok()
            .flatten()
    }

    /// What lane `lane` of VGPR value `v` holds: `Ok(None)` for a phi in
    /// `seen` (already being resolved), `Err` if not a spilled SGPR.
    fn spilled(&self, v: Value, lane: u32, seen: &mut Vec<Value>) -> Result<Option<Moved>, ()> {
        if let Some((b, i)) = self.phis[v.0 as usize] {
            if seen.contains(&v) {
                return Ok(None);
            }
            seen.push(v);
            let mut found = None;
            for &a in &self.p.blocks[b].phis[i].args {
                if let Some(m) = self.spilled(a, lane, seen)? {
                    if found.is_some_and(|f| f != m) {
                        return Err(());
                    }
                    found = Some(m);
                }
            }
            return Ok(found);
        }
        let inst = self.def(v).ok_or(())?;
        let Fields::Valu(w) = &inst.fields else {
            return Err(());
        };
        let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
            return Err(());
        };
        if inst.name() != "v_writelane_b32" {
            return Err(());
        }
        if lane_of(*at).ok_or(())? != lane {
            return self.spilled(inst.read(Reg::V(reg)).ok_or(())?, lane, seen);
        }
        match *value {
            Operand::Scalar { code, .. } if code != NULL => {
                inst.read(Reg::S(code)).map(Moved::Value)
            }
            Operand::Literal(x) => Some(Moved::Const(x)),
            Operand::Const(c) => inline(c).map(Moved::Const),
            _ => None,
        }
        .map(Some)
        .ok_or(())
    }
}

/// A lane select given as a constant.
pub(crate) fn lane_of(op: Operand) -> Option<u32> {
    match op {
        Operand::Const(c) => inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

/// What dword `k` of a move's result copies.
fn copy_source(inst: &Inst, k: usize) -> Option<Moved> {
    let Fields::Sop { src, imm, .. } = &inst.fields else {
        return None;
    };
    if inst.name() == "s_movk_i32" {
        return Some(Moved::Const(i32::from((*imm)? as i16) as u32));
    }
    // A mask of S0 bits from bit S1, as the upscaler's compiler makes an
    // immediate sampler's word.
    if let ("s_bfm_b32", 0, [Operand::Const(w), Operand::Const(at)]) = (inst.name(), k, &src[..]) {
        let (w, at) = (inline(*w)? & 31, inline(*at)? & 31);
        return Some(Moved::Const(((1u32 << w) - 1) << at));
    }
    // The same in 64 bits (a sampler's first two words, from the title's compiler).
    if let ("s_bfm_b64", [Operand::Const(w), Operand::Const(at)]) = (inst.name(), &src[..]) {
        let (w, at) = (inline(*w)? & 63, inline(*at)? & 63);
        return Some(Moved::Const((((1u64 << w) - 1) << at >> (32 * k)) as u32));
    }
    let src = src.first()?;
    match (inst.name(), *src) {
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { code, .. }) if code != NULL => {
            inst.read(Reg::S(code + k as u8)).map(Moved::Value)
        }
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { .. }) => Some(Moved::Const(0)),
        ("s_mov_b32", Operand::Literal(x)) => Some(Moved::Const(x)),
        ("s_mov_b32", Operand::Const(c)) => inline(c).map(Moved::Const),
        // shortcut: only the one-operand ways the compiler makes a constant
        // that has no inline form; fold more if descriptors need it.
        ("s_brev_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(x.reverse_bits())),
        ("s_not_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(!x)),
        // shortcut: a literal below 2^31, whose high dword is 0 whether the
        // hardware extends it with zeros or with its sign.
        ("s_mov_b64", Operand::Literal(x)) if x >> 31 == 0 => {
            Some(Moved::Const(if k == 0 { x } else { 0 }))
        }
        // 64-bit inline integers are sign-extended.
        ("s_mov_b64", Operand::Const(c @ 128..=208)) => {
            let low = inline(c)?;
            Some(Moved::Const(if k == 0 {
                low
            } else {
                ((low as i32) >> 31) as u32
            }))
        }
        _ => None,
    }
}

/// The 32-bit value of an inline constant operand.
pub(crate) fn inline(code: u8) -> Option<u32> {
    let float = |f: f32| Some(f.to_bits());
    match code {
        128..=192 => Some(u32::from(code) - 128),
        193..=208 => Some((192 - i32::from(code)) as u32),
        240 => float(0.5),
        241 => float(-0.5),
        242 => float(1.0),
        243 => float(-1.0),
        244 => float(2.0),
        245 => float(-2.0),
        246 => float(4.0),
        247 => float(-4.0),
        248 => Some(0x3E22_F983),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::native::RootType::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn track_words(words: &[u32], layout: &[RootParam], stage: HwStage) -> Resources {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        track(&p, layout, stage, lanes(&p).as_ref()).unwrap()
    }

    /// `p` as one lane runs it, if it structures.
    fn lanes(p: &Program) -> Option<Structured> {
        crate::structure::structure(p).ok()
    }

    fn param(root: RootType, user_data: u8, sgprs: u32) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot: 0,
        }
    }

    fn at(param: usize, offset: u32) -> Address {
        Address {
            param,
            offset,
            dynamic: None,
        }
    }

    #[test]
    fn a_mask_that_preserves_a_heap_address_carry_keeps_the_table_offset() {
        let run = |hi: u32, mask: u32| {
            let words = [
                0x8004_A004, // s_add_u32 s4, s4, 32
                hi,
                0x8705_FF05, mask, // s_and_b32 s5, s5, mask
                0xF408_0202, 0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
                0xE070_0000, 0x8002_0000, // buffer_store_dword v0, off, s[8:11], 0
                S_ENDPGM,
            ];
            let code = words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>();
            let p = crate::ir::build(&code, false, 256).unwrap();
            track(&p, &[param(Table, 4, 1)], HwStage::Cs, lanes(&p).as_ref())
        };
        let carried = 0x8205_8080; // s_addc_u32 s5, 0, 0
        assert_eq!(run(carried, 0xff).unwrap().scalar, [((0, 3), at(0, 32))]);
        // Clearing the carry changes the address.
        assert!(run(carried, 0xfe).is_err());
        // A masked arbitrary input can still name memory outside the heap.
        assert!(matches!(run(0xBE85_0300, 0xff), // s_mov_b32 s5, s0
            Err(ShaderError::Unsupported {
                what: "an address into a descriptor heap with an untracked high half", ..
            })
        ));
    }

    #[test]
    fn table_descriptors_are_loaded_from_their_range() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE89_0380, // s_mov_b32 s9, 0
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF408_0204,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
                0xF40C_0001,
                0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
                0xF080_0F08,
                0x0040_0002, // image_sample v[0:3], v[2:3], s[0:7], s[8:11]
                S_ENDPGM,
            ],
            &[param(Table, 8, 1), param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0)), ((0, 4), at(1, 0))]);
        let roles: Vec<_> = r.descriptors.iter().map(|d| (d.role, &d.source)).collect();
        assert_eq!(
            roles,
            [
                (
                    Role::Resource,
                    &Source::Loaded {
                        load: (0, 4),
                        first: 0
                    }
                ),
                (
                    Role::Sampler,
                    &Source::Loaded {
                        load: (0, 3),
                        first: 0
                    }
                ),
            ]
        );
    }

    #[test]
    fn root_descriptors_are_built_in_code() {
        let r = track_words(
            &[
                0x8801_FF0B,
                0x0010_0000, // s_or_b32 s1, s11, 0x100000
                0xBE80_030A, // s_mov_b32 s0, s10
                0xBE82_03C1, // s_mov_b32 s2, -1
                0xBE83_03FF,
                0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
                0xF430_0000,
                0xFA00_0000, // s_buffer_load_dwordx16 s[0:15], s[0:3], null
                S_ENDPGM,
            ],
            &[param(Descriptor, 10, 2)],
            HwStage::Vs,
        );
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(0, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC]
            }
        );
        assert_eq!(
            r.params.iter().map(|p| (p.1, p.2)).collect::<Vec<_>>(),
            [(0, 0), (0, 1)]
        );
    }

    #[test]
    fn heap_indices_from_root_constants_in_a_merged_stage() {
        // User data starts at s8: the constant (index 6) is s14, the heap
        // (index 10) s18.
        let r = track_words(
            &[
                0x8F08_850E, // s_lshl_b32 s8, s14, 5
                0xBE93_0380, // s_mov_b32 s19, 0
                0xF408_0209,
                0x1000_0000, // s_load_dwordx4 s[8:11], s[18:19], s8
                0xE000_2000,
                0x8002_0400, // buffer_load_format_x v4, v0, s[8:11], 0 idxen
                S_ENDPGM,
            ],
            &[param(Constants, 6, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ms,
        );
        let index = r.params.iter().find(|p| p.1 == 0).unwrap().0;
        let shift = r.scalar[0].1.dynamic.unwrap();
        assert_eq!(r.scalar[0].1.param, 1);
        // The computed term is the shifted constant, not the constant itself.
        assert_ne!(shift, index);
        assert_eq!(
            r.descriptors[0].source,
            Source::Loaded {
                load: (0, 2),
                first: 0
            }
        );
    }

    #[test]
    fn spilled_sgprs_are_followed_through_vgpr_lanes() {
        let r = track_words(
            &[
                0xD761_0000,
                0x0001_0609, // v_writelane_b32 v0, s9, 3
                0xD760_0002,
                0x0001_0700, // v_readlane_b32 s2, v0, 3
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF40C_0001,
                0xFA00_0020, // s_load_dwordx8 s[0:7], s[2:3], 0x20
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0x20))]);
    }

    #[test]
    fn descriptors_joined_by_control_flow_are_traced_per_edge() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE83_0380, // s_mov_b32 s3, 0
                0xBF06_8001, // s_cmp_eq_u32 s1, 0
                0xBF85_0003, // s_cbranch_scc1 -> 0x1c
                0xF408_0001,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[2:3], null
                0xBF82_0002, // s_branch -> 0x24
                0xF408_0001,
                0xFA00_0010, // 0x1c: s_load_dwordx4 s[0:3], s[2:3], 0x10
                0xF420_0100,
                0xFA00_0000, // 0x24: s_buffer_load_dword s4, s[0:3], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Phi { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        let offsets: Vec<_> = args
            .iter()
            .map(|a| match a {
                Source::Loaded { load, .. } => {
                    r.scalar.iter().find(|s| s.0 == *load).unwrap().1.offset
                }
                a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets.len(), 2);
        assert!(
            offsets.contains(&0) && offsets.contains(&0x10),
            "{offsets:?}"
        );
    }

    #[test]
    fn descriptors_selected_by_scc_are_traced_per_outcome() {
        // Two buffer descriptors loaded from the table at s9 and one
        // picked with s_cselect_b64 per half, as the compiler makes a
        // conditional choice of resources without a branch.
        let r = track_words(
            &[
                0xBE8C_0309, // s_mov_b32 s12, s9
                0xBE8D_0380, // s_mov_b32 s13, 0
                0xF408_0006,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[12:13], null
                0xF408_0106,
                0xFA00_0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
                0xBF06_800E, // s_cmp_eq_u32 s14, 0
                0x8588_0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
                0x858A_0602, // s_cselect_b64 s[10:11], s[2:3], s[6:7]
                0xF420_0404,
                0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Select { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        // SCC set picks the first operand.
        assert_eq!(
            args[..],
            [
                Source::Loaded {
                    load: (0, 2),
                    first: 0
                },
                Source::Loaded {
                    load: (0, 3),
                    first: 0
                },
            ]
        );
        assert_eq!(r.scalar, [((0, 2), at(0, 0)), ((0, 3), at(0, 0x10))]);
    }

    /// A loaded buffer descriptor changed before use is refused naming the first dword that is
    /// no longer the load's and how it was made: here records replaced by a constant, then the
    /// base address offset.
    #[test]
    fn a_descriptor_changed_after_loading_says_which_dword_and_how() {
        let run = |change: &[u32]| {
            let mut words = vec![
                0xBE88_0304, // s_mov_b32 s8, s4
                0xBE89_0380, // s_mov_b32 s9, 0
                0xF408_0204,
                0xFA00_0160, // s_load_dwordx4 s[8:11], s[8:9], 0x160
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            ];
            words.extend_from_slice(change);
            words.extend([
                0xE078_1000,
                0x8002_0007, // buffer_store_dwordx4 v[0:3], v7, s[8:11], 0 offen
                S_ENDPGM,
            ]);
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = crate::ir::build(&code, false, 256).unwrap();
            match track(&p, &[param(Table, 4, 1)], HwStage::Cs, lanes(&p).as_ref()) {
                Err(ShaderError::Unsupported { what, .. }) => what,
                r => panic!("{:?}", r.err()),
            }
        };
        assert_eq!(
            run(&[0xBE8A_03C0]), // s_mov_b32 s10, 64
            "a descriptor changed after loading: dword 2 is a constant"
        );
        assert_eq!(
            run(&[0x8008_9008, 0x8209_8009]), // s_add_u32 s8, s8, 16; s_addc_u32 s9, s9, 0
            "a descriptor changed after loading: dword 0 added to (s_add, s_sub...)"
        );
        // A mask or merge names the operation and its constant operands, if any.
        assert_eq!(
            run(&[0x8709_FF09, 0x3FFF_FFFF]), // s_and_b32 s9, s9, 0x3fffffff
            "a descriptor changed after loading: dword 1 masked or merged (s_and_b32 with \
             0x3fffffff)"
        );
        assert_eq!(
            run(&[0x8809_0409]), // s_or_b32 s9, s9, s4 (a program input)
            "a descriptor changed after loading: dword 1 masked or merged (s_or_b32 with \
             computed values)"
        );
    }

    /// The FidelityFX upscaler's byte accesses make a loaded buffer
    /// descriptor raw first (stride cleared, records times stride, bounds
    /// in bytes): the access reads the loaded descriptor. Any other change
    /// still fails.
    #[test]
    fn a_descriptor_made_raw_is_the_loaded_one() {
        let words = |mask: u32| -> Vec<u8> {
            [
                0xBE88_0304, // s_mov_b32 s8, s4
                0xBE89_0380, // s_mov_b32 s9, 0
                0xF408_0204,
                0xFA00_0160, // s_load_dwordx4 s[8:11], s[8:9], 0x160
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                0x938C_FF09,
                0x000E_0010, // s_bfe_u32 s12, s9, 0x000e0010
                0x848C_810C, // s_max_u32 s12, s12, 1
                0x8709_FF09,
                mask, // s_and_b32 s9, s9, mask
                0x870B_FF0B,
                0xCFFF_FFFF, // s_and_b32 s11, s11, 0xcfffffff
                0x930A_0C0A, // s_mul_i32 s10, s10, s12
                0x880B_FF0B,
                0x3000_0000, // s_or_b32 s11, s11, 0x30000000
                0xE078_1000,
                0x8002_0007, // buffer_store_dwordx4 v[0:3], v7, s[8:11], 0 offen
                S_ENDPGM,
            ]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect()
        };
        let run = |mask| {
            let p = crate::ir::build(&words(mask), false, 256).unwrap();
            track(&p, &[param(Table, 4, 1)], HwStage::Cs, lanes(&p).as_ref())
        };
        assert!(run(0xC000_FFFF).is_ok());
        assert_eq!(
            run(0xC000_FFFE).unwrap_err(),
            ShaderError::Unsupported {
                offset: 0x3c,
                what: "a descriptor changed after loading: dword 1 masked or merged (s_and_b32 \
                       with 0xc000fffe)"
            }
        );
    }

    /// The title's compiler swaps descriptor dwords between SGPRs in three XORs (32 and 64 bits):
    /// swapped there and back, the descriptor is the loaded one.
    #[test]
    fn descriptors_swapped_by_xors_are_the_loaded_ones() {
        let swap32 = [0x8909_0D09, 0x890D_0D09, 0x8909_0D09]; // s9 <-> s13
        let swap64 = [0x898A_0E0A, 0x898E_0E0A, 0x898A_0E0A]; // s[10:11] <-> s[14:15]
        let words = |swaps: &[u32]| -> Vec<u8> {
            [
                &[
                    0xBE88_0304, // s_mov_b32 s8, s4
                    0xBE89_0380, // s_mov_b32 s9, 0
                    0xF408_0204,
                    0xFA00_0160, // s_load_dwordx4 s[8:11], s[8:9], 0x160
                    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                    0xBE8D_0380, // s_mov_b32 s13, 0
                    0xBE8E_0480, // s_mov_b64 s[14:15], 0
                ][..],
                swaps,
                &[
                    0xE078_1000,
                    0x8002_0007, // buffer_store_dwordx4 v[0:3], v7, s[8:11], 0 offen
                    S_ENDPGM,
                ],
            ]
            .concat()
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect()
        };
        let run = |swaps: &[u32]| {
            let p = crate::ir::build(&words(swaps), false, 256).unwrap();
            track(&p, &[param(Table, 4, 1)], HwStage::Cs, lanes(&p).as_ref())
        };
        let there_and_back = [swap32, swap32, swap64, swap64].concat();
        run(&there_and_back).unwrap();
        // Swapped once, its second dword is the zero.
        assert!(run(&swap32).is_err());
    }

    /// Immediate sampler words the compilers make with bit masks (32 and
    /// 64 bits) and a 64-bit move of a literal.
    #[test]
    fn sampler_words_from_bit_masks_and_wide_literals_are_constants() {
        let code: Vec<u8> = [
            0xB010_3092, // s_movk_i32 s16, 0x3092
            0x9211_8C8C, // s_bfm_b32 s17, 12, 12
            0xBE92_04FF,
            0x1850_0000, // s_mov_b64 s[18:19], 0x18500000
            0x9294_B682, // s_bfm_b64 s[20:21], 2, 54
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w: &u32| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let layout = [param(Table, 4, 1)];
        let t = Tracer::new(&p, &layout, HwStage::Cs, None).unwrap();
        let block = &p.blocks[0];
        let value = |n: usize, k: usize| t.origin(block.insts[n].writes[k].1);
        assert_eq!(value(1, 0), Term::Const(0x00FF_F000));
        assert_eq!(value(2, 0), Term::Const(0x1850_0000));
        assert_eq!(value(2, 1), Term::Const(0));
        // The title's samplers: bits 54 and 55, "a sampler built from computed words" before.
        assert_eq!(value(3, 0), Term::Const(0));
        assert_eq!(value(3, 1), Term::Const(0x00C0_0000));
    }

    /// A register kept in a second one across an inner loop that restores
    /// it, inside a loop whose entry value is itself a loop's phi: s2 at
    /// 0x08 and at 0x0c is s2 at 0x04 whichever way it is reached. The
    /// origins must settle, and name no value the register does not hold.
    #[test]
    fn copies_restored_in_nested_loops_settle() {
        let words = [
            0xBE87_0380, // 0x00 s_mov_b32 s7, 0
            0xBF06_8006, // 0x04 s_cmp_eq_u32 s6, 0
            0xBE83_0302, // 0x08 s_mov_b32 s3, s2
            0xBF06_8004, // 0x0c s_cmp_eq_u32 s4, 0
            0xBF85_0002, // 0x10 s_cbranch_scc1 -> 0x1c
            0xBE82_0303, // 0x14 s_mov_b32 s2, s3
            0xBF82_FFFC, // 0x18 s_branch -> 0x0c
            0xBF06_8005, // 0x1c s_cmp_eq_u32 s5, 0
            0xBF85_FFF9, // 0x20 s_cbranch_scc1 -> 0x08
            0x8002_8102, // 0x24 s_add_u32 s2, s2, 1
            0xBF06_8006, // 0x28 s_cmp_eq_u32 s6, 0
            0xBF85_FFF5, // 0x2c s_cbranch_scc1 -> 0x04
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let t = Tracer::new(&p, &[], HwStage::Ps, None).unwrap();
        let block = |offset| &p.blocks[block_at(&words, offset)];
        let phi = |offset| {
            let phis = &block(offset).phis;
            phis.iter().find(|phi| phi.reg == Reg::S(2)).unwrap().value
        };
        let write = |offset: usize, name| {
            let inst = block(offset).insts.iter().find(|i| i.name() == name);
            inst.unwrap().writes[0].1
        };
        let (outer, middle, inner) = (phi(0x04), phi(0x08), phi(0x0c));
        let sum = write(0x24, "s_add_u32");
        // The outer phi joins the entry value and the sum, which is no copy.
        assert_eq!(t.origin(sum), Term::Value(sum));
        assert_eq!(t.origin(outer), Term::Value(outer));
        // Both inner phis hold the outer one's value; a phi may stand for itself.
        let held = [Term::Value(outer), Term::Value(middle)];
        assert!(held.contains(&t.origin(middle)));
        assert!(held.contains(&t.origin(inner)) || t.origin(inner) == Term::Value(inner));
        // The moves are copies of the middle phi.
        assert_eq!(t.origin(write(0x08, "s_mov_b32")), t.origin(middle));
        assert_eq!(t.origin(write(0x14, "s_mov_b32")), t.origin(middle));
    }

    #[test]
    fn a_selection_that_mixes_two_descriptors_fails() {
        // The high half picked from the first load whatever SCC is: when
        // clear, the descriptor is half of one load and half of the other.
        let code: Vec<u8> = [
            0xBE8C_0309, // s_mov_b32 s12, s9
            0xBE8D_0380, // s_mov_b32 s13, 0
            0xF408_0006,
            0xFA00_0000, // s_load_dwordx4 s[0:3], s[12:13], null
            0xF408_0106,
            0xFA00_0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
            0xBF06_800E, // s_cmp_eq_u32 s14, 0
            0x8588_0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
            0x858A_0202, // s_cselect_b64 s[10:11], s[2:3], s[2:3]
            0xF420_0404,
            0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w: &u32| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps, lanes(&p).as_ref()).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 36,
                what: "a descriptor changed after loading: dword 2 comes from another load or \
                       place in it"
            }
        );
    }

    /// A compute program whose root parameters do not fit its 16 user
    /// SGPRs loads them through the pointer in user data 0-1: the loads are
    /// user data (dword 16 at 0x40), their values the parameters'.
    #[test]
    fn root_parameters_past_the_user_sgprs_load_from_the_pointer() {
        let r = track_words(
            &[
                0xBE82_0300, // s_mov_b32 s2, s0
                0x8703_FF01,
                0x0000_00FF, // s_and_b32 s3, s1, 0xff
                0xF404_0101,
                0xFA00_0008, // s_load_dwordx2 s[4:5], s[2:3], 0x8 (user data 18)
                0xBE86_03C1, // s_mov_b32 s6, -1
                0xBE87_03FF,
                0x1101_4FAC, // s_mov_b32 s7, 0x11014fac
                0xF420_0202,
                0xFA00_0000, // s_buffer_load_dword s8, s[4:7], null
                S_ENDPGM,
            ],
            &[param(Constants, 16, 2), param(Descriptor, 18, 2)],
            HwStage::Cs,
        );
        assert_eq!(r.scalar[0].1, at(USER_DATA, 0x48));
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(1, 0),
                bits: 0,
                words: [u32::MAX, 0x1101_4FAC]
            }
        );
        let mut params: Vec<_> = r.params.iter().map(|p| (p.1, p.2)).collect();
        params.sort_unstable();
        assert_eq!(params, [(1, 0), (1, 1)]);
    }

    /// Pointers the program loads or takes from root constants address
    /// title memory; an SGPR no root parameter fills stays refused.
    #[test]
    fn computed_addresses_are_title_memory() {
        let r = track_words(
            &[
                0xF404_0105,
                0xFA00_0000, // s_load_dwordx2 s[4:5], s[10:11], null (constants)
                0xF400_0182,
                0xFA00_0008, // s_load_dword s6, s[4:5], 0x8 (the loaded pointer)
                S_ENDPGM,
            ],
            &[param(Constants, 10, 2)],
            HwStage::Cs,
        );
        assert_eq!(r.scalar, [((0, 0), at(MEMORY, 0)), ((0, 1), at(MEMORY, 0))]);
    }

    /// A live-in SGPR pair the layout does not declare holds a title-memory
    /// address (guessed: the console's user data carries raw addresses), so
    /// the load tracks to [`MEMORY`] and lowers through the memory space.
    #[test]
    fn an_address_in_an_sgpr_no_root_parameter_fills_is_title_memory() {
        let code: Vec<u8> = [0xF408_0001, 0xFA00_0000, S_ENDPGM] // s_load_dwordx4 s[0:3], s[2:3]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let r = track(&p, &[param(Table, 9, 1)], HwStage::Ps, lanes(&p).as_ref()).unwrap();
        assert_eq!(r.scalar, [((0, 0), at(MEMORY, 0))]);
        assert_eq!(r.memory_inputs, [2]);
    }

    /// A waterfall loop over an image descriptor each lane loads from the
    /// table at s0 (wave32, assembled with llvm-mc for gfx1030): lane i
    /// reads entry i with two `global_load_dwordx4`, the loop reads the
    /// first active lane's descriptor into s[12:19], samples with the lanes
    /// whose descriptor it is, and repeats for the others. The sampler comes
    /// from the table at s1; the result goes to LDS so the sample is needed.
    pub(crate) const WATERFALL_FETCHED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// [`WATERFALL_FETCHED_WAVE32`] in wave64: 64-bit lane masks.
    pub(crate) const WATERFALL_FETCHED_WAVE64: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0016,
        0x0002_080C, // v_cmp_eq_u64_e64 s[22:23], s[12:13], v[4:5]
        0xD4E2_0018,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s[24:25], s[14:15], v[6:7]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xD4E2_0018,
        0x0002_1010, // v_cmp_eq_u64_e64 s[24:25], s[16:17], v[8:9]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xD4E2_0018,
        0x0002_1412, // v_cmp_eq_u64_e64 s[24:25], s[18:19], v[10:11]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xBE96_2416, // s_and_saveexec_b64 s[22:23], s[22:23]
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x89FE_167E, // s_xor_b64 exec, exec, s[22:23]
        0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// The same waterfall over one of two image descriptors loaded from the
    /// table at s0, picked per lane by `v_cndmask_b32` (lanes below 16 take
    /// the first).
    pub(crate) const WATERFALL_PICKED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E08_0218, // v_mov_b32 v4, s24
        0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
        0x7E10_021C, // v_mov_b32 v8, s28
        0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
        0x7E12_021D, // v_mov_b32 v9, s29
        0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
        0x7E14_021E, // v_mov_b32 v10, s30
        0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
        0x7E16_021F, // v_mov_b32 v11, s31
        0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7E18_0504, // 0x7c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0x7c
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// The waterfall loop of [`WATERFALL_PICKED_WAVE32`] over one of two
    /// image descriptors loaded from the table at s0, chosen by control flow
    /// as the compiler lowers an if/else on a per-lane condition: lanes below
    /// 16 move the first into v[4:11], the others the second, each arm
    /// skipped when no lane takes it. The sample goes to the buffer at dword
    /// 16 of the table, at lane * 4.
    pub(crate) const WATERFALL_JOINED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0008, // s_cbranch_execz -> 0x70
        0x7E08_0218, // v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x7E10_021C, // v_mov_b32 v8, s28
        0x7E12_021D, // v_mov_b32 v9, s29
        0x7E14_021E, // v_mov_b32 v10, s30
        0x7E16_021F, // v_mov_b32 v11, s31
        0xBE94_3D14, // 0x70: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0008, // s_cbranch_execz -> 0x9c
        0x7E08_0220, // v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x7E10_0224, // v_mov_b32 v8, s36
        0x7E12_0225, // v_mov_b32 v9, s37
        0x7E14_0226, // v_mov_b32 v10, s38
        0x7E16_0227, // v_mov_b32 v11, s39
        0x887E_147E, // 0x9c: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0x7E18_0504, // 0xa4: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0xa4
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// The console compiler's waterfall (Grand Theft Auto VI's ray-tracing
    /// resolve and others): each lane holds its entry's byte offset in the
    /// table at s0 (v7 = (lane & 1) * 32); the loop reads the first active
    /// lane's offset, narrows EXEC to the lanes with the same one
    /// (`v_cmpx_eq_u32`), loads that entry with `s_load_dwordx8` and samples,
    /// then drops those lanes (`s_andn2_wrexec_b32`). The sample goes to the
    /// buffer at dword 16 of the table, at lane * 4.
    pub(crate) const WATERFALL_INDEXED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x360E_0081, // v_and_b32 v7, 1, v0
        0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBE95_037E, // s_mov_b32 s21, exec_lo
        0x7E28_0507, // 0x40: v_readfirstlane_b32 s20, v7
        0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
        0xF40C_0302,
        0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
        0xBF89_FFF6, // s_cbranch_execnz -> 0x40
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    #[test]
    fn the_console_waterfall_over_a_first_lane_offset_is_one() {
        let r = waterfall(WATERFALL_INDEXED_WAVE32, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| d.role == Role::Resource)
            .unwrap();
        let Source::Loaded { load, first: 0 } = image.source else {
            panic!("{:?}", image.source)
        };
        let (_, address) = r.scalar.iter().find(|s| s.0 == load).unwrap();
        assert!(address.dynamic.is_some(), "{address:?}");
        assert_eq!(r.waterfall.compares.len(), 1);
        assert_eq!(r.waterfall.reads.len(), 1);
        assert_eq!(r.waterfall.loads, [load]);
    }

    /// [`WATERFALL_JOINED_WAVE32`] with other sources on each path: lanes
    /// below 16 load table entry (lane & 1) with two `global_load_dwordx4`,
    /// the others pick per lane with `v_cndmask_b32` (lanes below 24 the
    /// first entry, the rest the second).
    pub(crate) const WATERFALL_JOINED_MIXED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x3602_0081, // v_and_b32 v1, 1, v0
        0x3402_0285, // v_lshlrev_b32 v1, 5, v1
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0004, // s_cbranch_execz -> 0x68
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xBE94_3D14, // 0x68: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0011, // s_cbranch_execz -> 0xb8
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0x7E08_0218, // v_mov_b32 v4, s24
        0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
        0x7E10_021C, // v_mov_b32 v8, s28
        0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
        0x7E12_021D, // v_mov_b32 v9, s29
        0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
        0x7E14_021E, // v_mov_b32 v10, s30
        0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
        0x7E16_021F, // v_mov_b32 v11, s31
        0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
        0x887E_147E, // 0xb8: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E18_0504, // 0xc4: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0xc4
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// A waterfall loop whose descriptor changes on each pass: v[4:11] holds
    /// the first table entry on entry and the second after every pass, so
    /// the first lane's dwords are phis of the loop header.
    pub(crate) const WATERFALL_LOOPED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7E08_0218, // v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x7E10_021C, // v_mov_b32 v8, s28
        0x7E12_021D, // v_mov_b32 v9, s29
        0x7E14_021E, // v_mov_b32 v10, s30
        0x7E16_021F, // v_mov_b32 v11, s31
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0x7E18_0504, // 0x64: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0x7E08_0220, // v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x7E10_0224, // v_mov_b32 v8, s36
        0x7E12_0225, // v_mov_b32 v9, s37
        0x7E14_0226, // v_mov_b32 v10, s38
        0x7E16_0227, // v_mov_b32 v11, s39
        0xBF89_FFDE, // s_cbranch_execnz -> 0x64
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// A waterfall loop over one of two buffer descriptors (four dwords)
    /// loaded from the table at s0, chosen by nested if/else: lanes below 16
    /// move the first into v[4:7], lanes from 16 to 23 the second, and the
    /// others keep what v[4:7] held. Only lanes below 24 (the same mask)
    /// run the loop, which loads dword `lane` of the chosen buffer, and
    /// store it to the buffer at dword 16 of the table.
    pub(crate) const WATERFALL_JOINED_BUFFER_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0602,
        0xFA00_0000, // s_load_dwordx4 s[24:27], s[4:5], 0x0
        0xF408_0802,
        0xFA00_0020, // s_load_dwordx4 s[32:35], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xD4C4_001E,
        0x0002_0098, // v_cmp_gt_u32_e64 s30, 24, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0004, // s_cbranch_execz -> 0x50
        0x7E08_0218, // 0x40: v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0xBE94_3D14, // 0x50: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0007, // s_cbranch_execz -> 0x78
        0xBE9F_3C1E, // s_and_saveexec_b32 s31, s30
        0xBF88_0004, // s_cbranch_execz -> 0x74
        0x7E08_0220, // 0x64: v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x887E_1F7E, // 0x74: s_or_b32 exec_lo, exec_lo, s31
        0x887E_147E, // 0x78: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_3C1E, // s_and_saveexec_b32 s23, s30
        0xBF88_0013, // s_cbranch_execz -> 0xd0
        0xBE9D_037E, // s_mov_b32 s29, exec_lo
        0x7E18_0504, // 0x88: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xE030_1000,
        0x8003_0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFF1, // s_cbranch_execnz -> 0x88
        0xBEFE_031D, // s_mov_b32 exec_lo, s29
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0x887E_177E, // 0xd0: s_or_b32 exec_lo, exec_lo, s23
        0xBF81_0000, // s_endpgm
    ];

    fn waterfall(words: &[u32], wave64: bool) -> Result<Resources, ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, wave64, 256).unwrap();
        let layout = [param(Table, 0, 1), param(Table, 1, 1)];
        track(&p, &layout, HwStage::Cs, lanes(&p).as_ref())
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_per_lane_loads_are_traced_to_them() {
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
        ] {
            let r = waterfall(words, wave64).unwrap();
            let image = r
                .descriptors
                .iter()
                .find(|d| d.role == Role::Resource)
                .unwrap();
            let Source::Fetched { load, first: 0 } = image.source else {
                panic!("{:?}", image.source)
            };
            // The first of the two loads: entry v0 * 32 of the table at s0.
            let a = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
            assert_eq!((a.param, a.offset), (0, 0));
            assert!(a.dynamic.is_some());
            // The sampler is the same for every lane.
            let sampler = r.descriptors.iter().find(|d| d.role == Role::Sampler);
            assert!(matches!(
                sampler.unwrap().source,
                Source::Loaded { first: 0, .. }
            ));
            // Four 64-bit compares of the first lane's dwords with the lane's
            // own; the eight reads become the lane's own dwords.
            assert_eq!(r.waterfall.compares.len(), 4);
            assert_eq!(r.waterfall.reads.len(), 8);
        }
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_a_per_lane_choice_are_traced_per_lane() {
        let r = waterfall(WATERFALL_PICKED_WAVE32, false).unwrap();
        let image = &r.descriptors[0];
        let Source::Lanes { args, .. } = &image.source else {
            panic!("{:?}", image.source)
        };
        // Lanes with their bit set take the first load (offset 0), the
        // others the second (0x20).
        let offsets: Vec<u32> = args
            .iter()
            .map(|a| match *a {
                Source::Loaded { load, first: 0 } => {
                    r.scalar.iter().find(|s| s.0 == load).unwrap().1.offset
                }
                ref a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets, [0, 0x20]);
        assert_eq!(r.waterfall.compares.len(), 4);
    }

    #[test]
    fn a_first_lane_read_outside_a_loop_is_no_waterfall() {
        // The waterfall without its back edge: the s_cbranch_execnz is an
        // s_nop, so lanes the first one does not match never get a pass.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let back = words.iter().position(|&w| w == 0xBF89_FFE6).unwrap();
        words[back] = 0xBF80_0000;
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane outside a waterfall loop",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// Why resource tracking refuses `words`.
    fn refusal(words: &[u32], wave64: bool) -> &'static str {
        match waterfall(words, wave64) {
            Err(ShaderError::Unsupported { what, .. }) => what,
            other => panic!("not refused: {other:?}"),
        }
    }

    const NO_LOOP: &str = "a descriptor read from the first lane outside a waterfall loop";
    const NOT_SELECTED: &str = "a waterfall loop whose compares do not select the access's lanes";
    const BAD_RETRY: &str = "a waterfall loop whose retry does not keep the lanes still waiting";
    const SPLIT: &str = "a waterfall loop nested in another or split over blocks";

    /// The index of `word` in `words`.
    fn word_at(words: &[u32], word: u32) -> usize {
        words.iter().position(|&w| w == word).unwrap()
    }

    #[test]
    fn a_first_lane_read_without_its_compare_is_no_waterfall() {
        // The loop without the compare of dwords 0 and 1: nothing makes
        // the first lane's dwords the lane's own there.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let cmp = word_at(&words, 0x0002_080C);
        words[cmp - 1..=cmp].copy_from_slice(&[0xBF80_0000, 0xBF80_0000]);
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn a_compare_that_does_not_mask_the_access_is_no_waterfall() {
        // The compares are all there, but their result never narrows EXEC:
        // every lane samples with the first lane's descriptor (reported by
        // ifBars on upstream PR #566).
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let save = word_at(&words, 0xBE95_3C15);
        words[save] = 0xBE95_037E; // s_mov_b32 s21, exec_lo
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn an_access_under_a_further_mask_is_no_waterfall() {
        // The access runs for the matching lanes that VCC also holds, yet
        // the retry removes every matching lane.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let nop = word_at(&words, 0xBE95_3C15) + 1;
        assert_eq!(words[nop], 0xBF80_0000);
        words[nop] = 0x877E_6A7E; // s_and_b32 exec_lo, exec_lo, vcc_lo
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn a_loop_that_drops_unmatched_lanes_is_no_waterfall() {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let retry = word_at(&words, 0x897E_157E);
        words[retry] = 0xBEFE_0380; // s_mov_b32 exec_lo, 0
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_retry_that_keeps_other_lanes_than_those_waiting_is_no_waterfall() {
        // The retry keeps the lanes of VCC that did not just sample, not
        // the waiting ones: lanes 16 and up never sample on the console.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let retry = word_at(&words, 0x897E_157E);
        words[retry] = 0x8A7E_7E6A; // s_andn2_b32 exec_lo, vcc_lo, exec_lo
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_wave64_retry_of_one_half_is_no_waterfall() {
        // Only the low half of EXEC loses the lanes just served.
        let mut words = WATERFALL_FETCHED_WAVE64.to_vec();
        let retry = word_at(&words, 0x89FE_167E);
        words[retry] = 0x897E_167E; // s_xor_b32 exec_lo, exec_lo, s22
        assert_eq!(refusal(&words, true), BAD_RETRY);
    }

    #[test]
    fn a_descriptor_used_after_exec_is_restored_is_no_waterfall() {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let sample = word_at(&words, 0xF09C_0F08);
        let restore = word_at(&words, 0xBEFE_0317);
        let image = [words[sample], words[sample + 1]];
        words[sample..sample + 2].fill(0xBF80_0000); // Keep the branch offsets.
        words.splice(restore + 1..restore + 1, image);
        assert_eq!(refusal(&words, false), NO_LOOP);
    }

    #[test]
    fn a_wave64_access_with_exec_hi_overwritten_is_no_waterfall() {
        let mut words = WATERFALL_FETCHED_WAVE64.to_vec();
        let save = word_at(&words, 0xBE96_2416);
        words[save + 1] = 0xBEFF_03C1; // s_mov_b32 exec_hi, -1
        assert_eq!(refusal(&words, true), NOT_SELECTED);
    }

    /// The waterfall with `v_cmpx_eq` narrowing EXEC itself, and the retry
    /// as `todo & !EXEC` (`folded`: in one `s_andn2_wrexec`), as LLVM's
    /// AMDGPU backend writes it for GFX10 and later.
    pub(crate) fn cmpx_waterfall(wave64: bool, folded: bool) -> Vec<u32> {
        let (base, todo, restore): (&[u32], u32, u32) = if wave64 {
            (WATERFALL_FETCHED_WAVE64, 0xBE94_047E, 0xBEFE_0414)
        } else {
            (WATERFALL_JOINED_WAVE32, 0xBE97_037E, 0xBEFE_0317)
        };
        let reads = word_at(base, 0x7E18_0504);
        let sample = word_at(base, 0xF09C_0F08);
        let mut w = base[..=word_at(base, todo)].to_vec();
        // s_mov_b32 s21, exec_lo / s_mov_b64 s[22:23], exec
        w.push(if wave64 { 0xBE96_047E } else { 0xBE95_037E });
        w.extend(&base[reads..reads + 8]); // v_readfirstlane_b32 s12..s19
        w.extend([
            0x7DE4_080C, // v_cmpx_eq_u64 s[12:13], v[4:5]
            0x7DE4_0C0E, // v_cmpx_eq_u64 s[14:15], v[6:7]
            0x7DE4_1010, // v_cmpx_eq_u64 s[16:17], v[8:9]
            0x7DE4_1412, // v_cmpx_eq_u64 s[18:19], v[10:11]
        ]);
        w.extend(&base[sample..sample + 2]); // image_sample_lz
        match (wave64, folded) {
            // s_andn2_wrexec_b32 s21, s21; s_cbranch_execnz
            (false, true) => w.extend([0xBE95_4715, 0xBF89_FFF0]),
            // s_andn2_b32 s21, s21, exec_lo; s_mov_b32 exec_lo, s21
            (false, false) => w.extend([0x8A15_7E15, 0xBEFE_0315, 0xBF89_FFEF]),
            // s_andn2_wrexec_b64 s[22:23], s[22:23]
            (true, true) => w.extend([0xBE96_3A16, 0xBF89_FFF0]),
            // s_andn2_b64 s[22:23], s[22:23], exec; s_mov_b64 exec, s[22:23]
            (true, false) => w.extend([0x8A96_7E16, 0xBEFE_0416, 0xBF89_FFEF]),
        }
        w.extend(&base[word_at(base, restore)..]);
        w
    }

    #[test]
    fn a_waterfall_narrowed_by_v_cmpx_is_one() {
        for wave64 in [false, true] {
            for folded in [false, true] {
                let r = waterfall(&cmpx_waterfall(wave64, folded), wave64)
                    .unwrap_or_else(|e| panic!("wave64 {wave64} folded {folded}: {e:?}"));
                assert_eq!(r.waterfall.compares.len(), 4, "{wave64} {folded}");
                assert_eq!(r.waterfall.reads.len(), 8, "{wave64} {folded}");
            }
        }
    }

    #[test]
    fn a_v_cmpx_waterfall_whose_retry_drops_lanes_is_none() {
        let mut words = cmpx_waterfall(false, false);
        let retry = word_at(&words, 0x8A15_7E15);
        words[retry] = 0x8A15_7E6A; // s_andn2_b32 s21, vcc_lo, exec_lo
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_waterfall_nested_in_another_is_refused_by_name() {
        // An image whose resource and sampler are both per lane, with one
        // loop per operand, the one for the sampler inside the one for the
        // resource (what the SelectionDAG legalisation in LLVM would give).
        // The console gives each lane its own pair. Refused for now: the
        // access of the inner loop reads the first-lane dwords of the outer
        // loop, which another block holds.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x3602_0081, // v_and_b32 v1, 1, v0
            0x3402_0285, // v_lshlrev_b32 v1, 5, v1
            0x3622_0080, // v_and_b32 v17, 0, v0
            0x3422_2284, // v_lshlrev_b32 v17, 4, v17
            0xDC38_8000,
            0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
            0xDC38_8010,
            0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
            0xDC38_8000,
            0x1406_0011, // global_load_dwordx4 v[20:23], v17, s[6:7]
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xBE94_037E, // s_mov_b32 s20, exec_lo
            // L1, the loop over the resource:
            0x7E18_0504, // v_readfirstlane_b32 s12, v4
            0x7E1A_0505,
            0x7E1C_0506,
            0x7E1E_0507,
            0x7E20_0508,
            0x7E22_0509,
            0x7E24_050A,
            0x7E26_050B, // v_readfirstlane_b32 s19, v11
            0xD4E2_0015,
            0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
            0xD4E2_0016,
            0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xD4E2_0016,
            0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xD4E2_0016,
            0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xBE95_3C15, // s_and_saveexec_b32 s21, s21
            0xBE9A_037E, // s_mov_b32 s26, exec_lo
            // L2, the loop over the sampler:
            0x7E10_0514, // v_readfirstlane_b32 s8, v20
            0x7E12_0515, // v_readfirstlane_b32 s9, v21
            0x7E14_0516, // v_readfirstlane_b32 s10, v22
            0x7E16_0517, // v_readfirstlane_b32 s11, v23
            0xD4E2_0018,
            0x0002_2808, // v_cmp_eq_u64_e64 s24, s[8:9], v[20:21]
            0xD4E2_0019,
            0x0002_2C0A, // v_cmp_eq_u64_e64 s25, s[10:11], v[22:23]
            0x8718_1918, // s_and_b32 s24, s24, s25
            0xBE98_3C18, // s_and_saveexec_b32 s24, s24
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            0x897E_187E, // s_xor_b32 exec_lo, exec_lo, s24
            0xBF89_FFF2, // s_cbranch_execnz L2
            0xBEFE_031A, // s_mov_b32 exec_lo, s26
            0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
            0xBF89_FFDA, // s_cbranch_execnz L1
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        assert_eq!(refusal(&words, false), SPLIT);
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_lds_fail_naming_it() {
        // The waterfall over descriptors each lane read from LDS:
        // ds_read_b128 v[4:7], v1 and ds_read_b128 v[8:11], v1 offset:16
        // in place of the global loads.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        words[5..9].copy_from_slice(&[0xDBFC_0000, 0x0400_0001, 0xDBFC_0010, 0x0800_0001]);
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of an LDS read",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// The block of `p` at byte `offset`.
    fn block_at(words: &[u32], offset: usize) -> usize {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        p.blocks.iter().position(|b| b.offset == offset).unwrap()
    }

    /// The table offset of what the scalar load of `s` reads.
    fn table_offset(r: &Resources, s: &Source) -> u32 {
        match *s {
            Source::Loaded { load, first: 0 } => {
                r.scalar.iter().find(|x| x.0 == load).unwrap().1.offset
            }
            ref s => panic!("{s:?}"),
        }
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_an_if_else_join_are_traced_per_path() {
        let words = WATERFALL_JOINED_WAVE32;
        let r = waterfall(words, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        // The join after the else arm: lanes that ran it (lanes from 16 on)
        // take the second entry (0x20); the others come through the flow
        // block, whose join gives lanes that ran the then arm the first one
        // (0). No lane that skipped the then arm comes that way, so the
        // dwords they held before (the inputs) are not traced.
        let Source::LanePhi { block, args } = &image.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0x9c));
        let [Some(flow), Some(other)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        let Source::LanePhi { block, args } = flow else {
            panic!("{flow:?}")
        };
        assert_eq!(*block, block_at(words, 0x70));
        let [None, Some(then)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        // The loop's compares are of the joined dwords; still recognised.
        assert_eq!(r.waterfall.compares.len(), 4);
        assert_eq!(r.waterfall.reads.len(), 8);
        // Without the lanes' paths the join is refused.
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let layout = [param(Table, 0, 1), param(Table, 1, 1)];
        let err = track(&p, &layout, HwStage::Cs, None).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of a phi, without the lanes' paths",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn descriptors_joined_from_a_per_lane_load_and_a_per_lane_choice_are_traced_per_path() {
        let words = WATERFALL_JOINED_MIXED_WAVE32;
        let r = waterfall(words, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        let Source::LanePhi { block, args } = &image.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0xb8));
        let [
            Some(Source::LanePhi { args: flow, .. }),
            Some(Source::Lanes { args: picked, .. }),
        ] = &args[..]
        else {
            panic!("{args:?}")
        };
        // The then arm loads each lane's entry from the table at s0.
        let [None, Some(Source::Fetched { load, first: 0 })] = flow[..] else {
            panic!("{flow:?}")
        };
        let a = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
        assert_eq!((a.param, a.offset), (0, 0));
        assert!(a.dynamic.is_some());
        // The else arm picks the first entry where vcc is set.
        let offsets: Vec<u32> = picked.iter().map(|s| table_offset(&r, s)).collect();
        assert_eq!(offsets, [0, 0x20]);
        assert_eq!(r.waterfall.compares.len(), 4);
    }

    #[test]
    fn a_descriptor_read_from_the_first_lane_of_a_loop_phi_fails_naming_it() {
        let err = waterfall(WATERFALL_LOOPED_WAVE32, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of a phi at a loop header",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_per_lane_descriptor_some_lanes_do_not_write_fails() {
        // The else arm of WATERFALL_JOINED_WAVE32 not skipped (its
        // s_cbranch_execz an s_nop): it runs with EXEC clear in the lanes
        // of the then arm, which keep the first entry; the second is not
        // every lane's.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let skip = words.iter().rposition(|&w| w == 0xBF88_0008).unwrap();
        words[skip] = 0xBF80_0000;
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a per-lane descriptor written where some lanes may keep the old one",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_buffer_descriptor_joined_per_lane_is_traced_on_the_paths_of_lanes_that_load() {
        let words = WATERFALL_JOINED_BUFFER_WAVE32;
        let r = waterfall(words, false).unwrap();
        let buffer = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        // The outer join: lanes from the else arm's inner join, or (below
        // 16) through the flow block.
        let Source::LanePhi { block, args } = &buffer.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0x78));
        let [
            Some(Source::LanePhi { args: flow, .. }),
            Some(Source::LanePhi {
                block: inner,
                args: nested,
            }),
        ] = &args[..]
        else {
            panic!("{args:?}")
        };
        let [None, Some(then)] = &flow[..] else {
            panic!("{flow:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        // Lanes from 24 on skip the inner arm and keep the inputs, but do
        // not load: that edge is not traced.
        assert_eq!(*inner, block_at(words, 0x74));
        let [None, Some(other)] = &nested[..] else {
            panic!("{nested:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        assert_eq!(r.waterfall.compares.len(), 2);
        assert_eq!(r.waterfall.reads.len(), 4);
    }

    #[test]
    fn root_constant_added_to_heap_base_is_dynamic_offset_not_second_base() {
        // s10 = ResourceHeap, s9 = Constants (dword 0): s_add_u32 s12, s10, s9 must trace to
        // ResourceHeap with dynamic offset s9 (not fail or collapse to MEMORY).
        let r = track_words(
            &[
                0x800C_090A, // s_add_u32 s12, s10, s9
                0xBE8D_0380, // s_mov_b32 s13, 0
                0xF408_0406,
                0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
                0xE030_2000,
                0x8004_0400, // buffer_load_dword v4, v0, s[16:19], 0 idxen
                S_ENDPGM,
            ],
            &[param(Constants, 9, 1), param(ResourceHeap, 10, 1)],
            HwStage::Vs,
        );
        let const_v = r.params.iter().find(|p| p.1 == 0).unwrap().0;
        assert_eq!(r.scalar[0].1.param, 1);
        assert_eq!(r.scalar[0].1.dynamic, Some(const_v));

        // Same when s9 (Constants) is passed directly as SMEM soffset.
        let r2 = track_words(
            &[
                0xBE8C_030A, // s_mov_b32 s12, s10
                0xBE8D_0380, // s_mov_b32 s13, 0
                0xF40C_0406,
                0x1200_0000, // s_load_dwordx8 s[16:23], s[12:13], s9
                S_ENDPGM,
            ],
            &[param(Constants, 9, 1), param(ResourceHeap, 10, 1)],
            HwStage::Vs,
        );
        let const_v2 = r2.params.iter().find(|p| p.1 == 0).unwrap().0;
        assert_eq!(r2.scalar[0].1.param, 1);
        assert_eq!(r2.scalar[0].1.dynamic, Some(const_v2));

        // s_lshl4_add_u32 s12, s9, s11 (s9 * 16 + s11 SamplerHeap): traces to SamplerHeap.
        let r3 = track_words(
            &[
                0x988C_0B09, // s_lshl4_add_u32 s12, s9, s11
                0xBE8D_0380, // s_mov_b32 s13, 0
                0xF408_0606,
                0xFA00_0000, // s_load_dwordx4 s[24:27], s[12:13], 0x0
                S_ENDPGM,
            ],
            &[param(Constants, 9, 1), param(SamplerHeap, 11, 1)],
            HwStage::Ps,
        );
        assert_eq!(r3.scalar[0].1.param, 1);
        assert!(r3.scalar[0].1.dynamic.is_some());

        // On a RootType::Descriptor (not a heap/table), s_lshl4_add_u32 must NOT set `dynamic`
        // to the sum (which would double-add the descriptor base); it stays MEMORY.
        let r4 = track_words(
            &[
                0x988E_0A09, // s_lshl4_add_u32 s14, s9, s10
                0xBE8F_030B, // s_mov_b32 s15, s11
                0xF408_0607,
                0xFA00_0000, // s_load_dwordx4 s[24:27], s[14:15], 0x0
                S_ENDPGM,
            ],
            &[param(Constants, 9, 1), param(Descriptor, 10, 2)],
            HwStage::Ps,
        );
        assert_eq!(r4.scalar[0].1.param, MEMORY);
    }

    #[test]
    fn infer_roots_finds_heaps_and_rejects_unknown_scalar_loads() {
        // s10 loads an image descriptor (hi=0, ResourceHeap) with dynamic offset s9 (Constants);
        // s11 loads a sampler descriptor (hi=0, SamplerHeap).
        let code: Vec<u8> = [
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
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w: &u32| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, true, 256).unwrap();
        let inf = infer_roots(&p, &[param(Constants, 9, 1)], HwStage::Vs).unwrap();
        assert_eq!(inf.heaps, [(10, ResourceHeap), (11, SamplerHeap)]);
        assert_eq!(inf.vertex_table, None);

        // Non-descriptor load from unknown s[6:7] produces no inferred heap or vertex table.
        let unknown: Vec<u8> = [0xF400_0203u32, 0xFA00_0000, S_ENDPGM]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let p2 = crate::ir::build(&unknown, true, 256).unwrap();
        let inf2 = infer_roots(&p2, &[], HwStage::Vs).unwrap();
        assert!(inf2.heaps.is_empty());
        assert_eq!(inf2.vertex_table, None);
    }

    #[test]
    fn heap_addresses_with_shifted_offsets_carries_and_joins_trace_cleanly() {
        // 1. Scalar 32-bit shift pair: s9 = Constants (unmasked index), s10 = ResourceHeap.
        //    off_lo = s9 << 5, off_hi = s9 >> 27, s[12:13] = s10 + off_64, masked to 48 bits.
        let shifted_32 = [
            0x8F0E_8509u32, // s_lshl_b32 s14, s9, 5
            0x900F_9B09,    // s_lshr_b32 s15, s9, 27
            0x800C_0E0A,    // s_add_u32 s12, s10, s14
            0x820D_0F80,    // s_addc_u32 s13, 0, s15
            0x938D_FF0D,
            0x0010_0000, // s_bfe_u32 s13, s13, 0x100000 (16 bits from 0)
            0xF408_0406,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
            0xE030_2000,
            0x8004_0400, // buffer_load_dword v4, v0, s[16:19], 0 idxen
            S_ENDPGM,
        ];
        let r1 = track_words(
            &shifted_32,
            &[param(Constants, 9, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ps,
        );
        assert_eq!(r1.scalar[0].1.param, 1);
        assert!(r1.scalar[0].1.dynamic.is_some());

        // Also inferred by `infer_roots` when s10 is undeclared!
        let code1: Vec<u8> = shifted_32.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p1 = crate::ir::build(&code1, false, 256).unwrap();
        let inf1 = infer_roots(&p1, &[param(Constants, 9, 1)], HwStage::Ps).unwrap();
        assert_eq!(inf1.heaps, [(10, ResourceHeap)]);

        // Negatives: unrelated SGPR, mismatched shift, mismatched bit-field (offset or width),
        // shared-parameter-only expression (`s9 >> 1` against `s9 * 4` or `s9`), and arbitrary or
        // multiplier-mismatched `v_mad_u64_u32` must all be refused and never inferred as a heap.
        let assert_refused_and_not_inferred = |words: &[u32], label: &str| {
            let bad_code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let bp = crate::ir::build(&bad_code, false, 256).unwrap();
            let err = track(
                &bp,
                &[param(Constants, 9, 1), param(ResourceHeap, 10, 1)],
                HwStage::Ps,
                lanes(&bp).as_ref(),
            )
            .unwrap_err();
            assert!(
                matches!(
                    err,
                    ShaderError::Unsupported {
                        what: "an address into a descriptor heap with an untracked high half",
                        ..
                    }
                ),
                "{label}: {err:?}"
            );
            let bad_inf = infer_roots(&bp, &[param(Constants, 9, 1)], HwStage::Ps).unwrap();
            assert!(bad_inf.heaps.is_empty(), "{label}: {:?}", bad_inf.heaps);
        };

        // 1a. Unrelated SGPR and mismatched right shift (`lo = s10 + (s9 << 5)`, `hi = addc(0, s9 >> 3)`).
        for (label, bad_hi) in [
            ("unrelated s8 >> 27", [0x900F_9B08u32, 0x800C_0E0A, 0x820D_0F80]),
            ("unrelated addc s8", [0xBF80_0000u32, 0x800C_0E0A, 0x820D_8008]),
            ("mismatched shift s9 >> 3", [0x900F_8309u32, 0x800C_0E0A, 0x820D_0F80]),
        ] {
            let mut bad = shifted_32;
            bad[1..4].copy_from_slice(&bad_hi);
            assert_refused_and_not_inferred(&bad, label);
        }

        // 1b. Matched vs mismatched bit-field extract on `lo = s10 + (s9 << 5)`:
        //     `s_bfe_u32 s15, s9, (5 << 16) | 27` succeeds; wrong width `3` or wrong offset `20`
        //     (`20 + 12 == 32`) are refused.
        let bfe_prog = |pack: u32| {
            [
                0x8F0E_8509u32, // s_lshl_b32 s14, s9, 5
                0x938F_FF09,
                pack,           // s_bfe_u32 s15, s9, pack
                0x800C_0E0A,    // s_add_u32 s12, s10, s14
                0x820D_0F80,    // s_addc_u32 s13, 0, s15
                0xF408_0406,
                0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
                0xE030_2000,
                0x8004_0400, // buffer_load_dword v4, v0, s[16:19], 0 idxen
                S_ENDPGM,
            ]
        };
        let r_bfe = track_words(
            &bfe_prog(0x0005_001B),
            &[param(Constants, 9, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ps,
        );
        assert_eq!(r_bfe.scalar[0].1.param, 1);
        assert_refused_and_not_inferred(&bfe_prog(0x0003_001B), "mismatched bfe width 3 at 27");
        assert_refused_and_not_inferred(&bfe_prog(0x000C_0014), "mismatched bfe offset 20 width 12");

        // 1c. Shared-parameter-only case (`lo = s10 + s9 * 4` or `s10 + s9`, `hi = addc(0, s9 >> 1)`):
        //     refused, whereas matched `s9 >> 30` against `s9 * 4` succeeds.
        let mul4_prog = |lo_op: u32, hi_shr: u32| {
            [
                lo_op,
                hi_shr,
                0x800C_0E0A, // s_add_u32 s12, s10, s14
                0x820D_0F80, // s_addc_u32 s13, 0, s15
                0xF408_0406,
                0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
                0xE030_2000,
                0x8004_0400, // buffer_load_dword v4, v0, s[16:19], 0 idxen
                S_ENDPGM,
            ]
        };
        let r_mul4 = track_words(
            &mul4_prog(0x930E_8409, 0x900F_9E09), // s14 = s9 * 4, s15 = s9 >> 30
            &[param(Constants, 9, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ps,
        );
        assert_eq!(r_mul4.scalar[0].1.param, 1);
        assert_refused_and_not_inferred(
            &mul4_prog(0x930E_8409, 0x900F_8109), // s14 = s9 * 4, s15 = s9 >> 1
            "shared parameter s9 * 4 vs s9 >> 1",
        );
        assert_refused_and_not_inferred(
            &mul4_prog(0xBE8E_0309, 0x900F_8109), // s14 = s9, s15 = s9 >> 1
            "shared parameter s9 vs s9 >> 1",
        );

        // 1d. Zero-addend `v_mad_u64_u32`: matched multiplier `12` (`lo = s10 + s9 * 12`,
        //     `hi = addc(0, readfirstlane((v_mad_u64_u32 s9, 12, 0).hi))`) succeeds; mismatched
        //     multiplier `20` on the same `s9`, unrelated `s8`, or direct `hi` without carry are
        //     all refused.
        let mad_prog = |mad_src: u32, with_addc: bool| {
            [
                0x930E_8C09u32, // s_mul_i32 s14, s9, 12
                0xD576_1602,
                mad_src,        // v_mad_u64_u32 v[2:3], s22, src0, src1, 0
                0x7E1E_0503,    // v_readfirstlane_b32 s15, v3
                0x800C_0E0A,    // s_add_u32 s12, s10, s14
                if with_addc {
                    0x820D_0F80 // s_addc_u32 s13, 0, s15
                } else {
                    0xBE8D_030F // s_mov_b32 s13, s15
                },
                0xF408_0406,
                0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
                0xE030_2000,
                0x8004_0400, // buffer_load_dword v4, v0, s[16:19], 0 idxen
                S_ENDPGM,
            ]
        };
        let r_mad = track_words(
            &mad_prog(0x0201_1809, true), // s9 * 12 + 0
            &[param(Constants, 9, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ps,
        );
        assert_eq!(r_mad.scalar[0].1.param, 1);
        assert_refused_and_not_inferred(
            &mad_prog(0x0201_2809, true), // s9 * 20 + 0 (wrong multiplier on shared s9)
            "mismatched mad multiplier 20 vs 12",
        );
        assert_refused_and_not_inferred(
            &mad_prog(0x0201_1808, true), // s8 * 12 + 0 (arbitrary mad source)
            "arbitrary mad source s8",
        );
        assert_refused_and_not_inferred(
            &mad_prog(0x0201_1809, false), // s13 = v3 directly without addc carry
            "direct zero-addend mad high half without addc",
        );

        // 2. Scalar `s_lshl_b64` of zero-extended index + `s_and_b64` 48-bit mask (`s_bfm_b64`).
        let shifted_64 = [
            0xBE8E_0309u32, // s_mov_b32 s14, s9
            0xBE8F_0380,    // s_mov_b32 s15, 0
            0x8F8E_850E,    // s_lshl_b64 s[14:15], s[14:15], 5
            0x800C_0E0A,    // s_add_u32 s12, s10, s14
            0x820D_800F,    // s_addc_u32 s13, s15, 0
            0x9286_80B0,    // s_bfm_b64 s[6:7], 48, 0
            0x878C_060C,    // s_and_b64 s[12:13], s[12:13], s[6:7]
            0xF408_0406,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
            0xE030_2000,
            0x8004_0400, // buffer_load_dword v4, v0, s[16:19], 0 idxen
            S_ENDPGM,
        ];
        let r2 = track_words(
            &shifted_64,
            &[param(Constants, 9, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ps,
        );
        assert_eq!(r2.scalar[0].1.param, 1);
        assert!(r2.scalar[0].1.dynamic.is_some());

        // 3. Control-flow phi joining `s_mov_b32 s5, 0` on one taken arm and `s_addc_u32 s5, 0, 0`
        //    on the other taken arm.
        let joined_hi = [
            0xBF06_8002u32, // s_cmp_eq_u32 s2, 0
            0xBF84_0003,    // s_cbranch_scc0 -> else (offset +3 dwords = dword 5)
            0xBE84_0300,    // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xBF82_0002,    // s_branch -> join (dword 7)
            0x8004_0200,    // else: s_add_u32 s4, s0, s2
            0x8205_8080,    // s_addc_u32 s5, 0, 0
            0xF408_0202,
            0xFA00_0000, // join: s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xE070_0000,
            0x8002_0000, // buffer_store_dword v0, off, s[8:11], 0
            S_ENDPGM,
        ];
        // When only the `s_addc_u32` arm is taken (or both arms compute `s4 = s0 + s2`),
        // `hi` joined across `0` and `s_addc_u32(0, 0)` traces cleanly.
        let mut both_added = joined_hi;
        both_added[2] = 0x8004_0200; // s_add_u32 s4, s0, s2 (so lo agrees across both arms)
        let r3 = track_words(
            &both_added,
            &[param(Table, 0, 1), param(Constants, 2, 1)],
            HwStage::Cs,
        );
        assert_eq!(r3.scalar[0].1.param, 0);
        assert!(r3.scalar[0].1.dynamic.is_some());

        // 4. Per-lane 64-bit VGPR heap address (`v_add_co_u32` + `v_add_co_ci_u32` with `saddr: off`)
        //    feeding a waterfall loop.
        let vgpr_heap = [
            0xBE86_0301u32, // s_mov_b32 s6, s1
            0xBE87_0380,    // s_mov_b32 s7, 0
            0xF408_0A03,
            0xFA00_0000, // s_load_dwordx4 s[40:43], s[6:7], 0x0
            0x3402_0085, // v_lshlrev_b32 v1, 5, v0
            0xD70F_0202,
            0x0002_0200, // v_add_co_u32 v2, s2, s0, v1
            0xD528_0203,
            0x0009_0080, // v_add_co_ci_u32 v3, s2, 0, 0, s2
            0xDC38_8000,
            0x047D_0002, // global_load_dwordx4 v[4:7], v[2:3], off
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xBE97_037E, // s_mov_b32 s23, exec_lo
            0xBE95_037E, // s_mov_b32 s21, exec_lo
            0x7E18_0504, // loop: v_readfirstlane_b32 s12, v4
            0x7E1A_0505, // v_readfirstlane_b32 s13, v5
            0x7E1C_0506, // v_readfirstlane_b32 s14, v6
            0x7E1E_0507, // v_readfirstlane_b32 s15, v7
            0x7DE4_080C, // v_cmpx_eq_u64 s[12:13], v[4:5]
            0x7DE4_0C0E, // v_cmpx_eq_u64 s[14:15], v[6:7]
            0xE030_0000,
            0x8003_0800, // buffer_load_dword v8, off, s[12:15], 0
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
            0xBF89_FFF4, // s_cbranch_execnz -> loop
            0xBEFE_0317, // s_mov_b32 exec_lo, s23
            0xE070_0000,
            0x800A_0800, // buffer_store_dword v8, off, s[40:43], 0
            S_ENDPGM,
        ];
        let r4 = waterfall(&vgpr_heap, false).unwrap();
        let buf = r4
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::Fetched { .. }))
            .unwrap();
        let Source::Fetched { load, first: 0 } = buf.source else {
            panic!("{:?}", buf.source);
        };
        let addr = r4.scalar.iter().find(|s| s.0 == load).unwrap().1;
        assert_eq!((addr.param, addr.offset), (0, 0));
        assert!(addr.dynamic.is_some());
    }

    #[test]
    fn deep_diamond_phi_and_add_chains_track_in_linear_time() {
        // 24 chained if/else diamonds (2^24 = 16,777,216 paths) inside an outer loop, updating
        // both a title-memory VGPR address (`v[2:3]`) and a heap SGPR address (`s[12:13]`),
        // exercising `flat_address -> address -> sum` and `joined_on_taken_edges` + `edge_taken`.
        let mut words = vec![
            0xBE8C_030Au32, // s_mov_b32 s12, s10 (ResourceHeap)
            0xBE8D_0380,    // s_mov_b32 s13, 0
            0x7E04_0202,    // v_mov_b32 v2, s2 (title-memory low)
            0x7E06_0203,    // v_mov_b32 v3, s3 (title-memory high)
            0xBE8E_0380,    // s_mov_b32 s14, 0 (loop counter)
        ];
        let loop_head = words.len();
        for step in 0..24u32 {
            let bit = 1u32 << (step & 7);
            words.extend([
                0x870F_FF09,
                bit,         // s_and_b32 s15, s9, bit
                0xBF06_800F, // s_cmp_eq_u32 s15, 0
                0xBF84_0002, // s_cbranch_scc0 -> else (+2 dwords)
                0x4A04_0484, // then: v_add_nc_u32 v2, 4, v2
                0xBF82_0001, // s_branch -> join (+1 dword)
                0x4A04_0488, // else: v_add_nc_u32 v2, 8, v2
            ]);
        }
        words.push(0x800E_810E); // s_add_u32 s14, s14, 1
        words.push(0xBF0C_820E); // s_cmp_lt_u32 s14, 2
        let rel = (loop_head as isize) - (words.len() as isize + 1);
        words.push(0xBF85_0000 | ((rel as i16 as u16) as u32)); // s_cbranch_scc1 -> loop_head
        // Plus a 24-diamond heap-offset phi chain outside the loop (`2^24` paths).
        for step in 0..24u32 {
            let bit = 1u32 << (step & 7);
            words.extend([
                0x870F_FF09,
                bit,         // s_and_b32 s15, s9, bit
                0xBF06_800F, // s_cmp_eq_u32 s15, 0
                0xBF84_0002, // s_cbranch_scc0 -> else (+2 dwords)
                0x800C_900C, // then: s_add_u32 s12, s12, 16
                0xBF82_0001, // s_branch -> join (+1 dword)
                0x800C_A00C, // else: s_add_u32 s12, s12, 32
            ]);
        }
        words.extend([
            0xDC30_8000,
            0x047D_0002, // global_load_dword v4, v[2:3], off
            0xF408_0406,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
            0xE070_0000,
            0x8004_0400, // buffer_store_dword v4, off, s[16:19], 0
            S_ENDPGM,
        ]);

        let t0 = std::time::Instant::now();
        let r = track_words(
            &words,
            &[param(Constants, 9, 1), param(ResourceHeap, 10, 1)],
            HwStage::Cs,
        );
        let elapsed = t0.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(250),
            "24-diamond phi chain took {elapsed:?}"
        );
        assert_eq!(r.scalar.len(), 2);
        assert_eq!(r.scalar[0].1.param, MEMORY);
        assert_eq!(r.scalar[1].1.param, 1);
    }
}
