//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::UserData;
use super::bitcode::{Attr, Bin, Cast, Module, Term, V, pred};
use super::container::SigElement;
use super::graphics;
use super::shader::{Shader, VIEW_INTEGER, VIEW_SIGNED, VIEW_STRIDE_BITS, VIEW_STRIDES, flags};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Elsewhere, Export, Input, Interp, Io, Target};
use crate::ir::{Block, EXEC, Exit, Inst, Phi, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, MEMORY, Resources, Role, Site, Source, USER_DATA};
use crate::structure::{Atom, Bool, LaneExit, Map, Set, Structured};

mod bvh;
mod halves;

type Why = &'static str;

/// The refusal texts of a cross-lane operation outside compute, in a pixel
/// and in a vertex program; the shared prefix keeps tallies grouping them.
macro_rules! outside_compute {
    ($op:literal) => {
        [
            concat!(
                "a cross-lane operation outside compute: ",
                $op,
                " in a pixel program"
            ),
            concat!(
                "a cross-lane operation outside compute: ",
                $op,
                " in a vertex program"
            ),
        ]
    };
}

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Pixel inputs split off an input read two ways: (original, new) semantic indices.
    pub splits: Vec<(u32, u32)>,
    /// Minimum entries needed from each descriptor table root parameter: `(slot, count)`.
    pub table_entries: Vec<(u8, u32)>,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub root_uavs: Vec<u8>,
    /// Reads title memory at computed addresses (the page table).
    pub memory: bool,
    /// Writes there (the written-pages bitmap).
    pub memory_writes: bool,
    /// Reads buffer views' addresses or texture views' sizes from
    /// descriptors (the view addresses).
    pub view_addresses: bool,
    /// Takes descriptors from the resource heap itself (root type 5, not
    /// a table), and whether as unordered access views.
    pub heap: bool,
    pub heap_writes: bool,
    /// Marks the pages it writes through buffer views (`emit`, "Write
    /// marks").
    pub view_marks: bool,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Heap handles written.
    #[cfg(test)]
    pub heap_handles: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    /// Whether it declares rasterizer-ordered views.
    #[cfg(test)]
    pub rovs: bool,
    /// Barriers added for narrow host waves ([`split_wave_barriers`]).
    #[cfg(test)]
    pub split_barriers: usize,
    /// The `WaveSize` the program declares.
    #[cfg(test)]
    pub wave_size: Option<u32>,
    /// Group barriers in the DXIL (the exchanges between host waves, and
    /// those for narrow waves).
    #[cfg(test)]
    pub barriers: usize,
    /// Whether the DXIL holds the group-shared array host waves exchange in.
    #[cfg(test)]
    pub exchanges: bool,
    /// The sample and gather calls, with their constant arguments.
    #[cfg(test)]
    pub texture_calls: Vec<(String, Vec<Option<i64>>)>,
    /// The wave prefix calls, with their constant arguments.
    #[cfg(test)]
    pub wave_prefix_calls: Vec<(String, Vec<Option<i64>>)>,
    /// The lane reads, with their constant arguments.
    #[cfg(test)]
    pub wave_read_lane_calls: Vec<(String, Vec<Option<i64>>)>,
    /// The wave reduction calls, with their constant arguments.
    #[cfg(test)]
    pub wave_active_calls: Vec<(String, Vec<Option<i64>>)>,
    /// Live f32 multiplies, including the inclusive and cross-wave prefix steps.
    #[cfg(test)]
    pub float_multiplies: usize,
    /// The output stores (`storeOutput`, `storeVertexOutput`, `storePrimitiveOutput`), with
    /// where each and the values it stores are written.
    #[cfg(test)]
    pub output_stores: Vec<super::bitcode::WrittenCall>,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
    /// Writes through buffer views mark their pages (`emit`, "Write marks").
    pub view_marks: bool,
    /// 64-bit image atomics run under per-texel locks (`emit`, "Atomic
    /// locks").
    pub atomic_locks: bool,
    /// Compute-stage biased samples run as gradient samples scaled by
    /// `2^bias` ([`Options::bias_as_gradients`](super::Options)).
    pub bias_as_gradients: bool,
    /// See [`Options::narrow_waves`](super::Options).
    pub narrow_waves: bool,
    /// See [`Options::wave32_host`](super::Options).
    pub wave32_host: bool,
    /// See [`Options::zero_copy_memory`](super::Options).
    pub zero_copy_memory: bool,
    /// See [`Options::rect_list`](super::Options).
    pub rect_list: bool,
    /// See [`Options::copy_params`](super::Options).
    pub copy_params: [u8; 32],
}

enum Stop {
    /// Phis whose incoming value has no DXIL form were read: redo with them missing. All that
    /// one pass finds, not the first: a pass for each made a program with some hundreds of
    /// them take that many times as long (tens of seconds).
    Retry(Vec<(Value, Why)>),
    Fail(ShaderError),
}

#[cfg(any(test, all(windows, feature = "host")))]
pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    lower_graphics(parts, false)
}

#[cfg(test)]
thread_local! {
    /// Lowering passes of this thread's last [`lower_graphics`].
    pub(super) static PASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn lower_graphics(parts: &Parts, guessed: bool) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    #[cfg(test)]
    PASSES.set(0);
    // Compute programs hold masks the structurisation cannot see into
    // (built from scalars, shifted) only with the console's lanes: asked
    // for when the first try finds one used.
    let mut console = false;
    loop {
        #[cfg(test)]
        PASSES.set(PASSES.get() + 1);
        match Lower::new(parts, &bad, console, guessed).run() {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(phis)) => {
                for (v, why) in phis {
                    if bad.insert(v, why).is_some() {
                        return Err(ShaderError::Unsupported {
                            offset: 0,
                            what: "a phi that does not settle",
                        });
                    }
                }
            }
            Err(Stop::Fail(ShaderError::Unsupported {
                what: "an opaque lane mask" | "a constant lane mask",
                ..
            })) if !console && parts.native.stage == HwStage::Cs => console = true,
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

const TWO_PI: f32 = std::f32::consts::TAU;
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

/// A descriptor load's address given by root parameters alone: root
/// parameter and byte offset.
type EntryAddress = (usize, u32);

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    guessed: bool,
    bad: &'a HashMap<Value, Why>,
    lane_fail: Map<Value, Why>,
    vals: Map<Value, Val>,
    floats: Map<Value, V>,
    /// Values (missing in `vals`) whose low 16 bits are known: a 16-bit
    /// result written over a value with no DXIL form keeps its high half.
    low16: Map<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: Map<(usize, Bool), V>,
    lane_bits: Map<Value, V>,
    /// IR phi location by value.
    phi_at: Map<Value, (usize, usize)>,
    mask_phis: Map<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    /// [`Lower::unneeded_edges`], made at the first join that asks.
    unneeded: std::cell::OnceCell<Set<(Value, usize)>>,
    /// [`Lower::load_use`], made at the first descriptor load read as data.
    load_uses: std::cell::OnceCell<Map<Site, (Site, u8)>>,
    data_phis: Vec<(V, usize, usize)>,
    /// [`Lower::spill_slot`]: the phi made for lane `.1` of VGPR phi `.0`, and those whose
    /// incoming values are still to fill (block, phi index, lane).
    spill_phis: Map<(Value, u32), V>,
    pending_spills: Vec<(V, usize, usize, u32)>,
    /// Phis that code reads. Left a std set: [`Lower::live_edges`] starts
    /// its walk in this set's iteration order, and few phis are read.
    read_phis: HashSet<Value>,
    params: Map<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    /// Minimum entries needed from each `RootType::Table` root slot.
    table_entries: BTreeMap<u8, u32>,
    /// Handles of root SRVs and UAVs read as bounded views, by user-data
    /// dword ([`Lower::root_view`]).
    root_views: Map<u32, V>,
    samplers: Vec<[u32; 4]>,
    descriptors: Map<(Site, bool), &'a Source>,
    scalar: Map<Site, Address>,
    exports: Map<Site, &'a Export>,
    interps: Map<Site, &'a Interp>,
    /// The lists of `x.res.waterfall` as sets: they are tested for every
    /// VALU instruction, and a bindless shader has hundreds of compares.
    waterfall_compares: Set<Site>,
    waterfall_reads: Set<Value>,
    waterfall_loads: Set<Site>,
    handles: Map<(usize, Site, u8, bool, u32), V>,
    /// Handles made in the entry block for descriptors at addresses the
    /// root parameters alone give (table and heap loads with no other
    /// term), by base value, byte offset, dword, offset register, sampler
    /// heap and properties ([`Lower::entry_address`]).
    entry_handles: Map<(EntryAddress, u8, bool, u32), V>,
    /// Handles of descriptors chosen by control flow, made in their join
    /// block, by the choice (its `Debug` text), sampler heap and properties.
    join_handles: Map<(String, bool, u32), V>,
    /// Blocks splitting the edge from one IR block to another.
    edges: Map<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// [`Lowered::heap`], [`Lowered::heap_writes`].
    heap: [bool; 2],
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    /// Console wave64 code on 32-lane host waves ([`Options::wave32_host`]).
    host32: bool,
    /// The same, where a console wave is two host waves and the program has
    /// cross-lane operations: lanes are `flat & 63`, and what reaches across
    /// the halves goes through group-shared memory (`lower/halves.rs`).
    halves: bool,
    /// With `halves`, blocks where both halves of a console wave arrive
    /// together: [`once_blocks`], or [`wave_blocks`] in a group of one wave.
    once: Vec<bool>,
    /// Why the program cannot run on a 32-lane host.
    refuse: Option<Why>,
    /// The group's lane in console order, made in the entry block.
    console_flat: Option<V>,
    /// Pixel: an ordered section (POPS), whose UAVs the host makes
    /// rasterizer-ordered views.
    pops: bool,
    /// The program uses title memory (addresses it computes, the BVH
    /// instruction, root parameters through the user-data pointer): its root
    /// SRVs and UAVs are read there too ([`Lower::raw_view`]).
    root_memory: bool,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: Set<usize>,
    /// Group barriers before these instructions (index = length: at the
    /// block's end), for a one-wave group split across host waves
    /// ([`split_wave_barriers`]).
    split_barriers: Set<(usize, usize)>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword ([`packed_half`]).
    half_outputs: HashMap<u8, bool>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: Map<Value, (V, V)>,
    /// SCC written by a scalar operation on masks, by value: the dwords of
    /// the mask result it says is not zero.
    mask_scc: Map<Value, (Value, Option<Value>)>,
    /// Mesh: SCC of a uniform compare ANDed with EXEC ([`Self::uniform_and_exec`]): the
    /// compare's result value.
    uniform_scc: Map<Value, Value>,
    /// Mesh: SCC of EXEC ANDed with (or without) a lane function of uniform atoms
    /// ([`Self::uniform_with_exec`]): the function, and whether SCC is its complement.
    uniform_bits: Map<Value, UniformBit>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: Set<Value>,
    /// Block writing each value.
    def_block: Map<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: Set<Value>,
    /// Constant-lane reads `(v, lane)` from outside `v`'s section, and their lowered values.
    section_lane_reads: Set<(Value, u32)>,
    section_lanes: Map<(Value, u32), Result<V, Why>>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// The 16/64-bit round mode (MODE bits 3:2) each block starts with, and
    /// the one in force at the instruction being lowered (see [`round16`]).
    round16: Vec<Option<u8>>,
    mode16: Option<u8>,
    /// Generic pixel inputs interpolated with more than one mode:
    /// `(original input id, mode)` -> split `SigElement` id on its own row.
    split_inputs: Map<(usize, u8), usize>,
    /// Debug (`VICTORY_DEBUG_VS_POS`): the last two-dword raw vertex fetch, as floats.
    debug_pos: Option<(V, V)>,
    /// Rectangle list: which stored corner holds the right angle, `(block, at v0, at v2)`
    /// (neither: v1), picked from the first two-channel float fetch (the position).
    rect_corner: Option<(usize, V, V)>,
}

/// The 16/64-bit round mode (MODE bits 3:2: 0 nearest even, 3 toward zero)
/// after `inst`, from `mode` before it (`None`: not known).
fn round16_after(inst: &Inst, mode: Option<u8>) -> Option<u8> {
    match (inst.name(), &inst.fields) {
        ("s_round_mode", Fields::Sopp { imm }) => Some((imm >> 2 & 3) as u8),
        (
            "s_setreg_imm32_b32" | "s_setreg_b32",
            Fields::Sop {
                src, imm: Some(h), ..
            },
        ) => {
            let (id, off, size) = (h & 0x3F, h >> 6 & 0x1F, (h >> 11 & 0x1F) + 1);
            if id != 1 || off >= 4 || off + size <= 2 {
                return mode;
            }
            match src.first() {
                Some(&Operand::Literal(x)) if off <= 2 && off + size >= 4 => {
                    Some((x >> (2 - off) & 3) as u8)
                }
                _ => None,
            }
        }
        _ => mode,
    }
}

/// The 16/64-bit round mode each block starts with: nearest even at the
/// program's start, then as the MODE writes on the way set it; `None` where
/// paths that set it differently join (or a write is not a constant).
fn round16(p: &crate::ir::Program) -> Vec<Option<u8>> {
    // None: not reached yet; Some(m): the mode (m None: not known).
    let mut entry: Vec<Option<Option<u8>>> = vec![None; p.blocks.len()];
    entry[0] = Some(Some(0));
    let order = crate::ir::reverse_postorder(&p.blocks);
    loop {
        let mut changed = false;
        for &b in &order {
            let Some(mut m) = entry[b] else { continue };
            for inst in &p.blocks[b].insts {
                m = round16_after(inst, m);
            }
            for s in p.blocks[b].exit.successors() {
                let new = match entry[s] {
                    None => Some(m),
                    Some(old) if old == m => continue,
                    Some(_) => Some(None),
                };
                if entry[s] != new {
                    entry[s] = new;
                    changed = true;
                }
            }
        }
        if !changed {
            return entry.into_iter().map(Option::flatten).collect();
        }
    }
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards). The limit is also what such a loop
/// costs: a dispatch lasts as long as its slowest lane, and one lane's 2^20
/// passes took 60 to 195 ms on an Apple M5 GPU, where 2^14 take 1 to 3 ms.
// shortcut: a genuine loop of more iterations than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 14;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
/// A page table lookup ([`Lower::memory`]).
struct MemoryEntry {
    handle: V,
    offset: V,
    /// Block × 4096 + page in the block.
    entry: V,
    block: V,
    /// The level-1 index: address bits 28-39.
    l1: V,
    /// No host copy holds the page.
    miss: V,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared. Only a mask's bit
    /// count is taken, over the host's wave (the console compiler's test of
    /// a value for uniformity, `WaveActiveAllEqual`): lane numbers stay the
    /// console's, so the first lane and the count below a lane are refused.
    None,
}

/// A mesh SCC's lane function, whether SCC is its complement, and the atoms EXEC decides.
type UniformBit = (Bool, bool, Vec<(Atom, bool)>);

/// What a value holding a lane mask read as a number stops with.
const LANE_MASK: Why = "a lane mask read as data";

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// Where half `h` (dword h / 2, the high half when odd) of the parameter
/// whose first element is in row `a` comes from, when the console packs
/// 16-bit values two per dword: a row and channel of `elements`, none for
/// padding. The console fills its parameters with the 16-bit elements taken
/// by first channel, then row: each takes whole dwords of the first
/// parameter with room for it (four dwords each), its channels in order.
fn packed_half(elements: &[SigElement], a: u8, h: u8) -> Option<(u8, u8)> {
    let mut halves: Vec<&SigElement> = elements
        .iter()
        .filter(|e| e.half && e.kind == 0 && e.row.is_some())
        .collect();
    halves.sort_by_key(|e| (e.col, e.row));
    // (first element's row, (row, channel) per half)
    type Param = (Option<u32>, Vec<Option<(u8, u8)>>);
    let mut params: Vec<Param> = Vec::new();
    for e in halves {
        let need = usize::from(e.cols.div_ceil(2) * 2);
        let i = match params.iter().position(|p| p.1.len() + need <= 8) {
            Some(i) => i,
            None => {
                params.push((e.row, Vec::new()));
                params.len() - 1
            }
        };
        let row = u8::try_from(e.row?).ok()?;
        let slots = &mut params[i].1;
        slots.extend((0..e.cols).map(|c| Some((row, e.col + c))));
        slots.resize(slots.len().next_multiple_of(2), None);
    }
    // Two parameters starting in one row would leave the half ambiguous.
    let mut found = params.iter().filter(|p| p.0 == Some(u32::from(a)));
    match (found.next(), found.next()) {
        (Some(p), None) => p.1.get(usize::from(h)).copied().flatten(),
        _ => None,
    }
}

/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]),
/// by row: whether two per dword.
fn half_outputs(io: &StageIo) -> HashMap<u8, bool> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    entries
        .iter()
        .filter(|o| o.param & 0x80 == 0 && o.fp16)
        .map(|o| (o.row, o.pack2))
        .collect()
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

/// The scalar instruction defining an SSA value in this block and its operands.
fn scalar_def<'a, const N: usize>(
    block: &'a Block,
    value: Value,
    op: &str,
) -> Option<(&'a Inst, [Operand; N])> {
    let inst = block
        .insts
        .iter()
        .find(|i| i.writes.iter().any(|w| w.1 == value))?;
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    (inst.name() == op).then_some((inst, src.as_slice().try_into().ok()?))
}

fn sgpr_operand(inst: &Inst, operand: Operand) -> Option<Value> {
    match operand {
        Operand::Scalar {
            code: code @ 0..=101,
            dwords: 1,
        } => inst.read(Reg::S(code)),
        _ => None,
    }
}

fn scalar_constant(operand: Operand) -> Option<u32> {
    match operand {
        Operand::Const(code) => crate::resource::inline(code),
        Operand::Literal(value) => Some(value),
        _ => None,
    }
}

/// The dwords an `s_load_dword*` reads.
fn smem_dwords(name: &str) -> Option<u32> {
    match name.strip_prefix("s_load_dword")? {
        "" => Some(1),
        "x2" => Some(2),
        "x4" => Some(4),
        "x8" => Some(8),
        "x16" => Some(16),
        _ => None,
    }
}

enum InterpUse<'a> {
    Phi(Value),
    Read(Reg, &'a Inst),
}

struct InterpUses<'a> {
    by_value: Map<Value, Vec<InterpUse<'a>>>,
    #[cfg(test)]
    visits: std::cell::Cell<usize>,
}

impl<'a> InterpUses<'a> {
    fn new(program: &'a Program) -> Self {
        let mut by_value: Map<_, Vec<_>> = Map::default();
        for block in &program.blocks {
            for phi in &block.phis {
                for arg in &phi.args {
                    by_value
                        .entry(*arg)
                        .or_default()
                        .push(InterpUse::Phi(phi.value));
                }
            }
            for inst in &block.insts {
                for &(reg, val) in &inst.reads {
                    by_value
                        .entry(val)
                        .or_default()
                        .push(InterpUse::Read(reg, inst));
                }
            }
        }
        Self {
            by_value,
            #[cfg(test)]
            visits: Default::default(),
        }
    }

    /// Whether a pixel-shader live-in SGPR `start` flows exclusively (through
    /// scalar copies, phis, and constant-lane VGPR spills) into `M0` for
    /// `v_interp_*` instructions and nowhere else (e.g. `PRIM_MASK` in a
    /// guessed pixel shader whose SGPR slot is not `initialized_sgprs - 2`).
    fn is_interp_m0_only(&self, start: Value) -> bool {
        let mut work = vec![(start, None::<u32>)];
        let mut seen = HashSet::new();
        seen.insert((start, None));
        let mut used_by_interp = false;
        while let Some((val, spilled)) = work.pop() {
            for site in self.by_value.get(&val).into_iter().flatten() {
                #[cfg(test)]
                self.visits.set(self.visits.get() + 1);
                let (reg, inst) = match *site {
                    InterpUse::Phi(result) => {
                        if seen.insert((result, spilled)) {
                            work.push((result, spilled));
                        }
                        continue;
                    }
                    InterpUse::Read(reg, inst) => (reg, inst),
                };
                let name = inst.name();
                match spilled {
                    Some(lane) => match (&inst.fields, name) {
                        (Fields::Valu(v), "v_writelane_b32") => {
                            // `val` must be the old dst VGPR, not src0.
                            let is_src0 = matches!(
                                v.src.first(),
                                Some(&Operand::Scalar { code, .. })
                                    if reg == Reg::S(code)
                            );
                            let Some(other) =
                                v.src.get(1).and_then(|&l| crate::resource::lane_of(l))
                            else {
                                return false;
                            };
                            if is_src0 {
                                return false;
                            }
                            if other != lane {
                                let Some(Operand::Vector { reg: d, .. }) = v.dst else {
                                    return false;
                                };
                                let Some(w) = inst.write(Reg::V(d)) else {
                                    return false;
                                };
                                if seen.insert((w, Some(lane))) {
                                    work.push((w, Some(lane)));
                                }
                            }
                        }
                        (Fields::Valu(v), "v_readlane_b32") => {
                            let Some(rlane) =
                                v.src.get(1).and_then(|&l| crate::resource::lane_of(l))
                            else {
                                return false;
                            };
                            if rlane == lane {
                                let Some(Operand::Scalar { code, .. }) = v.dst else {
                                    return false;
                                };
                                let Some(w) = inst.write(Reg::S(code)) else {
                                    return false;
                                };
                                if seen.insert((w, None)) {
                                    work.push((w, None));
                                }
                            }
                        }
                        _ => return false,
                    },
                    None if reg == Reg::S(crate::ir::M0) => {
                        if name.starts_with("v_interp_")
                            || matches!(inst.fields, Fields::Vintrp { .. })
                        {
                            used_by_interp = true;
                        } else if matches!(name, "s_sendmsg" | "s_sendmsghalt")
                            && matches!(inst.fields, Fields::Sopp { imm } if imm & 0xF == 7)
                        {
                            // `MSG_ORDERED_PS_DONE` ignores M0 in hardware, though
                            // `ir::build` records an implicit M0 read on all `s_sendmsg`.
                        } else if name == "s_mov_b32"
                            && let Fields::Sop {
                                dst: Some(Operand::Scalar { code, .. }),
                                ..
                            } = &inst.fields
                            && let Some(w) = inst.write(Reg::S(*code))
                        {
                            if seen.insert((w, None)) {
                                work.push((w, None));
                            }
                        } else {
                            return false;
                        }
                    }
                    None => match (&inst.fields, name) {
                        (
                            Fields::Sop {
                                dst: Some(Operand::Scalar { code, .. }),
                                ..
                            },
                            "s_mov_b32",
                        ) => {
                            let Some(w) = inst.write(Reg::S(*code)) else {
                                return false;
                            };
                            if seen.insert((w, None)) {
                                work.push((w, None));
                            }
                        }
                        (
                            Fields::Sop {
                                dst: Some(Operand::Scalar { code: dcode, .. }),
                                src,
                                ..
                            },
                            "s_mov_b64",
                        ) => {
                            let Some(&Operand::Scalar { code: scode, .. }) = src.first() else {
                                return false;
                            };
                            let half = if reg == Reg::S(scode) {
                                0
                            } else if reg == Reg::S(scode + 1) {
                                1
                            } else {
                                return false;
                            };
                            let Some(w) = inst.write(Reg::S(dcode + half)) else {
                                return false;
                            };
                            if seen.insert((w, None)) {
                                work.push((w, None));
                            }
                        }
                        (Fields::Valu(v), "v_writelane_b32") => {
                            let is_src0 = matches!(
                                v.src.first(),
                                Some(&Operand::Scalar { code, .. })
                                    if reg == Reg::S(code)
                            );
                            let Some(lane) =
                                v.src.get(1).and_then(|&l| crate::resource::lane_of(l))
                            else {
                                return false;
                            };
                            let Some(Operand::Vector { reg: d, .. }) = v.dst else {
                                return false;
                            };
                            if !is_src0 {
                                return false;
                            }
                            let Some(w) = inst.write(Reg::V(d)) else {
                                return false;
                            };
                            if seen.insert((w, Some(lane))) {
                                work.push((w, Some(lane)));
                            }
                        }
                        _ => return false,
                    },
                }
            }
        }
        used_by_interp
    }
}

impl<'a> Lower<'a> {
    fn new(x: &'a Parts<'a>, bad: &'a HashMap<Value, Why>, console: bool, guessed: bool) -> Self {
        let p = x.p;
        let mut phi_at = Map::default();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        });
        let root_memory = x
            .res
            .scalar
            .iter()
            .any(|s| matches!(s.1.param, MEMORY | USER_DATA))
            || insts().any(|i| i.name().starts_with("image_bvh"))
            // A view of the title's own memory from a root address on is linear in the pool past
            // the title page it starts in, and the title maps the next pages anywhere in the pool
            // (Grand Theft Auto VI's vertex fetch: a root SRV, vertex buffers in 64 KiB pages
            // mapped in descending pool order). Through the page table every access lands where
            // the title has it. Compute keeps the views: no case seen, and a lookup per access.
            || (x.zero_copy_memory && x.native.stage != HwStage::Cs);
        let mut console_lanes = console;
        let (mut host32, mut halves, mut once, mut refuse) = (false, false, Vec::new(), None);
        let mut split_barriers = Set::default();
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            s.lds_dwords = cs.lds_bytes() / 4;
            // Code reading other lanes needs the console's wave size and
            // lane order; so do lane masks read as numbers, and LDS shared
            // within one wave, where the console compiler leaves out barriers.
            console_lanes = console
                || insts().any(|i| {
                    let n = i.name();
                    n.starts_with("ds_swizzle")
                        || n.contains("permute")
                        || n.starts_with("v_permlane")
                        || n.starts_with("v_mbcnt")
                        || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                        // One atomic per console wave; addresses by console lane.
                        || matches!(n, "ds_append" | "ds_consume")
                        || n.contains("addtid")
                        || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
                        || reads_mask_as_data(x.st, i)
                })
                || {
                    // Masks held as numbers or constant lane sets: the lane
                    // tests its own bit by its console lane.
                    let whens = x.st.exits.iter().filter_map(|e| match *e {
                        LaneExit::Branch { when, .. } => Some(when),
                        _ => None,
                    });
                    let mut all: Vec<Bool> =
                        x.st.guards
                            .iter()
                            .flatten()
                            .flatten()
                            .copied()
                            .chain(whens)
                            .collect();
                    all.sort_unstable();
                    all.dedup();
                    all.into_iter().any(|g| {
                        x.st.bools
                            .support(g)
                            .iter()
                            .any(|a| matches!(a, Atom::Opaque(_) | Atom::Constant(_)))
                    })
                };
            let wave = if x.native.wave64 { 64 } else { 32 };
            let lanes = cs.threads.iter().product::<u32>();
            let one_wave = lanes <= wave;
            // The console's one wave is several host waves: LDS no longer
            // passes between its lanes without a barrier.
            let mut synced = false;
            host32 = x.wave32_host && x.native.wave64;
            if (x.narrow_waves || host32) && one_wave && lanes > 32 && s.lds_dwords > 0 {
                // shortcut: a read in divergent code of what divergent code
                // wrote gets no barrier (none could be placed there); the
                // program then runs as before.
                if x.st.lockstep {
                    split_barriers = super::lockstep::lds_barriers(p);
                    synced = true;
                } else if let Some(b) = split_wave_barriers(p, x.st) {
                    split_barriers = b;
                    synced = true;
                }
            }
            // A host whose waves are narrower than WaveSize(64) does not run
            // the dispatch (D3DMetal: nothing written): ask for the console's
            // width only where wave operations need it, or LDS is not synced.
            let fits = x.narrow_waves && (synced || lanes <= 32);
            if host32 {
                // Whole-wave sections are reductions over the console's
                // 64 lanes (`halves.rs`).
                halves = lanes > 32 && (console_lanes || super::wave::has_sections(p) || x.st.lockstep);
                // No host wave is 64 lanes: LDS a wave shares unsynced
                // cannot run.
                if one_wave && lanes > 32 && s.lds_dwords > 0 && !synced {
                    refuse = Some(SPLIT_LDS);
                }
                console_lanes |= halves;
                if console_lanes {
                    s.wave_size = Some(32);
                }
                if halves {
                    once = if one_wave { wave_blocks(p, x.st) } else { once_blocks(p) };
                    for (o, at_barrier) in once.iter_mut().zip(barrier_blocks(p, x.st)) {
                        *o |= at_barrier;
                    }
                    s.xchg_dwords = lanes.next_multiple_of(64);
                }
            } else if console_lanes || one_wave && s.lds_dwords > 0 && !fits {
                s.wave_size = Some(wave);
            }
        }
        let mut no_barrier = Set::default();
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        Lower {
            s,
            x,
            guessed,
            bad,
            lane_fail: Map::default(),
            vals: Map::default(),
            floats: Map::default(),
            low16: Map::default(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: Map::default(),
            lane_bits: Map::default(),
            phi_at,
            mask_phis: Map::default(),
            pending_masks: Vec::new(),
            unneeded: std::cell::OnceCell::new(),
            load_uses: std::cell::OnceCell::new(),
            data_phis: Vec::new(),
            spill_phis: Map::default(),
            pending_spills: Vec::new(),
            read_phis: HashSet::new(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            table_entries: BTreeMap::new(),
            root_views: Map::default(),
            samplers: Vec::new(),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            waterfall_compares: x.res.waterfall.compares.iter().copied().collect(),
            waterfall_reads: x.res.waterfall.reads.iter().copied().collect(),
            waterfall_loads: x.res.waterfall.loads.iter().copied().collect(),
            handles: Map::default(),
            entry_handles: Map::default(),
            join_handles: Map::default(),
            edges: Map::default(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            heap: [false; 2],
            src0: None,
            console_lanes,
            host32,
            halves,
            once,
            console_flat: None,
            refuse,
            pops: matches!(x.native.io, StageIo::Ps { db_shader_control, .. }
                if db_shader_control & 1 << 16 != 0),
            root_memory,
            flat: None,
            no_barrier,
            split_barriers,
            half_outputs: half_outputs(&x.native.io),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs => MaskData::Host,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: Map::default(),
            mask_scc: Map::default(),
            uniform_scc: Map::default(),
            uniform_bits: Map::default(),
            balloting: Set::default(),
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: Set::default(),
            section_lane_reads: Set::default(),
            section_lanes: Map::default(),
            code_table: None,
            counters: HashMap::new(),
            abort: None,
            round16: round16(p),
            mode16: Some(0),
            split_inputs: Map::default(),
            debug_pos: None,
            rect_corner: None,
        }
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| a.offset & !3)
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// A global memory access (`global_*`): loads from the program's
    /// constant table, from a root view at the address the title sets, or
    /// from title memory at an address the program computes; stores and
    /// atomics through a root view.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if !name.starts_with("global_load_") {
            return self.flat_write(inst, b, i);
        }
        let rest = &name["global_load_".len()..];
        let (n, sub) = match rest {
            "dword" => (1, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let c = self.cur;
        let bytes = sub.map_or(4 * n, |(width, _)| width / 8);
        let (h, off) = if a.param == MEMORY {
            let (lo, hi) = self.global_address(inst)?;
            let (h, off) = self.memory(lo, hi, bytes, false, guard);
            (Some(h), off)
        } else {
            let mut off = self.s.m.i32(a.offset);
            if let Some(d) = a.dynamic {
                let d = self.i32_of(d)?;
                off = self.s.m.bin(c, Bin::Add, off, d);
            }
            if a.param == CODE {
                (None, off)
            } else {
                let src = Source::Built {
                    address: a,
                    bits: 0,
                    words: [0, 0],
                };
                let (h, off) = self.raw_view((b, i), &src, off, bytes, (11, 0), false, guard)?;
                (Some(h), off)
            }
        };
        let c = self.cur;
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let c = l.cur;
                    let (u, mask, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i8(1), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, byte, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    Ok(l.as_int(x))
                }
            }
        };
        for k in 0..n {
            let x = match sub {
                None => {
                    let four = self.s.m.i32(4 * k);
                    let at = self.s.m.bin(c, Bin::Add, off, four);
                    dword(self, at)?
                }
                Some((width, signed)) => {
                    // shortcut: the field is taken not to cross a dword.
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let at = self.s.m.bin(c, Bin::And, off, not3);
                    let w = dword(self, at)?;
                    let byte = self.s.m.bin(c, Bin::And, off, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let x = self.s.m.bin(c, Bin::LShr, w, sh);
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    self.s
                        .m
                        .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left)
                }
            };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The 64-bit address a `global_*` instruction accesses in each lane:
    /// its SGPR pair plus its VGPR offset, or its VGPR pair, plus its
    /// immediate; the high dword cut to address bits 32-47.
    fn global_address(&mut self, inst: &Inst) -> Result<(V, V), Why> {
        const WHY: Why = "a global address";
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = inst.fields
        else {
            return Err(WHY);
        };
        let read = |l: &mut Self, r: Reg| -> Result<V, Why> {
            let v = inst.read(r).ok_or(WHY)?;
            l.i32_of(v)
        };
        let (lo, hi) = match saddr {
            Some(Operand::Scalar { code, .. }) if code != NULL => {
                let (lo, hi) = (read(self, Reg::S(code))?, read(self, Reg::S(code + 1))?);
                let x = read(self, Reg::V(reg))?;
                self.add64(lo, hi, x, false)
            }
            _ => (read(self, Reg::V(reg))?, read(self, Reg::V(reg + 1))?),
        };
        let k = self.s.m.i32(offset as i32 as u32);
        let (lo, hi) = self.add64(lo, hi, k, offset < 0);
        let mask = self.s.m.i32(0xFFFF);
        Ok((lo, self.s.m.bin(self.cur, Bin::And, hi, mask)))
    }

    /// A global store or atomic (`global_store_*`, `global_atomic_*`)
    /// through a root view at the address the title sets, or into title
    /// memory at an address the program computes.
    fn flat_write(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat {
            dst, data, cache, ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        if a.param == CODE {
            return Err("a global store into the code");
        }
        let n = match name.strip_prefix("global_store_") {
            None => 1,
            Some("dword") => 1,
            Some("dwordx2") => 2,
            Some("dwordx3") => 3,
            Some("dwordx4") => 4,
            Some(_) => return Err(name),
        };
        // DXIL atomic operations: add, and, or, xor, signed and unsigned min
        // and max, exchange; subtraction adds the negation. A float minimum
        // or maximum is the signed integer one where the value is positive
        // and the reversed unsigned one where negative (IEEE order of the
        // bits), as two atomics, each a no-op in the lanes the other serves.
        let ops: Option<&[(u32, u32)]> = match name.strip_prefix("global_atomic_") {
            None => None,
            Some("add" | "sub") => Some(&[(0, 0)]),
            Some("and") => Some(&[(1, 0)]),
            Some("or") => Some(&[(2, 0)]),
            Some("xor") => Some(&[(3, 0)]),
            Some("smin") => Some(&[(4, 0)]),
            Some("smax") => Some(&[(5, 0)]),
            Some("umin") => Some(&[(6, 0)]),
            Some("umax") => Some(&[(7, 0)]),
            Some("swap") => Some(&[(8, 0)]),
            Some("fmin") if !cache.glc => Some(&[(4, i32::MAX as u32), (7, 0)]),
            Some("fmax") if !cache.glc => Some(&[(5, i32::MIN as u32), (6, u32::MAX)]),
            Some(_) => return Err(name),
        };
        if ops.is_none() && !name.starts_with("global_store_") {
            return Err(name);
        }
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let d = data.as_ref().ok_or("global store data")?;
        let vals = self.vgprs(inst, d, n)?;
        let (memory, root) = if a.param == MEMORY {
            (Some(self.global_address(inst)?), None)
        } else {
            let mut off = self.s.m.i32(a.offset);
            if let Some(d) = a.dynamic {
                let d = self.i32_of(d)?;
                off = self.s.m.bin(c, Bin::Add, off, d);
            }
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            match self.root_address(&src) {
                Some((lo, hi)) => {
                    // As `raw_view`, looked up inside the guard below.
                    let (lo, hi) = self.add64(lo, hi, off, false);
                    let mask = self.s.m.i32(0xFFFF);
                    (Some((lo, self.s.m.bin(self.cur, Bin::And, hi, mask))), None)
                }
                None => {
                    let h = self.handle((b, i), &src, false, (11 | 0x1000, 0))?;
                    let off = match self.root_view_offset(&src) {
                        Some(at) => self.s.m.bin(self.cur, Bin::Add, off, at),
                        None => off,
                    };
                    (None, Some((h, off)))
                }
            }
        };
        let (iu, void, i32t) = (self.s.m.undef(self.s.t.i32), self.s.t.void, self.s.t.i32);
        // Title memory is looked up (and its pages marked written) only in
        // the lanes that write.
        let at = self.begin_guarded(guard);
        let (h, off) = match (memory, root) {
            (Some((lo, hi)), _) => self.memory(lo, hi, 4 * u32::from(n), true, None),
            (_, Some(r)) => r,
            _ => unreachable!(),
        };
        if name.starts_with("global_store_") {
            let mut args = vec![h, off, iu];
            args.extend(&vals);
            args.resize(3 + 4, iu);
            args.push(self.s.m.i8((1u8 << n) - 1));
            args.push(self.s.m.i32(4));
            self.s.op(
                self.cur,
                "rawBufferStore.i32",
                Attr::NoUnwind,
                void,
                140,
                &args,
            );
            self.mark_view_write(h, off, 4 * u32::from(n), None);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let ops = ops.expect("an atomic");
        let mut x = vals[0];
        let c = self.cur;
        if name == "global_atomic_sub" {
            let zero = self.s.m.i32(0);
            x = self.s.m.bin(c, Bin::Sub, zero, x);
        }
        let negative = {
            let zero = self.s.m.i32(0);
            self.s.m.cmp(c, pred::SLT, x, zero)
        };
        let returned = match *dst {
            Some(Operand::Vector { reg, .. }) if cache.glc => Some(reg),
            _ => None,
        };
        let mut first = None;
        for (k, &(op, neutral)) in ops.iter().enumerate() {
            let value = if ops.len() == 1 {
                x
            } else {
                let n = self.s.m.i32(neutral);
                // The first serves positive values, the second negative ones.
                if k == 0 {
                    self.s.m.select(self.cur, negative, n, x)
                } else {
                    self.s.m.select(self.cur, negative, x, n)
                }
            };
            let op = self.s.m.i32(op);
            let r = self.s.op(
                self.cur,
                "atomicBinOp.i32",
                Attr::NoUnwind,
                i32t,
                78,
                &[h, op, off, iu, iu, value],
            );
            first.get_or_insert(r);
        }
        self.mark_view_write(h, off, 4, None);
        let zero = self.s.m.i32(0);
        let r = self.end_guarded(at, &[(first.expect("an atomic"), zero)])[0];
        if let Some(reg) = returned {
            self.write(inst, Reg::V(reg), r, guard);
        }
        Ok(())
    }

    /// Whether an earlier whole-wave section left `v` the same in every lane
    /// (a reduction, a constant or one lane's value), so any lane read of it is `v`.
    fn section_uniform(&self, v: Value) -> bool {
        use super::wave::Shape;
        self.section_out.contains(&v)
            && self.sections.values().flat_map(|m| m.values()).any(|sec| {
                matches!(
                    sec.shapes.get(&v),
                    Some(Shape::K(_) | Shape::Active(..) | Shape::Half(..) | Shape::ReadLane(..))
                )
            })
    }

    /// Finds the whole-wave sections and the values they leave.
    fn find_sections(&mut self) {
        // Lockstep runs whole-wave code literally: every thread is there (`lockstep.rs`).
        if self.x.st.lockstep {
            return;
        }
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    // A passthrough step's results are lowered as themselves, not made from
                    // the simulation.
                    if super::wave::passthrough(&p.blocks[b].insts[i]) {
                        continue;
                    }
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                // What a passthrough step reads leaves the section: it is lowered from it.
                let passthrough = super::wave::passthrough(inst);
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && (passthrough || !(sb == b && i > start && i < end))
                    {
                        self.section_out.insert(r);
                        if inst.name() == "v_readlane_b32"
                            && let Fields::Valu(v) = &inst.fields
                            && let [Operand::Vector { reg, .. }, at] = v.src.as_slice()
                            && inst.read(Reg::V(*reg)) == Some(r)
                            && let Some(lane) = crate::resource::lane_of(*at)
                        {
                            self.section_lane_reads.insert((r, lane));
                        }
                    }
                }
            }
        }
        let more: Vec<(Value, u32)> = self
            .sections
            .values()
            .flat_map(|m| m.values())
            .flat_map(|sec| sec.shapes.values().chain(sec.lane_shapes.values()))
            .filter_map(|sh| match *sh {
                super::wave::Shape::ReadLane(r, lane) if inside.contains_key(&r) => Some((r, lane)),
                _ => None,
            })
            .collect();
        self.section_lane_reads.extend(more);
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    fn section_inst(&mut self, b: usize, i: usize, start: usize) -> Result<(), Stop> {
        let inst = &self.x.p.blocks[b].insts[i];
        if super::wave::passthrough(inst) {
            return self.inst(b, i);
        }
        if !super::wave::register_only(inst) {
            return Err(unsupported(inst.machine.offset, inst.name()));
        }
        // The first step the simulation does not model names the refusal.
        let sec = &self.sections[&b][&start];
        let unknown: Why = sec.why.unwrap_or(super::wave::NO_SHAPE);
        for &(_, w) in &inst.writes {
            let shape = self.sections[&b][&start].shapes.get(&w).copied();
            let val = if self.section_out.contains(&w) {
                match shape.ok_or(unknown).and_then(|sh| self.shape(sh)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(unknown)
            };
            self.vals.insert(w, val);
            if self.x.st.masks.contains_key(&w) && shape.is_none() {
                self.lane_fail.insert(w, unknown);
            }
            if self.section_out.contains(&w) && shape.is_none() {
                let lanes: Vec<(u32, super::wave::Shape)> = self.sections[&b][&start]
                    .lane_shapes
                    .iter()
                    .filter_map(|(&(v, lane), &sh)| {
                        (v == w && self.section_lane_reads.contains(&(w, lane)))
                            .then_some((lane, sh))
                    })
                    .collect();
                for (lane, sh) in lanes {
                    let res = self.shape(sh);
                    self.section_lanes.insert((w, lane), res);
                }
            }
        }
        Ok(())
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    fn shape(&mut self, shape: super::wave::Shape) -> Result<V, Why> {
        use super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::ReadLane(v, _) if self.section_uniform(v) => return self.i32_of(v),
            Shape::ReadLane(v, lane) if self.section_lanes.contains_key(&(v, lane)) => {
                return self.section_lanes[&(v, lane)];
            }
            Shape::ReadLane(v, lane) => {
                self.cross_lane(false, outside_compute!("v_readlane_b32 of a constant lane"))?;
                let x = self.i32_of(v)?;
                let lane = self.s.m.i32(lane);
                return self.wide_read_lane(x, lane);
            }
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
            Shape::Half(_, src, mask, ident, hi) => {
                // The high half is a console lane's: compute only.
                if self.mask_data != MaskData::Console {
                    return Err("a 16-bit wave reduction outside compute");
                }
                if self.halves {
                    return Err("a 16-bit wave reduction across the host's waves");
                }
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                let x = self.s.m.select(c, on, x, id);
                self.s.flags |= flags::WAVE_OPS;
                // shortcut: the f16 sum is taken in f32 and rounded once; the
                // console rounds each step, in its own order.
                let low = self.s.m.i32(0xFFFF);
                let h = self.s.m.bin(c, Bin::And, x, low);
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
                let (sum, signed) = (self.s.m.i8(0), self.s.m.i8(0));
                let r = self.s.op(
                    c,
                    "waveActiveOp.f32",
                    Attr::NoUnwind,
                    self.s.t.f32,
                    119,
                    &[f, sum, signed],
                );
                let r = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
                let r = self.s.m.bin(c, Bin::And, r, low);
                // Lane `hi` runs the section on the console even where it is
                // elsewhere on the host; there it holds the identity.
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let b = self
                    .s
                    .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
                let word = self.s.m.extract(c, b, hi / 32);
                let (sh, one, z) = (self.s.m.i32(hi % 32), self.s.m.i32(1), self.s.m.i32(0));
                let bit = self.s.m.bin(c, Bin::LShr, word, sh);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                let there = self.s.m.cmp(c, pred::NE, bit, z);
                let lane = self.s.m.i32(hi);
                let at = self.read_lane(x, lane);
                let top = self.s.m.select(c, there, at, id);
                let high = self.s.m.i32(0xFFFF_0000);
                let top = self.s.m.bin(c, Bin::And, top, high);
                return Ok(self.s.m.bin(c, Bin::Or, top, r));
            }
        };
        // A mesh program's lanes are numbered as the console's, its host
        // waves are not the console's: only a minimum or maximum (the loop
        // that serves each distinct value once) means the same on any wave.
        if self.mask_data == MaskData::None
            && (prefix.is_some()
                || !matches!(
                    op,
                    Op::MinU | Op::MaxU | Op::MinI | Op::MaxI | Op::MinF | Op::MaxF
                ))
        {
            return Err("a wave reduction in a mesh shader");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        let float = matches!(op, Op::AddF | Op::MulF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let mut half_prefix = None;
        let r = match (op, prefix) {
            (_, Some(inclusive)) => {
                let (kind, bin) = match op {
                    Op::AddU | Op::AddF => (0, Bin::Add),
                    Op::MulF => (1, Bin::Mul),
                    _ => return Err("a wave prefix of an operation other than a sum or product"),
                };
                let args = [arg, self.s.m.i8(kind), self.s.m.i8(u8::from(!float))];
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &args,
                );
                if self.halves {
                    let total = self.s.op(
                        c,
                        &format!("waveActiveOp.{sfx}"),
                        Attr::NoUnwind,
                        ty,
                        119,
                        &args,
                    );
                    half_prefix = Some((bin, total));
                }
                if inclusive {
                    self.s.m.bin(c, bin, r, arg)
                } else {
                    r
                }
            }
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MulF => (1, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        let r = if self.halves {
            self.halves_reduce(op, r, half_prefix)?
        } else {
            r
        };
        Ok(if float { self.as_int(r) } else { r })
    }

    fn run(mut self) -> Result<Lowered, Stop> {
        if let Some(why) = self.refuse {
            return Err(unsupported(0, why));
        }
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Loop headers count iterations (`LOOP_LIMIT`).
        for &b in &order {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.cur = self.s.entry;
        self.find_sections();
        self.inputs()?;
        if p.thread_counts {
            self.launched();
        }
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        }
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
            }
            let mut i = 0;
            self.mode16 = self.round16[b];
            while i < p.blocks[b].insts.len() {
                if let Some(&lanes) = self.x.st.discards.get(&(b, i)) {
                    // A pixel's discard: the lanes stay on as helpers.
                    let at = self.cur;
                    let offset = p.blocks[b].insts[i].machine.offset;
                    let kill = self
                        .boolean(lanes, at)
                        .map_err(|w| unsupported(offset, w))?;
                    self.s
                        .op(at, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
                }
                if self.split_barriers.contains(&(b, i)) {
                    self.group_barrier();
                }
                self.inst(b, i)?;
                self.mode16 = round16_after(&p.blocks[b].insts[i], self.mode16);
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    for j in i + 1..end {
                        self.section_inst(b, j, i)?;
                    }
                    i = end;
                    continue;
                }
                i += 1;
            }
            if self.split_barriers.contains(&(b, p.blocks[b].insts.len())) {
                self.group_barrier();
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            self.tail[b] = Some(self.cur);
            self.exit(b)?;
        }
        self.fill_phis()?;
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let root_uavs = self.s.root_uavs.iter().map(|r| r.0).collect();
        let memory = self.s.memory.is_some();
        let memory_writes = self.s.memory_dirty.is_some();
        let view_addresses = self.s.view_addresses.is_some();
        let cmp_samplers = self.s.cmp_samplers;
        let splits = self
            .split_inputs
            .iter()
            .map(|(&(orig, _), &new)| (self.s.inputs[orig].index, self.s.inputs[new].index))
            .collect();
        for u in &self.x.native.used {
            if u.root != RootType::Table {
                continue;
            }
            let slot = self
                .x
                .native
                .layout
                .iter()
                .find(|rp| rp.root == RootType::Table && rp.user_data == u.user_data)
                .map_or(u.slot, |rp| rp.slot);
            let step_dwords = if u.kind == ResourceType::Sampler { 4 } else { 8 };
            let end_dwords = u.table_offset_dwords.saturating_add(u32::from(u.dwords).max(1));
            let count = end_dwords.div_ceil(step_dwords);
            let entry = self.table_entries.entry(slot).or_insert(0);
            if count > *entry {
                *entry = count;
            }
        }
        let table_entries = self.table_entries.into_iter().collect();
        Ok(Lowered {
            splits,
            table_entries,
            cmp_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            memory,
            memory_writes,
            view_addresses,
            heap: self.heap[0],
            heap_writes: self.heap[1],
            view_marks: self.s.view_writes.is_some(),
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            heap_handles: self.s.m.written_calls("dx.op.createHandleFromHeap") as u32,
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            #[cfg(test)]
            rovs: self.s.flags & flags::ROVS != 0,
            #[cfg(test)]
            split_barriers: self.split_barriers.len(),
            #[cfg(test)]
            wave_size: self.s.wave_size,
            #[cfg(test)]
            barriers: self.s.m.written_calls("dx.op.barrier"),
            #[cfg(test)]
            exchanges: self.s.xchg_used(),
            #[cfg(test)]
            texture_calls: [
                self.s.m.written_call_args("dx.op.sample"),
                self.s.m.written_call_args("dx.op.textureGather"),
            ]
            .concat(),
            #[cfg(test)]
            wave_prefix_calls: self.s.m.written_call_args("dx.op.wavePrefixOp"),
            #[cfg(test)]
            wave_read_lane_calls: self.s.m.written_call_args("dx.op.waveReadLaneAt"),
            #[cfg(test)]
            wave_active_calls: self.s.m.written_call_args("dx.op.waveActiveOp"),
            #[cfg(test)]
            float_multiplies: self.s.m.written_bin_ops(self.s.t.f32, Bin::Mul),
            #[cfg(test)]
            output_stores: [
                self.s.m.written_call_places("dx.op.storeOutput."),
                self.s.m.written_call_places("dx.op.storeVertexOutput."),
                self.s.m.written_call_places("dx.op.storePrimitiveOutput."),
            ]
            .concat(),
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
        })
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    /// [`Options::thread_counts`](super::Options): each lane's bit of EXEC
    /// on entry, set where the thread is one the dispatch launches: below
    /// the thread count in each dimension, which the host passes in
    /// user-data dwords past the root signature's own.
    fn launched(&mut self) {
        let p = self.x.p;
        let (Some(&(_, exec)), Some(threads)) = (
            p.inputs.iter().find(|i| i.0 == Reg::S(EXEC)),
            self.s.threads,
        ) else {
            return;
        };
        let layout = &self.x.native.layout;
        let free = layout.iter().map(|rp| u32::from(rp.user_data) + rp.sgprs).max().unwrap_or(0);
        let e = self.s.entry;
        let mut on = self.s.m.bool(true);
        // A dimension of one thread a group has no partial group.
        for d in (0..3u8).filter(|&d| threads[usize::from(d)] > 1) {
            let (Val::I32(t), Val::I32(g)) =
                (self.thread_id(d), self.compute_id("groupId.i32", 94, d))
            else {
                unreachable!("compute ids are numbers")
            };
            let n = self.s.m.i32(threads[usize::from(d)]);
            let at = self.s.m.bin(e, Bin::Mul, g, n);
            let at = self.s.m.bin(e, Bin::Add, at, t);
            let count = self.user_at(UserData {
                index: free + u32::from(d),
                slot: super::DISPATCH_THREADS,
                root: RootType::Constants,
                dword: d.into(),
            });
            let below = self.s.m.cmp(e, pred::IULT, at, count);
            on = self.s.m.bin(e, Bin::And, on, below);
        }
        self.lane_bits.insert(exec, on);
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        let p = self.x.p;
        let interp_uses = std::cell::LazyCell::new(|| InterpUses::new(p));
        let prim_uses = std::cell::LazyCell::new(|| crate::io::PrimMaskUses::new(p));
        for &(reg, v) in &p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Guess (docs/native-host.md, "LDS instructions addressed by M0"): M0 a
                    // compute program reads before writing holds LDS address 0 (observed: a
                    // `ds_append` counter at 0 + offset that the program zeroes and reads there).
                    None if n == crate::ir::M0 && self.x.native.stage == HwStage::Cs => {
                        Val::I32(self.s.m.i32(0))
                    }
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    // The primitive mask (pixel), which console code only
                    // moves to M0 for its interpolation (spilled and reloaded
                    // on the way); the host interpolates by itself, so the
                    // value goes unread.
                    None if self.s.pixel
                        && ((self.x.io.initialized_sgprs == Some(n + 2)
                            && (!self.guessed || prim_uses.is_prim_mask_only(v, true)))
                            || interp_uses.is_interp_m0_only(v)) =>
                    {
                        Val::I32(self.s.m.i32(0))
                    }
                    // The ordered section's collision id, and (`operand`) the
                    // id of the last wave that left it: 0, so no wave waits;
                    // the host's rasterizer-ordered views order the accesses.
                    None if self.pops && self.x.io.initialized_sgprs == Some(n + 1) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { .. } => return Val::Missing("barycentrics read directly"),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::Ancillary => {
                let Some(e) = self.s.inputs.iter_mut().find(|e| e.kind == 12) else {
                    return Val::Missing("a pixel system input the signature lacks");
                };
                e.usage |= 1;
                // shortcut: only sample index bits 8..11 of the pixel ancillary
                // input are populated (face/prim-mask/slice bits stay zero);
                // upgrade if a pixel shader reads non-sample ancillary bits.
                let entry = self.s.entry;
                let idx =
                    self.s
                        .op(entry, "sampleIndex.i32", Attr::ReadNone, self.s.t.i32, 90, &[]);
                let k8 = self.s.m.i32(8);
                return Val::I32(self.s.m.bin(entry, Bin::Shl, idx, k8));
            }
            Input::SampleCoverage => {
                let Some(e) = self.s.inputs.iter_mut().find(|e| e.kind == 14) else {
                    return Val::Missing("a pixel system input the signature lacks");
                };
                e.usage |= 1;
                let entry = self.s.entry;
                return Val::I32(
                    self.s
                        .op(entry, "coverage.i32", Attr::ReadNone, self.s.t.i32, 91, &[]),
                );
            }
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return self.compute_id("groupId.i32", 94, c),
            Input::GroupInfo => {
                // Guess (RADV's reading of TG_SIZE): bits 0-5 the group's
                // wave count, 6-11 this wave's index in it; the rest 0.
                let f = if self.console_lanes { self.console_flat() } else { self.mesh_thread() };
                let lanes: u32 = self.s.threads.expect("compute").iter().product();
                let wave = self.wave();
                let e = self.s.entry;
                let (shift, k6, n) = (
                    self.s.m.i32(wave.trailing_zeros()),
                    self.s.m.i32(6),
                    self.s.m.i32(lanes.div_ceil(wave)),
                );
                let w = self.s.m.bin(e, Bin::LShr, f, shift);
                let w = self.s.m.bin(e, Bin::Shl, w, k6);
                return Val::I32(self.s.m.bin(e, Bin::Or, w, n));
            }
            Input::MeshGroup => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// A component of a compute thread or group id, read in the entry block.
    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// The vertex id input (debugging).
    fn get_input_vertex_id(&mut self) -> Result<Val, Why> {
        Ok(self.input(Input::VertexId))
    }

    /// What lane `lane` of VGPR value `v` holds when every path to it fills that lane with
    /// `v_writelane_b32` of an SGPR or constant (an SGPR spill): the value written, or a phi
    /// of them at a VGPR join. With `make` unset only checks (`seen`: joins on the way).
    fn spill_slot(
        &mut self,
        v: Value,
        lane: u32,
        seen: &mut Vec<Value>,
        make: bool,
    ) -> Result<V, Why> {
        const NOT: Why = "a VGPR lane no SGPR spill fills";
        let p = self.x.p;
        if let Some(&(b, i)) = self.phi_at.get(&v) {
            if let Some(&x) = self.spill_phis.get(&(v, lane)) {
                return Ok(x);
            }
            if !make {
                if seen.contains(&v) {
                    return Ok(self.s.m.i32(0));
                }
                seen.push(v);
                for &a in &p.blocks[b].phis[i].args {
                    self.spill_slot(a, lane, seen, false)?;
                }
                return Ok(self.s.m.i32(0));
            }
            let at = self.block[b].ok_or(NOT)?;
            let x = self.s.m.phi(at, self.s.t.i32);
            self.spill_phis.insert((v, lane), x);
            self.pending_spills.push((x, b, i, lane));
            return Ok(x);
        }
        let inst = self.def_inst(v).ok_or(NOT)?;
        let Fields::Valu(w) = &inst.fields else { return Err(NOT) };
        let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
            return Err(NOT);
        };
        if inst.name() != "v_writelane_b32" || w.sdwa.is_some() || w.dpp.is_some() {
            return Err(NOT);
        }
        if crate::resource::lane_of(*at).ok_or(NOT)? != lane {
            return self.spill_slot(inst.read(Reg::V(reg)).ok_or(NOT)?, lane, seen, make);
        }
        let c = match *value {
            Operand::Scalar { code: NULL, .. } => 0,
            Operand::Scalar { code, .. } => {
                let x = inst.read(Reg::S(code)).ok_or(NOT)?;
                // Values made by now (on a back edge, by the time phis are filled).
                return match self.vals.get(&x) {
                    Some(&Val::I32(x)) => Ok(x),
                    None if !make => Ok(self.s.m.i32(0)),
                    _ => Err(NOT),
                };
            }
            Operand::Const(c) => crate::resource::inline(c).ok_or(NOT)?,
            Operand::Literal(x) => x,
            _ => return Err(NOT),
        };
        Ok(self.s.m.i32(c))
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let found = self.vals.get(&v).copied();
        // Most values have a DXIL form, which is the answer whatever they
        // are; only one without needs to know whether it is a lane mask.
        if let Some(val @ (Val::I32(_) | Val::Bit(_))) = found {
            return Ok(val);
        }
        if self.mask_ok && (self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v)) {
            return self.mask_dword(v).map(Val::I32);
        }
        if let (Some(Val::Missing(_)), Some(&u)) = (found, self.uniform_scc.get(&v)) {
            let at = match self.def_block.get(&v) {
                None => self.s.entry,
                Some(&b) if b == self.cur_ir => self.cur,
                Some(&b) => self.tail[b].ok_or("SCC written after its use")?,
            };
            let any = Val::Bit(self.atom(Atom::Lane(u), at)?);
            self.vals.insert(v, any);
            return Ok(any);
        }
        if let (Some(Val::Missing(_)), Some((m, not, fixed))) =
            (found, self.uniform_bits.get(&v).cloned())
        {
            let at = match self.def_block.get(&v) {
                None => self.s.entry,
                Some(&b) if b == self.cur_ir => self.cur,
                Some(&b) => self.tail[b].ok_or("SCC written after its use")?,
            };
            let mut x = self.boolean_given(m, at, &fixed)?;
            if not {
                let t = self.s.m.bool(true);
                x = self.s.m.bin(at, Bin::Xor, x, t);
            }
            self.vals.insert(v, Val::Bit(x));
            return Ok(Val::Bit(x));
        }
        if let (Some(Val::Missing(_)), Some(&(lo, hi))) = (found, self.mask_scc.get(&v)) {
            // SCC of a mask operation read as data: some lane in the result.
            // Emit in the defining block so caching in `vals` dominates all uses,
            // and allow reading the mask dwords even when the consumer does not
            // itself set `mask_ok` (e.g. `s_cselect`, `s_addc`).
            let at = match self.def_block.get(&v) {
                None => self.s.entry,
                Some(&b) if b == self.cur_ir => self.cur,
                Some(&b) => self.tail[b].ok_or("SCC written after its use")?,
            };
            let was_ok = self.mask_ok;
            self.mask_ok = true;
            let res = (|| -> Result<V, Why> {
                let mut word = self.i32_of(lo)?;
                if let Some(hi) = hi {
                    let h = self.i32_of(hi)?;
                    word = self.s.m.bin(at, Bin::Or, word, h);
                }
                let z = self.s.m.i32(0);
                Ok(self.s.m.cmp(at, pred::NE, word, z))
            })();
            self.mask_ok = was_ok;
            let any = Val::Bit(res?);
            self.vals.insert(v, any);
            return Ok(any);
        }
        if let Some(Val::Missing(w)) = found {
            return Err(w);
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        Err("a value defined after its use")
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => Val::I32(self.user_dword(param, dword)),
            // Its address, where root views are title memory. Not the vertex buffer table of a
            // guessed vertex program ([`graphics::VERTEX_TABLE`]): no root argument holds its
            // address, and the vertex fetch is found by tracing loads through it.
            RootType::Descriptor if self.root_memory && rp.slot != graphics::VERTEX_TABLE => {
                Val::I32(self.user_dword(param, dword))
            }
            _ => Val::Missing("a root descriptor"),
        }
    }

    /// User-data dword `dword` of root parameter `param` as the host passes
    /// it (for a root descriptor: its GPU address, low dword first).
    fn user_dword(&mut self, param: usize, dword: u32) -> V {
        let rp = self.x.native.layout[param];
        let index = u32::from(rp.user_data) + dword;
        self.user_at(UserData {
            index,
            slot: rp.slot,
            root: rp.root,
            dword,
        })
    }

    /// A root SRV or UAV (`uav`) of root parameter `param`, in a program
    /// that does not read title memory: the resource-heap view the host
    /// makes of the host copy holding the buffer, whose index comes in the
    /// root descriptor's own first user-data dword (module notes, "Root SRVs
    /// and root UAVs"); offsets from the root address move by
    /// [`Lower::root_view_offset`]. Host root views have no bounds; a view's
    /// reads past the copy give 0 and its writes there are dropped.
    fn root_view(&mut self, param: usize, uav: bool) -> V {
        let rp = self.x.native.layout[param];
        // The descriptor's two user-data dwords hold its address on the
        // console; such a program never reads them as data (`user`).
        let index = u32::from(rp.user_data);
        if let Some(&h) = self.root_views.get(&index) {
            return h;
        }
        let i = self.user_at(UserData {
            index,
            slot: rp.slot,
            root: RootType::Descriptor,
            dword: if uav {
                super::ROOT_UAV_VIEW
            } else {
                super::ROOT_SRV_VIEW
            },
        });
        let props = if uav { (11 | 0x1000, 0) } else { (11, 0) };
        let h = self.s.heap_handle(self.s.entry, i, false, false, props);
        self.root_views.insert(index, h);
        h
    }

    /// For a root SRV or UAV read through the view of its host copy
    /// ([`Lower::root_view`]): the root address's byte offset in that view,
    /// from the descriptor's second user-data dword. None for other sources.
    fn root_view_offset(&mut self, src: &Source) -> Option<V> {
        let Source::Built { address, .. } = src else {
            return None;
        };
        let rp = *self.x.native.layout.get(address.param)?;
        let cbv = self
            .used_kind(src)
            .is_some_and(|u| u.kind == ResourceType::Cbv);
        if self.root_memory || rp.root != RootType::Descriptor || cbv {
            return None;
        }
        Some(self.user_at(UserData {
            index: u32::from(rp.user_data) + 1,
            slot: rp.slot,
            root: RootType::Descriptor,
            dword: super::ROOT_VIEW_OFFSET,
        }))
    }

    /// User-data dword `u.index`, which the host fills as `u` says, loaded
    /// in the entry block.
    fn user_at(&mut self, u: UserData) -> V {
        let index = u.index;
        self.user_reads.insert(index, u);
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        self.s.m.extract(entry, r, index % 4)
    }

    /// In a program that uses title memory, the address of the root SRV or
    /// UAV a descriptor `src` is built around: `(lo, hi)`, `hi` cut to bits
    /// 32-47. None for other descriptors and programs.
    fn root_address(&mut self, src: &Source) -> Option<(V, V)> {
        let Source::Built { address, .. } = src else {
            return None;
        };
        let rp = self.x.native.layout.get(address.param)?;
        let cbv = self
            .used_kind(src)
            .is_some_and(|u| u.kind == ResourceType::Cbv);
        if !self.root_memory || rp.root != RootType::Descriptor || cbv {
            return None;
        }
        let lo = self.user_dword(address.param, 0);
        let hi = self.user_dword(address.param, 1);
        let mask = self.s.m.i32(0xFFFF);
        Some((lo, self.s.m.bin(self.s.entry, Bin::And, hi, mask)))
    }

    /// A raw view `src` at byte offset `off` (`props` as [`Self::handle`]
    /// takes them): its handle and the offset in it. A root view of a
    /// program that uses title memory is title memory at the root address
    /// plus the offset ([`Self::memory`]: `bytes` accessed, `write`, the
    /// `active` lanes), so the access is bounds-checked (host root views
    /// have no bounds) and sees the copies the program's other accesses see.
    #[allow(clippy::too_many_arguments)]
    fn raw_view(
        &mut self,
        site: Site,
        src: &Source,
        off: V,
        bytes: u32,
        props: (u32, u32),
        write: bool,
        active: Option<V>,
    ) -> Result<(V, V), Why> {
        let Some((lo, hi)) = self.root_address(src) else {
            let h = self.handle(site, src, false, props)?;
            let off = match self.root_view_offset(src) {
                Some(at) => self.s.m.bin(self.cur, Bin::Add, off, at),
                None => off,
            };
            return Ok((h, off));
        };
        // The hardware adds the 32-bit offset to the 48-bit base.
        let (lo, hi) = self.add64(lo, hi, off, false);
        let mask = self.s.m.i32(0xFFFF);
        let hi = self.s.m.bin(self.cur, Bin::And, hi, mask);
        Ok(self.memory(lo, hi, bytes, write, active))
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        // `Atom::Quad` evaluates to `true` as an EXEC derivative shortcut (`atom`), not the
        // mask's quad-expanded bits: route any `s_wqm`-derived mask through `mask_number`.
        let has_quad = self
            .x
            .st
            .bools
            .support(m)
            .iter()
            .any(|a| matches!(a, Atom::Quad(_)));
        if (!self.x.st.active.contains(&lo) || has_quad) && !self.ballots.contains_key(&lo) {
            let xy = self.mask_number(lo)?;
            self.ballots.insert(lo, xy);
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => xy,
            None => {
                let at = match self.def_block.get(&lo) {
                    // Program inputs (EXEC on entry).
                    None => self.s.entry,
                    Some(&b) if b == self.cur_ir => self.cur,
                    Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                };
                if !self.balloting.insert(lo) {
                    return Err("a lane mask whose lanes depend on its own value");
                }
                let bit = self.boolean(m, at);
                self.balloting.remove(&lo);
                let bit = bit?;
                // Program inputs (EXEC on entry): the entry block.
                let ir = self.def_block.get(&lo).copied().unwrap_or(0);
                let xy = self.wide_ballot(at, ir, bit)?;
                self.ballots.insert(lo, xy);
                xy
            }
        };
        Ok(if high { y } else { x })
    }

    /// Both dwords of a mask holding lanes that do not reach the block
    /// writing it (a complement, a copy made where EXEC is not known), so
    /// no ballot there gives it: with console lanes, the scalar operation
    /// that writes it applied to its operands' numbers, as the console does;
    /// with host lanes (`MaskData::Host`), only wave-size-safe bitwise ops
    /// and selects on host ballots.
    fn mask_number(&mut self, lo: Value) -> Result<(V, V), Why> {
        // Each way out names where the mask comes from, so a refusal log says which is missing.
        const STAGE: Why = "a lane mask holding lanes elsewhere read as data: in a mesh, hull or \
                            geometry program";
        const INPUT: Why = "a lane mask holding lanes elsewhere read as data: a program input";
        const VECTOR: Why = "a lane mask holding lanes elsewhere read as data: not written by a \
                             scalar ALU instruction";
        const WIDTH: Why = "a lane mask holding lanes elsewhere read as data: written by a scalar \
                            instruction of the other width";
        const SAVED: Why = "a lane mask holding lanes elsewhere read as data: the EXEC a saveexec \
                            keeps, or a mask written with EXEC";
        const SHIFT: Why =
            "a lane mask holding lanes elsewhere read as data: shifted on the host's lanes";
        const OPERAND: Why = "a lane mask holding lanes elsewhere read as data: an operand no SGPR \
                              or constant gives";
        const OP: Why =
            "a lane mask holding lanes elsewhere read as data: a scalar operation not modelled";
        if self.mask_data == MaskData::None {
            return Err(STAGE);
        }
        let p = self.x.p;
        // A register the compiler reuses: a mask on one path, data on another.
        if let Some(&(b, i)) = self.phi_at.get(&lo)
            && p.blocks[b].phis[i]
                .args
                .iter()
                .any(|a| !self.x.st.masks.contains_key(a))
        {
            return Err("a register joining a lane mask and data, read as data");
        }
        let &b = self.def_block.get(&lo).ok_or(INPUT)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))
            .ok_or(INPUT)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return Err(VECTOR);
        };
        let wave64 = self.x.native.wave64;
        let width = if wave64 { "_b64" } else { "_b32" };
        let op = inst
            .name()
            .strip_prefix("s_")
            .and_then(|n| n.strip_suffix(width))
            .ok_or(WIDTH)?;
        let (save, f_op) = match (op.strip_suffix("_saveexec"), op.strip_suffix("_wrexec")) {
            (Some(f), _) => (Some(true), f),
            (_, Some(f)) => (Some(false), f),
            _ => (None, op),
        };
        let writes_dst = inst.write(Reg::S(*code)) == Some(lo);
        let writes_exec = inst.write(Reg::S(EXEC)) == Some(lo);
        // (EXEC written besides the destination: a saveexec's new mask, modelled on console lanes.)
        if !writes_dst && !writes_exec
            || (save.is_none() || self.mask_data != MaskData::Console)
                && *code != EXEC
                && inst.write(Reg::S(EXEC)).is_some()
        {
            return Err(SAVED);
        }
        if save.is_none()
            && self.mask_data != MaskData::Console
            && matches!(
                op,
                "lshl" | "lshr" | "bitset0" | "bitset1" | "bfm" | "brev" | "quadmask" | "wqm"
            )
        {
            return Err(SHIFT);
        }
        let at = if b == self.cur_ir {
            self.cur
        } else {
            self.tail[b].ok_or("a lane mask written after its use")?
        };
        let was_ok = self.mask_ok;
        self.mask_ok = true;
        let result = (|| -> Result<(V, V), Why> {
            // Dword `h` of source `k` (inline constants sign-extend).
            let word = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
                let x = match src.get(k) {
                    Some(&Operand::Scalar { code, .. }) if code != NULL => {
                        let v = inst.read(Reg::S(code + h)).ok_or(OPERAND)?;
                        return l.i32_of(v);
                    }
                    Some(&Operand::Const(c)) => {
                        let x = crate::resource::inline(c).ok_or(OPERAND)?;
                        match h {
                            0 => x,
                            _ if (x as i32) < 0 => u32::MAX,
                            _ => 0,
                        }
                    }
                    Some(&Operand::Literal(x)) if h == 0 => x,
                    Some(Operand::Literal(_)) => 0,
                    _ => return Err(OPERAND),
                };
                Ok(l.s.m.i32(x))
            };
            if save == Some(true) && !writes_exec {
                // `s_*_saveexec` writes the incoming `EXEC` into `sdst`.
                let lo_e = self.i32_of(inst.read(Reg::S(EXEC)).ok_or(OPERAND)?)?;
                let hi_e = if wave64 {
                    self.i32_of(inst.read(Reg::S(EXEC + 1)).ok_or(OPERAND)?)?
                } else {
                    self.s.m.i32(0)
                };
                return Ok((lo_e, hi_e));
            }
            if save.is_none() && matches!(op, "lshl" | "lshr") {
                // The shift count is the low 6 (5) bits of a plain number.
                let n = word(self, 1, 0)?;
                let a = word(self, 0, 0)?;
                let i64t = self.s.m.int(64);
                let k32 = self.s.m.int_const(i64t, 32);
                let kmask = self.s.m.i32(if wave64 { 63 } else { 31 });
                let n = self.s.m.bin(at, Bin::And, n, kmask);
                let f = if op == "lshl" { Bin::Shl } else { Bin::LShr };
                if !wave64 {
                    return Ok((self.s.m.bin(at, f, a, n), self.s.m.i32(0)));
                }
                self.s.flags |= flags::INT64;
                let b = word(self, 0, 1)?;
                let a = self.s.m.cast(at, Cast::ZExt, a, i64t);
                let b = self.s.m.cast(at, Cast::ZExt, b, i64t);
                let b = self.s.m.bin(at, Bin::Shl, b, k32);
                let x = self.s.m.bin(at, Bin::Or, a, b);
                let n = self.s.m.cast(at, Cast::ZExt, n, i64t);
                let r = self.s.m.bin(at, f, x, n);
                let rl = self.s.m.cast(at, Cast::Trunc, r, self.s.t.i32);
                let rh = self.s.m.bin(at, Bin::LShr, r, k32);
                let rh = self.s.m.cast(at, Cast::Trunc, rh, self.s.t.i32);
                return Ok((rl, rh));
            }
            if save.is_none() && matches!(op, "bitset0" | "bitset1") {
                let old_lo = self.i32_of(inst.read(Reg::S(*code)).ok_or(OPERAND)?)?;
                let bit = word(self, 0, 0)?;
                let (k31, one, z, ones) = (
                    self.s.m.i32(31),
                    self.s.m.i32(1),
                    self.s.m.i32(0),
                    self.s.m.i32(u32::MAX),
                );
                let idx = self.s.m.bin(at, Bin::And, bit, k31);
                let mask = self.s.m.bin(at, Bin::Shl, one, idx);
                if !wave64 {
                    let r = if op == "bitset1" {
                        self.s.m.bin(at, Bin::Or, old_lo, mask)
                    } else {
                        let nmask = self.s.m.bin(at, Bin::Xor, mask, ones);
                        self.s.m.bin(at, Bin::And, old_lo, nmask)
                    };
                    return Ok((r, z));
                }
                let old_hi = self.i32_of(inst.read(Reg::S(*code + 1)).ok_or(OPERAND)?)?;
                let k32 = self.s.m.i32(32);
                let hi_bit = self.s.m.bin(at, Bin::And, bit, k32);
                let upper = self.s.m.cmp(at, pred::NE, hi_bit, z);
                let lo_mask = self.s.m.select(at, upper, z, mask);
                let hi_mask = self.s.m.select(at, upper, mask, z);
                return Ok(if op == "bitset1" {
                    (
                        self.s.m.bin(at, Bin::Or, old_lo, lo_mask),
                        self.s.m.bin(at, Bin::Or, old_hi, hi_mask),
                    )
                } else {
                    let nlo = self.s.m.bin(at, Bin::Xor, lo_mask, ones);
                    let nhi = self.s.m.bin(at, Bin::Xor, hi_mask, ones);
                    (
                        self.s.m.bin(at, Bin::And, old_lo, nlo),
                        self.s.m.bin(at, Bin::And, old_hi, nhi),
                    )
                });
            }
            if save.is_none() && op == "bfm" {
                let (w, o) = (word(self, 0, 0)?, word(self, 1, 0)?);
                if !wave64 {
                    let (k31, one, z) = (self.s.m.i32(31), self.s.m.i32(1), self.s.m.i32(0));
                    let w = self.s.m.bin(at, Bin::And, w, k31);
                    let o = self.s.m.bin(at, Bin::And, o, k31);
                    let m = self.s.m.bin(at, Bin::Shl, one, w);
                    let m = self.s.m.bin(at, Bin::Sub, m, one);
                    return Ok((self.s.m.bin(at, Bin::Shl, m, o), z));
                }
                let i64t = self.s.m.int(64);
                self.s.flags |= flags::INT64;
                let k63 = self.s.m.i32(63);
                let w = self.s.m.bin(at, Bin::And, w, k63);
                let w = self.s.m.cast(at, Cast::ZExt, w, i64t);
                let o = self.s.m.bin(at, Bin::And, o, k63);
                let o = self.s.m.cast(at, Cast::ZExt, o, i64t);
                let one = self.s.m.int_const(i64t, 1);
                let m = self.s.m.bin(at, Bin::Shl, one, w);
                let m = self.s.m.bin(at, Bin::Sub, m, one);
                let r = self.s.m.bin(at, Bin::Shl, m, o);
                let k32 = self.s.m.int_const(i64t, 32);
                let rl = self.s.m.cast(at, Cast::Trunc, r, self.s.t.i32);
                let rh = self.s.m.bin(at, Bin::LShr, r, k32);
                let rh = self.s.m.cast(at, Cast::Trunc, rh, self.s.t.i32);
                return Ok((rl, rh));
            }
            if save.is_none() && op == "brev" {
                let i32t = self.s.t.i32;
                let a = word(self, 0, 0)?;
                let ra = self.s.op(at, "unary.i32", Attr::ReadNone, i32t, 30, &[a]);
                if !wave64 {
                    return Ok((ra, self.s.m.i32(0)));
                }
                let b = word(self, 0, 1)?;
                let rb = self.s.op(at, "unary.i32", Attr::ReadNone, i32t, 30, &[b]);
                return Ok((rb, ra));
            }
            if save.is_none() && op == "quadmask" {
                let a = word(self, 0, 0)?;
                let low = self.quad_mask(at, a);
                if !wave64 {
                    return Ok((low, self.s.m.i32(0)));
                }
                let b = word(self, 0, 1)?;
                let high = self.quad_mask(at, b);
                let k8 = self.s.m.i32(8);
                let high = self.s.m.bin(at, Bin::Shl, high, k8);
                return Ok((self.s.m.bin(at, Bin::Or, low, high), self.s.m.i32(0)));
            }
            let mut out = [self.s.m.i32(0); 2];
            for h in 0..1 + u8::from(wave64) {
                let a = word(self, 0, h)?;
                let ones = self.s.m.i32(u32::MAX);
                out[usize::from(h)] = match f_op {
                    "mov" if save.is_none() => a,
                    "not" if save.is_none() => self.s.m.bin(at, Bin::Xor, a, ones),
                    "wqm" if save.is_none() => self.wqm_word(at, a),
                    "cselect" | "cmov" if save.is_none() => {
                        let c = if f_op == "cmov" {
                            let old = inst.read(Reg::S(*code + h)).ok_or(OPERAND)?;
                            self.i32_of(old)?
                        } else {
                            word(self, 1, h)?
                        };
                        let scc = inst.read(Reg::Scc).ok_or(OPERAND)?;
                        let cond = match self.get(scc)? {
                            Val::Bit(x) => x,
                            Val::I32(x) => {
                                let z = self.s.m.i32(0);
                                self.s.m.cmp(at, pred::NE, x, z)
                            }
                            Val::Missing(w) => return Err(w),
                        };
                        self.s.m.select(at, cond, a, c)
                    }
                    _ => {
                        // (operation, complement of src0, complement of src1, complement of result)
                        let (f, not_a, not_c, not_r) = match f_op {
                            "and" => (Bin::And, false, false, false),
                            "or" => (Bin::Or, false, false, false),
                            "xor" => (Bin::Xor, false, false, false),
                            "andn2" => (Bin::And, false, true, false),
                            "orn2" => (Bin::Or, false, true, false),
                            "andn1" => (Bin::And, true, false, false),
                            "orn1" => (Bin::Or, true, false, false),
                            "nand" => (Bin::And, false, false, true),
                            "nor" => (Bin::Or, false, false, true),
                            "xnor" => (Bin::Xor, false, false, true),
                            _ => return Err(OP),
                        };
                        let a = if not_a {
                            self.s.m.bin(at, Bin::Xor, a, ones)
                        } else {
                            a
                        };
                        let c = if save.is_some() {
                            let e = inst.read(Reg::S(EXEC + h)).ok_or(OPERAND)?;
                            self.i32_of(e)?
                        } else {
                            word(self, 1, h)?
                        };
                        let c = if not_c {
                            self.s.m.bin(at, Bin::Xor, c, ones)
                        } else {
                            c
                        };
                        let r = self.s.m.bin(at, f, a, c);
                        if not_r {
                            self.s.m.bin(at, Bin::Xor, r, ones)
                        } else {
                            r
                        }
                    }
                };
            }
            if self.mask_data == MaskData::Host
                && wave64
                && matches!(f_op, "not" | "orn2" | "orn1" | "nand" | "nor" | "xnor")
            {
                // On a 32-lane host wave, complementing sets bits 32..63 that
                // belong to no lane; clear the high dword when wave size <= 32.
                self.s.flags |= flags::WAVE_OPS;
                let lanes = self
                    .s
                    .op(at, "waveGetLaneCount", Attr::ReadNone, self.s.t.i32, 112, &[]);
                let (k32, z) = (self.s.m.i32(32), self.s.m.i32(0));
                let wide = self.s.m.cmp(at, pred::IUGT, lanes, k32);
                out[1] = self.s.m.select(at, wide, out[1], z);
            }
            Ok((out[0], out[1]))
        })();
        self.mask_ok = was_ok;
        result
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            // src_scc: SCC as 0 or 1.
            Operand::Const(253) => {
                let bit = self.scc_in(inst)?;
                Ok(self.s.m.cast(self.cur, Cast::ZExt, bit, self.s.t.i32))
            }
            // `src_pops_exiting_wave_id` (see `inputs`).
            Operand::Const(239) if self.pops => Ok(self.s.m.i32(0)),
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// GFX10 puts an inline f16 constant in the low half of the operand.
    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => {
                let h = [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)];
                Ok(self.s.m.i32(h))
            }
            _ => self.operand(inst, op),
        }
    }

    /// VGPR source `k` read for its low 16 bits alone, when only those are
    /// known (`low16`); the caller applies the modifiers and selects.
    fn low16_src(&self, inst: &Inst, k: usize) -> Option<V> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        match v.src.get(k) {
            Some(&Operand::Vector { reg, .. }) if v.dpp.is_none() => inst
                .read(Reg::V(reg))
                .and_then(|o| self.low16.get(&o).copied()),
            _ => None,
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let sel = Self::sdwa_src(inst, k);
                // A byte or word select within the low half.
                let low = matches!(sel, Some((0 | 1 | 4, _)));
                let x = match self.low16_src(inst, k).filter(|_| low) {
                    Some(x) => x,
                    None => self.operand(inst, &op)?,
                };
                Ok(match sel {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// A 32-bit denormal as the console's float mode reads it: a zero of its
    /// sign, decided on the bits (the host's own denormal mode varies).
    fn ftz(&mut self, f: V) -> V {
        let c = self.cur;
        let x = self.as_int(f);
        let (exp, sign, zero) = (
            self.s.m.i32(0x7F80_0000),
            self.s.m.i32(0x8000_0000),
            self.s.m.i32(0),
        );
        let e = self.s.m.bin(c, Bin::And, x, exp);
        let den = self.s.m.cmp(c, pred::EQ, e, zero);
        let s = self.s.m.bin(c, Bin::And, x, sign);
        let r = self.s.m.select(c, den, s, x);
        self.as_float(r)
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but of the operands with denormals read as zeros and with
    /// -0 ordered below +0 (RDNA2 ISA, `v_max_f32`), where the compare and
    /// select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let (a, b) = (self.ftz(a), self.ftz(b));
        self.max_min_flushed(max, a, b)
    }

    /// [`Self::max_min`] of operands without denormals.
    fn max_min_flushed(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as an f16 in the low bits, rounded as the program's MODE says at
    /// this point: toward zero where it set that for 16-bit results (the
    /// title does around its 16-bit interpolation), else to nearest even;
    /// f32 denormals read as zero; on the bits, where the host's own
    /// conversion rounds as it likes (toward zero on the RX 6700 XT and WARP).
    // shortcut: rounding toward +-infinity, or a mode not known at this
    // point (paths that set it differently join), rounds to nearest even.
    fn narrow16(&mut self, f: V) -> V {
        let rtz = self.mode16 == Some(3);
        let c = self.cur;
        let x = self.as_int(f);
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k13, k16, k31) = (k(self, 13), k(self, 16), k(self, 31));
        let t = self.s.m.bin(c, Bin::LShr, x, k16);
        let m = k(self, 0x8000);
        let sign = self.s.m.bin(c, Bin::And, t, m);
        let m = k(self, 0x7FFF_FFFF);
        let a = self.s.m.bin(c, Bin::And, x, m);
        // Normal halves: the exponent rebased, the mantissa rounded.
        let bias = k(self, 0x3800_0000);
        let rebased = self.s.m.bin(c, Bin::Sub, a, bias);
        let normal = if rtz {
            self.s.m.bin(c, Bin::LShr, rebased, k13)
        } else {
            let one = k(self, 1);
            let lsb = self.s.m.bin(c, Bin::LShr, rebased, k13);
            let lsb = self.s.m.bin(c, Bin::And, lsb, one);
            let half = k(self, 0xFFF);
            let up = self.s.m.bin(c, Bin::Add, rebased, half);
            let up = self.s.m.bin(c, Bin::Add, up, lsb);
            self.s.m.bin(c, Bin::LShr, up, k13)
        };
        let (inf, big) = (k(self, 0x7C00), k(self, if rtz { 0x7BFF } else { 0x7C00 }));
        let over = self.s.m.cmp(c, pred::IUGE, normal, inf);
        let normal = self.s.m.select(c, over, big, normal);
        let infinity = k(self, 0x7F80_0000);
        let is_inf = self.s.m.cmp(c, pred::EQ, a, infinity);
        let normal = self.s.m.select(c, is_inf, inf, normal);
        // Denormal halves: the mantissa shifted into units of 2^-24.
        let k23 = k(self, 23);
        let e = self.s.m.bin(c, Bin::LShr, a, k23);
        let k126 = k(self, 126);
        let s = self.s.m.bin(c, Bin::Sub, k126, e);
        let far = self.s.m.cmp(c, pred::IUGT, s, k31);
        let s = self.s.m.select(c, far, k31, s);
        let (frac, hidden, zero) = (k(self, 0x7F_FFFF), k(self, 0x80_0000), k(self, 0));
        let mant = self.s.m.bin(c, Bin::And, a, frac);
        let mant = self.s.m.bin(c, Bin::Or, mant, hidden);
        let f32_denormal = self.s.m.cmp(c, pred::EQ, e, zero);
        let mant = self.s.m.select(c, f32_denormal, zero, mant);
        let q = self.s.m.bin(c, Bin::LShr, mant, s);
        let den = if rtz {
            q
        } else {
            let one = k(self, 1);
            let s1 = self.s.m.bin(c, Bin::Sub, s, one);
            let half = self.s.m.bin(c, Bin::LShr, mant, s1);
            let half = self.s.m.bin(c, Bin::And, half, one);
            let below = self.s.m.bin(c, Bin::Shl, one, s1);
            let below = self.s.m.bin(c, Bin::Sub, below, one);
            let rest = self.s.m.bin(c, Bin::And, mant, below);
            let rest = self.s.m.cmp(c, pred::NE, rest, zero);
            let rest = self.s.m.cast(c, Cast::ZExt, rest, self.s.t.i32);
            let lsb = self.s.m.bin(c, Bin::And, q, one);
            let tie = self.s.m.bin(c, Bin::Or, rest, lsb);
            let up = self.s.m.bin(c, Bin::And, half, tie);
            self.s.m.bin(c, Bin::Add, q, up)
        };
        let min_normal = k(self, 0x3880_0000);
        let small = self.s.m.cmp(c, pred::IULT, a, min_normal);
        let r = self.s.m.select(c, small, den, normal);
        let is_nan = self.s.m.cmp(c, pred::IUGT, a, infinity);
        // A NaN keeps the top of its payload, quieted (as the RX 6700 XT,
        // RDNA2 like the console, converts).
        let payload = self.s.m.bin(c, Bin::LShr, a, k13);
        let (low10, quiet) = (k(self, 0x3FF), k(self, 0x7E00));
        let payload = self.s.m.bin(c, Bin::And, payload, low10);
        let nan = self.s.m.bin(c, Bin::Or, payload, quiet);
        let r = self.s.m.select(c, is_nan, nan, r);
        self.s.m.bin(c, Bin::Or, r, sign)
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let (a, b, third) = (self.ftz(a), self.ftz(b), self.ftz(third));
        let lo = self.max_min_flushed(false, a, b);
        let hi = self.max_min_flushed(true, a, b);
        let t = self.max_min_flushed(false, hi, third);
        let median = self.max_min_flushed(true, lo, t);
        let least = self.max_min_flushed(false, lo, third);
        let ab = self.s.m.cmp(c, pred::UNO, a, b);
        let nan3 = self.s.m.cmp(c, pred::UNO, third, third);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan3);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        let r = self.s.m.select(c, exact, zero, r);
        // The hardware returns NaN for NaN and for infinity, whose reduced
        // part is NaN. Metal's sin and cos give 0 there, so that NaN is
        // passed through here instead of relying on the host.
        let nan = self.s.m.cmp(c, pred::UNO, part, part);
        let r = self.s.m.select(c, nan, part, r);
        if cosine {
            return r;
        }
        // sin(±0) is ±0. Some Metal GPUs (M2) drop the sign through the
        // host's sin or a float select, so a zero input is passed through
        // as integer bits (from nickorlabs's #30).
        let bits = self.as_int(x);
        let mask = self.s.m.i32(0x7fff_ffff);
        let magnitude = self.s.m.bin(c, Bin::And, bits, mask);
        let zero = self.s.m.i32(0);
        let is_zero = self.s.m.cmp(c, pred::EQ, magnitude, zero);
        let result = self.as_int(r);
        let result = self.s.m.select(c, is_zero, bits, result);
        self.as_float(result)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    /// [`Self::boolean`] with the atoms of `fixed` taken as its values.
    // shortcut: no sharing of nodes reached twice, as these functions are of a few atoms.
    fn boolean_given(&mut self, b: Bool, at: usize, fixed: &[(Atom, bool)]) -> Result<V, Why> {
        let Some((atom, lo, hi)) = self.x.st.bools.node(b).filter(|_| !fixed.is_empty()) else {
            return self.boolean(b, at);
        };
        if let Some(&(_, v)) = fixed.iter().find(|f| f.0 == atom) {
            return self.boolean_given(if v { hi } else { lo }, at, fixed);
        }
        let a = self.atom(atom, at)?;
        let h = self.boolean_given(hi, at, fixed)?;
        let l = self.boolean_given(lo, at, fixed)?;
        Ok(self.s.m.select(at, a, h, l))
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            // In pixel and vertex programs (`MaskData::Host`), allow opaque
            // masks whose value is host-wave-size independent (e.g.
            // `1 << s_ff1(mask)`, `mask & -mask`, `mask & (mask - 1)`,
            // `s_bitset0(mask, s_ff1(mask))`, uniform all-0/all-1 masks).
            Atom::Opaque(v) if self.mask_data == MaskData::Host => {
                let hi_val = self.pair_high(v);
                if !self.is_host_wave_mask(v, hi_val, &mut Set::default()) {
                    return Err("an opaque lane mask");
                }
                let lo = self.get(v)?;
                let hi = hi_val.map(|h| self.get(h)).transpose()?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                let hi = if self.x.wave32_host { None } else { hi };
                Ok(self.lane_bit(at, lo, hi))
            }
            // A mesh program's lanes are the console's too (`lane_index_at`).
            Atom::Constant(k) if self.console_lanes || self.s.mesh.is_some() => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            // Quad-periodic constants (same 4-bit pattern in every quad, e.g.
            // `0x5555_5555_5555_5555`) mean the same on any host wave size.
            Atom::Constant(k)
                if self.mask_data == MaskData::Host && self.is_quad_periodic(k) =>
            {
                let lo = self.s.m.i32(k as u32);
                Ok(self.lane_bit(at, lo, None))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// The bit of lane `lane` in the mask `lo`, `hi`.
    fn lane_bit_of(&mut self, at: usize, lane: V, lo: V, hi: V) -> V {
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let low = self.s.m.cmp(at, pred::IULT, lane, k32);
        let word = self.s.m.select(at, low, lo, hi);
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The bit position of byte LDS address `a` + `offset` in its dword.
    fn lds_shift(&mut self, a: V, offset: u32) -> V {
        let c = self.cur;
        let (o, three) = (self.s.m.i32(offset), self.s.m.i32(3));
        let at = self.s.m.bin(c, Bin::Add, a, o);
        let byte = self.s.m.bin(c, Bin::And, at, three);
        self.s.m.bin(c, Bin::Shl, byte, three)
    }

    /// Whether constant mask data `lo` (and in wave64 the value `hi` of its
    /// high dword) holds every lane (`Some(true)`) or none (`Some(false)`).
    fn all_or_none(&self, lo: V, hi: Option<Value>) -> Option<bool> {
        let k = |v: V| self.s.m.int_value(v).map(|x| x as u32);
        let lo = k(lo)?;
        let hi = if self.x.native.wave64 {
            match self.vals.get(&hi?) {
                Some(&Val::I32(h)) => k(h)?,
                _ => return None,
            }
        } else {
            lo
        };
        match (lo, hi) {
            (0, 0) => Some(false),
            (u32::MAX, u32::MAX) => Some(true),
            _ => None,
        }
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        if self.s.mesh.is_none() {
            self.s.flags |= flags::WAVE_OPS;
        }
        let lane = self.lane_index_at(at);
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        let hreg = pick(reg)?;
        if let Some(hi) = inst.write(hreg) {
            return Some(hi);
        }
        // Split 64-bit add/sub pair: `s_sub_u32` + `s_subb_u32` (or `s_add_u32`
        // + `s_addc_u32`) sharing SCC into adjacent SGPRs.
        if let Some(scc) = inst.write(Reg::Scc)
            && matches!(inst.name(), "s_sub_u32" | "s_add_u32")
            && let Some(hi) = p.blocks[b].insts.iter().find_map(|u| {
                (matches!(u.name(), "s_subb_u32" | "s_addc_u32")
                    && u.read(Reg::Scc) == Some(scc))
                .then(|| u.write(hreg))
                .flatten()
            })
        {
            return Some(hi);
        }
        // Fallback: a 2-dword scalar operand reading `lo` alongside `reg + 1`.
        let mut found = None;
        for block in &p.blocks {
            for u in &block.insts {
                let ops: &[Operand] = match &u.fields {
                    Fields::Sop { src, .. } => src,
                    Fields::Valu(v) => &v.src,
                    _ => continue,
                };
                for op in ops {
                    if let Operand::Scalar { code, dwords: 2 } = *op
                        && u.read(Reg::S(code)) == Some(lo)
                        && let Some(h) = u.read(Reg::S(code + 1))
                    {
                        match found {
                            None => found = Some(h),
                            Some(prev) if prev == h => {}
                            Some(_) => return None,
                        }
                    }
                }
            }
        }
        found
    }

    /// Whether a constant lane mask repeats identically in every 4-lane quad.
    fn is_quad_periodic(&self, k: u64) -> bool {
        let lo = k as u32;
        let hi = (k >> 32) as u32;
        lo.rotate_left(4) == lo && (!self.x.native.wave64 || lo == hi)
    }

    /// Whether `op` is provably a 0/1 scalar flag (so `0 - op` is all-0/all-1).
    fn is_scalar_flag(&self, inst: &Inst, op: Operand, depth: u8) -> bool {
        if depth == 0 {
            return false;
        }
        match op {
            Operand::Const(253) => true, // src_scc
            _ if matches!(scalar_constant(op), Some(0 | 1)) => true,
            Operand::Scalar { code, .. } if code != NULL => {
                let Some(v) = inst.read(Reg::S(code)) else {
                    return false;
                };
                let Some(def) = self.def_inst(v) else {
                    return false;
                };
                let Fields::Sop { src, .. } = &def.fields else {
                    return false;
                };
                match def.name() {
                    "s_mov_b32" => src
                        .first()
                        .is_some_and(|&s| self.is_scalar_flag(def, s, depth - 1)),
                    "s_and_b32" => src
                        .iter()
                        .any(|&s| scalar_constant(s) == Some(1)),
                    "s_lshr_b32" => src.get(1).and_then(|&s| scalar_constant(s)) == Some(31),
                    "s_bfe_u32" => src
                        .get(1)
                        .and_then(|&s| scalar_constant(s))
                        .is_some_and(|imm| (imm >> 16 & 0x7F) == 1),
                    "s_cselect_b32" => {
                        src.len() == 2
                            && src
                                .iter()
                                .all(|&s| self.is_scalar_flag(def, s, depth - 1))
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Whether 32-bit scalar `v` is provably uniform all-0 or all-1.
    fn is_uniform_bitmask32(&self, v: Value, depth: u8) -> bool {
        if depth == 0 {
            return false;
        }
        let Some(def) = self.def_inst(v) else {
            return false;
        };
        let Fields::Sop { dst, src, imm } = &def.fields else {
            return false;
        };
        let op_ok = |op: Operand| -> bool {
            if matches!(scalar_constant(op), Some(0 | u32::MAX)) {
                return true;
            }
            match op {
                Operand::Scalar { code, .. } if code != NULL => def
                    .read(Reg::S(code))
                    .is_some_and(|x| self.is_uniform_bitmask32(x, depth - 1)),
                _ => false,
            }
        };
        match def.name() {
            "s_mov_b32" | "s_not_b32" => src.first().is_some_and(|&s| op_ok(s)),
            "s_movk_i32" => matches!(imm.map(|x| i32::from(x as i16) as u32), Some(0 | u32::MAX)),
            "s_cselect_b32" => src.len() == 2 && src.iter().all(|&s| op_ok(s)),
            "s_cmov_b32" | "s_cmovk_i32" => {
                let src_ok = if def.name() == "s_cmovk_i32" {
                    matches!(imm.map(|x| i32::from(x as i16) as u32), Some(0 | u32::MAX))
                } else {
                    src.first().is_some_and(|&s| op_ok(s))
                };
                let old_ok = dst.is_some_and(op_ok);
                src_ok && old_ok
            }
            "s_ashr_i32" => src.get(1).and_then(|&s| scalar_constant(s)) == Some(31),
            "s_sub_i32" | "s_sub_u32" => {
                src.first().and_then(|&s| scalar_constant(s)) == Some(0)
                    && src
                        .get(1)
                        .is_some_and(|&s| self.is_scalar_flag(def, s, depth - 1))
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                src.len() == 2 && src.iter().all(|&s| op_ok(s))
            }
            _ => false,
        }
    }

    /// Whether scalar `v` is a host-wave lane index (`s_ff1` or `W - 1 - s_flbit`
    /// of a host-wave mask).
    fn is_host_lane_index(&self, v: Value, visited: &mut Set<Value>) -> bool {
        if !visited.insert(v) {
            return false;
        }
        let Some(def) = self.def_inst(v) else {
            return false;
        };
        let Fields::Sop { dst, src, .. } = &def.fields else {
            return false;
        };
        let wave64 = self.x.native.wave64;
        let mask_arg = |op: Operand, wide: bool, visited: &mut Set<Value>| -> bool {
            let Operand::Scalar { code, .. } = op else {
                return false;
            };
            if code == NULL {
                return false;
            }
            let Some(lo) = def.read(Reg::S(code)) else {
                return false;
            };
            let hi = if wide {
                match def.read(Reg::S(code + 1)) {
                    Some(h) => Some(h),
                    None => return false,
                }
            } else {
                None
            };
            self.is_host_wave_mask(lo, hi, visited)
        };
        let flbit_of = |op: Operand, visited: &mut Set<Value>| -> bool {
            let Operand::Scalar { code, .. } = op else {
                return false;
            };
            let Some(u) = def.read(Reg::S(code)) else {
                return false;
            };
            let Some(fdef) = self.def_inst(u) else {
                return false;
            };
            let Fields::Sop { src: fsrc, .. } = &fdef.fields else {
                return false;
            };
            let want = if wave64 {
                "s_flbit_i32_b64"
            } else {
                "s_flbit_i32_b32"
            };
            if fdef.name() != want {
                return false;
            }
            let Some(&Operand::Scalar { code: mc, .. }) = fsrc.first() else {
                return false;
            };
            if mc == NULL {
                return false;
            }
            let Some(lo) = fdef.read(Reg::S(mc)) else {
                return false;
            };
            let hi = if wave64 {
                match fdef.read(Reg::S(mc + 1)) {
                    Some(h) => Some(h),
                    None => return false,
                }
            } else {
                None
            };
            self.is_host_wave_mask(lo, hi, visited)
        };
        match def.name() {
            "s_ff1_i32_b64" if wave64 => {
                src.first().is_some_and(|&s| mask_arg(s, true, visited))
            }
            "s_ff1_i32_b32" if !wave64 => {
                src.first().is_some_and(|&s| mask_arg(s, false, visited))
            }
            "s_sub_i32" | "s_sub_u32" | "s_xor_b32" => {
                let top = if wave64 { 63 } else { 31 };
                (src.first().and_then(|&s| scalar_constant(s)) == Some(top)
                    && src.get(1).is_some_and(|&s| flbit_of(s, visited)))
                    || (def.name() == "s_xor_b32"
                        && src.get(1).and_then(|&s| scalar_constant(s)) == Some(top)
                        && src.first().is_some_and(|&s| flbit_of(s, visited)))
            }
            "s_mov_b32" => src.first().is_some_and(|&s| {
                matches!(s, Operand::Scalar { code, .. } if code != NULL
                    && def.read(Reg::S(code)).is_some_and(|x| self.is_host_lane_index(x, visited)))
            }),
            "s_cselect_b32" => {
                src.len() == 2
                    && src.iter().all(|&s| {
                        matches!(s, Operand::Scalar { code, .. } if code != NULL
                            && def.read(Reg::S(code)).is_some_and(|x| self.is_host_lane_index(x, visited)))
                    })
            }
            "s_cmov_b32" => {
                let s0 = src.first().is_some_and(|&s| {
                    matches!(s, Operand::Scalar { code, .. } if code != NULL
                        && def.read(Reg::S(code)).is_some_and(|x| self.is_host_lane_index(x, visited)))
                });
                let old = dst.is_some_and(|d| {
                    matches!(d, Operand::Scalar { code, .. } if code != NULL
                        && def.read(Reg::S(code)).is_some_and(|x| self.is_host_lane_index(x, visited)))
                });
                s0 && old
            }
            _ => false,
        }
    }

    /// Whether `(lo, hi)` is a host-wave-safe lane mask in `MaskData::Host`
    /// (produces the right per-lane boolean via `lane_bit` on both 32-lane and
    /// 64-lane host waves, without depending on console-wave64 lane layout).
    fn is_host_wave_mask(&self, lo: Value, hi: Option<Value>, visited: &mut Set<Value>) -> bool {
        let wave64 = self.x.native.wave64;
        if wave64 && hi.is_none() {
            return false;
        }
        if !visited.insert(lo) {
            return true;
        }
        if self.x.st.masks.contains_key(&lo) {
            return !wave64 || hi.is_some_and(|h| self.x.st.highs.get(&h) == Some(&lo));
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let lo_phi = &self.x.p.blocks[b].phis[i];
            let hi_phi = if wave64 {
                let h = hi.expect("checked");
                let &(hb, hj) = match self.phi_at.get(&h) {
                    Some(x) if x.0 == b => x,
                    _ => return false,
                };
                Some(&self.x.p.blocks[hb].phis[hj])
            } else {
                None
            };
            let mut non_cycle = false;
            for (k, &alo) in lo_phi.args.iter().enumerate() {
                let ahi = hi_phi.map(|hp| hp.args[k]);
                if !visited.contains(&alo) {
                    non_cycle = true;
                }
                if !self.is_host_wave_mask(alo, ahi, visited) {
                    return false;
                }
            }
            return non_cycle;
        }
        let Some(lo_inst) = self.def_inst(lo) else {
            return false;
        };
        let hi_inst = hi.and_then(|h| self.def_inst(h));
        let same_inst = match (hi, hi_inst) {
            (None, _) => !wave64,
            (Some(_), Some(hi_i)) => std::ptr::eq(lo_inst, hi_i),
            (Some(_), None) => false,
        };
        let op_mask = |inst: &Inst, op: Operand, visited: &mut Set<Value>| -> bool {
            if let Some(k) = scalar_constant(op) {
                if k == 0 || k == u32::MAX {
                    return true;
                }
                return !wave64 && k.rotate_left(4) == k;
            }
            let Operand::Scalar { code, .. } = op else {
                return false;
            };
            if code == NULL {
                return true;
            }
            let Some(alo) = inst.read(Reg::S(code)) else {
                return false;
            };
            let ahi = if wave64 {
                match inst.read(Reg::S(code + 1)) {
                    Some(h) => Some(h),
                    None => return false,
                }
            } else {
                None
            };
            self.is_host_wave_mask(alo, ahi, visited)
        };
        if same_inst {
            let Fields::Sop { dst, src, imm } = &lo_inst.fields else {
                return false;
            };
            let name = lo_inst.name();
            let width = if wave64 { "_b64" } else { "_b32" };
            if name == (if wave64 { "s_lshl_b64" } else { "s_lshl_b32" }) {
                let s0 = src.first().copied();
                let s1 = src.get(1).copied();
                // `1 << lane_index` (WaveIsFirstLane idiom).
                if s0.and_then(scalar_constant) == Some(1)
                    && let Some(Operand::Scalar { code, .. }) = s1
                    && code != NULL
                    && let Some(lv) = lo_inst.read(Reg::S(code))
                {
                    return self.is_host_lane_index(lv, &mut Set::default());
                }
                // No-op shift by 0 of a host-wave mask.
                if s1.and_then(scalar_constant) == Some(0)
                    && let Some(s0) = s0
                {
                    return op_mask(lo_inst, s0, visited);
                }
                return false;
            }
            if name == (if wave64 { "s_bitset0_b64" } else { "s_bitset0_b32" })
                || name == (if wave64 { "s_bitset1_b64" } else { "s_bitset1_b32" })
            {
                let Some(d) = *dst else { return false };
                let Some(&Operand::Scalar { code, .. }) = src.first() else {
                    return false;
                };
                if code == NULL {
                    return false;
                }
                let Some(lv) = lo_inst.read(Reg::S(code)) else {
                    return false;
                };
                return op_mask(lo_inst, d, visited)
                    && self.is_host_lane_index(lv, &mut Set::default());
            }
            if name == (if wave64 { "s_ashr_i64" } else { "s_ashr_i32" }) {
                let want = if wave64 { 63 } else { 31 };
                return src.get(1).and_then(|&s| scalar_constant(s)) == Some(want);
            }
            if !wave64 {
                match name {
                    "s_movk_i32" => {
                        return matches!(
                            imm.map(|x| i32::from(x as i16) as u32),
                            Some(0 | u32::MAX)
                        );
                    }
                    "s_cmovk_i32" => {
                        return matches!(
                            imm.map(|x| i32::from(x as i16) as u32),
                            Some(0 | u32::MAX)
                        ) && dst.is_some_and(|d| op_mask(lo_inst, d, visited));
                    }
                    "s_sub_u32" | "s_sub_i32" => {
                        // `0 - mask`, `0 - flag`, or `mask - 1`.
                        let s0 = src.first().copied();
                        let s1 = src.get(1).copied();
                        if s0.and_then(scalar_constant) == Some(0)
                            && let Some(s1) = s1
                        {
                            return op_mask(lo_inst, s1, visited)
                                || self.is_scalar_flag(lo_inst, s1, 8);
                        }
                        if s1.and_then(scalar_constant) == Some(1)
                            && let Some(s0) = s0
                        {
                            return op_mask(lo_inst, s0, visited);
                        }
                        return false;
                    }
                    "s_add_u32" | "s_add_i32" => {
                        // `mask + (-1)`.
                        let s0 = src.first().copied();
                        let s1 = src.get(1).copied();
                        if s1.and_then(scalar_constant) == Some(u32::MAX)
                            && let Some(s0) = s0
                        {
                            return op_mask(lo_inst, s0, visited);
                        }
                        if s0.and_then(scalar_constant) == Some(u32::MAX)
                            && let Some(s1) = s1
                        {
                            return op_mask(lo_inst, s1, visited);
                        }
                        return false;
                    }
                    _ => {}
                }
            }
            let Some(op) = name
                .strip_prefix("s_")
                .and_then(|n| n.strip_suffix(width))
            else {
                return false;
            };
            return match op {
                "mov" | "not" => src.first().is_some_and(|&s| op_mask(lo_inst, s, visited)),
                "cselect" | "and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" => {
                    src.len() == 2 && src.iter().all(|&s| op_mask(lo_inst, s, visited))
                }
                "cmov" => {
                    src.first().is_some_and(|&s| op_mask(lo_inst, s, visited))
                        && dst.is_some_and(|d| op_mask(lo_inst, d, visited))
                }
                _ => false,
            };
        }
        // Wave64 pair written by two separate 32-bit scalar instructions.
        let Some(hi_inst) = hi_inst else {
            return false;
        };
        let (
            Fields::Sop {
                src: lsrc,
                ..
            },
            Fields::Sop {
                src: hsrc,
                ..
            },
        ) = (&lo_inst.fields, &hi_inst.fields)
        else {
            return false;
        };
        let (lname, hname) = (lo_inst.name(), hi_inst.name());
        let read_s = |inst: &Inst, op: Operand| -> Option<Value> {
            match op {
                Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
                _ => None,
            }
        };
        let shared_scc = lo_inst
            .write(Reg::Scc)
            .is_some_and(|s| hi_inst.read(Reg::Scc) == Some(s));
        // 64-bit `-mask`: `s_sub_u32 0, m_lo` + `s_subb_u32 0, m_hi`.
        if shared_scc && lname == "s_sub_u32" && hname == "s_subb_u32" {
            let l0 = lsrc.first().and_then(|&s| scalar_constant(s));
            let h0 = hsrc.first().and_then(|&s| scalar_constant(s));
            let l1 = lsrc.get(1).copied();
            let h1 = hsrc.get(1).copied();
            if l0 == Some(0)
                && h0 == Some(0)
                && let (Some(l1), Some(h1)) = (l1, h1)
            {
                if let (Some(mlo), Some(mhi)) = (read_s(lo_inst, l1), read_s(hi_inst, h1)) {
                    return self.is_host_wave_mask(mlo, Some(mhi), visited);
                }
                // Uniform 64-bit `0 - flag` (`s_sub_u32 0, flag; s_subb_u32 0, 0`).
                if scalar_constant(h1) == Some(0) && self.is_scalar_flag(lo_inst, l1, 8) {
                    return true;
                }
            }
            // 64-bit `mask - 1`: `s_sub_u32 m_lo, 1` + `s_subb_u32 m_hi, 0`.
            if l1.and_then(scalar_constant) == Some(1)
                && h1.and_then(scalar_constant) == Some(0)
                && let (Some(mlo), Some(mhi)) = (
                    lsrc.first().and_then(|&s| read_s(lo_inst, s)),
                    hsrc.first().and_then(|&s| read_s(hi_inst, s)),
                )
            {
                return self.is_host_wave_mask(mlo, Some(mhi), visited);
            }
        }
        // 64-bit `mask + (-1)`: `s_add_u32 m_lo, -1` + `s_addc_u32 m_hi, -1`.
        if shared_scc && lname == "s_add_u32" && hname == "s_addc_u32" {
            let pick = |inst: &Inst, ops: &[Operand]| -> Option<Value> {
                let (s0, s1) = (*ops.first()?, *ops.get(1)?);
                if scalar_constant(s1) == Some(u32::MAX) {
                    read_s(inst, s0)
                } else if scalar_constant(s0) == Some(u32::MAX) {
                    read_s(inst, s1)
                } else {
                    None
                }
            };
            if let (Some(mlo), Some(mhi)) = (pick(lo_inst, lsrc), pick(hi_inst, hsrc)) {
                return self.is_host_wave_mask(mlo, Some(mhi), visited);
            }
        }
        // Uniform all-0/all-1 replicated into both dwords: `hi = s_mov_b32 lo`
        // (or vice versa), or matching `s_cselect_b32` on the same SCC.
        if hname == "s_mov_b32"
            && hsrc.first().and_then(|&s| read_s(hi_inst, s)) == Some(lo)
            && self.is_uniform_bitmask32(lo, 8)
        {
            return true;
        }
        if lname == "s_mov_b32"
            && let Some(h) = hi
            && lsrc.first().and_then(|&s| read_s(lo_inst, s)) == Some(h)
            && self.is_uniform_bitmask32(h, 8)
        {
            return true;
        }
        if lname == "s_cselect_b32"
            && hname == "s_cselect_b32"
            && lo_inst.read(Reg::Scc).is_some()
            && lo_inst.read(Reg::Scc) == hi_inst.read(Reg::Scc)
            && lsrc.len() == 2
            && hsrc.len() == 2
        {
            let pair_ok = |a: Operand, b: Operand| -> bool {
                match (scalar_constant(a), scalar_constant(b)) {
                    (Some(x), Some(y)) => x == y && matches!(x, 0 | u32::MAX),
                    _ => false,
                }
            };
            if pair_ok(lsrc[0], hsrc[0]) && pair_ok(lsrc[1], hsrc[1]) {
                return true;
            }
        }
        // Split 32-bit identical bitwise/move ops on matching halves of host masks.
        if lname == hname {
            let pair_arg = |k: usize, visited: &mut Set<Value>| -> bool {
                let (Some(&la), Some(&ha)) = (lsrc.get(k), hsrc.get(k)) else {
                    return false;
                };
                if let (Some(lc), Some(hc)) = (scalar_constant(la), scalar_constant(ha)) {
                    return lc == hc && lc.rotate_left(4) == lc;
                }
                let (Some(lv), Some(hv)) = (read_s(lo_inst, la), read_s(hi_inst, ha)) else {
                    return false;
                };
                self.is_host_wave_mask(lv, Some(hv), visited)
            };
            match lname {
                "s_mov_b32" | "s_not_b32" => return pair_arg(0, visited),
                "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
                | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                    return pair_arg(0, visited) && pair_arg(1, visited);
                }
                _ => {}
            }
        }
        false
    }

    /// Whether `inst` may read lane masks as data in `MaskData::Host`.
    fn host_mask_ok(&self, inst: &Inst) -> bool {
        let name = inst.name();
        if name.starts_with("s_bcnt1_i32_")
            || name.starts_with("s_ff1_i32_")
            || name.starts_with("s_flbit_i32_b")
            || name.starts_with("v_mbcnt_")
            || matches!(
                name,
                "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
            )
        {
            return true;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return false;
        };
        let is_mask_op = |op: Operand| -> bool {
            let Operand::Scalar { code, .. } = op else {
                return false;
            };
            code != NULL
                && inst.read(Reg::S(code)).is_some_and(|v| {
                    self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v)
                })
        };
        match name {
            "s_bitset0_b32" | "s_bitset0_b64" | "s_bitset1_b32" | "s_bitset1_b64" => {
                // Only when toggling a host-wave lane index (e.g. `s_ff1`).
                matches!(src.first(), Some(&Operand::Scalar { code, .. }) if code != NULL
                    && inst.read(Reg::S(code)).is_some_and(|v| self.is_host_lane_index(v, &mut Set::default())))
            }
            "s_sub_u32" | "s_sub_i32" => {
                let s0 = src.first().copied();
                let s1 = src.get(1).copied();
                (s0.and_then(scalar_constant) == Some(0) && s1.is_some_and(is_mask_op))
                    || (s1.and_then(scalar_constant) == Some(1) && s0.is_some_and(is_mask_op))
            }
            "s_subb_u32" => {
                let s0 = src.first().copied();
                let s1 = src.get(1).copied();
                (s0.and_then(scalar_constant) == Some(0) && s1.is_some_and(is_mask_op))
                    || (s1.and_then(scalar_constant) == Some(0) && s0.is_some_and(is_mask_op))
            }
            "s_add_u32" | "s_add_i32" | "s_addc_u32" => {
                let s0 = src.first().copied();
                let s1 = src.get(1).copied();
                (s1.and_then(scalar_constant) == Some(u32::MAX) && s0.is_some_and(is_mask_op))
                    || (s0.and_then(scalar_constant) == Some(u32::MAX)
                        && s1.is_some_and(is_mask_op))
            }
            // Split 32-bit bitwise ops on a wave64 mask (e.g. `mask & -mask`).
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" => {
                self.x.native.wave64
                    && src.len() == 2
                    && src.iter().any(|&s| is_mask_op(s))
                    && src.iter().all(|&s| {
                        is_mask_op(s)
                            || matches!(s, Operand::Scalar { code, .. } if code != NULL
                                && inst.read(Reg::S(code)).is_some_and(|v| {
                                    self.def_inst(v).is_some_and(|d| {
                                        matches!(
                                            d.name(),
                                            "s_sub_u32" | "s_subb_u32" | "s_add_u32" | "s_addc_u32"
                                        )
                                    })
                                }))
                    })
            }
            _ => false,
        }
    }

    /// Whether a phi `v` (either dword in wave64) belongs to a host-wave mask.
    fn is_host_wave_mask_phi(&self, v: Value) -> bool {
        if !self.x.native.wave64 {
            return self.is_host_wave_mask(v, None, &mut Set::default());
        }
        if let Some(hi) = self.pair_high(v)
            && self.is_host_wave_mask(v, Some(hi), &mut Set::default())
        {
            return true;
        }
        if let Some(&(b, i)) = self.phi_at.get(&v)
            && let Reg::S(c) = self.x.p.blocks[b].phis[i].reg
            && c > 0
            && let Some(lo) = self.x.p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == Reg::S(c - 1))
                .map(|q| q.value)
        {
            return self.is_host_wave_mask(lo, Some(v), &mut Set::default());
        }
        false
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        match self.x.st.guards[b][i] {
            None => Ok(None),
            Some(g) => Ok(Some(self.boolean(g, self.cur)?)),
        }
    }

    /// Whether every lane outside the guard of the export at (b, i) runs
    /// the `done` export later in block b without its own guard holding,
    /// i.e. is discarded there.
    fn discarded_at_done(&self, b: usize, i: usize) -> bool {
        let guards = &self.x.st.guards[b];
        let g = guards[i].unwrap_or(Bool::TRUE);
        self.exports.iter().any(|(&(eb, ei), e)| {
            e.done && eb == b && ei > i && {
                let done = guards[ei].unwrap_or(Bool::TRUE);
                !self.x.st.bools.meet(&[(done, true), (g, false)])
            }
        })
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let o = inst.read(reg);
                let read = o.is_some_and(|o| self.read_phis.contains(&o));
                let old = match o.map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                // What lanes keep is zero where it has no DXIL form (as
                // above), so keeping a phi does not make it live.
                if let Some(o) = o.filter(|_| !read) {
                    self.read_phis.remove(&o);
                }
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                // On the flushed result; a zero or denormal product comes out
                // +0 (measured on RDNA2 with IEEE mode off).
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.cur;
                let k = self.s.m.f32(factor.to_bits());
                let r = self.ftz(f);
                let r = self.s.m.bin(c, Bin::Mul, r, k);
                let x = self.as_int(r);
                let (exp, zero) = (self.s.m.i32(0x7F80_0000), self.s.m.i32(0));
                let e = self.s.m.bin(c, Bin::And, x, exp);
                let tiny = self.s.m.cmp(c, pred::EQ, e, zero);
                let x = self.s.m.select(c, tiny, zero, x);
                f = self.as_float(x);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad. `outside` names the operation
    /// in a pixel and in a vertex program ([`outside_compute!`]).
    fn cross_lane(&mut self, quad: bool, outside: [Why; 2]) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err(outside[usize::from(!self.s.pixel)]);
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// The thread's index in the group as console lanes number it: the
    /// first lane's flat id rounded down to the host wave size, plus the
    /// host lane.
    fn console_flat(&mut self) -> V {
        if let Some(f) = self.console_flat {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self
            .s
            .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        self.console_flat = Some(f);
        f
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let [x, y, _] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let f = self.console_flat();
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let guard = (threads > mesh.max_primitives).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        self.lane_index_at(self.cur)
    }

    /// [`Self::lane_index`] in DXIL block `at`. On a 32-lane host where a
    /// console wave is two host waves, the lane is the group index modulo 64.
    fn lane_index_at(&mut self, at: usize) -> V {
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        if self.halves {
            let f = self.console_flat();
            let m = self.s.m.i32(63);
            return self.s.m.bin(at, Bin::And, f, m);
        }
        self.s.op(
            at,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    /// Whether a quad-local read must reach helper lanes: in a pixel program
    /// every quad a triangle edge crosses has some (the console's whole-quad
    /// mode runs them; discarded pixels stay helpers too), and only quad
    /// reads see them on the host (wave reads give 0 from them on AMD: wrong
    /// derivatives of computed values along every triangle edge).
    fn helpers_read(&self, quad: bool) -> bool {
        quad && self.s.pixel
    }

    /// `v_readfirstlane_b32`'s result `first` of `own` in a pixel shader:
    /// a helper lane takes its own value instead.
    ///
    /// On the console, whole-quad mode puts helper lanes in EXEC, so a
    /// waterfall loop (read the first lane's value, run the lanes with that
    /// value, take them out of EXEC) serves helper lanes too. On the host,
    /// wave operations leave helper lanes out (DXIL before
    /// `WaveOpsIncludeHelperLanes`): once only helper lanes are left in such
    /// a loop, `WaveReadLaneFirst` reads no lane, no lane matches, and the
    /// loop runs to [`LOOP_LIMIT`] in every such lane: draws of 0.3 to 1.3 s
    /// each, then DEVICE_HUNG. With its own value a helper lane matches in
    /// the first pass and leaves; what a helper lane computes only feeds
    /// derivatives, and its stores are dropped.
    fn helper_reads_own(&mut self, own: V, first: V) -> V {
        if !self.s.pixel {
            return first;
        }
        let c = self.cur;
        let helper = self
            .s
            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        self.s.m.select(c, helper, own, first)
    }

    /// `x` of lane `src` where `src` shares this lane's 32 (DPP rows, quads
    /// and `ds_swizzle` groups never cross them): a host wave holds one half
    /// of a console wave, in the same order.
    fn read_in_half(&mut self, x: V, src: V) -> V {
        let src = if self.host32 {
            let k31 = self.s.m.i32(31);
            self.s.m.bin(self.cur, Bin::And, src, k31)
        } else {
            src
        };
        self.read_lane(x, src)
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // Bit 15 set: each lane of a quad picks a lane of it; else lanes
            // of each group of 32 combine their index with and/or/xor masks.
            let quad = offset & 0x8000 != 0
                || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            if !mesh_quad {
                let outside = if quad {
                    outside_compute!("ds_swizzle_b32 within a quad")
                } else {
                    outside_compute!("ds_swizzle_b32 across quads")
                };
                self.cross_lane(quad, outside)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let r = if mesh_quad || self.helpers_read(quad) {
                self.quad_pick(x, src)
            } else {
                self.read_in_half(x, src)
            };
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if name == "ds_bpermute_b32" {
            // Each lane reads `data` of the lane its address plus the offset
            // names in its own 32-lane half; 0 from a lane EXEC leaves out.
            self.cross_lane(false, outside_compute!("ds_bpermute_b32"))?;
            let a = self.vgprs(inst, addr.as_ref().ok_or("a permute address")?, 1)?[0];
            let x = self.vgprs(inst, data.first().ok_or("a permute source")?, 1)?[0];
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let (o, two, k31) = (k(self, offset), k(self, 2), k(self, 31));
            let at = self.s.m.bin(c, Bin::Add, a, o);
            let dword = self.s.m.bin(c, Bin::LShr, at, two);
            let local = self.s.m.bin(c, Bin::And, dword, k31);
            let src = if self.x.native.wave64 && !self.host32 {
                let lane = self.lane_index();
                let k32 = k(self, 32);
                let half = self.s.m.bin(c, Bin::And, lane, k32);
                self.s.m.bin(c, Bin::Or, half, local)
            } else {
                local
            };
            let r = self.read_lane(x, src);
            let on = guard.unwrap_or_else(|| self.s.m.bool(true));
            let (lo, hi) = self.host_ballot(c, on);
            let running = self.lane_bit_of(c, src, lo, hi);
            let z = k(self, 0);
            let r = self.s.m.select(c, running, r, z);
            let reg = dst_reg.ok_or("a permute result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        if matches!(name, "ds_append" | "ds_consume" | "ds_read_addtid_b32" | "ds_write_addtid_b32")
        {
            return self.ds_by_m0(inst, guard, offset, dst_reg);
        }
        // The `ds_gws` group's address is M0's too (GDS); refused by name.
        let no_address = || match name {
            n if n.starts_with("ds_gws") => "an LDS address: a ds_gws instruction has none (M0)",
            _ => "an LDS address",
        };
        let a = self.vgprs(inst, addr.as_ref().ok_or_else(no_address)?, 1)?[0];
        // shortcut: LDS addresses are taken as aligned to what they access
        // and in bounds (out-of-bounds console accesses read 0 and drop
        // writes).
        let dword = |l: &mut Self, bytes: u32| -> V {
            let c = l.cur;
            let o = l.s.m.i32(bytes);
            let at = l.s.m.bin(c, Bin::Add, a, o);
            let two = l.s.m.i32(2);
            let index = l.s.m.bin(c, Bin::LShr, at, two);
            let lds = l.s.lds();
            l.s.m.element(c, lds, index)
        };
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        if let Some(rest) = name.strip_prefix("ds_read") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                // A byte or half, extended (`_d16`: into the low half,
                // `_d16_hi`: the high half, the other half kept).
                let (bits, signed) = match rest.trim_end_matches("_hi").trim_end_matches("_d16") {
                    "_u8" => (8, false),
                    "_i8" => (8, true),
                    "_u16" => (16, false),
                    "_i16" => (16, true),
                    _ => return Err("an LDS read of bytes or halves"),
                };
                let reg = dst_reg.ok_or("an LDS read result")?;
                let p = dword(self, offset);
                let x = self.s.m.load(c, p);
                let sh = self.lds_shift(a, offset);
                let x = self.s.m.bin(c, Bin::LShr, x, sh);
                let left = self.s.m.i32(32 - bits);
                let x = self.s.m.bin(c, Bin::Shl, x, left);
                let x = self
                    .s
                    .m
                    .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left);
                let x = if rest.contains("_d16") {
                    let old = match inst.read(Reg::V(reg)) {
                        Some(o) => self.i32_of(o)?,
                        None => self.s.m.i32(0),
                    };
                    let hi = rest.ends_with("_hi");
                    let keep = self.s.m.i32(if hi { 0xFFFF } else { 0xFFFF_0000 });
                    let old = self.s.m.bin(c, Bin::And, old, keep);
                    let x = if hi {
                        let k16 = self.s.m.i32(16);
                        self.s.m.bin(c, Bin::Shl, x, k16)
                    } else {
                        let low = self.s.m.i32(0xFFFF);
                        self.s.m.bin(c, Bin::And, x, low)
                    };
                    self.s.m.bin(c, Bin::Or, old, x)
                } else {
                    x
                };
                self.write(inst, Reg::V(reg), x, guard);
                return Ok(());
            };
            let reg = dst_reg.ok_or("an LDS read result")?;
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let p = dword(self, off + 4 * k);
                    let x = self.s.m.load(self.cur, p);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                // A byte or half (`_d16_hi`: the high half of the data) into
                // its dword through atomics, so lanes writing the other parts
                // of the dword keep theirs.
                // shortcut: lanes writing the same part in one instruction
                // leave the OR of theirs (the console keeps one).
                let bits = match rest.trim_end_matches("_d16_hi") {
                    "_b8" => 8,
                    "_b16" => 16,
                    _ => return Err("an LDS write of bytes or halves"),
                };
                let x = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
                let x = if rest.ends_with("_d16_hi") {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let mask = self.s.m.i32((1 << bits) - 1);
                let x = self.s.m.bin(c, Bin::And, x, mask);
                let sh = self.lds_shift(a, offset);
                let set = self.s.m.bin(c, Bin::Shl, x, sh);
                let part = self.s.m.bin(c, Bin::Shl, mask, sh);
                let ones = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, part, ones);
                let p = dword(self, offset);
                let at = self.begin_guarded(guard);
                use super::bitcode::Rmw;
                self.s.m.atomic(self.cur, Rmw::And, p, keep);
                self.s.m.atomic(self.cur, Rmw::Or, p, set);
                self.end_guarded(at, &[]);
                return Ok(());
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let p = dword(self, off);
                self.s.m.store(self.cur, p, x);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        let op = name
            .strip_prefix("ds_")
            .map(|n| n.replace("_rtn", ""))
            .ok_or("an LDS operation")?;
        use super::bitcode::Rmw;
        let rmw = match op.as_str() {
            "add_u32" => Some(Rmw::Add),
            "sub_u32" => Some(Rmw::Sub),
            "and_b32" => Some(Rmw::And),
            "or_b32" => Some(Rmw::Or),
            "xor_b32" => Some(Rmw::Xor),
            "min_i32" => Some(Rmw::Min),
            "max_i32" => Some(Rmw::Max),
            "min_u32" => Some(Rmw::UMin),
            "max_u32" => Some(Rmw::UMax),
            "wrxchg_b32" => Some(Rmw::Xchg),
            "cmpst_b32" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.vgprs(inst, data.first().ok_or("an LDS atomic")?, 1)?[0];
        let v1 = match data.get(1) {
            Some(d) => Some(self.vgprs(inst, d, 1)?[0]),
            None => None,
        };
        let old_reg = dst_reg.and_then(|r| inst.read(Reg::V(r)).map(|v| (r, v)));
        let outside = match old_reg {
            Some((_, v)) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
            None => self.s.m.i32(0),
        };
        let p = dword(self, offset);
        let at = self.begin_guarded(guard);
        let r = match rmw {
            Some(op) => self.s.m.atomic(self.cur, op, p, v0),
            // Compare with data0, store data1.
            None => self
                .s
                .m
                .cmpxchg(self.cur, p, v0, v1.ok_or("an LDS compare")?),
        };
        let r = self.end_guarded(at, &[(r, outside)])[0];
        if let Some(reg) = dst_reg {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }

    /// The DS instructions whose LDS address comes from M0 (RDNA2 ISA 12.13, LDS only):
    /// * `ds_append` (`ds_consume`): one atomic add (subtract) of the count of the valid lanes at
    ///   M0 + offset, by the first valid lane, its pre-op value returned to every valid lane. M0
    ///   holds the byte address, as LLVM sets it (`SelectDSAppendConsume` copies the pointer to M0
    ///   unchanged; the ISA's "M0.base").
    /// * `ds_read_addtid_b32` (`ds_write_addtid_b32`): the dword at offset + M0[15:0] + 4 × the
    ///   lane's id in the wave (ISA 10.4, 12.13).
    // shortcut: like other LDS accesses, addresses are taken as aligned and in bounds (the ISA
    // clamps append and consume by M0's size).
    fn ds_by_m0(
        &mut self,
        inst: &Inst,
        guard: Option<V>,
        offset: u32,
        dst_reg: Option<u8>,
    ) -> Result<(), Why> {
        use super::bitcode::Rmw;
        let name = inst.name();
        let m0 = inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?;
        let m0 = self.i32_of(m0)?;
        let c = self.cur;
        let dword = |l: &mut Self, at: V| -> V {
            let c = l.cur;
            let two = l.s.m.i32(2);
            let index = l.s.m.bin(c, Bin::LShr, at, two);
            let lds = l.s.lds();
            l.s.m.element(c, lds, index)
        };
        if name.contains("addtid") {
            let low = self.s.m.i32(0xFFFF);
            let base = self.s.m.bin(c, Bin::And, m0, low);
            let o = self.s.m.i32(offset);
            let base = self.s.m.bin(c, Bin::Add, base, o);
            // The lane's id in the console's wave, also where a host wave is half of it.
            let lane = if self.host32 {
                let f = self.console_flat();
                let k63 = self.s.m.i32(63);
                self.s.m.bin(c, Bin::And, f, k63)
            } else {
                self.lane_index_at(c)
            };
            let four = self.s.m.i32(4);
            let at = self.s.m.bin(c, Bin::Mul, lane, four);
            let at = self.s.m.bin(c, Bin::Add, base, at);
            let Fields::Ds { data, .. } = &inst.fields else { unreachable!() };
            if name == "ds_write_addtid_b32" {
                let x = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
                let g = self.begin_guarded(guard);
                let p = dword(self, at);
                self.s.m.store(self.cur, p, x);
                self.end_guarded(g, &[]);
            } else {
                let reg = dst_reg.ok_or("an LDS read")?;
                let p = dword(self, at);
                let x = self.s.m.load(c, p);
                self.write(inst, Reg::V(reg), x, guard);
            }
            return Ok(());
        }
        // One atomic for the console's wave: a host wave must be all of it.
        let lanes: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        if self.halves || self.host32 && lanes > 32 {
            return Err("ds_append or ds_consume in a console wave the host splits");
        }
        let reg = dst_reg.ok_or("an append result")?;
        let outside = match inst.read(Reg::V(reg)) {
            Some(v) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
            None => self.s.m.i32(0),
        };
        let g = self.begin_guarded(guard);
        let c = self.cur;
        let i32t = self.s.t.i32;
        // The valid lanes: those running here.
        let on = self.s.m.bool(true);
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self.s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
        let (x, y) = (self.s.m.extract(c, ballot, 0), self.s.m.extract(c, ballot, 1));
        let nx = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[x]);
        let ny = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[y]);
        let n = self.s.m.bin(c, Bin::Add, nx, ny);
        let i1 = self.s.t.i1;
        let first = self.s.op(c, "waveIsFirstLane", Attr::NoUnwind, i1, 110, &[]);
        self.s.flags |= flags::WAVE_OPS;
        let o = self.s.m.i32(offset);
        let at = self.s.m.bin(c, Bin::Add, m0, o);
        let lead = self.begin_guarded(Some(first));
        let p = dword(self, at);
        let op = if name == "ds_append" { Rmw::Add } else { Rmw::Sub };
        let old = self.s.m.atomic(self.cur, op, p, n);
        let zero = self.s.m.i32(0);
        let old = self.end_guarded(lead, &[(old, zero)])[0];
        let c = self.cur;
        let r = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[old]);
        let r = self.end_guarded(g, &[(r, outside)])[0];
        self.write(inst, Reg::V(reg), r, None);
        Ok(())
    }

    /// Thread-group sync with a shared-memory fence, and a UAV fence when
    /// the program writes UAVs (the waits before an s_barrier make those
    /// writes visible on the console).
    fn group_barrier(&mut self) {
        let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
        let mode = self.s.m.i32(mode);
        let void = self.s.t.void;
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => self.host_mask_ok(inst),
            MaskData::None => name == "s_bcnt1_i32_b64",
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                self.group_barrier();
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } => Err("flat memory"),
        };
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    /// Each result dword of a swap's last XOR ([`crate::structure::xored_back`]) and the value
    /// it equals.
    fn xor_copies(&self, inst: &Inst) -> Option<Vec<(Value, Value)>> {
        let Fields::Sop { dst: Some(Operand::Scalar { code: d, .. }), .. } = &inst.fields else {
            return None;
        };
        let halves = if inst.name() == "s_xor_b64" { 2 } else { 1 };
        (0..halves)
            .map(|k| {
                let a = crate::structure::xored_back(inst, k, |v| self.def_inst(v))?;
                Some((inst.write(Reg::S(d + k))?, a))
            })
            .collect()
    }

    /// `v` as a 32-bit number, when it has one.
    fn i32_data(&mut self, v: Value) -> Result<V, Why> {
        match self.vals.get(&v) {
            Some(&Val::I32(x)) => Ok(x),
            _ => Err("not data"),
        }
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v))
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        // The last XOR of a swap by three (the console compiler's swap of two SGPRs, a mask's
        // and data's say) is a copy of the data: what it XORs back out.
        if let Some(copies) = self.xor_copies(inst)
            && copies.iter().all(|&(_, a)| !self.is_mask(Some(a)))
            && let Ok(vals) =
                copies.iter().map(|&(_, a)| self.i32_data(a)).collect::<Result<Vec<_>, _>>()
        {
            let mut any = None;
            for (&(w, _), x) in copies.iter().zip(vals) {
                self.vals.insert(w, Val::I32(x));
                any = Some(match any {
                    None => x,
                    Some(y) => self.s.m.bin(self.cur, Bin::Or, y, x),
                });
            }
            if let (Some(scc), Some(any)) = (scc, any) {
                let z = self.s.m.i32(0);
                let b = self.s.m.cmp(self.cur, pred::NE, any, z);
                self.vals.insert(scc, Val::Bit(b));
            }
            return Ok(());
        }
        if inst.write(Reg::S(EXEC)).is_some()
            || (self.x.native.wave64 && inst.write(Reg::S(EXEC + 1)).is_some())
            || self.is_mask(lo)
        {
            if let Some(scc) = scc {
                // The result is EXEC when written, else the destination.
                let pair = |l: u8| {
                    let lo = inst.write(Reg::S(l))?;
                    if !self.x.st.masks.contains_key(&lo) {
                        return None;
                    }
                    let hi = inst.write(Reg::S(l + 1)).filter(|_| self.x.native.wave64);
                    Some((lo, hi))
                };
                if self.mask_data != MaskData::None {
                    if let Some(r) = pair(EXEC).or_else(|| dst_code.and_then(pair)) {
                        self.mask_scc.insert(scc, r);
                    }
                } else if let Some(u) = self.uniform_and_exec(inst) {
                    // Mesh: no ballot, but a uniform mask needs none.
                    self.uniform_scc.insert(scc, u);
                } else if let Some(r) = self.uniform_with_exec(inst) {
                    self.uniform_bits.insert(scc, r);
                }
            }
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                // Of a signed operand, a literal sign-extends too.
                Some(&Operand::Literal(x)) if name.ends_with("_i64") => {
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        if let Some(op) = name.strip_prefix("s_bitcmp") {
            // SCC is whether bit src1 (mod 32, or mod 64 for the 64-bit form)
            // of src0 is clear (bitcmp0) or set (bitcmp1).
            let (bit, k31) = (src32(self, 1)?, self.s.m.i32(31));
            let idx = self.s.m.bin(c, Bin::And, bit, k31);
            let word = if op.ends_with("_b64") {
                let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                let k32 = self.s.m.i32(32);
                let upper = self.s.m.bin(c, Bin::And, bit, k32);
                let z = self.s.m.i32(0);
                let upper = self.s.m.cmp(c, pred::NE, upper, z);
                self.s.m.select(c, upper, b, a)
            } else {
                src32(self, 0)?
            };
            let shifted = self.s.m.bin(c, Bin::LShr, word, idx);
            let one = self.s.m.i32(1);
            let tested = self.s.m.bin(c, Bin::And, shifted, one);
            let want = self.s.m.i32(u32::from(op.starts_with('1')));
            let r = self.s.m.cmp(c, pred::EQ, tested, want);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64")
            || name.ends_with("_u64")
            || name.ends_with("_i64")
            || name == "s_bitreplicate_b64_b32";
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                "s_quadmask_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let low = self.quad_mask(c, a);
                    let high = self.quad_mask(c, b);
                    let k8 = self.s.m.i32(8);
                    let high = self.s.m.bin(c, Bin::Shl, high, k8);
                    let mask = self.s.m.bin(c, Bin::Or, low, high);
                    let zero = self.s.m.i32(0);
                    self.set(lo, Val::I32(mask));
                    self.set(hi, Val::I32(zero));
                    let nonzero = self.s.m.cmp(c, pred::NE, mask, zero);
                    self.set(scc, Val::Bit(nonzero));
                    Ok(())
                }
                "s_wqm_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let (rl, rh) = (self.wqm_word(c, a), self.wqm_word(c, b));
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let na = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
                    let nb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
                    let r = self.s.m.bin(c, Bin::Add, na, nb);
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_bcnt0_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let na = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
                    let nb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
                    let ones = self.s.m.bin(c, Bin::Add, na, nb);
                    let k64 = self.s.m.i32(64);
                    let r = self.s.m.bin(c, Bin::Sub, k64, ones);
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_flbit_i32_b64" | "s_flbit_i32_i64" => {
                    let (mut a, mut b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    if name == "s_flbit_i32_i64" {
                        // Turn bits differing from bit 63 into set bits in both halves.
                        let k31 = self.s.m.i32(31);
                        let sign = self.s.m.bin(c, Bin::AShr, b, k31);
                        a = self.s.m.bin(c, Bin::Xor, a, sign);
                        b = self.s.m.bin(c, Bin::Xor, b, sign);
                    }
                    // Leading zeros; -1 for none (DXIL's first bit high).
                    let i32t = self.s.t.i32;
                    let fa = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[a]);
                    let fb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let fa32 = self.s.m.bin(c, Bin::Add, fa, k32);
                    let none = self.s.m.cmp(c, pred::EQ, a, z);
                    let fa = self.s.m.select(c, none, fa, fa32);
                    let high = self.s.m.cmp(c, pred::NE, b, z);
                    let r = self.s.m.select(c, high, fb, fa);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_brev_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let ra = self.s.op(c, "unary.i32", Attr::ReadNone, i32t, 30, &[a]);
                    let rb = self.s.op(c, "unary.i32", Attr::ReadNone, i32t, 30, &[b]);
                    self.set(lo, Val::I32(rb));
                    self.set(hi, Val::I32(ra));
                    Ok(())
                }
                "s_ff1_i32_b64" | "s_ff0_i32_b64" => {
                    // The lowest set (clear) bit; -1 for none (as DXIL's first bit low).
                    let (mut a, mut b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    if name == "s_ff0_i32_b64" {
                        let ones = self.s.m.i32(u32::MAX);
                        a = self.s.m.bin(c, Bin::Xor, a, ones);
                        b = self.s.m.bin(c, Bin::Xor, b, ones);
                    }
                    let i32t = self.s.t.i32;
                    let fa = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
                    let fb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let fb = self.s.m.select(c, none, fb, fb32);
                    let low = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, low, fa, fb);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_bfm_b64" => {
                    // ((1 << S0[5:0]) - 1) << S1[5:0] (RDNA2 ISA, SOP2 opcode
                    // 37; the summary table's [4:0] is the 32-bit form's).
                    let (w, o) = (src32(self, 0)?, src32(self, 1)?);
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let k63 = self.s.m.i32(63);
                    let w = self.s.m.bin(c, Bin::And, w, k63);
                    let w = self.s.m.cast(c, Cast::ZExt, w, i64t);
                    let o = self.s.m.bin(c, Bin::And, o, k63);
                    let o = self.s.m.cast(c, Cast::ZExt, o, i64t);
                    let one = self.s.m.int_const(i64t, 1);
                    let m = self.s.m.bin(c, Bin::Shl, one, w);
                    let m = self.s.m.bin(c, Bin::Sub, m, one);
                    let r = self.s.m.bin(c, Bin::Shl, m, o);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    Ok(())
                }
                "s_bitreplicate_b64_b32" => {
                    // Bit i of S0 to bits 2i and 2i+1 (RDNA2 ISA, SOP1 opcode
                    // 59): each half of S0 spread to the even bits of a dword,
                    // then doubled.
                    let a = src32(self, 0)?;
                    let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
                    let halves = [
                        (self.s.m.bin(c, Bin::And, a, low), lo),
                        (self.s.m.bin(c, Bin::LShr, a, k16), hi),
                    ];
                    for (mut x, v) in halves {
                        for (d, keep) in [
                            (8, 0x00FF_00FF),
                            (4, 0x0F0F_0F0F),
                            (2, 0x3333_3333),
                            (1, 0x5555_5555),
                        ] {
                            let d = self.s.m.i32(d);
                            let t = self.s.m.bin(c, Bin::Shl, x, d);
                            let t = self.s.m.bin(c, Bin::Or, x, t);
                            let keep = self.s.m.i32(keep);
                            x = self.s.m.bin(c, Bin::And, t, keep);
                        }
                        let one = self.s.m.i32(1);
                        let t = self.s.m.bin(c, Bin::Shl, x, one);
                        let x = self.s.m.bin(c, Bin::Or, x, t);
                        self.set(v, Val::I32(x));
                    }
                    Ok(())
                }
                "s_mov_b64" => {
                    // Dword by dword: one without a value (the high dword of
                    // a 64-bit wave sum) leaves the other usable.
                    for (h, v) in [(0, lo), (1, hi)] {
                        let x = half(self, 0, h).map_or_else(Val::Missing, Val::I32);
                        self.set(v, x);
                    }
                    Ok(())
                }
                // The code is taken to sit at address 0: what the program
                // computes from the counter is a byte offset in the code,
                // which accesses there read through the code table
                // (`resource::CODE` with that offset as its computed term).
                "s_getpc_b64" => {
                    let next = self.s.m.i32(inst.machine.offset as u32 + 4);
                    let zero = self.s.m.i32(0);
                    self.set(lo, Val::I32(next));
                    self.set(hi, Val::I32(zero));
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                "s_cmov_b64" => {
                    let Some(code) = dst_code else {
                        return Err("a missing operand");
                    };
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let a = half(self, 0, h)?;
                        let old =
                            self.i32_of(inst.read(Reg::S(code + h)).ok_or("an unread operand")?)?;
                        let r = self.s.m.select(c, cond, a, old);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                "s_bitset0_b64" | "s_bitset1_b64" => {
                    let Some(code) = dst_code else {
                        return Err("a missing operand");
                    };
                    let old_lo = self.i32_of(inst.read(Reg::S(code)).ok_or("an unread operand")?)?;
                    let old_hi =
                        self.i32_of(inst.read(Reg::S(code + 1)).ok_or("an unread operand")?)?;
                    let bit = src32(self, 0)?;
                    let (k31, k32, one, z) = (
                        self.s.m.i32(31),
                        self.s.m.i32(32),
                        self.s.m.i32(1),
                        self.s.m.i32(0),
                    );
                    let idx = self.s.m.bin(c, Bin::And, bit, k31);
                    let mask = self.s.m.bin(c, Bin::Shl, one, idx);
                    let hi_bit = self.s.m.bin(c, Bin::And, bit, k32);
                    let upper = self.s.m.cmp(c, pred::NE, hi_bit, z);
                    let lo_mask = self.s.m.select(c, upper, z, mask);
                    let hi_mask = self.s.m.select(c, upper, mask, z);
                    let (rlo, rhi) = if name == "s_bitset1_b64" {
                        (
                            self.s.m.bin(c, Bin::Or, old_lo, lo_mask),
                            self.s.m.bin(c, Bin::Or, old_hi, hi_mask),
                        )
                    } else {
                        let ones = self.s.m.i32(u32::MAX);
                        let nlo = self.s.m.bin(c, Bin::Xor, lo_mask, ones);
                        let nhi = self.s.m.bin(c, Bin::Xor, hi_mask, ones);
                        (
                            self.s.m.bin(c, Bin::And, old_lo, nlo),
                            self.s.m.bin(c, Bin::And, old_hi, nhi),
                        )
                    };
                    self.set(lo, Val::I32(rlo));
                    self.set(hi, Val::I32(rhi));
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_quadmask_b32" => {
                let pixels = src32(self, 0)?;
                (self.quad_mask(c, pixels), None)
            }
            "s_wqm_b32" => {
                let pixels = src32(self, 0)?;
                (self.wqm_word(c, pixels), None)
            }
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_cmov_b32" | "s_cmovk_i32" => {
                let cond = self.scc_in(inst)?;
                let a = if name == "s_cmovk_i32" {
                    sext16(self)
                } else {
                    src32(self, 0)?
                };
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                (self.s.m.select(c, cond, a, old), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_sext_i32_i8" | "s_sext_i32_i16" => {
                let a = src32(self, 0)?;
                let k = self.s.m.i32(if name.ends_with("i8") { 24 } else { 16 });
                let up = self.s.m.bin(c, Bin::Shl, a, k);
                (self.s.m.bin(c, Bin::AShr, up, k), None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    self.s.m.cmp(c, pred::IULT, r, a)
                } else {
                    // Signed overflow: both sources of one sign, the sum of
                    // the other (RDNA2 ISA, SOP2 2 and SOPK 15).
                    let x = self.s.m.bin(c, Bin::Xor, a, r);
                    let y = self.s.m.bin(c, Bin::Xor, b, r);
                    self.sign_set(x, y)
                };
                (r, Some(carry))
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = if name == "s_sub_u32" {
                    self.s.m.cmp(c, pred::IULT, a, b)
                } else {
                    // Signed overflow: sources of different signs, the
                    // difference not of the first's (RDNA2 ISA, SOP2 3).
                    let x = self.s.m.bin(c, Bin::Xor, a, b);
                    let y = self.s.m.bin(c, Bin::Xor, a, r);
                    self.sign_set(x, y)
                };
                (r, Some(carry))
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff0_i32_b32" | "s_bcnt0_i32_b32" => {
                let mut a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                a = self.s.m.bin(c, Bin::Xor, a, ones);
                let op = if name == "s_ff0_i32_b32" { 32 } else { 31 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff1_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let a = src32(self, 0)?;
                let op = if name == "s_ff1_i32_b32" { 32 } else { 31 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                // SCC: whether (S0 << N) + S1, taken in 64 bits, reaches
                // 2^32 (RDNA2 ISA, SOP2 46-49): the add's carry, or a bit
                // the shift moved out. Mesa ACO ignores the latter.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let n = u32::from(name.as_bytes()[6] - b'0');
                let k = self.s.m.i32(n);
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                let (r, carry) = self.add_sub_carry(c, true, t, b, None);
                let k = self.s.m.i32(32 - n);
                let out = self.s.m.bin(c, Bin::LShr, a, k);
                let z = self.s.m.i32(0);
                let out = self.s.m.cmp(c, pred::NE, out, z);
                (r, Some(self.s.m.bin(c, Bin::Or, carry, out)))
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                // SCC: whether the first operand was chosen (RDNA2 ISA).
                (self.s.m.select(c, t, a, b), Some(t))
            }
            "s_abs_i32" => {
                let a = src32(self, 0)?;
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_absdiff_i32" => {
                // D = S0 - S1 (wrapping), negated if negative; SCC = D != 0 (RDNA2 ISA, SOP2).
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let z = self.s.m.i32(0);
                let d = self.s.m.bin(c, Bin::Sub, a, b);
                let neg = self.s.m.cmp(c, pred::SLT, d, z);
                let n = self.s.m.bin(c, Bin::Sub, z, d);
                let r = self.s.m.select(c, neg, n, d);
                (r, Some(self.s.m.cmp(c, pred::NE, r, z)))
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            "s_pack_ll_b32_b16" | "s_pack_lh_b32_b16" | "s_pack_hh_b32_b16" => {
                // D = {S1 half, S0 half}, `ll`/`lh`/`hh` naming the S0 and S1
                // halves (RDNA2 ISA, SOP2 opcodes 50-52); no SCC.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k16 = self.s.m.i32(16);
                let a = if name == "s_pack_hh_b32_b16" {
                    self.s.m.bin(c, Bin::LShr, a, k16)
                } else {
                    let low = self.s.m.i32(0xFFFF);
                    self.s.m.bin(c, Bin::And, a, low)
                };
                let b = if name == "s_pack_ll_b32_b16" {
                    self.s.m.bin(c, Bin::Shl, b, k16)
                } else {
                    let high = self.s.m.i32(0xFFFF_0000);
                    self.s.m.bin(c, Bin::And, b, high)
                };
                (self.s.m.bin(c, Bin::Or, a, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cselect_b32"
                        | "s_cmov_b32"
                        | "s_cmovk_i32"
                        | "s_brev_b32"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                None if name.starts_with("s_add") || name.starts_with("s_sub") => {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// Whether `x & y` has its sign bit set.
    fn sign_set(&mut self, x: V, y: V) -> V {
        let c = self.cur;
        let both = self.s.m.bin(c, Bin::And, x, y);
        let z = self.s.m.i32(0);
        self.s.m.cmp(c, pred::SLT, both, z)
    }

    /// `S_WQM_B32` on a 32-bit mask dword: each 4-bit quad with any set bit becomes `0xf`.
    fn wqm_word(&mut self, c: usize, a: V) -> V {
        let (odd, even) = (self.s.m.i32(0xAAAA_AAAA), self.s.m.i32(0x5555_5555));
        let (hi2, lo2) = (self.s.m.i32(0xCCCC_CCCC), self.s.m.i32(0x3333_3333));
        let (one, two) = (self.s.m.i32(1), self.s.m.i32(2));
        let lo_odd = self.s.m.bin(c, Bin::And, a, odd);
        let lo_even = self.s.m.bin(c, Bin::And, a, even);
        let down1 = self.s.m.bin(c, Bin::LShr, lo_odd, one);
        let up1 = self.s.m.bin(c, Bin::Shl, lo_even, one);
        let pair_or = self.s.m.bin(c, Bin::Or, down1, up1);
        let p = self.s.m.bin(c, Bin::Or, a, pair_or);
        let p_hi2 = self.s.m.bin(c, Bin::And, p, hi2);
        let p_lo2 = self.s.m.bin(c, Bin::And, p, lo2);
        let down2 = self.s.m.bin(c, Bin::LShr, p_hi2, two);
        let up2 = self.s.m.bin(c, Bin::Shl, p_lo2, two);
        let quad_or = self.s.m.bin(c, Bin::Or, down2, up2);
        self.s.m.bin(c, Bin::Or, p, quad_or)
    }

    /// Packs whether each consecutive group of four pixels has a set bit.
    fn quad_mask(&mut self, c: usize, pixels: V) -> V {
        let one = self.s.m.i32(1);
        let shifted = self.s.m.bin(c, Bin::LShr, pixels, one);
        let mut mask = self.s.m.bin(c, Bin::Or, pixels, shifted);
        // Fold each nibble into its bottom bit, then gather those bits into
        // the low byte in three stages without needing a 64-bit operation.
        for (distance, keep) in [
            (2, 0x1111_1111),
            (3, 0x0303_0303),
            (6, 0x000f_000f),
            (12, 0x0000_00ff),
        ] {
            let distance = self.s.m.i32(distance);
            let shifted = self.s.m.bin(c, Bin::LShr, mask, distance);
            let combined = self.s.m.bin(c, Bin::Or, mask, shifted);
            let keep = self.s.m.i32(keep);
            mask = self.s.m.bin(c, Bin::And, combined, keep);
        }
        mask
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    fn add_sub_carry(&mut self, c: usize, add: bool, a: V, b: V, cin: Option<V>) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        if let Some(a) = self
            .scalar
            .get(&site)
            .copied()
            .filter(|a| a.param == USER_DATA)
        {
            return self.user_data_smem(inst, a);
        }
        if let Some(a) = self
            .scalar
            .get(&site)
            .copied()
            .filter(|a| a.param == MEMORY)
        {
            return self.memory_smem(inst, a);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        if matches!(src, Source::BuiltMemory { .. }) {
            let byte = u32::try_from(*offset)
                .map_err(|_| "a computed buffer descriptor with a negative scalar immediate")?;
            let zero = self.s.m.i32(0);
            let extra = self.operand(inst, soffset)?;
            let words = self.computed_buffer(inst, src, None, (extra, byte), zero, count)?;
            for (k, x) in words.into_iter().enumerate() {
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let kind = self.used_kind(src);
        let c = self.cur;
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // Constant data through the resource heap itself is a raw buffer: the
        // host's constant buffer views need 256-byte alignment, the title's
        // 16 (module notes, "Constant buffers in the heap").
        let heap_constants = kind.is_none() && self.is_heap_direct(src);
        // So is constant data in a table: the title's slot holds a raw view
        // of it, so tables need no copies of their own per draw.
        let table_constants =
            matches!(kind, Some(u) if u.kind == ResourceType::Cbv && u.root == RootType::Table);
        let srv = heap_constants
            || table_constants
            || matches!(kind, Some(u) if u.kind == ResourceType::Srv);
        // A root UAV is read as title memory in a program that uses it.
        let root_uav = self.root_memory
            && matches!(kind, Some(u) if u.kind == ResourceType::Uav)
            && matches!(src, Source::Built { address, .. }
                if self.x.native.layout.get(address.param).is_some_and(|p| p.root == RootType::Descriptor));
        if srv || root_uav {
            let (h, addr) = self.raw_view(site, src, addr, 4 * count, (11, 0), false, None)?;
            let c = self.cur;
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let mask = self.s.m.i8(1);
                let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, a, u, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let h = self.handle(site, src, false, (13, 0x1_0000))?;
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let mut rows: HashMap<u32, V> = HashMap::new();
        for k in 0..count {
            let x = match constant {
                Some(base) => {
                    let byte = base + 4 * k;
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            };
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// Where the program uses the descriptor that load `site` reads, and the load's first
    /// dword of it: the first resource descriptor (in their order) with that load among its
    /// sources. One table for all loads: searching every descriptor's sources for each load
    /// read as data, and again for each load of the same address, took a pixel shader with
    /// some hundreds of each tens of seconds.
    fn load_use(&self, site: Site) -> Option<(Site, u8)> {
        let uses = self.load_uses.get_or_init(|| {
            let mut uses = Map::default();
            for d in self.x.res.descriptors.iter().filter(|d| d.role == Role::Resource) {
                for l in d.source.leaves() {
                    if let Source::Loaded { load, first } = *l {
                        uses.entry(load).or_insert((d.site, first));
                    }
                }
            }
            uses
        });
        uses.get(&site).copied()
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program.
    fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        // What the host has no value for, by dword, so a refusal log says which is read.
        const BUFFER_ADDRESS: Why =
            "a descriptor's address or format read as data: a buffer's base address (dword 0)";
        const BUFFER_DWORD1: Why = "a descriptor's address or format read as data: a buffer's \
                                    dword 1 (address high bits, stride, swizzle)";
        const BUFFER_FORMAT: Why =
            "a descriptor's address or format read as data: a buffer's dword 3 (format and type)";
        const IMAGE_ADDRESS: Why =
            "a descriptor's address or format read as data: an image's base address (dword 0)";
        const IMAGE_REST: Why =
            "a descriptor's address or format read as data: an image's dwords 5 to 7";
        const OUTSIDE: Why =
            "a descriptor's address or format read as data: a dword outside the descriptor";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // How the program uses the descriptor tells the view's kind; a load
        // read only as data takes it from a load of the same descriptor that
        // the program uses (the compiler loads a descriptor again rather than
        // keep it, and queries sizes in loops of their own).
        let use_of = |site: Site| self.load_use(site);
        // The VGPR a waterfall loop reads the first lane of.
        let lane_source = |v: Value| {
            let i = self.def_inst(v)?;
            match i.fields {
                Fields::Valu(ref f) if i.name() == "v_readfirstlane_b32" => match f.src[..] {
                    [Operand::Vector { reg, .. }] => i.read(Reg::V(reg)),
                    _ => None,
                },
                _ => None,
            }
        };
        let same = |a: &Address, b: &Address| {
            a.param == b.param
                && a.offset == b.offset
                && (a.dynamic == b.dynamic
                    || a.dynamic.and_then(lane_source).is_some()
                        && a.dynamic.and_then(lane_source) == b.dynamic.and_then(lane_source))
        };
        let address = self.scalar.get(&site);
        let use_at = use_of(site).or_else(|| {
            self.x
                .res
                .scalar
                .iter()
                .filter(|&(s, a)| *s != site && address.is_some_and(|b| same(a, b)))
                .find_map(|&(s, _)| use_of(s))
        });
        let Some((at, first)) = use_at else {
            return self.view_addresses(inst, site, code, count);
        };
        let src = Source::Loaded { load: site, first };
        let used = self.used_kind(&src);
        let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
        let un = ui.name();
        let uav = un.contains("store")
            || un.contains("atomic")
            || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let (props, image) = match &ui.fields {
            Fields::Image { dim, .. } => {
                let kind = match dim {
                    0 => 1,
                    1 => 2,
                    2 => 4,
                    4 => 6,
                    5 => 7,
                    _ => return Err("an image dimension (cube or MSAA)"),
                };
                ((kind | u, 0x409), Some(kind))
            }
            Fields::Buffer { idxen, format, .. } => {
                let stride = used.map_or(0, |u| u.stride);
                let props = if un.starts_with("buffer_load_format")
                    || un.starts_with("buffer_store_format")
                {
                    (10 | u, 0x409)
                } else if *idxen && format.is_none() && stride >= 4 && stride.is_multiple_of(4) {
                    (12 | u, stride)
                } else {
                    (11 | u, 0)
                };
                if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                    return Err("a constant buffer's descriptor read as data");
                }
                (props, None)
            }
            // A buffer the code reads with scalar loads (`buf[buf.Length - 1]`
            // at a uniform index): structured by the used-resource entry's
            // stride, so its record count is in elements, else raw (bytes).
            Fields::Smem { .. } => {
                let used = used.ok_or("a scalar-loaded buffer of unknown kind read as data")?;
                if used.kind == ResourceType::Cbv {
                    return Err("a constant buffer's descriptor read as data");
                }
                let props = match used.stride {
                    0 | 1 => (11 | u, 0),
                    s if s.is_multiple_of(4) => (12 | u, s),
                    _ => return Err("a scalar-loaded buffer of odd stride read as data"),
                };
                (props, None)
            }
            _ => return Err("a scalar memory read as data"),
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        // An image's fields are its view's own size, level 0 of the host view, writable or not:
        // Metal answers the undefined level DXC writes for a UAV with 0 (see `image_op`).
        let mip = if image.is_some() {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    _ => (field(self, 1), field(self, 2)),
                };
                let levels = if uav { one } else { field(self, 3) };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => vec![(2, field(self, 0))],
        };
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    let why = match (image.is_some(), rel) {
                        (false, 0) => BUFFER_ADDRESS,
                        (false, 1) => BUFFER_DWORD1,
                        (false, 3) => BUFFER_FORMAT,
                        (true, 0) => IMAGE_ADDRESS,
                        (true, 5..=7) => IMAGE_REST,
                        _ => OUTSIDE,
                    };
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(why));
                    }
                }
            }
        }
        Ok(())
    }

    /// `s_load_dword*` from the root signature's memory: user-data dwords.
    fn user_data_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = smem_dwords(inst.name()).ok_or("a scalar load from the root signature")?;
        for k in 0..count {
            let index = a.offset / 4 + k;
            let held = self.x.native.layout.iter().enumerate().find_map(|(i, p)| {
                let dword = index.checked_sub(p.user_data.into())?;
                (dword < p.sgprs).then_some((i, dword))
            });
            let val = match held {
                Some((param, dword)) => self.user(param, dword),
                None => Val::Missing("a root-signature dword no parameter holds"),
            };
            if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                self.vals.insert(v, val);
            }
        }
        Ok(())
    }

    /// `s_load_dword*` from title memory at an address the program computes.
    fn memory_smem(&mut self, inst: &Inst, _a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            base: Operand::Scalar { code: base, .. },
            offset,
            soffset,
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = smem_dwords(inst.name()).ok_or("a scalar load from title memory")?;
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst
                .read(Reg::S(base + k))
                .ok_or("a scalar memory address")?;
            l.i32_of(v)
        };
        let (lo, hi) = (word(self, 0)?, word(self, 1)?);
        // The base's high dword holds address bits 32-47.
        let mask = self.s.m.i32(0xFFFF);
        let hi = self.s.m.bin(self.cur, Bin::And, hi, mask);
        // The immediate is signed (21 bits), the offset register unsigned.
        let mut extra = vec![(self.s.m.i32(offset as u32), offset < 0)];
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            extra.push((self.operand(inst, &soffset)?, false));
        }
        let (mut lo, mut hi) = (lo, hi);
        for (x, negative) in extra {
            (lo, hi) = self.add64(lo, hi, x, negative);
        }
        let (h, off) = self.memory(lo, hi, 4 * count, false, None);
        for k in 0..count {
            let four = self.s.m.i32(4 * k);
            let at = self.s.m.bin(self.cur, Bin::Add, off, four);
            let x = self.raw_dword(h, at);
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// `(hi:lo) + x` with `x` zero-extended, or sign-extended when
    /// `negative` says its top bit means it.
    fn add64(&mut self, lo: V, hi: V, x: V, negative: bool) -> (V, V) {
        let c = self.cur;
        let sum = self.s.m.bin(c, Bin::Add, lo, x);
        let carry = self.s.m.cmp(c, pred::IULT, sum, lo);
        let carry = self.s.m.cast(c, Cast::ZExt, carry, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::Add, hi, carry);
        let hi = if negative {
            let all = self.s.m.i32(u32::MAX);
            self.s.m.bin(c, Bin::Add, hi, all)
        } else {
            hi
        };
        (sum, hi)
    }

    /// A 64-bit image atomic `op` of `value` (cmpswap: replacing
    /// `compare`) on the texel at `cs` of the globally coherent R32G32_UINT
    /// view `h`, under a lock word (`emit`, "Atomic locks"). The wave's
    /// lanes take turns in a loop over lane indices that every lane runs
    /// alike: in its turn a lane takes its texel's lock, reads, writes and
    /// frees it. Lanes of one wave never wait for each other's locks (that
    /// never finishes on Apple GPUs), and the loop needs no wave operation
    /// in divergent code (D3DMetal's return other lanes' values there).
    /// Returns the old value's low and high dwords.
    fn locked_atomic64(
        &mut self,
        h: V,
        cs: &[V],
        coords: usize,
        op: &str,
        value: V,
        compare: Option<V>,
    ) -> (V, V) {
        let i32t = self.s.t.i32;
        let i64t = self.s.m.int(64);
        let void = self.s.t.void;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        self.s.flags |= flags::WAVE_OPS | flags::TYPED_UAV_LOADS | flags::INT64;
        let c = self.cur;
        // The lock: a hash of the coordinates.
        let mut hash = k(self, 0);
        for (&x, m) in cs
            .iter()
            .take(coords)
            .zip([0x9E37_79B1u32, 0x85EB_CA77, 0xC2B2_AE3D])
        {
            let m = k(self, m);
            let x = self.s.m.bin(c, Bin::Mul, x, m);
            hash = self.s.m.bin(c, Bin::Xor, hash, x);
            let k15 = k(self, 15);
            let r = self.s.m.bin(c, Bin::LShr, hash, k15);
            hash = self.s.m.bin(c, Bin::Xor, hash, r);
        }
        let mask = k(self, super::ATOMIC_LOCKS - 1);
        let slot = self.s.m.bin(c, Bin::And, hash, mask);
        let k2 = k(self, 2);
        let slot = self.s.m.bin(c, Bin::Shl, slot, k2);
        let base = k(self, super::ATOMIC_LOCKS_AT);
        let lock = self.s.m.bin(c, Bin::Add, slot, base);
        let locks = self.s.view_writes_handle();
        let me = self
            .s
            .op(c, "waveGetLaneIndex", Attr::ReadNone, i32t, 111, &[]);
        let lanes = self
            .s
            .op(c, "waveGetLaneCount", Attr::ReadNone, i32t, 112, &[]);
        let mut blocks = [0; 5];
        for b in &mut blocks {
            *b = self.s.m.new_block();
        }
        let [head, spin, held, latch, done] = blocks;
        self.s.m.terminate(c, Term::Br(head));
        let iu = self.s.m.undef(i32t);
        let (zero, one) = (k(self, 0), k(self, 1));
        let fence = k(self, 2 | 8);

        // Head: lane `i`'s turn.
        let i = self.s.m.phi(head, i32t);
        let plo = self.s.m.phi(head, i32t);
        let phi_ = self.s.m.phi(head, i32t);
        let turn = self.s.m.cmp(head, pred::EQ, me, i);
        self.s.m.terminate(head, Term::CondBr(turn, spin, latch));

        // Spin, bounded (past it, on without the lock).
        let got = self.s.op(
            spin,
            "atomicCompareExchange.i32",
            Attr::NoUnwind,
            i32t,
            79,
            &[locks, lock, iu, iu, zero, one],
        );
        let won = self.s.m.cmp(spin, pred::EQ, got, zero);
        let tries = self.s.m.phi(spin, i32t);
        let more = self.s.m.bin(spin, Bin::Add, tries, one);
        self.s.m.set_phi(tries, vec![(zero, head), (more, spin)]);
        let limit = k(self, 1 << 16);
        let out = self.s.m.cmp(spin, pred::IUGE, more, limit);
        let leave = self.s.m.bin(spin, Bin::Or, won, out);
        self.s.m.terminate(spin, Term::CondBr(leave, held, spin));

        // Held: fence, read, write, fence, free.
        self.s
            .op(held, "barrier", Attr::NoDuplicate, void, 80, &[fence]);
        let mut args = vec![h, iu];
        args.extend(cs);
        args.extend([iu, iu, iu]);
        let r = self.s.op(
            held,
            "textureLoad.i32",
            Attr::ReadOnly,
            self.s.t.res_ret_i32,
            66,
            &args,
        );
        let lo = self.s.m.extract(held, r, 0);
        let hi = self.s.m.extract(held, r, 1);
        let l64 = self.s.m.cast(held, Cast::ZExt, lo, i64t);
        let h64 = self.s.m.cast(held, Cast::ZExt, hi, i64t);
        let k32 = self.s.m.int_const(i64t, 32);
        let h64 = self.s.m.bin(held, Bin::Shl, h64, k32);
        let old = self.s.m.bin(held, Bin::Or, l64, h64);
        let m = &mut self.s.m;
        let pick = |m: &mut Module, p: u8| {
            let keep = m.cmp(held, p, old, value);
            m.select(held, keep, old, value)
        };
        let new = match op {
            "add" | "sub" => m.bin(held, Bin::Add, old, value),
            "and" => m.bin(held, Bin::And, old, value),
            "or" => m.bin(held, Bin::Or, old, value),
            "xor" => m.bin(held, Bin::Xor, old, value),
            "smin" => pick(m, pred::SLT),
            "smax" => pick(m, pred::SGT),
            "umin" => pick(m, pred::IULT),
            "umax" => pick(m, pred::IUGT),
            "swap" => value,
            _ => {
                let same = m.cmp(held, pred::EQ, old, compare.expect("a compare value"));
                m.select(held, same, value, old)
            }
        };
        let nlo = self.s.m.cast(held, Cast::Trunc, new, i32t);
        let nhi = self.s.m.bin(held, Bin::LShr, new, k32);
        let nhi = self.s.m.cast(held, Cast::Trunc, nhi, i32t);
        let mut args = vec![h];
        args.extend(cs);
        // DXIL stores all four channels; an R32G32 view keeps two.
        args.extend([nlo, nhi, nlo, nhi, self.s.m.i8(0xF)]);
        self.s
            .op(held, "textureStore.i32", Attr::NoUnwind, void, 67, &args);
        self.s
            .op(held, "barrier", Attr::NoDuplicate, void, 80, &[fence]);
        // Frees it only if taken: an `and` with all ones changes nothing.
        let (and, all) = (k(self, 1), k(self, u32::MAX));
        let keep = self.s.m.select(held, won, zero, all);
        self.s.op(
            held,
            "atomicBinOp.i32",
            Attr::NoUnwind,
            i32t,
            78,
            &[locks, and, lock, iu, iu, keep],
        );
        self.s.m.terminate(held, Term::Br(latch));

        // Latch: the next lane.
        let olo = self.s.m.phi(latch, i32t);
        self.s.m.set_phi(olo, vec![(lo, held), (plo, head)]);
        let ohi = self.s.m.phi(latch, i32t);
        self.s.m.set_phi(ohi, vec![(hi, held), (phi_, head)]);
        let next = self.s.m.bin(latch, Bin::Add, i, one);
        let more = self.s.m.cmp(latch, pred::IULT, next, lanes);
        self.s.m.set_phi(i, vec![(zero, c), (next, latch)]);
        self.s.m.set_phi(plo, vec![(iu, c), (olo, latch)]);
        self.s.m.set_phi(phi_, vec![(iu, c), (ohi, latch)]);
        self.s.m.terminate(latch, Term::CondBr(more, head, done));
        self.cur = done;
        (olo, ohi)
    }

    /// The dword at byte `at` of raw buffer `h`.
    fn raw_dword(&mut self, h: V, at: V) -> V {
        let c = self.cur;
        let (u, mask, al) = (
            self.s.m.undef(self.s.t.i32),
            self.s.m.i8(1),
            self.s.m.i32(4),
        );
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, at, u, mask, al],
        );
        let x = self.s.m.extract(c, r, 0);
        self.as_int(x)
    }

    /// Marks the pages a write of `bytes` bytes at byte `x` of the unordered
    /// buffer view `h` touches, in the lanes `guard` lets through (module
    /// notes, "Write marks"): the bits of its first and last byte, each set
    /// only where still clear. Nothing for handles that are no heap view, or
    /// without [`Parts::view_marks`].
    fn mark_view_write(&mut self, h: V, x: V, bytes: u32, guard: Option<V>) {
        if !self.x.view_marks {
            return;
        }
        let Some(&index) = self.s.view_index.get(&h) else {
            return;
        };
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let table = self.s.view_marks_handle();
        let c = self.cur;
        let (eight, four) = (k(self, 8), k(self, 4));
        let at = self.s.m.bin(c, Bin::Mul, index, eight);
        let base = self.raw_dword(table, at);
        let at = self.s.m.bin(c, Bin::Add, at, four);
        let phase = self.raw_dword(table, at);
        let none = k(self, super::VIEW_MARK_NONE);
        let valid = self.s.m.cmp(c, pred::NE, base, none);
        let active = match guard {
            Some(g) => self.s.m.bin(c, Bin::And, g, valid),
            None => valid,
        };
        let first = self.s.m.bin(c, Bin::Add, phase, x);
        let span = k(self, bytes.max(1) - 1);
        let last = self.s.m.bin(c, Bin::Add, first, span);
        let (k16, cap) = (k(self, 16), k(self, super::VIEW_WRITE_BITS - 1));
        let mut bits = Vec::new();
        for byte in [first, last] {
            let page = self.s.m.bin(c, Bin::LShr, byte, k16);
            let n = self.s.m.bin(c, Bin::Add, base, page);
            let over = self.s.m.cmp(c, pred::IUGT, n, cap);
            bits.push(self.s.m.select(c, over, cap, n));
        }
        let bitmap = self.s.view_writes_handle();
        let outer = self.begin_guarded(Some(active));
        for (i, &n) in bits.iter().enumerate() {
            let c = self.cur;
            let (k5, k2, k31, one, zero) =
                (k(self, 5), k(self, 2), k(self, 31), k(self, 1), k(self, 0));
            let word = self.s.m.bin(c, Bin::LShr, n, k5);
            let word = self.s.m.bin(c, Bin::Shl, word, k2);
            let bit = self.s.m.bin(c, Bin::And, n, k31);
            let bit = self.s.m.bin(c, Bin::Shl, one, bit);
            // Most writes find their page marked: a load, not an atomic.
            let old = self.raw_dword(bitmap, word);
            let set = self.s.m.bin(c, Bin::And, old, bit);
            let mut clear = self.s.m.cmp(c, pred::EQ, set, zero);
            if i == 1 {
                let other = self.s.m.cmp(c, pred::NE, n, bits[0]);
                clear = self.s.m.bin(c, Bin::And, clear, other);
            }
            let inner = self.begin_guarded(Some(clear));
            let iu = self.s.m.undef(self.s.t.i32);
            // Operation 2 is OR.
            self.s.op(
                self.cur,
                "atomicBinOp.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                78,
                &[bitmap, k2, word, iu, iu, bit],
            );
            self.end_guarded(inner, &[]);
        }
        self.end_guarded(outer, &[]);
    }

    /// The byte stride of the buffer view at heap index `index`, from the
    /// view-stride table (`t`[`VIEW_STRIDES`]).
    fn stride_at(&mut self, index: V) -> V {
        let word = self.view_word(index);
        let bits = self.s.m.i32(VIEW_STRIDE_BITS);
        self.s.m.bin(self.cur, Bin::And, word, bits)
    }

    /// Dword `index` of the view-stride table: the stride and the format
    /// kind bits ([`VIEW_INTEGER`], [`VIEW_SIGNED`]).
    fn view_word(&mut self, index: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let at = self.s.m.bin(c, Bin::Mul, index, four);
        let table = self.s.root_srv_handle(VIEW_STRIDES);
        self.raw_dword(table, at)
    }

    /// Whether the typed buffer view a handle names has an integer format,
    /// and whether a signed one, from the view-stride table; `None` if the
    /// handle's heap index is not known.
    fn integer_view(&mut self, h: V) -> Option<(V, V)> {
        let index = *self.s.view_index.get(&h)?;
        let word = self.view_word(index);
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let [int, signed] = [VIEW_INTEGER, VIEW_SIGNED].map(|bit| {
            let bit = self.s.m.i32(bit);
            let x = self.s.m.bin(c, Bin::And, word, bit);
            self.s.m.cmp(c, pred::NE, x, zero)
        });
        Some((int, signed))
    }

    /// Title memory at a raw GPU address (decision 0005: the unified-memory
    /// item). Every access to an address the program computes goes through
    /// here: scalar and global loads, stores and atomics, the BVH node
    /// fetch, and in a program that uses title memory its root SRVs and UAVs
    /// ([`Self::raw_view`]). `(hi:lo)` is the 64-bit address of the access's
    /// first byte (`hi` holds bits 32-47), `bytes` its size, `write` whether
    /// it stores or does atomics, `active` the lanes that access (`None`:
    /// all; the others look up but mark nothing). Returns a raw buffer handle
    /// (read-write with `write`) and the address's byte offset in it; the
    /// access's dwords follow at `+ 4k`. Memory no host copy holds is the
    /// null view: reads give 0, writes are dropped. Contract (`emit` "Title
    /// memory"): a page table in the root SRV `t0` of
    /// [`super::shader::MEMORY_SPACE`]; the root UAV `u0` there takes a bit
    /// per page written and per page or 256 MiB region missed. May end in a
    /// new block: callers take [`Lower::cur`] afresh.
    // shortcut: one lookup per access, so its bytes are taken to lie in one
    // host copy of title memory (any access within a 64 KiB page does).
    pub(crate) fn memory(
        &mut self,
        lo: V,
        hi: V,
        bytes: u32,
        write: bool,
        active: Option<V>,
    ) -> (V, V) {
        let e = self.memory_entry(lo, hi, write);
        if self.x.zero_copy_memory {
            return (e.handle, e.offset);
        }
        let c = self.cur;
        // Writes always mark (written or missed pages), reads only misses.
        let marks = match (write, active) {
            (true, Some(a)) => Some(a),
            (true, None) => None,
            (false, Some(a)) => Some(self.s.m.bin(c, Bin::And, e.miss, a)),
            (false, None) => Some(e.miss),
        };
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        // A miss marks its page's entry, or with no block its region's
        // level-1 index (bits 0-4095, block 0's entries, which no copy has).
        let zero = k(self, 0);
        let empty = self.s.m.cmp(c, pred::EQ, e.block, zero);
        let missed = self.s.m.select(c, empty, e.l1, e.entry);
        let first = self.s.m.select(c, e.miss, missed, e.entry);
        let mut bits = vec![first];
        if write {
            // A write marks the pages of its first and last byte (one block).
            let (last, k16) = (k(self, bytes - 1), k(self, 16));
            let end = self.s.m.bin(c, Bin::Add, lo, last);
            let a = self.s.m.bin(c, Bin::LShr, lo, k16);
            let b = self.s.m.bin(c, Bin::LShr, end, k16);
            let pages = self.s.m.bin(c, Bin::Sub, b, a);
            let next = self.s.m.bin(c, Bin::Add, e.entry, pages);
            bits.push(self.s.m.select(c, e.miss, first, next));
        }
        let bitmap = self.s.memory_dirty_handle();
        let at = self.begin_guarded(marks);
        let c = self.cur;
        let (k5, k31, k2, one) = (k(self, 5), k(self, 31), k(self, 2), k(self, 1));
        let iu = self.s.m.undef(self.s.t.i32);
        for n in bits {
            // Byte offset of the bitmap dword; operation 2 is OR.
            let word = self.s.m.bin(c, Bin::LShr, n, k5);
            let word = self.s.m.bin(c, Bin::Shl, word, k2);
            let bit = self.s.m.bin(c, Bin::And, n, k31);
            let bit = self.s.m.bin(c, Bin::Shl, one, bit);
            self.s.op(
                c,
                "atomicBinOp.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                78,
                &[bitmap, k2, word, iu, iu, bit],
            );
        }
        self.end_guarded(at, &[]);
        (e.handle, e.offset)
    }

    /// [`Self::memory`]'s lookup in the page table.
    fn memory_entry(&mut self, lo: V, hi: V, write: bool) -> MemoryEntry {
        let c = self.cur;
        let map = self.s.memory_handle();
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        // Level 1: (hi:lo) >> 28, 4096 entries for 40-bit addresses.
        let (k4, k28, k256) = (k(self, 4), k(self, 28), k(self, 256));
        let top = self.s.m.bin(c, Bin::Shl, hi, k4);
        let low = self.s.m.bin(c, Bin::LShr, lo, k28);
        let l1 = self.s.m.bin(c, Bin::Or, top, low);
        let inside = self.s.m.cmp(c, pred::IULT, hi, k256);
        let zero = k(self, 0);
        let l1 = self.s.m.select(c, inside, l1, zero);
        let two = k(self, 2);
        let at = self.s.m.bin(c, Bin::Shl, l1, two);
        let block = self.raw_dword(map, at);
        let block = self.s.m.select(c, inside, block, zero);
        // Level 2: block b holds 4096 entries of two dwords after level 1.
        let (k16, k12, mask, k3, first) = (
            k(self, 16),
            k(self, 12),
            k(self, 0xFFF),
            k(self, 3),
            k(self, 4096 * 4),
        );
        let page = self.s.m.bin(c, Bin::LShr, lo, k16);
        let page = self.s.m.bin(c, Bin::And, page, mask);
        let entry = self.s.m.bin(c, Bin::Shl, block, k12);
        let entry = self.s.m.bin(c, Bin::Or, entry, page);
        let at = self.s.m.bin(c, Bin::Shl, entry, k3);
        let at = self.s.m.bin(c, Bin::Add, at, first);
        let view = self.raw_dword(map, at);
        let four = k(self, 4);
        let at = self.s.m.bin(c, Bin::Add, at, four);
        let start = self.raw_dword(map, at);
        // Block 0's first entry names the null pair: no copy holds the page.
        let null = self.raw_dword(map, first);
        let miss = self.s.m.cmp(c, pred::EQ, view, null);
        // Views come in pairs: the raw SRV, then the raw UAV of one copy.
        let (view, props) = if write {
            let one = k(self, 1);
            (self.s.m.bin(c, Bin::Add, view, one), (11 | 0x1000, 0))
        } else {
            (view, (11, 0))
        };
        let handle = self.s.heap_handle(c, view, false, true, props);
        // Pages written here are marked in the page table's bitmap.
        self.s.view_index.remove(&handle);
        MemoryEntry {
            handle,
            offset: self.s.m.bin(c, Bin::Sub, lo, start),
            entry,
            block,
            l1,
            miss,
        }
    }

    /// A descriptor load from a table or heap whose dwords the program only
    /// reads as numbers: the base address of a buffer view (dwords 0 and 1,
    /// as a pointer to title memory: the BVH's address in its SRV) or, for
    /// a whole image descriptor (eight dwords), also the size fields of
    /// dwords 1-3 (width, height, first and last level, type: size queries
    /// of a texture the program never samples). They come from the host's
    /// view addresses, `t1` in [`super::shader::MEMORY_SPACE`]: four dwords
    /// per heap index.
    // shortcut: dword 1 holds the address bits only (stride and swizzle read
    // 0) or a texture's width bits, dword 3 a texture's size fields (format,
    // tiling and swizzle choice read 0 or the identity); dword 4 on and a
    // buffer's dwords 2-3 (record count, format) stop the program when read;
    // the table offset's computed part is taken as whole descriptors
    // (32-byte multiples).
    fn view_addresses(&mut self, inst: &Inst, site: Site, code: u8, count: u32) -> Result<(), Why> {
        const OTHER: Why = "a descriptor read only as data";
        let a = *self
            .scalar
            .get(&site)
            .ok_or("a descriptor read only as data")?;
        if !matches!(
            self.x.native.layout.get(a.param).map(|p| p.root),
            Some(RootType::Table | RootType::ResourceHeap)
        ) {
            return Err("a descriptor read only as data");
        }
        // The compiler loads image descriptors whole, buffer ones as four
        // dwords.
        let known = if count == 8 && a.offset.is_multiple_of(32) {
            4
        } else {
            2
        };
        let views = self.s.view_addresses_handle();
        for k in 0..count {
            let word = (a.offset / 4 + k) % 8;
            let val = if word < known {
                let index = self.heap_index(site, k as u8, false)?;
                let c = self.cur;
                let four = self.s.m.i32(4);
                let at = self.s.m.bin(c, Bin::Shl, index, four);
                let w = self.s.m.i32(4 * word);
                let at = self.s.m.bin(c, Bin::Add, at, w);
                Val::I32(self.raw_dword(views, at))
            } else {
                Val::Missing(OTHER)
            };
            if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                self.vals.insert(v, val);
            }
        }
        Ok(())
    }

    /// `s_load_dword*` from the program's constant table.
    fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned.
    fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            // A normalized format's ends are exactly 1.0 and -1.0. The host
            // may divide through a reciprocal and land just below 1.0: RDNA
            // 4 loaded 16_16_UNORM's 65535 as the half 0x3BFF, not 0x3C00.
            let x = match numeric {
                Numeric::Unorm => {
                    let top = (1u64 << w) - 1;
                    let q = if w <= 16 {
                        let q = self.norm_quotient(field, w);
                        self.as_float(q)
                    } else {
                        let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                        let max = self.s.m.f32((top as f32).to_bits());
                        self.s.m.bin(c, Bin::SDiv, f, max)
                    };
                    let (high, one) = (self.s.m.i32(top as u32), self.s.m.f32(1.0f32.to_bits()));
                    let at_top = self.s.m.cmp(c, pred::EQ, field, high);
                    let q = self.s.m.select(c, at_top, one, q);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let top = (1u32 << (w - 1)) - 1;
                    let q = if w <= 17 {
                        // The magnitude's quotient, signed (codes at and
                        // below -top are -1.0 below).
                        let (zero, sign) = (self.s.m.i32(0), self.s.m.i32(0x8000_0000));
                        let neg = self.s.m.cmp(c, pred::SLT, s, zero);
                        let minus = self.s.m.bin(c, Bin::Sub, zero, s);
                        let m = self.s.m.select(c, neg, minus, s);
                        let q = self.norm_quotient(m, w - 1);
                        let sign = self.s.m.select(c, neg, sign, zero);
                        let q = self.s.m.bin(c, Bin::Or, q, sign);
                        self.as_float(q)
                    } else {
                        let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                        let max = self.s.m.f32((top as f32).to_bits());
                        self.s.m.bin(c, Bin::SDiv, f, max)
                    };
                    // The lowest value is -1.0 too, as is the one above it.
                    let (high, one) = (self.s.m.i32(top), self.s.m.f32(1.0f32.to_bits()));
                    let at_top = self.s.m.cmp(c, pred::SGE, s, high);
                    let q = self.s.m.select(c, at_top, one, q);
                    let low = self.s.m.i32(top.wrapping_neg());
                    let minus_one = self.s.m.f32((-1.0f32).to_bits());
                    let at_bottom = self.s.m.cmp(c, pred::SLE, s, low);
                    let q = self.s.m.select(c, at_bottom, minus_one, q);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// The bits of the float nearest to `a / d`, `d = 2^w - 1`, for codes
    /// `0 <= a < d` and `w <= 16`: what the console's buffer unit gives for
    /// a normalized code. The host's division is not exact (the RX 6700 XT
    /// returns 8-bit codes one ulp high and some 16-bit codes one ulp low),
    /// and titles truncate `x * d` back to the code (a material index in a
    /// vertex colour's alpha), which a quotient one ulp low turns into the
    /// code below.
    ///
    /// `Q = floor(a * 2^32 / d) = a * K + floor(a * r / d)` with `K =
    /// floor(2^32 / d)` and `r = 2^32 mod d` (no division when `r` is 1:
    /// widths dividing 32). `a / d` is never exact for `0 < a < d` (`d`
    /// odd, `r` coprime to it), so `Q` rounds to 24 bits up exactly when
    /// its highest dropped bit is set, and the result is that integer (at
    /// most 2^24, so its conversion is exact on any host) scaled by a power
    /// of two; `Q` itself when it fits, the quotient's further bits then
    /// lying below half an ulp. That holds for every code of every width but
    /// 11 and 12 (checked for all codes of widths 2 to 16), which take
    /// [`Self::exact_norm`]. The host's conversion of a larger integer is
    /// not relied on: WARP rounds those above 2^31 otherwise.
    fn norm_quotient(&mut self, a: V, w: u32) -> V {
        if matches!(w, 11 | 12) {
            return self.exact_norm(a, w);
        }
        let c = self.cur;
        let d = (1u64 << w) - 1;
        let (kk, r) = ((1u64 << 32) / d, (1u64 << 32) % d);
        let k = |l: &mut Self, v: u64| l.s.m.i32(v as u32);
        let (one, kv) = (k(self, 1), k(self, kk));
        let mut q = self.s.m.bin(c, Bin::Mul, a, kv);
        if r != 1 {
            let (rv, dv) = (k(self, r), k(self, d));
            let t = self.s.m.bin(c, Bin::Mul, a, rv);
            let more = self.s.m.bin(c, Bin::UDiv, t, dv);
            q = self.s.m.bin(c, Bin::Add, q, more);
        }
        // The bits to drop (1 to 8) from the exponent of `q >> 8`, which
        // converts exactly (fewer than 25 bits); none when `q` fits.
        let k8 = k(self, 8);
        let top = self.s.m.bin(c, Bin::LShr, q, k8);
        let top = self.s.m.cast(c, Cast::UiToFp, top, self.s.t.f32);
        let top = self.as_int(top);
        let k23 = k(self, 23);
        let e = self.s.m.bin(c, Bin::LShr, top, k23);
        let bias = k(self, 127 + 15);
        let shift = self.s.m.bin(c, Bin::Sub, e, bias);
        let (zero, b24) = (k(self, 0), k(self, 1 << 24));
        let fits = self.s.m.cmp(c, pred::IULT, q, b24);
        let shift = self.s.m.select(c, fits, zero, shift);
        // Rounded up when the highest dropped bit is set: `a / d` is never
        // exact, so a dropped half is more than half, never a tie.
        let up = self.s.m.bin(c, Bin::Shl, q, one);
        let up = self.s.m.bin(c, Bin::LShr, up, shift);
        let up = self.s.m.bin(c, Bin::And, up, one);
        let m = self.s.m.bin(c, Bin::LShr, q, shift);
        let m = self.s.m.bin(c, Bin::Add, m, up);
        // At most 2^24: exact as a float, and so its scale by 2^(shift - 32).
        let f = self.s.m.cast(c, Cast::UiToFp, m, self.s.t.f32);
        let b95 = k(self, 127 - 32);
        let scale = self.s.m.bin(c, Bin::Add, shift, b95);
        let scale = self.s.m.bin(c, Bin::Shl, scale, k23);
        let scale = self.as_float(scale);
        let f = self.s.m.bin(c, Bin::Mul, f, scale);
        self.as_int(f)
    }

    /// [`Self::norm_quotient`] in integers only, for any width up to 16: `Q`
    /// as there, 8 more bits when that leaves fewer than 25 (codes of wide
    /// formats near 0), then rounded to 24 bits to nearest even with the
    /// remainder as sticky bit, and assembled as float bits.
    fn exact_norm(&mut self, a: V, w: u32) -> V {
        let c = self.cur;
        let d = (1u64 << w) - 1;
        let (kk, r) = ((1u64 << 32) / d, (1u64 << 32) % d);
        let k = |l: &mut Self, v: u64| l.s.m.i32(v as u32);
        let (one, zero, dv, k8) = (k(self, 1), k(self, 0), k(self, d), k(self, 8));
        let kv = k(self, kk);
        let mut q = self.s.m.bin(c, Bin::Mul, a, kv);
        let mut rem = a;
        if r != 1 {
            let rv = k(self, r);
            let t = self.s.m.bin(c, Bin::Mul, a, rv);
            let more = self.s.m.bin(c, Bin::UDiv, t, dv);
            q = self.s.m.bin(c, Bin::Add, q, more);
            rem = self.s.m.bin(c, Bin::URem, t, dv);
        }
        // The exponent's bias less the 32 (or 40) fraction bits of `q`; the
        // shift below is added.
        let mut bias = k(self, 127 + 23 - 32);
        if kk < 1 << 24 {
            // Fewer than 25 bits for the smallest codes: 8 more.
            let t = self.s.m.bin(c, Bin::Shl, rem, k8);
            let more = self.s.m.bin(c, Bin::UDiv, t, dv);
            let rest = self.s.m.bin(c, Bin::URem, t, dv);
            let wide = self.s.m.bin(c, Bin::Shl, q, k8);
            let wide = self.s.m.bin(c, Bin::Or, wide, more);
            let limit = k(self, 1 << 24);
            let small = self.s.m.cmp(c, pred::IULT, q, limit);
            q = self.s.m.select(c, small, wide, q);
            rem = self.s.m.select(c, small, rest, rem);
            let b40 = k(self, 127 + 23 - 40);
            bias = self.s.m.select(c, small, b40, bias);
        }
        // Bits to drop: the bit length less 24, 1 to 8.
        let mut shift = one;
        for bit in 25..32 {
            let at = k(self, 1 << bit);
            let ge = self.s.m.cmp(c, pred::IUGE, q, at);
            let add = self.s.m.select(c, ge, one, zero);
            shift = self.s.m.bin(c, Bin::Add, shift, add);
        }
        let mant = self.s.m.bin(c, Bin::LShr, q, shift);
        let below = self.s.m.bin(c, Bin::Sub, shift, one);
        let half = self.s.m.bin(c, Bin::LShr, q, below);
        let half = self.s.m.bin(c, Bin::And, half, one);
        let low = self.s.m.bin(c, Bin::Shl, one, below);
        let low = self.s.m.bin(c, Bin::Sub, low, one);
        let low = self.s.m.bin(c, Bin::And, q, low);
        let rest = self.s.m.bin(c, Bin::Or, low, rem);
        let sticky = self.s.m.cmp(c, pred::NE, rest, zero);
        let odd = self.s.m.bin(c, Bin::And, mant, one);
        let tie = self.s.m.select(c, sticky, one, odd);
        let up = self.s.m.bin(c, Bin::And, half, tie);
        let mant = self.s.m.bin(c, Bin::Add, mant, up);
        // Rounding up to 2^24 moves to the next exponent.
        let k24 = k(self, 24);
        let carry = self.s.m.bin(c, Bin::LShr, mant, k24);
        let mant = self.s.m.bin(c, Bin::LShr, mant, carry);
        let e = self.s.m.bin(c, Bin::Add, bias, shift);
        let e = self.s.m.bin(c, Bin::Add, e, carry);
        let k23 = k(self, 23);
        let e = self.s.m.bin(c, Bin::Shl, e, k23);
        let fraction = k(self, 0x7F_FFFF);
        let m = self.s.m.bin(c, Bin::And, mant, fraction);
        let bits = self.s.m.bin(c, Bin::Or, e, m);
        let is_zero = self.s.m.cmp(c, pred::EQ, a, zero);
        self.s.m.select(c, is_zero, zero, bits)
    }

    /// Channels of buffer format `format`, as [`Self::format_load`] gives
    /// them, as d16 data: 16 bits each, two per dword, low half first (an
    /// odd last high half 0). Integer formats keep their low 16 bits; the
    /// others become f16.
    fn d16_channels(&mut self, format: u8, channels: Vec<V>) -> Result<Vec<V>, Why> {
        let (_, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let mut halves = Vec::new();
        for x in channels {
            halves.push(if matches!(numeric, Numeric::Uint | Numeric::Sint) {
                let m = self.s.m.i32(0xFFFF);
                self.s.m.bin(c, Bin::And, x, m)
            } else {
                let (f, i32t) = (self.as_float(x), self.s.t.i32);
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            });
        }
        let mut out = Vec::new();
        for pair in halves.chunks(2) {
            out.push(match *pair {
                [lo, hi] => {
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                }
                [lo] => lo,
                _ => unreachable!("chunks of one or two"),
            });
        }
        Ok(out)
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                let x = match address.dynamic {
                    Some(v) => Some(self.i32_of(v)?),
                    None => None,
                };
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    /// A guessed PS's linear buffer descriptor in SGPRs, addressed in title memory.
    // shortcut: constant control flags and nonnegative SMEM immediates only.
    // Subword reads assume SH_MEM_CONFIG.alignment_mode = UNALIGNED; the
    // guessed path does not yet decode that register. Dword alignment is unchanged.
    fn computed_buffer(
        &mut self,
        inst: &Inst,
        src: &Source,
        index: Option<V>,
        (voffset, immediate): (V, u32),
        soffset: V,
        count: u32,
    ) -> Result<Vec<V>, Why> {
        let Source::BuiltMemory { control, swizzled } = *src else {
            unreachable!();
        };
        if swizzled || control & 0xC080_0000 != 0 {
            return Err("a computed buffer descriptor with swizzling, ADD_TID or a non-buffer type");
        }
        let descriptor = match &inst.fields {
            Fields::Buffer { rsrc, .. } | Fields::Smem { base: rsrc, .. } => rsrc,
            _ => unreachable!(),
        };
        let Operand::Scalar { code, .. } = *descriptor else {
            return Err("a computed buffer descriptor outside SGPRs");
        };
        let word = |l: &mut Self, k| {
            l.i32_of(inst.read(Reg::S(code + k)).ok_or("a computed buffer base")?)
        };
        let (lo, hi) = (word(self, 0)?, word(self, 1)?);
        let c = self.cur;
        let mask = self.s.m.i32(0xFFFF);
        let base_hi = self.s.m.bin(c, Bin::And, hi, mask);
        let shift = self.s.m.i32(16);
        let stride = self.s.m.bin(c, Bin::LShr, hi, shift);
        let zero = self.s.m.i32(0);
        let index = index.unwrap_or(zero);
        let bounds = (control >> 28) & 3;
        let records = word(self, 2)?;

        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let widen = |l: &mut Self, v| l.s.m.cast(c, Cast::ZExt, v, i64t);
        let (ix, stride, off) = (widen(self, index), widen(self, stride), widen(self, voffset));
        let immediate = self.s.m.int_const(i64t, i64::from(immediate));
        let off = self.s.m.bin(c, Bin::Add, off, immediate);
        let row = self.s.m.bin(c, Bin::Mul, ix, stride);
        let byte = self.s.m.bin(c, Bin::Add, row, off);
        // SOFFSET moves the base without changing the bounds of the view.
        let (lo, hi) = self.add64(lo, base_hi, soffset, false);

        let record_ok = self.s.m.cmp(c, pred::IULT, index, records);
        let byte_count = widen(self, records);
        let (bytes, signed) = match inst.name() {
            "buffer_load_ubyte" => (1, false),
            "buffer_load_sbyte" => (1, true),
            "buffer_load_ushort" => (2, false),
            "buffer_load_sshort" => (2, true),
            _ => (4, false),
        };
        let mut values = Vec::new();
        for k in 0..count {
            let c = self.cur;
            let delta = self.s.m.int_const(i64t, i64::from(bytes * k));
            let at = self.s.m.bin(c, Bin::Add, byte, delta);
            let end = self.s.m.int_const(i64t, i64::from(bytes * (k + 1)));
            let in_bounds = match bounds {
                0 => {
                    // Mode 0 checks the component's starting offset, not its
                    // end (RDNA2 ISA table 35). Mode 3 checks the payload end.
                    let start = self.s.m.bin(c, Bin::Add, off, delta);
                    let fits = self.s.m.cmp(c, pred::IULT, start, stride);
                    Some(self.s.m.bin(c, Bin::And, record_ok, fits))
                }
                1 => Some(record_ok),
                // OOB_SELECT=2 disables range checks, but an empty descriptor
                // still returns zero (RDNA2 ISA section 8.1.5, table 35).
                2 => Some(self.s.m.cmp(c, pred::NE, records, zero)),
                _ => {
                    let end = self.s.m.bin(c, Bin::Add, byte, end);
                    Some(self.s.m.cmp(c, pred::IULE, end, byte_count))
                }
            };
            let start = self.begin_guarded(in_bounds);
            let (low, high) = self.split64(at);
            let (lo, hi) = self.add64(lo, hi, low, false);
            let hi = self.s.m.bin(self.cur, Bin::Add, hi, high);
            let value = if bytes == 4 {
                let (h, byte) = self.memory(lo, hi, 4, false, None);
                self.raw_dword(h, byte)
            } else {
                self.memory_subword(lo, hi, bytes, signed)
            };
            values.push(self.end_guarded(start, &[(value, zero)])[0]);
        }
        Ok(values)
    }

    /// Read each byte through its own page-table lookup, including a halfword
    /// straddling a dword, a 64 KiB page, or the low 32 bits of the address.
    /// Bounds have already been checked for the whole element, not its backing dwords.
    fn memory_subword(&mut self, lo: V, hi: V, bytes: u32, signed: bool) -> V {
        let mut value = self.s.m.i32(0);
        for k in 0..bytes {
            let delta = self.s.m.i32(k);
            let (low, high) = self.add64(lo, hi, delta, false);
            let not3 = self.s.m.i32(!3);
            let aligned = self.s.m.bin(self.cur, Bin::And, low, not3);
            let (h, at) = self.memory(aligned, high, 4, false, None);
            let word = self.raw_dword(h, at);
            let c = self.cur;
            let three = self.s.m.i32(3);
            let lane = self.s.m.bin(c, Bin::And, low, three);
            let shift = self.s.m.bin(c, Bin::Shl, lane, three);
            let byte = self.s.m.bin(c, Bin::LShr, word, shift);
            let mask = self.s.m.i32(0xFF);
            let byte = self.s.m.bin(c, Bin::And, byte, mask);
            let shift = self.s.m.i32(8 * k);
            let byte = self.s.m.bin(c, Bin::Shl, byte, shift);
            value = self.s.m.bin(c, Bin::Or, value, byte);
        }
        if signed {
            let shift = self.s.m.i32(32 - bytes * 8);
            value = self.s.m.bin(self.cur, Bin::Shl, value, shift);
            value = self.s.m.bin(self.cur, Bin::AShr, value, shift);
        }
        value
    }

    /// The byte stride of the buffer view a heap-indexed descriptor names
    /// (descriptors from the resource heap only): dword `index` of the
    /// host's view-stride table, root SRV [`VIEW_STRIDES`].
    fn view_stride(&mut self, src: &Source) -> Result<V, Why> {
        const WHY: Why = "a structured buffer of unknown stride";
        let heap = |load: &Site| {
            self.scalar
                .get(load)
                .and_then(|a| self.x.native.layout.get(a.param))
                .is_some_and(|p| matches!(p.root, RootType::ResourceHeap | RootType::Table))
        };
        let leaves = src.leaves();
        if leaves.is_empty()
            || !leaves.iter().all(|l| match l {
                Source::Loaded { load, .. } | Source::Fetched { load, .. } => heap(load),
                _ => false,
            })
        {
            return Err(WHY);
        }
        let index = match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, false)?,
            Source::Phi { .. } => return Err(WHY),
            _ => self.lane_heap_index(src, false)?,
        };
        let c = self.cur;
        let four = self.s.m.i32(4);
        let at = self.s.m.bin(c, Bin::Mul, index, four);
        let table = self.s.root_srv_handle(VIEW_STRIDES);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, at, iu, mask, four],
        );
        let x = self.s.m.extract(c, r, 0);
        let word = self.as_int(x);
        let bits = self.s.m.i32(VIEW_STRIDE_BITS);
        Ok(self.s.m.bin(c, Bin::And, word, bits))
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } =
            src
        {
            return self.used_kind(args.first()?);
        }
        if let Source::LanePhi { args, .. } = src {
            return args.iter().flatten().find_map(|a| self.used_kind(a));
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// A handle for the descriptor `src` read at `site`; the heap index of
    /// a texture shader view the program takes from the heap itself is noted
    /// for the instruction's read mark (handles are made once per block,
    /// often before the instruction). A view read through a table needs
    /// none: the host makes the slot for the draw.
    fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        let h = self.handle_made(site, src, sampler, props)?;
        if let Some(&index) = self.s.texture_index.get(&h)
            && self.is_heap_direct(src)
        {
            self.s.texture_reads.push(index);
        }
        Ok(h)
    }

    fn handle_made(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        let props = if self.pops && props.0 & 0x1000 != 0 {
            self.s.flags |= flags::ROVS;
            (props.0 | 0x2000, props.1)
        } else {
            props
        };
        // A descriptor from the resource heap itself: the host cannot tell
        // which slots the program reads, or writes through unordered access.
        if !sampler && self.is_heap_direct(src) {
            self.heap[0] = true;
            self.heap[1] |= props.0 & 0x1000 != 0;
        }
        match src {
            Source::Loaded { load, first } if let Some(address) = self.entry_address(*load) => {
                // The same descriptor wherever it is loaded: the console
                // compiler reloads one before every use rather than keep it
                // (ActivateRays: 2,399 loads of one), and one handle made in
                // the entry block serves them all.
                let key = (address, *first, sampler, props.0);
                if let Some(&h) = self.entry_handles.get(&key) {
                    return Ok(h);
                }
                let here = std::mem::replace(&mut self.cur, self.s.entry);
                let index = self.entry_heap_index(address, *first, sampler);
                let index = self.buffer_guard(src, index, sampler, props);
                let h = self.s.heap_handle(self.cur, index, sampler, false, props);
                self.cur = here;
                self.entry_handles.insert(key, h);
                Ok(h)
            }
            Source::Loaded { load, first } => {
                let key = (self.cur, *load, *first, sampler, props.0);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let index = self.heap_index(*load, *first, sampler)?;
                let index = self.buffer_guard(src, index, sampler, props);
                // A waterfall loop's load at the first lane's offset reads
                // each lane's own entry (see `resource::Waterfall`).
                let per_lane = self.waterfall_loads.contains(load);
                let h = self
                    .s
                    .heap_handle(self.cur, index, sampler, per_lane, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words
                    .get(..4)
                    .and_then(|w| w.try_into().ok())
                    .ok_or("a sampler constant")?;
                let comparison = props.0 & 0x8000 != 0;
                let slot = self.samplers.iter().enumerate().position(|(n, w)| {
                    *w == words && (self.s.cmp_samplers >> n & 1 != 0) == comparison
                });
                let n = match slot {
                    Some(n) => n,
                    None => {
                        let n = self.samplers.len();
                        if n >= u64::BITS as usize {
                            return Err("more than 64 immediate sampler slots");
                        }
                        self.samplers.push(words);
                        self.s.cmp_samplers |= u64::from(comparison) << n;
                        n
                    }
                };
                Ok(self.s.sampler_handle(self.cur, n as u32))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            // Only image instructions get here (buffer accesses take
            // `computed_buffer`): `resource` gives every four dwords it
            // cannot trace this source, a sampler's state words too.
            Source::BuiltMemory { .. } if sampler => Err("a sampler built from computed words"),
            Source::BuiltMemory { .. } => Err("a computed buffer descriptor used as a view"),
            Source::Built { address, .. } if address.param == MEMORY => {
                Err("a buffer descriptor built around a computed address")
            }
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) if !self.root_memory => {
                        Ok(self.root_view(address.param, false))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        Ok(self.s.root_srv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Uav) | None, 0x100B)
                    | (RootType::Descriptor, Some(ResourceType::Uav), 11)
                        if !self.root_memory =>
                    {
                        Ok(self.root_view(address.param, true))
                    }
                    (RootType::Descriptor, Some(ResourceType::Uav) | None, 0x100B)
                    | (RootType::Descriptor, Some(ResourceType::Uav), 11) => {
                        Ok(self.s.root_uav_handle(rp.slot))
                    }
                    _ => Err("a root UAV or typed root view"),
                }
            }
            Source::Phi { block, args }
                if sampler && matches!(args[..], [Source::Constant(_), Source::Constant(_)]) =>
            {
                // One of two immediate samplers, chosen by uniform control
                // flow: what reads it runs with both, the edge taken picks
                // the result (static samplers cannot be indexed).
                let [a, b] = &args[..] else { unreachable!() };
                let ha = self.handle(site, a, sampler, props)?;
                let hb = self.handle(site, b, sampler, props)?;
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (j, &q) in self.x.p.blocks[*block].preds.iter().enumerate() {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail = self.tail[q].ok_or("a descriptor chosen by control flow")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    incoming.push((self.s.m.bool(j == 0), at));
                }
                let phi = self.s.m.phi(join, self.s.t.i1);
                self.s.m.set_phi(phi, incoming);
                self.s.sampler_choices.push((ha, hb, phi));
                Ok(ha)
            }
            Source::Phi { block, .. } => {
                // One handle at the join for every access the choice reaches
                // (each is in a block the join dominates).
                let _ = site;
                let key = (format!("{src:?}"), sampler, props.0);
                if let Some(&h) = self.join_handles.get(&key) {
                    return Ok(h);
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let here = std::mem::replace(&mut self.cur, join);
                let index = self.uniform_heap_index(src, sampler);
                let h = index.map(|index| {
                    let index = self.buffer_guard(src, index, sampler, props);
                    self.s.heap_handle(self.cur, index, sampler, false, props)
                });
                self.cur = here;
                let h = h?;
                self.join_handles.insert(key, h);
                Ok(h)
            }
            Source::Select { scc, args } => {
                // The heap index of each choice, picked by SCC (uniform).
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, sampler)?;
                let clear = self.heap_index(b, fb, sampler)?;
                let index = self.s.m.select(self.cur, cond, set, clear);
                let index = self.buffer_guard(src, index, sampler, props);
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                let index = self.lane_heap_index(src, sampler)?;
                let index = self.buffer_guard(src, index, sampler, props);
                Ok(self.s.heap_handle(self.cur, index, sampler, true, props))
            }
        }
    }

    /// The heap index to read a buffer view at, for a descriptor `src` from
    /// the resource heap itself (module notes, "Buffer views at computed
    /// heap indices"): `index`, or the host's null raw buffer view when the
    /// view-stride table says the slot holds no buffer view. Indices come
    /// from data; one naming a texture slot read a texture descriptor as a
    /// buffer one (a wild address: the texture's address shifted 8 bits
    /// down) and faulted the device.
    fn buffer_guard(&mut self, src: &Source, index: V, sampler: bool, props: (u32, u32)) -> V {
        // Typed, raw and structured buffers (DXIL resource kinds 10-12).
        if sampler || !(10..=12).contains(&(props.0 & 0xFF)) || !self.is_heap_direct(src) {
            return index;
        }
        let c = self.cur;
        let last = self.s.m.i32(super::HEAP_VIEWS - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let four = self.s.m.i32(4);
        let at = self.s.m.bin(c, Bin::Mul, index, four);
        let table = self.s.root_srv_handle(VIEW_STRIDES);
        let (iu, mask) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(1));
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, at, iu, mask, four],
        );
        let x = self.s.m.extract(c, r, 0);
        let stride = self.as_int(x);
        let zero = self.s.m.i32(0);
        let none = self.s.m.cmp(c, pred::EQ, stride, zero);
        // The null unordered access view sits just below the null SRV.
        let null = super::NULL_BUFFER_VIEW - u32::from(props.0 & 0x1000 != 0);
        let null = self.s.m.i32(null);
        self.s.m.select(c, none, null, index)
    }

    /// Whether a descriptor comes from the resource heap itself (root type
    /// 5, not a table) on some path.
    fn is_heap_direct(&self, src: &Source) -> bool {
        src.leaves().iter().any(|l| match l {
            Source::Loaded { load, .. } | Source::Fetched { load, .. } => self
                .scalar
                .get(load)
                .and_then(|a| self.x.native.layout.get(a.param))
                .is_some_and(|p| p.root == RootType::ResourceHeap),
            _ => false,
        })
    }

    /// The address of the descriptor load at `load` when root parameters
    /// alone give it (a table or a heap, at a fixed offset: resource
    /// tracking found no other term), and not per lane.
    fn entry_address(&self, load: Site) -> Option<EntryAddress> {
        let a = self.scalar.get(&load)?;
        let fixed = a.dynamic.is_none() && !self.x.res.waterfall.loads.contains(&load);
        let root = self.x.native.layout.get(a.param)?.root;
        let heap = matches!(
            root,
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap
        );
        (fixed && heap).then_some((a.param, a.offset))
    }

    /// Records that a fixed-offset read from a descriptor table needs at least
    /// `(byte_offset >> shift) + 1` entries in that table's root slot.
    fn record_table_entry(&mut self, param: usize, byte_offset: u32, sampler: bool) {
        let Some(rp) = self.x.native.layout.get(param) else {
            return;
        };
        if rp.root != RootType::Table {
            return;
        }
        let shift = if sampler { 4 } else { 5 };
        let count = (byte_offset >> shift).saturating_add(1);
        let entry = self.table_entries.entry(rp.slot).or_insert(0);
        if count > *entry {
            *entry = count;
        }
    }

    /// [`Lower::heap_index`] of a descriptor at a fixed address
    /// ([`Lower::entry_address`]), in the entry block.
    fn entry_heap_index(&mut self, (param, offset): EntryAddress, first: u8, sampler: bool) -> V {
        let byte_offset = offset.wrapping_add(4 * u32::from(first));
        self.record_table_entry(param, byte_offset, sampler);
        let entry = self.s.entry;
        let base = match self.x.native.layout[param].root {
            RootType::Table => self.user_dword(param, 0),
            _ => self.s.m.i32(0),
        };
        let k = self.s.m.i32(byte_offset);
        let addr = self.s.m.bin(entry, Bin::Add, base, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        self.s.m.bin(entry, Bin::LShr, addr, shift)
    }

    /// The heap index of a descriptor every lane shares, computed in the
    /// current block. One chosen by control flow is a phi at its join of the
    /// index each predecessor computes at its end; an edge may bring another
    /// join's (nested if/else).
    fn uniform_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Phi { block, args } => {
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail = self.tail[q].ok_or("a descriptor chosen by control flow")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    // No lane comes this way (`resource` gave it another
                    // edge's source): any index will do.
                    if self.never_taken(q, *block) {
                        incoming.push((self.s.m.i32(0), at));
                        continue;
                    }
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = self.uniform_heap_index(arg, sampler);
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a descriptor chosen by control flow"),
        }
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block (or, chosen by control flow, at the
    /// join).
    fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    let &m = self
                        .x
                        .st
                        .masks
                        .get(mask)
                        .ok_or("a descriptor chosen per lane by an untracked mask")?;
                    self.boolean(m, self.cur)?
                } else {
                    match self.get(*mask)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let set = self.lane_heap_index(a, sampler)?;
                let clear = self.lane_heap_index(b, sampler)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            Source::LanePhi { block, args } => {
                // Each predecessor computes its heap index at its end (what
                // it reads is there: the dwords are, on that edge); the join
                // picks the one of the edge the lane came by. An edge no
                // lane reading it takes gives index 0.
                let join =
                    self.block[*block].ok_or("a per-lane descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail =
                        self.tail[q].ok_or("a per-lane descriptor joined from a later block")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let index = match arg {
                        Some(arg) => {
                            let here = std::mem::replace(&mut self.cur, at);
                            let index = self.lane_heap_index(arg, sampler);
                            self.cur = here;
                            index?
                        }
                        None => self.s.m.i32(0),
                    };
                    incoming.push((index, at));
                }
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size.
    fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        if a.dynamic.is_none() {
            self.record_table_entry(a.param, a.offset.wrapping_add(4 * u32::from(first)), sampler);
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, at, shift))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        if a.dynamic.is_none() {
            self.record_table_entry(a.param, a.offset.wrapping_add(4 * u32::from(first)), sampler);
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4 * u32::from(first)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, addr, shift))
    }

    fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.waterfall_compares.contains(&(b, i)) {
            // The first lane's dwords of a waterfall loop are each lane's
            // own (see `resource::Waterfall`).
            self.lane(inst, Some(true));
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = match v.dpp {
                None => self.compare(inst, rest),
                // 64-bit compares read a VGPR pair (`src64`), which DPP does not permute.
                Some(_) if rest.ends_with("64") => Err("a 64-bit compare with DPP"),
                Some(d) => match self.dpp(inst, d, self.x.st.guards[b][i]) {
                    // Lanes DPP leaves out keep their old bit: not modelled.
                    Ok((_, Some(_))) => Err("a compare with DPP row masks or bounds"),
                    Ok((x, None)) => {
                        self.src0 = Some(x);
                        let r = self.compare(inst, rest);
                        self.src0 = None;
                        r
                    }
                    Err(why) => Err(why),
                },
            };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        // A scalar result (`v_readlane`, `v_readfirstlane`) is written
        // whatever EXEC holds.
        let mut guard = match v.dst {
            Some(Operand::Scalar { .. }) => None,
            _ => self.guard(b, i)?,
        };
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d, self.x.st.guards[b][i])?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, guard);
        self.src0 = None;
        r
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    fn dpp(&mut self, inst: &Inst, dpp: Dpp, guard: Option<Bool>) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            // GFX10 DPP16 controls (RDNA2 ISA, DPP_CTRL); quad_perm is refused
            // only outside pixel programs.
            let outside = match dpp {
                Dpp::Lanes { .. } => outside_compute!("DPP8"),
                Dpp::Row { ctrl: 0..=0xFF, .. } => outside_compute!("DPP quad_perm"),
                Dpp::Row {
                    ctrl: 0x101..=0x12F,
                    ..
                } => {
                    outside_compute!("DPP row_shl, row_shr or row_ror")
                }
                Dpp::Row {
                    ctrl: 0x140 | 0x141,
                    ..
                } => {
                    outside_compute!("DPP row_mirror or row_half_mirror")
                }
                Dpp::Row {
                    ctrl: 0x150..=0x16F,
                    ..
                } => {
                    outside_compute!("DPP row_share or row_xmask")
                }
                Dpp::Row { .. } => outside_compute!("DPP with another row control"),
            };
            self.cross_lane(quad, outside)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, and whether it lies in the row.
        let (src, inside, masks, zero, fi) = match dpp {
            Dpp::Lanes { sel, fi } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, true, fi)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                fi,
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl, fi)
            }
        };
        let mut r = if mesh_quad || self.helpers_read(quad) {
            self.quad_pick(x, src)
        } else {
            self.read_in_half(x, src)
        };
        // A partial group or a divergent block can leave source lanes out even
        // when reaching this instruction is enough to enable a destination.
        let group = match dpp {
            Dpp::Lanes { .. } => 8,
            _ if quad => 4,
            _ => 16,
        };
        let partial = self.s.threads.is_some_and(|t| {
            !t.iter().product::<u32>().is_multiple_of(group)
        });
        let reach = self.x.st.reach[self.cur_ir];
        let divergent = !fi && reach != Bool::TRUE && self.x.st.bools.support(reach).into_iter()
            .any(|a| !(self.uniform_atom(a) || quad && matches!(a, Atom::Quad(_))));
        let inside = if !fi && (guard.is_some() || divergent || partial) {
            let on = self.boolean(guard.unwrap_or(Bool::TRUE), c)?;
            let active = if self.helpers_read(quad) {
                // Wave ballots omit pixel helpers; quad reads include them,
                // but only while the host quad executes together.
                if divergent {
                    return Err("a DPP quad read in divergent pixel control flow");
                }
                let on = self.s.m.cast(c, Cast::ZExt, on, self.s.t.i32);
                let bit = self.quad_pick(on, src);
                let z = k(self, 0);
                self.s.m.cmp(c, pred::NE, bit, z)
            } else {
                // DPP never crosses a 32-lane half. On a split wave the
                // local ballot word serves either half without an exchange.
                let four = self.s.m.named_struct("dx.types.fouri32", vec![self.s.t.i32; 4]);
                let ballot = self.s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
                let lo = self.s.m.extract(c, ballot, 0);
                let hi = if self.host32 { lo } else { self.s.m.extract(c, ballot, 1) };
                let at = if mesh_quad {
                    // Only the quad-relative index is shared with the host.
                    let host = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, self.s.t.i32, 111, &[]);
                    let (not3, three) = (k(self, !3), k(self, 3));
                    let base = self.s.m.bin(c, Bin::And, host, not3);
                    let pick = self.s.m.bin(c, Bin::And, src, three);
                    self.s.m.bin(c, Bin::Or, base, pick)
                } else {
                    src
                };
                self.lane_bit_of(c, at, lo, hi)
            };
            Some(match inside {
                Some(bound) => self.s.m.bin(c, Bin::And, bound, active),
                None => active,
            })
        } else {
            inside
        };
        let mut enable = None;
        if let Some(inside) = inside {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            let ri = self.s.m.bin(c, Bin::LShr, lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }

    fn valu_op(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        let f = |l: &mut Self, k| l.fsrc(inst, k);
        let s = |l: &mut Self, k| l.src(inst, k);
        let float = |l: &mut Self, r: V| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: V| l.write_i(inst, r, guard);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        match name {
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1)?;
                let i64t = self.s.m.int(64);
                let k63 = self.s.m.i32(63);
                let n = self.s.m.bin(c, Bin::And, n, k63);
                let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                let op = match name {
                    "v_lshlrev_b64" => Bin::Shl,
                    "v_lshrrev_b64" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let r = self.s.m.bin(c, op, x, n);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 64-bit result");
                };
                let (lo, hi) = self.split64(r);
                self.write(inst, Reg::V(reg), lo, guard);
                self.write(inst, Reg::V(reg + 1), hi, guard);
                Ok(())
            }
            "v_mad_u64_u32" | "v_mad_i64_i32" => {
                // 32 × 32 → 64-bit product plus a 64-bit addend (address
                // arithmetic); the carry out is a lane mask.
                let signed = name == "v_mad_i64_i32";
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let addend = self.src64(inst, 2)?;
                let i64t = self.s.m.int(64);
                let ext = if signed { Cast::SExt } else { Cast::ZExt };
                let a = self.s.m.cast(c, ext, a, i64t);
                let bb = self.s.m.cast(c, ext, bb, i64t);
                let p = self.s.m.bin(c, Bin::Mul, a, bb);
                let r = self.s.m.bin(c, Bin::Add, p, addend);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 64-bit result");
                };
                let (lo, hi) = self.split64(r);
                self.write(inst, Reg::V(reg), lo, guard);
                self.write(inst, Reg::V(reg + 1), hi, guard);
                if !signed
                    && let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    let carry = self.s.m.cmp(c, pred::IULT, r, addend);
                    self.lane_bits.insert(w, carry);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                // Cube-map face coordinates (AMD's ISA): the major axis is z
                // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
                // else x.
                let (x, y, z) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
                let zx = self.s.m.cmp(c, pred::OGE, az, ax);
                let zy = self.s.m.cmp(c, pred::OGE, az, ay);
                let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
                let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
                let zero = self.s.m.f32(0);
                let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
                let minus = |l: &mut Self, v: V| {
                    let z = l.s.m.f32(0x8000_0000);
                    l.s.m.bin(c, Bin::Sub, z, v)
                };
                let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
                    let t = l.s.m.select(c, ymaj, yv, xv);
                    l.s.m.select(c, zmaj, zv, t)
                };
                let r = match name {
                    "v_cubeid_f32" => {
                        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                        let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                        let (f5, f4, f3, f2, f1, f0) = (
                            k(self, 5.0),
                            k(self, 4.0),
                            k(self, 3.0),
                            k(self, 2.0),
                            k(self, 1.0),
                            k(self, 0.0),
                        );
                        let zf = self.s.m.select(c, nz, f5, f4);
                        let yf = self.s.m.select(c, ny, f3, f2);
                        let xf = self.s.m.select(c, nx, f1, f0);
                        pick(self, zf, yf, xf)
                    }
                    "v_cubesc_f32" => {
                        let (nz, nx) = (neg(self, z), neg(self, x));
                        let (mx, mz) = (minus(self, x), minus(self, z));
                        let zf = self.s.m.select(c, nz, mx, x);
                        let xf = self.s.m.select(c, nx, z, mz);
                        pick(self, zf, x, xf)
                    }
                    "v_cubetc_f32" => {
                        let ny = neg(self, y);
                        let (my, mz) = (minus(self, y), minus(self, z));
                        let yf = self.s.m.select(c, ny, mz, z);
                        pick(self, my, yf, my)
                    }
                    _ => {
                        let t = pick(self, z, y, x);
                        let two = self.s.m.f32(2.0f32.to_bits());
                        self.s.m.bin(c, Bin::Mul, t, two)
                    }
                };
                float(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the program's
                // VGPR count (the registers the instruction reads or writes).
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let indexed_src = name != "v_movreld_b32";
                let value = if indexed_src {
                    let mut r = self.s.m.i32(0);
                    for k in 0..=255u8.saturating_sub(sreg) {
                        let Some(x) = inst.read(Reg::V(sreg + k)) else {
                            break;
                        };
                        let x = self.i32_of(x)?;
                        let kk = self.s.m.i32(u32::from(k));
                        let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                        r = self.s.m.select(c, is, x, r);
                    }
                    r
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    let Some(_) = inst.write(Reg::V(dreg + k)) else {
                        break;
                    };
                    let old = match inst.read(Reg::V(dreg + k)) {
                        Some(o) => self.i32_of(o)?,
                        None => self.s.m.i32(0),
                    };
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::V(dreg + k), x, guard);
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                self.write_placed(inst, Reg::V(dreg), b, guard);
                if dreg != sreg {
                    self.write_placed(inst, Reg::V(sreg), a, guard);
                }
                Ok(())
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = s(self, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                int(self, r)
            }
            "v_dot4c_i32_i8" => {
                // D += the dot product of four signed bytes, wrapping: DXIL's
                // Dot4AddI8Packed.
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                let i32_ty = self.s.t.i32;
                let r = self.s.op(
                    c,
                    "dot4AddPacked.i32",
                    Attr::ReadNone,
                    i32_ty,
                    163,
                    &[acc, a, bb],
                );
                int(self, r)
            }
            "v_dot4_i32_i8" => {
                // VOP3P: op_sel picks the half each multiplicand's low half
                // reads, op_sel_hi its high half's; the accumulator is read
                // whole (as AMD's compilers fold constants into it).
                if v.clamp || v.neg_hi != 0 {
                    return Err("a saturating or negated dot product");
                }
                let mut xs = Vec::new();
                for k in 0..2 {
                    let x = s(self, k)?;
                    let (lo, hi) = (v.opsel >> k & 1 != 0, v.opsel_hi >> k & 1 != 0);
                    if (lo, hi) == (false, true) {
                        xs.push(x);
                        continue;
                    }
                    let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
                    let top = self.s.m.bin(c, Bin::LShr, x, k16);
                    let bottom = self.s.m.bin(c, Bin::And, x, low);
                    let hi = self
                        .s
                        .m
                        .bin(c, Bin::Shl, if hi { top } else { bottom }, k16);
                    xs.push(self.s.m.bin(c, Bin::Or, if lo { top } else { bottom }, hi));
                }
                let acc = s(self, 2)?;
                let i32_ty = self.s.t.i32;
                let r = self.s.op(
                    c,
                    "dot4AddPacked.i32",
                    Attr::ReadNone,
                    i32_ty,
                    163,
                    &[acc, xs[0], xs[1]],
                );
                int(self, r)
            }
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return float(self, a);
                }
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let r = match name {
                    "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
                    "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
                    "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
                    "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
                    "v_mul_legacy_f32" => self.legacy_mul(a, bb),
                    "v_max_f32" => self.max_min(true, a, bb),
                    _ => self.max_min(false, a, bb),
                };
                float(self, r)
            }
            "v_ldexp_f32" => {
                // a * 2^e, e a signed integer: three exact power-of-two factors of at most 2^100
                // each (e clamped to +-300, past which every finite f32 already overflows or
                // flushes), so no factor overflows where the product would not.
                let (a, e) = (f(self, 0)?, s(self, 1)?);
                let (lo, hi) = (self.s.m.i32(-300i32 as u32), self.s.m.i32(300));
                let below = self.s.m.cmp(c, pred::SLT, e, lo);
                let e = self.s.m.select(c, below, lo, e);
                let above = self.s.m.cmp(c, pred::SGT, e, hi);
                let e = self.s.m.select(c, above, hi, e);
                let (three, two) = (self.s.m.i32(3), self.s.m.i32(2));
                let p1 = self.s.m.bin(c, Bin::SDiv, e, three);
                let rest = self.s.m.bin(c, Bin::Sub, e, p1);
                let p2 = self.s.m.bin(c, Bin::SDiv, rest, two);
                let p3 = self.s.m.bin(c, Bin::Sub, rest, p2);
                let mut r = a;
                for p in [p1, p2, p3] {
                    let (bias, shift) = (self.s.m.i32(127), self.s.m.i32(23));
                    let biased = self.s.m.bin(c, Bin::Add, p, bias);
                    let bits = self.s.m.bin(c, Bin::Shl, biased, shift);
                    let scale = self.as_float(bits);
                    r = self.s.m.bin(c, Bin::Mul, r, scale);
                }
                float(self, r)
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fmad(a, bb, cc);
                float(self, r)
            }
            "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let acc = match self.floats.get(&acc) {
                    Some(&x) => x,
                    None => {
                        let x = self.i32_of(acc)?;
                        self.as_float(x)
                    }
                };
                let r = match name {
                    "v_fmac_f32" => self.fmad(a, bb, acc),
                    "v_mac_f32" => {
                        let m = self.s.m.bin(c, Bin::Mul, a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                    _ => {
                        let m = self.legacy_mul(a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                };
                float(self, r)
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" | "v_mad_legacy_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = if name == "v_mad_legacy_f32" {
                    self.legacy_mul(a, bb)
                } else {
                    self.s.m.bin(c, Bin::Mul, a, bb)
                };
                let r = self.s.m.bin(c, Bin::Add, m, cc);
                float(self, r)
            }
            "v_max3_f32" | "v_min3_f32" | "v_med3_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = match name {
                    "v_max3_f32" => {
                        let t = self.max_min(true, a, bb);
                        self.max_min(true, t, cc)
                    }
                    "v_min3_f32" => {
                        let t = self.max_min(false, a, bb);
                        self.max_min(false, t, cc)
                    }
                    _ => self.med3(a, bb, cc),
                };
                float(self, r)
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let a = f(self, 0)?;
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(c, Bin::SDiv, one, a);
                float(self, r)
            }
            "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32" | "v_exp_f32" | "v_fract_f32"
            | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_sin_f32"
            | "v_cos_f32" => {
                let a = f(self, 0)?;
                let op = match name {
                    "v_rsq_f32" => 25,
                    "v_sqrt_f32" => 24,
                    "v_log_f32" => 23,
                    "v_exp_f32" => 21,
                    "v_fract_f32" => 22,
                    "v_trunc_f32" => 29,
                    "v_floor_f32" => 27,
                    "v_ceil_f32" => 28,
                    "v_rndne_f32" => 26,
                    "v_sin_f32" => 13,
                    _ => 12,
                };
                let r = match op {
                    12 | 13 => self.trig_turns(a, op == 12),
                    22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
                    _ => self.unary(op, a),
                };
                float(self, r)
            }
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let mut a = f(self, 0)?;
                let floor = name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32";
                let x = a;
                if floor {
                    // floor(x), or floor(x + 0.5) with the half added exactly
                    // (x - floor(x) is exact in f32), of x with denormals read
                    // as zeros (measured on RDNA2 by the Leonida project).
                    let flushed = self.ftz(a);
                    let fl = self.unary(27, flushed);
                    a = fl;
                    if name == "v_cvt_rpi_i32_f32" {
                        let frac = self.s.m.bin(c, Bin::Sub, flushed, fl);
                        let h = self.s.m.f32(0.5f32.to_bits());
                        let up = self.s.m.cmp(c, pred::OGE, frac, h);
                        let one = self.s.m.f32(1.0f32.to_bits());
                        let next = self.s.m.bin(c, Bin::Add, fl, one);
                        a = self.s.m.select(c, up, next, fl);
                    }
                }
                // D3D's conversions saturate and give 0 for NaN, as the
                // hardware does (checked on the RX 6700 XT and WARP).
                let op = if name == "v_cvt_u32_f32" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                let mut r = self.s.m.cast(c, op, a, self.s.t.i32);
                if floor {
                    // These saturate a NaN by its sign (measured on RDNA2).
                    let xi = self.as_int(x);
                    let zero = self.s.m.i32(0);
                    let neg = self.s.m.cmp(c, pred::SLT, xi, zero);
                    let (lo, hi) = (self.s.m.i32(0x8000_0000), self.s.m.i32(0x7FFF_FFFF));
                    let sat = self.s.m.select(c, neg, lo, hi);
                    let nan = self.s.m.cmp(c, pred::UNO, x, x);
                    r = self.s.m.select(c, nan, sat, r);
                }
                int(self, r)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let mut a = s(self, 0)?;
                if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
                    let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
                    let ff = self.s.m.i32(0xff);
                    a = self.s.m.bin(c, Bin::LShr, a, sh);
                    a = self.s.m.bin(c, Bin::And, a, ff);
                }
                let op = if name == "v_cvt_f32_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let r = self.s.m.cast(c, op, a, self.s.t.f32);
                float(self, r)
            }
            "v_cvt_off_f32_i4" => {
                // The low nibble as a signed 4-bit integer, over 16.
                let a = s(self, 0)?;
                let k28 = self.s.m.i32(28);
                let a = self.s.m.bin(c, Bin::Shl, a, k28);
                let a = self.s.m.bin(c, Bin::AShr, a, k28);
                let f = self.s.m.cast(c, Cast::SiToFp, a, self.s.t.f32);
                let k = self.s.m.f32(0.0625f32.to_bits());
                let r = self.s.m.bin(c, Bin::Mul, f, k);
                float(self, r)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                // Each source is f32, or with op_sel_hi the low (op_sel: high)
                // f16 half; neg_hi is the absolute value here.
                let mut xs = Vec::new();
                for i in 0..3 {
                    let op = v.src.get(i).ok_or("a missing operand")?;
                    let low = v.opsel_hi >> i & 1 != 0 && v.opsel >> i & 1 == 0;
                    let x = match self.low16_src(inst, i).filter(|_| low) {
                        Some(x) => x,
                        None => self.operand(inst, op)?,
                    };
                    let mut x = if v.opsel_hi >> i & 1 != 0 {
                        let h = if v.opsel >> i & 1 != 0 {
                            let k16 = self.s.m.i32(16);
                            self.s.m.bin(c, Bin::LShr, x, k16)
                        } else {
                            x
                        };
                        self.s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
                    } else {
                        self.as_float(x)
                    };
                    if v.neg_hi >> i & 1 != 0 {
                        x = self.unary(6, x);
                    }
                    if v.neg >> i & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Sub, z, x);
                    }
                    xs.push(x);
                }
                let mut r = self.fmad(xs[0], xs[1], xs[2]);
                if name == "v_fma_mix_f32" {
                    return float(self, r);
                }
                if v.clamp {
                    r = self.unary(7, r);
                }
                // The f16 result goes to one half; the other half stays.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a mixed-precision result");
                };
                // A high half kept from a value with no DXIL form (a first
                // interpolation half) stays unknown: only the low half is.
                let mut unknown = None;
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => match self.i32_of(o) {
                        Ok(x) => x,
                        Err(w) if name == "v_fma_mixlo_f16" && guard.is_none() => {
                            unknown = Some(w);
                            self.s.m.i32(0)
                        }
                        Err(w) => return Err(w),
                    },
                    None => self.s.m.i32(0),
                };
                let h = self.narrow16(r);
                let (keep, h) = if name == "v_fma_mixhi_f16" {
                    let k16 = self.s.m.i32(16);
                    (0xFFFF, self.s.m.bin(c, Bin::Shl, h, k16))
                } else {
                    (0xFFFF_0000, h)
                };
                let keep = self.s.m.i32(keep);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let x = self.s.m.bin(c, Bin::Or, old, h);
                self.write(inst, Reg::V(reg), x, guard);
                if let (Some(w), Some(new)) = (unknown, inst.write(Reg::V(reg))) {
                    self.vals.insert(new, Val::Missing(w));
                    self.low16.insert(new, x);
                }
                Ok(())
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let lo = self.f16_toward_zero(a);
                let hi = self.f16_toward_zero(bb);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_cvt_pknorm_u16_f32" | "v_cvt_pknorm_i16_f32" => {
                // Two floats as 16-bit normalized integers, src0 in the low
                // half and src1 in the high one: round(clamp(x, lo, 1) * max)
                // to nearest even, the product rounded as an f32 first, as
                // Mesa's ACO uses the pair for packUnorm2x16 and
                // packSnorm2x16 (VOP3 only on GFX10).
                // Guess: a NaN gives 0 (D3D's float to normalized integer
                // rule); no public source states what the hardware gives.
                let signed = name == "v_cvt_pknorm_i16_f32";
                let (lo, max) = if signed {
                    (-1.0f32, 32767.0f32)
                } else {
                    (0.0, 65535.0)
                };
                let mut halves = Vec::new();
                for k in 0..2 {
                    let x = f(self, k)?;
                    let nan = self.s.m.cmp(c, pred::UNO, x, x);
                    let zero = self.s.m.f32(0);
                    let x = self.s.m.select(c, nan, zero, x);
                    let (lo, one) = (self.s.m.f32(lo.to_bits()), self.s.m.f32(1.0f32.to_bits()));
                    let x = self.binary(35, x, lo);
                    let x = self.binary(36, x, one);
                    let max = self.s.m.f32(max.to_bits());
                    let x = self.s.m.bin(c, Bin::Mul, x, max);
                    let x = self.unary(26, x);
                    let op = if signed { Cast::FpToSi } else { Cast::FpToUi };
                    let x = self.s.m.cast(c, op, x, self.s.t.i32);
                    let ffff = self.s.m.i32(0xFFFF);
                    halves.push(self.s.m.bin(c, Bin::And, x, ffff));
                }
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], sh);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cvt_f16_f32" if v.opsel == 0 => {
                // Rounded as MODE says (nearest even by default) into the low half.
                let a = f(self, 0)?;
                let r = self.narrow16(a);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 16-bit result outside a VGPR");
                };
                let r = self.low_half(inst, reg, r);
                int(self, r)
            }
            "v_pack_b32_f16" => {
                // Two f16 sources (op_sel: their high halves; abs and neg on
                // the sign bit) into the low and high half.
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.operand(inst, op)?;
                    if v.opsel >> k & 1 != 0 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
                    let mask = self.s.m.i32(mask);
                    x = self.s.m.bin(c, Bin::And, x, mask);
                    if v.neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        x = self.s.m.bin(c, Bin::Xor, x, sign);
                    }
                    halves.push(x);
                }
                let k16 = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cvt_f32_f16" if v.opsel == 0 => {
                let a = s(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[a]);
                float(self, r)
            }
            "v_cndmask_b32" => {
                let cond = match v.src.get(2) {
                    Some(&Operand::Scalar { code, .. }) => {
                        let m = inst
                            .read(Reg::S(code))
                            .and_then(|m| {
                                self.x
                                    .st
                                    .masks
                                    .get(&m)
                                    .or_else(|| self.x.st.other.get(&m))
                                    .copied()
                            })
                            .ok_or("a select on an untracked mask")?;
                        self.boolean(m, c)?
                    }
                    Some(&Operand::Const(k)) => {
                        let x =
                            crate::resource::inline(k).ok_or("a select on an untracked mask")?;
                        if x == 0 {
                            self.s.m.bool(false)
                        } else if x == u32::MAX {
                            self.s.m.bool(true)
                        } else if self.console_lanes || self.s.mesh.is_some() {
                            let lo = self.s.m.i32(x);
                            let hi = self.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 });
                            self.lane_bit(c, lo, Some(hi))
                        } else {
                            return Err("a constant lane mask");
                        }
                    }
                    _ => return Err("a select on an untracked mask"),
                };
                if v.abs != 0 || v.neg != 0 {
                    // Sign-bit moves on the bits a select passes through
                    // (a host `fabs`/`fneg` may flush a denormal).
                    let mut xs = Vec::new();
                    for k in 0..2 {
                        let mut x = match self.src0 {
                            Some(x) if k == 0 => x,
                            _ => {
                                let op = *v.src.get(k).ok_or("a missing operand")?;
                                let x = self.operand(inst, &op)?;
                                match Self::sdwa_src(inst, k) {
                                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                                    None => x,
                                }
                            }
                        };
                        if v.abs >> k & 1 != 0 {
                            let m = self.s.m.i32(0x7FFF_FFFF);
                            x = self.s.m.bin(c, Bin::And, x, m);
                        }
                        if v.neg >> k & 1 != 0 {
                            let m = self.s.m.i32(0x8000_0000);
                            x = self.s.m.bin(c, Bin::Xor, x, m);
                        }
                        xs.push(x);
                    }
                    let r = self.s.m.select(c, cond, xs[1], xs[0]);
                    return int(self, r);
                }
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let r = self.s.m.select(c, cond, bb, a);
                int(self, r)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                // The carry (borrow) out is a lane mask: this lane's bit.
                let (mut a, mut bb) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut bb);
                }
                let cin = if name.ends_with("_ci_u32") {
                    let m = match v.src.get(2) {
                        Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                        _ => None,
                    };
                    let m = m
                        .and_then(|m| self.x.st.masks.get(&m).copied())
                        .ok_or("a carry-in from an untracked mask")?;
                    Some(self.boolean(m, c)?)
                } else {
                    None
                };
                let (r, carry) = self.add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
                int(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, carry);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            // An SGPR spilled to a VGPR lane and read back is the SGPR, in
            // every stage (in compute the VGPR's other lanes may hold what has
            // no host value, a descriptor's address say).
            "v_readlane_b32"
                if inst
                    .writes
                    .first()
                    .is_some_and(|w| self.x.st.restored.contains_key(&w.1)) =>
            {
                let a = self.i32_of(self.x.st.restored[&inst.writes[0].1])?;
                int(self, a)
            }
            "v_writelane_b32"
                if !self.console_lanes
                    && v.src
                        .get(1)
                        .and_then(|&l| crate::resource::lane_of(l))
                        .is_some() =>
            {
                Err("a VGPR of spilled SGPRs read per lane")
            }
            // A constant lane of a VGPR only `v_writelane_b32` fills on every path: the SGPR (or
            // constant) last written there, a phi where paths wrote different ones.
            "v_readlane_b32"
                if let [Operand::Vector { reg, .. }, at] = v.src[..]
                    && let Some(lane) = crate::resource::lane_of(at)
                    && let Some(x) = inst.read(Reg::V(reg))
                    && self.spill_slot(x, lane, &mut Vec::new(), false).is_ok() =>
            {
                let a = self.spill_slot(x, lane, &mut Vec::new(), true)?;
                int(self, a)
            }
            "v_readlane_b32"
                if let Some(Operand::Vector { reg, .. }) = v.src.first()
                    && let Some(x) = inst.read(Reg::V(*reg))
                    && self.section_uniform(x) =>
            {
                let a = self.i32_of(x)?;
                int(self, a)
            }
            "v_readlane_b32"
                if let [Operand::Vector { reg, .. }, at] = v.src[..]
                    && let Some(lane) = crate::resource::lane_of(at)
                    && let Some(x) = inst.read(Reg::V(reg))
                    && let Some(&res) = self.section_lanes.get(&(x, lane)) =>
            {
                int(self, res?)
            }
            "v_readlane_b32" => {
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                // A computed lane in a pixel or vertex program is a host lane
                // (from a mask or `v_mbcnt`); a constant one is the console's.
                if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                    self.s.flags |= flags::WAVE_OPS;
                } else {
                    let outside = if matches!(lane, V::Const(_)) {
                        outside_compute!("v_readlane_b32 of a constant lane")
                    } else {
                        outside_compute!("v_readlane_b32 of a computed lane")
                    };
                    self.cross_lane(false, outside)?;
                }
                let r = if self.mask_data == MaskData::Host {
                    self.read_lane(a, lane)
                } else {
                    self.wide_read_lane(a, lane)?
                };
                int(self, r)
            }
            "v_writelane_b32" => {
                // The lane given takes the scalar; the others keep their value.
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                self.cross_lane(false, outside_compute!("v_writelane_b32"))?;
                let me = self.lane_index();
                let is = self.s.m.cmp(c, pred::EQ, me, lane);
                let r = self.s.m.select(c, is, a, old);
                int(self, r)
            }
            "v_readfirstlane_b32"
                if inst
                    .writes
                    .first()
                    .is_some_and(|w| self.waterfall_reads.contains(&w.1)) =>
            {
                // A waterfall loop's read of a descriptor dword: the lane's own.
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32" if self.x.st.lockstep => {
                // Lockstep: every thread runs, so the first lane is the wide EXEC's.
                let a = s(self, 0)?;
                let lane = self.lockstep_first_lane(inst)?;
                let r = self.wide_read_lane(a, lane)?;
                int(self, r)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                self.s.flags |= flags::WAVE_OPS;
                let r = self.s.op(
                    c,
                    "waveReadLaneFirst.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    118,
                    &[a],
                );
                let r = self.helper_reads_own(a, r);
                int(self, r)
            }
            "v_permlane16_b32" | "v_permlanex16_b32" => {
                // Each lane reads from its own 16-lane row (`v_permlane16`) or
                // the other row in the same 32-lane half (`v_permlanex16`),
                // using nibble `lane & 15` of `{src2, src1}` as the lane within
                // that row.
                self.cross_lane(
                    false,
                    outside_compute!("v_permlane16_b32 or v_permlanex16_b32"),
                )?;
                self.s.flags |= flags::WAVE_OPS;
                let (x, s1, s2) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let lane = self.lane_index();
                let k = |l: &mut Self, n: u32| l.s.m.i32(n);
                let (k0, k2, k7, k8, k15, not15) = (
                    k(self, 0),
                    k(self, 2),
                    k(self, 7),
                    k(self, 8),
                    k(self, 15),
                    k(self, !15),
                );
                let hi_bit = self.s.m.bin(c, Bin::And, lane, k8);
                let is_hi = self.s.m.cmp(c, pred::NE, hi_bit, k0);
                let word = self.s.m.select(c, is_hi, s2, s1);
                let nibble = self.s.m.bin(c, Bin::And, lane, k7);
                let shift = self.s.m.bin(c, Bin::Shl, nibble, k2);
                let pick = self.s.m.bin(c, Bin::LShr, word, shift);
                let pick = self.s.m.bin(c, Bin::And, pick, k15);
                let row = self.s.m.bin(c, Bin::And, lane, not15);
                let row = if name == "v_permlanex16_b32" {
                    let k16 = k(self, 16);
                    self.s.m.bin(c, Bin::Xor, row, k16)
                } else {
                    row
                };
                let src = self.s.m.bin(c, Bin::Or, row, pick);
                let mut r = self.read_in_half(x, src);
                // Without `fi:1` (opsel bit 0), reading an inactive lane yields 0
                // (`bound_ctrl:1`, opsel bit 1) or keeps the destination's old value.
                if v.opsel & 1 == 0
                    && let Some(on) = guard
                {
                    let i32t = self.s.t.i32;
                    let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                    let ballot = self
                        .s
                        .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
                    let (lo, hi) = (
                        self.s.m.extract(c, ballot, 0),
                        self.s.m.extract(c, ballot, 1),
                    );
                    let active = if self.host32 {
                        let k31 = k(self, 31);
                        let in_half = self.s.m.bin(c, Bin::And, src, k31);
                        let bit = self.s.m.bin(c, Bin::LShr, lo, in_half);
                        let one = k(self, 1);
                        let bit = self.s.m.bin(c, Bin::And, bit, one);
                        self.s.m.cmp(c, pred::NE, bit, k0)
                    } else {
                        self.lane_bit_of(c, src, lo, hi)
                    };
                    let fallback = if v.opsel & 2 != 0 {
                        k0
                    } else if let Some(Operand::Vector { reg, .. }) = v.dst
                        && let Some(old) = inst.read(Reg::V(reg))
                    {
                        self.i32_of(old)?
                    } else {
                        k0
                    };
                    r = self.s.m.select(c, active, r, fallback);
                }
                int(self, r)
            }
            _ => {
                let r = self.integer(inst, name)?;
                int(self, r)
            }
        }
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            let lo = l.s.m.cast(c, Cast::ZExt, lo, i64t);
            let hi = l.s.m.cast(c, Cast::ZExt, hi, i64t);
            let k32 = l.s.m.int_const(i64t, 32);
            let hi = l.s.m.bin(c, Bin::Shl, hi, k32);
            Ok(l.s.m.bin(c, Bin::Or, lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            // A signed 64-bit operand sign-extends its 32-bit literal, as
            // the hardware does (LLVM's AMDGPU operand rules).
            Operand::Literal(x) if inst.name().ends_with("_i64") => {
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals (of unsigned and bit operands) and the null
                // register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }

    /// The low and high dwords of an i64.
    fn split64(&mut self, x: V) -> (V, V) {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let k32 = self.s.m.int_const(i64t, 32);
        let lo = self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::LShr, x, k32);
        let hi = self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32);
        (lo, hi)
    }

    /// Source `k` of a 16-bit operation in the low 16 bits (the rest 0):
    /// op_sel (or an SDWA word select) picks the high half; inline float
    /// constants are f16 values here.
    fn half_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let mut high = v.opsel >> k & 1 != 0;
        if let Some(sd) = v.sdwa {
            match sd.src_sel.get(k).copied().unwrap_or(6) {
                4 => {}
                5 => high = true,
                6 => {}
                _ => return Err("a byte select on a 16-bit operation"),
            }
        }
        let x = match self.src0 {
            Some(x) if k == 0 => x,
            _ => match self.low16_src(inst, k).filter(|_| !high) {
                Some(x) => x,
                None => self.f16_operand(inst, &op)?,
            },
        };
        let c = self.cur;
        let x = if high {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// An f16 source as an f32 (exact), with its abs/neg modifiers.
    fn half_fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let h = self.half_src(inst, k)?;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    /// Writes a 16-bit result (low bits of `x`) into the destination's low
    /// half, or its high half with op_sel bit 3 (or an SDWA WORD_1 select);
    /// the other half is kept (GFX10).
    fn write_half(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 16-bit result outside a VGPR");
        };
        let mut high = v.opsel >> 3 & 1 != 0;
        // SDWA: the word written, the other one kept (preserve) or zero
        // (pad; a sign extension above the high word adds nothing).
        let mut keep = true;
        if let Some(sd) = v.sdwa {
            match (sd.dst_sel, sd.dst_unused) {
                (4, 2) => {}
                (5, 2) => high = true,
                (4 | 6, _) => keep = false,
                (5, _) => (high, keep) = (true, false),
                _ => return Err("an SDWA destination of a 16-bit operation"),
            }
        }
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) if keep => o,
            _ => self.s.m.i32(0),
        };
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let x = self.s.m.bin(c, Bin::And, x, low);
        let r = if high {
            let o = self.s.m.bin(c, Bin::And, old, low);
            let x = self.s.m.bin(c, Bin::Shl, x, k16);
            self.s.m.bin(c, Bin::Or, o, x)
        } else {
            let keep = self.s.m.i32(0xFFFF_0000);
            let o = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, o, x)
        };
        self.write_placed(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// 16-bit operations, computed in f32 or i32: f16 sums, differences and
    /// products are exact there and round once to f16 (f32 has the 2 x 11 +
    /// 2 bits that make double rounding harmless); `None` if `name` is none.
    fn valu16(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const FLOAT2: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
        ];
        const FLOAT3: &[&str] = &[
            "v_fma_f16",
            "v_mad_f16",
            "v_fmac_f16",
            "v_mac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
        ];
        const FLOAT1: &[&str] = &[
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_fract_f16",
            "v_trunc_f16",
            "v_floor_f16",
            "v_ceil_f16",
            "v_rndne_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_cvt_f16_u16",
            "v_cvt_f16_i16",
        ];
        const INT: &[&str] = &[
            "v_add_nc_i16",
            "v_sub_nc_i16",
            "v_add_nc_u16",
            "v_sub_nc_u16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
            "v_cvt_u16_f16",
            "v_cvt_i16_f16",
            "v_cvt_norm_i16_f16",
            "v_cvt_norm_u16_f16",
        ];
        let all = [FLOAT1, FLOAT2, FLOAT3, INT];
        if !all.iter().any(|l| l.contains(&name))
            && !matches!(name, "v_cvt_f16_f32" | "v_cvt_f32_f16")
        {
            return None;
        }
        Some(self.valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3))
    }

    #[allow(clippy::too_many_arguments)]
    fn valu16_op(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
        float1: &[&str],
        float2: &[&str],
        float3: &[&str],
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        if name == "v_cvt_f32_f16" {
            let f = self.half_fsrc(inst, 0)?;
            return self.write_f(inst, f, guard);
        }
        let float_result = |l: &mut Self, mut f: V| -> Result<(), Why> {
            if v.omod != 0 {
                return Err("an output modifier on a 16-bit operation");
            }
            if v.clamp {
                f = l.unary(7, f);
            }
            let h = l.narrow16(f);
            l.write_half(inst, h, guard)
        };
        if name == "v_cvt_f16_f32" {
            let f = self.fsrc(inst, 0)?;
            return float_result(self, f);
        }
        if float2.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let r = match name {
                "v_add_f16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_f16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_f16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_f16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_max_f16" => self.max_min(true, a, b),
                _ => self.max_min(false, a, b),
            };
            return float_result(self, r);
        }
        if float3.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let acc = if matches!(name, "v_fmac_f16" | "v_mac_f16") {
                // The accumulator is the destination's half that is written.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                let x = if v.opsel >> 3 & 1 != 0 {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let m = self.s.m.i32(0xFFFF);
                let x = self.s.m.bin(c, Bin::And, x, m);
                self.s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.half_fsrc(inst, 2)?
            };
            let r = match name {
                // shortcut: fused f16 multiply-adds as FMad in f32 (the
                // product is exact; the sum rounds twice).
                "v_fma_f16" | "v_fmac_f16" | "v_mad_f16" | "v_mac_f16" => self.fmad(a, b, acc),
                "v_max3_f16" => {
                    let t = self.max_min(true, a, b);
                    self.max_min(true, t, acc)
                }
                "v_min3_f16" => {
                    let t = self.max_min(false, a, b);
                    self.max_min(false, t, acc)
                }
                _ => self.med3(a, b, acc),
            };
            return float_result(self, r);
        }
        if float1.contains(&name) {
            let r = match name {
                "v_cvt_f16_u16" | "v_cvt_f16_i16" => {
                    let mut x = self.half_src(inst, 0)?;
                    let op = if name == "v_cvt_f16_i16" {
                        let k16 = self.s.m.i32(16);
                        let t = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, t, k16);
                        Cast::SiToFp
                    } else {
                        Cast::UiToFp
                    };
                    self.s.m.cast(c, op, x, f32t)
                }
                "v_rcp_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    let one = self.s.m.f32(1.0f32.to_bits());
                    self.s.m.bin(c, Bin::SDiv, one, a)
                }
                _ => {
                    let a = self.half_fsrc(inst, 0)?;
                    let op = match name {
                        "v_sqrt_f16" => 24,
                        "v_rsq_f16" => 25,
                        "v_log_f16" => 23,
                        "v_exp_f16" => 21,
                        "v_fract_f16" => 22,
                        "v_trunc_f16" => 29,
                        "v_floor_f16" => 27,
                        "v_ceil_f16" => 28,
                        "v_rndne_f16" => 26,
                        "v_sin_f16" => 13,
                        _ => 12,
                    };
                    match op {
                        12 | 13 => self.trig_turns(a, op == 12),
                        // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                        22 => self.fract(a, 1.0 - 1.0 / 4096.0, 1.0 - 1.0 / 2048.0),
                        _ => self.unary(op, a),
                    }
                }
            };
            return float_result(self, r);
        }
        if matches!(name, "v_cvt_norm_i16_f16" | "v_cvt_norm_u16_f16") {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let r = self.cvt_norm16(inst, name == "v_cvt_norm_i16_f16")?;
            return self.write_half(inst, r, guard);
        }
        // 16-bit integers: compute on the selected halves, then keep 16 bits.
        let signed_add_sub = matches!(name, "v_add_nc_i16" | "v_sub_nc_i16");
        if (v.clamp && !signed_add_sub) || v.omod != 0 || v.abs != 0 || v.neg != 0 {
            return Err("modifiers on a 16-bit integer operation");
        }
        let r = match name {
            "v_cvt_u16_f16" | "v_cvt_i16_f16" => {
                // Toward zero, saturating at the 16-bit range (infinities
                // included); NaN gives 0. Clamped in f32 first, where every
                // f16 is exact, so the conversion never leaves i32's range.
                let a = self.half_fsrc(inst, 0)?;
                let (lo, hi) = if name == "v_cvt_u16_f16" {
                    (0.0f32, 65535.0f32)
                } else {
                    (-32768.0, 32767.0)
                };
                let (lo, hi) = (self.s.m.f32(lo.to_bits()), self.s.m.f32(hi.to_bits()));
                let t = self.binary(35, a, lo);
                let t = self.binary(36, t, hi);
                let r = self.s.m.cast(c, Cast::FpToSi, t, i32t);
                let nan = self.s.m.cmp(c, pred::UNO, a, a);
                let z = self.s.m.i32(0);
                self.s.m.select(c, nan, z, r)
            }
            _ => {
                let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if name.ends_with("_i16") {
                    a = sext(self, a);
                    b = sext(self, b);
                }
                let k15 = self.s.m.i32(15);
                match name {
                    "v_add_nc_u16" | "v_add_nc_i16" => self.s.m.bin(c, Bin::Add, a, b),
                    "v_sub_nc_u16" | "v_sub_nc_i16" => self.s.m.bin(c, Bin::Sub, a, b),
                    "v_subrev_nc_u16" => self.s.m.bin(c, Bin::Sub, b, a),
                    "v_mul_lo_u16" => self.s.m.bin(c, Bin::Mul, a, b),
                    "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                        // src0 is the shift (4 bits), src1 the value.
                        let sh = self.s.m.bin(c, Bin::And, a, k15);
                        let x = if name == "v_ashrrev_i16" {
                            sext(self, b)
                        } else {
                            b
                        };
                        let op = match name {
                            "v_lshlrev_b16" => Bin::Shl,
                            "v_lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.s.m.bin(c, op, x, sh)
                    }
                    _ => {
                        let p = match name {
                            "v_max_u16" => pred::IUGT,
                            "v_min_u16" => pred::IULT,
                            "v_max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, a, b);
                        self.s.m.select(c, t, a, b)
                    }
                }
            }
        };
        let r = if v.clamp {
            let lo = self.s.m.i32(i16::MIN as u32);
            let hi = self.s.m.i32(i16::MAX as u32);
            let below = self.s.m.cmp(c, pred::SLT, r, lo);
            let r = self.s.m.select(c, below, lo, r);
            let above = self.s.m.cmp(c, pred::SGT, r, hi);
            self.s.m.select(c, above, hi, r)
        } else {
            r
        };
        self.write_half(inst, r, guard)
    }

    /// `v_cvt_norm_i16_f16` / `v_cvt_norm_u16_f16`: the f16 source (with its
    /// abs/neg modifiers) clamped to [-1, 1] or [0, 1], times 32767 or 65535,
    /// rounded to nearest even once from the exact product; NaN gives 0.
    /// Worked on the f16's bits in integers, because in f32 the product
    /// (up to 26 significant bits) would round before the rounding to an
    /// integer.
    fn cvt_norm16(&mut self, inst: &Inst, signed: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let mut h = self.half_src(inst, 0)?;
        if v.abs & 1 != 0 {
            let m = k(self, 0x7FFF);
            h = self.s.m.bin(c, Bin::And, h, m);
        }
        if v.neg & 1 != 0 {
            let m = k(self, 0x8000);
            h = self.s.m.bin(c, Bin::Xor, h, m);
        }
        let scale = if signed { 0x7FFF } else { 0xFFFF };
        let (k0, k1, k10) = (k(self, 0), k(self, 1), k(self, 10));
        let mag_mask = k(self, 0x7FFF);
        let mag = self.s.m.bin(c, Bin::And, h, mag_mask);
        let k15 = k(self, 15);
        let neg = self.s.m.bin(c, Bin::LShr, h, k15);
        let neg = self.s.m.cmp(c, pred::NE, neg, k0);
        // |x| < 1: x = m * 2^-s with m < 2^11 and 11 <= s <= 24.
        let exp = self.s.m.bin(c, Bin::LShr, mag, k10);
        let man_mask = k(self, 0x3FF);
        let man = self.s.m.bin(c, Bin::And, mag, man_mask);
        let implicit = k(self, 0x400);
        let normal = self.s.m.bin(c, Bin::Or, man, implicit);
        let sub = self.s.m.cmp(c, pred::EQ, exp, k0);
        let m = self.s.m.select(c, sub, man, normal);
        let k25 = k(self, 25);
        let s_norm = self.s.m.bin(c, Bin::Sub, k25, exp);
        let k24 = k(self, 24);
        let sh = self.s.m.select(c, sub, k24, s_norm);
        let ks = k(self, scale);
        let p = self.s.m.bin(c, Bin::Mul, m, ks);
        let q = self.s.m.bin(c, Bin::LShr, p, sh);
        let one_s = self.s.m.bin(c, Bin::Shl, k1, sh);
        let low_mask = self.s.m.bin(c, Bin::Sub, one_s, k1);
        let rem = self.s.m.bin(c, Bin::And, p, low_mask);
        let half = self.s.m.bin(c, Bin::LShr, one_s, k1);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let odd = self.s.m.bin(c, Bin::And, q, k1);
        let odd = self.s.m.cmp(c, pred::NE, odd, k0);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let r = self.s.m.select(c, up, k1, k0);
        let r = self.s.m.bin(c, Bin::Add, q, r);
        // |x| >= 1 (and infinity) saturates; NaN gives 0.
        let one_h = k(self, 0x3C00);
        let big = self.s.m.cmp(c, pred::IUGE, mag, one_h);
        let r = self.s.m.select(c, big, ks, r);
        let inf = k(self, 0x7C00);
        let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
        let r = self.s.m.select(c, nan, k0, r);
        let r = if signed {
            let n = self.s.m.bin(c, Bin::Sub, k0, r);
            self.s.m.select(c, neg, n, r)
        } else {
            self.s.m.select(c, neg, k0, r)
        };
        let m16 = k(self, 0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, r, m16))
    }

    /// Half `high` of source `k` of a packed (VOP3P) operation, in the low
    /// 16 bits: op_sel picks the half the low result reads, op_sel_hi the
    /// half the high result reads. Packed integer instructions see the f32
    /// bits of an inline float constant.
    fn packed_src(&mut self, inst: &Inst, k: usize, high: bool, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = if float {
            self.f16_operand(inst, &op)?
        } else {
            self.operand(inst, &op)?
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Half `high` of source `k` of `v_pk_fmac_f16`, in the low 16 bits: a
    /// VOP2 operation (no op_sel) whose third source is its destination. A
    /// constant is the 32-bit value its halves come from: an inline float
    /// constant's f16 with a high half of 0, an integer one's 32 bits, a
    /// literal's (llvm-mc's packed 16-bit operands: `1.0` stands for
    /// 0x00003C00, `-1` for 0xFFFFFFFF; 0x3C003C00 needs a literal).
    fn fmac_half(&mut self, inst: &Inst, k: usize, high: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = if k == 2 { v.dst } else { v.src.get(k).copied() };
        let op = op.ok_or("a missing operand")?;
        let x = self.f16_operand(inst, &op)?;
        let c = self.cur;
        let x = if high {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "fmac_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        if v.omod != 0 || !float && (v.clamp || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let f32t = self.s.t.f32;
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = if op == "fmac_f16" {
                    self.fmac_half(inst, k, high)?
                } else {
                    self.packed_src(inst, k, high, float)?
                };
                if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        f = self.s.m.bin(c, Bin::Sub, z, f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    // shortcut: FMad in f32, as v_fma_f16.
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.narrow16(f)
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    fn integer(&mut self, inst: &Inst, name: &'static str) -> Result<V, Why> {
        let c = self.cur;
        let s = |l: &mut Self, k| l.src(inst, k);
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let m31 = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            l.s.m.bin(c, Bin::And, x, k31)
        };
        Ok(match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_add_nc_u32" | "v_sub_nc_u32"
            | "v_subrev_nc_u32" | "v_mul_lo_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (op, a, b) = match name {
                    "v_and_b32" => (Bin::And, a, b),
                    "v_or_b32" => (Bin::Or, a, b),
                    "v_xor_b32" => (Bin::Xor, a, b),
                    "v_add_nc_u32" => (Bin::Add, a, b),
                    "v_sub_nc_u32" => (Bin::Sub, a, b),
                    "v_subrev_nc_u32" => (Bin::Sub, b, a),
                    _ => (Bin::Mul, a, b),
                };
                self.s.m.bin(c, op, a, b)
            }
            "v_xnor_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let x = self.s.m.bin(c, Bin::Xor, a, b);
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, x, ones)
            }
            "v_not_b32" => {
                let a = s(self, 0)?;
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                // A left shift by 16 or more reads the low half alone.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                let by = v.src.first().and_then(|&o| crate::resource::lane_of(o));
                let low =
                    name == "v_lshlrev_b32" && v.sdwa.is_none() && by.is_some_and(|n| n & 31 >= 16);
                let a = match self.low16_src(inst, 1).filter(|_| low) {
                    Some(x) => x,
                    None => s(self, 1)?,
                };
                let sh = s(self, 0)?;
                let sh = m31(self, sh);
                let op = match name {
                    "v_lshlrev_b32" => Bin::Shl,
                    "v_lshrrev_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, sh)
            }
            "v_lshl_add_u32" | "v_lshl_or_b32" | "v_add_lshl_u32" | "v_and_or_b32"
            | "v_or3_b32" | "v_xad_u32" | "v_add3_u32" | "v_xor3_b32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                match name {
                    "v_lshl_add_u32" | "v_lshl_or_b32" => {
                        let b = m31(self, b);
                        let t = self.s.m.bin(c, Bin::Shl, a, b);
                        let op = if name == "v_lshl_add_u32" {
                            Bin::Add
                        } else {
                            Bin::Or
                        };
                        self.s.m.bin(c, op, t, d)
                    }
                    "v_add_lshl_u32" => {
                        let t = self.s.m.bin(c, Bin::Add, a, b);
                        let d = m31(self, d);
                        self.s.m.bin(c, Bin::Shl, t, d)
                    }
                    _ => {
                        let (o1, o2) = match name {
                            "v_and_or_b32" => (Bin::And, Bin::Or),
                            "v_or3_b32" => (Bin::Or, Bin::Or),
                            "v_xad_u32" => (Bin::Xor, Bin::Add),
                            "v_add3_u32" => (Bin::Add, Bin::Add),
                            _ => (Bin::Xor, Bin::Xor),
                        };
                        let t = self.s.m.bin(c, o1, a, b);
                        self.s.m.bin(c, o2, t, d)
                    }
                }
            }
            "v_mul_u32_u24" | "v_mul_i32_i24" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mul_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                self.s.m.bin(c, Bin::Mul, a, b)
            }
            "v_mul_hi_u32_u24" | "v_mul_hi_i32_i24" => {
                // The high dword of the same 48-bit product: the low 24 bits
                // of each operand, zero- or sign-extended, multiplied.
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let signed = name == "v_mul_hi_i32_i24";
                let (sh, op) = (k(self, 8), if signed { Bin::AShr } else { Bin::LShr });
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                self.mul_hi(a, b, signed)
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let (a, off, w) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let w = m31(self, w);
                self.bfe(a, off, w, name == "v_bfe_i32")
            }
            "v_bfi_b32" => {
                let (m, a, b) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let ones = k(self, u32::MAX);
                let nm = self.s.m.bin(c, Bin::Xor, m, ones);
                let x = self.s.m.bin(c, Bin::And, m, a);
                let y = self.s.m.bin(c, Bin::And, nm, b);
                self.s.m.bin(c, Bin::Or, x, y)
            }
            "v_max_i32" | "v_min_i32" | "v_max_u32" | "v_min_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let p = match name {
                    "v_max_i32" => pred::SGT,
                    "v_min_i32" => pred::SLT,
                    "v_max_u32" => pred::IUGT,
                    _ => pred::IULT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, t, a, b)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                self.mul_hi(a, b, name == "v_mul_hi_i32")
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => {
                // The low dword of {a, b} >> shift.
                let (a, b, sh) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let sh = if name == "v_alignbyte_b32" {
                    let three = k(self, 3);
                    let t = self.s.m.bin(c, Bin::And, sh, three);
                    let k3 = k(self, 3);
                    self.s.m.bin(c, Bin::Shl, t, k3)
                } else {
                    m31(self, sh)
                };
                let k32 = k(self, 32);
                let lo = self.s.m.bin(c, Bin::LShr, b, sh);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let left = m31(self, left);
                let hi = self.s.m.bin(c, Bin::Shl, a, left);
                let zero = k(self, 0);
                let none = self.s.m.cmp(c, pred::EQ, sh, zero);
                let hi = self.s.m.select(c, none, zero, hi);
                self.s.m.bin(c, Bin::Or, lo, hi)
            }
            "v_bfrev_b32" => {
                let a = s(self, 0)?;
                self.s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "v_bcnt_u32_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[a]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            "v_mad_u32_u24" | "v_mad_i32_i24" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mad_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            "v_mad_u32_u16" => {
                // S0.u16 * S1.u16 + S2.u32, wrapping (RDNA2 ISA, VOP3 opcode
                // 883); op_sel bits 0 and 1 pick the high halves of S0 and S1
                // (table 21). Bits 2 and 3 would name the 32-bit S2 and D,
                // which the manual leaves undefined: refused, as are
                // integer source modifiers.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                if v.opsel & 0b1100 != 0 || (v.abs | v.neg) & 0b11 != 0 {
                    return Err("op_sel or modifiers on a 32-bit operand");
                }
                let (a, b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                let d = s(self, 2)?;
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            "v_sad_u8" | "v_sad_hi_u8" | "v_sad_u16" | "v_sad_u32" | "v_msad_u8" => {
                // src2 plus the sum of the unsigned differences |a - b| of
                // the operands' bytes, halves or whole dwords, wrapping at 32
                // bits. v_sad_hi_u8 adds the sum shifted to the high half;
                // v_msad_u8 leaves out a byte whose mask byte (src1) is 0.
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (width, parts) = match name {
                    "v_sad_u16" => (16, 2),
                    "v_sad_u32" => (32, 1),
                    _ => (8, 4),
                };
                let zero = k(self, 0);
                let mut sum = zero;
                for i in 0..parts {
                    let (mut x, mut y) = (a, b);
                    if width < 32 {
                        let (sh, mask) = (k(self, width * i), k(self, (1 << width) - 1));
                        x = self.s.m.bin(c, Bin::LShr, x, sh);
                        x = self.s.m.bin(c, Bin::And, x, mask);
                        y = self.s.m.bin(c, Bin::LShr, y, sh);
                        y = self.s.m.bin(c, Bin::And, y, mask);
                    }
                    let above = self.s.m.cmp(c, pred::IUGT, x, y);
                    let hi = self.s.m.select(c, above, x, y);
                    let lo = self.s.m.select(c, above, y, x);
                    let mut diff = self.s.m.bin(c, Bin::Sub, hi, lo);
                    if name == "v_msad_u8" {
                        let masked = self.s.m.cmp(c, pred::NE, y, zero);
                        diff = self.s.m.select(c, masked, diff, zero);
                    }
                    sum = self.s.m.bin(c, Bin::Add, sum, diff);
                }
                if name == "v_sad_hi_u8" {
                    let k16 = k(self, 16);
                    sum = self.s.m.bin(c, Bin::Shl, sum, k16);
                }
                self.s.m.bin(c, Bin::Add, sum, d)
            }
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_i32"
            | "v_med3_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let signed = name.ends_with("i32");
                let (lt, gt) = if signed {
                    (pred::SLT, pred::SGT)
                } else {
                    (pred::IULT, pred::IUGT)
                };
                let pick = |l: &mut Self, p, x, y| {
                    let t = l.s.m.cmp(c, p, x, y);
                    l.s.m.select(c, t, x, y)
                };
                match &name[2..6] {
                    "min3" => {
                        let t = pick(self, lt, a, b);
                        pick(self, lt, t, d)
                    }
                    "max3" => {
                        let t = pick(self, gt, a, b);
                        pick(self, gt, t, d)
                    }
                    _ => {
                        let lo = pick(self, lt, a, b);
                        let hi = pick(self, gt, a, b);
                        let t = pick(self, lt, hi, d);
                        pick(self, gt, lo, t)
                    }
                }
            }
            "v_perm_b32" => {
                // Byte i of the result is byte S of {src0, src1} (src1 the
                // low dword) for selector byte S below 8; 8-11 repeat the
                // sign of bytes 1 and 3 of src1, then of src0; 12 is 0x00,
                // higher 0xFF.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                let sel = match v.src.get(2) {
                    Some(&Operand::Literal(x)) => x,
                    Some(&Operand::Const(k)) => crate::resource::inline(k).ok_or(name)?,
                    _ => {
                        // A computed selector: each byte picked at run time.
                        let (a, b, sv) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                        let mut r = k(self, 0);
                        for i in 0..4u32 {
                            let (sh, ff) = (k(self, 8 * i), k(self, 0xFF));
                            let sel = self.s.m.bin(c, Bin::LShr, sv, sh);
                            let sel = self.s.m.bin(c, Bin::And, sel, ff);
                            // Select a dword and a byte within it.
                            // Later selects handle codes 8 and above.
                            let four = k(self, 4);
                            let zero = k(self, 0);
                            let high = self.s.m.bin(c, Bin::And, sel, four);
                            let high = self.s.m.cmp(c, pred::NE, high, zero);
                            let word = self.s.m.select(c, high, a, b);
                            let three = k(self, 3);
                            let index = self.s.m.bin(c, Bin::And, sel, three);
                            let bits = self.s.m.bin(c, Bin::Shl, index, three);
                            let x = self.s.m.bin(c, Bin::LShr, word, bits);
                            let mut byte = self.s.m.bin(c, Bin::And, x, ff);
                            // 8-11: the sign of bytes 1, 3 of src1, then of src0.
                            for (n, src, bit) in [(8, b, 15), (9, b, 31), (10, a, 15), (11, a, 31)]
                            {
                                let left = k(self, 31 - bit);
                                let t = self.s.m.bin(c, Bin::Shl, src, left);
                                let k31 = k(self, 31);
                                let t = self.s.m.bin(c, Bin::AShr, t, k31);
                                let t = self.s.m.bin(c, Bin::And, t, ff);
                                let kn = k(self, n);
                                let is = self.s.m.cmp(c, pred::EQ, sel, kn);
                                byte = self.s.m.select(c, is, t, byte);
                            }
                            let (k12, z) = (k(self, 12), k(self, 0));
                            let is12 = self.s.m.cmp(c, pred::EQ, sel, k12);
                            byte = self.s.m.select(c, is12, z, byte);
                            let above = self.s.m.cmp(c, pred::IUGT, sel, k12);
                            byte = self.s.m.select(c, above, ff, byte);
                            let at = k(self, 8 * i);
                            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                            r = self.s.m.bin(c, Bin::Or, r, byte);
                        }
                        return Ok(r);
                    }
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut r = k(self, 0);
                for i in 0..4 {
                    let pick = sel >> (8 * i) & 0xFF;
                    let byte = match pick {
                        0..=7 => {
                            let (src, n) = if pick < 4 { (b, pick) } else { (a, pick - 4) };
                            let sh = k(self, 8 * n);
                            let t = self.s.m.bin(c, Bin::LShr, src, sh);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        8..=11 => {
                            let src = if pick < 10 { b } else { a };
                            let bit = if pick % 2 == 0 { 15 } else { 31 };
                            let left = k(self, 31 - bit);
                            let t = self.s.m.bin(c, Bin::Shl, src, left);
                            let k31 = k(self, 31);
                            let t = self.s.m.bin(c, Bin::AShr, t, k31);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        12 => continue,
                        _ => k(self, 0xFF),
                    };
                    let at = k(self, 8 * i);
                    let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                    r = self.s.m.bin(c, Bin::Or, r, byte);
                }
                r
            }
            "v_bfm_b32" => {
                // ((1 << src0[4:0]) - 1) << src1[4:0]
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let one = k(self, 1);
                let (a, b) = (m31(self, a), m31(self, b));
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            "v_cvt_pk_u8_f32" => {
                // src0 rounded to nearest even and saturated to a byte, into
                // byte src1[1:0] of src2.
                let f = self.fsrc(inst, 0)?;
                let (b, d) = (s(self, 1)?, s(self, 2)?);
                let f = self.unary(26, f);
                let (zero, max) = (self.s.m.f32(0), self.s.m.f32(255.0f32.to_bits()));
                let f = self.binary(35, f, zero);
                let f = self.binary(36, f, max);
                let x = self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32);
                let (three, k3) = (k(self, 3), k(self, 3));
                let sh = self.s.m.bin(c, Bin::And, b, three);
                let sh = self.s.m.bin(c, Bin::Shl, sh, k3);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let ff = k(self, 0xFF);
                let hole = self.s.m.bin(c, Bin::Shl, ff, sh);
                let ones = k(self, u32::MAX);
                let hole = self.s.m.bin(c, Bin::Xor, hole, ones);
                let d = self.s.m.bin(c, Bin::And, d, hole);
                self.s.m.bin(c, Bin::Or, d, x)
            }
            "v_ffbl_b32" => {
                // -1 for zero, as DXIL's first-bit-low.
                let a = s(self, 0)?;
                self.s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[a])
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                // Bits of the mask's low (high) half below this lane, plus src1:
                // with an all-ones mask, the lane index.
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                // Mesh lanes are the console's by construction; pixel and
                // vertex lanes are the host's, as their masks are.
                if self.mask_data == MaskData::Host {
                    // Host lanes: the mask must be one (a ballot) or all lanes.
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let host = match v.src.first() {
                        Some(&Operand::Const(193)) => true,
                        Some(&Operand::Scalar { code, .. }) => {
                            inst.read(Reg::S(code)).is_some_and(|m| {
                                self.x.st.masks.contains_key(&m) || self.x.st.highs.contains_key(&m)
                            })
                        }
                        _ => false,
                    };
                    if !host {
                        return Err("v_mbcnt of a mask that is no ballot (console lane layout)");
                    }
                    self.s.flags |= flags::WAVE_OPS;
                } else if self.s.mesh.is_none() {
                    self.cross_lane(false, outside_compute!("v_mbcnt of a console lane mask"))?;
                }
                let lane = self.lane_index();
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    lane
                } else {
                    let (k32, z) = (k(self, 32), k(self, 0));
                    let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                    let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                    self.s.m.select(c, low, z, t)
                };
                // (1 << below) - 1, all ones from 32 on.
                let (one, k32, ones) = (k(self, 1), k(self, 32), k(self, u32::MAX));
                let sh = m31(self, below);
                let m = self.s.m.bin(c, Bin::Shl, one, sh);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                let full = self.s.m.cmp(c, pred::IUGE, below, k32);
                let m = self.s.m.select(c, full, ones, m);
                let x = self.s.m.bin(c, Bin::And, a, m);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[x]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            _ => return Err(name),
        })
    }

    fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let c = self.cur;
        let r = match ty {
            "f32" | "f16" if op == "class" => {
                // Bit k of src1 tests class k: signalling NaN, quiet NaN,
                // -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf.
                let f16 = ty == "f16";
                let x = if f16 {
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let mut h = self.half_src(inst, 0)?;
                    if v.abs & 1 != 0 {
                        let m = self.s.m.i32(0x7FFF);
                        h = self.s.m.bin(c, Bin::And, h, m);
                    }
                    if v.neg & 1 != 0 {
                        let s = self.s.m.i32(0x8000);
                        h = self.s.m.bin(c, Bin::Xor, h, s);
                    }
                    h
                } else {
                    let x = self.fsrc(inst, 0)?;
                    self.as_int(x)
                };
                let mask = self.src(inst, 1)?;
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit) = if f16 {
                    (10, 15, 0x1F, 0x03FF, 0x0200)
                } else {
                    (23, 31, 0xFF, 0x7F_FFFF, 0x40_0000)
                };
                let (k_exp, k_sign, k_emax, mant, quiet) = (
                    k(self, exp_sh),
                    k(self, sign_sh),
                    k(self, exp_mask),
                    k(self, mant_mask),
                    k(self, quiet_bit),
                );
                let e = self.s.m.bin(c, Bin::LShr, x, k_exp);
                let e = self.s.m.bin(c, Bin::And, e, k_emax);
                let m = self.s.m.bin(c, Bin::And, x, mant);
                let neg = self.s.m.bin(c, Bin::LShr, x, k_sign);
                let zero = k(self, 0);
                let (e0, emax) = (
                    self.s.m.cmp(c, pred::EQ, e, zero),
                    self.s.m.cmp(c, pred::EQ, e, k_emax),
                );
                let m0 = self.s.m.cmp(c, pred::EQ, m, zero);
                let q = self.s.m.bin(c, Bin::And, x, quiet);
                let q = self.s.m.cmp(c, pred::NE, q, zero);
                // Positive classes 6-9; negative ones are 11 minus those.
                let (normal, inf, zero_c, denorm) =
                    (k(self, 8), k(self, 9), k(self, 6), k(self, 7));
                let t = self.s.m.select(c, m0, zero_c, denorm);
                let t = self.s.m.select(c, e0, t, normal);
                let t = self.s.m.select(c, emax, inf, t);
                let eleven = k(self, 11);
                let mirrored = self.s.m.bin(c, Bin::Sub, eleven, t);
                let is_neg = self.s.m.cmp(c, pred::NE, neg, zero);
                let t = self.s.m.select(c, is_neg, mirrored, t);
                let (one, snan) = (k(self, 1), k(self, 0));
                let nan_class = self.s.m.select(c, q, one, snan);
                let yes = self.s.m.bool(true);
                let nan = self.s.m.bin(c, Bin::Xor, m0, yes);
                let nan = self.s.m.bin(c, Bin::And, nan, emax);
                let t = self.s.m.select(c, nan, nan_class, t);
                let bit = self.s.m.bin(c, Bin::LShr, mask, t);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                self.s.m.cmp(c, pred::NE, bit, zero)
            }
            "f32" | "f16" => {
                let (a, b) = if ty == "f16" {
                    (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
                } else {
                    (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?)
                };
                if op == "lg" {
                    // Ordered and not equal, false for a NaN, as `olt | ogt`.
                    // `fcmp one` means the same, but the AMD D3D12 driver
                    // runs it as unordered (true for a NaN) where NVIDIA
                    // and WARP do not; olt and ogt agree on all three.
                    let lt = self.s.m.cmp(c, pred::OLT, a, b);
                    let gt = self.s.m.cmp(c, pred::OGT, a, b);
                    self.s.m.bin(c, Bin::Or, lt, gt)
                } else if op == "f" || op == "tru" {
                    self.s.m.bool(op == "tru")
                } else {
                    let p = match op {
                        "f" => pred::FALSE,
                        "lt" => pred::OLT,
                        "eq" => pred::OEQ,
                        "le" => pred::OLE,
                        "gt" => pred::OGT,
                        "ge" => pred::OGE,
                        "o" => pred::ORD,
                        "u" => pred::UNO,
                        "nge" => pred::ULT,
                        "nlg" => pred::UEQ,
                        "ngt" => pred::ULE,
                        "nle" => pred::UGT,
                        "neq" => pred::UNE,
                        "nlt" => pred::UGE,
                        "tru" => pred::TRUE,
                        _ => return Err(inst.name()),
                    };
                    self.s.m.cmp(c, p, a, b)
                }
            }
            "u64" | "i64" => {
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let signed = ty == "i64";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    _ => return Err("a 64-bit compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            "u32" | "i32" | "u16" | "i16" => {
                let (a, b) = if ty.ends_with("16") {
                    let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                    if ty == "i16" {
                        let k16 = self.s.m.i32(16);
                        for x in [&mut a, &mut b] {
                            let t = self.s.m.bin(c, Bin::Shl, *x, k16);
                            *x = self.s.m.bin(c, Bin::AShr, t, k16);
                        }
                    }
                    (a, b)
                } else {
                    (self.src(inst, 0)?, self.src(inst, 1)?)
                };
                let signed = ty == "i32" || ty == "i16";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    "f" | "t" => {
                        self.lane(inst, Some(op == "t"));
                        return Ok(());
                    }
                    _ => return Err("an integer compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r);
        Ok(())
    }

    fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            return Err("the first half of an interpolation");
        };
        let (attribute, mode) = match interp {
            Interp::Interpolated { unless, .. } if !self.avoids(site, unless) => {
                return Err("an interpolation with computed barycentrics");
            }
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => (
                attribute,
                match barycentric {
                    Barycentric::PerspectiveCenter => 2,
                    Barycentric::PerspectiveCentroid => 3,
                    Barycentric::PerspectiveSample => 6,
                    Barycentric::LinearCenter => 4,
                    Barycentric::LinearCentroid => 5,
                    Barycentric::LinearSample => 7,
                    Barycentric::PerspectivePullModel => return Err("pull-model interpolation"),
                },
            ),
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, 1),
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
        };
        let dst = match &inst.fields {
            Fields::Vintrp { dst, .. } => *dst,
            Fields::Valu(v) => v.dst.ok_or("an interpolation")?,
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A channel of an input element, interpolated as `mode` says.
        let channel = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let (orig_id, col) = l
                .find_input(row, ch)
                .ok_or("an input the signature lacks")?;
            let id = if l.s.inputs[orig_id].interp != mode && l.s.inputs[orig_id].usage != 0 {
                if l.s.inputs[orig_id].kind != 0 {
                    return Err("one input interpolated two ways");
                }
                // shortcut: a generic input interpolated two ways is split onto
                // a fresh row and unused semantic index so DXIL keeps one
                // interpolation mode per row; the vertex shader only exports
                // the original semantic index unless extended to duplicate it.
                if let Some(&split_id) = l.split_inputs.get(&(orig_id, mode)) {
                    split_id
                } else {
                    let orig = l.s.inputs[orig_id].clone();
                    let next_index = l
                        .s
                        .inputs
                        .iter()
                        .filter(|x| x.name == orig.name)
                        .map(|x| x.index + 1)
                        .max()
                        .unwrap_or(0);
                    let free_row = l
                        .s
                        .inputs
                        .iter()
                        .filter_map(|x| x.row)
                        .map(|r| r + 1)
                        .max()
                        .unwrap_or(0);
                    l.s.inputs.push(SigElement {
                        name: orig.name,
                        index: next_index,
                        kind: orig.kind,
                        comp: orig.comp,
                        interp: mode,
                        row: Some(free_row),
                        col: orig.col,
                        cols: orig.cols,
                        usage: 0,
                        // Clear `half` so `packed_half` keeps its original slot layout.
                        half: false,
                    });
                    let split_id = l.s.inputs.len() - 1;
                    l.split_inputs.insert((orig_id, mode), split_id);
                    split_id
                }
            } else {
                l.s.inputs[orig_id].interp = mode;
                orig_id
            };
            match l.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            }
        };
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| i.pack2),
            _ => None,
        };
        let f16 = inst.name().ends_with("_f16");
        // Which read, so a refusal log says what the stage data lacks (an input it does not
        // list at all is refused before this).
        let unmarked = if attribute.high {
            "a 16-bit read of a 32-bit input: the high half of an input not marked 16-bit"
        } else {
            "a 16-bit read of a 32-bit input: an f16 interpolation of an input not marked 16-bit"
        };
        // A guessed pipeline's half: its own input element (`graphics::HALF`), an f32.
        let guessed_half = (self.guessed && packing.is_none() && f16)
            .then(|| {
                let index = 2 * u32::from(attribute.row) + u32::from(attribute.high);
                self.s.inputs.iter().find_map(|e| {
                    (e.name == super::graphics::HALF
                        && e.index == index
                        && (e.col..e.col + e.cols).contains(&attribute.channel))
                    .then_some(e.row)
                    .flatten()
                })
            })
            .flatten();
        if let Some(row) = guessed_half {
            let row = u8::try_from(row).map_err(|_| "an input row")?;
            let x = channel(self, row, attribute.channel)?;
            let f = self.as_float(x);
            let h = self.narrow16(f);
            let h = self.half_modifiers(inst, h);
            return self.write_half(inst, h, None);
        }
        let x = match packing {
            None if f16 || attribute.high => return Err(unmarked),
            None => channel(self, attribute.row, attribute.channel)?,
            Some(pack2) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 {
                        packed_half(&l.s.inputs, attribute.row, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    let Some((row, ch)) = src else {
                        return Ok(None);
                    };
                    let x = channel(l, row, ch)?;
                    let f = l.as_float(x);
                    Ok(Some(l.narrow16(f)))
                };
                let h = 2 * attribute.channel;
                let c = self.cur;
                if f16 {
                    let lo = half(self, h + u8::from(attribute.high))?
                        .ok_or("a packed input half no element holds")?;
                    // omod and clamp are part of the f16 instruction; op_sel[3]
                    // picks the destination half.
                    let lo = self.half_modifiers(inst, lo);
                    return self.write_half(inst, lo, None);
                } else if mode == 1 {
                    // The whole dword of a flat input: both halves.
                    let z = self.s.m.i32(0);
                    let lo = half(self, h)?.unwrap_or(z);
                    let hi = half(self, h + 1)?.unwrap_or(z);
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// Whether no lane that runs the instruction at `site` has been where
    /// `unless` says: a lane that came along such an edge or ran (skipped)
    /// such an instruction meets its condition, which the instruction's must
    /// contradict. Those lane bits must not change in between, so none of
    /// those blocks is on a cycle. A lane mask joined where such an edge
    /// enters holds, for the lanes that came along it, the edge's mask.
    fn avoids(&self, (b, i): Site, unless: &[Elsewhere]) -> bool {
        let st = self.x.st;
        let runs = [
            (st.reach[b], true),
            (st.guards[b][i].unwrap_or(Bool::TRUE), true),
        ];
        let mut atoms = st.bools.support(runs[0].0);
        atoms.extend(st.bools.support(runs[1].0));
        unless.iter().all(|&u| {
            let (mut all, at, mut links) = match u {
                Elsewhere::Edge(q, to) => {
                    let all = self.lane_edge(q, to);
                    let block = &self.x.p.blocks[to];
                    let n = block.preds.iter().position(|&p| p == q).expect("an edge");
                    let links: Vec<(Bool, Bool)> = block
                        .phis
                        .iter()
                        .filter(|ph| atoms.contains(&Atom::Phi(ph.value)))
                        .filter_map(|ph| {
                            Some((*st.masks.get(&ph.value)?, *st.masks.get(&ph.args[n])?))
                        })
                        .collect();
                    (all, to, links)
                }
                Elsewhere::Write((w, k)) | Elsewhere::Skip((w, k)) => {
                    let guard = st.guards[w][k].unwrap_or(Bool::TRUE);
                    let ran = matches!(u, Elsewhere::Write(_));
                    (vec![(st.reach[w], true), (guard, ran)], w, vec![])
                }
            };
            all.extend(runs);
            // Each joined mask equals the incoming one: try both values.
            links.truncate(6);
            let some_lane = (0..1u32 << links.len()).any(|bits| {
                let mut all = all.clone();
                for (k, &(joined, incoming)) in links.iter().enumerate() {
                    let v = bits >> k & 1 != 0;
                    all.extend([(joined, v), (incoming, v)]);
                }
                st.bools.meet(&all)
            });
            !self.on_cycle(at) && !some_lane
        })
    }

    /// What holds in a lane that goes from IR block `q` to `b`.
    fn lane_edge(&self, q: usize, b: usize) -> Vec<(Bool, bool)> {
        let st = self.x.st;
        let mut all = vec![(st.reach[q], true)];
        if let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = st.exits[q]
            && taken != not_taken
        {
            all.push((when, taken == b));
        }
        all
    }

    /// Every (join value, edge) whose incoming value no lane of that edge sees, as the greatest
    /// fixpoint: each instruction reading the join is an AND or NAND with EXEC (or an `s_orn2`
    /// by EXEC) with no lane of the edge in that EXEC as the edge brings it, or an `s_andn2` by
    /// EXEC with every lane of the edge in it (those lanes' bits of the result are 0, or 1,
    /// whatever the join holds for them), in the join's block or where neither runs again; and each
    /// later join that takes it is on an edge those lanes do not take from the block, or is in
    /// the set itself (a register the console compiler reuses on a path those lanes skip, as a
    /// nested if's else part writes it again before any read; a loop passing it round).
    fn unneeded_edges(&self) -> &Set<(Value, usize)> {
        self.unneeded.get_or_init(|| {
            let (p, st) = (self.x.p, self.x.st);
            let mut reads: Map<Value, Vec<(usize, usize)>> = Map::default();
            let mut uses: Map<Value, Vec<(usize, usize, usize)>> = Map::default();
            for (b, blk) in p.blocks.iter().enumerate() {
                for (i, inst) in blk.insts.iter().enumerate() {
                    for r in &inst.reads {
                        reads.entry(r.1).or_default().push((b, i));
                    }
                }
                for (i, phi) in blk.phis.iter().enumerate() {
                    for (k, &a) in phi.args.iter().enumerate() {
                        uses.entry(a).or_default().push((b, i, k));
                    }
                }
            }
            let mut cycle_cache: Vec<Option<bool>> = vec![None; p.blocks.len()];
            let mut on_cycle = |x: usize| -> bool {
                *cycle_cache[x].get_or_insert_with(|| self.on_cycle(x))
            };
            let mut set = Set::default();
            let mut read_execs = Vec::new();
            for (b, blk) in p.blocks.iter().enumerate() {
                for phi in blk.phis.iter().filter(|phi| matches!(phi.reg, Reg::S(_))) {
                    let reads = reads.get(&phi.value).map_or(&[][..], Vec::as_slice);
                    read_execs.clear();
                    let eligible = reads.iter().all(|&(bb, i)| {
                        let inst = &p.blocks[bb].insts[i];
                        let by_exec = matches!(&inst.fields, Fields::Sop { src, .. }
                            if matches!(src.get(1), Some(Operand::Scalar { code: EXEC, .. })));
                        // The lanes whose bits of the join the result keeps: EXEC's, or
                        // (`s_andn2` by EXEC) the others.
                        let keeps_exec = match inst.name() {
                            "s_and_b32" | "s_and_b64" | "s_nand_b32" | "s_nand_b64" => true,
                            "s_orn2_b32" | "s_orn2_b64" if by_exec => true,
                            "s_andn2_b32" | "s_andn2_b64" if by_exec => false,
                            _ => return false,
                        };
                        // In another block the edge's lanes are the same lanes only where
                        // neither block runs again (the else part of an if, further down).
                        let Some(exec) = inst
                            .read(Reg::S(EXEC))
                            .filter(|_| bb == b || !on_cycle(b) && !on_cycle(bb))
                        else {
                            return false;
                        };
                        read_execs.push((exec, keeps_exec));
                        true
                    });
                    if !eligible {
                        continue;
                    }
                    for n in 0..blk.preds.len() {
                        let off = read_execs.iter().all(|&(exec, keeps_exec)| {
                            let joined = self
                                .phi_at
                                .get(&exec)
                                .filter(|&&(pb, _)| pb == b)
                                .map(|&(_, pi)| &blk.phis[pi]);
                            let exec = joined.map_or(exec, |ph| ph.args[n]);
                            let Some(&m) = st.masks.get(&exec) else { return false };
                            let mut lanes = self.lane_edge(blk.preds[n], b);
                            lanes.push((m, keeps_exec));
                            !st.bools.meet(&lanes)
                        });
                        if off {
                            set.insert((phi.value, n));
                        }
                    }
                }
            }
            let mut relay_cache: Map<(usize, usize, usize), bool> = Map::default();
            let mut dead_relay = |b: usize, n: usize, e: usize| -> bool {
                *relay_cache.entry((b, n, e)).or_insert_with(|| {
                    let mut path = self.lane_edge(p.blocks[b].preds[n], b);
                    path.extend(self.lane_edge(b, e));
                    !st.bools.meet(&path)
                })
            };
            let mut dependents: Map<(Value, usize), Vec<(Value, usize)>> = Map::default();
            let mut worklist = Vec::new();
            for (b, blk) in p.blocks.iter().enumerate() {
                for phi in blk.phis.iter().filter(|phi| matches!(phi.reg, Reg::S(_))) {
                    let uses = uses.get(&phi.value).map_or(&[][..], Vec::as_slice);
                    for n in 0..blk.preds.len() {
                        let item = (phi.value, n);
                        if !set.contains(&item) {
                            continue;
                        }
                        for &(e, i, k) in uses {
                            if p.blocks[e].preds[k] == b && dead_relay(b, n, e) {
                                continue;
                            }
                            let next = (p.blocks[e].phis[i].value, k);
                            if !set.contains(&next) {
                                set.remove(&item);
                                worklist.push(item);
                                break;
                            }
                            dependents.entry(next).or_default().push(item);
                        }
                    }
                }
            }
            while let Some(bad) = worklist.pop() {
                if let Some(deps) = dependents.remove(&bad) {
                    for dep in deps {
                        if set.remove(&dep) {
                            worklist.push(dep);
                        }
                    }
                }
            }
            set
        })
    }

    /// Whether every lane enters IR block `b` along the edge the console's
    /// wave takes, so that an SGPR joined there holds what that edge
    /// brought: every branch into it is uniform (SCC values). The wave runs
    /// one path; a per-lane branch into the join (an EXEC skip, a loop's
    /// per-lane exit) is where lanes would take edges the wave did not, and
    /// the console compiler's if/else always skips into its join that way.
    /// Lanes that left EXEC inside a uniform loop (the BVH walks) take no
    /// edge into its header; the others take the wave's.
    fn uniform_join(&self, b: usize, uniform_scc: impl Fn(Value) -> bool) -> bool {
        let st = self.x.st;
        let preds = &self.x.p.blocks[b].preds;
        preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .all(|&q| match st.exits[q] {
                LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } if taken != not_taken => st
                    .bools
                    .support(when)
                    .iter()
                    .all(|a| matches!(a, Atom::Uniform(v) if uniform_scc(*v))),
                _ => true,
            })
    }

    /// Whether no lane goes from IR block `q` to `b`.
    fn never_taken(&self, q: usize, b: usize) -> bool {
        !self.x.st.bools.meet(&self.lane_edge(q, b))
    }

    /// Whether, at a per-lane join `b` of a lane mask whose register holds
    /// data along edge `n`, some edge brings a mask and no lane taking edge
    /// `n` is in any of them (a compare inside a region those lanes skip).
    /// The console wave runs one path: on a mask edge, the lanes of edge `n`
    /// read their bit of that mask, 0. The compiler merges no value for them,
    /// so it holds none for them either where the whole wave skips (the data
    /// there is a reused register): 0 is theirs.
    fn no_mask_lane_on_edge(&self, b: usize, args: &[Value], n: usize) -> bool {
        let st = self.x.st;
        let preds = &self.x.p.blocks[b].preds;
        let on_n = self.lane_edge(preds[n], b);
        let mut some = false;
        for (k, arg) in args.iter().enumerate() {
            let Some(&m) = st.masks.get(arg).filter(|_| k != n) else {
                continue;
            };
            some = true;
            let mut lanes = on_n.clone();
            lanes.push((m, true));
            if st.bools.meet(&lanes) {
                return false;
            }
        }
        some
    }

    /// Whether, at a per-lane join `b` that edge `n` reaches by a per-lane branch, another edge
    /// brings a mask that may differ from the lane function `x` in a lane taking edge `n`. The
    /// console wave runs every path some lane takes, so where that edge's block ran, the
    /// register holds its mask for the lanes of edge `n` too: their bit of it, which `x`, what
    /// edge `n` brings, then does not tell.
    fn mask_differs_on_edge(&self, b: usize, args: &[Value], n: usize, x: Bool) -> bool {
        let st = self.x.st;
        let preds = &self.x.p.blocks[b].preds;
        // Where the whole wave leaves by edge `n`, no other path runs before the join.
        let per_lane = matches!(st.exits[preds[n]], LaneExit::Branch { when, taken, not_taken }
            if taken != not_taken
                && st.bools.support(when).iter().any(|a| !matches!(a, Atom::Uniform(_))));
        let on_n = self.lane_edge(preds[n], b);
        // Whether a lane of edge `n` that `part` holds for, brought `x` (`None`: a value not
        // known, which differs wherever there is such a lane), differs from another edge's mask.
        let differs = |part: &[(Bool, bool)], x: Option<Bool>| {
            let on_n = [on_n.as_slice(), part].concat();
            let Some(x) = x else { return st.bools.meet(&on_n) };
            args.iter().enumerate().any(|(k, arg)| {
                // A back edge's block (one the join dominates) runs after the lanes of edge
                // `n` have gone on. An arm inside a loop's body is no back edge.
                k != n
                    && !self.dominates(b, preds[k])
                    && st.masks.get(arg).is_some_and(|&m| {
                        [true, false].into_iter().any(|set| {
                            let mut lanes = on_n.clone();
                            lanes.extend([(m, set), (x, !set)]);
                            st.bools.meet(&lanes)
                        })
                    })
            })
        };
        // Where what the edge brings is a join made in its own block (an if's then part writes
        // the register, and this edge skips the else part), `x` is an atom nothing is known
        // of: each edge into that block is weighed too, for its own lanes and what it brought.
        let p = self.x.p;
        per_lane
            && differs(&[], Some(x))
            && self.phi_at.get(&args[n]).is_none_or(|&(q, i)| {
                q != preds[n]
                    || self.on_cycle(q)
                    || p.blocks[q].phis[i].args.iter().zip(&p.blocks[q].preds).any(|(a, &from)| {
                        let known = st.masks.get(a).or_else(|| st.other.get(a)).copied();
                        differs(&self.lane_edge(from, q), known)
                    })
            })
    }

    /// Whether IR block `b` dominates `q`: no path from the entry gets to `q` but through `b`.
    fn dominates(&self, b: usize, q: usize) -> bool {
        let blocks = &self.x.p.blocks;
        let mut seen = vec![false; blocks.len()];
        seen[b] = true;
        let mut stack = vec![0];
        while let Some(x) = stack.pop() {
            if !std::mem::replace(&mut seen[x], true) {
                if x == q {
                    return false;
                }
                stack.extend(blocks[x].exit.successors());
            }
        }
        true
    }

    /// Whether IR block `b` can reach itself.
    fn on_cycle(&self, b: usize) -> bool {
        let blocks = &self.x.p.blocks;
        let mut seen = vec![false; blocks.len()];
        let mut stack = blocks[b].exit.successors();
        while let Some(x) = stack.pop() {
            if x == b {
                return true;
            }
            if !std::mem::replace(&mut seen[x], true) {
                stack.extend(blocks[x].exit.successors());
            }
        }
        false
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    /// VOP3 f16 output modifiers: omod, then clamp, applied in f32 so the
    /// half conversion rounds once (the order `write_f` uses).
    fn half_modifiers(&mut self, inst: &Inst, h: V) -> V {
        let Fields::Valu(v) = &inst.fields else {
            return h;
        };
        if v.omod == 0 && !v.clamp {
            return h;
        }
        let c = self.cur;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.omod != 0 {
            let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(v.omod)];
            let k = self.s.m.f32(factor.to_bits());
            f = self.s.m.bin(c, Bin::Mul, f, k);
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        self.s
            .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[f])
    }

    /// An image instruction, its reads of texture views marked (module
    /// notes, "Read marks") after it.
    fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        self.s.texture_reads.clear();
        let lowered = self.image_op(inst, b, i);
        for index in std::mem::take(&mut self.s.texture_reads) {
            if lowered.is_ok() {
                self.mark_view_read(index);
            }
        }
        lowered
    }

    /// Sets the read mark of resource-heap index `index` (module notes,
    /// "Read marks") where it is still clear. Nothing without
    /// [`Parts::view_marks`].
    fn mark_view_read(&mut self, index: V) {
        if !self.x.view_marks {
            return;
        }
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let bitmap = self.s.view_writes_handle();
        let c = self.cur;
        let (base, k5, k2, k31, one, zero) = (
            k(self, super::VIEW_READS),
            k(self, 5),
            k(self, 2),
            k(self, 31),
            k(self, 1),
            k(self, 0),
        );
        let n = self.s.m.bin(c, Bin::Add, index, base);
        let word = self.s.m.bin(c, Bin::LShr, n, k5);
        let word = self.s.m.bin(c, Bin::Shl, word, k2);
        let bit = self.s.m.bin(c, Bin::And, n, k31);
        let bit = self.s.m.bin(c, Bin::Shl, one, bit);
        // Most reads find their slot marked: a load, not an atomic.
        let old = self.raw_dword(bitmap, word);
        let set = self.s.m.bin(c, Bin::And, old, bit);
        let clear = self.s.m.cmp(c, pred::EQ, set, zero);
        let at = self.begin_guarded(Some(clear));
        let iu = self.s.m.undef(self.s.t.i32);
        // Operation 2 is OR.
        self.s.op(
            self.cur,
            "atomicBinOp.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            78,
            &[bitmap, k2, word, iu, iu, bit],
        );
        self.end_guarded(at, &[]);
    }

    fn image_op(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        if name.starts_with("image_bvh") {
            return self.bvh(inst, b, i);
        }
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // With a16, two 16-bit components per VGPR, low half first, after
        // the offset, bias and compare value, which keep a VGPR each (an f16
        // bias in its low half).
        let extra = name
            .strip_prefix("image_sample")
            .or_else(|| name.strip_prefix("image_gather4"))
            .map_or(0, |rest| {
                rest.split('_')
                    .filter(|t| matches!(*t, "o" | "b" | "c"))
                    .count()
            });
        let addr_val = |l: &mut Self, k: usize| -> Result<V, Why> {
            let packed = *a16 && k >= extra;
            let r = *regs
                .get(if packed { extra + (k - extra) / 2 } else { k })
                .ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            let x = l.i32_of(v)?;
            if !packed {
                return Ok(x);
            }
            let c = l.cur;
            let half = (k - extra) as u32 % 2;
            let (sh, mask) = (l.s.m.i32(16 * half), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // DXIL's atomic operations on a 32-bit integer texture UAV; the
            // old value returns with glc. Sub is the add of the negation.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            // 16-bit coordinates (a16) are read as `addr_val` reads them.
            if *d16 {
                return Err("a 16-bit image atomic");
            }
            // dmask: the data dwords (cmpswap: twice the value's); two per
            // value is a 64-bit atomic (an R32G32_UINT view, low dword in R).
            let values = if code == u32::MAX { 2 } else { 1 };
            let wide = match dmask.count_ones() / values {
                1 => false,
                2 => true,
                _ => return Err("an image atomic of more than two dwords"),
            };
            let n = if wide { 2 } else { 1 };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let vals = self.vgprs(inst, data, values as u8 * n)?;
            let i64t = self.s.m.int(64);
            // Value k of the data: one dword, or two as an i64.
            let value_at = |l: &mut Self, k: usize| -> V {
                if !wide {
                    return vals[k];
                }
                let lo = l.s.m.cast(c, Cast::ZExt, vals[2 * k], i64t);
                let hi = l.s.m.cast(c, Cast::ZExt, vals[2 * k + 1], i64t);
                let k32 = l.s.m.int_const(i64t, 32);
                let hi = l.s.m.bin(c, Bin::Shl, hi, k32);
                l.s.m.bin(c, Bin::Or, lo, hi)
            };
            let mut value = value_at(self, 0);
            if op == "sub" {
                let z = if wide {
                    self.s.m.int_const(i64t, 0)
                } else {
                    self.s.m.i32(0)
                };
                value = self.s.m.bin(c, Bin::Sub, z, value);
            }
            let mut old = Vec::new();
            for k in 0..n {
                old.push(match inst.read(Reg::V(reg + k)) {
                    Some(v) if cache.glc => self.i32_of(v)?,
                    _ => self.s.m.i32(0),
                });
            }
            if wide && self.x.atomic_locks {
                let compare = (code == u32::MAX).then(|| value_at(self, 1));
                let h = self.handle(site, src, false, (kind | 0x1000 | 0x4000, 0x205))?;
                let at = self.begin_guarded(guard);
                let (lo, hi) = self.locked_atomic64(h, &cs, coords, op, value, compare);
                let r = self.end_guarded(at, &[(lo, old[0]), (hi, old[1])]);
                if cache.glc {
                    for (k, x) in (0..).zip(r) {
                        self.write(inst, Reg::V(reg + k), x, None);
                    }
                }
                return Ok(());
            }
            if wide {
                // SM 6.6's 64-bit atomics on typed resources in the heap: the
                // host device must support both (D3D12 options 9 and 11).
                self.s.flags |= flags::INT64 | flags::ATOMIC64_TYPED | flags::ATOMIC64_HEAP;
            }
            let h = self.handle(site, src, false, (kind | 0x1000, 0x105))?;
            let at = self.begin_guarded(guard);
            let (ty, suffix) = if wide {
                (i64t, "i64")
            } else {
                (self.s.t.i32, "i32")
            };
            let mut args = vec![h];
            let r = if code == u32::MAX {
                // cmpswap: data is (new, compare).
                args.extend(cs);
                let (new, compare) = (value, value_at(self, 1));
                args.extend([compare, new]);
                self.s.op(
                    self.cur,
                    &format!("atomicCompareExchange.{suffix}"),
                    Attr::NoUnwind,
                    ty,
                    79,
                    &args,
                )
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                self.s.op(
                    self.cur,
                    &format!("atomicBinOp.{suffix}"),
                    Attr::NoUnwind,
                    ty,
                    78,
                    &args,
                )
            };
            let r = if wide {
                let (lo, hi) = self.split64(r);
                vec![lo, hi]
            } else {
                vec![r]
            };
            let inside: Vec<_> = r.into_iter().zip(old).collect();
            let r = self.end_guarded(at, &inside);
            if cache.glc {
                for (k, x) in (0..).zip(r) {
                    self.write(inst, Reg::V(reg + k), x, None);
                }
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            // shortcut: stores are taken to be of a float format (the
            // console converts by the view's format).
            // A store of fewer channels still writes the whole texel, the
            // data holding the dmask's channels in turn. With a prefix dmask
            // (0x1, 0x3, 0x7) the hardware stores 0 in the trailing channels
            // left out (GFX11 and older; LLVM 88421ea "[AMDGPU] Trim zero
            // components from buffer and image stores", Mesa 4a6ee2c4833
            // "aco: shrink buffer stores with undef/zero components"), as for
            // typed buffer stores. Guess (docs/native-host.md): a gapped
            // dmask (0x5, or one without red) gets 0 in its gaps too, which
            // that evidence does not cover. With no channel enabled the
            // hardware's behaviour is undocumented: refused.
            if matches!(inst.fields, Fields::Image { dmask: 0, .. }) {
                return Err("an image store of no channel");
            }
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            let mut k = 0u8;
            for ch in 0..4 {
                if dmask >> ch & 1 == 0 {
                    vals.push(self.s.m.f32(0));
                    continue;
                }
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
                k += 1;
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // `_o`: texel offsets, the first address (RDNA2 ISA 8.2.5), for any
        // sample or gather below.
        let base = match name.strip_suffix("_o") {
            Some(base) if base.starts_with("image_sample") || base.starts_with("image_gather4") => {
                base
            }
            _ => name,
        };
        let (op, sample) = match base {
            "image_sample" => (60, true),
            "image_sample_lz" | "image_sample_l" => (62, true),
            "image_sample_b" => (61, true),
            "image_sample_c_lz" => (65, true),
            "image_sample_d" | "image_sample_d_cl" => (63, true),
            "image_get_lod" if self.s.pixel => (81, true),
            // Implicit derivatives: pixel shaders only.
            "image_sample_c" if self.s.pixel => (64, true),
            "image_gather4" | "image_gather4_lz" | "image_gather4_l" => (73, true),
            // DXIL gathers from level 0 only, so with a comparison only `_lz`.
            "image_gather4_c_lz" => (74, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions; a channel DXIL leaves
            // out reads 1 (UAVs: one level).
            //
            // A writable image's query takes the program's level too, not the
            // undefined one DXC writes for a UAV: Metal answers that with 0,
            // which gave the title's sun-shadow min/max program a 0x0 image
            // (whole 16x16 tiles lit or shadowed together; seen under D3DMetal
            // in halfwave50/VIctory-macOS-halfwave50 0f3ed38, measured here by
            // `a_size_query_of_a_writable_image_reads_its_size`).
            // guess: on the console a level past a writable view's one reads
            // 0, as it does here; level 0, which those programs ask for, is
            // the view's own size on both.
            let h = self.handle(site, src, false, props)?;
            let mip = addr_val(self, 0)?;
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav || !present[ch as usize] {
                    one
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            if base == "image_gather4_l" {
                if *a16 || *d16 || *unorm || kind != 2 {
                    return Err(
                        "a gather at a level of 16-bit values, texel coordinates or not 2D",
                    );
                }
                let n = 3 + usize::from(name.ends_with("_o"));
                let a = (0..n)
                    .map(|k| addr_val(self, k))
                    .collect::<Result<Vec<_>, _>>()?;
                let lod = inst.read(Reg::V(regs[n - 1])).ok_or("an image address")?;
                return self.gather_at_level(inst, site, h, ss, &a, lod, guard);
            }
            // Comparison: the reference value is the first address after
            // the bias, and the sampler is a comparison sampler.
            let cmp = matches!(op, 64 | 65 | 74);
            if cmp && (*a16 || dmask.count_ones() != 1) {
                return Err("a comparison sample of 16-bit coordinates or several channels");
            }
            let sh = self.handle(site, ss, true, (14 | if cmp { 0x8000 } else { 0 }, 0))?;
            // Texel offsets (`_o`) come first: x in bits 0-5, y in 8-13,
            // signed (programmable offsets: gathers only).
            let offset_word = name.ends_with("_o");
            let first =
                usize::from(offset_word) + usize::from(base == "image_sample_b") + usize::from(cmp);
            // Coordinates the size divides (not the array slice); explicit
            // gradients (x then y) come before the coordinates.
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            // (A cube's are two per direction, on the face.)
            let grads = match op {
                63 if kind == 5 => 4,
                63 => 2 * spatial,
                _ => 0,
            };
            let mut cs = vec![fu; 4];
            // With a16, sampling coordinates and LOD are f16.
            let to_float = |l: &mut Self, x: V| {
                if *a16 {
                    let f = l.s.t.f32;
                    l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
                } else {
                    l.as_float(x)
                }
            };
            // A level query takes no array slice (the code may pass anything
            // there, e.g. an unwritten register).
            let read = if op == 81 { spatial } else { coords };
            for (k, slot) in cs.iter_mut().enumerate().take(read) {
                let x = addr_val(self, first + grads + k)?;
                *slot = to_float(self, x);
            }
            if kind == 5 {
                if *unorm || op == 81 {
                    return Err("a cube sample with texel coordinates or a level query");
                }
                let dir = self.cube_direction(cs[0], cs[1], cs[2]);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[h, sh, cs[0], cs[1], cs[2], clamped],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 {
                0
            } else {
                (coords - usize::from(kind == 6 || kind == 7)).min(3)
            };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            if offset_word && !matches!(op, 73 | 74) {
                if kind == 5 {
                    return Err("a cube sample with texel offsets");
                }
                // DXIL takes immediate offsets in -8..=7; the console's are
                // 6-bit signed (x in bits 0-5, y 8-13, z 16-21) and may be
                // computed.
                let w = addr_val(self, 0)?;
                let w = self
                    .s
                    .m
                    .int_value(w)
                    .ok_or("a sample offset that is not constant")? as u32;
                for (k, o) in os.iter_mut().enumerate().take(offsets) {
                    let x = (w >> (8 * k) & 0x3F) as i32;
                    let x = if x >= 32 { x - 64 } else { x };
                    if !(-8..=7).contains(&x) {
                        return Err("a sample offset past -8..7");
                    }
                    *o = self.s.m.i32(x as u32);
                }
            }
            let mut args = vec![h, sh];
            args.extend(&cs);
            match op {
                60 => {
                    args.extend(&os);
                    args.push(fu);
                    self.s
                        .op(c, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    // After the offset, if any.
                    let bias = addr_val(self, usize::from(offset_word))?;
                    let bias = to_float(self, bias);
                    if self.x.bias_as_gradients && self.x.native.stage == HwStage::Cs {
                        // D3DMetal ignores SampleBias in compute shaders
                        // (found by nickorlabs, #30). Scale explicit quad
                        // gradients by 2^bias instead:
                        // log2(rho * 2^bias) = log2(rho) + bias.
                        let scale = self.unary(21, bias);
                        let mut gradients = vec![fu; 6];
                        for k in 0..spatial {
                            let dx = self.unary(83, cs[k]);
                            let dy = self.unary(84, cs[k]);
                            gradients[k] = self.s.m.bin(c, Bin::Mul, dx, scale);
                            gradients[3 + k] = self.s.m.bin(c, Bin::Mul, dy, scale);
                        }
                        args.extend(&os);
                        args.extend(gradients);
                        args.push(fu);
                        self.s.op(
                            c,
                            "sampleGrad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            63,
                            &args,
                        )
                    } else {
                        args.extend(&os);
                        args.extend([bias, fu]);
                        self.s.op(
                            c,
                            "sampleBias.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            61,
                            &args,
                        )
                    }
                }
                63 if kind == 5 => {
                    // A cube's gradients are the face's: ds, dt per step in x,
                    // then in y, in face coordinates (a face spans 1). The level
                    // the hardware picks from them, isotropically: log2 of the
                    // longer gradient in texels of a face, then the clamp.
                    // shortcut: sampled at that level, so a sampler's
                    // anisotropy is lost and its LOD bias comes after the
                    // clamp (the console adds it before); D3D12 cube sampling
                    // takes 3D gradients only. Upgrade if cube lookups blur or
                    // a biased sampler meets a binding clamp.
                    let mut g = [fu; 4];
                    for (k, x) in g.iter_mut().enumerate() {
                        let v = addr_val(self, first + k)?;
                        *x = to_float(self, v);
                    }
                    let i32t = self.s.t.i32;
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let z = self.s.m.i32(0);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                    let w = self.s.m.extract(c, d, 0);
                    let w = self.s.m.cast(c, Cast::UiToFp, w, self.s.t.f32);
                    let mut len2 = [fu; 2];
                    for (k, l) in len2.iter_mut().enumerate() {
                        let a = self.s.m.bin(c, Bin::Mul, g[2 * k], g[2 * k]);
                        let b = self.s.m.bin(c, Bin::Mul, g[2 * k + 1], g[2 * k + 1]);
                        *l = self.s.m.bin(c, Bin::Add, a, b);
                    }
                    // FMax, Log (base 2): 0.5 log2(max(|dx|², |dy|²) w²).
                    let m = self.binary(35, len2[0], len2[1]);
                    let w2 = self.s.m.bin(c, Bin::Mul, w, w);
                    let m = self.s.m.bin(c, Bin::Mul, m, w2);
                    let half = self.s.m.f32(0.5f32.to_bits());
                    let l = self.unary(23, m);
                    let mut lod = self.s.m.bin(c, Bin::Mul, l, half);
                    if base.ends_with("_cl") {
                        let x = addr_val(self, first + grads + coords)?;
                        let clamp = to_float(self, x);
                        lod = self.binary(35, lod, clamp);
                    }
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                63 => {
                    // With a16 each direction's gradients start a VGPR.
                    if *a16 && spatial % 2 == 1 {
                        return Err("16-bit gradients of a 1D or 3D image");
                    }
                    let mut g = vec![fu; 6];
                    for k in 0..spatial {
                        let x = addr_val(self, first + k)?;
                        g[k] = to_float(self, x);
                        let y = addr_val(self, first + spatial + k)?;
                        g[3 + k] = to_float(self, y);
                    }
                    args.extend(&os);
                    args.extend(g);
                    // `_cl`: the least level of detail, after the coordinates.
                    let clamp = if base.ends_with("_cl") {
                        let x = addr_val(self, first + grads + coords)?;
                        to_float(self, x)
                    } else {
                        fu
                    };
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let reference = addr_val(self, first - 1)?;
                    let reference = self.as_float(reference);
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(fu);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                62 => {
                    // The level follows the coordinates.
                    let lod = if base == "image_sample_l" {
                        let x = addr_val(self, first + coords)?;
                        to_float(self, x)
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ => {
                    if base == "image_gather4" {
                        return Err("a gather with implicit level");
                    }
                    if dmask.count_ones() != 1 {
                        return Err("a gather of several channels");
                    }
                    // Guess: a comparison gather compares the channel dmask
                    // picks; DXIL's compares red only, so only red is taken.
                    if op == 74 && *dmask != 1 {
                        return Err("a comparison gather of another channel than red");
                    }
                    if offset_word {
                        let w = addr_val(self, 0)?;
                        for (k, at) in [(0, 26), (1, 18)] {
                            // Bits at..at+5 sign-extended: shift up, then down.
                            let left = self.s.m.i32(at);
                            let t = self.s.m.bin(c, Bin::Shl, w, left);
                            let k26 = self.s.m.i32(26);
                            os[k] = self.s.m.bin(c, Bin::AShr, t, k26);
                        }
                    }
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    let name = if op == 74 {
                        let reference = addr_val(self, first - 1)?;
                        args.push(self.as_float(reference));
                        "textureGatherCmp.f32"
                    } else {
                        "textureGather.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            )
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if matches!(op, 73 | 74) {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = self.s.m.extract(c, r, ch);
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        // Per face +x, -x, +y, -y, +z, -z.
        let table = [
            [one, mone, u, u, u, nu],
            [nv, nv, one, mone, nv, nv],
            [nu, u, v, nv, one, mone],
        ];
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }

    /// `image_gather4_l(_o)`: DXIL gathers only from a view's first level,
    /// so the four texels are loaded from the level. Only what the title's
    /// culling shaders use: an immediate sampler without bias whose border
    /// is transparent black (a texel outside reads as a load outside: 0),
    /// and a level of detail proven whole (which level a fraction picks is
    /// not known; a whole one is that level, clamped to the sampler's and
    /// the view's).
    #[allow(clippy::too_many_arguments)]
    fn gather_at_level(
        &mut self,
        inst: &Inst,
        site: Site,
        h: V,
        sampler: &Source,
        a: &[V],
        lod: Value,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let Fields::Image { data, dmask, .. } = &inst.fields else {
            unreachable!()
        };
        let Source::Constant(w) = sampler else {
            return Err("a gather at a level through a sampler from memory");
        };
        let &[w0, w1, w2, w3, ..] = &w[..] else {
            return Err("a sampler constant");
        };
        let bits = |word: u32, at: u32, n: u32| word >> at & ((1 << n) - 1);
        // Clamp to a border (x, y) of transparent black; no anisotropy,
        // unnormalized or truncated coordinates, reduction or bias; whole
        // level limits.
        if w0 & 0x3F != 0x36
            || w0 & (7 << 9 | 1 << 15 | 1 << 19 | 1 << 27 | 3 << 29) != 0
            || bits(w3, 30, 2) != 0
            || bits(w2, 0, 20) != 0
            || w1 & 0x0F_F0FF != 0
        {
            return Err("a gather at a level through a sampler other than a black border's");
        }
        // No channel enabled: the first.
        let dmask = if *dmask == 0 { 1 } else { *dmask };
        if dmask.count_ones() != 1 {
            return Err("a gather of several channels");
        }
        let mip_filter = bits(w2, 26, 2) != 0;
        if mip_filter && self.whole(lod, site, &mut HashSet::new()) != Some(true) {
            return Err("a gather at a level that may not be whole");
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims_t = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let zero = self.s.m.i32(0);
        let level = if mip_filter {
            // The sampler's limits, then the view's last level.
            let f = self.as_float(a[a.len() - 1]);
            let lo = self.s.m.f32((bits(w1, 0, 12) as f32 / 256.0).to_bits());
            let hi = self.s.m.f32((bits(w1, 12, 12) as f32 / 256.0).to_bits());
            let f = self.binary(35, f, lo);
            let f = self.binary(36, f, hi);
            let l = self.s.m.cast(c, Cast::FpToSi, f, i32t);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims_t, 72, &[h, zero]);
            let levels = self.s.m.extract(c, d, 3);
            let one = self.s.m.i32(1);
            let last = self.s.m.bin(c, Bin::Sub, levels, one);
            let past = self.s.m.cmp(c, pred::SGT, l, last);
            self.s.m.select(c, past, last, l)
        } else {
            zero
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims_t, 72, &[h, level]);
        let offset = (a.len() == 4).then(|| a[0]);
        let mut first = [zero; 2];
        for (k, at) in [(0, 26), (1, 18)] {
            let size = self.s.m.extract(c, d, k as u32);
            first[k] = self.gather_first(a[a.len() - 3 + k], size);
            if let Some(o) = offset {
                // Bits at..at+5 of the offset word sign-extended.
                let left = self.s.m.i32(at);
                let t = self.s.m.bin(c, Bin::Shl, o, left);
                let k26 = self.s.m.i32(26);
                let o = self.s.m.bin(c, Bin::AShr, t, k26);
                first[k] = self.s.m.bin(c, Bin::Add, first[k], o);
            }
        }
        // Texels (x, y + 1), (x + 1, y + 1), (x + 1, y), (x, y).
        let one = self.s.m.i32(1);
        let next = first.map(|f| self.s.m.bin(c, Bin::Add, f, one));
        let iu = self.s.m.undef(i32t);
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        for (k, (x, y)) in [
            (first[0], next[1]),
            (next[0], next[1]),
            (next[0], first[1]),
            (first[0], first[1]),
        ]
        .into_iter()
        .enumerate()
        {
            let args = [h, level, x, y, iu, iu, iu, iu];
            let r = self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            );
            let v = self.s.m.extract(c, r, dmask.trailing_zeros());
            let v = self.as_int(v);
            self.write(inst, Reg::V(reg + k as u8), v, guard);
        }
        Ok(())
    }

    /// The first texel of a gather's footprint along an axis of `size`
    /// texels at coordinate `u` (bits of a float): floor(u × size − 0.5)
    /// with the product exact and rounded to 1/256 of a texel, half up, as
    /// the hardware does (`gather_level_tests`), i.e.
    /// (floor(u × size × 512) − 255) >> 9, in 64-bit integers. NaN is 0;
    /// beyond ±2 every texel of the footprint is outside, as at ±2.
    fn gather_first(&mut self, u: V, size: V) -> V {
        let c = self.cur;
        let f = self.as_float(u);
        let num = self.s.m.cmp(c, pred::OEQ, f, f);
        let fz = self.s.m.f32(0);
        let f = self.s.m.select(c, num, f, fz);
        let (lo, hi) = (
            self.s.m.f32((-2.0f32).to_bits()),
            self.s.m.f32(2.0f32.to_bits()),
        );
        let f = self.binary(35, f, lo);
        let f = self.binary(36, f, hi);
        let x = self.as_int(f);
        // u = m × 2^(e − 150), e at least 1 (denormals).
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k23, kff) = (k(self, 23), k(self, 0xFF));
        let e = self.s.m.bin(c, Bin::LShr, x, k23);
        let e = self.s.m.bin(c, Bin::And, e, kff);
        let (kmant, zero, one) = (k(self, 0x7F_FFFF), k(self, 0), k(self, 1));
        let m = self.s.m.bin(c, Bin::And, x, kmant);
        let normal = self.s.m.cmp(c, pred::NE, e, zero);
        let implicit = k(self, 0x80_0000);
        let bit = self.s.m.select(c, normal, implicit, zero);
        let m = self.s.m.bin(c, Bin::Or, m, bit);
        let e = self.s.m.select(c, normal, e, one);
        // u × size × 512 = m × size × 2^(e − 141); |u| ≤ 2 keeps e ≤ 128.
        let k141 = k(self, 141);
        let shift = self.s.m.bin(c, Bin::Sub, k141, e);
        let k63 = k(self, 63);
        let far = self.s.m.cmp(c, pred::SGT, shift, k63);
        let shift = self.s.m.select(c, far, k63, shift);
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let m = self.s.m.cast(c, Cast::ZExt, m, i64t);
        let n = self.s.m.cast(c, Cast::ZExt, size, i64t);
        let p = self.s.m.bin(c, Bin::Mul, m, n);
        let z64 = self.s.m.int_const(i64t, 0);
        let neg = self.s.m.bin(c, Bin::Sub, z64, p);
        let negative = self.s.m.cmp(c, pred::SLT, x, zero);
        let p = self.s.m.select(c, negative, neg, p);
        let shift = self.s.m.cast(c, Cast::ZExt, shift, i64t);
        let g = self.s.m.bin(c, Bin::AShr, p, shift);
        let k255 = self.s.m.int_const(i64t, 255);
        let g = self.s.m.bin(c, Bin::Sub, g, k255);
        let k9 = self.s.m.int_const(i64t, 9);
        let g = self.s.m.bin(c, Bin::AShr, g, k9);
        self.s.m.cast(c, Cast::Trunc, g, self.s.t.i32)
    }

    /// Whether `v` holds, as a float, a whole number or an infinity in
    /// every lane that runs the instruction at `at` (which reads it):
    /// `Some(true)` and never NaN, `Some(false)` or NaN, `None` perhaps a
    /// fraction. Follows moves, selects, minimum and maximum, sums with
    /// constants, rounding and conversions from integers (a level of detail
    /// made with `ceil`), through phis and guarded writes' kept values. A
    /// guarded write's kept value counts only if a lane that runs `at` may
    /// have skipped the write: every step here is the lane's own, and a lane
    /// that reads the value at `at` went through the write's block.
    fn whole(&self, v: Value, at: Site, seen: &mut HashSet<Value>) -> Option<bool> {
        if !seen.insert(v) {
            // Around a loop: what enters it decides.
            return Some(true);
        }
        let p = self.x.p;
        if let Some(&(b, i)) = self.phi_at.get(&v) {
            let mut never_nan = true;
            for &arg in &p.blocks[b].phis[i].args {
                never_nan &= self.whole(arg, at, seen)?;
            }
            return Some(never_nan);
        }
        let Some(&b) = self.def_block.get(&v) else {
            // A VGPR no one set reads zero.
            let vgpr = p
                .inputs
                .iter()
                .any(|&(r, x)| x == v && matches!(r, Reg::V(_)));
            let io = self.x.io.inputs.iter().any(|i| i.0 == v);
            return (vgpr && !io).then_some(true);
        };
        let (i, inst) = p.blocks[b]
            .insts
            .iter()
            .enumerate()
            .find(|(_, i)| i.writes.iter().any(|w| w.1 == v))?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.omod != 0 || f.sdwa.is_some() || f.dpp.is_some() {
            return None;
        }
        let mut never_nan = true;
        if self.x.st.guards[b][i].is_some() && !self.avoids(at, &[Elsewhere::Skip((b, i))]) {
            let Some(Operand::Vector { reg, .. }) = f.dst else {
                return None;
            };
            never_nan = self.whole(inst.read(Reg::V(reg))?, at, seen)?;
        }
        // Per source: (never NaN, a finite constant).
        let mut src = |k: usize| -> Option<(bool, bool)> {
            let x = match *f.src.get(k)? {
                Operand::Vector { reg, .. } => {
                    return Some((self.whole(inst.read(Reg::V(reg))?, at, seen)?, false));
                }
                Operand::Const(c) => crate::resource::inline(c)?,
                Operand::Literal(x) => x,
                _ => return None,
            };
            let x = f32::from_bits(x);
            (x.is_infinite() || x.fract() == 0.0).then_some((true, x.is_finite()))
        };
        let r = match inst.name() {
            "v_cvt_f32_i32" | "v_cvt_f32_u32" => true,
            "v_floor_f32" | "v_ceil_f32" | "v_trunc_f32" | "v_rndne_f32" => false,
            "v_mov_b32" => src(0)?.0,
            "v_cndmask_b32" => src(0)?.0 & src(1)?.0,
            "v_max_f32" | "v_min_f32" => src(0)?.0 | src(1)?.0,
            "v_max3_f32" | "v_min3_f32" => src(0)?.0 | src(1)?.0 | src(2)?.0,
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" => {
                // Infinities of both signs make NaN.
                let (x, y) = (src(0)?, src(1)?);
                x.0 && y.0 && (x.1 || y.1)
            }
            _ => return None,
        };
        Some(r && never_nan)
    }

    /// The vertex buffer slot (the 16-byte descriptor's index) a descriptor is, when it is loaded
    /// from a guessed graphics description's vertex buffer table ([`graphics::VERTEX_TABLE`]).
    fn vertex_table_slot(&self, src: &Source) -> Option<u32> {
        let Source::Loaded { load, first } = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let param = self.x.native.layout.get(a.param)?;
        (param.slot == graphics::VERTEX_TABLE && a.dynamic.is_none())
            .then(|| (a.offset + 4 * u32::from(*first)) / 16)
    }

    /// A buffer load through a vertex buffer table descriptor, as the vertex attribute it is:
    /// a stage-in element `VB<kind><n>` of index `slot << 16 | offset` (kind F float, U uint, N
    /// unorm8; n channels) the host maps to the stream of that slot ([`graphics`]).
    /// shortcut: one index per vertex (the load's index is not examined), so no per-instance
    /// streams; formats are those the host's vertex layout has.
    #[allow(clippy::too_many_arguments)]
    fn vertex_fetch(
        &mut self,
        inst: &Inst,
        slot: u32,
        typed: bool,
        count: u32,
        format: Option<u8>,
        offset: u16,
        by_index: bool,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let Fields::Buffer { data: Some(Operand::Vector { reg, .. }), .. } = &inst.fields else {
            return Err("a vertex fetch without a result");
        };
        if !by_index || offset >= 0x1000 {
            return Err("a vertex fetch that is not indexed or has a large offset");
        }
        let (kind, comp, channels) = match (typed, format.and_then(buffer_format)) {
            (false, _) if inst.name() == "buffer_load_ubyte" => ('B', 5, 1),
            (false, _) if inst.name() == "buffer_load_sbyte" => ('C', 4, 1),
            (false, _) if inst.name() == "buffer_load_ushort" => ('W', 5, 1),
            (false, _) if inst.name() == "buffer_load_sshort" => ('K', 4, 1),
            (false, _) if (1..=4).contains(&count) => (
                if count == 1 { 'U' } else { 'F' },
                if count == 1 { 5 } else { 9 },
                count,
            ),
            (true, Some((w, num)))
                if !inst.name().contains("_format_d16_") && w.len() >= count as usize =>
            {
                let (k, c) = match (w, num) {
                    ([32] | [32, 32] | [32, 32, 32] | [32, 32, 32, 32], Numeric::Float) => {
                        ('F', 9)
                    }
                    ([32] | [32, 32] | [32, 32, 32] | [32, 32, 32, 32], Numeric::Uint) => ('U', 5),
                    ([32] | [32, 32] | [32, 32, 32] | [32, 32, 32, 32], Numeric::Sint) => ('I', 4),
                    ([16] | [16, 16] | [16, 16, 16, 16], Numeric::Float) => ('H', 9),
                    ([16] | [16, 16] | [16, 16, 16, 16], Numeric::Unorm) => ('M', 9),
                    ([16] | [16, 16] | [16, 16, 16, 16], Numeric::Snorm) => ('S', 9),
                    ([16] | [16, 16] | [16, 16, 16, 16], Numeric::Uint) => ('W', 5),
                    ([16] | [16, 16] | [16, 16, 16, 16], Numeric::Sint) => ('K', 4),
                    ([8] | [8, 8] | [8, 8, 8, 8], Numeric::Unorm) => ('N', 9),
                    ([8] | [8, 8] | [8, 8, 8, 8], Numeric::Snorm) => ('Q', 9),
                    ([8] | [8, 8] | [8, 8, 8, 8], Numeric::Uint) => ('B', 5),
                    ([8] | [8, 8] | [8, 8, 8, 8], Numeric::Sint) => ('C', 4),
                    ([10, 10, 10, 2], Numeric::Unorm) => ('P', 9),
                    ([10, 10, 10, 2], Numeric::Snorm) => ('T', 9),
                    ([11, 11, 10], Numeric::Float) => ('G', 9),
                    _ => return Err("a vertex fetch format the host's vertex layout lacks"),
                };
                (k, c, w.len() as u32)
            }
            _ => return Err("a vertex fetch format the host's vertex layout lacks"),
        };
        let name = format!("VB{kind}{channels}");
        let corners = if self.x.rect_list { 3 } else { 1 };
        let mut ids = Vec::new();
        for corner in 0..corners {
        let index = corner << 28 | slot << 16 | u32::from(offset);
        let id = match self.s.inputs.iter().position(|e| e.name == name && e.index == index) {
            Some(id) => id,
            None => {
                let row = self.s.inputs.iter().filter_map(|e| e.row).map(|r| r + 1).max().unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: name.clone(),
                    index,
                    kind: 0,
                    comp,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: channels as u8,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let cols = &mut self.s.inputs[id].cols;
        *cols = (*cols).max(count as u8);
        ids.push(id);
        }
        let mut loaded = Vec::new();
        for k in 0..count {
            let mut x = [None; 3];
            for (corner, &id) in ids.iter().enumerate() {
                let Val::I32(v) = self.load_any_input(id, k as u8) else {
                    return Err("a vertex input");
                };
                x[corner] = Some(v);
            }
            loaded.push(x);
        }
        let float = !matches!(kind, 'U' | 'I' | 'W' | 'K' | 'B' | 'C');
        if let ([[Some(a0), Some(b0), Some(d0)], [Some(a1), Some(b1), Some(d1)], ..], true, None) =
            (&loaded[..], float, self.rect_corner)
        {
            // The right angle is opposite the longest side (the corners may come in any order).
            let c = self.cur;
            let p: Vec<V> = [*a0, *a1, *b0, *b1, *d0, *d1].map(|v| self.as_float(v)).to_vec();
            let mut d2 = |i: usize, j: usize| {
                let dx = self.s.m.bin(c, Bin::Sub, p[2 * i], p[2 * j]);
                let dy = self.s.m.bin(c, Bin::Sub, p[2 * i + 1], p[2 * j + 1]);
                let xx = self.s.m.bin(c, Bin::Mul, dx, dx);
                let yy = self.s.m.bin(c, Bin::Mul, dy, dy);
                self.s.m.bin(c, Bin::Add, xx, yy)
            };
            let (ab, bd, ad) = (d2(0, 1), d2(1, 2), d2(0, 2));
            let ge1 = self.s.m.cmp(c, pred::OGE, bd, ab);
            let ge2 = self.s.m.cmp(c, pred::OGE, bd, ad);
            let at_a = self.s.m.bin(c, Bin::And, ge1, ge2);
            let ge3 = self.s.m.cmp(c, pred::OGT, ab, ad);
            let ge4 = self.s.m.cmp(c, pred::OGT, ab, bd);
            let at_d = self.s.m.bin(c, Bin::And, ge3, ge4);
            self.rect_corner = Some((c, at_a, at_d));
        }
        for (k, x) in loaded.into_iter().enumerate() {
            let Some(mut v) = x[0] else { return Err("a vertex input") };
            if let [Some(a), Some(b), Some(d)] = x {
                // Rectangle corner by vertex id: 1 and 2 are the stored ones, 3 is b + d - a.
                let Val::I32(vid) = self.input(Input::VertexId) else { return Err("a vertex id") };
                let c = self.cur;
                let is = |l: &mut Self, n: u32| {
                    let n = l.s.m.i32(n);
                    l.s.m.cmp(c, pred::EQ, vid, n)
                };
                // r: the right-angle corner, p and q the others (v1, v0 and v2 unless the
                // position says otherwise); the missing corner is p + q - r.
                let (p, r, q) = match self.rect_corner {
                    Some((block, at_a, at_d)) if block == c => {
                        let r = self.s.m.select(c, at_d, d, b);
                        let r = self.s.m.select(c, at_a, a, r);
                        let p = self.s.m.select(c, at_a, b, a);
                        let q = self.s.m.select(c, at_d, b, d);
                        (p, r, q)
                    }
                    _ => (a, b, d),
                };
                let ex = if float {
                    let (fp, fr, fq) = (self.as_float(p), self.as_float(r), self.as_float(q));
                    let sum = self.s.m.bin(c, Bin::Add, fp, fq);
                    let fourth = self.s.m.bin(c, Bin::Sub, sum, fr);
                    let fourth = self.as_int(fourth);
                    // A value two corners share is the fourth's exactly (no rounding, NaN or
                    // flushed denormal from the arithmetic).
                    let pr = self.s.m.cmp(c, pred::EQ, p, r);
                    let qr = self.s.m.cmp(c, pred::EQ, q, r);
                    let v = self.s.m.select(c, qr, p, fourth);
                    self.s.m.select(c, pr, q, v)
                } else {
                    q
                };
                // Strip order p, r, the missing corner, q: two triangles sharing the r diagonal.
                let (c1, c2, c3) = (is(self, 1), is(self, 2), is(self, 3));
                v = self.s.m.select(c, c1, r, p);
                v = self.s.m.select(c, c2, ex, v);
                v = self.s.m.select(c, c3, q, v);
            }
            if count == 2 && kind == 'F' {
                match (k, self.debug_pos) {
                    (0, _) => self.debug_pos = Some((v, v)),
                    (_, Some((x, _))) => self.debug_pos = Some((x, v)),
                    _ => {}
                }
            }
            self.write(inst, Reg::V(reg + k as u8), v, guard);
        }
        Ok(())
    }

    fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name == "buffer_gl0_inv" || name == "buffer_gl1_inv" {
            // Console cache coherence: gl0 writes back and invalidates the
            // per-CU shader L0, gl1 invalidates GL1, so one wavefront's
            // writes become visible to others. D3D12 already guarantees UAV
            // coherence across wavefronts and dispatches, DXIL has no
            // equivalent instruction, and this project's host-visible
            // writeback comes from command-stream GpuWrite ranges rather
            // than shader cache hints. Ordering is kept by the surrounding
            // s_waitcnt/s_barrier, which the translator does lower.
            return Ok(());
        }
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        // d16: two 16-bit channels per VGPR, low half first (buffer and
        // tbuffer formats alike).
        let d16 = name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            Some("ubyte" | "sbyte" | "ushort" | "sshort") if !store => (false, 1),
            Some("byte" | "short") if store => (false, 1),
            _ => return Err(name),
        };
        // Loads or stores of a byte or a half: its width and whether signed.
        let sub = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
        {
            Some("ubyte" | "byte") => Some((8, false)),
            Some("sbyte") => Some((8, true)),
            Some("ushort" | "short") => Some((16, false)),
            Some("sshort") => Some((16, true)),
            _ => None,
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        // A descriptor whose stride the program cleared adds no index (`Resources::unstrided`).
        let unstrided = self.x.res.unstrided.contains(&(b, i));
        let index = if *idxen && !unstrided { Some(vaddr(self, 0)?) } else { None };
        let voffset = if *offen { Some(vaddr(self, u8::from(*idxen))?) } else { None };
        let so = if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            Some(self.operand(inst, soffset)?)
        } else {
            None
        };
        // A typed store's scalar offset register proved to hold 0 (moves of
        // the constant) is no offset; any other makes `off` computed, which
        // the typed store below refuses.
        let so = so.filter(|_| {
            !(typed && store && self.konst_operand(&HashMap::new(), inst, soffset) == Some(0))
        });
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        if matches!(src, Source::BuiltMemory { .. }) {
            if typed || store || atomic.is_some() {
                return Err("a computed buffer descriptor used for an unsupported buffer access");
            }
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let zero = self.s.m.i32(0);
            let offset = (voffset.unwrap_or(zero), u32::from(*offset));
            let words = self.computed_buffer(inst, src, index, offset, so.unwrap_or(zero), count)?;
            for (k, x) in words.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let mut off = self.s.m.i32(u32::from(*offset));
        if let Some(vo) = voffset {
            off = self.s.m.bin(c, Bin::Add, off, vo);
        }
        if let Some(so) = so {
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        // A raw buffer descriptor built around a table in the code (stride 0, the record count
        // its size in bytes): each dword is the code's, as for scalar loads through it, and 0
        // past the record count (the hardware's bounds check of a raw access).
        if let Source::Built { address, bits, words } = src
            && address.param == CODE
        {
            if typed || store || atomic.is_some() || sub.is_some() || bits >> 16 & 0x3FFF != 0 {
                return Err("a descriptor built around a table in the code used for an unsupported buffer access");
            }
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let (base, extra) = self.built_offset(src)?;
            let b = self.s.m.i32(base);
            let mut at = self.s.m.bin(c, Bin::Add, off, b);
            if let Some(x) = extra {
                at = self.s.m.bin(c, Bin::Add, at, x);
            }
            let records = self.s.m.i32(words[0]);
            let zero = self.s.m.i32(0);
            for k in 0..count {
                let four_k = self.s.m.i32(4 * k);
                let byte = self.s.m.bin(c, Bin::Add, at, four_k);
                let x = self.code_dword(byte)?;
                let rel = self.s.m.bin(c, Bin::Add, off, four_k);
                let inside = self.s.m.cmp(c, pred::IULT, rel, records);
                let x = self.s.m.select(c, inside, x, zero);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if let Some(slot) = self.vertex_table_slot(src) {
            return self.vertex_fetch(inst, slot, typed, count, *format, *offset, index.is_some() && !*offen, guard);
        }
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        // A descriptor built around a root view has the stride the code puts
        // in dword 1 (bits 16-29) and no record limit (dword 2 is -1): an
        // index is a byte offset into the raw root view.
        let index = match (src, index) {
            (
                Source::Built {
                    address,
                    bits,
                    words,
                },
                Some(ix),
            ) if address.param < self.x.native.layout.len() && !typed => {
                if bits >> 30 != 0 || words[0] != u32::MAX {
                    return Err("a root view indexed with swizzling or a record limit");
                }
                let stride = self.s.m.i32(bits >> 16 & 0x3FFF);
                let byte = self.s.m.bin(c, Bin::Mul, ix, stride);
                off = self.s.m.bin(c, Bin::Add, off, byte);
                None
            }
            (_, ix) => ix,
        };
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let table = used.is_some_and(|u| u.root == RootType::Table);
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            if table {
                // A table's constant buffer is a raw view (scalar loads above).
                let Some(Operand::Vector { reg, .. }) = *data else {
                    return Err("a buffer result");
                };
                let (h, byte) = self.raw_view(site, src, byte, 4 * count, (11, 0), false, guard)?;
                for k in 0..count {
                    let four = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, byte, four);
                    let (mask, al) = (self.s.m.i8(1), self.s.m.i32(4));
                    let u = self.s.m.undef(self.s.t.i32);
                    let r = self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, a, u, mask, al],
                    );
                    let x = self.s.m.extract(c, r, 0);
                    let x = self.as_int(x);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = self.cbuffer_dword(h, a);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        // A heap-indexed buffer read by element has no used-resource entry,
        // so no stride: its bytes are addressed as the hardware does, the
        // element times the view's stride (the host's table) plus the offset.
        // So is one from a table whose entry (FSPX metadata, which gives no
        // stride) names none: the view's heap index is its table offset plus
        // the load's, in the same heap.
        let index = match index {
            Some(i) if used.is_none_or(|u| u.stride == 0) && !typed => {
                let stride = self.view_stride(src)?;
                let rows = self.s.m.bin(c, Bin::Mul, i, stride);
                off = self.s.m.bin(c, Bin::Add, rows, off);
                None
            }
            index => index,
        };
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            let (h, coords, byte) = match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride < 4 || !stride.is_multiple_of(4) {
                        return Err("a structured buffer of unknown stride");
                    }
                    let h = self.handle(site, src, false, (12 | u, stride))?;
                    let s = self.s.m.i32(stride);
                    let row = self.s.m.bin(self.cur, Bin::Mul, index, s);
                    let byte = self.s.m.bin(self.cur, Bin::Add, row, off);
                    (h, [index, off, iu], byte)
                }
                None => {
                    let (h, off) = self.raw_view(site, src, off, 4, (11 | u, 0), true, guard)?;
                    (h, [off, iu, iu], off)
                }
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let r = if op == u32::MAX {
                // cmpswap: data is (new, compare).
                let mut args = vec![h];
                args.extend(coords);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                let mut args = vec![h, self.s.m.i32(op)];
                args.extend(coords);
                args.push(vals[0]);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            self.mark_view_write(h, byte, 4, None);
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if let (true, Some((width, _))) = (store, sub) {
            // The field of the dword holding it, by two atomics on that dword
            // (clear, then set): other lanes' fields in it are left alone.
            // shortcut: the field is taken not to cross a dword.
            if index.is_some() {
                return Err("a byte or half buffer store by index");
            }
            let d = data.as_ref().ok_or("buffer store data")?;
            let x = self.vgprs(inst, d, 1)?[0];
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let field = self.s.m.i32(if width == 8 { 0xFF } else { 0xFFFF });
            let x = self.s.m.bin(c, Bin::And, x, field);
            let x = self.s.m.bin(c, Bin::Shl, x, sh);
            let mask = self.s.m.bin(c, Bin::Shl, field, sh);
            let all = self.s.m.i32(u32::MAX);
            let keep = self.s.m.bin(c, Bin::Xor, mask, all);
            let (h, at) = self.raw_view(site, src, at, 4, (11 | 0x1000, 0), true, guard)?;
            let begin = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            for (op, v) in [(1, keep), (2, x)] {
                let args = [h, self.s.m.i32(op), at, iu, iu, v];
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args);
            }
            self.mark_view_write(h, at, 4, None);
            self.end_guarded(begin, &[]);
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            let typed_h = if typed {
                Some(self.handle(site, src, false, (10 | u, 0x409))?)
            } else {
                None
            };
            if d16 {
                // d16 data is f16 for float and normalized formats, widened
                // for the 32-bit float store; for integer formats (the view's
                // kind, from the view-stride table) the 16-bit integer, zero-
                // or sign-extended (stored through the float-typed handle as
                // loads read: see there).
                let integer = typed_h.and_then(|h| self.integer_view(h));
                let c = self.cur;
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    let mut w = self.as_int(f);
                    if let Some((int, signed)) = integer {
                        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
                        let zero_ext = self.s.m.bin(c, Bin::And, x, low);
                        let up = self.s.m.bin(c, Bin::Shl, x, k16);
                        let sign_ext = self.s.m.bin(c, Bin::AShr, up, k16);
                        let n = self.s.m.select(c, signed, sign_ext, zero_ext);
                        w = self.s.m.select(c, int, n, w);
                    }
                    wide.push(w);
                }
                vals = wide;
            }
            let mut args = Vec::new();
            // Where the write starts in the view, and how many bytes it may
            // cover (for its marks).
            let (byte, bytes);
            let (op, name) = if typed {
                // A typed view has no bytes inside an element to address: a
                // computed offset is refused, as it is not known to be 0.
                if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                    return Err("a typed buffer store at an offset");
                }
                let h = typed_h.ok_or("a typed buffer store")?;
                let element = index.unwrap_or(self.s.m.i32(0));
                args.push(h);
                args.extend([element, iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                // A store of fewer channels (`_x`, `_xy`, `_xyz`) still
                // writes the whole element: the hardware stores 0 in the
                // channels the instruction does not name (GFX11 and older;
                // LLVM 88421ea "[AMDGPU] Trim zero components from buffer and
                // image stores", Mesa 4a6ee2c4833 "aco: shrink buffer stores
                // with undef/zero components"), so they are not kept. Channels
                // past the view's format are dropped by the host's store.
                let zero = self.s.m.f32(0);
                args.resize(3 + 4, zero);
                // The element's size is the view's stride (16 bytes at most).
                (byte, bytes) = match self.s.view_index.get(&h).copied() {
                    Some(at) if self.x.view_marks => {
                        let stride = self.stride_at(at);
                        (self.s.m.bin(self.cur, Bin::Mul, element, stride), 16)
                    }
                    _ => (element, 16),
                };
                (69, "bufferStore.f32")
            } else {
                match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride < 4 || !stride.is_multiple_of(4) {
                            return Err("a structured buffer of unknown stride");
                        }
                        args.push(self.handle(site, src, false, (12 | u, stride))?);
                        args.extend([index, off]);
                        let s = self.s.m.i32(stride);
                        let row = self.s.m.bin(self.cur, Bin::Mul, index, s);
                        (byte, bytes) = (self.s.m.bin(self.cur, Bin::Add, row, off), 4 * count);
                    }
                    None => {
                        let n = 4 * count;
                        let (h, off) =
                            self.raw_view(site, src, off, n, (11 | u, 0), true, guard)?;
                        args.extend([h, off, iu]);
                        (byte, bytes) = (off, n);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.mark_view_write(args[0], byte, bytes, None);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let (h, off) = self.raw_view(site, src, off, 4 * count, (11 | u, 0), false, guard)?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let mut xs = self.format_load(h, off, f, count)?;
            if d16 {
                xs = self.d16_channels(f, xs)?;
            }
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if let Some((width, signed)) = sub {
            // The dword holding the field, as `global_load_ubyte` and kin.
            // shortcut: the field is taken not to cross a dword.
            if index.is_some() {
                return Err("a byte or half buffer load by index");
            }
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let (h, at) = self.raw_view(site, src, at, 4, (11 | u, 0), false, guard)?;
            let c = self.cur;
            let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                139,
                &[h, at, iu, mask, align],
            );
            let w = self.s.m.extract(c, r, 0);
            let w = self.as_int(w);
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let x = self.s.m.bin(c, Bin::LShr, w, sh);
            let left = self.s.m.i32(32 - width);
            let x = self.s.m.bin(c, Bin::Shl, x, left);
            let x = self
                .s
                .m
                .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left);
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            self.write(inst, Reg::V(reg), x, guard);
            return Ok(());
        }
        let mut typed_h = None;
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            typed_h = Some(h);
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride < 4 || !stride.is_multiple_of(4) {
                        return Err("a structured buffer of unknown stride");
                    }
                    let h = self.handle(site, src, false, (12 | u, stride))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, index, off, mask, align],
                    )
                }
                None => {
                    let bytes = 4 * count;
                    let (h, off) =
                        self.raw_view(site, src, off, bytes, (11 | u, 0), false, guard)?;
                    self.s.op(
                        self.cur,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let c = self.cur;
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two 16-bit channels per VGPR (an odd last high half 0): f16 for
            // float and normalized formats; for integer formats (the view's
            // kind, from the view-stride table) the integer's low 16 bits.
            // shortcut: an integer view is read and written through the
            // float-typed handle, so its channels' bits travel as the driver
            // passes a mismatched component type (unchanged on the RX 6700
            // XT and WARP, `stride_tests`); an integer-typed second handle
            // for the slot would not rely on it.
            let integer = typed_h.and_then(|h| self.integer_view(h));
            let c = self.cur;
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let mut h = self
                        .s
                        .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                    if let Some((int, _)) = integer {
                        let bits = self.as_int(f);
                        let low = self.s.m.i32(0xFFFF);
                        let low = self.s.m.bin(c, Bin::And, bits, low);
                        h = self.s.m.select(c, int, low, h);
                    }
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        // Compressed exports are two 16-bit values per dword (FP16_ABGR, the colour format a
        // graphics description's pixel shader is guessed to export), the halves path below.
        if e.compressed && !matches!(e.target, Target::Color(_)) {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        let c = self.cur;
        // A pixel export whose lanes outside its guard the `done` export
        // after it discards: what those lanes write is lost, so it stores in
        // every lane.
        if let Some(g) = guard
            && !mesh
            && !(self.s.pixel && !e.done && self.discarded_at_done(b, i))
        {
            if !(e.done && self.s.pixel) {
                return Err("a conditional export");
            }
            let t = self.s.m.bool(true);
            let kill = self.s.m.bin(c, Bin::Xor, g, t);
            self.s
                .op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
        }
        if mesh && e.target == Target::Primitive {
            return self.emit_indices(e, guard);
        }
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        // (output element, channel) per exported channel; per-primitive
        // outputs are elements of `prims`.
        let mut stores = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.s.prims
        } else {
            &self.s.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, pack2)) = halves {
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 {
                        packed_half(outputs, row, h)
                    } else {
                        (!high).then_some((row, ch))
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Some(high)));
                    }
                }
            }
        }
        if let (true, Target::Color(n)) = (e.compressed, e.target) {
            let id = outputs.iter().position(|o| o.kind == 16 && o.row == Some(u32::from(n)));
            for ch in 0..4u8 {
                if let (Some(id), Some(value)) = (id, e.values[usize::from(ch)]) {
                    stores.push((id, ch, value, Some(ch & 1 == 1)));
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() || e.compressed {
                break;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| o.kind == 16 && o.row == Some(u32::from(n)))
                    .map(|id| (id, ch)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // shortcut: without the misc vector (point size, render
                    // target or viewport index) pos1 and pos2 carry the clip
                    // then cull distances, in semantic order.
                    if outputs.iter().any(|o| matches!(o.kind, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - 1) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                stores.push((id, col, value, None));
            }
        }
        // A guessed pipeline's parameter channels, each also as its two halves widened to f32
        // (`graphics::HALF`).
        if let (true, Target::Param { usage, row }) = (self.guessed, e.target)
            && let Some(r) = param_row(usage, row)
        {
            for high in [false, true] {
                let index = 2 * u32::from(r) + u32::from(high);
                let Some(id) = outputs.iter().position(|o| o.name == super::graphics::HALF && o.index == index) else {
                    continue;
                };
                let (col, cols) = (outputs[id].col, outputs[id].cols);
                for ch in col..col + cols {
                    if let Some(value) = e.values[usize::from(ch)] {
                        stores.push((id, ch - col, value, Some(high)));
                    }
                }
            }
        }
        // A guessed pipeline's parameter elements are four channels (`graphics::HALF`): the
        // channels this export leaves out get a copy of one it has (unread), since the
        // converter crashes linking a mesh output some channels of which are not written, or
        // are written a constant.
        // shortcut: a parameter exported twice with different channels keeps only the last
        // export's; upgrade if a guessed program is seen doing that.
        let mut pad = Vec::new();
        if let (true, false, Target::Param { usage, row }) = (self.guessed, primitive, e.target)
            && let Some(r) = param_row(usage, row)
        {
            let r = u32::from(r);
            for (id, o) in outputs.iter().enumerate() {
                let ours = (o.name == "TEXCOORD" && o.kind == 0 && o.row == Some(r) && o.index == r)
                    || (o.name == super::graphics::HALF && o.index / 2 == r);
                let half = (o.name == super::graphics::HALF).then_some(o.index % 2 == 1);
                for col in (0..o.cols).filter(|_| ours) {
                    if !stores.iter().any(|s| s.0 == id && s.1 == col)
                        && let Some(value) = e.values.iter().flatten().next()
                    {
                        pad.push((id, col, *value, half));
                    }
                }
            }
        }
        stores.extend(pad);
        // A parameter a split pixel input copies (`Options::copy_params`) is written with the
        // original's values.
        if self.x.copy_params != [0; 32] && !primitive {
            let mut copies = Vec::new();
            for &(id, col, value, half) in &stores {
                let src = &self.s.outputs[id];
                // `TEXCOORD`s only: a `HALF` of the copied index is another parameter's half,
                // and copying it would overwrite the `HALF` of the copy's index.
                if src.kind != 0 || src.name != "TEXCOORD" || src.index >= 32 {
                    continue;
                }
                for (m, &from) in self.x.copy_params.iter().enumerate() {
                    if from != 0
                        && u32::from(from) - 1 == src.index
                        && let Some(to) = self
                            .s
                            .outputs
                            .iter()
                            .position(|o| o.kind == 0 && o.name == src.name && o.index == m as u32)
                    {
                        copies.push((to, col, value, half));
                    }
                }
            }
            stores.extend(copies);
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| s.3.is_none() && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports.
        let at = if mesh {
            self.begin_guarded(guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        // Every component's value is made before the first store, and the stores then follow
        // each other with nothing between them. Apple's Metal Shader Converter gathers the
        // component stores of one output into a vector at the first of them: a value made
        // between two stores (a float cast, an f16 conversion) was then used before its
        // definition in the converted library, and Metal's compiler failed on the mesh
        // programs whose outputs a pixel stage reads.
        let mut ready: Vec<(usize, u8, &'static str, V)> = Vec::with_capacity(stores.len());
        for (id, col, value, half) in stores {
            let mut x = self.i32_of(value)?;
            if let Some(high) = half {
                // An f16 half, declared a 32-bit float on the host.
                let c = self.cur;
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                x = self.as_int(f);
            }
            if half.is_some() && elements(self, primitive)[id].comp != 9 {
                return Err("a 16-bit export to an integer output");
            }
            let el = &mut elements(self, primitive)[id];
            let el_kind = el.kind;
            el.usage |= 1 << col;
            let (suffix, x) = match el.comp {
                9 if half.is_some() => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let x = if !self.s.pixel && !mesh && el_kind == 3 && col < 2 && std::env::var_os("VICTORY_DEBUG_VS_TRI").is_some() {
                // Debug: a triangle covering the target, from the vertex id.
                let Ok(Val::I32(vid)) = self.get_input_vertex_id() else { return Err("a vertex id") };
                let c = self.cur;
                let sh = self.s.m.i32(u32::from(col));
                let bit = self.s.m.bin(c, Bin::LShr, vid, sh);
                let one = self.s.m.i32(1);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                let z = self.s.m.i32(0);
                let on = self.s.m.cmp(c, pred::NE, bit, z);
                let (hi, lo) = (self.s.m.i32(3.0f32.to_bits()), self.s.m.i32((-1.0f32).to_bits()));
                let b = self.s.m.select(c, on, hi, lo);
                self.as_float(b)
            } else {
                x
            };
            // Debug: VICTORY_DEBUG_VS_PARAM=<n,...> writes 1.0 to every channel of those vertex
            // parameters (the guessed signature's row n).
            let forced = !self.s.pixel
                && !mesh
                && std::env::var("VICTORY_DEBUG_VS_PARAM").is_ok_and(|v| {
                    v.split(',').filter_map(|p| p.parse::<u32>().ok()).any(|p| elements(self, primitive)[id].row == Some(p) && el_kind == 0)
                });
            // Debug: VICTORY_DEBUG_VS_POS=1 positions from the raw vertex xy (scale 1/32768).
            let x = match (self.debug_pos, el_kind == 3 && col < 2 && !self.s.pixel && !mesh) {
                (Some((vx, vy)), true) if std::env::var_os("VICTORY_DEBUG_VS_POS").is_some() => {
                    let raw = self.as_float(if col == 0 { vx } else { vy });
                    let k = self.s.m.f32(if col == 0 { 3.0517578e-5f32.to_bits() } else { (-3.0517578e-5f32).to_bits() });
                    self.s.m.bin(self.cur, Bin::Mul, raw, k)
                }
                _ => x,
            };
            let x = if forced && suffix == ".f32" { self.s.m.f32(1.0f32.to_bits()) } else { x };
            // Debug (`VICTORY_DEBUG_PS_RED`; `VICTORY_DEBUG_PS=white|alpha`): constant colours.
            let x = match (self.s.pixel && suffix == ".f32", std::env::var("VICTORY_DEBUG_PS").ok().as_deref()) {
                (true, Some(m)) if m.starts_with("in") && m[2..].parse::<u8>().is_ok() => {
                    // Debug: the raw pixel input of that row as the colour (channel = the export's).
                    let row = m[2..].parse::<u8>().unwrap_or(0);
                    match self.find_input(row, col) {
                        Some((iid, c)) => match self.load_any_input(iid, c) {
                            Val::I32(v) => self.as_float(v),
                            _ => self.s.m.f32(0),
                        },
                        None => self.s.m.f32(0),
                    }
                }
                (true, Some("white")) => self.s.m.f32(1.0f32.to_bits()),
                (true, Some("alpha")) if col == 3 => self.s.m.f32(1.0f32.to_bits()),
                (true, _) if std::env::var_os("VICTORY_DEBUG_PS_RED").is_some() => {
                    self.s.m.f32(if col == 0 || col == 3 { 1.0f32.to_bits() } else { 0 })
                }
                _ => x,
            };
            ready.push((id, col, suffix, x));
        }
        let thread = mesh.then(|| self.mesh_thread());
        for (id, col, suffix, x) in ready {
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            args.extend(thread);
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Each corner takes ten
    /// bits: a 9-bit vertex index and, above it, its edge flag (bits 9, 19
    /// and 29), which only the console's edge-flag rasterisation reads and
    /// which is no part of the index. Bit 31 culls it (all indices 0: no
    /// area).
    fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let culled = self.s.m.cmp(c, pred::SLT, x, z);
        let mut args = vec![self.mesh_thread()];
        for k in 0..3 {
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x1FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            args.push(self.s.m.select(c, culled, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .s
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, pred::IULT);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                // A back edge is taken only below the iteration limit.
                let back =
                    |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
                // (No `xor` of a compare: a loop exit testing one hung the
                // AMD driver.)
                let cond = if back(self, taken) {
                    let ok = self.count_iteration(taken, pred::IULT);
                    self.s.m.bin(self.cur, Bin::And, cond, ok)
                } else if back(self, not_taken) {
                    let stop = self.count_iteration(not_taken, pred::IUGE);
                    self.s.m.bin(self.cur, Bin::Or, cond, stop)
                } else {
                    cond
                };
                // Critical edges get a block of their own, so loops have
                // dedicated exits and preheaders as DXC emits them.
                let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
                for (to, via) in [(taken, t), (not_taken, f)] {
                    if !back(self, to) {
                        // From an edge block, or from here.
                        let from = if Some(via) == self.block[to] {
                            self.cur
                        } else {
                            via
                        };
                        self.enter_loop(b, to, from);
                    }
                }
                Term::CondBr(cond, t, f)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Ret
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    fn enter_loop(&mut self, _from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
    }

    /// A back edge to header `to` (a back edge keeps its block, no
    /// critical-edge block): the count goes up; the result compares it with
    /// [`LOOP_LIMIT`] by `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, to: usize, p: u8) -> V {
        let c = self.cur;
        let (phi, _) = self.counters[&to];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, phi, one);
        let limit = self.s.m.i32(super::loop_limit());
        self.counters
            .get_mut(&to)
            .expect("header")
            .1
            .push((next, c));
        self.s.m.cmp(c, p, next, limit)
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    /// Whether no lane entering IR block `b` from its `n`th predecessor
    /// reads what `reg` holds there before something overwrites it. The
    /// lane is followed along its per-lane branches, with the phis it
    /// passes taking that path's values; a branch whose compares have
    /// constant operands on the path is followed one way. Scalar arithmetic
    /// on the value passes it on; any other read of it, or a branch or guard
    /// that depends on it, makes it live. Paths meeting at a block with the
    /// same registers holding it go on with what they agree on. This is the
    /// console compiler's flag after a join (`if (c) { x = load; f = .. }
    /// if (f) use(x)`, with `f` 0 on the other path), where the register
    /// that holds `x` holds a stale value, often a lane mask, on that path.
    fn dead_on_edge(&self, b: usize, n: usize, reg: Reg) -> bool {
        let p = self.x.p;
        let st = self.x.st;
        // Values computed from the stale one on some path followed.
        let mut tainted = HashSet::new();
        let mut work = vec![(b, n, HashMap::new(), BTreeSet::from([reg]))];
        // Per block reached and registers holding tainted values on entry:
        // the phi values known there, over the paths so far.
        let mut at: HashMap<(usize, BTreeSet<Reg>), HashMap<Value, u32>> = HashMap::new();
        let mut visits = 0;
        while let Some((b, n, mut env, mut regs)) = work.pop() {
            let block = &p.blocks[b];
            let known: Vec<_> = block
                .phis
                .iter()
                .map(|phi| (phi.value, self.konst(&env, phi.args[n])))
                .collect();
            for (v, k) in known {
                match k {
                    Some(k) => env.insert(v, k),
                    None => env.remove(&v),
                };
            }
            for phi in &block.phis {
                if regs.contains(&phi.reg) {
                    tainted.insert(phi.value);
                }
            }
            let key = (b, regs.clone());
            if let Some(old) = at.get(&key) {
                let agreed: HashMap<Value, u32> = old
                    .iter()
                    .filter(|&(v, k)| env.get(v) == Some(k))
                    .map(|(&v, &k)| (v, k))
                    .collect();
                if agreed.len() == old.len() {
                    continue;
                }
                env = agreed;
            }
            at.insert(key, env.clone());
            visits += 1;
            if visits > 4 * p.blocks.len() + 64 {
                return false;
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if regs.is_empty() {
                    break;
                }
                let guard = st.guards[b][i];
                if guard.is_some_and(|g| self.touches(g, &env, &tainted)) {
                    return false;
                }
                // The closing branch is the lane's exit, tested below.
                if inst.reads.iter().any(|r| regs.contains(&r.0))
                    && !inst.name().starts_with("s_cbranch")
                {
                    // Scalar arithmetic only passes the value on.
                    let pure = matches!(inst.fields, Fields::Sop { .. })
                        && inst.write(Reg::S(EXEC)).is_none()
                        && !inst.name().starts_with("s_setreg");
                    if !pure {
                        return false;
                    }
                    for &(r, w) in &inst.writes {
                        regs.insert(r);
                        tainted.insert(w);
                    }
                } else if guard.is_none() {
                    for w in &inst.writes {
                        regs.remove(&w.0);
                    }
                }
            }
            if regs.is_empty() {
                continue;
            }
            let into = |t: usize| p.blocks[t].preds.iter().position(|&q| q == b);
            let mut go = |t: usize, env, regs| match into(t) {
                Some(n) => {
                    work.push((t, n, env, regs));
                    true
                }
                None => false,
            };
            let ok = match st.exits[b] {
                LaneExit::End => true,
                LaneExit::Indirect => false,
                LaneExit::Jump(t) => go(t, env, regs),
                LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } => {
                    let (no, yes) = self.can_be(when, &env, &mut HashMap::new());
                    !(no && yes && self.touches(when, &env, &tainted))
                        && (!yes || go(taken, env.clone(), regs.clone()))
                        && (!no || go(not_taken, env, regs))
                }
            };
            if !ok {
                return false;
            }
        }
        true
    }

    /// Whether deciding `b`, given the values in `env`, may test an atom
    /// of a `tainted` value.
    fn touches(&self, b: Bool, env: &HashMap<Value, u32>, tainted: &HashSet<Value>) -> bool {
        let mut stack = vec![b];
        let mut seen = HashSet::new();
        while let Some(b) = stack.pop() {
            let Some((atom, lo, hi)) = self.x.st.bools.node(b) else {
                continue;
            };
            if !seen.insert(b) {
                continue;
            }
            let v = match atom {
                Atom::Lane(v)
                | Atom::Uniform(v)
                | Atom::Phi(v)
                | Atom::Quad(v)
                | Atom::Opaque(v) => Some(v),
                Atom::Constant(_) => None,
            };
            if v.is_some_and(|v| tainted.contains(&v)) {
                return true;
            }
            match v.and_then(|v| self.compared(env, v)) {
                Some(true) => stack.push(hi),
                Some(false) => stack.push(lo),
                None => stack.extend([lo, hi]),
            }
        }
        false
    }

    /// Whether `b` can be false, and true, given the values in `env`.
    fn can_be(
        &self,
        b: Bool,
        env: &HashMap<Value, u32>,
        memo: &mut HashMap<Bool, (bool, bool)>,
    ) -> (bool, bool) {
        if b == Bool::TRUE || b == Bool::FALSE {
            return (b == Bool::FALSE, b == Bool::TRUE);
        }
        if let Some(&r) = memo.get(&b) {
            return r;
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let v = match atom {
            Atom::Lane(v) | Atom::Uniform(v) => Some(v),
            _ => None,
        };
        let r = match v.and_then(|v| self.compared(env, v)) {
            Some(true) => self.can_be(hi, env, memo),
            Some(false) => self.can_be(lo, env, memo),
            None => {
                let (a, c) = self.can_be(lo, env, memo);
                let (d, e) = self.can_be(hi, env, memo);
                (a || d, c || e)
            }
        };
        memo.insert(b, r);
        r
    }

    /// `s_and` of EXEC and a compare of SGPRs and constants made under the same EXEC (the
    /// console compiler's form of a wave-uniform boolean, `v_cmp_* vcc, s, k` then
    /// `s_and_b64 sN, vcc, exec`): the compare's result. The compare holds the same bit in every
    /// lane, so SCC (some lane set) is that bit, in every lane, those outside EXEC included, as
    /// the console's wave holds it.
    // Guess: EXEC is not empty where the title's programs do this (they branch around their
    // EXEC regions with `s_cbranch_execz`); with an empty EXEC the console's SCC is 0.
    fn uniform_and_exec(&self, inst: &Inst) -> Option<Value> {
        let width = if self.x.native.wave64 { "s_and_b64" } else { "s_and_b32" };
        let Fields::Sop { src, .. } = &inst.fields else { return None };
        if inst.name() != width || src.len() != 2 {
            return None;
        }
        let code = |k: usize| match src[k] {
            Operand::Scalar { code, .. } => Some(code),
            _ => None,
        };
        let other = match (code(0)?, code(1)?) {
            (EXEC, c) | (c, EXEC) if c != EXEC => c,
            _ => return None,
        };
        let u = inst.read(Reg::S(other))?;
        let cmp = self.def_inst(u)?;
        let same_exec = (EXEC..=EXEC + u8::from(self.x.native.wave64))
            .all(|r| cmp.read(Reg::S(r)).is_some() && cmp.read(Reg::S(r)) == inst.read(Reg::S(r)));
        (matches!(cmp.fields, Fields::Valu(_)) && same_exec && self.uniform_atom(Atom::Lane(u)))
            .then_some(u)
    }

    /// Mesh: SCC of `s_and` of EXEC and a mask, or `s_andn2` of EXEC by one, where the mask is
    /// a function of atoms [`Self::uniform_atom`] takes once EXEC decides the rest (the title's
    /// NGG vertex programs test such masks so: compares of SGPRs, which hold 0 outside the
    /// compare's EXEC, and `s_cselect` by SCC): the same bit in every lane, so SCC is the
    /// function's bit, or its complement for `s_andn2`, with the atoms EXEC decides.
    // Guess: EXEC is not empty there, as for `uniform_and_exec`.
    fn uniform_with_exec(&self, inst: &Inst) -> Option<UniformBit> {
        let width = if self.x.native.wave64 { "_b64" } else { "_b32" };
        let Fields::Sop { src, .. } = &inst.fields else { return None };
        let code = |k: usize| match src.get(k)? {
            Operand::Scalar { code, .. } => Some(*code),
            _ => None,
        };
        let (other, not) = match (inst.name().strip_suffix(width)?, code(0)?, code(1)?) {
            ("s_and", EXEC, c) | ("s_and", c, EXEC) if c != EXEC => (c, false),
            ("s_andn2", EXEC, c) if c != EXEC => (c, true),
            _ => return None,
        };
        let v = inst.read(Reg::S(other))?;
        let st = &self.x.st;
        let &m = st.masks.get(&v).or_else(|| st.other.get(&v))?;
        // Atoms EXEC decides (the EXEC a compare ANDs its lanes with) are those values in it.
        let exec = inst.read(Reg::S(EXEC)).and_then(|e| st.masks.get(&e));
        let mut fixed = Vec::new();
        for a in st.bools.support(m) {
            if !self.uniform_atom(a) {
                fixed.push((a, st.bools.decides(&[(*exec?, true)], a)?));
            }
        }
        Some((m, not, fixed))
    }

    /// Whether `a` holds the same bit in every lane of a console wave: an SCC value, or a
    /// compare (or carry) of SGPRs and constants, which a wave holds once for all its lanes.
    fn uniform_atom(&self, a: Atom) -> bool {
        let st = &self.x.st;
        match a {
            Atom::Uniform(_) => true,
            Atom::Lane(v) => self.def_inst(v).is_some_and(|i| {
                i.reads.iter().all(|&(r, x)| match r {
                    Reg::V(_) => false,
                    Reg::S(c) => {
                        c == EXEC
                            || c == EXEC + 1
                            || !st.masks.contains_key(&x) && !st.highs.contains_key(&x)
                    }
                    Reg::Scc => true,
                })
            }),
            _ => false,
        }
    }

    /// The instruction writing `v`.
    fn def_inst(&self, v: Value) -> Option<&'a Inst> {
        let b = *self.def_block.get(&v)?;
        let p = self.x.p;
        p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))
    }

    /// The constant `v` holds, `env` giving phi values: moves of constants.
    fn konst(&self, env: &HashMap<Value, u32>, v: Value) -> Option<u32> {
        if let Some(&k) = env.get(&v) {
            return Some(k);
        }
        let inst = self.def_inst(v)?;
        let op = match &inst.fields {
            Fields::Valu(w)
                if inst.name() == "v_mov_b32" && w.sdwa.is_none() && w.dpp.is_none() =>
            {
                w.src.first()?
            }
            Fields::Sop { src, .. } if inst.name() == "s_mov_b32" => src.first()?,
            Fields::Sop { imm, .. } if inst.name() == "s_movk_i32" => {
                return Some(i32::from((*imm)? as i16) as u32);
            }
            _ => return None,
        };
        self.konst_operand(env, inst, op)
    }

    fn konst_operand(&self, env: &HashMap<Value, u32>, inst: &Inst, op: &Operand) -> Option<u32> {
        match *op {
            Operand::Const(c) => crate::resource::inline(c),
            Operand::Literal(x) => Some(x),
            Operand::Scalar { code, .. } if code != NULL => {
                self.konst(env, inst.read(Reg::S(code))?)
            }
            Operand::Vector { reg, .. } => self.konst(env, inst.read(Reg::V(reg))?),
            _ => None,
        }
    }

    /// The lane's bit of the 32-bit integer compare writing `v` (a VALU
    /// result before EXEC masking, or SCC), when its operands are constants.
    fn compared(&self, env: &HashMap<Value, u32>, v: Value) -> Option<bool> {
        let inst = self.def_inst(v)?;
        let name = inst.name();
        let rest = ["v_cmpx_", "v_cmp_", "s_cmp_"]
            .iter()
            .find_map(|p| name.strip_prefix(p))?;
        let (op, ty) = rest.rsplit_once('_')?;
        let (a, b) = match &inst.fields {
            Fields::Valu(w) if w.sdwa.is_none() && w.dpp.is_none() => {
                (w.src.first()?, w.src.get(1)?)
            }
            Fields::Sop { src, .. } => (src.first()?, src.get(1)?),
            _ => return None,
        };
        let (a, b) = (
            self.konst_operand(env, inst, a)?,
            self.konst_operand(env, inst, b)?,
        );
        let (sa, sb) = (a as i32, b as i32);
        Some(match (op, ty) {
            ("f", _) => false,
            ("t", _) => true,
            ("eq", "u32" | "i32") => a == b,
            ("ne" | "lg", "u32" | "i32") => a != b,
            ("lt", "u32") => a < b,
            ("le", "u32") => a <= b,
            ("gt", "u32") => a > b,
            ("ge", "u32") => a >= b,
            ("lt", "i32") => sa < sb,
            ("le", "i32") => sa <= sb,
            ("gt", "i32") => sa > sb,
            ("ge", "i32") => sa >= sb,
            _ => return None,
        })
    }

    /// The incoming edges (phi, predecessor index) over which a value can
    /// reach a read. A phi that only another read phi consumes is live too
    /// (else a value with no DXIL form reaching a read through two joins
    /// would be replaced by zero), but only over edges a lane can take on
    /// its way there: each step back adds the branch a lane took on that
    /// edge, and a contradiction (the lane that skipped an arm is not one
    /// that ran it) leaves the edge dead. A back edge forgets the branches
    /// (the loop redefines their conditions). Fewer conditions are weaker,
    /// so keeping only the last few stays sound, and so does walking a phi
    /// knowing nothing once it has been walked with `MAX_PATHS` sets.
    fn live_edges(&self) -> Set<(Value, usize)> {
        const MAX_KNOWN: usize = 4;
        // A branchy pixel shader of the title reached thousands of sets per
        // phi, each compared with all before it, and never finished.
        const MAX_PATHS: usize = 64;
        let (p, st) = (self.x.p, &self.x.st);
        let mut work: Vec<(Value, Vec<(Bool, bool)>)> =
            self.read_phis.iter().map(|&v| (v, Vec::new())).collect();
        let mut live = Set::default();
        let mut seen = Map::<Value, Vec<Vec<(Bool, bool)>>>::default();
        while let Some((v, known)) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            let done = seen.entry(v).or_default();
            // Known of fewer lanes than before: nothing new is live.
            if done.iter().any(|d| d.iter().all(|c| known.contains(c))) {
                continue;
            }
            // Knowing nothing covers every later set, so this is the phi's last walk.
            let known = if done.len() < MAX_PATHS { known } else { Vec::new() };
            done.push(known.clone());
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                if self.tail[q].is_none() {
                    continue;
                }
                let mut k = Vec::new();
                // Forward edges out of a block that is no loop's latch (a
                // loop's exit is also taken at its iteration limit).
                if let LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } = st.exits[q]
                    && taken != not_taken
                    && self.rpo[taken] > self.rpo[q]
                    && self.rpo[not_taken] > self.rpo[q]
                {
                    k.clone_from(&known);
                    k.push((when, taken == b));
                    k.drain(..k.len().saturating_sub(MAX_KNOWN));
                    if !st.bools.meet(&k) {
                        continue;
                    }
                } else if self.rpo[q] < self.rpo[b] {
                    k.clone_from(&known);
                }
                live.insert((v, n));
                let arg = p.blocks[b].phis[i].args[n];
                if self.phi_at.contains_key(&arg) {
                    work.push((arg, k));
                }
            }
        }
        live
    }

    /// An unguarded indexed write replaces the incoming value before a
    /// consumer can observe it, either on this edge or in a counted loop.
    fn overwritten_movreld_edge(&self, b: usize, phi: &Phi, edge: usize) -> Option<()> {
        let Reg::V(reg) = phi.reg else { return None };
        let p = self.x.p;
        let block = &p.blocks[b];
        if self.sections.contains_key(&b)
            || self.sections.contains_key(&block.preds[edge])
            || p.blocks
                .iter()
                .flat_map(|bb| &bb.phis)
                .any(|ph| ph.args.contains(&phi.value))
        {
            return None;
        }
        let mut uses = p.blocks.iter().enumerate().flat_map(|(bb, block)| {
            block.insts.iter().enumerate().filter_map(move |(i, inst)| {
                inst.reads
                    .iter()
                    .any(|r| r.1 == phi.value)
                    .then_some((bb, i, inst))
            })
        });
        let (bb, i, mov) = uses.next()?;
        if uses.next().is_some()
            || bb != b
            || self.x.st.guards[b][i].is_some()
            || mov.name() != "v_movreld_b32"
            || mov.read(phi.reg) != Some(phi.value)
        {
            return None;
        }
        let Fields::Valu(v) = &mov.fields else {
            return None;
        };
        let (
            Some(Operand::Vector {
                reg: base,
                dwords: 1,
            }),
            [
                Operand::Vector {
                    reg: src,
                    dwords: 1,
                },
            ],
        ) = (v.dst, v.src.as_slice())
        else {
            return None;
        };
        if v.dpp.is_some()
            || v.sdwa.is_some()
            || mov.read(Reg::V(*src))? == phi.value
            || mov.write(phi.reg).is_none()
        {
            return None;
        }
        let index = u32::from(reg.checked_sub(base)?);
        // Follow M0 to a copy of a scalar phi in this block, then the
        // constant written on the corresponding predecessor.
        let (copy, [src]) = scalar_def(block, mov.read(Reg::S(crate::ir::M0))?, "s_mov_b32")?;
        let scalar = sgpr_operand(copy, src)?;
        let index_phi = block.phis.iter().find(|ph| ph.value == scalar)?;
        let (_, [initial]) = scalar_def(
            &p.blocks[block.preds[edge]],
            index_phi.args[edge],
            "s_mov_b32",
        )?;
        let known = scalar_constant(initial)?;
        if index == known {
            return Some(());
        }
        if index < known {
            return None; // Reaching this slot would require the index to wrap.
        }
        self.counted_movreld_index(b, edge, phi, index_phi, mov, index)
    }

    /// An unguarded move fills one more register on every iteration.
    /// Values from before the loop can only survive in slots the index skips.
    fn counted_movreld_index(
        &self,
        b: usize,
        edge: usize,
        phi: &Phi,
        indices: &Phi,
        mov: &Inst,
        slot: u32,
    ) -> Option<()> {
        let p = self.x.p;
        let st = self.x.st;
        let block = &p.blocks[b];
        let LaneExit::Branch {
            when,
            taken: latch,
            not_taken: exit,
        } = st.exits[b]
        else {
            return None;
        };
        let back = block.preds.iter().position(|&q| q == latch)?;
        if block.preds.len() != 2
            || back == edge
            || b == latch
            || b == exit
            || exit == latch
            || self.sections.contains_key(&latch)
            || st.exits[latch] != LaneExit::Jump(b)
            || p.blocks[latch].preds != [b]
            || p.blocks[exit].preds != [b]
            || st.exits[exit] != LaneExit::End
            || self.rpo[b] >= self.rpo[latch]
            || self.rpo[block.preds[edge]] >= self.rpo[b]
        {
            return None;
        }
        // The indexed result may only go around the loop as its own old
        // value or be consumed in the normal exit block.
        let result = mov.write(phi.reg)?;
        if phi.args[back] != result {
            return None;
        }
        for (bb, block) in p.blocks.iter().enumerate() {
            if bb != exit
                && block
                    .insts
                    .iter()
                    .any(|inst| inst.reads.iter().any(|r| r.1 == result))
            {
                return None;
            }
            if block.phis.iter().any(|other| {
                other
                    .args
                    .iter()
                    .enumerate()
                    .any(|(n, &arg)| arg == result && (other.value != phi.value || n != back))
            }) {
                return None;
            }
        }
        // Follow the latch copy to the unit increment, then require that
        // this very increment is compared for equality to the exit bound.
        let (copy, [src]) = scalar_def(&p.blocks[latch], indices.args[back], "s_mov_b32")?;
        let next = sgpr_operand(copy, src)?;
        let (increment, [from, step]) = scalar_def(block, next, "s_add_i32")?;
        if sgpr_operand(increment, from)? != indices.value || scalar_constant(step)? != 1 {
            return None;
        }
        let (Atom::Uniform(scc), Bool::TRUE, Bool::FALSE) = st.bools.node(when)? else {
            return None;
        };
        let branch = block.insts.last()?;
        if branch.name() != "s_cbranch_scc0" || branch.read(Reg::Scc) != Some(scc) {
            return None;
        }
        let (cmp, [tested, bound]) = scalar_def(block, scc, "s_cmp_eq_u32")?;
        if sgpr_operand(cmp, tested)? != next {
            return None;
        }
        let limit = scalar_constant(bound)?;
        // The initial index is below this slot. The loop must pass it before exit.
        (slot < limit).then_some(())
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        let live = self.live_edges();
        let mut retry = Vec::new();
        'phi: for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                let read = live.contains(&(phi.value, n));
                let x = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    // No lane from this edge reads it, or no lane comes
                    // along it at all: any value will do.
                    _ if read && (self.never_taken(q, b) || self.dead_on_edge(b, n, phi.reg)) => {
                        if bit {
                            self.s.m.bool(false)
                        } else {
                            self.s.m.i32(0)
                        }
                    }
                    // An input an unguarded counted `v_movreld` loop overwrites
                    // before anything reads it (PR #457).
                    (Some(Val::Missing("barycentrics read directly")), false)
                        if read && self.overwritten_movreld_edge(b, phi, n).is_some() =>
                    {
                        self.s.m.undef(self.s.t.i32)
                    }
                    (Some(Val::Bit(_)), false) if read => {
                        retry.push((phi.value, "an SCC joined with SGPR values"));
                        continue 'phi;
                    }
                    // A lane mask joined with data at a join every lane
                    // enters along one edge: the register holds the mask's
                    // number there (exact with console lanes, or for host-wave
                    // masks in `MaskData::Host`).
                    _ if read
                        && !bit
                        && (self.mask_data == MaskData::Console
                            || self.mask_data == MaskData::Host
                                && self.is_host_wave_mask_phi(phi.value))
                        && (self.x.st.masks.contains_key(&arg)
                            || self.x.st.highs.contains_key(&arg))
                        && self.uniform_join(b, |_| true) =>
                    {
                        match self.mask_dword(arg) {
                            Ok(x) => x,
                            Err(why) => {
                                retry.push((phi.value, why));
                                continue 'phi;
                            }
                        }
                    }
                    (Some(Val::Missing(why)), _) if read => {
                        retry.push((phi.value, why));
                        continue 'phi;
                    }
                    _ if read => {
                        retry.push((phi.value, "a join of mismatched values"));
                        continue 'phi;
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        if !retry.is_empty() {
            return Err(Stop::Retry(retry));
        }
        while let Some((x, b, i, lane)) = self.pending_spills.pop() {
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let a = p.blocks[b].phis[i].args[n];
                let y = self
                    .spill_slot(a, lane, &mut Vec::new(), true)
                    .map_err(|w| unsupported(p.blocks[b].offset, w))?;
                incoming.push((y, qb));
            }
            self.s.m.set_phi(x, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        let mut mesh_split = None;
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            // The high dword's join, in a wave64 program.
            let high = self
                .pair_high(phi.value)
                .and_then(|h| self.phi_at.get(&h))
                .map(|&(hb, hi)| &p.blocks[hb].phis[hi]);
            // A mesh thread knows its console lane. If a uniform branch skips a
            // single block, the whole wave brings either its mask or the scalar
            // data. Two unconditional jumps from separate arms prove no such thing.
            let preds = &p.blocks[b].preds;
            let mesh_branch = self.s.mesh.is_some()
                && matches!(preds.as_slice(), &[a, c] if a != b && c != b
                    && (p.blocks[a].preds == [c] || p.blocks[c].preds == [a]))
                && phi.args.iter().enumerate().any(|(n, &lo)| {
                    self.x.st.mask_bits(lo, high.map(|h| h.args[n])).is_some()
                });
            let uniform = self.uniform_join(b, |_| true);
            let mesh_number = uniform && mesh_branch && {
                // Each host half can read a different first lane.
                let split = mesh_split.get_or_insert_with(|| split_scc(p));
                !split.contains(&phi.value)
                    && high.is_none_or(|h| !split.contains(&h.value))
                    && self.uniform_join(b, |scc| !split.contains(&scc))
            };
            let number = self.console_lanes && uniform || mesh_number;
            let mut incoming = Vec::new();
            for (n, &q) in preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                // Each way out names what the mask is joined with, so a refusal log says which
                // is missing.
                const HOST_LANES: Why =
                    "a lane-mask join of an untracked mask: data, on the host's lanes";
                const DIVERGENT: Why =
                    "a lane-mask join of an untracked mask: data, at a join of several edges";
                const HIGH: Why =
                    "a lane-mask join of an untracked mask: data whose high dword is not known";
                const MISSING: Why =
                    "a lane-mask join of an untracked mask: a value the translation could not make";
                const BIT: Why = "a lane-mask join of an untracked mask: an SCC or lane bit";
                const NONE: Why = "a lane-mask join of an untracked mask: no value on that edge";
                // Data joined with masks where the wave takes one edge (the BVH
                // walks pop masks from memory): with console lanes the lane's
                // own bit of the number, as `Atom::Opaque`: of the low dword for
                // lanes below 32, of the high one above (a mask pair's lane `i`
                // is its bit `i`).
                let mask = if mesh_number {
                    self.x.st.mask_bits(arg, high.map(|h| h.args[n]))
                } else {
                    self.x.st.masks.get(&arg).copied()
                };
                let x = match (mask, self.vals.get(&arg)) {
                    (Some(m), _) => self.boolean(m, qb).map_err(|w| unsupported(offset, w))?,
                    (None, Some(&Val::I32(word))) if number => {
                        let hi = if self.x.native.wave64 {
                            match high.and_then(|h| self.vals.get(&h.args[n])) {
                                Some(&Val::I32(hi)) => Some(hi),
                                _ => return Err(unsupported(offset, HIGH)),
                            }
                        } else {
                            None
                        };
                        self.lane_bit(qb, word, hi)
                    }
                    (None, _) if !uniform && self.no_mask_lane_on_edge(b, &phi.args, n) => {
                        self.s.m.bool(false)
                    }
                    // Read only ANDed with an EXEC those lanes are not in, or not read at all
                    // but by joins that do not need it: any bit will do, so what the edge brings
                    // (data, or a value not made) is not needed.
                    (None, _) if self.unneeded_edges().contains(&(phi.value, n)) => {
                        self.s.m.bool(false)
                    }
                    // A lane function the structure knows though no mask tracks it (a constant,
                    // SCC, a join that resolved): each lane's bit of it, unless another edge's
                    // mask may differ from it in the lanes of this one (`mask_differs_on_edge`).
                    (None, _)
                        if let Some(&m) = self.x.st.other.get(&arg)
                            && (uniform || !self.mask_differs_on_edge(b, &phi.args, n, m))
                            && !self.x.st.bools.support(m).iter().any(|a| {
                                matches!(a, Atom::Opaque(_) | Atom::Quad(_))
                            }) =>
                    {
                        self.boolean(m, qb).map_err(|w| unsupported(offset, w))?
                    }
                    // Data that is all lanes or none (`s_mov_b64 s, -1`) is that bit on any lanes.
                    (None, Some(&Val::I32(word)))
                        if !self.console_lanes
                            && let Some(all) = self.all_or_none(word, high.map(|h| h.args[n])) =>
                    {
                        self.s.m.bool(all)
                    }
                    (None, Some(Val::I32(_))) if !self.console_lanes => {
                        return Err(unsupported(offset, HOST_LANES));
                    }
                    (None, Some(Val::I32(_))) => return Err(unsupported(offset, DIVERGENT)),
                    (None, Some(Val::Missing(_))) => return Err(unsupported(offset, MISSING)),
                    (None, Some(Val::Bit(_))) => return Err(unsupported(offset, BIT)),
                    (None, None) => return Err(unsupported(offset, NONE)),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }
}

/// The refusal of LDS a split console wave shares in divergent code.
pub(crate) const SPLIT_LDS: &str = "LDS shared in a wave the host splits, in divergent code";
/// The refusal of a cross-lane operation between host waves in divergent code.
pub(crate) const DIVERGENT_EXCHANGE: &str = "a cross-lane operation between host waves in divergent code";

/// Blocks each lane runs exactly once: no cycle through them, and on every
/// path from the entry to an end. Group barriers go only there.
pub(crate) fn once_blocks(p: &Program) -> Vec<bool> {
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    // Blocks reachable from `from`'s successors (or from `from` itself when
    // `start`), never entering `avoid`.
    let reach = |from: usize, start: bool, avoid: Option<usize>| {
        let mut seen = vec![false; n];
        let mut stack = if start { vec![from] } else { succ(from) };
        while let Some(b) = stack.pop() {
            if Some(b) == avoid || std::mem::replace(&mut seen[b], true) {
                continue;
            }
            stack.extend(succ(b));
        }
        seen
    };
    (0..n)
        .map(|b| {
            let ends_around = b != 0
                && reach(0, true, Some(b))
                    .iter()
                    .enumerate()
                    .any(|(q, &r)| r && succ(q).is_empty());
            !reach(b, false, None)[b] && !ends_around
        })
        .collect()
}

/// Blocks outside any cycle that hold one of the title's own `s_barrier`s. The console's barrier
/// waits for every wave of the group that has not ended, so the title brings every live wave
/// there together, and the host lowers it to a group barrier on that same promise: an exchange
/// between host waves in such a block is reached by the same threads, once (the G-buffer culling
/// shaders' whole-wave OR in groups of two console waves, whose culled waves end early).
///
/// Not a block reached past a branch on an SCC from [`split_scc`]: the console wave takes such a
/// branch whole, so the title may put one before its barrier (a wave-uniform early return), but
/// each host wave computes it from its own first lane, so one half could skip the block while
/// the other waits at the exchange's group barrier. As in [`wave_blocks`], such an SCC in the
/// block's reach condition excludes it.
pub(crate) fn barrier_blocks(p: &Program, st: &Structured) -> Vec<bool> {
    let n = p.blocks.len();
    let split = split_scc(p);
    (0..n)
        .map(|b| {
            if !p.blocks[b].insts.iter().any(|i| i.name() == "s_barrier") {
                return false;
            }
            let apart = |a: &Atom| matches!(a, Atom::Uniform(v) if split.contains(v));
            if st.bools.support(st.reach[b]).iter().any(apart) {
                return false;
            }
            let mut seen = vec![false; n];
            let mut stack = p.blocks[b].exit.successors();
            while let Some(x) = stack.pop() {
                if x == b {
                    return false;
                }
                if !std::mem::replace(&mut seen[x], true) {
                    stack.extend(p.blocks[x].exit.successors());
                }
            }
            true
        })
        .collect()
}

/// SCC values that can differ between the host waves of one console wave:
/// those computed from a `v_readfirstlane_b32` result, which each host wave
/// reads of its own first lane.
fn split_scc(p: &Program) -> Set<Value> {
    let mut from = Set::default();
    loop {
        let n = from.len();
        for b in &p.blocks {
            for phi in &b.phis {
                if phi.args.iter().any(|a| from.contains(a)) {
                    from.insert(phi.value);
                }
            }
            for i in &b.insts {
                if i.name() == "v_readfirstlane_b32" || i.reads.iter().any(|r| from.contains(&r.1)) {
                    from.extend(i.writes.iter().map(|w| w.1));
                }
            }
        }
        if from.len() == n {
            return from;
        }
    }
}

/// Blocks every lane of a one-wave group runs together, at most once: those
/// of [`once_blocks`], and those outside any cycle reached by uniform (SCC)
/// branches alone, which the whole console wave takes or skips at once (and
/// so do its host waves: no SCC from [`split_scc`]).
pub(crate) fn wave_blocks(p: &Program, st: &Structured) -> Vec<bool> {
    let n = p.blocks.len();
    let split = split_scc(p);
    let mut once = once_blocks(p);
    for (b, once) in once.iter_mut().enumerate() {
        let mut seen = vec![false; n];
        let mut stack = p.blocks[b].exit.successors();
        let mut cycle = false;
        while let Some(x) = stack.pop() {
            cycle |= x == b;
            if !std::mem::replace(&mut seen[x], true) {
                stack.extend(p.blocks[x].exit.successors());
            }
        }
        *once |= !cycle
            && st
                .bools
                .support(st.reach[b])
                .iter()
                .all(|a| matches!(a, Atom::Uniform(v) if !split.contains(v)));
    }
    once
}

/// Where a compute group that is one console wave needs group barriers
/// when the host splits it into narrower waves: (block, instruction) to
/// sync before, instruction index = the block's length for its end.
///
/// The console runs such a group in lockstep, so its compiler leaves out
/// the barrier between one lane's LDS write and another's read. Here a
/// read (or atomic) of LDS that a write may have changed since the last
/// barrier gets one before it. Barriers need every lane of the group, so
/// they go only in blocks each lane runs exactly once (no cycle through
/// them, on every path from the entry to an end); such a block still dirty
/// at its end syncs there when a successor is not one, so divergent code
/// after it reads clean LDS. `None`: a read in divergent code may see a
/// write made in divergent code since the last such block.
pub(crate) fn split_wave_barriers(p: &Program, st: &Structured) -> Option<Set<(usize, usize)>> {
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    let uniform = wave_blocks(p, st);
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut dirty_out = vec![false; n];
    loop {
        let mut at = Set::default();
        let mut changed = false;
        for &b in &order {
            let mut dirty = p.blocks[b].preds.iter().any(|&q| dirty_out[q]);
            for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                let name = inst.name();
                if name == "s_barrier" {
                    dirty = false;
                }
                if !matches!(inst.fields, Fields::Ds { .. })
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute")
                {
                    continue;
                }
                if dirty && !name.starts_with("ds_write") {
                    if !uniform[b] {
                        return None;
                    }
                    at.insert((b, i));
                }
                dirty = !name.starts_with("ds_read");
            }
            if dirty && uniform[b] && succ(b).iter().any(|&s| !uniform[s]) {
                at.insert((b, p.blocks[b].insts.len()));
                dirty = false;
            }
            changed |= std::mem::replace(&mut dirty_out[b], dirty) != dirty;
        }
        if !changed {
            return Some(at);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Lowered, Numeric, Parts, buffer_format, half_shift, lower, packed_half};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn interp_copy_chain(moves: usize, arithmetic: bool) -> (bool, usize) {
        let mut words = Vec::new();
        let mut src = 0u32;
        for _ in 0..moves {
            let dst = 1 - src;
            words.push(0xBE80_0300 | dst << 16 | src); // s_mov_b32 between s0 and s1
            src = dst;
        }
        words.push(0xBEFC_0300 | src); // s_mov_b32 m0, src
        words.push(0xC81A_0002); // v_interp_mov_f32 v6, p0, attr0.x
        if arithmetic {
            words.push(0x8002_8100 | src); // s_add_u32 s2, src, 1
        }
        words.push(S_ENDPGM);
        let code: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, false, 8).unwrap();
        let &(_, start) = p.inputs.iter().find(|(r, _)| *r == ir::Reg::S(0)).unwrap();
        let uses = super::InterpUses::new(&p);
        (uses.is_interp_m0_only(start), uses.visits.get())
    }

    #[test]
    fn interp_m0_only_visits_each_copy_and_interpolation_once() {
        let chains = [128, 256];
        assert_eq!(
            chains.map(|moves| interp_copy_chain(moves, false)),
            chains.map(|moves| (true, moves + 2))
        );
    }

    #[test]
    fn interp_m0_only_rejects_a_copy_also_read_by_arithmetic() {
        assert!(!interp_copy_chain(256, true).0);
    }

    /// Whether a synthetic compute program translates with these input registers.
    fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
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
            wave32_host: false,
            zero_copy_memory: false,
            rect_list: false,
            copy_params: [0; 32],
            view_marks: false,
        })
        .map(|_| ())
    }

    /// Lowers a compute program with the root parameters `layout` (in user
    /// SGPRs from s0 on) and the used resources `used`.
    fn lower_with(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_wave(words, layout, used, false)
    }

    /// [`lower_with`] in wave64 if `wave64`.
    fn lower_wave(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                // The layout's user SGPRs (at least two), 512 bytes of LDS.
                rsrc2: 1 << 15 | sgprs.max(2) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
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
            wave32_host: false,
            zero_copy_memory: false,
            rect_list: false,
            copy_params: [0; 32],
            view_marks: false,
        })
    }

    /// Root parameter `slot` in user SGPRs from `user_data` on.
    fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: if root == RootType::Descriptor { 2 } else { 1 },
            base_register: 0,
            user_data,
            slot,
        }
    }

    /// Lowers a one-thread compute program that copies four root constants
    /// (s0..s3) into v1..v4, runs `body`, and stores v0 to a raw buffer from a
    /// descriptor table, so the vector instructions in `body` have an effect.
    fn lowers_valu(body: &[u32]) -> Result<(), crate::ShaderError> {
        let mut words = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xFA00_0000,
        ];
        // v_mov_b32 v1..v4, s0..s3
        words.extend((0..4u32).map(|i| 0x7E00_0200 | (i + 1) << 17 | i));
        words.extend_from_slice(body);
        words.extend_from_slice(&[
            0xE070_0000, // buffer_store_dword v0, off, s[8:11], 0
            0x8002_0000,
            S_ENDPGM,
        ]);
        let constants = RootParam {
            sgprs: 4,
            ..root(RootType::Constants, 0, 0)
        };
        let buffer = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        lower_with(
            &words,
            vec![constants, root(RootType::Table, 4, 1)],
            vec![buffer],
        )
        .map(|_| ())
    }

    #[test]
    fn the_24_bit_multiplies_lower_both_halves() {
        // VOP2 `op v0, v1, v2`: v_mul_i32_i24 and v_mul_u32_u24 (the low
        // dword) and v_mul_hi_i32_i24 and v_mul_hi_u32_u24 (the high one).
        for op in [0x09u32, 0x0A, 0x0B, 0x0C] {
            let word = op << 25 | 2 << 9 | 0x101;
            lowers_valu(&[word]).unwrap_or_else(|e| panic!("VOP2 {op:#x}: {e}"));
        }
    }

    #[test]
    fn sums_of_absolute_differences_lower() {
        // VOP3 `op v0, v1, v2, v3`: v_sad_u8, v_sad_hi_u8, v_sad_u16,
        // v_sad_u32 and v_msad_u8.
        for op in [0x15Au32, 0x15B, 0x15C, 0x15D, 0x171] {
            let words = [0xD400_0000 | op << 16, 0x101 | 0x102 << 9 | 0x103 << 18];
            lowers_valu(&words).unwrap_or_else(|e| panic!("VOP3 {op:#x}: {e}"));
        }
    }

    #[test]
    fn buffer_cache_invalidations_lower_as_no_ops() {
        // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
        // host to invalidate its cached view of a buffer.
        assert!(
            lowers(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(),
            "buffer_gl0_inv"
        );
        assert!(
            lowers(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(),
            "buffer_gl1_inv"
        );
    }

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];

        for words in [add, sub] {
            lowers(&words, 0).unwrap();
        }
    }

    #[test]
    fn a_signed_overflow_reaches_the_select() {
        // SCC of s_add_i32, s_sub_i32 and s_addk_i32 is their signed
        // overflow; a select on it, and a branch on that (only code with an
        // effect needs its operands), used to be refused.
        for (what, op) in [
            ("s_add_i32 s0, s0, 1", 0x8100_8100),
            ("s_sub_i32 s0, s0, 1", 0x8180_8100),
            ("s_addk_i32 s0, 0x7fff", 0xB780_7FFF),
        ] {
            let words = [
                0xBE80_03C1, // s_mov_b32 s0, -1
                0xBE81_0380, // s_mov_b32 s1, 0
                op,
                0x8502_0001, // s_cselect_b32 s2, s1, s0
                0xBF07_8002, // s_cmp_lg_u32 s2, 0
                0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
                0xBE83_0380, // s_mov_b32 s3, 0
                S_ENDPGM,
            ];
            lowers(&words, 0).unwrap_or_else(|e| panic!("{what}: {e}"));
        }
    }

    #[test]
    fn v_cndmask_accepts_uniform_and_constant_masks() {
        // Uniform select (`s_cselect_b32`) and inline constant masks in `v_cndmask_b32`
        // (resolved in `Structured::other` or directly as constants).
        let words = [
            0xBF06_8000u32, // s_cmp_eq_u32 s0, 0
            0x8500_80C1,    // s_cselect_b32 s0, -1, 0
            0xD501_0001,
            0x0001_0080, // v_cndmask_b32 v1, 0, 1, s0
            0xD501_0002,
            0x0201_02C1, // v_cndmask_b32 v2, 1, v1, -1
            0xD834_0000,
            0x0000_0200, // ds_write_b32 v0, v2
            S_ENDPGM,
        ];
        lowers(&words, 0x8001).unwrap();
    }

    fn perm_features(selector: u32) -> u64 {
        // Synthetic compute shader:
        // v0 is the thread ID and can supply a dynamic selector.
        // Store the result in LDS to keep the permutation live.
        let words = [
            0x7E02_02FFu32,
            0x817F_34AB,
            0x7E04_02FF,
            0xFE01_56CD,
            0xD744_0003,
            257 | (258 << 9) | (selector << 18),
            0x3408_0082,
            0xD834_0000,
            0x0000_0304,
            S_ENDPGM,
        ];

        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [64, 1, 1],
            }),
            code,
        };

        let p = ir::build(&native.code, false, 256).unwrap();

        assert!(
            p.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.name() == "v_perm_b32")
        );

        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

        let io = io::raise(&p, &native).unwrap();

        let emitted = lower(&Parts {
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
            wave32_host: false,
            zero_copy_memory: false,
            rect_list: false,
            copy_params: [0; 32],
            view_marks: false,
        })
        .unwrap();

        let container = crate::Container::parse(&emitted.dxil).unwrap();

        let features = container.part(crate::FourCC::SFI0).unwrap();

        u64::from_le_bytes(features.data.try_into().unwrap())
    }

    #[test]
    fn perm_constant_selector_needs_no_int64() {
        assert_eq!(perm_features(128) & 0x8000, 0);
    }

    #[test]
    fn perm_dynamic_selector_needs_no_int64() {
        assert_eq!(
            perm_features(256) & 0x8000,
            0,
            "dynamic v_perm_b32 must not require INT64"
        );
    }

    #[test]
    fn a_descriptor_selected_by_scc_picks_its_heap_index() {
        // Two buffer descriptors loaded from the table at s0, one picked by
        // a root constant (s1) with s_cselect_b64 per half, and loaded
        // through by index and stored to LDS so the load is needed.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
            0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
            0xE030_2000,
            0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        // Both are structured buffers of dwords in the table's range.
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        lower_with(&words, layout(), used()).unwrap();
    }

    /// Tables of 2D textures at s0 and samplers at s1.
    fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let range = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
            user_data,
            slot,
        };
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
            vec![
                range(ResourceType::Srv, 0, 0),
                range(ResourceType::Sampler, 1, 1),
            ],
        )
    }

    #[test]
    fn a_descriptor_loaded_only_for_its_size_takes_the_kind_of_its_reload() {
        // The record count of the buffer at s0 (count - 1 as an index), then
        // the same descriptor loaded again for the indexed read.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0x810C_C10A, // s_add_i32 s12, s10, -1
            0xF408_0402,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[4:5], null
            0x7E04_020C, // v_mov_b32 v2, s12
            0xE030_2000,
            0x8004_0102, // buffer_load_dword v1, v2, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        let used = vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];
        lower_with(&words, layout, used).unwrap();
    }

    #[test]
    fn a_waterfall_loop_that_only_sizes_takes_the_kind_of_the_sampled_one() {
        // Each lane names table entry (lane & 1) by its byte offset in v7.
        // A first waterfall loop only reads the width bits of the entry
        // (`v_alignbit_b32`), a second one samples it; the width goes to the
        // buffer at dword 16 of the table.
        let words = [
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
            0x7E28_0507, // L1: v_readfirstlane_b32 s20, v7
            0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
            0xF40C_0302,
            0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xD54E_0008,
            0x0278_1A0E, // v_alignbit_b32 v8, s14, s13, 30
            0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
            0xBF89_FFF7, // s_cbranch_execnz L1
            0xBEFE_0317, // s_mov_b32 exec_lo, s23
            0xBE95_037E, // s_mov_b32 s21, exec_lo
            0x7E28_0507, // L2: v_readfirstlane_b32 s20, v7
            0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
            0xF40C_0302,
            0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
            0xBF89_FFF6, // s_cbranch_execnz L2
            0xBEFE_0317, // s_mov_b32 exec_lo, s23
            0xE070_1000,
            0x800A_0810, // buffer_store_dword v8, v16, s[40:43], 0 offen
            S_ENDPGM,
        ];
        let (layout, used) = texture_and_uav_tables();
        lower_wave(&words, layout, used, false).unwrap();
    }

    #[test]
    fn compute_spills_beside_a_descriptor_word_read_back() {
        // Dword 0 of a buffer descriptor (no host value) and a root constant
        // spilled to lanes 0 and 1 of v5 in a compute program; reading lane
        // 1 back is the constant (the descriptor's lane is never read).
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xD761_0005,
            0x0001_0008, // v_writelane_b32 v5, s8, 0
            0xD761_0005,
            0x0001_0201, // v_writelane_b32 v5, s1, 1
            0xD760_000C,
            0x0001_0305, // v_readlane_b32 s12, v5, 1
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0x7E06_020C, // v_mov_b32 v3, s12
            0xD834_0000,
            0x0000_0300, // ds_write_b32 v0, v3
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        let used = vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];
        lower_with(&words, layout, used).unwrap();
    }

    #[test]
    fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
        use crate::resource::tests::{
            WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
        };
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
            (WATERFALL_PICKED_WAVE32, false),
        ] {
            let (layout, used) = texture_tables();
            let l = lower_wave(words, layout, used, wave64).unwrap();
            // The loop's reads of the first lane became each lane's own
            // dwords: no wave operation is left.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    /// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
    pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let (layout, mut used) = texture_tables();
        used.push(UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 16,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        });
        (layout, used)
    }

    #[test]
    fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
        use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
        for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
            let (layout, used) = texture_and_uav_tables();
            let l = lower_wave(words, layout, used, false).unwrap();
            // One image handle, from the join's per-lane heap index.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    #[test]
    fn an_address_half_set_where_only_lanes_that_never_run_skip_is_known() {
        // As `indirect_light_probes:CS_Alpha_Schedule` after its texture: a
        // table address whose high half (s25 = 0) is set inside the region
        // that only a skip no lane takes bypasses; the skip edge would bring
        // the stale input s25.
        let words = [
            0xBE9A_037E, // s_mov_b32 s26, exec_lo
            0xBEFE_097E, // s_wqm_b32 exec_lo, exec_lo
            0xBE98_0300, // s_mov_b32 s24, s0
            0xBE86_3C1A, // s_and_saveexec_b32 s6, s26
            0xBF88_0001, // s_cbranch_execz -> join
            0xBE99_0380, // s_mov_b32 s25, 0
            0xBEFE_0306, // join: s_mov_b32 exec_lo, s6
            0xF408_0A0C,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[24:25], 0x40
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x7E06_0280, // v_mov_b32 v3, 0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE070_1000,
            0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
            S_ENDPGM,
        ];
        let (_, mut used) = texture_and_uav_tables();
        used.retain(|u| u.user_data == 0);
        let layout = vec![root(RootType::Table, 0, 0)];
        lower_with(&words, layout, used).expect("the high half is 0 wherever a lane runs");
    }

    #[test]
    fn a_descriptor_skipped_only_by_lanes_that_never_run_needs_no_source() {
        // As `indirect_light_probes:CS_Alpha_Schedule`: the texture is
        // loaded under the lanes that ran before `s_wqm`, skipped when there
        // are none, and sampled with the helper lanes after the join. With
        // every lane running from the start nothing takes the skip, so the
        // registers it would bring (no descriptor) are never read.
        let words = [
            0xBE9A_037E, // s_mov_b32 s26, exec_lo
            0xBEFE_097E, // s_wqm_b32 exec_lo, exec_lo
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0xBE86_3C1A, // s_and_saveexec_b32 s6, s26
            0xBF88_0002, // s_cbranch_execz -> join
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
            0xBEFE_0306, // join: s_mov_b32 exec_lo, s6
            0xBE88_03FF,
            0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0x7E04_02F0, // v_mov_b32 v2, 0.5
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ];
        let (_, mut used) = texture_and_uav_tables();
        used.retain(|u| u.user_data == 0);
        let layout = vec![root(RootType::Table, 0, 0)];
        let l = lower_with(&words, layout, used).unwrap();
        assert_eq!(l.nonuniform_handles, 0);
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil.clone())
                .expect("DXIL validation failed");
        }
    }

    #[test]
    fn descriptors_joined_by_nested_control_flow_lower_to_one_uniform_handle() {
        // The view table at s0 (2D textures at entries 0 and 1, a raw UAV at
        // dword 16) and a root constant at s1: entry 0 if s1 is 0, else
        // entry 1 if it is 1, else entry 0 again, the inner if/else joined
        // before the outer one. The join's edges are a load and a join.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0xBE88_03FF,
            0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0xBF84_0003, // s_cbranch_scc0 -> 0x38
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
            0xBF82_0008, // s_branch -> 0x58
            0xBF06_8101, // 0x38: s_cmp_eq_u32 s1, 1
            0xBF84_0003, // s_cbranch_scc0 -> 0x4c
            0xF40C_0302,
            0xFA00_0020, // s_load_dwordx8 s[12:19], s[4:5], 0x20
            0xBF82_0002, // s_branch -> 0x54
            0xF40C_0302,
            0xFA00_0000, // 0x4c: s_load_dwordx8 s[12:19], s[4:5], 0x0
            0xBE86_0301, // 0x54: s_mov_b32 s6, s1
            0x7E04_02F0, // 0x58: v_mov_b32 v2, 0.5
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ];
        let (_, mut used) = texture_and_uav_tables();
        used.retain(|u| u.user_data == 0);
        let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        let l = lower_with(&words, layout, used).unwrap();
        // One image handle from the joins' heap index, the same in all lanes.
        assert_eq!(l.nonuniform_handles, 0);
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil.clone())
                .expect("DXIL validation failed");
        }
    }

    #[test]
    fn descriptors_reloaded_from_a_root_table_share_one_handle() {
        // The view table at s0 (2D textures at entries 0 and 1, a raw UAV at
        // dword 16) and a root constant at s1. Entry 0 is loaded and sampled
        // in a branch, then loaded again and sampled after it (the console
        // compiler reloads a descriptor before each use rather than keep it).
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0xBE88_03FF,
            0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0x7E04_02F0, // v_mov_b32 v2, 0.5
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0xBF84_0005, // s_cbranch_scc0 -> the reload
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0 (again)
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ];
        let (_, mut used) = texture_and_uav_tables();
        used.retain(|u| u.user_data == 0);
        let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        let l = lower_with(&words, layout, used).unwrap();
        // One image handle made in the entry block for both loads, and the
        // UAV's.
        assert_eq!(l.heap_handles, 2);
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil.clone())
                .expect("DXIL validation failed");
        }
    }

    #[test]
    fn scalar_bit_tests_lower() {
        // Each tests one bit of a constant into SCC and branches on it (only
        // code with an effect needs its operands).
        let tail = [
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let bit_tests: [&[u32]; 4] = [
            &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
            &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
            ],
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
            ],
        ];
        for head in bit_tests {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers(&words, 0).unwrap();
        }
    }

    #[test]
    fn a_missing_value_stays_missing_through_two_live_joins() {
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // skip the first overwrite
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
            0xBF85_0001, // skip the second overwrite
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // use the result to branch
            0xBE83_0380,
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        // Compare to a direct read so future timestamp support works in both cases.
        let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[8..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
        words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
        lowers(&words, group_ids_xy).expect("dead timestamp");
    }

    #[test]
    fn a_value_no_lane_from_its_edge_reads_needs_no_form() {
        // s6 is a timestamp (no DXIL form) where the load is skipped and 5
        // where it runs; a flag in v1, set only with the load, guards the
        // read after the join (the console compiler's form, where the
        // skipped path leaves a stale lane mask in the register).
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0x7E02_0280, // v_mov_b32 v1, 0
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0002, // s_cbranch_scc1 -> join
            0xBE86_0385, // s_mov_b32 s6, 5
            0x7E02_02C1, // v_mov_b32 v1, -1
            0x7D84_02C1, // join: v_cmp_eq_u32 vcc_lo, -1, v1
            0xBF86_0003, // s_cbranch_vccz -> end
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // s_cbranch_scc1 -> end
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        lowers(&words, group_ids_xy).expect("the timestamp is never read");
        // With the flag set on both paths the timestamp is read: as refused
        // (or accepted) as a direct read.
        words[2] = 0x7E02_02C1; // v_mov_b32 v1, -1
        let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[9..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
    }

    #[test]
    fn a_mask_joined_with_data_needs_a_uniform_join() {
        // s6: lanes below 24, then shifted (data) on each back edge; read
        // after the loop. Lanes leaving the loop at different iterations
        // (per lane) see different numbers than the console's register:
        // refused. The uniform loop is `synthetic_tests::
        // a_mask_joined_with_data_at_a_uniform_loop_is_its_number`.
        let mut words = [
            0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
            0x8706_7E6A, // s_and_b32 s6, vcc_lo, exec_lo
            0xBE87_0380, // s_mov_b32 s7, 0
            0x9006_8406, // header: s_lshr_b32 s6, s6, 4
            0x8007_8107, // s_add_u32 s7, s7, 1
            0x7D88_0007, // v_cmp_gt_u32 vcc_lo, s7, v0
            0x8A7E_6A7E, // s_andn2_b32 exec_lo, exec_lo, vcc_lo
            0xBF89_FFFB, // s_cbranch_execnz header
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // s_cbranch_scc1 +1
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let Err(crate::ShaderError::Unsupported { what, .. }) = lowers(&words, 1 << 7) else {
            panic!("a per-lane loop's mask join translated");
        };
        assert!(what.contains("lane mask"), "{what}");
        // Leaving on a uniform condition instead: the mask's number.
        words[5] = 0xBF0A_8207; // s_cmp_lt_u32 s7, 2
        words[6] = 0xBF80_0000; // s_nop 0
        words[7] = 0xBF85_FFFB; // s_cbranch_scc1 header
        lowers(&words, 1 << 7).expect("a uniform loop's mask join");
    }

    /// s6 holds a constant where lanes 16 and up skip the arm, which is `arm`; in a uniform
    /// loop's body if `looped`.
    fn skip_edge_constant(constant: u32, arm: [u32; 2], looped: bool) -> Vec<u32> {
        let mut words = vec![
            0x7E10_0280, // v_mov_b32 v8, 0
            0x7E12_0280, // v_mov_b32 v9, 0
            0xBE87_0380, // s_mov_b32 s7, 0
            constant,    // loop: s_mov_b32 s6, 5 (or 0)
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
            0xBE88_3C6A, // s_and_saveexec_b32 s8, vcc_lo
            0xBF88_0002, // s_cbranch_execz +2
            arm[0],
            arm[1],
            0xBEFE_0308, // s_mov_b32 exec_lo, s8
            0xD501_0003,
            0x0019_0280, // v_cndmask_b32 v3, 0, 1, s6
        ];
        if looped {
            words.extend([
                0x8007_8107, // s_add_u32 s7, s7, 1
                0xBF0A_8207, // s_cmp_lt_u32 s7, 2
                0xBF85_FFF4, // s_cbranch_scc1 loop
            ]);
        }
        words.extend([
            0xDC70_8000,
            0x007D_0308, // global_store_dword v[8:9], v3
            S_ENDPGM,
        ]);
        words
    }

    /// `s_mov_b32 s6, s8; s_nop 0`: the saved EXEC, which holds the skipping lanes.
    const SAVED_EXEC: [u32; 2] = [0xBE86_0308, 0xBF80_0000];

    #[test]
    fn a_skip_edge_constant_is_not_the_bit_where_the_arm_keeps_the_skipping_lanes() {
        // With a compare in the arm the skipping lanes read 0, their bit of it. With the saved
        // EXEC there the console's register holds their bit set, which the constant 5 does not
        // say: refused, in a uniform loop's body too.
        const COMPARE: [u32; 2] = [0xD4C4_0006, 0x0002_0088]; // v_cmp_gt_u32 s6, 8, v0
        for looped in [false, true] {
            lowers(&skip_edge_constant(0xBE86_0385, COMPARE, looped), 1 << 7)
                .unwrap_or_else(|e| panic!("a compare the skipping lanes are not in: {e}"));
            let Err(crate::ShaderError::Unsupported { what, .. }) =
                lowers(&skip_edge_constant(0xBE86_0385, SAVED_EXEC, looped), 1 << 7)
            else {
                panic!("5 joined with the saved EXEC translated (loop: {looped})");
            };
            assert!(what.contains("lane-mask join"), "{what}");
        }
    }

    #[test]
    fn a_then_parts_constant_on_the_else_skip_edge_is_weighed_for_its_own_lanes() {
        // An if's then part (lanes below 8) writes s6, and the else part writes a mask there.
        // The edge that skips the else part brings a join of the then part's value and what s6
        // held before, of which the then lanes hold the first. With -1, and an else part that
        // sets the bits of the lanes outside it (`s_orn2` by EXEC), they read 1 either way. With
        // 5, and the saved EXEC in the else part, they read 1 on the console and their bit of 5
        // from the join: refused.
        let program = |then: u32, other: [u32; 3]| {
            [
                0x7E10_0280, // v_mov_b32 v8, 0
                0x7E12_0280, // v_mov_b32 v9, 0
                0xBE86_0385, // s_mov_b32 s6, 5
                0xBE89_037E, // s_mov_b32 s9, exec_lo
                0x7DA8_0088, // v_cmpx_gt_u32 8, v0
                0xBF88_0001, // s_cbranch_execz else
                then,        // s_mov_b32 s6, -1 (or 5)
                0x8A7E_7E09, // else: s_andn2_b32 exec_lo, s9, exec_lo
                0xBF88_0003, // s_cbranch_execz end
                other[0],
                other[1],
                other[2],
                0xBEFE_0309, // end: s_mov_b32 exec_lo, s9
                0xD501_0003,
                0x0019_0280, // v_cndmask_b32 v3, 0, 1, s6
                0xDC70_8000,
                0x007D_0308, // global_store_dword v[8:9], v3
                S_ENDPGM,
            ]
        };
        // v_cmp_gt_u32 s6, 12, v0; s_orn2_b32 s6, s6, exec_lo
        const OTHERS_SET: [u32; 3] = [0xD4C4_0006, 0x0002_008C, 0x8B06_7E06];
        // s_mov_b32 s6, s9; s_nop 0; s_nop 0
        const SAVED: [u32; 3] = [0xBE86_0309, 0xBF80_0000, 0xBF80_0000];
        lowers(&program(0xBE86_03C1, OTHERS_SET), 1 << 7)
            .unwrap_or_else(|e| panic!("-1 for lanes whose bits the else part sets: {e}"));
        let Err(crate::ShaderError::Unsupported { what, .. }) =
            lowers(&program(0xBE86_0385, SAVED), 1 << 7)
        else {
            panic!("5 joined with the saved EXEC translated");
        };
        assert!(what.contains("lane-mask join"), "{what}");
    }

    #[test]
    #[ignore = "known gap: on the host's lanes a constant 0 or -1 is taken as every lane's bit unchecked"]
    fn a_skip_edge_constant_of_no_lane_is_not_the_bit_either() {
        let Err(crate::ShaderError::Unsupported { what, .. }) =
            lowers(&skip_edge_constant(0xBE86_0380, SAVED_EXEC, false), 1 << 7)
        else {
            panic!("0 joined with the saved EXEC translated");
        };
        assert!(what.contains("lane-mask join"), "{what}");
    }

    #[test]
    fn a_scalar_result_needs_no_exec() {
        // s6 is 0 where every lane skips and the saved EXEC where some run:
        // no lane mask at the per-lane join. EXEC taken from it guards only
        // a `v_readlane` (written whatever EXEC holds) before it is restored
        // (`CS_IrradianceCache_Shade`).
        let words = [
            0xBE86_0380, // s_mov_b32 s6, 0
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
            0xBE84_3C6A, // s_and_saveexec_b32 s4, vcc_lo
            0xBF88_0001, // s_cbranch_execz +1
            0xBE86_0304, // s_mov_b32 s6, s4
            0xBEFE_0306, // s_mov_b32 exec_lo, s6
            0xD760_0002, // v_readlane_b32 s2, v0, 3
            0x0001_0700,
            0xBEFE_0304, // s_mov_b32 exec_lo, s4
            S_ENDPGM,
        ];
        lowers(&words, 1 << 7).expect("a lane read under an EXEC with no mask");
    }

    #[test]
    fn a_missing_value_on_a_path_no_lane_takes_is_dead() {
        let refusal = |code: &[u32]| match lowers(code, 1 << 7) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        // The timestamp stays in s6 only where s0 == 0, and those lanes
        // overwrite it before the second join: it never reaches the read.
        let mut joins = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // s_cbranch_scc1: s0 == 0 keeps the timestamp
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF84_0001, // s_cbranch_scc0: s0 != 0 keeps that 0
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001,
            0xBE83_0380,
            S_ENDPGM,
        ];
        assert_eq!(refusal(&joins), None);
        // Tested on the other side, the timestamp reaches the read.
        joins[5] = 0xBF85_0001; // s_cbranch_scc1
        assert!(refusal(&joins).is_some());
        // The same through a VGPR whose old value a guarded write keeps
        // (lanes outside it get 0 for a value with no DXIL form, as for any
        // such write): keeping does not make the join live.
        let kept = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0x7E02_0206, // v_mov_b32 v1, s6
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0001, // s_cbranch_scc1: s0 == 0 keeps the timestamp
            0x7E02_0280, // v_mov_b32 v1, 0
            0xBF84_0004, // s_cbranch_scc0: s0 != 0 keeps that 0
            0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
            0x7E02_0282, // v_mov_b32 v1, 2 (the other lanes keep v1)
            0xBEFE_03C1, // s_mov_b32 exec_lo, -1
            0x7E02_0283, // v_mov_b32 v1, 3
            0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
            0xBF87_0001, // s_cbranch_vccnz
            0xBE83_0380,
            S_ENDPGM,
        ];
        assert_eq!(refusal(&kept), None);
    }

    #[test]
    fn liveness_through_joins_of_many_conditions_ends() {
        // Six joins in a row, each reached by twelve branches on bits of the group id; s6 passes
        // every one to the read at the end, so the deeper joins' phis were walked back once per
        // set of the last four conditions on the way, 12^4 sets each, every new one compared with
        // all the ones before it (minutes; a title pixel shader never finished). Past
        // `MAX_PATHS` sets a phi is walked knowing nothing.
        const K: u32 = 12;
        const D: u32 = 6;
        let mut words = vec![0xBE86_0380]; // s_mov_b32 s6, 0
        for j in 0..D {
            for i in 0..K {
                let bit = j * K + i;
                // s_bitcmp1_b32 s0, s1 or s2 (group id X, Y, Z), its bit `bit % 32`
                words.push(0xBF0D_0000 | ((0x80 + bit % 32) << 8) | (bit / 32));
                words.push(0xBF85_0000 | (2 * (K - 1 - i) + 1)); // s_cbranch_scc1 to the join
            }
            words.push(0xBE86_0380 | (j + 1)); // s_mov_b32 s6, j + 1 (the fall-through)
        }
        words.extend([
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // s_cbranch_scc1
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ]);
        let start = std::time::Instant::now();
        lowers(&words, 1 << 7 | 1 << 8 | 1 << 9).expect("lowers");
        assert!(start.elapsed() < std::time::Duration::from_secs(10), "{:?}", start.elapsed());
    }

    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(
            buffer_format(50),
            Some((&[10, 10, 10, 2][..], Numeric::Unorm))
        );
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(
            buffer_format(77),
            Some((&[32, 32, 32, 32][..], Numeric::Float))
        );
        // Invalid, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(78), None);
    }

    #[test]
    fn packed_buffer_formats_by_number() {
        // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
        // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
        // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
        assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
        assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
        assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
        assert_eq!(
            buffer_format(44),
            Some((&[2, 10, 10, 10][..], Numeric::Unorm))
        );
        assert_eq!(
            buffer_format(45),
            Some((&[2, 10, 10, 10][..], Numeric::Snorm))
        );
        assert_eq!(
            buffer_format(46),
            Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
        );
        assert_eq!(
            buffer_format(47),
            Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
        );
        assert_eq!(
            buffer_format(48),
            Some((&[2, 10, 10, 10][..], Numeric::Uint))
        );
        assert_eq!(
            buffer_format(49),
            Some((&[2, 10, 10, 10][..], Numeric::Sint))
        );
    }

    /// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
    /// `legacyF16ToF32` widens it.
    fn half(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 if m == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn small_floats_widen_as_halves() {
        // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
        // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
        let f11 = |x: u32| half(x << half_shift(11));
        let f10 = |x: u32| half(x << half_shift(10));
        assert_eq!(f11(0), 0.0);
        assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
        assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
        assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
        assert_eq!(f11(0x3C0), 1.0);
        assert_eq!(f11(0x3E0), 1.5);
        assert_eq!(f11(0x7BF), 65024.0); // largest
        assert_eq!(f11(0x7C0), f32::INFINITY);
        assert!(f11(0x7C1).is_nan());
        assert_eq!(f10(0x001), 2f32.powi(-19));
        assert_eq!(f10(0x020), 2f32.powi(-14));
        assert_eq!(f10(0x1E0), 1.0);
        assert_eq!(f10(0x3DF), 64512.0);
        assert_eq!(f10(0x3E0), f32::INFINITY);
        assert!(f10(0x3FF).is_nan());
        // A 16-bit float is a half already, sign included.
        assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
    }

    #[test]
    fn packed_halves_fill_parameters_first_fit() {
        // 16-bit rows as a title's signature has them: row 0 holds two
        // elements (x y and z w), row 3 is 32-bit, row 4 has one channel at w.
        let el = |row: u32, col: u8, cols: u8, half: bool| super::SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(row),
            col,
            cols,
            usage: 0,
            half,
        };
        let sig = [
            el(0, 0, 2, true),
            el(0, 2, 2, true),
            el(1, 0, 4, true),
            el(2, 0, 3, true),
            el(3, 0, 4, false),
            el(4, 0, 3, true),
            el(4, 3, 1, true),
            el(5, 0, 3, true),
        ];
        // Each half as row and channel ("-": padding or past the end).
        let param = |a| {
            (0..8)
                .map(|h| match packed_half(&sig, a, h) {
                    Some((r, c)) => format!("{r}{}", ["x", "y", "z", "w"][usize::from(c)]),
                    None => "-".into(),
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        // By first channel, then row: row 0's x y, all of row 1, then (last,
        // as it starts at z) row 0's z w.
        assert_eq!(param(0), "0x 0y 1x 1y 1z 1w 0z 0w");
        // Three channels take two dwords; row 4's w fills the first
        // parameter with room, after row 5.
        assert_eq!(param(2), "2x 2y 2z - 4x 4y 4z -");
        assert_eq!(param(5), "5x 5y 5z - 4w - - -");
        // No parameter starts in rows 1, 3 or 4.
        for a in [1, 3, 4] {
            assert_eq!(param(a), "- - - - - - - -");
        }
    }

    // ---- one console wave split across host waves ----

    const V_LSHLREV_V1_2_V0: u32 = 0x3402_0082; // v_lshlrev_b32 v1, 2, v0
    const DS_WRITE_V1_V0: [u32; 2] = [0xD834_0000, 0x0000_0001]; // ds_write_b32 v1, v0
    const DS_READ_V2_V1_4: [u32; 2] = [0xD8D8_0004, 0x0200_0001]; // ds_read_b32 v2, v1 offset:4
    const SAVE_EXEC: u32 = 0xBE80_047E; // s_mov_b64 s[0:1], exec
    const CMPX_V0_EQ_0: u32 = 0x7DA4_0080; // v_cmpx_eq_u32 exec, 0, v0
    const RESTORE_EXEC: u32 = 0xBEFE_0400; // s_mov_b64 exec, s[0:1]

    /// `s_cbranch_execz` over the next `words` words.
    const fn skip(words: u32) -> u32 {
        0xBF88_0000 | words
    }

    fn wave64(words: &[u32]) -> ir::Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        ir::build(&code, true, 256).unwrap()
    }

    /// Lowers a wave64 compute program of `lanes` threads with 512 bytes of
    /// LDS: the barriers it got for narrow host waves.
    fn split_barriers(words: &[u32], lanes: u32, narrow_waves: bool) -> usize {
        lowered(words, lanes, narrow_waves, false).split_barriers
    }

    /// The same program, lowered.
    fn lowered(words: &[u32], lanes: u32, narrow_waves: bool, wave32_host: bool) -> super::Lowered {
        lowered_for(words, lanes, narrow_waves, wave32_host, false)
    }

    /// [`lowered`], for dispatches of thread counts or not.
    fn lowered_for(
        words: &[u32],
        lanes: u32,
        narrow_waves: bool,
        wave32_host: bool,
        thread_counts: bool,
    ) -> super::Lowered {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15 | 2 << 1,
                threads: [lanes, 1, 1],
            }),
            code,
        };
        let mut p = ir::build(&native.code, true, 256).unwrap();
        p.thread_counts = thread_counts;
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
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
            narrow_waves,
            wave32_host,
            zero_copy_memory: false,
            rect_list: false,
            copy_params: [0; 32],
            view_marks: false,
        })
        .expect("lower")
    }

    #[test]
    fn host32_simulates_each_whole_wave_section_once() {
        let words = [
            0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
            0x7E02_0280, // v_mov_b32 v1, 0
            RESTORE_EXEC,
            S_ENDPGM,
        ];
        let simulations = &crate::emit::wave::SIMULATIONS;
        let before = simulations.get();
        let shader = lowered(&words, 64, false, true);
        assert_eq!(shader.wave_size, Some(32));
        assert_eq!(simulations.get() - before, 1);
    }

    #[test]
    fn a_program_for_thread_counts_reads_the_count_of_each_dimension_with_lanes() {
        // `v_mov_b32 v1, 0`: nothing reads user data, but the lanes' EXEC on
        // entry comes from the dispatch's thread count in x, which the host
        // passes in the first dword past the root signature's (none here).
        let w = [0x7E02_0280, S_ENDPGM];
        let count = |dword| super::UserData {
            index: dword,
            slot: crate::emit::DISPATCH_THREADS,
            root: RootType::Constants,
            dword,
        };
        for split in [false, true] {
            assert_eq!(lowered_for(&w, 64, false, split, true).user_data, [count(0)]);
            assert_eq!(lowered_for(&w, 64, false, split, false).user_data, []);
        }
        // A group of one thread has no partial group: nothing to read.
        assert_eq!(lowered_for(&w, 1, false, true, true).user_data, []);
    }

    #[test]
    fn a_read_after_another_lanes_write_syncs_in_a_split_wave() {
        let mut w = vec![V_LSHLREV_V1_2_V0];
        w.extend(DS_WRITE_V1_V0);
        w.extend(DS_READ_V2_V1_4);
        w.push(S_ENDPGM);
        let at = super::split_wave_barriers(&wave64(&w), &structure::structure(&wave64(&w)).unwrap()).unwrap();
        assert_eq!(at.into_iter().collect::<Vec<_>>(), [(0, 2)]);
        assert_eq!(split_barriers(&w, 64, true), 1);
        // Option off, or a group one host wave holds: as before.
        assert_eq!(split_barriers(&w, 64, false), 0);
        assert_eq!(split_barriers(&w, 32, true), 0);
        // Synced, it no longer asks for 64-lane waves a 32-lane host lacks
        // (D3DMetal runs nothing then); without the option it still does.
        assert_eq!(lowered(&w, 64, true, false).wave_size, None);
        assert_eq!(lowered(&w, 64, false, false).wave_size, Some(64));
        assert_eq!(lowered(&w, 32, true, false).wave_size, None);
    }

    #[test]
    fn a_divergent_write_syncs_at_the_join_before_the_read() {
        // 34d88edf583f3778's shape: lanes with v0 = 0 write, all read.
        let mut w = vec![V_LSHLREV_V1_2_V0, SAVE_EXEC, CMPX_V0_EQ_0, skip(2)];
        w.extend(DS_WRITE_V1_V0);
        w.push(RESTORE_EXEC);
        w.extend(DS_READ_V2_V1_4);
        w.push(S_ENDPGM);
        let p = wave64(&w);
        let at: Vec<_> = super::split_wave_barriers(&p, &structure::structure(&p).unwrap())
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(at.len(), 1);
        let (b, i) = at[0];
        assert!(p.blocks[b].insts[i].name().starts_with("ds_read"));
        assert_eq!(p.blocks[b].preds.len(), 2, "the join");
        assert_eq!(split_barriers(&w, 64, true), 1);
    }

    #[test]
    fn a_write_before_divergent_reads_syncs_where_all_lanes_run() {
        let mut w = vec![V_LSHLREV_V1_2_V0];
        w.extend(DS_WRITE_V1_V0);
        w.extend([SAVE_EXEC, CMPX_V0_EQ_0, skip(2)]);
        w.extend(DS_READ_V2_V1_4);
        w.push(RESTORE_EXEC);
        w.push(S_ENDPGM);
        let p = wave64(&w);
        let at: Vec<_> = super::split_wave_barriers(&p, &structure::structure(&p).unwrap())
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(at, [(0, p.blocks[0].insts.len())], "the end of the entry");
        assert_eq!(split_barriers(&w, 64, true), 1);
    }

    #[test]
    fn no_barrier_in_divergent_code() {
        // Write and read both under the branch: no barrier can go there.
        let mut w = vec![V_LSHLREV_V1_2_V0, SAVE_EXEC, CMPX_V0_EQ_0, skip(4)];
        w.extend(DS_WRITE_V1_V0);
        w.extend(DS_READ_V2_V1_4);
        w.push(RESTORE_EXEC);
        w.push(S_ENDPGM);
        assert!(super::split_wave_barriers(&wave64(&w), &structure::structure(&wave64(&w)).unwrap()).is_none());
        assert_eq!(split_barriers(&w, 64, true), 0);
    }

    #[test]
    fn dpp_inactive_source_modifiers_change_the_lowered_shader() {
        let shader = |selector: u32, extension: u32| {
            let words = [
                V_LSHLREV_V1_2_V0,
                0x4A06_0086, // v_add_nc_u32 v3, 6, v0
                0x7E04_02FF, 99, // v_mov_b32 v2, 99
                SAVE_EXEC,
                CMPX_V0_EQ_0,
                0x7E04_0200 | selector, extension | 3, // v_mov_b32_dpp v2, v3
                RESTORE_EXEC,
                0xD834_0000, 0x0000_0201, // ds_write_b32 v1, v2
                S_ENDPGM,
            ];
            let shader = lowered(&words, 64, false, true);
            assert!(!shader.exchanges);
            shader.dxil
        };
        // Lane 0 selects lane 1 after EXEC has disabled it.
        assert!(shader(0xE9, 0x100) != shader(0xEA, 0x100));
        assert!(shader(0xFA, 0xFF00_0100) != shader(0xFA, 0xFF08_0100));
    }

    #[test]
    fn a_whole_wave_lane_read_uses_the_original_source_lane() {
        // v_mov_b32_dpp v2, v0 quad_perm:[1,0,3,2]
        let swap_neighbours = [0x7E04_02FA, 0xFF00_B100];
        for (shuffle, vsrc, lane) in [(&[][..], 0, 35), (&swap_neighbours[..], 2, 34)] {
            let mut words = vec![
                V_LSHLREV_V1_2_V0,
                0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
            ];
            words.extend_from_slice(shuffle);
            words.extend([
                0xD760_0002, // v_readlane_b32 s2, vsrc, 35
                0x0001_4700 | vsrc,
                RESTORE_EXEC,
                0x7E00_0202, // v_mov_b32 v0, s2
            ]);
            words.extend(DS_WRITE_V1_V0);
            words.push(S_ENDPGM);

            let normal = lowered(&words, 64, false, false);
            assert_eq!(
                normal.wave_read_lane_calls,
                [(
                    "dx.op.waveReadLaneAt.i32".into(),
                    vec![Some(117), None, Some(lane)]
                )]
            );

            let split = lowered(&words, 64, false, true);
            assert!(split.exchanges);
            assert_eq!(split.barriers, 2);
            assert!(split.wave_read_lane_calls.is_empty());
        }
    }

    #[test]
    fn float_wave_prefix_sum_passes_signed_zero_flag() {
        let sop1 = |sdst: u8, op: u8, ssrc0: u8| -> u32 {
            0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | ssrc0 as u32
        };
        let vop2 = |vdst: u8, op: u8, src0: u16, vsrc1: u8| -> u32 {
            (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
        };
        let dpp16 = |vsrc0: u8, ctrl: u16| -> u32 {
            (0xFu32 << 28) | (0xFu32 << 24) | (ctrl as u32) << 8 | vsrc0 as u32
        };
        let vop3 = |vdst: u8, op: u16, src0: u16, src1: u16, src2: u16| -> [u32; 2] {
            [
                0xD400_0000 | (op as u32) << 16 | vdst as u32,
                (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
            ]
        };

        // Wave32 inclusive prefix sum over v0, storing the result to LDS so
        // the `dx.op.wavePrefixOp.*` call stays live.
        let prefix_words = |add_vop2: u8| -> Vec<u32> {
            let mut w = vec![sop1(0, 0x3D, 193)]; // s_or_saveexec_b32 s0, -1
            w.extend(vop3(1, 0x101, 128, 256, 0)); // v_cndmask_b32 v1, 0, v0, s0
            for shift in [1u16, 2, 4, 8] {
                w.push(vop2(1, add_vop2, 0xFA, 1));
                w.push(dpp16(1, 0x110 + shift));
            }
            w.extend(vop3(2, 0x360, 257, 128 + 15, 0)); // v_readlane_b32 s2, v1, 15
            w.push(sop1(126, 0x03, 255)); // s_mov_b32 exec_lo, 0xFFFF0000
            w.push(0xFFFF_0000);
            w.push(vop2(1, add_vop2, 2, 1)); // add s2 into upper 16 lanes
            w.push(sop1(126, 0x03, 0)); // s_mov_b32 exec_lo, s0
            w.push(0x3404_0082); // v_lshlrev_b32 v2, 2, v0
            w.extend([0xD834_0000, 0x0000_0102]); // ds_write_b32 v2, v1
            w.push(S_ENDPGM);
            w
        };

        let lower_cs = |words: &[u32]| {
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 8,
                layout: vec![],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 1 << 15, // 512 bytes LDS
                    threads: [32, 1, 1],
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = crate::ir::build(&native.code, false, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            super::lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                view_marks: false,
                atomic_locks: false,
                bias_as_gradients: false,
                narrow_waves: false,
                wave32_host: true,
                zero_copy_memory: false,
                rect_list: false,
                copy_params: [0; 32],
            })
            .unwrap()
        };

        // `v_add_f32` (VOP2 0x03): floating-point prefix sum passes unsigned=0.
        let float_l = lower_cs(&prefix_words(0x03));
        assert_eq!(
            float_l.wave_prefix_calls,
            vec![(
                "dx.op.wavePrefixOp.f32".to_string(),
                vec![Some(121), None, Some(0), Some(0)]
            )]
        );

        // `v_add_nc_u32` (VOP2 0x25): unsigned integer prefix sum passes unsigned=1.
        let uint_l = lower_cs(&prefix_words(0x25));
        assert_eq!(
            uint_l.wave_prefix_calls,
            vec![(
                "dx.op.wavePrefixOp.i32".to_string(),
                vec![Some(121), None, Some(0), Some(1)]
            )]
        );
    }

    #[test]
    fn unneeded_edges_propagates_across_deep_sgpr_phi_chains_without_quadratic_scans() {
        // 160 sequential per-lane diamonds carrying 32 live SGPRs (`s10..s41`) as SSA phis at
        // every merge block before reading them at the end, plus one genuine `unneeded_edges`
        // mask join (`s6`) read only ANDed with the taken arm's EXEC.
        let mut w = Vec::new();
        for r in 10..42u32 {
            w.push(0xBE80_0300 | (r << 16) | 0x81); // s_mov_b32 sr, 1
        }
        w.push(0xBE86_0300); // s_mov_b32 s6, s0 (data on skip edge)
        w.push(0x7D88_0090); // v_cmp_gt_u32 vcc_lo, 16, v0
        w.push(0xBE88_3C6A); // s_and_saveexec_b32 s8, vcc_lo
        w.push(0xBF88_0002); // s_cbranch_execz +2
        w.extend([0xD4C4_0006, 0x0002_0088]); // v_cmp_gt_u32 s6, 8, v0
        w.push(0x8706_7E06); // join: s_and_b32 s6, s6, exec_lo
        w.push(0xBEFE_0308); // s_mov_b32 exec_lo, s8

        for k in 0..160u32 {
            let s = 10 + (k % 32);
            w.push(0x7D88_0084); // v_cmp_gt_u32 vcc_lo, 4, v0
            w.push(0xBE88_3C6A); // s_and_saveexec_b32 s8, vcc_lo
            w.push(0xBF88_0001); // s_cbranch_execz +1
            w.push(0x8000_8100 | (s << 16) | s); // s_add_u32 ss, ss, 1
            w.push(0xBEFE_0308); // s_mov_b32 exec_lo, s8
        }

        w.extend([0xD501_0001, 0x0019_0280]); // v_cndmask_b32 v1, 0, 1, s6
        for r in 10..42u32 {
            w.push(0x4A02_0200 | r); // v_add_nc_u32 v1, sr, v1
        }
        w.push(0x3404_0082); // v_lshlrev_b32 v2, 2, v0
        w.extend([0xD834_0000, 0x0000_0102]); // ds_write_b32 v2, v1
        w.push(S_ENDPGM);

        let started = std::time::Instant::now();
        lowers(&w, 1 << 15).unwrap();
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "deep SGPR phi chain took {:?}",
            started.elapsed()
        );
    }
}
