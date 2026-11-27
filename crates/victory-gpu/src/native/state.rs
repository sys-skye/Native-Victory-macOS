//! A title's graphics pipeline state, read from its pipeline-state stream (the subobjects of
//! `CreatePipelineState`: blend, rasterizer, depth and stencil, input layout, target formats) and
//! turned into what the backend's render pipeline takes. The same reading as
//! `pipeline/host.rs`'s `GraphicsState::read` (the D3D12 host keeps the raw D3D12 structures), in
//! plain Rust, so it needs no host API. Subobjects the stream leaves out keep D3D12's defaults.

use super::descriptors::format;
use victory_backend::resources::*;

/// A `CB_BLEND<n>_CONTROL` register as the D3D12 description of a target (all channels written);
/// `None` for a factor D3D12 has no value for (`BOTH_SRC_ALPHA`, `BOTH_INV_SRC_ALPHA`).
fn cb_blend(r: u32) -> Option<TargetBlend> {
    if r & 1 << 30 == 0 {
        return Some(TargetBlend::default());
    }
    // The hardware's factors are D3D12's less one; its 11 and 12 (both-alpha factors) have none.
    let factor = |f: u32| match f {
        0..=10 => Some(f + 1),
        13..=20 => Some(f + 1),
        _ => None,
    };
    // Hardware: add, source minus destination, min, max, destination minus source.
    let op = |o: u32| [1, 2, 4, 5, 3].get(o as usize).copied();
    let (src, comb, dst) = (r & 0x1f, r >> 5 & 7, r >> 8 & 0x1f);
    let (src_a, comb_a, dst_a) = if r & 1 << 29 != 0 { (r >> 16 & 0x1f, r >> 21 & 7, r >> 24 & 0x1f) } else { (src, comb, dst) };
    Some(TargetBlend {
        enable: true,
        src: factor(src)?,
        dst: factor(dst)?,
        op: op(comb)?,
        src_alpha: factor(src_a)?,
        dst_alpha: factor(dst_a)?,
        op_alpha: op(comb_a)?,
        mask: 0xf,
    })
}

/// A `DB_STENCIL_CONTROL` operation as D3D12's (keep, zero, replace, increment and decrement
/// clamped, invert, increment and decrement wrapped); KEEP for those D3D12 has none of.
fn stencil_op_hw(op: u32) -> u32 {
    match op {
        1 => 2,
        3 | 4 => 3,
        5 => 4,
        6 => 5,
        7 => 6,
        8 => 7,
        9 => 8,
        _ => 1,
    }
}

/// One colour target of the blend description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TargetBlend {
    enable: bool,
    src: u32,
    dst: u32,
    op: u32,
    src_alpha: u32,
    dst_alpha: u32,
    op_alpha: u32,
    mask: u8,
}

impl Default for TargetBlend {
    fn default() -> Self {
        TargetBlend {
            enable: false,
            src: 2,
            dst: 1,
            op: 1,
            src_alpha: 2,
            dst_alpha: 1,
            op_alpha: 1,
            mask: 0xf,
        }
    }
}

/// An input element: its slot, byte offset (`APPEND_ALIGNED` is resolved), DXGI format, semantic,
/// and whether it steps per instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    pub name: String,
    pub index: u32,
    pub format: u32,
    pub slot: u32,
    pub offset: u32,
    pub per_instance: bool,
}

const APPEND_ALIGNED: u32 = u32::MAX;

#[derive(Clone, Debug, PartialEq)]
pub struct Graphics {
    independent: bool,
    blend: [TargetBlend; 8],
    fill: u32,
    cull: u32,
    front_ccw: bool,
    bias: (i32, f32, f32),
    depth_clip: bool,
    depth_enable: bool,
    depth_write: bool,
    depth_func: u32,
    stencil_enable: bool,
    stencil_masks: (u8, u8),
    /// Front then back: fail, depth fail, pass, compare.
    stencil_faces: [[u32; 4]; 2],
    inputs: Vec<Input>,
    pub topology_type: u32,
    rt_formats: Vec<u32>,
    depth_format: u32,
    pub samples: u32,
    /// The formats of the targets bound at the draw, for pipelines whose description has none
    /// (the console's; see [`Graphics::with_targets`]).
    bound: Option<(Vec<Format>, Option<Format>)>,
    /// A guessed pipeline (`VICTORY_GUESS_GRAPHICS`, [`Graphics::guess_base`]): its vertex layout
    /// and strides come from the draw ([`Graphics::for_draw`]).
    guess: Option<Guess>,
    /// A described pipeline's state for rectangle-list draws ([`Graphics::for_rects`]): each
    /// slot's vertex stride, a stream stepping three vertices an instance.
    rects: Option<Vec<u32>>,
}

/// What a draw gives a guessed pipeline that the description does not.
#[derive(Clone, Debug, PartialEq, Default)]
struct Guess {
    strides: Vec<u32>,
    /// A rectangle-list draw: every attribute is read for three corners, one instance a rectangle.
    rect: bool,
}

impl Default for Graphics {
    fn default() -> Self {
        Graphics {
            independent: false,
            blend: [TargetBlend::default(); 8],
            fill: 3,
            cull: 3,
            front_ccw: false,
            bias: (0, 0.0, 0.0),
            depth_clip: true,
            depth_enable: true,
            depth_write: true,
            depth_func: 2,
            stencil_enable: false,
            stencil_masks: (0xff, 0xff),
            stencil_faces: [[1, 1, 1, 8]; 2],
            inputs: Vec::new(),
            topology_type: 0,
            rt_formats: Vec::new(),
            depth_format: 0,
            samples: 1,
            bound: None,
            guess: None,
            rects: None,
        }
    }
}

fn word(at: *const u8, byte: usize) -> u32 {
    // SAFETY: the caller passes a whole subobject, whose layout names `byte`.
    unsafe { at.add(byte).cast::<u32>().read_unaligned() }
}

impl Graphics {
    /// Takes in one subobject of a stream: its type, then its data at +4 (+8 for those holding
    /// pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`, whose pointers (input elements, their
    /// names) are valid.
    pub unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject; offsets follow the D3D12 structures.
        unsafe {
            let data = at.add(4);
            match kind {
                // D3D12_BLEND_DESC: alpha to coverage, independent blend, 8 targets of 10 words.
                8 => {
                    self.independent = word(data, 4) != 0;
                    for (i, t) in self.blend.iter_mut().enumerate() {
                        let w = |k: usize| word(data, 8 + i * 40 + k * 4);
                        *t = TargetBlend {
                            enable: w(0) != 0,
                            src: w(2),
                            dst: w(3),
                            op: w(4),
                            src_alpha: w(5),
                            dst_alpha: w(6),
                            op_alpha: w(7),
                            mask: w(9) as u8,
                        };
                    }
                }
                // D3D12_RASTERIZER_DESC.
                10 => {
                    self.fill = word(data, 0);
                    self.cull = word(data, 4);
                    self.front_ccw = word(data, 8) != 0;
                    self.bias = (
                        word(data, 12) as i32,
                        f32::from_bits(word(data, 16)),
                        f32::from_bits(word(data, 20)),
                    );
                    self.depth_clip = word(data, 24) != 0;
                }
                // D3D12_DEPTH_STENCIL_DESC, and its DESC1 (the same up to the depth bounds).
                11 | 21 => {
                    self.depth_enable = word(data, 0) != 0;
                    self.depth_write = word(data, 4) != 0;
                    self.depth_func = word(data, 8);
                    self.stencil_enable = word(data, 12) != 0;
                    let masks = word(data, 16);
                    self.stencil_masks = (masks as u8, (masks >> 8) as u8);
                    for (face, base) in [20usize, 36].into_iter().enumerate() {
                        self.stencil_faces[face] = [
                            word(data, base),
                            word(data, base + 4),
                            word(data, base + 8),
                            word(data, base + 12),
                        ];
                    }
                }
                12 => {
                    let ptr = at.add(8).cast::<*const u8>().read_unaligned();
                    let count = word(at, 16) as usize;
                    self.inputs.clear();
                    if ptr.is_null() {
                        return;
                    }
                    let mut next = [0u32; 32];
                    for i in 0..count {
                        let e = ptr.add(i * 32);
                        let name = e.cast::<*const std::ffi::c_char>().read_unaligned();
                        if name.is_null() {
                            continue;
                        }
                        let slot = word(e, 16).min(31);
                        let format = crate::resource::pc_format(word(e, 12));
                        let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                        // An appended element starts at the next multiple of min(4, its size).
                        let offset = match word(e, 20) {
                            APPEND_ALIGNED => next[slot as usize]
                                .checked_next_multiple_of(bytes.clamp(1, 4))
                                .unwrap_or(u32::MAX),
                            o => o,
                        };
                        next[slot as usize] = offset.saturating_add(bytes);
                        self.inputs.push(Input {
                            name: std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned(),
                            index: word(e, 8),
                            format,
                            slot,
                            offset,
                            per_instance: word(e, 24) == 1,
                        });
                    }
                }
                14 => self.topology_type = word(data, 0),
                15 => {
                    let count = (word(data, 32) as usize).min(8);
                    self.rt_formats = (0..count)
                        .map(|i| crate::resource::pc_format(word(data, i * 4)))
                        .collect();
                }
                16 => self.depth_format = crate::resource::pc_format(word(data, 0)),
                17 => self.samples = word(data, 0).max(1),
                _ => {}
            }
        }
    }

    /// The state of a pipeline made by device slot 0x1c8, whose description has none
    /// (docs/native-host.md, "Guessed graphics pipelines"): no blend, no culling, depth test
    /// less-equal with writes. Variants made by slot 0x1b8 apply their records over it.
    pub fn guess_base() -> Graphics {
        Graphics { cull: 1, depth_func: 4, guess: Some(Guess::default()), ..Graphics::default() }
    }

    /// [`Graphics::guess_base`] with the blend state of the 0x1c8 register image `image` (pair 0
    /// of the description's slot), where it is found.
    ///
    /// Decoded from 23,000 images of one title: word 1 is 4 or 6 (sections that come before), and
    /// the eight `CB_BLEND<n>_CONTROL` registers start at byte 0xd0 or 0x124 respectively, with
    /// DISABLE_ROP3 (bit 31) set in each and 0xffffffff after them (then 3 words and the eight
    /// `SX_MRT<n>_BLEND_OPT`). shortcut: any other image keeps the guessed "no blend".
    pub fn guess_from(image: &[u32]) -> Graphics {
        let mut g = Graphics::guess_base();
        let at = match image.get(1) {
            Some(4) => 0xd0 / 4,
            Some(6) => 0x124 / 4,
            _ => return g,
        };
        let (Some(regs), Some(&0xffff_ffff)) = (image.get(at..at + 8), image.get(at + 8)) else {
            return g;
        };
        if regs.iter().any(|r| r >> 31 == 0) {
            return g;
        }
        g.independent = true;
        for (t, &r) in g.blend.iter_mut().zip(regs) {
            if let Some(b) = cb_blend(r) {
                *t = b;
            }
        }
        // Debugging: the blend alone, the rest guessed as before.
        if env_set!("VICTORY_GUESS_BLEND_ONLY") {
            return g;
        }
        // The registers before the blend ones (same offsets in every image): PA_SU_SC_MODE_CNTL,
        // then DB_DEPTH_CONTROL, DB_STENCIL_CONTROL, DB_STENCILREFMASK(_BF), CB_TARGET_MASK.
        let reg = |back: usize| image[at - back / 4];
        let mode = reg(0x30);
        g.cull = match mode & 3 {
            1 => 2,
            2 => 3,
            _ => 1,
        };
        // FACE 0: counter-clockwise is front.
        g.front_ccw = mode & 4 == 0;
        // Guess (docs/native-host.md, "Polygon offset"): the three floats are
        // PA_SU_POLY_OFFSET_CLAMP, _FRONT_SCALE and _FRONT_OFFSET in register order, written as
        // AMD's PAL writes them: the slope scale in the hardware's 1/16ths, D3D12's integer bias as
        // a float. Only with POLY_OFFSET_FRONT_ENABLE, which every image seen sets.
        if mode & 1 << 11 != 0 {
            let float = |back: usize| f32::from_bits(reg(back));
            g.bias = (float(0x1c) as i32, float(0x24), float(0x20) / 16.0);
        }
        let depth = reg(0x18);
        g.depth_enable = depth & 2 != 0;
        g.depth_write = depth & 4 != 0;
        // The hardware's compare functions are D3D12's less one.
        g.depth_func = (depth >> 4 & 7) + 1;
        g.stencil_enable = depth & 1 != 0;
        let (stencil, refmask) = (reg(0x14), reg(0x10));
        g.stencil_masks = ((refmask >> 8) as u8, (refmask >> 16) as u8);
        // Front, then back (the front's unless BACKFACE_ENABLE): fail, pass and depth fail ops.
        let front = (stencil & 0xfff, depth >> 8 & 7);
        let back = if depth & 0x80 != 0 { (stencil >> 12 & 0xfff, depth >> 20 & 7) } else { front };
        for (face, (ops, func)) in g.stencil_faces.iter_mut().zip([front, back]) {
            *face = [stencil_op_hw(ops & 0xf), stencil_op_hw(ops >> 8 & 0xf), stencil_op_hw(ops >> 4 & 0xf), func + 1];
        }
        let mask = reg(0x4);
        for (t, b) in g.blend.iter_mut().enumerate() {
            b.mask = (mask >> (4 * t) & 0xf) as u8;
        }
        g
    }

    /// This guessed state for a draw: the targets and vertex strides bound, and the input
    /// elements the vertex shader's `VB<kind><n>` semantics are (`reads`, as [`vertex_inputs`]
    /// gives them; index: corner << 28 | slot << 16 | byte offset, corner 0 unless `rect`).
    pub fn for_draw(
        &self,
        colors: Vec<Format>,
        depth: Option<Format>,
        strides: Vec<u32>,
        reads: &[(String, u32)],
        rect: bool,
    ) -> Graphics {
        let mut g = self.clone().with_targets(colors, depth);
        g.inputs.clear();
        for (name, index) in reads {
            let Some((format, _)) = guessed_vertex_format(name) else {
                continue;
            };
            let (corner, slot) = (index >> 28, (index >> 16 & 0xfff).min(31));
            let stride = strides.get(slot as usize).copied().unwrap_or(0);
            g.inputs.push(Input {
                name: name.clone(),
                index: *index,
                format,
                slot,
                offset: (index & 0xffff) + corner * stride,
                per_instance: rect,
            });
        }
        g.guess = Some(Guess { strides, rect });
        g
    }

    /// This described state for rectangle-list draws of the vertex buffers bound at `strides`
    /// (topology 17, drawn as four-vertex strip instances, one a rectangle): every input element
    /// is read per instance, for each of the rectangle's three corners. Corner k's copy (semantic
    /// index `index | k << 28`, as the vertex shader translated with `Options::rect_list` declares
    /// it) is at the element's offset plus k vertices, and each stream steps three vertices an
    /// instance. A slot's vertex stride is the one bound, or where its elements end if none is.
    pub fn for_rects(&self, strides: &[u32]) -> Graphics {
        let mut g = self.clone();
        let slots = self.inputs.iter().map(|e| e.slot as usize + 1).max().unwrap_or(0);
        let mut vertex = vec![0u32; slots];
        for e in &self.inputs {
            let bytes = crate::resource::element(e.format).map_or(0, |(b, _)| b as u32);
            let v = &mut vertex[e.slot as usize];
            *v = (*v).max(e.offset + bytes);
        }
        for (slot, v) in vertex.iter_mut().enumerate() {
            *v = match strides.get(slot) {
                Some(&bound) if bound != 0 => bound,
                _ => v.next_multiple_of(4),
            };
        }
        g.inputs.clear();
        for k in 0..3u32 {
            for e in &self.inputs {
                g.inputs.push(Input {
                    index: e.index | k << 28,
                    offset: e.offset + k * vertex[e.slot as usize],
                    per_instance: true,
                    ..e.clone()
                });
            }
        }
        g.rects = Some(vertex);
        g
    }

    /// Whether this is a guessed state, to be made at the first draw.
    pub fn is_guessed(&self) -> bool {
        self.guess.is_some()
    }

    /// Whether the state names the targets' formats (a stream's does, the console's does not).
    pub fn names_targets(&self) -> bool {
        !self.rt_formats.is_empty() || self.depth_format != 0
    }

    /// This state for a draw into the given targets: the console's pipelines name no formats, the
    /// draw's bound targets do.
    pub fn with_targets(mut self, color: Vec<Format>, depth: Option<Format>) -> Self {
        self.bound = Some((color, depth));
        self
    }

    /// The input elements, in the title's order.
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    /// The colour target formats the host knows; `Err` contains the first unknown DXGI format.
    pub fn color_formats(&self) -> Result<Vec<Format>, u32> {
        // If the state has explicit bound formats (used for console draws), they are already known.
        if let Some((color, _)) = &self.bound {
            return Ok(color.clone());
        }
        // Convert each DXGI format number to a host `Format`. If any conversion fails we return the offending DXGI value.
        self.rt_formats
            .iter()
            .map(|&f| format(f).ok_or(f))
            .collect()
    }

    /// The depth format; `Ok(None)` for no depth target, `Err` if the host does not know it.
    pub fn depth_format(&self) -> Result<Option<Format>, u32> {
        if let Some((_, depth)) = &self.bound {
            return Ok(*depth);
        }
        match self.depth_format {
            0 => Ok(None),
            f => format(f).map(Some).ok_or(f),
        }
    }

    /// The attributes and streams of the input layout for a vertex shader reading `reads`
    /// (semantic, index) in its input signature's order. A stream's stride is where its elements
    /// end. `None`: the shader reads an input the layout does not have.
    pub fn vertex_layout(
        &self,
        reads: &[(String, u32)],
    ) -> Option<(Vec<VertexAttr>, Vec<VertexStream>)> {
        let mut attributes = Vec::new();
        for (name, index) in reads {
            let e = self
                .inputs
                .iter()
                .find(|e| e.name.eq_ignore_ascii_case(name) && e.index == *index)?;
            let format = vertex_format(e.format)
                .or_else(|| self.guess.as_ref().and_then(|_| guessed_vertex_format(&e.name).map(|(_, f)| f)))?;
            attributes.push(VertexAttr {
                format,
                slot: e.slot as usize,
                offset: e.offset as usize,
            });
        }
        let mut streams: Vec<VertexStream> = Vec::new();
        for e in &self.inputs {
            let bytes = crate::resource::element(e.format)
                .map(|(b, _)| b as u32)
                .or_else(|| {
                    self.guess
                        .as_ref()
                        .and_then(|_| guessed_vertex_format(&e.name).map(|(_, f)| f.byte_size() as u32))
                })
                .unwrap_or(0);
            let end = (e.offset + bytes) as usize;
            match streams.iter_mut().find(|s| s.slot == e.slot as usize) {
                Some(s) => s.stride = s.stride.max(end),
                None => streams.push(VertexStream {
                    slot: e.slot as usize,
                    stride: end,
                    per_instance: e.per_instance,
                }),
            }
        }
        for s in &mut streams {
            s.stride = s.stride.next_multiple_of(4);
            // A rectangle-list state's streams step a rectangle's three vertices.
            if let Some(&vertex) = self.rects.as_ref().and_then(|r| r.get(s.slot)) {
                s.stride = 3 * vertex as usize;
            }
            // A guessed pipeline's streams have the strides of the buffers bound.
            if let Some(&bound) = self.guess.as_ref().and_then(|g| g.strides.get(s.slot)) {
                s.stride = if self.guess.as_ref().is_some_and(|g| g.rect) {
                    3 * bound as usize
                } else {
                    s.stride.max(bound as usize)
                };
            }
        }
        Some((attributes, streams))
    }

    /// Blend and write mask of each colour target (target 0's description for all when the blend
    /// description is not independent).
    pub fn color_state(&self) -> Vec<ColorState> {
        let targets = self.bound.as_ref().map_or(self.rt_formats.len(), |b| b.0.len());
        (0..targets.max(1))
            .map(|i| {
                let t = &self.blend[if self.independent { i } else { 0 }];
                ColorState {
                    blend: t.enable.then(|| Blend {
                        src: blend_factor(t.src),
                        dst: blend_factor(t.dst),
                        op: blend_op(t.op),
                        src_alpha: blend_factor(t.src_alpha),
                        dst_alpha: blend_factor(t.dst_alpha),
                        op_alpha: blend_op(t.op_alpha),
                    }),
                    write_mask: t.mask & 0xf,
                }
            })
            .collect()
    }

    pub fn depth_test(&self) -> Option<(bool, Compare)> {
        // Debugging: no depth test at all.
        if self.guess.is_some() && env_set!("VICTORY_DEBUG_NO_DEPTH") {
            return None;
        }
        self.depth_enable
            .then(|| (self.depth_write, compare(self.depth_func)))
    }

    pub fn stencil(&self) -> Option<StencilState> {
        let face = |f: [u32; 4]| StencilFace {
            fail: stencil_op(f[0]),
            depth_fail: stencil_op(f[1]),
            pass: stencil_op(f[2]),
            compare: compare(f[3]),
        };
        self.stencil_enable.then(|| StencilState {
            read_mask: self.stencil_masks.0,
            write_mask: self.stencil_masks.1,
            front: face(self.stencil_faces[0]),
            back: face(self.stencil_faces[1]),
        })
    }

    pub fn cull(&self) -> Cull {
        match self.cull {
            1 => Cull::None,
            2 => Cull::Front,
            _ => Cull::Back,
        }
    }

    pub fn raster(&self) -> Raster {
        Raster {
            front_ccw: self.front_ccw,
            wireframe: self.fill == 2,
            depth_clip: self.depth_clip,
            depth_bias: (self.bias.0 as f32, self.bias.2, self.bias.1),
        }
    }
}

/// `D3D12_BLEND`.
pub fn blend_factor(f: u32) -> BlendFactor {
    match f {
        1 => BlendFactor::Zero,
        3 => BlendFactor::SrcColor,
        4 => BlendFactor::InvSrcColor,
        5 => BlendFactor::SrcAlpha,
        6 => BlendFactor::InvSrcAlpha,
        7 => BlendFactor::DestAlpha,
        8 => BlendFactor::InvDestAlpha,
        9 => BlendFactor::DestColor,
        10 => BlendFactor::InvDestColor,
        11 => BlendFactor::SrcAlphaSat,
        14 => BlendFactor::Constant,
        15 => BlendFactor::InvConstant,
        16 => BlendFactor::Src1Color,
        17 => BlendFactor::InvSrc1Color,
        18 => BlendFactor::Src1Alpha,
        19 => BlendFactor::InvSrc1Alpha,
        _ => BlendFactor::One,
    }
}

/// `D3D12_BLEND_OP`.
pub fn blend_op(o: u32) -> BlendOp {
    match o {
        2 => BlendOp::Subtract,
        3 => BlendOp::ReverseSubtract,
        4 => BlendOp::Min,
        5 => BlendOp::Max,
        _ => BlendOp::Add,
    }
}

/// `D3D12_COMPARISON_FUNC`.
pub fn compare(c: u32) -> Compare {
    match c {
        1 => Compare::Never,
        2 => Compare::Less,
        3 => Compare::Equal,
        4 => Compare::LessEqual,
        5 => Compare::Greater,
        6 => Compare::NotEqual,
        7 => Compare::GreaterEqual,
        _ => Compare::Always,
    }
}

/// `D3D12_STENCIL_OP`.
pub fn stencil_op(o: u32) -> StencilOp {
    match o {
        2 => StencilOp::Zero,
        3 => StencilOp::Replace,
        4 => StencilOp::IncrementClamp,
        5 => StencilOp::DecrementClamp,
        6 => StencilOp::Invert,
        7 => StencilOp::IncrementWrap,
        8 => StencilOp::DecrementWrap,
        _ => StencilOp::Keep,
    }
}

/// A guessed vertex semantic `VB<tag><n>` as `(dxgi, VertexFormat)`.
/// `dxgi` is `0` for `VBT4` (`Snorm1010102`), which matches GFX10 `2_10_10_10` snorm and Metal's
/// `Int1010102Normalized` but has no PC DXGI format number.
fn guessed_vertex_format(name: &str) -> Option<(u32, VertexFormat)> {
    let [b'V', b'B', tag, ch] = name.as_bytes() else {
        return None;
    };
    let (dxgi, fmt) = match (tag, ch) {
        (b'F', b'1') => (41, VertexFormat::Float),
        (b'F', b'2') => (16, VertexFormat::Float2),
        (b'F', b'3') => (6, VertexFormat::Float3),
        (b'F', b'4') => (2, VertexFormat::Float4),
        (b'U', b'1') => (42, VertexFormat::Uint),
        (b'U', b'2') => (17, VertexFormat::Uint2),
        (b'U', b'3') => (7, VertexFormat::Uint3),
        (b'U', b'4') => (3, VertexFormat::Uint4),
        (b'I', b'1') => (43, VertexFormat::Sint),
        (b'I', b'2') => (18, VertexFormat::Sint2),
        (b'I', b'3') => (8, VertexFormat::Sint3),
        (b'I', b'4') => (4, VertexFormat::Sint4),
        (b'H', b'1') => (54, VertexFormat::Half),
        (b'H', b'2') => (34, VertexFormat::Half2),
        (b'H', b'4') => (10, VertexFormat::Half4),
        (b'M', b'1') => (56, VertexFormat::Unorm16),
        (b'M', b'2') => (35, VertexFormat::Unorm16x2),
        (b'M', b'4') => (11, VertexFormat::Unorm16x4),
        (b'S', b'1') => (58, VertexFormat::Snorm16),
        (b'S', b'2') => (37, VertexFormat::Snorm16x2),
        (b'S', b'4') => (13, VertexFormat::Snorm16x4),
        (b'W', b'1') => (57, VertexFormat::Uint16),
        (b'W', b'2') => (36, VertexFormat::Uint16x2),
        (b'W', b'4') => (12, VertexFormat::Uint16x4),
        (b'K', b'1') => (59, VertexFormat::Sint16),
        (b'K', b'2') => (38, VertexFormat::Sint16x2),
        (b'K', b'4') => (14, VertexFormat::Sint16x4),
        (b'N', b'1') => (61, VertexFormat::Unorm8),
        (b'N', b'2') => (49, VertexFormat::Unorm8x2),
        (b'N', b'4') => (28, VertexFormat::Unorm8x4),
        (b'Q', b'1') => (63, VertexFormat::Snorm8),
        (b'Q', b'2') => (51, VertexFormat::Snorm8x2),
        (b'Q', b'4') => (31, VertexFormat::Snorm8x4),
        (b'B', b'1') => (62, VertexFormat::Uint8),
        (b'B', b'2') => (50, VertexFormat::Uint8x2),
        (b'B', b'4') => (30, VertexFormat::Uint8x4),
        (b'C', b'1') => (64, VertexFormat::Sint8),
        (b'C', b'2') => (52, VertexFormat::Sint8x2),
        (b'C', b'4') => (32, VertexFormat::Sint8x4),
        (b'P', b'4') => (24, VertexFormat::Unorm1010102),
        (b'T', b'4') => (0, VertexFormat::Snorm1010102),
        (b'G', b'3') => (26, VertexFormat::FloatRG11B10),
        _ => return None,
    };
    Some((dxgi, fmt))
}

/// An input element's DXGI format (PC numbering) as a vertex format.
pub fn vertex_format(dxgi: u32) -> Option<VertexFormat> {
    Some(match dxgi {
        41 => VertexFormat::Float,
        16 => VertexFormat::Float2,
        6 => VertexFormat::Float3,
        2 => VertexFormat::Float4,
        42 => VertexFormat::Uint,
        17 => VertexFormat::Uint2,
        7 => VertexFormat::Uint3,
        3 => VertexFormat::Uint4,
        43 => VertexFormat::Sint,
        18 => VertexFormat::Sint2,
        8 => VertexFormat::Sint3,
        4 => VertexFormat::Sint4,
        54 => VertexFormat::Half,
        34 => VertexFormat::Half2,
        10 => VertexFormat::Half4,
        56 => VertexFormat::Unorm16,
        35 => VertexFormat::Unorm16x2,
        11 => VertexFormat::Unorm16x4,
        58 => VertexFormat::Snorm16,
        37 => VertexFormat::Snorm16x2,
        13 => VertexFormat::Snorm16x4,
        57 => VertexFormat::Uint16,
        36 => VertexFormat::Uint16x2,
        12 => VertexFormat::Uint16x4,
        59 => VertexFormat::Sint16,
        38 => VertexFormat::Sint16x2,
        14 => VertexFormat::Sint16x4,
        61 => VertexFormat::Unorm8,
        49 => VertexFormat::Unorm8x2,
        28 => VertexFormat::Unorm8x4,
        63 => VertexFormat::Snorm8,
        51 => VertexFormat::Snorm8x2,
        31 => VertexFormat::Snorm8x4,
        62 => VertexFormat::Uint8,
        50 => VertexFormat::Uint8x2,
        30 => VertexFormat::Uint8x4,
        64 => VertexFormat::Sint8,
        52 => VertexFormat::Sint8x2,
        32 => VertexFormat::Sint8x4,
        24 => VertexFormat::Unorm1010102,
        26 => VertexFormat::FloatRG11B10,
        _ => return None,
    })
}

/// `IASetPrimitiveTopology`'s value as the backend's topology: lists and strips of triangles, lines
/// and points. Patches, adjacency and the console's rectangle and quad lists are not.
pub fn topology(d3d: u32) -> Option<Topology> {
    Some(match d3d {
        1 => Topology::Points,
        2 => Topology::Lines,
        3 => Topology::LineStrip,
        4 => Topology::Triangles,
        5 => Topology::TriangleStrip,
        _ => return None,
    })
}

/// The semantics a vertex shader reads, in its input signature's order (system values, which the
/// pipeline supplies, left out), from its DXIL container's `ISG1` part.
pub fn vertex_inputs(dxil: &[u8]) -> Option<Vec<(String, u32)>> {
    use victory_shader::{Container, FourCC};
    let container = Container::parse(dxil).ok()?;
    let part = container.part(FourCC::ISG1)?;
    let d = part.data;
    let count = u32::from_le_bytes(d.get(..4)?.try_into().ok()?) as usize;
    let mut reads = Vec::new();
    for i in 0..count {
        let row = d.get(8 + 32 * i..8 + 32 * i + 32)?;
        let w = |k: usize| u32::from_le_bytes(row[k * 4..k * 4 + 4].try_into().unwrap());
        let (name_at, index, system) = (w(1) as usize, w(2), w(3));
        if system != 0 {
            continue;
        }
        let tail = d.get(name_at..)?;
        let end = tail.iter().position(|&b| b == 0)?;
        reads.push((String::from_utf8_lossy(&tail[..end]).into_owned(), index));
    }
    Some(reads)
}

/// The `TEXCOORD` indices of a translated shader's `ISG1` (inputs) or `OSG1` (outputs) as a mask.
pub fn texcoords(dxil: &[u8], fourcc: victory_shader::FourCC) -> u32 {
    use victory_shader::Container;
    let Some(part) = Container::parse(dxil).ok().and_then(|c| c.part(fourcc).map(|p| p.data.to_vec())) else {
        return 0;
    };
    let word = |at: usize| part.get(at..at + 4).map_or(0, |w| u32::from_le_bytes(w.try_into().unwrap()));
    let mut mask = 0;
    for i in 0..word(0) as usize {
        let at = 8 + 32 * i;
        let name = part.get(word(at + 4) as usize..).and_then(|t| t.split(|&b| b == 0).next()).unwrap_or(&[]);
        if name == b"TEXCOORD" && word(at + 12) == 0 && word(at + 8) < 32 {
            mask |= 1 << word(at + 8);
        }
    }
    mask
}

/// The render targets a pixel shader writes (`SV_Target` index and write mask), from its DXIL
/// container's `OSG1` part.
pub fn pixel_targets(dxil: &[u8]) -> Vec<(u32, u8)> {
    use victory_shader::{Container, FourCC};
    let Some(part) = Container::parse(dxil).ok().and_then(|c| c.part(FourCC::OSG1).map(|p| p.data.to_vec())) else {
        return Vec::new();
    };
    let word = |at: usize| part.get(at..at + 4).map_or(0, |w| u32::from_le_bytes(w.try_into().unwrap()));
    (0..word(0) as usize)
        .filter_map(|i| {
            let at = 8 + 32 * i;
            // System value 64 is a render target.
            (word(at + 12) == 64).then(|| (word(at + 8), part.get(at + 24).copied().unwrap_or(0)))
        })
        .collect()
}

/// Whether a pipeline's colour comes through dual-source blending: target 0's blend reads a second
/// source (`SRC1_*`), which the pixel shader supplies as one more output than there are targets
/// (`SV_Target1` for the one target, not a second attachment). The console does this for particles
/// and decals; Metal expresses it as `[[color(0) index(1)]]`, which its converter only emits when
/// told the pipeline is dual-source.
pub fn dual_source(colors: usize, targets: &[(u32, u8)], states: &[ColorState]) -> bool {
    targets.len() == colors + 1
        && targets.iter().enumerate().all(|(i, (index, _))| *index as usize == i)
        && states
            .first()
            .and_then(|s| s.blend)
            .is_some_and(|b| b.second_source())
}

/// The positions of the system values in a vertex shader's input signature that come before its
/// last attribute. The converter numbers the stage-in attributes by signature position, system
/// values included, so these positions are attribute slots no element fills.
pub fn system_gaps(dxil: &[u8]) -> Vec<usize> {
    use victory_shader::{Container, FourCC};
    let Some(part) = Container::parse(dxil).ok().and_then(|c| c.part(FourCC::ISG1).map(|p| p.data.to_vec())) else {
        return Vec::new();
    };
    let count = part.get(..4).map_or(0, |c| u32::from_le_bytes(c.try_into().unwrap()) as usize);
    let system: Vec<bool> = (0..count)
        .map(|i| part.get(8 + 32 * i + 12..8 + 32 * i + 16).is_some_and(|w| w != [0; 4]))
        .collect();
    let last = system.iter().rposition(|s| !s).unwrap_or(0);
    (0..last).filter(|&i| system[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A described state for rectangle-list draws reads each element for three corners, per
    /// instance: corner k's (`index | k << 28`) k vertices on, each stream stepping three; the
    /// vertex stride is the one bound, or where the elements end when none is.
    #[test]
    fn a_rectangle_list_state_reads_three_corners_an_instance() {
        let element = |name: &str, format, offset| Input {
            name: name.into(),
            index: 0,
            format,
            slot: 0,
            offset,
            per_instance: false,
        };
        // R32G32B32_FLOAT, R8G8B8A8_UNORM, R32G32_FLOAT: 24 bytes.
        let g = Graphics {
            inputs: vec![element("POSITION", 6, 0), element("COLOR", 28, 12), element("TEXCOORD", 16, 16)],
            ..Graphics::default()
        };
        let reads: Vec<(String, u32)> = (0..3u32)
            .flat_map(|k| ["POSITION", "COLOR", "TEXCOORD"].map(|n| (n.to_owned(), k << 28)))
            .collect();
        for (bound, vertex) in [(&[36u32][..], 36usize), (&[][..], 24)] {
            let r = g.for_rects(bound);
            assert!(r.inputs().iter().all(|e| e.per_instance));
            let (attrs, streams) = r.vertex_layout(&reads).expect("every corner's element");
            let offsets: Vec<usize> = attrs.iter().map(|a| a.offset).collect();
            let want: Vec<usize> = (0..3).flat_map(|k| [0, 12, 16].map(|o| o + k * vertex)).collect();
            assert_eq!(offsets, want, "bound {bound:?}");
            assert_eq!(streams.len(), 1);
            assert_eq!((streams[0].stride, streams[0].per_instance), (3 * vertex, true), "bound {bound:?}");
        }
        // The state itself is unchanged: one vertex an element.
        let (attrs, streams) = g.vertex_layout(&reads[..3]).unwrap();
        assert_eq!((attrs.len(), streams[0].stride, streams[0].per_instance), (3, 24, false));
    }

    /// Dual-source blending: the shader gives target 0 one more output than there are targets, and
    /// target 0's blend reads it as its second source. Metal refuses the pipeline otherwise.
    #[test]
    fn a_target_blended_with_its_second_source_is_dual_source() {
        let blend = |src, dst, src_alpha, dst_alpha| ColorState {
            blend: Some(Blend { src, dst, op: BlendOp::Add, src_alpha, dst_alpha, op_alpha: BlendOp::Add }),
            write_mask: 0xf,
        };
        let second = blend(
            BlendFactor::Src1Color,
            BlendFactor::InvSrc1Color,
            BlendFactor::Src1Alpha,
            BlendFactor::InvSrc1Alpha,
        );
        let alpha = blend(
            BlendFactor::SrcAlpha,
            BlendFactor::InvSrcAlpha,
            BlendFactor::One,
            BlendFactor::InvSrcAlpha,
        );
        let two = [(0, 0xf), (1, 0xf)];
        let one = [(0, 0xf)];
        // The console's particle pass: two shader outputs, one target.
        assert!(dual_source(1, &two, &[second]));
        // Two outputs into two targets: a second attachment, not a second source.
        assert!(!dual_source(2, &two, &[second]));
        // One output cannot feed both of the target's sources.
        assert!(!dual_source(1, &one, &[second]));
        // The extra output must be the one after the targets, not a gap.
        assert!(!dual_source(1, &[(0, 0xf), (2, 0xf)], &[second]));
        // Ordinary alpha blending reads no second source.
        assert!(!dual_source(1, &two, &[alpha]));
    }

    /// Any of the four factors reading source 1 makes a blend dual-source.
    #[test]
    fn any_second_source_factor_needs_dual_source() {
        let base = Blend {
            src: BlendFactor::One,
            dst: BlendFactor::One,
            op: BlendOp::Add,
            src_alpha: BlendFactor::One,
            dst_alpha: BlendFactor::One,
            op_alpha: BlendOp::Add,
        };
        assert!(!base.second_source());
        for f in [
            BlendFactor::Src1Color,
            BlendFactor::InvSrc1Color,
            BlendFactor::Src1Alpha,
            BlendFactor::InvSrc1Alpha,
        ] {
            let picks = [
                Blend { src: f, ..base },
                Blend { dst: f, ..base },
                Blend { src_alpha: f, ..base },
                Blend { dst_alpha: f, ..base },
            ];
            assert!(picks.iter().all(Blend::second_source), "{f:?} in one place must count");
        }
    }

    /// A stream's subobject bytes: the type, then the data at +4.
    fn sub(kind: u32, data: &[u32]) -> Vec<u8> {
        let mut v = kind.to_le_bytes().to_vec();
        v.extend(data.iter().flat_map(|w| w.to_le_bytes()));
        v.resize(v.len().next_multiple_of(8), 0);
        v
    }

    /// The text pipeline's image: straight alpha on target 0 (`e5010504`), the rest disabled.
    #[test]
    fn blend_is_read_from_the_register_image() {
        for (sections, at) in [(4, 0xd0 / 4), (6, 0x124 / 4)] {
            let mut image = vec![0u32; at + 20];
            image[1] = sections;
            image[at..at + 8].copy_from_slice(&[0xe501_0504, 0x8000_0000, 0x8000_0000, 0x8000_0000, 0x8000_0000, 0x8000_0000, 0x8000_0000, 0xe001_0161]);
            image[at + 8] = 0xffff_ffff;
            // Cull back; depth on with writes, GREATER_EQUAL; stencil on, EQUAL, pass REPLACE,
            // read mask 0xff, write mask 0x80; targets 0 and 7 write all channels, target 1 red.
            image[at - 12] = 0x0020_1a42;
            image[at - 6] = 0x0020_02e7;
            image[at - 5] = 0x0003_0030;
            image[at - 4] = 0x0180_ff00;
            image[at - 1] = 0xf000_001f;
            // Polygon offset: clamp 0.003, slope scale 16 (1.0 in D3D12's units), bias 29.
            image[at - 9] = 0.003f32.to_bits();
            image[at - 8] = 16.0f32.to_bits();
            image[at - 7] = 29.0f32.to_bits();
            let g = Graphics::guess_from(&image);
            assert!(g.independent);
            assert_eq!((g.cull, g.front_ccw), (3, true));
            assert_eq!((g.depth_enable, g.depth_write, g.depth_func), (true, true, 7));
            assert_eq!((g.stencil_enable, g.stencil_masks), (true, (0xff, 0x80)));
            assert_eq!(g.stencil_faces, [[1, 1, 3, 3]; 2]);
            assert_eq!((g.blend[1].mask, g.blend[2].mask), (1, 0));
            assert_eq!(g.blend[0], TargetBlend { enable: true, src: 5, dst: 6, op: 1, src_alpha: 2, dst_alpha: 6, op_alpha: 1, mask: 0xf });
            assert_eq!(g.blend[1], TargetBlend { mask: 1, ..TargetBlend::default() });
            // Max of one and one for colour, alpha added.
            assert_eq!(g.blend[7], TargetBlend { enable: true, src: 2, dst: 2, op: 5, src_alpha: 2, dst_alpha: 1, op_alpha: 1, mask: 0xf });
            assert_eq!(g.bias, (29, 0.003, 1.0));
            // Metal's order: bias, slope scale, clamp.
            assert_eq!(g.raster().depth_bias, (29.0, 1.0, 0.003));
            // Without POLY_OFFSET_FRONT_ENABLE there is no bias.
            image[at - 12] &= !(1 << 11);
            assert_eq!(Graphics::guess_from(&image).bias, (0, 0.0, 0.0));
            // Without the marker after the registers, nothing is read.
            image[at + 8] = 0;
            assert_eq!(Graphics::guess_from(&image), Graphics::guess_base());
        }
    }

    #[test]
    fn unknown_color_format_is_propagated() {
        // This value is not a known DXGI format.
        let unknown_dxgi = 0xFFFF_FFFFu32;
        let g = Graphics { rt_formats: vec![unknown_dxgi], ..Default::default() };
        assert_eq!(g.color_formats(), Err(unknown_dxgi));
    }

    #[test]
    fn blend_depth_stencil_and_raster_subobjects_are_read() {
        let mut g = Graphics::default();
        // Blend: target 0 on, (SRC_ALPHA, INV_SRC_ALPHA, ADD / ONE, ZERO, ADD), write mask 0b0111;
        // target 1 independent (off, mask 0b1000).
        let mut blend = vec![0u32; 2 + 80];
        blend[1] = 1;
        blend[2..12].copy_from_slice(&[1, 0, 5, 6, 1, 2, 1, 1, 0, 0b0111]);
        blend[12..22].copy_from_slice(&[0, 0, 2, 1, 1, 2, 1, 1, 0, 0b1000]);
        let rast = [3, 2, 1, 5, 0.5f32.to_bits(), 2.0f32.to_bits(), 0, 0, 0, 0, 0];
        // Depth: on, no write, LESS_EQUAL; stencil on, masks 0x0f/0xf0, front (KEEP, KEEP, REPLACE,
        // EQUAL), back (ZERO, INVERT, KEEP, NEVER).
        let depth = [1, 0, 4, 1, 0xf0_0f, 1, 1, 3, 3, 2, 6, 1, 1];
        for s in [sub(8, &blend), sub(10, &rast), sub(11, &depth)] {
            // SAFETY: whole subobjects without pointers.
            unsafe { g.read(u32::from_le_bytes(s[..4].try_into().unwrap()), s.as_ptr()) };
        }
        let mut formats = [0u32; 9];
        formats[0] = 28;
        formats[1] = 10;
        formats[8] = 2;
        let s = sub(15, &formats);
        // SAFETY: as above.
        unsafe { g.read(15, s.as_ptr()) };
        let s = sub(16, &[40]);
        // SAFETY: as above.
        unsafe { g.read(16, s.as_ptr()) };

        let state = g.color_state();
        assert_eq!(state.len(), 2);
        assert_eq!(
            state[0].blend,
            Some(Blend {
                src: BlendFactor::SrcAlpha,
                dst: BlendFactor::InvSrcAlpha,
                op: BlendOp::Add,
                src_alpha: BlendFactor::One,
                dst_alpha: BlendFactor::Zero,
                op_alpha: BlendOp::Add,
            })
        );
        assert_eq!((state[0].write_mask, state[1].blend, state[1].write_mask), (0b0111, None, 0b1000));
        assert_eq!(g.color_formats(), Ok(vec![Format::Rgba8Unorm, Format::Rgba16Float]));
        assert_eq!(g.depth_format(), Ok(Some(Format::D32Float)));
        assert_eq!(g.depth_test(), Some((false, Compare::LessEqual)));
        let st = g.stencil().unwrap();
        assert_eq!((st.read_mask, st.write_mask), (0x0f, 0xf0));
        assert_eq!(st.front.pass, StencilOp::Replace);
        assert_eq!(st.front.compare, Compare::Equal);
        assert_eq!((st.back.fail, st.back.depth_fail, st.back.compare), (StencilOp::Zero, StencilOp::Invert, Compare::Never));
        assert_eq!(g.cull(), Cull::Front);
        let r = g.raster();
        assert!(r.front_ccw && !r.wireframe && !r.depth_clip);
        assert_eq!(r.depth_bias, (5.0, 2.0, 0.5));
    }

    #[test]
    fn input_layouts_resolve_offsets_and_strides_in_signature_order() {
        // POSITION float3 at 0, COLOR rgba8 appended, TEXCOORD float2 appended; slot 1: a
        // per-instance float4 at 16.
        let names = [c"POSITION", c"COLOR", c"TEXCOORD", c"OFFSET"];
        let el = |n: usize, format: u32, slot: u32, offset: u32, class: u32| {
            let mut e = [0u8; 32];
            e[..8].copy_from_slice(&(names[n].as_ptr() as u64).to_le_bytes());
            for (k, w) in [0, format, slot, offset, class, 0].into_iter().enumerate() {
                e[8 + 4 * k..12 + 4 * k].copy_from_slice(&u32::to_le_bytes(w));
            }
            e
        };
        let mut elements = Vec::new();
        for e in [
            el(0, 6, 0, 0, 0),
            el(1, 28, 0, u32::MAX, 0),
            el(2, 16, 0, u32::MAX, 0),
            el(3, 2, 1, 16, 1),
        ] {
            elements.extend(e);
        }
        let mut s = 12u32.to_le_bytes().to_vec();
        s.extend([0; 4]);
        s.extend((elements.as_ptr() as u64).to_le_bytes());
        s.extend(4u32.to_le_bytes());
        s.extend([0; 4]);
        let mut g = Graphics::default();
        // SAFETY: a whole input layout subobject over live elements and names.
        unsafe { g.read(12, s.as_ptr()) };
        assert_eq!(
            g.inputs().iter().map(|i| (i.slot, i.offset)).collect::<Vec<_>>(),
            [(0, 0), (0, 12), (0, 16), (1, 16)]
        );
        // The shader reads TEXCOORD, then POSITION, then OFFSET.
        let reads = [("TEXCOORD".to_owned(), 0), ("POSITION".to_owned(), 0), ("OFFSET".to_owned(), 0)];
        let (attrs, streams) = g.vertex_layout(&reads).unwrap();
        assert_eq!(attrs.iter().map(|a| (a.slot, a.offset)).collect::<Vec<_>>(), [(0, 16), (0, 0), (1, 16)]);
        assert_eq!(attrs[0].format, VertexFormat::Float2);
        assert_eq!(
            streams.iter().map(|s| (s.slot, s.stride, s.per_instance)).collect::<Vec<_>>(),
            [(0, 24, false), (1, 32, true)]
        );
        assert!(g.vertex_layout(&[("NORMAL".to_owned(), 0)]).is_none());
    }

    #[test]
    fn a_rectangle_list_reads_three_corners_per_instance() {
        // One float3 at slot 0 offset 0 of a 24-byte vertex, read at corners 0 to 2.
        let reads: Vec<(String, u32)> = (0..3).map(|c| ("VBF3".to_owned(), c << 28)).collect();
        let g = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![24], &reads, true);
        let (attrs, streams) = g.vertex_layout(&reads).unwrap();
        assert_eq!(attrs.iter().map(|a| a.offset).collect::<Vec<_>>(), [0, 24, 48]);
        assert_eq!((streams[0].stride, streams[0].per_instance), (72, true));
    }

    #[test]
    fn topologies_map_and_the_rest_are_refused() {
        assert_eq!(topology(4), Some(Topology::Triangles));
        assert_eq!(topology(3), Some(Topology::LineStrip));
        assert_eq!(topology(5), Some(Topology::TriangleStrip));
        // Adjacency, patches, the console's rectangle lists.
        assert_eq!([6, 33, 35, 100].map(topology), [None; 4]);
    }

    #[test]
    fn guessed_vertex_formats_cover_all_supported_tags_and_widths() {
        let cases = [
            ("VBF1", 41, VertexFormat::Float, 4),
            ("VBF2", 16, VertexFormat::Float2, 8),
            ("VBF3", 6, VertexFormat::Float3, 12),
            ("VBF4", 2, VertexFormat::Float4, 16),
            ("VBU1", 42, VertexFormat::Uint, 4),
            ("VBU2", 17, VertexFormat::Uint2, 8),
            ("VBU3", 7, VertexFormat::Uint3, 12),
            ("VBU4", 3, VertexFormat::Uint4, 16),
            ("VBI1", 43, VertexFormat::Sint, 4),
            ("VBI2", 18, VertexFormat::Sint2, 8),
            ("VBI3", 8, VertexFormat::Sint3, 12),
            ("VBI4", 4, VertexFormat::Sint4, 16),
            ("VBH1", 54, VertexFormat::Half, 2),
            ("VBH2", 34, VertexFormat::Half2, 4),
            ("VBH4", 10, VertexFormat::Half4, 8),
            ("VBM1", 56, VertexFormat::Unorm16, 2),
            ("VBM2", 35, VertexFormat::Unorm16x2, 4),
            ("VBM4", 11, VertexFormat::Unorm16x4, 8),
            ("VBS1", 58, VertexFormat::Snorm16, 2),
            ("VBS2", 37, VertexFormat::Snorm16x2, 4),
            ("VBS4", 13, VertexFormat::Snorm16x4, 8),
            ("VBW1", 57, VertexFormat::Uint16, 2),
            ("VBW2", 36, VertexFormat::Uint16x2, 4),
            ("VBW4", 12, VertexFormat::Uint16x4, 8),
            ("VBK1", 59, VertexFormat::Sint16, 2),
            ("VBK2", 38, VertexFormat::Sint16x2, 4),
            ("VBK4", 14, VertexFormat::Sint16x4, 8),
            ("VBN1", 61, VertexFormat::Unorm8, 1),
            ("VBN2", 49, VertexFormat::Unorm8x2, 2),
            ("VBN4", 28, VertexFormat::Unorm8x4, 4),
            ("VBQ1", 63, VertexFormat::Snorm8, 1),
            ("VBQ2", 51, VertexFormat::Snorm8x2, 2),
            ("VBQ4", 31, VertexFormat::Snorm8x4, 4),
            ("VBB1", 62, VertexFormat::Uint8, 1),
            ("VBB2", 50, VertexFormat::Uint8x2, 2),
            ("VBB4", 30, VertexFormat::Uint8x4, 4),
            ("VBC1", 64, VertexFormat::Sint8, 1),
            ("VBC2", 52, VertexFormat::Sint8x2, 2),
            ("VBC4", 32, VertexFormat::Sint8x4, 4),
            ("VBP4", 24, VertexFormat::Unorm1010102, 4),
            ("VBT4", 0, VertexFormat::Snorm1010102, 4),
            ("VBG3", 26, VertexFormat::FloatRG11B10, 4),
        ];
        for (name, dxgi, expected, bytes) in cases {
            assert_eq!(expected.byte_size(), bytes, "{name}");
            if dxgi != 0 {
                assert_eq!(vertex_format(dxgi), Some(expected), "{name} dxgi {dxgi}");
                assert_eq!(crate::resource::element(dxgi), Some((bytes as u64, 1)), "{name} element");
            }
            let reads = [(name.to_owned(), 0u32)];
            let g = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &reads, false);
            let (attrs, streams) = g.vertex_layout(&reads).unwrap_or_else(|| panic!("layout for {name}"));
            assert_eq!(attrs.len(), 1);
            assert_eq!(attrs[0].format, expected, "{name}");
            assert_eq!(streams[0].stride, bytes.next_multiple_of(4), "{name}");
        }

        // Unknown tags, invalid channel counts, and non-VB semantics must be refused.
        for bad in ["VBF0", "VBF5", "VBU0", "VBU5", "VBH3", "VBM3", "VBN3", "VBP3", "VBT3", "VBG4", "VBX4", "FOO4", "VBF40"] {
            let reads = [(bad.to_owned(), 0u32)];
            let g = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &reads, false);
            assert!(g.vertex_layout(&reads).is_none(), "{bad} should be refused");
        }
    }

    #[test]
    fn explicit_input_layouts_support_packed_and_narrow_vertex_formats() {
        let names = [c"POS", c"NORM", c"TAN", c"UV", c"JOINTS"];
        let el = |n: usize, format: u32, offset: u32| {
            let mut e = [0u8; 32];
            e[..8].copy_from_slice(&(names[n].as_ptr() as u64).to_le_bytes());
            for (k, w) in [0, format, 0, offset, 0, 0].into_iter().enumerate() {
                e[8 + 4 * k..12 + 4 * k].copy_from_slice(&u32::to_le_bytes(w));
            }
            e
        };
        let mut elements = Vec::new();
        for e in [
            el(0, 10, 0),            // R16G16B16A16_FLOAT (8 bytes -> 0..8)
            el(1, 24, u32::MAX),     // R10G10B10A2_UNORM (4 bytes -> 8..12)
            el(2, 26, u32::MAX),     // R11G11B10_FLOAT (4 bytes -> 12..16)
            el(3, 37, u32::MAX),     // R16G16_SNORM (4 bytes -> 16..20)
            el(4, 30, u32::MAX),     // R8G8B8A8_UINT (4 bytes -> 20..24)
        ] {
            elements.extend(e);
        }
        let mut s = 12u32.to_le_bytes().to_vec();
        s.extend([0; 4]);
        s.extend((elements.as_ptr() as u64).to_le_bytes());
        s.extend(5u32.to_le_bytes());
        s.extend([0; 4]);
        let mut g = Graphics::default();
        // SAFETY: a whole input layout subobject over live elements and names.
        unsafe { g.read(12, s.as_ptr()) };
        let reads = [
            ("POS".to_owned(), 0),
            ("NORM".to_owned(), 0),
            ("TAN".to_owned(), 0),
            ("UV".to_owned(), 0),
            ("JOINTS".to_owned(), 0),
        ];
        let (attrs, streams) = g.vertex_layout(&reads).unwrap();
        assert_eq!(
            attrs.iter().map(|a| (a.format, a.offset)).collect::<Vec<_>>(),
            [
                (VertexFormat::Half4, 0),
                (VertexFormat::Unorm1010102, 8),
                (VertexFormat::FloatRG11B10, 12),
                (VertexFormat::Snorm16x2, 16),
                (VertexFormat::Uint8x4, 20),
            ]
        );
        assert_eq!(streams[0].stride, 24);
    }
}
