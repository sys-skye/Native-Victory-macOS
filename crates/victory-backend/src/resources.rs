//! Descriptor heaps, textures, samplers and render passes: the second slice of the seam (`backend`
//! has queues, fences and compute). Platform neutral, no Metal types.

use crate::backend::{ComputeBackend, Dispatch, Heaps, RootArgs, RootLayout};

/// Which descriptor heap type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeapKind {
    /// `D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV`: bound at slot 0.
    Resource,
    /// `D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER`: bound at slot 1.
    Sampler,
}

/// The DXGI formats the title's textures use.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Format {
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgba16Float,
    R32Float,
    D32Float,
    /// 32-bit depth and 8-bit stencil (DXGI `D32_FLOAT_S8X24_UINT`, and the console's 24-bit
    /// depth formats, which Apple GPUs do not have).
    D32FloatS8,
    Bc1,
    Bc2,
    Bc3,
    Bc4,
    Bc5,
    /// BC6H of unsigned half floats (DXGI `BC6H_UF16`).
    Bc6h,
    Bc7,
    R8Unorm,
    /// Alpha only: samples as (0, 0, 0, a), as D3D's `A8_UNORM` (UI font atlases).
    A8Unorm,
    R16Unorm,
    Rgba32Float,
    Rg32Float,
    Rgb10a2Unorm,
    Rg11b10Float,
    Rgba8Uint,
    Rg16Float,
    Rg16Unorm,
    R32Uint,
    R8Uint,
    /// Shared-exponent RGB (DXGI `R9G9B9E5_SHAREDEXP`).
    Rgb9e5Float,
    Bc7Srgb,
    Rgba32Uint,
    Rgba16Unorm,
    Rgba16Snorm,
    Rg32Uint,
    Rg16Uint,
    Rg8Unorm,
    Rg8Uint,
    Rg8Snorm,
    R16Float,
    R16Uint,
    D16Unorm,
    Bc1Srgb,
    Bc3Srgb,
    /// Metal's `BGR5A1Unorm`.
    Bgr5a1Unorm,
    /// Metal's `ABGR4Unorm`.
    Abgr4Unorm,
    /// BC6H of signed half floats (DXGI `BC6H_SF16`).
    Bc6hSigned,
    Rgba16Uint,
    Rgba8Snorm,
    Rg16Snorm,
    R16Snorm,
    R8Snorm,
    Rgb10a2Uint,
}

impl Format {
    /// (block width and height in texels, bytes per block): 1 for uncompressed, 4 for BC.
    pub fn block(self) -> (usize, usize) {
        use Format::*;
        match self {
            Rgba8Unorm | Rgba8UnormSrgb | Bgra8Unorm | Bgra8UnormSrgb | R32Float | D32Float => (1, 4),
            Rgb10a2Unorm | Rg11b10Float | Rgba8Uint | Rg16Float | Rg16Unorm | R32Uint | Rgb9e5Float => (1, 4),
            Rg16Uint | Rgba8Snorm | Rg16Snorm | Rgb10a2Uint => (1, 4),
            D32FloatS8 => (1, 8),
            Rgba16Float | Rg32Float | Rgba16Unorm | Rgba16Snorm | Rg32Uint | Rgba16Uint => (1, 8),
            Bc1 | Bc1Srgb | Bc4 => (4, 8),
            Bc2 | Bc3 | Bc3Srgb | Bc5 | Bc6h | Bc6hSigned | Bc7 | Bc7Srgb => (4, 16),
            R8Unorm | A8Unorm | R8Uint | R8Snorm => (1, 1),
            R16Unorm | R16Float | R16Uint | R16Snorm | D16Unorm | Rg8Unorm | Rg8Uint | Rg8Snorm | Bgr5a1Unorm | Abgr4Unorm => (1, 2),
            Rgba32Float | Rgba32Uint => (1, 16),
        }
    }

    /// An integer format: its render targets cannot blend.
    pub fn is_integer(self) -> bool {
        use Format::*;
        matches!(
            self,
            Rgba8Uint
                | R32Uint
                | R8Uint
                | Rgba32Uint
                | Rg32Uint
                | Rg16Uint
                | Rg8Uint
                | R16Uint
                | Rgba16Uint
                | Rgb10a2Uint
        )
    }

    /// A depth or depth-stencil format.
    pub fn is_depth(self) -> bool {
        use Format::*;
        matches!(self, D32Float | D32FloatS8 | D16Unorm)
    }

    /// Bytes of one row of `width` texels (of blocks, for BC).
    pub fn row_bytes(self, width: usize) -> usize {
        let (b, n) = self.block();
        width.div_ceil(b) * n
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TexDim {
    D2,
    D2Array,
    D3,
}

#[derive(Clone, Copy, Debug)]
pub struct TextureDesc {
    pub dim: TexDim,
    pub format: Format,
    pub width: usize,
    pub height: usize,
    /// Array length for `D2Array`, depth for `D3`, else 1.
    pub depth_or_layers: usize,
    pub mips: usize,
    /// `ALLOW_RENDER_TARGET` / `ALLOW_DEPTH_STENCIL`.
    pub render_target: bool,
    /// `ALLOW_UNORDERED_ACCESS`.
    pub unordered_access: bool,
}

impl TextureDesc {
    pub fn d2(format: Format, width: usize, height: usize) -> TextureDesc {
        TextureDesc {
            dim: TexDim::D2,
            format,
            width,
            height,
            depth_or_layers: 1,
            mips: 1,
            render_target: false,
            unordered_access: false,
        }
    }
}

/// A view of part of a texture, possibly in another (size-compatible) format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewDesc {
    pub format: Format,
    /// First mip, count.
    pub mips: (usize, usize),
    /// First slice, count.
    pub slices: (usize, usize),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Address {
    Wrap,
    Mirror,
    Clamp,
    Border,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerDesc {
    pub min: Filter,
    pub mag: Filter,
    pub mip: Filter,
    pub address: [Address; 3],
    pub border_color: BorderColor,
    pub max_anisotropy: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Compare {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

/// An allocation a heap keeps resident ([`ResourceBackend::heap_allocations`]): its identity (for
/// counting it once over several heaps), bytes, latest use of an entry naming it, and kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapAllocation {
    pub id: usize,
    pub bytes: u64,
    pub latest: u32,
    pub texture: bool,
}

/// Descriptor heaps, textures and samplers. Heap entries are the converter's 24-byte
/// `IRDescriptorTableEntry`; a root descriptor table is the address of its first entry
/// (`heap_address`), pushed into `RootArgs` like a root view. Resources a heap entry names are kept
/// alive and resident by the heap until it is dropped (`MTLResidencySet`, or `useResource` per
/// encoder where residency sets are unavailable).
pub trait ResourceBackend: ComputeBackend {
    type Texture;
    type Sampler;

    fn create_heap(&self, kind: HeapKind, count: usize) -> Result<Self::Heap, String>;
    /// The GPU address of entry `index`: what a root descriptor table holds.
    fn heap_address(&self, heap: &Self::Heap, index: usize) -> u64;
    /// A raw buffer SRV or UAV over `size` bytes at `title` inside `buffer`.
    fn write_buffer(
        &self,
        heap: &Self::Heap,
        index: usize,
        buffer: &Self::Buffer,
        title: *const u8,
        size: u32,
    );
    /// A buffer SRV or UAV that also carries a typed `texture_buffer` view (`format`) when the
    /// backend supports one, so both raw/structured accesses (`gpuVA`) and typed `bufferLoad` /
    /// `bufferStore` (`textureViewID`) work through the same heap entry. Returns whether a typed
    /// `texture_buffer` view now lives in `index` (`false` when falling back to `write_buffer`).
    #[allow(clippy::too_many_arguments)]
    fn write_typed_buffer(
        &self,
        heap: &Self::Heap,
        index: usize,
        buffer: &Self::Buffer,
        title: *const u8,
        size: u32,
        format: Format,
        writable: bool,
    ) -> bool {
        let _ = (format, writable);
        self.write_buffer(heap, index, buffer, title, size);
        false
    }
    /// A texture SRV or UAV.
    fn write_texture(&self, heap: &Self::Heap, index: usize, tex: &Self::Texture, min_lod: f32);
    fn write_sampler(&self, heap: &Self::Heap, index: usize, s: &Self::Sampler, lod_bias: f32);
    /// Entry `index` names nothing any more: it reads as one never written, and what it kept
    /// alive and resident is let go.
    fn clear_entry(&self, heap: &Self::Heap, index: usize);
    /// Entries in `heap` that currently hold a live `texture_buffer` view.
    fn heap_typed_views(&self, _heap: &Self::Heap) -> usize {
        0
    }
    /// Diagnostics: the distinct allocations `heap` keeps resident, each with its bytes, the latest
    /// use of the entries naming it (`last_used[index]`, a frame number; 0: never) and whether it
    /// is a texture. `None` where the backend keeps nothing resident for its heaps.
    fn heap_allocations(&self, _heap: &Self::Heap, _last_used: &[u32]) -> Option<Vec<HeapAllocation>> {
        None
    }

    fn create_texture(&self, desc: &TextureDesc) -> Result<Self::Texture, String>;
    fn create_texture_view(
        &self,
        tex: &Self::Texture,
        view: &ViewDesc,
    ) -> Result<Self::Texture, String>;
    /// A 2D, single-mip texture whose texels are the title memory at `title` (zero copy): no upload,
    /// CPU and GPU see the same bytes. `title` and `row_pitch` must be multiples of `linear_alignment`.
    fn create_texture_over_memory(
        &self,
        buffer: &Self::Buffer,
        title: *const u8,
        desc: &TextureDesc,
        row_pitch: usize,
    ) -> Result<Self::Texture, String>;
    /// Alignment (bytes) of the offset and row pitch of a texture over memory.
    fn linear_alignment(&self, format: Format) -> usize;
    /// The whole mip of one slice (for 3D, of the whole volume); rows `row_pitch` apart, slices `slice_pitch`.
    fn upload_texture(
        &self,
        tex: &Self::Texture,
        mip: usize,
        slice: usize,
        data: &[u8],
        row_pitch: usize,
        slice_pitch: usize,
    );
    fn read_texture(
        &self,
        tex: &Self::Texture,
        mip: usize,
        slice: usize,
        out: &mut [u8],
        row_pitch: usize,
        slice_pitch: usize,
    );
    fn create_sampler(&self, desc: &SamplerDesc) -> Result<Self::Sampler, String>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VertexFormat {
    Float,
    Float2,
    Float3,
    Float4,
    Uint,
    Uint2,
    Uint3,
    Uint4,
    Sint,
    Sint2,
    Sint3,
    Sint4,
    Half,
    Half2,
    Half4,
    Unorm16,
    Unorm16x2,
    Unorm16x4,
    Snorm16,
    Snorm16x2,
    Snorm16x4,
    Uint16,
    Uint16x2,
    Uint16x4,
    Sint16,
    Sint16x2,
    Sint16x4,
    Unorm8,
    Unorm8x2,
    Unorm8x4,
    Snorm8,
    Snorm8x2,
    Snorm8x4,
    Uint8,
    Uint8x2,
    Uint8x4,
    Sint8,
    Sint8x2,
    Sint8x4,
    Unorm1010102,
    Snorm1010102,
    FloatRG11B10,
}

impl VertexFormat {
    /// Size of one element in bytes.
    pub fn byte_size(self) -> usize {
        use VertexFormat::*;
        match self {
            Unorm8 | Snorm8 | Uint8 | Sint8 => 1,
            Half | Unorm16 | Snorm16 | Uint16 | Sint16 | Unorm8x2 | Snorm8x2 | Uint8x2
            | Sint8x2 => 2,
            Float | Uint | Sint | Half2 | Unorm16x2 | Snorm16x2 | Uint16x2 | Sint16x2
            | Unorm8x4 | Snorm8x4 | Uint8x4 | Sint8x4 | Unorm1010102 | Snorm1010102
            | FloatRG11B10 => 4,
            Float2 | Uint2 | Sint2 | Half4 | Unorm16x4 | Snorm16x4 | Uint16x4 | Sint16x4 => 8,
            Float3 | Uint3 | Sint3 => 12,
            Float4 | Uint4 | Sint4 => 16,
        }
    }
}

/// One input element (`D3D12_INPUT_ELEMENT_DESC`), in the order of the vertex shader's input signature.
#[derive(Clone, Copy, Debug)]
pub struct VertexAttr {
    pub format: VertexFormat,
    pub slot: usize,
    pub offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct VertexStream {
    pub slot: usize,
    pub stride: usize,
    pub per_instance: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Cull {
    None,
    Front,
    #[default]
    Back,
}

/// `D3D12_BLEND`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    InvSrcColor,
    SrcAlpha,
    InvSrcAlpha,
    DestAlpha,
    InvDestAlpha,
    DestColor,
    InvDestColor,
    SrcAlphaSat,
    /// The constant of `OMSetBlendFactor`.
    Constant,
    InvConstant,
    Src1Color,
    InvSrc1Color,
    Src1Alpha,
    InvSrc1Alpha,
}

/// `D3D12_BLEND_OP`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// The blend equation of one colour target (`D3D12_RENDER_TARGET_BLEND_DESC` with blending on).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Blend {
    pub src: BlendFactor,
    pub dst: BlendFactor,
    pub op: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub op_alpha: BlendOp,
}

impl Blend {
    /// Whether a factor reads the pixel shader's second output (`SRC1_*`): dual-source blending,
    /// where one target takes two shader outputs and the second is a blend input only.
    pub fn second_source(&self) -> bool {
        use BlendFactor::*;
        [self.src, self.dst, self.src_alpha, self.dst_alpha]
            .into_iter()
            .any(|f| matches!(f, Src1Color | InvSrc1Color | Src1Alpha | InvSrc1Alpha))
    }
}

/// One colour target's output state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ColorState {
    pub blend: Option<Blend>,
    /// Channels written, bit 0 red to bit 3 alpha.
    pub write_mask: u8,
}

impl Default for ColorState {
    fn default() -> Self {
        ColorState {
            blend: None,
            write_mask: 0xf,
        }
    }
}

/// `D3D12_STENCIL_OP`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StencilFace {
    pub fail: StencilOp,
    pub depth_fail: StencilOp,
    pub pass: StencilOp,
    pub compare: Compare,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StencilState {
    pub read_mask: u8,
    pub write_mask: u8,
    pub front: StencilFace,
    pub back: StencilFace,
}

/// Rasterizer state beyond culling (`D3D12_RASTERIZER_DESC`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Raster {
    pub front_ccw: bool,
    pub wireframe: bool,
    pub depth_clip: bool,
    /// Constant bias in depth units, slope scale, clamp; 0 0 0 is none.
    pub depth_bias: (f32, f32, f32),
}

impl Default for Raster {
    fn default() -> Self {
        Raster {
            front_ccw: false,
            wireframe: false,
            depth_clip: true,
            depth_bias: (0.0, 0.0, 0.0),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Topology {
    Triangles,
    TriangleStrip,
    Lines,
    LineStrip,
    Points,
}

#[derive(Default)]
pub struct RenderPipelineDesc<'a> {
    pub vs: &'a [u8],
    pub ps: Option<&'a [u8]>,
    /// Explicit root layouts when the converter needs the host's argument contract.
    pub vs_layout: Option<&'a RootLayout>,
    pub ps_layout: Option<&'a RootLayout>,
    pub attributes: &'a [VertexAttr],
    pub streams: &'a [VertexStream],
    pub color: &'a [Format],
    pub depth: Option<Format>,
    /// Depth test `(write, compare)`; `None` = off.
    pub depth_test: Option<(bool, Compare)>,
    pub cull: Cull,
    /// Blend and write mask per colour target; targets past the end are opaque, all channels.
    pub color_state: &'a [ColorState],
    /// The pixel shader writes a second output for target 0, used only as a blend input
    /// (dual-source blending). Metal wants that declared when the shader is converted, or the
    /// pipeline is refused for not writing `color(0), index(1)`.
    pub dual_source: bool,
    pub stencil: Option<StencilState>,
    pub raster: Raster,
}

pub struct ColorTarget<'a, B: GraphicsBackend + ?Sized> {
    pub texture: &'a B::Texture,
    /// `None` loads the previous contents.
    pub clear: Option<[f32; 4]>,
}

pub struct DepthTarget<'a, B: GraphicsBackend + ?Sized> {
    pub texture: &'a B::Texture,
    pub clear: Option<f32>,
    /// Independent stencil load action; `None` preserves the existing plane.
    pub clear_stencil: Option<u8>,
}

#[derive(Clone, Copy, Debug)]
pub enum DrawCall {
    Arrays {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    /// `u32` indices when `wide`, else `u16`, starting `first_index` into the index buffer.
    Indexed {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
        wide: bool,
    },
    /// `DispatchMesh` of a mesh pipeline: thread groups in x, y, z.
    Mesh { groups: [u32; 3] },
    /// `ExecuteIndirect` of a rectangle list (indirect only): the record (`D3D12_DRAW_ARGUMENTS`)
    /// counts the vertices of three-corner rectangles, and is drawn as a direct rectangle list is,
    /// one four-vertex strip instance a rectangle (`victory_shader::emit::Options::rect_list`).
    Rects,
}

/// A viewport as `RSSetViewports` takes it: top left, size, depth range.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

/// Per-draw state a title sets with its own calls. `None` / zero: the whole target, no constant.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct DrawState {
    pub viewport: Option<Viewport>,
    /// `RSSetScissorRects`: left, top, right, bottom (texels); clipped to the target.
    pub scissor: Option<[i32; 4]>,
    pub stencil_ref: u32,
    pub blend_factor: [f32; 4],
}

pub struct Draw<'a, B: GraphicsBackend + ?Sized> {
    pub pipeline: &'a B::RenderPipeline,
    pub topology: Topology,
    pub state: DrawState,
    pub root: RootArgs,
    /// The pixel stage's own argument block, if it has one (else it reads `root`).
    pub fragment_root: Option<RootArgs>,
    pub heaps: Heaps<'a, B>,
    pub resources: Vec<&'a B::Buffer>,
    /// `(input slot, buffer, title address)` for each bound vertex stream.
    pub vertex_buffers: Vec<(usize, &'a B::Buffer, *const u8)>,
    pub index_buffer: Option<(&'a B::Buffer, *const u8)>,
    pub call: DrawCall,
    /// `ExecuteIndirect`: the call's counts are read on the GPU from this title address (the
    /// D3D12 argument records are Metal's indirect ones), and `call` gives only its kind.
    pub indirect: Option<(&'a B::Buffer, *const u8)>,
    /// `ExecuteIndirect` with a count buffer: the count's title address and this record's index.
    /// The record runs only if the index is below the count, which the GPU reads when it runs.
    pub count: Option<(&'a B::Buffer, *const u8, u32)>,
}

pub struct RenderPass<'a, B: GraphicsBackend + ?Sized> {
    pub color: Vec<ColorTarget<'a, B>>,
    pub depth: Option<DepthTarget<'a, B>>,
    pub draws: Vec<Draw<'a, B>>,
}

/// Signed view-relative bounds, with exclusive right and bottom edges.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ClearRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// A GPU-side copy or fill operation, ordered with the passes around it.
pub enum BlitOp<'a, B: ResourceBackend + ?Sized> {
    Copy {
        src: (&'a B::Buffer, *const u8),
        dst: (&'a B::Buffer, *const u8),
        len: usize,
    },
    Fill {
        dst: (&'a B::Buffer, *const u8),
        len: usize,
        value: u8,
    },
    CopyTexture {
        src: &'a B::Texture,
        src_mip: usize,
        src_slice: usize,
        src_origin: [usize; 3],
        size: [usize; 3],
        dst: &'a B::Texture,
        dst_mip: usize,
        dst_slice: usize,
        dst_origin: [usize; 3],
    },
    CopyBufferToTexture {
        src: (&'a B::Buffer, *const u8),
        len: usize,
        row_pitch: usize,
        slice_pitch: usize,
        size: [usize; 3],
        dst: &'a B::Texture,
        dst_mip: usize,
        dst_slice: usize,
        dst_origin: [usize; 3],
    },
    CopyTextureToBuffer {
        src: &'a B::Texture,
        src_mip: usize,
        src_slice: usize,
        src_origin: [usize; 3],
        size: [usize; 3],
        dst: (&'a B::Buffer, *const u8),
        len: usize,
        row_pitch: usize,
        slice_pitch: usize,
    },
}

pub enum Pass<'a, B: GraphicsBackend + ?Sized> {
    Compute(&'a [Dispatch<'a, B>]),
    Render(RenderPass<'a, B>),
    Blit(Vec<BlitOp<'a, B>>),
    ClearColorRects {
        target: &'a B::Texture,
        color: [f32; 4],
        rects: &'a [ClearRect],
    },
}

/// Like `Submission`, with render passes among the dispatches, run in order.
pub struct PassSubmission<'a, B: GraphicsBackend + ?Sized> {
    pub waits: &'a [(&'a B::Fence, u64)],
    pub passes: &'a [Pass<'a, B>],
    pub signals: &'a [(&'a B::Fence, u64)],
}

/// Why a submission was not made as submitted, and what of it did not run: the caller pays for
/// the lost work (a dropped pass's value writes never happen, and a queue waiting on one has no
/// other way to go on).
#[derive(Debug)]
pub struct SubmitFailure {
    /// The command buffer committed without the passes in `refused`. `false` when nothing was
    /// committed at all, so every pass of the submission was lost.
    pub committed: bool,
    /// Indices into `PassSubmission::passes` dropped from a committed command buffer (a pass the
    /// backend could not encode).
    pub refused: Vec<usize>,
    /// What failed: the first refusal's message, or why nothing was committed.
    pub error: String,
}

impl SubmitFailure {
    /// Nothing was committed: every pass of the submission was lost.
    pub fn uncommitted(error: impl Into<String>) -> Self {
        Self {
            committed: false,
            refused: Vec::new(),
            error: error.into(),
        }
    }
}

impl std::fmt::Display for SubmitFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.committed {
            write!(f, "passes {:?} were refused: {}", self.refused, self.error)
        } else {
            write!(f, "nothing was committed: {}", self.error)
        }
    }
}

impl std::error::Error for SubmitFailure {}

pub trait GraphicsBackend: ResourceBackend {
    type RenderPipeline;

    /// Converted VS (+ PS) DXIL to a render pipeline: input layout, target formats, depth state, culling.
    fn create_render_pipeline(
        &self,
        desc: &RenderPipelineDesc<'_>,
    ) -> Result<Self::RenderPipeline, String>;
    /// As `ComputeBackend::create_compute_pipeline_within`: `None` while the pipeline is still
    /// being made after `wait`. The default makes it at once, however long.
    fn create_render_pipeline_within(
        &self,
        desc: &RenderPipelineDesc<'_>,
        _wait: std::time::Duration,
    ) -> Result<Option<Self::RenderPipeline>, String> {
        self.create_render_pipeline(desc).map(Some)
    }
    fn submit_passes(
        &self,
        queue: &Self::Queue,
        work: &PassSubmission<'_, Self>,
    ) -> Result<(), SubmitFailure>;
    /// Names a render pipeline for diagnostics (`VICTORY_TRACE_GPU_FAILS`); the default does nothing.
    fn name_render_pipeline(&self, _p: &Self::RenderPipeline, _name: &str) {}
}

/// Showing a texture of the title in a window.
pub trait PresentBackend: GraphicsBackend {
    /// Where frames go: a layer of the app's window.
    type Surface;

    /// A surface over the app's drawable layer (`CAMetalLayer` on Metal), passed by address; the
    /// layer must stay alive while the surface does.
    fn create_surface(&self, layer: *mut std::ffi::c_void) -> Result<Self::Surface, String>;
    /// Attempts presentation without blocking drawable acquisition. False keeps the FIFO head
    /// pending; callers must retry before submitting later work on this queue.
    fn try_present(&self, queue: &Self::Queue, surface: &Self::Surface, sources: &[&Self::Texture]) -> Result<bool, String> {
        self.present(queue, surface, sources).map(|()| true)
    }

    /// Draws `sources` over the surface's next drawable (each scaled to it, whatever the formats)
    /// and presents it after the work already submitted to `queue`: the first as it is, each
    /// later one blended over by its alpha (straight). Never waits for the GPU.
    fn present(
        &self,
        queue: &Self::Queue,
        surface: &Self::Surface,
        sources: &[&Self::Texture],
    ) -> Result<(), String>;
}
