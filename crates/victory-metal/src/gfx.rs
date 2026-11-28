//! Descriptor heaps, textures, samplers, render pipelines and passes on Metal, plus the one encoder
//! loop (`submit_passes`) that compute `submit` also goes through.
//!
//! Descriptor encoding is `metal_irconverter_runtime.h` re-implemented from its documented layout
//! (`IRDescriptorTableSetBuffer` / `SetTexture` / `SetSampler`, 24-byte entries), so no Apple header
//! is needed at build time.

use crate::backend::{ComputeBackend, Heaps, RootArgs};
use crate::metal::{Buffer, Metal, Queue};
use crate::resources::*;
use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::*;
use std::cell::{Cell, RefCell};
use std::ptr::NonNull;

type Res = Retained<ProtocolObject<dyn MTLResource>>;

/// A fence chaining the encoders of one command buffer, and whether one has run: what a pass waits
/// for and updates when the queue's buffers are made resident by a set instead of named.
struct Order {
    fence: Option<Retained<ProtocolObject<dyn MTLFence>>>,
    started: Cell<bool>,
}

type Ordering<'a> = Option<(&'a ProtocolObject<dyn MTLFence>, &'a Cell<bool>)>;

/// `IRDescriptorTableEntry`: `{ gpuVA, textureViewID, metadata }`.
const ENTRY: usize = 24;

/// `kIRDescriptorHeapBindPoint`, `kIRSamplerHeapBindPoint`, `kIRArgumentBufferBindPoint`,
/// `kIRArgumentBufferDrawArgumentsBindPoint`, `kIRArgumentBufferUniformsBindPoint`,
/// `kIRVertexBufferBindPoint`, `kIRStageInAttributeStartIndex` (`metal_irconverter_runtime.h`).
const HEAP_SLOT: [usize; 2] = [0, 1];
const ARGUMENT_BUFFER: usize = 2;
const DRAW_ARGUMENTS: usize = 4;
const UNIFORMS: usize = 5;
const VERTEX_BUFFERS: usize = 6;
const STAGE_IN_ATTRIBUTES: usize = 11;

pub struct Heap {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kind: HeapKind,
    count: usize,
    /// What each entry names: `(resident_allocation, optional_buffer_backed_texture_view)`.
    /// A buffer-backed `TypeTextureBuffer` shares its parent `MTLBuffer`'s backing allocation, so
    /// retaining it here keeps its `gpuResourceID` live without duplicating the pool's pages in
    /// `alloc_counts` / `MTLResidencySet` / per-encoder `useResource`.
    refs: RefCell<std::collections::HashMap<usize, (Res, Option<Retained<ProtocolObject<dyn MTLTexture>>>)>>,
    /// Entries in `refs` that currently hold `Some(texture_buffer)`.
    typed_views: Cell<usize>,
    /// The distinct allocations `refs` names, by raw `MTLResource` pointer, with how many entries
    /// name each: an allocation is added to `set` once and removed + committed before its last
    /// `refs` entry is dropped, and without a set each is named to an encoder once.
    alloc_counts: RefCell<std::collections::HashMap<usize, (usize, Res)>>,
    /// Residency set holding the heap buffer and everything its entries name.
    set: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    dirty: Cell<bool>,
    /// The set asked to stay resident (`VICTORY_METAL_REQUEST_RESIDENCY`): it ends that when dropped.
    requested: bool,
}

impl Drop for Heap {
    fn drop(&mut self) {
        // MTLResidencySet does not retain allocations: detach everything and commit before
        // `buf` and `refs` are dropped.
        if let Some(set) = self.set.take() {
            if self.requested {
                crate::metal::residency_timed("end", Some(&set), || set.endResidency());
            }
            set.removeAllAllocations();
            crate::metal::residency_timed("drop_commit", Some(&set), || set.commit());
        }
    }
}

pub struct Texture {
    tex: Retained<ProtocolObject<dyn MTLTexture>>,
    /// What the pixels are as Metal reads them: sizes the CPU transfers' bounds check.
    format: Format,
}

/// What a CPU transfer of a `w` x `h` x `d` mip needs of `format`: `(row bytes, total bytes)`, or the
/// reason the pitches or the `len` bytes cannot hold it. Metal asserts (and may write past the buffer)
/// when the row pitch is short, so a refused transfer is skipped.
fn transfer_need(format: Format, [w, h, d]: [usize; 3], row_pitch: usize, slice_pitch: usize, len: usize) -> Result<usize, (usize, usize)> {
    let row = format.row_bytes(w);
    let rows = h.div_ceil(format.block().0);
    let image = (rows - 1) * row_pitch + row;
    let total = if d > 1 { (d - 1) * slice_pitch + image } else { image };
    if row_pitch < row || (d > 1 && slice_pitch < image) || len < total {
        return Err((row, total));
    }
    Ok(total)
}

/// Whether the transfer fits; else logs it (the backtrace once) and says to skip it.
#[allow(clippy::too_many_arguments)]
fn transfer_fits(what: &str, t: &Texture, mip: usize, dims: [usize; 3], row_pitch: usize, slice_pitch: usize, len: usize) -> bool {
    static FIRST: std::sync::Once = std::sync::Once::new();
    let Err((row, total)) = transfer_need(t.format, dims, row_pitch, slice_pitch, len) else {
        return true;
    };
    tracing::warn!(
        "{what} skipped: {:?} {:?} mip {mip} region {}x{}x{} row_pitch {row_pitch} slice_pitch {slice_pitch} buffer {len} bytes, needs row {row} total {total}",
        t.format, t.tex.pixelFormat(), dims[0], dims[1], dims[2]
    );
    FIRST.call_once(|| tracing::warn!("first refused transfer, called from:\n{}", std::backtrace::Backtrace::force_capture()));
    false
}

impl Texture {
    /// The texture as the bytes it holds when it is sRGB (a view in the plain format), else itself.
    pub(crate) fn view_without_srgb(&self, _m: &Metal) -> Retained<ProtocolObject<dyn MTLTexture>> {
        let plain = match self.tex.pixelFormat() {
            MTLPixelFormat::RGBA8Unorm_sRGB => MTLPixelFormat::RGBA8Unorm,
            MTLPixelFormat::BGRA8Unorm_sRGB => MTLPixelFormat::BGRA8Unorm,
            _ => return self.tex.clone(),
        };
        self.tex
            .newTextureViewWithPixelFormat(plain)
            .unwrap_or_else(|| self.tex.clone())
    }
}

pub struct Sampler(Retained<ProtocolObject<dyn MTLSamplerState>>);

pub struct RenderPipeline {
    pso: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    depth: Option<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    cull: MTLCullMode,
    raster: Raster,
    /// The pipeline's target has a stencil plane (the pass attaches it too).
    stencil: bool,
    /// A mesh pipeline: its thread-group size (the converter's reflection of the mesh shader).
    mesh: Option<MTLSize>,
    /// Each stage's static samplers, which the stage's argument buffer points at.
    vs_static: Option<StaticSamplers>,
    ps_static: Option<StaticSamplers>,
}

/// One shader stage's static samplers, as Apple's converter reads them: not built into the
/// shader, but one table of `IRDescriptorTableEntry` (sampler, 0, LOD bias), `s0` first, whose
/// address the stage's top-level argument buffer holds at `offset`, right after the root
/// parameters. Nothing wrote that address before: the shader read a sampler through whatever
/// bytes followed the parameters, which sampled zeros on M1 and M2 (Apple7, Apple8) GPUs and
/// left the window black at the title's present pass (Native-Victory-macOS #26).
pub(crate) struct StaticSamplers {
    table: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: usize,
    /// The table holds only their resource IDs.
    _samplers: Vec<Retained<ProtocolObject<dyn MTLSamplerState>>>,
}

impl StaticSamplers {
    /// `root` with the table's address at its offset.
    fn args<'a>(t: Option<&Self>, root: &'a [u8]) -> std::borrow::Cow<'a, [u8]> {
        let Some(t) = t else { return root.into() };
        let mut v = root.to_vec();
        if v.len() < t.offset + 8 {
            v.resize(t.offset + 8, 0);
        }
        v[t.offset..t.offset + 8].copy_from_slice(&t.table.gpuAddress().to_le_bytes());
        v.into()
    }

    fn table(t: Option<&Self>) -> Option<&ProtocolObject<dyn MTLResource>> {
        t.map(|t| ProtocolObject::from_ref(&*t.table))
    }
}

fn pixel_format(f: Format) -> MTLPixelFormat {
    match f {
        Format::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        Format::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Format::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        Format::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        Format::Rgba16Float => MTLPixelFormat::RGBA16Float,
        Format::R32Float => MTLPixelFormat::R32Float,
        Format::D32Float => MTLPixelFormat::Depth32Float,
        Format::D32FloatS8 => MTLPixelFormat::Depth32Float_Stencil8,
        Format::Bc1 => MTLPixelFormat::BC1_RGBA,
        Format::Bc2 => MTLPixelFormat::BC2_RGBA,
        Format::Bc3 => MTLPixelFormat::BC3_RGBA,
        Format::Bc4 => MTLPixelFormat::BC4_RUnorm,
        Format::Bc5 => MTLPixelFormat::BC5_RGUnorm,
        // Metal's `RGBFloat` is the signed BC6H, `RGBUfloat` the unsigned one.
        Format::Bc6h => MTLPixelFormat::BC6H_RGBUfloat,
        Format::Bc6hSigned => MTLPixelFormat::BC6H_RGBFloat,
        Format::Bc7 => MTLPixelFormat::BC7_RGBAUnorm,
        Format::R8Unorm => MTLPixelFormat::R8Unorm,
        Format::A8Unorm => MTLPixelFormat::A8Unorm,
        Format::R16Unorm => MTLPixelFormat::R16Unorm,
        Format::Rgba32Float => MTLPixelFormat::RGBA32Float,
        Format::Rg32Float => MTLPixelFormat::RG32Float,
        Format::Rgb10a2Unorm => MTLPixelFormat::RGB10A2Unorm,
        Format::Rg11b10Float => MTLPixelFormat::RG11B10Float,
        Format::Rgba8Uint => MTLPixelFormat::RGBA8Uint,
        Format::Rg16Float => MTLPixelFormat::RG16Float,
        Format::Rg16Unorm => MTLPixelFormat::RG16Unorm,
        Format::R32Uint => MTLPixelFormat::R32Uint,
        Format::R8Uint => MTLPixelFormat::R8Uint,
        Format::Rgb9e5Float => MTLPixelFormat::RGB9E5Float,
        Format::Bc7Srgb => MTLPixelFormat::BC7_RGBAUnorm_sRGB,
        Format::Rgba32Uint => MTLPixelFormat::RGBA32Uint,
        Format::Rgba16Unorm => MTLPixelFormat::RGBA16Unorm,
        Format::Rgba16Snorm => MTLPixelFormat::RGBA16Snorm,
        Format::Rg32Uint => MTLPixelFormat::RG32Uint,
        Format::Rg16Uint => MTLPixelFormat::RG16Uint,
        Format::Rg8Unorm => MTLPixelFormat::RG8Unorm,
        Format::Rg8Uint => MTLPixelFormat::RG8Uint,
        Format::Rg8Snorm => MTLPixelFormat::RG8Snorm,
        Format::R16Float => MTLPixelFormat::R16Float,
        Format::R16Uint => MTLPixelFormat::R16Uint,
        Format::D16Unorm => MTLPixelFormat::Depth16Unorm,
        Format::Bc1Srgb => MTLPixelFormat::BC1_RGBA_sRGB,
        Format::Bc3Srgb => MTLPixelFormat::BC3_RGBA_sRGB,
        Format::Bgr5a1Unorm => MTLPixelFormat::BGR5A1Unorm,
        Format::Abgr4Unorm => MTLPixelFormat::ABGR4Unorm,
        Format::Rgba16Uint => MTLPixelFormat::RGBA16Uint,
        Format::Rgba8Snorm => MTLPixelFormat::RGBA8Snorm,
        Format::Rg16Snorm => MTLPixelFormat::RG16Snorm,
        Format::R16Snorm => MTLPixelFormat::R16Snorm,
        Format::R8Snorm => MTLPixelFormat::R8Snorm,
        Format::Rgb10a2Uint => MTLPixelFormat::RGB10A2Uint,
    }
}

/// Whether a texture of `from` may be viewed as `to`: uncompressed formats of one texel size, or
/// a block format and its sRGB or signed twin. Never wider than what Metal casts (it asserts on
/// the rest); a depth view of a colour texture is not one of them.
fn castable(from: Format, to: Format) -> bool {
    use Format::*;
    let twin = |f| match f {
        Bc1Srgb => Bc1,
        Bc3Srgb => Bc3,
        Bc7Srgb => Bc7,
        Bc6hSigned => Bc6h,
        f => f,
    };
    if matches!(to, D32Float | D32FloatS8 | D16Unorm) {
        return false;
    }
    if from.block().0 == 1 { from.block() == to.block() } else { twin(from) == twin(to) }
}

fn tex_type(d: TexDim) -> MTLTextureType {
    match d {
        TexDim::D2 => MTLTextureType::Type2D,
        TexDim::D2Array => MTLTextureType::Type2DArray,
        TexDim::D3 => MTLTextureType::Type3D,
    }
}

fn compare(c: Compare) -> MTLCompareFunction {
    match c {
        Compare::Never => MTLCompareFunction::Never,
        Compare::Less => MTLCompareFunction::Less,
        Compare::Equal => MTLCompareFunction::Equal,
        Compare::LessEqual => MTLCompareFunction::LessEqual,
        Compare::Greater => MTLCompareFunction::Greater,
        Compare::NotEqual => MTLCompareFunction::NotEqual,
        Compare::GreaterEqual => MTLCompareFunction::GreaterEqual,
        Compare::Always => MTLCompareFunction::Always,
    }
}

fn blend_factor(f: BlendFactor) -> MTLBlendFactor {
    match f {
        BlendFactor::Zero => MTLBlendFactor::Zero,
        BlendFactor::One => MTLBlendFactor::One,
        BlendFactor::SrcColor => MTLBlendFactor::SourceColor,
        BlendFactor::InvSrcColor => MTLBlendFactor::OneMinusSourceColor,
        BlendFactor::SrcAlpha => MTLBlendFactor::SourceAlpha,
        BlendFactor::InvSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        BlendFactor::DestAlpha => MTLBlendFactor::DestinationAlpha,
        BlendFactor::InvDestAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        BlendFactor::DestColor => MTLBlendFactor::DestinationColor,
        BlendFactor::InvDestColor => MTLBlendFactor::OneMinusDestinationColor,
        BlendFactor::SrcAlphaSat => MTLBlendFactor::SourceAlphaSaturated,
        BlendFactor::Constant => MTLBlendFactor::BlendColor,
        BlendFactor::InvConstant => MTLBlendFactor::OneMinusBlendColor,
        BlendFactor::Src1Color => MTLBlendFactor::Source1Color,
        BlendFactor::InvSrc1Color => MTLBlendFactor::OneMinusSource1Color,
        BlendFactor::Src1Alpha => MTLBlendFactor::Source1Alpha,
        BlendFactor::InvSrc1Alpha => MTLBlendFactor::OneMinusSource1Alpha,
    }
}

fn blend_op(o: BlendOp) -> MTLBlendOperation {
    match o {
        BlendOp::Add => MTLBlendOperation::Add,
        BlendOp::Subtract => MTLBlendOperation::Subtract,
        BlendOp::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOp::Min => MTLBlendOperation::Min,
        BlendOp::Max => MTLBlendOperation::Max,
    }
}

fn stencil_op(o: StencilOp) -> MTLStencilOperation {
    match o {
        StencilOp::Keep => MTLStencilOperation::Keep,
        StencilOp::Zero => MTLStencilOperation::Zero,
        StencilOp::Replace => MTLStencilOperation::Replace,
        StencilOp::IncrementClamp => MTLStencilOperation::IncrementClamp,
        StencilOp::DecrementClamp => MTLStencilOperation::DecrementClamp,
        StencilOp::Invert => MTLStencilOperation::Invert,
        StencilOp::IncrementWrap => MTLStencilOperation::IncrementWrap,
        StencilOp::DecrementWrap => MTLStencilOperation::DecrementWrap,
    }
}

fn stencil_face(f: &StencilFace, s: &StencilState) -> Retained<MTLStencilDescriptor> {
    let d = MTLStencilDescriptor::new();
    d.setStencilFailureOperation(stencil_op(f.fail));
    d.setDepthFailureOperation(stencil_op(f.depth_fail));
    d.setDepthStencilPassOperation(stencil_op(f.pass));
    d.setStencilCompareFunction(compare(f.compare));
    d.setReadMask(u32::from(s.read_mask));
    d.setWriteMask(u32::from(s.write_mask));
    d
}

fn address(a: Address) -> MTLSamplerAddressMode {
    match a {
        Address::Wrap => MTLSamplerAddressMode::Repeat,
        Address::Mirror => MTLSamplerAddressMode::MirrorRepeat,
        Address::Clamp => MTLSamplerAddressMode::ClampToEdge,
        Address::Border => MTLSamplerAddressMode::ClampToBorderColor,
    }
}

fn filter(f: Filter) -> MTLSamplerMinMagFilter {
    match f {
        Filter::Nearest => MTLSamplerMinMagFilter::Nearest,
        Filter::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

fn dynamic_sampler_descriptor(d: &SamplerDesc) -> Retained<MTLSamplerDescriptor> {
    let sd = MTLSamplerDescriptor::new();
    sd.setMinFilter(filter(d.min));
    sd.setMagFilter(filter(d.mag));
    sd.setMipFilter(match d.mip {
        Filter::Nearest => MTLSamplerMipFilter::Nearest,
        Filter::Linear => MTLSamplerMipFilter::Linear,
    });
    sd.setSAddressMode(address(d.address[0]));
    sd.setTAddressMode(address(d.address[1]));
    sd.setRAddressMode(address(d.address[2]));
    sd.setBorderColor(match d.border_color {
        BorderColor::TransparentBlack => MTLSamplerBorderColor::TransparentBlack,
        BorderColor::OpaqueBlack => MTLSamplerBorderColor::OpaqueBlack,
        BorderColor::OpaqueWhite => MTLSamplerBorderColor::OpaqueWhite,
    });
    sd.setMaxAnisotropy(d.max_anisotropy.max(1));
    // Heap entries store resource IDs, which exist only for argument-buffer samplers.
    sd.setSupportArgumentBuffers(true);
    sd
}

/// A sampler for a static sampler's `D3D12_STATIC_SAMPLER_DESC` words up to `MaxLOD` (filter,
/// address U, V, W, LOD bias, anisotropy, comparison, border, minimum and maximum LOD). The LOD
/// bias is the table entry's. Minimum and maximum filtering (filter bits 0x100) have no
/// equivalent here and filter normally.
fn static_sampler(
    dev: &ProtocolObject<dyn MTLDevice>,
    w: &[u32; 10],
) -> Result<Retained<ProtocolObject<dyn MTLSamplerState>>, String> {
    let [
        filter,
        u,
        v,
        ww,
        _,
        anisotropy,
        comparison,
        border,
        min_lod,
        max_lod,
    ] = *w;
    let aniso = filter & 0x40 != 0;
    let linear = |bit: u32| {
        if aniso || filter & bit != 0 {
            MTLSamplerMinMagFilter::Linear
        } else {
            MTLSamplerMinMagFilter::Nearest
        }
    };
    let mode = |m: u32| match m {
        1 => MTLSamplerAddressMode::Repeat,
        2 => MTLSamplerAddressMode::MirrorRepeat,
        4 => MTLSamplerAddressMode::ClampToBorderColor,
        5 => MTLSamplerAddressMode::MirrorClampToEdge,
        _ => MTLSamplerAddressMode::ClampToEdge,
    };
    let sd = MTLSamplerDescriptor::new();
    sd.setMinFilter(linear(0x10));
    sd.setMagFilter(linear(0x4));
    sd.setMipFilter(if aniso || filter & 1 != 0 {
        MTLSamplerMipFilter::Linear
    } else {
        MTLSamplerMipFilter::Nearest
    });
    sd.setSAddressMode(mode(u));
    sd.setTAddressMode(mode(v));
    sd.setRAddressMode(mode(ww));
    sd.setMaxAnisotropy(if aniso {
        anisotropy.clamp(1, 16) as usize
    } else {
        1
    });
    if filter & 0x80 != 0 {
        // `D3D12_COMPARISON_FUNC`, 1 (never) to 8 (always).
        let c = [
            Compare::Never,
            Compare::Less,
            Compare::Equal,
            Compare::LessEqual,
            Compare::Greater,
            Compare::NotEqual,
            Compare::GreaterEqual,
            Compare::Always,
        ];
        sd.setCompareFunction(compare(c[(comparison.clamp(1, 8) - 1) as usize]));
    }
    sd.setBorderColor(match border {
        1 => MTLSamplerBorderColor::OpaqueBlack,
        2 => MTLSamplerBorderColor::OpaqueWhite,
        _ => MTLSamplerBorderColor::TransparentBlack,
    });
    let (lo, hi) = (f32::from_bits(min_lod), f32::from_bits(max_lod));
    let lo = if lo.is_nan() { 0.0 } else { lo.max(0.0) };
    sd.setLodMinClamp(lo);
    sd.setLodMaxClamp(if hi.is_nan() { f32::MAX } else { hi.max(lo) });
    // The table holds the sampler's resource ID, which exists only for argument-buffer samplers.
    sd.setSupportArgumentBuffers(true);
    dev.newSamplerStateWithDescriptor(&sd)
        .ok_or_else(|| "newSamplerState failed for a static sampler".into())
}

fn vertex_format(f: VertexFormat) -> MTLVertexFormat {
    match f {
        VertexFormat::Float => MTLVertexFormat::Float,
        VertexFormat::Float2 => MTLVertexFormat::Float2,
        VertexFormat::Float3 => MTLVertexFormat::Float3,
        VertexFormat::Float4 => MTLVertexFormat::Float4,
        VertexFormat::Uint => MTLVertexFormat::UInt,
        VertexFormat::Uint2 => MTLVertexFormat::UInt2,
        VertexFormat::Uint3 => MTLVertexFormat::UInt3,
        VertexFormat::Uint4 => MTLVertexFormat::UInt4,
        VertexFormat::Sint => MTLVertexFormat::Int,
        VertexFormat::Sint2 => MTLVertexFormat::Int2,
        VertexFormat::Sint3 => MTLVertexFormat::Int3,
        VertexFormat::Sint4 => MTLVertexFormat::Int4,
        VertexFormat::Half => MTLVertexFormat::Half,
        VertexFormat::Half2 => MTLVertexFormat::Half2,
        VertexFormat::Half4 => MTLVertexFormat::Half4,
        VertexFormat::Unorm16 => MTLVertexFormat::UShortNormalized,
        VertexFormat::Unorm16x2 => MTLVertexFormat::UShort2Normalized,
        VertexFormat::Unorm16x4 => MTLVertexFormat::UShort4Normalized,
        VertexFormat::Snorm16 => MTLVertexFormat::ShortNormalized,
        VertexFormat::Snorm16x2 => MTLVertexFormat::Short2Normalized,
        VertexFormat::Snorm16x4 => MTLVertexFormat::Short4Normalized,
        VertexFormat::Uint16 => MTLVertexFormat::UShort,
        VertexFormat::Uint16x2 => MTLVertexFormat::UShort2,
        VertexFormat::Uint16x4 => MTLVertexFormat::UShort4,
        VertexFormat::Sint16 => MTLVertexFormat::Short,
        VertexFormat::Sint16x2 => MTLVertexFormat::Short2,
        VertexFormat::Sint16x4 => MTLVertexFormat::Short4,
        VertexFormat::Unorm8 => MTLVertexFormat::UCharNormalized,
        VertexFormat::Unorm8x2 => MTLVertexFormat::UChar2Normalized,
        VertexFormat::Unorm8x4 => MTLVertexFormat::UChar4Normalized,
        VertexFormat::Snorm8 => MTLVertexFormat::CharNormalized,
        VertexFormat::Snorm8x2 => MTLVertexFormat::Char2Normalized,
        VertexFormat::Snorm8x4 => MTLVertexFormat::Char4Normalized,
        VertexFormat::Uint8 => MTLVertexFormat::UChar,
        VertexFormat::Uint8x2 => MTLVertexFormat::UChar2,
        VertexFormat::Uint8x4 => MTLVertexFormat::UChar4,
        VertexFormat::Sint8 => MTLVertexFormat::Char,
        VertexFormat::Sint8x2 => MTLVertexFormat::Char2,
        VertexFormat::Sint8x4 => MTLVertexFormat::Char4,
        VertexFormat::Unorm1010102 => MTLVertexFormat::UInt1010102Normalized,
        VertexFormat::Snorm1010102 => MTLVertexFormat::Int1010102Normalized,
        VertexFormat::FloatRG11B10 => MTLVertexFormat::FloatRG11B10,
    }
}

/// `MTLResourceID` is a `#[repr(transparent)]` u64 whose field is crate-private in `objc2-metal`.
fn resource_id(id: MTLResourceID) -> u64 {
    // SAFETY: MTLResourceID is one u64 (`Encode`d as such).
    unsafe { std::mem::transmute::<MTLResourceID, u64>(id) }
}

fn origin(a: [usize; 3]) -> MTLOrigin {
    MTLOrigin {
        x: a[0],
        y: a[1],
        z: a[2],
    }
}

fn size(a: [usize; 3]) -> MTLSize {
    MTLSize {
        width: a[0],
        height: a[1],
        depth: a[2],
    }
}

impl Metal {
    fn heap_entry(
        &self,
        heap: &Heap,
        index: usize,
        words: [u64; 3],
        keep: Option<(Res, Option<Retained<ProtocolObject<dyn MTLTexture>>>)>,
    ) {
        assert!(
            index < heap.count,
            "descriptor index {index} out of {}",
            heap.count
        );
        // SAFETY: `index` is inside the heap buffer, which is shared memory the GPU is not
        // reading while the title writes descriptors (D3D12 has the same rule).
        unsafe {
            let p = heap
                .buf
                .contents()
                .as_ptr()
                .cast::<u64>()
                .add(index * ENTRY / 8);
            words
                .iter()
                .enumerate()
                .for_each(|(i, w)| p.add(i).write_volatile(*w));
        }
        let added_typed = matches!(keep, Some((_, Some(_))));
        let old = match keep {
            Some((r, tex_view)) => {
                let ptr = (&*r as *const ProtocolObject<dyn MTLResource>).cast::<()>() as usize;
                let mut counts = heap.alloc_counts.borrow_mut();
                let (count, _) = counts.entry(ptr).or_insert_with(|| (0, r.clone()));
                if *count == 0
                    && let Some(set) = &heap.set
                {
                    set.addAllocation(ProtocolObject::from_ref(&*r));
                    heap.dirty.set(true);
                }
                *count += 1;
                drop(counts);
                heap.refs.borrow_mut().insert(index, (r, tex_view))
            }
            None => heap.refs.borrow_mut().remove(&index),
        };
        let removed_typed = matches!(old, Some((_, Some(_))));
        if added_typed != removed_typed {
            let n = heap.typed_views.get();
            heap.typed_views
                .set(if added_typed { n + 1 } else { n - 1 });
        }
        if let Some((old, old_tex_view)) = old {
            let old_ptr = (&*old as *const ProtocolObject<dyn MTLResource>).cast::<()>() as usize;
            let mut counts = heap.alloc_counts.borrow_mut();
            let last = match counts.get_mut(&old_ptr) {
                Some((n, _)) if *n > 1 => {
                    *n -= 1;
                    false
                }
                Some(_) => {
                    counts.remove(&old_ptr);
                    true
                }
                None => false,
            };
            drop(counts);
            if last && let Some(set) = &heap.set {
                set.removeAllocation(ProtocolObject::from_ref(&*old));
                crate::metal::residency_timed("remove_commit", Some(set), || set.commit());
                heap.dirty.set(false);
            }
            drop(old_tex_view);
            drop(old);
        }
    }

    /// A command buffer of `queue`, with encoder execution status when tracing failures.
    fn new_command_buffer(&self, queue: &Queue) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, String> {
        if self.trace_fails {
            let d = MTLCommandBufferDescriptor::new();
            d.setErrorOptions(MTLCommandBufferErrorOption::EncoderExecutionStatus);
            queue.0.commandBufferWithDescriptor(&d)
        } else {
            queue.0.commandBuffer()
        }
        .ok_or_else(|| "no command buffer".to_string())
    }

    /// The command buffer for the next piece of a split submission (`VICTORY_PASSES_PER_BUFFER`).
    /// A residency set keeps its allocations resident only for the command buffers it was given
    /// to, so every heap is attached again (`attached_heaps` is emptied): without that, pieces 2
    /// and later read heap entries with nothing keeping them resident, and the GPU faulted
    /// (`kIOGPUCommandBufferCallbackErrorPageFault`). The submission's waits are encoded again too,
    /// so no piece starts before what the submission waits for (values already reached cost
    /// nothing), and the piece waits for the one before it (`after`: the submission's piece event
    /// and the value that piece raises it to, by the GPU or, if the GPU fails it, by its recovery).
    fn next_piece(
        &self,
        queue: &Queue,
        waits: &[(&crate::metal::Fence, u64)],
        attached_heaps: &mut Vec<*const Heap>,
        after: Option<(&ProtocolObject<dyn MTLSharedEvent>, u64)>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, String> {
        let cb = self.new_command_buffer(queue)?;
        attached_heaps.clear();
        for (f, v) in waits {
            cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*f.0), *v);
        }
        if let Some((event, v)) = after {
            cb.encodeWaitForEvent_value(ProtocolObject::from_ref(event), v);
        }
        Ok(cb)
    }

    /// With `VICTORY_METAL_REQUEST_RESIDENCY`, asks that a heap's set, as last committed, stay
    /// resident between the command buffers that use it. Without it the system may make the
    /// set's allocations non-resident whenever no command buffer in flight uses the set, and
    /// make them resident again for the next one: in a run on a 48 GB Mac the wired memory fell
    /// from about 22 GB to 6 GB and came back within 10 s in the world, then command buffers
    /// timed out (#79). Measured since (`VICTORY_TRACE_RESIDENCY`, docs/native-host.md): the
    /// wired memory falls only after the host has committed no command buffer for a few seconds,
    /// this request does not stop that, and making the set resident again cost 0.1 to 0.3 s: the
    /// fall follows a stall, it does not cause it.
    fn request(&self, set: &ProtocolObject<dyn MTLResidencySet>) {
        if self.request_residency {
            crate::metal::residency_timed("request", Some(set), || set.requestResidency());
            self.count(|c| c.requests += 1);
        }
    }

    /// A set covers all passes in the command buffer. Later uses can still commit new entries.
    fn use_heap_cb(
        &self,
        heap: &Heap,
        attached_heaps: &mut Vec<*const Heap>,
        attach: impl FnOnce(&ProtocolObject<dyn MTLResidencySet>),
    ) {
        if let Some(set) = &heap.set {
            if heap.dirty.replace(false) {
                crate::metal::residency_timed("add_commit", Some(set), || set.commit());
                self.request(set);
                self.count(|c| {
                    c.heap_entries = c.heap_entries.max(heap.refs.borrow().len() as u64);
                    c.heap_allocations = c.heap_allocations.max(heap.alloc_counts.borrow().len() as u64);
                    c.heap_set_bytes = c.heap_set_bytes.max(set.allocatedSize());
                });
            }
            if !attached_heaps.contains(&std::ptr::from_ref(heap)) {
                attached_heaps.push(heap);
                attach(set);
            }
        }
    }

    /// Residency without sets: the heap buffer and each distinct allocation its entries name, once
    /// per encoder (`named` holds the heaps this encoder was given; residency lasts the encoder).
    /// Naming every entry at every draw cost a `useResource` per entry per draw, and the title fills
    /// more entries as the world streams in, so frame time grew with play time.
    fn use_heap_encoder(
        &self,
        heap: &Heap,
        named: &mut Vec<*const Heap>,
        mut use_res: impl FnMut(&ProtocolObject<dyn MTLResource>),
    ) {
        if heap.set.is_some() || named.contains(&(heap as *const Heap)) {
            return;
        }
        named.push(heap);
        use_res(ProtocolObject::from_ref(&*heap.buf));
        let refs = heap.refs.borrow().len() as u64;
        let allocations = heap.alloc_counts.borrow();
        allocations.values().for_each(|(_, r)| use_res(r));
        self.count(|c| {
            c.heap_resources += 1 + allocations.len() as u64;
            c.heap_entries = c.heap_entries.max(refs);
            c.heap_allocations = c.heap_allocations.max(allocations.len() as u64);
        });
    }

    /// The static samplers of one stage's `layout`, if it declares any.
    pub(crate) fn static_samplers(
        &self,
        layout: Option<&victory_backend::backend::RootLayout>,
    ) -> Result<Option<StaticSamplers>, String> {
        let Some(l) = layout.filter(|l| !l.static_samplers.is_empty()) else {
            return Ok(None);
        };
        // The converter's layout names the table as a table resource with no register.
        // With no table in the layout there is nothing to fill: the pipeline is still made.
        let Some(offset) = self
            .msc()?
            .locations(l)
            .map_err(|e| e.to_string())?
            .iter()
            .find(|&&(kind, space, slot, ..)| kind == 0 && space == u32::MAX && slot == u32::MAX)
            .map(|&(.., offset, _)| offset as usize)
        else {
            return Ok(None);
        };
        let table = self
            .dev
            .newBufferWithLength_options(
                l.static_samplers.len() * ENTRY,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or("no static sampler table")?;
        let mut samplers = Vec::new();
        for (n, w) in l.static_samplers.iter().enumerate() {
            let s = static_sampler(&self.dev, w)?;
            let bias = f32::from_bits(w[4]);
            let bias = if bias.is_nan() {
                0.0
            } else {
                bias.clamp(-16.0, 15.99)
            };
            let entry = [resource_id(s.gpuResourceID()), 0, u64::from(bias.to_bits())];
            // SAFETY: entry `n` is inside the table, which no GPU work reads yet.
            unsafe {
                let p = table.contents().as_ptr().cast::<u64>().add(n * ENTRY / 8);
                entry
                    .iter()
                    .enumerate()
                    .for_each(|(i, w)| p.add(i).write(*w));
            }
            samplers.push(s);
        }
        Ok(Some(StaticSamplers {
            table,
            offset,
            _samplers: samplers,
        }))
    }

    fn bind_root_compute(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        root: &RootArgs,
        heaps: &Heaps<'_, Metal>,
        statics: Option<&StaticSamplers>,
        named: &mut Vec<*const Heap>,
    ) {
        let root = StaticSamplers::args(statics, &root.0);
        if !root.is_empty() {
            // SAFETY: setBytes copies `root` (at most 4 KiB) before returning.
            unsafe {
                e.setBytes_length_atIndex(
                    NonNull::new(root.as_ptr().cast_mut().cast()).unwrap(),
                    root.len(),
                    ARGUMENT_BUFFER,
                )
            };
        }
        if let Some(t) = StaticSamplers::table(statics) {
            e.useResource_usage(t, MTLResourceUsage::Read);
        }
        for (h, slot) in [
            (heaps.resource, HEAP_SLOT[0]),
            (heaps.sampler, HEAP_SLOT[1]),
        ] {
            if let Some(h) = h {
                // SAFETY: a live buffer at an in-range slot.
                unsafe { e.setBuffer_offset_atIndex(Some(&h.buf), 0, slot) };
                self.use_heap_encoder(h, named, |r| {
                    e.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write)
                });
            }
        }
    }

    fn bind_root_render(
        &self,
        e: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        root: &RootArgs,
        fragment: Option<&RootArgs>,
        heaps: &Heaps<'_, Metal>,
        pipeline: &RenderPipeline,
        named: &mut Vec<*const Heap>,
    ) {
        let fragment = fragment.unwrap_or(root);
        let (vs, ps) = (pipeline.vs_static.as_ref(), pipeline.ps_static.as_ref());
        let (root, fragment) = (
            StaticSamplers::args(vs, &root.0),
            StaticSamplers::args(ps, &fragment.0),
        );
        // A mesh pipeline's first stage takes what a vertex shader would.
        let mesh = pipeline.mesh.is_some();
        let first = if mesh { MTLRenderStages::Mesh } else { MTLRenderStages::Vertex };
        // SAFETY: setBytes copies the bytes before returning.
        unsafe {
            if !root.is_empty() {
                let p = NonNull::new(root.as_ptr().cast_mut().cast()).unwrap();
                if mesh {
                    e.setMeshBytes_length_atIndex(p, root.len(), ARGUMENT_BUFFER);
                } else {
                    e.setVertexBytes_length_atIndex(p, root.len(), ARGUMENT_BUFFER);
                }
            }
            if !fragment.is_empty() {
                let p = NonNull::new(fragment.as_ptr().cast_mut().cast()).unwrap();
                e.setFragmentBytes_length_atIndex(p, fragment.len(), ARGUMENT_BUFFER);
            }
        }
        for (t, stage) in [
            (vs, first),
            (ps, MTLRenderStages::Fragment),
        ] {
            if let Some(t) = StaticSamplers::table(t) {
                e.useResource_usage_stages(t, MTLResourceUsage::Read, stage);
            }
        }
        for (h, slot) in [
            (heaps.resource, HEAP_SLOT[0]),
            (heaps.sampler, HEAP_SLOT[1]),
        ] {
            if let Some(h) = h {
                // SAFETY: a live buffer at an in-range slot.
                unsafe {
                    if mesh {
                        e.setMeshBuffer_offset_atIndex(Some(&h.buf), 0, slot);
                    } else {
                        e.setVertexBuffer_offset_atIndex(Some(&h.buf), 0, slot);
                    }
                    e.setFragmentBuffer_offset_atIndex(Some(&h.buf), 0, slot);
                }
                self.use_heap_encoder(h, named, |r| {
                    e.useResource_usage_stages(
                        r,
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                        first | MTLRenderStages::Fragment,
                    )
                });
            }
        }
    }
}

impl ResourceBackend for Metal {
    type Texture = Texture;
    type Sampler = Sampler;

    fn create_heap(&self, kind: HeapKind, count: usize) -> Result<Heap, String> {
        let buf = self
            .dev
            .newBufferWithLength_options(count * ENTRY, MTLResourceOptions::StorageModeShared)
            .ok_or("descriptor heap allocation failed")?;
        let set = if self.residency_sets {
            let d = MTLResidencySetDescriptor::new();
            let s = self
                .dev
                .newResidencySetWithDescriptor_error(&d)
                .map_err(|e| e.localizedDescription().to_string())?;
            s.addAllocation(ProtocolObject::from_ref(&*buf));
            crate::metal::residency_timed("made", Some(&s), || s.commit());
            self.request(&s);
            Some(track_residency_set(s))
        } else {
            None
        };
        Ok(Heap {
            buf,
            kind,
            count,
            refs: RefCell::default(),
            typed_views: Cell::new(0),
            alloc_counts: RefCell::default(),
            requested: set.is_some() && self.request_residency,
            set,
            dirty: Cell::new(false),
        })
    }

    fn heap_address(&self, heap: &Heap, index: usize) -> u64 {
        heap.buf.gpuAddress() + (index * ENTRY) as u64
    }

    fn write_buffer(
        &self,
        heap: &Heap,
        index: usize,
        buffer: &Buffer,
        title: *const u8,
        size: u32,
    ) {
        debug_assert_eq!(heap.kind, HeapKind::Resource);
        let gpu = self.gpu_address(buffer, title);
        let keep: Res = ProtocolObject::from_ref(&*buffer.buf).retain();
        // IRDescriptorTableSetBuffer(entry, gpu, metadata = size & 0xffffffff).
        self.heap_entry(heap, index, [gpu, 0, size as u64], Some((keep, None)));
    }

    fn write_typed_buffer(
        &self,
        heap: &Heap,
        index: usize,
        buffer: &Buffer,
        title: *const u8,
        size: u32,
        format: Format,
        writable: bool,
    ) -> bool {
        debug_assert_eq!(heap.kind, HeapKind::Resource);
        let (bw, elem_bytes) = format.block();
        if bw != 1 || format.is_depth() || size == 0 {
            self.write_buffer(heap, index, buffer, title, size);
            return false;
        }
        let pf = pixel_format(format);
        let align = self
            .dev
            .minimumTextureBufferAlignmentForPixelFormat(pf)
            .max(elem_bytes);
        let offset = title as usize - buffer.base as usize;
        let max_elems = (buffer.buf.length().saturating_sub(offset)) / elem_bytes;
        let width = ((size as usize) / elem_bytes).min(max_elems).min(1 << 26);
        if width == 0 {
            self.write_buffer(heap, index, buffer, title, size);
            return false;
        }
        if !offset.is_multiple_of(align) {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    offset = format!("{offset:#x}"),
                    align,
                    ?format,
                    size,
                    "typed buffer view offset is not aligned for MTLTextureTypeTextureBuffer; falling back to raw buffer entry (first time)"
                );
            } else {
                tracing::debug!(
                    offset = format!("{offset:#x}"),
                    align,
                    ?format,
                    size,
                    "typed buffer view offset is not aligned for MTLTextureTypeTextureBuffer"
                );
            }
            self.write_buffer(heap, index, buffer, title, size);
            return false;
        }
        let td = MTLTextureDescriptor::new();
        td.setTextureType(MTLTextureType::TypeTextureBuffer);
        td.setPixelFormat(pf);
        // SAFETY: `width` is in `1..=(1 << 26)` and fits inside `buffer.buf`.
        unsafe {
            td.setWidth(width);
        }
        td.setStorageMode(MTLStorageMode::Shared);
        let mut usage = MTLTextureUsage::ShaderRead;
        if writable {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        td.setUsage(usage);
        let Some(tex) = buffer
            .buf
            .newTextureWithDescriptor_offset_bytesPerRow(&td, offset, width * elem_bytes)
        else {
            self.write_buffer(heap, index, buffer, title, size);
            return false;
        };
        let gpu = self.gpu_address(buffer, title);
        let tex_id = resource_id(tex.gpuResourceID());
        let keep_buf: Res = ProtocolObject::from_ref(&*buffer.buf).retain();
        self.heap_entry(
            heap,
            index,
            [gpu, tex_id, size as u64],
            Some((keep_buf, Some(tex))),
        );
        true
    }

    fn write_texture(&self, heap: &Heap, index: usize, t: &Texture, min_lod: f32) {
        debug_assert_eq!(heap.kind, HeapKind::Resource);
        let id = resource_id(t.tex.gpuResourceID());
        let keep: Res = ProtocolObject::from_ref(&*t.tex).retain();
        // IRDescriptorTableSetTexture(entry, tex, minLODClamp, metadata = 0).
        self.heap_entry(heap, index, [0, id, min_lod.to_bits() as u64], Some((keep, None)));
    }

    fn write_sampler(&self, heap: &Heap, index: usize, s: &Sampler, lod_bias: f32) {
        debug_assert_eq!(heap.kind, HeapKind::Sampler);
        let id = resource_id(s.0.gpuResourceID());
        // IRDescriptorTableSetSampler(entry, sampler, lodBias): samplers need no residency.
        self.heap_entry(heap, index, [id, 0, lod_bias.to_bits() as u64], None);
    }

    fn clear_entry(&self, heap: &Heap, index: usize) {
        self.heap_entry(heap, index, [0; 3], None);
    }

    fn heap_typed_views(&self, heap: &Heap) -> usize {
        heap.typed_views.get()
    }

    fn heap_allocations(&self, heap: &Heap, last_used: &[u32]) -> Option<Vec<HeapAllocation>> {
        use objc2_foundation::NSObjectProtocol;
        heap.set.as_ref()?;
        // Each allocation's latest use over the entries naming it.
        let mut latest = std::collections::HashMap::<usize, u32>::new();
        for (&index, (r, _)) in heap.refs.borrow().iter() {
            let ptr = (&**r as *const ProtocolObject<dyn MTLResource>).cast::<()>() as usize;
            let used = last_used.get(index).copied().unwrap_or(0);
            let e = latest.entry(ptr).or_insert(0);
            *e = (*e).max(used);
        }
        // A view's (or a texture over a buffer's) memory is its parent's: count the allocation under
        // it, once, with the latest use over every object made of it.
        let mut under = std::collections::HashMap::<usize, HeapAllocation>::new();
        for (&id, (_, r)) in heap.alloc_counts.borrow().iter() {
            let used = latest.get(&id).copied().unwrap_or(0);
            let texture = r.respondsToSelector(objc2::sel!(textureType));
            let root: Res = if texture {
                // SAFETY: it answers `textureType`, so it is an MTLTexture.
                let mut t: Retained<ProtocolObject<dyn MTLTexture>> = unsafe { Retained::cast_unchecked(r.clone()) };
                while let Some(p) = t.parentTexture() {
                    t = p;
                }
                match t.buffer() {
                    Some(b) => ProtocolObject::from_retained(b),
                    None => ProtocolObject::from_retained(t),
                }
            } else {
                r.clone()
            };
            let rid = (&*root as *const ProtocolObject<dyn MTLResource>).cast::<()>() as usize;
            under
                .entry(rid)
                .and_modify(|a| a.latest = a.latest.max(used))
                .or_insert(HeapAllocation { id: rid, bytes: MTLResource::allocatedSize(&*root) as u64, latest: used, texture });
        }
        Some(under.into_values().collect())
    }

    fn create_texture(&self, d: &TextureDesc) -> Result<Texture, String> {
        let td = MTLTextureDescriptor::new();
        td.setTextureType(tex_type(d.dim));
        td.setPixelFormat(pixel_format(d.format));
        // SAFETY: plain size properties of a fresh descriptor; Metal validates them at creation.
        unsafe {
            td.setWidth(d.width);
            td.setHeight(d.height);
            td.setDepth(if d.dim == TexDim::D3 {
                d.depth_or_layers
            } else {
                1
            });
            td.setArrayLength(if d.dim == TexDim::D2Array {
                d.depth_or_layers
            } else {
                1
            });
            td.setMipmapLevelCount(d.mips);
        }
        let mut usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView;
        if d.unordered_access {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        if d.render_target {
            usage |= MTLTextureUsage::RenderTarget;
        }
        td.setUsage(usage);
        td.setStorageMode(MTLStorageMode::Shared);
        let tex = self
            .dev
            .newTextureWithDescriptor(&td)
            .ok_or("newTextureWithDescriptor failed")?;
        Ok(Texture { tex, format: d.format })
    }

    fn create_texture_view(&self, t: &Texture, v: &ViewDesc) -> Result<Texture, String> {
        let ty = t.tex.textureType();
        // A depth texture cannot be viewed as a color format (D3D's R32_FLOAT
        // over a depth buffer): it keeps its own, and shaders read it as the
        // float it holds.
        let source = t.tex.pixelFormat();
        let format = if is_depth(source) {
            source
        } else {
            pixel_format(v.format)
        };
        let title_format = if is_depth(source) { t.format } else { v.format };
        // Metal asserts on a format it cannot cast the texture to, too: a title view left in a
        // heap over a resource object the title has since made anew, with another format.
        if format != source && !castable(t.format, v.format) {
            return Err(format!("a {:?} view of a {:?} texture", v.format, t.format));
        }
        // Metal asserts (it does not return nil) on a range past the texture, which a
        // title view asks for over a texture we made with fewer levels or layers (one
        // over title memory has one level): the range is cut to what is there.
        let fit = |(start, len): (usize, usize), count: usize| {
            (start < count).then(|| NSRange::new(start, len.min(count - start)))
        };
        let slices = if ty == MTLTextureType::Type2DArray {
            t.tex.arrayLength()
        } else {
            1
        };
        let (Some(levels), Some(layers)) =
            (fit(v.mips, t.tex.mipmapLevelCount()), fit(v.slices, slices))
        else {
            return Err(format!(
                "view {:?} {:?} is past the texture",
                v.mips, v.slices
            ));
        };
        // SAFETY: the ranges are inside the texture.
        let tex = unsafe {
            t.tex
                .newTextureViewWithPixelFormat_textureType_levels_slices(format, ty, levels, layers)
        }
        .ok_or("newTextureView refused (format not view-compatible)")?;
        Ok(Texture { tex, format: title_format })
    }

    fn create_texture_over_memory(
        &self,
        buffer: &Buffer,
        title: *const u8,
        d: &TextureDesc,
        row_pitch: usize,
    ) -> Result<Texture, String> {
        // SAFETY: plain sizes; Metal validates them at creation.
        let td = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                pixel_format(d.format),
                d.width,
                d.height,
                false,
            )
        };
        td.setStorageMode(MTLStorageMode::Shared);
        let mut usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView;
        if d.unordered_access {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        td.setUsage(usage);
        let offset = title as usize - buffer.base as usize;
        let tex = buffer
            .buf
            .newTextureWithDescriptor_offset_bytesPerRow(&td, offset, row_pitch)
            .ok_or("buffer-backed texture refused (offset, pitch or format)")?;
        Ok(Texture { tex, format: d.format })
    }

    fn linear_alignment(&self, f: Format) -> usize {
        self.dev
            .minimumLinearTextureAlignmentForPixelFormat(pixel_format(f))
    }

    fn upload_texture(
        &self,
        t: &Texture,
        mip: usize,
        slice: usize,
        data: &[u8],
        row_pitch: usize,
        slice_pitch: usize,
    ) {
        let (w, h, d) = (
            (t.tex.width() >> mip).max(1),
            (t.tex.height() >> mip).max(1),
            (t.tex.depth() >> mip).max(1),
        );
        if !transfer_fits("upload_texture", t, mip, [w, h, d], row_pitch, slice_pitch, data.len()) {
            return;
        }
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: size([w, h, d]),
        };
        // SAFETY: `data` covers the region at the given pitches (callers size it by `Format::row_bytes`).
        unsafe {
            t.tex
                .replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    mip,
                    slice,
                    NonNull::new(data.as_ptr().cast_mut().cast()).unwrap(),
                    row_pitch,
                    slice_pitch,
                )
        };
    }

    fn read_texture(
        &self,
        t: &Texture,
        mip: usize,
        slice: usize,
        out: &mut [u8],
        row_pitch: usize,
        slice_pitch: usize,
    ) {
        let (w, h, d) = (
            (t.tex.width() >> mip).max(1),
            (t.tex.height() >> mip).max(1),
            (t.tex.depth() >> mip).max(1),
        );
        if !transfer_fits("read_texture", t, mip, [w, h, d], row_pitch, slice_pitch, out.len()) {
            return;
        }
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: size([w, h, d]),
        };
        // SAFETY: `out` covers the region at the given pitches.
        unsafe {
            t.tex
                .getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                    NonNull::new(out.as_mut_ptr().cast()).unwrap(),
                    row_pitch,
                    slice_pitch,
                    region,
                    mip,
                    slice,
                )
        };
    }

    fn create_sampler(&self, d: &SamplerDesc) -> Result<Sampler, String> {
        if let Some((_, s)) = self.samplers.borrow().iter().find(|(k, _)| k == d) {
            return Ok(Sampler(s.clone()));
        }
        let sd = dynamic_sampler_descriptor(d);
        let s = self
            .dev
            .newSamplerStateWithDescriptor(&sd)
            .ok_or("newSamplerState failed")?;
        self.samplers.borrow_mut().push((*d, s.clone()));
        Ok(Sampler(s))
    }
}

/// What a compiler crash is remembered by: the two stages' DXIL (the crash is in compiling them,
/// whatever the targets and state).
fn crash_key(d: &RenderPipelineDesc<'_>) -> String {
    crate::disk::key(&[d.vs, d.ps.unwrap_or_default()])
}

impl GraphicsBackend for Metal {
    type RenderPipeline = RenderPipeline;

    fn create_render_pipeline(&self, d: &RenderPipelineDesc<'_>) -> Result<RenderPipeline, String> {
        let (pso, mesh) = self.compiler()?.render(d)?;
        self.finish_render(d, pso, mesh)
    }

    /// As `create_compute_pipeline_within`: converted and compiled on a `pipeline-compile`
    /// thread, and picked up by the same call later when `wait` was not enough.
    fn create_render_pipeline_within(
        &self,
        d: &RenderPipelineDesc<'_>,
        wait: std::time::Duration,
    ) -> Result<Option<RenderPipeline>, String> {
        let state = format!(
            "{} {:?}",
            d.vs.len(),
            (d.vs_layout, d.ps_layout, d.attributes, d.streams, d.color, d.depth, d.depth_test, d.cull, d.color_state, d.dual_source, d.stencil, d.raster)
        );
        let key = *blake3::Hasher::new().update(d.vs).update(d.ps.unwrap_or_default()).update(state.as_bytes()).finalize().as_bytes();
        let made = crate::metal::within(&self.compiling_render, key, wait, || {
            let (compiler, owned) = (self.compiler()?, OwnedDesc::of(d));
            Ok(move || compiler.render(&owned.desc()))
        })?;
        made.map(|made| made.and_then(|(pso, mesh)| self.finish_render(d, pso, mesh))).transpose()
    }

    fn name_render_pipeline(&self, p: &RenderPipeline, name: &str) {
        if self.trace_fails {
            let key = Retained::as_ptr(&p.pso) as *const u8 as usize;
            self.pipe_names.borrow_mut().insert(key, name.to_string());
        }
    }

    fn submit_passes(&self, queue: &Queue, work: &PassSubmission<'_, Self>) -> Result<(), SubmitFailure> {
        if self.trace_fails {
            report_in_flight();
            self.trace_enc.borrow_mut().clear();
            self.trace_args.borrow_mut().clear();
        }
        let mut cb = self.new_command_buffer(queue).map_err(SubmitFailure::uncommitted)?;
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut label = String::new();
        if self.trace_fails {
            label = format!(
                "q{:x} #{} t={:.3} waits={} signals={} passes={}",
                Retained::as_ptr(&queue.0) as *const u8 as usize,
                serial,
                crate::metal::since_start(),
                work.waits.len(),
                work.signals.len(),
                work.passes.len()
            );
            cb.setLabel(Some(&objc2_foundation::NSString::from_str(&label)));
        }
        for (f, v) in work.waits {
            cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*f.0), *v);
        }
        // With residency sets the encoders do not name what they reach through a heap (nor, with
        // the pool's set on the queue, the title's buffers), so Metal cannot order them: this
        // command buffer starts after the previous one finished, and each encoder after the one
        // before.
        let set_on_queue = queue.1.get();
        let order = set_on_queue.then(|| {
            let n = queue.3.get();
            queue.3.set(n + 1);
            if n > 0 {
                cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*queue.2), n);
            }
            Order {
                fence: self.dev.newFence(),
                started: Cell::new(false),
            }
        });
        let order = order.as_ref().and_then(|o| o.fence.as_ref().map(|f| (&**f, &o.started)));
        // A pass that cannot be encoded must not take the submission's fence signals and
        // bottom-of-pipe value writes with it: the title's compute queues wait for those writes,
        // and a lost one is `Unsignaled CrossQueueBarrier` (the title then dies). The failing pass
        // is dropped, the command buffer still commits, and the caller is told which passes were
        // dropped, so it can pay only their lost work.
        let mut refused: Vec<usize> = Vec::new();
        let mut first_error: Option<String> = None;
        let mut attached_heaps = Vec::new();
        // A submission's passes can go in command buffers of at most `passes_per_buffer` passes
        // (`VICTORY_PASSES_PER_BUFFER`; default 0: all in one). macOS stops a command buffer that
        // runs too long on the GPU (`kIOGPUCommandBufferCallbackErrorTimeout`): a world
        // submission of 72 to 80 passes ran 6 to 21 s with the ray-traced passes on, while none of
        // its passes alone took more than 1.4 s. The earlier pieces are committed as they fill.
        // Each piece owes its own value writes, and raises the submission's piece event for the
        // next piece, which waits for it; the last piece owes the fence signals and the order
        // event. A piece the GPU fails pays its own share on the CPU, once ([`Owed`]), so the
        // later pieces still run and nothing is paid twice.
        let per = self.passes_per_buffer;
        let pieces = if per > 0 && work.passes.len() > per {
            Some(track_event(self.dev.newSharedEvent().ok_or_else(|| SubmitFailure::uncommitted("newSharedEvent failed"))?))
        } else {
            None
        };
        let queue_id = Retained::as_ptr(&queue.0) as *const u8 as usize;
        let mut first = 0;
        for (n, pass) in work.passes.iter().enumerate() {
            let encoded = match pass {
                Pass::Compute(dispatches) => {
                    self.encode_compute(&cb, dispatches, set_on_queue, order, &mut attached_heaps)
                }
                Pass::Render(rp) => {
                    self.encode_render(&cb, rp, set_on_queue, order, &mut attached_heaps)
                }
                Pass::Blit(ops) => self.encode_blit(&cb, ops, order),
                Pass::ClearColorRects { target, color, rects } => {
                    self.encode_color_clear_rects(&cb, target, *color, rects, set_on_queue, order, &mut attached_heaps)
                }
            };
            if let Err(e) = encoded {
                // A pass that cannot be encoded (a render target with no attachments, a piece
                // whose root arguments were never set) is dropped; the passes after it still run,
                // so one bad pass does not cost the frame's whole picture.
                refused.push(n);
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
            if let Some(event) = &pieces
                && (n + 1) % per == 0
                && n + 1 < work.passes.len()
            {
                let piece = ((n + 1) / per) as u64;
                cb.encodeSignalEvent_value(ProtocolObject::from_ref(&**event), piece);
                let owed = Owed::of(vec![(event.clone(), piece)], &work.passes[first..=n], 0);
                // The trace names what this piece holds, not the whole submission.
                let tinfo = self.trace_fails.then(|| {
                    let enc = std::mem::take(&mut *self.trace_enc.borrow_mut()).join(" | ");
                    (format!("{label} piece {piece} | {enc}"), std::mem::take(&mut *self.trace_args.borrow_mut()))
                });
                self.commit_piece(queue, &cb, owed, piece, tinfo);
                first = n + 1;
                cb = match self.next_piece(queue, work.waits, &mut attached_heaps, Some((&**event, piece))) {
                    Ok(cb) => cb,
                    Err(e) => {
                        // The current piece never started: every pass from `first` on was lost,
                        // and the earlier pieces did commit.
                        refused.extend(first..work.passes.len());
                        return Err(SubmitFailure {
                            committed: true,
                            refused,
                            error: e,
                        });
                    }
                };
                if self.trace_fails {
                    let piece = format!("{label} piece {}", piece + 1);
                    cb.setLabel(Some(&objc2_foundation::NSString::from_str(&piece)));
                }
            }
        }
        if set_on_queue {
            cb.encodeSignalEvent_value(ProtocolObject::from_ref(&*queue.2), queue.3.get());
        }
        for (f, v) in work.signals {
            cb.encodeSignalEvent_value(ProtocolObject::from_ref(&*f.0), *v);
        }
        let mut events = Vec::with_capacity(work.signals.len() + 1);
        if set_on_queue {
            events.push((queue.2.clone(), queue.3.get()));
        }
        events.extend(work.signals.iter().map(|(f, v)| (f.0.clone(), *v)));
        let owed = std::sync::Arc::new(Owed::of(events, &work.passes[first..], work.signals.len()));
        flights().push_back(Flight { serial, queue: queue_id, since: std::time::Instant::now(), owed: owed.clone() });
        // A command buffer the GPU fails is logged (the first few), not silently dropped.
        let trace = self.trace_fails;
        let tinfo = trace.then(|| (serial, label.clone(), self.trace_enc.borrow().join(" | ")));
        self.trace_commit(serial, queue_id, &label);
        if let Some((s, l, enc)) = &tinfo {
            IN_FLIGHT.lock().unwrap().insert(*s, (std::time::Instant::now(), format!("{l} | {enc}")));
            GATED.lock().unwrap().insert(*s, std::mem::take(&mut *self.trace_args.borrow_mut()));
        }
        let failed = block2::RcBlock::new(move |done: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            // SAFETY: Metal passes the command buffer that completed.
            let done = unsafe { done.as_ref() };
            trace_done(serial, done);
            landed(serial, queue_id, done.error().is_some());
            if let Some((s, l, enc)) = &tinfo {
                let started = IN_FLIGHT.lock().unwrap().remove(s).map(|(t, _)| t);
                gated_args(*s, true);
                if done.error().is_none()
                    && let Some(t) = started
                    && t.elapsed().as_secs_f64() > 1.0
                    && TRACE_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 100
                {
                    tracing::warn!(
                        "metal: slow command buffer {l} took {:.3}s (gpu {:.3}s, kernel {:.3}s) | {enc}",
                        t.elapsed().as_secs_f64(),
                        done.GPUEndTime() - done.GPUStartTime(),
                        done.kernelEndTime() - done.kernelStartTime()
                    );
                }
            }
            if let Some(e) = done.error() {
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < if trace { 200 } else { 20 } {
                    if trace {
                        trace_failure(done, &e);
                    } else {
                        eprintln!("metal: command buffer failed: {e:?} ({})", e.localizedDescription());
                    }
                }
                owed.recover();
            }
        });
        // SAFETY: the block is retained by Metal until it has run.
        unsafe { cb.addCompletedHandler(&*failed as *const _ as *mut _) };
        crate::metal::residency_timed("cb_commit", None, || cb.commit());
        self.committed.set(self.committed.get() + 1);
        match first_error {
            Some(e) => Err(SubmitFailure {
                committed: true,
                refused,
                error: e,
            }),
            None => Ok(()),
        }
    }
}

impl RenderPipeline {
    /// A mesh pipeline's thread-group size (else `None`: a vertex pipeline).
    pub fn mesh_threads(&self) -> Option<[usize; 3]> {
        self.mesh.map(|m| [m.width, m.height, m.depth])
    }
}

/// Formats, write masks and blending of the colour targets.
fn color_attachments(arr: &MTLRenderPipelineColorAttachmentDescriptorArray, d: &RenderPipelineDesc<'_>) {
    for (i, f) in d.color.iter().enumerate() {
        // SAFETY: colour attachment index below 8.
        let a = unsafe { arr.objectAtIndexedSubscript(i) };
        a.setPixelFormat(pixel_format(*f));
        let state = d.color_state.get(i).copied().unwrap_or_default();
        let mut mask = MTLColorWriteMask::empty();
        for (bit, channel) in [
            MTLColorWriteMask::Red,
            MTLColorWriteMask::Green,
            MTLColorWriteMask::Blue,
            MTLColorWriteMask::Alpha,
        ]
        .into_iter()
        .enumerate()
        {
            if state.write_mask >> bit & 1 != 0 {
                mask |= channel;
            }
        }
        a.setWriteMask(mask);
        if let Some(b) = state.blend.filter(|_| !f.is_integer()) {
            a.setBlendingEnabled(true);
            a.setSourceRGBBlendFactor(blend_factor(b.src));
            a.setDestinationRGBBlendFactor(blend_factor(b.dst));
            a.setRgbBlendOperation(blend_op(b.op));
            a.setSourceAlphaBlendFactor(blend_factor(b.src_alpha));
            a.setDestinationAlphaBlendFactor(blend_factor(b.dst_alpha));
            a.setAlphaBlendOperation(blend_op(b.op_alpha));
        }
    }
}

impl Metal {
    /// Commits piece `piece` (from 1) of a submission, not its last ([`GraphicsBackend::submit_passes`]),
    /// with what it owes: its value writes and the piece event raised to `piece`, which the next
    /// piece waits for. One the GPU fails is logged as the last piece's failure is and pays that on
    /// the CPU, so the next piece runs. It is in flight as any command buffer is: one Metal never
    /// reports on is paid by [`recover_lost`].
    fn commit_piece(
        &self,
        queue: &Queue,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        owed: Owed,
        piece: u64,
        tinfo: Option<(String, Vec<(String, usize, usize)>)>,
    ) {
        let owed = std::sync::Arc::new(owed);
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let queue_id = Retained::as_ptr(&queue.0) as *const u8 as usize;
        // Tests: a piece the GPU drops, as one it fails (its passes and its event signal not run).
        #[cfg(test)]
        let dropped = DROP_PIECE.with(|d| d.get()) == Some(piece);
        #[cfg(not(test))]
        let dropped = {
            let _ = piece;
            false
        };
        let empty;
        let cb = if dropped {
            empty = self.new_command_buffer(queue).expect("a command buffer");
            &*empty
        } else {
            cb
        };
        flights().push_back(Flight { serial, queue: queue_id, since: std::time::Instant::now(), owed: owed.clone() });
        self.trace_commit(serial, queue_id, &format!("piece {piece}"));
        let trace = self.trace_fails;
        let tinfo = tinfo.map(|(d, args)| {
            IN_FLIGHT.lock().unwrap().insert(serial, (std::time::Instant::now(), d.clone()));
            GATED.lock().unwrap().insert(serial, args);
            d
        });
        let failed = block2::RcBlock::new(move |done: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            // SAFETY: Metal passes the command buffer that completed.
            let done = unsafe { done.as_ref() };
            trace_done(serial, done);
            let error = done.error();
            landed(serial, queue_id, dropped || error.is_some());
            // A piece is traced as a whole command buffer is: slow ones, and what a failed one held
            // (a time-out carries no encoder states, so the label alone does not name the pass).
            if let Some(d) = &tinfo {
                let started = IN_FLIGHT.lock().unwrap().remove(&serial).map(|(t, _)| t);
                let secs = started.map_or(0.0, |t| t.elapsed().as_secs_f64());
                let args = gated_args(serial, true);
                if (error.is_some() || secs > 1.0)
                    && TRACE_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 100
                {
                    tracing::warn!(
                        "metal: {} piece {d} took {secs:.3}s (gpu {:.3}s, kernel {:.3}s){args}",
                        if error.is_some() { "failed" } else { "slow" },
                        done.GPUEndTime() - done.GPUStartTime(),
                        done.kernelEndTime() - done.kernelStartTime()
                    );
                }
            }
            if let Some(e) = &error
                && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < if trace { 200 } else { 20 }
            {
                if trace {
                    trace_failure(done, e);
                } else {
                    eprintln!("metal: command buffer failed: {e:?} ({})", e.localizedDescription());
                }
            }
            if dropped || error.is_some() {
                owed.recover();
            }
        });
        // SAFETY: the block is retained by Metal until it has run.
        unsafe { cb.addCompletedHandler(&*failed as *const _ as *mut _) };
        crate::metal::residency_timed("cb_commit", None, || cb.commit());
        self.committed.set(self.committed.get() + 1);
    }

    /// The pipeline of a compiled state `pso`: what the passes set beside it.
    fn finish_render(&self, d: &RenderPipelineDesc<'_>, pso: RenderState, mesh: Option<MTLSize>) -> Result<RenderPipeline, String> {
        let has_stencil = d.depth == Some(Format::D32FloatS8);
        let depth = (d.depth_test.is_some() || (d.stencil.is_some() && has_stencil)).then(|| {
            let dd = MTLDepthStencilDescriptor::new();
            let (write, cmp) = d.depth_test.unwrap_or((false, Compare::Always));
            dd.setDepthCompareFunction(compare(cmp));
            dd.setDepthWriteEnabled(write);
            if let Some(s) = d.stencil.filter(|_| has_stencil) {
                dd.setFrontFaceStencil(Some(&stencil_face(&s.front, &s)));
                dd.setBackFaceStencil(Some(&stencil_face(&s.back, &s)));
            }
            self.dev
                .newDepthStencilStateWithDescriptor(&dd)
                .expect("depth state")
        });
        let cull = match d.cull {
            Cull::None => MTLCullMode::None,
            Cull::Front => MTLCullMode::Front,
            Cull::Back => MTLCullMode::Back,
        };
        Ok(RenderPipeline {
            pso,
            depth,
            cull,
            raster: d.raster,
            stencil: has_stencil,
            mesh,
            vs_static: self.static_samplers(d.vs_layout)?,
            ps_static: if d.ps.is_some() {
                self.static_samplers(d.ps_layout)?
            } else {
                None
            },
        })
    }
}

pub(crate) type RenderState = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;

/// A `RenderPipelineDesc` that owns what it names, for the compile thread.
struct OwnedDesc {
    vs: Vec<u8>,
    ps: Option<Vec<u8>>,
    vs_layout: Option<victory_backend::backend::RootLayout>,
    ps_layout: Option<victory_backend::backend::RootLayout>,
    attributes: Vec<VertexAttr>,
    streams: Vec<VertexStream>,
    color: Vec<Format>,
    color_state: Vec<ColorState>,
    depth: Option<Format>,
    depth_test: Option<(bool, Compare)>,
    cull: Cull,
    dual_source: bool,
    stencil: Option<StencilState>,
    raster: Raster,
}

impl OwnedDesc {
    fn of(d: &RenderPipelineDesc<'_>) -> OwnedDesc {
        OwnedDesc {
            vs: d.vs.to_vec(),
            ps: d.ps.map(<[u8]>::to_vec),
            vs_layout: d.vs_layout.cloned(),
            ps_layout: d.ps_layout.cloned(),
            attributes: d.attributes.to_vec(),
            streams: d.streams.to_vec(),
            color: d.color.to_vec(),
            color_state: d.color_state.to_vec(),
            depth: d.depth,
            depth_test: d.depth_test,
            cull: d.cull,
            dual_source: d.dual_source,
            stencil: d.stencil,
            raster: d.raster,
        }
    }

    fn desc(&self) -> RenderPipelineDesc<'_> {
        RenderPipelineDesc {
            vs: &self.vs,
            ps: self.ps.as_deref(),
            vs_layout: self.vs_layout.as_ref(),
            ps_layout: self.ps_layout.as_ref(),
            attributes: &self.attributes,
            streams: &self.streams,
            color: &self.color,
            color_state: &self.color_state,
            depth: self.depth,
            depth_test: self.depth_test,
            cull: self.cull,
            dual_source: self.dual_source,
            stencil: self.stencil,
            raster: self.raster,
        }
    }
}

/// The part of a render pipeline Metal compiles: all of it may run on another thread.
impl crate::metal::Compiler {
    /// The converted function of a stage and the converter's stage number (`IRShaderStage`).
    fn stage_function(
        &self,
        dxil: &[u8],
        layout: Option<&victory_backend::backend::RootLayout>,
        dual_source: bool,
    ) -> Result<(i32, Retained<ProtocolObject<dyn MTLFunction>>), String> {
        let (stage, lib) = self.msc.convert_with(dxil, None, layout, dual_source).map_err(|e| e.to_string())?;
        let l = crate::metal::load_library(&self.dev, &lib)?;
        let name = l
            .functionNames()
            .iter()
            .next()
            .ok_or("converted library is empty")?;
        Ok((stage, l.newFunctionWithName(&name).ok_or("no function")?))
    }

    /// The pipeline state of `d` and, for a mesh pipeline, its thread-group size.
    pub(crate) fn render(&self, d: &RenderPipelineDesc<'_>) -> Result<(RenderState, Option<MTLSize>), String> {
        // A pipeline that crashed Metal's compiler on an earlier run would only crash it again.
        if crate::disk::crashed(crash_key(d).as_bytes()) {
            return Err("Metal's compiler crashed on it on an earlier run".into());
        }
        let (stage, vf) = self.stage_function(d.vs, d.vs_layout, false)?;
        let ff = d
            .ps
            .map(|ps| self.stage_function(ps, d.ps_layout, d.dual_source))
            .transpose()?
            .map(|f| f.1);
        let has_stencil = d.depth == Some(Format::D32FloatS8);
        let (pso, mesh) = if stage == crate::msc::STAGE_MESH {
            let [x, y, z] = self.msc.mesh_threads(d.vs, d.vs_layout).map_err(|e| e.to_string())?;
            let md = MTLMeshRenderPipelineDescriptor::new();
            // SAFETY: both functions are the converter's, of the stages they are set as.
            unsafe {
                md.setMeshFunction(Some(&vf));
                md.setFragmentFunction(ff.as_deref());
            }
            color_attachments(&md.colorAttachments(), d);
            if let Some(f) = d.depth {
                md.setDepthAttachmentPixelFormat(pixel_format(f));
                if has_stencil {
                    md.setStencilAttachmentPixelFormat(pixel_format(f));
                }
            }
            let crash_key = crash_key(d);
            let pso = match &self.archive {
                Some(a) => a.mesh(&self.dev, &md, &crash_key),
                None => crate::archive::compile_mesh(&self.dev, &md, &crash_key),
            }?;
            (pso, Some(size([x as usize, y as usize, z as usize])))
        } else {
            (self.vertex_pipeline(d, &vf, ff.as_deref())?, None)
        };
        Ok((pso, mesh))
    }

    /// The render pipeline state of a vertex and fragment function, with the stage-in layout.
    fn vertex_pipeline(
        &self,
        d: &RenderPipelineDesc<'_>,
        vf: &ProtocolObject<dyn MTLFunction>,
        ff: Option<&ProtocolObject<dyn MTLFunction>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
        let pd = MTLRenderPipelineDescriptor::new();
        pd.setVertexFunction(Some(vf));
        pd.setFragmentFunction(ff);
        // The converter's stage-in function takes its inputs at attribute indices from
        // `kIRStageInAttributeStartIndex`, in input-signature order; buffers from `kIRVertexBufferBindPoint`.
        let vd = MTLVertexDescriptor::new();
        // SAFETY: indices are below the attribute and buffer limits for sane layouts; plain sizes.
        unsafe {
            for (i, a) in d.attributes.iter().enumerate() {
                let at = vd
                    .attributes()
                    .objectAtIndexedSubscript(STAGE_IN_ATTRIBUTES + i);
                at.setFormat(vertex_format(a.format));
                at.setOffset(a.offset);
                at.setBufferIndex(VERTEX_BUFFERS + a.slot);
            }
            for s in d.streams {
                let l = vd
                    .layouts()
                    .objectAtIndexedSubscript(VERTEX_BUFFERS + s.slot);
                l.setStride(s.stride);
                l.setStepFunction(if s.per_instance {
                    MTLVertexStepFunction::PerInstance
                } else {
                    MTLVertexStepFunction::PerVertex
                });
            }
        }
        pd.setVertexDescriptor(Some(&vd));
        color_attachments(&pd.colorAttachments(), d);
        if let Some(f) = d.depth {
            pd.setDepthAttachmentPixelFormat(pixel_format(f));
            if d.depth == Some(Format::D32FloatS8) {
                pd.setStencilAttachmentPixelFormat(pixel_format(f));
            }
        }
        let crash_key = crash_key(d);
        match &self.archive {
            Some(a) => a.render(&self.dev, &pd, &crash_key),
            None => crate::archive::compile_render(&self.dev, &pd, &crash_key),
        }
    }
}


/// What a submission owes the title once it completes: the order event, the title's fence signals
/// and its value writes (fills). A command buffer the GPU fails does none of them, so the queue and
/// every fence and barrier word waiting on it would hang; `recover` pays them from the CPU.
///
/// The fills retain their buffer wrappers so the pool keeps the title memory mapped.
pub(crate) struct Owed {
    recovery: std::sync::Once,
    events: Vec<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
    /// Buffer, offset, length, byte.
    #[allow(clippy::type_complexity)]
    fills: Vec<(Retained<ProtocolObject<dyn MTLBuffer>>, usize, usize, u8)>,
    signals: usize,
}

impl Owed {
    pub(crate) fn new(queue: &Queue, work: &PassSubmission<'_, Metal>, set_on_queue: bool) -> Owed {
        let mut events = Vec::with_capacity(work.signals.len() + 1);
        if set_on_queue {
            events.push((queue.2.clone(), queue.3.get()));
        }
        events.extend(work.signals.iter().map(|(f, v)| (f.0.clone(), *v)));
        Owed::of(events, work.passes, work.signals.len())
    }

    /// What `passes` owe if they fail: their value writes, then `events` raised (`signals` of them
    /// the submission's fence signals, for the log).
    #[allow(clippy::type_complexity)]
    pub(crate) fn of(
        events: Vec<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
        passes: &[Pass<'_, Metal>],
        signals: usize,
    ) -> Owed {
        let mut fills = Vec::new();
        for pass in passes {
            if let Pass::Blit(ops) = pass {
                for op in ops {
                    // shortcut: copies are not replayed (one may have partly run and its source
                    // may have changed since), when a title needs a failed copy's data upgrade
                    // to a replay that checks the source is unchanged.
                    if let BlitOp::Fill { dst, len, value } = op {
                        fills.push((dst.0.buf.clone(), dst.1 as usize - dst.0.base as usize, *len, *value));
                    }
                }
            }
        }
        Owed { recovery: std::sync::Once::new(), events, fills, signals }
    }

    /// Writes each fill, in order, then raises each event (never lowers it): a waiter that sees
    /// its fence reached reads the values the submission wrote before it.
    pub(crate) fn recover(&self) {
        self.recovery.call_once(|| {
            static WARNED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            for (buf, off, len, value) in &self.fills {
                // SAFETY: the range lies inside the wrapped title memory (the caller resolved it).
                unsafe { std::ptr::write_bytes(buf.contents().as_ptr().cast::<u8>().add(*off), *value, *len) };
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            for (e, v) in &self.events {
                if e.signaledValue() < *v {
                    e.setSignaledValue(*v);
                }
            }
            if WARNED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
                tracing::warn!(
                    "metal: a command buffer failed; paid {} fence signals and {} value writes on the CPU",
                    self.signals,
                    self.fills.len()
                );
            }
        });
    }
}

// SAFETY: Metal allows the retained shared events and buffers to be used from either thread.
unsafe impl Send for Owed {}
// SAFETY: the fills and events are immutable lists. The Once lets only one recovery write the
// buffers and signal the events, even if the timeout and completion threads both call recover.
unsafe impl Sync for Owed {}

/// A command buffer committed and not yet reported on by Metal: its queue, when, what it owes.
struct Flight {
    serial: u64,
    queue: usize,
    since: std::time::Instant,
    owed: std::sync::Arc<Owed>,
}

/// Tests only. The list below is the process's, and a recovery with no age pays every command
/// buffer in it, other tests' too: their fences are signalled before their work has run. So the
/// test that recovers takes this for writing, and the tests that depend on when their own command
/// buffers complete take it for reading.
#[cfg(test)]
pub(crate) static FLIGHTS_IN_TESTS: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Committed command buffers Metal has not reported on, oldest first, and per queue the newest
/// serial Metal failed and when.
static FLIGHTS: std::sync::Mutex<std::collections::VecDeque<Flight>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
static FAILURES: std::sync::Mutex<Vec<(usize, u64, std::time::Instant)>> = std::sync::Mutex::new(Vec::new());

fn flights() -> std::sync::MutexGuard<'static, std::collections::VecDeque<Flight>> {
    FLIGHTS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Metal reported on command buffer `serial` of `queue` (completed, or `failed`).
fn landed(serial: u64, queue: usize, failed: bool) {
    flights().retain(|f| f.serial != serial);
    if failed {
        let mut failures = FAILURES.lock().unwrap_or_else(|e| e.into_inner());
        match failures.iter_mut().find(|(q, ..)| *q == queue) {
            Some(f) if f.1 < serial => *f = (queue, serial, std::time::Instant::now()),
            Some(_) => {}
            None => failures.push((queue, serial, std::time::Instant::now())),
        }
    }
}

/// How long after Metal failed a command buffer the ones committed before it on its queue, and
/// still not reported on, count as lost; and how long any command buffer may go unreported.
const LOST_BEHIND_A_FAILURE: std::time::Duration = std::time::Duration::from_secs(5);
const LOST_ANYWAY: std::time::Duration = std::time::Duration::from_secs(30);

/// Pays what lost command buffers owe (fence signals, the order event, value writes) from the CPU
/// and returns how many were lost. The host calls it while work stays held.
///
/// Observed with memory running out (two games at once on a 48 GB Mac, 41 GB wired): Metal fails
/// some command buffers of a queue with `kIOGPUCommandBufferCallbackErrorTimeout` and never
/// reports on others committed before them, neither completed nor failed. Their value writes
/// never land, so every title queue stays held on its barrier word and the title spins for good.
///
/// Guess: a command buffer older than one Metal failed on the same queue and unreported 5 s
/// later, or unreported for 30 s, will never run. If one does run after all, its draws land late
/// and its signals and writes repeat what was paid. The console has no lost command buffers, so
/// this is our recovery, not console behaviour: the title goes on and the lost frames' draws are
/// missing.
pub fn recover_lost() -> usize {
    recover_lost_after(LOST_BEHIND_A_FAILURE, LOST_ANYWAY)
}

fn recover_lost_after(behind_a_failure: std::time::Duration, anyway: std::time::Duration) -> usize {
    let failures = FAILURES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let lost: Vec<Flight> = {
        let mut flying = flights();
        let is_lost = |f: &Flight| {
            f.since.elapsed() >= anyway
                || failures.iter().any(|(q, s, at)| *q == f.queue && f.serial < *s && at.elapsed() >= behind_a_failure)
        };
        let (lost, kept): (Vec<Flight>, Vec<Flight>) = flying.drain(..).partition(is_lost);
        *flying = kept.into();
        lost
    };
    for f in &lost {
        f.owed.recover();
    }
    if !lost.is_empty() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::error!(
                "metal: {} command buffers were never reported on by the GPU (the first unreported for {:.1}s); \
                 their fence signals and value writes are paid on the CPU and their draws are lost. \
                 This is what running out of memory looks like: close other apps, or other games. (First time; later ones are counted.)",
                lost.len(),
                lost[0].since.elapsed().as_secs_f64()
            );
        });
    }
    lost.len()
}

/// The process's shared events (queue order events, title fences, piece events), weakly, so a
/// stop can wake every command buffer waiting on one ([`stop_gpu`]).
struct TrackedEvents(Vec<objc2::rc::Weak<ProtocolObject<dyn MTLSharedEvent>>>);
// SAFETY: shared events may be used from any thread; the list is only touched under its lock.
unsafe impl Send for TrackedEvents {}
static EVENTS: std::sync::Mutex<TrackedEvents> = std::sync::Mutex::new(TrackedEvents(Vec::new()));

/// Remembers `event` for [`stop_gpu`] and returns it.
pub(crate) fn track_event(
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
) -> Retained<ProtocolObject<dyn MTLSharedEvent>> {
    let mut events = EVENTS.lock().unwrap_or_else(|e| e.into_inner());
    if events.0.len() % 1024 == 1023 {
        events.0.retain(|w| w.load().is_some());
    }
    events.0.push(objc2::rc::Weak::from_retained(&event));
    event
}

/// The process's residency sets (each heap's and the pool's), weakly, so a stop can give back what
/// they hold before the process exits ([`stop_gpu`]).
struct TrackedSets(Vec<objc2::rc::Weak<ProtocolObject<dyn MTLResidencySet>>>);
// SAFETY: residency sets may be used from any thread; the list is only touched under its lock.
unsafe impl Send for TrackedSets {}
static SETS: std::sync::Mutex<TrackedSets> = std::sync::Mutex::new(TrackedSets(Vec::new()));

/// Remembers `set` for [`stop_gpu`] and returns it.
pub(crate) fn track_residency_set(
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
) -> Retained<ProtocolObject<dyn MTLResidencySet>> {
    let mut sets = SETS.lock().unwrap_or_else(|e| e.into_inner());
    if sets.0.len() % 256 == 255 {
        sets.0.retain(|w| w.load().is_some());
    }
    sets.0.push(objc2::rc::Weak::from_retained(&set));
    set
}

/// Empties and commits `sets`: what they held is no longer kept resident. Returns the bytes they
/// held.
fn release_sets(sets: &[Retained<ProtocolObject<dyn MTLResidencySet>>]) -> u64 {
    let mut bytes = 0;
    for set in sets {
        bytes += set.allocatedSize();
        set.removeAllAllocations();
        set.commit();
    }
    bytes
}

/// A value above any a command buffer waits for: an event raised to it lets every wait on it pass.
const STOP_VALUE: u64 = 1 << 62;

/// Stops the process's GPU work cleanly before it exits: raises every shared event a command
/// buffer may wait on ([`track_event`]), waits up to `wait` for the command buffers in flight to
/// complete, and pays what the ones that did not still owe on the CPU. Returns how many were in
/// flight and how many of them did not complete.
///
/// A process killed while its command buffers sleep in the kernel on events it never signals can
/// leave the GPU memory they hold allocated after it exits: on macOS 26.5 an M5 Max kept about
/// 20 GB "in use" by the GPU driver, owned by no process, until a reboot (#357). Woken first, they
/// finish and the process exits with nothing pending.
pub fn stop_gpu(wait: std::time::Duration) -> (usize, usize) {
    let events: Vec<_> = EVENTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .0
        .iter()
        .filter_map(objc2::rc::Weak::load)
        .collect();
    let drained = wake_and_drain(&events, wait);
    // Then what the residency sets keep resident is given back before the process goes, not left
    // to the driver's teardown: a run that hung the GPU at world entry with ~16 GB in its heaps'
    // sets left 19.7 GB "in use" after it exited (#357). Process exit skips `Heap`'s `Drop`.
    let sets: Vec<_> = SETS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .0
        .iter()
        .filter_map(objc2::rc::Weak::load)
        .collect();
    let bytes = release_sets(&sets);
    tracing::warn!(sets = sets.len(), mb = bytes >> 20, "stop: residency sets emptied");
    drained
}

/// [`stop_gpu`] for `events` (the tests wake only their own).
fn wake_and_drain(
    events: &[Retained<ProtocolObject<dyn MTLSharedEvent>>],
    wait: std::time::Duration,
) -> (usize, usize) {
    let in_flight = flights().len();
    for e in events {
        if e.signaledValue() < STOP_VALUE {
            e.setSignaledValue(STOP_VALUE);
        }
    }
    let deadline = std::time::Instant::now() + wait;
    while !flights().is_empty() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let left: Vec<Flight> = flights().drain(..).collect();
    for f in &left {
        f.owed.recover();
    }
    (in_flight, left.len())
}

/// `VICTORY_TRACE_COMMITS=1` (with `VICTORY_TRACE_GPU_FAILS` for the encoder names): a line for
/// each command buffer as it is committed (serial, queue, its encoders: pipelines by shader hash,
/// draw counts, dispatch sizes) and one as it completes, so a command buffer the GPU never
/// finishes is the one committed and never done (#79).
fn commits_traced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TRACE_COMMITS").is_ok_and(|v| v.trim() == "1"))
}

/// The completion line of [`commits_traced`].
fn trace_done(serial: u64, done: &ProtocolObject<dyn MTLCommandBuffer>) {
    if commits_traced() {
        tracing::info!(
            "cb done #{serial} t={:.3} gpu_ms={:.1} failed={}",
            crate::metal::since_start(),
            (done.GPUEndTime() - done.GPUStartTime()) * 1000.0,
            done.error().is_some()
        );
    }
}

static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
std::thread_local! {
    /// Tests: the piece of this thread's split submissions that the GPU drops, as one it fails.
    static DROP_PIECE: Cell<Option<u64>> = const { Cell::new(None) };
}
/// Tracing: command buffers committed and not yet completed, by serial: when, and what they carry.
static IN_FLIGHT: std::sync::Mutex<std::collections::BTreeMap<u64, (std::time::Instant, String)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());
/// Tracing: the gate buffers of a command buffer's indirect dispatches and draws (pipeline name or
/// the draw's place in its pass, contents address, words), by serial. The command buffer keeps
/// the buffers alive while it is in flight.
static GATED: std::sync::Mutex<std::collections::BTreeMap<u64, Vec<(String, usize, usize)>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The group counts the GPU gave command buffer `serial`'s indirect dispatches, as its gates
/// wrote them (zeros for a gate that has not run); `forget` when the command buffer has landed.
fn gated_args(serial: u64, forget: bool) -> String {
    let mut gated = GATED.lock().unwrap();
    let args = if forget { gated.remove(&serial) } else { gated.get(&serial).cloned() };
    let mut out = String::new();
    for (name, at, words) in args.unwrap_or_default() {
        // SAFETY: a shared buffer of `words` words, alive while its command buffer is in flight; a
        // landed one is read in its completion handler, before Metal releases it.
        let w: Vec<u32> = (0..words).map(|k| unsafe { (at as *const u32).add(k).read_unaligned() }).collect();
        // A dispatch's gate holds what it runs, then what the record asked for.
        let asked = if words == 6 { &w[3..] } else { &w[..] };
        out.push_str(&format!(" | gpu read {name} {asked:?}"));
        // A record past the most groups the gate runs, which ran nothing: each pipeline's first few.
        if words == 6 && asked.iter().map(|g| u128::from(*g)).product::<u128>() > 1 << 22 {
            static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap();
            if seen.iter().filter(|n| **n == name).count() < 4 {
                seen.push(name.clone());
                tracing::warn!("metal: an indirect dispatch past the most groups ran nothing: {name} {asked:?}");
            }
        }
    }
    out
}
static REPORTED: std::sync::Mutex<std::collections::BTreeSet<u64>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());
/// Slow and in-flight lines printed so far (each capped at 100).
static TRACE_LINES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static FLIGHT_LINES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Logs each command buffer in flight for over 3 s, once (the locks are not held while logging),
/// and returns how many are in flight. Only `VICTORY_TRACE_GPU_FAILS=1` records them. Called at
/// each submission, and by the host when work stays held: nothing is submitted then, so a
/// command buffer the GPU never finishes would otherwise go unreported.
pub fn report_in_flight() -> usize {
    let (count, old): (usize, Vec<(u64, f64, String)>) = {
        let flying = IN_FLIGHT.lock().unwrap();
        let old = flying
            .iter()
            .filter(|(_, (t, _))| t.elapsed().as_secs_f64() > 3.0)
            .map(|(s, (t, d))| (*s, t.elapsed().as_secs_f64(), d.clone()))
            .collect();
        (flying.len(), old)
    };
    for (s, secs, d) in old {
        if REPORTED.lock().unwrap().insert(s)
            && FLIGHT_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 100
        {
            tracing::warn!("metal: command buffer in flight for {secs:.1}s: {d}{}", gated_args(s, false));
        }
    }
    count
}

/// One `tracing::warn!` line for a failed command buffer: its label and each encoder's state.
fn trace_failure(cb: &ProtocolObject<dyn MTLCommandBuffer>, e: &objc2_foundation::NSError) {
    let label = cb.label().map(|l| l.to_string()).unwrap_or_default();
    let mut encoders = String::new();
    // SAFETY: the key is a Metal constant; the value is an NSArray of encoder infos.
    let infos = e.userInfo().objectForKey(unsafe { MTLCommandBufferEncoderInfoErrorKey });
    if let Some(infos) = infos {
        // SAFETY: Metal documents the value as an array of `MTLCommandBufferEncoderInfo`.
        let infos: Retained<objc2_foundation::NSArray<ProtocolObject<dyn MTLCommandBufferEncoderInfo>>> =
            unsafe { Retained::cast_unchecked(infos) };
        for i in infos.iter() {
            let st = match i.errorState() {
                MTLCommandEncoderErrorState::Completed => "Completed",
                MTLCommandEncoderErrorState::Pending => "Pending",
                MTLCommandEncoderErrorState::Faulted => "Faulted",
                MTLCommandEncoderErrorState::Affected => "Affected",
                _ => "Unknown",
            };
            // Metal leaves an unlabelled encoder's label nil, which the binding's `label()`
            // declares never is (it panicked here after a failed command buffer).
            // SAFETY: `label` is an `MTLCommandBufferEncoderInfo` property returning an NSString.
            let label: Option<Retained<objc2_foundation::NSString>> =
                unsafe { objc2::msg_send![&*i, label] };
            let label = label.map(|l| l.to_string()).unwrap_or_default();
            encoders.push_str(&format!(" | [{st}] {label}"));
        }
    }
    tracing::warn!(
        "metal: command buffer failed: [{label}] now={:.3} ({}){encoders}",
        crate::metal::since_start(),
        e.localizedDescription()
    );
}

/// Caps an encoder label.
fn cap_label_str(mut s: String) -> String {
    if s.len() > 600 {
        let mut n = 600;
        while !s.is_char_boundary(n) {
            n -= 1;
        }
        s.truncate(n);
    }
    s
}

impl Metal {
    /// The capped label for an encoder, kept for the command buffer's description.
    /// The command buffer's description keeps the whole label when `VICTORY_TRACE_COMMITS` logs it
    /// (a capped one hid the end of a 17-dispatch pass, #79).
    fn cap_label(&self, s: String) -> Retained<objc2_foundation::NSString> {
        let capped = cap_label_str(s.clone());
        let n = objc2_foundation::NSString::from_str(&capped);
        self.trace_enc.borrow_mut().push(if commits_traced() { s } else { capped });
        n
    }

    /// The commit line of [`commits_traced`]: this command buffer's encoders, which are then
    /// forgotten so the next piece lists only its own.
    fn trace_commit(&self, serial: u64, queue_id: usize, what: &str) {
        if commits_traced() {
            let enc = std::mem::take(&mut *self.trace_enc.borrow_mut()).join(" | ");
            tracing::warn!("cb commit #{serial} q{queue_id:x} t={:.3} {what} | {enc}", crate::metal::since_start());
        }
    }

    fn encode_blit(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        ops: &[BlitOp<'_, Metal>],
        order: Ordering<'_>,
    ) -> Result<(), String> {
        let nonzero = |op: &BlitOp<'_, Metal>| match op {
            BlitOp::Copy { len, .. } | BlitOp::Fill { len, .. } => *len > 0,
            BlitOp::CopyTexture { size, .. } => !size.contains(&0),
            BlitOp::CopyBufferToTexture { len, size, .. }
            | BlitOp::CopyTextureToBuffer { len, size, .. } => *len > 0 && !size.contains(&0),
        };
        if !ops.iter().any(nonzero) {
            return Ok(());
        }
        let e = cb.blitCommandEncoder().ok_or("no blit encoder")?;
        if let Some((f, started)) = order
            && started.get()
        {
            e.waitForFence(f);
        }
        let off = |b: &Buffer, p: *const u8| p as usize - b.base as usize;
        let tex_dims = |t: &ProtocolObject<dyn MTLTexture>, mip: usize| {
            [(t.width() >> mip).max(1), (t.height() >> mip).max(1), (t.depth() >> mip).max(1)]
        };
        let tex_slices = |t: &ProtocolObject<dyn MTLTexture>| {
            if t.textureType() == MTLTextureType::Type2DArray { t.arrayLength() } else { 1 }
        };
        let fits_region = |t: &ProtocolObject<dyn MTLTexture>, mip: usize, slice: usize, at: [usize; 3], sz: [usize; 3]| {
            if sz.contains(&0) || mip >= t.mipmapLevelCount() || slice >= tex_slices(t) {
                return false;
            }
            let dims = tex_dims(t, mip);
            (0..3).all(|i| at[i].saturating_add(sz[i]) <= dims[i])
        };
        let fits_buffer_transfer = |what: &str, buf: (&Buffer, *const u8), t: &Texture, mip: usize, slice: usize, at: [usize; 3], sz: [usize; 3], row_pitch: usize, slice_pitch: usize, len: usize| -> Option<(usize, usize)> {
            // A depth and stencil texture's rows are its 4-byte depth plane (`DepthFromDepthStencil`).
            let plane = t.format == Format::D32FloatS8;
            let align = if plane { 4 } else { t.format.block().1 };
            let buf_off = off(buf.0, buf.1);
            let rows_fit = if plane {
                (sz[2] - 1) * slice_pitch + (sz[1] - 1) * row_pitch + sz[0] * 4 <= len && row_pitch >= sz[0] * 4
            } else {
                transfer_fits(what, t, mip, sz, row_pitch, slice_pitch, len)
            };
            if (is_depth(t.tex.pixelFormat()) && !matches!(t.format, Format::D32FloatS8 | Format::D32Float | Format::D16Unorm))
                || !fits_region(&t.tex, mip, slice, at, sz)
                || buf_off % align != 0
                || row_pitch % align != 0
                || buf_off.saturating_add(len) > buf.0.buf.length()
                || !rows_fit
            {
                return None;
            }
            let slice_bytes = if sz[2] > 1 || t.tex.textureType() == MTLTextureType::Type3D {
                slice_pitch
            } else {
                0
            };
            (slice_bytes % align == 0).then_some((buf_off, slice_bytes))
        };
        for op in ops.iter().filter(|op| nonzero(op)) {
            match op {
                BlitOp::Copy { src, dst, len } => {
                    // SAFETY: both ranges lie inside live buffers (the caller resolved them).
                    unsafe {
                        e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &src.0.buf,
                            off(src.0, src.1),
                            &dst.0.buf,
                            off(dst.0, dst.1),
                            *len,
                        )
                    };
                }
                BlitOp::Fill { dst, len, value } => {
                    let o = off(dst.0, dst.1);
                    e.fillBuffer_range_value(&dst.0.buf, NSRange::new(o, *len), *value);
                }
                BlitOp::CopyTexture {
                    src,
                    src_mip,
                    src_slice,
                    src_origin,
                    size: sz,
                    dst,
                    dst_mip,
                    dst_slice,
                    dst_origin,
                } => {
                    if !direct_copy(src.format, dst.format) {
                        // Block-compressed against plain texels of the block's size, or a depth
                        // plane against a plain 4-byte format: no view joins them, so rows go
                        // through a buffer of our own (the command buffer keeps it alive). `sz` is
                        // in source texels.
                        let ((sw, sb), (dw, db)) = (src.format.block(), dst.format.block());
                        let bytes = |f: Format, b: usize| if f == Format::D32FloatS8 { 4 } else { b };
                        if bytes(src.format, sb) != bytes(dst.format, db) || !fits_region(&src.tex, *src_mip, *src_slice, *src_origin, *sz) {
                            continue;
                        }
                        let (row, rows) = (sz[0].div_ceil(sw) * bytes(src.format, sb), sz[1].div_ceil(sw));
                        let dims = tex_dims(&dst.tex, *dst_mip);
                        let d: [usize; 3] = std::array::from_fn(|i| match i {
                            2 => sz[2],
                            _ if dw > sw => (sz[i] * dw).min(dims[i].saturating_sub(dst_origin[i])),
                            _ if sw > dw => sz[i].div_ceil(sw),
                            _ => sz[i],
                        });
                        if !fits_region(&dst.tex, *dst_mip, *dst_slice, *dst_origin, d) {
                            continue;
                        }
                        let Some(scratch) = self.dev.newBufferWithLength_options((row * rows * sz[2]).max(16), MTLResourceOptions::StorageModePrivate) else {
                            continue;
                        };
                        let pitch = if sz[2] > 1 { row * rows } else { 0 };
                        // SAFETY: the regions lie inside both subresources (checked above) and the
                        // buffer holds their rows.
                        unsafe {
                            e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                                &src.tex, *src_slice, *src_mip, origin(*src_origin), size(*sz), &scratch, 0, row, pitch, depth_plane(src),
                            );
                            e.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                                &scratch, 0, row, pitch, size(d), &dst.tex, *dst_slice, *dst_mip, origin(*dst_origin), depth_plane(dst),
                            );
                        }
                        continue;
                    }
                    if !fits_region(&src.tex, *src_mip, *src_slice, *src_origin, *sz)
                        || !fits_region(&dst.tex, *dst_mip, *dst_slice, *dst_origin, *sz)
                    {
                        continue;
                    }
                    let view = if src.tex.pixelFormat() == dst.tex.pixelFormat() {
                        None
                    } else if !is_depth(src.tex.pixelFormat())
                        && !is_depth(dst.tex.pixelFormat())
                        && castable(src.format, dst.format)
                        && src.tex.textureType() == dst.tex.textureType()
                    {
                        src.tex.newTextureViewWithPixelFormat(dst.tex.pixelFormat())
                    } else {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| {
                            tracing::warn!(
                                src = ?src.format,
                                dst = ?dst.format,
                                "metal: skipping incompatible CopyTexture formats"
                            );
                        });
                        continue;
                    };
                    if src.tex.pixelFormat() != dst.tex.pixelFormat() && view.is_none() {
                        continue;
                    }
                    let src_tex: &ProtocolObject<dyn MTLTexture> = view.as_deref().unwrap_or(&*src.tex);
                    // SAFETY: bounds, mip/slice ranges, and equal pixel formats are checked above.
                    unsafe {
                        e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                            src_tex,
                            *src_slice,
                            *src_mip,
                            origin(*src_origin),
                            size(*sz),
                            &dst.tex,
                            *dst_slice,
                            *dst_mip,
                            origin(*dst_origin),
                        )
                    };
                }
                BlitOp::CopyBufferToTexture {
                    src,
                    len,
                    row_pitch,
                    slice_pitch,
                    size: sz,
                    dst,
                    dst_mip,
                    dst_slice,
                    dst_origin,
                } => {
                    let Some((src_off, slice_bytes)) = fits_buffer_transfer(
                        "copy_buffer_to_texture",
                        *src,
                        dst,
                        *dst_mip,
                        *dst_slice,
                        *dst_origin,
                        *sz,
                        *row_pitch,
                        *slice_pitch,
                        *len,
                    ) else {
                        continue;
                    };
                    // SAFETY: buffer bounds, pitches, alignments, and texture region are validated above.
                    unsafe {
                        e.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                            &src.0.buf,
                            src_off,
                            *row_pitch,
                            slice_bytes,
                            size(*sz),
                            &dst.tex,
                            *dst_slice,
                            *dst_mip,
                            origin(*dst_origin),
                            depth_plane(dst),
                        )
                    };
                }
                BlitOp::CopyTextureToBuffer {
                    src,
                    src_mip,
                    src_slice,
                    src_origin,
                    size: sz,
                    dst,
                    len,
                    row_pitch,
                    slice_pitch,
                } => {
                    let Some((dst_off, slice_bytes)) = fits_buffer_transfer(
                        "copy_texture_to_buffer",
                        *dst,
                        src,
                        *src_mip,
                        *src_slice,
                        *src_origin,
                        *sz,
                        *row_pitch,
                        *slice_pitch,
                        *len,
                    ) else {
                        continue;
                    };
                    // SAFETY: buffer bounds, pitches, alignments, and texture region are validated above.
                    unsafe {
                        e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                            &src.tex,
                            *src_slice,
                            *src_mip,
                            origin(*src_origin),
                            size(*sz),
                            &dst.0.buf,
                            dst_off,
                            *row_pitch,
                            slice_bytes,
                            depth_plane(src),
                        )
                    };
                }
            }
        }
        if let Some((f, started)) = order {
            e.updateFence(f);
            started.set(true);
        }
        if self.trace_fails {
            let fills = ops.iter().filter(|o| matches!(o, BlitOp::Fill { .. })).count();
            e.setLabel(Some(&self.cap_label(format!("blit: {} ops ({fills} fills)", ops.len()))));
        }
        e.endEncoding();
        Ok(())
    }
}

/// The blit option a copy between `t` and buffer rows takes: a depth and stencil texture's rows
/// are its depth plane.
fn depth_plane(t: &Texture) -> MTLBlitOption {
    if t.format == Format::D32FloatS8 { MTLBlitOption::DepthFromDepthStencil } else { MTLBlitOption::None }
}

/// Whether a texture copies into another with one texture-to-texture blit: the same format, or
/// two plain formats of one texel size (through a view).
pub fn direct_copy(a: Format, b: Format) -> bool {
    let plain = |f: Format| f.block().0 == 1 && !f.is_depth();
    a == b || (plain(a) && plain(b) && a.block() == b.block())
}

/// `ExecuteIndirect` with a count buffer runs record `index` only while `index < count`, both read
/// on the GPU: this one-thread kernel copies the record's `words`, or zeros (no groups, no
/// vertices or instances), into a buffer of its own, which the indirect command then reads.
/// A rectangle list's record (`DrawCall::Rects`) becomes the four-vertex strip instances a direct
/// one draws: (4, vertices / 3, 0, first vertex / 3).
/// A dispatch's record, or a mesh dispatch's, runs nothing if it asks for more than 2^22 groups in
/// all: such a record is what its writer left unfinished in title memory, and Metal would run it
/// (510 x 1,182,027,346 x 1,314,076,063 groups in GTA VI's world: that command buffer never
/// ended, and every one behind it on the queue timed out). A guard of ours, not a console rule:
/// the title's own records ask for far less (9,192 groups at most seen here), but for more than
/// D3D12's 65,535 in one dimension on larger targets, so the limit is on the product.
/// Bound at 20 and up, apart from the converter's slots. `at`: the record's index, its words,
/// whether a count applies, 1 for a rectangle list's record or 2 for a dispatch's or a mesh
/// dispatch's (six words out: the three it runs, then the three the record asked for, for
/// `VICTORY_TRACE_GPU_FAILS`).
const GATE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void gate(device const uint *count [[buffer(20)]],
                 device const uint *record [[buffer(21)]],
                 device uint *out [[buffer(22)]],
                 constant uint4 &at [[buffer(23)]]) {
    bool run = at.z == 0u || at.x < count[0];
    if (at.w == 2u) {
        // Two steps: the product of three words does not fit 64 bits.
        ulong plane = ulong(record[0]) * record[1];
        run = run && plane <= 4194304ul && plane * record[2] <= 4194304ul;
        for (uint w = 0; w < 3u; w++) {
            out[w] = run ? record[w] : 0u;
            out[3u + w] = record[w];
        }
        return;
    }
    for (uint w = 0; w < at.y; w++) {
        out[w] = run ? record[w] : 0u;
    }
    if (at.w == 1u) {
        // shortcut: an instanced rectangle list draws nothing (the rectangle translation takes the
        // instance for the rectangle); draw its instances when a title is seen to use them.
        out[0] = 4u;
        out[1] = run && record[1] == 1u ? record[0] / 3u : 0u;
        out[2] = 0u;
        out[3] = run ? record[2] / 3u : 0u;
    }
}
"#;

impl Metal {
    fn gate_pipeline(&self) -> Result<&ProtocolObject<dyn MTLComputePipelineState>, String> {
        self.gate
            .get_or_init(|| {
                let lib = self
                    .dev
                    .newLibraryWithSource_options_error(
                        &objc2_foundation::NSString::from_str(GATE),
                        None,
                    )
                    .map_err(|e| e.localizedDescription().to_string())?;
                let f = lib
                    .newFunctionWithName(&objc2_foundation::NSString::from_str("gate"))
                    .ok_or("no gate function")?;
                self.dev
                    .newComputePipelineStateWithFunction_error(&f)
                    .map_err(|e| e.localizedDescription().to_string())
            })
            .as_deref()
            .map_err(Clone::clone)
    }

    /// Encodes the gate of one counted (`count`: buffer, address, index), rectangle-list (`kind` 1)
    /// or dispatch or mesh dispatch (`kind` 2) record (`words` long at `record`) into `e`; returns the
    /// buffer the indirect command reads instead of the record. Tracked by Metal (unlike the
    /// title's pools), so whatever reads it later in the command buffer waits for the write.
    /// shortcut: one gate dispatch and one small buffer per record; gate all the records of one
    /// `ExecuteIndirect` in a single dispatch if a large `MaxCommandCount` shows up in profiles.
    fn encode_gate(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        count: Option<(&Buffer, *const u8, u32)>,
        (record, title): (&Buffer, *const u8),
        words: u32,
        kind: u32,
        set_on_queue: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let out = self
            .dev
            .newBufferWithLength_options(
                // A dispatch's gate also keeps what the record asked for.
                if kind == 2 { 24 } else { words as usize * 4 },
                // Tracing reads the record the GPU used back (`GATED`).
                if self.trace_fails { MTLResourceOptions::StorageModeShared } else { MTLResourceOptions::StorageModePrivate },
            )
            .ok_or("no buffer for a gated indirect record")?;
        let counted = u32::from(count.is_some());
        // Without a count, slot 20 is bound to the record (not read).
        let (count, at, index) = count.unwrap_or((record, title, 0));
        for b in [count, record] {
            if self.names(b, set_on_queue) {
                e.useResource_usage(ProtocolObject::from_ref(&*b.buf), MTLResourceUsage::Read);
            }
        }
        // What the work before wrote (the count, the record) is seen.
        e.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        e.setComputePipelineState(self.gate_pipeline()?);
        let args = [index, words, counted, kind];
        // SAFETY: both title ranges lie inside live buffers (the caller resolved them); the bytes
        // are copied before the call returns.
        unsafe {
            e.setBuffer_offset_atIndex(Some(&count.buf), at as usize - count.base as usize, 20);
            e.setBuffer_offset_atIndex(
                Some(&record.buf),
                title as usize - record.base as usize,
                21,
            );
            e.setBuffer_offset_atIndex(Some(&out), 0, 22);
            e.setBytes_length_atIndex(
                NonNull::new(args.as_ptr().cast_mut().cast()).unwrap(),
                16,
                23,
            );
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(size([1, 1, 1]), size([1, 1, 1]));
        e.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        Ok(out)
    }

    /// Whether encoder `use` calls are needed for `b`: not when the queue has the pool residency set
    /// and the buffer is in it.
    fn names(&self, b: &Buffer, set_on_queue: bool) -> bool {
        let needed = !(set_on_queue && b.is_resident());
        if needed {
            self.count(|c| c.use_resource += 1);
        }
        needed
    }

    pub(crate) fn encode_compute(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        dispatches: &[crate::backend::Dispatch<'_, Metal>],
        set_on_queue: bool,
        order: Ordering<'_>,
        attached_heaps: &mut Vec<*const Heap>,
    ) -> Result<(), String> {
        let runs = |d: &crate::backend::Dispatch<'_, Metal>| {
            d.indirect.is_some() || !d.groups.contains(&0)
        };
        if !dispatches.iter().any(runs) {
            return Ok(());
        }
        let e = cb.computeCommandEncoder().ok_or("no compute encoder")?;
        if let Some((f, started)) = order
            && started.get()
        {
            e.waitForFence(f);
        }
        let mut named = Vec::new();
        for d in dispatches.iter().filter(|d| runs(d)) {
            // Every indirect record goes through the gate: the count buffer's limit, and the most
            // groups a record may ask for.
            let gated = match d.indirect {
                // An open encoder left behind makes Metal abort at the next pass's encoder.
                Some(record) => Some(
                    self.encode_gate(&e, d.count, record, 3, 2, set_on_queue)
                        .inspect_err(|_| e.endEncoding())?,
                ),
                None => None,
            };
            if let (true, Some(g)) = (self.trace_fails, &gated) {
                let k = Retained::as_ptr(&d.pipeline.0) as *const u8 as usize;
                let name = self.pipe_names.borrow().get(&k).cloned().unwrap_or_else(|| format!("{k:#x}"));
                self.trace_args.borrow_mut().push((name, g.contents().as_ptr() as usize, 6));
            }
            e.setComputePipelineState(&d.pipeline.0);
            for h in [d.heaps.resource, d.heaps.sampler].into_iter().flatten() {
                self.use_heap_cb(h, attached_heaps, |set| cb.useResidencySet(set));
            }
            self.bind_root_compute(&e, &d.root, &d.heaps, d.pipeline.1.as_ref(), &mut named);
            for b in d.resources.iter().filter(|b| self.names(b, set_on_queue)) {
                e.useResource_usage(
                    ProtocolObject::from_ref(&*b.buf),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
            match &gated {
                // SAFETY: the gate wrote three words to its own buffer.
                Some(g) => unsafe {
                    e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                        g,
                        0,
                        size(d.threads_per_group),
                    )
                },
                None => e.dispatchThreadgroups_threadsPerThreadgroup(size(d.groups), size(d.threads_per_group)),
            }
        }
        if let Some((f, started)) = order {
            e.updateFence(f);
            started.set(true);
        }
        if self.trace_fails {
            let names = self.pipe_names.borrow();
            let mut l = String::from("compute: ");
            for d in dispatches.iter().filter(|d| runs(d)) {
                let k = Retained::as_ptr(&d.pipeline.0) as *const u8 as usize;
                match names.get(&k) {
                    Some(n) => l.push_str(n),
                    None => l.push_str(&format!("{k:#x}")),
                }
                match d.indirect {
                    // The record as title memory holds it now: what earlier command buffers
                    // left there, not what this one's earlier passes will write.
                    Some((_, title)) => {
                        // SAFETY: the caller resolved three words at `title` inside the live buffer.
                        let w: [u32; 3] = unsafe { title.cast::<[u32; 3]>().read_unaligned() };
                        l.push_str(&format!(" indirect{w:?}; "));
                    }
                    None => l.push_str(&format!(" {}x{}x{}; ", d.groups[0], d.groups[1], d.groups[2])),
                }
            }
            e.setLabel(Some(&self.cap_label(l)));
        }
        e.endEncoding();
        Ok(())
    }

    fn encode_color_clear_rects(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        target: &Texture,
        color: [f32; 4],
        rects: &[ClearRect],
        set_on_queue: bool,
        order: Ordering<'_>,
        attached_heaps: &mut Vec<*const Heap>,
    ) -> Result<(), String> {
        let clip = |p, limit| usize::try_from(p).unwrap_or_default().min(limit);
        let (width, height) = (target.tex.width(), target.tex.height());
        let regions: Vec<_> = rects
            .iter()
            .filter_map(|r| {
                let (left, right) = (clip(r.left, width), clip(r.right, width));
                let (top, bottom) = (clip(r.top, height), clip(r.bottom, height));
                (left < right && top < bottom).then(|| ([left, top, 0], [right - left, bottom - top, 1]))
            })
            .collect();
        if regions.is_empty() {
            return Ok(());
        }
        if !matches!(target.tex.textureType(), MTLTextureType::Type2D | MTLTextureType::Type2DArray)
            || target.tex.sampleCount() != 1
        {
            return Err("rectangular color clear requires a single-sampled 2D color target".into());
        }
        if target.format.is_depth()
            || !target.tex.usage().contains(MTLTextureUsage::RenderTarget)
        {
            return Err("rectangular color clear requires a renderable color format".into());
        }

        // Use Metal's clear conversion in the view format, including sRGB and integer targets.
        // A retained command buffer keeps this private scratch alive until the blits finish.
        let (max_width, max_height) = regions.iter().fold((0, 0), |(w, h), (_, size)| {
            (w.max(size[0]), h.max(size[1]))
        });
        let td = MTLTextureDescriptor::new();
        td.setPixelFormat(target.tex.pixelFormat());
        td.setStorageMode(MTLStorageMode::Private);
        td.setUsage(MTLTextureUsage::RenderTarget);
        // SAFETY: these nonzero dimensions fit the selected view.
        unsafe {
            td.setWidth(max_width);
            td.setHeight(max_height);
        }
        let scratch = Texture {
            tex: self.dev.newTextureWithDescriptor(&td).ok_or("no rectangular clear texture")?,
            format: target.format,
        };
        self.encode_render(
            cb,
            &RenderPass {
                color: vec![ColorTarget { texture: &scratch, clear: Some(color) }],
                depth: None,
                draws: vec![],
            },
            set_on_queue,
            order,
            attached_heaps,
        )?;
        let copies: Vec<_> = regions
            .into_iter()
            .map(|(dst_origin, size)| BlitOp::CopyTexture {
                src: &scratch,
                src_mip: 0,
                src_slice: 0,
                src_origin: [0, 0, 0],
                size,
                dst: target,
                dst_mip: 0,
                dst_slice: 0,
                dst_origin,
            })
            .collect();
        self.encode_blit(cb, &copies, order)
    }

    fn encode_render(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        rp: &RenderPass<'_, Metal>,
        set_on_queue: bool,
        order: Ordering<'_>,
        attached_heaps: &mut Vec<*const Heap>,
    ) -> Result<(), String> {
        // Attachments must agree in sample count. Their sizes may differ (a 1920x1080 colour
        // target over a 3840x2160 depth target in GTA VI's story load): the console draws into
        // the area all of them cover, and so does the pass, constrained to that intersection.
        let mut samples = None;
        let mut rt_size = (usize::MAX, usize::MAX);
        for t in rp
            .color
            .iter()
            .map(|c| &c.texture.tex)
            .chain(rp.depth.as_ref().map(|d| &d.texture.tex))
        {
            let n = t.sampleCount();
            if let Some(want) = samples
                && want != n
            {
                return Err(format!("render targets of different sample counts: {want} and {n}"));
            }
            samples = Some(n);
            rt_size = (rt_size.0.min(t.width()), rt_size.1.min(t.height()));
        }
        if rp.color.is_empty() && rp.depth.is_none() {
            // The title drew with no render targets. D3D12 runs it (it writes nothing visible);
            // Metal needs an attachment to make an encoder, and Metal refuses one without. The
            // pass is skipped rather than refused, so the rest of the submission still runs.
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "metal: a pass with no render targets is skipped ({} draws, writes nothing)",
                    rp.draws.len()
                );
            });
            return Ok(());
        }
        let desc = MTLRenderPassDescriptor::new();
        desc.setRenderTargetWidth(rt_size.0);
        desc.setRenderTargetHeight(rt_size.1);
        for (i, c) in rp.color.iter().enumerate() {
            // SAFETY: colour attachment index below 8.
            let a = unsafe { desc.colorAttachments().objectAtIndexedSubscript(i) };
            a.setTexture(Some(&c.texture.tex));
            a.setStoreAction(MTLStoreAction::Store);
            match c.clear {
                Some(v) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearColor(MTLClearColor {
                        red: v[0] as f64,
                        green: v[1] as f64,
                        blue: v[2] as f64,
                        alpha: v[3] as f64,
                    });
                }
                None => a.setLoadAction(MTLLoadAction::Load),
            }
        }
        // Counted indirect draws, indirect rectangle lists and indirect mesh dispatches (for the
        // most groups they may ask for) are gated first, in a compute pass of their own; when
        // tracing, every indirect draw is, so what the GPU read is known.
        let gated = |d: &Draw<'_, Metal>| {
            d.indirect.is_some()
                && (d.count.is_some() || matches!(d.call, DrawCall::Rects | DrawCall::Mesh { .. }) || self.trace_fails)
        };
        let mut gates = Vec::new();
        if rp.draws.iter().any(gated) {
            let c = cb.computeCommandEncoder().ok_or("no compute encoder")?;
            if let Some((f, started)) = order
                && started.get()
            {
                c.waitForFence(f);
            }
            for d in &rp.draws {
                let words = match d.call {
                    DrawCall::Indexed { .. } => 5,
                    DrawCall::Arrays { .. } | DrawCall::Rects => 4,
                    DrawCall::Mesh { .. } => 3,
                };
                let kind = match d.call {
                    DrawCall::Rects => 1,
                    DrawCall::Mesh { .. } => 2,
                    _ => 0,
                };
                gates.push(match d.indirect {
                    Some(record) if gated(d) => {
                        let g = self.encode_gate(&c, d.count, record, words, kind, set_on_queue).inspect_err(|_| c.endEncoding())?;
                        if self.trace_fails {
                            let name = format!("{} {}", if kind == 2 { "mesh draw" } else { "draw" }, gates.len());
                            self.trace_args.borrow_mut().push((name, g.contents().as_ptr() as usize, if kind == 2 { 6 } else { words as usize }));
                        }
                        Some(g)
                    }
                    _ => None,
                });
            }
            if let Some((f, started)) = order {
                c.updateFence(f);
                started.set(true);
            }
            c.endEncoding();
        }
        if let Some(dp) = &rp.depth {
            let a = desc.depthAttachment();
            a.setTexture(Some(&dp.texture.tex));
            a.setStoreAction(MTLStoreAction::Store);
            match dp.clear {
                Some(v) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearDepth(v as f64);
                }
                None => a.setLoadAction(MTLLoadAction::Load),
            }
            // A combined depth and stencil texture is attached as both.
            if dp.texture.tex.pixelFormat() == MTLPixelFormat::Depth32Float_Stencil8 {
                let a = desc.stencilAttachment();
                a.setTexture(Some(&dp.texture.tex));
                a.setStoreAction(MTLStoreAction::Store);
                match dp.clear_stencil {
                    Some(v) => {
                        a.setLoadAction(MTLLoadAction::Clear);
                        a.setClearStencil(u32::from(v));
                    }
                    None => a.setLoadAction(MTLLoadAction::Load),
                }
            }
        }
        let e = cb.renderCommandEncoderWithDescriptor(&desc).ok_or_else(|| {
            // Metal refuses an encoder whose attachments disagree (size, sample count, format
            // kind); the caller's log then says which targets the pass named.
            let size = |t: &ProtocolObject<dyn MTLTexture>| {
                format!("{}x{} fmt {}", t.width(), t.height(), t.pixelFormat().0)
            };
            let color: Vec<String> = rp.color.iter().map(|c| size(&c.texture.tex)).collect();
            let depth = rp.depth.as_ref().map(|d| size(&d.texture.tex));
            format!(
                "no render encoder: {} colour {color:?}, depth {depth:?}, {} draws",
                rp.color.len(),
                rp.draws.len()
            )
        })?;
        if let Some((f, started)) = order
            && started.get()
        {
            e.waitForFence_beforeStages(f, MTLRenderStages::Vertex | MTLRenderStages::Object | MTLRenderStages::Mesh);
        }
        let mut named = Vec::new();
        let mut depth_set = false;
        let mut default_depth = None;
        for (k, d) in rp.draws.iter().enumerate() {
            // The stage that reads the draw's buffers before the pixel shader.
            let first = match (d.pipeline.mesh, d.call) {
                (Some(_), DrawCall::Mesh { .. }) => MTLRenderStages::Mesh,
                (None, DrawCall::Arrays { .. } | DrawCall::Indexed { .. } | DrawCall::Rects) => MTLRenderStages::Vertex,
                _ => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| eprintln!("metal: a mesh draw of a vertex pipeline, or the reverse, is dropped"));
                    continue;
                }
            };
            if d.indirect.is_none() && empty_draw(&d.call) {
                continue;
            }
            for h in [d.heaps.resource, d.heaps.sampler].into_iter().flatten() {
                self.use_heap_cb(h, attached_heaps, |set| cb.useResidencySet(set));
            }
            e.setRenderPipelineState(&d.pipeline.pso);
            e.setCullMode(d.pipeline.cull);
            let r = &d.pipeline.raster;
            e.setFrontFacingWinding(if r.front_ccw {
                MTLWinding::CounterClockwise
            } else {
                MTLWinding::Clockwise
            });
            e.setTriangleFillMode(if r.wireframe {
                MTLTriangleFillMode::Lines
            } else {
                MTLTriangleFillMode::Fill
            });
            e.setDepthClipMode(if r.depth_clip {
                MTLDepthClipMode::Clip
            } else {
                MTLDepthClipMode::Clamp
            });
            e.setDepthBias_slopeScale_clamp(r.depth_bias.0, r.depth_bias.1, r.depth_bias.2);
            // Metal validation forbids passing nil to `setDepthStencilState:`: bind a default
            // state object when returning from a depth/stencil pipeline to one without.
            if let Some(depth) = d.pipeline.depth.as_deref() {
                e.setDepthStencilState(Some(depth));
                depth_set = true;
            } else if depth_set {
                let depth = default_depth.get_or_insert_with(|| {
                    self.dev
                        .newDepthStencilStateWithDescriptor(&MTLDepthStencilDescriptor::new())
                        .expect("default depth state")
                });
                e.setDepthStencilState(Some(depth));
                depth_set = false;
            }
            if d.pipeline.stencil {
                e.setStencilReferenceValue(d.state.stencil_ref);
            }
            let [r, g, b, a] = d.state.blend_factor;
            e.setBlendColorRed_green_blue_alpha(r, g, b, a);
            // The target's size bounds the scissor (Metal refuses a rectangle outside it).
            let (tw, th) = rt_size;
            let viewport = match d.state.viewport {
                Some(v) if v.width <= 0.0 || v.height <= 0.0 => continue,
                Some(v) => MTLViewport {
                    originX: v.x as f64,
                    originY: v.y as f64,
                    width: v.width as f64,
                    height: v.height as f64,
                    znear: v.min_depth as f64,
                    zfar: v.max_depth as f64,
                },
                None => MTLViewport {
                    originX: 0.0,
                    originY: 0.0,
                    width: tw as f64,
                    height: th as f64,
                    znear: 0.0,
                    zfar: 1.0,
                },
            };
            e.setViewport(viewport);
            let scissor = match d.state.scissor {
                Some([l, t, r, b]) => {
                    let (l, t) = (l.clamp(0, tw as i32) as usize, t.clamp(0, th as i32) as usize);
                    let (r, b) = (r.clamp(0, tw as i32) as usize, b.clamp(0, th as i32) as usize);
                    MTLScissorRect {
                        x: l,
                        y: t,
                        width: r.saturating_sub(l),
                        height: b.saturating_sub(t),
                    }
                }
                None => MTLScissorRect {
                    x: 0,
                    y: 0,
                    width: tw,
                    height: th,
                },
            };
            if scissor.width == 0 || scissor.height == 0 {
                // Nothing of the target is inside the rectangle: nothing is drawn.
                continue;
            }
            e.setScissorRect(scissor);
            self.bind_root_render(&e, &d.root, d.fragment_root.as_ref(), &d.heaps, d.pipeline, &mut named);
            for b in d.resources.iter().filter(|b| self.names(b, set_on_queue)) {
                e.useResource_usage_stages(
                    ProtocolObject::from_ref(&*b.buf),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                    first | MTLRenderStages::Fragment,
                );
            }
            for &(slot, b, title) in &d.vertex_buffers {
                let off = title as usize - b.base as usize;
                // SAFETY: the buffer is live and `off` is inside it.
                unsafe {
                    e.setVertexBuffer_offset_atIndex(Some(&b.buf), off, VERTEX_BUFFERS + slot)
                };
                if self.names(b, set_on_queue) {
                    e.useResource_usage_stages(
                        ProtocolObject::from_ref(&*b.buf),
                        MTLResourceUsage::Read,
                        MTLRenderStages::Vertex,
                    );
                }
            }
            let prim = match d.topology {
                Topology::Triangles => MTLPrimitiveType::Triangle,
                Topology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
                Topology::Lines => MTLPrimitiveType::Line,
                Topology::LineStrip => MTLPrimitiveType::LineStrip,
                Topology::Points => MTLPrimitiveType::Point,
            };
            // What `IRRuntimeDrawPrimitives` / `IRRuntimeDrawIndexedPrimitives` do: draw arguments
            // (`IRRuntimeDrawParams`, 5 words) at slot 4 and the index type at slot 5 (0 = not indexed,
            // else Metal's index type + 1), then the draw itself.
            if let Some((ab, title)) = d.indirect {
                // What the converter runtime's indirect `IRRuntimeDraw(Indexed)Primitives` do: the
                // argument record itself is the shader's draw arguments (D3D12's records are
                // `IRRuntimeDraw(Indexed)Argument`s and Metal's indirect ones, word for word).
                if self.names(ab, set_on_queue) {
                    e.useResource_usage_stages(
                        ProtocolObject::from_ref(&*ab.buf),
                        MTLResourceUsage::Read,
                        first,
                    );
                }
                // A counted record is read from its gate's buffer.
                let (args, at) = match gates.get(k).and_then(Option::as_ref) {
                    Some(g) => (&**g, 0),
                    None => (&*ab.buf, title as usize - ab.base as usize),
                };
                if let (Some(threads), DrawCall::Mesh { .. }) = (d.pipeline.mesh, d.call) {
                    // SAFETY: the record (three group counts) is inside the live buffer.
                    unsafe {
                        e.drawMeshThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                            args,
                            at,
                            size([1, 1, 1]),
                            threads,
                        )
                    };
                    continue;
                }
                // SAFETY: the record is inside the live buffer (the caller resolved it).
                unsafe { e.setVertexBuffer_offset_atIndex(Some(args), at, DRAW_ARGUMENTS) };
                match (d.call, d.index_buffer) {
                    (DrawCall::Arrays { .. } | DrawCall::Rects, _) => {
                        set_index_type(&e, 0);
                        // SAFETY: as above.
                        unsafe { e.drawPrimitives_indirectBuffer_indirectBufferOffset(prim, args, at) };
                    }
                    (DrawCall::Indexed { wide, .. }, Some((b, title))) => {
                        set_index_type(&e, if wide { 2 } else { 1 });
                        if self.names(b, set_on_queue) {
                            e.useResource_usage_stages(
                                ProtocolObject::from_ref(&*b.buf),
                                MTLResourceUsage::Read,
                                MTLRenderStages::Vertex,
                            );
                        }
                        // SAFETY: both buffers are live; the record's start index counts from `title`.
                        unsafe {
                            e.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                                prim,
                                if wide { MTLIndexType::UInt32 } else { MTLIndexType::UInt16 },
                                &b.buf,
                                title as usize - b.base as usize,
                                args,
                                at,
                            )
                        };
                    }
                    (DrawCall::Indexed { .. }, None) => {
                        e.endEncoding();
                        return Err("indexed draw without an index buffer".into());
                    }
                    (DrawCall::Mesh { .. }, _) => unreachable!("a mesh draw of a vertex pipeline"),
                }
                continue;
            }
            let mut dp = [0u32; 5];
            let ib: u16;
            match d.call {
                DrawCall::Mesh { groups } => {
                    let threads = d.pipeline.mesh.expect("a mesh pipeline");
                    let [x, y, z] = groups.map(|n| n as usize);
                    e.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                        size([x, y, z]),
                        size([1, 1, 1]),
                        threads,
                    );
                }
                DrawCall::Arrays {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    dp[..4].copy_from_slice(&[
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    ]);
                    ib = 0;
                    set_draw_bytes(&e, &dp, ib);
                    // SAFETY: arguments are plain counts.
                    unsafe {
                        e.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                            prim,
                            first_vertex as usize,
                            vertex_count as usize,
                            instance_count as usize,
                            first_instance as usize,
                        )
                    };
                }
                DrawCall::Rects => unreachable!("an indirect-only call, and so empty here"),
                DrawCall::Indexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                    wide,
                } => {
                    // A draw that names no index buffer is dropped, not the pass: returning here
                    // would leave the render encoder open and, now that a refused pass still
                    // commits (for its fence signals), Metal would assert on the open encoder.
                    let Some((b, title)) = d.index_buffer else {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| {
                            eprintln!("metal: an indexed draw without an index buffer is dropped")
                        });
                        continue;
                    };
                    let width = if wide { 4 } else { 2 };
                    let off = title as usize - b.base as usize + first_index as usize * width;
                    dp = [
                        index_count,
                        instance_count,
                        off as u32,
                        base_vertex as u32,
                        first_instance,
                    ];
                    ib = if wide { 2 } else { 1 };
                    set_draw_bytes(&e, &dp, ib);
                    if self.names(b, set_on_queue) {
                        e.useResource_usage_stages(
                            ProtocolObject::from_ref(&*b.buf),
                            MTLResourceUsage::Read,
                            MTLRenderStages::Vertex,
                        );
                    }
                    // SAFETY: the index buffer is live and holds `index_count` indices at `off`.
                    unsafe {
                        e.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                            prim,
                            index_count as usize,
                            if wide { MTLIndexType::UInt32 } else { MTLIndexType::UInt16 },
                            &b.buf,
                            off,
                            instance_count as usize,
                            base_vertex as isize,
                            first_instance as usize,
                        )
                    };
                }
            }
        }
        if let Some((f, started)) = order {
            e.updateFence_afterStages(f, MTLRenderStages::Fragment);
            started.set(true);
        }
        if self.trace_fails {
            // Each distinct pipeline once, with its draw count: names the pass's shaders.
            let names = self.pipe_names.borrow();
            let mut seen: Vec<(usize, usize)> = Vec::new();
            for d in &rp.draws {
                let k = Retained::as_ptr(&d.pipeline.pso) as *const u8 as usize;
                match seen.iter_mut().find(|(p, _)| *p == k) {
                    Some((_, n)) => *n += 1,
                    None => seen.push((k, 1)),
                }
            }
            let mut l = format!("render: {} draws", rp.draws.len());
            for (k, n) in seen {
                match names.get(&k) {
                    Some(name) => l.push_str(&format!("; {name} x{n}")),
                    None => l.push_str(&format!("; {k:#x} x{n}")),
                }
            }
            drop(names);
            e.setLabel(Some(&self.cap_label(l)));
        }
        e.endEncoding();
        Ok(())
    }
}

/// Whether a direct draw has a zero vertex, index, instance or mesh-group count.
fn empty_draw(call: &DrawCall) -> bool {
    match *call {
        DrawCall::Arrays { vertex_count, instance_count, .. } => {
            vertex_count == 0 || instance_count == 0
        }
        DrawCall::Indexed { index_count, instance_count, .. } => {
            index_count == 0 || instance_count == 0
        }
        DrawCall::Mesh { groups } => groups.contains(&0),
        // Only an indirect call.
        DrawCall::Rects => true,
    }
}

fn set_draw_bytes(e: &ProtocolObject<dyn MTLRenderCommandEncoder>, dp: &[u32; 5], index_type: u16) {
    // SAFETY: setVertexBytes copies the bytes before returning.
    unsafe {
        e.setVertexBytes_length_atIndex(
            NonNull::new(dp.as_ptr().cast_mut().cast()).unwrap(),
            20,
            DRAW_ARGUMENTS,
        );
    }
    set_index_type(e, index_type);
}

/// The converter's index type word (0 = not indexed, else Metal's index type + 1).
fn set_index_type(e: &ProtocolObject<dyn MTLRenderCommandEncoder>, index_type: u16) {
    // SAFETY: setVertexBytes copies the bytes before returning.
    unsafe {
        e.setVertexBytes_length_atIndex(
            NonNull::new((&raw const index_type).cast_mut().cast()).unwrap(),
            2,
            UNIFORMS,
        );
    }
}

/// Whether a pixel format is a depth or stencil one.
fn is_depth(f: MTLPixelFormat) -> bool {
    [
        MTLPixelFormat::Depth16Unorm,
        MTLPixelFormat::Depth32Float,
        MTLPixelFormat::Stencil8,
        MTLPixelFormat::Depth24Unorm_Stencil8,
        MTLPixelFormat::Depth32Float_Stencil8,
    ]
    .contains(&f)
}

#[cfg(test)]
mod transfer_tests {
    use super::*;

    #[test]
    fn transfers_are_checked_against_the_format() {
        // RGBA16Float 4x4: a row is 32 bytes.
        assert!(transfer_need(Format::Rgba16Float, [4, 4, 1], 16, 0, 1 << 20).is_err());
        assert_eq!(transfer_need(Format::Rgba16Float, [4, 4, 1], 32, 0, 128), Ok(128));
        // BC1 8x8: 2 block rows of 16 bytes; the last row needs no padding.
        assert_eq!(transfer_need(Format::Bc1, [8, 8, 1], 16, 0, 32), Ok(32));
        assert!(transfer_need(Format::Bc1, [8, 8, 1], 8, 0, 1 << 20).is_err());
        // Too short a buffer, and a 3D slice pitch below one image.
        assert!(transfer_need(Format::Rgba8Unorm, [4, 4, 1], 16, 0, 63).is_err());
        assert!(transfer_need(Format::Rgba8Unorm, [4, 4, 2], 16, 32, 1 << 20).is_err());
        assert_eq!(transfer_need(Format::Rgba8Unorm, [4, 4, 2], 16, 64, 128), Ok(128));
    }
}

#[cfg(test)]
mod render_target_tests {
    use super::*;
    use objc2_foundation::NSString;

    fn draw_green_triangles(metal: &Metal, target: &Texture, clear: Option<[f32; 4]>, viewports: &[Option<Viewport>]) {
        let _flight = FLIGHTS_IN_TESTS.read().unwrap_or_else(|e| e.into_inner());
        let library = metal.dev.newLibraryWithSource_options_error(&NSString::from_str(r#"
            #include <metal_stdlib>
            using namespace metal;
            vertex float4 test_vertex(uint i [[vertex_id]]) {
                return float4(i == 1 ? 3.0 : -1.0, i == 2 ? 3.0 : -1.0, 0.0, 1.0);
            }
            fragment float4 test_fragment() { return float4(0.0, 1.0, 0.0, 1.0); }
        "#), None).expect("compile generated Metal shaders");
        let pipeline_desc = MTLRenderPipelineDescriptor::new();
        let vertex = library.newFunctionWithName(&NSString::from_str("test_vertex")).unwrap();
        let fragment = library.newFunctionWithName(&NSString::from_str("test_fragment")).unwrap();
        pipeline_desc.setVertexFunction(Some(&vertex));
        pipeline_desc.setFragmentFunction(Some(&fragment));
        // SAFETY: the descriptor has a color attachment at index 0.
        unsafe { pipeline_desc.colorAttachments().objectAtIndexedSubscript(0) }
            .setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        let pipeline = RenderPipeline {
            pso: metal.dev.newRenderPipelineStateWithDescriptor_error(&pipeline_desc).unwrap(),
            depth: None,
            cull: MTLCullMode::None,
            raster: Raster::default(),
            stencil: false,
            mesh: None,
            vs_static: None,
            ps_static: None,
        };
        let queue = metal.create_queue().unwrap();
        let fence = metal.create_fence(0).unwrap();
        let passes = [Pass::Render(RenderPass {
            color: vec![ColorTarget { texture: target, clear }],
            depth: None,
            draws: viewports.iter().map(|&viewport| Draw {
                pipeline: &pipeline,
                topology: Topology::Triangles,
                state: DrawState { viewport, ..Default::default() },
                root: RootArgs::default(),
                fragment_root: None,
                heaps: Heaps::default(),
                resources: vec![],
                vertex_buffers: vec![],
                index_buffer: None,
                call: DrawCall::Arrays {
                    vertex_count: 3,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
                indirect: None,
                count: None,
            }).collect(),
        })];
        metal.submit_passes(&queue, &PassSubmission {
            passes: &passes,
            waits: &[],
            signals: &[(&fence, 1)],
        }).expect("draw green triangles");
        assert!(metal.wait(&fence, 1, 2000), "draw did not complete");
    }

    #[test]
    fn a_draw_into_a_narrowed_render_target_view_preserves_neighbors() {
        let metal = Metal::new().expect("Metal device");
        let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 8, 8);
        desc.dim = TexDim::D2Array;
        desc.depth_or_layers = 2;
        desc.mips = 2;
        desc.render_target = true;
        let texture = metal.create_texture(&desc).unwrap();
        let sentinel = |mip, slice| [17 + (mip * 40 + slice * 80) as u8, 31, 47, 255];
        for slice in 0..2 {
            for mip in 0..2 {
                let n = 8 >> mip;
                metal.upload_texture(&texture, mip, slice, &sentinel(mip, slice).repeat(n * n), n * 4, 0);
            }
        }
        let target = metal.create_texture_view(&texture, &ViewDesc {
            format: Format::Rgba8Unorm,
            mips: (1, 1),
            slices: (1, 1),
        }).expect("narrowed render target");
        assert_eq!((target.tex.width(), target.tex.height()), (4, 4));

        draw_green_triangles(&metal, &target, None, &[None]);
        for slice in 0..2 {
            for mip in 0..2 {
                let n = 8 >> mip;
                let mut data = vec![0; n * n * 4];
                metal.read_texture(&texture, mip, slice, &mut data, n * 4, 0);
                let expected = if (mip, slice) == (1, 1) { [0, 255, 0, 255] } else { sentinel(mip, slice) };
                assert!(data.as_chunks::<4>().0.iter().all(|p| *p == expected),
                        "mip {mip} slice {slice}: expected {expected:?}, got {:?}", &data[..4]);
            }
        }
    }

    #[test]
    fn default_viewport_restores_the_full_target_in_the_same_pass() {
        let metal = Metal::new().expect("Metal device");
        let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 16, 16);
        desc.render_target = true;
        let target = metal.create_texture(&desc).unwrap();
        let half = Viewport {
            x: 0.0,
            y: 0.0,
            width: 8.0,
            height: 16.0,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        draw_green_triangles(&metal, &target, Some([0.0; 4]), &[Some(half), None]);

        let mut pixels = [0; 16 * 16 * 4];
        metal.read_texture(&target, 0, 0, &mut pixels, 16 * 4, 0);
        assert_eq!(pixels.as_chunks::<4>().0[8 * 16 + 12], [0, 255, 0, 255],
                   "the second draw must reach (12, 8) outside the first viewport");
    }
}

#[cfg(test)]
mod owed_tests {
    use super::*;
    use crate::backend::ComputeBackend;
    use crate::mem::{Region, page_size};

    /// A submission of more passes than a command buffer takes goes in several, in order: every
    /// pass runs and the fence is signalled once the last has.
    #[test]
    fn a_long_submission_is_cut_into_command_buffers_of_at_most_n_passes() {
        let m = Metal::new().unwrap().with_passes_per_buffer(8);
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        // SAFETY: inside the region; the GPU is idle.
        let at = |o: usize| unsafe { region.ptr.add(o) as *const u8 };
        let q = m.create_queue().unwrap();
        let fence = m.create_fence(0).unwrap();
        // Each pass fills its own 8 bytes; the last one also overwrites the first pass's.
        let mut passes: Vec<_> = (0..20u8)
            .map(|n| Pass::Blit(vec![BlitOp::Fill { dst: (&buf, at(usize::from(n) * 8)), len: 8, value: n + 1 }]))
            .collect();
        passes.push(Pass::Blit(vec![BlitOp::Fill { dst: (&buf, at(0)), len: 8, value: 0xEE }]));
        let signals = [(&fence, 1)];
        let before = m.command_buffers();
        m.submit_passes(&q, &PassSubmission { waits: &[], passes: &passes, signals: &signals }).unwrap();
        assert_eq!(m.command_buffers() - before, 3, "21 passes, 8 to a command buffer");
        assert!(m.wait(&fence, 1, 5000), "the last piece signals");
        // SAFETY: inside the region; the GPU is done with it.
        let bytes = unsafe { std::slice::from_raw_parts(region.ptr, 20 * 8) };
        assert_eq!(&bytes[..8], &[0xEE; 8], "the pieces ran in order");
        for n in 1..20 {
            assert_eq!(&bytes[n * 8..][..8], &[n as u8 + 1; 8], "pass {n}");
        }

        // 0: the whole submission in one command buffer.
        let m = m.with_passes_per_buffer(0);
        let before = m.command_buffers();
        m.submit_passes(&q, &PassSubmission { waits: &[], passes: &passes, signals: &[(&fence, 2)] }).unwrap();
        assert_eq!(m.command_buffers() - before, 1);
        assert!(m.wait(&fence, 2, 5000));
    }

    /// A piece of a split submission that the GPU fails (here one the test drops: its passes and
    /// its release of the next piece do not run, as when the GPU fails it) pays its own value
    /// writes on the CPU and lets the next piece go; the later pieces still run on the GPU, and the
    /// fence is signalled by the last one. Nothing is paid twice: a write made after the fence was
    /// reached stays.
    #[test]
    fn a_failed_earlier_piece_pays_its_own_writes_and_the_later_pieces_still_run() {
        let _shared = FLIGHTS_IN_TESTS.read().unwrap_or_else(|e| e.into_inner());
        let m = Metal::new().unwrap().with_passes_per_buffer(4);
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        // SAFETY: inside the region; the GPU is idle.
        let at = |o: usize| unsafe { region.ptr.add(o) as *const u8 };
        let q = m.create_queue().unwrap();
        q.1.set(true); // as with the pool's residency set: the order event is in play
        let fence = m.create_fence(0).unwrap();
        // 12 passes, three pieces of 4: pass n fills its own 8 bytes with n + 1.
        let passes: Vec<_> = (0..12u8)
            .map(|n| Pass::Blit(vec![BlitOp::Fill { dst: (&buf, at(usize::from(n) * 8)), len: 8, value: n + 1 }]))
            .collect();
        let check = |first: u8, signal: u64| {
            DROP_PIECE.with(|d| d.set(Some(1)));
            let before = m.command_buffers();
            let signals = [(&fence, signal)];
            m.submit_passes(&q, &PassSubmission { waits: &[], passes: &passes, signals: &signals }).unwrap();
            DROP_PIECE.with(|d| d.set(None));
            assert_eq!(m.command_buffers() - before, 3);
            assert!(m.wait(&fence, signal, 5000), "the later pieces did not run");
            // SAFETY: inside the region; the GPU is done with it.
            let bytes = unsafe { std::slice::from_raw_parts(region.ptr, 12 * 8) };
            for n in 0..12 {
                assert_eq!(&bytes[n * 8..][..8], &[n as u8 + first; 8], "pass {n}");
            }
        };
        check(1, 1);
        // The title writes over the first piece's words once it has seen the fence: a second
        // payment of the failed piece would undo that.
        // SAFETY: inside the region; the GPU is done with it.
        unsafe { std::ptr::write_bytes(region.ptr, 0xCD, 8) };
        let after = [(&fence, 2)];
        m.submit_passes(&q, &PassSubmission { waits: &[], passes: &[], signals: &after }).unwrap();
        assert!(m.wait(&fence, 2, 5000));
        // SAFETY: inside the region.
        assert_eq!(unsafe { std::slice::from_raw_parts(region.ptr, 8) }, [0xCD; 8], "paid twice");
        assert_eq!(q.2.signaledValue(), 2, "the order event went past what the submissions owe");
        // Again from zeros: the failed piece's words come from the CPU, the rest from the GPU.
        // SAFETY: inside the region; the GPU is idle.
        unsafe { std::ptr::write_bytes(region.ptr, 0, 12 * 8) };
        check(1, 3);
    }

    #[test]
    fn a_failed_command_buffer_is_paid_on_the_cpu() {
        let m = Metal::new().unwrap();
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        // SAFETY: inside the region; the GPU is idle.
        let at = |o: usize| unsafe { region.ptr.add(o) };
        let q = m.create_queue().unwrap();
        q.1.set(true); // as with the pool's residency set: the order event is in play
        let fence = m.create_fence(0).unwrap();
        let passes = [Pass::Blit(vec![
            BlitOp::Fill { dst: (&buf, at(64) as *const u8), len: 8, value: 0xAB },
            BlitOp::Copy { src: (&buf, at(0) as *const u8), dst: (&buf, at(256) as *const u8), len: 16 },
        ])];
        let (s5, s7, s9) = ([(&fence, 5)], [(&fence, 7)], [(&fence, 9)]);
        let sub = |signals| PassSubmission { waits: &[], passes: &passes, signals };
        // The recovery a submission signalling the fence to 7 would owe, built as `submit_passes` does.
        q.3.set(3);
        let owed = Owed::new(&q, &sub(&s7), true);
        owed.recover();
        // SAFETY: bounds as above.
        let bytes = unsafe { std::slice::from_raw_parts(at(60), 16) };
        assert_eq!(bytes, [0, 0, 0, 0, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0, 0, 0, 0]);
        // SAFETY: inside the region.
        assert_eq!(unsafe { *at(256) }, 0, "copies are not replayed");
        assert_eq!(m.completed(&fence), 7);
        assert_eq!(q.2.signaledValue(), 3);
        // An older submission failing later does not lower anything.
        Owed::new(&q, &sub(&s5), false).recover();
        assert_eq!(m.completed(&fence), 7);
        // A normal submission after it still completes: the queue is not stuck behind the order value.
        m.submit_passes(&q, &sub(&s9)).unwrap();
        assert!(m.wait(&fence, 9, 5000), "queue stuck after recovery");
    }

    #[test]
    fn repeated_recovery_preserves_the_waiters_write() {
        let m = Metal::new().unwrap();
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        let q = m.create_queue().unwrap();
        let fence = m.create_fence(0).unwrap();
        let passes = [Pass::Blit(vec![BlitOp::Fill { dst: (&buf, region.ptr), len: 8, value: 0xAB }])];
        let signals = [(&fence, 7)];
        let owed = Owed::new(&q, &PassSubmission { waits: &[], passes: &passes, signals: &signals }, false);
        let word = region.ptr.cast::<[u8; 8]>();

        owed.recover();
        assert_eq!(m.completed(&fence), 7);
        // SAFETY: these eight bytes are inside the live region. Only this thread accesses them.
        unsafe {
            assert_eq!(word.read(), [0xAB; 8]);
            word.write([0xCD; 8]);
        }
        owed.recover();
        // SAFETY: same region and exclusive access as above.
        assert_eq!(unsafe { word.read() }, [0xCD; 8]);
    }

    /// A command buffer Metal never reports on (here one waiting for an event nobody signals) is
    /// found once it is old enough, and what it owes is paid: the fill and the fence signal the
    /// title's queues wait for.
    #[test]
    fn a_lost_command_buffer_pays_what_it_owes() {
        let _alone = FLIGHTS_IN_TESTS.write().unwrap_or_else(|e| e.into_inner());
        let m = Metal::new().unwrap();
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        let q = m.create_queue().unwrap();
        let (never, fence) = (m.create_fence(0).unwrap(), m.create_fence(0).unwrap());
        let passes = [Pass::Blit(vec![BlitOp::Fill { dst: (&buf, region.ptr as *const u8), len: 8, value: 0x5A }])];
        let (waits, signals) = ([(&never, 1)], [(&fence, 3)]);
        m.submit_passes(&q, &PassSubmission { waits: &waits, passes: &passes, signals: &signals }).unwrap();
        // Not old enough by either rule: nothing is lost.
        let (long, zero) = (std::time::Duration::from_secs(3600), std::time::Duration::ZERO);
        assert_eq!(recover_lost_after(long, long), 0);
        assert_eq!(m.completed(&fence), 0);
        // Other tests' command buffers may be in flight too (those that do not mind): at least ours.
        assert!(recover_lost_after(long, zero) >= 1);
        // SAFETY: inside the region.
        assert_eq!(unsafe { std::slice::from_raw_parts(region.ptr, 8) }, [0x5A; 8]);
        assert_eq!(m.completed(&fence), 3);
        // Let it finish: a late completion of a recovered command buffer changes nothing.
        never.0.setSignaledValue(1);
        let after = [(&fence, 4)];
        m.submit_passes(&q, &PassSubmission { waits: &[], passes: &[], signals: &after }).unwrap();
        assert!(m.wait(&fence, 4, 5000), "the queue runs on behind the recovered command buffer");
    }

    /// A stop wakes a command buffer asleep on an event nobody signals, as a title's queues are
    /// when the process is told to stop: it completes on the GPU (its fence is reached by the GPU,
    /// not paid), and nothing is left in flight.
    #[test]
    fn a_stop_wakes_command_buffers_waiting_on_events_nobody_signals() {
        let _alone = FLIGHTS_IN_TESTS.write().unwrap_or_else(|e| e.into_inner());
        let m = Metal::new().unwrap();
        let q = m.create_queue().unwrap();
        let (never, fence) = (m.create_fence(0).unwrap(), m.create_fence(0).unwrap());
        let (waits, signals) = ([(&never, 1)], [(&fence, 3)]);
        m.submit_passes(&q, &PassSubmission { waits: &waits, passes: &[], signals: &signals }).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(m.completed(&fence), 0, "the command buffer sleeps on its wait");
        let (in_flight, left) = wake_and_drain(&[never.0.clone(), q.2.clone()], std::time::Duration::from_secs(5));
        assert!(in_flight >= 1);
        assert_eq!(left, 0, "every command buffer completed on the GPU");
        assert_eq!(m.completed(&fence), 3);
        // The events a stop wakes are tracked as they are made, and let go with them.
        let tracked = |e: &Retained<ProtocolObject<dyn MTLSharedEvent>>| {
            EVENTS.lock().unwrap().0.iter().any(|w| w.load().is_some_and(|t| Retained::as_ptr(&t) == Retained::as_ptr(e)))
        };
        assert!(tracked(&never.0) && tracked(&q.2));
    }

    /// A stop empties the residency sets it tracks, so nothing it made stays resident.
    #[test]
    fn a_stop_empties_the_residency_sets() {
        use objc2_foundation::NSObjectProtocol;
        let m = Metal::new().unwrap();
        if !m.dev.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
            return;
        }
        let set = track_residency_set(m.dev.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).unwrap());
        let buf = m.dev.newBufferWithLength_options(1 << 20, MTLResourceOptions::StorageModePrivate).unwrap();
        set.addAllocation(ProtocolObject::from_ref(&*buf));
        set.commit();
        assert_eq!(set.allocationCount(), 1);
        assert!(SETS.lock().unwrap().0.iter().any(|w| w.load().is_some_and(|t| Retained::as_ptr(&t) == Retained::as_ptr(&set))));
        assert!(release_sets(std::slice::from_ref(&set)) >= 1 << 20);
        assert_eq!(set.allocationCount(), 0);
    }

    #[test]
    fn recovery_writes_the_fills_before_it_signals() {
        struct Send<T>(T);
        // SAFETY: test only; the recovery runs on one other thread while this one only reads.
        unsafe impl<T> std::marker::Send for Send<T> {}
        let m = Metal::new().unwrap();
        let len = 64 << 20;
        let region = Region::new(len, None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        let q = m.create_queue().unwrap();
        let fence = m.create_fence(0).unwrap();
        let passes = [Pass::Blit(vec![BlitOp::Fill { dst: (&buf, region.ptr as *const u8), len, value: 1 }])];
        let signals = [(&fence, 1)];
        let owed = Send(Owed::new(&q, &PassSubmission { waits: &[], passes: &passes, signals: &signals }, false));
        let t = std::thread::spawn(move || {
            let owed = owed;
            owed.0.recover();
        });
        while fence.0.signaledValue() < 1 {
            std::hint::spin_loop();
        }
        // SAFETY: inside the region; read as the title would once the fence is reached.
        let last = unsafe { std::ptr::read_volatile(region.ptr.add(len - 1)) };
        t.join().unwrap();
        assert_eq!(last, 1, "the fence was reached before the fill it covers was written");
    }
}

#[cfg(test)]
mod sampler_tests {
    use super::*;

    #[test]
    fn dynamic_samplers_apply_the_metal_border_descriptor() {
        for (color, expected) in [
            (BorderColor::TransparentBlack, MTLSamplerBorderColor::TransparentBlack),
            (BorderColor::OpaqueBlack, MTLSamplerBorderColor::OpaqueBlack),
            (BorderColor::OpaqueWhite, MTLSamplerBorderColor::OpaqueWhite),
        ] {
            let desc = dynamic_sampler_descriptor(&SamplerDesc {
                min: Filter::Nearest,
                mag: Filter::Nearest,
                mip: Filter::Nearest,
                address: [Address::Border, Address::Clamp, Address::Mirror],
                border_color: color,
                max_anisotropy: 1,
            });
            assert_eq!(desc.sAddressMode(), MTLSamplerAddressMode::ClampToBorderColor);
            assert_eq!(desc.tAddressMode(), MTLSamplerAddressMode::ClampToEdge);
            assert_eq!(desc.rAddressMode(), MTLSamplerAddressMode::MirrorRepeat);
            assert_eq!(desc.borderColor(), expected);
        }
    }

    #[test]
    fn dynamic_samplers_sample_their_border_color_and_keep_distinct_cache_keys() {
        use objc2_foundation::NSString;

        const GREEN: [u8; 4] = [0, 255, 0, 255];
        let metal = Metal::new().expect("Metal device");
        let source = metal.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        metal.upload_texture(&source, 0, 0, &GREEN.repeat(4), 8, 0);
        let mut output_desc = TextureDesc::d2(Format::Rgba8Unorm, 4, 1);
        output_desc.render_target = true;
        let output = metal.create_texture(&output_desc).unwrap();

        let library = metal.dev.newLibraryWithSource_options_error(&NSString::from_str(r#"
            #include <metal_stdlib>
            using namespace metal;

            vertex float4 test_vertex(uint i [[vertex_id]]) {
                return float4(i == 1 ? 3.0 : -1.0, i == 2 ? 3.0 : -1.0, 0.0, 1.0);
            }
            fragment float4 test_fragment(
                float4 position [[position]],
                texture2d<float> tex [[texture(0)]],
                sampler s [[sampler(0)]]
            ) {
                const float2 uv[] = {
                    float2(-0.25, 0.25), float2(0.25, 0.25),
                    float2(1.25, 0.25), float2(0.25, -0.25),
                };
                return tex.sample(s, uv[uint(position.x)]);
            }
        "#), None).expect("compile generated Metal shader source");

        let desc = MTLRenderPipelineDescriptor::new();
        let vertex = library.newFunctionWithName(&NSString::from_str("test_vertex")).unwrap();
        let fragment = library.newFunctionWithName(&NSString::from_str("test_fragment")).unwrap();
        desc.setVertexFunction(Some(&vertex));
        desc.setFragmentFunction(Some(&fragment));
        // SAFETY: color attachment 0 exists.
        unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) }
            .setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        let pipeline = metal.dev.newRenderPipelineStateWithDescriptor_error(&desc).unwrap();
        let queue = metal.dev.newCommandQueue().unwrap();

        let sample = |sampler: &ProtocolObject<dyn MTLSamplerState>| {
            let pass = MTLRenderPassDescriptor::new();
            // SAFETY: color attachment 0 exists.
            let a = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            a.setTexture(Some(&output.tex));
            a.setLoadAction(MTLLoadAction::Clear);
            a.setClearColor(MTLClearColor { red: 1.0, green: 0.0, blue: 1.0, alpha: 1.0 });
            a.setStoreAction(MTLStoreAction::Store);
            let cb = queue.commandBuffer().unwrap();
            let e = cb.renderCommandEncoderWithDescriptor(&pass).unwrap();
            e.setRenderPipelineState(&pipeline);
            // SAFETY: the generated fragment shader uses texture and sampler slot 0, and
            // the vertex shader draws a fullscreen triangle using vertex IDs 0 through 2.
            unsafe {
                e.setFragmentTexture_atIndex(Some(&source.tex), 0);
                e.setFragmentSamplerState_atIndex(Some(sampler), 0);
                e.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
            e.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed, "{:?}", cb.error());

            let mut bytes = [0; 16];
            metal.read_texture(&output, 0, 0, &mut bytes, 16, 0);
            *bytes.as_chunks::<4>().0.first_chunk::<4>().unwrap()
        };

        // Independently construct the edge sampler the old dynamic path used.
        let edge_desc = MTLSamplerDescriptor::new();
        edge_desc.setMinFilter(MTLSamplerMinMagFilter::Nearest);
        edge_desc.setMagFilter(MTLSamplerMinMagFilter::Nearest);
        edge_desc.setMipFilter(MTLSamplerMipFilter::Nearest);
        edge_desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        edge_desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        edge_desc.setRAddressMode(MTLSamplerAddressMode::ClampToEdge);
        edge_desc.setSupportArgumentBuffers(true);
        let legacy_edge = metal.dev.newSamplerStateWithDescriptor(&edge_desc).unwrap();
        assert_eq!(sample(&legacy_edge), [GREEN; 4]);

        let cases = [
            (Address::Border, BorderColor::OpaqueWhite, [255; 4]),
            (Address::Border, BorderColor::OpaqueBlack, [0, 0, 0, 255]),
            (Address::Border, BorderColor::TransparentBlack, [0; 4]),
            (Address::Clamp, BorderColor::TransparentBlack, GREEN),
        ];
        let cached_before = metal.samplers.borrow().len();
        let mut ids = std::collections::HashSet::new();
        for (mode, border_color, outside) in cases {
            let key = SamplerDesc {
                min: Filter::Nearest,
                mag: Filter::Nearest,
                mip: Filter::Nearest,
                address: [mode, Address::Clamp, Address::Clamp],
                border_color,
                max_anisotropy: 1,
            };
            let sampler = metal.create_sampler(&key).unwrap();
            let id = resource_id(sampler.0.gpuResourceID());
            assert!(ids.insert(id), "two distinct samplers shared an ID");
            let repeated = metal.create_sampler(&key).unwrap();
            assert_eq!(resource_id(repeated.0.gpuResourceID()), id, "repeat lookup changed cache identity");

            let pixels = sample(&sampler.0);
            assert_eq!(pixels, [outside, GREEN, outside, GREEN], "{mode:?}, {border_color:?}");
            println!("{mode:?} {border_color:?}: {pixels:?}");
        }
        assert_eq!(metal.samplers.borrow().len(), cached_before + cases.len());
    }

    #[test]
    fn heap_samplers_keep_distinct_ids_for_independent_min_and_mag_filters() {
        use MTLSamplerMinMagFilter as MetalFilter;

        let m = Metal::new().unwrap();
        let nearest = SamplerDesc {
            min: Filter::Nearest,
            mag: Filter::Nearest,
            mip: Filter::Nearest,
            address: [Address::Clamp; 3],
            border_color: BorderColor::TransparentBlack,
            max_anisotropy: 1,
        };
        let cases = [
            (nearest, (MetalFilter::Nearest, MetalFilter::Nearest)),
            (SamplerDesc { min: Filter::Linear, ..nearest }, (MetalFilter::Linear, MetalFilter::Nearest)),
            (SamplerDesc { mag: Filter::Linear, ..nearest }, (MetalFilter::Nearest, MetalFilter::Linear)),
        ];
        let id = |key: &SamplerDesc| resource_id(m.create_sampler(key).unwrap().0.gpuResourceID());
        // Like a heap entry, keep just the ID after creating each sampler.
        let mut ids = std::collections::HashSet::new();
        let first = id(&nearest);
        for (key, expected) in cases {
            let desc = dynamic_sampler_descriptor(&key);
            assert_eq!((desc.minFilter(), desc.magFilter()), expected);
            assert!(ids.insert(id(&key)), "distinct filter pairs shared a sampler ID");
        }
        assert_eq!(id(&nearest), first, "samplers are shared by descriptor");
    }
}

#[cfg(test)]
mod heap_set_tests {
    use super::*;

    #[test]
    fn each_command_buffer_attaches_each_heap_once_and_commits_later_changes() {
        let m = Metal::new().unwrap().with_heap_sets();
        let a = m.create_heap(HeapKind::Resource, 1).unwrap();
        let b = m.create_heap(HeapKind::Sampler, 1).unwrap();
        let Some(set_a) = &a.set else {
            eprintln!("skipped: no residency sets on this system");
            return;
        };
        let attaches = |heap: &Heap, attached_heaps: &mut Vec<*const Heap>| {
            let mut called = false;
            m.use_heap_cb(heap, attached_heaps, |_| called = true);
            called
        };
        let mut first = Vec::new();
        assert!(attaches(&a, &mut first));
        assert!(attaches(&b, &mut first));

        // Metal reports allocatedSize at the last commit.
        let before = set_a.allocatedSize();
        let texture = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 128, 128)).unwrap();
        m.write_texture(&a, 0, &texture, 0.0);
        assert!(a.dirty.get());
        assert!(!attaches(&a, &mut first));
        assert!(!a.dirty.get());
        assert!(set_a.allocatedSize() > before);

        let mut second = Vec::new();
        assert!(attaches(&a, &mut second));
        assert!(!attaches(&a, &mut second));
    }

    /// Each piece of a split submission is a command buffer of its own, and a residency set keeps
    /// its allocations resident only for the command buffers it was given to: the next piece
    /// attaches every heap again.
    #[test]
    fn the_next_piece_of_a_split_submission_attaches_the_heaps_again() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        if heap.set.is_none() {
            eprintln!("skipped: no residency sets on this system");
            return;
        }
        let q = m.create_queue().unwrap();
        let attaches = |attached: &mut Vec<*const Heap>| {
            let mut called = false;
            m.use_heap_cb(&heap, attached, |_| called = true);
            called
        };
        let mut attached = Vec::new();
        assert!(attaches(&mut attached), "the first piece");
        assert!(!attaches(&mut attached), "once per command buffer");
        let _next = m.next_piece(&q, &[], &mut attached, None).unwrap();
        assert!(attaches(&mut attached), "the next piece is given the heap's set too");
    }

    /// With `VICTORY_METAL_REQUEST_RESIDENCY` a heap's set asks to stay resident when it is made
    /// and after each commit a use makes; without it, never.
    #[test]
    fn a_heaps_set_asks_to_stay_resident_only_when_switched_on() {
        for on in [false, true] {
            let m = Metal::new().unwrap().with_heap_sets().with_requested_residency(on);
            let heap = m.create_heap(HeapKind::Resource, 4).unwrap();
            if heap.set.is_none() {
                eprintln!("skipped: no residency sets on this system");
                return;
            }
            let before = m.residency_counts().requests;
            let t = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
            m.write_texture(&heap, 0, &t, 0.0);
            let mut attached = Vec::new();
            m.use_heap_cb(&heap, &mut attached, |_| {});
            m.use_heap_cb(&heap, &mut attached, |_| {});
            let made = u64::from(on);
            assert_eq!(before, made, "on={on}: when the set is made");
            assert_eq!(m.residency_counts().requests, 2 * made, "on={on}: again after the one commit");
            drop(heap); // ends the residency it asked for
        }
    }

    /// A heap's residency set holds the heap buffer and each allocation its entries name, once,
    /// and lets an allocation go when the last entry naming it is overwritten: the set is what a
    /// command buffer that uses the heap keeps resident.
    #[test]
    fn a_heaps_set_follows_its_entries() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 4).unwrap();
        let Some(set) = &heap.set else {
            eprintln!("skipped: no residency sets on this system");
            return;
        };
        let held = || {
            set.commit();
            set.allocationCount()
        };
        let texture = || m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        let (t1, t2) = (texture(), texture());
        m.write_texture(&heap, 0, &t1, 0.0);
        m.write_texture(&heap, 1, &t1, 0.0);
        assert_eq!(held(), 2, "the heap buffer and the texture two entries name");
        m.write_texture(&heap, 0, &t2, 0.0);
        assert_eq!(held(), 3);
        m.write_texture(&heap, 1, &t2, 0.0);
        assert_eq!(held(), 2, "the first texture went with its last entry");
        m.clear_entry(&heap, 0);
        assert_eq!(held(), 2, "an entry still names the second texture");
        m.clear_entry(&heap, 1);
        assert_eq!(held(), 1, "cleared entries let their texture go");
        // SAFETY: the heap buffer holds four entries of `ENTRY` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(heap.buf.contents().as_ptr().cast::<u8>(), 2 * ENTRY) };
        assert!(bytes.iter().all(|b| *b == 0), "a cleared entry reads as one never written");

        let heap_only = set.allocatedSize();
        let arena = crate::mem::Alias::new(65536).unwrap();
        let buf = victory_backend::backend::ComputeBackend::wrap_title_memory(
            &m,
            arena.primary,
            arena.len,
        )
        .unwrap();
        let entry_words = |i: usize| -> [u64; 3] {
            // SAFETY: the heap buffer holds four entries of `ENTRY` (24) bytes.
            unsafe {
                heap.buf
                    .contents()
                    .as_ptr()
                    .cast::<[u64; 3]>()
                    .add(i)
                    .read_unaligned()
            }
        };
        assert!(m.write_typed_buffer(&heap, 0, &buf, arena.primary, 1024, Format::Rgba32Uint, true));
        assert_eq!(m.heap_typed_views(&heap), 1);
        assert_ne!(entry_words(0)[1], 0, "aligned typed buffer writes textureViewID");
        // SAFETY: `1024` and `4` are inside the 64 KiB `arena` allocation.
        let second = unsafe { arena.primary.add(1024) };
        let unaligned = unsafe { arena.primary.add(4) };
        assert!(m.write_typed_buffer(&heap, 1, &buf, second, 1024, Format::Rgba32Float, true));
        assert_eq!(m.heap_typed_views(&heap), 2);
        assert!(!m.write_typed_buffer(&heap, 2, &buf, unaligned, 1024, Format::Rgba32Float, true));
        assert_eq!(m.heap_typed_views(&heap), 2, "unaligned fallback holds no typed view");
        assert_ne!(entry_words(2)[0], 0, "unaligned fallback preserves gpuVA in word 0");
        assert_eq!(entry_words(2)[1], 0, "unaligned fallback leaves textureViewID null");
        assert_eq!(entry_words(2)[2], 1024);
        assert_eq!(held(), 2, "typed buffer views over one pool add only the pool allocation");
        assert_eq!(
            set.allocatedSize(),
            heap_only + arena.len as u64,
            "buffer-backed texture views do not multiply the pool's residency-set size"
        );
        m.write_buffer(&heap, 0, &buf, arena.primary, 1024);
        assert_eq!(m.heap_typed_views(&heap), 1, "overwriting with write_buffer drops the typed view");
        assert_eq!(entry_words(0)[1], 0, "write_buffer zeroes textureViewID");
        m.clear_entry(&heap, 0);
        m.clear_entry(&heap, 2);
        assert_eq!(held(), 2);
        assert_eq!(m.heap_typed_views(&heap), 1);
        m.clear_entry(&heap, 1);
        assert_eq!(m.heap_typed_views(&heap), 0, "clear_entry drops the last typed view");
        assert_eq!(held(), 1, "cleared typed buffer entries release the pool allocation");
    }

    #[test]
    #[ignore = "benchmark: cargo test -p victory-metal typed_buffer_view_cost_benchmark -- --ignored --nocapture"]
    fn typed_buffer_view_cost_benchmark() {
        fn phys_footprint() -> usize {
            const TASK_VM_INFO: i32 = 22;
            unsafe extern "C" {
                fn mach_task_self() -> u32;
                fn task_info(task: u32, flavor: i32, info: *mut u64, count: *mut u32) -> i32;
            }
            let mut info = [0u64; 64];
            let mut count = 128u32;
            // SAFETY: local buffer of `count` 32-bit words for `TASK_VM_INFO`.
            let kr = unsafe { task_info(mach_task_self(), TASK_VM_INFO, info.as_mut_ptr(), &mut count) };
            if kr == 0 && count >= 38 { info[18] as usize } else { 0 }
        }

        let m = Metal::new().unwrap().with_heap_sets();
        for fmt in [
            Format::R8Unorm,
            Format::R16Float,
            Format::R32Uint,
            Format::Rg32Float,
            Format::Rgba16Uint,
            Format::Rgba32Uint,
            Format::Rgba32Float,
        ] {
            let pf = pixel_format(fmt);
            let align = m.dev.minimumTextureBufferAlignmentForPixelFormat(pf);
            println!("align {fmt:?} (MTLPixelFormat {:?}): {align} B", pf);
        }

        const TOTAL: usize = 40_000;
        let arena_len = TOTAL * 64;
        let arena = crate::mem::Alias::new(arena_len).unwrap();
        // Touch pages upfront so pool page faults do not pollute descriptor measurements.
        // SAFETY: `arena.primary` is a writable `arena_len` allocation.
        unsafe { std::ptr::write_bytes(arena.primary, 0, arena_len) };
        let buf = victory_backend::backend::ComputeBackend::wrap_title_memory(
            &m,
            arena.primary,
            arena.len,
        )
        .unwrap();
        let heap = m.create_heap(HeapKind::Resource, TOTAL).unwrap();
        if let Some(set) = &heap.set {
            set.commit();
        }

        let fp0 = phys_footprint();
        let t0 = std::time::Instant::now();
        for i in 0..TOTAL {
            // SAFETY: `i * 64 < arena_len`.
            let ptr = unsafe { arena.primary.add(i * 64) };
            m.write_buffer(&heap, i, &buf, ptr, 64);
        }
        let raw_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let fp_raw = phys_footprint();
        println!(
            "40,000 write_buffer (lazy mirror): {raw_ms:.2} ms, phys_footprint delta = {:.2} KiB",
            (fp_raw as isize - fp0 as isize) as f64 / 1024.0
        );

        for count in [64usize, 512, 1_000, 10_000, 40_000] {
            for i in 0..count {
                // SAFETY: `i * 64 < arena_len`.
                let ptr = unsafe { arena.primary.add(i * 64) };
                m.write_buffer(&heap, i, &buf, ptr, 64);
            }
            let before = phys_footprint();
            let start = std::time::Instant::now();
            for i in 0..count {
                // SAFETY: `i * 64 < arena_len`.
                let ptr = unsafe { arena.primary.add(i * 64) };
                let fmt = match i % 3 {
                    0 => Format::Rgba32Float,
                    1 => Format::Rg32Float,
                    _ => Format::R32Uint,
                };
                m.write_typed_buffer(&heap, i, &buf, ptr, 64, fmt, true);
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            let after = phys_footprint();
            let set_bytes = heap.set.as_ref().map_or(0, |s| {
                s.commit();
                s.allocatedSize()
            });
            let delta_kib = (after as isize - before as isize) as f64 / 1024.0;
            println!(
                "materialize {count:>6} write_typed_buffer: {ms:>7.2} ms ({:.2} us/view), phys_footprint delta = {delta_kib:>8.1} KiB ({:.1} B/view), heap_set = {:.2} MiB",
                ms * 1000.0 / count as f64,
                (after as isize - before as isize) as f64 / count as f64,
                set_bytes as f64 / (1024.0 * 1024.0)
            );
            for i in 0..count {
                // SAFETY: `i * 64 < arena_len`.
                let ptr = unsafe { arena.primary.add(i * 64) };
                m.write_buffer(&heap, i, &buf, ptr, 64);
            }
        }
    }
}

#[cfg(test)]
mod heap_usage_tests {
    use super::*;

    #[test]
    fn heap_allocations_give_each_allocation_once_with_its_latest_use() {
        use objc2_foundation::NSObjectProtocol;
        let Ok(mut m) = Metal::new() else { return };
        if !m.dev.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
            return;
        }
        m.residency_sets = true;
        let heap = m.create_heap(HeapKind::Resource, 8).unwrap();
        let tex = |w| m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, w, w)).unwrap();
        let (a, b, c) = (tex(64), tex(128), tex(256));
        m.write_texture(&heap, 0, &a, 0.0);
        m.write_texture(&heap, 1, &a, 0.0);
        m.write_texture(&heap, 2, &b, 0.0);
        m.write_texture(&heap, 3, &c, 0.0);
        // A view of `c` is `c`'s memory: one allocation, its latest use theirs.
        let view = m.create_texture_view(&c, &ViewDesc { format: Format::Rgba8Unorm, mips: (0, 1), slices: (0, 1) }).unwrap();
        m.write_texture(&heap, 4, &view, 0.0);
        // `a` is named by entries last used at frames 10 and 50, `b` at 20, `c` never, its view at 5.
        let mut all = m.heap_allocations(&heap, &[10, 50, 20, 0, 5, 0, 0, 0]).unwrap();
        all.sort_by_key(|a| a.bytes);
        assert_eq!(all.iter().map(|a| a.latest).collect::<Vec<_>>(), [50, 20, 5], "{all:?}");
        assert!(all.iter().all(|a| a.texture && a.bytes > 0), "{all:?}");
        // Without a residency set nothing is kept resident by the heap.
        m.residency_sets = false;
        let plain = m.create_heap(HeapKind::Resource, 1).unwrap();
        assert_eq!(m.heap_allocations(&plain, &[0]), None);
    }
}

impl Metal {
    /// The depth of a 32-bit depth (or depth-stencil) texture's first mip as floats, read through
    /// a blit on a queue of its own and waited for (local diagnostic: target dumps).
    pub fn read_depth(&self, t: &Texture) -> Option<Vec<f32>> {
        let (w, h) = (t.tex.width(), t.tex.height());
        let buf = self.dev.newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModeShared)?;
        let q = self.dev.newCommandQueue()?;
        let cb = q.commandBuffer()?;
        let e = cb.blitCommandEncoder()?;
        let opts = if t.tex.pixelFormat() == MTLPixelFormat::Depth32Float_Stencil8 { MTLBlitOption::DepthFromDepthStencil } else { MTLBlitOption::empty() };
        // SAFETY: the region is the texture's first mip; the buffer holds it at 4 bytes a texel.
        unsafe {
            e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                &t.tex, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, size([w, h, 1]), &buf, 0, w * 4, w * h * 4, opts,
            );
        }
        e.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        // SAFETY: the shared buffer holds w * h floats, written by the finished blit.
        Some(unsafe { std::slice::from_raw_parts(buf.contents().as_ptr().cast::<f32>(), w * h) }.to_vec())
    }
}
