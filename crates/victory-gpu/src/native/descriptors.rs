//! The title's descriptors, decoded for a native backend.
//!
//! A title writes a view with `CreateShaderResourceView` and friends; `descriptor::record` keeps the
//! description (in the host's layout, as words) and leaves `[tag, index, generation]` in the 32-byte
//! (16 for samplers) slot, so that titles that copy slots as bytes move whole descriptors. The D3D12
//! host builds its views from those records (`descriptor::host`); this module reads the same records
//! and gives the backend's `write_*` calls what they need. Placed views identify a texture by its title address
//! and resource description.

use crate::descriptor::View;
use std::collections::HashMap;
use victory_backend::resources::{Address, BorderColor, Filter, Format, SamplerDesc};

/// A registered resource: title address and size for buffers, the native texture for images.
pub(super) enum Res<T> {
    Buffer(u64, u64),
    Texture(T),
}

/// What a heap entry holds, in backend terms.
#[derive(Debug, Clone, PartialEq)]
pub enum Decoded {
    /// A raw buffer SRV, UAV or CBV: `size` bytes at title address `addr`.
    Buffer {
        addr: u64,
        size: u32,
    },
    /// A texture SRV or UAV of a registered resource: optional `(first, count)` mip and array-slice
    /// ranges and an optional format override. A count of `u32::MAX` includes every remaining mip
    /// or slice; the backend clips it to the texture. With no mip, slice, or format override, bind
    /// the original texture, including every mip and slice.
    Image {
        resource: usize,
        mips: Option<(u32, u32)>,
        slices: Option<(u32, u32)>,
        format: Option<Format>,
    },
    Sampler {
        desc: SamplerDesc,
        lod_bias: f32,
    },
}

/// SRV and UAV `ViewDimension` of a buffer.
const BUFFER: u32 = 1;
/// SRV `ViewDimension` of `Texture2D`; UAV's numbering is the same up to 3D.
const TEXTURE_2D: u32 = 4;
const TEXTURE_2D_ARRAY: u32 = 5;
const FLAG_RAW: u32 = 1;

/// Bytes per element of a buffer view and the range it covers from `base`.
fn buffer_view(
    base: u64,
    format: u32,
    stride: u32,
    flags: u32,
    first: u64,
    count: u32,
) -> Option<(u64, u32)> {
    let element = if stride > 0 {
        u64::from(stride)
    } else if flags & FLAG_RAW != 0 {
        4
    } else {
        crate::resource::element(crate::resource::pc_format(format))?.0
    };
    let size = (u64::from(count) * element).min(u64::from(u32::MAX));
    Some((base + first * element, size as u32))
}

/// The DXGI formats of textures the native host knows.
pub fn format(dxgi: u32) -> Option<Format> {
    Some(match crate::resource::pc_format(dxgi) {
        28 | 27 => Format::Rgba8Unorm,
        39 => Format::R32Float,
        29 => Format::Rgba8UnormSrgb,
        87 => Format::Bgra8Unorm,
        91 => Format::Bgra8UnormSrgb,
        10 => Format::Rgba16Float,
        41 => Format::R32Float,
        40 => Format::D32Float,
        // D32_FLOAT_S8X24_UINT (19), its shader views (20) and the console's variant (45).
        19 | 20 | 45 => Format::D32FloatS8,
        71 => Format::Bc1,
        74 => Format::Bc2,
        77 => Format::Bc3,
        80 => Format::Bc4,
        83 => Format::Bc5,
        95 => Format::Bc6h,
        96 => Format::Bc6hSigned,
        98 => Format::Bc7,
        61 => Format::R8Unorm,
        65 => Format::A8Unorm,
        56 => Format::R16Unorm,
        2 => Format::Rgba32Float,
        16 => Format::Rg32Float,
        24 => Format::Rgb10a2Unorm,
        26 => Format::Rg11b10Float,
        30 => Format::Rgba8Uint,
        34 => Format::Rg16Float,
        35 => Format::Rg16Unorm,
        42 => Format::R32Uint,
        62 => Format::R8Uint,
        67 => Format::Rgb9e5Float,
        99 => Format::Bc7Srgb,
        3 => Format::Rgba32Uint,
        11 => Format::Rgba16Unorm,
        13 => Format::Rgba16Snorm,
        17 => Format::Rg32Uint,
        36 => Format::Rg16Uint,
        49 => Format::Rg8Unorm,
        50 => Format::Rg8Uint,
        51 => Format::Rg8Snorm,
        54 => Format::R16Float,
        // R16_TYPELESS: a depth target when the texture is one (`TextureInfo::desc`).
        53 => Format::R16Unorm,
        57 => Format::R16Uint,
        55 => Format::D16Unorm,
        72 => Format::Bc1Srgb,
        78 => Format::Bc3Srgb,
        86 => Format::Bgr5a1Unorm,
        // B4G4R4A4: its title bytes are converted at upload (`textures::to_host`).
        115 => Format::Abgr4Unorm,
        _ => return None,
    })
}

/// The view-stride table's dword for `view` (`victory_shader::emit`, "Buffer views at computed
/// heap indices"; what `descriptor::host::view_stride` and `view_kind` give the D3D12 host): a
/// buffer view's element size in bytes (its structure stride, 4 for a raw view or a constant
/// buffer, else its format's element size; at least 1), with `VIEW_INTEGER` (and `VIEW_SIGNED`)
/// for a typed view of an integer format. 0 for anything [`decode`] makes no buffer of: a shader
/// that takes a buffer at a heap index it computes reads the null view there.
pub(super) fn stride_word<T>(view: &View, resources: &HashMap<usize, Res<T>>) -> u32 {
    use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED, VIEW_STRIDE_BITS};
    let buffer = |r: &usize| matches!(resources.get(r), Some(Res::Buffer(..)));
    let (format, stride, flags) = match view {
        View::Constant(..) => return 4,
        View::Shader(r, Some(d)) if d[1] == BUFFER && buffer(r) => (d[0], d[7], d[8]),
        View::Unordered(r, _, Some(d)) if d[1] == BUFFER && buffer(r) => (d[0], d[5], d[8]),
        View::Placed(true, res, d) if res[0] == BUFFER => (d[0], d[5], d[8]),
        View::Placed(false, res, d) if res[0] == BUFFER => (d[0], d[7], d[8]),
        // A whole buffer: raw.
        View::Shader(r, None) | View::Unordered(r, _, None) if buffer(r) => return 4,
        _ => return 0,
    };
    if stride > 0 {
        return stride & VIEW_STRIDE_BITS;
    }
    if flags & FLAG_RAW != 0 {
        return 4;
    }
    let format = crate::resource::pc_format(format);
    let element = crate::resource::element(format).map_or(0, |(bytes, _)| bytes).max(1) as u32;
    // `UINT` and `SINT` formats (the D3D12 host's `integer_format`).
    let kind = match format {
        3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => VIEW_INTEGER,
        4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => VIEW_INTEGER | VIEW_SIGNED,
        _ => 0,
    };
    element | kind
}

/// Whether `view` is a buffer view the console describes with no stride (a raw view, a constant
/// buffer, a whole buffer): its record count is in bytes.
pub(super) fn raw_buffer(view: &View) -> bool {
    match view {
        View::Constant(..) | View::Shader(_, None) | View::Unordered(_, _, None) => true,
        View::Shader(_, Some(d)) if d[1] == BUFFER => d[7] == 0 && d[8] & FLAG_RAW != 0,
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => d[5] == 0 && d[8] & FLAG_RAW != 0,
        View::Placed(true, res, d) if res[0] == BUFFER => d[5] == 0 && d[8] & FLAG_RAW != 0,
        View::Placed(false, res, d) if res[0] == BUFFER => d[7] == 0 && d[8] & FLAG_RAW != 0,
        _ => false,
    }
}

/// Object and placed views have the same SRV/UAV union layout.
fn texture_view(resource: usize, unordered: bool, d: &[u32]) -> Option<Decoded> {
    let (mips, layers) = if unordered {
        ((d[2], 1), (d[3], d[4]))
    } else {
        ((d[4], d[5]), (d[6], d[7]))
    };
    let format = format(d[0]);
    let slices = if d[1] == TEXTURE_2D_ARRAY {
        // An empty or unrepresentable array view must not fall back to the whole texture.
        if layers.1 == 0 || mips.1 == 0 || format.is_none() {
            return None;
        }
        Some(layers)
    } else {
        None
    };
    Some(Decoded::Image {
        resource,
        mips: Some(mips),
        slices,
        format,
    })
}

pub(super) fn decode<T>(view: &View, resources: &HashMap<usize, Res<T>>) -> Option<Decoded> {
    let buffer_base = |r: &usize| match resources.get(r)? {
        Res::Buffer(addr, _) => Some(*addr),
        Res::Texture(_) => None,
    };
    match view {
        View::Constant(addr, size) => Some(Decoded::Buffer {
            addr: *addr,
            size: *size,
        }),
        // SRV words: format, dimension, mapping, pad, then the union (buffer: first element (8
        // bytes), count, stride, flags; 2D: first mip, mip count).
        View::Shader(r, Some(d)) if d[1] == BUFFER => {
            let base = buffer_base(r)?;
            let first = u64::from(d[4]) | u64::from(d[5]) << 32;
            let (addr, size) = buffer_view(base, d[0], d[7], d[8], first, d[6])?;
            Some(Decoded::Buffer { addr, size })
        }
        // UAV words: format, dimension, then the union (buffer: first element (8 bytes), count,
        // stride, counter offset (8 bytes), flags; 2D: mip slice, plane).
        View::Unordered(r, _, Some(d)) if d[1] == BUFFER => {
            let base = buffer_base(r)?;
            let first = u64::from(d[2]) | u64::from(d[3]) << 32;
            let (addr, size) = buffer_view(base, d[0], d[5], d[8], first, d[4])?;
            Some(Decoded::Buffer { addr, size })
        }
        View::Placed(unordered, res, d) if res[0] == BUFFER => {
            let base = u64::from(d[10]) | u64::from(d[11]) << 32;
            let w: &[u32] = &d[..10];
            if *unordered {
                let first = u64::from(w[2]) | u64::from(w[3]) << 32;
                let (addr, size) = buffer_view(base, w[0], w[5], w[8], first, w[4])?;
                Some(Decoded::Buffer { addr, size })
            } else {
                let first = u64::from(w[4]) | u64::from(w[5]) << 32;
                let (addr, size) = buffer_view(base, w[0], w[7], w[8], first, w[6])?;
                Some(Decoded::Buffer { addr, size })
            }
        }
        // A texture the title placed in its own memory: named by address and description.
        View::Placed(unordered, res, d) if res[0] > BUFFER => {
            let base = u64::from(d[10]) | u64::from(d[11]) << 32;
            texture_view(super::textures::placed_key(base, res), *unordered, d)
        }
        View::Shader(r, None) | View::Unordered(r, _, None) => match *resources.get(r)? {
            Res::Buffer(addr, size) => Some(Decoded::Buffer {
                addr,
                size: size.min(u64::from(u32::MAX)) as u32,
            }),
            Res::Texture(_) if matches!(view, View::Shader(..)) => Some(Decoded::Image {
                resource: *r,
                mips: None,
                slices: None,
                format: None,
            }),
            Res::Texture(_) => None,
        },
        View::Shader(r, Some(d)) if matches!(d[1], TEXTURE_2D | TEXTURE_2D_ARRAY) => {
            texture_view(*r, false, d)
        }
        View::Unordered(r, _, Some(d)) if matches!(d[1], TEXTURE_2D | TEXTURE_2D_ARRAY) => {
            texture_view(*r, true, d)
        }
        View::Shader(r, Some(d)) | View::Unordered(r, _, Some(d)) => Some(Decoded::Image {
            resource: *r,
            mips: None,
            slices: None,
            format: format(d[0]),
        }),
        View::ConsoleSampler(w) => Some(sampler(&crate::sampler::console_fields(w))),
        // shortcut: the plain D3D12 sampler description (13 words) is not decoded; titles on this
        // console use `CreateSamplerX`.
        View::Sampler(_) | View::Placed(..) => None,
    }
}

fn sampler(f: &crate::sampler::Fields) -> Decoded {
    let linear = |v: u32| {
        if v == 1 || v == 3 {
            Filter::Linear
        } else {
            Filter::Nearest
        }
    };
    let aniso = f.mag >= 2 || f.min >= 2;
    // Metal has three fixed border colors. Keep the existing edge clamp for other colors.
    let border_color = if f.address.contains(&4) {
        match f.border {
            [0.0, 0.0, 0.0, 0.0] => Some(BorderColor::TransparentBlack),
            [0.0, 0.0, 0.0, 1.0] => Some(BorderColor::OpaqueBlack),
            [1.0, 1.0, 1.0, 1.0] => Some(BorderColor::OpaqueWhite),
            _ => None,
        }
    } else {
        None
    };
    let mode = |m: u32| match m {
        1 => Address::Wrap,
        2 => Address::Mirror,
        4 if border_color.is_some() => Address::Border,
        _ => Address::Clamp,
    };
    Decoded::Sampler {
        desc: SamplerDesc {
            min: linear(f.min),
            mag: linear(f.mag),
            mip: if f.mip == 2 {
                Filter::Linear
            } else {
                Filter::Nearest
            },
            address: f.address.map(mode),
            border_color: border_color.unwrap_or(BorderColor::TransparentBlack),
            max_anisotropy: if aniso {
                f.anisotropy.clamp(1, 16) as usize
            } else {
                1
            },
        },
        lod_bias: if f.bias.is_nan() {
            0.0
        } else {
            f.bias.clamp(-16.0, 15.99)
        },
    }
}

/// The `texture_buffer` `(format, writable, explicit)` of `view` when shaders may access it with
/// `buffer_load_format_*` / `buffer_store_format_*` (`explicit` is `true` for typed buffer views
/// with `stride == 0`, `false` for 4-, 8-, and 16-byte structured buffer views).
pub(super) fn typed_buffer<T>(
    view: &View,
    resources: &HashMap<usize, Res<T>>,
) -> Option<(Format, bool, bool)> {
    let buffer = |r: &usize| matches!(resources.get(r), Some(Res::Buffer(..)));
    let (dxgi, stride, flags, writable) = match view {
        View::Shader(r, Some(d)) if d[1] == BUFFER && buffer(r) => (d[0], d[7], d[8], false),
        View::Unordered(r, _, Some(d)) if d[1] == BUFFER && buffer(r) => (d[0], d[5], d[8], true),
        View::Placed(true, res, d) if res[0] == BUFFER => (d[0], d[5], d[8], true),
        View::Placed(false, res, d) if res[0] == BUFFER => (d[0], d[7], d[8], false),
        _ => return None,
    };
    if flags & FLAG_RAW != 0 {
        return None;
    }
    if stride > 0 {
        // Guess: on RDNA2 a structured buffer V# of 4-, 8-, or 16-byte elements carries a 1-, 2-,
        // or 4-dword 32-bit data format, so shaders may access it with `buffer_load_format_*` or
        // `buffer_store_format_*` as well as `buffer_load_dword*` / `buffer_store_dword*`.
        // Materialized lazily only when a bound shader uses DXIL `TypedBuffer` operations.
        let fmt = match stride {
            4 => Format::R32Float,
            8 => Format::Rg32Float,
            16 => Format::Rgba32Float,
            _ => return None,
        };
        return Some((fmt, writable, false));
    }
    let fmt = match crate::resource::pc_format(dxgi) {
        // `victory-shader` lowers all formatted buffer loads and stores through float-typed DXIL
        // handles (`TypedBuffer<float4>`, `bitcast i32 <-> float`), while Metal's texture hardware
        // unpacks/packs according to the `MTLTexture` view's `MTLPixelFormat`. For 32-bit `SINT`
        // (`4`, `18`, `43`) there is no sub-32-bit sign extension or clamping, so `Uint` transfers
        // all 32 two's-complement bits identically; 8- and 16-bit `SINT` stay out because `Uint`
        // would zero-extend instead of sign-extending.
        4 => Format::Rgba32Uint,
        12 => Format::Rgba16Uint,
        18 => Format::Rg32Uint,
        25 => Format::Rgb10a2Uint,
        31 => Format::Rgba8Snorm,
        37 => Format::Rg16Snorm,
        43 => Format::R32Uint,
        58 => Format::R16Snorm,
        63 => Format::R8Snorm,
        _ => format(dxgi)?,
    };
    Some((fmt, writable, true))
}

#[cfg(test)]
mod tests {
    /// The view-stride table's dwords: a structured view's stride, 4 for raw and constant
    /// views, a typed view's element size with its integer kind, and 0 for what is no buffer view
    /// (a texture view, a view of an object the host has no buffer for).
    #[test]
    fn stride_words_of_buffer_views() {
        use super::{BUFFER, FLAG_RAW, Res, TEXTURE_2D, stride_word};
        use crate::descriptor::View;
        use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
        let resources: std::collections::HashMap<usize, Res<()>> = [(7, Res::Buffer(0x1000, 0x1000)), (9, Res::Texture(()))].into();
        let word = |view: View| stride_word(&view, &resources);
        let srv = |format: u32, dimension: u32, stride: u32, flags: u32| {
            let mut d = [0u32; 10];
            (d[0], d[1], d[6], d[7], d[8]) = (format, dimension, 16, stride, flags);
            d
        };
        let uav = |format: u32, stride: u32, flags: u32| {
            let mut d = [0u32; 10];
            (d[0], d[1], d[4], d[5], d[8]) = (format, BUFFER, 16, stride, flags);
            d
        };
        assert_eq!(word(View::Constant(0x1000, 256)), 4);
        assert_eq!(word(View::Shader(7, Some(srv(0, BUFFER, 48, 0)))), 48);
        assert_eq!(word(View::Shader(7, Some(srv(39, BUFFER, 0, FLAG_RAW)))), 4);
        assert_eq!(word(View::Shader(7, None)), 4);
        assert_eq!(word(View::Unordered(7, 0, Some(uav(42, 0, 0)))), 4 | VIEW_INTEGER, "R32_UINT");
        assert_eq!(word(View::Unordered(7, 0, Some(uav(59, 0, 0)))), 2 | VIEW_INTEGER | VIEW_SIGNED, "R16_SINT");
        assert_eq!(word(View::Unordered(7, 0, Some(uav(2, 0, 0)))), 16, "R32G32B32A32_FLOAT");
        // Not buffer views: a texture view, a whole texture, a buffer view of an object the host
        // does not know.
        assert_eq!(word(View::Shader(7, Some(srv(28, TEXTURE_2D, 0, 0)))), 0);
        assert_eq!(word(View::Shader(9, None)), 0);
        assert_eq!(word(View::Shader(8, Some(srv(0, BUFFER, 48, 0)))), 0);
        let mut resource = [0u32; 14];
        resource[0] = BUFFER;
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&uav(0, 12, 0));
        assert_eq!(word(View::Placed(true, resource, placed)), 12);
        resource[0] = 3;
        assert_eq!(word(View::Placed(true, resource, placed)), 0);
    }

    #[test]
    fn host_formats_have_the_title_formats_element_size() {
        for dxgi in [2, 3, 10, 11, 13, 16, 17, 24, 26, 28, 30, 34, 35, 36, 41, 42, 49, 50, 51, 53, 54, 55, 56, 57, 61, 62, 65, 67, 71, 72, 77, 78, 86, 91, 98, 99, 115] {
            let f = super::format(dxgi).expect("a host format");
            let (bytes, block) = crate::resource::element(dxgi).expect("a title format");
            assert_eq!(f.block(), (block as usize, bytes as usize), "DXGI {dxgi}");
        }
        assert!(super::format(42).unwrap().is_integer() && !super::format(26).unwrap().is_integer());
    }

    /// Both BC6H formats, apart: the unsigned one (95) and the signed one (96), 16-byte blocks of
    /// 4x4 texels as the title's.
    #[test]
    fn bc6h_keeps_its_signedness() {
        use victory_backend::resources::Format;
        assert_eq!(super::format(95), Some(Format::Bc6h));
        assert_eq!(super::format(96), Some(Format::Bc6hSigned));
        for dxgi in [95, 96] {
            let (bytes, block) = crate::resource::element(dxgi).expect("a title format");
            assert_eq!(
                super::format(dxgi).unwrap().block(),
                (block as usize, bytes as usize)
            );
        }
    }

    #[test]
    fn r8g8_formats_keep_their_numeric_kind() {
        use victory_backend::resources::Format;
        assert_eq!(super::format(49), Some(Format::Rg8Unorm));
        assert_eq!(super::format(50), Some(Format::Rg8Uint));
        assert_eq!(super::format(51), Some(Format::Rg8Snorm));
        assert!(super::format(50).unwrap().is_integer());
        assert!(!super::format(51).unwrap().is_integer());
    }

    #[test]
    fn bgra8_srgb_has_a_host_format() {
        use victory_backend::resources::Format;
        assert_eq!(super::format(91), Some(Format::Bgra8UnormSrgb));
        assert!(!super::format(91).unwrap().is_integer());
    }

    #[test]
    fn b4g4r4a4_texels_become_abgr4() {
        // DXGI: blue 1 in bits 0-3, green 2, red 3, alpha 4 in bits 12-15. Metal: alpha low, red top.
        let mut t = 0x4321u16.to_le_bytes().to_vec();
        crate::native::textures::to_host(115, &mut t);
        assert_eq!(u16::from_le_bytes([t[0], t[1]]), 0x3214);
        let mut t = 0x4321u16.to_le_bytes().to_vec();
        crate::native::textures::to_host(86, &mut t);
        assert_eq!(u16::from_le_bytes([t[0], t[1]]), 0x4321);
    }

    use super::*;

    type Resources = HashMap<usize, Res<()>>;

    #[test]
    fn texture_srvs_keep_all_remaining_mips_from_the_first_requested() {
        let mut srv = [0; 10];
        (srv[0], srv[1], srv[4], srv[5]) = (28, TEXTURE_2D, 2, u32::MAX);
        let mut placed = [0; 16];
        placed[..srv.len()].copy_from_slice(&srv);
        let mut resource = [0; 14];
        resource[0] = 3; // D3D12_RESOURCE_DIMENSION_TEXTURE2D

        for view in [View::Shader(7, Some(srv)), View::Placed(false, resource, placed)] {
            let Some(Decoded::Image { mips, .. }) = decode(&view, &Resources::new()) else {
                panic!("expected a texture view");
            };
            assert_eq!(mips, Some((2, u32::MAX)));
        }
    }

    #[test]
    fn default_srv_uses_the_whole_registered_resource() {
        let resources = Resources::from([(7, Res::Texture(())), (8, Res::Buffer(0x1000, 256))]);
        assert_eq!(
            decode(&View::Shader(7, None), &resources),
            Some(Decoded::Image { resource: 7, mips: None, slices: None, format: None })
        );
        assert_eq!(
            decode(&View::Shader(8, None), &resources),
            Some(Decoded::Buffer { addr: 0x1000, size: 256 })
        );
        assert_eq!(decode(&View::Shader(9, None), &resources), None);
        assert_eq!(decode(&View::Unordered(7, 0, None), &resources), None);
    }

    #[test]
    fn default_texture_srv_replaces_the_texture_in_the_backend_slot() {
        use crate::native::{host::NativeHost, vertex_tests::Capture};
        use victory_backend::resources::{HeapKind, TexDim, TextureDesc};

        let host = NativeHost::new(Capture::default());
        let mut slot = [0u32; 8];
        let handle = slot.as_mut_ptr() as u64;
        let heap = host.heap(HeapKind::Resource, handle, 1).unwrap();
        let a = host.texture(&TextureDesc {
            dim: TexDim::D3,
            depth_or_layers: 2,
            ..TextureDesc::d2(Format::R8Unorm, 4, 4)
        }).unwrap();
        let key_a = std::rc::Rc::as_ptr(&a) as usize;
        host.register_texture(key_a, &a);

        let mut srv = [0u32; 10];
        srv[0] = 61; // R8_UNORM
        srv[1] = 8; // TEXTURE3D
        srv[2] = 5768; // default component mapping
        srv[5] = 1;
        crate::descriptor::create_srv(std::ptr::null_mut(), key_a as _, &srv, handle);
        assert!(host.mirror(handle).unwrap());
        assert_eq!(heap.heap.borrow()[0].unwrap().format, Format::R8Unorm);

        for format in [Format::Rgba8UnormSrgb, Format::D32Float] {
            let b = host.texture(&TextureDesc {
                dim: TexDim::D2Array,
                mips: 4,
                depth_or_layers: 3,
                render_target: format.is_depth(),
                ..TextureDesc::d2(format, 8, 8)
            }).unwrap();
            let key_b = std::rc::Rc::as_ptr(&b) as usize;
            host.register_texture(key_b, &b);
            crate::descriptor::create_srv(std::ptr::null_mut(), key_b as _, std::ptr::null(), handle);
            assert!(host.mirror(handle).unwrap());
            let bound = heap.heap.borrow()[0].unwrap();
            assert_eq!(bound.format, format);
            assert_eq!((bound.dim, bound.mips, bound.depth_or_layers), (TexDim::D2Array, 4, 3));
            crate::descriptor::forget(key_b);
        }
        crate::descriptor::forget(key_a);
    }

    #[test]
    fn array_texture_views_preserve_remaining_counts_and_refuse_empty_ranges() {
        let mut resource = [0; 14];
        resource[0] = 3;
        for unordered in [false, true] {
            for count in [0, 1, u32::MAX] {
                let mut words = [0; 10];
                words[..2].copy_from_slice(&[28, TEXTURE_2D_ARRAY]);
                let object = if unordered {
                    words[2..5].copy_from_slice(&[1, 2, count]);
                    View::Unordered(7, 0, Some(words))
                } else {
                    words[4..8].copy_from_slice(&[1, u32::MAX, 2, count]);
                    View::Shader(7, Some(words))
                };
                let mut placed = [0; 16];
                placed[..10].copy_from_slice(&words);
                for view in [object, View::Placed(unordered, resource, placed)] {
                    let decoded = decode(&view, &Resources::new());
                    if count == 0 {
                        assert_eq!(decoded, None);
                    } else {
                        let Some(Decoded::Image { mips, slices, .. }) = decoded else {
                            panic!("expected a texture array view");
                        };
                        assert_eq!(mips, Some((1, if unordered { 1 } else { u32::MAX })));
                        assert_eq!(slices, Some((2, count)));
                    }
                }
            }
        }
    }

    #[test]
    fn buffer_views_decode_to_raw_ranges() {
        let base = Resources::from([(7, Res::Buffer(0x1000_0000, 0x1_0000))]);
        // A raw SRV: 16 elements of 4 bytes from element 4.
        let mut srv = [0u32; 10];
        srv[1] = BUFFER;
        srv[4] = 4;
        srv[6] = 16;
        srv[8] = FLAG_RAW;
        assert_eq!(
            decode(&View::Shader(7, Some(srv)), &base),
            Some(Decoded::Buffer {
                addr: 0x1000_0010,
                size: 64
            })
        );
        // A structured UAV: 5 elements of 12 bytes.
        let mut uav = [0u32; 10];
        uav[1] = BUFFER;
        uav[4] = 5;
        uav[5] = 12;
        assert_eq!(
            decode(&View::Unordered(7, 0, Some(uav)), &base),
            Some(Decoded::Buffer {
                addr: 0x1000_0000,
                size: 60
            })
        );
        // A typed R32_FLOAT view: the format gives the element size.
        srv[0] = 41;
        srv[4] = 0;
        srv[8] = 0;
        assert_eq!(
            decode(&View::Shader(7, Some(srv)), &base),
            Some(Decoded::Buffer {
                addr: 0x1000_0000,
                size: 64
            })
        );
        assert_eq!(decode(&View::Shader(8, Some(srv)), &base), None);
        assert_eq!(
            decode(&View::Constant(0x2000, 256), &base),
            Some(Decoded::Buffer {
                addr: 0x2000,
                size: 256
            })
        );
    }

    #[test]
    fn console_samplers_preserve_supported_border_modes_and_colors() {
        let sampler = |mode, rgba: [f32; 4]| {
            let mut words = [0; 27];
            words[6..9].copy_from_slice(&[mode, 3, 3]);
            words[17..21].copy_from_slice(&rgba.map(f32::to_bits));
            let Some(Decoded::Sampler { desc, .. }) = decode(&View::ConsoleSampler(words), &Resources::new()) else {
                panic!("expected a sampler");
            };
            desc
        };

        for (rgba, color) in [
            ([0.0; 4], BorderColor::TransparentBlack),
            ([0.0, 0.0, 0.0, 1.0], BorderColor::OpaqueBlack),
            ([1.0; 4], BorderColor::OpaqueWhite),
        ] {
            let border = sampler(4, rgba);
            let edge = sampler(3, rgba);
            assert_eq!(border.address, [Address::Border, Address::Clamp, Address::Clamp]);
            assert_eq!(border.border_color, color);
            assert_eq!(edge.address, [Address::Clamp; 3]);
            assert_eq!(edge.border_color, BorderColor::TransparentBlack);
            assert_ne!(border, edge);
        }
        let custom = [0.25, 0.5, 0.0, 1.0];
        assert_eq!(sampler(4, custom), sampler(3, custom), "unsupported color retains edge clamp");
    }

    #[test]
    fn console_samplers_preserve_minification_with_nearest_magnification() {
        let sampler = |min| {
            let mut words = [0; 27];
            words[2] = min;
            let Some(Decoded::Sampler { desc, .. }) = decode(&View::ConsoleSampler(words), &Resources::new())
            else {
                panic!("expected a sampler");
            };
            desc
        };
        let nearest = sampler(0);
        let linear = sampler(1);
        assert_eq!((nearest.min, nearest.mag), (Filter::Nearest, Filter::Nearest));
        assert_eq!((linear.min, linear.mag), (Filter::Linear, Filter::Nearest));
        assert_ne!(nearest, linear);
    }

    #[test]
    fn console_samplers_decode() {
        let mut w = [0u32; 27];
        w[1] = 1; // bilinear magnification
        w[3] = 2; // linear mips
        w[6..9].copy_from_slice(&[1, 2, 3]);
        let Some(Decoded::Sampler { desc, .. }) = decode(&View::ConsoleSampler(w), &Resources::new())
        else {
            panic!("not a sampler");
        };
        assert_eq!((desc.min, desc.mag), (Filter::Nearest, Filter::Linear));
        assert_eq!(desc.mip, Filter::Linear);
        assert_eq!(
            desc.address,
            [Address::Wrap, Address::Mirror, Address::Clamp]
        );
    }

    #[test]
    fn typed_and_structured_buffer_views_carry_texture_buffer_formats() {
        let resources = Resources::from([(8, Res::Buffer(0x1000, 4096))]);
        let srv = |fmt, stride, flags| {
            let mut d = [0u32; 10];
            (d[0], d[1], d[6], d[7], d[8]) = (fmt, BUFFER, 8, stride, flags);
            View::Shader(8, Some(d))
        };
        let uav = |fmt, stride, flags| {
            let mut d = [0u32; 10];
            (d[0], d[1], d[4], d[5], d[8]) = (fmt, BUFFER, 8, stride, flags);
            View::Unordered(8, 0, Some(d))
        };
        assert_eq!(
            super::typed_buffer(&srv(0, 16, 0), &resources),
            Some((Format::Rgba32Float, false, false))
        );
        assert_eq!(
            super::typed_buffer(&uav(0, 8, 0), &resources),
            Some((Format::Rg32Float, true, false))
        );
        assert_eq!(
            super::typed_buffer(&srv(0, 4, 0), &resources),
            Some((Format::R32Float, false, false))
        );
        assert_eq!(super::typed_buffer(&uav(0, 12, 0), &resources), None);
        assert_eq!(super::typed_buffer(&uav(41, 0, FLAG_RAW), &resources), None);
        assert_eq!(super::typed_buffer(&View::Constant(0x1000, 256), &resources), None);
        // Typed-buffer-only DXGI formats decode on buffers without turning into texture formats.
        assert_eq!(super::format(12), None);
        assert_eq!(
            super::typed_buffer(&uav(12, 0, 0), &resources),
            Some((Format::Rgba16Uint, true, true))
        );
        assert_eq!(
            super::typed_buffer(&srv(4, 0, 0), &resources),
            Some((Format::Rgba32Uint, false, true))
        );
        // Sub-32-bit SINT formats stay unmapped so Uint does not zero-extend signed channels.
        for sint in [14, 32, 38, 52, 59, 64] {
            assert_eq!(super::typed_buffer(&srv(sint, 0, 0), &resources), None, "DXGI {sint}");
        }
        for dxgi in [12, 25, 31, 37, 58, 63] {
            let (fmt, _, explicit) =
                super::typed_buffer(&srv(dxgi, 0, 0), &resources).expect("typed buffer format");
            assert!(explicit, "DXGI {dxgi}");
            let (bytes, block) = crate::resource::element(dxgi).expect("title element");
            assert_eq!(fmt.block(), (block as usize, bytes as usize), "DXGI {dxgi}");
        }
    }
}
