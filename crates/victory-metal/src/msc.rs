//! Metal Shader Converter (`libmetalirconverter`) loaded at run time: DXIL container -> metallib.
//!
//! The library is Apple's and not redistributable here, so it is dlopen'ed from a path the user
//! supplies (`VICTORY_MSC_DYLIB`). The C API is declared by hand from `metal_irconverter.h`.

use std::ffi::{CStr, CString, c_char, c_void};
use victory_backend::backend::{RootLayout, RootParam};

type Create = unsafe extern "C" fn() -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type FromDxil = unsafe extern "C" fn(*const u8, usize, i32) -> *mut c_void;
type Compile =
    unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_void, *mut *mut c_void) -> *mut c_void;
type Stage = unsafe extern "C" fn(*const c_void) -> i32;
type GetLib = unsafe extern "C" fn(*const c_void, i32, *mut c_void) -> bool;
type LibSize = unsafe extern "C" fn(*const c_void) -> usize;
type LibBytes = unsafe extern "C" fn(*const c_void, *mut u8) -> usize;
type ErrCode = unsafe extern "C" fn(*mut c_void) -> u32;
type SetEntry = unsafe extern "C" fn(*mut c_void, *const c_char);
type Payload = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type MakeSignature = unsafe extern "C" fn(*const Descriptor, *mut *mut c_void) -> *mut c_void;
type SetSignature = unsafe extern "C" fn(*mut c_void, *const c_void);
type SetFlags = unsafe extern "C" fn(*mut c_void, u32);
type SetDualSource = unsafe extern "C" fn(*mut c_void, u32);
type GetReflection = unsafe extern "C" fn(*const c_void, i32, *mut c_void) -> bool;
type CopyMeshInfo = unsafe extern "C" fn(*const c_void, u32, *mut MeshInfo) -> bool;
type ReleaseMeshInfo = unsafe extern "C" fn(*mut MeshInfo) -> bool;

/// `IRDualSourceBlendingConfigurationForceEnabled`: one target's two sources are shader outputs
/// 0 and 1, both `[[color(0)]]` (`index(0)` and `index(1)`).
const DUAL_SOURCE_ON: u32 = 1;

/// `IRShaderStageMesh`.
pub const STAGE_MESH: i32 = 5;

/// `IRVersionedMSInfo` holding `IRMSInfo_1_0`.
#[repr(C)]
#[derive(Default)]
struct MeshInfo {
    version: u32,
    max_vertices: u32,
    max_primitives: u32,
    topology: u32,
    payload: u32,
    threads: [u32; 3],
}

/// `IRRootParameter1`: type, a 16-byte union (constants `{register, space, count}`, descriptor
/// `{register, space, flags}`), shader visibility (all).
#[repr(C)]
struct Parameter {
    kind: u32,
    union: [u64; 2],
    visibility: u32,
}

/// `IRVersionedRootSignatureDescriptor` with version 1.1 (`IRRootSignatureDescriptor1`, no static
/// samplers).
#[repr(C)]
struct Descriptor {
    version: u32,
    /// The union that follows is 8-aligned.
    _pad: u32,
    count: u32,
    parameters: *const Parameter,
    samplers: u32,
    static_samplers: *const c_void,
    flags: u32,
}

/// `IRRootSignatureFlagCBVSRVUAVHeapDirectlyIndexed | IRRootSignatureFlagSamplerHeapDirectlyIndexed`.
const HEAPS_DIRECTLY_INDEXED: u32 = 0x400 | 0x800;
/// `IRCompatibilityFlagBoundsCheck`.
const BOUNDS_CHECK: u32 = 1;

impl Parameter {
    fn of(p: &RootParam) -> Parameter {
        // IRRootParameterType: table 0, 32-bit constants 1, CBV 2, SRV 3, UAV 4.
        let (kind, register, space, extra) = match *p {
            RootParam::Constants {
                register,
                space,
                dwords,
            } => (1, register, space, dwords),
            RootParam::Cbv { register, space } => (2, register, space, 0),
            RootParam::Srv { register, space } => (3, register, space, 0),
            RootParam::Uav { register, space } => (4, register, space, 0),
        };
        Parameter {
            kind,
            union: [u64::from(register) | u64::from(space) << 32, u64::from(extra)],
            visibility: 0,
        }
    }
}

/// Logs a slow conversion (and whether the disk cache had it).
fn note(started: std::time::Instant, dxil: usize, hit: bool) {
    let ms = started.elapsed().as_millis();
    if ms >= 20 {
        eprintln!("metal: library conversion {ms} ms (dxil {dxil} bytes, from the disk cache: {hit})");
    }
}

/// `IRResourceLocation`: where the converter puts one resource of a root signature in the
/// top-level argument buffer.
#[repr(C)]
struct Location {
    kind: u32,
    space: u32,
    slot: u32,
    offset: u32,
    size: u64,
    name: *const c_char,
}

type Count = unsafe extern "C" fn(*const c_void) -> usize;
type Locations = unsafe extern "C" fn(*const c_void, *mut Location);

/// The converter's root signature for `l` (destroyed with `IRRootSignatureDestroy`), or the code
/// it refused it with.
///
/// # Safety
/// `h` is the converter's dlopen handle.
unsafe fn make_signature(h: *mut c_void, l: &RootLayout) -> Result<*mut c_void, u32> {
    let params: Vec<Parameter> = l.params.iter().map(Parameter::of).collect();
    // `IRStaticSamplerDescriptor` (`D3D12_STATIC_SAMPLER_DESC`): the layout's ten words,
    // then register n, space 0, visibility all.
    let samplers: Vec<[u32; 13]> = l
        .static_samplers
        .iter()
        .enumerate()
        .map(|(n, w)| {
            let mut d = [0u32; 13];
            d[..10].copy_from_slice(w);
            d[10] = n as u32;
            d
        })
        .collect();
    let d = Descriptor {
        version: 2,
        _pad: 0,
        count: params.len() as u32,
        parameters: params.as_ptr(),
        samplers: samplers.len() as u32,
        static_samplers: samplers.as_ptr().cast(),
        flags: HEAPS_DIRECTLY_INDEXED,
    };
    let mut err = std::ptr::null_mut();
    // SAFETY: the converter's documented C API; `d` and what it points at outlive the call.
    unsafe {
        let signature =
            sym::<MakeSignature>(h, "IRRootSignatureCreateFromDescriptor")(&d, &mut err);
        if signature.is_null() {
            return Err(if err.is_null() {
                0
            } else {
                let code = sym::<ErrCode>(h, "IRErrorGetCode")(err);
                sym::<Destroy>(h, "IRErrorDestroy")(err);
                code
            });
        }
        Ok(signature)
    }
}

pub struct Msc {
    h: *mut c_void,
    /// Which converter this is (its path, size and modification time): part of the library cache's keys.
    id: String,
}

// SAFETY: the handle is a dlopen handle; every conversion makes and destroys its own IR objects.
unsafe impl Send for Msc {}
// SAFETY: as above, no state is shared between calls.
unsafe impl Sync for Msc {}

/// A converter failure: the `IRError` code and the converter's own message.
#[derive(Debug)]
pub struct ConvError {
    pub code: u32,
    pub msg: String,
}

impl std::fmt::Display for ConvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IRError code {}: {}", self.code, self.msg)
    }
}

fn sym<T: Copy>(h: *mut c_void, name: &str) -> T {
    let n = CString::new(name).unwrap();
    // SAFETY: h is a live dlopen handle and T is the function-pointer type of the named C symbol.
    let p = unsafe { libc::dlsym(h, n.as_ptr()) };
    assert!(!p.is_null(), "missing symbol {name}");
    // SAFETY: T is a pointer-sized fn pointer type.
    unsafe { std::mem::transmute_copy(&p) }
}

/// Where Apple's installer puts the converter (universal binary); `VICTORY_MSC_DYLIB` overrides.
pub const DEFAULT_DYLIB: &str = "/usr/local/lib/libmetalirconverter.dylib";

/// The converter to load, and whether `VICTORY_MSC_DYLIB` named it (an empty value counts as
/// unset, as in `native-setup.sh`).
fn dylib_path(var: Option<String>) -> (String, bool) {
    match var.filter(|v| !v.is_empty()) {
        Some(v) => (v, true),
        None => (DEFAULT_DYLIB.into(), false),
    }
}

/// The start-up line naming the loaded converter (`id`: its path, size and modification time),
/// and a warning when it is not the installed default. The library exposes no version to log:
/// its Mach-O current version is 0.0.0 and the C API has no version call.
fn load_note(path: &str, from_var: bool, id: &str) -> (String, Option<String>) {
    let source = if from_var { "VICTORY_MSC_DYLIB" } else { "the installed default" };
    let note = format!("metal: shader converter {id} (from {source})");
    let warn = (path != DEFAULT_DYLIB).then(|| {
        format!(
            "metal: the shader converter is not {DEFAULT_DYLIB}; a copy other than Apple's package \
             may convert shaders differently (a non-Apple copy drew no loading-screen art, #295)"
        )
    });
    (note, warn)
}

/// An export that Apple's converter has from 3.0 on (function constants), and no earlier release
/// has: 1.0, 1.1 and 2.1 lack it, 3.0 and 4.0 beta 3 export it. The C API has no version call,
/// so the generation is told by what the library exports.
const FIRST_IN_3_0: &str = "IRCompilerSetFunctionConstantResourceSpace";

/// Why the converter at `path` cannot be used, if it is older than 3.0 (`has`: whether the library
/// exports a symbol). Converters before 3.0 take every 1D and 2D texture as an array type (their
/// manual, "Texture arrays"), and 3.0 made plain types the default. The host binds a 2D texture
/// as a plain 2D view, which is what 3.0 and later expect: through an older converter, a compute
/// shader's writes to such a texture are dropped and the memory stays zero, with no error from
/// Metal (`a_texture_compute_writes_is_copied_back_into_title_memory` fails on 1.1 and 2.1, and
/// passes on them with the views made as 2D arrays, which in turn fails on 3.0). That is the
/// missing loading art of #295, whose converter (Unreal Engine 5.6's, `D3DRendererMetal-30.2`) is
/// older than Apple's 2.1 (`31.2`).
fn too_old(path: &str, has: impl Fn(&str) -> bool) -> Option<String> {
    (!has(FIRST_IN_3_0)).then(|| {
        format!(
            "{path} is a Metal Shader Converter older than 3.0 (it does not export {FIRST_IN_3_0}). \
             Converters before 3.0 need every 1D and 2D texture bound as an array, and this host \
             binds plain ones, so their shaders would read and write nothing (#295). Install \
             Apple's 3.0 or newer from https://developer.apple.com/metal/shader-converter/"
        )
    })
}

impl Msc {
    /// Opens `VICTORY_MSC_DYLIB` if set, else the installed default.
    /// Logs, once per run, which converter it loaded (#295: a non-Apple copy drew no loading art),
    /// or why none was.
    pub fn try_open_default() -> Result<Msc, String> {
        let (path, from_var) = dylib_path(std::env::var("VICTORY_MSC_DYLIB").ok());
        let opened = Msc::open(&path);
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| match &opened {
            Ok(msc) => {
                let (note, warn) = load_note(&path, from_var, &msc.id);
                tracing::info!("{note}");
                if let Some(w) = warn {
                    tracing::warn!("{w}");
                }
            }
            Err(e) => tracing::error!("metal: no shader converter: {e}"),
        });
        opened
    }

    /// Opens `VICTORY_MSC_DYLIB` if set, else the installed default, panicking on failure.
    pub fn open_default() -> Msc {
        Self::try_open_default().unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn open(path: &str) -> Result<Msc, String> {
        let c = CString::new(path).unwrap();
        // SAFETY: dlopen of a user-named library.
        let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW) };
        if h.is_null() {
            // SAFETY: dlerror returns a C string or null.
            let e = unsafe { libc::dlerror() };
            let e = if e.is_null() {
                "?".into()
            } else {
                // SAFETY: non-null NUL-terminated string from dlerror.
                unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
            };
            return Err(format!("dlopen {path}: {e}"));
        }
        let has = |name: &str| {
            let n = CString::new(name).unwrap();
            // SAFETY: h is a live dlopen handle.
            !unsafe { libc::dlsym(h, n.as_ptr()) }.is_null()
        };
        if let Some(e) = too_old(path, has) {
            // SAFETY: h came from dlopen above and nothing from it is kept.
            unsafe { libc::dlclose(h) };
            return Err(e);
        }
        let id = std::fs::metadata(path)
            .map(|m| format!("{path} {} {:?}", m.len(), m.modified().ok()))
            .unwrap_or_else(|_| path.to_owned());
        Ok(Msc { h, id })
    }

    /// DXIL (a DXBC-wrapped container) -> (shader stage, metallib bytes). Stage 8 is compute (`IRShaderStage`).
    pub fn convert(&self, dxil: &[u8], entry: Option<&str>) -> Result<(i32, Vec<u8>), ConvError> {
        self.convert_with(dxil, entry, None, false)
    }

    /// Where the converter lays out `l` in the top-level argument buffer: (resource type, space,
    /// register, byte offset, size) per resource, as `IRRootSignatureGetResourceLocations` gives.
    pub fn locations(&self, l: &RootLayout) -> Result<Vec<(u32, u32, u32, u32, u64)>, ConvError> {
        let h = self.h;
        // SAFETY: the converter's documented C API; the signature is destroyed below.
        unsafe {
            let signature = make_signature(h, l).map_err(|code| ConvError {
                code,
                msg: "the root signature was refused".into(),
            })?;
            let n = sym::<Count>(h, "IRRootSignatureGetResourceCount")(signature);
            let mut v: Vec<Location> = (0..n)
                .map(|_| Location {
                    kind: 0,
                    space: 0,
                    slot: 0,
                    offset: 0,
                    size: 0,
                    name: std::ptr::null(),
                })
                .collect();
            sym::<Locations>(h, "IRRootSignatureGetResourceLocations")(signature, v.as_mut_ptr());
            sym::<Destroy>(h, "IRRootSignatureDestroy")(signature);
            Ok(v.iter()
                .map(|l| (l.kind, l.space, l.slot, l.offset, l.size))
                .collect())
        }
    }

    /// [`Self::convert`] with the top-level argument buffer laid out as `layout` says (for DXIL
    /// that carries no root signature), and out-of-range raw buffer accesses made safe if it asks.
    /// `dual_source`: the pixel shader's second output is target 0's second blend source, which
    /// the converter only emits as `[[color(0) index(1)]]` when told.
    pub fn convert_with(
        &self,
        dxil: &[u8],
        entry: Option<&str>,
        layout: Option<&RootLayout>,
        dual_source: bool,
    ) -> Result<(i32, Vec<u8>), ConvError> {
        let Some(dir) = crate::disk::dir() else {
            return self.convert_uncached(dxil, entry, layout, dual_source);
        };
        let started = std::time::Instant::now();
        let layout_text = format!("{layout:?}");
        // A dual-source conversion is a different shader laid out for the same DXIL, so it gets
        // an entry of its own; the other case keeps the key it had, so those entries stay warm.
        let mut parts: Vec<&[u8]> = Vec::new();
        if dual_source {
            parts.push(b"dual-source");
        }
        parts.extend([
            dxil,
            entry.unwrap_or("").as_bytes(),
            layout_text.as_bytes(),
            self.id.as_bytes(),
        ]);
        let key = crate::disk::key(&parts);
        if let Some((stage, lib)) = crate::disk::read(dir, &key).and_then(|b| {
            let (stage, lib) = b.split_at_checked(4)?;
            Some((i32::from_le_bytes(stage.try_into().ok()?), lib.to_vec()))
        }) {
            note(started, dxil.len(), true);
            return Ok((stage, lib));
        }
        let (stage, lib) = self.convert_uncached(dxil, entry, layout, dual_source)?;
        crate::disk::write(dir, &key, &[&stage.to_le_bytes()[..], &lib].concat());
        note(started, dxil.len(), false);
        Ok((stage, lib))
    }

    /// The thread-group size of a mesh shader, from the converter's reflection of it (cached on
    /// disk as conversions are).
    pub fn mesh_threads(&self, dxil: &[u8], layout: Option<&RootLayout>) -> Result<[u32; 3], ConvError> {
        let layout_text = format!("{layout:?}");
        let dir = crate::disk::dir();
        let key = crate::disk::key(&[b"mesh threads", dxil, layout_text.as_bytes(), self.id.as_bytes()]);
        if let Some(t) = dir.and_then(|d| crate::disk::read(d, &key)).and_then(|b| {
            let w = |k: usize| Some(u32::from_le_bytes(b.get(k * 4..k * 4 + 4)?.try_into().ok()?));
            Some([w(0)?, w(1)?, w(2)?])
        }) {
            return Ok(t);
        }
        let h = self.h;
        let threads = self.compiled(dxil, None, layout, false, |out| {
            // SAFETY: the converter's documented C API (`IRShaderReflection*`, `IRObjectGetReflection`,
            // `IRShaderReflectionCopyMeshInfo` with `IRReflectionVersion_1_0`); the reflection is
            // destroyed before returning.
            unsafe {
                let r = sym::<Create>(h, "IRShaderReflectionCreate")();
                let mut info = MeshInfo::default();
                let ok = sym::<GetReflection>(h, "IRObjectGetReflection")(out, STAGE_MESH, r)
                    && sym::<CopyMeshInfo>(h, "IRShaderReflectionCopyMeshInfo")(r, 1, &mut info);
                if ok {
                    sym::<ReleaseMeshInfo>(h, "IRShaderReflectionReleaseMeshInfo")(&mut info);
                }
                sym::<Destroy>(h, "IRShaderReflectionDestroy")(r);
                if ok {
                    Ok(info.threads)
                } else {
                    Err(ConvError { code: u32::MAX, msg: "no mesh reflection".into() })
                }
            }
        })?;
        if let Some(d) = dir {
            crate::disk::write(d, &key, &threads.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>());
        }
        Ok(threads)
    }

    fn convert_uncached(
        &self,
        dxil: &[u8],
        entry: Option<&str>,
        layout: Option<&RootLayout>,
        dual_source: bool,
    ) -> Result<(i32, Vec<u8>), ConvError> {
        let h = self.h;
        self.compiled(dxil, entry, layout, dual_source, |out| {
            // SAFETY: the converter's documented C API; the binary is destroyed below.
            unsafe {
                let stage = sym::<Stage>(h, "IRObjectGetMetalIRShaderStage")(out);
                let lib = sym::<Create>(h, "IRMetalLibBinaryCreate")();
                let r = if sym::<GetLib>(h, "IRObjectGetMetalLibBinary")(out, stage, lib) {
                    let n = sym::<LibSize>(h, "IRMetalLibGetBytecodeSize")(lib);
                    let mut v = vec![0u8; n];
                    sym::<LibBytes>(h, "IRMetalLibGetBytecode")(lib, v.as_mut_ptr());
                    Ok((stage, v))
                } else {
                    Err(ConvError {
                        code: u32::MAX,
                        msg: "no metallib for stage".into(),
                    })
                };
                sym::<Destroy>(h, "IRMetalLibBinaryDestroy")(lib);
                r
            }
        })
    }

    /// Compiles `dxil` and hands the converter's output object to `f`.
    fn compiled<T>(
        &self,
        dxil: &[u8],
        entry: Option<&str>,
        layout: Option<&RootLayout>,
        dual_source: bool,
        f: impl FnOnce(*mut c_void) -> Result<T, ConvError>,
    ) -> Result<T, ConvError> {
        let h = self.h;
        // SAFETY: the symbols are the converter's documented C API; every object is destroyed below.
        unsafe {
            let comp = sym::<Create>(h, "IRCompilerCreate")();
            // Older converters have no dual-source switch; those keep refusing such pipelines.
            // Every other conversion leaves the option alone, as before.
            if dual_source {
                let set = libc::dlsym(h, c"IRCompilerSetDualSourceBlendingConfiguration".as_ptr());
                if !set.is_null() {
                    let set: SetDualSource = std::mem::transmute_copy(&set);
                    set(comp, DUAL_SOURCE_ON);
                }
            }
            let mut signature = std::ptr::null_mut();
            if let Some(l) = layout {
                signature = match make_signature(h, l) {
                    Ok(s) => s,
                    Err(code) => {
                        sym::<Destroy>(h, "IRCompilerDestroy")(comp);
                        return Err(ConvError {
                            code,
                            msg: "the root signature was refused".into(),
                        });
                    }
                };
                sym::<SetSignature>(h, "IRCompilerSetGlobalRootSignature")(comp, signature);
                if l.bounds_checked {
                    sym::<SetFlags>(h, "IRCompilerSetCompatibilityFlags")(comp, BOUNDS_CHECK);
                }
            }
            if let Some(e) = entry {
                let e = CString::new(e).unwrap();
                sym::<SetEntry>(h, "IRCompilerSetEntryPointName")(comp, e.as_ptr());
            }
            // IRBytecodeOwnershipNone = 0: the converter borrows `dxil` for the call.
            let input = sym::<FromDxil>(h, "IRObjectCreateFromDXIL")(dxil.as_ptr(), dxil.len(), 0);
            let mut err: *mut c_void = std::ptr::null_mut();
            let out = sym::<Compile>(h, "IRCompilerAllocCompileAndLink")(
                comp,
                std::ptr::null(),
                input,
                &mut err,
            );
            let res = if out.is_null() {
                let (code, msg) = if err.is_null() {
                    (0, String::new())
                } else {
                    // The payload is the message text (the library formats "<Name> : <detail>").
                    let p = sym::<Payload>(h, "IRErrorGetPayload")(err);
                    let msg = if p.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(p).to_string_lossy().into_owned()
                    };
                    (sym::<ErrCode>(h, "IRErrorGetCode")(err), msg)
                };
                Err(ConvError { code, msg })
            } else {
                let r = f(out);
                sym::<Destroy>(h, "IRObjectDestroy")(out);
                r
            };
            if !err.is_null() {
                sym::<Destroy>(h, "IRErrorDestroy")(err);
            }
            sym::<Destroy>(h, "IRObjectDestroy")(input);
            sym::<Destroy>(h, "IRCompilerDestroy")(comp);
            if !signature.is_null() {
                sym::<Destroy>(h, "IRRootSignatureDestroy")(signature);
            }
            res
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_converter_path_is_the_override_if_set_else_the_default() {
        assert_eq!(dylib_path(None), (DEFAULT_DYLIB.into(), false));
        assert_eq!(dylib_path(Some(String::new())), (DEFAULT_DYLIB.into(), false));
        assert_eq!(dylib_path(Some("/x/libmetalirconverter.dylib".into())), ("/x/libmetalirconverter.dylib".into(), true));
    }

    #[test]
    fn the_load_note_names_its_source_and_warns_only_off_the_default_path() {
        let (note, warn) = load_note(DEFAULT_DYLIB, false, "id");
        assert!(note.contains("id (from the installed default)"), "{note}");
        assert!(warn.is_none());
        // The variable naming the default path itself is not warned about.
        let (note, warn) = load_note(DEFAULT_DYLIB, true, "id");
        assert!(note.contains("from VICTORY_MSC_DYLIB"), "{note}");
        assert!(warn.is_none());
        let (_, warn) = load_note("/ue/libmetalirconverter.dylib", true, "id");
        assert!(warn.unwrap().contains(DEFAULT_DYLIB));
    }

    /// The exports of Apple's releases decide: 2.1 (and 1.x) lack the 3.0 export and are refused
    /// with a reason that names the path and the fix; 3.0 and later are taken.
    #[test]
    fn a_converter_older_than_3_is_refused_by_its_exports() {
        let before_3 = |name: &str| name == "IRCompilerAllocCompileAndLink";
        let from_3 = |name: &str| matches!(name, "IRCompilerAllocCompileAndLink" | FIRST_IN_3_0);
        let e = too_old("/ue/libmetalirconverter.dylib", before_3).expect("2.1 is refused");
        assert!(e.contains("/ue/libmetalirconverter.dylib"), "{e}");
        assert!(e.contains("older than 3.0") && e.contains("3.0 or newer"), "{e}");
        assert_eq!(too_old(DEFAULT_DYLIB, from_3), None);
    }
}
