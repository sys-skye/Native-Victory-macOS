//! Xbox Series shaders to host shaders.
//!
//! An Xbox Series shader is a standard DXBC container ([`dxbc`]). Its `DXIL`
//! part ([`dxil`]) may carry Xbox additions around the LLVM bitcode: an
//! `XDAT` table before it indexing blobs after it, among them `xbop`
//! (compiler options) and the zlib-compressed GFX10 code. The bitcode may use
//! console-only intrinsics, so the GFX10 code is translated in general.

pub mod dxbc;
pub mod dxil;
pub mod emit;
pub mod gfx10;
pub mod io;
pub mod ir;
#[cfg(test)]
mod ir_tests;
pub mod native;
pub mod range;
#[cfg(test)]
mod range_tests;
pub mod resource;
pub mod structure;

pub use dxbc::{Container, FourCC, Part};
pub use dxil::{DxilProgram, ShaderKind, XboxExtras};
pub use native::{
    HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, native_programs,
};

/// A hash of what this translator is built from: its sources, manifest and
/// build script, the lock file's entries of the packages it depends on and the
/// compiler's version. Caches
/// of translations key on it, so a changed translator never serves an old
/// translation.
pub const SOURCE_HASH: &str = env!("VICTORY_SHADER_SOURCE_HASH");

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ShaderError {
    #[error("not a DXBC container")]
    NotDxbc,
    #[error("malformed shader: {0}")]
    Malformed(&'static str),
    #[error("GFX10 code at {offset:#x}: {reason}")]
    Isa { offset: usize, reason: &'static str },
    /// Valid code the translator cannot handle yet: an opcode name or a construct.
    #[error("GFX10 code at {offset:#x}: {what} is not supported")]
    Unsupported { offset: usize, what: &'static str },
    /// An FSPX/FSPG metadata resource record whose root type the reader does not take.
    #[error("FSPX metadata resource of root type {root}: {what}")]
    FspxRoot { root: u8, what: &'static str },
}

#[cfg(test)]
mod lock_tests;

#[cfg(test)]
mod tests {
    /// `build.rs` leaves `*_tests.rs` out of [`super::SOURCE_HASH`]: each must be a module only a
    /// test build compiles, or a change to it could change translations the cache still serves.
    #[test]
    fn test_modules_are_test_only() {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                if e.path().is_dir() { walk(&e.path(), out) } else { out.push(e.path()) }
            }
        }
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk(&src, &mut files);
        let texts: Vec<String> = files.iter().filter_map(|f| std::fs::read_to_string(f).ok()).collect();
        for f in files.iter().filter(|f| f.to_string_lossy().ends_with("_tests.rs")) {
            let module = f.file_stem().unwrap().to_string_lossy();
            let declared = format!("mod {module};");
            let gated = texts.iter().any(|t| {
                let lines: Vec<&str> = t.lines().map(str::trim).collect();
                lines.iter().enumerate().any(|(i, l)| {
                    l.ends_with(&declared)
                        && i > 0
                        && lines[i - 1].starts_with("#[cfg(")
                        && lines[i - 1].contains("test")
                })
            });
            assert!(gated, "{} is not declared under a cfg(test)", f.display());
        }
    }
}
