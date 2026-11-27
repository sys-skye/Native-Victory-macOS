//! Implementations, one module per area of the console API.
//!
//! Adding an API: write an `extern "system"` function named exactly like
//! the import, then list it with `hle_export!` in the module's `module()`.
//! Run `victory census` over your titles to see what is still missing.
//!
//! Planned areas (driven by census results, not by completeness):
//! `file` (package paths), `thread` (TLS callbacks for title threads),
//! `xaudio2`, `xstore` stubs.

#[cfg(any(windows, all(target_os = "macos", target_arch = "x86_64")))]
pub mod apu;
pub mod audio;
pub mod d3d12x;
pub mod display;
pub mod dsp;
#[cfg(any(windows, target_os = "macos"))]
pub mod dstorage;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
mod read_cache;
#[cfg(windows)]
pub mod fidelityfx;
#[cfg(windows)]
pub mod files;
pub mod gameinput;
pub mod kernel;
#[cfg(windows)]
pub mod lifecycle;
#[cfg(windows)]
pub mod media;
#[cfg(windows)]
pub mod net;
pub mod rpcrt4;
pub mod speech;
pub mod storage;
pub mod xasync;
pub mod xg;
pub mod xgame;
pub mod xgameruntime;
pub mod xgamesave;
pub mod xgameui;
#[cfg(any(windows, target_os = "macos"))]
pub mod xmem;
#[cfg(any(windows, target_os = "macos"))]
pub mod xmem_physical;
#[cfg(target_os = "macos")]
mod xmem_posix;
pub mod xnetworking;
pub mod xpackage;
pub mod xstore;
pub mod xsystem;
pub mod xuser;

use crate::HleModule;

pub(crate) fn all() -> Vec<HleModule> {
    let mut all = vec![
        kernel::module(),
        xgameruntime::module(),
        gameinput::module(),
        d3d12x::module(),
        audio::module(),
        dsp::module(),
        xg::module(),
    ];
    #[cfg(any(windows, all(target_os = "macos", target_arch = "x86_64")))]
    all.push(apu::module());
    #[cfg(windows)]
    all.push(media::module());
    #[cfg(any(windows, target_os = "macos"))]
    all.push(dstorage::module());
    #[cfg(windows)]
    all.push(files::module());
    #[cfg(windows)]
    all.push(lifecycle::module());
    #[cfg(windows)]
    all.push(net::module());
    all.push(rpcrt4::module());
    #[cfg(any(windows, target_os = "macos"))]
    all.push(xmem::module());
    #[cfg(windows)]
    all.push(xsystem::module());
    #[cfg(windows)]
    if fidelityfx::serving() {
        all.push(fidelityfx::module());
    }
    all
}
