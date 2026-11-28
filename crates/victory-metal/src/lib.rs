//! Native Metal backend for victory-gpu (macOS only; empty on other targets).
//!
//! `backend` is the platform-neutral seam, `metal` its Metal implementation, `msc` the Metal Shader
//! Converter loaded at run time, `mem` page-aligned host reservations for tests and probes.

pub use victory_backend::{backend, resources};
#[cfg(target_os = "macos")]
mod archive;
#[cfg(target_os = "macos")]
pub use archive::{CRASHES_REMEMBERED, is_compiler_crash};
pub mod disk;
#[cfg(target_os = "macos")]
pub mod gfx;
#[cfg(target_os = "macos")]
pub mod mem;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "macos")]
pub mod msc;
#[cfg(target_os = "macos")]
pub mod present;
