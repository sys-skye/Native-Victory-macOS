//! The native host: victory-gpu's work for a host GPU API reached through `victory_backend`'s
//! traits instead of `windows::Win32::Graphics::Direct3D12`. Platform neutral and generic over the
//! backend, so it builds (and is linted) on the Windows target too; on macOS it is instantiated with
//! `victory_metal::metal::Metal` ([`Host`]). The D3D12 host (`host.rs`, `draw.rs`, ...) is untouched.
//!
//! Module selection is by `cfg`, not at run time: `crate::host` and everything under it exist only
//! with `cfg(windows)`, and this module is what a macOS build links the title-facing objects
//! (`list.rs`, `queue.rs`, `fence.rs`) to.
//!
//! [`memory`] title memory as zero-copy pools (no shadow, watch or copy-back), [`descriptors`]
//! the title's views for the backend, [`list`] a command list, [`host`] queues, pipelines, heaps and
//! submission, [`title`] the title's queues, fences and lists on top of them; `glue` (macOS) is
//! the one instance the title-facing objects call, and its [`pool_observer`] is what the runtime's
//! physical-memory observer forwards to.

/// Whether environment variable `$name` is set, read once per call site (draws ask per draw).
macro_rules! env_set {
    ($name:literal) => {{
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os($name).is_some())
    }};
}

/// Whether the pass trace's window is open (`VICTORY_TRACE_PASSES`, see `glue::pass_present`):
/// false unless that variable is set, so every pass-trace line costs one relaxed load otherwise.
static PASS_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn pass_on() -> bool {
    PASS_ON.load(std::sync::atomic::Ordering::Relaxed)
}

pub mod addr;
pub mod console_state;
pub(crate) mod descriptor_trace;
pub mod descriptors;
pub mod freed;
#[cfg(target_os = "macos")]
pub(crate) mod glue;
pub mod host;
pub mod list;
pub mod memory;
pub mod shaders;
pub mod state;
pub mod textures;
pub mod title;

#[cfg(target_os = "macos")]
pub type Host = host::NativeHost<victory_metal::metal::Metal>;

#[cfg(target_os = "macos")]
pub use glue::{
    create_layer, frames_presented, last_frame, pool_observer, pools, set_event_hook, set_layer,
    shader_stats,
};

/// Reads `out.len()` bytes at `at` from this process, false if they cannot be read (unmapped
/// memory: the kernel checks, nothing faults).
pub fn safe_read(at: *const u8, out: &mut [u8]) -> bool {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            static mach_task_self_: u32;
            fn mach_vm_read_overwrite(
                task: u32,
                address: u64,
                size: u64,
                data: u64,
                out_size: *mut u64,
            ) -> i32;
        }
        let mut got = 0u64;
        // SAFETY: `out` is writable for its length; the kernel validates the source range.
        let r = unsafe {
            mach_vm_read_overwrite(
                mach_task_self_,
                at as u64,
                out.len() as u64,
                out.as_mut_ptr() as u64,
                &mut got,
            )
        };
        r == 0 && got == out.len() as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (at, out);
        false
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests;

#[cfg(test)]
mod vertex_tests;
