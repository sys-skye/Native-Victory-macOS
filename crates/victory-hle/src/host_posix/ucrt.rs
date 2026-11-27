//! The Universal C runtime for the native host: `api-ms-win-crt-*` and
//! `ucrtbase.dll` on macOS, in Rust over libSystem.
//!
//! Nothing here hands a Windows `va_list`, `wchar_t` or `FILE` to macOS:
//! `wchar_t` is two bytes (UTF-16), `va_list` is a pointer to 8-byte slots,
//! and `FILE*` is an opaque handle into our own stream table over
//! `host_posix::files` paths. Functions with a `float` or `double` argument
//! are `extern "win64"` (declared to the bridge), the rest are `extern
//! "system"` and the bridge remaps their integer arguments. Callbacks into
//! title code (`qsort`, `_initterm`, `atexit`) go through `call_win64`.
//!
//! Simplifications, all of the "C" locale: strings are UTF-8 (the process
//! code page), `setlocale` accepts only "C", the invalid-parameter handler
//! is called if the title set one and the call then returns its error code
//! (no fast-fail), text-mode files strip `\r` on reads and write `\n` as is.
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![cfg_attr(
    test,
    allow(
        clippy::approx_constant,
        clippy::manual_c_str_literals,
        clippy::undocumented_unsafe_blocks
    )
)]

use std::cell::UnsafeCell;

mod convert;
mod format;
mod fs;
mod heap;
mod math;
mod runtime;
mod stdio;
mod string;
mod time;

pub use runtime::set_command_line;

/// The CRT `exit` path's `atexit`/`onexit` handlers and stream flush, for the HLE `exit`.
pub(crate) fn run_exit_handlers() {
    runtime::_cexit();
}

/// libSystem functions this module calls (the allocator and `libm`).
pub(super) mod libc {
    use std::ffi::{c_int, c_void};
    unsafe extern "C" {
        pub fn malloc(n: usize) -> *mut c_void;
        pub fn calloc(n: usize, m: usize) -> *mut c_void;
        pub fn realloc(p: *mut c_void, n: usize) -> *mut c_void;
        pub fn free(p: *mut c_void);
        pub fn posix_memalign(out: *mut *mut c_void, align: usize, n: usize) -> c_int;
        pub fn malloc_size(p: *const c_void) -> usize;
        pub fn arc4random_buf(p: *mut c_void, n: usize);
        pub fn _exit(code: c_int) -> !;
        pub fn erf(x: f64) -> f64;
        pub fn erfc(x: f64) -> f64;
        pub fn tgamma(x: f64) -> f64;
        pub fn lgamma(x: f64) -> f64;
        pub fn j0(x: f64) -> f64;
        pub fn j1(x: f64) -> f64;
        pub fn y0(x: f64) -> f64;
        pub fn y1(x: f64) -> f64;
        pub fn jn(n: c_int, x: f64) -> f64;
        pub fn yn(n: c_int, x: f64) -> f64;
        pub fn remainder(x: f64, y: f64) -> f64;
        pub fn nextafter(x: f64, y: f64) -> f64;
        pub fn nextafterf(x: f32, y: f32) -> f32;
        pub fn scalbn(x: f64, n: c_int) -> f64;
        pub fn logb(x: f64) -> f64;
        pub fn ilogb(x: f64) -> c_int;
        pub fn fmod(x: f64, y: f64) -> f64;
    }
}

pub(super) const ENOENT: i32 = 2;
pub(super) const EBADF: i32 = 9;
pub(super) const ENOMEM: i32 = 12;
pub(super) const EACCES: i32 = 13;
pub(super) const EEXIST: i32 = 17;
pub(super) const EINVAL: i32 = 22;
pub(super) const EMFILE: i32 = 24;
pub(super) const ENOSPC: i32 = 28;
pub(super) const EDOM: i32 = 33;
pub(super) const ERANGE: i32 = 34;
pub(super) const EILSEQ: i32 = 42;
pub(super) const STRUNCATE: i32 = 80;
/// `_TRUNCATE` and `SIZE_MAX`.
pub(super) const TRUNCATE: usize = usize::MAX;

thread_local! {
    static ERRNO: UnsafeCell<i32> = const { UnsafeCell::new(0) };
}

pub(super) fn set_errno(e: i32) {
    ERRNO.with(|c| {
        // SAFETY: this thread's own cell.
        unsafe { *c.get() = e }
    });
}

pub(super) fn errno() -> i32 {
    // SAFETY: this thread's own cell.
    ERRNO.with(|c| unsafe { *c.get() })
}

/// `errno` as a pointer, which is what `_errno()` hands out.
pub(super) fn errno_ptr() -> *mut i32 {
    ERRNO.with(UnsafeCell::get)
}

/// `errno = e` and `e` back, for `return fail(EINVAL)`.
pub(super) fn fail(e: i32) -> i32 {
    set_errno(e);
    e
}

/// The bytes of a NUL-terminated string.
pub(super) unsafe fn bytes<'a>(p: *const u8) -> &'a [u8] {
    // SAFETY: the caller passes a NUL-terminated string.
    unsafe { units(p) }
}

/// The units of a NUL-terminated wide string.
pub(super) unsafe fn wide<'a>(p: *const u16) -> &'a [u16] {
    // SAFETY: the caller passes a NUL-terminated wide string.
    unsafe { units(p) }
}

/// A character unit of narrow (`u8`) or wide (`u16`) strings.
pub(super) trait Ch: Copy + Eq + Ord + Default + 'static {
    fn get(self) -> u32;
    fn make(v: u32) -> Self;
}

impl Ch for u8 {
    fn get(self) -> u32 {
        u32::from(self)
    }
    fn make(v: u32) -> Self {
        v as u8
    }
}

impl Ch for u16 {
    fn get(self) -> u32 {
        u32::from(self)
    }
    fn make(v: u32) -> Self {
        v as u16
    }
}

/// The units of a NUL-terminated string of either width.
pub(super) unsafe fn units<'a, C: Ch>(p: *const C) -> &'a [C] {
    // SAFETY: the caller passes a NUL-terminated string.
    unsafe { std::slice::from_raw_parts(p, (0..).take_while(|&i| (*p.add(i)).get() != 0).count()) }
}

/// Leaks `bytes` plus a NUL: a string the title may keep for good.
pub(super) fn leak_cstr(bytes: &[u8]) -> *mut u8 {
    let mut v = bytes.to_vec();
    v.push(0);
    Box::leak(v.into_boxed_slice()).as_mut_ptr()
}

/// Leaks `units` plus a NUL.
pub(super) fn leak_wstr(units: &[u16]) -> *mut u16 {
    let mut v = units.to_vec();
    v.push(0);
    Box::leak(v.into_boxed_slice()).as_mut_ptr()
}

/// A `fn(&str) -> Option<usize>` over functions named as their exports,
/// plus `"export" => function` aliases.
macro_rules! exports {
    ($vis:vis fn $lookup:ident; $($name:ident),* $(,)?; $($alias:literal => $f:ident),* $(,)?) => {
        $vis fn $lookup(name: &str) -> Option<usize> {
            match name {
                $(stringify!($name) => Some($name as *const () as usize),)*
                $($alias => Some($f as *const () as usize),)*
                _ => None,
            }
        }
    };
}

/// As [`exports!`], for `extern "win64"` functions: each is declared to the
/// bridge when looked up.
macro_rules! exports64 {
    ($vis:vis fn $lookup:ident; $($name:ident),* $(,)?; $($alias:literal => $f:ident),* $(,)?) => {
        $vis fn $lookup(name: &str) -> Option<usize> {
            let address = match name {
                $(stringify!($name) => $name as *const () as usize,)*
                $($alias => $f as *const () as usize,)*
                _ => return None,
            };
            victory_loader::native::declare_win64(address);
            Some(address)
        }
    };
}
pub(super) use {exports, exports64};

/// Whether the C runtime serves `dll`: the API sets and `ucrtbase`. Not the
/// Visual C++ runtimes (`vcruntime140`, `msvcp140`): the title ships those,
/// with the C++ runtime and `__CxxFrameHandler`, and they are loaded from its folder.
pub(super) fn serves(dll: &str) -> bool {
    dll == "ucrtbase.dll"
        || dll.starts_with("api-ms-win-crt-")
        // `vcruntime140.dll`'s memory and string functions are the ucrt's
        // (its `memset` and `memcpy` came back as no-op stubs in a failed
        // load, #104); a title's own DLL of that name still wins.
        || (dll.starts_with("vcruntime140")
            && victory_loader::native::dll::find(dll).is_none())
}

pub(super) fn lookup(dll: &str, name: &str) -> Option<usize> {
    if !serves(dll) {
        return None;
    }
    heap::lookup(name)
        .or_else(|| math::lookup(name))
        .or_else(|| string::lookup(name))
        .or_else(|| convert::lookup(name))
        .or_else(|| stdio::lookup(name))
        .or_else(|| runtime::lookup(name))
        .or_else(|| time::lookup(name))
        .or_else(|| fs::lookup(name))
}

/// A mutable static the title gets a pointer to (`__p___argc`, `_fmode`).
pub(super) struct Shared<T>(UnsafeCell<T>);

// SAFETY: written by the host at startup, then by the title the way it
// would write the C runtime's own variables in one process.
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    pub(super) const fn new(v: T) -> Self {
        Shared(UnsafeCell::new(v))
    }
    pub(super) fn ptr(&self) -> *mut T {
        self.0.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msvcp140_ucrt_imports_are_served() {
        let imports: &[(&str, &[&str])] = &[
            ("environment", &["_wgetcwd"]),
            ("locale", &["_lock_locales", "_unlock_locales"]),
            ("runtime", &["_endthreadex"]),
            ("stdio", &["ungetwc"]),
            ("string", &["__strncnt"]),
            (
                "time",
                &[
                    "_Getdays",
                    "_Getmonths",
                    "_Gettnames",
                    "_Strftime",
                    "_W_Getdays",
                    "_W_Getmonths",
                    "_W_Gettnames",
                    "_Wcsftime",
                ],
            ),
        ];
        for (category, names) in imports {
            let dll = format!("api-ms-win-crt-{category}-l1-1-0.dll");
            for name in *names {
                assert!(lookup(&dll, name).is_some(), "{dll}!{name}");
            }
        }
    }

    /// With `VICTORY_UCRT_IMPORTS` naming a `--list-imports` dry-run output,
    /// every UCRT import in it must resolve (the HLE registry's own `exit`
    /// and `abort` are served there first, not here).
    /// `vcruntime140.dll`'s memory functions came back as no-op stubs in a
    /// failed load (#104): no title DLL of that name is loaded in a test,
    /// so the host serves the ucrt's implementations.
    #[test]
    fn vcruntime140s_memory_and_string_imports_are_served() {
        assert!(serves("vcruntime140.dll"));
        assert!(serves("vcruntime140_1.dll"));
        assert!(lookup("vcruntime140.dll", "memset").is_some());
        assert!(lookup("vcruntime140.dll", "memcpy").is_some());
        assert!(lookup("vcruntime140.dll", "memmove").is_some());
        assert!(lookup("vcruntime140.dll", "memcmp").is_some());
        assert!(lookup("vcruntime140.dll", "strlen").is_some());
        assert!(!serves("msvcp140.dll"));
        assert!(!serves("kernel32.dll"));
    }

    #[test]
    fn every_ucrt_import_of_a_dry_run_is_served() {
        let Some(list) = std::env::var_os("VICTORY_UCRT_IMPORTS") else {
            return;
        };
        let text = std::fs::read_to_string(list).unwrap();
        let mut missing = Vec::new();
        let mut checked = 0;
        for line in text.lines().filter(|l| l.contains("api-ms-win-crt-")) {
            let Some(item) = line.split_whitespace().last() else {
                continue;
            };
            let Some((dll, name)) = item.split_once('!') else {
                continue;
            };
            checked += 1;
            if lookup(dll, name).is_none() {
                missing.push(item.to_string());
            }
        }
        assert!(checked > 100, "only {checked} UCRT imports in the list");
        assert!(missing.is_empty(), "not served: {missing:?}");
    }

    #[test]
    fn only_the_ucrt_and_an_unloaded_vcruntime_are_served() {
        assert!(lookup("kernel32.dll", "strlen").is_none());
        assert!(lookup("ucrtbase.dll", "strlen").is_some());
        assert!(lookup("api-ms-win-crt-string-l1-1-0.dll", "strlen").is_some());
        assert!(lookup("ucrtbase.dll", "NoSuchFunction").is_none());
        // `vcruntime140.dll`'s memory and string functions are the ucrt's
        // (its `memset` and `memcpy` came back as no-op stubs in a failed
        // load, #104), so they are served while the title has not mapped
        // its own DLL of that name. A bare test process has an empty
        // loader registry, so `serves` answers yes; a title that did map
        // its own copy would make `find` answer and the ucrt step aside.
        assert!(lookup("vcruntime140.dll", "memcpy").is_some());
        assert!(lookup("vcruntime140_1.dll", "strlen").is_some());
        // `msvcp140.dll` is not the C runtime and is never ours to serve.
        assert!(lookup("msvcp140.dll", "strlen").is_none());
    }
}
