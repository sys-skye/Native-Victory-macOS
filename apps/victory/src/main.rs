//! `victory`: the command-line front end.
//!
//! Every command reads the user's own files and writes only to stdout or to
//! the user's cache directory (outside the repository).

mod census;
mod inspect;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
mod native_run;
#[cfg(windows)]
mod prebuild;
mod run;
mod shader;
#[cfg(windows)]
mod translate;
#[cfg(target_os = "macos")]
mod warm;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
mod window;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "victory",
    version,
    about = "Run an Xbox Series title natively on Windows"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarise one PE image: headers, sections, imports, TLS, build id.
    Inspect {
        file: PathBuf,
        /// List every imported function.
        #[arg(long)]
        imports: bool,
        /// List every exported function.
        #[arg(long)]
        exports: bool,
    },
    /// Aggregate the imports of every PE image under the given paths and
    /// show which ones our runtime covers. Defaults to the configured
    /// game and test-title directories.
    Census {
        paths: Vec<PathBuf>,
        /// Do not count host pass-through as covered.
        #[arg(long)]
        strict: bool,
        /// List every missing `dll!function`, most used first.
        #[arg(long)]
        missing: bool,
    },
    /// List the `dll!export` pairs an import library describes.
    LibExports {
        lib: PathBuf,
        /// List every export, not just per-DLL counts.
        #[arg(long)]
        list: bool,
    },
    /// Find shader containers in files or directories (also inside engine
    /// archives, if uncompressed) and report their layout. Shader packs
    /// among the paths are listed last, their stages named from the
    /// containers found.
    ShaderInfo {
        paths: Vec<PathBuf>,
        /// Print one line per container found (and per pack pipeline).
        #[arg(long)]
        verbose: bool,
        /// Also list the decoded native instructions (implies --verbose).
        #[arg(long)]
        disasm: bool,
        /// Instead list the effects of shader databases: technique, pass,
        /// shaders (name, container hash, engine hash) and pass state.
        #[arg(long)]
        effects: bool,
    },
    /// Run every shader found (containers, SGD5 files) through the
    /// translator and report what it cannot handle yet.
    Translate {
        paths: Vec<PathBuf>,
        /// Check emitted DXIL with dxil.dll's validator (`VICTORY_DXIL`,
        /// else the newest Windows SDK's).
        #[arg(long)]
        validate: bool,
        /// Also create a host D3D12 pipeline for each emitted shader. With
        /// `--cache`: then make every pipeline of the title's shader pack
        /// into the host pipeline library `run` reads (on this GPU and
        /// driver; a second time only what is not there yet).
        #[arg(long)]
        pipelines: bool,
        /// Print one line per program and outcome.
        #[arg(long)]
        list: bool,
        /// Worker threads (default: every logical core).
        #[arg(long)]
        jobs: Option<usize>,
        /// Only translate into the shader cache `run` reads (the warm-up):
        /// every container found, once, nothing else checked or listed
        /// (with `--pipelines`, then the shader pack's pipelines).
        #[arg(long, conflicts_with_all = ["validate", "list"])]
        cache: bool,
    },
    /// Prepare, map and bind a title executable; with --execute, run it.
    Run {
        exe: PathBuf,
        /// What unimplemented imports do when called.
        #[arg(long, value_enum, default_value_t = Unimplemented::Trap)]
        unimplemented: Unimplemented,
        /// Disable host pass-through (everything must be implemented by us).
        #[arg(long)]
        strict: bool,
        /// Actually start the title. Without it, stop after binding.
        #[arg(long)]
        execute: bool,
        /// macOS native host: load and bind, print every import as served
        /// or stubbed, and exit without running the entry point.
        #[arg(long, conflicts_with = "execute")]
        list_imports: bool,
        /// The console model the title sees (device type, memory).
        #[arg(long, value_enum, default_value_t = Console::SeriesX)]
        console: Console,
        /// macOS native host: no window; frames are presented to an offscreen layer.
        #[arg(long)]
        headless: bool,
        /// Arguments for the title, after `--` (it reads the whole command
        /// line; a title that relaunches itself passes some).
        #[arg(last = true)]
        title_args: Vec<String>,
    },
    /// Show the resolved configuration and check it.
    Config,
}

#[derive(Clone, Copy, ValueEnum)]
enum Unimplemented {
    Trap,
    Zero,
}

#[derive(Clone, Copy, ValueEnum)]
enum Console {
    SeriesX,
    SeriesS,
}

/// SIGTERM, SIGINT and SIGHUP stop the process after its GPU work is woken and drained
/// (`victory_gpu::stop_gpu`), not in the middle of it: a process killed while its command buffers
/// sleep on events can leave the GPU memory they hold allocated until a reboot. The signals are
/// blocked here, before any other thread starts (threads inherit the mask), and taken by a thread
/// of their own with `sigwait`, so the stop runs as ordinary code, not in a signal handler.
#[cfg(target_os = "macos")]
fn stop_gpu_on_signals() {
    unsafe extern "C" {
        fn pthread_sigmask(how: i32, set: *const u32, old: *mut u32) -> i32;
        fn sigwait(set: *const u32, sig: *mut i32) -> i32;
    }
    const SIG_BLOCK: i32 = 1;
    const SIG_UNBLOCK: i32 = 2;
    let set: u32 = [1u32 /* SIGHUP */, 2 /* SIGINT */, 15 /* SIGTERM */]
        .iter()
        .fold(0, |set, sig| set | 1 << (sig - 1));
    // SAFETY: libSystem's documented calls with a valid set (`sigset_t` is 32 bits on macOS).
    if unsafe { pthread_sigmask(SIG_BLOCK, &set, std::ptr::null_mut()) } != 0 {
        return;
    }
    let spawned = std::thread::Builder::new().name("stop-on-signal".into()).spawn(move || {
        let mut sig = 0;
        // SAFETY: as above; `sig` receives the signal taken.
        if unsafe { sigwait(&set, &mut sig) } != 0 {
            return;
        }
        // A second signal stops the process at once, should the drain or the exit hang.
        // SAFETY: as above.
        unsafe { pthread_sigmask(SIG_UNBLOCK, &set, std::ptr::null_mut()) };
        tracing::warn!(signal = sig, "stopping: waking and draining the GPU's queued work first");
        let (in_flight, left) = victory_gpu::stop_gpu(std::time::Duration::from_secs(3));
        tracing::warn!(in_flight, left, "stopped the GPU's work; exiting");
        std::process::exit(128 + sig);
    });
    if spawned.is_err() {
        // No thread to take them: leave the signals their default action.
        // SAFETY: as above; no other thread has started yet.
        unsafe { pthread_sigmask(SIG_UNBLOCK, &set, std::ptr::null_mut()) };
    }
}

/// Diagnostics: the heap bytes Rust code holds, for the statistics.
#[cfg(all(windows, feature = "diagnostics"))]
#[global_allocator]
static ALLOC: victory_gpu::CountingAlloc = victory_gpu::CountingAlloc;

fn main() -> Result<()> {
    victory_core::logging::init("info");
    #[cfg(target_os = "macos")]
    stop_gpu_on_signals();
    // The audio recorder `run` starts for `VICTORY_AUDIO_CAPTURE`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if victory_hle::modules::audio::capture::child() {
        return Ok(());
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect {
            file,
            imports,
            exports,
        } => inspect::run(&file, imports, exports),
        Command::Census {
            paths,
            strict,
            missing,
        } => census::run(paths, strict, missing),
        Command::LibExports { lib, list } => inspect::lib_exports(&lib, list),
        Command::ShaderInfo {
            paths,
            verbose,
            disasm,
            effects,
        } => shader::run(paths, verbose || disasm, disasm, effects),
        Command::Translate {
            paths,
            jobs,
            pipelines,
            cache: true,
            ..
        } => {
            #[cfg(windows)]
            {
                translate::warm(paths, jobs, pipelines)
            }
            // The native host's warm-up goes from the containers earlier runs
            // kept, and its Metal libraries and pipelines are made in the run.
            #[cfg(target_os = "macos")]
            {
                anyhow::ensure!(!pipelines, "--pipelines needs a Windows host for now");
                anyhow::ensure!(
                    paths.is_empty(),
                    "on a Mac the warm-up translates the containers earlier runs kept, not files: give no paths"
                );
                warm::native(jobs)
            }
            #[cfg(not(any(windows, target_os = "macos")))]
            {
                let _ = (paths, jobs, pipelines);
                anyhow::bail!("translate needs a Windows host for now")
            }
        }
        Command::Translate {
            paths,
            validate,
            pipelines,
            list,
            jobs,
            cache: false,
        } => {
            #[cfg(windows)]
            {
                translate::run(paths, validate, pipelines, list, jobs)
            }
            // Translation checks shaders with the Windows DXC host, which
            // other hosts do not have yet.
            #[cfg(not(windows))]
            {
                let _ = (paths, validate, pipelines, list, jobs);
                anyhow::bail!("translate needs a Windows host for now")
            }
        }
        Command::Run {
            exe,
            unimplemented,
            strict,
            execute,
            list_imports,
            console,
            headless,
            title_args,
        } => {
            let code = run::run(
                &exe,
                unimplemented,
                strict,
                execute,
                list_imports,
                console,
                &title_args,
                headless,
            )?;
            if code != 0 {
                // Preserve all 32 bits of the Windows process status.
                std::process::exit(code as i32);
            }
            Ok(())
        }
        Command::Config => show_config(),
    }
}

fn show_config() -> Result<()> {
    let cfg = victory_core::Config::load()?;
    let status = |p: &Option<PathBuf>| match p {
        Some(p) if p.exists() => format!("{}", p.display()),
        Some(p) => format!("{}  (does not exist)", p.display()),
        None => "(not set)".to_owned(),
    };
    println!(
        "config file : {}",
        cfg.source
            .as_ref()
            .map_or("(none)".into(), |p| p.display().to_string())
    );
    println!("game_dir    : {}", status(&cfg.game_dir));
    println!("cache_dir   : {}", cfg.cache_dir.display());
    println!("data_dir    : {}", cfg.data_dir.display());
    println!("fidelityfx  : {}", status(&cfg.fidelityfx_dir));
    for t in &cfg.test_titles {
        println!("test_title  : {}", status(&Some(t.clone())));
    }
    for p in cfg
        .game_dir
        .iter()
        .chain(&cfg.test_titles)
        .chain([&cfg.cache_dir, &cfg.data_dir])
    {
        if let Err(e) = victory_core::ensure_outside_repo(p) {
            println!("POLICY VIOLATION: {e}");
        }
    }
    Ok(())
}

/// Paths given on the command line, or the configured game + test titles.
pub(crate) fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !paths.is_empty() {
        return Ok(paths);
    }
    let cfg = victory_core::Config::load()?;
    let all: Vec<PathBuf> = cfg.game_dir.into_iter().chain(cfg.test_titles).collect();
    anyhow::ensure!(
        !all.is_empty(),
        "no paths given and none configured (see victory.example.toml)"
    );
    Ok(all)
}
