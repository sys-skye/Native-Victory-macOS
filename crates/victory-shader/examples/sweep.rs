//! Translates every shader container of a folder as the native host does, one line each:
//! `cargo run --release -p victory-shader --example sweep -- <folder> [-j N] [--list FILE] [--show] [--timeout SECONDS] > sweep.tsv`.
//! A line is `<file> \t ok|refused|panic|timeout|missing \t <output fingerprint or reason> \t <ms>`;
//! `tools/corpus/compare.sh` tallies one sweep or compares two (docs/native-host.md, "Sweeping a
//! shader corpus"). The folder holds game data: keep it and `--show` output outside the repo.

use std::hash::{Hash, Hasher};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use victory_shader::ShaderError;
use victory_shader::emit::{HostShader, Options, translate_with};

const USAGE: &str = "usage: sweep <folder> [-j N] [--list FILE] [--show] [--timeout SECONDS]";

fn main() {
    let usage = || -> ! {
        eprintln!("{USAGE}");
        std::process::exit(2)
    };
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next().filter(|a| !a.starts_with('-')) else { usage() };
    let (mut jobs, mut list, mut show, mut limit) = (4usize, None, false, 120u64);
    while let Some(a) = args.next() {
        let mut number = || args.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or_else(|| usage());
        match a.as_str() {
            "-j" => jobs = number() as usize,
            // The files to translate, one name a line, instead of every `.bin` of the folder.
            "--list" => list = Some(args.next().unwrap_or_else(|| usage())),
            // The translation's host-side description (root parameters, flags), not its fingerprint.
            "--show" => show = true,
            // Seconds one shader may take: past that its line says "timeout" and the sweep goes on.
            "--timeout" => limit = number(),
            _ => usage(),
        }
    }
    let mut names: Vec<String> = match list {
        Some(f) => std::fs::read_to_string(f).expect("the list file").lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect(),
        None => std::fs::read_dir(&dir)
            .expect("the folder of containers")
            .map(|e| e.expect("a folder entry").file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".bin"))
            .collect(),
    };
    names.sort();
    names.dedup();
    // A translation that panics is a line of the sweep, not a message for each on stderr.
    std::panic::set_hook(Box::new(|_| {}));
    // The native host's options (`victory-gpu`, `native::shaders::translate`).
    let options = Options { wave32_host: true, zero_copy_memory: true, ..Default::default() };
    let lines = sweep(dir, names, jobs, Duration::from_secs(limit), show, move |c| translate_with(c, options));
    let mut text = String::new();
    let mut counts = std::collections::BTreeMap::new();
    for l in &lines {
        *counts.entry(l.split('\t').nth(1).unwrap_or_default()).or_insert(0usize) += 1;
        text.push_str(l);
        text.push('\n');
    }
    use std::io::Write;
    std::io::stdout().write_all(text.as_bytes()).expect("writable output");
    eprintln!("sweep: {} shaders:{}", lines.len(), counts.iter().map(|(k, v)| format!(" {k} {v}")).collect::<String>());
    // Not a return: a translation given up on may still be running.
    std::process::exit(0);
}

struct State<F> {
    dir: String,
    names: Vec<String>,
    show: bool,
    translate: F,
    next: AtomicUsize,
    out: Vec<Mutex<Option<String>>>,
    /// Milliseconds since `began` plus one, when a worker took the shader; 0 before.
    started: Vec<AtomicU64>,
    /// Given up on: the worker that still holds it ends when it returns.
    left: Vec<AtomicBool>,
    began: Instant,
}

/// One line for each of `names` (files of `dir`), in that order, translated on `jobs` threads.
fn sweep<F>(dir: String, names: Vec<String>, jobs: usize, limit: Duration, show: bool, translate: F) -> Vec<String>
where
    F: Fn(&[u8]) -> Result<HostShader, ShaderError> + Send + Sync + 'static,
{
    let n = names.len();
    let st = Arc::new(State {
        dir,
        names,
        show,
        translate,
        next: AtomicUsize::new(0),
        out: (0..n).map(|_| Mutex::new(None)).collect(),
        started: (0..n).map(|_| AtomicU64::new(0)).collect(),
        left: (0..n).map(|_| AtomicBool::new(false)).collect(),
        began: Instant::now(),
    });
    for _ in 0..jobs.max(1) {
        let st = st.clone();
        std::thread::spawn(move || work(&st));
    }
    // A shader that takes longer than the limit is given up on (its thread is left running and a
    // new one takes its place), so one endless translation does not hold the sweep.
    let limit = limit.as_millis() as u64;
    loop {
        let now = st.began.elapsed().as_millis() as u64;
        let mut open = 0;
        for i in 0..n {
            let mut slot = st.out[i].lock().unwrap();
            if slot.is_some() {
                continue;
            }
            open += 1;
            let at = st.started[i].load(Ordering::Relaxed);
            if at != 0 && now.saturating_sub(at - 1) > limit {
                *slot = Some(format!("{}\ttimeout\tno result after {} s\t{limit}", st.names[i], limit / 1000));
                st.left[i].store(true, Ordering::Relaxed);
                open -= 1;
                let st = st.clone();
                std::thread::spawn(move || work(&st));
            }
        }
        if open == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    st.out.iter().map(|slot| slot.lock().unwrap().clone().unwrap_or_default()).collect()
}

/// What two sweeps compare: a hash of the DXIL and of everything else the host is told about the
/// shader (the `Debug` form of the other fields). Unchanged since the first sweeps posted, so
/// fingerprints of one toolchain stay comparable; a new `HostShader` field changes every one.
fn fingerprint(mut h: HostShader) -> (u64, String) {
    let mut k = std::collections::hash_map::DefaultHasher::new();
    std::mem::take(&mut h.dxil).hash(&mut k);
    let rest = format!("{h:?}");
    rest.hash(&mut k);
    (k.finish(), rest)
}

fn work<F: Fn(&[u8]) -> Result<HostShader, ShaderError>>(st: &State<F>) {
    loop {
        let i = st.next.fetch_add(1, Ordering::Relaxed);
        let Some(name) = st.names.get(i) else { break };
        st.started[i].store(st.began.elapsed().as_millis() as u64 + 1, Ordering::Relaxed);
        let clean = |s: String| s.replace(['\t', '\n', '\r'], " ");
        let line = match std::fs::read(std::path::Path::new(&st.dir).join(name)) {
            Err(e) => format!("{name}\tmissing\t{}\t0", clean(e.to_string())),
            Ok(bytes) => {
                let began = Instant::now();
                let r = catch_unwind(AssertUnwindSafe(|| (st.translate)(&bytes)));
                let ms = began.elapsed().as_millis();
                match r {
                    Ok(Ok(h)) => {
                        let (hash, rest) = fingerprint(h);
                        if st.show { format!("{name}\tok\t{}\t{ms}", clean(rest)) } else { format!("{name}\tok\t{hash:016x}\t{ms}") }
                    }
                    Ok(Err(e)) => format!("{name}\trefused\t{}\t{ms}", clean(e.to_string())),
                    Err(p) => {
                        let m = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                        format!("{name}\tpanic\t{}\t{ms}", clean(m))
                    }
                }
            }
        };
        let mut slot = st.out[i].lock().unwrap();
        if slot.is_none() {
            *slot = Some(line);
        }
        drop(slot);
        if st.left[i].load(Ordering::Relaxed) {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::HwStage;

    /// A guessed pixel container around a program made up for these tests.
    fn pixel(colour: u32) -> Vec<u8> {
        let code = [
            0x7E00_0280 | colour, // v_mov_b32_e32 v0, <colour>
            0xF800_180F,          // exp mrt0 v0, v0, v0, v0 done vm
            0x0000_0000,
            0xBF81_0000, // s_endpgm
        ];
        let code: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        victory_shader::emit::graphics::container(HwStage::Ps, &code, &[0u8; 0x10]).unwrap()
    }

    /// A folder of its own under the system's temporary one, holding `files`.
    fn folder(test: &str, files: &[(&str, &[u8])]) -> String {
        let dir = std::env::temp_dir().join(format!("victory-sweep-{}-{test}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, bytes) in files {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
        dir.to_string_lossy().into_owned()
    }

    fn host(c: &[u8]) -> Result<HostShader, ShaderError> {
        translate_with(c, Options { wave32_host: true, zero_copy_memory: true, ..Default::default() })
    }

    fn fields(line: &str) -> Vec<&str> {
        line.split('\t').collect()
    }

    #[test]
    fn each_container_gets_a_line_in_the_order_asked() {
        let (one, two) = (pixel(1), pixel(2));
        let dir = folder("lines", &[("pixel_0000000000000001.bin", &one), ("pixel_0000000000000002.bin", &two), ("vertex_0000000000000003.bin", b"not a container")]);
        let names = ["pixel_0000000000000001.bin", "pixel_0000000000000002.bin", "vertex_0000000000000003.bin", "vertex_0000000000000004.bin"];
        let names: Vec<String> = names.iter().map(|n| n.to_string()).collect();
        let lines = sweep(dir.clone(), names.clone(), 3, Duration::from_secs(120), false, host);
        assert_eq!(lines.len(), 4);
        for (l, n) in lines.iter().zip(&names) {
            assert_eq!(fields(l).len(), 4, "{l}");
            assert_eq!(fields(l)[0], n);
        }
        // Translated: a fingerprint of 16 hex digits, another for another program.
        let (a, b) = (fields(&lines[0]), fields(&lines[1]));
        assert_eq!((a[1], b[1]), ("ok", "ok"));
        assert!(a[2].len() == 16 && a[2].chars().all(|c| c.is_ascii_hexdigit()), "{}", lines[0]);
        assert_ne!(a[2], b[2]);
        // Refused: the translator's own words. A file that is not there says so and costs nothing.
        assert_eq!(fields(&lines[2])[1..3], ["refused", "not a DXBC container"]);
        assert_eq!((fields(&lines[3])[1], fields(&lines[3])[3]), ("missing", "0"));
        // A sweep of the same build gives the same lines again, but for the times.
        let again = sweep(dir.clone(), names, 1, Duration::from_secs(120), false, host);
        let untimed = |ls: &[String]| ls.iter().map(|l| fields(l)[..3].join("\t")).collect::<Vec<_>>();
        assert_eq!(untimed(&lines), untimed(&again));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn show_prints_the_host_side_description_in_one_field() {
        let one = pixel(1);
        let dir = folder("show", &[("pixel_0000000000000001.bin", &one)]);
        let lines = sweep(dir.clone(), vec!["pixel_0000000000000001.bin".into()], 1, Duration::from_secs(120), true, host);
        let f = fields(&lines[0]);
        assert_eq!((f.len(), f[1]), (4, "ok"));
        assert!(f[2].starts_with("HostShader { stage: Ps, dxil: [], "), "{}", f[2]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_fingerprint_covers_the_dxil_and_what_the_host_is_told() {
        let h = host(&pixel(1)).unwrap();
        let print = fingerprint(h.clone()).0;
        assert_eq!(print, fingerprint(h.clone()).0);
        let mut other = h.clone();
        *other.dxil.last_mut().unwrap() ^= 1;
        assert_ne!(print, fingerprint(other).0);
        let mut other = h.clone();
        other.memory = !other.memory;
        assert_ne!(print, fingerprint(other).0);
        // The definition, on its own: the DXIL as a byte vector, then the rest's `Debug` text, in
        // std's default hasher. Sweeps already posted were made with it, so it must not move.
        let mut k = std::collections::hash_map::DefaultHasher::new();
        h.dxil.hash(&mut k);
        format!("{:?}", HostShader { dxil: Vec::new(), ..h.clone() }).hash(&mut k);
        assert_eq!(print, k.finish());
        let mut k = std::collections::hash_map::DefaultHasher::new();
        vec![1u8, 2, 3].hash(&mut k);
        "rest".to_string().hash(&mut k);
        assert_eq!(k.finish(), PINNED);
    }

    /// `DefaultHasher` over the bytes 1, 2, 3 as a vector and then the string "rest" (Rust 1.98.1).
    /// A toolchain that gives another value gives other fingerprints too: sweeps made before it
    /// compare with none made after, so sweep the base again and put the new value here.
    const PINNED: u64 = 0x9ED0_8042_2937_F2C2;

    #[test]
    fn a_panic_is_a_line_and_the_sweep_goes_on() {
        let dir = folder("panic", &[("compute_0000000000000001.bin", b"panics"), ("compute_0000000000000002.bin", b"refused")]);
        let names = vec!["compute_0000000000000001.bin".to_string(), "compute_0000000000000002.bin".to_string()];
        let lines = sweep(dir.clone(), names, 1, Duration::from_secs(120), false, |c| {
            assert!(c != b"panics", "made up\tfor the test");
            Err(ShaderError::Malformed("made up"))
        });
        assert_eq!(fields(&lines[0])[1..3], ["panic", "made up for the test"]);
        assert_eq!(fields(&lines[1])[1..3], ["refused", "malformed shader: made up"]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_translation_past_the_limit_is_given_up_on_and_the_sweep_goes_on() {
        let dir = folder("limit", &[("compute_0000000000000001.bin", b"endless"), ("compute_0000000000000002.bin", b"refused")]);
        let names = vec!["compute_0000000000000001.bin".to_string(), "compute_0000000000000002.bin".to_string()];
        // One worker, and it is stuck in the first shader: the second needs the worker's stand-in.
        let lines = sweep(dir.clone(), names, 1, Duration::from_secs(1), false, |c| {
            if c == b"endless" {
                std::thread::sleep(Duration::from_secs(20));
            }
            Err(ShaderError::Malformed("made up"))
        });
        assert_eq!(lines[0], "compute_0000000000000001.bin\ttimeout\tno result after 1 s\t1000");
        assert_eq!(fields(&lines[1])[1], "refused");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
