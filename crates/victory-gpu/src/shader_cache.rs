//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by what the
//! translation depends on ([`build`]), so a changed translator never serves
//! an old translation, while rebuilding the rest of VIctory keeps them.
//! Each translator's entries are in a folder of their own, named by the
//! start of that key: older translators' folders can be deleted whole.
//! Entries go through `victory-pipeline`'s content-addressed cache, which
//! writes them atomically. Each entry starts with the blake3 sum of the rest,
//! as in `victory_metal::disk`: an entry that is damaged, cut short or does
//! not read back is translated again and rewritten. A container that does not translate is kept too, with the reason,
//! so it is not tried again on every run.
//!
//! `victory translate --cache` fills the same cache ahead of a run, through
//! [`host_shader`].
//!
//! The entry is our own encoding of [`HostShader`]; this file's text is part
//! of the key, so changing the encoding invalidates every entry.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in a folder of `dir` from now on: the folder
/// of this translator, `None` if it cannot be opened.
pub fn use_shader_cache(dir: PathBuf) -> Option<PathBuf> {
    let dir = dir.join(&blake3::Hash::from(*build()).to_hex()[..16]);
    match Cache::open(&dir) {
        Ok(cache) => {
            tracing::info!(dir = %dir.display(), "shader cache");
            let _ = CACHE.set(cache);
            Some(dir)
        }
        Err(e) => {
            tracing::warn!(error = %e, "no shader cache");
            None
        }
    }
}

/// The native host's translations: a folder of their own ([`use_native_cache`]).
static NATIVE: OnceLock<Cache> = OnceLock::new();

/// Keeps the native host's translations in a folder of `dir` from now on (one per translator,
/// like [`use_shader_cache`]), and the Metal libraries made from them beside it. Does nothing
/// with `VICTORY_NO_SHADER_CACHE` set. `None` if it is off or the folder cannot be opened.
pub fn use_native_cache(dir: PathBuf) -> Option<PathBuf> {
    if std::env::var_os("VICTORY_NO_SHADER_CACHE").is_some_and(|v| !v.is_empty() && v != "0") {
        tracing::info!("native shader cache off (VICTORY_NO_SHADER_CACHE)");
        return None;
    }
    let parent = dir;
    let dir = parent.join(&blake3::Hash::from(*build()).to_hex()[..16]);
    match Cache::open(dir.join("translated")) {
        Ok(cache) => {
            tracing::info!(dir = %dir.display(), "native shader cache");
            let _ = NATIVE.set(cache);
            let _ = FIRST_BINDS.set(parent.join(FIRST_BINDS_FILE));
            let _ = std::fs::write(dir.join(LAST_USED), b"");
            // Deleting a build's ~20,000 entries takes a while: not on the start-up path.
            let (keep, current) = (parent.clone(), dir.clone());
            let _ = std::thread::Builder::new()
                .name("shader-cache-prune".into())
                .spawn(move || prune(&keep, &current, KEEP_BUILDS));
            #[cfg(target_os = "macos")]
            victory_metal::disk::use_dir(dir.join("metal"));
            Some(dir)
        }
        Err(e) => {
            tracing::warn!(error = %e, "no native shader cache");
            None
        }
    }
}

/// The containers the native host bound, by their short hash, one per line, in the order of
/// their first bind (`pipeline::prewarm` reads it to translate those first). Beside the build
/// folders: it names containers, not translations, so it outlives a changed translator.
static FIRST_BINDS: OnceLock<PathBuf> = OnceLock::new();
const FIRST_BINDS_FILE: &str = "first-binds.txt";

/// Where the native host keeps its list of first binds, if the cache is on.
pub fn first_binds_file() -> Option<&'static std::path::Path> {
    FIRST_BINDS.get().map(PathBuf::as_path)
}

/// Written into a build folder each time a run opens it: its time orders the folders for [`prune`].
const LAST_USED: &str = "last-used";
/// Build folders kept: this build's and the newest others (about 400 MB each).
const KEEP_BUILDS: usize = 3;

/// A folder stamped this recently is never deleted: another run (the other `game-run.sh` slot
/// shares the cache folder) may still be using it.
const IN_USE: std::time::Duration = std::time::Duration::from_secs(60 * 60);

/// Deletes the build folders of `parent` (named by 16 hex digits) but the `keep` used last,
/// `current` always among them, and any stamped within [`IN_USE`]. A folder never stamped counts
/// by its own time.
fn prune(parent: &std::path::Path, current: &std::path::Path, keep: usize) {
    prune_at(parent, current, keep, std::time::SystemTime::now());
}

fn prune_at(parent: &std::path::Path, current: &std::path::Path, keep: usize, now: std::time::SystemTime) {
    let Ok(entries) = std::fs::read_dir(parent) else { return };
    let mut builds: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.len() == 16 && n.bytes().all(|b| b.is_ascii_hexdigit()))
        })
        .map(|p| {
            let used = std::fs::metadata(p.join(LAST_USED)).or_else(|_| std::fs::metadata(&p)).and_then(|m| m.modified());
            (used.unwrap_or(std::time::UNIX_EPOCH), p)
        })
        .collect();
    builds.sort_by_key(|b| std::cmp::Reverse(b.0));
    let mut kept = usize::from(builds.iter().any(|(_, p)| p == current));
    for (used, p) in builds {
        if p == current || now.duration_since(used).is_ok_and(|age| age < IN_USE) {
            continue;
        }
        if kept < keep {
            kept += 1;
            continue;
        }
        tracing::info!(dir = %p.display(), "deleting an old build's shader cache");
        let _ = std::fs::remove_dir_all(&p);
    }
}

/// A translation for the native host, from the cache when there is one. `settings` is whatever
/// else the translation depends on besides the container (its options); the closure translates.
pub fn native_shader(
    container: &[u8],
    settings: &str,
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let Some(cache) = NATIVE.get() else {
        return translate(container);
    };
    let started = std::time::Instant::now();
    let (made, hit) = cached(cache, &native_key(settings), container, translate);
    let ms = started.elapsed().as_millis();
    if ms >= 20 {
        tracing::info!(ms, hit, size = container.len(), "native shader translation");
    }
    made
}

/// The native host's build key for translations made with `settings`: [`build`] and the settings.
fn native_key(settings: &str) -> [u8; 32] {
    *blake3::Hasher::new().update(build()).update(settings.as_bytes()).finalize().as_bytes()
}

/// Whether the native host's cache holds the translation of `container` with `settings` (as
/// [`native_shader`] would key it), without reading or translating it.
pub fn native_present(container: &[u8], settings: &str) -> bool {
    let Some(cache) = NATIVE.get() else { return false };
    let converter = Translate { translate: |_: &[u8]| Err::<HostShader, String>(String::new()), ran: AtomicBool::new(false) };
    let input = [&native_key(settings)[..], container].concat();
    cache.path_for(&victory_pipeline::CacheKey::new(&converter, &input)).is_file()
}

/// Shaders found in the cache and translated since the start (both hosts). The slow-translation
/// log line shows only translations of 20 ms or more, which a hit never is.
pub fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What a translation depends on besides its container: the translator
/// (`victory_shader::SOURCE_HASH`: its sources, manifest, the lock file and
/// the compiler), the entry's encoding (this file's text) and the
/// translation settings: the loop limit, which a diagnostics build can
/// change at run time, and the translator's environment switches
/// (`victory_shader::emit::TRANSLATION_SWITCHES`: `VICTORY_DEBUG_*`,
/// `VICTORY_NO_LOCKSTEP`).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        let switches = victory_shader::emit::translation_switches();
        if !switches.is_empty() {
            tracing::warn!(
                switches = switches.trim_end().replace('\n', " "),
                "translation switches set: translations are cached apart from those made without them"
            );
        }
        key(victory_shader::SOURCE_HASH, ENCODING, limit, switches)
    })
}

/// This file, whose `encode` and `decode` define the entries.
const ENCODING: &[u8] = include_bytes!("shader_cache.rs");

fn key(translator: &str, encoding: &[u8], limit: u32, switches: &str) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(translator.as_bytes());
    h.update(blake3::hash(encoding).as_bytes());
    h.update(&limit.to_le_bytes());
    h.update(switches.as_bytes());
    *h.finalize().as_bytes()
}

/// Whether translations put 64-bit image atomics under locks: under Wine,
/// whose D3DMetal claims the atomics and leaves the texels untouched.
/// `VICTORY_ATOMIC_LOCKS=0` or `1` overrides.
fn atomic_locks() -> bool {
    static LOCKS: OnceLock<bool> = OnceLock::new();
    *LOCKS.get_or_init(|| match std::env::var("VICTORY_ATOMIC_LOCKS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => under_wine(),
    })
}

/// Whether compute shaders' biased samples run as scaled gradient samples:
/// under Wine, whose D3DMetal ignores the bias there (#30).
/// `VICTORY_BIAS_AS_GRADIENTS=0` or `1` overrides.
fn bias_as_gradients() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_BIAS_AS_GRADIENTS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => under_wine(),
    })
}

/// Whether translations take wave64 programs to run in 32-lane host waves:
/// under Wine, whose D3DMetal reports 32 lanes and ignores `WaveSize(64)`.
/// `VICTORY_NARROW_WAVES=0` or `1` overrides.
fn narrow_waves() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_NARROW_WAVES").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => under_wine(),
    })
}

/// Whether this runs under Wine (on a Mac: D3DMetal's D3D12).
#[cfg(windows)]
pub(crate) fn under_wine() -> bool {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    // SAFETY: ntdll is always loaded; both names end with a NUL.
    unsafe {
        GetProcAddress(
            GetModuleHandleA(c"ntdll.dll".as_ptr().cast()),
            c"wine_get_version".as_ptr().cast(),
        )
    }
    .is_some()
}

#[cfg(not(windows))]
pub(crate) fn under_wine() -> bool {
    false
}

/// [`build`] for translations with host workarounds (atomic locks, bias
/// as gradients, narrow waves): their own cache entries, one per
/// combination.
fn options_build(locks: bool, bias: bool, narrow: bool) -> &'static [u8; 32] {
    static BUILDS: [OnceLock<[u8; 32]>; 8] = [const { OnceLock::new() }; 8];
    let n = usize::from(locks) | usize::from(bias) << 1 | usize::from(narrow) << 2;
    BUILDS[n].get_or_init(|| {
        let mut h = blake3::Hasher::new();
        h.update(build());
        if locks {
            h.update(b"atomic locks");
        }
        if bias {
            h.update(b"bias as gradients");
        }
        if narrow {
            h.update(b"narrow waves");
        }
        *h.finalize().as_bytes()
    })
}

/// The translation of a console shader container, from the cache when
/// there is one, and whether it came from there.
pub fn host_shader(container: &[u8]) -> (Result<HostShader, String>, bool) {
    // Writes through views mark their pages: copy-backs compare only those
    // (`crate::shadow::copy_back_marks`).
    // Under Wine (D3DMetal), 64-bit image atomics run under locks: its own
    // leave the texels untouched (`emit`, "Atomic locks").
    let options = victory_shader::emit::Options {
        view_marks: true,
        atomic_locks: atomic_locks(),
        bias_as_gradients: bias_as_gradients(),
        narrow_waves: narrow_waves(),
        // The host's waves are the console's size.
        wave32_host: false,
        // The D3D12 host copies title memory: it needs the pages marked.
        zero_copy_memory: false,
        rect_list: false,
        copy_params: [0; 32],
        pixel_reads: None,
        extra_params: 0,
        uint_targets: 0,
        // `DispatchX` runs thread counts as group counts here (`list::dispatch_x`).
        thread_counts: false,
        // shortcut: this key has no options; integer targets are made on the native host only.
        integer_targets: 0,
    };
    let translate =
        |c: &[u8]| victory_shader::emit::translate_with(c, options).map_err(|e| e.to_string());
    let build = if options.atomic_locks || options.bias_as_gradients || options.narrow_waves {
        options_build(
            options.atomic_locks,
            options.bias_as_gradients,
            options.narrow_waves,
        )
    } else {
        build()
    };
    match CACHE.get() {
        Some(cache) => cached(cache, build, container, translate),
        None => (translate(container), false),
    }
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation (or refusal) out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        Ok(seal(encode(&(self.translate)(&input[32..]))))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> (Result<HostShader, String>, bool) {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => {
                tracing::warn!(error = format!("{e:#}"), "shader cache entry not written");
                break;
            }
        };
        if let Some(shader) = crate::read_file(&path).ok().and_then(|b| decode(unseal(&b)?)) {
            let hit = !converter.ran.load(Ordering::Relaxed);
            let (counter, what) = if hit {
                (&HITS, "from the cache")
            } else {
                (&MISSES, "translated and cached")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return (shader, hit);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    ((converter.translate)(container), false)
}

const MAGIC: &[u8; 4] = b"VSH2";
/// A container that does not translate: the reason follows.
const REFUSED: &[u8; 4] = b"VSHE";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists; a refusal is its
/// reason's text.
fn encode(translation: &Result<HostShader, String>) -> Vec<u8> {
    let s = match translation {
        Ok(s) => s,
        Err(reason) => return [REFUSED.as_slice(), reason.as_bytes()].concat(),
    };
    // Every field by name: a field added to `HostShader` must be added here
    // (and to `decode`).
    let HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        memory,
        memory_writes,
        view_addresses,
        heap,
        heap_writes,
        view_marks,
        splits,
        table_entries,
    } = s;
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|x| x == stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, dxil.len() as u32);
    out.extend(dxil);
    word(&mut out, user_data.len() as u32);
    for u in user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, samplers.len() as u32);
    for sampler in samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, static_samplers.len() as u32);
    for t in static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [root_cbvs, root_srvs, root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(
        &mut out,
        u32::from(*memory)
            | u32::from(*memory_writes) << 1
            | u32::from(*view_addresses) << 2
            | u32::from(*heap) << 3
            | u32::from(*heap_writes) << 4
            | u32::from(*view_marks) << 5,
    );
    word(&mut out, splits.len() as u32);
    for &(a, b) in splits {
        word(&mut out, a);
        word(&mut out, b);
    }
    word(&mut out, table_entries.len() as u32);
    for &(slot, n) in table_entries {
        word(&mut out, u32::from(slot));
        word(&mut out, n);
    }
    out
}

/// An entry: the blake3 sum of the encoding, then the encoding.
fn seal(encoding: Vec<u8>) -> Vec<u8> {
    [blake3::hash(&encoding).as_bytes().as_slice(), &encoding].concat()
}

/// The encoding in an entry; `None` if the sum does not match (a damaged or cut-short file).
fn unseal(entry: &[u8]) -> Option<&[u8]> {
    let (sum, encoding) = entry.split_at_checked(32)?;
    (blake3::hash(encoding).as_bytes() == sum).then_some(encoding)
}

fn decode(bytes: &[u8]) -> Option<Result<HostShader, String>> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    if let Some(reason) = bytes.strip_prefix(REFUSED) {
        return Some(Err(String::from_utf8(reason.to_vec()).ok()?));
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let memory = r.word()?;
    let n = r.word()? as usize;
    let mut splits = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        splits.push((r.word()?, r.word()?));
    }
    let n = r.word()? as usize;
    let mut table_entries = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        table_entries.push((u8::try_from(r.word()?).ok()?, r.word()?));
    }
    r.0.is_empty().then_some(Ok(HostShader {
        splits,
        table_entries,
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        memory: memory & 1 != 0,
        memory_writes: memory & 2 != 0,
        view_addresses: memory & 4 != 0,
        heap: memory & 8 != 0,
        heap_writes: memory & 16 != 0,
        view_marks: memory & 32 != 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            splits: vec![(3, 9), (4, 10)],
            table_entries: vec![(1, 7), (5, 130)],
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![4],
            memory: true,
            memory_writes: true,
            view_addresses: true,
            heap: true,
            heap_writes: true,
            view_marks: true,
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(
            cached(&cache, &[8; 32], container, translate).0,
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH2 broken").unwrap();
        }
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Refusals are stored with their reason, and not tried again.
        let fails = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            fails.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported".to_owned())
        };
        let refused = (Err("unsupported".to_owned()), false);
        assert_eq!(cached(&cache, &build, b"other", fail), refused);
        assert_eq!(cached(&cache, &build, b"other", fail), (refused.0.clone(), true));
        assert_eq!(fails.load(Ordering::Relaxed), 1);

        // A cut-short refusal still decodes (to a shorter reason); the sum
        // makes it a miss, translated and rewritten.
        for e in walk(dir.path()) {
            let bytes = std::fs::read(&e).unwrap();
            std::fs::write(&e, &bytes[..bytes.len() - 1]).unwrap();
        }
        assert_eq!(cached(&cache, &build, b"other", fail), refused);
        assert_eq!(fails.load(Ordering::Relaxed), 2);
        assert_eq!(cached(&cache, &build, b"other", fail), (refused.0, true));
        assert_eq!(fails.load(Ordering::Relaxed), 2);
    }

    /// The key's parts: a change to any of them (the translator's sources,
    /// this file's encoding, the loop limit, the translation switches) makes
    /// another key, so no entry written before it is read.
    #[test]
    fn the_key_covers_translator_encoding_and_settings() {
        let k = key("0123456789abcdef", b"encoding", 1 << 20, "");
        assert_eq!(k, key("0123456789abcdef", b"encoding", 1 << 20, ""));
        assert_ne!(k, key("0123456789abcdee", b"encoding", 1 << 20, ""));
        assert_ne!(k, key("0123456789abcdef", b"encoding.", 1 << 20, ""));
        assert_ne!(k, key("0123456789abcdef", b"encoding", 1 << 19, ""));
        // A debug switch's translations are not those of a run without it.
        let with = |switches| key("0123456789abcdef", b"encoding", 1 << 20, switches);
        assert_eq!(k, with(""));
        assert_ne!(k, with("VICTORY_DEBUG_PS_RED=1\n"));
        assert_ne!(
            with("VICTORY_DEBUG_PS_RED=1\n"),
            with("VICTORY_DEBUG_VS_TRI=1\n")
        );
        assert_eq!(
            *build(),
            key(
                victory_shader::SOURCE_HASH,
                ENCODING,
                victory_shader::emit::loop_limit(),
                victory_shader::emit::translation_switches()
            )
        );
        // The encoding part is this file as built.
        assert!(
            std::str::from_utf8(ENCODING)
                .unwrap()
                .contains("fn encode(translation: &Result<HostShader, String>)")
        );
    }

    /// Two runs of one build: each makes the key afresh and opens the folder anew, and the second
    /// finds what the first translated. Settings that change the output change the key.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_second_run_reads_the_first_runs_translations() {
        let dir = tempfile::tempdir().unwrap();
        let settings = || format!("{:?}", crate::native::shaders::options());
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let first = Cache::open(dir.path()).unwrap();
        let input = [&native_key(&settings())[..], b"container"].concat();
        let probe = Translate { translate: |_: &[u8]| Err::<HostShader, String>(String::new()), ran: AtomicBool::new(false) };
        let path = first.path_for(&victory_pipeline::CacheKey::new(&probe, &input));
        assert!(!path.is_file());
        assert_eq!(cached(&first, &native_key(&settings()), b"container", translate).1, false);
        // Where `native_present` looks: the entry `cached` wrote.
        assert!(path.is_file());
        drop(first);
        let second = Cache::open(dir.path()).unwrap();
        assert_eq!(cached(&second, &native_key(&settings()), b"container", translate).1, true);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        let rect = format!("{:?}", victory_shader::emit::Options { rect_list: true, ..crate::native::shaders::options() });
        assert_ne!(native_key(&settings()), native_key(&rect));
        // A pixel shader's integer-target variants are entries of their own.
        let int = |n| format!("{:?}", victory_shader::emit::Options { integer_targets: n, ..crate::native::shaders::options() });
        assert_eq!(int(0), format!("{:?}", crate::native::shaders::options()));
        assert_ne!(native_key(&int(0b10)), native_key(&int(0)));
        assert_ne!(native_key(&int(0b10)), native_key(&int(0b11)));
    }

    /// The current build and the two used last stay; older ones, by their stamp, go. Other
    /// folders and files are left alone.
    #[test]
    fn old_build_folders_are_pruned() {
        let dir = tempfile::tempdir().unwrap();
        let t = |s: u64| std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000 + s);
        let builds: Vec<PathBuf> = (0..5).map(|n| dir.path().join(format!("{n:016x}"))).collect();
        for (n, b) in builds.iter().enumerate() {
            std::fs::create_dir_all(b.join("translated")).unwrap();
            let f = std::fs::File::create(b.join(LAST_USED)).unwrap();
            f.set_modified(t(n as u64)).unwrap();
        }
        std::fs::create_dir(dir.path().join("not-a-build")).unwrap();
        std::fs::write(dir.path().join(FIRST_BINDS_FILE), b"x").unwrap();
        // The oldest is the one in use. Stamps are from 1970; a run within the hour keeps all.
        prune_at(dir.path(), &builds[0], 3, t(10));
        assert!(builds.iter().all(|b| b.exists()), "used within the hour");
        prune_at(dir.path(), &builds[0], 3, t(4 + 2 * 3600));
        let left: Vec<bool> = builds.iter().map(|b| b.exists()).collect();
        assert_eq!(left, [true, false, false, true, true]);
        assert!(dir.path().join("not-a-build").exists() && dir.path().join(FIRST_BINDS_FILE).exists());
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&Ok(shader()));
        assert_eq!(decode(&bytes), Some(Ok(shader())));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH1"), None);
        let refusal = Err("GFX10 code at 0x10: x is not supported".to_owned());
        assert_eq!(decode(&encode(&refusal)), Some(refusal));
        assert_eq!(decode(&[b"VSHE".as_slice(), &[0xff]].concat()), None);
    }
}
