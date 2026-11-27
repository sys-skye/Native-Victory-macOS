//! What the shader prewarm pool (`pipeline::prewarm`) works on first: the containers earlier runs
//! bound (the list of first binds, `shader_cache::first_binds_file`), then the rest in the order
//! the title registers them. A cold first world frame binds about 1,300 shaders; registered in
//! order, they wait behind most of the title's ~21,600.
//!
//! And the containers a run translates, kept as the title passed them ([`keep`]), so that
//! `victory translate --cache` can translate them again after the translator changed, with no
//! title running. On the native host the title hands the device pieces (code, metadata) and the
//! container is put together here, so the containers in the game's files are other bytes with
//! other cache keys: the warm-up has to go from what a run saw.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Entries the list keeps: far more than a world frame binds.
const LIMIT: usize = 16_384;
/// New first binds between two writes of the list: a run the watchdog ends loses at most these.
const WRITE_EVERY: usize = 64;

/// Workers in the pool: the performance cores but two (the title's render thread and one more),
/// at least two; `cores` (all of them) where the kinds of core are not known. On a cold cache a
/// pool of all cores but two (16 on an 18-core M5 Max) took the cores the title's frames need: the
/// boot movie and menus dropped about 216 frames in 50 s, against about 54 with 4 workers, and the
/// world still loaded (late shaders are put off, `glue::put_off`).
pub(crate) fn pool_size(cores: usize) -> usize {
    pool_for(cores, performance_cores())
}

/// [`pool_size`] given the core counts.
fn pool_for(cores: usize, performance: Option<usize>) -> usize {
    let fast = performance.unwrap_or(cores).min(cores);
    fast.saturating_sub(2).max(2).min(cores).max(1)
}

/// The performance cores (`hw.perflevel0.physicalcpu`), on macOS.
fn performance_cores() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                old: *mut std::ffi::c_void,
                old_len: *mut usize,
                new: *mut std::ffi::c_void,
                new_len: usize,
            ) -> i32;
        }
        let mut n: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        // SAFETY: a documented sysctl name and an int-sized output.
        let ok = unsafe {
            sysctlbyname(c"hw.perflevel0.physicalcpu".as_ptr(), (&raw mut n).cast(), &mut len, std::ptr::null_mut(), 0)
        } == 0;
        (ok && n > 0).then_some(n as usize)
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// How long a frame may run, from its first new host pipeline, before pipelines whose shaders
/// are not translated yet are put off ([`FrameBudget`]): well under the title's 30 s watchdog.
pub(crate) const FIRST_BIND_BUDGET: Duration = Duration::from_secs(10);

/// When the current frame made its first new host pipeline. Wall time, not time spent making
/// pipelines: the frame's thread also translates inline elsewhere (rectangle lists, extra
/// parameters, thread counts), and the watchdog counts all of it.
pub(crate) struct FrameBudget {
    frame: u64,
    started: Option<Instant>,
    /// [`FrameBudget::spend`] was called for this frame: none of its budget is left, whatever
    /// time has passed, and a compile crashed Metal's compiler to take it.
    crashed: bool,
}

impl FrameBudget {
    pub(crate) const fn new() -> Self {
        FrameBudget { frame: 0, started: None, crashed: false }
    }

    /// What frame `frame`, making a new pipeline at `now`, has left of `budget` since its first
    /// (zero: spent).
    pub(crate) fn left(&mut self, frame: u64, now: Instant, budget: Duration) -> Duration {
        if frame != self.frame || self.started.is_none() {
            *self = FrameBudget { frame, started: Some(now), crashed: false };
        }
        if self.crashed {
            return Duration::ZERO;
        }
        self.started.map_or(budget, |t| budget.saturating_sub(now.saturating_duration_since(t)))
    }

    /// Spends what frame `frame` has left of its budget: [`Self::left`] is zero for it from `now`
    /// on, and the next frame starts with the full budget again, as one that never spends. A
    /// compile that crashed Metal's compiler spends its frame: the frame's other new pipelines
    /// are put off rather than each paying a crash of its own (issue #181).
    pub(crate) fn spend(&mut self, frame: u64, now: Instant) {
        if frame != self.frame || self.started.is_none() {
            *self = FrameBudget { frame, started: Some(now), crashed: false };
        }
        self.crashed = true;
    }

    /// Whether a compile that crashed Metal's compiler spent this frame, rather than the frame
    /// running out of time. The compiler service may be down for the rest of such a frame, so it
    /// gets no new pipeline at all; a frame merely past its time still gets the short wait a
    /// pipeline the archive holds takes. False for the next frame.
    pub(crate) fn spent_by_crash(&self) -> bool {
        self.crashed
    }
}

/// Two queues: the shaders earlier runs bound first, then the others.
pub(crate) struct Queue<T> {
    lists: Mutex<(VecDeque<T>, VecDeque<T>)>,
    ready: Condvar,
}

impl<T> Queue<T> {
    pub(crate) const fn new() -> Self {
        Queue { lists: Mutex::new((VecDeque::new(), VecDeque::new())), ready: Condvar::new() }
    }

    pub(crate) fn push(&self, item: T, bound_before: bool) {
        let mut l = self.lists.lock().unwrap_or_else(|e| e.into_inner());
        if bound_before { l.0.push_back(item) } else { l.1.push_back(item) }
        self.ready.notify_one();
    }

    /// The next shader, waiting for one.
    pub(crate) fn pop(&self) -> T {
        let mut l = self.lists.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(item) = l.0.pop_front().or_else(|| l.1.pop_front()) {
                return item;
            }
            l = self.ready.wait(l).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// The list of first binds: what the last runs bound, then what this one binds that they did not.
#[derive(Default)]
pub(crate) struct FirstBinds {
    known: HashSet<String>,
    order: Vec<String>,
    /// Entries in the file as last written (or read).
    written: usize,
}

impl FirstBinds {
    /// The list in `text` (one short hash per line; anything else is skipped).
    pub(crate) fn parse(text: &str) -> Self {
        let mut list = FirstBinds::default();
        for line in text.lines().map(str::trim) {
            if line.len() == 16 && line.bytes().all(|b| b.is_ascii_hexdigit()) {
                list.add(line);
            }
        }
        list.written = list.order.len();
        list
    }

    pub(crate) fn len(&self) -> usize {
        self.order.len()
    }

    pub(crate) fn contains(&self, hash: &str) -> bool {
        self.known.contains(hash)
    }

    /// Adds `hash`; whether it was new.
    pub(crate) fn add(&mut self, hash: &str) -> bool {
        if self.order.len() >= LIMIT || !self.known.insert(hash.to_owned()) {
            return false;
        }
        self.order.push(hash.to_owned());
        true
    }

    /// The text to write, if enough is new since the last write.
    pub(crate) fn due(&mut self) -> Option<String> {
        if self.order.len() < self.written + WRITE_EVERY {
            return None;
        }
        self.written = self.order.len();
        Some(self.order.iter().map(|h| format!("{h}\n")).collect())
    }
}

/// Writes `text` to `path` through a temporary name. Failures are not fatal: the next run
/// prewarms in the title's order.
pub(crate) fn write(path: &Path, text: &str) {
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    if std::fs::write(&tmp, text).and_then(|()| std::fs::rename(&tmp, path)).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The folder of kept containers: beside the list of first binds, outside the build folders, so
/// it outlives a changed translator. `None` with the shader cache off.
pub(crate) fn kept_dir() -> Option<PathBuf> {
    Some(crate::shader_cache::first_binds_file()?.parent()?.join("containers"))
}

/// Keeps `container` in `dir` under `hash` (its short hash), once, through a temporary name.
/// Failures are not fatal: a warm-up translates what was kept, the run the rest.
pub(crate) fn keep(dir: &Path, hash: &str, container: &[u8]) {
    let path = dir.join(hash);
    if container.is_empty() || path.exists() {
        return;
    }
    let tmp = dir.join(format!("{hash}.{}.tmp", std::process::id()));
    let made = std::fs::create_dir_all(dir)
        .and_then(|()| std::fs::write(&tmp, container))
        .and_then(|()| std::fs::rename(&tmp, &path));
    if made.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The kept containers' files in `dir`: those named by a short hash.
pub(crate) fn kept(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.len() == 16 && n.bytes().all(|b| b.is_ascii_hexdigit())))
        .collect();
    files.sort();
    files
}

/// The container kept in `file`; `None` if its bytes are not the ones its name says (cut short
/// or damaged).
pub(crate) fn read_kept(file: &Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(file).ok()?;
    (file.file_name()?.to_str()? == &blake3::hash(&bytes).to_hex()[..16]).then_some(bytes)
}

/// Runs the calling thread at the lowest priority of the default QoS class: below the title's
/// threads, so its frame work goes first when cores are short. Not the utility class, which
/// keeps a thread on the efficiency cores: there a pool of 16 translated each shader about twice
/// as slowly as six at the default (220 against 100 ms in a run), with the title's thread waiting
/// on the shaders it needed.
#[cfg(target_os = "macos")]
pub(crate) fn lower_priority() {
    // `QOS_CLASS_DEFAULT` in <sys/qos.h>; -15 is the lowest relative priority it allows.
    const QOS_CLASS_DEFAULT: u32 = 0x15;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
    }
    // SAFETY: libSystem's documented call, for the calling thread only.
    unsafe { pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, -15) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_bound_before_come_first() {
        let q = Queue::new();
        q.push("a", false);
        q.push("b", true);
        q.push("c", false);
        q.push("d", true);
        assert_eq!([q.pop(), q.pop(), q.pop(), q.pop()], ["b", "d", "a", "c"]);
    }

    #[test]
    fn each_frame_has_its_own_budget() {
        let mut b = FrameBudget::new();
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let budget = Duration::from_millis(100);
        assert_eq!(b.left(5, at(0), budget), budget);
        assert_eq!(b.left(5, at(99), budget), Duration::from_millis(1));
        assert!(b.left(5, at(100), budget).is_zero());
        // The next frame starts from its own first pipeline.
        assert_eq!(b.left(6, at(150), budget), budget);
        assert!(!b.left(6, at(249), budget).is_zero());
        assert!(b.left(6, at(250), budget).is_zero());
    }

    #[test]
    fn a_spent_frame_starts_the_next_one_whole() {
        // A compile that crashed Metal's compiler spends the frame's budget (issue #181: nine
        // crashes held one frame for 7.3 s): the rest of that frame's new pipelines are put off,
        // and the next frame tries again with the full budget, as one that never spent. A frame
        // that is merely past its time is told apart: it still gets the short wait a pipeline the
        // archive holds takes.
        let mut b = FrameBudget::new();
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let budget = Duration::from_millis(100);
        // A frame that has made no pipeline yet can spend too.
        b.spend(4, at(0));
        assert!(b.left(4, at(0), budget).is_zero());
        assert!(b.spent_by_crash());
        assert_eq!(b.left(5, at(10), budget), budget);
        assert!(!b.spent_by_crash(), "the next frame is not spent by the crash");
        assert_eq!(b.left(5, at(99), budget), Duration::from_millis(11));
        b.spend(5, at(99));
        assert!(b.left(5, at(99), budget).is_zero());
        assert!(b.left(5, at(150), budget).is_zero(), "spent outlasts the frame's remaining time");
        assert!(b.spent_by_crash());
        assert_eq!(b.left(6, at(200), budget), budget);
        assert!(!b.spent_by_crash());
        // A frame that runs out of time is spent too, but not by a crash.
        assert_eq!(b.left(6, at(249), budget), Duration::from_millis(51));
        assert!(b.left(6, at(300), budget).is_zero());
        assert!(!b.spent_by_crash(), "a frame out of time did not crash");
    }

    #[test]
    fn the_pool_is_the_performance_cores_but_two() {
        // The performance cores but two, at least two: 4 of an M5 Max's 6, 2 of an M1's 4.
        assert_eq!(pool_for(18, Some(6)), 4);
        assert_eq!(pool_for(8, Some(4)), 2);
        // Kinds of core unknown: all cores but two.
        assert_eq!(pool_for(8, None), 6);
        assert_eq!(pool_for(2, None), 2);
        assert_eq!(pool_for(1, None), 1);
    }

    #[test]
    fn the_list_keeps_earlier_runs_and_adds_this_ones() {
        let mut list = FirstBinds::parse("00000000000000aa\nnot a hash\n00000000000000bb\n");
        assert!(list.contains("00000000000000aa") && list.contains("00000000000000bb"));
        assert!(!list.add("00000000000000aa"));
        assert_eq!(list.due(), None);
        for n in 0..WRITE_EVERY {
            assert!(list.add(&format!("{n:016x}")));
        }
        let text = list.due().expect("64 new binds");
        assert!(text.starts_with("00000000000000aa\n00000000000000bb\n0000000000000000\n"));
        assert_eq!(list.due(), None);
        // A second run reads what the first wrote.
        let again = FirstBinds::parse(&text);
        assert_eq!(again.order.len(), 2 + WRITE_EVERY);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("first-binds.txt");
        write(&path, &text);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn kept_containers_read_back_and_damaged_ones_do_not() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("containers");
        let short = |b: &[u8]| blake3::hash(b).to_hex()[..16].to_owned();
        let (a, b) = (b"one container".as_slice(), b"another".as_slice());
        keep(&store, &short(a), a);
        keep(&store, &short(b), b);
        // Kept once: a second run leaves the file alone.
        std::fs::write(store.join(short(b)), b"cut").unwrap();
        keep(&store, &short(b), b);
        // Nothing to keep, and a file that is no container's.
        keep(&store, &short(b""), b"");
        std::fs::write(store.join("notes.txt"), b"x").unwrap();
        let files = kept(&store);
        assert_eq!(files.len(), 2);
        let read: Vec<Option<Vec<u8>>> = files.iter().map(|f| read_kept(f)).collect();
        assert!(read.contains(&Some(a.to_vec())) && read.contains(&None), "{read:?}");
        assert!(kept(&dir.path().join("absent")).is_empty());
    }
}
