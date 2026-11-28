//! Compiled pipeline states kept across runs in an `MTLBinaryArchive` (one file in the library
//! cache's folder, `disk::use_dir`). A pipeline is looked up in the archive first (failing on a
//! miss); a miss is compiled as before and added. The archive is written back every few seconds
//! while pipelines are being added (on a thread of its own: a write takes about half a second
//! and the frame's thread used to wait for it), and when the backend is dropped, so a run that
//! is killed loses at most the last few seconds' pipelines. A file Metal refuses (another GPU or
//! OS) is replaced by an empty archive.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString, NSURL};
use objc2_metal::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

const SAVE_EVERY: Duration = Duration::from_secs(5);

pub(crate) struct Archive {
    archive: Retained<ProtocolObject<dyn MTLBinaryArchive>>,
    list: Retained<NSArray<ProtocolObject<dyn MTLBinaryArchive>>>,
    path: std::path::PathBuf,
    added: AtomicU32,
    /// Held for writing to add a pipeline, for reading to look one up or to write the file.
    /// Guess: Apple does not say an archive may be added to while another thread uses it.
    changing: RwLock<()>,
}

// SAFETY: Metal's devices and archives may be used from any thread; `changing` orders the adds.
unsafe impl Send for Archive {}
// SAFETY: as above.
unsafe impl Sync for Archive {}

type Dev = Retained<ProtocolObject<dyn MTLDevice>>;

impl Archive {
    /// The archive for this GPU, if libraries are kept on disk.
    pub(crate) fn open(dev: &Dev) -> Option<Arc<Archive>> {
        Self::open_in(dev, crate::disk::dir()?)
    }

    fn open_in(dev: &Dev, dir: &std::path::Path) -> Option<Arc<Archive>> {
        let gpu = blake3::hash(dev.name().to_string().as_bytes()).to_hex();
        let path = dir.join(format!("pipelines-{}.metalarchive", &gpu[..8]));
        let make = |load: bool| {
            let d = MTLBinaryArchiveDescriptor::new();
            if load {
                d.setUrl(Some(&url(&path)));
            }
            dev.newBinaryArchiveWithDescriptor_error(&d)
        };
        let archive = match path.exists().then(|| make(true)) {
            Some(Ok(a)) => a,
            _ => {
                let _ = std::fs::remove_file(&path);
                make(false).ok()?
            }
        };
        let list = NSArray::from_retained_slice(&[archive.clone()]);
        let archive = Arc::new(Archive { archive, list, path, added: AtomicU32::new(0), changing: RwLock::new(()) });
        let weak = Arc::downgrade(&archive);
        let _ = std::thread::Builder::new().name("pipeline-archive".into()).spawn(move || {
            loop {
                std::thread::sleep(SAVE_EVERY);
                let Some(a) = weak.upgrade() else { return };
                a.save();
            }
        });
        Some(archive)
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, ()> {
        self.changing.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Counts a pipeline `add` put in the archive.
    fn add(&self, what: &str, add: impl FnOnce() -> bool) {
        let started = Instant::now();
        let _changing = self.changing.write().unwrap_or_else(|e| e.into_inner());
        if add() {
            self.added.fetch_add(1, Ordering::Relaxed);
        }
        note(started, what);
    }

    /// The pipeline of `d`: from the archive, else compiled and added.
    pub(crate) fn render(
        &self,
        dev: &Dev,
        d: &MTLRenderPipelineDescriptor,
        crash_key: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
        let started = Instant::now();
        d.setBinaryArchives(Some(&self.list));
        let reading = self.read();
        if let Ok(p) = dev.newRenderPipelineStateWithDescriptor_options_reflection_error(
            d,
            MTLPipelineOption::FailOnBinaryArchiveMiss,
            None,
        ) {
            note(started, "render pipeline from the archive");
            return Ok(p);
        }
        drop(reading);
        // No archive to look in this time: the miss is known, and the compile would read the
        // archive without holding `changing`.
        d.setBinaryArchives(None);
        let p = compile_render(dev, d, crash_key)?;
        self.add("render pipeline added to the archive", || {
            self.archive.addRenderPipelineFunctionsWithDescriptor_error(d).is_ok()
        });
        Ok(p)
    }

    /// As [`Self::render`], for a mesh pipeline. Without this each run compiled every mesh pipeline
    /// again on the frame that first draws with it: seconds each at the first frame of the world.
    pub(crate) fn mesh(&self, dev: &Dev, d: &MTLMeshRenderPipelineDescriptor, crash_key: &str) -> Result<RenderPso, String> {
        let started = Instant::now();
        if let Some(p) = self.archived_mesh(dev, d) {
            note(started, "mesh pipeline from the archive");
            return Ok(p);
        }
        d.setBinaryArchives(None);
        let p = compile_mesh(dev, d, crash_key)?;
        self.add("mesh pipeline added to the archive", || {
            self.archive.addMeshRenderPipelineFunctionsWithDescriptor_error(d).is_ok()
        });
        Ok(p)
    }

    fn archived_mesh(&self, dev: &Dev, d: &MTLMeshRenderPipelineDescriptor) -> Option<RenderPso> {
        d.setBinaryArchives(Some(&self.list));
        let _reading = self.read();
        dev.newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
            d,
            MTLPipelineOption::FailOnBinaryArchiveMiss,
            None,
        )
        .ok()
    }

    /// As [`Self::render`], for a compute function.
    pub(crate) fn compute(
        &self,
        dev: &Dev,
        d: &MTLComputePipelineDescriptor,
        crash_key: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        let started = Instant::now();
        d.setBinaryArchives(Some(&self.list));
        let reading = self.read();
        if let Ok(p) = dev.newComputePipelineStateWithDescriptor_options_reflection_error(
            d,
            MTLPipelineOption::FailOnBinaryArchiveMiss,
            None,
        ) {
            note(started, "compute pipeline from the archive");
            return Ok(p);
        }
        drop(reading);
        d.setBinaryArchives(None);
        let p = compile_compute(dev, d, crash_key)?;
        self.add("compute pipeline added to the archive", || {
            self.archive.addComputePipelineFunctionsWithDescriptor_error(d).is_ok()
        });
        Ok(p)
    }

    /// Writes the archive back (to a temporary name, then renamed) if pipelines were added.
    fn save(&self) {
        // One writer at a time: the temporary name is the process's.
        static WRITING: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _writing = WRITING.lock().unwrap_or_else(|e| e.into_inner());
        if self.added.swap(0, Ordering::Relaxed) == 0 {
            return;
        }
        let started = Instant::now();
        let _reading = self.read();
        let tmp = self.path.with_extension(format!("{}.tmp", std::process::id()));
        if self.archive.serializeToURL_error(&url(&tmp)).is_err() || std::fs::rename(&tmp, &self.path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        note(started, "pipeline archive written");
    }
}

impl Drop for Archive {
    fn drop(&mut self) {
        self.save();
    }
}

/// How long a render pipeline may compile before it is refused for this run. Metal's compiler
/// service crashes on some pipelines, and Metal retries before it reports the crash: about 30 s a
/// pipeline on macOS 15 (0.7 s on macOS 27), on the frame. A normal compile takes well under this.
/// shortcut: a pipeline that compiles slower than this is refused for the run, and the compile it
/// finishes is dropped (not archived); keep it in the archive if a real one ever needs longer.
/// It was 10 s while the frame's thread waited for it. Compiles now run side by side on the
/// `pipeline-compile` threads, where a cold world frame's took up to 15 s each, and the frame
/// waits for them only inside its own budget (`metal::within`).
const COMPILE_DEADLINE: Duration = Duration::from_secs(60);

type RenderPso = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;
type ComputePso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Metal objects handed to the compile thread: Metal's devices, descriptors and pipeline states
/// may be used from any thread.
pub(crate) struct Sendable<T>(pub(crate) T);
// SAFETY: as above; only the compile thread uses them once moved.
unsafe impl<T> Send for Sendable<T> {}

fn compile_with_deadline<P: 'static>(
    what: &str,
    crash_key: &str,
    compile: impl FnOnce() -> Result<P, String> + Send + 'static,
) -> Result<P, String> {
    let job = move || Sendable(compile());
    let late_key = crash_key.to_owned();
    let started = Instant::now();
    let made = with_deadline(COMPILE_DEADLINE, job, move |r| remember_crash(&r.0, &late_key));
    note(started, what);
    let Some(Sendable(r)) = made else {
        return Err(format!(
            "Metal's compiler took over {} s (refused for this run; a crash is remembered when it ends)",
            COMPILE_DEADLINE.as_secs()
        ));
    };
    remember_crash(&r, crash_key);
    r
}

/// Compiles the render pipeline `d` on a thread of its own, waiting at most `COMPILE_DEADLINE`.
/// A compile that ends in a compiler crash, in time or after, is remembered under `crash_key`
/// ([`crate::disk::remember_crash`]), so later runs refuse it at once.
pub(crate) fn compile_render(dev: &Dev, d: &MTLRenderPipelineDescriptor, crash_key: &str) -> Result<RenderPso, String> {
    use objc2_foundation::NSCopying;
    let job = Sendable((dev.clone(), d.copy()));
    compile_with_deadline("render pipeline compiled", crash_key, move || {
        let job = job;
        let (dev, d) = &job.0;
        dev.newRenderPipelineStateWithDescriptor_error(d)
            .map_err(|e| e.localizedDescription().to_string())
    })
}

/// As [`compile_render`], for a mesh render pipeline descriptor.
pub(crate) fn compile_mesh(dev: &Dev, d: &MTLMeshRenderPipelineDescriptor, crash_key: &str) -> Result<RenderPso, String> {
    use objc2_foundation::NSCopying;
    let job = Sendable((dev.clone(), d.copy()));
    compile_with_deadline("mesh pipeline compiled", crash_key, move || {
        let job = job;
        let (dev, d) = &job.0;
        dev.newRenderPipelineStateWithMeshDescriptor_options_reflection_error(d, MTLPipelineOption::None, None)
            .map_err(|e| e.localizedDescription().to_string())
    })
}

/// As [`compile_render`], for a compute pipeline descriptor.
pub(crate) fn compile_compute(dev: &Dev, d: &MTLComputePipelineDescriptor, crash_key: &str) -> Result<ComputePso, String> {
    use objc2_foundation::NSCopying;
    let job = Sendable((dev.clone(), d.copy()));
    compile_with_deadline("compute pipeline compiled", crash_key, move || {
        let job = job;
        let (dev, d) = &job.0;
        dev.newComputePipelineStateWithDescriptor_options_reflection_error(d, MTLPipelineOption::None, None)
            .map_err(|e| e.localizedDescription().to_string())
    })
}

/// Whether `error`, the text of a pipeline that would not compile, is Metal's compiler dying on
/// the pipeline (its service crashed and the compile was retried, or it reported an internal
/// error) rather than a shader or a pipeline state the host cannot run. The first kind is the
/// pipeline's fault and worth remembering and retrying later; the second is the translation's.
pub fn is_compiler_crash(error: &str) -> bool {
    error.contains("XPC_ERROR_CONNECTION_INTERRUPTED") || error.contains("Compiler encountered an internal error")
}

/// Logs slow pipeline work, as `msc` does a slow conversion: the frame's thread waits for it.
fn note(started: Instant, what: &str) {
    let ms = started.elapsed().as_millis();
    if ms >= 20 {
        eprintln!("metal: {what}: {ms} ms");
    }
}

/// Crashes of Metal's compiler on one pipeline in this run before it is remembered on disk
/// ([`crate::disk::remember_crash`]). One crash of the compiler service fails every compile it
/// had in flight, so a single crash does not say which pipeline caused it: GTA VI's G-buffer lost
/// a dozen pipelines a run with one crash report, and remembered at the first failure each was
/// refused for good on every later run.
pub const CRASHES_REMEMBERED: u32 = 3;

fn remember_crash<T>(r: &Result<T, String>, key: &str) {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashMap<String, u32>>> = Mutex::new(None);
    if let Err(e) = r
        && is_compiler_crash(e)
    {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let n = seen.get_or_insert_with(HashMap::new).entry(key.to_string()).or_insert(0);
        *n += 1;
        if *n == CRASHES_REMEMBERED {
            crate::disk::remember_crash(key.as_bytes());
        }
    }
}

/// `f`'s result if it comes within `deadline`; else `None`, and `late` gets it when it comes.
fn with_deadline<T: Send + 'static>(
    deadline: Duration,
    f: impl FnOnce() -> T + Send + 'static,
    late: impl FnOnce(T) + Send + 'static,
) -> Option<T> {
    use std::sync::{Arc, Mutex, mpsc};
    let gave_up = Arc::new(Mutex::new(false));
    let (tx, rx) = mpsc::channel();
    let flag = gave_up.clone();
    std::thread::spawn(move || {
        let r = f();
        let gave_up = flag.lock().unwrap_or_else(|e| e.into_inner());
        if *gave_up {
            late(r);
        } else {
            let _ = tx.send(r);
        }
    });
    match rx.recv_timeout(deadline) {
        Ok(r) => Some(r),
        Err(_) => {
            *gave_up.lock().unwrap_or_else(|e| e.into_inner()) = true;
            // The result may have been sent between the timeout and the flag.
            rx.try_recv().ok()
        }
    }
}

fn url(p: &std::path::Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&p.to_string_lossy()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn a_result_past_its_deadline_goes_to_the_late_handler() {
        let (tx, rx) = mpsc::channel();
        let quick = with_deadline(Duration::from_secs(5), || 1, |_| panic!("not late"));
        assert_eq!(quick, Some(1));
        let slow = with_deadline(
            Duration::from_millis(10),
            || {
                std::thread::sleep(Duration::from_millis(200));
                2
            },
            move |r| tx.send(r).unwrap(),
        );
        assert_eq!(slow, None);
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)), Ok(2));
    }

    #[test]
    fn recognizes_reported_metal_compiler_failures() {
        assert!(is_compiler_crash("Compiler encountered an internal error"));
        assert!(is_compiler_crash("Compilation failed: Compiler encountered an internal error."));
        assert!(is_compiler_crash(
            "Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED"
        ));
        assert!(!is_compiler_crash("Fragment input mismatching vertex shader output type"));
        assert!(!is_compiler_crash("not enough memory to create pipeline"));
    }

    fn mesh_descriptor(dev: &Dev) -> Retained<MTLMeshRenderPipelineDescriptor> {
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            struct Corner { float4 position [[position]]; };
            using Triangle = metal::mesh<Corner, void, 3, 1, metal::topology::triangle>;
            [[mesh]] void test_mesh(Triangle out) {
                out.set_primitive_count(1);
                for (uint i = 0; i < 3; i++) {
                    Corner c;
                    c.position = float4(i == 1 ? 3.0 : -1.0, i == 2 ? 3.0 : -1.0, 0.0, 1.0);
                    out.set_vertex(i, c);
                    out.set_index(i, i);
                }
            }
            fragment float4 test_fragment() { return float4(0.0, 1.0, 0.0, 1.0); }
        "#;
        let library = dev.newLibraryWithSource_options_error(&NSString::from_str(source), None).expect("test shaders");
        let d = MTLMeshRenderPipelineDescriptor::new();
        // SAFETY: the functions are of the stages they are set as; the descriptor has attachment 0.
        unsafe {
            d.setMeshFunction(library.newFunctionWithName(&NSString::from_str("test_mesh")).as_deref());
            d.setFragmentFunction(library.newFunctionWithName(&NSString::from_str("test_fragment")).as_deref());
            d.colorAttachments().objectAtIndexedSubscript(0).setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        }
        d
    }

    /// A mesh pipeline is compiled once: a later run finds it in the archive's file.
    #[test]
    fn a_mesh_pipeline_is_kept_in_the_archive() {
        let Some(dev) = MTLCreateSystemDefaultDevice() else { return };
        let dir = tempfile::tempdir().unwrap();
        let first = Archive::open_in(&dev, dir.path()).unwrap();
        assert!(first.archived_mesh(&dev, &mesh_descriptor(&dev)).is_none());
        first.mesh(&dev, &mesh_descriptor(&dev), "mesh").unwrap();
        // Written here, not left to the drop: the writing thread may hold the archive just then.
        first.save();
        drop(first);
        let second = Archive::open_in(&dev, dir.path()).unwrap();
        assert!(second.archived_mesh(&dev, &mesh_descriptor(&dev)).is_some());
    }

    #[test]
    fn compute_archive_hits_after_flush_across_reloaded_libraries() {
        let Some(m) = crate::metal::Metal::new().ok() else { return };
        let Some(msc) = m.msc().ok() else { return };
        let affine = include_bytes!("../tests/affine.dxil");
        let (_, lib) = msc.convert(affine, None).unwrap();
        let dir = tempfile::tempdir().unwrap();

        let l1 = crate::metal::load_library(&m.dev, &lib).unwrap();
        let name1 = l1.functionNames().iter().next().unwrap();
        let f1 = l1.newFunctionWithName(&name1).unwrap();
        let d1 = MTLComputePipelineDescriptor::new();
        d1.setComputeFunction(Some(&f1));

        let a1 = Archive::open_in(&m.dev, dir.path()).unwrap();
        a1.compute(&m.dev, &d1, "affine").unwrap();
        // What the `pipeline-archive` thread does every `SAVE_EVERY`, with no drop (`Title` is in a static).
        a1.save();
        assert_eq!(a1.added.load(Ordering::Relaxed), 0);
        assert!(a1.path.exists());

        // Reload the converted metallib from a fresh temporary URL (as a second run does)
        // and verify FailOnBinaryArchiveMiss hits on the reopened archive.
        let l2 = crate::metal::load_library(&m.dev, &lib).unwrap();
        let name2 = l2.functionNames().iter().next().unwrap();
        let f2 = l2.newFunctionWithName(&name2).unwrap();
        let d2 = MTLComputePipelineDescriptor::new();
        d2.setComputeFunction(Some(&f2));
        let a2 = Archive::open_in(&m.dev, dir.path()).unwrap();
        d2.setBinaryArchives(Some(&a2.list));
        m.dev
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &d2,
                MTLPipelineOption::FailOnBinaryArchiveMiss,
                None,
            )
            .expect("reopened archive should hit FailOnBinaryArchiveMiss");
    }
}
