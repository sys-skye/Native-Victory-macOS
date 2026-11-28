//! Title shaders on the native host: the console container translated to DXIL for a 32-lane
//! host (`Options { wave32_host: true, .. }`: the console's wave64 compute programs run as two host
//! waves), DXIL to a Metal library through the converter (`create_compute_pipeline` and
//! `create_render_pipeline` on the backend: every library is its own file name, because Metal
//! keeps libraries by URL), pipelines cached by the content of their DXIL, and what the host
//! cannot run refused, logged and counted instead of failing the submission.
//!
//! Title memory at addresses the shader computes is found through the page table of
//! `victory_shader`'s "Title memory", which `super::addr` fills with views of the pools
//! themselves (the pool-offset contract). What a translated shader needs
//! from the host that this one does not give yet, and is refused for: buffer views read from the
//! heap by index and the write marks and view addresses of the shadow-copy design. See
//! docs/metal-backend.md.

use super::list::{Binding, ComputePso};
use super::state::{self, Graphics};
use super::title::{Pipe, Title};
use std::collections::HashMap;
use std::rc::Rc;
use victory_backend::resources::*;
use victory_shader::emit::{HostShader, Options};

/// The translation settings of the native host.
pub fn options() -> Options {
    Options {
        wave32_host: true,
        zero_copy_memory: true,
        ..Options::default()
    }
}

/// A console shader container as DXIL for the native host.
pub fn translate(container: &[u8]) -> Result<HostShader, String> {
    translate_cached(container, options())
}

/// Whether [`translate`]'s translation is in the disk cache already.
pub fn cached(container: &[u8]) -> bool {
    crate::shader_cache::native_present(container, &format!("{:?}", options()))
}

/// Keeps `container`, as the run is about to translate it, for later warm-ups
/// (`crate::prewarm::keep`); nothing with the shader cache off.
#[cfg(target_os = "macos")]
pub fn keep(container: &[u8]) {
    if let Some(dir) = crate::prewarm::kept_dir() {
        crate::prewarm::keep(&dir, &blake3::hash(container).to_hex()[..16], container);
    }
}

/// The files of the containers earlier runs kept ([`keep`]).
#[cfg(target_os = "macos")]
pub fn kept() -> Vec<std::path::PathBuf> {
    crate::prewarm::kept_dir().map(|dir| crate::prewarm::kept(&dir)).unwrap_or_default()
}

/// The container in one of [`kept`]'s files; `None` if it is damaged.
#[cfg(target_os = "macos")]
pub fn read_kept(file: &std::path::Path) -> Option<Vec<u8>> {
    crate::prewarm::read_kept(file)
}

/// [`victory_shader::emit::translate_with`] through the on-disk cache (`shader_cache::native_shader`):
/// `o` is part of the key.
fn translate_cached(container: &[u8], o: Options) -> Result<HostShader, String> {
    #[cfg(any(windows, target_os = "macos"))]
    return crate::shader_cache::native_shader(container, &format!("{o:?}"), |c| {
        victory_shader::emit::translate_with(c, o).map_err(|e| e.to_string())
    });
    #[cfg(not(any(windows, target_os = "macos")))]
    return victory_shader::emit::translate_with(container, o).map_err(|e| e.to_string());
}

/// A compute shader's container translated for dispatches of thread counts
/// (`Options::thread_counts`).
pub fn translate_threads(container: &[u8]) -> Result<HostShader, String> {
    translate_cached(container, Options { thread_counts: true, ..options() })
}

/// A guessed vertex shader's container translated to draw rectangle lists (`Options::rect_list`).
pub fn translate_rect(container: &[u8]) -> Result<HostShader, String> {
    translate_cached(container, Options { rect_list: true, ..options() })
}

/// A guessed vertex shader's container translated for rectangle lists or not, writing the
/// `TEXCOORD` parameters in `extra` (bit n) too: those its pixel shader reads and it does not write.
pub fn translate_extra(container: &[u8], rect: bool, extra: u32, copy_params: [u8; 32]) -> Result<HostShader, String> {
    translate_cached(container, Options { rect_list: rect, extra_params: extra, copy_params, ..options() })
}

/// Whether [`translate_extra`]'s translation is in the disk cache already.
/// A guessed pixel shader's container translated for draws into unsigned-integer targets (bit n:
/// colour target n), which Metal pairs only with integer outputs (`Options::uint_targets`).
pub fn translate_uint(container: &[u8], uint_targets: u8) -> Result<HostShader, String> {
    translate_cached(container, Options { uint_targets, ..options() })
}

/// Whether [`translate_uint`] would find its translation on disk.
pub fn uint_cached(container: &[u8], uint_targets: u8) -> bool {
    let o = Options { uint_targets, ..options() };
    crate::shader_cache::native_present(container, &format!("{o:?}"))
}

pub fn extra_cached(container: &[u8], rect: bool, extra: u32, copy_params: [u8; 32]) -> bool {
    let o = Options { rect_list: rect, extra_params: extra, copy_params, ..options() };
    #[cfg(any(windows, target_os = "macos"))]
    return crate::shader_cache::native_present(container, &format!("{o:?}"));
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = (container, rect, extra, copy_params, o);
        false
    }
}

/// A pixel shader's container translated for colour targets of integer formats (bit n: target n).
pub fn translate_targets(container: &[u8], integer: u8) -> Result<HostShader, String> {
    translate_cached(container, Options { integer_targets: integer, ..options() })
}

/// Whether [`translate_targets`]'s translation is in the disk cache already.
pub fn targets_cached(container: &[u8], integer: u8) -> bool {
    let o = Options { integer_targets: integer, ..options() };
    #[cfg(any(windows, target_os = "macos"))]
    return crate::shader_cache::native_present(container, &format!("{o:?}"));
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = (container, o);
        false
    }
}

/// Bit n set: colour target n has an integer format.
pub fn integer_targets(color: &[Format]) -> u8 {
    color.iter().enumerate().filter(|(n, f)| *n < 8 && f.is_integer()).fold(0, |m, (n, _)| m | 1 << n)
}

/// What the host cannot give a translated shader (the reason is the refusal's); `memory`: it can
/// give the page table (every pipeline of the Metal host: compute, guessed and described graphics). A shader that indexes the resource
/// heap itself is not refused here: the bound heaps are at the converter's heap slots for every
/// stage (`bind_root_render`, `bind_root` in `victory-metal`).
pub fn unsupported(shader: &HostShader, memory: bool) -> Option<&'static str> {
    if shader.memory_writes {
        return Some("marks the pages it writes (translated without zero_copy_memory)");
    }
    if shader.memory && !memory {
        return Some("reads title memory through the page table in a graphics pipeline");
    }
    if shader.view_marks {
        return Some("needs write marks");
    }
    None
}

/// Why a vertex shader and pixel shader cannot be a pipeline of the host, if they cannot.
fn refusal(vs: &HostShader, ps: Option<&HostShader>, memory: bool) -> Option<&'static str> {
    for s in [Some(vs), ps].into_iter().flatten() {
        if let Some(why) = unsupported(s, memory) {
            return Some(why);
        }
    }
    None
}

/// The thread-group size a compute container declares (`PSV0`'s runtime info).
pub fn threads_of(dxil: &[u8]) -> Option<[usize; 3]> {
    use victory_shader::{Container, FourCC};
    let container = Container::parse(dxil).ok()?;
    let part = container.part(FourCC::PSV0)?;
    // The runtime info is 48 bytes from version 2 on; the group size is its last three words.
    if u32::from_le_bytes(part.data.get(..4)?.try_into().ok()?) < 48 {
        return None;
    }
    let d = part.data.get(40..52)?;
    let n = |k: usize| u32::from_le_bytes(d[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
    Some([n(0), n(1), n(2)]).filter(|t| t.iter().all(|&n| n > 0))
}

/// Counts of what the shader path did, and why it refused what it did.
#[derive(Default, Debug, Clone)]
pub struct ShaderStats {
    /// Pipelines made.
    pub pipelines: u64,
    /// Pipelines served from the library cache.
    pub cached: u64,
    /// Times a pipeline was not compiled within `Title::compile_wait`: its work is dropped that
    /// time and nothing is cached, so a later use picks the pipeline up.
    pub put_off: u64,
    /// Refusals, by reason, with how often.
    pub refused: Vec<(String, u64)>,
    /// Pipelines whose compile died with Metal's compiler, by name, with how often
    /// ([`Self::crashed`]).
    pub crashes: HashMap<String, u32>,
}

impl ShaderStats {
    /// All refusals.
    pub fn refusals(&self) -> u64 {
        self.refused.iter().map(|r| r.1).sum()
    }

    /// Whether the compile of pipeline `what` that ended in `error` is put off rather than
    /// refused: Metal's compiler died on it (`victory_metal::is_compiler_crash`) fewer than
    /// `victory_metal::CRASHES_REMEMBERED` times. One crash of the compiler service fails every
    /// compile it had in flight, most of them pipelines it would have compiled, so those are
    /// tried again on a later use and only a pipeline that keeps crashing it is refused.
    #[cfg(target_os = "macos")]
    fn crashed(&mut self, what: &str, error: &str) -> bool {
        if !victory_metal::is_compiler_crash(error) {
            return false;
        }
        let n = self.crashes.entry(what.to_string()).or_insert(0);
        *n += 1;
        *n < victory_metal::CRASHES_REMEMBERED
    }

    fn refuse(&mut self, what: &str, reason: String) {
        let key = format!("{what}: {reason}");
        match self.refused.iter_mut().find(|r| r.0 == key) {
            Some(r) => r.1 += 1,
            None => {
                tracing::warn!(shader = what, %reason, "title shader refused; its work is dropped");
                if self.refused.len() < 256 {
                    self.refused.push((key, 1));
                }
            }
        }
    }
}

/// The pipelines made, by the content of their DXIL (and, for graphics, their state).
pub struct Libraries<B: GraphicsBackend> {
    compute: HashMap<[u8; 32], Pipe<B>>,
    render: HashMap<[u8; 32], Pipe<B>>,
}

impl<B: GraphicsBackend> Default for Libraries<B> {
    fn default() -> Self {
        Libraries {
            compute: HashMap::new(),
            render: HashMap::new(),
        }
    }
}

impl<B: PresentBackend> Title<B> {
    /// The compute pipeline of a translated shader: from the cache when its DXIL was converted
    /// before, else converted now. [`Pipe::Refused`] (counted) when the host cannot run it.
    pub fn compute_pipe(&mut self, shader: &HostShader, name: &str) -> Pipe<B> {
        // Shaders indexing the heap themselves run: the heaps are bound for compute as for render
        // (`heap_tests::a_compute_shader_loads_a_texture_by_heap_index`).
        if let Some(why) = unsupported(shader, true) {
            self.shaders.refuse(name, why.into());
            return Pipe::Refused;
        }
        let key = *blake3::hash(&shader.dxil).as_bytes();
        if let Some(p) = self.libraries.compute.get(&key) {
            self.shaders.cached += 1;
            return p.clone();
        }
        let put_off = self.shaders.put_off;
        let made = self.make_compute(shader, name);
        if self.shaders.put_off == put_off {
            self.libraries.compute.insert(key, made.clone());
        }
        made
    }

    fn make_compute(&mut self, shader: &HostShader, name: &str) -> Pipe<B> {
        let Some(threads) = threads_of(&shader.dxil) else {
            self.shaders.refuse(name, "no thread group size".into());
            return Pipe::Refused;
        };
        // Translator output carries no root signature: the argument buffer is laid out by the
        // binding the host fills it by.
        let binding = Binding { name: name.into(), ..Binding::of(shader) };
        let made = self.host.backend.create_compute_pipeline_within(&shader.dxil, binding.layout().as_ref(), self.compile_wait);
        match made {
            Ok(None) => self.put_off(),
            Ok(Some(pso)) => {
                self.shaders.pipelines += 1;
                self.host.backend.name_pipeline(&pso, name);
                Pipe::Compute(Rc::new(ComputePso {
                    pso,
                    threads,
                    binding: Some(binding),
                }))
            }
            Err(e) => {
                #[cfg(target_os = "macos")]
                {
                    self.compile_failed(name, &e);
                    if self.shaders.crashed(name, &e) {
                        return self.put_off();
                    }
                }
                self.shaders.refuse(name, format!("conversion: {e}"));
                Pipe::Refused
            }
        }
    }

    /// The render pipeline of a translated vertex shader, pixel shader and graphics state.
    pub fn render_pipe(
        &mut self,
        vs: &HostShader,
        ps: Option<&HostShader>,
        graphics: &Graphics,
        name: &str,
    ) -> Pipe<B> {
        // A described pipeline binds as a guessed one does (`make_render`, and the draw's
        // `memory_args` for either stage), so either gets the page table.
        if let Some(why) = refusal(vs, ps, true) {
            self.shaders.refuse(name, why.into());
            return Pipe::Refused;
        }
        let mut h = blake3::Hasher::new();
        h.update(&vs.dxil);
        h.update(ps.map_or(&[][..], |p| &p.dxil));
        h.update(format!("{graphics:?}").as_bytes());
        // The arguments are part of the pipeline's binding.
        h.update(format!("{:?}{:?}", vs.user_data, ps.map(|p| &p.user_data)).as_bytes());
        let key = *h.finalize().as_bytes();
        if let Some(p) = self.libraries.render.get(&key) {
            self.shaders.cached += 1;
            return p.clone();
        }
        let put_off = self.shaders.put_off;
        let made = self.make_render(vs, ps, graphics, name);
        if self.shaders.put_off == put_off {
            self.libraries.render.insert(key, made.clone());
        }
        made
    }

    /// A pipeline Metal is still compiling: no pipeline this time, and the caller caches nothing.
    fn put_off(&mut self) -> Pipe<B> {
        self.shaders.put_off += 1;
        if self.shaders.put_off.is_power_of_two() {
            tracing::info!(put_off = self.shaders.put_off, "pipelines put off for their compiles so far");
        }
        Pipe::Refused
    }

    fn make_render(
        &mut self,
        vs: &HostShader,
        ps: Option<&HostShader>,
        g: &Graphics,
        name: &str,
    ) -> Pipe<B> {
        let Some(reads) = state::vertex_inputs(&vs.dxil) else {
            self.shaders.refuse(name, "no input signature".into());
            return Pipe::Refused;
        };
        let Some((mut attributes, streams)) = g.vertex_layout(&reads) else {
            self.shaders
                .refuse(name, "the vertex shader reads an input the layout lacks".into());
            return Pipe::Refused;
        };
        // A system value ahead of the attributes takes an attribute number: described as the
        // attribute after it, which the converter's function ignores.
        for at in state::system_gaps(&vs.dxil) {
            if let Some(next) = attributes.get(at).copied() {
                attributes.insert(at, next);
            }
        }
        // Resolve colour and depth formats, reporting unknown formats as refusals.
        let (color, depth) = match (g.color_formats(), g.depth_format()) {
            (Ok(c), Ok(d)) => (c, d),
            (Err(dxgi), _) => {
                self.shaders
                    .refuse(name, format!("unknown target format {}", dxgi));
                return Pipe::Refused;
            }
            (_, Err(dxgi)) => {
                self.shaders
                    .refuse(name, format!("unknown depth format {}", dxgi));
                return Pipe::Refused;
            }
        };
        if g.samples > 1 {
            self.shaders.refuse(name, "multisampled targets".into());
            return Pipe::Refused;
        }
        let color_state = g.color_state();
        // A pixel shader with one output more than there are targets feeds target 0's second
        // blend source; the converter must be told, or Metal refuses the pipeline.
        let targets = ps.map_or_else(Vec::new, |p| state::pixel_targets(&p.dxil));
        let dual_source = state::dual_source(color.len(), &targets, &color_state);
        if env_set!("VICTORY_TRACE_PIPES") {
            tracing::info!(name, stencil = ?g.stencil(), depth = ?g.depth_test(), ?color_state, "guessed pipeline state");
        }
        let ps_full = ps.map(|ps| Binding { name: name.into(), ..Binding::of(ps) });
        let mut vs_binding = Binding { name: name.into(), ..Binding::of(vs) };
        if let Some(p) = &ps_full {
            vs_binding.typed_buffers |= p.typed_buffers;
            vs_binding.heap_typed_buffers |= p.heap_typed_buffers;
        }
        // As for compute: without the host's layout the converter makes the user-data block a
        // root CBV (a pointer), while the host writes its dwords inline, so the shader reads its
        // user data (descriptor table offsets, root constants) from address 0.
        let vs_layout = vs_binding.layout();
        let ps_layout = ps_full.as_ref().and_then(Binding::layout);
        let ps_binding = ps_full
            .filter(|b| b.memory || b.dwords > 0 || !b.views.is_empty());
        let desc = RenderPipelineDesc {
            vs: &vs.dxil,
            ps: ps.map(|p| &p.dxil[..]),
            vs_layout: vs_layout.as_ref(),
            ps_layout: ps_layout.as_ref(),
            attributes: &attributes,
            streams: &streams,
            color: &color,
            depth,
            depth_test: depth.and(g.depth_test()),
            cull: g.cull(),
            color_state: &color_state,
            dual_source,
            stencil: depth.and(g.stencil()),
            raster: g.raster(),
        };
        match self.host.backend.create_render_pipeline_within(&desc, self.compile_wait) {
            Ok(None) => self.put_off(),
            Ok(Some(p)) => {
                self.shaders.pipelines += 1;
                self.host.backend.name_render_pipeline(&p, name);
                Pipe::Render(Rc::new(p), Some(Rc::new(Binding {
                    ps: ps_binding.map(Rc::new),
                    ..vs_binding
                })))
            }
            Err(e) => {
                #[cfg(target_os = "macos")]
                {
                    self.compile_failed(name, &e);
                    if self.shaders.crashed(name, &e) {
                        return self.put_off();
                    }
                }
                self.shaders.refuse(name, format!("conversion: {e}"));
                Pipe::Refused
            }
        }
    }

    /// A translation that failed, counted.
    pub fn refused_translation(&mut self, name: &str, reason: String) {
        self.shaders.refuse(name, format!("translation: {reason}"));
    }

    /// A pipeline compile that ended in `error`: if Metal's compiler died on it rather than
    /// refusing a shader the host cannot run, the frame spends the rest of its pipeline budget
    /// (`glue::compile_failed`), so the frame's other new pipelines are put off instead of each
    /// waiting about a second for a compiler that crashes (issue #181).
    #[cfg(target_os = "macos")]
    pub(crate) fn compile_failed(&mut self, name: &str, error: &str) {
        super::glue::compile_failed(name, error);
    }
}

/// A pipeline whose compile died with Metal's compiler is put off, not refused, until it has done
/// so `CRASHES_REMEMBERED` times: one crash of the compiler service fails every compile it had in
/// flight. Any other compile error is refused at once.
#[cfg(all(test, target_os = "macos"))]
#[test]
fn compiles_the_compiler_crash_took_down_are_tried_again() {
    let crash = "Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED.";
    let mut s = ShaderStats::default();
    let tries: Vec<bool> = (0..victory_metal::CRASHES_REMEMBERED).map(|_| s.crashed("pipeline a", crash)).collect();
    assert_eq!(tries.iter().filter(|&&t| t).count() as u32, victory_metal::CRASHES_REMEMBERED - 1);
    assert!(!tries.last().unwrap(), "the last crash refuses it");
    assert!(s.crashed("pipeline b", crash), "counted per pipeline");
    assert!(!s.crashed("pipeline c", "Fragment input mismatching vertex shader output type"));
}
