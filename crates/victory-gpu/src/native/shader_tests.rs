//! The shader path of the native host: translation settings, the library cache, pipelines from
//! translated shaders and graphics state, and refusals that are counted instead of failing a
//! submission. Our own HLSL (`victory-metal/tests/fixtures`) stands in for translator output: a
//! `HostShader` is plain data. Skipped without the Metal Shader Converter.

use crate::native::memory::PoolTable;
use crate::native::shaders::{self, ShaderStats};
use crate::native::state::{self, Graphics};
use crate::native::title::{Pipe, Title};
use std::sync::Arc;
use victory_metal::metal::Metal;
use victory_shader::HwStage;
use victory_shader::emit::HostShader;

fn read(path: &str) -> Vec<u8> {
    let p = format!("{}/../victory-metal/{path}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn title() -> Option<Title<Metal>> {
    if !std::path::Path::new(victory_metal::msc::DEFAULT_DYLIB).exists()
        && std::env::var_os("VICTORY_MSC_DYLIB").is_none()
    {
        eprintln!("skipped: Metal Shader Converter not installed");
        return None;
    }
    Some(Title::new(Metal::new().unwrap(), Arc::new(PoolTable::default())))
}

/// A translated shader as plain data: our own DXIL, no title arguments.
fn host_shader(stage: HwStage, dxil: Vec<u8>) -> HostShader {
    HostShader {
        stage,
        dxil,
        user_data: vec![],
        samplers: vec![],
        static_samplers: vec![],
        root_cbvs: vec![],
        root_srvs: vec![],
        root_uavs: vec![],
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: false,
        heap_writes: false,
        view_marks: false,
        splits: vec![],
        table_entries: vec![],
    }
}

#[test]
fn the_host_translates_for_32_lane_waves() {
    let o = shaders::options();
    assert!(o.wave32_host);
    // Everything else is what D3D12 gets by default, except the workarounds `wave32_host` implies.
    assert!(!o.view_marks && !o.atomic_locks && !o.bias_as_gradients);
    assert!(shaders::translate(b"not a container").is_err());
}

#[test]
fn group_sizes_come_from_the_converted_container() {
    assert_eq!(shaders::threads_of(&read("tests/affine.dxil")), Some([64, 1, 1]));
    assert_eq!(shaders::threads_of(b"junk"), None);
}

/// One conversion per distinct DXIL: a second pipeline of the same shader (another title pipeline
/// object with the same container) is served from the library cache.
#[test]
fn compute_pipelines_are_cached_by_the_content_of_their_dxil() {
    let Some(mut t) = title() else { return };
    let cs = host_shader(HwStage::Cs, read("tests/affine.dxil"));
    let first = t.compute_pipe(&cs, "a");
    let again = t.compute_pipe(&cs.clone(), "b");
    let (Pipe::Compute(a), Pipe::Compute(b)) = (first, again) else {
        panic!("not made: {:?}", t.shaders)
    };
    assert!(std::rc::Rc::ptr_eq(&a, &b));
    assert_eq!((t.shaders.pipelines, t.shaders.cached), (1, 1));
    assert_eq!(a.threads, [64, 1, 1]);
    let other = host_shader(HwStage::Cs, read("tests/fixtures/heap_rw.dxil"));
    assert!(matches!(t.compute_pipe(&other, "c"), Pipe::Compute(_)));
    assert_eq!(t.shaders.pipelines, 2);
}

/// A compute pipeline Metal has not compiled within the frame's wait is put off, not refused: a
/// later use picks it up, and each miss is counted.
#[test]
fn a_compute_pipeline_put_off_is_made_by_a_later_use() {
    let Some(mut t) = title() else { return };
    let cs = host_shader(HwStage::Cs, read("tests/affine.dxil"));
    t.compile_wait = std::time::Duration::ZERO;
    let mut asks = 0;
    while !matches!(t.compute_pipe(&cs, "a"), Pipe::Compute(_)) {
        asks += 1;
        assert!(asks < 3000, "never made: {:?}", t.shaders);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!((t.shaders.put_off, t.shaders.pipelines, t.shaders.refusals()), (asks, 1, 0));
}

/// What the host cannot give a shader is refused, counted by reason, and logged once; nothing
/// fails.
#[test]
fn shaders_the_host_cannot_run_are_refused_and_counted() {
    let Some(mut t) = title() else { return };
    let dxil = read("tests/affine.dxil");
    let mut paged = host_shader(HwStage::Cs, dxil.clone());
    paged.memory_writes = true;
    let garbage = host_shader(HwStage::Cs, b"DXBC but not really".to_vec());
    for s in [&paged, &paged, &garbage] {
        // The same refusal twice is two refusals of one reason.
        assert!(matches!(t.compute_pipe(s, "x"), Pipe::Refused));
    }
    let ShaderStats { refused, pipelines, .. } = t.shaders.clone();
    assert_eq!(pipelines, 0);
    assert_eq!(refused.len(), 2, "{refused:?}");
    assert!(refused.iter().any(|(why, _)| why.contains("marks the pages")));
    assert!(refused.iter().any(|(why, _)| why.contains("thread group") || why.contains("conversion")));
    // A described graphics pipeline gets the page table as compute and guessed ones do; a
    // stage that marks the pages it writes is refused there too.
    let mut vs = host_shader(HwStage::Vs, read("tests/fixtures/tri_vs.dxil"));
    vs.memory = true;
    let g = Graphics::default();
    assert!(!g.is_guessed());
    let _ = t.render_pipe(&vs, None, &g, "vs");
    assert!(!t.shaders.refused.iter().any(|(why, _)| why.contains("page table")), "{:?}", t.shaders.refused);
    let before = t.shaders.refusals();
    vs.memory_writes = true;
    assert!(matches!(t.render_pipe(&vs, None, &g, "marks"), Pipe::Refused));
    t.refused_translation("y", "no DXIL part".into());
    assert_eq!(t.shaders.refusals(), before + 2, "{:?}", t.shaders.refused);
}

/// A graphics pipeline from a vertex and pixel shader and a pipeline-state stream's state:
/// the input layout in the shader's signature order, formats, blend, depth, stencil and raster.
#[test]
fn render_pipelines_come_from_the_translated_shaders_and_the_streams_state() {
    let Some(mut t) = title() else { return };
    let vs = host_shader(HwStage::Vs, read("tests/fixtures/tri_vs.dxil"));
    let ps = host_shader(HwStage::Ps, read("tests/fixtures/tri_ps.dxil"));
    // The vertex shader reads POSITION, TEXCOORD, COLOR, in the signature's order.
    let reads = state::vertex_inputs(&vs.dxil).unwrap();
    assert_eq!(
        reads,
        [("POSITION".to_owned(), 0), ("TEXCOORD".to_owned(), 0), ("COLOR".to_owned(), 0)]
    );

    // A stream with the layout, a target and blending on.
    let names = [c"POSITION", c"TEXCOORD", c"COLOR"];
    let formats = [16u32, 16, 2];
    let mut elements = Vec::new();
    for (i, n) in names.iter().enumerate() {
        let mut e = [0u8; 32];
        e[..8].copy_from_slice(&(n.as_ptr() as u64).to_le_bytes());
        for (k, w) in [0, formats[i], 0, u32::MAX, 0, 0].into_iter().enumerate() {
            e[8 + 4 * k..12 + 4 * k].copy_from_slice(&w.to_le_bytes());
        }
        elements.extend(e);
    }
    let mut layout = 12u32.to_le_bytes().to_vec();
    layout.extend([0; 4]);
    layout.extend((elements.as_ptr() as u64).to_le_bytes());
    layout.extend(3u32.to_le_bytes());
    layout.extend([0; 4]);
    let mut rts = 15u32.to_le_bytes().to_vec();
    for f in [28u32, 0, 0, 0, 0, 0, 0, 0, 1] {
        rts.extend(f.to_le_bytes());
    }
    rts.extend([0; 4]);
    let mut g = Graphics::default();
    for (kind, s) in [(12u32, &layout), (15, &rts)] {
        // SAFETY: whole subobjects over live elements and names.
        unsafe { g.read(kind, s.as_ptr()) };
    }
    // With no time to wait it is put off while Metal compiles it, not refused: nothing is cached
    // and a later use picks it up.
    t.compile_wait = std::time::Duration::ZERO;
    let mut asks = 0;
    let pipe = loop {
        let pipe = t.render_pipe(&vs, Some(&ps), &g, "tri");
        if matches!(pipe, Pipe::Render(..)) {
            break pipe;
        }
        asks += 1;
        assert!(asks < 3000, "never made: {:?}", t.shaders);
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert!(asks > 0);
    assert_eq!((t.shaders.put_off, t.shaders.refusals()), (asks, 0));
    t.compile_wait = std::time::Duration::MAX;
    assert!(matches!(t.render_pipe(&vs, Some(&ps), &g, "tri again"), Pipe::Render(..)));
    assert_eq!((t.shaders.pipelines, t.shaders.cached), (1, 1));

    // A different state is a different pipeline; a layout the shader's input is missing from is
    // refused, not drawn wrong.
    let mut no_layout = Graphics::default();
    // SAFETY: as above, only the formats.
    unsafe { no_layout.read(15, rts.as_ptr()) };
    assert!(matches!(t.render_pipe(&vs, Some(&ps), &no_layout, "bare"), Pipe::Refused));
    // A pixel stage with arguments of its own is converted with the host's layout of them: this
    // fixture (plain HLSL with a texture at t0, not translator output) is then refused, as its
    // texture is in no host layout.
    let mut args = ps.clone();
    args.user_data.push(victory_shader::emit::UserData {
        index: 0,
        slot: 0,
        root: victory_shader::RootType::Constants,
        dword: 0,
    });
    assert!(matches!(t.render_pipe(&vs, Some(&args), &g, "args"), Pipe::Refused));
    assert!(t.shaders.refused.iter().any(|(why, _)| why.starts_with("args") && why.contains("root signature")));
    assert!(t.shaders.refused.iter().any(|(why, _)| why.contains("layout lacks")));
}
