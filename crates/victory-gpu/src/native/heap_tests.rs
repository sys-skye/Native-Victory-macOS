//! Guessed graphics shaders that take descriptors from the resource heap themselves (an SGPR the
//! translator infers as a `ResourceHeap` base, not a table): hand-assembled GFX10 drawn on Metal,
//! with the title's heap bound for the vertex, mesh and pixel stages; and a compute program with
//! no heap record whose user SGPR the translator infers as the heap (`native::infer_heaps`).

use super::addr_tests::title;
use crate::native::list::{NativeHeap, NativeList};
use crate::native::shaders;
use crate::native::state::Graphics;
use crate::native::title::{Pipe, Title};
use std::rc::Rc;
use victory_backend::resources::*;
use victory_metal::metal::Metal;
use victory_shader::emit::HostShader;
use victory_shader::native::HwStage;

fn translate(stage: HwStage, words: &[u32]) -> HostShader {
    let mut meta = [0; 0x10];
    meta[12] = 0x40;
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let c = victory_shader::emit::graphics::container(stage, &bytes, &meta).unwrap();
    shaders::translate(&c).unwrap()
}

/// A title resource heap of two 1x1 RGBA8 textures: entry 0 red, entry 1 green. A shader that
/// reads entry 1 sees green; one that read entry 0 (or nothing) would not. Entry 2 is left for a
/// test's own view.
struct Heap {
    heap: Rc<NativeHeap<Metal>>,
    _records: Vec<u32>,
}

impl Heap {
    fn new(t: &Title<Metal>) -> Heap {
        let records = vec![0u32; 8 * 3];
        let start = records.as_ptr() as u64;
        let heap = t.host.heap(HeapKind::Resource, start, 3).unwrap();
        for (i, texel) in [[255, 0, 0, 255], [0, 255, 0, 255]].into_iter().enumerate() {
            let tex = t.host.texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1)).unwrap();
            t.host.backend.upload_texture(&tex, 0, 0, &texel, 4, 0);
            let key = 0x4EA9_0000 + i;
            t.host.register_texture(key, &tex);
            // DXGI_FORMAT_R8G8B8A8_UNORM, TEXTURE2D, one mip.
            let mut srv = [0u32; 10];
            (srv[0], srv[1], srv[5]) = (28, 4, 1);
            let slot = start + 32 * i as u64;
            crate::descriptor::create_srv(std::ptr::null(), key as _, &srv, slot);
            assert!(t.host.mirror(slot).unwrap());
        }
        Heap { heap, _records: records }
    }
}

/// Draws `vs` and `ps` as a guessed pipeline into an 8x8 RGBA8 target cleared to black, with the
/// heap bound and `record` issuing the draw; the target's pixels.
fn draw(
    t: &mut Title<Metal>,
    heap: &Heap,
    vs: &HostShader,
    ps: &HostShader,
    record: impl FnOnce(&mut NativeList<Metal>),
) -> Vec<[u8; 4]> {
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(vs, Some(ps), &state, "heap") else {
        panic!("{:?}", t.shaders);
    };
    let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 8, 8);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    let mut list = NativeList::default();
    list.set_render_targets(std::slice::from_ref(&color), None);
    list.clear_render_target(&color, [0.0; 4]);
    list.set_heaps(Some(&heap.heap), None);
    list.set_graphics_pipeline(&pso, Topology::Triangles);
    list.set_graphics_binding(binding);
    record(&mut list);
    super::run(&t.host, &[&list]);
    let mut p = [0; 8 * 8 * 4];
    t.host.backend.read_texture(&color, 0, 0, &mut p, 8 * 4, 0);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    p.chunks(4).map(|c| c.try_into().unwrap()).collect()
}

/// A vertex shader whose vertices 0-2 are the corners of a triangle covering the target.
const COVER: &[u32] = &[
    0x7E14_02F3, // v_mov_b32 v10, -1.0
    0x7E16_02FF, 0x4040_0000, // v_mov_b32 v11, 3.0
    0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
    0x0204_170A, // v_cndmask_b32 v2, v10, v11, vcc
    0x7D84_0082, // v_cmp_eq_u32 vcc, 2, v0
    0x0206_170A, // v_cndmask_b32 v3, v10, v11, vcc
    0x7E08_0280, // v_mov_b32 v4, 0
    0x7E0C_02F2, // v_mov_b32 v6, 1.0
    0xF800_08CF, 0x0604_0302, // exp pos0 v2, v3, v4, v6 done
    0xBF81_0000, // s_endpgm
];

/// A pixel input the vertex shader does not write, added to it (`translate_extra`) with another
/// parameter's values, beside a `HALF` of the same index: the pair converts, the input gets the
/// copied values and the `HALF` keeps its own. The vertex shader exports param1 and param3 as
/// packed halves (0.25 and 0.5: `HALF2`, `HALF6`) and param2 as 1.0; `TEXCOORD6` is written with
/// param2's values. The pixel shader writes attr6.x (flat) to red and blue, and attr3.x's low half
/// to green. Before, the converter refused the pair (`TEXCOORD6` was taken for written), and the
/// copy then overwrote `HALF6` with `HALF2` (green 64).
#[test]
fn an_extra_parameter_beside_a_half_of_its_index_links_with_its_own_values() {
    let mut vs_words = COVER.to_vec();
    let at = vs_words.iter().position(|&w| w == 0xF800_08CF).unwrap(); // before exp pos0
    vs_words.splice(
        at..at,
        [
            0x7E0A_02F0, // v_mov_b32 v5, 0.5
            0x7E0E_02FF,
            0x3E80_0000, // v_mov_b32 v7, 0.25
            0x5E10_0B05, // v_cvt_pkrtz_f16_f32 v8, v5, v5
            0x5E12_0F07, // v_cvt_pkrtz_f16_f32 v9, v7, v7
            0x1018_0D06, // v_mul_f32 v12, v6, v6 (1.0)
            // param3 first: a copy made at param1's export would be the last write to `HALF6`.
            0xF800_0231,
            0x0000_0008, // exp param3 v8
            0xF800_0211,
            0x0000_0009, // exp param1 v9
            0xF800_0221,
            0x0000_000C, // exp param2 v12
        ],
    );
    let ps_words = [
        0xC80E_1802, // v_interp_mov_f32 v3, p0, attr6.x
        0xD742_0002,
        0x0202_0003, // v_interp_p1ll_f16 v2, v0, attr3.x
        0xD75A_0004,
        0x040A_0203, // v_interp_p2_f16 v4, v1, attr3.x, v2
        0x7E08_1704, // v_cvt_f32_f16 v4, v4
        0xF800_180F,
        0x0303_0403, // exp mrt0 v3, v4, v3, v3 done vm
        0xBF81_0000, // s_endpgm
    ];
    let mut meta = [0; 0x10];
    meta[12] = 0x40;
    let bytes: Vec<_> = vs_words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let c = victory_shader::emit::graphics::container(HwStage::Vs, &bytes, &meta).unwrap();
    let mut copy = [0; 32];
    copy[6] = 3; // TEXCOORD2 + 1
    let vs = shaders::translate_extra(&c, false, 1 << 6, copy).unwrap();
    let ps = translate(HwStage::Ps, &ps_words);
    let Some(mut t) = title() else { return };
    let heap = Heap::new(&t);
    let pixels = draw(&mut t, &heap, &vs, &ps, |l| {
        l.draw_instanced(3, 1, 0, 0).unwrap()
    });
    assert!(
        pixels.iter().all(|p| *p == [255, 128, 255, 255]),
        "{pixels:?}"
    );
}

/// A pixel shader that loads the image descriptor at heap entry 1 (byte 0x20 from the heap base in
/// s2, high half 0) and writes its texel (0, 0).
const PS_HEAP: &[u32] = &[
    0xBE83_0380, // s_mov_b32 s3, 0
    0xF40C_0101, 0xFA00_0020, // s_load_dwordx8 s[4:11], s[2:3], 0x20
    0x7E00_0280, // v_mov_b32 v0, 0
    0x7E02_0280, // v_mov_b32 v1, 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF000_1F08, 0x0001_0000, // image_load v[0:3], v[0:1], s[4:11] dmask:0xf dim:2D unorm
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xF800_180F, 0x0302_0100, // exp mrt0 v0, v1, v2, v3 done vm
    0xBF81_0000, // s_endpgm
];

/// A pixel shader indexing the heap itself is not refused, and reads the entry it names.
#[test]
fn a_pixel_shader_loads_a_texture_by_heap_index() {
    let (vs, ps) = (translate(HwStage::Vs, COVER), translate(HwStage::Ps, PS_HEAP));
    assert!(ps.heap && !vs.heap);
    let Some(mut t) = title() else { return };
    let heap = Heap::new(&t);
    let pixels = draw(&mut t, &heap, &vs, &ps, |l| l.draw_instanced(3, 1, 0, 0).unwrap());
    assert!(pixels.iter().all(|p| *p == [0, 255, 0, 255]), "{pixels:?}");
}

#[test]
fn a_pixel_shader_reads_the_heap_entry_after_masking_the_address_carry() {
    // Address entry 1 explicitly, then run the same image load and export as PS_HEAP.
    let mut words = vec![
        0x8002_A002, // s_add_u32 s2, s2, 32
        0x8203_8080, // s_addc_u32 s3, 0, 0
        0x8703_FF03, 0xff, // s_and_b32 s3, s3, 0xff
        0xF40C_0101, 0xFA00_0000, // s_load_dwordx8 s[4:11], s[2:3], 0
    ];
    words.extend_from_slice(&PS_HEAP[3..]);
    let ps = translate(HwStage::Ps, &words);
    assert!(ps.heap);
    let Some(mut t) = title() else { return };
    let heap = Heap::new(&t);
    let vs = translate(HwStage::Vs, COVER);
    let pixels = draw(&mut t, &heap, &vs, &ps, |l| l.draw_instanced(3, 1, 0, 0).unwrap());
    assert!(pixels.iter().all(|p| *p == [0, 255, 0, 255]), "{pixels:?}");
}

/// Loads the image descriptor at heap entry 1 (heap base in `base`, high half 0) into s[12:19] and
/// its texel (0, 0) into v[12:15], then scales the position in v2, v3 by its green channel: 1.0
/// for entry 1, 0.0 (no area, nothing drawn) for entry 0 or an unbound heap.
fn scaled_by_heap(base: u8) -> Vec<u32> {
    let b = u32::from(base);
    vec![
        0xBE80_0380 | (b + 1) << 16, // s_mov_b32 s[base+1], 0
        0xF40C_0300 | b >> 1, 0xFA00_0020, // s_load_dwordx8 s[12:19], s[base:base+1], 0x20
        0x7E10_0280, // v_mov_b32 v8, 0
        0x7E12_0280, // v_mov_b32 v9, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF000_1F08, 0x0003_0C08, // image_load v[12:15], v[8:9], s[12:19] dmask:0xf dim:2D unorm
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x1004_050D, // v_mul_f32 v2, v13, v2
        0x1006_070D, // v_mul_f32 v3, v13, v3
    ]
}

/// `words` with `scaled_by_heap(base)` before its `v_mov_b32 v4, 0` (after the corners).
fn with_heap_scale(words: &[u32], base: u8) -> Vec<u32> {
    let mut words = words.to_vec();
    let at = words.iter().position(|&w| w == 0x7E08_0280).unwrap();
    words.splice(at..at, scaled_by_heap(base));
    words
}

/// Writes (1.0, 0.5, 0.5, 1.0) to mrt0.
const RED: &[u32] = &[
    0x7E00_02F2, // v_mov_b32 v0, 1.0
    0x7E02_02F0, // v_mov_b32 v1, 0.5
    0xF800_180F, 0x0001_0100, // exp mrt0 v0, v1, v1, v0 done vm
    0xBF81_0000, // s_endpgm
];

/// A vertex shader indexing the heap itself (base in s2) reads the entry it names.
#[test]
fn a_vertex_shader_loads_a_texture_by_heap_index() {
    let vs = translate(HwStage::Vs, &with_heap_scale(COVER, 2));
    let ps = translate(HwStage::Ps, RED);
    assert!(vs.heap && vs.stage == HwStage::Vs);
    let Some(mut t) = title() else { return };
    let heap = Heap::new(&t);
    let pixels = draw(&mut t, &heap, &vs, &ps, |l| l.draw_instanced(3, 1, 0, 0).unwrap());
    assert!(pixels.iter().all(|p| *p == [255, 128, 128, 255]), "{pixels:?}");
}

/// A mesh shader indexing the heap itself (base in s10, user data 2 of a mesh program) reads the
/// entry it names: the heap is bound for the mesh stage.
#[test]
fn a_mesh_shader_loads_a_texture_by_heap_index() {
    let ms = translate(HwStage::Vs, &with_heap_scale(super::mesh_tests::TRIANGLE, 10));
    let ps = translate(HwStage::Ps, RED);
    assert!(ms.heap && ms.stage == HwStage::Ms);
    let Some(mut t) = title() else { return };
    let heap = Heap::new(&t);
    let pixels = draw(&mut t, &heap, &ms, &ps, |l| l.dispatch_mesh(1, 1, 1).unwrap());
    assert!(pixels.iter().all(|p| *p == [255, 128, 128, 255]), "{pixels:?}");
}

/// A texture the title releases leaves the heap: the entries showing it are cleared, where they
/// used to keep the host texture alive and resident until the title wrote another view there.
/// Entries of other textures stay, and a texture only forgotten (another description placed at
/// its address) keeps its entries.
#[test]
fn a_released_texture_leaves_the_heap() {
    let Some(t) = title() else { return };
    let heap = Heap::new(&t);
    let (red, green) = (0x4EA9_0000, 0x4EA9_0001);
    t.host.forget_resource(red);
    assert_eq!(t.host.cleared(), 0, "forgetting a texture leaves its entries");
    t.host.register_texture(red, &t.host.texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1)).unwrap());
    t.host.release_resource(green);
    assert_eq!(t.host.cleared(), 1);
    assert!(!t.host.knows(green) && t.host.knows(red));
    assert!(heap.heap.mirrored.borrow().contains_key(&1), "its old record is not mirrored again");
    assert!(heap.heap.take_entries(green).is_empty() && heap.heap.take_entries(red) == [0]);
    // Released again, or a buffer: nothing to clear.
    t.host.release_resource(green);
    assert_eq!(t.host.cleared(), 1);
    // A texture at the released one's address: the entry is checked again, the other is not.
    t.host.register_texture(green, &t.host.texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1)).unwrap());
    assert!(!heap.heap.mirrored.borrow().contains_key(&1) && heap.heap.mirrored.borrow().contains_key(&0));
}

/// A compute program of the title's shape: a raw UAV in the table at user data 4, and the
/// resource heap's base in s6, which no metadata record names. It loads the image descriptor at
/// heap entry 1 and stores its texel's green channel (1.0; 0.0 for entry 0) through the UAV.
const CS_HEAP: &[u32] = &[
    0xBE8A_0304, // s_mov_b32 s10, s4
    0xBE8B_0380, // s_mov_b32 s11, 0
    0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
    0xBE80_0306, // s_mov_b32 s0, s6
    0xBE81_0380, // s_mov_b32 s1, 0
    0xF40C_0400, 0xFA00_0020, // s_load_dwordx8 s[16:23], s[0:1], 0x20
    0x7E00_0280, // v_mov_b32 v0, 0
    0x7E02_0280, // v_mov_b32 v1, 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF000_1F08, 0x0004_0200, // image_load v[2:5], v[0:1], s[16:23] dmask:0xf dim:2D unorm
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_0000, 0x8003_0300, // buffer_store_dword v3, off, s[12:15], 0
    0xBF81_0000, // s_endpgm
];

/// A compute program that indexes the heap through an undeclared user SGPR is translated with
/// that SGPR as the heap, is not refused, and reads the entry it names.
#[test]
fn a_compute_shader_loads_a_texture_by_heap_index() {
    // A raw UAV (kind 3) in a table (root type 3) at user data 4, slot 0.
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
    let c = crate::pipeline::test_compute_container(CS_HEAP, &[uav], 7);
    let cs = shaders::translate(&c).unwrap();
    assert!(cs.heap);
    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    let heap = Heap::new(&t);
    // Entry 2: a raw buffer UAV of 64 dwords at `VA_A` (pool offset 0x2_0000).
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[4], view[8]) = (1, 64, 1);
    [view[10], view[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    let slot = heap._records.as_ptr() as u64 + 64;
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, slot);
    assert!(t.host.mirror(slot).unwrap());
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "heap") else { panic!("{:?}", t.shaders) };
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap.heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_table(0, &heap.heap, 2);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(f.get(0x2_0000, 2), [0x3F80_0000, super::addr_tests::CANARY]);
}

/// A compute program that takes its output buffer from the resource heap itself (entry 2, by
/// the heap's base in s6): no table names that entry. It stores the green channel of entry 1's
/// texel (1.0) at bytes 0 and 8.
const CS_HEAP_BUFFER: &[u32] = &[
    0xBE80_0306, // s_mov_b32 s0, s6
    0xBE81_0380, // s_mov_b32 s1, 0
    0xF408_0300, 0xFA00_0040, // s_load_dwordx4 s[12:15], s[0:1], 0x40
    0xF40C_0400, 0xFA00_0020, // s_load_dwordx8 s[16:23], s[0:1], 0x20
    0x7E00_0280, // v_mov_b32 v0, 0
    0x7E02_0280, // v_mov_b32 v1, 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF000_1F08, 0x0004_0200, // image_load v[2:5], v[0:1], s[16:23] dmask:0xf dim:2D unorm
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_0000, 0x8003_0300, // buffer_store_dword v3, off, s[12:15], 0
    0xE070_0008, 0x8003_0300, // buffer_store_dword v3, off, s[12:15], 8
    0xBF81_0000, // s_endpgm
];

/// A compute program that takes a buffer view from the resource heap itself reads the heap's
/// view-stride table (the translator's root SRV 255) before the entry: it runs, where its
/// dispatch used to be dropped ("root parameter 255 was never set"). While the entry holds no
/// buffer view the table holds 0 there and the program reads the host's null view (nothing is
/// stored); once the title writes one, the mirrored write brings the table's dword up to date
/// and the stores land.
#[test]
fn a_compute_shader_stores_through_a_buffer_view_it_takes_from_the_heap() {
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
    let c = crate::pipeline::test_compute_container(CS_HEAP_BUFFER, &[uav], 7);
    let cs = shaders::translate(&c).unwrap();
    assert!(cs.heap && cs.root_srvs == [victory_shader::emit::VIEW_STRIDES]);
    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    let heap = Heap::new(&t);
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "heap buffer") else { panic!("{:?}", t.shaders) };
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    let run = |t: &Title<Metal>| {
        let mut l = NativeList::default();
        l.set_heaps(Some(&heap.heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_table(0, &heap.heap, 0);
        l.dispatch(1, 1, 1).unwrap();
        super::run(&t.host, &[&l]);
        f.get(0x2_0000, 3)
    };
    let canary = super::addr_tests::CANARY;
    // Entry 2 is empty: stride 0, so the null view.
    assert_eq!(run(&t), [canary; 3]);
    assert_eq!((t.host.unset_roots(), t.host.failed()), (0, 0), "the dispatch ran");
    // Entry 2: a raw buffer UAV of 64 dwords at `VA_A` (pool offset 0x2_0000).
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[4], view[8]) = (1, 64, 1);
    [view[10], view[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    let slot = heap._records.as_ptr() as u64 + 64;
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, slot);
    assert!(t.host.mirror(slot).unwrap());
    assert_eq!(run(&t), [0x3F80_0000, canary, 0x3F80_0000]);
    assert_eq!((t.host.unset_roots(), t.host.unmapped(), t.host.failed()), (0, 0, 0));
}

/// A compute program asking a writable image for its size: a table at user data 4 holds the raw
/// output UAV (dwords 0-3) and a texture UAV (dwords 8-15), as the title's metadata declares them.
/// `image_get_resinfo` at level 0 (v0) returns width and height, which go to the output's dwords 0
/// and 1, and the height the descriptor's own bits hold (less 1) goes to dword 2; the program also
/// stores to the image, as the title's programs that query it do.
const CS_UAV_SIZE: &[u32] = &[
    0xBE8A_0304, // s_mov_b32 s10, s4
    0xBE8B_0380, // s_mov_b32 s11, 0
    0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
    0xF40C_0405, 0xFA00_0020, // s_load_dwordx8 s[16:23], s[10:11], 0x20
    0x7E00_0280, // v_mov_b32 v0, 0
    0x7E02_0280, // v_mov_b32 v1, 0
    0x7E0C_0280, // v_mov_b32 v6, 0
    0x7E0E_0280, // v_mov_b32 v7, 0
    0x7E10_0280, // v_mov_b32 v8, 0
    0x7E12_0280, // v_mov_b32 v9, 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF038_1308, 0x0004_0200, // image_get_resinfo v[2:3], v0, s[16:23] dmask:0x3 dim:2D unorm
    0xF020_1F08, 0x0004_0600, // image_store v[6:9], v[0:1], s[16:23] dmask:0xf dim:2D unorm
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_0000, 0x8003_0200, // buffer_store_dword v2, off, s[12:15], 0
    0xE070_0004, 0x8003_0300, // buffer_store_dword v3, off, s[12:15], 0 offset:4
    // Its height read from the descriptor's own bits (dword 2 bits 14-29: height - 1).
    0x9018_8E12, // s_lshr_b32 s24, s18, 14
    0x8718_FF18, 0x0000_3FFF, // s_and_b32 s24, s24, 0x3fff
    0x7E08_0218, // v_mov_b32 v4, s24
    0xE070_0008, 0x8003_0400, // buffer_store_dword v4, off, s[12:15], 0 offset:8
    0xBF81_0000, // s_endpgm
];

/// The size a program reads of a writable image is its own (16x8 here), not 0: a writable view has
/// one level, so the query is of level 0. (Under Wine, D3DMetal answered the query with no level
/// at all with 0, and the title's sun-shadow min/max program then lit or shadowed whole 16x16
/// tiles: halfwave50/VIctory-macOS-halfwave50 0f3ed38.)
#[test]
fn a_size_query_of_a_writable_image_reads_its_size() {
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
    // The texture UAV (kind 3) in the same table (root type 3), 8 dwords from its start, 8 dwords.
    let mut image = [0u8; 32];
    image[0] = 0x41;
    image[4..6].copy_from_slice(&[3, 3]);
    image[8] = 8;
    image[16] = 1;
    image[20..23].copy_from_slice(&[8, 4, 1]);
    let c = crate::pipeline::test_compute_container(CS_UAV_SIZE, &[uav, image], 7);
    let cs = shaders::translate(&c).unwrap();
    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    // Entry 0: a raw buffer UAV of 64 dwords at `VA_A`; entry 1: a 16x8 RGBA8 texture UAV.
    let records = vec![0u32; 8 * 2];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 2).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[4], view[8]) = (1, 64, 1);
    [view[10], view[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, start);
    assert!(t.host.mirror(start).unwrap());
    let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 16, 8);
    desc.unordered_access = true;
    let tex = t.host.texture(&desc).unwrap();
    let key = 0x4EA9_0100;
    t.host.register_texture(key, &tex);
    // DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_UAV_DIMENSION_TEXTURE2D, mip 0.
    let mut uav_desc = [0u32; 10];
    (uav_desc[0], uav_desc[1]) = (28, 4);
    crate::descriptor::create_uav(std::ptr::null(), key as _, std::ptr::null(), &uav_desc, start + 32);
    assert!(t.host.mirror(start + 32).unwrap());
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "uav size") else { panic!("{:?}", t.shaders) };
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    assert_eq!(f.get(0x2_0000, 3), [16, 8, 7], "width, height and height - 1 from the descriptor");
    drop(records);
}
