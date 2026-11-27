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

/// A compute program whose lane 0 subtracts 3 from the first dword of the raw UAV in its table
/// (`buffer_atomic_sub` with `glc`) and stores the value it returns, the old one, in the second.
const CS_ATOMIC_SUB: &[u32] = &[
    0x7DA4_0080, // v_cmpx_eq_u32 0, v0
    0xBE8A_0304, // s_mov_b32 s10, s4
    0xBE8B_0380, // s_mov_b32 s11, 0
    0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
    0x7E04_0283, // v_mov_b32 v2, 3
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE0CC_4000, 0x8003_0200, // buffer_atomic_sub v2, off, s[12:15], 0 glc
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_0004, 0x8003_0200, // buffer_store_dword v2, off, s[12:15], 0 offset:4
    0xBF81_0000, // s_endpgm
];

/// `buffer_atomic_sub` has no DXIL operation; it runs as an add of the negated data, which leaves
/// the same dword and returns the same old value, wrapping below zero as the console does.
#[test]
fn a_buffer_atomic_sub_leaves_the_difference_and_returns_the_old_value() {
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
    let c = crate::pipeline::test_compute_container(CS_ATOMIC_SUB, &[uav], 7);
    let cs = shaders::translate(&c).expect("buffer_atomic_sub translates");
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
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "atomic sub") else { panic!("{:?}", t.shaders) };
    for (before, after) in [(10u32, 7u32), (1, u32::MAX - 1)] {
        f.put(0x2_0000, &[before]);
        let mut l = NativeList::default();
        l.set_heaps(Some(&heap.heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_table(0, &heap.heap, 2);
        l.dispatch(1, 1, 1).unwrap();
        super::run(&t.host, &[&l]);
        assert_eq!(f.get(0x2_0000, 2), [after, before], "{before} - 3");
    }
}

/// A compute program of our own with a raw UAV in the table at user data 4: it loads the dword at
/// byte `at` of the view, swaps it for one more if it still holds what was loaded
/// (`buffer_atomic_cmpswap`, which returns what the dword held), and stores what it loaded at byte
/// 0 and what the swap returned at byte 4. The title's counted appends loop on that pair until
/// the swap returns what was loaded.
fn cs_load_then_swap(at: u32) -> Vec<u32> {
    vec![
        0xBE8A_0304, // s_mov_b32 s10, s4
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
        0x7E08_02FF, at, // v_mov_b32 v4, at
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_1000, 0x8003_0104, // buffer_load_dword v1, v4, s[12:15], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x4A0C_0281, // v_add_nc_u32 v6, 1, v1
        0x7E0E_0301, // v_mov_b32 v7, v1
        0xE0C4_5000, 0x8003_0604, // buffer_atomic_cmpswap v[6:7], v4, s[12:15], 0 offen glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_0000, 0x8003_0100, // buffer_store_dword v1, off, s[12:15], 0
        0xE070_0004, 0x8003_0600, // buffer_store_dword v6, off, s[12:15], 4
        0xBF81_0000, // s_endpgm
    ]
}

/// A load and a compare-and-swap of the same dword of a view agree on what it holds, inside the
/// view and past its end: a swap that returned something else than the load read would keep a
/// retry loop going for good (the GPU never finishes that command buffer). Guess: past the end of
/// a view the console reads 0 and its atomics write nothing and return 0, as its loads and stores
/// do.
#[test]
fn a_swap_agrees_with_a_load_of_the_same_dword_inside_a_view_and_past_its_end() {
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
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
    // Byte 0x10 is in the view (256 bytes), byte 0x110 is title memory after it.
    for (at, held, loaded, after) in [(0x10u32, 41u32, 41u32, 42u32), (0x110, 77, 0, 77)] {
        let c = crate::pipeline::test_compute_container(&cs_load_then_swap(at), &[uav], 7);
        let cs = shaders::translate(&c).unwrap();
        let Pipe::Compute(pso) = t.compute_pipe(&cs, "swap") else { panic!("{:?}", t.shaders) };
        f.put(0x2_0000, &[0u32, 0]);
        f.put(0x2_0000 + at as usize, &[held]);
        let mut l = NativeList::default();
        l.set_heaps(Some(&heap.heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_table(0, &heap.heap, 2);
        l.dispatch(1, 1, 1).unwrap();
        super::run(&t.host, &[&l]);
        let out = f.get(0x2_0000, 2);
        assert_eq!(out, [loaded, loaded], "byte {at:#x}: what the load read, what the swap returned");
        assert_eq!(f.get(0x2_0000 + at as usize, 1), [after], "byte {at:#x}: the dword after the swap");
    }
}

/// A compute program of our own, of the shape of the title's counted appends: every lane adds one
/// to the dword at byte 16 of the raw UAV in the table at user data 4, by loading it and swapping
/// it for one more until the swap returns what was loaded. A lane leaves the loop once the count
/// or its own tries pass 60 (the title's leave at a full count; the tries are there so that a
/// swap that never agrees with the load ends the test instead of hanging the GPU), and stores
/// its tries at byte 64 + 4 x lane.
const CS_APPEND: &[u32] = &[
        0xBE8A_0304, // s_mov_b32 s10, s4
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
        0x7E08_0290, // v_mov_b32 v4, 16
        0x7E12_0280, // v_mov_b32 v9, 0
        0xBE90_037E, // s_mov_b32 s16, exec_lo
        0xBE91_0380, // s_mov_b32 s17, 0
        0xBE92_037E, // s_mov_b32 s18, exec_lo
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_1000, 0x8003_0104, // buffer_load_dword v1, v4, s[12:15], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x2814_1301, // v_max_u32 v10, v1, v9
        0x7DA2_14BC, // v_cmpx_lt_u32 exec_lo, 60, v10
        0x8811_7E11, // s_or_b32 s17, s17, exec_lo
        0x8A10_7E10, // s_andn2_b32 s16, s16, exec_lo
        0xBF84_000A, // s_cbranch_scc0 exit
        0xBEFE_0310, // s_mov_b32 exec_lo, s16
        0x4A0C_0281, // v_add_nc_u32 v6, 1, v1
        0x7E0E_0301, // v_mov_b32 v7, v1
        0x4A12_1281, // v_add_nc_u32 v9, 1, v9
        0xE0C4_5000, 0x8003_0604, // buffer_atomic_cmpswap v[6:7], v4, s[12:15], 0 offen glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7D24_0D01, // v_cmpx_eq_i32 exec_lo, v1, v6
        0xBE90_4710, // s_andn2_wrexec_b32 s16, s16
        0xBF85_FFEE, // s_cbranch_scc1 loop
        0xBEFE_0312, // s_mov_b32 exec_lo, s18
        0x340A_0082, // v_lshlrev_b32 v5, 2, v0
        0xE070_1040, 0x8003_0905, // buffer_store_dword v9, v5, s[12:15], 0 offen offset:0x40
        0xBF81_0000, // s_endpgm
];

/// Lanes that append to one count at once all get in: each try, the lane whose swap finds what it
/// loaded wins and the others load again. The count is the lanes, and no lane runs out of tries.
#[test]
fn lanes_that_swap_one_count_at_once_all_append() {
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
    let c = crate::pipeline::test_compute_container(CS_APPEND, &[uav], 7);
    let cs = shaders::translate(&c).unwrap();
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
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "append") else { panic!("{:?}", t.shaders) };
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    f.put(0x2_0000, &[0u32; 64]);
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap.heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_table(0, &heap.heap, 2);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    let tries = f.get(0x2_0000 + 64, 32);
    assert_eq!(f.get(0x2_0000 + 16, 1), [32], "the count; tries a lane: {tries:?}");
    assert!(tries.iter().all(|n| (1..=32).contains(n)), "{tries:?}");
}

/// [`CS_APPEND`] in 64-lane waves, as the title's is: lane `i` appends to the count at dword
/// `i & 7`, and stores its tries at byte 256 + 4 x lane.
const CS_APPEND_64: &[u32] = &[
    0xBE8A_0304, // s_mov_b32 s10, s4
    0xBE8B_0380, // s_mov_b32 s11, 0
    0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
    0x3608_0087, // v_and_b32 v4, 7, v0
    0x3408_0882, // v_lshlrev_b32 v4, 2, v4
    0x7E12_0280, // v_mov_b32 v9, 0
    0xBE90_047E, // s_mov_b64 s[16:17], exec
    0xBE92_0480, // s_mov_b64 s[18:19], 0
    0xBE94_047E, // s_mov_b64 s[20:21], exec
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE030_1000, 0x8003_0104, // buffer_load_dword v1, v4, s[12:15], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x2814_1301, // v_max_u32 v10, v1, v9
    0x7DA2_14BC, // v_cmpx_lt_u32 exec, 60, v10
    0x8892_7E12, // s_or_b64 s[18:19], s[18:19], exec
    0x8A90_7E10, // s_andn2_b64 s[16:17], s[16:17], exec
    0xBF84_000A, // s_cbranch_scc0 exit
    0xBEFE_0410, // s_mov_b64 exec, s[16:17]
    0x4A0C_0281, // v_add_nc_u32 v6, 1, v1
    0x7E0E_0301, // v_mov_b32 v7, v1
    0x4A12_1281, // v_add_nc_u32 v9, 1, v9
    0xE0C4_5000, 0x8003_0604, // buffer_atomic_cmpswap v[6:7], v4, s[12:15], 0 offen glc
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7D24_0D01, // v_cmpx_eq_i32 exec, v1, v6
    0xBE90_3A10, // s_andn2_wrexec_b64 s[16:17], s[16:17]
    0xBF85_FFEE, // s_cbranch_scc1 loop
    0xBEFE_0414, // s_mov_b64 exec, s[20:21]
    0x340A_0082, // v_lshlrev_b32 v5, 2, v0
    0xE070_1100, 0x8003_0905, // buffer_store_dword v9, v5, s[12:15], 0 offen offset:0x100
    0xBF81_0000, // s_endpgm
];

/// The same in the title's shape: 256 lanes a group in 64-lane waves, which the host runs as
/// eight waves of 32. Each of the eight counts is appended to by 32 lanes, eight from each
/// console wave.
#[test]
fn lanes_of_split_waves_that_swap_counts_at_once_all_append() {
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
    let c = crate::pipeline::test_compute_container_of(CS_APPEND_64, &[uav], 7, 256, true);
    let cs = shaders::translate(&c).unwrap();
    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    let heap = Heap::new(&t);
    // Entry 2: a raw buffer UAV of 512 dwords at `VA_A` (pool offset 0x2_0000).
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[4], view[8]) = (1, 512, 1);
    [view[10], view[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    let slot = heap._records.as_ptr() as u64 + 64;
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, slot);
    assert!(t.host.mirror(slot).unwrap());
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "append 64") else { panic!("{:?}", t.shaders) };
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    f.put(0x2_0000, &[0u32; 512]);
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap.heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_table(0, &heap.heap, 2);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    let tries = f.get(0x2_0000 + 256, 256);
    assert_eq!(f.get(0x2_0000, 8), [32; 8], "the counts; tries a lane: {tries:?}");
    assert!(tries.iter().all(|n| (1..=32).contains(n)), "{tries:?}");
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

/// #264 review: the title zeroes a heap record with plain stores (no device call). The record is
/// in a bound table and a view was mirrored there, so the next submission mirrors it: its
/// view-stride dword goes to 0 and the shader reads the null view, not the old buffer.
#[test]
fn a_heap_record_the_title_zeroes_with_plain_stores_reads_the_null_view_at_the_next_dispatch() {
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
    let c = crate::pipeline::test_compute_container(CS_HEAP_BUFFER, &[uav], 7);
    let cs = shaders::translate(&c).unwrap();
    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    let heap = Heap::new(&t);
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "heap buffer") else { panic!("{:?}", t.shaders) };
    let canary = super::addr_tests::CANARY;
    let run = |t: &Title<Metal>| {
        f.put(0x2_0000, &[canary; 3]);
        let mut l = NativeList::default();
        l.set_heaps(Some(&heap.heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_table(0, &heap.heap, 0);
        l.dispatch(1, 1, 1).unwrap();
        super::run(&t.host, &[&l]);
        f.get(0x2_0000, 3)
    };
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[4], view[8]) = (1, 64, 1);
    [view[10], view[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    let slot = heap._records.as_ptr() as u64 + 64;
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, slot);
    assert!(t.host.mirror(slot).unwrap());
    let stored = [0x3F80_0000, canary, 0x3F80_0000];
    assert_eq!(run(&t), stored);
    // The title zeroes entry 2's record with plain stores.
    // SAFETY: entry 2's 32 bytes inside `heap._records`, which the heap's records are.
    unsafe { std::ptr::write_bytes(slot as *mut u8, 0, 32) };
    assert!(crate::descriptor::view(slot).is_none(), "an empty record");
    // Seen at the dispatch: the stride is 0 and the stores go to the null view.
    assert_eq!(run(&t), [canary; 3], "the zeroed record's old view");
    assert!(!heap.heap.mirrored.borrow().contains_key(&2), "nothing mirrored there any more");
    assert_eq!(run(&t), [canary; 3], "a second dispatch");
    // A view written there again with plain stores is mirrored again.
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, slot);
    assert_eq!(run(&t), stored, "a view written there again");
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

/// `buffer_store_format_xyzw` and `buffer_load_format_xyzw` with `idxen` through a 16-byte
/// structured writable buffer view, an `R32G32B32A32_UINT` typed writable buffer view, an
/// `R32G32B32A32_FLOAT` view, and an `R8G8B8A8_UNORM` view write and read back through the same
/// heap entry that `buffer_store_dwordx4` writes (`#255`).
#[test]
fn formatted_buffer_stores_and_loads_through_structured_and_typed_uavs() {
    // Table at s4: entry 0 (dwords 0..4) is the tested UAV, entry 1 (dwords 4..8) is a raw output UAV.
    // Lane 0 stores `[v1..v4]` to element 1 of entry 0 with `buffer_store_format_xyzw ... idxen`,
    // loads element 1 back with `buffer_load_format_xyzw ... idxen` into `v[6:9]`, and writes
    // `[v6..v9, s14 (num_records)]` to entry 1 via `buffer_store_dwordx4` + `buffer_store_dword`.
    const CS_FMT_STORE: &[u32] = &[
        0xBE8A_0304, // s_mov_b32 s10, s4
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
        0xF408_0405, 0xFA00_0020, // s_load_dwordx4 s[16:19], s[10:11], 0x20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E00_0281, // v_mov_b32 v0, 1 (element index 1)
        0xE038_0800, 0x8004_0100, // buffer_load_dwordx4 v[1:4], off, s[16:19], 0 offset:2048
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE01C_2000, 0x8003_0100, // buffer_store_format_xyzw v[1:4], v0, s[12:15], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE00C_2000, 0x8003_0600, // buffer_load_format_xyzw v[6:9], v0, s[12:15], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_0000, 0x8004_0600, // buffer_store_dwordx4 v[6:9], off, s[16:19], 0
        0x7E14_020E, // v_mov_b32 v10, s14 (getDimensions)
        0xE070_0010, 0x8004_0A00, // buffer_store_dword v10, off, s[16:19], 0 offset:16
        0xBF81_0000, // s_endpgm
    ];
    // Raw `buffer_store_dwordx4` through the same entry 0 descriptor (`gpuVA` in word[0]).
    const CS_RAW_STORE: &[u32] = &[
        0xBE8A_0304, // s_mov_b32 s10, s4
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
        0xF408_0405, 0xFA00_0020, // s_load_dwordx4 s[16:19], s[10:11], 0x20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE038_0800, 0x8004_0100, // buffer_load_dwordx4 v[1:4], off, s[16:19], 0 offset:2048
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_0020, 0x8003_0100, // buffer_store_dwordx4 v[1:4], off, s[12:15], 0 offset:32
        0xBF81_0000, // s_endpgm
    ];

    let mut uav0 = [0u8; 32];
    uav0[0] = 0x41;
    uav0[4..6].copy_from_slice(&[3, 3]);
    uav0[16] = 1;
    uav0[20..23].copy_from_slice(&[4, 4, 0]);
    let mut uav1 = uav0;
    uav1[8] = 8; // +8 dwords = +32 bytes (entry 1)
    let c_fmt = crate::pipeline::test_compute_container(CS_FMT_STORE, &[uav0, uav1], 7);
    let cs_fmt = shaders::translate(&c_fmt).unwrap();
    let c_raw = crate::pipeline::test_compute_container(CS_RAW_STORE, &[uav0, uav1], 7);
    let cs_raw = shaders::translate(&c_raw).unwrap();
    let mut img_uav = uav1;
    img_uav[20..23].copy_from_slice(&[8, 4, 1]);
    let c_img = crate::pipeline::test_compute_container(CS_UAV_SIZE, &[uav0, img_uav], 7);
    let cs_img = shaders::translate(&c_img).unwrap();
    assert!(
        crate::native::list::Binding::of(&cs_fmt).typed_buffers,
        "bufferLoad/bufferStore marks TypedBuffer use"
    );
    assert!(
        !crate::native::list::Binding::of(&cs_raw).typed_buffers,
        "rawBufferLoad/rawBufferStore does not mark TypedBuffer use"
    );
    assert!(
        !crate::native::list::Binding::of(&cs_img).typed_buffers,
        "Texture2D getDimensions does not mark TypedBuffer use"
    );

    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    let records = vec![0u32; 8 * 20];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 20).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1; // BUFFER

    // Entry 1: raw output/input UAV of 1024 dwords (4096 B) at `VA_A + 0x1000`.
    let mut raw_out = [0u32; 16];
    (raw_out[1], raw_out[4], raw_out[8]) = (1, 1024, 1);
    [raw_out[10], raw_out[11]] = super::addr_tests::halves(super::addr_tests::VA_A + 0x1000);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &raw_out, start + 32);
    assert!(t.host.mirror(start + 32).unwrap());

    // Entry 19 (past the 16-entry default table check): an unbound 16-byte structured UAV must
    // never materialize a Metal texture view.
    let mut unbound_struct = [0u32; 16];
    (unbound_struct[1], unbound_struct[4], unbound_struct[5]) = (1, 8, 16);
    [unbound_struct[10], unbound_struct[11]] = super::addr_tests::halves(super::addr_tests::VA_A + 0x2000);
    crate::descriptor::create_placed_uav(
        std::ptr::null_mut(),
        &resource,
        &unbound_struct,
        start + 19 * 32,
    );
    assert!(t.host.mirror(start + 19 * 32).unwrap());
    assert_eq!(t.host.typed_views(), 0, "mirror() makes no Metal texture_buffer views");

    let Pipe::Compute(pso_fmt) = t.compute_pipe(&cs_fmt, "fmt store") else {
        panic!("{:?}", t.shaders)
    };
    let Pipe::Compute(pso_raw) = t.compute_pipe(&cs_raw, "raw store") else {
        panic!("{:?}", t.shaders)
    };

    let bits = [0x0000_0001u32, 0xDEAD_BEEF, 0x7F80_0001, 0x3F80_0000];
    for (idx, (label, fmt, stride)) in [
        ("structured 16B", 0u32, 16u32),
        ("R32G32B32A32_UINT", 3, 0),
        ("R32G32B32A32_FLOAT", 2, 0),
    ]
    .into_iter()
    .enumerate()
    {
        f.put(0x2_0000, &[0u32; 16]);
        f.put(0x2_1000, &[0u32; 8]);
        f.put(0x2_1800, &bits);
        let mut view = [0u32; 16];
        (view[0], view[1], view[4], view[5]) = (fmt, 1, 8, stride);
        [view[10], view[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
        crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, start);
        assert!(t.host.mirror(start).unwrap());
        let views_before = t.host.typed_views();
        assert!(heap.typed_pending.borrow().contains(0));

        // Running `pso_raw` first must not materialize `texture_buffer` views.
        let mut l_raw = NativeList::default();
        l_raw.set_heaps(Some(&heap), None);
        l_raw.set_compute_pipeline(&pso_raw);
        l_raw.set_compute_root_table(0, &heap, 0);
        l_raw.dispatch(1, 1, 1).unwrap();
        super::run(&t.host, &[&l_raw]);
        assert_eq!(
            t.host.typed_views(),
            views_before,
            "{label}: raw shader leaves typed_pending unmaterialized"
        );
        assert!(heap.typed_pending.borrow().contains(0));

        let mut l = NativeList::default();
        l.set_heaps(Some(&heap), None);
        l.set_compute_pipeline(&pso_fmt);
        l.set_compute_root_table(0, &heap, 0);
        l.dispatch(1, 1, 1).unwrap();
        super::run(&t.host, &[&l]);
        assert_eq!(
            t.host.typed_views(),
            views_before + 1,
            "{label}: formatted shader materializes only the bound candidate (case {idx})"
        );
        assert_eq!(
            t.host.live_typed_views(),
            1,
            "{label}: overwriting slot 0 drops its previous view so only 1 stays live"
        );
        assert!(!heap.typed_pending.borrow().contains(0));
        assert!(
            heap.typed_pending.borrow().contains(19),
            "{label}: unbound structured UAV outside the table stays unmaterialized"
        );
        assert_eq!(f.get(0x2_0000, 4), [0; 4], "{label}: element 0 untouched");
        assert_eq!(f.get(0x2_0010, 4), bits, "{label}: element 1 written by buffer_store_format_xyzw");
        assert_eq!(f.get(0x2_0020, 4), bits, "{label}: element 2 written by buffer_store_dwordx4");
        assert_eq!(
            f.get(0x2_1000, 5),
            [bits[0], bits[1], bits[2], bits[3], 8],
            "{label}: buffer_load_format_xyzw + getDimensions"
        );
    }

    // Format-converting typed UAVs: `R8G8B8A8_UNORM` (28) and `R16G16B16A16_FLOAT` (10).
    f.put(0x2_0000, &[0u32; 16]);
    f.put(0x2_1000, &[0u32; 8]);
    let unorm_in = [0.0f32, 1.0, 1.0, 0.0].map(f32::to_bits);
    f.put(0x2_1800, &unorm_in);
    let mut view_unorm = [0u32; 16];
    (view_unorm[0], view_unorm[1], view_unorm[4]) = (28, 1, 8);
    [view_unorm[10], view_unorm[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view_unorm, start);
    assert!(t.host.mirror(start).unwrap());
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso_fmt);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(f.get(0x2_0000, 2), [0, 0x00FF_FF00], "R8G8B8A8_UNORM packed texel 1");
    assert_eq!(
        f.get(0x2_1000, 5),
        [unorm_in[0], unorm_in[1], unorm_in[2], unorm_in[3], 8],
        "R8G8B8A8_UNORM readback + getDimensions"
    );

    f.put(0x2_0000, &[0u32; 16]);
    f.put(0x2_1000, &[0u32; 8]);
    let f16_in = [1.0f32, -2.0, 0.5, 4.0].map(f32::to_bits);
    f.put(0x2_1800, &f16_in);
    let mut view_f16 = [0u32; 16];
    (view_f16[0], view_f16[1], view_f16[4]) = (10, 1, 8);
    [view_f16[10], view_f16[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view_f16, start);
    assert!(t.host.mirror(start).unwrap());
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso_fmt);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(
        f.get(0x2_0000, 4),
        [0, 0, 0xC000_3C00, 0x4400_3800],
        "R16G16B16A16_FLOAT packed texel 1"
    );
    assert_eq!(
        f.get(0x2_1000, 5),
        [f16_in[0], f16_in[1], f16_in[2], f16_in[3], 8],
        "R16G16B16A16_FLOAT readback + getDimensions"
    );

    f.put(0x2_0000, &[0u32; 16]);
    f.put(0x2_1000, &[0u32; 8]);
    let u16_in = [0x0000_0012u32, 0x0000_3456, 0x0000_789A, 0x0000_BCDE];
    f.put(0x2_1800, &u16_in);
    let mut view_u16 = [0u32; 16];
    (view_u16[0], view_u16[1], view_u16[4]) = (12, 1, 8);
    [view_u16[10], view_u16[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view_u16, start);
    assert!(t.host.mirror(start).unwrap());
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso_fmt);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(
        f.get(0x2_0000, 4),
        [0, 0, 0x3456_0012, 0xBCDE_789A],
        "R16G16B16A16_UINT packed texel 1"
    );
    assert_eq!(
        f.get(0x2_1000, 5),
        [u16_in[0], u16_in[1], u16_in[2], u16_in[3], 8],
        "R16G16B16A16_UINT readback + getDimensions"
    );

    // 1-channel signed 32-bit integer `R32_SINT` (43 -> `R32Uint`) and 2-channel `R16G16_SNORM` (37).
    f.put(0x2_0000, &[0u32; 16]);
    f.put(0x2_1000, &[0u32; 8]);
    f.put(0x2_1800, &[0xDEAD_BEEFu32, 0x1111_2222, 0x3333_4444, 0x5555_6666]);
    let mut view_i32 = [0u32; 16];
    (view_i32[0], view_i32[1], view_i32[4]) = (43, 1, 8);
    [view_i32[10], view_i32[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view_i32, start);
    assert!(t.host.mirror(start).unwrap());
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso_fmt);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(f.get(0x2_0000, 3), [0, 0xDEAD_BEEF, 0], "R32_SINT single-channel texel 1");
    let i32_out = f.get(0x2_1000, 5);
    assert_eq!((i32_out[0], i32_out[4]), (0xDEAD_BEEF, 8), "R32_SINT negative i32 + getDimensions");

    f.put(0x2_0000, &[0u32; 16]);
    f.put(0x2_1000, &[0u32; 8]);
    let snorm_in = [-1.0f32, 1.0, 0.5, -0.5].map(f32::to_bits);
    f.put(0x2_1800, &snorm_in);
    let mut view_snorm = [0u32; 16];
    (view_snorm[0], view_snorm[1], view_snorm[4]) = (37, 1, 8);
    [view_snorm[10], view_snorm[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view_snorm, start);
    assert!(t.host.mirror(start).unwrap());
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso_fmt);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(f.get(0x2_0000, 3), [0, 0x7FFF_8001, 0], "R16G16_SNORM packed [-1.0, 1.0] texel 1");
    let snorm_out = f.get(0x2_1000, 5);
    assert_eq!(
        (snorm_out[0], snorm_out[1], snorm_out[4]),
        ((-1.0f32).to_bits(), 1.0f32.to_bits(), 8),
        "R16G16_SNORM readback + getDimensions"
    );
    assert_eq!(t.host.typed_views(), 8, "8 promotions across the 8 mirrored descriptors");
    assert_eq!(t.host.live_typed_views(), 1, "only 1 entry holds a typed view at the end");

    // A typed view over a committed buffer (host memory no pool holds) is made over the host's
    // wrap of that buffer ([`NativeHost::resolve`]), at the right offset into it.
    let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
    // SAFETY: non-zero size; freed below, after the GPU is done with it.
    let mem = unsafe { std::alloc::alloc_zeroed(layout) };
    let at = mem as u64 + 0x100;
    t.host.register_own_buffer(0xC0FFEE, mem as u64, layout.size() as u64);
    f.put(0x2_1000, &[0u32; 8]);
    f.put(0x2_1800, &bits);
    let mut view_own = [0u32; 16];
    (view_own[0], view_own[1], view_own[4]) = (3, 1, 8);
    [view_own[10], view_own[11]] = super::addr_tests::halves(at);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view_own, start);
    assert!(t.host.mirror(start).unwrap());
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso_fmt);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(t.host.typed_views(), 9, "promoted over the committed buffer");
    assert_eq!(t.host.unmapped(), 0);
    // SAFETY: inside the allocation; read after the run waited for the GPU.
    let own = unsafe { std::slice::from_raw_parts(at as *const u32, 8) };
    assert_eq!((&own[..4], &own[4..8]), (&[0; 4][..], &bits[..]), "texel 1 in committed memory");
    assert_eq!(f.get(0x2_1000, 5), [bits[0], bits[1], bits[2], bits[3], 8], "committed: readback");
    t.host.release_resource(0xC0FFEE);
    // SAFETY: allocated above with this layout; the GPU is idle.
    unsafe { std::alloc::dealloc(mem, layout) };
    drop(records);
}

/// If `Binding`'s DXIL scan ever missed a `TypedBuffer` operation (`typed_buffers == false` on
/// purpose), `NativeHost::mirror` wrote `[gpu, 0, size]` (`textureViewID == 0`), so both SRV and
/// UAV `texture_buffer` loads and `getDimensions` read zero cleanly on Metal without faulting —
/// both on a fresh heap slot and after overwriting a previously promoted slot.
#[test]
fn wrong_typed_buffer_scan_reads_zero_without_crashing() {
    // Entry 0 (dwords 0..4) is a read-only SRV `Buffer<float4>`, entry 1 (dwords 8..12) is a raw output UAV.
    const CS_FMT_SRV_LOAD: &[u32] = &[
        0xBE8A_0304, // s_mov_b32 s10, s4
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
        0xF408_0405, 0xFA00_0020, // s_load_dwordx4 s[16:19], s[10:11], 0x20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E00_0281, // v_mov_b32 v0, 1 (element index 1)
        0xE00C_2000, 0x8003_0600, // buffer_load_format_xyzw v[6:9], v0, s[12:15], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_0000, 0x8004_0600, // buffer_store_dwordx4 v[6:9], off, s[16:19], 0
        0x7E14_020E, // v_mov_b32 v10, s14 (getDimensions)
        0xE070_0010, 0x8004_0A00, // buffer_store_dword v10, off, s[16:19], 0 offset:16
        0xBF81_0000, // s_endpgm
    ];
    // Entry 0 is a read-write UAV `RWBuffer<float4>`; stores to element 1, loads element 1, and queries size.
    const CS_FMT_UAV_RW: &[u32] = &[
        0xBE8A_0304, // s_mov_b32 s10, s4
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
        0xF408_0405, 0xFA00_0020, // s_load_dwordx4 s[16:19], s[10:11], 0x20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E00_0281, // v_mov_b32 v0, 1 (element index 1)
        0xE038_0800, 0x8004_0100, // buffer_load_dwordx4 v[1:4], off, s[16:19], 0 offset:2048
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE01C_2000, 0x8003_0100, // buffer_store_format_xyzw v[1:4], v0, s[12:15], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE00C_2000, 0x8003_0600, // buffer_load_format_xyzw v[6:9], v0, s[12:15], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_0000, 0x8004_0600, // buffer_store_dwordx4 v[6:9], off, s[16:19], 0
        0x7E14_020E, // v_mov_b32 v10, s14 (getDimensions)
        0xE070_0010, 0x8004_0A00, // buffer_store_dword v10, off, s[16:19], 0 offset:16
        0xBF81_0000, // s_endpgm
    ];

    let mut srv0 = [0u8; 32];
    srv0[0] = 0x41;
    srv0[4..6].copy_from_slice(&[1, 3]); // SRV in table
    srv0[16] = 1;
    srv0[20..23].copy_from_slice(&[4, 4, 0]);
    let mut uav0 = srv0;
    uav0[4] = 3; // UAV in table
    let mut uav1 = uav0;
    uav1[8] = 8; // +8 dwords (entry 1)

    let cs_srv = shaders::translate(&crate::pipeline::test_compute_container(
        CS_FMT_SRV_LOAD,
        &[srv0, uav1],
        7,
    ))
    .unwrap();
    let cs_uav = shaders::translate(&crate::pipeline::test_compute_container(
        CS_FMT_UAV_RW,
        &[uav0, uav1],
        7,
    ))
    .unwrap();
    assert!(crate::native::list::Binding::of(&cs_srv).typed_buffers);
    assert!(crate::native::list::Binding::of(&cs_uav).typed_buffers);

    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    let records = vec![0u32; 8 * 4];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 4).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1; // BUFFER

    let mut raw_out = [0u32; 16];
    (raw_out[1], raw_out[4], raw_out[8]) = (1, 1024, 1);
    [raw_out[10], raw_out[11]] = super::addr_tests::halves(super::addr_tests::VA_A + 0x1000);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &raw_out, start + 32);
    assert!(t.host.mirror(start + 32).unwrap());

    let make_wrong_pso = |t: &Title<Metal>, cs: &HostShader, name: &str| {
        let mut b = crate::native::list::Binding {
            name: name.into(),
            ..crate::native::list::Binding::of(cs)
        };
        // Force the scan flags wrong on purpose so `materialize_typed_buffers` skips promotion.
        b.typed_buffers = false;
        b.heap_typed_buffers = false;
        let threads = shaders::threads_of(&cs.dxil).unwrap();
        let pso = victory_backend::backend::ComputeBackend::create_compute_pipeline_with(
            &t.host.backend,
            &cs.dxil,
            b.layout().as_ref().unwrap(),
        )
        .unwrap();
        Rc::new(crate::native::list::ComputePso {
            pso,
            threads,
            binding: Some(b),
        })
    };
    let Pipe::Compute(pso_srv_right) = t.compute_pipe(&cs_srv, "srv right") else {
        panic!("{:?}", t.shaders)
    };
    let pso_srv_wrong = make_wrong_pso(&t, &cs_srv, "srv wrong");
    let pso_uav_wrong = make_wrong_pso(&t, &cs_uav, "uav wrong");

    let elem0 = [0x1111_1111u32, 0x2222_2222, 0x3333_3333, 0x4444_4444];
    let elem1 = [1.5f32, -2.0, 0.5, 4.0].map(f32::to_bits);
    f.put(0x2_0000, &elem0);
    f.put(0x2_0010, &elem1);
    f.put(0x2_1000, &[0xCAFE_BABEu32; 8]);

    // 1. Fresh SRV slot (`mirror()` wrote `[gpu, 0, size]`): wrong scan reads zeros, no crash.
    let mut srv_desc = [0u32; 16];
    (srv_desc[0], srv_desc[1], srv_desc[6]) = (2, 1, 8); // RGBA32_FLOAT, BUFFER, 8 elements
    [srv_desc[10], srv_desc[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, &srv_desc, start);
    assert!(t.host.mirror(start).unwrap());
    assert_eq!(t.host.live_typed_views(), 0);

    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso_srv_wrong);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(t.host.typed_views(), 0);
    assert_eq!(t.host.live_typed_views(), 0);
    assert_eq!(
        f.get(0x2_1000, 5),
        [0, 0, 0, 0, 0],
        "fresh SRV with forced-false scan reads [0, 0, 0, 0] and width 0 over canary"
    );

    // 2. Promote the SRV slot with the real scan (`live_typed_views == 1`), verify non-zero read,
    // then overwrite slot 0 via `mirror()` (`live_typed_views == 0`) and run the wrong-scan UAV
    // shader: store is dropped cleanly, load and `getDimensions` read `[0, 0, 0, 0, 0]`.
    let mut l_ok = NativeList::default();
    l_ok.set_heaps(Some(&heap), None);
    l_ok.set_compute_pipeline(&pso_srv_right);
    l_ok.set_compute_root_table(0, &heap, 0);
    l_ok.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l_ok]);
    assert_eq!(t.host.typed_views(), 1);
    assert_eq!(t.host.live_typed_views(), 1);
    assert_eq!(f.get(0x2_1000, 5), [elem1[0], elem1[1], elem1[2], elem1[3], 8]);

    f.put(0x2_1000, &[0xCAFE_BABEu32; 8]);
    f.put(0x2_1800, &[0xDEAD_0001u32, 0xDEAD_0002, 0xDEAD_0003, 0xDEAD_0004]);
    let mut uav_desc = [0u32; 16];
    (uav_desc[0], uav_desc[1], uav_desc[4]) = (2, 1, 8); // RGBA32_FLOAT UAV, 8 elements
    [uav_desc[10], uav_desc[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav_desc, start);
    assert!(t.host.mirror(start).unwrap());
    assert_eq!(
        t.host.live_typed_views(),
        0,
        "re-mirroring slot 0 drops its previous texture_buffer view"
    );

    let mut l_uav = NativeList::default();
    l_uav.set_heaps(Some(&heap), None);
    l_uav.set_compute_pipeline(&pso_uav_wrong);
    l_uav.set_compute_root_table(0, &heap, 0);
    l_uav.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l_uav]);
    assert_eq!(t.host.typed_views(), 1, "wrong-scan UAV did not promote slot 0");
    assert_eq!(t.host.live_typed_views(), 0);
    assert_eq!(
        f.get(0x2_0010, 4),
        elem1,
        "null texture_buffer UAV store leaves title memory untouched"
    );
    assert_eq!(
        f.get(0x2_1000, 5),
        [0, 0, 0, 0, 0],
        "overwritten UAV with forced-false scan reads [0, 0, 0, 0] and width 0"
    );
    drop(records);
}

#[test]
#[ignore = "benchmark: cargo test -p victory-gpu world_heap_mirror_lazy_benchmark -- --ignored --nocapture"]
fn world_heap_mirror_lazy_benchmark() {
    fn phys_footprint() -> usize {
        const TASK_VM_INFO: i32 = 22;
        unsafe extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(task: u32, flavor: i32, info: *mut u64, count: *mut u32) -> i32;
        }
        let mut info = [0u64; 64];
        let mut count = 128u32;
        // SAFETY: local buffer of `count` 32-bit words for `TASK_VM_INFO`.
        let kr = unsafe { task_info(mach_task_self(), TASK_VM_INFO, info.as_mut_ptr(), &mut count) };
        if kr == 0 && count >= 38 { info[18] as usize } else { 0 }
    }

    let Some(t) = title() else { return };
    let _f = super::addr_tests::Fixture::new(&t);
    const TOTAL: usize = 40_000;
    let records = vec![0u32; 8 * TOTAL];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, TOTAL).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1; // BUFFER

    let mut raw_view = [0u32; 16];
    (raw_view[1], raw_view[4], raw_view[8]) = (1, 16, 1); // FLAG_RAW (`main` baseline: 0 pending bits)
    [raw_view[10], raw_view[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    for i in 0..TOTAL {
        crate::descriptor::create_placed_uav(
            std::ptr::null_mut(),
            &resource,
            &raw_view,
            start + (i as u64) * 32,
        );
    }
    assert!(t.host.mirror(start).unwrap());
    let fp_base0 = phys_footprint();
    let t_base = std::time::Instant::now();
    for i in 0..TOTAL {
        assert!(t.host.mirror(start + (i as u64) * 32).unwrap());
    }
    let base_ms = t_base.elapsed().as_secs_f64() * 1000.0;
    let fp_base1 = phys_footprint();
    println!(
        "baseline NativeHost::mirror 40,000 raw entries: {base_ms:.2} ms ({:.2} us/entry), phys_footprint = {:.2} MiB (cold delta {:.1} KiB)",
        base_ms * 1000.0 / TOTAL as f64,
        fp_base1 as f64 / (1024.0 * 1024.0),
        (fp_base1 as isize - fp_base0 as isize) as f64 / 1024.0,
    );

    for i in 0..TOTAL {
        let mut view = [0u32; 16];
        match i % 4 {
            0 => (view[0], view[1], view[4], view[5]) = (0, 1, 16, 16), // structured 16B
            1 => (view[0], view[1], view[4], view[5]) = (0, 1, 16, 8),  // structured 8B
            2 => (view[0], view[1], view[4], view[5]) = (0, 1, 16, 4),  // structured 4B
            _ => (view[0], view[1], view[4], view[5]) = (2, 1, 16, 0),  // typed RGBA32_FLOAT
        }
        [view[10], view[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
        crate::descriptor::create_placed_uav(
            std::ptr::null_mut(),
            &resource,
            &view,
            start + (i as u64) * 32,
        );
    }

    let fp_before = phys_footprint();
    let t0 = std::time::Instant::now();
    for i in 0..TOTAL {
        assert!(t.host.mirror(start + (i as u64) * 32).unwrap());
    }
    let mirror_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let fp_after = phys_footprint();
    println!(
        "lazy NativeHost::mirror 40,000 structured+typed entries: {mirror_ms:.2} ms ({:.2} us/entry), typed_views = {}, pending = {}, phys_footprint delta over raw = {:.1} KiB",
        mirror_ms * 1000.0 / TOTAL as f64,
        t.host.typed_views(),
        heap.typed_pending.borrow().count(),
        (fp_after as isize - fp_before as isize) as f64 / 1024.0,
    );
    assert_eq!(t.host.typed_views(), 0);
    assert_eq!(heap.typed_pending.borrow().count(), TOTAL);
    drop(records);
}

/// #264 review (cost): what a resource heap as long as `HEAP_VIEWS` and its view-stride table add
/// to the process footprint and to Metal's allocated size, beside a heap of 40,000 title entries
/// plus the reserved pairs (its length before #264).
#[test]
#[ignore = "benchmark: cargo test -p victory-gpu long_resource_heap_cost_benchmark -- --ignored --nocapture"]
fn long_resource_heap_cost_benchmark() {
    use victory_backend::backend::ComputeBackend;
    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    let mb = |b: usize| b as f64 / (1 << 20) as f64;
    let measure = |t: &Title<Metal>| (victory_platform::mem::footprint(), t.host.backend.allocated_bytes() as usize);
    let report = |what: &str, a: (usize, usize), b: (usize, usize)| {
        println!("{what}: footprint {:+.1} MiB, Metal allocated {:+.1} MiB", mb(b.0) - mb(a.0), mb(b.1) - mb(a.1));
    };
    let m0 = measure(&t);
    let short = t.host.backend.create_heap(HeapKind::Resource, 40_000 + crate::native::addr::RESERVED).unwrap();
    let m1 = measure(&t);
    report("heap of 40,000 + reserved entries (before #264)", m0, m1);
    let long = t.host.backend.create_heap(HeapKind::Resource, victory_shader::emit::HEAP_VIEWS as usize).unwrap();
    let m2 = measure(&t);
    report("heap of HEAP_VIEWS entries (#264)", m1, m2);
    // A host heap (HEAP_VIEWS long) bound to a shader that reads the stride table: the table is
    // made, the null views written and the heap made resident for the dispatch.
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
    let c = crate::pipeline::test_compute_container(CS_HEAP_BUFFER, &[uav], 7);
    let cs = shaders::translate(&c).unwrap();
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "heap buffer") else { panic!("{:?}", t.shaders) };
    let dispatch = |t: &Title<Metal>, heap: &Heap| {
        let mut l = NativeList::default();
        l.set_heaps(Some(&heap.heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_table(0, &heap.heap, 0);
        l.dispatch(1, 1, 1).unwrap();
        super::run(&t.host, &[&l]);
    };
    // The first dispatch also pays for the device's first command buffer and the fixture's pool
    // becoming resident: measured apart, on a heap of its own.
    let m3 = measure(&t);
    let first = Heap::new(&t);
    dispatch(&t, &first);
    let m4 = measure(&t);
    report("first host heap and first dispatch (device warm-up included)", m3, m4);
    for round in 0..3 {
        let a = measure(&t);
        let heap = Heap::new(&t);
        let b = measure(&t);
        report(&format!("another host resource heap made (HEAP_VIEWS long), #{round}"), a, b);
        dispatch(&t, &heap);
        let c = measure(&t);
        report(&format!("its first dispatch (stride table, null views, residency), #{round}"), b, c);
        dispatch(&t, &heap);
        report(&format!("its second dispatch, #{round}"), c, measure(&t));
        std::mem::forget(heap);
    }
    drop((short, long, f));
}

/// Loads an image SRV from the table at `s4` by adding a `v_readfirstlane_b32` byte offset (`32`,
/// entry 1) with `s_add_u32 s0, s4, s8; s_add_u32 s1, 0, src_scc`, and stores its green channel
/// (`1.0`) to the raw UAV at byte offset `64` (entry 2) of the same table (`#258`).
const CS_TABLE_SCC_CARRY: &[u32] = &[
    0xBE8A_0304, // s_mov_b32 s10, s4
    0xBE8B_0380, // s_mov_b32 s11, 0
    0xF408_0305, 0xFA00_0040, // s_load_dwordx4 s[12:15], s[10:11], 0x40
    0x7E08_02A0, // v_mov_b32 v4, 32 (entry 1 byte offset)
    0x7E10_0504, // v_readfirstlane_b32 s8, v4
    0x8000_0804, // s_add_u32 s0, s4, s8
    0x8001_FD80, // s_add_u32 s1, 0, src_scc
    0xF40C_0400, 0xFA00_0000, // s_load_dwordx8 s[16:23], s[0:1], 0x0
    0x7E00_0280, // v_mov_b32 v0, 0
    0x7E02_0280, // v_mov_b32 v1, 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF000_1F08, 0x0004_0200, // image_load v[2:5], v[0:1], s[16:23] dmask:0xf dim:2D unorm
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_0000, 0x8003_0300, // buffer_store_dword v3, off, s[12:15], 0
    0xBF81_0000, // s_endpgm
];

#[test]
fn a_compute_shader_loads_a_texture_with_scc_carry() {
    // SRV range (kind 2) of 2 textures at dwords 0..16, and raw UAV (kind 3) at dword 16, in
    // table (root type 3) at user data 4, slot 0.
    let mut srv = [0u8; 32];
    srv[0] = 0x41;
    srv[4..6].copy_from_slice(&[2, 3]);
    srv[12] = 1; // upper_bound = 1 (2 entries: 0 and 1)
    srv[16] = 2; // count = 2
    srv[20..23].copy_from_slice(&[8, 4, 0]);
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[8] = 16; // table_offset_dwords = 16 (byte 64, entry 2)
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
    let c = crate::pipeline::test_compute_container(CS_TABLE_SCC_CARRY, &[srv, uav], 7);
    let cs = shaders::translate(&c).unwrap();
    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    let heap = Heap::new(&t);
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[4], view[8]) = (1, 64, 1);
    [view[10], view[11]] = super::addr_tests::halves(super::addr_tests::VA_A);
    let slot = heap._records.as_ptr() as u64 + 64;
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, slot);
    assert!(t.host.mirror(slot).unwrap());
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "table scc carry") else { panic!("{:?}", t.shaders) };
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap.heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_table(0, &heap.heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(f.get(0x2_0000, 2), [0x3F80_0000, super::addr_tests::CANARY]);
}

/// A compute program that walks title memory from a buffer view's own address, as the BVH walks
/// of the title's ray-traced passes do: a table at user data 4 holds the raw output UAV (dwords
/// 0-3) and a buffer SRV (dwords 8-11); the SRV's descriptor dwords 0-1 are read as its title
/// address (dword 1's low 16 bits), and the word there is loaded and stored to the output.
const CS_VIEW_ADDRESS: &[u32] = &[
    0xBE8A_0304, // s_mov_b32 s10, s4
    0xBE8B_0380, // s_mov_b32 s11, 0
    0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
    0xF408_0405, 0xFA00_0020, // s_load_dwordx4 s[16:19], s[10:11], 0x20
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x8711_FF11, 0x0000_FFFF, // s_and_b32 s17, s17, 0xffff
    0x7E00_0280, // v_mov_b32 v0, 0
    0xDC30_8000, 0x0110_0000, // global_load_dword v1, v0, s[16:17]
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_0000, 0x8003_0100, // buffer_store_dword v1, off, s[12:15], 0
    0xBF81_0000, // s_endpgm
];

/// The heap's view addresses (`t1` of the memory space) give a program the title address of a
/// buffer view in the heap (#306). The native host bound no such table and refused the program
/// ("needs view addresses or write marks"), as it did 7 compute programs of the world.
#[test]
fn a_compute_shader_reads_title_memory_at_a_heap_views_address() {
    // A raw UAV and a raw SRV (kinds 3 and 0) in a table (root type 3) at user data 4.
    let record = |kind: u8, offset: u8, register: u8| {
        let mut r = [0u8; 32];
        r[0] = 0x41;
        r[4..6].copy_from_slice(&[kind, 3]);
        (r[8], r[12], r[16]) = (offset, register, 1);
        r[20..23].copy_from_slice(&[4, 4, 0]);
        r
    };
    let c = crate::pipeline::test_compute_container(CS_VIEW_ADDRESS, &[record(3, 0, 0), record(0, 8, 1)], 7);
    let cs = shaders::translate(&c).unwrap();
    assert!(cs.view_addresses && cs.memory, "{cs:?}");
    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    // The word the program finds by the view's address: pool offset 0x2_1000 (`VA_A + 0x1000`).
    f.put(0x2_1000, &[0x1234_5678u32]);
    let records = vec![0u32; 8 * 2];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 2).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let view_at = |address: u64| {
        let mut v = [0u32; 16];
        (v[1], v[4], v[6], v[8]) = (1, 64, 64, 1);
        [v[10], v[11]] = super::addr_tests::halves(address);
        v
    };
    // Entry 0: the output, 64 raw dwords at `VA_A` (pool offset 0x2_0000); entry 1: the SRV.
    let mut uav = view_at(super::addr_tests::VA_A);
    uav[6] = 0;
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav, start);
    let mut srv = view_at(super::addr_tests::VA_A + 0x1000);
    srv[4] = 0;
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, &srv, start + 32);
    assert!(t.host.mirror(start).unwrap());
    assert!(t.host.mirror(start + 32).unwrap());
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "view address") else { panic!("{:?}", t.shaders) };
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(f.get(0x2_0000, 2), [0x1234_5678, super::addr_tests::CANARY]);
    drop(records);
}

/// A compute program that needs both of the host's per-heap tables: the view-stride table (root
/// SRV 255, among its root views) for the output buffer it takes from the resource heap itself
/// (entry 2, by the heap's base in s6), and the view addresses (`t1` of the memory space, after
/// the page table) for the title address of the SRV in its table (entry 1). The converter lays
/// the argument buffer out by `Binding::layout` and the host fills it in the same order (user
/// data, root views, page table, view addresses): were two swapped, the shader would read one
/// table as the other and store nothing, or the wrong word.
const CS_STRIDES_AND_ADDRESSES: &[u32] = &[
    0xBE8A_0304, // s_mov_b32 s10, s4
    0xBE8B_0380, // s_mov_b32 s11, 0
    0xF408_0405, 0xFA00_0000, // s_load_dwordx4 s[16:19], s[10:11], 0x0
    0xBE80_0306, // s_mov_b32 s0, s6
    0xBE81_0380, // s_mov_b32 s1, 0
    0xF408_0300, 0xFA00_0040, // s_load_dwordx4 s[12:15], s[0:1], 0x40
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x8711_FF11, 0x0000_FFFF, // s_and_b32 s17, s17, 0xffff
    0x7E00_0280, // v_mov_b32 v0, 0
    0xDC30_8000, 0x0110_0000, // global_load_dword v1, v0, s[16:17]
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_0000, 0x8003_0100, // buffer_store_dword v1, off, s[12:15], 0
    0xBF81_0000, // s_endpgm
];

#[test]
fn a_compute_shader_reads_both_the_view_strides_and_the_view_addresses() {
    // A raw SRV (kind 0) in a table (root type 3) at user data 4.
    let mut record = [0u8; 32];
    record[0] = 0x41;
    record[4..6].copy_from_slice(&[0, 3]);
    record[16] = 1;
    record[20..23].copy_from_slice(&[4, 4, 0]);
    let c = crate::pipeline::test_compute_container(CS_STRIDES_AND_ADDRESSES, &[record], 7);
    let cs = shaders::translate(&c).unwrap();
    assert!(cs.view_addresses && cs.memory && cs.heap, "{cs:?}");
    assert_eq!(cs.root_srvs, [victory_shader::emit::VIEW_STRIDES]);
    let Some(mut t) = title() else { return };
    let f = super::addr_tests::Fixture::new(&t);
    // The word the program finds by the SRV's address: pool offset 0x2_1000 (`VA_A + 0x1000`);
    // a decoy at `VA_A + 0x2000`, which entry 0 views.
    f.put(0x2_1000, &[0x1234_5678u32]);
    f.put(0x2_2000, &[0xDEC0_DEC0u32]);
    let records = vec![0u32; 8 * 3];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 3).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let view_at = |address: u64| {
        let mut v = [0u32; 16];
        (v[1], v[4], v[8]) = (1, 64, 1);
        [v[10], v[11]] = super::addr_tests::halves(address);
        v
    };
    // Entry 0: the decoy; entry 1: the SRV (the table starts there); entry 2: the output, 64 raw
    // dwords at `VA_A` (pool offset 0x2_0000). A program that read the page table as the view
    // addresses, or the other way round, would not find the SRV's word.
    let srv = |address: u64| {
        let mut v = view_at(address);
        (v[4], v[6]) = (0, 64);
        v
    };
    let views = [
        srv(super::addr_tests::VA_A + 0x2000),
        srv(super::addr_tests::VA_A + 0x1000),
    ];
    for (entry, view) in views.iter().enumerate() {
        let at = start + 32 * entry as u64;
        crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, view, at);
        assert!(t.host.mirror(at).unwrap());
    }
    let uav = view_at(super::addr_tests::VA_A);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav, start + 64);
    assert!(t.host.mirror(start + 64).unwrap());
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "strides and addresses") else {
        panic!("{:?}", t.shaders)
    };
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_table(0, &heap, 1);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(f.get(0x2_0000, 2), [0x1234_5678, super::addr_tests::CANARY]);
    assert_eq!((t.host.unset_roots(), t.host.failed()), (0, 0));
    drop(records);
}
