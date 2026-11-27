//! Mesh-style (NGG) vertex programs of guessed pipelines on the native host: hand-assembled GFX10
//! translated as mesh shaders, made into Metal mesh pipelines and drawn.

use super::addr_tests::title;
use crate::native::list::{Binding, NativeList};
use crate::native::shaders;
use crate::native::state::Graphics;
use crate::native::title::{Pipe, Title};
use std::rc::Rc;
use victory_backend::resources::{Format, GraphicsBackend, ResourceBackend, TextureDesc, Topology};
use victory_metal::metal::Metal;
use victory_shader::emit::HostShader;
use victory_shader::native::HwStage;

type Texture = <Metal as ResourceBackend>::Texture;

/// A mesh-style vertex program as the title's start (lane 0 alone takes the group id, shifted by
/// 7: groups of 128 lanes), whose lanes 0-2 export the corners of a triangle covering the target
/// and whose lane 0 exports primitive (0, 1, 2).
pub(super) const TRIANGLE: &[u32] = &[
    0xBEFE_0481, // s_mov_b64 exec, 1
    0x2C0A_0A87, // v_lshrrev_b32 v5, 7, v5
    0xBEFE_04C1, // s_mov_b64 exec, -1
    0xD765_0001, 0x0201_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
    0xD766_0001, 0x0202_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
    0x7E14_02F3, // v_mov_b32 v10, -1.0
    0x7E16_02FF, 0x4040_0000, // v_mov_b32 v11, 3.0
    0x7D84_0281, // v_cmp_eq_u32 vcc, 1, v1
    0x0204_170A, // v_cndmask_b32 v2, v10, v11, vcc
    0x7D84_0282, // v_cmp_eq_u32 vcc, 2, v1
    0x0206_170A, // v_cndmask_b32 v3, v10, v11, vcc
    0x7E08_0280, // v_mov_b32 v4, 0
    0x7E0C_02F2, // v_mov_b32 v6, 1.0
    0xF800_08CF, 0x0604_0302, // exp pos0 v2, v3, v4, v6 done
    0x7E0E_02FF, 0x0020_0400, // v_mov_b32 v7, 0x200400 (corners 0, 1, 2)
    0x7DA4_0280, // v_cmpx_eq_u32 0, v1
    0xF800_0941, 0x0000_0007, // exp prim v7 done
    0xBF81_0000, // s_endpgm
];

/// Writes (1.0, 0.5, 0.5, 1.0) to mrt0.
pub(super) const RED: &[u32] = &[
    0x7E00_02F2, // v_mov_b32 v0, 1.0
    0x7E02_02F0, // v_mov_b32 v1, 0.5
    0xF800_180F, 0x0001_0100, // exp mrt0 v0, v1, v1, v0 done vm
    0xBF81_0000, // s_endpgm
];

pub(super) fn translate(stage: HwStage, words: &[u32]) -> HostShader {
    let mut meta = [0; 0x10];
    meta[12] = 0x40;
    translate_with(stage, words, &meta)
}

fn translate_with(stage: HwStage, words: &[u32], meta: &[u8]) -> HostShader {
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let c = victory_shader::emit::graphics::container(stage, &bytes, meta).unwrap();
    shaders::translate(&c).unwrap()
}

/// A guessed pipeline whose vertex program is mesh-style becomes a Metal mesh pipeline, with the
/// thread-group size the converter reflects for its mesh shader.
#[test]
fn a_mesh_style_vertex_program_makes_a_mesh_pipeline() {
    let ms = translate(HwStage::Vs, TRIANGLE);
    assert_eq!(ms.stage, HwStage::Ms);
    let ps = translate(HwStage::Ps, RED);
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, _) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders);
    };
    assert_eq!(pso.mesh_threads(), Some([128, 1, 1]));
}

/// An 8x8 RGBA8 target cleared to black, `pso` and `binding` set on a list.
fn list_into(
    t: &Title<Metal>,
    pso: &Rc<<Metal as GraphicsBackend>::RenderPipeline>,
    binding: Option<Rc<Binding>>,
) -> (Rc<Texture>, NativeList<Metal>) {
    let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 8, 8);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    let mut list = NativeList::default();
    list.set_render_targets(std::slice::from_ref(&color), None);
    list.clear_render_target(&color, [0.0; 4]);
    list.set_graphics_pipeline(pso, Topology::Triangles);
    list.set_graphics_binding(binding);
    (color, list)
}

fn pixels(t: &Title<Metal>, color: &Texture) -> Vec<[u8; 4]> {
    let mut p = [0; 8 * 8 * 4];
    t.host.backend.read_texture(color, 0, 0, &mut p, 8 * 4, 0);
    p.chunks(4).map(|c| c.try_into().unwrap()).collect()
}

/// `DispatchMesh` of one group of the triangle program colours the whole target.
#[test]
fn a_mesh_draw_colours_the_target() {
    let (ms, ps) = (translate(HwStage::Vs, TRIANGLE), translate(HwStage::Ps, RED));
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders);
    };
    let (color, mut list) = list_into(&t, &pso, binding);
    list.dispatch_mesh(1, 1, 1).unwrap();
    super::run(&t.host, &[&list]);
    assert!(pixels(&t, &color).iter().all(|p| *p == [255, 128, 128, 255]), "{:?}", pixels(&t, &color));
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}

/// The mesh stage gets its root arguments: the triangle program with its corners scaled by the
/// root constant in user data dword 0 (s8 of a mesh program), 1.0. Without them it reads 0 and
/// draws nothing.
#[test]
fn a_mesh_shader_reads_its_root_constants() {
    let mut words = TRIANGLE.to_vec();
    let at = words.iter().position(|&w| w == 0x7E08_0280).unwrap(); // before v_mov_b32 v4, 0
    words.splice(at..at, [
        0x1004_0408, // v_mul_f32 v2, s8, v2
        0x1006_0608, // v_mul_f32 v3, s8, v3
    ]);
    let mut meta = [0; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // Root constants in slot 0, user data dword 0.
    let ms = translate_with(HwStage::Vs, &words, &meta);
    let ps = translate(HwStage::Ps, RED);
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders);
    };
    let (color, mut list) = list_into(&t, &pso, binding);
    list.set_graphics_root_constants(0, &[1.0f32.to_bits()]);
    list.dispatch_mesh(1, 1, 1).unwrap();
    super::run(&t.host, &[&list]);
    assert!(pixels(&t, &color).iter().all(|p| *p == [255, 128, 128, 255]), "{:?}", pixels(&t, &color));
}

/// `ExecuteIndirect` of `DispatchMesh` with a count buffer: the record (one group) and the count
/// (1) are read on the GPU, so record 0 colours the target and record 1 of the same count, not
/// below it, leaves it black.
#[test]
fn a_counted_indirect_mesh_dispatch_runs_only_the_records_below_the_count() {
    let (ms, ps) = (translate(HwStage::Vs, TRIANGLE), translate(HwStage::Ps, RED));
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders);
    };
    let pool = super::Pool::new(&t.host, 1);
    let (records, count) = (0, 64);
    pool.write(records, &[1u32, 1, 1, 1, 1, 1]);
    pool.write(count, &[1u32]);
    for (i, want) in [(0, [255, 128, 128, 255]), (1, [0; 4])] {
        let (color, mut list) = list_into(&t, &pso, binding.clone());
        list.dispatch_mesh_indirect(pool.va(records + 12 * i as usize), Some((pool.va(count), i)))
            .unwrap();
        super::run(&t.host, &[&list]);
        assert!(pixels(&t, &color).iter().all(|p| *p == want), "record {i}: {:?}", pixels(&t, &color));
    }
}

/// A mesh dispatch record that asks for more than 2^22 groups in all (D3D12's most for
/// `DispatchMesh`) is what a culling pass left unfinished in title memory: it draws nothing
/// (Metal would run it, for as long as it takes), and one below that still draws.
#[test]
fn an_indirect_mesh_dispatch_past_the_most_groups_draws_nothing() {
    let (ms, ps) = (translate(HwStage::Vs, TRIANGLE), translate(HwStage::Ps, RED));
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders);
    };
    let pool = super::Pool::new(&t.host, 1);
    let red = [255, 128, 128, 255];
    for (record, want) in [([1u32, 1, 1], red), ([256, 256, 65], [0; 4]), ([1, 1_182_027_346, 1_314_076_063], [0; 4]), ([1, 1, 64], red)] {
        pool.write(0, &record);
        let (color, mut list) = list_into(&t, &pso, binding.clone());
        list.dispatch_mesh_indirect(pool.va(0), None).unwrap();
        super::run(&t.host, &[&list]);
        assert!(pixels(&t, &color).iter().all(|p| *p == want), "record {record:?}: {:?}", pixels(&t, &color));
    }
}

/// 16-bit parameters across guessed stages (`graphics::HALF`): the triangle program exports
/// param0 as two packed dwords (x: 1.0 high, 0.5 low; y: 0.25 high, 0.75 low), and the pixel
/// program interpolates each half with `v_interp_p1ll_f16`/`v_interp_p2_f16` (low and `high`)
/// and writes them to mrt0 as (x.lo, x.hi, y.lo, y.hi). Without a cross-stage convention the
/// pixel program is refused.
#[test]
fn packed_half_parameters_reach_the_pixel_program() {
    let mut words = TRIANGLE.to_vec();
    let at = words.iter().position(|&w| w == 0xF800_08CF).unwrap(); // before exp pos0
    words.splice(at..at, [
        0x7E10_02FF, 0x3C00_3800, // v_mov_b32 v8, 0x3c003800 (1.0, 0.5)
        0x7E12_02FF, 0x3400_3A00, // v_mov_b32 v9, 0x34003a00 (0.25, 0.75)
        0xF800_0203, 0x0000_0908, // exp param0 v8, v9
    ]);
    let ps_words = [
        0xD742_0002, 0x0202_0000, // v_interp_p1ll_f16 v2, v0, attr0.x
        0xD75A_0003, 0x040A_0200, // v_interp_p2_f16 v3, v1, attr0.x, v2
        0xD742_0002, 0x0202_0100, // v_interp_p1ll_f16 v2, v0, attr0.x high
        0xD75A_0004, 0x040A_0300, // v_interp_p2_f16 v4, v1, attr0.x high, v2
        0xD742_0002, 0x0202_0040, // v_interp_p1ll_f16 v2, v0, attr0.y
        0xD75A_0005, 0x040A_0240, // v_interp_p2_f16 v5, v1, attr0.y, v2
        0xD742_0002, 0x0202_0140, // v_interp_p1ll_f16 v2, v0, attr0.y high
        0xD75A_0006, 0x040A_0340, // v_interp_p2_f16 v6, v1, attr0.y high, v2
        0x7E06_1703, // v_cvt_f32_f16 v3, v3
        0x7E08_1704, // v_cvt_f32_f16 v4, v4
        0x7E0A_1705, // v_cvt_f32_f16 v5, v5
        0x7E0C_1706, // v_cvt_f32_f16 v6, v6
        0xF800_180F, 0x0605_0403, // exp mrt0 v3, v4, v5, v6 done vm
        0xBF81_0000, // s_endpgm
    ];
    let (ms, ps) = (translate(HwStage::Vs, &words), translate(HwStage::Ps, &ps_words));
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders);
    };
    let (color, mut list) = list_into(&t, &pso, binding);
    list.dispatch_mesh(1, 1, 1).unwrap();
    super::run(&t.host, &[&list]);
    assert!(pixels(&t, &color).iter().all(|p| *p == [128, 255, 191, 64]), "{:?}", pixels(&t, &color));
}

/// The uniform-mask SCC rule on the GPU (`Lower::uniform_and_exec`): inside an EXEC region of
/// lanes 0-2 (`v_cmpx`), the triangle program compares the root constant (s8) with 0, ANDs the
/// mask with EXEC and keeps SCC in s20; only when it is set does it keep the triangle's size
/// (else it scales the corners by 0 and draws nothing). s8 = 1.0 colours the target, -1.0
/// leaves it black.
#[test]
fn a_mesh_program_branches_on_the_scc_of_a_uniform_lane_mask() {
    let mut words = TRIANGLE.to_vec();
    let at = words.iter().position(|&w| w == 0x7E08_0280).unwrap(); // before v_mov_b32 v4, 0
    words.splice(at..at, [
        0x7E1A_0280, // v_mov_b32 v13, 0
        0x7DA8_0283, // v_cmpx_gt_u32 3, v1
        0xBF88_0007, // s_cbranch_execz 7
        0xD406_006A, 0x0001_0008, // v_cmp_ge_f32 vcc, s8, 0
        0x8796_7E6A, // s_and_b64 s[22:23], vcc, exec
        0xBE94_03FD, // s_mov_b32 s20, src_scc
        0xBF01_8014, // s_cmp_lg_i32 s20, 0
        0xBF84_0001, // s_cbranch_scc0 1
        0x7E1A_02F2, // v_mov_b32 v13, 1.0
        0xBEFE_04C1, // s_mov_b64 exec, -1
        0x1004_050D, // v_mul_f32 v2, v13, v2
        0x1006_070D, // v_mul_f32 v3, v13, v3
    ]);
    let mut meta = [0; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // Root constants in slot 0, user data dword 0.
    let ms = translate_with(HwStage::Vs, &words, &meta);
    let ps = translate(HwStage::Ps, RED);
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders);
    };
    for (s8, want) in [(1.0f32, [255, 128, 128, 255]), (-1.0, [0; 4])] {
        let (color, mut list) = list_into(&t, &pso, binding.clone());
        list.set_graphics_root_constants(0, &[s8.to_bits()]);
        list.dispatch_mesh(1, 1, 1).unwrap();
        super::run(&t.host, &[&list]);
        assert!(pixels(&t, &color).iter().all(|p| *p == want), "s8 = {s8}: {:?}", pixels(&t, &color));
    }
}

/// As above, but SCC is that of EXEC without the compare ANDed with an `s_cselect` by bit 29 of
/// s8 (`s_andn2_b64 sN, exec, mask`, as the NGG vertex programs ahead of the G-buffer test it):
/// set, so the triangle drawn, unless s8 is at least 0 with bit 29 set. 1.0 leaves the target
/// black; -1.0 (below 0) and 2.0 (bit 29 clear) colour it.
#[test]
fn a_mesh_program_branches_on_the_scc_of_exec_without_a_uniform_lane_mask() {
    let mut words = TRIANGLE.to_vec();
    let at = words.iter().position(|&w| w == 0x7E08_0280).unwrap(); // before v_mov_b32 v4, 0
    words.splice(at..at, [
        0x7E1A_0280, // v_mov_b32 v13, 0
        0x7DA8_0283, // v_cmpx_gt_u32 3, v1
        0xBF88_000A, // s_cbranch_execz 10
        0xD406_006A, 0x0001_0008, // v_cmp_ge_f32 vcc, s8, 0
        0xBF0D_9D08, // s_bitcmp1_b32 s8, 29
        0x8598_80C1, // s_cselect_b64 s[24:25], -1, 0
        0x8798_186A, // s_and_b64 s[24:25], vcc, s[24:25]
        0x8A96_187E, // s_andn2_b64 s[22:23], exec, s[24:25]
        0xBE94_03FD, // s_mov_b32 s20, src_scc
        0xBF01_8014, // s_cmp_lg_i32 s20, 0
        0xBF84_0001, // s_cbranch_scc0 1
        0x7E1A_02F2, // v_mov_b32 v13, 1.0
        0xBEFE_04C1, // s_mov_b64 exec, -1
        0x1004_050D, // v_mul_f32 v2, v13, v2
        0x1006_070D, // v_mul_f32 v3, v13, v3
    ]);
    let mut meta = [0; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // Root constants in slot 0, user data dword 0.
    let ms = translate_with(HwStage::Vs, &words, &meta);
    let ps = translate(HwStage::Ps, RED);
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders);
    };
    let red = [255, 128, 128, 255];
    for (s8, want) in [(1.0f32, [0; 4]), (-1.0, red), (2.0, red)] {
        let (color, mut list) = list_into(&t, &pso, binding.clone());
        list.set_graphics_root_constants(0, &[s8.to_bits()]);
        list.dispatch_mesh(1, 1, 1).unwrap();
        super::run(&t.host, &[&list]);
        assert!(pixels(&t, &color).iter().all(|p| *p == want), "s8 = {s8}: {:?}", pixels(&t, &color));
    }
}

/// An SGPR spilled to a VGPR lane on two paths and read back after the join, then swapped with
/// a mask by three XORs and read as data (the NGG vertex programs ahead of the G-buffer do
/// both): the triangle's scale is 1.0 when s8 is not negative, else 0. 1.0 colours the target,
/// -1.0 leaves it black.
#[test]
fn a_mesh_program_reads_back_a_spill_joined_and_swapped_with_a_mask() {
    let mut words = TRIANGLE.to_vec();
    let at = words.iter().position(|&w| w == 0x7E08_0280).unwrap(); // before v_mov_b32 v4, 0
    words.splice(at..at, [
        0xBE94_03F2, // s_mov_b32 s20, 1.0
        0xBE95_0380, // s_mov_b32 s21, 0
        0xBF0D_9F08, // s_bitcmp1_b32 s8, 31
        0xBF85_0003, // s_cbranch_scc1 3
        0xD761_000E, 0x0001_0614, // v_writelane_b32 v14, s20, 3
        0xBF82_0002, // s_branch 2
        0xD761_000E, 0x0001_0615, // v_writelane_b32 v14, s21, 3
        0xD760_0016, 0x0001_070E, // v_readlane_b32 s22, v14, 3
        0xBE98_047E, // s_mov_b64 s[24:25], exec
        0x8918_1618, // s_xor_b32 s24, s24, s22
        0x8916_1618, // s_xor_b32 s22, s24, s22
        0x8918_1618, // s_xor_b32 s24, s24, s22
        0x7E1A_0218, // v_mov_b32 v13, s24
        0x1004_050D, // v_mul_f32 v2, v13, v2
        0x1006_070D, // v_mul_f32 v3, v13, v3
    ]);
    let mut meta = [0; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // Root constants in slot 0, user data dword 0.
    let ms = translate_with(HwStage::Vs, &words, &meta);
    let ps = translate(HwStage::Ps, RED);
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders);
    };
    for (s8, want) in [(1.0f32, [255, 128, 128, 255]), (-1.0, [0; 4])] {
        let (color, mut list) = list_into(&t, &pso, binding.clone());
        list.set_graphics_root_constants(0, &[s8.to_bits()]);
        list.dispatch_mesh(1, 1, 1).unwrap();
        super::run(&t.host, &[&list]);
        assert!(pixels(&t, &color).iter().all(|p| *p == want), "s8 = {s8}: {:?}", pixels(&t, &color));
    }
}

/// A pixel program reading the halves of fewer channels than the vertex side exports still links
/// (parameter elements are four channels on both sides, those the export leaves out written a
/// copy of one it has): only x's low half, of a parameter of x and y. Before, the converter
/// refused such pairs as mismatching types, and crashed when the channels were left unwritten.
#[test]
fn halves_of_some_channels_link() {
    let mut words = TRIANGLE.to_vec();
    let at = words.iter().position(|&w| w == 0xF800_08CF).unwrap(); // before exp pos0
    words.splice(at..at, [
        0x7E10_02FF, 0x3C00_3800, // v_mov_b32 v8, 0x3c003800 (1.0, 0.5)
        0x7E12_02FF, 0x3400_3A00, // v_mov_b32 v9, 0x34003a00 (0.25, 0.75)
        0xF800_0203, 0x0000_0908, // exp param0 v8, v9
    ]);
    let ps_words = [
        0xD742_0002, 0x0202_0000, // v_interp_p1ll_f16 v2, v0, attr0.x
        0xD75A_0003, 0x040A_0200, // v_interp_p2_f16 v3, v1, attr0.x, v2
        0x7E06_1703, // v_cvt_f32_f16 v3, v3
        0xF800_180F, 0x0303_0303, // exp mrt0 v3, v3, v3, v3 done vm
        0xBF81_0000, // s_endpgm
    ];
    let (ms, ps) = (translate(HwStage::Vs, &words), translate(HwStage::Ps, &ps_words));
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders.refused);
    };
    let (color, mut list) = list_into(&t, &pso, binding);
    list.dispatch_mesh(1, 1, 1).unwrap();
    super::run(&t.host, &[&list]);
    assert!(pixels(&t, &color).iter().all(|p| *p == [128; 4]), "{:?}", pixels(&t, &color));
}

/// The halves fit the converter's mesh size (32 KiB of outputs): ten parameters a literal fills
/// (so each may hold halves) still make a mesh pipeline, those past the room left without them.
#[test]
fn halves_of_many_parameters_stay_within_the_mesh_size() {
    let mut words = TRIANGLE.to_vec();
    let at = words.iter().position(|&w| w == 0xF800_08CF).unwrap(); // before exp pos0
    words.splice(at..at, [0x7E10_02FF, 0x3C00_3800]); // v_mov_b32 v8, 0x3c003800
    let at = at + 2;
    // exp param<n> v8, v8, v8, v8
    words.splice(at..at, (0..10u32).flat_map(|n| [0xF800_020F | n << 4, 0x0808_0808]));
    let (ms, ps) = (translate(HwStage::Vs, &words), translate(HwStage::Ps, RED));
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(..) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders.refused);
    };
}

/// A parameter only a packing instruction makes (`v_cvt_pkrtz_f16_f32`: 0.5 low, 1.0 high) is
/// exported as its halves alone, and the pixel program's f16 interpolations of both halves of
/// x read them.
#[test]
fn a_parameter_of_halves_only_reaches_the_pixel_program() {
    let mut words = TRIANGLE.to_vec();
    let at = words.iter().position(|&w| w == 0xF800_08CF).unwrap(); // before exp pos0
    words.splice(at..at, [
        0x5E10_0CF0, // v_cvt_pkrtz_f16_f32 v8, 0.5, v6 (1.0)
        0xF800_0201, 0x0000_0008, // exp param0 v8
    ]);
    let ps_words = [
        0xD742_0002, 0x0202_0000, // v_interp_p1ll_f16 v2, v0, attr0.x
        0xD75A_0003, 0x040A_0200, // v_interp_p2_f16 v3, v1, attr0.x, v2
        0xD742_0002, 0x0202_0100, // v_interp_p1ll_f16 v2, v0, attr0.x high
        0xD75A_0004, 0x040A_0300, // v_interp_p2_f16 v4, v1, attr0.x high, v2
        0x7E06_1703, // v_cvt_f32_f16 v3, v3
        0x7E08_1704, // v_cvt_f32_f16 v4, v4
        0xF800_180F, 0x0403_0403, // exp mrt0 v3, v4, v3, v4 done vm
        0xBF81_0000, // s_endpgm
    ];
    let (ms, ps) = (translate(HwStage::Vs, &words), translate(HwStage::Ps, &ps_words));
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mesh") else {
        panic!("{:?}", t.shaders.refused);
    };
    let (color, mut list) = list_into(&t, &pso, binding);
    list.dispatch_mesh(1, 1, 1).unwrap();
    super::run(&t.host, &[&list]);
    assert!(pixels(&t, &color).iter().all(|p| *p == [128, 255, 128, 255]), "{:?}", pixels(&t, &color));
}

/// The library the converter writes for a mesh program is valid IR: the triangle program also
/// exports the lane's number, a value of the run, as param0's two dwords, so each store of that
/// output takes a cast or a 16-bit conversion made for it. The converter gathers the stores of
/// one output into a vector at the first of them; with a value made between two stores the
/// library used it before its definition ("Instruction does not dominate all uses" from
/// `metal-as`, and compiler failures in Metal for larger programs). The library is taken apart
/// with `metal-objdump` and put together again with `metal-as`, which verifies it. Skipped
/// without the converter or those tools.
#[test]
fn a_converted_mesh_program_with_run_time_parameters_is_valid_ir() {
    use std::process::Command;
    let mut words = TRIANGLE.to_vec();
    let at = words.iter().position(|&w| w == 0xF800_08CF).unwrap(); // before exp pos0
    words.splice(at..at, [
        0xF800_0203, 0x0000_0101, // exp param0 v1, v1
    ]);
    let ms = translate(HwStage::Vs, &words);
    assert_eq!(ms.stage, HwStage::Ms);
    let Ok(msc) = victory_metal::msc::Msc::try_open_default() else {
        eprintln!("skipped: Metal Shader Converter not installed");
        return;
    };
    let tool = |name: &str| {
        let found = Command::new("xcrun").args(["-f", name]).output().ok()?;
        found.status.success().then(|| String::from_utf8_lossy(&found.stdout).trim().to_owned())
    };
    let (Some(objdump), Some(assembler)) = (tool("metal-objdump"), tool("metal-as")) else {
        eprintln!("skipped: no metal-objdump or metal-as (the Metal toolchain is not installed)");
        return;
    };
    let layout = Binding::of(&ms).layout();
    let (stage, lib) = msc.convert_with(&ms.dxil, None, layout.as_ref(), false).unwrap();
    assert_eq!(stage, victory_metal::msc::STAGE_MESH);
    let dir = tempfile::tempdir().unwrap();
    let path = |name: &str| dir.path().join(name);
    std::fs::write(path("mesh.metallib"), lib).unwrap();
    let listing = Command::new(objdump).arg("--disassemble").arg(path("mesh.metallib")).output().unwrap();
    assert!(listing.status.success(), "{}", String::from_utf8_lossy(&listing.stderr));
    // The module's text follows the tool's own heading.
    let text = String::from_utf8_lossy(&listing.stdout);
    let module = &text[text.find("source_filename").expect("a module in the listing")..];
    assert!(module.contains("air.set_vertex_data_mesh"), "{module}");
    std::fs::write(path("mesh.ll"), module).unwrap();
    let assembled = Command::new(assembler).arg(path("mesh.ll")).arg("-o").arg(path("mesh.air")).output().unwrap();
    assert!(assembled.status.success(), "{}", String::from_utf8_lossy(&assembled.stderr));
}

/// A legacy vertex program that takes its position from the vertex id alone (no vertex fetch):
/// id 1 is (1, -1), id 2 (-1, 1), any other (-1, -1). As a rectangle list's three corners
/// (0, 1, 2), the rectangle is the whole target.
const RECT_BY_ID: &[u32] = &[
    0x7E14_02F3, // v_mov_b32 v10, -1.0
    0x7E16_02F2, // v_mov_b32 v11, 1.0
    0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
    0x0204_170A, // v_cndmask_b32 v2, v10, v11, vcc
    0x7D84_0082, // v_cmp_eq_u32 vcc, 2, v0
    0x0206_170A, // v_cndmask_b32 v3, v10, v11, vcc
    0x7E08_0280, // v_mov_b32 v4, 0
    0x7E0C_02F2, // v_mov_b32 v6, 1.0
    0xF800_08CF, 0x0604_0302, // exp pos0 v2, v3, v4, v6 done
    0xBF81_0000, // s_endpgm
];

/// A rectangle-list pipeline of [`RECT_BY_ID`] translated as `guess` does (`Options::rect_list`),
/// and a list into a black 8x8 target with it set as a strip.
fn rect_list_into(t: &mut Title<Metal>) -> (Rc<Texture>, NativeList<Metal>) {
    let mut meta = [0; 0x10];
    meta[12] = 0x40;
    let bytes: Vec<_> = RECT_BY_ID.iter().flat_map(|w| w.to_le_bytes()).collect();
    let c = victory_shader::emit::graphics::container(HwStage::Vs, &bytes, &meta).unwrap();
    let vs = shaders::translate_rect(&c).unwrap();
    assert_eq!(vs.stage, HwStage::Vs);
    let ps = translate(HwStage::Ps, RED);
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], true);
    let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &state, "rect") else {
        panic!("{:?}", t.shaders);
    };
    let (color, mut list) = list_into(t, &pso, binding);
    list.set_graphics_pipeline(&pso, Topology::TriangleStrip);
    (color, list)
}

fn lit(t: &Title<Metal>, color: &Texture) -> usize {
    pixels(t, color).iter().filter(|p| **p == [255, 128, 128, 255]).count()
}

/// A rectangle-list pipeline whose vertex program takes its corners from `SV_VertexID` alone runs
/// the program for console vertex ids `3 * instance + 0..2`, finds the right-angle corner from
/// `SV_Position.xy`, and extrapolates the fourth corner `p + q - r`, covering all 64 texels of the
/// 8x8 target instead of only the lower-left 28 texels.
#[test]
fn a_rectangle_of_a_vertex_id_program_covers_half_its_rectangle() {
    let Some(mut t) = title() else { return };
    let (color, mut list) = rect_list_into(&mut t);
    list.draw_instanced(4, 1, 0, 0).unwrap();
    super::run(&t.host, &[&list]);
    let n = lit(&t, &color);
    assert_eq!(n, 64, "{n} of 64 lit");
}

/// Multi-rectangle `SV_VertexID` draws (`vertices = 3 * N`, direct and `ExecuteIndirect`) advance
/// `v0` by `3 * instance + corner`, handle half-size and quarter-size tiles regardless of which
/// corner holds the right angle, and extrapolate `TEXCOORD` varyings linearly across the fourth
/// corner.
#[test]
fn vertex_id_rectangle_lists_draw_half_tiles_multiple_rects_and_interpolated_params() {
    // Rectangle 0 (`v0 = 0..2`, right angle at corner 0): left half `x in [-1, 0], y in [-1, 1]`,
    // `param0 = (u, v)` in `[0, 0.5] x [0.5, 1.0]` (32 of 64 texels).
    // Rectangle 1 (`v0 = 3..5`, right angle at corner 1): top-right quarter `x in [0, 1], y in [0, 1]`,
    // `param0 = (u, v)` in `[0.5, 1.0] x [0.0, 0.5]` (16 of 64 texels).
    const TILE_RECTS_BY_ID: &[u32] = &[
        0x7E10_0280, // v_mov_b32 v8, 0.0
        0x7E12_02F0, // v_mov_b32 v9, 0.5
        0x7E14_02F3, // v_mov_b32 v10, -1.0
        0x7E16_02F2, // v_mov_b32 v11, 1.0
        0x7E04_030A, // v_mov_b32 v2, v10 (-1.0)
        0x7E06_030A, // v_mov_b32 v3, v10 (-1.0)
        0x7E0E_0308, // v_mov_b32 v7, v8  (0.0)
        0x7E18_0309, // v_mov_b32 v12, v9 (0.5)
        0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
        0x0204_1102, // v_cndmask_b32 v2, v2, v8, vcc
        0x020E_1307, // v_cndmask_b32 v7, v7, v9, vcc
        0x7D84_0082, // v_cmp_eq_u32 vcc, 2, v0
        0x0206_1703, // v_cndmask_b32 v3, v3, v11, vcc
        0x0218_170C, // v_cndmask_b32 v12, v12, v11, vcc
        0x7D84_0083, // v_cmp_eq_u32 vcc, 3, v0
        0x0204_1102, // v_cndmask_b32 v2, v2, v8, vcc
        0x0206_1703, // v_cndmask_b32 v3, v3, v11, vcc
        0x020E_1307, // v_cndmask_b32 v7, v7, v9, vcc
        0x0218_110C, // v_cndmask_b32 v12, v12, v8, vcc
        0x7D84_0084, // v_cmp_eq_u32 vcc, 4, v0
        0x0204_1702, // v_cndmask_b32 v2, v2, v11, vcc
        0x0206_1703, // v_cndmask_b32 v3, v3, v11, vcc
        0x020E_1707, // v_cndmask_b32 v7, v7, v11, vcc
        0x0218_110C, // v_cndmask_b32 v12, v12, v8, vcc
        0x7D84_0085, // v_cmp_eq_u32 vcc, 5, v0
        0x0204_1702, // v_cndmask_b32 v2, v2, v11, vcc
        0x0206_1103, // v_cndmask_b32 v3, v3, v8, vcc
        0x020E_1707, // v_cndmask_b32 v7, v7, v11, vcc
        0x0218_130C, // v_cndmask_b32 v12, v12, v9, vcc
        0x7E08_0280, // v_mov_b32 v4, 0
        0x7E0C_02F2, // v_mov_b32 v6, 1.0
        0xF800_0203, 0x0000_0C07, // exp param0 v7, v12
        0xF800_08CF, 0x0604_0302, // exp pos0 v2, v3, v4, v6 done
        0xBF81_0000, // s_endpgm
    ];
    const INTERP_UV_PS: &[u32] = &[
        0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
        0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
        0xC80C_0100, // v_interp_p1_f32 v3, v0, attr0.y
        0xC80D_0101, // v_interp_p2_f32 v3, v1, attr0.y
        0x7E08_02F0, // v_mov_b32 v4, 0.5
        0x7E0A_02F2, // v_mov_b32 v5, 1.0
        0xF800_180F, 0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
        0xBF81_0000, // s_endpgm
    ];

    let Some(mut t) = title() else { return };
    let mut meta = [0u8; 0x10];
    meta[12] = 0x40;
    let vs_bytes: Vec<_> = TILE_RECTS_BY_ID.iter().flat_map(|w| w.to_le_bytes()).collect();
    let vs_c = victory_shader::emit::graphics::container(HwStage::Vs, &vs_bytes, &meta).unwrap();
    let vs = shaders::translate_rect(&vs_c).unwrap();
    let ps = translate(HwStage::Ps, INTERP_UV_PS);
    let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], true);
    let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &state, "tile_rect") else {
        panic!("{:?}", t.shaders.refused);
    };

    // 1. Half-size tile alone (`1` instance = 3 console vertices): exact 32 of 64 texels (`x < 4`).
    let (color, mut list) = list_into(&t, &pso, binding.clone());
    list.set_graphics_pipeline(&pso, Topology::TriangleStrip);
    list.draw_instanced(4, 1, 0, 0).unwrap();
    super::run(&t.host, &[&list]);
    let half = pixels(&t, &color);
    let half_lit = half.iter().filter(|p| p[3] == 255).count();
    assert_eq!(half_lit, 32, "half-size tile must light 32 of 64 texels, got {half_lit}");
    for y in 0..8usize {
        for x in 0..8usize {
            let p = half[y * 8 + x];
            if x < 4 {
                let want_u = ((x as f32 + 0.5) / 8.0 * 255.0).round() as i32;
                let want_v = ((1.0 - (y as f32 + 0.5) / 16.0) * 255.0).round() as i32;
                assert!((p[0] as i32 - want_u).abs() <= 2 && (p[1] as i32 - want_v).abs() <= 2 && p[3] == 255, "half ({x},{y})={p:?}");
            } else {
                assert_eq!(p, [0; 4], "half outside ({x},{y})={p:?}");
            }
        }
    }

    // 2. Both tiles (`2` instances = 6 console vertices), checked for both direct and indirect draws:
    //    32 + 16 = 48 texels lit, with the bottom-right 4x4 untouched.
    let pool = super::Pool::new(&t.host, 1);
    pool.write(0, &[6u32, 1, 0, 0]);
    for indirect in [false, true] {
        let (color, mut list) = list_into(&t, &pso, binding.clone());
        list.set_graphics_pipeline(&pso, Topology::TriangleStrip);
        if indirect {
            list.draw_rects_indirect(pool.va(0), None).unwrap();
        } else {
            list.draw_instanced(4, 2, 0, 0).unwrap();
        }
        super::run(&t.host, &[&list]);
        let got = pixels(&t, &color);
        let count_lit = got.iter().filter(|p| p[3] == 255).count();
        assert_eq!(count_lit, 48, "indirect={indirect}: expected 48 of 64 texels, got {count_lit}");
        for y in 0..8usize {
            for x in 0..8usize {
                let p = got[y * 8 + x];
                if x < 4 {
                    assert_eq!(p, half[y * 8 + x], "indirect={indirect} rect0 ({x},{y})");
                } else if y < 4 {
                    // Top-right quarter: `u` in `[0.5, 1.0]` across `x = 4..8`, `v` in `[0.0, 0.5]`
                    // top-to-bottom across `y = 0..4`.
                    let want_u = ((0.5 + (x as f32 - 3.5) / 8.0) * 255.0).round() as i32;
                    let want_v = (((y as f32 + 0.5) / 8.0) * 255.0).round() as i32;
                    assert!(
                        (p[0] as i32 - want_u).abs() <= 2 && (p[1] as i32 - want_v).abs() <= 2 && p[3] == 255,
                        "indirect={indirect} rect1 ({x},{y})={p:?} want=({want_u},{want_v})"
                    );
                } else {
                    assert_eq!(p, [0; 4], "indirect={indirect} unlit ({x},{y})={p:?}");
                }
            }
        }
    }
}

/// `ExecuteIndirect` of a rectangle list draws each record as the direct path draws its arguments
/// (the record rewritten on the GPU to four-vertex instances): a record of 3 vertices lights what
/// the direct draw does, one of 0 vertices, an instanced one, or one past the count, nothing.
#[test]
fn an_indirect_rectangle_list_draws_its_records_as_a_direct_one() {
    let Some(mut t) = title() else { return };
    let direct = {
        let (color, mut list) = rect_list_into(&mut t);
        list.draw_instanced(4, 1, 0, 0).unwrap();
        super::run(&t.host, &[&list]);
        pixels(&t, &color)
    };
    let pool = super::Pool::new(&t.host, 1);
    let count = 64;
    // Records: 3 vertices; 0 vertices; 3 vertices of 2 instances.
    pool.write(0, &[3u32, 1, 0, 0, 0, 1, 0, 0, 3, 2, 0, 0]);
    pool.write(count, &[1u32]);
    for (record, counted, draws) in [(0, None, true), (1, None, false), (2, None, false), (0, Some(0), true), (0, Some(1), false)] {
        let (color, mut list) = rect_list_into(&mut t);
        list.draw_rects_indirect(pool.va(16 * record), counted.map(|i| (pool.va(count), i))).unwrap();
        super::run(&t.host, &[&list]);
        let got = pixels(&t, &color);
        if draws {
            assert_eq!(got, direct, "record {record} {counted:?}");
        } else {
            assert!(got.iter().all(|p| *p == [0; 4]), "record {record} {counted:?}");
        }
    }
}

/// The deferred-lighting shape: a pixel program writing a float colour to mrt0 and, in mrt1.x,
/// the bits of a float it made (pi, 0x40490fdb) into an R32Uint target. Translated for the
/// integer target (`Options::integer_targets`, the mask `glue::make_host_pipe` computes from the
/// formats), Metal makes the pipeline and the target holds the bits, not 3 (a conversion).
#[test]
fn a_pixel_program_writes_exact_bits_to_an_r32uint_target() {
    const BITS: &[u32] = &[
        0x7E00_02F2, // v_mov_b32 v0, 1.0
        0x7E02_02F0, // v_mov_b32 v1, 0.5
        0x7E06_02FF, 0x4049_0FDB, // v_mov_b32 v3, pi
        0x0604_0680, // v_add_f32 v2, 0, v3
        0xF800_100F, 0x0001_0100, // exp mrt0 v0, v1, v1, v0 vm
        0xF800_1811, 0x0000_0002, // exp mrt1 v2, off, off, off done vm
        0xBF81_0000, // s_endpgm
    ];
    let formats = [Format::Rgba16Float, Format::R32Uint];
    let integer = shaders::integer_targets(&formats);
    assert_eq!(integer, 0b10);
    let bytes: Vec<_> = BITS.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut meta = [0; 0x10];
    meta[12] = 0x40;
    let c = victory_shader::emit::graphics::container(HwStage::Ps, &bytes, &meta).unwrap();
    let ms = translate(HwStage::Vs, TRIANGLE);
    let Some(mut t) = title() else { return };
    let state = Graphics::guess_base().for_draw(formats.to_vec(), None, vec![], &[], false);
    // Translated as today (float outputs), Metal refuses it.
    let float = shaders::translate(&c).unwrap();
    assert!(matches!(t.render_pipe(&ms, Some(&float), &state, "float"), Pipe::Refused));
    assert!(t.shaders.refused.iter().any(|(why, _)| why.contains("color attachment")), "{:?}", t.shaders.refused);
    let ps = shaders::translate_targets(&c, integer).unwrap();
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "integer") else {
        panic!("{:?}", t.shaders);
    };
    let target = |f| {
        let mut desc = TextureDesc::d2(f, 8, 8);
        desc.render_target = true;
        t.host.texture(&desc).unwrap()
    };
    let (color, ids) = (target(Format::Rgba16Float), target(Format::R32Uint));
    let mut list = NativeList::default();
    list.set_render_targets(&[color.clone(), ids.clone()], None);
    list.clear_render_target(&color, [0.0; 4]);
    list.clear_render_target(&ids, [0.0; 4]);
    list.set_graphics_pipeline(&pso, Topology::Triangles);
    list.set_graphics_binding(binding);
    list.dispatch_mesh(1, 1, 1).unwrap();
    super::run(&t.host, &[&list]);
    let mut p = [0u8; 8 * 8 * 4];
    t.host.backend.read_texture(&ids, 0, 0, &mut p, 8 * 4, 0);
    let words: Vec<u32> = p.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert!(words.iter().all(|&w| w == 0x4049_0FDB), "{words:x?}");
    let mut h = [0u8; 8 * 8 * 8];
    t.host.backend.read_texture(&color, 0, 0, &mut h, 8 * 8, 0);
    // 1.0, 0.5, 0.5, 1.0 as halves.
    assert_eq!(&h[..8], &[0x00, 0x3c, 0x00, 0x38, 0x00, 0x38, 0x00, 0x3c]);
}

/// The bits of input VGPR `v` an R32Uint target gets, per texel of an 8x8 target, from a pixel
/// program whose register image gives input enables `ena`.
fn exported_input(ena: u32, v: u32) -> Option<Vec<u32>> {
    let code = [
        0xF800_1801, v, // exp mrt0 v<v>, off, off, off done vm
        0xBF81_0000, // s_endpgm
    ];
    let bytes: Vec<_> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut meta = [0; 0x10];
    meta[12] = 0x40;
    let c = victory_shader::emit::graphics::container_with_inputs(HwStage::Ps, &bytes, &meta, Some(ena)).unwrap();
    let ms = translate(HwStage::Vs, TRIANGLE);
    let mut t = title()?;
    let state = Graphics::guess_base().for_draw(vec![Format::R32Uint], None, vec![], &[], false);
    let ps = shaders::translate_targets(&c, 1).unwrap();
    let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "pixel input") else {
        panic!("{:?}", t.shaders);
    };
    let mut desc = TextureDesc::d2(Format::R32Uint, 8, 8);
    desc.render_target = true;
    let ids = t.host.texture(&desc).unwrap();
    let mut list = NativeList::default();
    list.set_render_targets(std::slice::from_ref(&ids), None);
    list.clear_render_target(&ids, [1.0; 4]);
    list.set_graphics_pipeline(&pso, Topology::Triangles);
    list.set_graphics_binding(binding);
    list.dispatch_mesh(1, 1, 1).unwrap();
    super::run(&t.host, &[&list]);
    let mut p = [0u8; 8 * 8 * 4];
    t.host.backend.read_texture(&ids, 0, 0, &mut p, 8 * 4, 0);
    Some(p.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// A pixel program whose register image enables the fixed-point position (`SPI_PS_INPUT_ENA`
/// 0x8080: line stipple in v0, unread, the fixed-point position in v1) gets the pixel's x | y << 16
/// in v1: each texel holds its own coordinates. Guessed from the VGPRs it reads, v1 was the float y
/// position.
#[test]
fn a_pixel_program_reads_its_fixed_point_position_where_the_register_image_says() {
    let Some(words) = exported_input(0x8080, 1) else { return };
    let want: Vec<u32> = (0..64u32).map(|i| i % 8 | (i / 8) << 16).collect();
    assert_eq!(words, want);
}

/// Line stipple alone enabled (0x80): v0 is its coordinate, 0 for a triangle (a guess).
#[test]
fn a_pixel_program_reads_a_zero_line_stipple_coordinate_on_a_triangle() {
    let Some(words) = exported_input(0x80, 0) else { return };
    assert_eq!(words, vec![0; 64]);
}

/// A pixel program that packs its colour as the title's do. It interpolates param0 with the
/// barycentric inputs v0 and v1. Then `v_fma_mixlo_f16` writes the low half of v1 and keeps the
/// high half, which still holds the barycentric, and `v_fma_mixhi_f16` replaces that high half
/// (the test also runs the two in the other order).
/// `exp ... compr` writes the pairs. The barycentric is never read for the colour, so the program
/// translates (it was refused as "barycentrics read directly") and draws (1.0, 0.5, 1.0, 1.0).
const PACKED_BY_MIX: &[u32] = &[
    0xC80C_0000, // v_interp_p1_f32 v3, v0, attr0.x
    0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
    0xCC21_0001, 0x0201_E4F2, // v_fma_mixlo_f16 v1, 1.0, 1.0, 0
    0xCC22_0001, 0x0201_E4F0, // v_fma_mixhi_f16 v1, 0.5, 1.0, 0
    0x7E04_02FF, 0x3C00_3C00, // v_mov_b32 v2, 0x3c003c00 (two f16 1.0)
    0xF800_1C0F, 0x0000_0201, // exp mrt0 v1, v1, v2, v2 done compr vm (vsrc0 v1, vsrc1 v2)
    0xBF81_0000, // s_endpgm
];

#[test]
fn a_half_kept_by_mixlo_and_replaced_by_mixhi_draws_without_the_barycentric() {
    // The triangle program, exporting param0 for the pixel program's interpolation.
    let mut words = TRIANGLE.to_vec();
    let at = words.iter().position(|&w| w == 0xF800_08CF).unwrap(); // before exp pos0
    words.splice(at..at, [0xF800_0201, 0x0000_0006]); // exp param0 v6 (1.0)
    // Also in the other order: `mixhi` first keeps the low half, `mixlo` replaces it.
    let mut swapped = PACKED_BY_MIX.to_vec();
    let at = swapped.iter().position(|&w| w == 0xCC21_0001).unwrap();
    let (lo, hi) = (swapped[at..at + 2].to_vec(), swapped[at + 2..at + 4].to_vec());
    swapped.splice(at..at + 4, hi.into_iter().chain(lo));
    let ms = translate(HwStage::Vs, &words);
    let Some(mut t) = title() else { return };
    for (order, ps_words) in [("mixlo, mixhi", PACKED_BY_MIX), ("mixhi, mixlo", &swapped[..])] {
        let ps = translate(HwStage::Ps, ps_words);
        let state = Graphics::guess_base().for_draw(vec![Format::Rgba8Unorm], None, vec![], &[], false);
        let Pipe::Render(pso, binding) = t.render_pipe(&ms, Some(&ps), &state, "mix") else {
            panic!("{order}: {:?}", t.shaders);
        };
        let (color, mut list) = list_into(&t, &pso, binding);
        list.dispatch_mesh(1, 1, 1).unwrap();
        super::run(&t.host, &[&list]);
        let got = pixels(&t, &color);
        assert!(got.iter().all(|p| *p == [255, 128, 255, 255]), "{order}: {got:?}");
    }
}
