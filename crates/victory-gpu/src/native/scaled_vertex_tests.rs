//! Synthetic GFX10 vertex fetches through translation, guessed input layouts and Metal readback.
use super::{Pool, host};
use crate::native::{
    list::NativeList,
    shaders, state,
    title::{Pipe, Title},
};
use victory_backend::backend::ComputeBackend;
use victory_backend::resources::{Format, ResourceBackend, TextureDesc, Topology};
use victory_shader::{emit::graphics, native::HwStage};

fn translate(stage: HwStage, words: &[u32], rect: bool) -> victory_shader::emit::HostShader {
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let container = graphics::container(stage, &bytes, &[0; 0x10]).unwrap();
    shaders::translate_extra(&container, rect, 0, [0; 32]).unwrap()
}

// Assembled with llvm-mc -triple=amdgcn -mcpu=gfx1030 -show-encoding.
fn vertex(format: u32, count: u32, rect: bool) -> victory_shader::emit::HostShader {
    translate(
        HwStage::Vs,
        &[
            0xF408_0100,
            0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xEA6B_2000,
            0x8001_0100, // tbuffer_load_format_xyzw v[1:4], ... FLOAT
            0x7E0A_02FF,
            0x4188_0000, // v5 = 17.0 (unloaded channels stay unchanged)
            0x7E0C_02FF,
            0x4188_0000,
            0x7E0E_02FF,
            0x4188_0000,
            0x7E10_02FF,
            0x4188_0000,
            0xE800_2010 | format << 19 | (count - 1) << 16,
            0x8001_0500,
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xF800_08CF,
            0x0403_0201, // exp pos0 v1, v2, v3, v4 done
            0xF800_020F,
            0x0807_0605, // exp param0 v5, v6, v7, v8
            0xBF81_0000,
        ],
        rect,
    )
}

#[test]
fn scaled_vertex_loads_reach_float_render_targets() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 1);
    let memory = h.memory();
    let mut t = Title::new(h.backend, memory);
    let ps = translate(
        HwStage::Ps,
        &[
            0xBEFC_0300, // s_mov_b32 m0, s0
            0xC806_0002,
            0xC80A_0102,
            0xC80E_0202,
            0xC812_0302, // flat attr0.xyzw
            0xF800_180F,
            0x0403_0201, // exp mrt0 v1, v2, v3, v4 done vm
            0xBF81_0000,
        ],
        false,
    );
    let mut desc = TextureDesc::d2(Format::Rgba32Float, 4, 4);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    let queue = t.host.backend.create_queue().unwrap();
    let fence = t.host.backend.create_fence(0).unwrap();
    let mut serial = 0;
    for (format, channels) in [(3, 1), (4, 1), (16, 2), (17, 2), (58, 4), (59, 4)] {
        for count in 1..=channels {
            let vs = vertex(format, count, false);
            let reads = state::vertex_inputs(&vs.dxil).unwrap();
            let g = state::Graphics::guess_base().for_draw(
                vec![Format::Rgba32Float],
                None,
                vec![20],
                &reads,
                false,
            );
            let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &g, "scaled vertex")
            else {
                panic!("{:?}", t.shaders);
            };
            // Each channel visits all byte values, with distinct channel values to catch swaps.
            for value in 0..=255u8 {
                let data = [
                    value,
                    value.wrapping_add(67),
                    value.wrapping_add(131),
                    value.wrapping_add(193),
                ];
                for (i, xy) in [[-1.0f32, -1.0], [3.0, -1.0], [-1.0, 3.0]]
                    .iter()
                    .enumerate()
                {
                    pool.write(i * 20, &[xy[0], xy[1], 0.0, 1.0]);
                    pool.write(i * 20 + 16, &data);
                }
                let mut list = NativeList::default();
                list.set_render_targets(std::slice::from_ref(&color), None);
                list.clear_render_target(&color, [-999.0; 4]);
                list.set_graphics_pipeline(&pso, Topology::Triangles);
                list.set_graphics_binding(binding.clone());
                list.set_vertex_buffer(0, pool.va(0));
                list.draw_instanced(3, 1, 0, 0).unwrap();
                serial += 1;
                t.host
                    .execute(&queue, &[&list], &[], &[(&fence, serial)])
                    .unwrap();
                assert!(
                    t.host.backend.wait(&fence, serial, 10_000),
                    "GPU did not finish"
                );
                let mut pixels = [0u8; 4 * 4 * 16];
                t.host
                    .backend
                    .read_texture(&color, 0, 0, &mut pixels, 4 * 16, 0);
                for (channel, &byte) in data.iter().enumerate() {
                    let expected = if channel >= count as usize {
                        17.0f32
                    } else if matches!(format, 3 | 16 | 58) {
                        f32::from(byte)
                    } else {
                        f32::from(byte as i8)
                    };
                    let at = (2 * 4 + 2) * 16 + channel * 4;
                    let actual = u32::from_le_bytes(pixels[at..at + 4].try_into().unwrap());
                    assert_eq!(
                        actual,
                        expected.to_bits(),
                        "format={format} count={count} value={value} channel={channel}"
                    );
                }
            }
        }
    }
    eprintln!("verified 3584 Metal draws, all 256 byte values in each loaded channel");
}

#[test]
fn scaled_vertex_rectangles_match_explicit_float_corners() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 1);
    let memory = h.memory();
    let mut t = Title::new(h.backend, memory);
    let ps = translate(
        HwStage::Ps,
        &[
            0xBEFC_0300,
            0xC808_0000,
            0xC809_0001, // v_interp_p1/p2_f32 v2, v0/v1, attr0.x
            0xC80C_0100,
            0xC80D_0101,
            0xC810_0200,
            0xC811_0201,
            0xC814_0300,
            0xC815_0301,
            0xF800_180F,
            0x0504_0302,
            0xBF81_0000,
        ],
        false,
    );
    let mut desc = TextureDesc::d2(Format::Rgba32Float, 4, 4);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    for format in [58, 59] {
        let bytes = [[136u8, 156, 176, 196], [216, 236, 0, 20], [40, 60, 80, 100]];
        let values = bytes.map(|v| {
            v.map(|b| {
                if format == 58 {
                    f32::from(b)
                } else {
                    f32::from(b as i8)
                }
            })
        });
        let fourth: [f32; 4] = std::array::from_fn(|k| values[1][k] + values[2][k] - values[0][k]);
        // Three source corners: right angle, bottom right, top left.
        for (i, xy) in [[-1.0f32, -1.0], [1.0, -1.0], [-1.0, 1.0]]
            .iter()
            .enumerate()
        {
            pool.write(i * 20, &[xy[0], xy[1], 0.0, 1.0]);
            pool.write(i * 20 + 16, &bytes[i]);
        }
        // Independent reference: explicit four-corner strip, with already-converted floats.
        for (i, (xy, data)) in [
            ([1.0f32, -1.0], values[1]),
            ([-1.0, -1.0], values[0]),
            ([1.0, 1.0], fourth),
            ([-1.0, 1.0], values[2]),
        ]
        .iter()
        .enumerate()
        {
            pool.write(1024 + i * 32, &[xy[0], xy[1], 0.0, 1.0]);
            pool.write(1024 + i * 32 + 16, data);
        }
        let mut results = Vec::new();
        for rect in [false, true] {
            let vs = vertex(if rect { format } else { 77 }, 4, rect);
            let reads = state::vertex_inputs(&vs.dxil).unwrap();
            let g = state::Graphics::guess_base().for_draw(
                vec![Format::Rgba32Float],
                None,
                vec![if rect { 20 } else { 32 }],
                &reads,
                rect,
            );
            let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &g, "scaled rectangle")
            else {
                panic!("{:?}", t.shaders);
            };
            let mut list = NativeList::default();
            list.set_render_targets(std::slice::from_ref(&color), None);
            list.clear_render_target(&color, [-999.0; 4]);
            list.set_graphics_pipeline(&pso, Topology::TriangleStrip);
            list.set_graphics_binding(binding);
            list.set_vertex_buffer(0, pool.va(if rect { 0 } else { 1024 }));
            list.draw_instanced(4, 1, 0, 0).unwrap();
            super::run(&t.host, &[&list]);
            let mut pixels = [0u8; 4 * 4 * 16];
            t.host
                .backend
                .read_texture(&color, 0, 0, &mut pixels, 4 * 16, 0);
            assert!(
                pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|v| *v != (-999.0f32).to_le_bytes()),
                "uncovered pixel"
            );
            results.push(pixels);
        }
        assert_eq!(
            results[0], results[1],
            "format={format}: rectangle differs from explicit quad"
        );
    }
    eprintln!("verified signed and unsigned rectangle pixels against explicit float quads");
}
