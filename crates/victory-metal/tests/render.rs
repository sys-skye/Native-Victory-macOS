//! Render passes on Metal: a vertex and a pixel shader (our own HLSL, dxc, Metal Shader Converter)
//! draw offscreen with an input layout read from zero-copy vertex memory, a descriptor-table texture
//! and sampler in the pixel shader, a colour and a depth attachment; the pixels are read back.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use victory_metal::backend::{ComputeBackend, Dispatch, Heaps, RootArgs};
use victory_metal::metal::Metal;
use victory_metal::resources::*;

const W: usize = 16;

/// Everything one test draws with.
struct Scene<'a> {
    m: &'a Metal,
    pipe: <Metal as GraphicsBackend>::RenderPipeline,
    arena: Arena,
    res: <Metal as ComputeBackend>::Heap,
    smp: <Metal as ComputeBackend>::Heap,
    color: <Metal as ResourceBackend>::Texture,
    depth: <Metal as ResourceBackend>::Texture,
}

type Vertex = [f32; 8];

fn v(x: f32, y: f32, c: [f32; 4]) -> Vertex {
    [x, y, 0.5, 0.5, c[0], c[1], c[2], c[3]]
}

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

impl<'a> Scene<'a> {
    fn pipeline(
        m: &Metal,
        depth_test: Option<(bool, Compare)>,
    ) -> <Metal as GraphicsBackend>::RenderPipeline {
        let attrs = [
            VertexAttr {
                format: VertexFormat::Float2,
                slot: 0,
                offset: 0,
            },
            VertexAttr {
                format: VertexFormat::Float2,
                slot: 0,
                offset: 8,
            },
            VertexAttr {
                format: VertexFormat::Float4,
                slot: 0,
                offset: 16,
            },
        ];
        m.create_render_pipeline(&RenderPipelineDesc {
            vs: &fixture("tri_vs"),
            ps: Some(&fixture("tri_ps")),
            vs_layout: None,
            ps_layout: None,
            attributes: &attrs,
            streams: &[VertexStream {
                slot: 0,
                stride: 32,
                per_instance: false,
            }],
            color: &[Format::Rgba8Unorm],
            depth: Some(Format::D32Float),
            depth_test,
            cull: Cull::None,
            ..Default::default()
        })
        .unwrap()
    }

    fn new(m: &'a Metal, depth_test: Option<(bool, Compare)>) -> Scene<'a> {
        let pipe = Self::pipeline(m, depth_test);
        let mut rt = TextureDesc::d2(Format::Rgba8Unorm, W, W);
        rt.render_target = true;
        let color = m.create_texture(&rt).unwrap();
        rt.format = Format::D32Float;
        let depth = m.create_texture(&rt).unwrap();
        // The pixel shader multiplies by a 1x1 white texture sampled through the heaps.
        let white = m
            .create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1))
            .unwrap();
        m.upload_texture(&white, 0, 0, &[255; 4], 4, 0);
        let s = m
            .create_sampler(&SamplerDesc {
                min: Filter::Nearest,
                mag: Filter::Nearest,
                mip: Filter::Nearest,
                address: [Address::Clamp; 3],
                border_color: BorderColor::TransparentBlack,
                max_anisotropy: 1,
            })
            .unwrap();
        let res = m.create_heap(HeapKind::Resource, 8).unwrap();
        let smp = m.create_heap(HeapKind::Sampler, 2).unwrap();
        m.write_texture(&res, 0, &white, 0.0);
        m.write_sampler(&smp, 0, &s, 0.0);
        Scene {
            m,
            pipe,
            arena: Arena::new(m, 2),
            res,
            smp,
            color,
            depth,
        }
    }

    fn root(&self, z: f32) -> RootArgs {
        RootArgs::default()
            .constants(&[z.to_bits(), 0, 0, 0])
            .address(self.m.heap_address(&self.res, 0))
            .address(self.m.heap_address(&self.smp, 0))
    }

    fn draw(&self, z: f32, call: DrawCall, verts: usize, idx: bool) -> Draw<'_, Metal> {
        Draw {
            pipeline: &self.pipe,
            topology: Topology::Triangles,
            state: DrawState::default(),
            root: self.root(z),
            fragment_root: None,
            heaps: Heaps {
                resource: Some(&self.res),
                sampler: Some(&self.smp),
            },
            resources: vec![],
            vertex_buffers: vec![(0, &self.arena.buf, self.arena.at(verts))],
            index_buffer: idx.then(|| (&self.arena.buf, self.arena.at(4096) as *const u8)),
            call,
            indirect: None,
            count: None,
        }
    }

    fn pass<'b>(&'b self, draws: Vec<Draw<'b, Metal>>, clear: [f32; 4]) -> Pass<'b, Metal> {
        Pass::Render(RenderPass {
            color: vec![ColorTarget {
                texture: &self.color,
                clear: Some(clear),
            }],
            depth: Some(DepthTarget {
                texture: &self.depth,
                clear: Some(1.0),
                clear_stencil: Some(0),
            }),
            draws,
        })
    }

    fn submit(&self, passes: &[Pass<'_, Metal>]) {
        let q = self.m.create_queue().unwrap();
        let f = self.m.create_fence(0).unwrap();
        self.m
            .submit_passes(
                &q,
                &PassSubmission {
                    waits: &[],
                    passes,
                    signals: &[(&f, 1)],
                },
            )
            .unwrap();
        assert!(self.m.wait(&f, 1, 10_000), "GPU did not finish");
    }

    fn pixels(&self) -> Vec<[u8; 4]> {
        let mut out = vec![0u8; W * W * 4];
        self.m.read_texture(&self.color, 0, 0, &mut out, W * 4, 0);
        out.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
    }
}

fn px(p: &[[u8; 4]], x: usize, y: usize) -> [u8; 4] {
    p[y * W + x]
}

#[test]
fn triangle_offscreen() {
    let Some(m) = backend() else { return };
    let s = Scene::new(&m, Some((true, Compare::Less)));
    s.arena.write(
        0,
        &[v(-0.5, -0.5, RED), v(0.5, -0.5, RED), v(0.0, 0.5, RED)],
    );
    let d = s.draw(
        0.5,
        DrawCall::Arrays {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
        0,
        false,
    );
    s.submit(&[s.pass(vec![d], [0.0, 0.0, 1.0, 1.0])]);
    let p = s.pixels();
    assert_eq!(px(&p, 8, 8), [255, 0, 0, 255], "inside the triangle");
    assert_eq!(px(&p, 0, 0), [0, 0, 255, 255], "cleared corner");
    assert_eq!(px(&p, 15, 15), [0, 0, 255, 255], "cleared corner");
    // Direct 'up' check: the apex is at y = +0.5, which is the upper rows of the image (row 4),
    // the base at y = -0.5 the lower rows (row 12): wide at the bottom, narrow at the top.
    let width = |y| (0..W).filter(|x| px(&p, *x, y) == [255, 0, 0, 255]).count();
    assert!(
        width(11) > width(5) && width(5) > 0,
        "{} {}",
        width(11),
        width(5)
    );
}

#[test]
fn indexed_draw_with_depth_test_and_depth_sampling() {
    let Some(m) = backend() else { return };
    let s = Scene::new(&m, Some((true, Compare::Less)));
    // Left half: an indexed quad (4 vertices, 6 u16 indices) at z 0.25, green. Drawn second, a
    // full-screen red triangle at z 0.75 is behind it where they overlap and in front of nothing else.
    s.arena.write(
        0,
        &[
            v(-1.0, -1.0, GREEN),
            v(0.0, -1.0, GREEN),
            v(0.0, 1.0, GREEN),
            v(-1.0, 1.0, GREEN),
        ],
    );
    s.arena.write(4096, &[0u16, 1, 2, 0, 2, 3]);
    s.arena.write(
        512,
        &[v(-1.0, -1.0, RED), v(3.0, -1.0, RED), v(-1.0, 3.0, RED)],
    );
    let quad = s.draw(
        0.25,
        DrawCall::Indexed {
            index_count: 6,
            instance_count: 1,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
            wide: false,
        },
        0,
        true,
    );
    let tri = s.draw(
        0.75,
        DrawCall::Arrays {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
        512,
        false,
    );

    // After the render pass, a compute pass samples the depth attachment itself (Metal cannot view
    // Depth32Float as R32Float: "not castable"), as `Texture2D<float4>`; depth is in .x.
    let cells = W * W;
    let out = Arena::new(&m, 1);
    let p = m.create_compute_pipeline(&fixture("sample2d")).unwrap();
    let res = m.create_heap(HeapKind::Resource, 2).unwrap();
    m.write_texture(&res, 0, &s.depth, 0.0);
    m.write_buffer(&res, 1, &out.buf, out.at(0), (cells * 16) as u32);
    let sm = m
        .create_sampler(&SamplerDesc {
            min: Filter::Nearest,
            mag: Filter::Nearest,
            mip: Filter::Nearest,
            address: [Address::Clamp; 3],
            border_color: BorderColor::TransparentBlack,
            max_anisotropy: 1,
        })
        .unwrap();
    m.write_sampler(&s.smp, 1, &sm, 0.0);
    let sample = [Dispatch {
        pipeline: &p,
        root: RootArgs::default()
            .constants(&[W as u32, W as u32])
            .address(m.heap_address(&res, 0))
            .address(m.heap_address(&s.smp, 1)),
        resources: vec![],
        heaps: Heaps {
            resource: Some(&res),
            sampler: Some(&s.smp),
        },
        groups: [W / 4, W / 4, 1],
        threads_per_group: [4, 4, 1],
        indirect: None,
        count: None,
    }];
    s.submit(&[
        s.pass(vec![quad, tri], [0.0, 0.0, 0.0, 1.0]),
        Pass::Compute(&sample),
    ]);

    let p = s.pixels();
    assert_eq!(px(&p, 4, 8), [0, 255, 0, 255], "quad (z 0.25) in front");
    assert_eq!(
        px(&p, 12, 8),
        [255, 0, 0, 255],
        "triangle where only it covers"
    );
    let depth: Vec<f32> = out.read(0, cells * 4);
    let d = |x: usize, y: usize| depth[(y * W + x) * 4];
    assert!((d(4, 8) - 0.25).abs() < 1e-6, "depth left {}", d(4, 8));
    assert!((d(12, 8) - 0.75).abs() < 1e-6, "depth right {}", d(12, 8));
}

#[test]
fn depth_test_rejects_a_draw_behind() {
    let Some(m) = backend() else { return };
    let s = Scene::new(&m, Some((true, Compare::Less)));
    // The same full-screen triangle twice: green at z 0.25 first, red at z 0.75 second. Red fails.
    s.arena.write(
        0,
        &[
            v(-1.0, -1.0, GREEN),
            v(3.0, -1.0, GREEN),
            v(-1.0, 3.0, GREEN),
        ],
    );
    s.arena.write(
        512,
        &[v(-1.0, -1.0, RED), v(3.0, -1.0, RED), v(-1.0, 3.0, RED)],
    );
    let call = DrawCall::Arrays {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    let (a, b) = (s.draw(0.25, call, 0, false), s.draw(0.75, call, 512, false));
    s.submit(&[s.pass(vec![a, b], [0.0, 0.0, 0.0, 1.0])]);
    assert!(s.pixels().iter().all(|p| *p == [0, 255, 0, 255]));
}

#[test]
fn switching_to_disabled_depth_clears_the_previous_test() {
    let Some(m) = backend() else { return };
    let s = Scene::new(&m, Some((true, Compare::Never)));
    let disabled = Scene::pipeline(&m, None);
    s.arena.write(
        0,
        &[v(-1.0, -1.0, RED), v(3.0, -1.0, RED), v(-1.0, 3.0, RED)],
    );
    let call = DrawCall::Arrays {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    let first = s.draw(0.5, call, 0, false);
    let mut second = s.draw(0.5, call, 0, false);
    second.pipeline = &disabled;
    second.state.scissor = Some([0, 0, 8, 16]);
    s.submit(&[s.pass(vec![first, second], [0.0, 0.0, 0.0, 1.0])]);

    let p = s.pixels();
    assert_eq!(
        px(&p, 12, 8),
        [0, 0, 0, 255],
        "Compare::Never rejects the first draw"
    );
    assert_eq!(
        px(&p, 4, 8),
        [255, 0, 0, 255],
        "the depth-disabled draw reaches the target"
    );
    let mut depth = [0u8; W * W * 4];
    m.read_texture(&s.depth, 0, 0, &mut depth, W * 4, 0);
    assert!(
        depth
            .as_chunks::<4>()
            .0
            .iter()
            .all(|d| f32::from_ne_bytes(*d) == 1.0),
        "the depth-disabled draw preserves the clear depth"
    );
}

#[test]
fn a_count_buffer_limits_the_indirect_draws_that_run() {
    let Some(m) = backend() else { return };
    for (count, red) in [(1u32, false), (2, true), (0, false)] {
        let s = Scene::new(&m, Some((true, Compare::Less)));
        // The records of `indirect_draws_read_their_records_from_title_memory`: a green quad
        // (record 0, indexed) and a red triangle (record 1) over the right half, with the count of
        // records to run in title memory too (D3D12 runs min(count, records)).
        s.arena.write(
            0,
            &[
                v(-1.0, -1.0, GREEN),
                v(0.0, -1.0, GREEN),
                v(0.0, 1.0, GREEN),
                v(-1.0, 1.0, GREEN),
            ],
        );
        s.arena.write(4096, &[0u16, 1, 2, 0, 2, 3]);
        s.arena.write(
            512,
            &[v(-1.0, -1.0, RED), v(3.0, -1.0, RED), v(-1.0, 3.0, RED)],
        );
        s.arena.write(6144, &[6u32, 1, 0, 0, 0]);
        s.arena.write(6164, &[3u32, 1, 0, 0]);
        s.arena.write(6200, &[count]);
        let none = DrawCall::Arrays {
            vertex_count: 0,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        let record = |at: usize| Some((&s.arena.buf, s.arena.at(at) as *const u8));
        let counted = |i: u32| Some((&s.arena.buf, s.arena.at(6200) as *const u8, i));
        let indexed = DrawCall::Indexed {
            index_count: 0,
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
            wide: false,
        };
        let mut quad = s.draw(0.25, indexed, 0, true);
        (quad.indirect, quad.count) = (record(6144), counted(0));
        let mut tri = s.draw(0.75, none, 512, false);
        (tri.indirect, tri.count) = (record(6164), counted(1));
        s.submit(&[s.pass(vec![quad, tri], [0.0, 0.0, 0.0, 1.0])]);
        let p = s.pixels();
        let green = if count > 0 {
            [0, 255, 0, 255]
        } else {
            [0, 0, 0, 255]
        };
        assert_eq!(px(&p, 4, 8), green, "count {count}: the quad");
        let right = if red {
            [255, 0, 0, 255]
        } else {
            [0, 0, 0, 255]
        };
        assert_eq!(px(&p, 12, 8), right, "count {count}: the triangle");
    }
}

#[test]
fn indirect_draws_read_their_records_from_title_memory() {
    let Some(m) = backend() else { return };
    let s = Scene::new(&m, Some((true, Compare::Less)));
    // As `indexed_draw_with_depth_test_and_depth_sampling`, with the counts in D3D12 argument
    // records (`D3D12_DRAW_INDEXED_ARGUMENTS`, `D3D12_DRAW_ARGUMENTS`) the GPU reads. A third
    // record with no instances draws a red triangle over everything, or nothing.
    s.arena.write(
        0,
        &[
            v(-1.0, -1.0, GREEN),
            v(0.0, -1.0, GREEN),
            v(0.0, 1.0, GREEN),
            v(-1.0, 1.0, GREEN),
        ],
    );
    s.arena.write(4096, &[0u16, 1, 2, 0, 2, 3]);
    s.arena.write(
        512,
        &[v(-1.0, -1.0, RED), v(3.0, -1.0, RED), v(-1.0, 3.0, RED)],
    );
    s.arena.write(6144, &[6u32, 1, 0, 0, 0]);
    s.arena.write(6164, &[3u32, 1, 0, 0]);
    s.arena.write(6180, &[3u32, 0, 0, 0]);
    let none = DrawCall::Arrays {
        vertex_count: 0,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    let record = |at: usize| Some((&s.arena.buf, s.arena.at(at) as *const u8));
    let mut quad = s.draw(
        0.25,
        DrawCall::Indexed {
            index_count: 0,
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
            wide: false,
        },
        0,
        true,
    );
    quad.indirect = record(6144);
    let mut tri = s.draw(0.75, none, 512, false);
    tri.indirect = record(6164);
    let mut skipped = s.draw(0.0, none, 512, false);
    skipped.indirect = record(6180);
    s.submit(&[s.pass(vec![quad, tri, skipped], [0.0, 0.0, 0.0, 1.0])]);
    let p = s.pixels();
    assert_eq!(px(&p, 4, 8), [0, 255, 0, 255], "indexed record (z 0.25) in front");
    assert_eq!(px(&p, 12, 8), [255, 0, 0, 255], "plain record where only it covers");
}

#[test]
fn a_refused_pass_keeps_its_fence_signal_and_the_passes_after_it() {
    // The submission commits even when a pass cannot be encoded (its fence signals and the value
    // writes other queues wait on must still go out), and the passes after it still run.
    let Some(m) = backend() else { return };
    let s = Scene::new(&m, Some((true, Compare::Less)));
    s.arena.write(
        0,
        &[v(-0.5, -0.5, GREEN), v(0.5, -0.5, GREEN), v(0.0, 0.5, GREEN)],
    );
    let args = || DrawCall::Arrays {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    // An indexed record without an index buffer: the host refuses that pass.
    s.arena.write(6144, &[3u32, 1, 0, 0, 0]);
    let indexed = DrawCall::Indexed {
        index_count: 0,
        instance_count: 0,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
        wide: false,
    };
    let mut bad_draw = s.draw(0.5, indexed, 0, false);
    bad_draw.indirect = Some((&s.arena.buf, s.arena.at(6144) as *const u8));
    let bad = s.pass(vec![bad_draw], [0.0, 0.0, 1.0, 1.0]);
    let good = s.pass(vec![s.draw(0.5, args(), 0, false)], [0.0, 0.0, 0.0, 1.0]);
    let q = m.create_queue().unwrap();
    let f = m.create_fence(0).unwrap();
    let refused = m.submit_passes(
        &q,
        &PassSubmission {
            waits: &[],
            passes: &[bad, good],
            signals: &[(&f, 1)],
        },
    );
    let err = refused.expect_err("the indexed record without an index buffer is refused");
    assert_eq!(err.refused, [0], "only the bad pass was refused");
    assert!(err.committed, "the command buffer still committed");
    assert!(m.wait(&f, 1, 10_000), "the fence signal still landed");
    let p = s.pixels();
    assert!(
        p.iter().any(|c| c[1] > 100),
        "the pass after the refused one drew: {:?}",
        &p[..4]
    );
}

#[test]
fn a_pass_refused_mid_encoding_closes_its_encoder() {
    // An indexed record without an index buffer is refused after the render encoder is open; the
    // encoder must still be ended, or the next pass cannot open its own.
    let Some(m) = backend() else { return };
    let s = Scene::new(&m, Some((true, Compare::Less)));
    s.arena.write(0, &[v(-0.5, -0.5, GREEN), v(0.5, -0.5, GREEN), v(0.0, 0.5, GREEN)]);
    s.arena.write(6144, &[3u32, 1, 0, 0, 0]);
    let indexed = DrawCall::Indexed {
        index_count: 0,
        instance_count: 0,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
        wide: false,
    };
    let mut bad = s.draw(0.5, indexed, 0, false);
    bad.indirect = Some((&s.arena.buf, s.arena.at(6144) as *const u8));
    let args = DrawCall::Arrays {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    let q = m.create_queue().unwrap();
    let f = m.create_fence(0).unwrap();
    let refused = m.submit_passes(
        &q,
        &PassSubmission {
            waits: &[],
            passes: &[
                s.pass(vec![bad], [0.0, 0.0, 1.0, 1.0]),
                s.pass(vec![s.draw(0.5, args, 0, false)], [0.0, 0.0, 0.0, 1.0]),
            ],
            signals: &[(&f, 1)],
        },
    );
    let err = refused.expect_err("the indexed record without an index buffer is refused");
    assert_eq!(err.refused, [0], "only the bad pass was refused");
    assert!(err.committed, "the command buffer still committed");
    assert!(m.wait(&f, 1, 10_000), "the fence signal still landed");
    assert!(s.pixels().iter().any(|c| c[1] > 100), "the pass after the refused one drew");
}

/// A title's present pass: a target drawn in one pass is sampled through a descriptor table by
/// the pixel shader of the next pass, in the same submission (both residency modes).
#[test]
fn a_target_drawn_in_one_pass_is_sampled_by_the_next() {
    for m in backends() {
        let s = Scene::new(&m, None);
        const WHITE: [f32; 4] = [1.0; 4];
        // Pass 1: green over all of `s.color` (times the white texture).
        s.arena.write(0, &[v(-1.0, -1.0, GREEN), v(3.0, -1.0, GREEN), v(-1.0, 3.0, GREEN)]);
        // Pass 2: white over all of `out`, times `s.color` sampled at its centre.
        s.arena.write(512, &[v(-1.0, -1.0, WHITE), v(3.0, -1.0, WHITE), v(-1.0, 3.0, WHITE)]);
        let call = DrawCall::Arrays {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        };
        let mut rt = TextureDesc::d2(Format::Rgba8Unorm, W, W);
        rt.render_target = true;
        let out = m.create_texture(&rt).unwrap();
        m.write_texture(&s.res, 1, &s.color, 0.0);
        let present = Draw {
            pipeline: &s.pipe,
            topology: Topology::Triangles,
            state: DrawState::default(),
            root: RootArgs::default()
                .constants(&[0.5f32.to_bits(), 0, 0, 0])
                .address(m.heap_address(&s.res, 1))
                .address(m.heap_address(&s.smp, 0)),
            fragment_root: None,
            heaps: Heaps {
                resource: Some(&s.res),
                sampler: Some(&s.smp),
            },
            resources: vec![],
            vertex_buffers: vec![(0, &s.arena.buf, s.arena.at(512))],
            index_buffer: None,
            call,
            indirect: None,
            count: None,
        };
        let second = Pass::Render(RenderPass {
            color: vec![ColorTarget {
                texture: &out,
                clear: Some([0.0, 0.0, 1.0, 1.0]),
            }],
            depth: Some(DepthTarget {
                texture: &s.depth,
                clear: Some(1.0),
                clear_stencil: Some(0),
            }),
            draws: vec![present],
        });
        s.submit(&[s.pass(vec![s.draw(0.5, call, 0, false)], [1.0, 0.0, 0.0, 1.0]), second]);
        assert!(s.pixels().iter().all(|p| *p == [0, 255, 0, 255]), "pass 1 drew green");
        let mut px = vec![0u8; W * W * 4];
        m.read_texture(&out, 0, 0, &mut px, W * 4, 0);
        assert_eq!(&px[(8 * W + 8) * 4..][..4], &[0, 255, 0, 255], "pass 2 sampled pass 1's green");
    }
}


/// The present pass as translated shaders bind it: pass 1 draws green into a target, then a pixel
/// shader takes that target straight from the descriptor heap at `index` (SM 6.6, no table) and
/// samples it with static sampler `s0` (`sampler`, a `D3D12_STATIC_SAMPLER_DESC`'s ten words).
fn heap_sampled(m: &Metal, ps: &str, index: usize, sampler: [u32; 10]) -> [u8; 4] {
    use victory_metal::backend::{RootLayout, RootParam};
    let s = Scene::new(m, None);
    const WHITE: [f32; 4] = [1.0; 4];
    s.arena.write(
        0,
        &[
            v(-1.0, -1.0, GREEN),
            v(3.0, -1.0, GREEN),
            v(-1.0, 3.0, GREEN),
        ],
    );
    s.arena.write(
        512,
        &[
            v(-1.0, -1.0, WHITE),
            v(3.0, -1.0, WHITE),
            v(-1.0, 3.0, WHITE),
        ],
    );
    let call = DrawCall::Arrays {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    let layout = RootLayout {
        params: vec![RootParam::Constants {
            register: 0,
            space: 0,
            dwords: 4,
        }],
        bounds_checked: true,
        static_samplers: vec![sampler],
    };
    let attrs = [
        VertexAttr {
            format: VertexFormat::Float2,
            slot: 0,
            offset: 0,
        },
        VertexAttr {
            format: VertexFormat::Float2,
            slot: 0,
            offset: 8,
        },
        VertexAttr {
            format: VertexFormat::Float4,
            slot: 0,
            offset: 16,
        },
    ];
    let pipe = m
        .create_render_pipeline(&RenderPipelineDesc {
            vs: &fixture("tri_vs"),
            ps: Some(&fixture(ps)),
            vs_layout: None,
            ps_layout: Some(&layout),
            attributes: &attrs,
            streams: &[VertexStream {
                slot: 0,
                stride: 32,
                per_instance: false,
            }],
            color: &[Format::Rgba8Unorm],
            depth: Some(Format::D32Float),
            depth_test: None,
            cull: Cull::None,
            ..Default::default()
        })
        .unwrap();
    let mut rt = TextureDesc::d2(Format::Rgba8Unorm, W, W);
    rt.render_target = true;
    let out = m.create_texture(&rt).unwrap();
    let heap = m.create_heap(HeapKind::Resource, index + 8).unwrap();
    m.write_texture(&heap, index, &s.color, 0.0);
    let present = Draw {
        pipeline: &pipe,
        topology: Topology::Triangles,
        state: DrawState::default(),
        root: s.root(0.5),
        fragment_root: Some(RootArgs::default().constants(&[index as u32, 0, 0, 0])),
        heaps: Heaps {
            resource: Some(&heap),
            sampler: Some(&s.smp),
        },
        resources: vec![],
        vertex_buffers: vec![(0, &s.arena.buf, s.arena.at(512))],
        index_buffer: None,
        call,
        indirect: None,
        count: None,
    };
    let second = Pass::Render(RenderPass {
        color: vec![ColorTarget {
            texture: &out,
            clear: Some([0.0, 0.0, 1.0, 1.0]),
        }],
        depth: Some(DepthTarget {
            texture: &s.depth,
            clear: Some(1.0),
            clear_stencil: Some(0),
        }),
        draws: vec![present],
    });
    s.submit(&[
        s.pass(vec![s.draw(0.5, call, 0, false)], [1.0, 0.0, 0.0, 1.0]),
        second,
    ]);
    let mut px = vec![0u8; W * W * 4];
    m.read_texture(&out, 0, 0, &mut px, W * 4, 0);
    px[(8 * W + 8) * 4..][..4].try_into().unwrap()
}

/// Static samplers come from a table the backend fills (the converter does not build them into
/// the shader): without it the sample read zeros on M1 and M2 GPUs, the title's black window.
#[test]
fn a_static_sampler_samples_a_heap_texture() {
    // Point and clamp; anisotropic 4x with wrap and LOD 0 to 12, as the title's present pass.
    let point = [0, 3, 3, 3, 0, 1, 1, 0, 0, f32::MAX.to_bits()];
    let aniso = [0x55, 1, 1, 1, 0, 4, 1, 0, 0, 12.0f32.to_bits()];
    for m in backends() {
        for smp in [point, aniso] {
            for index in [3, 1_010_071] {
                assert_eq!(
                    heap_sampled(&m, "heap_ps", index, smp),
                    [0, 255, 0, 255],
                    "{smp:?} {index}"
                );
            }
        }
    }
}

/// Where the converter puts the static sampler table: one pointer after the root parameters.
#[test]
fn the_static_sampler_table_follows_the_root_parameters() {
    use victory_metal::backend::{RootLayout, RootParam};
    if backend().is_none() {
        return;
    }
    let msc = victory_metal::msc::Msc::open_default();
    let point = [0, 3, 3, 3, 0, 1, 1, 0, 0, f32::MAX.to_bits()];
    let constants = |dwords| RootParam::Constants {
        register: 0,
        space: 0,
        dwords,
    };
    for (params, samplers, at) in [
        (vec![constants(4)], 1, 16),
        (vec![constants(4)], 2, 16),
        (
            vec![
                constants(12),
                RootParam::Srv {
                    register: 0,
                    space: 9,
                },
            ],
            2,
            56,
        ),
        (vec![], 1, 0),
    ] {
        let l = RootLayout {
            params,
            bounds_checked: true,
            static_samplers: vec![point; samplers],
        };
        let table = msc
            .locations(&l)
            .unwrap()
            .into_iter()
            .find(|r| r.0 == 0 && r.1 == u32::MAX);
        assert_eq!(table.map(|r| (r.3, r.4)), Some((at, 8)), "{:?}", l.params);
    }
}

#[test]
fn zero_count_draws_do_not_fail_validation() {
    for m in backends() {
        let s = Scene::new(&m, None);
        let d1 = s.draw(
            0.5,
            DrawCall::Arrays {
                vertex_count: 0,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            0,
            false,
        );
        let d2 = s.draw(
            0.5,
            DrawCall::Arrays {
                vertex_count: 3,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            },
            0,
            false,
        );
        let d3 = s.draw(
            0.5,
            DrawCall::Indexed {
                index_count: 0,
                instance_count: 1,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
                wide: false,
            },
            0,
            true,
        );
        let d4 = s.draw(
            0.5,
            DrawCall::Indexed {
                index_count: 3,
                instance_count: 0,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
                wide: false,
            },
            0,
            true,
        );
        let mut d5 = s.draw(
            0.5,
            DrawCall::Arrays {
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            0,
            false,
        );
        d5.state.viewport = Some(Viewport {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 16.0,
            min_depth: 0.0,
            max_depth: 1.0,
        });
        s.submit(&[s.pass(vec![d1, d2, d3, d4, d5], [0.0; 4])]);
    }
}
