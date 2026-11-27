//! The order of output stores: every component's value is made before the first store of its
//! output, for `storeOutput` (pixel and vertex programs), `storeVertexOutput` and
//! `storePrimitiveOutput` (mesh programs). Apple's Metal Shader Converter gathers the component
//! stores of one output into a vector at the first of them, so a value made between two stores
//! is used before its definition in the library it writes. Hand-assembled programs
//! (`llvm-mc -mcpu=gfx1030` encodings).

use super::container::SigElement;
use super::lower::{self, Lowered, Parts};
use crate::native::{HwStage, MeshIo, NativeProgram, StageIo, Usage, VsOutput};
use crate::{ShaderError, io, ir, resource, structure};

fn element(name: &str, index: u32, kind: u8, row: u32, cols: u8, half: bool) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: 0,
        cols,
        usage: 0,
        half,
    }
}

fn lowered(
    stage: HwStage,
    stage_io: StageIo,
    words: &[u32],
    outputs: Vec<SigElement>,
    prims: Vec<SigElement>,
) -> Result<Lowered, ShaderError> {
    let native = NativeProgram {
        stage,
        wave64: false,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: stage_io,
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs,
        prims,
        mesh_stride: 32,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        view_marks: false,
        rect_list: false,
        copy_params: [0; 32],
    })
}

/// The stores named `dx.op.<name>.*`, checked: per block and output element, no stored value is
/// made after the first store. Returns how many stores there are and how many of them store a
/// value an instruction made (a constant has no place to be late at).
fn check(l: &Lowered, name: &str) -> (usize, usize) {
    let prefix = format!("dx.op.{name}.");
    let stores: Vec<_> = l
        .output_stores
        .iter()
        .filter(|s| s.name.starts_with(&prefix))
        .collect();
    let mut made = 0;
    for s in &stores {
        // Arguments after the opcode: the element, its row, the column, the value.
        let (Some(id), _) = s.args[0] else {
            panic!("{name}: an element that is no constant: {s:?}")
        };
        let first = stores
            .iter()
            .filter(|o| o.block == s.block && o.args[0].0 == Some(id))
            .map(|o| o.place)
            .min()
            .expect("its own store");
        if let (_, Some(value)) = s.args[3] {
            made += 1;
            assert!(
                value < first,
                "{name}: element {id}'s value at place {value} is made after its first store at place {first}: {stores:#?}"
            );
        }
    }
    (stores.len(), made)
}

#[test]
fn a_pixel_export_makes_every_value_before_its_first_store() {
    let words = [
        0x7E00_02F2, // v_mov_b32 v0, 1.0
        0x7E02_02F0, // v_mov_b32 v1, 0.5
        0xF800_180F,
        0x0001_0100, // exp mrt0 v0, v1, v1, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    let io = StageIo::Ps {
        inputs: vec![],
        input_addr: 0,
        db_shader_control: 0,
        patches: vec![],
    };
    let target = element("SV_Target", 0, 16, 0, 4, false);
    let l = lowered(HwStage::Ps, io, &words, vec![target], vec![]).unwrap();
    // Four channels, each a dword made a float by a cast.
    assert_eq!(check(&l, "storeOutput"), (4, 4));
}

/// An integer colour target (declared uint, `Options::integer_targets`) stores the bits of the
/// float a program made as they are, as an `i32`: a bit-cast, not a conversion. The float target
/// beside it stores floats.
#[test]
fn an_integer_target_stores_the_bits() {
    let words = [
        0x7E00_02F2, // v_mov_b32 v0, 1.0
        0x7E06_02FF, 0x4049_0FDB, // v_mov_b32 v3, pi
        0x0604_0680, // v_add_f32 v2, 0, v3
        0xF800_100F, 0x0000_0000, // exp mrt0 v0, v0, v0, v0 vm
        0xF800_1811, 0x0000_0002, // exp mrt1 v2, off, off, off done vm
        0xBF81_0000, // s_endpgm
    ];
    let io = || StageIo::Ps {
        inputs: vec![],
        input_addr: 0,
        db_shader_control: 0,
        patches: vec![],
    };
    let float = element("SV_Target", 0, 16, 0, 4, false);
    let uint = SigElement { comp: 5, ..element("SV_Target", 1, 16, 1, 1, false) };
    let l = lowered(HwStage::Ps, io(), &words, vec![float.clone(), uint], vec![]).unwrap();
    let stores = |l: &Lowered| -> Vec<(u32, String)> {
        l.output_stores
            .iter()
            .filter(|s| s.name.starts_with("dx.op.storeOutput."))
            .map(|s| (s.args[0].0.expect("an element constant") as u32, s.name.clone()))
            .collect()
    };
    let f32s = |n: usize| vec![(0u32, "dx.op.storeOutput.f32".to_owned()); n];
    let mut want = f32s(4);
    want.push((1, "dx.op.storeOutput.i32".into()));
    assert_eq!(stores(&l), want);
    // Declared float (no mask), mrt1 stores the float.
    let l = lowered(HwStage::Ps, io(), &words, vec![float.clone(), element("SV_Target", 1, 16, 1, 1, false)], vec![]).unwrap();
    let mut want = f32s(4);
    want.push((1, "dx.op.storeOutput.f32".into()));
    assert_eq!(stores(&l), want);
}

#[test]
fn a_vertex_export_makes_every_value_before_its_first_store() {
    let words = [
        0x7E00_02F2, // v_mov_b32 v0, 1.0
        0x7E02_02F0, // v_mov_b32 v1, 0.5
        0xF800_08CF,
        0x0001_0100, // exp pos0 v0, v1, v1, v0 done
        0xBF81_0000, // s_endpgm
    ];
    let io = StageIo::Vs {
        inputs: vec![],
        outputs: vec![],
    };
    let position = element("SV_Position", 0, 3, 0, 4, false);
    let l = lowered(HwStage::Vs, io, &words, vec![position], vec![]).unwrap();
    assert_eq!(check(&l, "storeOutput"), (4, 4));
}

#[test]
fn a_vertex_id_rect_list_export_makes_every_value_before_its_first_store() {
    let words: [u32; 13] = [
        0x7E14_02F3, // v_mov_b32 v10, -1.0
        0x7E16_02F2, // v_mov_b32 v11, 1.0
        0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
        0x0204_170A, // v_cndmask_b32 v2, v10, v11, vcc
        0x7D84_0082, // v_cmp_eq_u32 vcc, 2, v0
        0x0206_170A, // v_cndmask_b32 v3, v10, v11, vcc
        0x7E08_0280, // v_mov_b32 v4, 0
        0x7E0C_02F2, // v_mov_b32 v6, 1.0
        0xF800_0203,
        0x0000_0302, // exp param0 v2, v3, off, off
        0xF800_08CF,
        0x0604_0302, // exp pos0 v2, v3, v4, v6 done
        0xBF81_0000, // s_endpgm
    ];
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut meta = [0u8; 0x10];
    meta[12] = 0x40;
    let raw = super::graphics::container(HwStage::Vs, &bytes, &meta).unwrap();
    let c = crate::Container::parse(&raw).unwrap();
    let dxil = c.part(crate::FourCC::DXIL).or(c.part(crate::FourCC::ACCC)).unwrap();
    let native = crate::native_programs(dxil.data).unwrap().into_iter().next().unwrap();
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track_graphics(&p, &native.layout, native.stage, Some(&st), true).unwrap();
    let io = io::raise_graphics(&p, &native, true).unwrap();
    let outputs = super::signature(c.part(crate::FourCC::OSG1).unwrap().data).unwrap();
    let l = lower::lower_graphics(
        &Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs,
            prims: vec![],
            mesh_stride: 0,
            atomic_locks: false,
            bias_as_gradients: false,
            narrow_waves: false,
            wave32_host: true,
            zero_copy_memory: true,
            view_marks: false,
            rect_list: true,
            copy_params: [0; 32],
        },
        true,
    )
    .unwrap();
    // SV_Position (4) + TEXCOORD0 padded to 4 = 8 stores, all computed before their element's
    // first `storeOutput`.
    assert_eq!(check(&l, "storeOutput"), (8, 8));
}

/// A one-wave triangle mesh program. Each lane exports, as its vertex: the position
/// (two computed floats, which need no cast, then two dwords, which do), parameter 0 (four
/// dwords declared floats) and parameter 2 (four 16-bit values, each converted to a float);
/// and as its primitive: parameter 1 (two dwords declared floats) and the corners.
#[test]
fn a_mesh_export_makes_every_value_before_its_first_store() {
    let words = [
        0xD765_0001u32,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x3604_0281, // v_and_b32 v2, 1, v1
        0x3606_0282, // v_and_b32 v3, 2, v1
        0x7E0C_02F0, // v_mov_b32 v6, 0.5
        0x7E0E_02F2, // v_mov_b32 v7, 1.0
        0xF800_020F,
        0x0706_0201, // exp param0 v1, v2, v6, v7
        0xF800_0213,
        0x0000_0103, // exp param1 v3, v1, off, off
        0xF800_022F,
        0x0103_0201, // exp param2 v1, v2, v3, v1
        0x7E04_0D02, // v_cvt_f32_u32 v2, v2
        0x7E06_0D03, // v_cvt_f32_u32 v3, v3
        0x1008_04F6, // v_mul_f32 v4, 4.0, v2
        0x0608_08F3, // v_add_f32 v4, -1.0, v4
        0x100A_06F4, // v_mul_f32 v5, 2.0, v3
        0x060A_0AF3, // v_add_f32 v5, -1.0, v5
        0x7E12_02FF,
        0x0020_0400, // v_mov_b32 v9, 0x200400 (corners 0, 1, 2)
        0xF800_0941,
        0x0000_0009, // exp prim v9, off, off, off done
        0xF800_08CF,
        0x0706_0504, // exp pos0 v4, v5, v6, v7 done
        0xBF81_0000, // s_endpgm
    ];
    let output = |row, param, fp16| VsOutput {
        usage: Usage::Generic,
        row,
        param,
        fp16,
        pack2: false,
    };
    let io = StageIo::Ms(MeshIo {
        // Bit 7: a per-primitive output, its row one of the primitive signature.
        outputs: vec![
            output(1, 0, false),
            output(0, 0x81, false),
            output(2, 2, true),
        ],
        clip_cntl: 0,
        threads: 32,
        max_vertices: 32,
        max_primitives: 32,
        topology: 2,
        lds_bytes: 0,
    });
    let outputs = vec![
        element("SV_Position", 0, 3, 0, 4, false),
        element("TEXCOORD", 0, 0, 1, 4, false),
        element("TEXCOORD", 1, 0, 2, 4, true),
    ];
    let prims = vec![element("TEXCOORD", 2, 0, 0, 2, false)];
    let l = lowered(HwStage::Ms, io, &words, outputs, prims).unwrap();
    // Position (two of its values are casts), parameter 0 (four casts), parameter 2 (four
    // conversions); the primitive's two casts.
    assert_eq!(check(&l, "storeVertexOutput"), (12, 12));
    assert_eq!(check(&l, "storePrimitiveOutput"), (2, 2));
    // Mesh stores name the lane's vertex or primitive, taken once for all of them.
    assert!(
        l.output_stores
            .iter()
            .all(|s| s.args.len() == 5 && s.args[4] == l.output_stores[0].args[4])
    );
}
