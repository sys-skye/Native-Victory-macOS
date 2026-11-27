//! The graphics state of the console's pipelines.
//!
//! Observed at run time (`VICTORY_TRACE_MISSING=<bytes>` on device slot 0x1b8): a description made
//! by device method 0x1c8 carries shaders and a register image but no blend, rasterizer or depth
//! state. The title adds those with device method 0x1b8, `(this, base, count, records, iid, out)`:
//! `base` is the pipeline 0x1c8 made, `records` is `count` records of [`RECORD_WORDS`] dwords, and
//! the result (the pipeline state's IID) is the pipeline to draw with. A record is a tag and then
//! the D3D12 structure of its kind, as the PC API lays it out:
//!
//! | tag | structure |
//! |-----|-----------|
//! | 0 | `D3D12_DEPTH_STENCIL_DESC` (stencil masks packed in one dword, as the stream's) |
//! | 1 | `D3D12_RASTERIZER_DESC` |
//! | 2 | `D3D12_BLEND_DESC` |
//! | 5 | seen with depth records only; its words are not understood and it is ignored |
//!
//! What a record leaves out keeps D3D12's defaults ([`Graphics::default`]). Target formats and
//! topology are in no record: a draw has them from its bound targets and `IASetPrimitiveTopology`
//! ([`graphics_state_for`]).

use super::state::Graphics;
use victory_backend::resources::Format;

/// Dwords of a record: the tag and the 82 dwords of the biggest structure, the blend description.
pub const RECORD_WORDS: usize = 83;

/// The most records a call carries (three were seen).
pub const MAX_RECORDS: usize = 8;

/// The subobject type (`D3D12_PIPELINE_STATE_SUBOBJECT_TYPE`, which [`Graphics::read`] takes) of a
/// record tag.
fn subobject(tag: u32) -> Option<u32> {
    match tag {
        0 => Some(11),
        1 => Some(10),
        2 => Some(8),
        _ => None,
    }
}

/// `base` with the state of `records` (a whole number of them) applied in order. Returns the tags
/// that were not understood, which are skipped.
pub fn apply(base: &mut Graphics, records: &[u32]) -> Vec<u32> {
    let mut unknown = Vec::new();
    for record in records.as_chunks::<RECORD_WORDS>().0.iter().take(MAX_RECORDS) {
        let Some(kind) = subobject(record[0]) else {
            unknown.push(record[0]);
            continue;
        };
        let mut subobject = [0u32; RECORD_WORDS];
        subobject.copy_from_slice(record);
        subobject[0] = kind;
        // SAFETY: a whole subobject of `kind` (at most 82 dwords of data are read) with no pointers.
        unsafe { base.read(kind, subobject.as_ptr().cast()) };
    }
    unknown
}

/// The state to build a draw's pipeline with: `base` (a variant's state, [`apply`]) for a draw into
/// targets of these formats.
pub fn graphics_state_for(base: &Graphics, color: Vec<Format>, depth: Option<Format>) -> Graphics {
    base.clone().with_targets(color, depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_backend::resources::*;

    /// A record: its tag, then `words` and the default blend description's tail as the title's
    /// stack leaves it.
    fn record(tag: u32, words: &[u32]) -> Vec<u32> {
        let mut r = vec![tag];
        r.extend_from_slice(words);
        r.resize(RECORD_WORDS, 0);
        r
    }

    /// Words 1.. of a blend record: the two flags, then 8 targets of 10 (enable, logic enable,
    /// source, destination, op, source alpha, destination alpha, alpha op, logic op, mask).
    fn blend(first: [u32; 10]) -> Vec<u32> {
        let mut w = vec![0, 0];
        w.extend(first);
        for _ in 1..8 {
            w.extend([0, 0, 2, 1, 1, 2, 1, 1, 4, 15]);
        }
        record(2, &w)
    }

    #[test]
    fn blend_record_sets_the_first_target() {
        let mut g = Graphics::default();
        // The title's alpha blend: source alpha, inverse source alpha, add; alpha: one, inverse.
        let unknown = apply(&mut g, &blend([1, 0, 5, 6, 1, 2, 6, 1, 4, 15]));
        assert!(unknown.is_empty());
        let target = g.color_state()[0];
        assert_eq!(
            target.blend,
            Some(Blend {
                src: BlendFactor::SrcAlpha,
                dst: BlendFactor::InvSrcAlpha,
                op: BlendOp::Add,
                src_alpha: BlendFactor::One,
                dst_alpha: BlendFactor::InvSrcAlpha,
                op_alpha: BlendOp::Add,
            })
        );
        assert_eq!(target.write_mask, 0xf);
        // The default record turns blending off.
        apply(&mut g, &blend([0, 0, 2, 1, 1, 2, 1, 1, 4, 15]));
        assert_eq!(g.color_state()[0].blend, None);
    }

    #[test]
    fn depth_record_sets_depth_and_stencil() {
        let mut g = Graphics::default();
        // Reverse-Z test and write, stencil on with both masks (0xffff), keep/keep/increment,
        // not-equal, in both faces (a record the title makes).
        apply(&mut g, &record(0, &[1, 1, 7, 1, 65535, 1, 1, 7, 6, 1, 1, 7, 6, 0]));
        assert_eq!(g.depth_test(), Some((true, Compare::GreaterEqual)));
        let stencil = g.stencil().expect("stencil on");
        assert_eq!((stencil.read_mask, stencil.write_mask), (0xff, 0xff));
        assert_eq!(stencil.front.pass, StencilOp::IncrementWrap);
        assert_eq!(stencil.front.compare, Compare::NotEqual);
        assert_eq!(stencil.back.compare, Compare::NotEqual);
        // Depth off, always: what most of the title's records say.
        apply(&mut g, &record(0, &[0, 0, 8, 0, 255, 1, 1, 1, 8, 1, 1, 1, 8, 0]));
        assert_eq!(g.depth_test(), None);
        assert!(g.stencil().is_none());
    }

    #[test]
    fn raster_record_sets_culling() {
        let mut g = Graphics::default();
        // Solid, no culling, front counter-clockwise, depth clip.
        apply(&mut g, &record(1, &[3, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]));
        assert_eq!(g.cull(), Cull::None);
        assert!(g.raster().front_ccw);
        assert!(g.raster().depth_clip);
        apply(&mut g, &record(1, &[3, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0]));
        assert_eq!(g.cull(), Cull::Back);
        assert!(!g.raster().front_ccw);
    }

    #[test]
    fn records_apply_in_order_and_unknown_tags_are_returned() {
        let mut g = Graphics::default();
        let mut records = record(0, &[0, 0, 8, 0, 255, 1, 1, 1, 8, 1, 1, 1, 8, 0]);
        records.extend(blend([1, 0, 9, 6, 1, 2, 6, 1, 4, 15]));
        records.extend(record(5, &[0, 0, 0, 0, 40]));
        assert_eq!(apply(&mut g, &records), [5]);
        assert_eq!(g.depth_test(), None);
        assert_eq!(g.color_state()[0].blend.expect("blend").src, BlendFactor::DestColor);
        // A partial record at the end is not read.
        let mut g = Graphics::default();
        apply(&mut g, &blend([1, 0, 9, 6, 1, 2, 6, 1, 4, 15])[..RECORD_WORDS - 1]);
        assert_eq!(g, Graphics::default());
    }

    #[test]
    fn a_draw_names_its_target_formats() {
        let mut g = Graphics::default();
        apply(&mut g, &blend([1, 0, 5, 6, 1, 2, 6, 1, 4, 15]));
        let d = graphics_state_for(&g, vec![Format::Bgra8Unorm, Format::Rgba16Float], Some(Format::D32Float));
        assert_eq!(d.color_formats(), Ok(vec![Format::Bgra8Unorm, Format::Rgba16Float]));
        assert_eq!(d.depth_format(), Ok(Some(Format::D32Float)));
        assert_eq!(d.color_state().len(), 2);
        // The same blend on every target (the description is not independent).
        assert_eq!(d.color_state()[1].blend, d.color_state()[0].blend);
        assert_eq!(graphics_state_for(&g, vec![], None).depth_format(), Ok(None));
    }
}
