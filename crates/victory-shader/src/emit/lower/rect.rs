//! Rectangle-list lowering for guessed vertex programs that compute corner
//! positions or varyings from `SV_VertexID` (`v0`).
//!
//! When `Options::rect_list` draws a rectangle list as four-vertex triangle-strip
//! instances (`(4, vertices / 3, 0, first / 3)`), vertex-buffer attributes fetch
//! three stored corners per instance via `Lower::vertex_fetch`. A program that
//! instead derives its corners from `v0` (for example `RECT_BY_ID`, fullscreen
//! passes, or screen-space tile lists indexing an SRV by `v0 / 3` and `v0 % 3`)
//! has no fetched float position from which `vertex_fetch` can extrapolate the
//! missing fourth corner, and each strip instance past the first would see
//! `SV_VertexID` = `0..3` again.
//!
//! Such a program runs its body three times in one invocation — for console
//! vertex ids `3 * instance + 0`, `3 * instance + 1` and `3 * instance + 2` —
//! finds the right-angle corner opposite the longest `SV_Position.xy` side,
//! extrapolates the fourth corner `p + q - r` across every float output, and
//! selects `(p, r, fourth, q)` by the strip vertex id `0..3`.

use super::super::bitcode::{Attr, Bin, Term, V, pred};
use super::{Lower, Stop, Val, unsupported};
use crate::gfx10::{Fields, Operand};
use crate::io::Input;
use crate::ir::{Reg, Value};
use crate::native::HwStage;

pub(super) struct RectPass {
    pub(super) corner: u32,
    pub(super) vid_val: Value,
    pub(super) corner_vid: V,
    pub(super) done_block: usize,
    pub(super) zeros: Vec<V>,
    pub(super) cur_out: Vec<V>,
    pub(super) block_out: Vec<Option<Vec<V>>>,
    pub(super) out_phis: Vec<(Vec<V>, usize)>,
    pub(super) end_exits: Vec<(Vec<V>, usize)>,
}

impl<'a> Lower<'a> {
    /// Returns the `Reg::V(0)` (`Input::VertexId`) entry value when this is a guessed
    /// `Options::rect_list` vertex program that reads `v0` outside the ignored vertex-index
    /// operand of a `VERTEX_TABLE` stage-in fetch.
    pub(super) fn rect_vertex_id(&self) -> Option<Value> {
        if !(self.guessed && self.x.rect_list && self.x.native.stage == HwStage::Vs) {
            return None;
        }
        let p = self.x.p;
        let &(_, vid) = p.inputs.iter().find(|&&(r, v)| {
            r == Reg::V(0) && self.x.io.inputs.contains(&(v, Input::VertexId))
        })?;
        let used = p.blocks.iter().enumerate().any(|(b, block)| {
            if block.phis.iter().any(|phi| phi.args.contains(&vid)) {
                return true;
            }
            block.insts.iter().enumerate().any(|(i, inst)| {
                let ignored_fetch_idx = match &inst.fields {
                    Fields::Buffer {
                        addr: Operand::Vector { reg: areg, .. },
                        idxen: true,
                        offen: false,
                        lds: false,
                        tfe: false,
                        ..
                    } if !inst.name().starts_with("buffer_store_")
                        && !inst.name().starts_with("buffer_atomic_")
                        && !self.x.res.unstrided.contains(&(b, i))
                        && self
                            .descriptors
                            .get(&((b, i), false))
                            .is_some_and(|src| self.vertex_table_slot(src).is_some()) =>
                    {
                        Some(Reg::V(*areg))
                    }
                    _ => None,
                };
                inst.reads
                    .iter()
                    .any(|&(r, v)| v == vid && Some(r) != ignored_fetch_idx)
            })
        });
        used.then_some(vid)
    }

    /// Resets per-pass SSA maps between the three corner evaluations while preserving function-
    /// and entry-block definitions (`s`, `loaded_inputs`, `entry_handles`, `root_views`,
    /// `user_data`, `user_reads`, `samplers`, `code_table`).
    fn reset_for_rect_corner(&mut self) {
        self.vals.clear();
        self.floats.clear();
        self.low16.clear();
        self.block.fill(None);
        self.bools.clear();
        self.lane_bits.clear();
        self.mask_phis.clear();
        self.pending_masks.clear();
        self.data_phis.clear();
        self.spill_phis.clear();
        self.pending_spills.clear();
        self.read_phis.clear();
        self.handles.clear();
        self.join_handles.clear();
        self.edges.clear();
        self.exported.fill(false);
        self.tail.fill(None);
        self.src0 = None;
        self.ballots.clear();
        self.mask_scc.clear();
        self.uniform_scc.clear();
        self.uniform_bits.clear();
        self.balloting.clear();
        self.section_lanes.clear();
        self.counters.clear();
        self.abort = None;
        self.debug_pos = None;
        self.rect_corner = None;
    }

    /// Lowers all three corners of one rectangle (`v0 = 3 * SV_InstanceID + 0..2`) and emits the
    /// reconstructed four-vertex strip outputs.
    pub(super) fn run_rect_by_id(&mut self, order: &[usize], vid_val: Value) -> Result<(), Stop> {
        if self.writes_uavs {
            return Err(unsupported(
                0,
                "a rectangle-list vertex program with unordered-access writes",
            ));
        }
        let Val::I32(strip_vid) = self.input(Input::VertexId) else {
            return Err(unsupported(0, "a vertex id"));
        };
        let Val::I32(inst_id) = self.input(Input::InstanceId) else {
            return Err(unsupported(0, "an instance id"));
        };
        let k3 = self.s.m.i32(3);
        let base_vid = self.s.m.bin(self.s.entry, Bin::Mul, inst_id, k3);

        let mut zeros = Vec::with_capacity(4 * self.s.outputs.len());
        for o in &self.s.outputs {
            let z = if o.comp == 9 {
                self.s.m.f32(0)
            } else {
                self.s.m.i32(0)
            };
            zeros.extend([z; 4]);
        }

        let mut corner_outs: Vec<Vec<V>> = Vec::with_capacity(3);
        let mut entry_block = self.s.entry;
        for corner in 0..3u32 {
            if corner > 0 {
                self.reset_for_rect_corner();
            }
            let done_block = self.s.m.new_block();
            let corner_vid = if corner == 0 {
                base_vid
            } else {
                let c = self.s.m.i32(corner);
                self.s.m.bin(self.s.entry, Bin::Add, base_vid, c)
            };
            self.rect_pass = Some(RectPass {
                corner,
                vid_val,
                corner_vid,
                done_block,
                zeros: zeros.clone(),
                cur_out: zeros.clone(),
                block_out: vec![None; self.x.p.blocks.len()],
                out_phis: Vec::new(),
                end_exits: Vec::new(),
            });
            self.lower_blocks(order, entry_block)?;
            self.fill_phis()?;

            let rp = self.rect_pass.take().expect("active rect pass");
            let p = self.x.p;
            for (phis, b) in rp.out_phis {
                for (slot, phi_v) in phis.into_iter().enumerate() {
                    let mut incoming = Vec::new();
                    for &q in &p.blocks[b].preds {
                        let Some(qb) = self.tail[q] else { continue };
                        let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                        let val = rp.block_out[q].as_ref().expect("reachable pred")[slot];
                        incoming.push((val, qb));
                    }
                    self.s.m.set_phi(phi_v, incoming);
                }
            }
            let out = match rp.end_exits.as_slice() {
                [] => rp.zeros,
                [(only, _)] => only.clone(),
                many => (0..rp.zeros.len())
                    .map(|slot| {
                        let ty = self.s.m.type_of(rp.zeros[slot]);
                        let phi = self.s.m.phi(done_block, ty);
                        let incoming = many.iter().map(|(v, qb)| (v[slot], *qb)).collect();
                        self.s.m.set_phi(phi, incoming);
                        phi
                    })
                    .collect(),
            };
            corner_outs.push(out);
            entry_block = done_block;
        }

        self.cur = entry_block;
        self.emit_rect_exports(strip_vid, &corner_outs);
        self.s.m.terminate(self.cur, Term::Ret);
        Ok(())
    }

    pub(super) fn begin_rect_block(&mut self, b: usize, preds: &[usize]) {
        let is_header = self.counters.contains_key(&b);
        let Some(rp) = self.rect_pass.as_mut() else {
            return;
        };
        if preds.is_empty() {
            rp.cur_out = rp.zeros.clone();
        } else if preds.len() == 1 && !is_header {
            rp.cur_out = rp.block_out[preds[0]]
                .as_ref()
                .expect("RPO predecessor")
                .clone();
        } else {
            let phis: Vec<V> = rp
                .zeros
                .iter()
                .map(|&z| {
                    let ty = self.s.m.type_of(z);
                    self.s.m.phi(self.cur, ty)
                })
                .collect();
            rp.cur_out = phis.clone();
            rp.out_phis.push((phis, b));
        }
    }

    pub(super) fn record_rect_export(
        &mut self,
        ready: Vec<(usize, u8, &'static str, V)>,
        guard: Option<V>,
    ) {
        let c = self.cur;
        for (id, col, _, x) in ready {
            let slot = 4 * id + usize::from(col);
            let val = match guard {
                Some(g) => {
                    let prev = self.rect_pass.as_ref().expect("rect pass").cur_out[slot];
                    self.s.m.select(c, g, x, prev)
                }
                None => x,
            };
            self.rect_pass.as_mut().expect("rect pass").cur_out[slot] = val;
        }
    }

    fn emit_rect_exports(&mut self, strip_vid: V, corners: &[Vec<V>]) {
        let c = self.cur;
        let [out0, out1, out2] = [&corners[0], &corners[1], &corners[2]];
        let pos_xy = self
            .s
            .outputs
            .iter()
            .position(|o| o.kind == 3 && o.comp == 9 && o.usage & 0b11 == 0b11)
            .map(|pos| {
                let p = [
                    out0[4 * pos],
                    out0[4 * pos + 1],
                    out1[4 * pos],
                    out1[4 * pos + 1],
                    out2[4 * pos],
                    out2[4 * pos + 1],
                ];
                let mut d2 = |i: usize, j: usize| {
                    let dx = self.s.m.bin(c, Bin::Sub, p[2 * i], p[2 * j]);
                    let dy = self.s.m.bin(c, Bin::Sub, p[2 * i + 1], p[2 * j + 1]);
                    let xx = self.s.m.bin(c, Bin::Mul, dx, dx);
                    let yy = self.s.m.bin(c, Bin::Mul, dy, dy);
                    self.s.m.bin(c, Bin::Add, xx, yy)
                };
                let (ab, bd, ad) = (d2(0, 1), d2(1, 2), d2(0, 2));
                let ge1 = self.s.m.cmp(c, pred::OGE, bd, ab);
                let ge2 = self.s.m.cmp(c, pred::OGE, bd, ad);
                let at_a = self.s.m.bin(c, Bin::And, ge1, ge2);
                let ge3 = self.s.m.cmp(c, pred::OGT, ab, ad);
                let ge4 = self.s.m.cmp(c, pred::OGT, ab, bd);
                let at_d = self.s.m.bin(c, Bin::And, ge3, ge4);
                (at_a, at_d)
            });

        let is = |l: &mut Self, n: u32| {
            let k = l.s.m.i32(n);
            l.s.m.cmp(c, pred::EQ, strip_vid, k)
        };
        let (c1, c2, c3) = (is(self, 1), is(self, 2), is(self, 3));

        let mut ready: Vec<(usize, u8, &'static str, V)> = Vec::new();
        for id in 0..self.s.outputs.len() {
            let usage = self.s.outputs[id].usage;
            let comp = self.s.outputs[id].comp;
            for col in 0..4u8 {
                if usage >> col & 1 == 0 {
                    continue;
                }
                let slot = 4 * id + usize::from(col);
                let (a, b, d) = (out0[slot], out1[slot], out2[slot]);
                let (p, r, q) = match pos_xy {
                    Some((at_a, at_d)) => {
                        let r = self.s.m.select(c, at_d, d, b);
                        let r = self.s.m.select(c, at_a, a, r);
                        let p = self.s.m.select(c, at_a, b, a);
                        let q = self.s.m.select(c, at_d, b, d);
                        (p, r, q)
                    }
                    None => (a, b, d),
                };
                let ex = if comp == 9 {
                    let sum = self.s.m.bin(c, Bin::Add, p, q);
                    let fourth = self.s.m.bin(c, Bin::Sub, sum, r);
                    let (pi, ri, qi, fi) = (
                        self.as_int(p),
                        self.as_int(r),
                        self.as_int(q),
                        self.as_int(fourth),
                    );
                    let pr = self.s.m.cmp(c, pred::EQ, pi, ri);
                    let qr = self.s.m.cmp(c, pred::EQ, qi, ri);
                    let ex_i = self.s.m.select(c, qr, pi, fi);
                    let ex_i = self.s.m.select(c, pr, qi, ex_i);
                    self.as_float(ex_i)
                } else {
                    q
                };
                let v = self.s.m.select(c, c1, r, p);
                let v = self.s.m.select(c, c2, ex, v);
                let v = self.s.m.select(c, c3, q, v);
                ready.push((id, col, if comp == 9 { ".f32" } else { ".i32" }, v));
            }
        }

        let void = self.s.t.void;
        for (id, col, suffix, x) in ready {
            let args = [
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            self.s.op(
                c,
                &format!("storeOutput{suffix}"),
                Attr::NoUnwind,
                void,
                5,
                &args,
            );
        }
    }
}
