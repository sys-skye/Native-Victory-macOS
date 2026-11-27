//! Capture a native host submission at the backend boundary without a GPU or shader converter.

use super::host::NativeHost;
use super::list::NativeList;
use super::memory::{PoolInfo, PoolObserver};
use std::cell::RefCell;
use std::rc::Rc;
use victory_backend::backend::{ComputeBackend, RootLayout, Submission};
use victory_backend::resources::*;

#[test]
fn a_draw_with_only_vertex_slot_two_submits_its_buffer() {
    let host = NativeHost::new(Capture::default());
    let pool = PoolInfo {
        id: 1,
        base: 0x40000,
        len: 0x4000,
        memory_entry: None,
    };
    host.memory().mapped(0x80000, pool.len, &pool, 0);
    let mut list = NativeList::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(&[Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 1, 1))], None);
    list.set_vertex_buffer(2, 0x80100);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    host.execute(&(), &[&list], &[], &[]).unwrap();

    assert_eq!(host.unmapped(), 0, "empty vertex slots dropped the draw");
    assert_eq!(*host.backend.draws.borrow(), [vec![(2, 0x40000, 0x40100)]]);
}

#[derive(Default)]
pub(super) struct Capture {
    pub(super) views: RefCell<Vec<ViewDesc>>,
    // For each draw, the (slot, pool base, resolved address) bindings received.
    draws: RefCell<Vec<Vec<(usize, usize, usize)>>>,
    host_buffers: RefCell<Vec<Box<[u8]>>>,
    // (heap index, pool base, resolved address, size) for each buffer descriptor write.
    pub(super) buffer_writes: RefCell<Vec<(usize, usize, usize, u32)>>,
    // When set, `submit_passes` fails with it: (whether the command buffer committed, the pass
    // indices dropped from it).
    pub(super) refuse: std::cell::RefCell<Option<(bool, Vec<usize>)>>,
}

impl GraphicsBackend for Capture {
    type RenderPipeline = ();

    fn submit_passes(&self, _: &(), work: &PassSubmission<'_, Self>) -> Result<(), SubmitFailure> {
        if let Some((committed, refused)) = self.refuse.borrow().as_ref() {
            return Err(SubmitFailure {
                committed: *committed,
                refused: refused.clone(),
                error: "refused for the test".to_string(),
            });
        }
        for pass in work.passes {
            let Pass::Render(render) = pass else {
                continue;
            };
            for draw in &render.draws {
                self.draws.borrow_mut().push(
                    draw.vertex_buffers
                        .iter()
                        .map(|&(slot, base, address)| (slot, *base, address as usize))
                        .collect(),
                );
            }
        }
        Ok(())
    }

    fn create_render_pipeline(&self, _: &RenderPipelineDesc<'_>) -> Result<(), String> {
        unreachable!()
    }
}

impl ComputeBackend for Capture {
    type Queue = ();
    type Fence = ();
    type Buffer = usize;
    type Pipeline = ();
    type Heap = RefCell<Vec<Option<TextureDesc>>>;

    fn wrap_title_memory(&self, base: *mut u8, _: usize) -> Result<usize, String> {
        Ok(base as usize)
    }

    fn create_queue(&self) -> Result<(), String> {
        unreachable!()
    }
    fn create_fence(&self, _: u64) -> Result<(), String> {
        unreachable!()
    }
    fn host_buffer(&self, len: usize) -> Result<(usize, *mut u8), String> {
        let mut bytes = vec![0; len].into_boxed_slice();
        let ptr = bytes.as_mut_ptr();
        self.host_buffers.borrow_mut().push(bytes);
        Ok((ptr as usize, ptr))
    }
    fn gpu_address(&self, _: &usize, _: *const u8) -> u64 {
        unreachable!()
    }
    fn create_compute_pipeline(&self, _: &[u8]) -> Result<(), String> {
        unreachable!()
    }
    fn create_compute_pipeline_with(&self, _: &[u8], _: &RootLayout) -> Result<(), String> {
        unreachable!()
    }
    fn submit(&self, _: &(), _: &Submission<'_, Self>) -> Result<(), String> {
        unreachable!()
    }
    fn signal(&self, _: &(), _: u64) {
        unreachable!()
    }
    fn completed(&self, _: &()) -> u64 {
        unreachable!()
    }
    fn wait(&self, _: &(), _: u64, _: u64) -> bool {
        unreachable!()
    }
}

impl ResourceBackend for Capture {
    type Texture = TextureDesc;
    type Sampler = ();

    fn create_heap(&self, _: HeapKind, count: usize) -> Result<Self::Heap, String> {
        Ok(RefCell::new(vec![None; count]))
    }
    fn heap_address(&self, _: &Self::Heap, index: usize) -> u64 {
        (index * 32) as u64
    }
    fn write_buffer(&self, _: &Self::Heap, index: usize, base: &usize, address: *const u8, size: u32) {
        self.buffer_writes.borrow_mut().push((index, *base, address as usize, size));
    }
    fn write_texture(&self, heap: &Self::Heap, slot: usize, tex: &TextureDesc, _: f32) {
        heap.borrow_mut()[slot] = Some(*tex);
    }
    fn write_sampler(&self, _: &Self::Heap, _: usize, _: &(), _: f32) {
        unreachable!()
    }
    fn clear_entry(&self, _: &Self::Heap, _: usize) {
        unreachable!()
    }
    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureDesc, String> {
        Ok(*desc)
    }
    fn create_texture_view(&self, tex: &TextureDesc, view: &ViewDesc) -> Result<TextureDesc, String> {
        self.views.borrow_mut().push(*view);
        Ok(*tex)
    }
    fn create_texture_over_memory(
        &self,
        _: &usize,
        _: *const u8,
        _: &TextureDesc,
        _: usize,
    ) -> Result<TextureDesc, String> {
        unreachable!()
    }
    fn linear_alignment(&self, _: Format) -> usize {
        unreachable!()
    }
    fn upload_texture(&self, _: &TextureDesc, _: usize, _: usize, _: &[u8], _: usize, _: usize) {
        unreachable!()
    }
    fn read_texture(&self, _: &TextureDesc, _: usize, _: usize, _: &mut [u8], _: usize, _: usize) {
        unreachable!()
    }
    fn create_sampler(&self, _: &SamplerDesc) -> Result<(), String> {
        unreachable!()
    }
}

/// A pool of real memory shown to the "title" at an alias, the way `title_tests` makes one.
fn aliased_pool(host: &NativeHost<Capture>, mem: &[u8]) -> u64 {
    let base = mem.as_ptr() as usize;
    let alias = (base + 0x8000_0000) & !0xfff;
    let pool = PoolInfo { id: 1, base, len: mem.len(), memory_entry: None };
    host.memory().mapped(alias, pool.len, &pool, 0);
    alias as u64
}

#[test]
fn a_submission_that_never_committed_pays_its_value_writes_on_the_cpu() {
    let backend = Capture::default();
    backend.refuse.replace(Some((false, Vec::new())));
    let host = NativeHost::new(backend);
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.fill(alias + 0x100, 8, 0xAB);
    list.fill(alias + 0x200, 4, 0x5A);
    let err = host.execute(&(), &[&list], &[], &[]);
    assert!(err.is_err(), "the backend refused the submission");
    assert_eq!(&mem[0x100..0x108], &[0xAB; 8], "the lost submission's first fill was not paid");
    assert_eq!(&mem[0x200..0x204], &[0x5A; 4], "the lost submission's second fill was not paid");
}

#[test]
fn only_the_fills_of_refused_passes_are_paid_on_the_cpu() {
    let backend = Capture::default();
    // The command buffer commits, but pass 0 (the first fill) and pass 1 (the draw) are dropped.
    backend.refuse.replace(Some((true, vec![0, 1])));
    let host = NativeHost::new(backend);
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(&[Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 1, 1))], None);
    list.fill(alias + 0x100, 8, 0xAB); // pass 0: refused
    list.set_vertex_buffer(2, alias + 0x300);
    list.draw_instanced(3, 1, 0, 0).unwrap(); // pass 1: refused
    list.fill(alias + 0x200, 4, 0x5A); // pass 2: kept, runs on the GPU
    let err = host.execute(&(), &[&list], &[], &[]);
    assert!(err.is_err(), "the backend refused passes 0 and 1");
    assert_eq!(&mem[0x100..0x108], &[0xAB; 8], "the refused pass's fill was not paid");
    assert_eq!(&mem[0x200..0x204], &[0; 4], "a kept pass's fill must stay with the GPU");
}

#[test]
fn a_submitted_piece_leaves_its_value_writes_to_the_gpu() {
    let host = NativeHost::new(Capture::default());
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.fill(alias + 0x100, 8, 0xAB);
    host.execute(&(), &[&list], &[], &[]).unwrap();
    assert_eq!(&mem[0x100..0x108], &[0; 8], "the host wrote a fill the GPU was given");
}
