//! victory-gpu on Metal, end to end, with command streams of our own (the HLSL under
//! `victory-metal/tests/fixtures`, our own data): the calls a title list receives, recorded into a
//! [`NativeList`], submitted through [`Host`], read back from title memory. Title memory is a pool
//! (`Alias::primary`, what the GPU buffer wraps) shown to the "title" at a second address
//! (`Alias::alias`) through the pool table, so every address the list names is translated.
//! Tests using `host()` skip when Apple's Metal Shader Converter is not installed.

use super::Host;
use super::list::NativeList;
use super::memory::{PoolInfo, PoolObserver};
use std::rc::Rc;
use victory_backend::backend::ComputeBackend;
use victory_backend::resources::*;
use victory_metal::mem::{Alias, page_size};
use victory_metal::metal::Metal;

#[path = "addr_tests.rs"]
mod addr_tests;
#[path = "heap_tests.rs"]
mod heap_tests;
#[path = "legacy_fma_tests.rs"]
mod legacy_fma_tests;
#[path = "mesh_tests.rs"]
mod mesh_tests;
#[path = "scaled_vertex_tests.rs"]
mod scaled_vertex_tests;
#[path = "shader_tests.rs"]
mod shader_tests;
#[path = "split_wave_tests.rs"]
mod split_wave_tests;
#[path = "stale_tests.rs"]
mod stale_tests;
#[path = "state_tests.rs"]
mod state_tests;
#[path = "texture_tests.rs"]
mod texture_tests;
#[path = "title_tests.rs"]
mod title_tests;

fn host() -> Option<Host> {
    if !std::path::Path::new(victory_metal::msc::DEFAULT_DYLIB).exists()
        && std::env::var_os("VICTORY_MSC_DYLIB").is_none()
    {
        eprintln!("skipped: Metal Shader Converter not installed");
        return None;
    }
    Some(Host::new(Metal::new().unwrap()))
}

#[test]
fn array_texture_ranges_reach_the_backend_for_object_and_placed_views() {
    use crate::descriptor::{View, record};
    let host = super::host::NativeHost::new(super::vertex_tests::Capture::default());
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    host.heap(HeapKind::Resource, handle, 1).unwrap();

    let mut resource = [0; 14];
    resource[0] = 3; // D3D12_RESOURCE_DIMENSION_TEXTURE2D
    let address = 0x12000;
    let texture = Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 4, 4));
    let key = Rc::as_ptr(&texture) as usize;
    host.register_texture(key, &texture);
    host.register_texture(super::textures::placed_key(address, &resource), &texture);

    for unordered in [false, true] {
        let mut desc = [0; 10];
        desc[..2].copy_from_slice(&[28, 5]); // RGBA8_UNORM, TEXTURE2DARRAY
        let object = if unordered {
            desc[2..5].copy_from_slice(&[1, 2, 1]);
            View::Unordered(key, 0, Some(desc))
        } else {
            desc[4..8].copy_from_slice(&[1, 3, 2, 4]);
            View::Shader(key, Some(desc))
        };
        let mut placed = [0; 16];
        placed[..10].copy_from_slice(&desc);
        placed[10] = address as u32;
        for view in [object, View::Placed(unordered, resource, placed)] {
            record(handle, view);
            assert!(host.mirror(handle).unwrap());
            assert_eq!(
                host.backend.views.borrow_mut().pop(),
                Some(ViewDesc {
                    format: Format::Rgba8Unorm,
                    mips: (1, if unordered { 1 } else { 3 }),
                    slices: (2, if unordered { 1 } else { 4 }),
                })
            );
        }
    }
}

fn fixture(name: &str) -> Vec<u8> {
    let p = format!(
        "{}/../victory-metal/tests/fixtures/{name}.dxil",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

/// A pool of `pages` pages that the title sees at `alias`; GPU buffers wrap `primary`.
struct Pool {
    mem: Alias,
}

impl Pool {
    fn new(h: &Host, pages: usize) -> Pool {
        // Tests assume at least 16 KiB per page (e.g. offsets 16384, 32768); under x86_64
        // Rosetta `page_size()` reports 4 KiB, so clamp the unit to at least 16 KiB.
        let mem = Alias::new(pages * page_size().max(16384)).unwrap();
        let info = PoolInfo {
            id: 1,
            base: mem.primary as usize,
            len: mem.len,
            memory_entry: None,
        };
        h.memory().pool_created(&info);
        h.memory().mapped(mem.alias as usize, mem.len, &info, 0);
        Pool { mem }
    }

    /// The title's address of byte `at`.
    fn va(&self, at: usize) -> u64 {
        self.mem.alias as u64 + at as u64
    }

    fn write<T: Copy>(&self, at: usize, v: &[T]) {
        // SAFETY: inside the mapping; the GPU is idle while the test writes.
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), self.mem.alias.add(at).cast(), v.len())
        };
    }

    fn read<T: Copy>(&self, at: usize, n: usize) -> Vec<T> {
        // SAFETY: inside the mapping; read after the fence wait.
        unsafe { std::slice::from_raw_parts(self.mem.alias.add(at).cast::<T>(), n).to_vec() }
    }
}

unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn mach_vm_allocate(task: u32, addr: *mut u64, size: u64, flags: i32) -> i32;
    fn mach_vm_deallocate(task: u32, addr: u64, size: u64) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn mach_vm_remap(
        target: u32,
        addr: *mut u64,
        size: u64,
        mask: u64,
        flags: i32,
        src_task: u32,
        src_addr: u64,
        copy: i32,
        cur: *mut i32,
        max: *mut i32,
        inherit: u32,
    ) -> i32;
}

/// Title pages made of pool pages in any order, mapped for real at a title address of this
/// process (as the title maps its physical pages), and registered with the host's pool table.
struct TitlePages {
    va: u64,
    len: usize,
    /// Pages of `page` bytes unmapped since, by index: no longer ours to deallocate.
    gone: std::cell::RefCell<Vec<(usize, usize)>>,
}

impl TitlePages {
    /// One title page of `page` bytes per pool offset in `from`, in that order.
    fn new(h: &Host, pool: &Alias, info: &PoolInfo, from: &[usize], page: usize) -> TitlePages {
        let len = from.len() * page;
        let mut va = 0u64;
        // SAFETY: Mach VM calls on our own task; each remap overwrites a page of our reservation
        // with a view of a page of the pool's own mapping.
        unsafe {
            let t = mach_task_self();
            assert_eq!(mach_vm_allocate(t, &mut va, len as u64, 1), 0);
            for (k, &off) in from.iter().enumerate() {
                let (mut at, mut cur, mut max) = (va + (k * page) as u64, 0, 0);
                // VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, shared (not copied).
                let r = mach_vm_remap(t, &mut at, page as u64, 0, 0x4000, t, pool.primary as u64 + off as u64, 0, &mut cur, &mut max, 0);
                assert_eq!(r, 0, "mach_vm_remap");
                h.memory().mapped(at as usize, page, info, off);
            }
        }
        TitlePages { va, len, gone: Default::default() }
    }

    /// The title maps title page `k` (of `page` bytes) from pool offset `off` instead.
    fn remap(&self, h: &Host, info: &PoolInfo, pool: &Alias, k: usize, off: usize, page: usize) {
        let (mut at, mut cur, mut max) = (self.va + (k * page) as u64, 0, 0);
        // SAFETY: Mach VM calls on our own task over a page of the reservation, as in `new`.
        let r = unsafe { mach_vm_remap(mach_task_self(), &mut at, page as u64, 0, 0x4000, mach_task_self(), pool.primary as u64 + off as u64, 0, &mut cur, &mut max, 0) };
        assert_eq!(r, 0, "mach_vm_remap");
        h.memory().mapped(at as usize, page, info, off);
    }

    /// The title unmaps title page `k` (of `page` bytes).
    fn unmap(&self, h: &Host, k: usize, page: usize) {
        let at = self.va + (k * page) as u64;
        // SAFETY: a page of the reservation, which is not used again (`gone`).
        assert_eq!(unsafe { mach_vm_deallocate(mach_task_self(), at, page as u64) }, 0);
        h.memory().unmapped(at as usize, page);
        self.gone.borrow_mut().push((k * page, page));
    }

    fn write<T: Copy>(&self, at: usize, v: &[T]) {
        // SAFETY: inside the mapping; the GPU is idle while the test writes.
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), (self.va as *mut u8).add(at).cast(), v.len()) };
    }

    fn read<T: Copy>(&self, at: usize, n: usize) -> Vec<T> {
        // SAFETY: inside the mapping; read after the fence wait.
        unsafe { std::slice::from_raw_parts((self.va as *const u8).add(at).cast::<T>(), n).to_vec() }
    }
}

impl Drop for TitlePages {
    fn drop(&mut self) {
        // The pages given back may hold someone else's mapping by now (the GPU driver's
        // notification queue was seen there): deallocate only the rest.
        let mut gone = self.gone.borrow().clone();
        gone.sort_unstable();
        let mut at = 0;
        for (from, len) in gone.into_iter().chain([(self.len, 0)]) {
            if from > at {
                // SAFETY: part of the range `new` allocated, still ours.
                unsafe { mach_vm_deallocate(mach_task_self(), self.va + at as u64, (from - at) as u64) };
            }
            at = at.max(from + len);
        }
    }
}

/// A buffer view over two title pages the title mapped from pool pages in descending order (the
/// G-buffer culling programs' 174 MB instance buffer, #258): not one linear run of the pool, so
/// it was the null view and the shader read zeros. It now wraps the title's own mapping.
#[test]
fn a_buffer_view_across_title_pages_out_of_pool_order_reads_the_title_mapping() {
    let Some(h) = host() else { return };
    let page = page_size().max(16384);
    let pool = Alias::new(4 * page).unwrap();
    let info = PoolInfo { id: 1, base: pool.primary as usize, len: pool.len, memory_entry: None };
    h.memory().pool_created(&info);
    // Title pages 0 and 1 are pool pages 2 and 1; title page 2, pool page 3, takes the output.
    let title = TitlePages::new(&h, &pool, &info, &[2 * page, page, 3 * page], page);
    let (src, dst) = (page - 128, 2 * page);
    title.write(src, &(0..64u32).collect::<Vec<_>>());
    let slots = vec![0u32; 8 * 8];
    let start = slots.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 8).unwrap();
    let placed = |at: usize| {
        let mut d = [0u32; 16];
        d[1] = 1; // buffer view
        d[6] = 64; // elements
        d[8] = 1; // raw (SRV flags word)
        d[10] = (title.va + at as u64) as u32;
        d[11] = ((title.va + at as u64) >> 32) as u32;
        d
    };
    let mut resource = [0u32; 14];
    resource[0] = 1; // buffer
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, &placed(src), start);
    let mut uav = placed(dst);
    (uav[4], uav[6]) = (64, 0);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav, start + 32);
    assert!(h.mirror(start).unwrap(), "the source view was not mirrored");
    assert!(h.mirror(start + 32).unwrap());
    let pso = h.compute_pipeline(&fixture("heap_rw"), [64, 1, 1]).unwrap();
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[3, 1]);
    l.set_compute_root_table(1, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    run(&h, &[&l]);
    assert_eq!(title.read::<u32>(dst, 64), (0..64).map(|i| i * 3 + 1).collect::<Vec<_>>());
    drop(slots);
}

/// A wrap keeps the physical pages it was made over: once the title maps other pool pages at
/// those title addresses, or unmaps them, the GPU still reads the old pages through it (no fault,
/// no zeros; seen on an M5 Pro, macOS 27). The host mirrors such a heap entry again before the
/// next work it submits (`NativeHost::follow_title_pages`), so the view reads the new page, and
/// then zeros (the null view) once part of its range is unmapped.
#[test]
fn a_wrapped_view_follows_title_pages_remapped_or_unmapped_under_it() {
    let Some(h) = host() else { return };
    let page = page_size().max(16384);
    let pool = Alias::new(5 * page).unwrap();
    let info = PoolInfo { id: 1, base: pool.primary as usize, len: pool.len, memory_entry: None };
    h.memory().pool_created(&info);
    // Title pages 0 and 1 are pool pages 2 and 1 (the source, across them); title page 2, pool
    // page 3, takes the output.
    let title = TitlePages::new(&h, &pool, &info, &[2 * page, page, 3 * page], page);
    let (src, dst) = (page - 128, 2 * page);
    title.write(src, &(0..64u32).collect::<Vec<_>>());
    // Pool page 4 holds other words where title page 1's are.
    // SAFETY: inside the pool's own mapping; the GPU is idle.
    unsafe { std::ptr::copy_nonoverlapping((100..132u32).collect::<Vec<_>>().as_ptr(), pool.primary.add(4 * page).cast(), 32) };
    let slots = vec![0u32; 8 * 8];
    let start = slots.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 8).unwrap();
    let placed = |at: usize| {
        let mut d = [0u32; 16];
        d[1] = 1; // buffer view
        d[6] = 64; // elements
        d[8] = 1; // raw (SRV flags word)
        d[10] = (title.va + at as u64) as u32;
        d[11] = ((title.va + at as u64) >> 32) as u32;
        d
    };
    let mut resource = [0u32; 14];
    resource[0] = 1; // buffer
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, &placed(src), start);
    let mut uav = placed(dst);
    (uav[4], uav[6]) = (64, 0);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav, start + 32);
    assert!(h.mirror(start).unwrap());
    assert!(h.mirror(start + 32).unwrap());
    let pso = h.compute_pipeline(&fixture("heap_rw"), [64, 1, 1]).unwrap();
    // dst[i] = src[i] * 3 + 1 through the heap's views; dst starts at a canary.
    let pass = || {
        title.write(dst, &[0xCAFE_CAFEu32; 64]);
        let mut l = NativeList::default();
        l.set_heaps(Some(&heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_constants(0, &[3, 1]);
        l.set_compute_root_table(1, &heap, 0);
        l.dispatch(1, 1, 1).unwrap();
        run(&h, &[&l]);
        title.read::<u32>(dst, 64)
    };
    let expect = |page1: &dyn Fn(u32) -> u32| (0..64u32).map(|i| if i < 32 { i * 3 + 1 } else { page1(i) * 3 + 1 }).collect::<Vec<_>>();
    assert_eq!(pass(), expect(&|i| i));
    title.remap(&h, &info, &pool, 1, 4 * page, page);
    assert_eq!(pass(), expect(&|i| i + 68), "the view still read the old pool page");
    title.unmap(&h, 1, page);
    assert_eq!(pass(), vec![1; 64], "a view with an unmapped page is the null view");
    // A null view (address 0) written over a followed entry is not followed any more.
    assert!(heap.title_views.borrow().contains_key(&0));
    crate::descriptor::create_cbv(std::ptr::null_mut(), &[0, 0, 256, 0], start);
    assert!(h.mirror(start).unwrap());
    assert!(!heap.title_views.borrow().contains_key(&0), "a null view's pages are followed");
    drop(slots);
}

/// The title copies one view into many heap entries; following the pages under them walks the
/// pool table once for that range a look, not once an entry, and every copy still follows a
/// remap (#267 review: 54 copies of the 174 MiB view cost about 3 ms a look).
#[test]
fn copies_of_a_wrapped_view_are_followed_with_one_walk() {
    let Some(h) = host() else { return };
    let page = page_size().max(16384);
    let pool = Alias::new(5 * page).unwrap();
    let info = PoolInfo { id: 1, base: pool.primary as usize, len: pool.len, memory_entry: None };
    h.memory().pool_created(&info);
    let title = TitlePages::new(&h, &pool, &info, &[2 * page, page, 3 * page], page);
    let (src, dst) = (page - 128, 2 * page);
    title.write(src, &(0..64u32).collect::<Vec<_>>());
    // SAFETY: inside the pool's own mapping; the GPU is idle.
    unsafe { std::ptr::copy_nonoverlapping((100..132u32).collect::<Vec<_>>().as_ptr(), pool.primary.add(4 * page).cast(), 32) };
    let slots = vec![0u32; 8 * 8];
    let start = slots.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 8).unwrap();
    let placed = |at: usize| {
        let mut d = [0u32; 16];
        (d[1], d[6], d[8]) = (1, 64, 1);
        d[10] = (title.va + at as u64) as u32;
        d[11] = ((title.va + at as u64) >> 32) as u32;
        d
    };
    let mut resource = [0u32; 14];
    resource[0] = 1;
    // Entries 0 to 2: three copies of the source view; entry 3: the output.
    for k in 0..3 {
        crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, &placed(src), start + 32 * k);
        assert!(h.mirror(start + 32 * k).unwrap());
    }
    let mut uav = placed(dst);
    (uav[4], uav[6]) = (64, 0);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav, start + 96);
    assert!(h.mirror(start + 96).unwrap());
    let walks = || super::host::SIGNATURES.with(|n| n.replace(0));
    // A pool-table change elsewhere: one walk for the three copies, none moved.
    h.memory().pool_created(&PoolInfo { id: 2, base: 0x1000, len: 0x1000, memory_entry: None });
    walks();
    run(&h, &[]);
    assert_eq!(walks(), 1);
    // Remapped under them: still one walk, and the copy the program reads follows.
    let pso = h.compute_pipeline(&fixture("heap_rw"), [64, 1, 1]).unwrap();
    title.remap(&h, &info, &pool, 1, 4 * page, page);
    title.write(dst, &[0xCAFE_CAFEu32; 64]);
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[3, 1]);
    l.set_compute_root_table(1, &heap, 2);
    l.dispatch(1, 1, 1).unwrap();
    walks();
    run(&h, &[&l]);
    assert_eq!(walks(), 1);
    let expect: Vec<u32> = (0..64u32).map(|i| if i < 32 { i * 3 + 1 } else { (i + 68) * 3 + 1 }).collect();
    assert_eq!(title.read::<u32>(dst, 64), expect);
    drop(slots);
}

fn run(h: &Host, lists: &[&NativeList<Metal>]) {
    let q = h.backend.create_queue().unwrap();
    let f = h.backend.create_fence(0).unwrap();
    h.execute(&q, lists, &[], &[(&f, 1)]).unwrap();
    assert!(h.backend.wait(&f, 1, 10_000), "GPU did not finish");
}

const N: usize = 1024;

/// dst[i] = src[i] * 5 + 7, root views to title memory; the source arrives through a GPU copy and
/// the destination is cleared first with a GPU fill, all in one list.
#[test]
fn compute_dispatch_writes_title_memory() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (staged, src, dst) = (0, 16384, 32768);
    pool.write(staged, &(0..N as u32).collect::<Vec<_>>());
    pool.write(dst, &[0xEEEE_EEEEu32; N]);
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();

    let mut l = NativeList::default();
    l.copy_buffer_region(pool.va(src), pool.va(staged), N * 4);
    l.fill(pool.va(dst), N * 4, 0);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[5, 7]);
    l.set_compute_root_view(1, pool.va(src));
    l.set_compute_root_view(2, pool.va(dst));
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    run(&h, &[&l]);

    let out: Vec<u32> = pool.read(dst, N);
    assert!(
        out.iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7),
        "{:?}",
        &out[..8]
    );
}

fn fixture_affine() -> Vec<u8> {
    let p = format!(
        "{}/../victory-metal/tests/affine.dxil",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(p).unwrap()
}

/// Descriptors the way a title writes them: `CreateShaderResourceView`-family device calls leave a
/// record in 32-byte heap slots, which `mirror` decodes into the backend's heap; the dispatch finds
/// its buffers through a descriptor table. Two submissions on one queue, the second waiting on the
/// first's fence value (no CPU wait between).
#[test]
fn descriptor_table_buffers_decode_into_metal_heap() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, dst) = (0, 16384);
    pool.write(src, &(0..64u32).collect::<Vec<_>>());
    let slots = vec![0u32; 8 * 8]; // 8 views of 32 bytes: the title's descriptor memory
    let start = slots.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 8).unwrap();

    // CreateShaderResourceView / UnorderedAccessView of memory the title placed itself (raw buffers).
    let placed = |at: usize| {
        let mut d = [0u32; 16];
        d[1] = 1; // buffer view
        d[6] = 64; // elements
        d[8] = 1; // raw (SRV flags word)
        d[10] = pool.va(at) as u32;
        d[11] = (pool.va(at) >> 32) as u32;
        d
    };
    let mut resource = [0u32; 14];
    resource[0] = 1; // buffer
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, &placed(src), start);
    let mut uav = placed(dst);
    (uav[4], uav[6]) = (64, 0); // UAV words: element count at 4, stride at 5, flags at 8
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav, start + 32);
    assert!(h.mirror(start).unwrap() && h.mirror(start + 32).unwrap());

    let pso = h.compute_pipeline(&fixture("heap_rw"), [64, 1, 1]).unwrap();
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[3, 1]);
    l.set_compute_root_table(1, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    // A second list on the same queue reuses the output as its input through a root view.
    let aff = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();
    let mut l2 = NativeList::default();
    l2.set_compute_pipeline(&aff);
    l2.set_compute_root_constants(0, &[2, 0]);
    l2.set_compute_root_view(1, pool.va(dst));
    l2.set_compute_root_view(2, pool.va(dst + 4096));
    l2.dispatch(1, 1, 1).unwrap();

    let q = h.backend.create_queue().unwrap();
    let (a, b) = (
        h.backend.create_fence(0).unwrap(),
        h.backend.create_fence(0).unwrap(),
    );
    h.execute(&q, &[&l], &[], &[(&a, 1)]).unwrap();
    h.execute(&q, &[&l2], &[(&a, 1)], &[(&b, 1)]).unwrap();
    assert!(h.backend.wait(&b, 1, 10_000));
    assert_eq!(
        pool.read::<u32>(dst, 64),
        (0..64).map(|i| i * 3 + 1).collect::<Vec<_>>()
    );
    assert_eq!(
        pool.read::<u32>(dst + 4096, 64),
        (0..64).map(|i| (i * 3 + 1) * 2).collect::<Vec<_>>()
    );
    drop(slots);
}

/// A title copies view records between heap slots with plain stores (no device call): the
/// submission mirrors a bound table's entries whose record changed.
#[test]
fn table_records_copied_by_the_title_are_mirrored_at_submission() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, dst) = (0, 16384);
    pool.write(src, &(0..64u32).collect::<Vec<_>>());
    let mut slots = vec![0u32; 8 * 8];
    let start = slots.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 8).unwrap();
    let placed = |at: usize| {
        let mut d = [0u32; 16];
        (d[1], d[6], d[8]) = (1, 64, 1); // raw buffer view of 64 elements
        d[10] = pool.va(at) as u32;
        d[11] = (pool.va(at) >> 32) as u32;
        d
    };
    let mut resource = [0u32; 14];
    resource[0] = 1;
    // Made in slots 4 and 5 (mirrored there), then copied by the title to 0 and 1.
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, &placed(src), start + 4 * 32);
    let mut uav = placed(dst);
    (uav[4], uav[6]) = (64, 0);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav, start + 5 * 32);
    assert!(h.mirror(start + 4 * 32).unwrap() && h.mirror(start + 5 * 32).unwrap());
    slots.copy_within(4 * 8..6 * 8, 0);

    let pso = h.compute_pipeline(&fixture("heap_rw"), [64, 1, 1]).unwrap();
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[3, 1]);
    l.set_compute_root_table(1, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    let q = h.backend.create_queue().unwrap();
    let done = h.backend.create_fence(0).unwrap();
    h.execute(&q, &[&l], &[], &[(&done, 1)]).unwrap();
    assert!(h.backend.wait(&done, 1, 10_000));
    assert_eq!(pool.read::<u32>(dst, 64), (0..64).map(|i| i * 3 + 1).collect::<Vec<_>>());
    drop(slots);
}

/// What one encoder is named for a heap (two entries naming one pool buffer: one allocation), for
/// one dispatch through the heap and then for five; the dispatches' results are checked.
fn named_for_a_heap(h: &Host) -> [u64; 2] {
    let pool = Pool::new(h, 4);
    let (src, dst) = (0, 16384);
    pool.write(src, &(0..64u32).collect::<Vec<_>>());
    let slots = vec![0u32; 8 * 8];
    let start = slots.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 8).unwrap();
    let placed = |at: usize| {
        let mut d = [0u32; 16];
        (d[1], d[6], d[8]) = (1, 64, 1); // raw buffer view of 64 elements
        d[10] = pool.va(at) as u32;
        d[11] = (pool.va(at) >> 32) as u32;
        d
    };
    let mut resource = [0u32; 14];
    resource[0] = 1;
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, &placed(src), start);
    let mut uav = placed(dst);
    (uav[4], uav[6]) = (64, 0);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav, start + 32);
    assert!(h.mirror(start).unwrap() && h.mirror(start + 32).unwrap());

    let pso = h.compute_pipeline(&fixture("heap_rw"), [64, 1, 1]).unwrap();
    let named = |dispatches: u32| {
        let before = h.backend.residency_counts().heap_resources;
        let mut l = NativeList::default();
        l.set_heaps(Some(&heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_constants(0, &[3, 1]);
        l.set_compute_root_table(1, &heap, 0);
        for _ in 0..dispatches {
            l.dispatch(1, 1, 1).unwrap();
        }
        run(h, &[&l]);
        h.backend.residency_counts().heap_resources - before
    };
    let named = [named(1), named(5)];
    assert_eq!(pool.read::<u32>(dst, 64), (0..64).map(|i| i * 3 + 1).collect::<Vec<_>>());
    drop(slots);
    named
}

/// Without residency sets, a heap's resources are named to an encoder once, not at every dispatch:
/// naming each entry at each draw made frame time grow with the entries the title fills. Two
/// entries naming one pool buffer are one allocation, so a dispatch names the heap and that buffer.
#[test]
fn a_heap_is_named_once_per_encoder() {
    if host().is_none() {
        return;
    }
    let h = Host::new(Metal::new().unwrap().with_residency_sets(false));
    // The heap buffer and the one pool buffer, for one dispatch as for five in one encoder.
    assert_eq!(named_for_a_heap(&h), [2, 2]);
}

/// With the heaps' sets (the default on a Mac with 32 GiB or more) what a heap names is resident
/// through the heap's own set, given to the command buffer: nothing is named to the encoders for
/// it (in GTA VI's world that was 37,000 allocations for every pass, about 155 ms of kernel time
/// per drawing render pass), and the title's pools are in no set of their own (all of title
/// memory would stay resident).
#[test]
fn a_heap_is_resident_through_its_set() {
    if host().is_none() {
        return;
    }
    let h = Host::new(Metal::new().unwrap().with_heap_sets());
    if !h.backend.residency_sets() {
        eprintln!("skipped: no residency sets on this system");
        return;
    }
    assert_eq!(named_for_a_heap(&h), [0, 0]);
    let c = h.backend.residency_counts();
    assert_eq!((c.heap_allocations, c.added, c.queues), (1, 0, 0), "{c:?}");
    // The heap buffer and the pool buffer its entries name.
    assert!(c.heap_set_bytes >= 4 * 16384, "{c:?}");
}

type Vertex = [f32; 8];

/// Issue #15 (writes that land elsewhere): a descriptor the title reuses for a buffer the host
/// cannot reach kept the host entry of the buffer it named before, so a shader went on writing
/// that buffer. The entry now shows nothing: the dispatch still runs, and the old buffer stays as
/// it was.
#[test]
fn a_view_of_unmapped_memory_no_longer_shows_the_previous_buffer() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, dst) = (0, 16384);
    pool.write(src, &(0..64u32).collect::<Vec<_>>());
    let slots = vec![0u32; 8 * 8];
    let start = slots.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 8).unwrap();
    let view = |va: u64| {
        let mut d = [0u32; 16];
        (d[1], d[6], d[8]) = (1, 64, 1); // raw buffer view of 64 elements
        (d[10], d[11]) = (va as u32, (va >> 32) as u32);
        d
    };
    let mut resource = [0u32; 14];
    resource[0] = 1;
    crate::descriptor::create_placed_srv(
        std::ptr::null_mut(),
        &resource,
        &view(pool.va(src)),
        start,
    );
    let uav = |va: u64| {
        let mut d = view(va);
        (d[4], d[6]) = (64, 0);
        crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &d, start + 32);
    };
    uav(pool.va(dst));
    assert!(h.mirror(start).unwrap() && h.mirror(start + 32).unwrap());

    let pso = h.compute_pipeline(&fixture("heap_rw"), [64, 1, 1]).unwrap();
    let dispatch = || {
        let mut l = NativeList::default();
        l.set_heaps(Some(&heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_constants(0, &[3, 1]);
        l.set_compute_root_table(1, &heap, 0);
        l.dispatch(1, 1, 1).unwrap();
        run(&h, &[&l]);
    };
    dispatch();
    assert_eq!(
        pool.read::<u32>(dst, 64),
        (0..64).map(|i| i * 3 + 1).collect::<Vec<_>>()
    );

    // The same slot, now a view of memory in no pool.
    pool.write(dst, &[0xEEEE_EEEEu32; 64]);
    uav(0x10_0000_0000);
    assert!(h.mirror(start + 32).is_err(), "not in a pool");
    dispatch();
    assert_eq!(
        pool.read::<u32>(dst, 64),
        [0xEEEE_EEEEu32; 64],
        "written through a stale entry"
    );

    // Mapped memory again: the real view comes back.
    uav(pool.va(dst));
    assert!(h.mirror(start + 32).unwrap());
    dispatch();
    assert_eq!(
        pool.read::<u32>(dst, 64),
        (0..64).map(|i| i * 3 + 1).collect::<Vec<_>>()
    );
    drop(slots);
}

#[test]
fn submission_retries_an_unchanged_buffer_descriptor_after_mapping() {
    let host = super::host::NativeHost::new(super::vertex_tests::Capture::default());
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = host.heap(HeapKind::Resource, handle, 1).unwrap();
    let address = 0x80100;
    crate::descriptor::create_cbv(std::ptr::null_mut(), &[address, 0, 256, 0], handle);

    let mut list = NativeList::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(
        &[Rc::new(victory_backend::resources::TextureDesc::d2(victory_backend::resources::Format::Rgba8Unorm, 1, 1))],
        None,
    );
    list.set_heaps(Some(&heap), None);
    list.set_graphics_root_table(0, &heap, 0);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    let submit = || host.execute(&(), &[&list], &[], &[]).unwrap();

    submit();
    assert!(matches!(
        host.backend.buffer_writes.borrow().as_slice(),
        [(0, _, _, 0)]
    ));

    // While the pool table has not changed, later draws do not re-encode the null view.
    submit();
    assert_eq!(host.backend.buffer_writes.borrow().len(), 1);

    let pool = PoolInfo {
        id: 1,
        base: 0x40000,
        len: 0x4000,
        memory_entry: None,
    };
    host.memory().mapped(0x80000, pool.len, &pool, 0);
    submit();
    assert_eq!(
        &host.backend.buffer_writes.borrow()[1..],
        &[(0, pool.base, pool.base + 0x100, 256)]
    );

    submit();
    assert_eq!(host.backend.buffer_writes.borrow().len(), 2);
}

/// A buffer view of address 0 (a null view) is the null view for good: a change to the pool table
/// (world streaming maps pages between almost every submission) does not make it be tried again.
#[test]
fn a_null_buffer_view_is_not_retried_when_the_pool_table_changes() {
    let host = super::host::NativeHost::new(super::vertex_tests::Capture::default());
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = host.heap(HeapKind::Resource, handle, 1).unwrap();
    crate::descriptor::create_cbv(std::ptr::null_mut(), &[0, 0, 256, 0], handle);

    let mut list = NativeList::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(
        &[Rc::new(victory_backend::resources::TextureDesc::d2(victory_backend::resources::Format::Rgba8Unorm, 1, 1))],
        None,
    );
    list.set_heaps(Some(&heap), None);
    list.set_graphics_root_table(0, &heap, 0);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    let submit = || host.execute(&(), &[&list], &[], &[]).unwrap();

    submit();
    assert!(matches!(
        host.backend.buffer_writes.borrow().as_slice(),
        [(0, _, _, 0)]
    ));

    // Pages mapped elsewhere move the pool table's generation on.
    for (n, at) in [0x80000usize, 0x90000].into_iter().enumerate() {
        let pool = PoolInfo {
            id: n + 1,
            base: 0x40000 + n * 0x4000,
            len: 0x4000,
            memory_entry: None,
        };
        host.memory().mapped(at, pool.len, &pool, 0);
        submit();
    }
    assert_eq!(host.backend.buffer_writes.borrow().len(), 1);
}

#[test]
fn submission_retries_an_unrealized_resource_descriptor_once_registered() {
    let host = super::host::NativeHost::new(super::vertex_tests::Capture::default());
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = host.heap(HeapKind::Resource, handle, 1).unwrap();
    let res_key = 0xDEAD_BEEFusize;
    crate::descriptor::create_srv(std::ptr::null_mut(), res_key as _, std::ptr::null(), handle);

    let pool = PoolInfo {
        id: 1,
        base: 0x40000,
        len: 0x4000,
        memory_entry: None,
    };
    host.memory().mapped(0x80000, pool.len, &pool, 0);

    let mut list = NativeList::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(
        &[Rc::new(victory_backend::resources::TextureDesc::d2(victory_backend::resources::Format::Rgba8Unorm, 1, 1))],
        None,
    );
    list.set_heaps(Some(&heap), None);
    list.set_graphics_root_table(0, &heap, 0);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    let submit = || host.execute(&(), &[&list], &[], &[]).unwrap();

    // Before the resource is registered, mirror returns Ok(false) and marks pending_mirror.
    submit();
    submit();
    assert!(host.backend.buffer_writes.borrow().is_empty());

    // Once registered, resources_epoch advances and the next submission mirrors the view.
    host.register_buffer(res_key, 0x80200, 512);
    submit();
    assert_eq!(
        host.backend.buffer_writes.borrow().as_slice(),
        &[(0, pool.base, pool.base + 0x200, 512)]
    );
    crate::descriptor::forget(res_key);
}

fn vertex(x: f32, y: f32, c: [f32; 4]) -> Vertex {
    [x, y, 0.5, 0.5, c[0], c[1], c[2], c[3]]
}

/// Clear a render target, draw a triangle with a texture and sampler reached through descriptor
/// tables (written the title's way, then decoded), and read the pixels back.
#[test]
fn clear_and_draw_into_render_target() {
    let Some(h) = host() else { return };
    const W: usize = 16;
    let pool = Pool::new(&h, 2);
    pool.write(
        0,
        &[
            vertex(-0.5, -0.5, [1.0, 0.0, 0.0, 1.0]),
            vertex(0.5, -0.5, [1.0, 0.0, 0.0, 1.0]),
            vertex(0.0, 0.5, [1.0, 0.0, 0.0, 1.0]),
        ],
    );

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
    let pso = h
        .render_pipeline(&RenderPipelineDesc {
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
            depth: None,
            depth_test: None,
            cull: Cull::None,
            ..Default::default()
        })
        .unwrap();
    let mut rt = TextureDesc::d2(Format::Rgba8Unorm, W, W);
    rt.render_target = true;
    let color = h.texture(&rt).unwrap();
    let white = h
        .texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1))
        .unwrap();
    h.backend.upload_texture(&white, 0, 0, &[255; 4], 4, 0);
    h.register_texture(0xA11CE, &white);

    let views = vec![0u32; 8 * 4];
    let smp = vec![0u32; 4 * 4];
    let (vstart, sstart) = (views.as_ptr() as u64, smp.as_ptr() as u64);
    let vheap = h.heap(HeapKind::Resource, vstart, 4).unwrap();
    let sheap = h.heap(HeapKind::Sampler, sstart, 4).unwrap();
    // CreateShaderResourceView(texture, Texture2D, mips 0..1) and CreateSamplerX (bilinear, clamp).
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null_mut(), 0xA11CE as _, &srv, vstart);
    let mut sw = [0u32; 27];
    (sw[1], sw[2], sw[6], sw[7], sw[8]) = (1, 1, 3, 3, 3);
    crate::descriptor::create_sampler_x(std::ptr::null_mut(), &sw, sstart);
    assert!(h.mirror(vstart).unwrap() && h.mirror(sstart).unwrap());

    let mut l = NativeList::default();
    l.set_render_targets(std::slice::from_ref(&color), None);
    l.clear_render_target(&color, [0.0, 0.0, 1.0, 1.0]);
    l.set_graphics_pipeline(&pso, Topology::Triangles);
    l.set_heaps(Some(&vheap), Some(&sheap));
    l.set_graphics_root_constants(0, &[0.5f32.to_bits(), 0, 0, 0]);
    l.set_graphics_root_table(1, &vheap, 0);
    l.set_graphics_root_table(2, &sheap, 0);
    l.set_vertex_buffer(0, pool.va(0));
    l.draw_instanced(3, 1, 0, 0).unwrap();
    run(&h, &[&l]);

    let mut out = vec![0u8; W * W * 4];
    h.backend.read_texture(&color, 0, 0, &mut out, W * 4, 0);
    let px = |x: usize, y: usize| <[u8; 4]>::try_from(&out[(y * W + x) * 4..][..4]).unwrap();
    assert_eq!(px(8, 8), [255, 0, 0, 255], "inside the triangle");
    assert_eq!(px(0, 0), [0, 0, 255, 255], "cleared corner");
    assert_eq!(px(15, 15), [0, 0, 255, 255], "cleared corner");
    let _ = Rc::strong_count(&color);
    drop((views, smp));
}

/// Compute writes a depth texture through an unordered view of its depth plane, and a draw after
/// it in the same list tests against what compute wrote. A full-screen draw at depth 0.02 with
/// `LessEqual` passes where `tex_fill` stored x / 255 >= 0.02 (columns 6 and up) and fails left of
/// them. Without the twin the unordered view is of the depth texture itself, and the depth test
/// does not see the compute writes: every pixel failed on this Mac (M4 Pro).
#[test]
fn a_draw_tests_against_the_depth_compute_wrote_through_an_unordered_view() {
    let Some(h) = host() else { return };
    const W: usize = 16;
    let pool = Pool::new(&h, 4);
    const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    pool.write(0, &[vertex(-1.0, -1.0, GREEN), vertex(3.0, -1.0, GREEN), vertex(-1.0, 3.0, GREEN)]);
    let attrs = [
        VertexAttr { format: VertexFormat::Float2, slot: 0, offset: 0 },
        VertexAttr { format: VertexFormat::Float2, slot: 0, offset: 8 },
        VertexAttr { format: VertexFormat::Float4, slot: 0, offset: 16 },
    ];
    let pso = h
        .render_pipeline(&RenderPipelineDesc {
            vs: &fixture("tri_vs"),
            ps: Some(&fixture("tri_ps")),
            vs_layout: None,
            ps_layout: None,
            attributes: &attrs,
            streams: &[VertexStream { slot: 0, stride: 32, per_instance: false }],
            color: &[Format::Rgba8Unorm],
            depth: Some(Format::D32Float),
            depth_test: Some((false, Compare::LessEqual)),
            cull: Cull::None,
            ..Default::default()
        })
        .unwrap();
    let fill = h.compute_pipeline(&fixture("tex_fill"), [8, 8, 1]).unwrap();
    let mut rt = TextureDesc::d2(Format::Rgba8Unorm, W, W);
    rt.render_target = true;
    let color = h.texture(&rt).unwrap();
    // A title depth texture: DXGI 40 (D32_FLOAT); flags 2 | 4, a depth target compute also writes.
    let res = crate::resource::test_texture_at(pool.va(16384), (W as u64, W as u32), 1, 40, victory_formats::texture::Swizzle::LINEAR.0, 2 | 4);
    let desc = crate::resource::texture_info(res).unwrap().desc().unwrap();
    assert_eq!(desc.format, Format::D32Float);
    let depth = h.texture(&desc).unwrap();
    h.register_texture(res as usize, &depth);
    let white = h.texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1)).unwrap();
    h.backend.upload_texture(&white, 0, 0, &[255; 4], 4, 0);
    h.register_texture(0xA11CF, &white);

    // Entry 0: the pixel shader's texture; entry 1: the R32_FLOAT unordered view of the depth.
    let views = vec![0u32; 8 * 4];
    let smp = vec![0u32; 4 * 4];
    let (vstart, sstart) = (views.as_ptr() as u64, smp.as_ptr() as u64);
    let vheap = h.heap(HeapKind::Resource, vstart, 4).unwrap();
    let sheap = h.heap(HeapKind::Sampler, sstart, 4).unwrap();
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null_mut(), 0xA11CF as _, &srv, vstart);
    crate::descriptor::create_uav(std::ptr::null_mut(), res, std::ptr::null_mut(), &[41u32, 4, 0, 0, 0, 0, 0, 0, 0, 0], vstart + 32);
    let mut sw = [0u32; 27];
    (sw[1], sw[2], sw[6], sw[7], sw[8]) = (1, 1, 3, 3, 3);
    crate::descriptor::create_sampler_x(std::ptr::null_mut(), &sw, sstart);
    assert!(h.mirror(vstart).unwrap() && h.mirror(vstart + 32).unwrap() && h.mirror(sstart).unwrap());
    assert!(h.twin_of(&depth).is_some(), "the unordered view made a twin");

    let mut l = NativeList::default();
    l.set_render_targets(std::slice::from_ref(&color), Some(&depth));
    l.clear_render_target(&color, [0.0, 0.0, 0.0, 1.0]);
    l.clear_depth(&depth, 1.0);
    l.set_heaps(Some(&vheap), Some(&sheap));
    l.set_compute_pipeline(&fill);
    l.set_compute_root_table(0, &vheap, 1);
    l.dispatch((W / 8) as u32, (W / 8) as u32, 1).unwrap();
    l.set_graphics_pipeline(&pso, Topology::Triangles);
    l.set_graphics_root_constants(0, &[0.02f32.to_bits(), 0, 0, 0]);
    l.set_graphics_root_table(1, &vheap, 0);
    l.set_graphics_root_table(2, &sheap, 0);
    l.set_vertex_buffer(0, pool.va(0));
    l.draw_instanced(3, 1, 0, 0).unwrap();
    run(&h, &[&l]);

    let mut out = vec![0u8; W * W * 4];
    h.backend.read_texture(&color, 0, 0, &mut out, W * 4, 0);
    let row: Vec<u8> = (0..W).map(|x| out[(8 * W + x) * 4 + 1]).collect();
    let want: Vec<u8> = (0..W).map(|x| if x >= 6 { 255 } else { 0 }).collect();
    assert_eq!(row, want, "green where the depth compute wrote is 0.02 or more");
    drop((views, smp));
}

/// A draw left with no render target (the title's targets were of formats the host does not
/// represent) is dropped by itself: Metal opens no pass without an attachment, and the value
/// write recorded with it still runs.
#[test]
fn a_draw_with_no_render_target_is_dropped_alone() {
    let Some(h) = host() else { return };
    const W: usize = 16;
    let pool = Pool::new(&h, 2);
    pool.write(
        0,
        &[
            vertex(-0.5, -0.5, [1.0, 0.0, 0.0, 1.0]),
            vertex(0.5, -0.5, [1.0, 0.0, 0.0, 1.0]),
            vertex(0.0, 0.5, [1.0, 0.0, 0.0, 1.0]),
        ],
    );

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
    let pso = h
        .render_pipeline(&RenderPipelineDesc {
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
            depth: None,
            depth_test: None,
            cull: Cull::None,
            ..Default::default()
        })
        .unwrap();
    let flag = 16384;
    pool.write(flag, &[0u64]);
    let mut l = NativeList::default();
    l.set_render_targets(&[], None);
    l.set_graphics_pipeline(&pso, Topology::Triangles);
    l.set_graphics_root_constants(0, &[0.5f32.to_bits(), 0, 0, 0]);
    l.set_vertex_buffer(0, pool.va(0));
    l.draw_instanced(3, 1, 0, 0).unwrap();
    l.write_value(pool.va(flag), 0xD6D, 8);
    run(&h, &[&l]);

    assert_eq!(h.no_targets(), 1);
    assert_eq!(pool.read::<u64>(flag, 1), [0xD6D], "the value write ran");
}

/// `ExecuteIndirect` with a count buffer: three dispatch records (2, 4 and 8 groups), and the count
/// of records to run (2) written by a copy earlier in the same list, both read on the GPU when the
/// dispatches run: only the first two run, so 256 values are written, not 512 (or none, if the
/// count were read before the copy).
#[test]
fn a_count_written_earlier_on_the_gpu_limits_the_indirect_dispatches() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, dst, records, staged, count) = (0, 16384, 32768, 32816, 32832);
    pool.write(src, &(0..N as u32).collect::<Vec<_>>());
    pool.write(dst, &[0u32; N]);
    pool.write(records, &[2u32, 1, 1, 4, 1, 1, 8, 1, 1]);
    pool.write(staged, &[2u32]);
    pool.write(count, &[0u32]);
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();

    let mut l = NativeList::default();
    l.copy_buffer_region(pool.va(count), pool.va(staged), 4);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[5, 7]);
    l.set_compute_root_view(1, pool.va(src));
    l.set_compute_root_view(2, pool.va(dst));
    for i in 0..3 {
        l.dispatch_indirect_counted(pool.va(records + 12 * i as usize), pool.va(count), i)
            .unwrap();
    }
    run(&h, &[&l]);

    let out: Vec<u32> = pool.read(dst, N);
    let want: Vec<u32> = (0..N as u32)
        .map(|i| if i < 256 { i * 5 + 7 } else { 0 })
        .collect();
    assert_eq!(out, want);
}

/// `ExecuteIndirect` of a dispatch: the record is read on the GPU when the dispatch runs, so a copy
/// earlier in the same list that writes it (as a culling pass would) decides the group count.
#[test]
fn an_indirect_dispatch_reads_a_record_written_earlier_on_the_gpu() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, dst, staged, record) = (0, 16384, 32768, 32800);
    pool.write(src, &(0..N as u32).collect::<Vec<_>>());
    pool.write(dst, &[0u32; N]);
    pool.write(staged, &[(N / 128) as u32, 1, 1]);
    pool.write(record, &[0u32; 3]);
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();

    let mut l = NativeList::default();
    l.copy_buffer_region(pool.va(record), pool.va(staged), 12);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[5, 7]);
    l.set_compute_root_view(1, pool.va(src));
    l.set_compute_root_view(2, pool.va(dst));
    l.dispatch_indirect(pool.va(record)).unwrap();
    run(&h, &[&l]);

    let out: Vec<u32> = pool.read(dst, N);
    assert!(out[..N / 2].iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7), "{:?}", &out[..8]);
    assert!(out[N / 2..].iter().all(|v| *v == 0), "only the record's groups ran");
}

/// A command that names title memory outside every mapped pool is not encoded (no GPU fault) and
/// is dropped by itself: the value writes, and the dispatch, recorded around it still run.
#[test]
fn a_command_at_unmapped_title_memory_is_dropped_alone() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, dst, flag, done) = (0, 16384, 32768, 32776);
    pool.write(src, &(0..N as u32).collect::<Vec<_>>());
    pool.write(dst, &[0xEEEE_EEEEu32; N]);
    pool.write(flag, &[0u64, 0]);
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();

    let mut l = NativeList::default();
    l.write_value(pool.va(flag), 0xD6D, 8);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[5, 7]);
    // Its source is no title memory the host knows.
    l.set_compute_root_view(1, 0xDEAD_0000);
    l.set_compute_root_view(2, pool.va(dst));
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    l.set_compute_root_view(1, pool.va(src));
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    // A copy from no title memory the host knows.
    l.copy_buffer_region(pool.va(src), 0xDEAD_1000, 64);
    l.write_value(pool.va(done), 0xD0E, 8);
    run(&h, &[&l]);

    assert_eq!(h.unmapped(), 2);
    assert_eq!(pool.read::<u64>(flag, 2), [0xD6D, 0xD0E], "the value writes ran");
    let out: Vec<u32> = pool.read(dst, N);
    assert!(out.iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7), "the second dispatch ran: {:?}", &out[..4]);
    let kept: Vec<u32> = pool.read(src, 16);
    assert_eq!(kept, (0..16).collect::<Vec<u32>>(), "the copy was dropped, not run");
}

/// A committed buffer's memory is the host's own (no pool holds it): root views and copies into it
/// reach it while the buffer lives, and are refused as unmapped once it is released.
#[test]
fn root_views_and_copies_reach_committed_buffer_memory() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 1);
    pool.write(0, &(0..N as u32).collect::<Vec<_>>());
    // What `XMemAllocDefault` gives a committed buffer: host heap memory, 64 KiB aligned.
    let layout = std::alloc::Layout::from_size_align(3 * N * 4, 1 << 16).unwrap();
    // SAFETY: non-zero size; freed below, after the GPU is done with it.
    let mem = unsafe { std::alloc::alloc_zeroed(layout) };
    let at = mem as u64;
    let (src, dst) = (at + (N * 4) as u64, at + (2 * N * 4) as u64);
    let key = 0xC0FFEE;
    h.register_own_buffer(key, at, layout.size() as u64);
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();

    let mut l = NativeList::default();
    l.copy_buffer_region(src, pool.va(0), N * 4);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[5, 7]);
    l.set_compute_root_view(1, src);
    l.set_compute_root_view(2, dst);
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    run(&h, &[&l]);
    assert_eq!(h.unmapped(), 0);
    // SAFETY: inside the allocation; read after the fence wait.
    let out = unsafe { std::slice::from_raw_parts(dst as *const u32, N) };
    assert!(out.iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7), "{:?}", &out[..4]);

    h.release_resource(key);
    run(&h, &[&l]);
    assert_eq!(h.unmapped(), 2, "the copy and the dispatch were refused once the buffer went");
    // SAFETY: allocated above with this layout; the GPU is idle.
    unsafe { std::alloc::dealloc(mem, layout) };
}

/// A dispatch that reads a root parameter nothing was set in is dropped by itself: the value
/// write recorded before it and the dispatch after it still run (a queue may wait for that value).
#[test]
fn a_dispatch_with_an_unset_root_parameter_is_dropped_alone() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, dst, flag) = (0, 16384, 32768);
    pool.write(src, &(0..N as u32).collect::<Vec<_>>());
    pool.write(dst, &[0xEEEE_EEEEu32; N]);
    pool.write(flag, &[0u64]);
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();

    let mut l = NativeList::default();
    l.write_value(pool.va(flag), 0xD6D, 8);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[5, 7]);
    // Root parameter 1 is not set yet: this dispatch cannot be given its arguments.
    l.set_compute_root_view(2, pool.va(dst));
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    l.set_compute_root_view(1, pool.va(src));
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    run(&h, &[&l]);

    assert_eq!(h.unset_roots(), 1);
    assert_eq!(pool.read::<u64>(flag, 1), [0xD6D], "the value write ran");
    let out: Vec<u32> = pool.read(dst, N);
    assert!(out.iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7), "the second dispatch ran: {:?}", &out[..4]);
}

/// A dispatch the host cannot give its arguments for another reason (here a root parameter that is
/// not what its shader reads it as) is dropped by itself, not the whole piece: the value write
/// recorded after it still runs.
#[test]
fn a_dispatch_the_host_cannot_prepare_is_dropped_alone() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let flag = 32768;
    pool.write(flag, &[0u64]);
    let pso = Rc::new(super::list::ComputePso {
        pso: h.backend.create_compute_pipeline(&fixture_affine()).unwrap(),
        threads: [64, 1, 1],
        binding: Some(super::list::Binding {
            dwords: 4,
            user_data: vec![victory_shader::emit::UserData {
                index: 0,
                slot: 0,
                root: victory_shader::RootType::Constants,
                dword: 0,
            }],
            ..Default::default()
        }),
    });

    let mut l = NativeList::default();
    l.set_compute_pipeline(&pso);
    // Slot 0 holds a root view where the shader reads root constants.
    l.set_compute_root_view(0, pool.va(0));
    l.dispatch(1, 1, 1).unwrap();
    l.write_value(pool.va(flag), 0xD6D, 8);
    run(&h, &[&l]);

    assert_eq!(h.failed(), 1);
    assert_eq!(pool.read::<u64>(flag, 1), [0xD6D], "the value write ran");
}

/// The triangle pipeline of [`clear_and_draw_into_render_target`].
fn triangle_pipeline(h: &Host) -> Rc<<Metal as GraphicsBackend>::RenderPipeline> {
    let attr = |format, offset| VertexAttr { format, slot: 0, offset };
    h.render_pipeline(&RenderPipelineDesc {
        vs: &fixture("tri_vs"),
        ps: Some(&fixture("tri_ps")),
        vs_layout: None,
        ps_layout: None,
        attributes: &[attr(VertexFormat::Float2, 0), attr(VertexFormat::Float2, 8), attr(VertexFormat::Float4, 16)],
        streams: &[VertexStream {
            slot: 0,
            stride: 32,
            per_instance: false,
        }],
        color: &[Format::Rgba8Unorm],
        depth: None,
        depth_test: None,
        cull: Cull::None,
        ..Default::default()
    })
    .unwrap()
}

/// A dropped command still ends the pass it would have ended. Two dispatches with a draw between
/// them are two compute passes, the second reading what the first wrote; they stay two when the
/// draw is dropped (it has no render target). A dropped dispatch between two dispatches never
/// ended a pass, and does not start to.
#[test]
fn a_dropped_command_between_two_dispatches_keeps_them_in_two_passes() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, mid, dst) = (0, 16384, 32768);
    pool.write(src, &(0..N as u32).collect::<Vec<_>>());
    pool.write(mid, &[0xEEEE_EEEEu32; N]);
    pool.write(dst, &[0xEEEE_EEEEu32; N]);
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();
    let tri = triangle_pipeline(&h);

    // The producer, a draw with no target (dropped), the consumer.
    let mut l = NativeList::default();
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[5, 7]);
    l.set_compute_root_view(1, pool.va(src));
    l.set_compute_root_view(2, pool.va(mid));
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    l.set_render_targets(&[], None);
    l.set_graphics_pipeline(&tri, Topology::Triangles);
    l.draw_instanced(3, 1, 0, 0).unwrap();
    l.set_compute_root_constants(0, &[3, 1]);
    l.set_compute_root_view(1, pool.va(mid));
    l.set_compute_root_view(2, pool.va(dst));
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    run(&h, &[&l]);

    assert_eq!(h.no_targets(), 1);
    assert_eq!(h.passes(), 2, "the dropped draw still parts the two dispatches");
    let out: Vec<u32> = pool.read(dst, N);
    assert!(out.iter().enumerate().all(|(i, v)| *v == (i as u32 * 5 + 7) * 3 + 1), "the consumer read the producer's output: {:?}", &out[..4]);

    // The same two dispatches around a dispatch dropped for an unset root parameter: one pass, as
    // the three would have been.
    let mut first = NativeList::default();
    first.set_compute_pipeline(&pso);
    first.set_compute_root_constants(0, &[5, 7]);
    first.set_compute_root_view(1, pool.va(src));
    first.set_compute_root_view(2, pool.va(mid));
    first.dispatch((N / 64) as u32, 1, 1).unwrap();
    let mut unset = NativeList::default();
    unset.set_compute_pipeline(&pso);
    unset.set_compute_root_constants(0, &[5, 7]);
    unset.set_compute_root_view(2, pool.va(mid));
    unset.dispatch((N / 64) as u32, 1, 1).unwrap();
    let mut last = NativeList::default();
    last.set_compute_pipeline(&pso);
    last.set_compute_root_constants(0, &[2, 0]);
    last.set_compute_root_view(1, pool.va(mid));
    last.set_compute_root_view(2, pool.va(dst));
    last.dispatch((N / 64) as u32, 1, 1).unwrap();
    run(&h, &[&first, &unset, &last]);

    assert_eq!(h.unset_roots(), 1);
    assert_eq!(h.passes(), 3, "a dropped dispatch between dispatches parts nothing");
    let out: Vec<u32> = pool.read(dst, N);
    assert!(out.iter().enumerate().all(|(i, v)| *v == (i as u32 * 5 + 7) * 2), "{:?}", &out[..4]);
}

/// Two draws into the same target with a dispatch between them are two render passes, and stay two
/// when the dispatch is dropped (it reads a root parameter nothing was set in): one encoder for
/// both would lose the ordering of their pool-buffer accesses the backend keeps between passes.
#[test]
fn a_dropped_dispatch_between_two_draws_keeps_them_in_two_passes() {
    let Some(h) = host() else { return };
    const W: usize = 16;
    let pool = Pool::new(&h, 2);
    let triangle = |x: f32, c: [f32; 4]| [vertex(x - 0.4, -0.5, c), vertex(x + 0.4, -0.5, c), vertex(x, 0.5, c)];
    pool.write(0, &triangle(-0.5, [1.0, 0.0, 0.0, 1.0]));
    pool.write(4096, &triangle(0.5, [0.0, 1.0, 0.0, 1.0]));
    let tri = triangle_pipeline(&h);
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();
    let mut rt = TextureDesc::d2(Format::Rgba8Unorm, W, W);
    rt.render_target = true;
    let color = h.texture(&rt).unwrap();
    let white = h.texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1)).unwrap();
    h.backend.upload_texture(&white, 0, 0, &[255; 4], 4, 0);
    h.register_texture(0xA11CF, &white);

    let views = vec![0u32; 8 * 4];
    let smp = vec![0u32; 4 * 4];
    let (vstart, sstart) = (views.as_ptr() as u64, smp.as_ptr() as u64);
    let vheap = h.heap(HeapKind::Resource, vstart, 4).unwrap();
    let sheap = h.heap(HeapKind::Sampler, sstart, 4).unwrap();
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null_mut(), 0xA11CF as _, &srv, vstart);
    let mut sw = [0u32; 27];
    (sw[1], sw[2], sw[6], sw[7], sw[8]) = (1, 1, 3, 3, 3);
    crate::descriptor::create_sampler_x(std::ptr::null_mut(), &sw, sstart);
    assert!(h.mirror(vstart).unwrap() && h.mirror(sstart).unwrap());

    let mut l = NativeList::default();
    l.set_render_targets(std::slice::from_ref(&color), None);
    l.clear_render_target(&color, [0.0, 0.0, 1.0, 1.0]);
    l.set_graphics_pipeline(&tri, Topology::Triangles);
    l.set_heaps(Some(&vheap), Some(&sheap));
    l.set_graphics_root_constants(0, &[0.5f32.to_bits(), 0, 0, 0]);
    l.set_graphics_root_table(1, &vheap, 0);
    l.set_graphics_root_table(2, &sheap, 0);
    l.set_vertex_buffer(0, pool.va(0));
    l.draw_instanced(3, 1, 0, 0).unwrap();
    // Root parameter 1 of the dispatch is never set.
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[5, 7]);
    l.set_compute_root_view(2, pool.va(8192));
    l.dispatch(1, 1, 1).unwrap();
    l.set_vertex_buffer(0, pool.va(4096));
    l.draw_instanced(3, 1, 0, 0).unwrap();
    run(&h, &[&l]);

    assert_eq!(h.unset_roots(), 1);
    assert_eq!(h.passes(), 3, "the clear, and a pass for each draw");
    let mut out = vec![0u8; W * W * 4];
    h.backend.read_texture(&color, 0, 0, &mut out, W * 4, 0);
    let px = |x: usize, y: usize| <[u8; 4]>::try_from(&out[(y * W + x) * 4..][..4]).unwrap();
    assert_eq!(px(4, 8), [255, 0, 0, 255], "the first draw");
    assert_eq!(px(11, 8), [0, 255, 0, 255], "the second draw");
    assert_eq!(px(0, 0), [0, 0, 255, 255], "cleared corner");
    drop((views, smp));
}

/// Pool buffers are made resident through one `MTLResidencySet`, added to each queue once, instead
/// of a `useResource` per dispatch; a buffer leaves it with its pool. The same work with sets off
/// names every buffer on every dispatch, and gives the same results.
#[test]
fn pool_buffers_are_resident_through_one_set_per_queue() {
    if host().is_none() {
        return;
    }
    let on = Host::new(Metal::new().unwrap().with_residency_sets(true));
    if !on.backend.residency_sets() {
        eprintln!("skipped: no residency sets on this system");
        return;
    }
    let off = Host::new(Metal::new().unwrap().with_residency_sets(false));

    let work = |h: &Host, queues: usize, submissions: usize| {
        let pool = Pool::new(h, 4);
        let (src, dst) = (0, 16384);
        pool.write(src, &(0..N as u32).collect::<Vec<_>>());
        let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();
        let qs: Vec<_> = (0..queues).map(|_| h.backend.create_queue().unwrap()).collect();
        let f = h.backend.create_fence(0).unwrap();
        let mut value = 0;
        for i in 0..submissions {
            let mut l = NativeList::default();
            l.set_compute_pipeline(&pso);
            l.set_compute_root_constants(0, &[i as u32 + 2, 1]);
            l.set_compute_root_view(1, pool.va(src));
            l.set_compute_root_view(2, pool.va(dst));
            l.dispatch((N / 64) as u32, 1, 1).unwrap();
            value += 1;
            h.execute(&qs[i % queues], &[&l], &[(&f, value - 1)], &[(&f, value)]).unwrap();
        }
        assert!(h.backend.wait(&f, value, 10_000));
        let last = submissions as u32 + 1;
        let out: Vec<u32> = pool.read(dst, N);
        assert!(out.iter().enumerate().all(|(i, v)| *v == i as u32 * last + 1), "{:?}", &out[..4]);
        pool
    };

    let _pool = work(&on, 2, 6);
    let c = on.backend.residency_counts();
    // One pool buffer, in the set once; the set is on both queues; no dispatch names a buffer.
    assert_eq!(c, victory_metal::metal::ResidencyCounts { use_resource: 0, added: 1, queues: 2, ..Default::default() });

    let _pool = work(&off, 2, 6);
    let c = off.backend.residency_counts();
    assert_eq!((c.added, c.queues), (0, 0));
    assert!(c.use_resource >= 6 * 2, "{c:?}");
}

/// A pool that goes away takes its buffer out of the set (and does not break the next work).
#[test]
fn a_released_pool_leaves_the_residency_set() {
    if host().is_none() {
        return;
    }
    let h = Host::new(Metal::new().unwrap().with_residency_sets(true));
    if !h.backend.residency_sets() {
        return;
    }
    let pool = Pool::new(&h, 2);
    pool.write(0, &(0..64u32).collect::<Vec<_>>());
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();
    let dispatch = |pool: &Pool| {
        let mut l = NativeList::default();
        l.set_compute_pipeline(&pso);
        l.set_compute_root_constants(0, &[2, 0]);
        l.set_compute_root_view(1, pool.va(0));
        l.set_compute_root_view(2, pool.va(4096));
        l.dispatch(1, 1, 1).unwrap();
        run(&h, &[&l]);
    };
    dispatch(&pool);
    assert_eq!(h.backend.residency_counts().added, 1);
    let info = PoolInfo {
        id: 1,
        base: pool.mem.primary as usize,
        len: pool.mem.len,
        memory_entry: None,
    };
    h.memory().pool_released(&info);
    // The next resolve drops the released pool's buffer.
    let second = {
        let mem = Alias::new(2 * page_size().max(16384)).unwrap();
        let info = PoolInfo {
            id: 2,
            base: mem.primary as usize,
            len: mem.len,
            memory_entry: None,
        };
        h.memory().pool_created(&info);
        h.memory().mapped(mem.alias as usize, mem.len, &info, 0);
        Pool { mem }
    };
    second.write(0, &(0..64u32).collect::<Vec<_>>());
    dispatch(&second);
    assert_eq!(h.backend.residency_counts().added, 2);
    assert_eq!(second.read::<u32>(4096, 64), (0..64).map(|i| i * 2).collect::<Vec<_>>());
}

#[test]
fn zero_count_commands_are_not_recorded_and_still_check_pipeline() {
    let Some(h) = host() else { return };
    let mut l = NativeList::default();
    assert!(l.dispatch(0, 1, 1).is_err());
    assert!(l.dispatch_threads([0, 1, 1]).is_err());
    assert!(l.draw_instanced(0, 1, 0, 0).is_err());
    assert!(l.dispatch_mesh(0, 1, 1).is_err());

    let cs = h.compute_pipeline(&fixture("heap_rw"), [64, 1, 1]).unwrap();
    l.set_compute_pipeline(&cs);
    l.dispatch(0, 1, 1).unwrap();
    l.dispatch(1, 0, 1).unwrap();
    l.dispatch(1, 1, 0).unwrap();
    l.dispatch_threads([0, 1, 1]).unwrap();
    l.dispatch_threads([64, 0, 1]).unwrap();
    l.copy_buffer_region(0x1000, 0x2000, 0);
    l.fill(0x1000, 0, 0xAB);

    let gfx = h
        .render_pipeline(&RenderPipelineDesc {
            vs: &fixture("tri_vs"),
            ps: Some(&fixture("tri_ps")),
            vs_layout: None,
            ps_layout: None,
            attributes: &[
                VertexAttr { format: VertexFormat::Float2, slot: 0, offset: 0 },
                VertexAttr { format: VertexFormat::Float2, slot: 0, offset: 8 },
                VertexAttr { format: VertexFormat::Float4, slot: 0, offset: 16 },
            ],
            streams: &[VertexStream {
                slot: 0,
                stride: 32,
                per_instance: false,
            }],
            color: &[Format::Rgba8Unorm],
            depth: None,
            depth_test: None,
            cull: Cull::None,
            ..Default::default()
        })
        .unwrap();
    l.set_graphics_pipeline(&gfx, Topology::Triangles);
    l.set_index_buffer(0x1000, false);
    l.draw_instanced(0, 1, 0, 0).unwrap();
    l.draw_instanced(3, 0, 0, 0).unwrap();
    l.draw_indexed_instanced(0, 1, 0, 0, 0).unwrap();
    l.draw_indexed_instanced(3, 0, 0, 0, 0).unwrap();
    l.dispatch_mesh(0, 1, 1).unwrap();
    assert!(l.is_empty());
}
