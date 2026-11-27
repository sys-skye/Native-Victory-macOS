//! The seam victory-gpu's `host.rs` / `queue.rs` / `fence.rs` would call instead of D3D12, first
//! slice: queues, fences, compute pipelines from DXIL, compute dispatch over title memory.
//!
//! Platform neutral on purpose (no Metal types), so the D3D12 host can implement it too and the
//! title-facing objects (`queue.rs`, `fence.rs`, `hold.rs`) stay as they are. One submission is one
//! `ExecuteCommandLists` worth of work: waits, then dispatches in order, then signals. Nothing in it
//! waits for the GPU; completion is observed through fences.

/// How a dispatch's top-level argument buffer is laid out: root constants inline, root SRV/UAV/CBV
/// as 64-bit GPU addresses, in root-parameter order (what `IRCompiler` makes of a root signature).
#[derive(Default, Clone, Debug)]
pub struct RootArgs(pub Vec<u8>);

impl RootArgs {
    pub fn constants(mut self, words: &[u32]) -> Self {
        words.iter().for_each(|w| self.0.extend(w.to_le_bytes()));
        self
    }

    /// A root view (SRV, UAV or CBV): a GPU address, 8-byte aligned.
    pub fn address(mut self, gpu: u64) -> Self {
        self.0.resize(self.0.len().next_multiple_of(8), 0);
        self.0.extend(gpu.to_le_bytes());
        self
    }
}

/// One root parameter of a [`RootLayout`]: what a translated shader declares at a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RootParam {
    /// `dwords` root constants at `b<register>` of `space`: inline in the argument buffer.
    Constants { register: u32, space: u32, dwords: u32 },
    /// A root CBV (`b`), SRV (`t`) or UAV (`u`): a 64-bit GPU address in the argument buffer.
    Cbv { register: u32, space: u32 },
    Srv { register: u32, space: u32 },
    Uav { register: u32, space: u32 },
}

/// The root signature of translated shaders, which carry none: parameters in the order
/// [`RootArgs`] is filled in, descriptor heaps indexed directly. Without one a backend takes the
/// signature the shader itself embeds (DXIL compiled by dxc), or its own default layout.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RootLayout {
    pub params: Vec<RootParam>,
    /// The shader reads the memory of a raw buffer view out of range as 0 and drops its writes
    /// (`D3D12` hardware does; Metal needs the converter told).
    pub bounds_checked: bool,
    /// Static samplers, `s<n>` at position n in space 0: the words of a
    /// `D3D12_STATIC_SAMPLER_DESC` up to `MaxLOD` (filter, address U, V, W, LOD bias as f32 bits,
    /// anisotropy, comparison, border, minimum and maximum LOD as f32 bits).
    pub static_samplers: Vec<[u32; 10]>,
}

/// One `Dispatch` call: the pipeline, its root arguments, the memory it touches (for residency),
/// and the group counts and sizes.
pub struct Dispatch<'a, B: ComputeBackend + ?Sized> {
    pub pipeline: &'a B::Pipeline,
    pub root: RootArgs,
    /// Every title buffer the shader reaches through `root` (heap-referenced resources are made
    /// resident by the heaps themselves).
    pub resources: Vec<&'a B::Buffer>,
    pub heaps: Heaps<'a, B>,
    pub groups: [usize; 3],
    pub threads_per_group: [usize; 3],
    /// `ExecuteIndirect`: the group counts are read on the GPU from this title address (three
    /// words) instead of `groups`. A record that asks for more than 2^22 groups in all
    /// dispatches nothing.
    pub indirect: Option<(&'a B::Buffer, *const u8)>,
    /// `ExecuteIndirect` with a count buffer: the count's title address and this record's index.
    /// The record runs only if the index is below the count, which the GPU reads when it runs.
    pub count: Option<(&'a B::Buffer, *const u8, u32)>,
}

/// The descriptor heaps a draw or dispatch binds (`SetDescriptorHeaps`): slot 0 and slot 1.
pub struct Heaps<'a, B: ComputeBackend + ?Sized> {
    pub resource: Option<&'a B::Heap>,
    pub sampler: Option<&'a B::Heap>,
}

impl<B: ComputeBackend + ?Sized> Default for Heaps<'_, B> {
    fn default() -> Self {
        Heaps {
            resource: None,
            sampler: None,
        }
    }
}

/// Work for one queue: wait for fence values, run dispatches in order, signal fence values.
pub struct Submission<'a, B: ComputeBackend + ?Sized> {
    pub waits: &'a [(&'a B::Fence, u64)],
    pub dispatches: &'a [Dispatch<'a, B>],
    pub signals: &'a [(&'a B::Fence, u64)],
}

pub trait ComputeBackend {
    /// `ID3D12CommandQueue`.
    type Queue;
    /// `ID3D12Fence`: a 64-bit value that only rises.
    type Fence;
    /// A GPU-visible view of title memory (a pool, wrapped once).
    type Buffer;
    /// A compute pipeline state.
    type Pipeline;
    /// A descriptor heap (`ID3D12DescriptorHeap`): resource (CBV/SRV/UAV) or sampler.
    type Heap;

    fn create_queue(&self) -> Result<Self::Queue, String>;
    fn create_fence(&self, initial: u64) -> Result<Self::Fence, String>;
    /// Wrap `ptr..ptr+len` (page-aligned) without copying; it stays mapped while the buffer lives.
    fn wrap_title_memory(&self, ptr: *mut u8, len: usize) -> Result<Self::Buffer, String>;
    /// The bytes of GPU memory the device holds for this process now, where the backend knows
    /// (Metal's `currentAllocatedSize`), for memory measurements.
    fn allocated_bytes(&self) -> u64 {
        0
    }
    /// A zeroed buffer of the host's own (`len` bytes) and where the CPU writes it. It lives
    /// while the caller or work submitted with it holds it: memory the GPU reads after the
    /// caller let go (a page table that was replaced) must not be the caller's to free.
    fn host_buffer(&self, len: usize) -> Result<(Self::Buffer, *mut u8), String>;
    /// Makes `buffers` resident for everything submitted to `queue` from now on, so that dispatches
    /// and draws need not name them one by one (Metal: an `MTLResidencySet` of the title's pools,
    /// made once, committed when a pool joins, added to each queue once). Idempotent; a buffer
    /// leaves the set when it is dropped. The default does nothing (the backend names buffers per
    /// dispatch).
    fn make_resident(&self, _queue: &Self::Queue, _buffers: &[&Self::Buffer]) -> Result<(), String> {
        Ok(())
    }
    /// The GPU address of a title address inside `buffer` (`gpu = title + delta`).
    fn gpu_address(&self, buffer: &Self::Buffer, title: *const u8) -> u64;
    /// DXIL container (`victory_shader::emit` output) to a compute pipeline.
    fn create_compute_pipeline(&self, dxil: &[u8]) -> Result<Self::Pipeline, String>;
    /// The same for DXIL without a root signature, laid out as `layout` says.
    fn create_compute_pipeline_with(
        &self,
        dxil: &[u8],
        layout: &RootLayout,
    ) -> Result<Self::Pipeline, String>;
    /// As the two above, waiting at most `wait` for the pipeline: `None` while it is still being
    /// made, and the same call later picks it up. The default makes it at once, however long.
    fn create_compute_pipeline_within(
        &self,
        dxil: &[u8],
        layout: Option<&RootLayout>,
        _wait: std::time::Duration,
    ) -> Result<Option<Self::Pipeline>, String> {
        match layout {
            Some(l) => self.create_compute_pipeline_with(dxil, l),
            None => self.create_compute_pipeline(dxil),
        }
        .map(Some)
    }
    /// Names a pipeline for diagnostics (`VICTORY_TRACE_GPU_FAILS`); the default does nothing.
    fn name_pipeline(&self, _p: &Self::Pipeline, _name: &str) {}
    /// Queue the work and return; never waits for the GPU.
    fn submit(&self, queue: &Self::Queue, work: &Submission<'_, Self>) -> Result<(), String>;
    /// `ID3D12Fence::Signal` from the CPU: queues waiting for `value` go on.
    fn signal(&self, fence: &Self::Fence, value: u64);
    /// `GetCompletedValue`.
    fn completed(&self, fence: &Self::Fence) -> u64;
    /// Block until the fence reaches `value`; false on timeout.
    fn wait(&self, fence: &Self::Fence, value: u64, timeout_ms: u64) -> bool;
}
