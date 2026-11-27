//! Where the mixer's frames go: an audio output behind a trait, with
//! CoreAudio (an AudioToolbox `AudioQueue`, called through a small FFI of
//! our own: no crate is added) and a silent one that keeps time when there is
//! no device.
//!
//! All [`CoreAudio`] streams of the same `(rate, channels)` (XAudio2, WASAPI
//! and Spatial Audio all open 48 kHz stereo float) share one hardware
//! `AudioQueue` and mix in userspace (`#336`). Re-activating a spatial or
//! WASAPI stream under world-load memory pressure attaches to the running
//! queue without calling `AudioQueueNewOutput` / `AudioQueueStart` again, so a
//! transient `'what'` (`2003329396`) from `coreaudiod` cannot strand the new
//! stream on [`Silent`] while the earlier queue keeps running.
//!
//! An [`Output`] asks for frames: the render function is called on the
//! device's thread with a buffer of interleaved 32-bit floats to fill, and
//! stops being called when the output is dropped.

use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

/// Fills a buffer of interleaved float frames or handles an output failure.
pub type Render = Box<dyn FnMut(io::Result<&mut [f32]>) + Send>;

/// An open output: it plays until it is dropped.
pub trait Output: Send {}

/// Opens an output of `channels` at `rate` that pulls frames from `render`.
pub type Opener = fn(rate: u32, channels: u32, render: Render) -> io::Result<Box<dyn Output>>;

/// CoreAudio, or the silent clock if it cannot open.
pub fn open_default(rate: u32, channels: u32, render: Render) -> io::Result<Box<dyn Output>> {
    match CoreAudio::open(rate, channels, render) {
        Ok(output) => Ok(Box::new(output)),
        Err((e, render)) => {
            tracing::warn!(error = %e, "no audio device: sound is mixed and dropped");
            Ok(Box::new(Silent::open(rate, channels, render)))
        }
    }
}

fn quantum_frames(rate: u32) -> u32 {
    (rate / 100).max(1) // 10 ms
}

// AudioToolbox's AudioQueue.

type OSStatus = i32;
type AudioQueueRef = *mut c_void;

#[repr(C)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

#[repr(C)]
struct AudioQueueBuffer {
    capacity: u32,
    data: *mut c_void,
    byte_size: u32,
    user_data: *mut c_void,
    packet_description_capacity: u32,
    packet_descriptions: *mut c_void,
    packet_description_count: u32,
}

type OutputCallback = extern "C" fn(*mut c_void, AudioQueueRef, *mut AudioQueueBuffer);

#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    fn AudioQueueNewOutput(
        format: *const AudioStreamBasicDescription,
        callback: OutputCallback,
        user: *mut c_void,
        run_loop: *mut c_void,
        run_loop_mode: *const c_void,
        flags: u32,
        queue: *mut AudioQueueRef,
    ) -> OSStatus;
    fn AudioQueueAllocateBuffer(
        queue: AudioQueueRef,
        bytes: u32,
        buffer: *mut *mut AudioQueueBuffer,
    ) -> OSStatus;
    fn AudioQueueEnqueueBuffer(
        queue: AudioQueueRef,
        buffer: *mut AudioQueueBuffer,
        packets: u32,
        descriptions: *const c_void,
    ) -> OSStatus;
    fn AudioQueueStart(queue: AudioQueueRef, start_time: *const c_void) -> OSStatus;
    fn AudioQueueStop(queue: AudioQueueRef, immediate: u8) -> OSStatus;
    fn AudioQueueDispose(queue: AudioQueueRef, immediate: u8) -> OSStatus;
}

/// `'lpcm'`, linear PCM; float, packed.
const FORMAT_LINEAR_PCM: u32 = 0x6C70_636D;
const FLAGS_FLOAT_PACKED: u32 = 1 | 8;
const BUFFERS: usize = 3;
/// How many times `AudioQueueStart` is tried, and how long between tries.
const RETRIES: usize = 3;
const START_RETRY_MS: u64 = 250;

/// AudioQueue entry points, injectable in tests so retry and failure paths run deterministically.
#[derive(Clone, Copy)]
struct QueueApi {
    new_output: unsafe extern "C" fn(
        *const AudioStreamBasicDescription,
        OutputCallback,
        *mut c_void,
        *mut c_void,
        *const c_void,
        u32,
        *mut AudioQueueRef,
    ) -> OSStatus,
    allocate_buffer:
        unsafe extern "C" fn(AudioQueueRef, u32, *mut *mut AudioQueueBuffer) -> OSStatus,
    enqueue_buffer:
        unsafe extern "C" fn(AudioQueueRef, *mut AudioQueueBuffer, u32, *const c_void) -> OSStatus,
    start: unsafe extern "C" fn(AudioQueueRef, *const c_void) -> OSStatus,
    stop: unsafe extern "C" fn(AudioQueueRef, u8) -> OSStatus,
    dispose: unsafe extern "C" fn(AudioQueueRef, u8) -> OSStatus,
    retry_delay: Duration,
}

const REAL_API: QueueApi = QueueApi {
    new_output: AudioQueueNewOutput,
    allocate_buffer: AudioQueueAllocateBuffer,
    enqueue_buffer: AudioQueueEnqueueBuffer,
    start: AudioQueueStart,
    stop: AudioQueueStop,
    dispose: AudioQueueDispose,
    retry_delay: Duration::from_millis(START_RETRY_MS),
};

struct SlotState {
    render: Option<Render>,
    closed: bool,
    in_call: Option<ThreadId>,
}

struct Slot {
    id: u64,
    state: Mutex<SlotState>,
    done: Condvar,
}

#[derive(Default)]
struct MixerSlots {
    next_id: u64,
    active: Vec<Arc<Slot>>,
}

struct MixerState {
    slots: Mutex<MixerSlots>,
    failed: AtomicBool,
}

/// Owned 1-to-1 by one `AudioQueueRef`: mutated only by its serial `refill` callback, then freed
/// after synchronous `AudioQueueDispose`.
struct QueueCtx {
    state: Arc<MixerState>,
    scratch: UnsafeCell<Vec<f32>>,
    snapshot: UnsafeCell<Vec<Arc<Slot>>>,
    bytes: u32,
    started: AtomicBool,
    closing: AtomicBool,
    enqueue:
        unsafe extern "C" fn(AudioQueueRef, *mut AudioQueueBuffer, u32, *const c_void) -> OSStatus,
}

struct MixerQueue {
    queue: AudioQueueRef,
    ctx: *mut QueueCtx,
    state: Arc<MixerState>,
    api: QueueApi,
}

// SAFETY: `AudioQueueRef` is thread-safe and `ctx` is touched only by the serial callback and
// by `Drop` after `AudioQueueDispose(..., 1)` has stopped all callbacks.
unsafe impl Send for MixerQueue {}
unsafe impl Sync for MixerQueue {}

impl Drop for MixerQueue {
    fn drop(&mut self) {
        // SAFETY: our queue and context; after synchronous dispose no callback runs.
        unsafe {
            (*self.ctx).closing.store(true, Ordering::Release);
            (self.api.stop)(self.queue, 1);
            (self.api.dispose)(self.queue, 1);
            drop(Box::from_raw(self.ctx));
        }
    }
}

struct Hub {
    api: QueueApi,
    queues: Mutex<BTreeMap<(u32, u32), Arc<MixerQueue>>>,
}

static DEFAULT_HUB: LazyLock<Hub> = LazyLock::new(|| Hub::new(REAL_API));

fn lock_unpoisoned<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

extern "C" fn refill(user: *mut c_void, queue: AudioQueueRef, buffer: *mut AudioQueueBuffer) {
    // SAFETY: `user` is our `QueueCtx`, alive until the queue is disposed;
    // `buffer` is one of ours, and AudioToolbox serialises callbacks per queue.
    unsafe {
        let ctx = &*user.cast::<QueueCtx>();
        if ctx.closing.load(Ordering::Acquire) {
            return;
        }
        let started = ctx.started.load(Ordering::Acquire);
        let buffer = &mut *buffer;
        let sample_count = ctx.bytes as usize / size_of::<f32>();
        let samples = std::slice::from_raw_parts_mut(buffer.data.cast::<f32>(), sample_count);
        samples.fill(0.0);

        let snap = &mut *ctx.snapshot.get();
        snap.clear();
        if started {
            let slots = lock_unpoisoned(&ctx.state.slots);
            snap.extend(slots.active.iter().cloned());
        }

        let tid = std::thread::current().id();
        let scratch = &mut *ctx.scratch.get();
        for slot in snap.iter() {
            let mut render = {
                let mut st = lock_unpoisoned(&slot.state);
                if st.closed {
                    continue;
                }
                let Some(r) = st.render.take() else {
                    continue;
                };
                st.in_call = Some(tid);
                r
            };
            scratch[..sample_count].fill(0.0);
            render(Ok(&mut scratch[..sample_count]));
            for (dst, &src) in samples.iter_mut().zip(&scratch[..sample_count]) {
                *dst += src;
            }
            let drop_after = {
                let mut st = lock_unpoisoned(&slot.state);
                st.in_call = None;
                let discard = if st.closed {
                    Some(render)
                } else {
                    st.render = Some(render);
                    None
                };
                slot.done.notify_all();
                discard
            };
            drop(drop_after);
        }

        buffer.byte_size = ctx.bytes;
        let status = (ctx.enqueue)(queue, buffer, 0, std::ptr::null());
        // shortcut: runtime output failure is detected on enqueue, monitor queue state if a device can stop without another refill.
        if status != 0 && started && !ctx.closing.swap(true, Ordering::AcqRel) {
            ctx.state.failed.store(true, Ordering::Release);
            for slot in snap.iter() {
                let mut render = {
                    let mut st = lock_unpoisoned(&slot.state);
                    if st.closed {
                        continue;
                    }
                    let Some(r) = st.render.take() else {
                        continue;
                    };
                    st.in_call = Some(tid);
                    r
                };
                render(Err(io::Error::other(format!(
                    "AudioQueueEnqueueBuffer failed with OSStatus {status}"
                ))));
                let drop_after = {
                    let mut st = lock_unpoisoned(&slot.state);
                    st.in_call = None;
                    let discard = if st.closed {
                        Some(render)
                    } else {
                        st.render = Some(render);
                        None
                    };
                    slot.done.notify_all();
                    discard
                };
                drop(drop_after);
            }
        }
        snap.clear();
    }
}

impl MixerQueue {
    fn open_once(
        api: QueueApi,
        format: &AudioStreamBasicDescription,
        bytes: u32,
        state: Arc<MixerState>,
    ) -> Result<MixerQueue, (&'static str, OSStatus)> {
        let sample_count = bytes as usize / size_of::<f32>();
        let ctx = Box::into_raw(Box::new(QueueCtx {
            state: state.clone(),
            scratch: UnsafeCell::new(vec![0.0; sample_count]),
            snapshot: UnsafeCell::new(Vec::with_capacity(8)),
            bytes,
            started: AtomicBool::new(false),
            closing: AtomicBool::new(false),
            enqueue: api.enqueue_buffer,
        }));
        let mut queue: AudioQueueRef = std::ptr::null_mut();
        // SAFETY: a valid format, our callback and its 1-to-1 `QueueCtx`.
        let status = unsafe {
            (api.new_output)(
                format,
                refill,
                ctx.cast(),
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                &mut queue,
            )
        };
        let cleanup = |queue: AudioQueueRef| {
            // SAFETY: `ctx` belongs to this attempt alone; dispose stops callbacks before free.
            unsafe {
                (*ctx).closing.store(true, Ordering::Release);
                if !queue.is_null() {
                    (api.dispose)(queue, 1);
                }
                drop(Box::from_raw(ctx));
            }
        };
        if status != 0 {
            cleanup(std::ptr::null_mut());
            return Err(("AudioQueueNewOutput", status));
        }
        for _ in 0..BUFFERS {
            let mut buffer: *mut AudioQueueBuffer = std::ptr::null_mut();
            // SAFETY: the queue just made.
            let status = unsafe { (api.allocate_buffer)(queue, bytes, &mut buffer) };
            if status != 0 {
                cleanup(queue);
                return Err(("AudioQueueAllocateBuffer", status));
            }
            // Prime with silence. Title callbacks begin once `started` is set.
            let status = unsafe {
                (*buffer).data.cast::<u8>().write_bytes(0, bytes as usize);
                (*buffer).byte_size = bytes;
                (api.enqueue_buffer)(queue, buffer, 0, std::ptr::null())
            };
            if status != 0 {
                cleanup(queue);
                return Err(("AudioQueueEnqueueBuffer", status));
            }
        }
        // SAFETY: the queue just primed.
        let status = unsafe { (api.start)(queue, std::ptr::null()) };
        if status != 0 {
            cleanup(queue);
            return Err(("AudioQueueStart", status));
        }
        // SAFETY: `ctx` stays alive until `MixerQueue::drop` disposes `queue`.
        unsafe { (*ctx).started.store(true, Ordering::Release) };
        Ok(MixerQueue {
            queue,
            ctx,
            state,
            api,
        })
    }

    fn open(api: QueueApi, rate: u32, channels: u32) -> io::Result<Arc<MixerQueue>> {
        if rate == 0 || channels == 0 {
            return Err(io::Error::other(format!(
                "invalid CoreAudio stream format: {rate} Hz, {channels} channels"
            )));
        }
        let bytes = quantum_frames(rate) * channels * 4;
        let format = AudioStreamBasicDescription {
            sample_rate: f64::from(rate),
            format_id: FORMAT_LINEAR_PCM,
            format_flags: FLAGS_FLOAT_PACKED,
            bytes_per_packet: channels * 4,
            frames_per_packet: 1,
            bytes_per_frame: channels * 4,
            channels_per_frame: channels,
            bits_per_channel: 32,
            reserved: 0,
        };
        let state = Arc::new(MixerState {
            slots: Mutex::new(MixerSlots::default()),
            failed: AtomicBool::new(false),
        });
        // `AudioQueueStart` can fail transiently with OSStatus 'what' (0x77686174, #140 / #336)
        // when the output HAL is reconfiguring. Dispose the failed queue handle and open a fresh
        // one on each retry instead of calling `AudioQueueStart` again on the broken handle.
        let mut last = ("AudioQueueStart", 0);
        for attempt in 0..RETRIES {
            match Self::open_once(api, &format, bytes, state.clone()) {
                Ok(q) => return Ok(Arc::new(q)),
                Err((what, status)) => {
                    last = (what, status);
                    if what != "AudioQueueStart" || attempt + 1 == RETRIES {
                        break;
                    }
                    tracing::warn!(
                        attempt,
                        status,
                        "AudioQueueStart failed; disposing queue and retrying"
                    );
                    if !api.retry_delay.is_zero() {
                        std::thread::sleep(api.retry_delay);
                    }
                }
            }
        }
        Err(io::Error::other(format!(
            "{} failed with OSStatus {}",
            last.0, last.1
        )))
    }

    fn attach(self: &Arc<Self>, render: Render) -> CoreAudio {
        let mut slots = lock_unpoisoned(&self.state.slots);
        let id = slots.next_id;
        slots.next_id += 1;
        let slot = Arc::new(Slot {
            id,
            state: Mutex::new(SlotState {
                render: Some(render),
                closed: false,
                in_call: None,
            }),
            done: Condvar::new(),
        });
        slots.active.push(slot.clone());
        drop(slots);
        CoreAudio {
            queue: self.clone(),
            slot,
        }
    }
}

impl Hub {
    const fn new(api: QueueApi) -> Self {
        Self {
            api,
            queues: Mutex::new(BTreeMap::new()),
        }
    }

    fn open(&self, rate: u32, channels: u32, render: Render) -> Result<CoreAudio, (io::Error, Render)> {
        let key = (rate, channels);
        if let Some(existing) = lock_unpoisoned(&self.queues)
            .get(&key)
            .filter(|q| !q.state.failed.load(Ordering::Acquire))
            .cloned()
        {
            return Ok(existing.attach(render));
        }
        let created = match MixerQueue::open(self.api, rate, channels) {
            Ok(q) => q,
            Err(e) => return Err((e, render)),
        };
        let mut queues = lock_unpoisoned(&self.queues);
        let queue = match queues
            .get(&key)
            .filter(|q| !q.state.failed.load(Ordering::Acquire))
            .cloned()
        {
            Some(winner) => winner,
            None => {
                queues.insert(key, created.clone());
                created
            }
        };
        drop(queues);
        Ok(queue.attach(render))
    }
}

pub struct CoreAudio {
    queue: Arc<MixerQueue>,
    slot: Arc<Slot>,
}

impl Output for CoreAudio {}

impl CoreAudio {
    fn open(rate: u32, channels: u32, render: Render) -> Result<CoreAudio, (io::Error, Render)> {
        DEFAULT_HUB.open(rate, channels, render)
    }
}

impl Drop for CoreAudio {
    fn drop(&mut self) {
        lock_unpoisoned(&self.queue.state.slots)
            .active
            .retain(|s| s.id != self.slot.id);
        let me = std::thread::current().id();
        let dropped = {
            let mut st = lock_unpoisoned(&self.slot.state);
            st.closed = true;
            let r = st.render.take();
            while st.in_call.is_some_and(|tid| tid != me) {
                st = self.slot.done.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            r
        };
        drop(dropped);
    }
}

/// No device: frames are pulled at the pace of the clock and dropped, so the
/// title's voices still advance and its callbacks still come.
pub struct Silent {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Output for Silent {}

impl Silent {
    pub fn open(rate: u32, channels: u32, mut render: Render) -> Silent {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let chunk = quantum_frames(rate) as usize;
        let period = Duration::from_secs_f64(chunk as f64 / f64::from(rate));
        let thread = std::thread::spawn(move || {
            let mut buffer = vec![0.0f32; chunk * channels as usize];
            let mut next = Instant::now();
            while !flag.load(Ordering::Acquire) {
                render(Ok(&mut buffer));
                next += period;
                std::thread::sleep(next.saturating_duration_since(Instant::now()));
            }
        });
        Silent {
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for Silent {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn the_device_opens_pulls_frames_and_closes() {
        let pulled = Arc::new(AtomicUsize::new(0));
        let count = pulled.clone();
        let render: Render = Box::new(move |out| {
            let out = out.unwrap();
            // A 440 Hz tone, as a title's mixer would give it.
            for (i, frame) in out.chunks_exact_mut(2).enumerate() {
                let x = (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.01;
                frame.fill(x);
            }
            count.fetch_add(out.len() / 2, Ordering::Relaxed);
        });
        match CoreAudio::open(48_000, 2, render) {
            Ok(device) => {
                std::thread::sleep(Duration::from_millis(200));
                assert!(pulled.load(Ordering::Relaxed) >= 480);
                drop(device);
                let after = pulled.load(Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(100));
                // Closed: nothing pulls any more.
                assert_eq!(pulled.load(Ordering::Relaxed), after);
            }
            // A machine with no output device (CI) says so; the silent
            // clock takes over, and that is tested next.
            Err((e, _)) => eprintln!("no CoreAudio output here: {e}"),
        }
    }

    #[test]
    fn the_silent_output_keeps_time() {
        let pulled = Arc::new(AtomicUsize::new(0));
        let count = pulled.clone();
        let silent = Silent::open(
            48_000,
            2,
            Box::new(move |out| {
                count.fetch_add(out.unwrap().len() / 2, Ordering::Relaxed);
            }),
        );
        std::thread::sleep(Duration::from_millis(300));
        drop(silent);
        let frames = pulled.load(Ordering::Relaxed);
        // About 14,400 frames in 300 ms.
        assert!((7_000..30_000).contains(&frames), "pulled {frames} frames");
    }

    /// Mock AudioQueue state so multi-stream mixing, 32 spatial stream re-activations,
    /// `'what'` (`0x77686174`) start retries, runtime enqueue errors, and re-entrant drops
    /// are tested deterministically even on headless runners with no audio hardware.
    struct MockDriver {
        next_handle: usize,
        new_calls: usize,
        start_calls: usize,
        stop_calls: usize,
        dispose_calls: Vec<usize>,
        fail_starts_left: usize,
        enqueue_fail_status: OSStatus,
        active_queue: usize,
        active_cb: Option<(OutputCallback, *mut c_void)>,
        buffers: Vec<(*mut AudioQueueBuffer, Vec<u8>)>,
    }

    // SAFETY: test-only mock guarded by `MOCK`'s `Mutex`.
    unsafe impl Send for MockDriver {}

    static MOCK: Mutex<MockDriver> = Mutex::new(MockDriver {
        next_handle: 0x1000,
        new_calls: 0,
        start_calls: 0,
        stop_calls: 0,
        dispose_calls: Vec::new(),
        fail_starts_left: 0,
        enqueue_fail_status: 0,
        active_queue: 0,
        active_cb: None,
        buffers: Vec::new(),
    });
    static MOCK_TEST_LOCK: Mutex<()> = Mutex::new(());

    impl MockDriver {
        fn reset(&mut self) {
            for (hdr, _) in self.buffers.drain(..) {
                // SAFETY: allocated by `mock_alloc_buf`.
                unsafe { drop(Box::from_raw(hdr)) };
            }
            self.next_handle = 0x1000;
            self.new_calls = 0;
            self.start_calls = 0;
            self.stop_calls = 0;
            self.dispose_calls.clear();
            self.fail_starts_left = 0;
            self.enqueue_fail_status = 0;
            self.active_queue = 0;
            self.active_cb = None;
        }
    }

    unsafe extern "C" fn mock_new(
        _format: *const AudioStreamBasicDescription,
        callback: OutputCallback,
        user: *mut c_void,
        _run_loop: *mut c_void,
        _mode: *const c_void,
        _flags: u32,
        out: *mut AudioQueueRef,
    ) -> OSStatus {
        let mut m = lock_unpoisoned(&MOCK);
        m.next_handle += 1;
        let q = m.next_handle;
        m.new_calls += 1;
        m.active_queue = q;
        m.active_cb = Some((callback, user));
        // SAFETY: caller's out pointer.
        unsafe { out.write(q as AudioQueueRef) };
        0
    }

    unsafe extern "C" fn mock_alloc_buf(
        _queue: AudioQueueRef,
        bytes: u32,
        out: *mut *mut AudioQueueBuffer,
    ) -> OSStatus {
        let mut m = lock_unpoisoned(&MOCK);
        let mut backing = vec![0u8; bytes as usize];
        let hdr = Box::into_raw(Box::new(AudioQueueBuffer {
            capacity: bytes,
            data: backing.as_mut_ptr().cast(),
            byte_size: 0,
            user_data: std::ptr::null_mut(),
            packet_description_capacity: 0,
            packet_descriptions: std::ptr::null_mut(),
            packet_description_count: 0,
        }));
        m.buffers.push((hdr, backing));
        // SAFETY: caller's out pointer.
        unsafe { out.write(hdr) };
        0
    }

    unsafe extern "C" fn mock_enqueue(
        _queue: AudioQueueRef,
        _buffer: *mut AudioQueueBuffer,
        _packets: u32,
        _desc: *const c_void,
    ) -> OSStatus {
        lock_unpoisoned(&MOCK).enqueue_fail_status
    }

    unsafe extern "C" fn mock_start(_queue: AudioQueueRef, _time: *const c_void) -> OSStatus {
        let mut m = lock_unpoisoned(&MOCK);
        m.start_calls += 1;
        if m.fail_starts_left > 0 {
            m.fail_starts_left -= 1;
            return 0x7768_6174; // 'what' (2003329396)
        }
        0
    }

    unsafe extern "C" fn mock_stop(_queue: AudioQueueRef, _immediate: u8) -> OSStatus {
        lock_unpoisoned(&MOCK).stop_calls += 1;
        0
    }

    unsafe extern "C" fn mock_dispose(queue: AudioQueueRef, _immediate: u8) -> OSStatus {
        let mut m = lock_unpoisoned(&MOCK);
        let q = queue as usize;
        m.dispose_calls.push(q);
        if m.active_queue == q {
            m.active_queue = 0;
            m.active_cb = None;
        }
        0
    }

    const MOCK_API: QueueApi = QueueApi {
        new_output: mock_new,
        allocate_buffer: mock_alloc_buf,
        enqueue_buffer: mock_enqueue,
        start: mock_start,
        stop: mock_stop,
        dispose: mock_dispose,
        retry_delay: Duration::ZERO,
    };

    fn pump_once() -> Vec<f32> {
        let (cb, user, q, buf, len) = {
            let m = lock_unpoisoned(&MOCK);
            let (cb, user) = m.active_cb.expect("active mock queue");
            let &(buf, ref bytes) = m.buffers.last().expect("allocated buffer");
            (cb, user, m.active_queue as AudioQueueRef, buf, bytes.len() / 4)
        };
        cb(user, q, buf);
        // SAFETY: `buf` points into `m.buffers`'s `Vec<u8>` of `len * 4` bytes.
        unsafe { std::slice::from_raw_parts((*buf).data.cast::<f32>(), len).to_vec() }
    }

    #[test]
    fn streams_share_one_queue_mix_and_survive_repeated_reactivations() {
        let _guard = lock_unpoisoned(&MOCK_TEST_LOCK);
        lock_unpoisoned(&MOCK).reset();
        let hub = Hub::new(MOCK_API);

        let a = hub
            .open(48_000, 2, Box::new(|out| out.unwrap().fill(0.25)))
            .ok()
            .unwrap();
        let b = hub
            .open(48_000, 2, Box::new(|out| out.unwrap().fill(0.5)))
            .ok()
            .unwrap();
        {
            let m = lock_unpoisoned(&MOCK);
            assert_eq!(m.new_calls, 1, "both streams share one AudioQueue");
            assert_eq!(m.start_calls, 1);
        }
        let mixed = pump_once();
        assert!(mixed.iter().all(|&s| (s - 0.75).abs() < 1e-6));

        drop(a);
        let b_only = pump_once();
        assert!(b_only.iter().all(|&s| (s - 0.5).abs() < 1e-6));

        drop(b);
        let idle = pump_once();
        assert!(idle.iter().all(|&s| s == 0.0));
        assert!(lock_unpoisoned(&MOCK).dispose_calls.is_empty());

        // Simulate 32 mid-run spatial audio stream tear-downs and re-activations (#8, #258, #336):
        // zero additional `AudioQueueNewOutput` / `AudioQueueStart` calls occur.
        for step in 1..=32 {
            let val = step as f32 * 0.01;
            let s = hub
                .open(48_000, 2, Box::new(move |out| out.unwrap().fill(val)))
                .ok()
                .unwrap();
            let got = pump_once();
            assert!((got[0] - val).abs() < 1e-6, "step {step}: got {}", got[0]);
            drop(s);
        }
        {
            let m = lock_unpoisoned(&MOCK);
            assert_eq!(m.new_calls, 1);
            assert_eq!(m.start_calls, 1);
        }
        drop(hub);
        assert_eq!(lock_unpoisoned(&MOCK).dispose_calls, vec![0x1001]);
    }

    #[test]
    fn start_retry_disposes_failed_queue_and_recovers_after_runtime_error() {
        let _guard = lock_unpoisoned(&MOCK_TEST_LOCK);
        {
            let mut m = lock_unpoisoned(&MOCK);
            m.reset();
            // Fail attempt 0 with 'what' (2003329396); attempt 1 must dispose 0x1001 and start 0x1002.
            m.fail_starts_left = 1;
        }
        let hub = Hub::new(MOCK_API);
        let errors = Arc::new(AtomicUsize::new(0));
        let err_count = errors.clone();
        let s1 = hub
            .open(
                48_000,
                2,
                Box::new(move |res| match res {
                    Ok(out) => out.fill(0.125),
                    Err(_) => {
                        err_count.fetch_add(1, Ordering::Relaxed);
                    }
                }),
            )
            .ok()
            .unwrap();
        {
            let m = lock_unpoisoned(&MOCK);
            assert_eq!(m.new_calls, 2, "failed start recreated a fresh AudioQueue");
            assert_eq!(m.start_calls, 2);
            assert_eq!(m.dispose_calls, vec![0x1001], "broken handle 0x1001 was disposed");
            assert_eq!(m.active_queue, 0x1002);
        }
        assert!(pump_once().iter().all(|&s| (s - 0.125).abs() < 1e-6));

        // Inject a runtime `AudioQueueEnqueueBuffer` error: active streams get `Err(...)` and
        // the next `open` replaces the failed queue with `0x1003`.
        lock_unpoisoned(&MOCK).enqueue_fail_status = -50;
        let _ = pump_once();
        assert_eq!(errors.load(Ordering::Relaxed), 1);
        lock_unpoisoned(&MOCK).enqueue_fail_status = 0;

        let s2 = hub
            .open(48_000, 2, Box::new(|out| out.unwrap().fill(0.375)))
            .ok()
            .unwrap();
        drop(s1);
        assert_eq!(lock_unpoisoned(&MOCK).active_queue, 0x1003);
        assert!(pump_once().iter().all(|&s| (s - 0.375).abs() < 1e-6));
        drop(s2);
        drop(hub);
    }

    #[test]
    fn reentrant_drop_inside_render_callback_does_not_deadlock() {
        let _guard = lock_unpoisoned(&MOCK_TEST_LOCK);
        lock_unpoisoned(&MOCK).reset();
        let hub = Hub::new(MOCK_API);
        let self_handle: Arc<Mutex<Option<CoreAudio>>> = Arc::new(Mutex::new(None));
        let inside = self_handle.clone();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_in = calls.clone();
        let out = hub
            .open(
                48_000,
                2,
                Box::new(move |buf| {
                    calls_in.fetch_add(1, Ordering::Relaxed);
                    buf.unwrap().fill(0.5);
                    // Drop our own `CoreAudio` handle from inside the callback thread.
                    let _ = lock_unpoisoned(&inside).take();
                }),
            )
            .ok()
            .unwrap();
        *lock_unpoisoned(&self_handle) = Some(out);
        let first = pump_once();
        assert!(first.iter().all(|&s| (s - 0.5).abs() < 1e-6));
        let second = pump_once();
        assert!(second.iter().all(|&s| s == 0.0), "dropped stream is not called again");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        drop(hub);
    }
}
