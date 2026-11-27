//! The console's audio DSP (`chad.dll`): streams that convolve blocks of
//! float samples with an impulse response (reverb), or run other block
//! transforms, on a coprocessor.
//!
//! A title connects once with a memory block for the DSP, activates
//! streams, and submits one command per block (gains, an input and an
//! output buffer). Activation hands back a status block whose second word is
//! the ticket of the last completed command and whose fourth is its result;
//! the title checks it one block later and treats a late command as an
//! overrun. Commands complete when submitted here.
//!
//! The transforms themselves are not computed from the title's own impulse
//! response: the hardware takes it already in its own partitioned
//! frequency-domain layout, which is not known, so `StreamDesc::ir` is not
//! read. With `VICTORY_GUESS_DSP=1`, a convolution stream instead gets a
//! short synthetic decaying-noise tail (see `synthetic_ir`), carried across
//! blocks so it does not reset at every block boundary; this is a CPU
//! fallback approximation of reverb coloration, not the title's actual
//! effect, and without the variable set the output is silence as before.
//! Other block transforms stay silent either way: their shape is unknown.
//!
//! With `VICTORY_DUMP_DSP=<dir>` set at activation, each stream's own
//! impulse response is captured to `<dir>/stream<n>_ir.f32` and every
//! command's gains, input block and output block are appended to
//! `<dir>/stream<n>_blocks.f32` / `_out.f32`, so the guessed convolution can
//! be checked against what the title actually sends and hears.
// shortcut: convolution streams get one fixed synthetic decay instead of the
// title's impulse response, behind VICTORY_GUESS_DSP=1 since it is unverified
// against a real run; other transform kinds stay silent regardless. Upgrade
// once the hardware's partitioned IR layout is documented.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_POINTER: HRESULT = 0x8000_4003_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;

/// Cap on how many floats a `VICTORY_DUMP_DSP` capture of `StreamDesc::ir`
/// reads, so a bogus `ir_floats` cannot make the diagnostic allocate
/// gigabytes; a larger count is logged as truncated.
const DUMP_IR_MAX: usize = 0x100_0000;

const CONVOLUTION: u32 = 1;
/// Option bits that need a stereo stream, and one convolution refuses.
const STEREO_OPTIONS: u32 = 2 | 8;
const NOT_FOR_CONVOLUTION: u32 = 4;

pub struct Dsp {
    max_streams: u32,
    active: Mutex<u32>,
}

/// What activation describes.
#[repr(C, packed)]
pub struct StreamDesc {
    kind: u32,
    options: u32,
    frames: u32,
    channels: u32,
    ir_floats: u32,
    ir: *const f32,
}

/// The status block a stream's owner polls.
#[repr(C)]
pub struct Status {
    submitted: u32,
    completed: u32,
    reserved: u32,
    result: HRESULT,
}

pub struct Stream {
    dsp: *mut Dsp,
    frames: u32,
    channels: u32,
    kind: u32,
    /// Uniformly partitioned overlap-save FFT convolver for `CONVOLUTION`
    /// streams (preallocated at activation, zero allocations per block).
    convolver: Option<PartitionedConvolver>,
    /// Capture directory and stream number when `VICTORY_DUMP_DSP` is set.
    dump: Option<(PathBuf, u32)>,
    status: Box<Status>,
}

/// Streams activated so far under `VICTORY_DUMP_DSP`, numbering capture
/// files.
static DUMP_ID: AtomicU32 = AtomicU32::new(0);

fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("VICTORY_DUMP_DSP").map(PathBuf::from)
}

fn append_f32(path: &std::path::Path, floats: &[f32]) {
    use std::io::Write as _;
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    else {
        tracing::warn!(?path, "audio DSP dump write failed");
        return;
    };
    let mut bytes = Vec::with_capacity(floats.len() * 4);
    bytes.extend(floats.iter().flat_map(|s| s.to_le_bytes()));
    let _ = file.write_all(&bytes);
}

/// Writes a convolution stream's own impulse response — the title's bytes,
/// not the synthetic tail — to `<dir>/stream<n>_ir.f32`, to help settle
/// the hardware's partitioned IR layout (see `trace_ir`). Off by default;
/// reads are bounded by `ir_floats` and `DUMP_IR_MAX`.
fn dump_ir(dir: &std::path::Path, n: u32, desc: &StreamDesc) {
    let (kind, ir_floats) = (desc.kind, desc.ir_floats as usize);
    if kind != CONVOLUTION || desc.ir.is_null() {
        tracing::info!(n, kind, "audio DSP stream has no IR to dump");
        return;
    }
    let len = ir_floats.min(DUMP_IR_MAX);
    if len < ir_floats {
        tracing::warn!(ir_floats, dumped = len, "IR dump truncated");
    }
    // SAFETY: `len` is bounded by the title's own `ir_floats` count.
    let ir = unsafe { std::slice::from_raw_parts(desc.ir, len) };
    append_f32(&dir.join(format!("stream{n}_ir.f32")), ir);
    tracing::info!(
        n,
        ir_floats,
        "dumped audio DSP stream's own impulse response"
    );
}

/// Appends one command's gains and input block to
/// `<dir>/stream<n>_blocks.f32`, so a capture shows what the title fed the
/// DSP in block order.
fn dump_block(dir: &std::path::Path, n: u32, gains: [f32; 2], input: &[f32]) {
    let mut block = Vec::with_capacity(2 + input.len());
    block.extend_from_slice(&gains);
    block.extend_from_slice(input);
    append_f32(&dir.join(format!("stream{n}_blocks.f32")), &block);
}

/// Appends one command's output block to `<dir>/stream<n>_out.f32`, after
/// the stream has filled it, so a capture shows either the silence or the
/// `VICTORY_GUESS_DSP` result the title actually heard.
fn dump_output(dir: &std::path::Path, n: u32, output: &[f32]) {
    append_f32(&dir.join(format!("stream{n}_out.f32")), output);
}

/// Taps of the synthetic reverb tail: a fixed decaying-noise burst,
/// deterministic so the same stream always gets the same tail. About 100 ms
/// at 48 kHz, convolved in the frequency domain via uniformly partitioned
/// overlap-save FFT so the audio thread pays O(N log N) instead of O(N * TAPS).
const TAPS: usize = 4800;

fn synthetic_ir(len: usize) -> Vec<f32> {
    let mut seed = 0x9E37_79B9_u32;
    (0..len)
        .map(|k| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let noise = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let decay = (-(k as f32) / (len as f32 / 5.0)).exp();
            noise * decay
        })
        .collect()
}

/// Complex sample `(re, im)` for the radix-2 FFT.
#[derive(Clone, Copy, Default)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    #[inline]
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// Precomputed bit-reversal and twiddle tables for an in-place radix-2 FFT of
/// size `n` (a power of two).
struct FftPlan {
    n: usize,
    bit_rev: Vec<usize>,
    /// Per-stage twiddles for forward FFT (`exp(-i * 2*pi*k / len)`), packed
    /// sequentially for `half = 1, 2, 4, ..., n/2`.
    twiddles: Vec<Complex>,
}

impl FftPlan {
    fn new(n: usize) -> Self {
        debug_assert!(n.is_power_of_two() && n >= 2);
        let bits = n.trailing_zeros();
        let bit_rev = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        let mut twiddles = Vec::with_capacity(n.saturating_sub(1));
        let mut half = 1usize;
        while half < n {
            let step = -std::f64::consts::PI / (half as f64);
            for k in 0..half {
                let angle = step * (k as f64);
                twiddles.push(Complex::new(angle.cos() as f32, angle.sin() as f32));
            }
            half <<= 1;
        }
        Self {
            n,
            bit_rev,
            twiddles,
        }
    }

    fn transform(&self, buf: &mut [Complex], inverse: bool) {
        let n = self.n;
        debug_assert_eq!(buf.len(), n);
        for (i, &j) in self.bit_rev.iter().enumerate() {
            if i < j {
                buf.swap(i, j);
            }
        }
        let mut half = 1usize;
        let mut tw_offset = 0usize;
        while half < n {
            let step = half << 1;
            let stage_tw = &self.twiddles[tw_offset..tw_offset + half];
            for base in (0..n).step_by(step) {
                for (k, &tw_raw) in stage_tw.iter().enumerate() {
                    let tw = if inverse {
                        Complex::new(tw_raw.re, -tw_raw.im)
                    } else {
                        tw_raw
                    };
                    let u = buf[base + k];
                    let v = buf[base + k + half].mul(tw);
                    buf[base + k] = u.add(v);
                    buf[base + k + half] = u.sub(v);
                }
            }
            tw_offset += half;
            half = step;
        }
        if inverse {
            let scale = 1.0 / (n as f32);
            for c in buf.iter_mut() {
                c.re *= scale;
                c.im *= scale;
            }
        }
    }
}

/// Uniformly partitioned overlap-save FFT convolver: exact causal FIR
/// convolution per channel with zero heap allocations on the audio path.
struct PartitionedConvolver {
    block: usize,
    channels: usize,
    partitions: usize,
    plan: FftPlan,
    /// `partitions * fft_size` precomputed IR partition spectra.
    ir_spectra: Vec<Complex>,
    /// Per-channel frequency-domain delay line: `channels * partitions * fft_size`.
    fdl: Vec<Complex>,
    /// Current write slot in the circular delay line (`0..partitions`).
    cursor: usize,
    /// Previous `block` input samples per channel (`channels * block`).
    prev_input: Vec<f32>,
    /// Reusable FFT/accumulation scratch buffers (`fft_size` each).
    fft_buf: Vec<Complex>,
    acc_buf: Vec<Complex>,
}

impl PartitionedConvolver {
    fn new(block: usize, channels: usize, ir: &[f32]) -> Self {
        debug_assert!(block.is_power_of_two() && block >= 1);
        debug_assert!((1..=2).contains(&channels));
        let fft_size = block * 2;
        let partitions = ir.len().div_ceil(block).max(1);
        let plan = FftPlan::new(fft_size);
        let mut ir_spectra = vec![Complex::default(); partitions * fft_size];
        for p in 0..partitions {
            let dst = &mut ir_spectra[p * fft_size..(p + 1) * fft_size];
            let start = p * block;
            let end = (start + block).min(ir.len());
            for (i, &h) in ir[start..end].iter().enumerate() {
                dst[i] = Complex::new(h, 0.0);
            }
            plan.transform(dst, false);
        }
        Self {
            block,
            channels,
            partitions,
            plan,
            ir_spectra,
            fdl: vec![Complex::default(); channels * partitions * fft_size],
            cursor: 0,
            prev_input: vec![0.0; channels * block],
            fft_buf: vec![Complex::default(); fft_size],
            acc_buf: vec![Complex::default(); fft_size],
        }
    }

    /// Processes one interleaved block (`block * channels` floats) into `out`.
    fn process(&mut self, input: &[f32], gains: [f32; 2], out: &mut [f32]) {
        self.process_channel_io(gains, |c, i, ch| input[i * ch + c], |c, i, ch, v| {
            out[i * ch + c] = v;
        });
    }

    /// Processes one interleaved block in-place (`command.input == command.output`)
    /// with zero allocations: each channel's input is loaded into `fft_buf`
    /// before that channel's output is written back.
    fn process_in_place(&mut self, buf: &mut [f32], gains: [f32; 2]) {
        let block = self.block;
        let channels = self.channels;
        debug_assert_eq!(buf.len(), block * channels);
            for (c, &gain) in gains.iter().enumerate().take(channels) {
                self.load_and_convolve_channel(c, |i| buf[i * channels + c]);
                for i in 0..block {
                    buf[i * channels + c] = self.acc_buf[block + i].re * gain;
                }
            }
        self.cursor = (self.cursor + 1) % self.partitions;
    }

    fn process_channel_io(
        &mut self,
        gains: [f32; 2],
        read_in: impl Fn(usize, usize, usize) -> f32,
        mut write_out: impl FnMut(usize, usize, usize, f32),
    ) {
        let block = self.block;
        let channels = self.channels;
        for (c, &gain) in gains.iter().enumerate().take(channels) {
            self.load_and_convolve_channel(c, |i| read_in(c, i, channels));
            for i in 0..block {
                write_out(c, i, channels, self.acc_buf[block + i].re * gain);
            }
        }
        self.cursor = (self.cursor + 1) % self.partitions;
    }

    fn load_and_convolve_channel(&mut self, c: usize, read_sample: impl Fn(usize) -> f32) {
        let block = self.block;
        let fft_size = block * 2;
        let partitions = self.partitions;

        let prev = &mut self.prev_input[c * block..(c + 1) * block];
        for (i, &s) in prev.iter().enumerate() {
            self.fft_buf[i] = Complex::new(s, 0.0);
        }
        for (i, p) in prev.iter_mut().enumerate() {
            let s = read_sample(i);
            self.fft_buf[block + i] = Complex::new(s, 0.0);
            *p = s;
        }
        self.plan.transform(&mut self.fft_buf, false);

        let fdl_base = c * partitions * fft_size;
        let slot_start = fdl_base + self.cursor * fft_size;
        self.fdl[slot_start..slot_start + fft_size].copy_from_slice(&self.fft_buf);

        self.acc_buf.fill(Complex::default());
        for p in 0..partitions {
            let slot = (self.cursor + partitions - p) % partitions;
            let x = &self.fdl[fdl_base + slot * fft_size..fdl_base + (slot + 1) * fft_size];
            let h = &self.ir_spectra[p * fft_size..(p + 1) * fft_size];
            for k in 0..fft_size {
                let prod = x[k].mul(h[k]);
                self.acc_buf[k].re += prod.re;
                self.acc_buf[k].im += prod.im;
            }
        }

        self.plan.transform(&mut self.acc_buf, true);
    }
}

/// Direct scalar reference convolution used by unit tests to verify the
/// partitioned FFT engine matches causal per-channel FIR convolution.
#[cfg(test)]
fn convolve_merged(
    merged: &[f32],
    hist_frames: usize,
    channels: usize,
    ir: &[f32],
    gains: [f32; 2],
    out: &mut [f32],
) {
    let frames = out.len() / channels;
    for c in 0..channels {
        let gain = gains[c];
        for n in 0..frames {
            let base = hist_frames + n;
            let mut acc = 0.0f32;
            for (k, &h) in ir.iter().enumerate() {
                if k > base {
                    break;
                }
                acc += merged[(base - k) * channels + c] * h;
            }
            out[n * channels + c] = acc * gain;
        }
    }
}

/// One block of work.
#[repr(C)]
pub struct Command {
    gains: [f32; 2],
    input: *const f32,
    output: *mut f32,
    reserved: u64,
}

pub extern "system" fn XDspConnectWithMaximumStreamLimit(
    base: *mut c_void,
    base_length: u32,
    ir_seconds: u32,
    max_streams: u32,
    max_channels: u32,
    out: *mut *mut Dsp,
) -> HRESULT {
    if base.is_null()
        || !(0x1_0000..=0x1F40_0000).contains(&base_length)
        || ir_seconds > 0x80
        || max_streams > 0x100
        || max_channels > 2
    {
        return E_INVALIDARG;
    }
    if out.is_null() {
        return E_POINTER;
    }
    tracing::info!(
        base_length,
        ir_seconds,
        max_streams,
        max_channels,
        "audio DSP connected"
    );
    let dsp = Box::new(Dsp {
        max_streams,
        active: Mutex::new(0),
    });
    // SAFETY: checked non-null.
    unsafe { out.write(Box::into_raw(dsp)) };
    S_OK
}

pub extern "system" fn XDspDisconnect(dsp: *mut Dsp) -> HRESULT {
    if dsp.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspConnectWithMaximumStreamLimit`; disconnected
    // once, after its streams.
    drop(unsafe { Box::from_raw(dsp) });
    S_OK
}

/// Whether the synthetic convolution tail (module docs) is turned on; off by
/// default since it is unverified against a real run, leaving the title's
/// reverb silent as before. Checked fresh each call, not cached, so tests
/// can toggle it.
fn guess_dsp() -> bool {
    std::env::var_os("VICTORY_GUESS_DSP").is_some()
}

/// Logs the leading floats of a convolution stream's own impulse response at
/// activation, to help settle what format `StreamDesc::ir` is actually in
/// (see native-host.md). Off by default; reads at most 8 floats, bounded by
/// `ir_floats`.
fn trace_ir(desc: &StreamDesc) {
    if desc.kind != CONVOLUTION
        || desc.ir.is_null()
        || std::env::var_os("VICTORY_TRACE_DSP").is_none()
    {
        return;
    }
    let ir_floats = desc.ir_floats;
    let len = (ir_floats as usize).min(8);
    // SAFETY: `len` is bounded by the title's own `ir_floats` count.
    let head = unsafe { std::slice::from_raw_parts(desc.ir, len) };
    tracing::info!(ir_floats, ?head, "audio DSP stream's own impulse response");
}

fn valid(desc: &StreamDesc) -> bool {
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    let shape = if kind == CONVOLUTION {
        (1..=2).contains(&channels)
            && matches!(frames, 0x200 | 0x400)
            && options & NOT_FOR_CONVOLUTION == 0
    } else {
        channels == 1 && matches!(frames, 0x200 | 0x400 | 0x800)
    };
    shape && (options & STEREO_OPTIONS == 0 || channels == 2)
}

pub extern "system" fn XDspActivate(
    dsp: *mut Dsp,
    desc: *const StreamDesc,
    status: *mut *const Status,
    stream: *mut *mut Stream,
) -> HRESULT {
    if status.is_null() || stream.is_null() {
        return E_POINTER;
    }
    if dsp.is_null() || desc.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes its description.
    let desc = unsafe { &*desc };
    if !valid(desc) {
        return E_INVALIDARG;
    }
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    tracing::debug!(
        kind,
        options,
        frames,
        channels,
        "audio DSP stream activated"
    );
    trace_ir(desc);
    let dump = dump_dir().map(|dir| {
        if std::fs::create_dir_all(&dir).is_ok() {
            let n = DUMP_ID.fetch_add(1, Ordering::Relaxed);
            dump_ir(&dir, n, desc);
            Some((dir, n))
        } else {
            tracing::warn!(?dir, "audio DSP dump directory unusable");
            None
        }
    });
    let dump = dump.flatten();
    {
        // SAFETY: a connection from `XDspConnectWithMaximumStreamLimit`.
        let dsp = unsafe { &*dsp };
        let mut active = dsp.active.lock().unwrap_or_else(|e| e.into_inner());
        if *active >= dsp.max_streams {
            return E_OUTOFMEMORY;
        }
        *active += 1;
    }
    let convolver = (kind == CONVOLUTION).then(|| {
        let ir = synthetic_ir(TAPS);
        PartitionedConvolver::new(frames as usize, channels as usize, &ir)
    });
    let new = Box::new(Stream {
        dsp,
        frames: desc.frames,
        channels: desc.channels,
        kind,
        convolver,
        dump,
        status: Box::new(Status {
            submitted: 0,
            completed: 0,
            reserved: 0,
            result: S_OK,
        }),
    });
    // SAFETY: checked non-null.
    unsafe {
        status.write(&*new.status);
        stream.write(Box::into_raw(new));
    }
    S_OK
}

pub extern "system" fn XDspDeactivate(stream: *mut Stream) -> HRESULT {
    if stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspActivate`; deactivated once.
    let stream = unsafe { Box::from_raw(stream) };
    // SAFETY: the connection outlives its streams.
    let dsp = unsafe { &*stream.dsp };
    *dsp.active.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
    S_OK
}

/// Completes the command at once: a convolution stream with an input gets
/// the synthetic reverb tail (see module docs); everything else stays
/// silence.
pub extern "system" fn XDspSubmitCommand(
    stream: *mut Stream,
    command: *const Command,
    ticket: *mut u32,
) -> HRESULT {
    if stream.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a stream from `XDspActivate`, the title's command.
    let (stream, command) = unsafe { (&mut *stream, &*command) };
    let samples = (stream.frames * stream.channels) as usize;
    if let Some((dir, n)) = &stream.dump {
        let input = if command.input.is_null() {
            &[][..]
        } else {
            // SAFETY: the title's input buffer holds one block of the stream.
            unsafe { std::slice::from_raw_parts(command.input, samples) }
        };
        dump_block(dir, *n, command.gains, input);
    }
    if !command.output.is_null() {
        if guess_dsp()
            && stream.kind == CONVOLUTION
            && !command.input.is_null()
            && let Some(convolver) = stream.convolver.as_mut()
        {
            if std::ptr::eq(command.input, command.output) {
                // SAFETY: `command.output` holds `samples` floats for one block.
                let buf = unsafe { std::slice::from_raw_parts_mut(command.output, samples) };
                convolver.process_in_place(buf, command.gains);
            } else {
                // SAFETY: `command.input` holds `samples` floats for one block.
                let input = unsafe { std::slice::from_raw_parts(command.input, samples) };
                // SAFETY: `command.output` holds `samples` floats for one block
                // and does not equal `command.input`.
                let out = unsafe { std::slice::from_raw_parts_mut(command.output, samples) };
                convolver.process(input, command.gains, out);
            }
        } else {
            // SAFETY: the title's output buffer holds one block of the stream.
            unsafe { std::ptr::write_bytes(command.output, 0, samples) };
        }
    }
    if let (Some((dir, n)), false) = (&stream.dump, command.output.is_null()) {
        // SAFETY: the title's output buffer, just filled above.
        let out = unsafe { std::slice::from_raw_parts(command.output, samples) };
        dump_output(dir, *n, out);
    }
    let status = &mut *stream.status;
    let next = status.submitted.wrapping_add(1);
    // SAFETY: the status block is read by title code on other threads.
    unsafe {
        std::ptr::write_volatile(&mut status.submitted, next);
        std::ptr::write_volatile(&mut status.result, S_OK);
        std::ptr::write_volatile(&mut status.completed, next);
    }
    if !ticket.is_null() {
        // SAFETY: optional out parameter.
        unsafe { ticket.write(next) };
    }
    S_OK
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dsp",
        dlls: &["chad.dll"],
        exports: vec![
            hle_export!(XDspConnectWithMaximumStreamLimit),
            hle_export!(XDspDisconnect),
            hle_export!(XDspActivate),
            hle_export!(XDspDeactivate),
            hle_export!(XDspSubmitCommand),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_stream_as_the_title_drives_it() {
        // SAFETY: no other test reads VICTORY_GUESS_DSP.
        unsafe { std::env::set_var("VICTORY_GUESS_DSP", "1") };
        let mut base = vec![0u8; 0x10_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(
            XDspConnectWithMaximumStreamLimit(
                base.as_mut_ptr().cast(),
                0x10_0000,
                24,
                0x80,
                1,
                &mut dsp
            ),
            S_OK
        );
        let ir = vec![0f32; 0x800];
        // Stereo convolution with option 2, as the title asks for.
        let desc = StreamDesc {
            kind: CONVOLUTION,
            options: 2,
            frames: 0x400,
            channels: 2,
            ir_floats: 0x800,
            ir: ir.as_ptr(),
        };
        let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
        assert_eq!(XDspActivate(dsp, &desc, &mut status, &mut stream), S_OK);
        let mut output = vec![1f32; 0x800];
        let input = vec![0.5f32; 0x800];
        let command = Command {
            gains: [1.0, 1.0],
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            reserved: 0,
        };
        let mut ticket = 0;
        assert_eq!(XDspSubmitCommand(stream, &command, &mut ticket), S_OK);
        // SAFETY: the stream's status block.
        let (completed, result) = unsafe { ((*status).completed, (*status).result) };
        assert_eq!((completed, result), (ticket, S_OK));
        // A convolution stream with real input no longer returns digital
        // silence: it gets the synthetic reverb tail.
        assert!(output.iter().any(|&s| s != 0.0));
        assert_ne!(output, vec![1f32; 0x800]);
        assert_eq!(XDspDeactivate(stream), S_OK);

        // Mono streams refuse stereo options; transforms are mono only.
        let bad = StreamDesc {
            channels: 1,
            ..desc
        };
        assert_eq!(
            XDspActivate(dsp, &bad, &mut status, &mut stream),
            E_INVALIDARG
        );
        let transform = StreamDesc {
            kind: 2,
            options: 0,
            channels: 1,
            ir_floats: 0,
            ir: std::ptr::null(),
            ..desc
        };
        assert_eq!(
            XDspActivate(dsp, &transform, &mut status, &mut stream),
            S_OK
        );
        assert_eq!(XDspDeactivate(stream), S_OK);
        assert_eq!(XDspDisconnect(dsp), S_OK);
        // SAFETY: set at the top of this test; nothing else reads it.
        unsafe { std::env::remove_var("VICTORY_GUESS_DSP") };
    }

    #[test]
    fn convolve_merged_is_a_causal_fir_per_channel_scaled_by_gain() {
        // Stereo, no history (hist_frames 0), 3 frames/channel: interleaved
        // L0 R0 L1 R1 L2 R2.
        let merged = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let ir = [1.0, 0.5, 0.25];
        let mut out = [0.0; 6];
        convolve_merged(&merged, 0, 2, &ir, [2.0, 3.0], &mut out);
        // Left channel is [1,0,0] convolved with ir, times gain 2: [1, 0.5, 0.25] * 2.
        // Right channel is [2,0,0] convolved with ir, times gain 3: [2, 1, 0.5] * 3.
        assert_eq!(out, [2.0, 6.0, 1.0, 3.0, 0.5, 1.5]);
    }

    #[test]
    fn convolve_merged_reads_history_as_the_frames_before_the_block() {
        // Mono, one frame of history (5.0), then the block [0.0, 0.0].
        let merged = [5.0, 0.0, 0.0];
        let ir = [1.0, 0.5, 0.25];
        let mut out = [0.0; 2];
        convolve_merged(&merged, 1, 1, &ir, [1.0, 0.0], &mut out);
        // n=0 reaches one frame back into history: 0*1 + 5*0.5 = 2.5.
        // n=1 reaches two frames back: 0*1 + 0*0.5 + 5*0.25 = 1.25.
        assert_eq!(out, [2.5, 1.25]);
    }

    #[test]
    fn a_block_split_matches_one_convolution_over_the_whole_signal() {
        // The point of carrying history across calls: splitting one signal
        // into two blocks and convolving them in sequence, carrying the
        // tail, must give the same result as convolving it in one go. This
        // is the property that stops a reverb tail from clicking at every
        // block boundary.
        let ir = synthetic_ir(5);
        let channels = 1;
        let whole = [1.0, 0.3, -0.2, 0.0, 0.5, 0.1, -0.4, 0.2];
        let gains = [1.0, 0.0];

        let mut reference = [0.0; 8];
        convolve_merged(&whole, 0, channels, &ir, gains, &mut reference);

        let (first, second) = whole.split_at(4);
        let mut history = vec![0.0; ir.len() - 1];
        let mut out1 = [0.0; 4];
        let mut scratch: Vec<f32> = history.clone();
        scratch.extend_from_slice(first);
        convolve_merged(&scratch, history.len(), channels, &ir, gains, &mut out1);
        let keep = history.len();
        history.copy_from_slice(&scratch[scratch.len() - keep..]);

        let mut out2 = [0.0; 4];
        scratch.clear();
        scratch.extend_from_slice(&history);
        scratch.extend_from_slice(second);
        convolve_merged(&scratch, history.len(), channels, &ir, gains, &mut out2);

        let streamed: Vec<f32> = out1.iter().chain(out2.iter()).copied().collect();
        assert_eq!(streamed, reference);
    }

    #[test]
    fn dump_captures_the_titles_ir_blocks_and_output() {
        let dir = std::env::temp_dir().join(format!("victory-dsp-dump-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // The dump captures whatever the stream actually wrote, so this
        // test does not touch VICTORY_GUESS_DSP, which the other test
        // toggles.
        // SAFETY: no other test reads VICTORY_DUMP_DSP.
        unsafe { std::env::set_var("VICTORY_DUMP_DSP", &dir) };
        let mut base = vec![0u8; 0x10_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(
            XDspConnectWithMaximumStreamLimit(
                base.as_mut_ptr().cast(),
                0x10_0000,
                24,
                0x80,
                1,
                &mut dsp
            ),
            S_OK
        );
        // A recognisable pattern, to prove the dump holds the title's own
        // IR bytes and not the synthetic tail.
        let ir: Vec<f32> = (0..0x800).map(|i| i as f32 * 0.25).collect();
        let desc = StreamDesc {
            kind: CONVOLUTION,
            options: 2,
            frames: 0x400,
            channels: 2,
            ir_floats: 0x800,
            ir: ir.as_ptr(),
        };
        let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
        assert_eq!(XDspActivate(dsp, &desc, &mut status, &mut stream), S_OK);
        let mut output = vec![0f32; 0x800];
        let input = vec![0.5f32; 0x800];
        let command = Command {
            gains: [1.0, 0.5],
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            reserved: 0,
        };
        let mut ticket = 0;
        assert_eq!(XDspSubmitCommand(stream, &command, &mut ticket), S_OK);

        let read_f32 = |name: &str| -> Vec<f32> {
            let bytes = std::fs::read(dir.join(name)).unwrap();
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        assert_eq!(read_f32("stream0_ir.f32"), ir);
        let blocks = read_f32("stream0_blocks.f32");
        assert_eq!(&blocks[..2], &[1.0, 0.5]);
        assert_eq!(&blocks[2..], &input[..]);
        assert_eq!(read_f32("stream0_out.f32"), output);

        assert_eq!(XDspDeactivate(stream), S_OK);
        assert_eq!(XDspDisconnect(dsp), S_OK);
        // SAFETY: set at the top of this test; nothing else reads it.
        unsafe { std::env::remove_var("VICTORY_DUMP_DSP") };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthetic_ir_is_deterministic_and_decays() {
        let a = synthetic_ir(16);
        let b = synthetic_ir(16);
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
        // The tail dies out: the back half carries much less energy than the front half.
        let front: f32 = a[..8].iter().map(|s| s * s).sum();
        let back: f32 = a[8..].iter().map(|s| s * s).sum();
        assert!(back < front);
    }

    #[test]
    fn partitioned_fft_matches_scalar_reference_across_blocks_and_channels() {
        for &(block, channels, ir_len, num_blocks, gains) in &[
            (4usize, 1usize, 3usize, 5usize, [1.0f32, 0.0]),
            (4, 2, 11, 6, [2.0, 0.5]),
            (8, 2, 16, 5, [0.75, 1.25]),
            (64, 1, 200, 6, [1.5, 0.0]),
            (512, 2, TAPS, 12, [0.8, 1.2]),
        ] {
            let ir = synthetic_ir(ir_len);
            let total_frames = block * num_blocks;
            let whole: Vec<f32> = (0..total_frames * channels)
                .map(|i| ((i as f32 * 0.137).sin()) * 0.5)
                .collect();
            let mut reference = vec![0.0f32; whole.len()];
            convolve_merged(&whole, 0, channels, &ir, gains, &mut reference);

            let mut conv = PartitionedConvolver::new(block, channels, &ir);
            let mut streamed = Vec::with_capacity(whole.len());
            let mut out = vec![0.0f32; block * channels];
            for b in 0..num_blocks {
                let start = b * block * channels;
                conv.process(&whole[start..start + block * channels], gains, &mut out);
                streamed.extend_from_slice(&out);
            }
            let max_err = streamed
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-3,
                "block={block} ch={channels} ir_len={ir_len}: max_err={max_err}"
            );
        }
    }

    #[test]
    fn partitioned_fft_unit_impulse_reproduces_ir_across_partitions() {
        let ir = synthetic_ir(1300);
        let block = 512;
        let channels = 2;
        let gains = [1.0, -0.5];
        let mut conv = PartitionedConvolver::new(block, channels, &ir);
        let mut collected = Vec::new();
        let mut out = vec![0.0f32; block * channels];
        for b in 0..4 {
            let mut input = vec![0.0f32; block * channels];
            if b == 0 {
                input[0] = 1.0; // L impulse
                input[1] = 2.0; // R impulse
            }
            conv.process(&input, gains, &mut out);
            collected.extend_from_slice(&out);
        }
        for (i, &h) in ir.iter().enumerate() {
            let l = collected[i * 2];
            let r = collected[i * 2 + 1];
            assert!((l - h * gains[0]).abs() < 1e-4, "L mismatch at {i}: {l} vs {}", h * gains[0]);
            assert!(
                (r - h * 2.0 * gains[1]).abs() < 1e-4,
                "R mismatch at {i}: {r} vs {}",
                h * 2.0 * gains[1]
            );
        }
        // Past the end of the IR, output returns to 0.
        for i in ir.len()..(block * 4) {
            assert!(collected[i * 2].abs() < 1e-4, "L tail non-zero at {i}");
            assert!(collected[i * 2 + 1].abs() < 1e-4, "R tail non-zero at {i}");
        }
    }

    #[test]
    fn partitioned_fft_in_place_matches_out_of_place() {
        let ir = synthetic_ir(1500);
        let block = 512;
        let channels = 2;
        let gains = [0.6, 1.4];
        let mut conv_oop = PartitionedConvolver::new(block, channels, &ir);
        let mut conv_ip = PartitionedConvolver::new(block, channels, &ir);
        for b in 0..5 {
            let input: Vec<f32> = (0..block * channels)
                .map(|i| (((b * block * channels + i) as f32) * 0.23).cos())
                .collect();
            let mut out_oop = vec![0.0f32; block * channels];
            conv_oop.process(&input, gains, &mut out_oop);
            let mut buf_ip = input.clone();
            conv_ip.process_in_place(&mut buf_ip, gains);
            assert_eq!(out_oop, buf_ip);
        }
    }
}
