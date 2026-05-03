//! Criterion benchmarks for the VP8 decoder hot paths.
//!
//! Runs against the docs/video/vp8/fixtures/ corpus — the same set the
//! integration tests in `tests/docs_corpus.rs` consume. Each benchmark
//! decodes one of the well-characterised fixtures end-to-end through
//! [`Vp8Decoder`] (header parse, token partitions, IDCT/WHT,
//! intra/inter reconstruction, loop filter, output crop). Measuring at
//! the public-API boundary keeps the numbers honest: any micro-opt has
//! to actually move the needle on a real frame, not just a synthetic
//! micro-bench.
//!
//! Fixtures targeted (rationale):
//! * `tiny-i-only-16x16` — single-MB intra. Isolates per-MB constant
//!   overhead (header parse, BoolDecoder priming, allocations).
//! * `i-only-64x64` — 16-MB intra with B_PRED neighbour propagation;
//!   exercises `predict_4x4`, `idct4x4` × 16, the per-MB scratch
//!   allocations.
//! * `i-only-loopfilter-high` — same shape as i-only-64x64 but with a
//!   high filter level so the normal-mode 4-tap edge filter runs at
//!   every sub-block boundary. Loopfilter is typically the single
//!   largest cost in a VP8 frame.
//! * `q-high` — 128×128 mandelbrot at qindex 127. Maximises residual
//!   coefficient density → stresses the bool-coder + token tree path
//!   in `tokens::decode_block`.
//! * `i-frame-then-p-frame-64x64` — single I + single P decode through
//!   a stateful `Vp8Decoder`. Exercises motion compensation
//!   (`sixtap_predict`, `bilinear_predict`) on top of the intra path.
//!
//! Run with:
//!     cargo bench -p oxideav-vp8 --bench decode

use std::fs;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::{CodecId, Decoder, Packet, TimeBase};
use oxideav_vp8::bool_decoder::BoolDecoder;
use oxideav_vp8::decoder::Vp8Decoder;

const IVF_HEADER_LEN: usize = 32;
const IVF_FRAME_HEADER_LEN: usize = 12;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/video/vp8/fixtures").join(name)
}

/// Pull every elementary VP8 frame out of an IVF buffer. Same parser
/// shape as `tests/docs_corpus.rs::ivf_frames`.
fn ivf_frames(ivf: &[u8]) -> Vec<Vec<u8>> {
    assert!(ivf.len() >= IVF_HEADER_LEN, "IVF too short");
    assert_eq!(&ivf[0..4], b"DKIF");
    let header_len = u16::from_le_bytes([ivf[6], ivf[7]]) as usize;
    let mut off = header_len;
    let mut out = Vec::new();
    while off + IVF_FRAME_HEADER_LEN <= ivf.len() {
        let size =
            u32::from_le_bytes([ivf[off], ivf[off + 1], ivf[off + 2], ivf[off + 3]]) as usize;
        off += IVF_FRAME_HEADER_LEN;
        if off + size > ivf.len() {
            break;
        }
        out.push(ivf[off..off + size].to_vec());
        off += size;
    }
    out
}

/// Read every elementary frame for a fixture. Returns `None` if the
/// fixture file is missing — benches just skip rather than fail (matches
/// the corpus-tests pattern for CI-without-docs setups).
fn load_fixture_frames(name: &str) -> Option<Vec<Vec<u8>>> {
    let path = fixture_dir(name).join("input.ivf");
    let bytes = fs::read(&path).ok()?;
    Some(ivf_frames(&bytes))
}

/// Decode every frame in `frames` through a freshly-initialised decoder.
/// Drains output frames so the inter-frame state machine is exercised
/// end-to-end (matters for the I+P bench).
fn decode_all(frames: &[Vec<u8>]) {
    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    for (i, fb) in frames.iter().enumerate() {
        let mut pkt = Packet::new(0, TimeBase::new(1, 1000), fb.clone());
        pkt.pts = Some(i as i64);
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(_) => continue,
                Err(oxideav_core::Error::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e:?}"),
            }
        }
    }
}

fn bench_decode_i_frame(c: &mut Criterion) {
    let Some(frames) = load_fixture_frames("tiny-i-only-16x16") else {
        eprintln!("skip decode_i_frame: tiny-i-only-16x16 fixture missing");
        return;
    };
    // 1 frame, 16×16 intra → tightest possible per-frame loop. Throughput
    // reported per frame so multi-frame benches can be compared apples-to-
    // apples (criterion converts to elements/sec).
    let mut g = c.benchmark_group("decode_i_frame");
    g.throughput(Throughput::Elements(frames.len() as u64));
    g.bench_function(BenchmarkId::from_parameter("tiny-i-only-16x16"), |b| {
        b.iter(|| decode_all(&frames));
    });
    g.finish();
}

fn bench_decode_i_only_64x64(c: &mut Criterion) {
    let Some(frames) = load_fixture_frames("i-only-64x64") else {
        eprintln!("skip decode_i_only_64x64: fixture missing");
        return;
    };
    // 16 MBs of B_PRED intra → exercises the per-MB pred + IDCT + scratch
    // allocation pattern that intra-heavy keyframes hit.
    let mut g = c.benchmark_group("decode_i_only_64x64");
    g.throughput(Throughput::Bytes(64 * 64 * 3 / 2));
    g.bench_function(BenchmarkId::from_parameter("i-only-64x64"), |b| {
        b.iter(|| decode_all(&frames));
    });
    g.finish();
}

fn bench_decode_with_loopfilter(c: &mut Criterion) {
    let Some(frames) = load_fixture_frames("i-only-loopfilter-high") else {
        eprintln!("skip decode_with_loopfilter: i-only-loopfilter-high missing");
        return;
    };
    // Same MB count as i-only-64x64 but with the loopfilter cranked to
    // a level that triggers the wide-MB filter on every edge. Subtract
    // the i-only-64x64 number to estimate raw loopfilter cost.
    let mut g = c.benchmark_group("decode_with_loopfilter");
    g.throughput(Throughput::Bytes(64 * 64 * 3 / 2));
    g.bench_function(BenchmarkId::from_parameter("i-only-loopfilter-high"), |b| {
        b.iter(|| decode_all(&frames));
    });
    g.finish();
}

fn bench_decode_q_high(c: &mut Criterion) {
    let Some(frames) = load_fixture_frames("q-high") else {
        eprintln!("skip decode_q_high: fixture missing");
        return;
    };
    // 128×128 mandelbrot at qindex 127 → fattest residual stream the
    // corpus carries. Stresses tokens::decode_block + bool_decoder.
    let mut g = c.benchmark_group("decode_q_high");
    g.throughput(Throughput::Bytes(128 * 128 * 3 / 2));
    g.bench_function(BenchmarkId::from_parameter("q-high"), |b| {
        b.iter(|| decode_all(&frames));
    });
    g.finish();
}

fn bench_decode_i_then_p(c: &mut Criterion) {
    let Some(frames) = load_fixture_frames("i-frame-then-p-frame-64x64") else {
        eprintln!("skip decode_i_then_p: i-frame-then-p-frame-64x64 missing");
        return;
    };
    // Single I + single P → exercises the inter reconstruction path
    // (sixtap_predict / bilinear_predict / find_near_mvs) on top of the
    // intra path. P-frame motion comp is one of the bigger inter costs.
    let mut g = c.benchmark_group("decode_i_then_p");
    g.throughput(Throughput::Elements(frames.len() as u64));
    g.bench_function(BenchmarkId::from_parameter("i-frame-then-p-frame-64x64"), |b| {
        b.iter(|| decode_all(&frames));
    });
    g.finish();
}

/// Micro-bench: drive the boolean (arithmetic) decoder over a fixed
/// blob of bytes with a deterministic mix of probabilities. Catches
/// regressions in the per-bit hot path independently of the rest of
/// the decoder. Probability schedule cycles through {16, 64, 128, 192,
/// 240} so renormalisation runs on every other bit on average.
fn bench_bool_coder_throughput(c: &mut Criterion) {
    // Deterministic 4 KiB pseudo-random source — the bool decoder treats
    // out-of-range reads as zero so the exact distribution doesn't
    // matter, only the size + the prob schedule applied per bit.
    let mut buf = vec![0u8; 4096];
    let mut state: u32 = 0xdead_beef;
    for b in buf.iter_mut() {
        // xorshift32 — small + branch-free, no hidden criterion dep.
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *b = (state & 0xff) as u8;
    }
    let probs: [u32; 5] = [16, 64, 128, 192, 240];
    let mut g = c.benchmark_group("bool_coder_throughput");
    g.throughput(Throughput::Bytes(buf.len() as u64));
    g.bench_function("4kib_mixed_probs", |b| {
        b.iter(|| {
            let mut dec = BoolDecoder::new(&buf).expect("bool dec init");
            // 8 bits/byte is the cap; cycle the probability schedule so
            // we don't pin ourselves to a single fast-path branch.
            let mut sink: u32 = 0;
            for i in 0..(buf.len() * 8) {
                let p = probs[i % probs.len()];
                sink = sink.wrapping_add(dec.read_bool(p) as u32);
            }
            criterion::black_box(sink)
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_decode_i_frame,
    bench_decode_i_only_64x64,
    bench_decode_with_loopfilter,
    bench_decode_q_high,
    bench_decode_i_then_p,
    bench_bool_coder_throughput,
);
criterion_main!(benches);
