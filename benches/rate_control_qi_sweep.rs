//! Round 194: rate-control knob sweep on `KeyframeParams::y_ac_qi`.
//!
//! This bench wires the public §9.6 baseline quantiser index
//! (`KeyframeParams::y_ac_qi`, range `0..=127`) into a 10-point criterion
//! sweep on a fixed 320×240 deterministic I420 source. The same source +
//! macroblock-mode picker is shared with `keyframe_encode` so the two
//! benches measure complementary halves of the same dial: `keyframe_encode`
//! pins `y_ac_qi = 32` and measures end-to-end encode wall time,
//! `rate_control_qi_sweep` walks `y_ac_qi` across ten representative
//! values (`8, 16, 24, 32, 40, 48, 56, 72, 96, 120`) so the run also
//! produces the output-byte / wall-time pairing readers need to pick a
//! starting point for tuning.
//!
//! `y_ac_qi` is the principal rate-control knob exposed by the encoder
//! per RFC 6386 §9.6 — every other §9.6 quantiser delta defaults to 0
//! in `KeyframeParams`, so the sweep here moves the entire DC + AC luma
//! / chroma quantiser bank in lockstep through `y_ac_qi`. Lower values
//! produce larger output (higher quality); higher values produce smaller
//! output (lower quality). The per-call output size is captured via a
//! one-shot encode in `setup_*` and printed on stderr at run time; the
//! same numbers + Mpx/s throughput are tabulated in `BENCHMARKS.md`
//! under "Round 194 — rate-control qi sweep".

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::{encode_keyframe, I420Frame, KeyframeParams};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// Sweep points across the §9.6 `y_ac_qi` field. Ten representative
/// values covering the full range 0..=127:
///   *  8 — near-lossless, large output
///   * 16 / 24 / 32 — high-quality cluster (32 is the default)
///   * 40 / 48 / 56 — mid-quality cluster
///   * 72 / 96 — low-quality cluster
///   * 120 — near-floor, smallest output
const SWEEP_QI: &[u8] = &[8, 16, 24, 32, 40, 48, 56, 72, 96, 120];

/// Build the same deterministic synthetic 320×240 I420 picture shared
/// with `keyframe_encode` (mixed luma gradient + centre flat-128 square
/// + gentle chroma gradients). Identical pixels across both benches.
fn synthesise() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let cw = (WIDTH as usize).div_ceil(2);
    let ch = (HEIGHT as usize).div_ceil(2);

    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut v = ((col * 256 / w + row * 256 / h) / 2) as u8;
            if col >= w / 4 && col < w * 3 / 4 && row >= h / 4 && row < h * 3 / 4 {
                v = 128;
            }
            y[row * w + col] = v;
        }
    }
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            u[row * cw + col] = (120 + (col * 16 / cw)) as u8;
            v[row * cw + col] = (130 + (row * 16 / ch)) as u8;
        }
    }
    (y, u, v)
}

fn bench_qi_sweep(c: &mut Criterion) {
    let (y, u, v) = synthesise();

    let mut group = c.benchmark_group("rate_control_qi_sweep");
    group.throughput(Throughput::Elements((WIDTH * HEIGHT) as u64));
    group.sample_size(20);

    for &qi in SWEEP_QI {
        let params = KeyframeParams {
            y_ac_qi: qi,
            ..KeyframeParams::default()
        };

        // One-shot encode at this qi so the bench prologue prints the
        // output size that goes with the timed throughput — readers
        // need both axes (wall time + output bytes) to make a tuning
        // call. The number is deterministic across runs because the
        // source picture is fixed and the encoder is determined by
        // KeyframeParams.
        let frame = I420Frame::packed(WIDTH, HEIGHT, &y, &u, &v);
        let probe = encode_keyframe(&frame, &params).expect("probe encode");
        eprintln!(
            "rate_control_qi_sweep: qi={:>3} -> {:>6} bytes ({:>5.2} bpp)",
            qi,
            probe.len(),
            (probe.len() as f64) * 8.0 / (WIDTH as f64 * HEIGHT as f64),
        );

        let name = format!("encode_320x240_qi{qi:03}");
        group.bench_function(name, |b| {
            b.iter(|| {
                let frame = I420Frame::packed(WIDTH, HEIGHT, &y, &u, &v);
                let bytes = encode_keyframe(black_box(&frame), black_box(&params))
                    .expect("encode_keyframe");
                black_box(bytes)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_qi_sweep);
criterion_main!(benches);
