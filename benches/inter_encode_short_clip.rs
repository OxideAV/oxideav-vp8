//! Four-frame inter encode bench — one keyframe + three P-frames at
//! 128×128, driven through `Vp8InterStreamEncoder`. This is the
//! shortest workload that exercises the §16 inter path (selection of
//! reference + §17 motion vectors + §15.1 inter-frame loop filter) on
//! top of the keyframe path the `keyframe_encode` bench measures.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::{I420Frame, KeyframeParams, Vp8InterStreamEncoder};

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;
const FRAMES: usize = 4;

/// Build a deterministic 4-frame I420 clip. Each frame is a translated
/// version of the previous one (one-pixel-per-frame drift) so the
/// inter path has actual motion to compensate.
fn synthesise() -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let cw = (WIDTH as usize).div_ceil(2);
    let ch = (HEIGHT as usize).div_ceil(2);

    (0..FRAMES)
        .map(|t| {
            let mut y = vec![0u8; w * h];
            for row in 0..h {
                for col in 0..w {
                    let v = (((col + t) * 256 / w + (row + t) * 256 / h) / 2) as u8;
                    y[row * w + col] = v;
                }
            }
            let mut u = vec![128u8; cw * ch];
            let mut v = vec![128u8; cw * ch];
            for row in 0..ch {
                for col in 0..cw {
                    u[row * cw + col] = (120 + (col * 16 / cw) + t) as u8;
                    v[row * cw + col] = (130 + (row * 16 / ch) + t) as u8;
                }
            }
            (y, u, v)
        })
        .collect()
}

fn bench_inter_clip_128x128(c: &mut Criterion) {
    let clip = synthesise();
    let params = KeyframeParams {
        y_ac_qi: 32,
        ..KeyframeParams::default()
    };

    let mut g = c.benchmark_group("inter_encode_short_clip");
    g.throughput(Throughput::Elements(
        (WIDTH * HEIGHT) as u64 * FRAMES as u64,
    ));
    g.sample_size(10);
    g.bench_function("inter_encode_4f_128x128_qi32", |b| {
        b.iter(|| {
            // keyframe_interval = FRAMES so we get exactly one K + three P.
            let mut enc =
                Vp8InterStreamEncoder::new(params, FRAMES as u64).expect("Vp8InterStreamEncoder");
            let mut total = 0usize;
            for (yi, ui, vi) in &clip {
                let frame = I420Frame::packed(WIDTH, HEIGHT, yi, ui, vi);
                let pkt = enc
                    .encode_frame(black_box(&frame))
                    .expect("inter encode_frame");
                total += pkt.bytes.len();
            }
            black_box(total)
        });
    });
    g.finish();
}

criterion_group!(benches, bench_inter_clip_128x128);
criterion_main!(benches);
