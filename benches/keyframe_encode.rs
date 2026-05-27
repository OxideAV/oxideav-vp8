//! Whole-keyframe encode bench — `oxideav_vp8::encode_keyframe` driving
//! a 320×240 synthetic I420 source at `y_ac_qi = 32`. The source is
//! deterministic (a mixed gradient + flat-square pattern) and shared
//! with `keyframe_decode` so the two benches measure complementary
//! halves of the same picture's roundtrip cost.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::{encode_keyframe, I420Frame, KeyframeParams};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// Build a deterministic synthetic 320×240 I420 picture. Mix of a
/// diagonal luma gradient (B_PRED-friendly) with a centre flat-128
/// square (DC_PRED-friendly) and gentle chroma gradients — exercises
/// both intra paths so the encoder's per-MB mode pick does real work.
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

fn bench_encode_keyframe_320x240(c: &mut Criterion) {
    let (y, u, v) = synthesise();
    let params = KeyframeParams {
        y_ac_qi: 32,
        ..KeyframeParams::default()
    };

    let mut g = c.benchmark_group("keyframe_encode");
    // Throughput in pixels — lets readers compute MPx/s.
    g.throughput(Throughput::Elements((WIDTH * HEIGHT) as u64));
    g.sample_size(20);
    g.bench_function("encode_keyframe_320x240_qi32", |b| {
        b.iter(|| {
            let frame = I420Frame::packed(WIDTH, HEIGHT, &y, &u, &v);
            let bytes =
                encode_keyframe(black_box(&frame), black_box(&params)).expect("encode_keyframe");
            black_box(bytes)
        });
    });
    g.finish();
}

criterion_group!(benches, bench_encode_keyframe_320x240);
criterion_main!(benches);
