//! Whole-keyframe decode bench — `oxideav_vp8::decode_vp8` consuming a
//! 320×240 keyframe that the crate's own `encode_keyframe` produced.
//! Pairs with `keyframe_encode` so the two halves of a roundtrip are
//! measured independently.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::{decode_vp8, encode_keyframe, I420Frame, KeyframeParams};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

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

fn bench_decode_keyframe_320x240(c: &mut Criterion) {
    let (y, u, v) = synthesise();
    let frame = I420Frame::packed(WIDTH, HEIGHT, &y, &u, &v);
    let params = KeyframeParams {
        y_ac_qi: 32,
        ..KeyframeParams::default()
    };
    let bytes = encode_keyframe(&frame, &params).expect("encode_keyframe");

    let mut g = c.benchmark_group("keyframe_decode");
    g.throughput(Throughput::Elements((WIDTH * HEIGHT) as u64));
    g.sample_size(20);
    g.bench_function("decode_keyframe_320x240_qi32", |b| {
        b.iter(|| {
            let decoded = decode_vp8(black_box(&bytes)).expect("decode_vp8");
            black_box(decoded)
        });
    });
    g.finish();
}

criterion_group!(benches, bench_decode_keyframe_320x240);
criterion_main!(benches);
