//! Four-frame inter decode bench — `Vp8DecoderState` consuming the
//! one-keyframe + three-P-frame 128×128 stream that the crate's own
//! `Vp8InterStreamEncoder` produces from the same deterministic drift
//! clip the `inter_encode_short_clip` bench encodes. This is the decode
//! half of the §16 inter roundtrip: §16.1 ref selection, §16.2/§17 MV
//! decode, §18 motion compensation (whole-pixel copies + §18.3 six-tap
//! sub-pixel synthesis), §14 transforms, and the §15.1 inter-frame loop
//! filter all run per iteration. Pairs with `keyframe_decode` the same
//! way `inter_encode_short_clip` pairs with `keyframe_encode`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::{I420Frame, KeyframeParams, Vp8DecoderState, Vp8InterStreamEncoder};

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;
const FRAMES: usize = 4;

/// Build the same deterministic 4-frame I420 clip the
/// `inter_encode_short_clip` bench drives: each frame is a translated
/// version of the previous one (one-pixel-per-frame drift) so the
/// decoded stream carries real motion vectors, not all-ZEROMV.
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

/// Encode the clip once (setup, unmeasured) into the 4-packet stream
/// every decode iteration replays.
fn encode_stream() -> Vec<Vec<u8>> {
    let clip = synthesise();
    let params = KeyframeParams {
        y_ac_qi: 32,
        ..KeyframeParams::default()
    };
    // keyframe_interval = FRAMES so the stream is exactly one K + three P.
    let mut enc = Vp8InterStreamEncoder::new(params, FRAMES as u64).expect("Vp8InterStreamEncoder");
    clip.iter()
        .map(|(yi, ui, vi)| {
            let frame = I420Frame::packed(WIDTH, HEIGHT, yi, ui, vi);
            enc.encode_frame(&frame).expect("inter encode_frame").bytes
        })
        .collect()
}

fn bench_inter_decode_128x128(c: &mut Criterion) {
    let packets = encode_stream();

    let mut g = c.benchmark_group("inter_decode_short_clip");
    g.throughput(Throughput::Elements(
        (WIDTH * HEIGHT) as u64 * FRAMES as u64,
    ));
    g.sample_size(20);
    g.bench_function("inter_decode_4f_128x128_qi32", |b| {
        b.iter(|| {
            let mut dec = Vp8DecoderState::new();
            let mut acc = 0usize;
            for pkt in &packets {
                let frame = dec.decode_frame(black_box(pkt)).expect("decode_frame");
                acc += frame.y.len();
            }
            black_box(acc)
        });
    });
    g.finish();
}

criterion_group!(benches, bench_inter_decode_128x128);
criterion_main!(benches);
