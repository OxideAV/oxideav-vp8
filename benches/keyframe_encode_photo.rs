//! Whole-keyframe encode on photo-like content, per effort level and
//! thread budget — the round-486 latency bench.
//!
//! The picture is the deterministic integer-synthesised scene of
//! `tests/support/photo.rs` at 800×533 and 2400×1600, encoded at the
//! quality-80-equivalent operating point (`y_ac_qi = 25` with the
//! matching trellis strength). Each case is one
//! `encode_keyframe_with_options` call; the output size of every case is
//! printed once so time and bytes can be read together.
//!
//! `examples/keyframe_timing.rs` prints the same matrix as a plain table
//! (with PSNR and a byte digest) without the criterion machinery.

#[path = "../tests/support/photo.rs"]
mod photo;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use oxideav_vp8::encoder::quality_to_trellis_strength;
use oxideav_vp8::{
    EncodeEffort, I420Frame, KeyframeEncodeOptions, KeyframeParams, encode_keyframe_with_options,
};

fn bench_photo(c: &mut Criterion) {
    let params = KeyframeParams::default()
        .with_y_ac_qi(25)
        .with_trellis_strength(quality_to_trellis_strength(80.0));
    for (w, h) in [(800usize, 533usize), (2400, 1600)] {
        let (y, u, v) = photo::photo_like_i420(w, h);
        let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
        let mut g = c.benchmark_group(format!("keyframe_encode_photo_{w}x{h}"));
        g.throughput(Throughput::Elements((w * h) as u64));
        g.sample_size(10);
        for effort in EncodeEffort::ALL.into_iter().rev() {
            for threads in [1usize, 4] {
                let options = KeyframeEncodeOptions::new()
                    .with_effort(effort)
                    .with_threads(threads);
                let bytes = encode_keyframe_with_options(&frame, &params, &options)
                    .expect("encode")
                    .len();
                eprintln!(
                    "{w}x{h} {} / {threads} thread(s): {bytes} bytes",
                    effort.name()
                );
                g.bench_function(
                    BenchmarkId::new(effort.name(), format!("{threads}t")),
                    |b| {
                        b.iter(|| {
                            black_box(
                                encode_keyframe_with_options(
                                    black_box(&frame),
                                    black_box(&params),
                                    &options,
                                )
                                .expect("encode"),
                            )
                        });
                    },
                );
            }
        }
        g.finish();
    }
}

criterion_group!(benches, bench_photo);
criterion_main!(benches);
