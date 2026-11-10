//! Timed keyframe encode on the deterministic photo-like picture.
//!
//! `cargo run --release --example keyframe_timing -- [WIDTH HEIGHT [QI [RUNS]]]`
//!
//! Prints, per configuration, the minimum and median wall time over `RUNS` encodes,
//! the output size, the luma PSNR of this crate's own decode, and an
//! FNV-1a digest of the bytes (the byte-identity pin).

#[path = "../tests/support/photo.rs"]
mod photo;

use std::time::Instant;

use oxideav_vp8::encoder::quality_to_trellis_strength;
use oxideav_vp8::{I420Frame, KeyframeParams, decode_vp8, encode_keyframe};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric argument"))
        .collect();
    let w = args.first().copied().unwrap_or(800);
    let h = args.get(1).copied().unwrap_or(533);
    let qi = args.get(2).copied().unwrap_or(25) as u8;
    let runs = args.get(3).copied().unwrap_or(5).max(1);

    let (y, u, v) = photo::photo_like_i420(w, h);
    let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
    let params = KeyframeParams::default()
        .with_y_ac_qi(qi)
        .with_trellis_strength(quality_to_trellis_strength(80.0));

    let mut times = Vec::with_capacity(runs);
    let mut bytes = Vec::new();
    for _ in 0..runs {
        let t = Instant::now();
        bytes = encode_keyframe(&frame, &params).expect("encode");
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let decoded = decode_vp8(&bytes).expect("decode");
    println!(
        "{w}x{h} qi {qi}: min {:8.2} ms  median {:8.2} ms  {:8} bytes  {:.2} dB  {:016x}",
        times[0],
        times[times.len() / 2],
        bytes.len(),
        photo::psnr(&decoded.y, &y),
        photo::fnv1a64(&bytes)
    );
}
