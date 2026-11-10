//! Timed keyframe encode on the deterministic photo-like picture.
//!
//! `cargo run --release --example keyframe_timing -- [WIDTH HEIGHT [QI [RUNS [THREADS...]]]]`
//!
//! For every effort level and each requested thread budget (default
//! `1 2 4`) prints the minimum and median wall time over `RUNS` encodes,
//! the output size, the luma PSNR of this crate's own decode, and an
//! FNV-1a digest of the bytes. The digest must not change with the
//! thread budget; the example asserts it. Set `VP8_ONLY_EFFORT` to a
//! level name to time just that level (handy under a profiler).

#[path = "../tests/support/photo.rs"]
mod photo;

use std::time::Instant;

use oxideav_vp8::encoder::quality_to_trellis_strength;
use oxideav_vp8::{
    EncodeEffort, I420Frame, KeyframeEncodeOptions, KeyframeParams, decode_vp8,
    encode_keyframe_with_options,
};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric argument"))
        .collect();
    let w = args.first().copied().unwrap_or(800);
    let h = args.get(1).copied().unwrap_or(533);
    let qi = args.get(2).copied().unwrap_or(25) as u8;
    let runs = args.get(3).copied().unwrap_or(5).max(1);
    let thread_budgets: Vec<usize> = if args.len() > 4 {
        args[4..].to_vec()
    } else {
        vec![1, 2, 4]
    };

    let (y, u, v) = photo::photo_like_i420(w, h);
    let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
    let params = KeyframeParams::default()
        .with_y_ac_qi(qi)
        .with_trellis_strength(quality_to_trellis_strength(80.0));

    println!("{w}x{h}, y_ac_qi {qi}, {runs} runs per cell");
    println!("effort    threads   min ms  median ms     bytes  PSNR-Y dB  digest");
    let only = std::env::var("VP8_ONLY_EFFORT")
        .ok()
        .and_then(|name| EncodeEffort::parse(&name));
    for effort in EncodeEffort::ALL.into_iter().rev() {
        if only.is_some_and(|e| e != effort) {
            continue;
        }
        let mut serial_digest = None;
        for &threads in &thread_budgets {
            let options = KeyframeEncodeOptions::new()
                .with_effort(effort)
                .with_threads(threads);
            let mut times = Vec::with_capacity(runs);
            let mut bytes = Vec::new();
            for _ in 0..runs {
                let t = Instant::now();
                bytes = encode_keyframe_with_options(&frame, &params, &options).expect("encode");
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let decoded = decode_vp8(&bytes).expect("decode");
            let digest = photo::fnv1a64(&bytes);
            assert_eq!(
                *serial_digest.get_or_insert(digest),
                digest,
                "output changed with the thread budget"
            );
            println!(
                "{:<9} {threads:>7} {:>8.2} {:>10.2} {:>9} {:>10.2}  {digest:016x}",
                effort.name(),
                times[0],
                times[times.len() / 2],
                bytes.len(),
                photo::psnr(&decoded.y, &y),
            );
        }
    }
}
