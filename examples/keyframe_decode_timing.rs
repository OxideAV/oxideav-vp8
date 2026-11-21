//! Timed key-frame decode of the deterministic photo-like picture.
//!
//! `cargo run --release --example keyframe_decode_timing -- [WIDTH HEIGHT [QI [RUNS [THREADS...]]]]`
//!
//! Encodes the picture once (default options plus the sampled loop
//! filter, so the decode includes the §15 pass), then for each thread
//! budget (default `1 2 4`) prints the minimum and median wall time of
//! `decode_vp8_with_options` over `RUNS` decodes and asserts that the
//! picture equals the serial one.

#[path = "../tests/support/photo.rs"]
mod photo;

use std::time::Instant;

use oxideav_vp8::{
    I420Frame, KeyframeEncodeOptions, KeyframeParams, Vp8DecodeOptions, decode_vp8_with_options,
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
    let runs = args.get(3).copied().unwrap_or(9).max(1);
    let thread_budgets: Vec<usize> = if args.len() > 4 {
        args[4..].to_vec()
    } else {
        vec![1, 2, 4]
    };
    let partitions = std::env::var("VP8_PARTITIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1u8);

    let (y, u, v) = photo::photo_like_i420(w, h);
    let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
    let params = KeyframeParams::default()
        .with_y_ac_qi(qi)
        .with_nbr_of_dct_partitions(partitions);
    let options = KeyframeEncodeOptions::new().with_sampled_loop_filter(true);
    let bytes = encode_keyframe_with_options(&frame, &params, &options).expect("encode");
    println!(
        "{w}x{h}, y_ac_qi {qi}, {} bytes, {partitions} token partition(s), {runs} runs per cell",
        bytes.len()
    );
    println!("threads   min ms  median ms");
    let mut serial = None;
    for &threads in &thread_budgets {
        let options = Vp8DecodeOptions::new().with_threads(threads);
        let mut times = Vec::with_capacity(runs);
        let mut picture = None;
        for _ in 0..runs {
            let t = Instant::now();
            picture = Some(decode_vp8_with_options(&bytes, &options).expect("decode"));
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let picture = picture.expect("at least one run");
        assert_eq!(
            *serial.get_or_insert_with(|| picture.clone()),
            picture,
            "picture changed with the thread budget"
        );
        println!(
            "{threads:>7} {:>8.2} {:>10.2}",
            times[0],
            times[times.len() / 2]
        );
    }
}
