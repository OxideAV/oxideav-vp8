//! Timed inter-stream encode and decode on the deterministic moving clip
//! (or a raw I420 file).
//!
//! `cargo run --release --example inter_timing -- [WIDTH HEIGHT [FRAMES [QI [RUNS [THREADS...]]]]]`
//!
//! Environment:
//! * `VP8_CLIP=<path>` — read the frames from a raw I420 file of the given
//!   size instead of synthesising the clip;
//! * `VP8_ONLY_EFFORT=<name>` — time just that effort level;
//! * `VP8_KEYINT=<n>` — key-frame interval (default: the
//!   `Vp8EncoderConfig` default);
//! * `VP8_AUTO_LF=1` — RD-select the loop-filter level per frame;
//! * `VP8_LF=<level>` — fixed loop-filter level;
//! * `VP8_BITRATE=<bits/s>` — one-pass rate control target at 30 fps;
//! * `VP8_PARTITIONS=<n>` — token partition count;
//! * `VP8_DECODE_ONLY=1` — encode once at the fastest level, then time
//!   only the decode at each thread budget (for profiling the decoder);
//! * `VP8_DUMP=<path>` — write the encoded stream as an IVF file.
//!
//! For every effort level and thread budget it prints encode fps
//! (median of `RUNS`), stream size as kbit/s at 30 fps, luma PSNR of this
//! crate's decode against the source, and a digest of the stream, which
//! must not change with the thread budget (asserted). The decode section
//! prints frames per second at each thread budget and asserts identical
//! pictures.

#[path = "../tests/support/photo.rs"]
mod photo;

#[path = "../tests/support/clip.rs"]
mod clip;

use std::time::Instant;

use oxideav_vp8::{
    AltrefPacketKind, EncodeEffort, I420Frame, Vp8DecoderState, Vp8Encoder, Vp8EncoderConfig,
};

fn env_num(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn median(times: &mut [f64]) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    times[times.len() / 2]
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric argument"))
        .collect();
    let w = args.first().copied().unwrap_or(1280);
    let h = args.get(1).copied().unwrap_or(720);
    let frames = args.get(2).copied().unwrap_or(10);
    let qi = args.get(3).copied().unwrap_or(50) as u8;
    let runs = args.get(4).copied().unwrap_or(3).max(1);
    let thread_budgets: Vec<usize> = if args.len() > 5 {
        args[5..].to_vec()
    } else {
        vec![1]
    };

    let source = match std::env::var("VP8_CLIP") {
        Ok(path) => clip::read_i420_file(&path, w, h, frames),
        Err(_) => clip::moving_clip(w, h, frames),
    };
    let frames = source.len();
    let views: Vec<I420Frame<'_>> = source
        .iter()
        .map(|(y, u, v)| I420Frame::packed(w as u32, h as u32, y, u, v))
        .collect();

    let mut base = Vp8EncoderConfig::default().with_qindex(qi);
    if let Some(n) = env_num("VP8_KEYINT") {
        base = base.with_golden_interval(n as u32);
    }
    if env_num("VP8_AUTO_LF") == Some(1) {
        base = base.with_auto_loop_filter(true);
    }
    if let Some(level) = env_num("VP8_LF") {
        base = base.with_lf_level(level as u8);
    }
    if let Some(bps) = env_num("VP8_BITRATE") {
        base = base.with_target_bitrate_bps(bps as u32);
    }
    if let Some(n) = env_num("VP8_PARTITIONS") {
        base = base.with_partitions(n as u8);
    }
    let decode_only = env_num("VP8_DECODE_ONLY") == Some(1);

    println!("{w}x{h}, {frames} frames, qindex {qi}, {runs} runs per cell (median)");
    println!("effort    threads  enc ms/frame  enc fps    kbit/s  PSNR-Y dB  digest");
    let only = std::env::var("VP8_ONLY_EFFORT")
        .ok()
        .and_then(|name| EncodeEffort::parse(&name));
    let mut last_stream: Vec<Vec<u8>> = Vec::new();
    for effort in EncodeEffort::ALL {
        if decode_only && effort != EncodeEffort::Fastest {
            continue;
        }
        if !decode_only && only.is_some_and(|e| e != effort) {
            continue;
        }
        let mut serial_digest = None;
        for &threads in &thread_budgets {
            let config = base
                .with_effort(effort)
                .with_inter_effort(effort)
                .with_threads(threads);
            let mut times = Vec::with_capacity(runs);
            let mut packets = Vec::new();
            for _ in 0..if decode_only { 1 } else { runs } {
                let mut enc = Vp8Encoder::new(config);
                let t = Instant::now();
                packets = enc.encode_sequence(&views).expect("encode");
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let ms = median(&mut times);

            let mut dec = Vp8DecoderState::new();
            let mut total_bytes = 0usize;
            let mut sse = 0f64;
            let mut digest = 0xCBF2_9CE4_8422_2325u64;
            for p in &packets {
                total_bytes += p.bytes.len();
                digest ^= photo::fnv1a64(&p.bytes);
                digest = digest.wrapping_mul(0x0000_0100_0000_01B3);
                let picture = dec.decode_frame(&p.bytes).expect("self-decode");
                if p.kind != AltrefPacketKind::AltrefUpdate {
                    let index = p.source_index.expect("visible packet") as usize;
                    let psnr = photo::psnr(&picture.y, &source[index].0);
                    sse += 10f64.powf(-psnr / 10.0);
                }
            }
            let psnr = -10.0 * (sse / frames as f64).log10();
            assert_eq!(
                *serial_digest.get_or_insert(digest),
                digest,
                "stream changed with the thread budget"
            );
            println!(
                "{:<9} {threads:>7} {:>13.2} {:>8.2} {:>9.1} {:>10.2}  {digest:016x}",
                effort.name(),
                ms / frames as f64,
                frames as f64 * 1e3 / ms,
                total_bytes as f64 * 8.0 * 30.0 / frames as f64 / 1e3,
                psnr,
            );
            last_stream = packets.into_iter().map(|p| p.bytes).collect();
            if decode_only {
                break;
            }
        }
    }

    if let Ok(path) = std::env::var("VP8_DUMP") {
        let mut header = oxideav_vp8::ivf::IvfHeader::vp8(w as u32, h as u32, 30, 1);
        header.frame_count = last_stream.len() as u32;
        let mut ivf = oxideav_vp8::ivf::write_header(&header);
        for (n, packet) in last_stream.iter().enumerate() {
            oxideav_vp8::ivf::write_frame(&mut ivf, n as u64, packet);
        }
        std::fs::write(&path, ivf).expect("writable dump path");
    }

    // Decode timing on the last stream encoded above.
    println!("decode   threads  ms/packet  packets/s");
    let mut reference: Option<u64> = None;
    for &threads in &thread_budgets {
        let reps = if decode_only { runs.max(5) } else { runs };
        let mut times = Vec::with_capacity(reps);
        let mut digest = 0u64;
        for _ in 0..reps {
            let mut dec = Vp8DecoderState::new().with_threads(threads);
            digest = 0xCBF2_9CE4_8422_2325u64;
            let t = Instant::now();
            for packet in &last_stream {
                let picture = dec.decode_frame(packet).expect("decode");
                digest ^= photo::fnv1a64(&picture.y[..picture.y.len().min(4096)]);
                digest = digest
                    .wrapping_mul(0x0000_0100_0000_01B3)
                    .wrapping_add(picture.y.iter().step_by(97).map(|&b| b as u64).sum::<u64>());
            }
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        assert_eq!(
            *reference.get_or_insert(digest),
            digest,
            "decoded pictures changed with the thread budget"
        );
        let ms = median(&mut times);
        println!(
            "         {threads:>7} {:>10.3} {:>10.1}",
            ms / last_stream.len() as f64,
            last_stream.len() as f64 * 1e3 / ms
        );
    }
}
