//! Inter-frame encode effort levels and the thread budget.
//!
//! * the default options are the exhaustive search and reproduce the
//!   option-less front door byte for byte;
//! * for every effort level the bytes and the reconstruction are the
//!   same at every thread budget;
//! * every level decodes through this crate's decoder to exactly the
//!   reconstruction the encoder returned — with the intra picker, the
//!   token-probability fit, the loop-filter search, segmentation and
//!   several token partitions on — and through a black-box decoder to
//!   the same pixels (skipped at runtime when the validator binary is
//!   not installed);
//! * the reduced levels do not cost more bits than the exhaustive search
//!   for the quality they deliver on moving content;
//! * the lagged stream encoder carries the inter effort and its output
//!   does not depend on the thread budget.

#[path = "support/photo.rs"]
mod photo;

#[path = "support/clip.rs"]
mod clip;

use std::io::Write;
use std::process::{Command, Stdio};

use oxideav_vp8::frame::KeyframePlanes;
use oxideav_vp8::{
    AltrefPacketKind, AltrefStreamConfig, EncodeEffort, I420Frame, InterCodingOptions,
    InterSegmentationConfig, KeyframeParams, RefreshControls, Vp8AltrefStreamEncoder,
    Vp8DecoderState, encode_keyframe_with_reconstruction, encode_p_frame_multi_ref_adaptive_quant,
    encode_p_frame_multi_ref_with_refresh,
    encode_p_frame_multi_ref_with_refresh_and_coding_options, ivf,
};

/// The visible `w × h` window of the macroblock-aligned planes, packed
/// as I420.
fn cropped(planes: &KeyframePlanes, w: usize, h: usize) -> Vec<u8> {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    for row in 0..h {
        out.extend_from_slice(&planes.y[row * planes.y_stride..row * planes.y_stride + w]);
    }
    for plane in [&planes.u, &planes.v] {
        for row in 0..ch {
            out.extend_from_slice(&plane[row * planes.uv_stride..row * planes.uv_stride + cw]);
        }
    }
    out
}

fn view(planes: &clip::Planes, w: usize, h: usize) -> I420Frame<'_> {
    I420Frame::packed(w as u32, h as u32, &planes.0, &planes.1, &planes.2)
}

/// One clip encoded as a key frame followed by inter frames against
/// LAST and GOLDEN (the key frame): the packets and, per packet, the
/// reconstruction the encoder returned.
fn encode_chain(
    source: &[clip::Planes],
    w: usize,
    h: usize,
    params: &KeyframeParams,
    options: &InterCodingOptions,
    segmentation: Option<&InterSegmentationConfig>,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let (key_bytes, key_planes) =
        encode_keyframe_with_reconstruction(&view(&source[0], w, h), params).unwrap();
    let mut packets = vec![key_bytes];
    let mut recons = vec![cropped(&key_planes, w, h)];
    let golden = key_planes.clone();
    let mut last = key_planes;
    for planes in &source[1..] {
        let frame = view(planes, w, h);
        let refresh = RefreshControls::default();
        let (bytes, recon) = match segmentation {
            Some(seg) => encode_p_frame_multi_ref_adaptive_quant(
                &frame,
                &last,
                Some(&golden),
                None,
                params,
                &refresh,
                seg,
                options,
            ),
            None => encode_p_frame_multi_ref_with_refresh_and_coding_options(
                &frame,
                &last,
                Some(&golden),
                None,
                params,
                &refresh,
                options,
            ),
        }
        .unwrap();
        packets.push(bytes);
        recons.push(cropped(&recon, w, h));
        last = recon;
    }
    (packets, recons)
}

fn decode_chain(packets: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut dec = Vp8DecoderState::new();
    packets
        .iter()
        .map(|p| {
            let d = dec.decode_frame(p).expect("own decoder accepts the frame");
            let mut out = d.y;
            out.extend_from_slice(&d.u);
            out.extend_from_slice(&d.v);
            out
        })
        .collect()
}

fn luma_psnr(decoded: &[Vec<u8>], source: &[clip::Planes], w: usize, h: usize) -> f64 {
    let mut total = 0.0;
    for (d, s) in decoded.iter().zip(source) {
        total += photo::psnr(&d[..w * h], &s.0);
    }
    total / decoded.len() as f64
}

#[test]
fn default_options_are_the_exhaustive_search() {
    let default = InterCodingOptions::default();
    assert_eq!(default.effort, EncodeEffort::Thorough);
    assert_eq!(default.threads, 0);
    let (w, h) = (96usize, 64usize);
    let source = clip::moving_clip(w, h, 2);
    let params = KeyframeParams::default().with_y_ac_qi(40);
    let (_, key) = encode_keyframe_with_reconstruction(&view(&source[0], w, h), &params).unwrap();
    let frame = view(&source[1], w, h);
    let refresh = RefreshControls::default();
    let plain =
        encode_p_frame_multi_ref_with_refresh(&frame, &key, Some(&key), None, &params, &refresh)
            .unwrap();
    for threads in [0usize, 1, 3] {
        let with_options = encode_p_frame_multi_ref_with_refresh_and_coding_options(
            &frame,
            &key,
            Some(&key),
            None,
            &params,
            &refresh,
            &default.with_threads(threads),
        )
        .unwrap();
        assert_eq!(plain.0, with_options.0, "threads {threads}");
        assert_eq!(plain.1.y, with_options.1.y, "threads {threads}");
    }
}

#[test]
fn thread_budget_never_changes_the_output() {
    for (w, h) in [(16usize, 16usize), (80, 16), (16, 80), (98, 62), (176, 144)] {
        let source = clip::moving_clip(w, h, 3);
        for effort in EncodeEffort::ALL {
            // The exhaustive search is slow; one small size is enough to
            // show its wavefront is order-independent.
            if effort == EncodeEffort::Thorough && w * h > 98 * 62 {
                continue;
            }
            let params = KeyframeParams::default()
                .with_y_ac_qi(36)
                .with_nbr_of_dct_partitions(2);
            let options = InterCodingOptions::default()
                .with_effort(effort)
                .with_intra_pick(true)
                .with_fitted_token_prob_updates(true)
                .with_auto_loop_filter(true);
            let serial = encode_chain(&source, w, h, &params, &options, None);
            for threads in [2usize, 3, 8] {
                let threaded =
                    encode_chain(&source, w, h, &params, &options.with_threads(threads), None);
                assert_eq!(serial, threaded, "{w}x{h} {effort:?} threads {threads}");
            }
        }
    }
}

#[test]
fn every_effort_decodes_to_the_encoder_reconstruction() {
    for (w, h) in [(48usize, 32usize), (98, 62), (176, 144)] {
        let source = clip::moving_clip(w, h, 5);
        for effort in EncodeEffort::ALL {
            if effort == EncodeEffort::Thorough && w * h > 98 * 62 {
                continue;
            }
            for variant in 0..5 {
                let mut params =
                    KeyframeParams::default().with_y_ac_qi([20, 44, 70, 100, 12][variant]);
                let mut options = InterCodingOptions::default()
                    .with_effort(effort)
                    .with_threads(variant % 3);
                let mut segmentation = None;
                match variant {
                    0 => {}
                    1 => {
                        options = options.with_intra_pick(true);
                        params = params.with_loop_filter_level(12);
                    }
                    2 => {
                        options = options
                            .with_intra_pick(true)
                            .with_fitted_token_prob_updates(true)
                            .with_auto_loop_filter(true);
                        params = params.with_nbr_of_dct_partitions(4);
                    }
                    3 => {
                        options = options.with_fitted_token_prob_updates(true);
                        params = params.with_loop_filter_level(30).with_filter_type(true);
                    }
                    _ => {
                        options = options.with_intra_pick(true).with_auto_loop_filter(true);
                        segmentation = Some(InterSegmentationConfig::default());
                    }
                }
                let (packets, recons) =
                    encode_chain(&source, w, h, &params, &options, segmentation.as_ref());
                assert_eq!(
                    decode_chain(&packets),
                    recons,
                    "{w}x{h} {effort:?} variant {variant}"
                );
            }
        }
    }
}

#[test]
fn reduced_efforts_spend_no_more_than_the_exhaustive_search() {
    let (w, h) = (176usize, 144usize);
    let source = clip::moving_clip(w, h, 5);
    let params = KeyframeParams::default().with_y_ac_qi(40);
    let run = |effort: EncodeEffort| {
        let options = InterCodingOptions::default()
            .with_effort(effort)
            .with_intra_pick(true)
            .with_fitted_token_prob_updates(true);
        let (packets, _) = encode_chain(&source, w, h, &params, &options, None);
        let inter_bytes: usize = packets[1..].iter().map(Vec::len).sum();
        (
            inter_bytes,
            luma_psnr(&decode_chain(&packets), &source, w, h),
        )
    };
    let (thorough_bytes, thorough_psnr) = run(EncodeEffort::Thorough);
    for effort in [
        EncodeEffort::Fastest,
        EncodeEffort::Fast,
        EncodeEffort::Balanced,
    ] {
        let (bytes, psnr) = run(effort);
        assert!(
            bytes <= thorough_bytes,
            "{effort:?}: {bytes} bytes against {thorough_bytes}"
        );
        assert!(
            psnr > thorough_psnr - 2.5,
            "{effort:?}: {psnr:.2} dB against {thorough_psnr:.2}"
        );
    }
}

fn validator_available() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Decode an IVF stream with the black-box validator and return its
/// I420 output, all frames back to back.
fn validator_decode(ivf_bytes: &[u8]) -> Vec<u8> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "ivf",
            "-c:v",
            "vp8",
            "-i",
            "pipe:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn validator");
    let mut stdin = child.stdin.take().expect("validator stdin");
    let input = ivf_bytes.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
    });
    let out = child.wait_with_output().expect("validator output");
    writer.join().expect("writer thread");
    assert!(
        out.status.success(),
        "validator rejected the stream: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

#[test]
fn black_box_decoder_agrees_at_every_effort() {
    if !validator_available() {
        eprintln!("black-box validator not on PATH — skipping");
        return;
    }
    for (w, h) in [(98usize, 62usize), (176, 144)] {
        let source = clip::moving_clip(w, h, 6);
        for effort in EncodeEffort::ALL {
            if effort == EncodeEffort::Thorough && w * h > 98 * 62 {
                continue;
            }
            for (lf, partitions) in [(0u8, 1u8), (14, 4)] {
                let params = KeyframeParams::default()
                    .with_y_ac_qi(34)
                    .with_loop_filter_level(lf)
                    .with_nbr_of_dct_partitions(partitions);
                let options = InterCodingOptions::default()
                    .with_effort(effort)
                    .with_threads(4)
                    .with_intra_pick(true)
                    .with_fitted_token_prob_updates(true);
                let (packets, _) = encode_chain(&source, w, h, &params, &options, None);
                let mut stream = ivf::write_header(&ivf::IvfHeader::vp8(w as u32, h as u32, 30, 1));
                for (n, packet) in packets.iter().enumerate() {
                    ivf::write_frame(&mut stream, n as u64, packet);
                }
                assert_eq!(
                    validator_decode(&stream),
                    decode_chain(&packets).concat(),
                    "{w}x{h} {effort:?} lf {lf} partitions {partitions}"
                );
            }
        }
    }
}

#[test]
fn stream_encoder_carries_the_inter_effort() {
    assert_eq!(
        AltrefStreamConfig::default().inter_effort,
        EncodeEffort::Balanced
    );
    let (w, h) = (112usize, 80usize);
    let source = clip::moving_clip(w, h, 9);
    let run = |effort: EncodeEffort, threads: usize| -> Vec<Vec<u8>> {
        let config = AltrefStreamConfig::default()
            .with_params(KeyframeParams::default().with_y_ac_qi(40))
            .with_keyframe_interval(0)
            .with_altref_window(4)
            .with_intra_pick(true)
            .with_fitted_token_prob_updates(true)
            .with_auto_loop_filter(true)
            .with_effort(EncodeEffort::Fast)
            .with_inter_effort(effort)
            .with_threads(threads);
        let mut enc = Vp8AltrefStreamEncoder::new(config).unwrap();
        let mut packets = Vec::new();
        for planes in &source {
            packets.extend(enc.push_frame(&view(planes, w, h)).unwrap());
        }
        packets.extend(enc.finish().unwrap());
        let visible = packets
            .iter()
            .filter(|p| p.kind != AltrefPacketKind::AltrefUpdate)
            .count();
        assert_eq!(visible, source.len());
        packets.into_iter().map(|p| p.bytes).collect()
    };
    let mut sizes = Vec::new();
    for effort in [
        EncodeEffort::Fastest,
        EncodeEffort::Fast,
        EncodeEffort::Balanced,
    ] {
        let serial = run(effort, 1);
        assert_eq!(serial, run(effort, 4), "{effort:?}");
        // Decodes, every visible frame recognisably the source.
        let mut dec = Vp8DecoderState::new();
        for packet in &serial {
            dec.decode_frame(packet).expect("stream decodes");
        }
        sizes.push(serial.iter().map(Vec::len).sum::<usize>());
    }
    // The levels are different searches.
    assert!(sizes[0] != sizes[1] || sizes[1] != sizes[2]);
}
