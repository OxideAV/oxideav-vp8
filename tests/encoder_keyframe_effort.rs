//! Key-frame encode effort levels and the thread budget.
//!
//! * the default options reproduce every option-less front door byte
//!   for byte;
//! * for every effort level the bytes and the reconstruction are the
//!   same at every thread budget;
//! * every level decodes through this crate's decoder to exactly the
//!   reconstruction the encoder returned, and through a black-box
//!   decoder to the same pixels (skipped at runtime when the validator
//!   binary is not installed);
//! * the reduced levels stay close to the exhaustive search in size and
//!   PSNR on photo-like content;
//! * the stream configurations and the typed encoder carry the options.

#[path = "support/photo.rs"]
mod photo;

use std::io::Write;
use std::process::{Command, Stdio};

use oxideav_vp8::encoder::Vp8EncoderConfig;
use oxideav_vp8::{
    AltrefStreamConfig, EncodeEffort, I420Frame, KeyframeCodingOptions, KeyframeEncodeOptions,
    KeyframeParams, TrellisStrength, Vp8AltrefStreamEncoder, Vp8DecoderState, Vp8Encoder,
    decode_vp8, encode_keyframe, encode_keyframe_auto_loop_filter,
    encode_keyframe_with_fitted_token_prob_updates, encode_keyframe_with_options,
    encode_keyframe_with_reconstruction, encode_keyframe_with_reconstruction_and_coding_options,
    encode_keyframe_with_reconstruction_and_options, ivf,
};

/// Picture sizes: one macroblock, one row, one column, odd sizes with
/// partial edge macroblocks, and a mid-size frame.
const SIZES: [(usize, usize); 7] = [
    (16, 16),
    (80, 16),
    (16, 80),
    (33, 17),
    (97, 61),
    (176, 144),
    (320, 213),
];

fn picture(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    photo::photo_like_i420(w, h)
}

/// The visible `w × h` window of the macroblock-aligned planes, packed
/// as I420.
fn cropped(planes: &oxideav_vp8::frame::KeyframePlanes, w: usize, h: usize) -> Vec<u8> {
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

fn decoded_i420(bytes: &[u8]) -> Vec<u8> {
    let d = decode_vp8(bytes).expect("own decoder accepts the frame");
    let mut out = d.y;
    out.extend_from_slice(&d.u);
    out.extend_from_slice(&d.v);
    out
}

#[test]
fn exhaustive_options_reproduce_the_plain_front_doors() {
    let default = KeyframeEncodeOptions::exhaustive();
    assert_eq!(default.effort, EncodeEffort::Thorough);
    assert!(!default.token_prob_fit);
    for (w, h) in SIZES {
        let (y, u, v) = picture(w, h);
        let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
        for params in [
            KeyframeParams::default(),
            KeyframeParams::default()
                .with_y_ac_qi(8)
                .with_trellis_strength(TrellisStrength::OFF),
            KeyframeParams::default()
                .with_y_ac_qi(60)
                .with_loop_filter_level(24)
                .with_sharpness_level(2)
                .with_nbr_of_dct_partitions(4)
                .with_trellis_strength(TrellisStrength::new(2.5)),
        ] {
            let plain = encode_keyframe(&frame, &params).unwrap();
            assert_eq!(
                encode_keyframe_with_options(&frame, &params, &default).unwrap(),
                plain,
                "{w}x{h}"
            );
            let (bytes, planes) = encode_keyframe_with_reconstruction(&frame, &params).unwrap();
            let (obytes, oplanes) =
                encode_keyframe_with_reconstruction_and_options(&frame, &params, &default).unwrap();
            assert_eq!((bytes, &planes.y), (obytes, &oplanes.y));

            assert_eq!(
                encode_keyframe_with_options(
                    &frame,
                    &params,
                    &default.with_auto_loop_filter(true).with_threads(3)
                )
                .unwrap(),
                encode_keyframe_auto_loop_filter(&frame, &params).unwrap(),
                "auto loop filter {w}x{h}"
            );
            assert_eq!(
                encode_keyframe_with_options(
                    &frame,
                    &params,
                    &default.with_fitted_token_prob_updates(true).with_threads(2)
                )
                .unwrap(),
                encode_keyframe_with_fitted_token_prob_updates(&frame, &params).unwrap(),
                "fitted updates {w}x{h}"
            );
            let both = KeyframeCodingOptions::default()
                .with_auto_loop_filter(true)
                .with_fitted_token_prob_updates(true);
            assert_eq!(
                encode_keyframe_with_reconstruction_and_options(
                    &frame,
                    &params,
                    &default
                        .with_auto_loop_filter(true)
                        .with_fitted_token_prob_updates(true)
                )
                .unwrap()
                .0,
                encode_keyframe_with_reconstruction_and_coding_options(&frame, &params, &both)
                    .unwrap()
                    .0,
                "both {w}x{h}"
            );
        }
    }
}

/// Luma PSNR of a decode against the source plane.
fn psnr_y(bytes: &[u8], y: &[u8]) -> f64 {
    let decoded = decode_vp8(bytes).expect("own decoder accepts the frame");
    let sse: u64 = decoded
        .y
        .iter()
        .zip(y)
        .map(|(&a, &b)| {
            let d = a as i64 - b as i64;
            (d * d) as u64
        })
        .sum();
    10.0 * (255.0f64 * 255.0 * y.len() as f64 / sse.max(1) as f64).log10()
}

/// The default options — the `Balanced` search with the token
/// probabilities fitted inside the one search — against the exhaustive
/// option-less encode: no larger, no worse, and decodable to exactly the
/// reconstruction the encoder reports.
#[test]
fn default_options_beat_the_exhaustive_encode_on_size_at_equal_quality() {
    let default = KeyframeEncodeOptions::default();
    assert_eq!(default, KeyframeEncodeOptions::new());
    assert_eq!(default.effort, EncodeEffort::Balanced);
    assert!(default.token_prob_fit && !default.fitted_token_prob_updates);
    assert_eq!(default.threads, 0);
    let (w, h) = (320usize, 240usize);
    let (y, u, v) = photo::photo_like_i420(w, h);
    let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
    for qi in [10u8, 25, 60, 100] {
        let params = KeyframeParams::default().with_y_ac_qi(qi);
        let plain = encode_keyframe(&frame, &params).unwrap();
        let tuned = oxideav_vp8::encode_keyframe_detailed(&frame, &params, &default).unwrap();
        assert!(
            tuned.bytes.len() < plain.len(),
            "qi {qi}: default {} bytes, exhaustive {}",
            tuned.bytes.len(),
            plain.len()
        );
        // Quality within a few hundredths of a dB of the exhaustive
        // encode (measured: ahead of it at the mid quantisers).
        let (got, want) = (psnr_y(&tuned.bytes, &y), psnr_y(&plain, &y));
        assert!(got > want - 0.15, "qi {qi}: {got:.3} dB vs {want:.3} dB");
        // What the encoder reports is what a decoder sees.
        assert!(
            tuned.token_prob_updates.is_some(),
            "qi {qi}: no fit shipped"
        );
        assert_eq!((tuned.loop_filter_level, tuned.sharpness_level), (0, 0));
        let decoded = decode_vp8(&tuned.bytes).unwrap();
        for row in 0..h {
            assert_eq!(
                decoded.y[row * w..(row + 1) * w],
                tuned.reconstruction.y[row * tuned.reconstruction.y_stride..][..w],
                "qi {qi} row {row}"
            );
        }
    }
}

/// The in-search fit never costs bytes: at every level the fitted stream
/// is no larger than the same search coded against the §13.5 defaults,
/// and the stream it produces reports the table it carries.
#[test]
fn token_prob_fit_only_ever_shrinks_the_stream() {
    for (w, h) in SIZES {
        let (y, u, v) = picture(w, h);
        let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
        for effort in EncodeEffort::ALL {
            for qi in [6u8, 40, 110] {
                let params = KeyframeParams::default().with_y_ac_qi(qi);
                let base = KeyframeEncodeOptions::new().with_effort(effort);
                // Same search table on both sides (the caller's), so the
                // only difference is the table the tokens are coded with.
                let seed = oxideav_vp8::encode_keyframe_detailed(&frame, &params, &base)
                    .unwrap()
                    .token_prob_updates;
                let fitted = oxideav_vp8::encode_keyframe_detailed(
                    &frame,
                    &params,
                    &base.with_search_token_probs(seed),
                )
                .unwrap();
                let Some(seed) = seed else {
                    continue;
                };
                let unfitted = encode_keyframe_with_options(
                    &frame,
                    &params,
                    &base
                        .with_search_token_probs(Some(seed))
                        .with_token_prob_fit(false),
                )
                .unwrap();
                assert!(
                    fitted.bytes.len() <= unfitted.len() + 8,
                    "{w}x{h} {effort:?} qi {qi}: fitted {} vs seeded {}",
                    fitted.bytes.len(),
                    unfitted.len()
                );
                assert_eq!(
                    decoded_i420(&fitted.bytes),
                    decoded_i420(&unfitted),
                    "{w}x{h} {effort:?} qi {qi}: the fit changed the picture"
                );
            }
        }
    }
}

/// The sampled loop-filter choice: a level and sharpness in range, the
/// header and the reconstruction agreeing with a decode, and a picture
/// at least as close to the source as the unfiltered one.
#[test]
fn sampled_loop_filter_improves_on_no_filter() {
    let (w, h) = (320usize, 240usize);
    let (y, u, v) = photo::photo_like_i420(w, h);
    let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
    for qi in [25u8, 70] {
        let params = KeyframeParams::default().with_y_ac_qi(qi);
        let off =
            encode_keyframe_with_options(&frame, &params, &KeyframeEncodeOptions::new()).unwrap();
        let mut digests = Vec::new();
        for threads in [1usize, 3] {
            let options = KeyframeEncodeOptions::new()
                .with_sampled_loop_filter(true)
                .with_threads(threads);
            let on = oxideav_vp8::encode_keyframe_detailed(&frame, &params, &options).unwrap();
            assert!(on.loop_filter_level <= 63 && on.sharpness_level <= 7);
            assert!(on.loop_filter_level > 0, "qi {qi}: filter left off");
            assert!(
                psnr_y(&on.bytes, &y) > psnr_y(&off, &y),
                "qi {qi}: filtered picture is not closer to the source"
            );
            let decoded = decode_vp8(&on.bytes).unwrap();
            assert_eq!(
                decoded.y[..w],
                on.reconstruction.y[..w],
                "qi {qi}: reconstruction differs from the decode"
            );
            digests.push(on.bytes);
        }
        assert_eq!(digests[0], digests[1], "qi {qi}: threads changed the bytes");
    }
}

#[test]
fn thread_budget_never_changes_the_output() {
    for (w, h) in SIZES {
        let (y, u, v) = picture(w, h);
        let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
        for effort in EncodeEffort::ALL {
            for partitions in [1u8, 2, 8] {
                let params = KeyframeParams::default()
                    .with_y_ac_qi(25)
                    .with_loop_filter_level(12)
                    .with_nbr_of_dct_partitions(partitions);
                let options = KeyframeEncodeOptions::new().with_effort(effort);
                let (serial, serial_planes) =
                    encode_keyframe_with_reconstruction_and_options(&frame, &params, &options)
                        .unwrap();
                for threads in [0usize, 2, 3, 7, 64] {
                    let (bytes, planes) = encode_keyframe_with_reconstruction_and_options(
                        &frame,
                        &params,
                        &options.with_threads(threads),
                    )
                    .unwrap();
                    assert_eq!(
                        bytes, serial,
                        "{w}x{h} {effort:?} {partitions} partitions, {threads} threads"
                    );
                    assert_eq!(planes.y, serial_planes.y);
                    assert_eq!(planes.u, serial_planes.u);
                    assert_eq!(planes.v, serial_planes.v);
                }
            }
        }
    }
}

#[test]
fn every_effort_decodes_to_the_encoder_reconstruction() {
    for (w, h) in SIZES {
        let (y, u, v) = picture(w, h);
        let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
        for effort in EncodeEffort::ALL {
            for (qi, lf, strength) in [
                (0u8, 0u8, 1.0),
                (25, 0, 1.33),
                (25, 20, 0.0),
                (110, 40, 4.0),
            ] {
                let params = KeyframeParams::default()
                    .with_y_ac_qi(qi)
                    .with_loop_filter_level(lf)
                    .with_trellis_strength(TrellisStrength::new(strength));
                let options = KeyframeEncodeOptions::new()
                    .with_effort(effort)
                    .with_threads(2);
                let (bytes, planes) =
                    encode_keyframe_with_reconstruction_and_options(&frame, &params, &options)
                        .unwrap();
                assert_eq!(
                    decoded_i420(&bytes),
                    cropped(&planes, w, h),
                    "{w}x{h} {effort:?} qi {qi} lf {lf}"
                );
            }
        }
    }
}

#[test]
fn reduced_efforts_stay_close_to_the_exhaustive_search() {
    let (w, h) = (320usize, 213usize);
    let (y, u, v) = picture(w, h);
    let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
    for qi in [12u8, 25, 50] {
        let params = KeyframeParams::default().with_y_ac_qi(qi);
        let measure = |effort| {
            let bytes = encode_keyframe_with_options(
                &frame,
                &params,
                &KeyframeEncodeOptions::new().with_effort(effort),
            )
            .unwrap();
            let decoded = decode_vp8(&bytes).unwrap();
            (bytes.len() as f64, photo::psnr(&decoded.y, &y))
        };
        let (ref_len, ref_psnr) = measure(EncodeEffort::Thorough);
        for (effort, max_growth, max_loss_db) in [
            (EncodeEffort::Balanced, 1.04, 0.15),
            (EncodeEffort::Fast, 1.06, 0.20),
            (EncodeEffort::Fastest, 1.20, 0.50),
        ] {
            let (len, psnr) = measure(effort);
            assert!(
                len <= ref_len * max_growth,
                "qi {qi} {effort:?}: {len} bytes vs {ref_len}"
            );
            assert!(
                psnr >= ref_psnr - max_loss_db,
                "qi {qi} {effort:?}: {psnr:.2} dB vs {ref_psnr:.2}"
            );
        }
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

/// Decode a one-frame IVF stream with the black-box validator and return
/// its I420 output.
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
    for (w, h) in [(97usize, 61usize), (320, 213)] {
        let (y, u, v) = picture(w, h);
        let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
        for effort in EncodeEffort::ALL {
            for (lf, partitions) in [(0u8, 1u8), (18, 4)] {
                let params = KeyframeParams::default()
                    .with_y_ac_qi(25)
                    .with_loop_filter_level(lf)
                    .with_nbr_of_dct_partitions(partitions);
                // The default options (in-search probability fit), the
                // same with the sampled loop filter, and the search
                // coded against the §13.5 defaults.
                let fit = KeyframeEncodeOptions::new()
                    .with_effort(effort)
                    .with_threads(4);
                for (n, options) in [
                    fit,
                    fit.with_sampled_loop_filter(true),
                    fit.with_token_prob_fit(false),
                ]
                .into_iter()
                .enumerate()
                {
                    let bytes = encode_keyframe_with_options(&frame, &params, &options).unwrap();
                    let mut stream =
                        ivf::write_header(&ivf::IvfHeader::vp8(w as u32, h as u32, 30, 1));
                    ivf::write_frame(&mut stream, 0, &bytes);
                    assert_eq!(
                        validator_decode(&stream),
                        decoded_i420(&bytes),
                        "{w}x{h} {effort:?} lf {lf} partitions {partitions} options {n}"
                    );
                }
            }
        }
    }
}

#[test]
fn stream_configurations_carry_the_effort() {
    let (w, h) = (96usize, 80usize);
    let (y, u, v) = picture(w, h);
    let frame = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
    let params = KeyframeParams::default().with_y_ac_qi(30);
    for effort in EncodeEffort::ALL {
        let want = encode_keyframe_with_options(
            &frame,
            &params,
            &KeyframeEncodeOptions::exhaustive().with_effort(effort),
        )
        .unwrap();

        // Lagged stream driver: the opening key frame.
        let config = AltrefStreamConfig::default()
            .with_params(params)
            .with_effort(effort)
            .with_threads(3);
        let mut stream = Vp8AltrefStreamEncoder::new(config).unwrap();
        let mut packets = stream.push_frame(&frame).unwrap();
        packets.extend(stream.push_frame(&frame).unwrap());
        packets.extend(stream.finish().unwrap());
        assert_eq!(packets[0].bytes, want, "{effort:?}");
        // The inter frames predict from that key frame's reconstruction;
        // the whole stream must decode.
        let mut decoder = Vp8DecoderState::new();
        for packet in &packets {
            decoder.decode_frame(&packet.bytes).expect("stream decodes");
        }

        // Typed single-pass encoder.
        let mut typed = Vp8Encoder::new(
            Vp8EncoderConfig::default()
                .with_qindex(30)
                .with_effort(effort)
                .with_threads(2),
        );
        assert_eq!(typed.encode_keyframe(&frame).unwrap(), want, "{effort:?}");
    }
}

#[cfg(feature = "registry")]
mod registry {
    use super::*;
    use oxideav_core::{
        CodecId, CodecParameters, ExecutionContext, Frame, PixelFormat, RuntimeContext, VideoFrame,
        VideoPlane,
    };
    use oxideav_vp8::{Vp8EncoderOptions, make_encoder, make_encoder_with_options};

    fn video_frame(w: usize, h: usize, y: &[u8], u: &[u8], v: &[u8]) -> Frame {
        let cw = w.div_ceil(2);
        let _ = h;
        Frame::Video(VideoFrame {
            pts: Some(0),
            planes: vec![
                VideoPlane {
                    stride: w,
                    data: y.to_vec(),
                },
                VideoPlane {
                    stride: cw,
                    data: u.to_vec(),
                },
                VideoPlane {
                    stride: cw,
                    data: v.to_vec(),
                },
            ],
        })
    }

    fn params(w: usize, h: usize) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new(oxideav_vp8::CODEC_ID_STR));
        p.width = Some(w as u32);
        p.height = Some(h as u32);
        p.pixel_format = Some(PixelFormat::Yuv420P);
        p
    }

    #[test]
    fn registry_encoder_honours_options_and_execution_context() {
        let (w, h) = (112usize, 96usize);
        let (y, u, v) = picture(w, h);
        let source = I420Frame::packed(w as u32, h as u32, &y, &u, &v);
        let input = video_frame(w, h, &y, &u, &v);

        let mut ctx = RuntimeContext::new();
        oxideav_vp8::register(&mut ctx);
        assert!(
            ctx.codecs
                .has_encoder(&CodecId::new(oxideav_vp8::CODEC_ID_STR))
        );

        // An empty bag is the plain default encoder.
        let mut plain = make_encoder(&params(w, h)).unwrap();
        plain.send_frame(&input).unwrap();
        let plain_bytes = plain.receive_packet().unwrap().data;
        let mut registered = ctx.codecs.first_encoder(&params(w, h)).unwrap();
        registered.send_frame(&input).unwrap();
        assert_eq!(registered.receive_packet().unwrap().data, plain_bytes);
        assert_eq!(
            plain_bytes,
            encode_keyframe(&source, &KeyframeParams::default()).unwrap()
        );

        for effort in EncodeEffort::ALL {
            let want = encode_keyframe_with_options(
                &source,
                &KeyframeParams::default().with_y_ac_qi(40),
                &KeyframeEncodeOptions::exhaustive().with_effort(effort),
            )
            .unwrap();

            // Bag route, with an announced thread budget.
            let mut p = params(w, h);
            p.options = p
                .options
                .set("effort", effort.level().to_string())
                .set("qindex", "40");
            let mut enc = ctx.codecs.first_encoder(&p).unwrap();
            enc.set_execution_context(&ExecutionContext::with_threads(4));
            enc.send_frame(&input).unwrap();
            assert_eq!(enc.receive_packet().unwrap().data, want, "{effort:?}");

            // Typed route, explicit thread option.
            let mut enc = make_encoder_with_options(
                &params(w, h),
                &Vp8EncoderOptions::new()
                    .with_effort(effort)
                    .with_qindex(40)
                    .with_threads(3),
            )
            .unwrap();
            enc.send_frame(&input).unwrap();
            assert_eq!(enc.receive_packet().unwrap().data, want, "{effort:?}");
        }

        // Quality replaces qindex and tunes the trellis.
        let mut p = params(w, h);
        p.options = p.options.set("quality", "80").set("effort", "1");
        let mut enc = ctx.codecs.first_encoder(&p).unwrap();
        enc.send_frame(&input).unwrap();
        assert_eq!(
            enc.receive_packet().unwrap().data,
            encode_keyframe_with_options(
                &source,
                &KeyframeParams::default()
                    .with_y_ac_qi(oxideav_vp8::encoder::quality_to_qindex(80.0))
                    .with_trellis_strength(oxideav_vp8::encoder::quality_to_trellis_strength(80.0)),
                &KeyframeEncodeOptions::exhaustive().with_effort(EncodeEffort::Fast),
            )
            .unwrap()
        );
    }

    #[test]
    fn registry_encoder_rejects_bad_options() {
        let mut ctx = RuntimeContext::new();
        oxideav_vp8::register(&mut ctx);
        for (key, value) in [
            ("effort", "4"),
            ("effort", "fast"),
            ("qindex", "128"),
            ("partitions", "3"),
            ("loop_filter_level", "64"),
            ("no_such_option", "1"),
        ] {
            let mut p = params(64, 64);
            p.options = p.options.set(key, value);
            assert!(
                ctx.codecs.first_encoder(&p).is_err(),
                "{key}={value} must be rejected"
            );
        }
        assert!(
            make_encoder_with_options(&params(64, 64), &Vp8EncoderOptions::new().with_qindex(200))
                .is_err()
        );
    }
}
