//! Round-387 coverage of the lagged framework encoder adapter behind
//! [`oxideav_vp8::make_encoder_with_config`] — the `Box<dyn Encoder>`
//! face of the auto-altref stream driver.
//!
//! Before this round the config path re-keyed every `send_frame`; now
//! it follows the standard lagged `Encoder` protocol. What this pins:
//!
//!   1. **lag semantics** — `receive_packet` surfaces `NeedMore` while
//!      the lookahead group fills; a completed group emits its packets;
//!      `flush` drains the tail group and arms `Eof`;
//!   2. **packet metadata** — visible packets carry their source
//!      frame's `pts` (in order) and only key frames set
//!      `flags.keyframe`; invisible anchor packets carry `pts = None`;
//!   3. **stream shape** — more packets than source frames (one
//!      invisible anchor per completed multi-frame group);
//!   4. **full framework loop** — every emitted packet decodes through
//!      the registry-side [`Vp8Decoder`] in emission order;
//!   5. **zero-lag mode** — `alt_ref_interval = 0` restores immediate
//!      per-frame packet availability (K/P streaming, no anchors);
//!   6. the zero-latency still-image doors (`make_encoder`,
//!      `make_encoder_with_quality`) keep their one-keyframe-per-frame
//!      immediate contract.
//!
//! Black-box: encoder output feeds the crate's own registry decoder.

#![cfg(feature = "registry")]

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Decoder as _, Encoder, Error, Frame, MediaType, Packet, PixelFormat,
    VideoFrame,
};
use oxideav_vp8::{
    make_encoder, make_encoder_with_config, make_encoder_with_quality, Vp8Decoder, Vp8EncoderConfig,
};

const W: u32 = 64;
const H: u32 = 64;

fn vp8_params() -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new("vp8"));
    params.media_type = MediaType::Video;
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params
}

/// Textured drifting source frame `f` with a pts of `f * 3000`.
fn video_frame(f: usize) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let (uvw, uvh) = (w / 2, h / 2);
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let sc = c + 2 * f;
            y[r * w + c] = (60 + ((r * 3 + sc * 2) % 130) as i32 + (((sc * 11) % 23) as i32 - 11))
                .clamp(0, 255) as u8;
        }
    }
    let u = vec![110u8; uvw * uvh];
    let v = vec![135u8; uvw * uvh];
    VideoFrame {
        pts: Some(f as i64 * 3000),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: uvw,
                data: u,
            },
            VideoPlane {
                stride: uvw,
                data: v,
            },
        ],
    }
}

/// Drain every currently-available packet.
fn drain(enc: &mut Box<dyn Encoder>) -> Vec<Packet> {
    let mut out = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => out.push(p),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }
    out
}

#[test]
fn config_path_lags_groups_and_flushes_the_tail() {
    let config = Vp8EncoderConfig {
        qindex: 44,
        alt_ref_interval: 4,
        lookahead_window: 8,
        golden_interval: 0, // key only frame 0
        ..Vp8EncoderConfig::default()
    };
    let mut enc = make_encoder_with_config(&vp8_params(), config).expect("build");

    let mut packets: Vec<Packet> = Vec::new();
    for f in 0..6 {
        enc.send_frame(&Frame::Video(video_frame(f))).expect("send");
        let got = drain(&mut enc);
        if f < 3 {
            // The first group (4 frames) is still filling.
            assert!(
                got.is_empty(),
                "frame {f}: no packets may surface while the group fills, got {}",
                got.len()
            );
            assert!(
                matches!(enc.receive_packet(), Err(Error::NeedMore)),
                "mid-group receive_packet must surface NeedMore"
            );
        } else if f == 3 {
            // Group complete: K + invisible anchor + 3 P = 5 packets.
            assert_eq!(got.len(), 5, "completed group must emit K + anchor + 3P");
        }
        packets.extend(got);
    }

    // Two tail frames are still buffered; flush drains them.
    enc.flush().expect("flush");
    let tail = drain(&mut enc);
    assert!(!tail.is_empty(), "flush must drain the buffered tail group");
    packets.extend(tail);
    assert!(
        matches!(enc.receive_packet(), Err(Error::Eof)),
        "post-flush drained receive_packet must surface Eof"
    );

    // Stream shape: 6 sources → 1 K + 5 P + 2 anchors (two multi-frame
    // groups: 4 + 2) = 8 packets.
    assert_eq!(packets.len(), 8, "6 sources must emit 8 packets");

    // Metadata: visible packets carry the source pts in order; anchors
    // carry none; exactly one keyframe (the first packet).
    let visible_pts: Vec<i64> = packets.iter().filter_map(|p| p.pts).collect();
    assert_eq!(
        visible_pts,
        (0..6).map(|f| f * 3000).collect::<Vec<i64>>(),
        "visible pts must map 1:1 onto source pts in order"
    );
    let anchors = packets.iter().filter(|p| p.pts.is_none()).count();
    assert_eq!(anchors, 2, "one invisible anchor per multi-frame group");
    assert!(packets[0].flags.keyframe, "first packet is the key frame");
    assert_eq!(
        packets.iter().filter(|p| p.flags.keyframe).count(),
        1,
        "only the key frame sets flags.keyframe"
    );

    // Full framework decode loop.
    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    for (i, p) in packets.iter().enumerate() {
        dec.send_packet(p).expect("send_packet");
        let frame = dec
            .receive_frame()
            .unwrap_or_else(|e| panic!("packet {i} must decode: {e:?}"));
        match frame {
            Frame::Video(v) => {
                assert_eq!(v.planes.len(), 3, "packet {i}: I420 output");
            }
            _ => panic!("packet {i}: video frame expected"),
        }
    }
}

#[test]
fn zero_alt_ref_interval_restores_immediate_packets() {
    let config = Vp8EncoderConfig {
        qindex: 44,
        alt_ref_interval: 0, // no anchors, group size 1 — zero lag
        golden_interval: 0,
        ..Vp8EncoderConfig::default()
    };
    let mut enc = make_encoder_with_config(&vp8_params(), config).expect("build");

    let mut sizes = Vec::new();
    for f in 0..4 {
        enc.send_frame(&Frame::Video(video_frame(f))).expect("send");
        let got = drain(&mut enc);
        assert_eq!(
            got.len(),
            1,
            "zero-lag mode must emit exactly one packet per frame"
        );
        assert_eq!(got[0].pts, Some(f as i64 * 3000));
        assert_eq!(got[0].flags.keyframe, f == 0, "K then P ladder");
        sizes.push(got[0].data.len());
    }
    // P-frames actually exploit inter prediction: smaller than the key.
    assert!(
        sizes[1] < sizes[0] && sizes[2] < sizes[0] && sizes[3] < sizes[0],
        "P-frames must undercut the key frame ({sizes:?})"
    );

    enc.flush().expect("flush");
    assert!(
        matches!(enc.receive_packet(), Err(Error::Eof)),
        "nothing left after flush in zero-lag mode"
    );
}

#[test]
fn send_frame_after_flush_is_rejected() {
    let mut enc = make_encoder_with_config(
        &vp8_params(),
        Vp8EncoderConfig {
            qindex: 44,
            ..Vp8EncoderConfig::default()
        },
    )
    .expect("build");
    enc.send_frame(&Frame::Video(video_frame(0))).expect("send");
    enc.flush().expect("flush");
    assert!(
        enc.send_frame(&Frame::Video(video_frame(1))).is_err(),
        "send_frame after flush must be rejected"
    );
}

#[test]
fn still_image_doors_keep_the_immediate_keyframe_contract() {
    // make_encoder — one keyframe per send_frame, available at once.
    let mut enc = make_encoder(&vp8_params()).expect("make_encoder");
    enc.send_frame(&Frame::Video(video_frame(0))).expect("send");
    let pkt = enc.receive_packet().expect("immediate packet");
    assert!(pkt.flags.keyframe);
    assert_eq!(pkt.pts, Some(0));

    // make_encoder_with_quality — same contract.
    let mut enc = make_encoder_with_quality(&vp8_params(), 75.0).expect("with_quality");
    enc.send_frame(&Frame::Video(video_frame(1))).expect("send");
    enc.send_frame(&Frame::Video(video_frame(2))).expect("send");
    let a = enc.receive_packet().expect("immediate packet 1");
    let b = enc.receive_packet().expect("immediate packet 2");
    assert!(a.flags.keyframe && b.flags.keyframe, "every frame re-keys");
}
