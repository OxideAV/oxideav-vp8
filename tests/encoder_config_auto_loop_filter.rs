//! `Vp8EncoderConfig::auto_loop_filter` wired through the typed
//! direct-API handle (`Vp8Encoder`) and the framework factory
//! (`make_encoder_with_config`) — round 373.
//!
//! Both honour the config's §9.4 `lf_level` and the `auto_loop_filter`
//! RD-selection switch; when the switch is set the §9.4 level / sharpness
//! are chosen per frame and `lf_level` is ignored. These tests confirm the
//! config flows through (output decodes; the auto path engages the filter
//! on a blocky source).

use oxideav_vp8::{decode_vp8, I420Frame, Vp8Encoder, Vp8EncoderConfig};

/// Blocky 80×64 I420 source matching the proven engagement case in
/// `tests/encoder_auto_loop_filter.rs` (per-MB plateaus + a faint
/// intra-MB ramp): the coarse-quantiser reconstruction carries
/// pronounced MB-edge error the §15 filter reduces.
fn blocky() -> (u32, u32, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (80usize, 64usize);
    let (cw, ch) = (40usize, 32usize);
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let mb = ((r / 16) * 7 + (c / 16) * 13) % 200;
            let ramp = ((r % 16) + (c % 16)) / 4;
            y[r * w + c] = (20 + mb + ramp).min(255) as u8;
        }
    }
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            let mb = ((r / 8) * 11 + (c / 8) * 5) % 80;
            u[r * cw + c] = (110 + mb) as u8;
            v[r * cw + c] = (140 + mb) as u8;
        }
    }
    (w as u32, h as u32, y, u, v)
}

/// Parse the §9.4 6-bit `loop_filter_level` from a key-frame bitstream.
fn keyframe_filter_level(bytes: &[u8]) -> u8 {
    use oxideav_vp8::bool_decoder::BoolDecoder;
    let mut bd = BoolDecoder::init(&bytes[10..]).expect("init first partition");
    let _ = bd.read_bool(128).unwrap();
    let _ = bd.read_bool(128).unwrap();
    let _ = bd.read_bool(128).unwrap();
    let _ = bd.read_bool(128).unwrap();
    bd.read_literal(6).unwrap() as u8
}

#[test]
fn typed_encoder_auto_loop_filter_engages_and_decodes() {
    let (w, h, y, u, v) = blocky();
    let frame = I420Frame::packed(w, h, &y, &u, &v);

    let cfg = Vp8EncoderConfig {
        qindex: 110,
        lf_level: 0,
        auto_loop_filter: true,
        ..Vp8EncoderConfig::default()
    };
    let mut enc = Vp8Encoder::new(cfg);
    let bytes = enc
        .encode_keyframe(&frame)
        .expect("auto-LF keyframe encode");

    // Decodes cleanly.
    let dec = decode_vp8(&bytes).expect("decode");
    assert_eq!(dec.width, w);
    assert_eq!(dec.height, h);

    // The selector engaged (non-zero level reached the wire), proving the
    // config flag flows through the typed handle.
    let level = keyframe_filter_level(&bytes);
    assert!(
        level > 0,
        "auto-LF must engage on this blocky source, got {level}"
    );

    // Stats updated.
    assert_eq!(enc.stats().keyframes_emitted, 1);
}

#[test]
fn typed_encoder_fixed_lf_level_reaches_wire() {
    // With auto off, the config's lf_level is written verbatim.
    let (w, h, y, u, v) = blocky();
    let frame = I420Frame::packed(w, h, &y, &u, &v);
    let cfg = Vp8EncoderConfig {
        qindex: 48,
        lf_level: 24,
        auto_loop_filter: false,
        ..Vp8EncoderConfig::default()
    };
    let mut enc = Vp8Encoder::new(cfg);
    let bytes = enc
        .encode_keyframe(&frame)
        .expect("fixed-LF keyframe encode");
    assert_eq!(
        keyframe_filter_level(&bytes),
        24,
        "fixed lf_level reaches wire"
    );
    let dec = decode_vp8(&bytes).expect("decode");
    assert_eq!(dec.width, w);
}

#[cfg(feature = "registry")]
#[test]
fn make_encoder_with_config_auto_loop_filter_flows_through() {
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame};
    use oxideav_vp8::make_encoder_with_config;

    let (w, h, y, u, v) = blocky();

    let mut params = CodecParameters::video(CodecId::new("vp8"));
    params.media_type = MediaType::Video;
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::Yuv420P);

    let cfg = Vp8EncoderConfig {
        qindex: 110,
        lf_level: 0,
        auto_loop_filter: true,
        ..Vp8EncoderConfig::default()
    };
    let mut enc = make_encoder_with_config(&params, cfg).expect("build encoder");

    let uvw = w.div_ceil(2) as usize;
    let vframe = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y.clone(),
            },
            VideoPlane {
                stride: uvw,
                data: u.clone(),
            },
            VideoPlane {
                stride: uvw,
                data: v.clone(),
            },
        ],
    };
    enc.send_frame(&Frame::Video(vframe)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    assert!(pkt.flags.keyframe);

    let dec = decode_vp8(&pkt.data).expect("decode factory output");
    assert_eq!(dec.width, w);
    let level = keyframe_filter_level(&pkt.data);
    assert!(
        level > 0,
        "auto-LF must engage through the factory, got {level}"
    );
}
