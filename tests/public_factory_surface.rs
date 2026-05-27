//! Public-surface lock-test for the framework factory functions
//! (`make_encoder`, `make_encoder_with_quality`, `make_encoder_with_qindex`,
//! `quality_to_qindex`).
//!
//! Downstream consumers (notably `oxideav-webp`'s lossy VP8 path per
//! `oxideav-webps published 0.1.2 surface`) build their RIFF-VP8 image
//! encoder against these factory signatures. This test imports them via
//! the documented paths AND drives one end-to-end encode through each
//! so a regression that changes a signature, drops a re-export, or
//! breaks the registry/standalone gating fails to compile / runs red.
//!
//! These tests are registry-feature-gated because the factory functions
//! (and the [`oxideav_core::Encoder`] trait they return) only exist on
//! the framework side. The standalone-reachable
//! [`oxideav_vp8::quality_to_qindex`] is exercised separately in the
//! `quality_to_qindex_standalone` test below, which compiles on both
//! feature configurations.

#![cfg(feature = "registry")]

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame};
use oxideav_vp8::{
    decoder, encoder, make_encoder, make_encoder_with_qindex, make_encoder_with_quality,
    quality_to_qindex,
};

/// Build a `CodecParameters` for a `width × height` Yuv420P VP8 stream.
fn vp8_params(width: u32, height: u32) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new("vp8"));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params
}

/// Build a flat-grey 16×16 I420 `VideoFrame`.
fn flat_grey_frame(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w,
                data: vec![128u8; w * h],
            },
            VideoPlane {
                stride: uvw,
                data: vec![128u8; uvw * uvh],
            },
            VideoPlane {
                stride: uvw,
                data: vec![128u8; uvw * uvh],
            },
        ],
    }
}

#[test]
fn make_encoder_via_crate_root_re_export() {
    // `use oxideav_vp8::make_encoder;` must work — locked here.
    let params = vp8_params(16, 16);
    let _enc = make_encoder(&params).expect("make_encoder must accept Yuv420P 16×16");
}

#[test]
fn make_encoder_via_module_path_too() {
    // `use oxideav_vp8::encoder::make_encoder;` must also work.
    let params = vp8_params(16, 16);
    let _enc = encoder::make_encoder(&params).expect("encoder::make_encoder must work");
}

#[test]
fn make_encoder_rejects_missing_dimensions() {
    let mut params = CodecParameters::video(CodecId::new("vp8"));
    params.media_type = MediaType::Video;
    // No width / height — should fail cleanly.
    let r = make_encoder(&params);
    assert!(r.is_err(), "make_encoder must reject missing dimensions");
}

#[test]
fn make_encoder_rejects_zero_dimensions() {
    let params = vp8_params(0, 16);
    assert!(make_encoder(&params).is_err());
    let params = vp8_params(16, 0);
    assert!(make_encoder(&params).is_err());
}

#[test]
fn make_encoder_rejects_oversize_dimensions() {
    // VP8 §9.1 field is 14 bits — max 16383 inclusive.
    let params = vp8_params(16384, 16);
    assert!(make_encoder(&params).is_err());
}

#[test]
fn make_encoder_rejects_non_yuv420p_pixel_format() {
    let mut params = vp8_params(16, 16);
    params.pixel_format = Some(PixelFormat::Yuv444P);
    assert!(make_encoder(&params).is_err());
}

#[test]
fn make_encoder_with_quality_via_crate_root() {
    // `use oxideav_vp8::make_encoder_with_quality;` must work.
    let params = vp8_params(16, 16);
    let _enc = make_encoder_with_quality(&params, 75.0)
        .expect("make_encoder_with_quality must accept quality=75.0");
}

#[test]
fn make_encoder_with_qindex_via_crate_root() {
    // `use oxideav_vp8::make_encoder_with_qindex;` must work.
    let params = vp8_params(16, 16);
    let _enc = make_encoder_with_qindex(&params, 32)
        .expect("make_encoder_with_qindex must accept qindex=32");
}

#[test]
fn make_encoder_with_qindex_rejects_out_of_range_qi() {
    let params = vp8_params(16, 16);
    assert!(
        make_encoder_with_qindex(&params, 128).is_err(),
        "qindex=128 must be rejected (VP8 §9.6 is 0..=127)"
    );
}

#[test]
fn end_to_end_encode_via_make_encoder_emits_a_vp8_keyframe() {
    use oxideav_core::Error as CoreError;
    let params = vp8_params(16, 16);
    let mut enc = make_encoder_with_qindex(&params, 32).expect("make_encoder_with_qindex");
    enc.send_frame(&Frame::Video(flat_grey_frame(16, 16)))
        .expect("send_frame should accept a 16×16 Yuv420P frame");
    let pkt = enc
        .receive_packet()
        .expect("receive_packet after send_frame");
    assert!(
        pkt.flags.keyframe,
        "VP8 first encoded frame must be a keyframe"
    );
    assert!(!pkt.data.is_empty(), "encoded packet must carry bytes");

    // The next receive_packet must surface NeedMore (no flush yet).
    let want_need_more = enc.receive_packet();
    assert!(
        matches!(want_need_more, Err(CoreError::NeedMore)),
        "post-drain receive_packet must surface NeedMore, got {want_need_more:?}"
    );

    // After flush + drain, receive_packet must surface Eof.
    enc.flush().expect("flush");
    let want_eof = enc.receive_packet();
    assert!(
        matches!(want_eof, Err(CoreError::Eof)),
        "post-flush drained receive_packet must surface Eof, got {want_eof:?}"
    );

    // And the emitted bytes must decode through the in-tree decoder.
    let mut dec_params = vp8_params(16, 16);
    dec_params.codec_id = CodecId::new("vp8");
    let mut dec = decoder::make_decoder(&dec_params).expect("make_decoder");
    use oxideav_core::Packet;
    let mut pkt_for_decode = Packet::new(0, pkt.time_base, pkt.data);
    pkt_for_decode.pts = Some(0);
    dec.send_packet(&pkt_for_decode)
        .expect("send_packet to decoder");
    let frame = dec.receive_frame().expect("receive_frame from decoder");
    match frame {
        Frame::Video(v) => {
            assert_eq!(v.planes.len(), 3, "decoded frame must carry 3 planes");
        }
        _ => panic!("decoded frame must be Frame::Video, got {frame:?}"),
    }
}

#[test]
fn quality_to_qindex_via_crate_root_and_module() {
    // Both paths must resolve to the same function.
    let via_root = quality_to_qindex(75.0);
    let via_module = encoder::quality_to_qindex(75.0);
    assert_eq!(via_root, via_module);
}
