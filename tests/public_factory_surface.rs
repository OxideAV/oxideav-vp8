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

// ───────────────────── round-408 fuzz-regression tests ─────────────────────
//
// The `encoder_trait_frame_lifecycle` fuzz target proved that the
// framework `send_frame` plane-extraction seam accepted `VideoFrame`
// shapes whose row repack then sliced out of bounds. Each hostile
// shape below must be answered with `Err`, never a panic, and must
// leave the encoder usable for a subsequent well-formed frame.

/// Build a 3-plane `VideoFrame` from explicit per-plane
/// `(stride, byte_len)` pairs.
fn shaped_frame(y: (usize, usize), u: (usize, usize), v: (usize, usize)) -> VideoFrame {
    let plane = |(stride, len): (usize, usize)| VideoPlane {
        stride,
        data: vec![0x80u8; len],
    };
    VideoFrame {
        pts: Some(0),
        planes: vec![plane(y), plane(u), plane(v)],
    }
}

#[test]
fn send_frame_rejects_stride_narrower_than_row_width() {
    // 16×16 luma with stride 15 and exactly stride*rows bytes: the old
    // `len < stride * h` check passed (240 >= 240), then the row
    // repack read `15*15 .. 15*15+16` = 225..241 out of a 240-byte
    // buffer — an out-of-bounds slice panic reachable from the public
    // `Encoder` trait.
    let mut enc = make_encoder(&vp8_params(16, 16)).expect("make_encoder");
    let bad = shaped_frame((15, 15 * 16), (8, 8 * 8), (8, 8 * 8));
    assert!(
        enc.send_frame(&Frame::Video(bad)).is_err(),
        "stride < row width must be rejected, not sliced out of bounds"
    );
    // The rejection must not have wedged the encoder.
    enc.send_frame(&Frame::Video(flat_grey_frame(16, 16)))
        .expect("well-formed frame after a rejected one must encode");
}

#[test]
fn send_frame_rejects_stride_times_rows_overflow() {
    // `stride * height` used to be computed unchecked: usize::MAX/2+1
    // × 16 wraps, letting the bogus plane through to the repack.
    let mut enc = make_encoder(&vp8_params(16, 16)).expect("make_encoder");
    let bad = shaped_frame((usize::MAX / 2 + 1, 16 * 16), (8, 8 * 8), (8, 8 * 8));
    assert!(
        enc.send_frame(&Frame::Video(bad)).is_err(),
        "stride x rows overflow must be a clean Err"
    );
}

#[test]
fn send_frame_rejects_short_plane_buffer() {
    let mut enc = make_encoder(&vp8_params(16, 16)).expect("make_encoder");
    let bad = shaped_frame((16, 16 * 16 - 1), (8, 8 * 8), (8, 8 * 8));
    assert!(
        enc.send_frame(&Frame::Video(bad)).is_err(),
        "a plane shorter than its repack footprint must be rejected"
    );
}

#[test]
fn send_frame_rejects_zero_stride_chroma() {
    let mut enc = make_encoder(&vp8_params(16, 16)).expect("make_encoder");
    let bad = shaped_frame((16, 16 * 16), (0, 8 * 8), (8, 8 * 8));
    assert!(
        enc.send_frame(&Frame::Video(bad)).is_err(),
        "zero-stride chroma must be rejected (stride < row width)"
    );
}

#[test]
fn send_frame_rejects_missing_planes() {
    let mut enc = make_encoder(&vp8_params(16, 16)).expect("make_encoder");
    let two_planes = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: 16,
                data: vec![0x80u8; 16 * 16],
            },
            VideoPlane {
                stride: 8,
                data: vec![0x80u8; 8 * 8],
            },
        ],
    };
    assert!(
        enc.send_frame(&Frame::Video(two_planes)).is_err(),
        "fewer than 3 planes must be rejected"
    );
}

#[test]
fn send_frame_accepts_last_row_without_stride_padding() {
    // The repack footprint is stride*(rows-1) + row_width: a frame
    // whose final row is not padded out to the full stride is legal.
    let mut enc = make_encoder(&vp8_params(16, 16)).expect("make_encoder");
    let ok = shaped_frame((20, 20 * 15 + 16), (10, 10 * 7 + 8), (10, 10 * 7 + 8));
    enc.send_frame(&Frame::Video(ok))
        .expect("unpadded final row must be accepted");
    assert!(
        enc.receive_packet().is_ok(),
        "the accepted frame must produce a packet"
    );
}
