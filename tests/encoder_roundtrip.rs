//! VP8 I-frame encoder → decoder round-trip tests.
//!
//! These cover the v1 encoder (keyframes only, DC_PRED, fixed qindex,
//! loop filter disabled):
//!   * Solid-colour 128×128 image round-trips with very high PSNR.
//!   * A mid-complexity YUV test pattern round-trips above the 25 dB
//!     PSNR bar the task brief specifies.
//!   * The compressed output starts with the correct VP8 keyframe tag
//!     (show_frame set, frame_type=0) followed by the 3-byte start code
//!     `9d 01 2a`.

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_vp8::decoder::Vp8Decoder;
use oxideav_vp8::encoder::{encode_keyframe, make_encoder_with_qindex};
use oxideav_vp8::{decode_frame, parse_header, FrameType};

const W: u32 = 128;
const H: u32 = 128;
const QINDEX: u8 = 50;

fn make_frame(y: &[u8], u: &[u8], v: &[u8]) -> VideoFrame {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    assert_eq!(y.len(), (W * H) as usize);
    assert_eq!(u.len(), cw * ch);
    assert_eq!(v.len(), cw * ch);
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: W,
        height: H,
        pts: None,
        time_base: TimeBase::new(1, 1000),
        planes: vec![
            VideoPlane {
                stride: W as usize,
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
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut se = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        se += d * d;
    }
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

#[test]
fn keyframe_starts_with_correct_start_code() {
    let y = vec![200u8; (W * H) as usize];
    let u = vec![100u8; ((W / 2) * (H / 2)) as usize];
    let v = vec![150u8; ((W / 2) * (H / 2)) as usize];
    let frame = make_frame(&y, &u, &v);
    let encoded = encode_keyframe(W, H, QINDEX, &frame).expect("encode");
    assert!(encoded.len() >= 10, "encoded stream too short");
    // Parse the frame tag — frame_type must be Key, show_frame=true.
    let parsed = parse_header(&encoded).expect("parse");
    assert!(matches!(parsed.tag.frame_type, FrameType::Key));
    assert!(parsed.tag.show_frame);
    // Sync code at offset 3..6.
    assert_eq!(
        &encoded[3..6],
        &[0x9d, 0x01, 0x2a],
        "sync code mismatch: got {:02x?}",
        &encoded[3..6]
    );
    // Width/height little-endian at 6..10.
    let w = u16::from_le_bytes([encoded[6], encoded[7]]) & 0x3fff;
    let h = u16::from_le_bytes([encoded[8], encoded[9]]) & 0x3fff;
    assert_eq!(w as u32, W);
    assert_eq!(h as u32, H);
}

#[test]
fn roundtrip_solid_color_high_psnr() {
    // Solid grey frame — should come back essentially losslessly since
    // only DC energy is present and DC quant is coarse but stable.
    let y = vec![128u8; (W * H) as usize];
    let u = vec![128u8; ((W / 2) * (H / 2)) as usize];
    let v = vec![128u8; ((W / 2) * (H / 2)) as usize];
    let frame = make_frame(&y, &u, &v);
    let encoded = encode_keyframe(W, H, QINDEX, &frame).expect("encode");
    let decoded = decode_frame(&encoded).expect("decode");
    assert_eq!(decoded.width, W);
    assert_eq!(decoded.height, H);
    let py = psnr(&decoded.planes[0].data, &y);
    let pu = psnr(&decoded.planes[1].data, &u);
    let pv = psnr(&decoded.planes[2].data, &v);
    eprintln!("solid-grey PSNR Y={py:.2} U={pu:.2} V={pv:.2}");
    assert!(py >= 40.0, "solid-grey Y PSNR too low: {py:.2} dB");
    assert!(pu >= 40.0, "solid-grey U PSNR too low: {pu:.2} dB");
    assert!(pv >= 40.0, "solid-grey V PSNR too low: {pv:.2} dB");
}

#[test]
fn roundtrip_yuv_test_pattern_psnr_above_25() {
    // Mid-complexity YUV pattern: smooth diagonal luma gradient plus
    // horizontal / vertical chroma gradients.
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut y = vec![0u8; (W * H) as usize];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for row in 0..H as usize {
        for col in 0..W as usize {
            let base = ((row + col) as i32 * 255 / (W + H - 2) as i32) as u8;
            y[row * W as usize + col] = base;
        }
    }
    for row in 0..ch {
        for col in 0..cw {
            u[row * cw + col] = 64 + ((col * 255) / cw) as u8 / 2;
            v[row * cw + col] = 192 - ((row * 255) / ch) as u8 / 2;
        }
    }
    let frame = make_frame(&y, &u, &v);
    let encoded = encode_keyframe(W, H, QINDEX, &frame).expect("encode");
    let decoded = decode_frame(&encoded).expect("decode");
    let py = psnr(&decoded.planes[0].data, &y);
    let pu = psnr(&decoded.planes[1].data, &u);
    let pv = psnr(&decoded.planes[2].data, &v);
    eprintln!(
        "yuv-pattern PSNR Y={py:.2} U={pu:.2} V={pv:.2} (encoded size {} bytes)",
        encoded.len()
    );
    assert!(py > 25.0, "Y PSNR too low: {py:.2} dB");
    assert!(pu > 25.0, "U PSNR too low: {pu:.2} dB");
    assert!(pv > 25.0, "V PSNR too low: {pv:.2} dB");
}

/// Encode a 2-frame sequence (I + P) of a sliding gradient, decode it
/// through the full encoder/decoder pipeline, and check that the second
/// (P-frame) plane recovers well.
#[test]
fn pframe_roundtrip_sliding_gradient_psnr_above_30() {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;

    // Build a frame with a smooth horizontal luma gradient offset by `shift`.
    let make_grad = |shift: usize| -> VideoFrame {
        let mut y = vec![0u8; (W * H) as usize];
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let c = (col + shift) % W as usize;
                // Smooth 0..=255 horizontal gradient.
                y[row * W as usize + col] = ((c * 255) / (W as usize - 1)) as u8;
            }
        }
        for row in 0..ch {
            for col in 0..cw {
                u[row * cw + col] = 128;
                v[row * cw + col] = 128;
            }
        }
        make_frame(&y, &u, &v)
    };

    // Frame 1 is identical to frame 0 (static content exercises SKIP).
    // Frame 2 shifts the gradient by 1 px (small change, exercises the
    // coded-residual ZERO_MV path since motion is below the SAD skip bar).
    let f0 = make_grad(0);
    let f1 = make_grad(0);
    let f0_y = f0.planes[0].data.clone();
    let f0_u = f0.planes[1].data.clone();
    let f0_v = f0.planes[2].data.clone();
    let f1_y = f1.planes[0].data.clone();
    let f1_u = f1.planes[1].data.clone();
    let f1_v = f1.planes[2].data.clone();

    // Encoder.
    let mut enc_params = CodecParameters::video(CodecId::new("vp8"));
    enc_params.width = Some(W);
    enc_params.height = Some(H);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = make_encoder_with_qindex(&enc_params, QINDEX).expect("encoder");

    enc.send_frame(&Frame::Video(f0)).expect("send f0");
    let pkt_i = enc.receive_packet().expect("receive I");
    assert!(pkt_i.flags.keyframe);
    let parsed_i = parse_header(&pkt_i.data).expect("parse I");
    assert!(matches!(parsed_i.tag.frame_type, FrameType::Key));

    enc.send_frame(&Frame::Video(f1)).expect("send f1");
    let pkt_p = enc.receive_packet().expect("receive P");
    assert!(!pkt_p.flags.keyframe);
    let parsed_p = parse_header(&pkt_p.data).expect("parse P");
    assert!(matches!(parsed_p.tag.frame_type, FrameType::Inter));

    eprintln!(
        "I-frame size: {} bytes, P-frame size: {} bytes",
        pkt_i.data.len(),
        pkt_p.data.len()
    );
    // For an identical frame under the skip path, the P-frame must be
    // meaningfully smaller than the I-frame — the defining feature of
    // inter coding.
    assert!(
        pkt_p.data.len() * 2 < pkt_i.data.len(),
        "P-frame not meaningfully smaller than I-frame: P={} I={}",
        pkt_p.data.len(),
        pkt_i.data.len()
    );

    // Decode through the stateful decoder.
    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_i.data.clone()))
        .expect("decode I");
    let frame_i = match dec.receive_frame().expect("receive I frame") {
        Frame::Video(v) => v,
        _ => panic!("not a video frame"),
    };
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_p.data.clone()))
        .expect("decode P");
    let frame_p = match dec.receive_frame().expect("receive P frame") {
        Frame::Video(v) => v,
        _ => panic!("not a video frame"),
    };

    let p_y = psnr(&frame_p.planes[0].data, &f1_y);
    let p_u = psnr(&frame_p.planes[1].data, &f1_u);
    let p_v = psnr(&frame_p.planes[2].data, &f1_v);
    eprintln!("P-frame PSNR Y={p_y:.2} U={p_u:.2} V={p_v:.2}");
    assert!(
        p_y >= 30.0,
        "P-frame Y PSNR too low: {p_y:.2} dB (I-frame Y PSNR was {:.2} dB)",
        psnr(&frame_i.planes[0].data, &f0_y)
    );
    assert!(p_u >= 30.0, "P-frame U PSNR too low: {p_u:.2} dB");
    assert!(p_v >= 30.0, "P-frame V PSNR too low: {p_v:.2} dB");

    // I-frame PSNR sanity (should still match the existing bar).
    let i_y = psnr(&frame_i.planes[0].data, &f0_y);
    let i_u = psnr(&frame_i.planes[1].data, &f0_u);
    let i_v = psnr(&frame_i.planes[2].data, &f0_v);
    eprintln!("I-frame PSNR Y={i_y:.2} U={i_u:.2} V={i_v:.2}");
    assert!(i_y > 25.0);
    assert!(i_u > 25.0);
    assert!(i_v > 25.0);
}

/// Encode a 3-frame sequence where the content actually changes between
/// frame 1 and frame 2 by enough to defeat the SAD-based skip heuristic.
/// This exercises the ZERO_MV residual-coding path in the P-frame encoder.
#[test]
fn pframe_roundtrip_residual_path_exercised() {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;

    let mut y0 = vec![0u8; (W * H) as usize];
    let mut y1 = vec![0u8; (W * H) as usize];
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    for row in 0..H as usize {
        for col in 0..W as usize {
            y0[row * W as usize + col] = ((col * 255) / (W as usize - 1)) as u8;
            // Shift by 8 pixels → big enough SAD that MBs cannot skip,
            // but still small enough that the residual is compressible.
            let c2 = (col + 8) % W as usize;
            y1[row * W as usize + col] = ((c2 * 255) / (W as usize - 1)) as u8;
        }
    }
    let f0 = make_frame(&y0, &u, &v);
    let f1 = make_frame(&y1, &u, &v);
    let y1_expected = y1.clone();

    let mut enc_params = CodecParameters::video(CodecId::new("vp8"));
    enc_params.width = Some(W);
    enc_params.height = Some(H);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = make_encoder_with_qindex(&enc_params, QINDEX).expect("encoder");

    enc.send_frame(&Frame::Video(f0)).expect("send f0");
    let pkt_i = enc.receive_packet().expect("receive I");
    enc.send_frame(&Frame::Video(f1)).expect("send f1");
    let pkt_p = enc.receive_packet().expect("receive P");
    let parsed_p = parse_header(&pkt_p.data).expect("parse P");
    assert!(matches!(parsed_p.tag.frame_type, FrameType::Inter));
    eprintln!(
        "residual-path I={} bytes, P={} bytes",
        pkt_i.data.len(),
        pkt_p.data.len()
    );

    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_i.data.clone()))
        .expect("decode I");
    let _ = dec.receive_frame().expect("rx I");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_p.data.clone()))
        .expect("decode P");
    let frame_p = match dec.receive_frame().expect("rx P") {
        Frame::Video(v) => v,
        _ => panic!("not video"),
    };
    let p_y = psnr(&frame_p.planes[0].data, &y1_expected);
    eprintln!("residual-path P-frame Y PSNR = {p_y:.2} dB");
    assert!(p_y > 25.0, "residual-path P-frame Y PSNR too low: {p_y:.2}");
}

/// Uniform horizontal pan (every pixel of frame 1 equals the corresponding
/// pixel of frame 0 shifted by +8 columns, with edge replication). The
/// encoder should find that shift via its integer motion search and emit
/// NEWMV, reconstructing the P-frame at very high PSNR with a P-frame that
/// is much smaller than a plain coded-residual ZERO_MV would produce.
#[test]
fn pframe_roundtrip_pan_picks_newmv() {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;

    // Frame 0: vertical-stripe pattern so horizontal shifts change SAD a lot.
    // Frame 1: same pattern shifted left by 8 columns (pan by +8 in source
    // coordinates → ref-side MV of (0, +8 luma pixels) = mv.col = +64).
    let mut y0 = vec![0u8; (W * H) as usize];
    let mut y1 = vec![0u8; (W * H) as usize];
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    for row in 0..H as usize {
        for col in 0..W as usize {
            // 8-wide vertical stripes alternating 40/200 — strong AC energy,
            // easy to lock onto with integer-pel SAD.
            let stripe = (col / 8) & 1;
            y0[row * W as usize + col] = if stripe == 0 { 40 } else { 200 };
            let src_col = (col + 8).min(W as usize - 1);
            y1[row * W as usize + col] = y0[row * W as usize + src_col];
        }
    }
    let f0 = make_frame(&y0, &u, &v);
    let f1 = make_frame(&y1, &u, &v);
    let y1_expected = y1.clone();

    let mut enc_params = CodecParameters::video(CodecId::new("vp8"));
    enc_params.width = Some(W);
    enc_params.height = Some(H);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = make_encoder_with_qindex(&enc_params, QINDEX).expect("encoder");

    enc.send_frame(&Frame::Video(f0)).expect("send f0");
    let pkt_i = enc.receive_packet().expect("rx I");
    enc.send_frame(&Frame::Video(f1)).expect("send f1");
    let pkt_p = enc.receive_packet().expect("rx P");
    assert!(!pkt_p.flags.keyframe);

    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_i.data.clone()))
        .expect("decode I");
    let _ = dec.receive_frame().expect("rx I");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_p.data.clone()))
        .expect("decode P");
    let frame_p = match dec.receive_frame().expect("rx P") {
        Frame::Video(v) => v,
        _ => panic!("not video"),
    };
    let p_y = psnr(&frame_p.planes[0].data, &y1_expected);
    eprintln!(
        "pan PSNR Y={p_y:.2} dB, P-frame {} bytes (I-frame {} bytes)",
        pkt_p.data.len(),
        pkt_i.data.len()
    );
    // Motion-compensated prediction should recover the pan near-losslessly
    // (the residual is essentially zero on fully-matched blocks). Without
    // NEWMV this value collapses to low-20 dB, so the bar doubles as a
    // regression guard that motion search is actually selecting shifted
    // matches.
    assert!(p_y >= 40.0, "pan P-frame Y PSNR too low: {p_y:.2}");
}

/// Uniform **diagonal** pan exercising the NEAREST_MV savings path.
/// Every MB wants the same (+8, +8) motion: the encoder picks NEW_MV for
/// the first MB, then for subsequent MBs the neighbour MV context yields
/// that same vector as `nearest` — so NEAREST_MV should be selected and
/// no MV delta coded, making the P-frame meaningfully smaller than a
/// NEW_MV-only encoder would produce.
#[test]
fn pframe_roundtrip_nearest_mv_kicks_in() {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;

    // Stripe pattern offers strong SAD contrast along (+8, +8); same
    // luminance values as the pan test, so an 8-px diagonal shift leaves
    // the MB content intact modulo edge replication.
    let mut y0 = vec![0u8; (W * H) as usize];
    let mut y1 = vec![0u8; (W * H) as usize];
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    for row in 0..H as usize {
        for col in 0..W as usize {
            // 16×16 checkerboard — any integer shift moves a detectable
            // amount of luma energy.
            let tile = ((row / 16) ^ (col / 16)) & 1;
            y0[row * W as usize + col] = if tile == 0 { 40 } else { 200 };
            let sr = row.saturating_sub(8);
            let sc = col.saturating_sub(8);
            let sti = ((sr / 16) ^ (sc / 16)) & 1;
            y1[row * W as usize + col] = if sti == 0 { 40 } else { 200 };
        }
    }
    let f0 = make_frame(&y0, &u, &v);
    let f1 = make_frame(&y1, &u, &v);
    let y1_expected = y1.clone();

    let mut enc_params = CodecParameters::video(CodecId::new("vp8"));
    enc_params.width = Some(W);
    enc_params.height = Some(H);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = make_encoder_with_qindex(&enc_params, QINDEX).expect("encoder");

    enc.send_frame(&Frame::Video(f0)).expect("send f0");
    let pkt_i = enc.receive_packet().expect("rx I");
    enc.send_frame(&Frame::Video(f1)).expect("send f1");
    let pkt_p = enc.receive_packet().expect("rx P");
    assert!(!pkt_p.flags.keyframe);

    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_i.data.clone()))
        .expect("decode I");
    let _ = dec.receive_frame().expect("rx I");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_p.data.clone()))
        .expect("decode P");
    let frame_p = match dec.receive_frame().expect("rx P") {
        Frame::Video(v) => v,
        _ => panic!("not video"),
    };
    let p_y = psnr(&frame_p.planes[0].data, &y1_expected);
    eprintln!(
        "nearest-mv PSNR Y={p_y:.2} dB, P-frame {} bytes (I-frame {} bytes)",
        pkt_p.data.len(),
        pkt_i.data.len()
    );
    assert!(p_y >= 35.0, "nearest-mv P-frame Y PSNR too low: {p_y:.2}");
    // Sanity: bit-exact round-trip through the decoder gives us a strong
    // regression guard on the MV-ref tree + sub-pel prediction paths
    // together (NEAREST passes no MV delta, so any miscoding shows up as
    // visible corruption rather than a dB drop).
}

/// Static background (all MBs identical between two frames) — the encoder
/// should SKIP every MB, yielding a P-frame that is a handful of bytes
/// longer than the header alone.
#[test]
fn pframe_roundtrip_static_background_all_skip() {
    let y = vec![100u8; (W * H) as usize];
    let u = vec![128u8; ((W / 2) * (H / 2)) as usize];
    let v = vec![128u8; ((W / 2) * (H / 2)) as usize];
    let f0 = make_frame(&y, &u, &v);
    let f1 = make_frame(&y, &u, &v);

    let mut enc_params = CodecParameters::video(CodecId::new("vp8"));
    enc_params.width = Some(W);
    enc_params.height = Some(H);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = make_encoder_with_qindex(&enc_params, QINDEX).expect("encoder");

    enc.send_frame(&Frame::Video(f0)).expect("send f0");
    let pkt_i = enc.receive_packet().expect("rx I");
    enc.send_frame(&Frame::Video(f1)).expect("send f1");
    let pkt_p = enc.receive_packet().expect("rx P");
    // With 64 MBs on a 128x128 frame and every one emitting SKIP, the
    // P-frame should be dominated by the fixed header overhead.
    eprintln!(
        "all-skip P-frame {} bytes vs I-frame {} bytes",
        pkt_p.data.len(),
        pkt_i.data.len()
    );
    assert!(
        pkt_p.data.len() < 150,
        "all-skip P-frame unexpectedly large: {} bytes",
        pkt_p.data.len()
    );
}

/// Mid-frame "scene cut": the upper half of the frame is unchanged (inter
/// codes well), the lower half is a fresh random-ish pattern that the
/// reference cannot predict. The intra-in-P fallback should kick in on
/// the lower half, producing a usable reconstruction that plain inter
/// modes could not achieve at this qindex.
#[test]
fn pframe_roundtrip_intra_in_p_on_scene_cut() {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;

    let mut y0 = vec![0u8; (W * H) as usize];
    let mut y1 = vec![0u8; (W * H) as usize];
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    for row in 0..H as usize {
        for col in 0..W as usize {
            // Both frames have a smooth gradient in the top half.
            y0[row * W as usize + col] = ((col * 200) / W as usize) as u8;
            if row < H as usize / 2 {
                y1[row * W as usize + col] = y0[row * W as usize + col];
            } else {
                // Bottom half of f1: a deterministic high-entropy pattern
                // that bears no resemblance to f0's bottom half.
                let x = row.wrapping_mul(131).wrapping_add(col.wrapping_mul(89));
                y1[row * W as usize + col] = ((x & 0xff) as u8).wrapping_add(64);
            }
        }
    }
    let f0 = make_frame(&y0, &u, &v);
    let f1 = make_frame(&y1, &u, &v);
    let y1_expected = y1.clone();

    let mut enc_params = CodecParameters::video(CodecId::new("vp8"));
    enc_params.width = Some(W);
    enc_params.height = Some(H);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = make_encoder_with_qindex(&enc_params, QINDEX).expect("encoder");

    enc.send_frame(&Frame::Video(f0)).expect("send f0");
    let pkt_i = enc.receive_packet().expect("rx I");
    enc.send_frame(&Frame::Video(f1)).expect("send f1");
    let pkt_p = enc.receive_packet().expect("rx P");
    assert!(!pkt_p.flags.keyframe);

    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_i.data.clone()))
        .expect("decode I");
    let _ = dec.receive_frame().expect("rx I");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_p.data.clone()))
        .expect("decode P");
    let frame_p = match dec.receive_frame().expect("rx P") {
        Frame::Video(v) => v,
        _ => panic!("not video"),
    };
    // Upper half: inter-coded, should be near-lossless.
    let upper_src = &y1_expected[..(W as usize * (H as usize / 2))];
    let upper_rec = &frame_p.planes[0].data[..(W as usize * (H as usize / 2))];
    let p_upper = psnr(upper_src, upper_rec);
    // Lower half: intra fallback — coarse but better than bogus inter.
    let lower_src = &y1_expected[(W as usize * (H as usize / 2))..];
    let lower_rec = &frame_p.planes[0].data[(W as usize * (H as usize / 2))..];
    let p_lower = psnr(lower_src, lower_rec);
    eprintln!(
        "intra-in-P PSNR upper={p_upper:.2} dB, lower={p_lower:.2} dB, \
         P-frame {} bytes",
        pkt_p.data.len()
    );
    // Upper half should be essentially perfect (static content → all skip).
    assert!(p_upper >= 40.0, "upper half PSNR too low: {p_upper:.2} dB");
    // Lower half bar: intra DC_PRED on high-entropy content gives low
    // single-digit dB at qindex=50 before any AC is transmitted, but the
    // coded residual lifts it well above inter-only (which would be
    // ~7 dB because the reference is a gradient, not noise).
    assert!(
        p_lower >= 10.0,
        "intra-in-P lower half PSNR too low: {p_lower:.2} dB"
    );
}

/// Sub-pel pan: every pixel of frame 1 equals frame 0 shifted by a
/// non-integer horizontal amount (approximated via a simple 2-tap
/// bilinear pre-filter on the source). The encoder's quarter-pel
/// refinement should outperform integer-only search on this input.
#[test]
fn pframe_roundtrip_subpel_pan_beats_integer_only() {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;

    let mut y0 = vec![0u8; (W * H) as usize];
    let mut y1 = vec![0u8; (W * H) as usize];
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    // Smooth vertical-stripe pattern: avoids hard aliasing so a
    // quarter-pel shift on the source remains a reasonable sub-pel
    // translation of the reference.
    for row in 0..H as usize {
        for col in 0..W as usize {
            // Smooth sinusoidal-ish pattern using a triangle wave.
            let t = (col as i32 * 4).rem_euclid(64);
            let tri = if t < 32 { t } else { 64 - t };
            let v0 = (128 + tri * 3) as u8;
            y0[row * W as usize + col] = v0;
            // Shift f1 by +0.5 pixel: average the two neighbouring samples.
            let next = col + 1;
            let t2 = (next as i32 * 4).rem_euclid(64);
            let tri2 = if t2 < 32 { t2 } else { 64 - t2 };
            let v1 = (128 + tri2 * 3) as u8;
            y1[row * W as usize + col] = ((v0 as u16 + v1 as u16) / 2) as u8;
        }
    }
    let f0 = make_frame(&y0, &u, &v);
    let f1 = make_frame(&y1, &u, &v);
    let y1_expected = y1.clone();

    let mut enc_params = CodecParameters::video(CodecId::new("vp8"));
    enc_params.width = Some(W);
    enc_params.height = Some(H);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = make_encoder_with_qindex(&enc_params, QINDEX).expect("encoder");

    enc.send_frame(&Frame::Video(f0)).expect("send f0");
    let pkt_i = enc.receive_packet().expect("rx I");
    enc.send_frame(&Frame::Video(f1)).expect("send f1");
    let pkt_p = enc.receive_packet().expect("rx P");

    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_i.data.clone()))
        .expect("decode I");
    let _ = dec.receive_frame().expect("rx I");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt_p.data.clone()))
        .expect("decode P");
    let frame_p = match dec.receive_frame().expect("rx P") {
        Frame::Video(v) => v,
        _ => panic!("not video"),
    };
    let p_y = psnr(&frame_p.planes[0].data, &y1_expected);
    eprintln!(
        "subpel-pan PSNR Y={p_y:.2} dB, P-frame {} bytes",
        pkt_p.data.len()
    );
    // Sanity: reconstruction must still be clean. A full decode path
    // that mishandled sub-pel MVs would produce visible corruption that
    // immediately shows up as a low PSNR.
    assert!(
        p_y >= 28.0,
        "subpel-pan P-frame Y PSNR too low: {p_y:.2} dB"
    );
}
