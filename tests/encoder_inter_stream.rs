//! Multi-frame I + P VP8 stream-encoder round-trip
//! (RFC 6386 §9 / §9.7 / §9.8 / §16 / §17 / §18 — keyframe-anchored
//! GOP with ZERO_MV P-frames in between).
//!
//! Drives a synthetic 10-frame I420 sequence with keyframe interval 4
//! through [`oxideav_vp8::Vp8InterStreamEncoder`]:
//!
//!   * Frames 0, 4, 8 → key frames (independently decodable).
//!   * Frames 1, 2, 3, 5, 6, 7, 9 → ZERO_MV P-frames against LAST.
//!
//! The emitted bytes are replayed through
//! [`oxideav_vp8::Vp8DecoderState::decode_frame`] (the stateful
//! multi-frame decoder a real consumer would use). Every recovered
//! frame must reach the round's PSNR ≥ 30 dB bar against its
//! corresponding input picture at a mid quantiser (`yac_qi = 32`).
//!
//! The 10-frame interleave pins:
//!
//!   * the §9 K-vs-P scheduling (the `frame_type` bit and absence of
//!     the keyframe start code on P-frames);
//!   * the §9.7 reference-slot refresh ladder across both K and P
//!     transitions (LAST reflects the most-recent frame, GOLDEN /
//!     ALTREF hold the most-recent K-frame's reconstruction);
//!   * the §16.3 census-driven inter-mode probability evolution
//!     across multiple consecutive P-frames (each P-frame inherits
//!     the LAST slot the previous frame produced);
//!   * the §17.2 MV-context carry-state across P-frames inside a
//!     single GOP;
//!   * the §18 identity-copy prediction at MV (0, 0) producing a
//!     usable picture for a low-motion synthetic source.
//!
//! Self-contained black-box test: no external codec consulted, the
//! encoder's output is fed straight into the crate's own decoder.

use oxideav_vp8::{
    FrameKind, I420Frame, KeyframeParams, Vp8DecodedFrame, Vp8DecoderState, Vp8InterStreamEncoder,
};

/// Packed I420 source picture with row-major planes.
struct Source {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Source {
    fn frame(&self) -> I420Frame<'_> {
        I420Frame::packed(self.width, self.height, &self.y, &self.u, &self.v)
    }
}

/// Build a smoothly-varying I420 picture parameterised by frame
/// index. The motion model is *slow translation* — every P-frame is
/// nearly equal to the previous one, so the ZERO_MV §18 identity copy
/// is a good prediction and the residual stays inside the §14
/// quantiser's distortion envelope.
///
/// * Luma is a diagonal gradient whose origin drifts by 1 pixel per
///   frame on both axes (small enough that a (0, 0) MV residual
///   absorbs the change at `yac_qi = 32`).
/// * Chroma is a constant DC slowly walking (`130 + frame_idx`,
///   `120 - frame_idx`).
fn synthetic_frame(width: u32, height: u32, frame_idx: usize) -> Source {
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;

    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            // Slow diagonal drift: per-frame offset of 1 pixel.
            let v = 40 + (r as i16) + (c as i16) + (frame_idx as i16);
            y[r * w + c] = v.clamp(0, 255) as u8;
        }
    }

    let u_val = (130 + frame_idx as i32).clamp(0, 255) as u8;
    let v_val = (120 - frame_idx as i32).clamp(0, 255) as u8;
    let u = vec![u_val; cw * ch];
    let v = vec![v_val; cw * ch];

    Source {
        width,
        height,
        y,
        u,
        v,
    }
}

fn plane_mse(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "plane length mismatch");
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum();
    sum / a.len() as f64
}

/// Whole-frame PSNR across the three planes combined (8-bit peak = 255).
fn frame_psnr(src: &Source, dec: &Vp8DecodedFrame) -> f64 {
    let total = src.y.len() + src.u.len() + src.v.len();
    let combined_se = plane_mse(&src.y, &dec.y) * src.y.len() as f64
        + plane_mse(&src.u, &dec.u) * src.u.len() as f64
        + plane_mse(&src.v, &dec.v) * src.v.len() as f64;
    let mse = combined_se / total as f64;
    if mse <= f64::EPSILON {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// The round target. A 10-frame synthetic I+P+P+P+I+P+P+P+I+P
/// sequence at `keyframe_interval = 4`. Every per-frame self-decode
/// through `Vp8DecoderState` must clear 30 dB.
#[test]
fn ten_frame_inter_stream_mid_quant_meets_30db_per_frame() {
    let width = 48u32;
    let height = 32u32;
    let qi = 32u8;
    let n_frames = 10usize;
    let keyframe_interval = 4u64;

    let sources: Vec<Source> = (0..n_frames)
        .map(|i| synthetic_frame(width, height, i))
        .collect();

    let mut enc = Vp8InterStreamEncoder::new(
        KeyframeParams {
            y_ac_qi: qi,
            loop_filter_level: 0,
            sharpness_level: 0,
            nbr_of_dct_partitions: 1,
            filter_type: false,
        },
        keyframe_interval,
    )
    .expect("non-zero interval");

    // Encode + remember the K-vs-P classification for each frame.
    let mut encoded = Vec::with_capacity(n_frames);
    for (i, src) in sources.iter().enumerate() {
        let frame = src.frame();
        let out = enc
            .encode_frame(&frame)
            .unwrap_or_else(|e| panic!("encode_frame[{i}]: {e}"));
        assert!(!out.bytes.is_empty(), "frame {i} bytes non-empty");
        encoded.push(out);
    }
    assert_eq!(enc.frame_count(), n_frames as u64);
    assert_eq!(enc.dimensions(), Some((width, height)));

    // Confirm the expected K/P interleave: 0,4,8 are K; rest are P.
    for (i, e) in encoded.iter().enumerate() {
        let expected = if i as u64 % keyframe_interval == 0 {
            FrameKind::Key
        } else {
            FrameKind::InterZeroMv
        };
        assert_eq!(
            e.kind, expected,
            "frame {i} expected {expected:?}, got {:?}",
            e.kind
        );
        assert_eq!(e.frame_index, i as u64);
    }

    // Replay through the multi-frame decoder driver.
    let mut dec = Vp8DecoderState::new();
    let mut psnrs = Vec::with_capacity(n_frames);
    for (i, e) in encoded.iter().enumerate() {
        let out = dec
            .decode_frame(&e.bytes)
            .unwrap_or_else(|err| panic!("decode_frame[{i}] ({:?}): {err:?}", e.kind));
        assert_eq!(out.width, width, "frame {i} decoded width");
        assert_eq!(out.height, height, "frame {i} decoded height");

        let psnr = frame_psnr(&sources[i], &out);
        eprintln!("frame {i} ({:?}) self-decode PSNR = {psnr:.2} dB", e.kind);
        assert!(
            psnr >= 30.0,
            "frame {i} ({:?}) PSNR {psnr:.2} dB below the 30.0 dB target",
            e.kind
        );
        psnrs.push(psnr);
    }
    let mean: f64 = psnrs.iter().sum::<f64>() / psnrs.len() as f64;
    eprintln!("10-frame mean self-decode PSNR = {mean:.2} dB");
}

/// Verify the §9.1 `frame_type` bit on every emitted frame matches
/// the encoder's classification: K-frames carry bit = 0 plus the
/// 0x9d 0x01 0x2a start code at offset 3; P-frames carry bit = 1 and
/// do not carry the start code at that offset.
#[test]
fn inter_stream_frame_tag_matches_classification() {
    let width = 32u32;
    let height = 32u32;
    let mut enc =
        Vp8InterStreamEncoder::new(KeyframeParams::default(), 3).expect("non-zero interval");
    for i in 0..6usize {
        let src = synthetic_frame(width, height, i);
        let out = enc.encode_frame(&src.frame()).expect("encode");
        let key_bit = out.bytes[0] & 0x01;
        match out.kind {
            FrameKind::Key => {
                assert_eq!(key_bit, 0, "frame {i} K-frame must have key bit 0");
                assert_eq!(
                    &out.bytes[3..6],
                    &[0x9d, 0x01, 0x2a],
                    "frame {i} K-frame must carry §9.1 start code"
                );
            }
            FrameKind::InterZeroMv => {
                assert_eq!(key_bit, 1, "frame {i} P-frame must have key bit 1");
                if out.bytes.len() >= 6 {
                    assert_ne!(
                        &out.bytes[3..6],
                        &[0x9d, 0x01, 0x2a],
                        "frame {i} P-frame must not carry the start code"
                    );
                }
            }
        }
    }
}

/// Verify a forced keyframe in the middle of a GOP both produces a
/// structurally-valid K frame and re-anchors the keyframe interval.
#[test]
fn inter_stream_forced_keyframe_decodes_and_reanchors() {
    let width = 32u32;
    let height = 32u32;
    let mut enc =
        Vp8InterStreamEncoder::new(KeyframeParams::default(), 5).expect("non-zero interval");

    // Frame 0: K (first ever)
    let src0 = synthetic_frame(width, height, 0);
    let _ = enc.encode_frame(&src0.frame()).expect("frame 0");
    // Frame 1: P (no force)
    let src1 = synthetic_frame(width, height, 1);
    let f1 = enc.encode_frame(&src1.frame()).expect("frame 1");
    assert_eq!(f1.kind, FrameKind::InterZeroMv);
    // Frame 2: force K
    let src2 = synthetic_frame(width, height, 2);
    let f2 = enc
        .encode_frame_with_force(&src2.frame(), true)
        .expect("frame 2");
    assert_eq!(f2.kind, FrameKind::Key);
    // Frames 3..6: P (re-anchor means next K is at frame 7)
    for i in 3..=6usize {
        let src = synthetic_frame(width, height, i);
        let f = enc.encode_frame(&src.frame()).expect("p frame");
        assert_eq!(f.kind, FrameKind::InterZeroMv, "frame {i}");
    }
    // Frame 7: K (2 + 5 = 7)
    let src7 = synthetic_frame(width, height, 7);
    let f7 = enc.encode_frame(&src7.frame()).expect("frame 7");
    assert_eq!(
        f7.kind,
        FrameKind::Key,
        "interval should re-anchor after the forced K"
    );

    // Quick sanity: the whole sequence decodes cleanly through the
    // stateful decoder.
    let mut dec = Vp8DecoderState::new();
    // Re-encode the same sequence in a fresh encoder so we have all
    // the bytes in hand to feed the decoder (the loop above consumed
    // them).
    let mut enc2 =
        Vp8InterStreamEncoder::new(KeyframeParams::default(), 5).expect("non-zero interval");
    let mut bytes_per_frame = Vec::new();
    for i in 0..=7usize {
        let src = synthetic_frame(width, height, i);
        let out = enc2
            .encode_frame_with_force(&src.frame(), i == 2)
            .expect("encode2");
        bytes_per_frame.push(out.bytes);
    }
    for (i, b) in bytes_per_frame.iter().enumerate() {
        let out = dec
            .decode_frame(b)
            .unwrap_or_else(|e| panic!("decode_frame[{i}]: {e:?}"));
        assert_eq!(out.width, width);
        assert_eq!(out.height, height);
    }
}
