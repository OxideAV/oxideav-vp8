//! Round-387 coverage of the §10 per-segment **loop-filter feature**
//! on the adaptive-quant keyframe path
//! ([`encode_keyframe_adaptive_quant_with_segment_lf_deltas`]) — the
//! second half of the §9.3 `update_segment_feature_data` block (the
//! quantiser half landed with the original AQ round).
//!
//! What this pins:
//!
//!   1. the §9.3 wire carries all four `loop_filter_update` values and
//!      round-trips through [`Vp8CodedHeader::parse`] slot-for-slot;
//!   2. encoder ↔ decoder **pixel lockstep**: the decoder resolves each
//!      MB's §15 level through the same §20.6 segment override the
//!      encoder's post-walk pass used;
//!   3. the deltas actually *do* something: the segment-LF picture
//!      differs from the flat-level picture on real content, and both
//!      still self-decode exactly;
//!   4. the §15 whole-frame skip mirrors the decoder: frame base level
//!      0 ⇒ identical pixels to the no-feature encode even with
//!      non-zero deltas on the wire;
//!   5. out-of-range deltas (|d| > 63) are rejected;
//!   6. black-box: `ffmpeg` decodes the segment-LF keyframe to
//!      byte-identical pixels vs our own decoder.

use std::io::Write as _;
use std::process::{Command, Stdio};

use oxideav_vp8::{
    decode_vp8, encode_keyframe_adaptive_quant_with_reconstruction,
    encode_keyframe_adaptive_quant_with_segment_lf_deltas, AdaptiveQuantConfig, EncodeError,
    I420Frame, Vp8CodedHeader, Vp8FrameHeader,
};

const W: usize = 96;
const H: usize = 96;

/// Mixed-activity source: flat sky band (segment 0), gentle gradient
/// (low segments), busy texture + hard edges (high segments) — content
/// that actually spreads macroblocks across all four variance segments.
fn source() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for r in 0..H {
        for c in 0..W {
            let v = if r < H / 4 {
                200 // flat band
            } else if r < H / 2 {
                (80 + (c / 2)) as i32 as u8 // gentle gradient
            } else {
                // busy texture with edges
                let t = ((r * 13 + c * 7) % 61) as i32 * 3 - 90;
                let e = if (c / 4) % 2 == 0 { 50 } else { -50 };
                (128 + t / 2 + e).clamp(0, 255) as u8
            };
            y[r * W + c] = v;
        }
    }
    let (cw, ch) = (W / 2, H / 2);
    let u = vec![118u8; cw * ch];
    let v = vec![132u8; cw * ch];
    (y, u, v)
}

fn aq_config(lf_level: u8) -> AdaptiveQuantConfig {
    AdaptiveQuantConfig {
        base_y_ac_qi: 40,
        loop_filter_level: lf_level,
        ..AdaptiveQuantConfig::default()
    }
}

const DELTAS: [i8; 4] = [16, 6, -6, -16];

#[test]
fn segment_lf_feature_round_trips_through_header_parser() {
    let (y, u, v) = source();
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let (bytes, _) =
        encode_keyframe_adaptive_quant_with_segment_lf_deltas(&frame, &aq_config(20), &DELTAS)
            .expect("segment-LF encode");

    let hdr = Vp8FrameHeader::parse(&bytes).expect("tag");
    let partition = &bytes
        [hdr.header_bytes_consumed..hdr.header_bytes_consumed + hdr.first_partition_size as usize];
    let coded = Vp8CodedHeader::parse(partition, true).expect("coded header");
    assert!(coded.segmentation_enabled);
    let seg = coded.update_segmentation.expect("update_segmentation");
    assert!(seg.update_segment_feature_data);
    assert!(!seg.segment_feature_mode_absolute, "delta mode");
    for (s, &delta) in DELTAS.iter().enumerate() {
        assert_eq!(
            seg.loop_filter_update[s],
            Some(i16::from(delta)),
            "segment {s} loop_filter_update"
        );
        assert!(seg.quantizer_update[s].is_some(), "quant feature intact");
    }
}

#[test]
fn segment_lf_encode_decodes_in_pixel_lockstep_and_changes_the_filtering() {
    let (y, u, v) = source();
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);

    let (bytes, planes) =
        encode_keyframe_adaptive_quant_with_segment_lf_deltas(&frame, &aq_config(20), &DELTAS)
            .expect("segment-LF encode");
    let dec = decode_vp8(&bytes).expect("decode");
    // 96×96 is MB-aligned: decoded picture == reconstruction planes.
    assert_eq!(dec.y, planes.y, "luma lockstep");
    assert_eq!(dec.u, planes.u, "U lockstep");
    assert_eq!(dec.v, planes.v, "V lockstep");

    // Same content, same base level, no segment feature ⇒ different
    // §15 output (the deltas must actually steer the filter).
    let (flat_bytes, flat_planes) =
        encode_keyframe_adaptive_quant_with_reconstruction(&frame, &aq_config(20))
            .expect("flat-LF encode");
    let flat_dec = decode_vp8(&flat_bytes).expect("decode flat");
    assert_eq!(flat_dec.y, flat_planes.y, "flat encode lockstep too");
    assert_ne!(
        dec.y, flat_dec.y,
        "non-zero segment LF deltas must change the filtered picture"
    );
}

#[test]
fn frame_level_zero_skips_the_filter_despite_deltas() {
    let (y, u, v) = source();
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);

    // Base level 0 + positive deltas: the §15 page-84 whole-frame skip
    // wins (mirrors the decoder's gate), so pixels match the
    // no-feature level-0 encode exactly even though the wires differ.
    let (bytes, planes) =
        encode_keyframe_adaptive_quant_with_segment_lf_deltas(&frame, &aq_config(0), &DELTAS)
            .expect("level-0 segment-LF encode");
    let (flat_bytes, flat_planes) =
        encode_keyframe_adaptive_quant_with_reconstruction(&frame, &aq_config(0))
            .expect("level-0 flat encode");
    assert_ne!(bytes, flat_bytes, "wires differ (feature travels)");
    assert_eq!(
        planes.y, flat_planes.y,
        "frame base level 0 skips the filter regardless of segment deltas"
    );
    let dec = decode_vp8(&bytes).expect("decode");
    assert_eq!(dec.y, planes.y, "lockstep holds on the skip path");
}

#[test]
fn out_of_range_deltas_are_rejected() {
    let (y, u, v) = source();
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let r = encode_keyframe_adaptive_quant_with_segment_lf_deltas(
        &frame,
        &aq_config(20),
        &[64, 0, 0, 0],
    );
    assert!(
        matches!(r, Err(EncodeError::LoopFilterLevelOutOfRange { value: 64 })),
        "delta magnitude 64 must be rejected, got {r:?}"
    );
    let r = encode_keyframe_adaptive_quant_with_segment_lf_deltas(
        &frame,
        &aq_config(20),
        &[0, -64, 0, 0],
    );
    assert!(
        matches!(r, Err(EncodeError::LoopFilterLevelOutOfRange { value: 64 })),
        "delta magnitude -64 must be rejected, got {r:?}"
    );
}

// ───────────────────────── ffmpeg black-box leg ─────────────────────────

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn wrap_in_ivf(vp8_frame: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + 12 + vp8_frame.len());
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"VP80");
    out.extend_from_slice(&(W as u16).to_le_bytes());
    out.extend_from_slice(&(H as u16).to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(vp8_frame.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(vp8_frame);
    out
}

#[test]
fn ffmpeg_cross_decodes_segment_lf_keyframe_byte_exact() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping black-box cross-decode");
        return;
    }
    let (y, u, v) = source();
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let (bytes, planes) =
        encode_keyframe_adaptive_quant_with_segment_lf_deltas(&frame, &aq_config(20), &DELTAS)
            .expect("segment-LF encode");
    let ivf = wrap_in_ivf(&bytes);

    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "ivf",
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
        .expect("spawn ffmpeg");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&ivf)
        .expect("write ivf");
    let out = child.wait_with_output().expect("ffmpeg run");
    assert!(
        out.status.success(),
        "ffmpeg must accept the segment-LF keyframe: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let frame_bytes = W * H + 2 * (W / 2) * (H / 2);
    assert_eq!(out.stdout.len(), frame_bytes, "one decoded picture");
    assert_eq!(
        &out.stdout[..W * H],
        planes.y.as_slice(),
        "ffmpeg luma must match our reconstruction byte-exactly"
    );
    assert_eq!(
        &out.stdout[W * H..W * H + (W / 2) * (H / 2)],
        planes.u.as_slice(),
        "ffmpeg U plane byte-exact"
    );
    assert_eq!(
        &out.stdout[W * H + (W / 2) * (H / 2)..],
        planes.v.as_slice(),
        "ffmpeg V plane byte-exact"
    );
}
