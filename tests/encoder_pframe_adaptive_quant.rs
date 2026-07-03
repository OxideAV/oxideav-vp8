//! Round-387 coverage of §9.3 / §10 **segment-based adaptive
//! quantisation on P-frames**
//! ([`oxideav_vp8::encode_p_frame_multi_ref_adaptive_quant`]) — the
//! inter mirror of the keyframe AQ path, previously a keyframe-only
//! feature.
//!
//! What this pins:
//!
//!   1. a P-frame carrying the full §9.3 `update_segmentation()` block
//!      (per-segment quant deltas + fitted `mb_segment_tree_probs` +
//!      per-MB `segment_id` prefix in the §11/§16 mode layer)
//!      self-decodes in **pixel lockstep** through [`Vp8DecoderState`]
//!      — the decoder's per-segment dequant resolution sees exactly
//!      the factors the encoder priced and quantised with;
//!   2. the header round-trips through [`Vp8CodedHeader::parse`]
//!      feature-for-feature;
//!   3. the optional per-segment loop-filter deltas ride along and
//!      keep lockstep (the §20.6 inter segment override);
//!   4. the layer composes with the full [`InterCodingOptions`] toggle
//!      set (intra pick + auto LF + fitted §13.4 updates);
//!   5. out-of-range `lf_delta` magnitudes are rejected;
//!   6. black-box: `ffmpeg` decodes the K + segmented-P stream to
//!      byte-identical pixels vs our own decoder.

use std::io::Write as _;
use std::process::{Command, Stdio};

use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref_adaptive_quant,
    encode_p_frame_multi_ref_with_refresh, EncodeError, I420Frame, InterCodingOptions,
    InterSegmentationConfig, KeyframeParams, KeyframePlanes, RefreshControls, Vp8CodedHeader,
    Vp8DecoderState, Vp8FrameHeader,
};

const W: usize = 96;
const H: usize = 96;

/// Mixed-activity scene shifted by `dx`: flat band, gentle gradient,
/// busy texture — spreads macroblocks across all four variance
/// segments, with trackable motion for the P-frame.
fn source(dx: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for r in 0..H {
        for c in 0..W {
            let cc = c + dx;
            let v = if r < H / 4 {
                200
            } else if r < H / 2 {
                (80 + (cc / 2) % 100) as u8
            } else {
                let t = ((r * 13 + cc * 7) % 61) as i32 * 3 - 90;
                let e = if (cc / 4) % 2 == 0 { 50 } else { -50 };
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

fn params() -> KeyframeParams {
    KeyframeParams {
        y_ac_qi: 44,
        loop_filter_level: 16,
        ..KeyframeParams::default()
    }
}

/// Encode the shared keyframe; return (bytes, reconstruction).
fn keyed_reference() -> (Vec<u8>, KeyframePlanes) {
    let (y, u, v) = source(0);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    encode_keyframe_with_reconstruction(&frame, &params()).expect("keyframe")
}

fn seg_config(lf: Option<[i8; 4]>) -> InterSegmentationConfig {
    InterSegmentationConfig {
        quant_delta: [10, 2, -4, -10],
        lf_delta: lf,
        ..InterSegmentationConfig::default()
    }
}

/// Parse the coded header of an inter frame.
fn inter_coded_header(bytes: &[u8]) -> Vp8CodedHeader {
    let hdr = Vp8FrameHeader::parse(bytes).expect("tag");
    let part = &bytes
        [hdr.header_bytes_consumed..hdr.header_bytes_consumed + hdr.first_partition_size as usize];
    Vp8CodedHeader::parse(part, false).expect("coded header")
}

#[test]
fn segmented_p_frame_decodes_in_pixel_lockstep() {
    let (kf_bytes, kf) = keyed_reference();
    let (y, u, v) = source(3);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);

    let (bytes, planes) = encode_p_frame_multi_ref_adaptive_quant(
        &frame,
        &kf,
        None,
        None,
        &params(),
        &RefreshControls::default(),
        &seg_config(None),
        &InterCodingOptions::default(),
    )
    .expect("segmented P-frame");

    // Wire actually differs from the unsegmented encode.
    let (plain, _) = encode_p_frame_multi_ref_with_refresh(
        &frame,
        &kf,
        None,
        None,
        &params(),
        &RefreshControls::default(),
    )
    .expect("plain P-frame");
    assert_ne!(bytes, plain, "segmentation must change the wire");

    let mut dec = Vp8DecoderState::new();
    dec.decode_frame(&kf_bytes).expect("keyframe decodes");
    let dp = dec.decode_frame(&bytes).expect("segmented P decodes");
    // 96×96 is MB-aligned: decoded picture == reconstruction planes.
    assert_eq!(dp.y, planes.y, "luma lockstep");
    assert_eq!(dp.u, planes.u, "U lockstep");
    assert_eq!(dp.v, planes.v, "V lockstep");
}

#[test]
fn segmented_p_frame_header_round_trips() {
    let (_, kf) = keyed_reference();
    let (y, u, v) = source(3);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let cfg = seg_config(Some([6, 2, -2, -6]));

    let (bytes, _) = encode_p_frame_multi_ref_adaptive_quant(
        &frame,
        &kf,
        None,
        None,
        &params(),
        &RefreshControls::default(),
        &cfg,
        &InterCodingOptions::default(),
    )
    .expect("segmented P-frame");

    let coded = inter_coded_header(&bytes);
    assert!(coded.segmentation_enabled);
    let seg = coded.update_segmentation.expect("update_segmentation");
    assert!(seg.update_mb_segmentation_map, "map travels every frame");
    assert!(seg.update_segment_feature_data);
    assert!(!seg.segment_feature_mode_absolute, "delta mode");
    for s in 0..4 {
        assert_eq!(
            seg.quantizer_update[s],
            Some(i16::from(cfg.quant_delta[s])),
            "segment {s} quantizer_update"
        );
        assert_eq!(
            seg.loop_filter_update[s],
            Some(i16::from(cfg.lf_delta.unwrap()[s])),
            "segment {s} loop_filter_update"
        );
    }
    // The fitted tree probs are explicit (all three nodes written).
    for n in 0..3 {
        assert!(seg.segment_prob[n].is_some(), "tree node {n} prob written");
    }
}

#[test]
fn segmented_p_frame_with_lf_deltas_stays_lockstep() {
    let (kf_bytes, kf) = keyed_reference();
    let (y, u, v) = source(3);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);

    let (bytes, planes) = encode_p_frame_multi_ref_adaptive_quant(
        &frame,
        &kf,
        None,
        None,
        &params(),
        &RefreshControls::default(),
        &seg_config(Some([10, 4, -4, -10])),
        &InterCodingOptions::default(),
    )
    .expect("segmented P-frame with LF deltas");

    let mut dec = Vp8DecoderState::new();
    dec.decode_frame(&kf_bytes).expect("keyframe");
    let dp = dec.decode_frame(&bytes).expect("P decodes");
    assert_eq!(dp.y, planes.y, "luma lockstep with segment LF override");
    assert_eq!(dp.u, planes.u, "U lockstep");
    assert_eq!(dp.v, planes.v, "V lockstep");
}

#[test]
fn segmentation_composes_with_all_coding_options() {
    let (kf_bytes, kf) = keyed_reference();
    let (y, u, v) = source(3);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let options = InterCodingOptions {
        intra_pick: true,
        auto_loop_filter: true,
        fitted_token_prob_updates: true,
    };
    let no_fit = InterCodingOptions {
        fitted_token_prob_updates: false,
        ..options
    };

    let (baseline, _) = encode_p_frame_multi_ref_adaptive_quant(
        &frame,
        &kf,
        None,
        None,
        &params(),
        &RefreshControls::default(),
        &seg_config(Some([8, 2, -2, -8])),
        &no_fit,
    )
    .expect("no-fit baseline");
    let (bytes, planes) = encode_p_frame_multi_ref_adaptive_quant(
        &frame,
        &kf,
        None,
        None,
        &params(),
        &RefreshControls::default(),
        &seg_config(Some([8, 2, -2, -8])),
        &options,
    )
    .expect("all toggles + segmentation");
    assert!(
        bytes.len() <= baseline.len(),
        "fitted pass must never grow the wire ({} > {})",
        bytes.len(),
        baseline.len()
    );

    let mut dec = Vp8DecoderState::new();
    dec.decode_frame(&kf_bytes).expect("keyframe");
    let dp = dec.decode_frame(&bytes).expect("P decodes");
    assert_eq!(dp.y, planes.y, "lockstep under the full toggle set");
}

#[test]
fn out_of_range_lf_delta_is_rejected() {
    let (_, kf) = keyed_reference();
    let (y, u, v) = source(3);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let r = encode_p_frame_multi_ref_adaptive_quant(
        &frame,
        &kf,
        None,
        None,
        &params(),
        &RefreshControls::default(),
        &seg_config(Some([64, 0, 0, 0])),
        &InterCodingOptions::default(),
    );
    assert!(
        matches!(r, Err(EncodeError::LoopFilterLevelOutOfRange { value: 64 })),
        "lf_delta magnitude 64 must be rejected, got {r:?}"
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

fn wrap_stream_in_ivf(frames: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"VP80");
    out.extend_from_slice(&(W as u16).to_le_bytes());
    out.extend_from_slice(&(H as u16).to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for (i, f) in frames.iter().enumerate() {
        out.extend_from_slice(&(f.len() as u32).to_le_bytes());
        out.extend_from_slice(&(i as u64).to_le_bytes());
        out.extend_from_slice(f);
    }
    out
}

#[test]
fn ffmpeg_cross_decodes_segmented_p_frame_byte_exact() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping black-box cross-decode");
        return;
    }
    let (kf_bytes, kf) = keyed_reference();
    let (y, u, v) = source(3);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let (p_bytes, p_planes) = encode_p_frame_multi_ref_adaptive_quant(
        &frame,
        &kf,
        None,
        None,
        &params(),
        &RefreshControls::default(),
        &seg_config(Some([10, 4, -4, -10])),
        &InterCodingOptions::default(),
    )
    .expect("segmented P-frame");

    let ivf = wrap_stream_in_ivf(&[&kf_bytes, &p_bytes]);
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
        "ffmpeg must accept the segmented stream: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let frame_bytes = W * H + 2 * (W / 2) * (H / 2);
    assert_eq!(out.stdout.len(), 2 * frame_bytes, "two decoded pictures");
    // Frame 1 (the segmented P) must match our reconstruction exactly.
    let base = frame_bytes;
    assert_eq!(
        &out.stdout[base..base + W * H],
        p_planes.y.as_slice(),
        "ffmpeg P-frame luma byte-exact"
    );
    assert_eq!(
        &out.stdout[base + W * H..base + W * H + (W / 2) * (H / 2)],
        p_planes.u.as_slice(),
        "ffmpeg P-frame U byte-exact"
    );
    assert_eq!(
        &out.stdout[base + W * H + (W / 2) * (H / 2)..],
        p_planes.v.as_slice(),
        "ffmpeg P-frame V byte-exact"
    );
}
