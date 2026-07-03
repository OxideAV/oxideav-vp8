//! Round-387 coverage of the [`Vp8AltrefStreamEncoder`] per-frame
//! feature toggles ([`AltrefStreamConfig::auto_loop_filter`] /
//! [`AltrefStreamConfig::fitted_token_prob_updates`] /
//! [`AltrefStreamConfig::intra_pick`]) and the
//! [`Vp8Encoder::encode_sequence`] batch front door that now enables
//! them.
//!
//! What this pins:
//!
//!   1. a fully-toggled anchored stream (auto-LF + fitted §13.4
//!      updates + §11 intra pick) keeps the exact packet structure and
//!      decodes packet-for-packet through [`Vp8DecoderState`] with the
//!      correct visibility mirror — the §9.7 slot ladder survives every
//!      per-frame feature;
//!   2. the §13.4 fitter measurably shrinks the stream: with only
//!      `fitted_token_prob_updates` toggled, total bytes come in below
//!      the all-off baseline on textured content;
//!   3. auto-LF actually writes non-zero §9.4 levels chosen per frame
//!      (readable back off the wire) even though `params` asks for
//!      level 0;
//!   4. `encode_sequence` produces a decodable anchored stream and
//!      reports honest stats;
//!   5. black-box: the toggled stream (invisible anchors included)
//!      wrapped in IVF decodes through `ffmpeg` to **byte-identical**
//!      pixels vs our own decoder, frame for visible frame.
//!
//! The ffmpeg leg is skipped (eprintln + return, never `#[ignore]`)
//! when the binary is absent.

use std::io::Write as _;
use std::process::{Command, Stdio};

use oxideav_vp8::{
    AltrefPacketKind, AltrefStreamConfig, AltrefStreamPacket, ArnrConfig, I420Frame,
    KeyframeParams, Vp8AltrefStreamEncoder, Vp8DecoderState, Vp8Encoder, Vp8EncoderConfig,
};

const W: usize = 64;
const H: usize = 64;
const FRAMES: usize = 12;

/// Slowly-translating textured scene with deterministic per-frame
/// "noise" (position-and-frame-keyed hash), the ARNR content class.
fn source_frame(f: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for r in 0..H {
        for c in 0..W {
            let sr = r + f; // slow diagonal drift
            let sc = c + 2 * f;
            let tex = ((sr * 7 + sc * 3) % 97) as i32 + ((sc * 13) % 31) as i32;
            let noise = (((r * 31 + c * 17 + f * 101) % 11) as i32) - 5;
            y[r * W + c] = (70 + tex + noise).clamp(0, 255) as u8;
        }
    }
    let (cw, ch) = (W / 2, H / 2);
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            u[r * cw + c] = (100 + ((r + c + f) % 50)) as u8;
            v[r * cw + c] = (130 + ((r * 2 + c) % 40)) as u8;
        }
    }
    (y, u, v)
}

fn encode_stream(config: AltrefStreamConfig) -> Vec<AltrefStreamPacket> {
    let mut enc = Vp8AltrefStreamEncoder::new(config).expect("window >= 1");
    let mut packets = Vec::new();
    for f in 0..FRAMES {
        let (y, u, v) = source_frame(f);
        let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
        packets.extend(enc.push_frame(&frame).expect("push_frame"));
    }
    packets.extend(enc.finish().expect("finish"));
    packets
}

fn base_config() -> AltrefStreamConfig {
    AltrefStreamConfig {
        params: KeyframeParams {
            y_ac_qi: 44,
            ..KeyframeParams::default()
        },
        keyframe_interval: 0,
        altref_window: 4,
        arnr: ArnrConfig::default(),
        // Keep scene-cut detection out of the way — the drift source
        // is continuous.
        scene_cut_mad_threshold: 0.0,
        ..AltrefStreamConfig::default()
    }
}

/// Decode every packet in order, asserting the visibility mirror, and
/// return the visible pictures (in display order).
fn decode_all(packets: &[AltrefStreamPacket]) -> Vec<oxideav_vp8::Vp8Frame> {
    let mut dec = Vp8DecoderState::new();
    let mut visible = Vec::new();
    let mut next_src = 0u64;
    for (i, p) in packets.iter().enumerate() {
        let pic = dec
            .decode_frame(&p.bytes)
            .unwrap_or_else(|e| panic!("packet {i} ({:?}) must decode: {e:?}", p.kind));
        assert_eq!(
            dec.last_frame_shown(),
            Some(p.is_visible()),
            "visibility mirror drift at packet {i}"
        );
        if let Some(src) = p.source_index {
            assert_eq!(src, next_src, "visible order drift at packet {i}");
            next_src += 1;
            visible.push(pic);
        }
    }
    visible
}

fn luma_psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

#[test]
fn fully_toggled_stream_keeps_structure_and_decodes_lockstep() {
    let config = AltrefStreamConfig {
        auto_loop_filter: true,
        fitted_token_prob_updates: true,
        intra_pick: true,
        ..base_config()
    };
    let packets = encode_stream(config);

    // 12 frames / window 4 → 3 groups → 1 K + 3 anchors + 11 P = 15.
    assert_eq!(packets.len(), 15, "packet structure must be preserved");
    assert_eq!(packets[0].kind, AltrefPacketKind::Key);
    let anchors = packets
        .iter()
        .filter(|p| p.kind == AltrefPacketKind::AltrefUpdate)
        .count();
    assert_eq!(anchors, 3, "one invisible anchor per group");

    let visible = decode_all(&packets);
    assert_eq!(visible.len(), FRAMES);
    for (f, pic) in visible.iter().enumerate() {
        let (sy, _, _) = source_frame(f);
        let psnr = luma_psnr(&sy, &pic.y);
        assert!(
            psnr >= 30.0,
            "visible frame {f} PSNR-Y {psnr:.2} dB below 30 dB floor"
        );
    }
}

#[test]
fn fitted_prob_updates_shrink_the_stream() {
    let baseline: usize = encode_stream(base_config())
        .iter()
        .map(|p| p.bytes.len())
        .sum();
    let fitted: usize = encode_stream(AltrefStreamConfig {
        fitted_token_prob_updates: true,
        ..base_config()
    })
    .iter()
    .map(|p| p.bytes.len())
    .sum();
    eprintln!("altref stream bytes: baseline {baseline}, fitted §13.4 updates {fitted}");
    assert!(
        fitted < baseline,
        "the §13.4 fitter must shrink the anchored stream ({fitted} >= {baseline})"
    );
}

#[test]
fn auto_loop_filter_writes_selected_levels_onto_the_wire() {
    // params ask for level 0 (no filtering); the RD selector must still
    // be free to pick per-frame non-zero levels — proving the header
    // value is chosen, not copied.
    let packets = encode_stream(AltrefStreamConfig {
        auto_loop_filter: true,
        ..base_config()
    });
    let mut nonzero_levels = 0usize;
    for p in &packets {
        // §9.1 frame tag: 3 bytes (+7 more on key frames). The §9.4
        // loop_filter_level lives in the boolean-coded header — read it
        // back through the crate's own parser.
        let parsed = oxideav_vp8::parse_header(&p.bytes).expect("tag parses");
        let is_key = parsed.keyframe.is_some();
        if decode_lf_level(&p.bytes, is_key) != 0 {
            nonzero_levels += 1;
        }
    }
    assert!(
        nonzero_levels > 0,
        "auto-LF must have selected a non-zero §9.4 level on at least one frame"
    );
    // And the stream still decodes in lockstep.
    let visible = decode_all(&packets);
    assert_eq!(visible.len(), FRAMES);
}

/// Pull the §9.4 `loop_filter_level` back off one frame's wire via the
/// crate's own coded-header parser. The first (control) partition
/// starts after the 3-byte §9.1 tag (plus the 7-byte start-code /
/// dimensions block on key frames).
fn decode_lf_level(bytes: &[u8], is_key: bool) -> u8 {
    let first_partition = if is_key { &bytes[10..] } else { &bytes[3..] };
    let header =
        oxideav_vp8::Vp8CodedHeader::parse(first_partition, is_key).expect("coded header parses");
    header.loop_filter_level
}

#[test]
fn encode_sequence_batch_front_door_is_decodable_and_counted() {
    let mut enc = Vp8Encoder::new(Vp8EncoderConfig {
        qindex: 44,
        alt_ref_interval: 4,
        lookahead_window: 8,
        golden_interval: 0, // key frame 0 only
        auto_loop_filter: true,
        ..Vp8EncoderConfig::default()
    });
    let sources: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..FRAMES).map(source_frame).collect();
    let frames: Vec<I420Frame<'_>> = sources
        .iter()
        .map(|(y, u, v)| I420Frame::packed(W as u32, H as u32, y, u, v))
        .collect();
    let packets = enc.encode_sequence(&frames).expect("encode_sequence");
    assert_eq!(packets.len(), 15, "12 sources / window 4 → 15 packets");
    assert_eq!(enc.stats().frames_encoded, 15);
    assert_eq!(enc.stats().keyframes_emitted, 1);
    assert_eq!(
        enc.stats().bytes_emitted,
        packets.iter().map(|p| p.bytes.len() as u64).sum::<u64>()
    );
    let visible = decode_all(&packets);
    assert_eq!(visible.len(), FRAMES);
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

/// Wrap a whole packet stream (invisible anchors included) in IVF.
/// `frame_count` in the header counts every packet — IVF is a plain
/// length-prefixed framing, one record per elementary-stream frame.
fn wrap_stream_in_ivf(packets: &[AltrefStreamPacket]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"VP80");
    out.extend_from_slice(&(W as u16).to_le_bytes());
    out.extend_from_slice(&(H as u16).to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for (i, p) in packets.iter().enumerate() {
        out.extend_from_slice(&(p.bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(i as u64).to_le_bytes());
        out.extend_from_slice(&p.bytes);
    }
    out
}

#[test]
fn ffmpeg_cross_decodes_fully_toggled_stream_byte_exact() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping black-box cross-decode");
        return;
    }
    let packets = encode_stream(AltrefStreamConfig {
        auto_loop_filter: true,
        fitted_token_prob_updates: true,
        intra_pick: true,
        ..base_config()
    });
    let ivf = wrap_stream_in_ivf(&packets);

    // ffmpeg decodes the whole stream; invisible frames (show_frame=0)
    // produce no output picture, so with `-fps_mode passthrough` (no
    // CFR duplication over the invisible frames' timestamp gaps) the
    // rawvideo stream carries exactly the visible frames in order.
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "ivf",
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
        "ffmpeg must accept the toggled stream: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let frame_bytes = W * H + 2 * (W / 2) * (H / 2);
    assert_eq!(
        out.stdout.len(),
        frame_bytes * FRAMES,
        "ffmpeg must emit exactly the {FRAMES} visible frames"
    );

    // Byte-exact agreement with our own stateful decode, per plane.
    let ours = decode_all(&packets);
    for (f, pic) in ours.iter().enumerate() {
        let base = f * frame_bytes;
        let ff_y = &out.stdout[base..base + W * H];
        let ff_u = &out.stdout[base + W * H..base + W * H + (W / 2) * (H / 2)];
        let ff_v = &out.stdout[base + W * H + (W / 2) * (H / 2)..base + frame_bytes];
        assert_eq!(pic.y.as_slice(), ff_y, "luma mismatch on visible frame {f}");
        assert_eq!(pic.u.as_slice(), ff_u, "U mismatch on visible frame {f}");
        assert_eq!(pic.v.as_slice(), ff_v, "V mismatch on visible frame {f}");
    }
}
