//! Black-box resolution-switch splices: `ffmpeg -c:v vp8` encodes
//! independent IVF segments at different sizes; their VP8 frames are
//! spliced into one elementary stream (each segment opens with a §9.1 key
//! frame carrying its size) and decoded through the registry
//! [`Vp8Decoder`]. The oracle is `ffmpeg`'s own decode of each segment on
//! its own (`-fps_mode passthrough -f rawvideo -pix_fmt yuv420p`),
//! concatenated.
//!
//! Pinned per splice: every shown picture byte-identical to the oracle,
//! the shown-picture count equal to the oracle's, and exactly one
//! `OutputParamsWatch` change (counter and in-band event both) per size
//! switch, each event immediately before the first picture of its size.
//! The two-pass `-auto-alt-ref 1` leg carries §9.1 `show_frame = 0`
//! altref frames, which must not surface as pictures.
//!
//! Skips (`eprintln!` + return, never `#[ignore]`) when `ffmpeg` or its
//! VP8 encoder is absent.

#![cfg(feature = "registry")]

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Error, Frame, OutputParamsWatch, Packet, TimeBase,
};
use oxideav_vp8::Vp8Decoder;

fn vp8_encoder_available() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .stderr(Stdio::null())
        .output()
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("(codec vp8)"))
        .unwrap_or(false)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "oxideav-vp8-resswitch-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// One `ffmpeg`-encoded IVF segment of `testsrc` at `w × h`.
fn encode_segment(dir: &Path, name: &str, w: u32, h: u32, frames: u32, altref: bool) -> Vec<u8> {
    let out = dir.join(format!("{name}.ivf"));
    let src = format!("testsrc=size={w}x{h}:rate=10");
    let frames = frames.to_string();
    let run = |extra: &[&str]| {
        let status = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi"])
            .args(["-i", &src, "-frames:v", &frames])
            .args(["-c:v", "vp8", "-b:v", "300k"])
            .args(extra)
            .args(["-f", "ivf"])
            .arg(&out)
            .status()
            .expect("spawn ffmpeg");
        assert!(status.success(), "ffmpeg encode {name} failed");
    };
    if altref {
        let log = dir.join(format!("{name}-pass"));
        let log = log.to_str().unwrap();
        for pass in ["1", "2"] {
            run(&[
                "-auto-alt-ref",
                "1",
                "-lag-in-frames",
                "16",
                "-pass",
                pass,
                "-passlogfile",
                log,
            ]);
        }
    } else {
        run(&[]);
    }
    std::fs::read(&out).expect("read segment")
}

/// `ffmpeg`'s decode of one IVF segment as packed I420.
fn oracle_decode(ivf: &[u8]) -> Vec<u8> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "ivf",
            "-i",
            "pipe:0",
        ])
        // Passthrough: one output picture per decoded shown picture (no
        // constant-rate duplication into the timestamp gap a hidden frame
        // leaves behind).
        .args(["-fps_mode", "passthrough"])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn ffmpeg");
    let mut stdin = child.stdin.take().unwrap();
    let input = ivf.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
    });
    let out = child.wait_with_output().expect("ffmpeg decode");
    writer.join().unwrap();
    assert!(out.status.success(), "ffmpeg decode failed");
    out.stdout
}

fn ivf_frames(ivf: &[u8]) -> Vec<Vec<u8>> {
    let hdr = oxideav_vp8::ivf::parse_header(ivf).expect("IVF header");
    assert_eq!(&hdr.fourcc, b"VP80");
    let mut off = oxideav_vp8::ivf::IVF_HEADER_LEN;
    let mut out = Vec::new();
    while off < ivf.len() {
        let fh = oxideav_vp8::ivf::parse_frame_header(&ivf[off..]).expect("frame header");
        off += oxideav_vp8::ivf::IVF_FRAME_HEADER_LEN;
        out.push(ivf[off..off + fh.size as usize].to_vec());
        off += fh.size as usize;
    }
    out
}

struct Segment {
    w: u32,
    h: u32,
    frames: Vec<Vec<u8>>,
    oracle: Vec<u8>,
}

fn segment(dir: &Path, name: &str, (w, h): (u32, u32), n: u32, altref: bool) -> Segment {
    let ivf = encode_segment(dir, name, w, h, n, altref);
    let oracle = oracle_decode(&ivf);
    Segment {
        w,
        h,
        frames: ivf_frames(&ivf),
        oracle,
    }
}

/// Splice the segments, decode them through the trait, and check the
/// whole contract against the oracle. Returns the number of hidden
/// (§9.1 show_frame = 0) frames the splice carried.
fn check_splice(segments: &[Segment]) -> usize {
    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    let mut watch = OutputParamsWatch::new();
    let mut event_watch = OutputParamsWatch::new();
    let mut stream = CodecParameters::video(CodecId::new("vp8"));
    let mut event_stream = CodecParameters::video(CodecId::new("vp8"));
    let mut hidden = 0;
    let mut pts = 0i64;
    // (segment index, decoded bytes) per shown picture, in order; events
    // recorded as the index of the picture they precede.
    let mut shown: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut events: Vec<usize> = Vec::new();
    let mut counter_changes = 0;
    let mut first = true;

    for (si, seg) in segments.iter().enumerate() {
        for f in &seg.frames {
            if f[0] & 0x10 == 0 {
                hidden += 1;
            }
            let mut pkt = Packet::new(0, TimeBase::new(1, 10), f.clone());
            pkt.pts = Some(pts);
            pts += 1;
            dec.send_packet(&pkt).expect("send_packet");
            loop {
                let frame = match dec.receive_frame() {
                    Ok(fr) => fr,
                    Err(Error::NeedMore) => break,
                    Err(e) => panic!("segment {si}: decode error {e:?}"),
                };
                let changed = watch.observe(&dec, &mut stream);
                let op = dec.output_params().unwrap();
                match &frame {
                    Frame::StreamParams(p) => {
                        assert!(changed, "counter moved with the event");
                        assert_eq!((p.width, p.height), (op.width, op.height));
                        assert_eq!((p.width, p.height), (Some(seg.w), Some(seg.h)));
                        assert!(event_watch.observe_frame(&frame, &mut event_stream));
                        counter_changes += 1;
                        events.push(shown.len());
                    }
                    Frame::Video(v) => {
                        if first {
                            assert!(changed, "first observe resolves the shape");
                            event_stream.merge_output_params(&op);
                            first = false;
                        } else {
                            assert!(!changed, "pictures never report a change");
                        }
                        assert_eq!((op.width, op.height), (Some(seg.w), Some(seg.h)));
                        let mut bytes = Vec::new();
                        for plane in &v.planes {
                            bytes.extend_from_slice(&plane.data);
                        }
                        shown.push((si, bytes));
                    }
                    other => panic!("unexpected {other:?}"),
                }
                assert_eq!(stream.width, Some(seg.w));
                assert_eq!(event_stream.width, Some(seg.w));
            }
        }
    }

    // Exactly one change per size switch, each right before the first
    // picture of the new segment.
    let mut want_events = Vec::new();
    let mut idx = 0;
    for (si, seg) in segments.iter().enumerate() {
        let frame_len = (seg.w * seg.h + 2 * seg.w.div_ceil(2) * seg.h.div_ceil(2)) as usize;
        let n = seg.oracle.len() / frame_len;
        assert_eq!(seg.oracle.len() % frame_len, 0);
        if si > 0 && (seg.w, seg.h) != (segments[si - 1].w, segments[si - 1].h) {
            want_events.push(idx);
        }
        // Byte-identical against the oracle, shown count included.
        let ours: Vec<u8> = shown
            .iter()
            .filter(|(s, _)| *s == si)
            .flat_map(|(_, b)| b.iter().copied())
            .collect();
        assert_eq!(
            ours.len(),
            seg.oracle.len(),
            "segment {si}: shown-picture count ({} oracle pictures)",
            n
        );
        assert!(ours == seg.oracle, "segment {si}: byte-identical to oracle");
        idx += n;
    }
    assert_eq!(events, want_events, "event positions");
    assert_eq!(counter_changes, want_events.len());
    assert_eq!(dec.output_params_generation(), want_events.len() as u64);
    hidden
}

#[test]
fn ffmpeg_three_segment_size_splice() {
    if !vp8_encoder_available() {
        eprintln!("ffmpeg VP8 encoder not available — skipping");
        return;
    }
    let dir = scratch("three");
    let segs = [
        segment(&dir, "a", (64, 48), 6, false),
        segment(&dir, "b", (96, 80), 6, false),
        // Odd, non-macroblock-aligned size: exercises the visible crop.
        segment(&dir, "c", (50, 38), 5, false),
        segment(&dir, "d", (96, 80), 4, false),
    ];
    let hidden = check_splice(&segs);
    assert_eq!(hidden, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ffmpeg_same_size_splice_has_no_change() {
    if !vp8_encoder_available() {
        eprintln!("ffmpeg VP8 encoder not available — skipping");
        return;
    }
    let dir = scratch("same");
    let segs = [
        segment(&dir, "a", (64, 48), 4, false),
        segment(&dir, "b", (64, 48), 4, false),
    ];
    check_splice(&segs);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ffmpeg_auto_alt_ref_splice_hides_invisible_frames() {
    if !vp8_encoder_available() {
        eprintln!("ffmpeg VP8 encoder not available — skipping");
        return;
    }
    let dir = scratch("arf");
    let segs = [
        segment(&dir, "a", (96, 80), 20, true),
        segment(&dir, "b", (64, 48), 20, true),
    ];
    let hidden = check_splice(&segs);
    assert!(hidden >= 1, "two-pass auto-alt-ref must emit hidden frames");
    let _ = std::fs::remove_dir_all(&dir);
}
