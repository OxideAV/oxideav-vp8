//! Black-box external-decoder validation of the Phase 1 silent
//! keyframe encoder.
//!
//! Builds a minimal IVF wrapper around a single emitted VP8 frame and
//! pipes it through `ffmpeg -c:v vp8 -f rawvideo`, then asserts ffmpeg
//! produced a YUV420P picture of the expected byte length. The exact
//! pixel content is implementation-determined (every macroblock is
//! coded as `mb_skip_coeff = 1` with `DC_PRED` luma + chroma, so the
//! decoder fills with whatever its §12 DC default produces — `127` for
//! the top-row / left-column boundary case); we only verify ffmpeg
//! accepts the bitstream and emits a structurally-correct frame.
//!
//! The test is gated on `ffmpeg` being present on `$PATH`; if it
//! isn't (e.g. CI image without ffmpeg installed) the test is skipped
//! via `eprintln! + return` rather than `#[ignore]` — keeping the
//! "guardrail #3: NEVER `#[ignore]`" rule honoured. `ffmpeg` is
//! available on every developer image and on the CI runners the
//! workspace uses, so the skip path is a defensive fallback rather
//! than a routine outcome.

use std::io::Write;
use std::process::{Command, Stdio};

use oxideav_vp8::{encode_silent_keyframe, SilentKeyframeParams};

/// Build a single-frame IVF container around `vp8_frame`. IVF is the
/// standard raw-VP8 wrapper: 32-byte DKIF header + per-frame (4-byte
/// LE size + 8-byte LE pts) prefix.
fn wrap_in_ivf(vp8_frame: &[u8], width: u16, height: u16) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(32 + 12 + vp8_frame.len());
    // DKIF header.
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes()); // version
    out.extend_from_slice(&32u16.to_le_bytes()); // header length
    out.extend_from_slice(b"VP80"); // FourCC
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes()); // timebase denominator (fps)
    out.extend_from_slice(&1u32.to_le_bytes()); // timebase numerator
    out.extend_from_slice(&1u32.to_le_bytes()); // frame count
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved

    // Per-frame header.
    let size = vp8_frame.len() as u32;
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // pts
    out.extend_from_slice(vp8_frame);
    out
}

/// Check whether `ffmpeg` is on `$PATH`. We do not run ffmpeg unless
/// it is.
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

/// Decode `ivf_bytes` through `ffmpeg -c:v vp8 -f rawvideo -` and
/// return the raw YUV420P bytes ffmpeg emitted on stdout.
fn decode_with_ffmpeg(ivf_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "ivf",
            "-c:v",
            "vp8",
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
        .map_err(|e| format!("spawn ffmpeg: {e}"))?;
    child
        .stdin
        .as_mut()
        .ok_or("ffmpeg stdin unavailable")?
        .write_all(ivf_bytes)
        .map_err(|e| format!("write ivf to ffmpeg stdin: {e}"))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("wait ffmpeg: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "ffmpeg returned {}: stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out.stdout)
}

/// Encode → wrap → decode-via-ffmpeg pipeline. Returns the YUV420P
/// bytes ffmpeg produced.
fn encode_and_external_decode(width: u32, height: u32) -> Result<Vec<u8>, String> {
    let frame = encode_silent_keyframe(SilentKeyframeParams::new(width, height))
        .map_err(|e| e.to_string())?;
    let ivf = wrap_in_ivf(&frame, width as u16, height as u16);
    decode_with_ffmpeg(&ivf)
}

#[test]
fn ffmpeg_accepts_silent_keyframe_16x16() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping external-decode test");
        return;
    }
    let yuv = encode_and_external_decode(16, 16).expect("ffmpeg must accept emitted frame");
    // 16×16 YUV420P = 256 Y + 64 U + 64 V = 384 bytes.
    assert_eq!(yuv.len(), 16 * 16 + 8 * 8 + 8 * 8);
}

#[test]
fn ffmpeg_accepts_silent_keyframe_64x64() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping external-decode test");
        return;
    }
    let yuv = encode_and_external_decode(64, 64).expect("ffmpeg must accept emitted frame");
    assert_eq!(yuv.len(), 64 * 64 + 32 * 32 + 32 * 32);
}

#[test]
fn ffmpeg_accepts_silent_keyframe_non_square() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping external-decode test");
        return;
    }
    // 48×32 — exercises non-square dimensions and a sub-MB chroma
    // boundary (24 chroma pixels).
    let yuv = encode_and_external_decode(48, 32).expect("ffmpeg must accept emitted frame");
    assert_eq!(yuv.len(), 48 * 32 + 24 * 16 + 24 * 16);
}
