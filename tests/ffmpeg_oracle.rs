//! Cross-validation of the VP8 codec against `ffmpeg` as a black-box
//! oracle.
//!
//! The crate's existing `tests/encoder_external_decode.rs` only pipes
//! the Phase-1 silent keyframe (`encode_silent_keyframe`) through
//! `ffmpeg` to verify it accepts the bitstream length. This file
//! extends that coverage in two directions:
//!
//! 1. **Our encode → ffmpeg decode** — drives the *real* `encode_keyframe`
//!    path (the §11 + §13 + §14 RD encoder) on a synthetic gradient
//!    source, wraps the emitted bytes in an IVF container, and pipes
//!    them through `ffmpeg -i pipe:0 -f rawvideo -pix_fmt yuv420p
//!    pipe:1`. The recovered YUV420P planes must match the source's
//!    dimensions and clear PSNR-Y ≥ 30 dB versus the source.
//!
//! 2. **ffmpeg encode → our decode** — pipes the same synthetic source
//!    through `ffmpeg -f rawvideo -c:v vp8 -f ivf pipe:1`, unwraps the
//!    IVF using the crate's own `ivf::parse_header` /
//!    `ivf::parse_frame_header`, and walks every frame through
//!    `Vp8DecoderState::decode_frame`. Each decoded picture must clear
//!    PSNR-Y ≥ 25 dB versus the source (ffmpeg's vp8 encoder is lossy;
//!    the floor is the round target, not a hard spec number).
//!
//! Both directions skip via `eprintln! + return` when `ffmpeg` isn't on
//! `$PATH` — never `#[ignore]`, per the crate's guardrail #3. `ffmpeg`
//! is the black-box validator the workspace allows for VP8 (RFC 6386);
//! its *source* stays off-limits, only its stdout / stderr / exit code
//! are consulted.
//!
//! No other external tool, library, or web resource is touched.

use std::io::Write;
use std::process::{Command, Stdio};

use oxideav_vp8::ivf::{
    parse_frame_header, parse_header, write_frame, write_header, IvfHeader, IVF_FRAME_HEADER_LEN,
    IVF_HEADER_LEN,
};
use oxideav_vp8::{
    decode_vp8, encode_keyframe, I420Frame, KeyframeParams, Vp8DecodedFrame, Vp8DecoderState,
};

// ───────────────────── ffmpeg availability + synthetic source ─────────────

/// `ffmpeg --version` succeeds on `$PATH`.
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

/// Source I420 picture with tightly-packed planes.
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

    /// Concatenated YUV420P planar bytes — what `ffmpeg -f rawvideo`
    /// consumes on `pipe:0` and emits on `pipe:1`.
    fn yuv420p_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.y.len() + self.u.len() + self.v.len());
        out.extend_from_slice(&self.y);
        out.extend_from_slice(&self.u);
        out.extend_from_slice(&self.v);
        out
    }
}

/// Synthetic gradient + flat-square I420 source, the same shape the
/// crate's other roundtrip tests use.
fn synthetic_source(width: u32, height: u32) -> Source {
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;

    let mut y = vec![0u8; w * h];
    for (row, chunk) in y.chunks_mut(w).enumerate() {
        for (col, px) in chunk.iter_mut().enumerate() {
            let mut v = ((col * 256 / w + row * 256 / h) / 2) as u8;
            if col >= w / 4 && col < w * 3 / 4 && row >= h / 4 && row < h * 3 / 4 {
                v = 128;
            }
            *px = v;
        }
    }

    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            u[row * cw + col] = (120 + (col * 16 / cw)) as u8;
            v[row * cw + col] = (130 + (row * 16 / ch)) as u8;
        }
    }

    Source {
        width,
        height,
        y,
        u,
        v,
    }
}

// ───────────────────── PSNR helpers ─────────────────────────────────────

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

fn luma_psnr(src: &[u8], dec: &[u8]) -> f64 {
    let mse = plane_mse(src, dec);
    if mse <= f64::EPSILON {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

// ─────────────────── Direction A: our encode → ffmpeg decode ────────────

/// Wrap a single VP8 keyframe in an IVF container using the crate's own
/// `ivf` module — exercising the standalone API directly.
fn wrap_in_ivf(vp8_frame: &[u8], width: u32, height: u32) -> Vec<u8> {
    let hdr = IvfHeader::vp8(width, height, 30, 1);
    let mut out = write_header(&hdr);
    write_frame(&mut out, 0, vp8_frame);
    out
}

/// Decode IVF bytes through ffmpeg and return the raw YUV420P payload.
fn ffmpeg_decode_ivf(ivf_bytes: &[u8]) -> Result<Vec<u8>, String> {
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

fn our_encode_ffmpeg_decode(width: u32, height: u32) -> Result<f64, String> {
    let src = synthetic_source(width, height);
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let vp8 =
        encode_keyframe(&src.frame(), &params).map_err(|e| format!("encode_keyframe: {e}"))?;
    let ivf = wrap_in_ivf(&vp8, width, height);
    let yuv = ffmpeg_decode_ivf(&ivf)?;

    let y_size = (width * height) as usize;
    let uv_size = (width.div_ceil(2) * height.div_ceil(2)) as usize;
    let expected_len = y_size + 2 * uv_size;
    if yuv.len() != expected_len {
        return Err(format!(
            "ffmpeg emitted {} bytes, expected {expected_len} ({}×{})",
            yuv.len(),
            width,
            height
        ));
    }
    let dec_y = &yuv[..y_size];
    let psnr_y = luma_psnr(&src.y, dec_y);
    Ok(psnr_y)
}

#[test]
fn direction_a_our_encode_ffmpeg_decode_64x64() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping our-encode/ffmpeg-decode test");
        return;
    }
    let psnr = our_encode_ffmpeg_decode(64, 64).expect("ffmpeg must accept our keyframe");
    eprintln!("direction A (our-encode → ffmpeg-decode) 64x64: PSNR-Y = {psnr:.2} dB");
    assert!(
        psnr >= 30.0,
        "direction-A PSNR-Y {psnr:.2} dB below 30 dB floor"
    );
}

#[test]
fn direction_a_our_encode_ffmpeg_decode_320x240() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping our-encode/ffmpeg-decode test");
        return;
    }
    let psnr = our_encode_ffmpeg_decode(320, 240).expect("ffmpeg must accept our keyframe");
    eprintln!("direction A (our-encode → ffmpeg-decode) 320x240: PSNR-Y = {psnr:.2} dB");
    assert!(
        psnr >= 30.0,
        "direction-A PSNR-Y {psnr:.2} dB below 30 dB floor"
    );
}

// ─────────────────── Direction B: ffmpeg encode → our decode ────────────

/// Pipe raw YUV420P bytes through `ffmpeg -c:v vp8` and capture the
/// resulting IVF bytes on stdout. Returns the raw IVF blob.
fn ffmpeg_encode_yuv(raw_yuv: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let size_arg = format!("{width}x{height}");
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            &size_arg,
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-c:v",
            "vp8",
            // Force a keyframe-only stream so the decoder doesn't need to
            // walk an inter chain we can't validate sample-exactly. (The
            // direction-A test already exercises inter-less keyframe
            // roundtrip with our encoder; here we want a clean
            // ffmpeg-encode → our-decode lane.)
            "-g",
            "1",
            "-deadline",
            "good",
            "-cpu-used",
            "0",
            "-b:v",
            "2M",
            "-f",
            "ivf",
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
        .write_all(raw_yuv)
        .map_err(|e| format!("write raw yuv to ffmpeg stdin: {e}"))?;
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

/// Walk an IVF stream and return every VP8 frame payload as an owned
/// `Vec<u8>` plus the parsed global header.
fn parse_ivf_frames(ivf: &[u8]) -> Result<(IvfHeader, Vec<Vec<u8>>), String> {
    let hdr = parse_header(ivf).map_err(|e| format!("parse_header: {e}"))?;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut cursor = IVF_HEADER_LEN;
    while cursor + IVF_FRAME_HEADER_LEN <= ivf.len() {
        let fh = parse_frame_header(&ivf[cursor..])
            .map_err(|e| format!("parse_frame_header at {cursor}: {e}"))?;
        let start = cursor + IVF_FRAME_HEADER_LEN;
        let end = start
            .checked_add(fh.size as usize)
            .ok_or_else(|| format!("frame size overflow at {cursor}"))?;
        if end > ivf.len() {
            return Err(format!(
                "truncated payload: end={end} > ivf.len={}",
                ivf.len()
            ));
        }
        frames.push(ivf[start..end].to_vec());
        cursor = end;
    }
    Ok((hdr, frames))
}

fn ffmpeg_encode_our_decode(width: u32, height: u32) -> Result<Vec<f64>, String> {
    let src = synthetic_source(width, height);
    // One frame — we set `-g 1` above so ffmpeg emits a key-only
    // stream; a single source frame keeps the test fast.
    let raw = src.yuv420p_bytes();
    let ivf = ffmpeg_encode_yuv(&raw, width, height)?;
    let (hdr, frames) = parse_ivf_frames(&ivf)?;
    if hdr.width != width || hdr.height != height {
        return Err(format!(
            "ffmpeg IVF dimensions {}x{} differ from source {width}x{height}",
            hdr.width, hdr.height
        ));
    }
    if frames.is_empty() {
        return Err("ffmpeg produced an empty IVF (no frames)".into());
    }

    // First frame must be a keyframe (we forced -g 1); use the
    // single-shot `decode_vp8` for it so the keyframe path is hit
    // independently of the stateful decoder.
    let first: Vp8DecodedFrame =
        decode_vp8(&frames[0]).map_err(|e| format!("decode_vp8(first): {e:?}"))?;
    let mut psnrs: Vec<f64> = Vec::with_capacity(frames.len());
    psnrs.push(luma_psnr(&src.y, &first.y));

    // Replay the stream through the stateful decoder too, so subsequent
    // frames (when present — usually only one with -g 1, but ffmpeg may
    // emit a flush packet) decode through the same path a real consumer
    // would use.
    let mut state = Vp8DecoderState::new();
    for (i, payload) in frames.iter().enumerate() {
        let dec = state
            .decode_frame(payload)
            .map_err(|e| format!("Vp8DecoderState::decode_frame[{i}]: {e:?}"))?;
        if dec.width != width || dec.height != height {
            return Err(format!(
                "decoded frame {i} dimensions {}x{} differ from source {width}x{height}",
                dec.width, dec.height
            ));
        }
        if i > 0 {
            psnrs.push(luma_psnr(&src.y, &dec.y));
        }
    }
    Ok(psnrs)
}

#[test]
fn direction_b_ffmpeg_encode_our_decode_64x64() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping ffmpeg-encode/our-decode test");
        return;
    }
    let psnrs = ffmpeg_encode_our_decode(64, 64).expect("our decoder must accept ffmpeg vp8");
    assert!(!psnrs.is_empty(), "must decode at least one frame");
    for (i, psnr) in psnrs.iter().enumerate() {
        eprintln!(
            "direction B (ffmpeg-encode → our-decode) 64x64 frame {i}: PSNR-Y = {psnr:.2} dB"
        );
        assert!(
            *psnr >= 25.0,
            "direction-B frame {i} PSNR-Y {psnr:.2} dB below 25 dB floor"
        );
    }
}

#[test]
fn direction_b_ffmpeg_encode_our_decode_320x240() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH — skipping ffmpeg-encode/our-decode test");
        return;
    }
    let psnrs = ffmpeg_encode_our_decode(320, 240).expect("our decoder must accept ffmpeg vp8");
    assert!(!psnrs.is_empty(), "must decode at least one frame");
    for (i, psnr) in psnrs.iter().enumerate() {
        eprintln!(
            "direction B (ffmpeg-encode → our-decode) 320x240 frame {i}: PSNR-Y = {psnr:.2} dB"
        );
        assert!(
            *psnr >= 25.0,
            "direction-B frame {i} PSNR-Y {psnr:.2} dB below 25 dB floor"
        );
    }
}
