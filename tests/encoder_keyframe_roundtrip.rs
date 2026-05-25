//! Whole-frame VP8 key-frame encoder roundtrip (RFC 6386 §9 / §11 /
//! §19.2 raster driver).
//!
//! Builds a small synthetic I420 picture (gradients + flat regions),
//! encodes it with [`oxideav_vp8::encode_keyframe`], decodes the
//! resulting bitstream back through the crate's own
//! [`oxideav_vp8::decode_vp8`], and asserts the reconstructed picture
//! reaches a whole-frame PSNR ≥ 30 dB at a mid quantiser — proving the
//! per-MB neighbour-strip threading, the §11 macroblock-mode layer, and
//! the §13.3 token partition all round-trip end-to-end.
//!
//! This is a fully self-contained black-box check: no external codec is
//! consulted, only the crate's own encode + decode entry points.

use oxideav_vp8::{decode_vp8, encode_keyframe, I420Frame, KeyframeParams};

/// A source I420 picture with tightly-packed planes.
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

/// Build a synthetic `width × height` I420 picture combining a smooth
/// luma gradient with a flat chroma background and a flat luma square —
/// a mix of structured and uniform regions that exercises both the
/// whole-block and `B_PRED` intra paths.
fn synthetic_source(width: u32, height: u32) -> Source {
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;

    let mut y = vec![0u8; w * h];
    for (row, chunk) in y.chunks_mut(w).enumerate() {
        for (col, px) in chunk.iter_mut().enumerate() {
            // Diagonal gradient over most of the frame…
            let mut v = ((col * 256 / w + row * 256 / h) / 2) as u8;
            // …with a flat-128 square in the centre quadrant.
            if col >= w / 4 && col < w * 3 / 4 && row >= h / 4 && row < h * 3 / 4 {
                v = 128;
            }
            *px = v;
        }
    }

    // Gentle chroma gradients so the chroma planes aren't pure DC.
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

/// Mean-squared error between two equal-length byte planes.
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

/// Whole-frame PSNR across the three planes (combined MSE over all
/// luma + chroma samples). 8-bit peak = 255.
fn frame_psnr(src: &Source, dec: &oxideav_vp8::Vp8DecodedFrame) -> f64 {
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

/// Encode → decode → PSNR for a `width × height` synthetic frame at the
/// given quantiser, asserting the dimensions round-trip and the PSNR
/// clears `min_psnr`.
fn roundtrip_at(width: u32, height: u32, y_ac_qi: u8, min_psnr: f64) -> f64 {
    let src = synthetic_source(width, height);
    let params = KeyframeParams {
        y_ac_qi,
        loop_filter_level: 0,
    };
    let bytes = encode_keyframe(&src.frame(), &params).expect("encode_keyframe");
    let dec = decode_vp8(&bytes).expect("decode_vp8 of our own keyframe");

    assert_eq!(dec.width, width, "decoded width");
    assert_eq!(dec.height, height, "decoded height");
    assert_eq!(dec.y.len(), (width * height) as usize, "luma plane size");
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert_eq!(dec.u.len(), (cw * ch) as usize, "U plane size");
    assert_eq!(dec.v.len(), (cw * ch) as usize, "V plane size");

    let psnr = frame_psnr(&src, &dec);
    assert!(
        psnr >= min_psnr,
        "PSNR {psnr:.2} dB below the {min_psnr:.1} dB target ({width}x{height}, qi={y_ac_qi})"
    );
    psnr
}

#[test]
fn keyframe_48x32_midquant_meets_30db() {
    // The round target: a small synthetic frame, mid quantiser, whole-
    // frame PSNR ≥ 30 dB through our own decode path.
    let psnr = roundtrip_at(48, 32, 32, 30.0);
    eprintln!("48x32 qi=32 whole-frame PSNR = {psnr:.2} dB");
}

#[test]
fn keyframe_32x32_midquant_meets_30db() {
    let psnr = roundtrip_at(32, 32, 32, 30.0);
    eprintln!("32x32 qi=32 whole-frame PSNR = {psnr:.2} dB");
}

#[test]
fn keyframe_lower_quant_raises_psnr() {
    // A lower quantiser index (finer steps) should not reduce fidelity;
    // qi=8 must clear a comfortably higher bar than the qi=32 target.
    let psnr = roundtrip_at(48, 32, 8, 36.0);
    eprintln!("48x32 qi=8 whole-frame PSNR = {psnr:.2} dB");
}

#[test]
fn keyframe_non_multiple_of_16_dimensions_roundtrip() {
    // A frame whose width and height are not multiples of 16 exercises
    // the partial right / bottom macroblock edge-replication padding.
    let psnr = roundtrip_at(40, 24, 32, 30.0);
    eprintln!("40x24 qi=32 whole-frame PSNR = {psnr:.2} dB");
}
