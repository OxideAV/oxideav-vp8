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
    roundtrip_at_partitioned(width, height, y_ac_qi, 1, min_psnr)
}

/// Encode → decode → PSNR for a `width × height` synthetic frame at the
/// given quantiser and §9.5 DCT-partition count, asserting the
/// dimensions round-trip and the PSNR clears `min_psnr`.
fn roundtrip_at_partitioned(
    width: u32,
    height: u32,
    y_ac_qi: u8,
    nbr_of_dct_partitions: u8,
    min_psnr: f64,
) -> f64 {
    let src = synthetic_source(width, height);
    let params = KeyframeParams {
        y_ac_qi,
        loop_filter_level: 0,
        nbr_of_dct_partitions,
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

/// §9.5 multi-partition self-decode round-trip.
///
/// The §9.5 DCT-coefficient partition count is a layout-only choice:
/// the residual coding inside each partition is bit-identical to the
/// 1-partition case, the decoder rebuilds the same reconstruction
/// regardless of how rows were distributed (§20.4 round-robin), and the
/// decoded picture matches across all four legal values 1 / 2 / 4 / 8.
/// This test pins that invariant by encoding the same source at every
/// partition count and asserting the self-decode PSNR is identical
/// against the 1-partition baseline.
#[test]
fn keyframe_multi_partition_psnr_matches_single_partition_baseline() {
    // A frame tall enough that every partition count routes work to
    // every partition (mb_rows = 8 ≥ 8). 128×128 keeps the test fast.
    let (width, height, qi) = (128u32, 128u32, 32u8);
    let src = synthetic_source(width, height);

    let mut measured = [0f64; 4];
    let mut byte_lens = [0usize; 4];
    for (slot, &count) in [1u8, 2, 4, 8].iter().enumerate() {
        let params = KeyframeParams {
            y_ac_qi: qi,
            loop_filter_level: 0,
            nbr_of_dct_partitions: count,
        };
        let bytes = encode_keyframe(&src.frame(), &params).expect("encode_keyframe");
        let dec = decode_vp8(&bytes).expect("decode_vp8 of our own keyframe");
        assert_eq!(dec.width, width);
        assert_eq!(dec.height, height);
        let psnr = frame_psnr(&src, &dec);
        eprintln!(
            "{width}x{height} qi={qi} partitions={count}: {} B, PSNR = {psnr:.4} dB",
            bytes.len()
        );
        measured[slot] = psnr;
        byte_lens[slot] = bytes.len();
    }

    // The 1-partition number is the baseline; the other three must match
    // it exactly. Multi-partition is a byte-layout reorganisation, not a
    // residual-coding change — the reconstructed sample values cannot
    // differ.
    let baseline = measured[0];
    for (slot, &count) in [1u8, 2, 4, 8].iter().enumerate() {
        let psnr = measured[slot];
        assert!(
            (psnr - baseline).abs() < 1e-9,
            "partitions={count} PSNR {psnr} differs from 1-partition baseline {baseline}",
        );
    }

    // Each additional partition pays a §7.3 4-byte flush trailer plus a
    // 3-byte §9.5 size-table entry (the last partition's size is implied),
    // so the encoded length grows monotonically with partition count
    // (since the residual data is the same).
    for w in byte_lens.windows(2) {
        assert!(
            w[1] >= w[0],
            "byte length must not shrink as partition count grows: {byte_lens:?}",
        );
    }
}

/// §9.5 short-frame coverage: a frame with fewer macroblock rows than
/// partitions still encodes and round-trips. With 32×32 the frame has
/// 2 macroblock rows; at `nbr_of_dct_partitions = 8`, six of the eight
/// partitions are never written to (the §20.4 round-robin only reaches
/// partitions 0 and 1). The encoder must still emit a valid frame whose
/// unused partitions are minimal §7.3 flush trailers — the decoder leaves
/// them untouched per the §13.3 row-walk.
#[test]
fn keyframe_multi_partition_short_frame_roundtrip() {
    let (width, height, qi) = (32u32, 32u32, 32u8);
    let src = synthetic_source(width, height);
    let baseline = {
        let params = KeyframeParams {
            y_ac_qi: qi,
            loop_filter_level: 0,
            nbr_of_dct_partitions: 1,
        };
        let bytes = encode_keyframe(&src.frame(), &params).expect("encode_keyframe");
        let dec = decode_vp8(&bytes).expect("decode_vp8 of our own keyframe");
        frame_psnr(&src, &dec)
    };
    for count in [2u8, 4, 8] {
        let params = KeyframeParams {
            y_ac_qi: qi,
            loop_filter_level: 0,
            nbr_of_dct_partitions: count,
        };
        let bytes = encode_keyframe(&src.frame(), &params).expect("encode_keyframe");
        let dec = decode_vp8(&bytes).expect("decode_vp8 of our own keyframe");
        let psnr = frame_psnr(&src, &dec);
        assert!(
            (psnr - baseline).abs() < 1e-9,
            "short-frame partitions={count} PSNR {psnr} differs from baseline {baseline}",
        );
    }
}

/// The encoder rejects partition counts outside the §9.5 four-value
/// table (1, 2, 4, 8) before running the long mode-pick walk.
#[test]
fn keyframe_invalid_partition_count_rejected() {
    use oxideav_vp8::EncodeError;
    let src = synthetic_source(32, 32);
    for bad in [0u8, 3, 5, 6, 7, 9, 16, 255] {
        let params = KeyframeParams {
            y_ac_qi: 32,
            loop_filter_level: 0,
            nbr_of_dct_partitions: bad,
        };
        match encode_keyframe(&src.frame(), &params) {
            Err(EncodeError::InvalidDctPartitionCount { value }) => assert_eq!(value, bad),
            other => panic!("expected InvalidDctPartitionCount for {bad}, got {other:?}"),
        }
    }
}
