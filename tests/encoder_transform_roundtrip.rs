//! VP8 §14 forward-transform + §13 TokenEncoder roundtrip — Phase 2 begin.
//!
//! This integration test wires the new `forward_dct_4x4` /
//! `forward_wht_4x4` primitives (`src/forward_transform.rs`) into the
//! existing §13 `TokenEncoder` and proves end-to-end fidelity on a
//! synthetic flat-color block:
//!
//! 1. Build a 4×4 source block of uniform pixel values (the macroblock
//!    residual after a perfect DC predictor would be zero, so we use
//!    the *signed* residual that a real encoder would feed — a flat
//!    `value` block reads as a single DC coefficient under the §14
//!    transform, exactly what the roundtrip needs to prove the
//!    encode-decode chain wires the right lanes).
//! 2. Forward-transform → quantize → raster-to-scan reorder →
//!    `TokenEncoder::encode_block`.
//! 3. Finalise the bool encoder, hand the bytes back to a
//!    `BoolDecoder` + `decode_block`, scan-to-raster reorder, apply
//!    the §14.1 dequantization factors, run the §14.4 inverse DCT (or
//!    §14.3 inverse WHT for the Y2 path), recover the block.
//! 4. Compute PSNR against the original block. Per the round-131
//!    target the chain must reach ≥ 35 dB on the synthetic flat block.
//!
//! Per-MB block-set wiring (Y2-collected DC seeding across 16 Y
//! sub-blocks, the 24/25-block walk, the RD-driven quant-step and
//! mode-pick) is the next round; this round proves only the
//! per-block primitive chain.

use oxideav_vp8::{
    add_residue_4x4, dequant_block, forward_dct_4x4, forward_wht_4x4, inverse_dct_4x4,
    inverse_wht_4x4, raster_to_scan, BlockType, BoolDecoder, TokenEncoder, Y1DequantFactors,
    AC_QLOOKUP, DC_QLOOKUP, DEFAULT_COEFF_PROBS, ZIGZAG,
};

/// Reorder a decoder-produced scan-order block back into raster order
/// — mirrors the private `scan_to_raster` inside
/// `dct_tokens::decode_mb_coeffs`. Kept inline so the test is fully
/// self-contained.
fn scan_to_raster(scan: &[i16; 16]) -> [i16; 16] {
    let mut raster = [0i16; 16];
    for (c, &v) in scan.iter().enumerate() {
        raster[ZIGZAG[c]] = v;
    }
    raster
}

/// Quantize a raster-order coefficient block with the §14.1
/// DC-vs-AC factor split: `q[0] = round_div(c[0], dc)`,
/// `q[i] = round_div(c[i], ac)` for i ≥ 1. Round-half-away-from-zero
/// to match the natural inverse of `dequant_block`.
fn quantize_block(coeffs: &mut [i16; 16], dc: i16, ac: i16) {
    coeffs[0] = round_div(coeffs[0] as i32, dc as i32);
    for c in coeffs.iter_mut().skip(1) {
        *c = round_div(*c as i32, ac as i32);
    }
}

fn round_div(num: i32, den: i32) -> i16 {
    debug_assert!(den > 0, "dequant factor must be positive");
    let r = if num >= 0 {
        (num + den / 2) / den
    } else {
        -(((-num) + den / 2) / den)
    };
    r as i16
}

/// Mean-squared-error PSNR for 8-bit pixel blocks.
fn psnr_8bit(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: f64 = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x as f64 - y as f64;
        sse += d * d;
    }
    if sse == 0.0 {
        return f64::INFINITY;
    }
    let mse = sse / a.len() as f64;
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

/// Encode one 4×4 residual block through FDCT → quantize → TokenEncoder
/// and decode it back through `BoolDecoder` → dequant → IDCT, returning
/// the recovered pixel residual.
///
/// `block_type` picks the §13.3 plane index; for a single 4×4 sub-block
/// the natural choice is `YNoY2` (a `B_PRED`-style Y sub-block whose
/// own DC coefficient is carried in the block) or `UV` (a chroma 4×4).
fn fdct_token_roundtrip(
    pixels: &[i16; 16],
    block_type: BlockType,
    yac_qi: i32,
    ydc_delta: i32,
) -> [i16; 16] {
    // 1. Forward DCT.
    let mut raster = [0i16; 16];
    forward_dct_4x4(pixels, &mut raster);

    // 2. Quantize against the §14.1 Y1 DC/AC factors.
    let factors = Y1DequantFactors::from_indices(yac_qi, ydc_delta);
    quantize_block(&mut raster, factors.dc, factors.ac);

    // 3. Raster → scan reorder for the §13.3 token walk.
    let scan = raster_to_scan(&raster);

    // 4. Encode with `TokenEncoder`.
    let mut enc = TokenEncoder::new(DEFAULT_COEFF_PROBS);
    enc.encode_block(block_type, false, false, &scan)
        .expect("encode_coeff_block accepts in-range coefficients");
    let bytes = enc.finish();

    // 5. Decode bytes back into scan-order coefficients.
    let mut dec = BoolDecoder::init(&bytes).expect("encoder emits ≥ 2 bytes");
    let mut recovered_scan = [0i16; 16];
    oxideav_vp8::decode_block(
        &mut dec,
        block_type,
        &DEFAULT_COEFF_PROBS,
        false,
        false,
        &mut recovered_scan,
    )
    .expect("decode_block consumes the encoded byte stream");

    // 6. Scan → raster.
    let mut recovered_raster = scan_to_raster(&recovered_scan);

    // 7. Dequantize.
    dequant_block(&mut recovered_raster, factors.dc, factors.ac);

    // 8. Inverse DCT.
    let mut recovered_pixels = [0i16; 16];
    inverse_dct_4x4(&recovered_raster, &mut recovered_pixels);
    recovered_pixels
}

/// Pre-roundtrip sanity: at `yac_qi = 0`, the DC lookup is 4 and the
/// AC lookup is 4 (per `DC_QLOOKUP[0] == AC_QLOOKUP[0] == 4`); a flat
/// block with DC = 8*v will quantize to round_div(8*v, 4) = 2*v with
/// exact dequant back to 8*v.
#[test]
fn lookup_table_assumptions_used_below() {
    assert_eq!(DC_QLOOKUP[0], 4);
    assert_eq!(AC_QLOOKUP[0], 4);
}

/// Synthetic flat residual block (value = 16) at `yac_qi = 0`,
/// `BlockType::YNoY2`. The FDCT picks DC = 128, quant divides by 4 →
/// 32, encoder emits one DCT4-bearing token then EOB, decode →
/// dequant → IDCT recovers an 8-bit block of pixel value 16 exactly.
/// PSNR must be ∞ (zero MSE) under this single-DC scenario.
#[test]
fn flat_residual_block_roundtrips_losslessly_at_q0() {
    let input = [16i16; 16];
    let recovered = fdct_token_roundtrip(&input, BlockType::YNoY2, 0, 0);
    assert_eq!(
        recovered, input,
        "lossless single-DC flat-block roundtrip failed"
    );
}

/// Same flat-block roundtrip across a sweep of pixel values at
/// `yac_qi = 0`. Each value must come back exact (the DC term is
/// `8 * value`, quant factor 4 → `2 * value`, dequant → `8 * value`,
/// IDCT round `(8v + 4) >> 3 = v + ((4) >> 3) = v` for v ≥ 0).
#[test]
fn flat_residual_block_sweep_roundtrips_at_q0() {
    for value in 0..=64i16 {
        let input = [value; 16];
        let recovered = fdct_token_roundtrip(&input, BlockType::YNoY2, 0, 0);
        assert_eq!(recovered, input, "flat-block value={value} failed");
    }
}

/// PSNR ≥ 35 dB target on a flat block (value = 32) at a non-trivial
/// quantizer (`yac_qi = 32`). With `DC_QLOOKUP[32] = 27` and the
/// flat-block DC = 8*32 = 256, quant gives round_div(256, 27) = 9,
/// dequant gives 243, IDCT gives (243+4)>>3 = 30 — an error of 2 per
/// pixel ≈ MSE 4, PSNR ≈ 42 dB, well above the 35 dB target.
#[test]
fn flat_residual_block_psnr_above_35db_at_q32() {
    let original_value = 32i16;
    let input = [original_value; 16];
    let recovered = fdct_token_roundtrip(&input, BlockType::YNoY2, 32, 0);
    // Convert i16 residuals to u8 for PSNR (shift by 128 so signed
    // residuals map into the byte range).
    let to_u8 = |v: i16| -> u8 { (v as i32 + 128).clamp(0, 255) as u8 };
    let orig_u8: Vec<u8> = input.iter().copied().map(to_u8).collect();
    let rec_u8: Vec<u8> = recovered.iter().copied().map(to_u8).collect();
    let psnr = psnr_8bit(&orig_u8, &rec_u8);
    assert!(
        psnr >= 35.0,
        "flat-block roundtrip PSNR {psnr:.2} dB < 35 dB target (input={input:?}, recovered={recovered:?})"
    );
}

/// PSNR ≥ 35 dB on a non-flat (gradient) block at `yac_qi = 0`. The
/// FDCT spreads energy into a couple of AC coefficients; the round-trip
/// loses at most a few LSBs to the §14.4 finite-precision inverse.
#[test]
fn gradient_block_psnr_above_35db_at_q0() {
    let mut input = [0i16; 16];
    for row in 0..4 {
        for col in 0..4 {
            input[row * 4 + col] = ((row * 8 + col * 2) as i16) + 16;
        }
    }
    let recovered = fdct_token_roundtrip(&input, BlockType::YNoY2, 0, 0);
    let to_u8 = |v: i16| -> u8 { (v as i32 + 128).clamp(0, 255) as u8 };
    let orig_u8: Vec<u8> = input.iter().copied().map(to_u8).collect();
    let rec_u8: Vec<u8> = recovered.iter().copied().map(to_u8).collect();
    let psnr = psnr_8bit(&orig_u8, &rec_u8);
    assert!(
        psnr >= 35.0,
        "gradient-block roundtrip PSNR {psnr:.2} dB < 35 dB target"
    );
}

/// `forward_wht_4x4` round-trip through `inverse_wht_4x4` on a flat
/// block — the WHT-side primitive partner to the DCT roundtrip above.
/// The Y2 block carries the DC coefficients of the sixteen Y
/// sub-blocks; this proves the FWHT picks the right DC concentration
/// and the IWHT recovers it bit-exact.
#[test]
fn fwht_iwht_flat_block_roundtrips() {
    for v in -64..=64i16 {
        let input = [v; 16];
        let mut coeffs = [0i16; 16];
        forward_wht_4x4(&input, &mut coeffs);
        let mut recovered = [0i16; 16];
        inverse_wht_4x4(&coeffs, &mut recovered);
        assert_eq!(recovered, input, "fwht/iwht flat v={v} failed");
    }
}

/// End-to-end PSNR ≥ 35 dB target as stated in the round goal —
/// reconstructed pixels (residual added to a zero prediction, clamped
/// at 255) against the original pixel block, on the synthetic flat-color
/// 4×4 used as the canonical regression case for Phase 2 begin.
#[test]
fn end_to_end_psnr_target_meets_round131_goal() {
    // Pick a pixel value in the middle of the byte range so the
    // residual is naturally signed but small enough that no §14.5
    // clamp fires.
    let pixel_value = 128u8;
    let prediction = [pixel_value; 16];
    // Treat the pixels themselves as the residual basis (zero
    // prediction). The synthetic flat-color check is about the FDCT
    // / quant / encoder / decoder / dequant / IDCT chain, not the
    // §11 / §12 prediction layer.
    let residual_in = [0i16; 16]; // a zero-residual block is the trivial
                                  // success case; the meaningful PSNR
                                  // test uses a non-zero residual to
                                  // actually exercise the token path.
    let recovered_residual = fdct_token_roundtrip(&residual_in, BlockType::YNoY2, 32, 0);
    let mut recovered_pixels = [0u8; 16];
    add_residue_4x4(&prediction, &recovered_residual, &mut recovered_pixels);
    let original_pixels = [pixel_value; 16];
    let psnr = psnr_8bit(&original_pixels, &recovered_pixels);
    assert!(
        psnr >= 35.0,
        "end-to-end zero-residual PSNR {psnr:.2} dB below round target"
    );

    // Non-zero residual: drive the token path.
    let residual_in = [12i16; 16];
    let recovered_residual = fdct_token_roundtrip(&residual_in, BlockType::YNoY2, 32, 0);
    let mut original = [0u8; 16];
    add_residue_4x4(&prediction, &residual_in, &mut original);
    let mut recovered = [0u8; 16];
    add_residue_4x4(&prediction, &recovered_residual, &mut recovered);
    let psnr = psnr_8bit(&original, &recovered);
    assert!(
        psnr >= 35.0,
        "end-to-end non-zero residual PSNR {psnr:.2} dB below 35 dB target"
    );
    // Report the achieved number in stderr for the round-131 record.
    eprintln!("round-131 flat-block PSNR: {psnr:.2} dB");
}
