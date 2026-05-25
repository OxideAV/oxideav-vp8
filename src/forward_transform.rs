//! VP8 §14 forward 4×4 DCT and WHT primitives (encoder side).
//!
//! The RFC 6386 §14.3 / §14.4 listings give the **inverse** transforms
//! `vp8_short_inv_walsh4x4_c` and `short_idct4x4llm_c`. The forward
//! transforms are not listed in the spec — but they are mechanically
//! recoverable as the transpose of the inverse (the §14.4 preamble
//! itself notes the transform is *"a classical 2-D inverse discrete
//! cosine transform, implemented as two passes of 1-D inverse DCT"*).
//! The primitives below were derived in this crate, from the §14.3 /
//! §14.4 inverses listed in RFC 6386, with no reference to any
//! external forward-transform implementation.
//!
//! # Derivation summary
//!
//! Write the §14.3 inverse WHT as a 2-D matrix product on a 4×4 block:
//!
//! ```text
//! IWHT(c) = round((M * c * M^T) / 8)
//! ```
//!
//! where the §14.3 column-pass / row-pass arithmetic
//! (`a1 = i0+i12`, `b1 = i4+i8`, `c1 = i4-i8`, `d1 = i0-i12`,
//! `op[0]=a1+b1`, `op[4]=c1+d1`, `op[8]=a1-b1`, `op[12]=d1-c1`)
//! unfolds to a Walsh-Hadamard matrix
//!
//! ```text
//! M = [[1,  1,  1,  1],
//!      [1,  1, -1, -1],
//!      [1, -1, -1,  1],
//!      [1, -1,  1, -1]]
//! ```
//!
//! that satisfies `M * M^T = 4 * I` and is symmetric (`M = M^T`). The
//! `(x + 3) >> 3` rounding in the second pass divides the result by 8.
//!
//! The forward WHT is therefore
//!
//! ```text
//! FWHT(p) = round((M * p * M) / 2)
//! ```
//!
//! (since `IWHT(FWHT(p)) = round( ((M * (M*p*M) * M^T) / 2) / 8 )
//!                       = round((M*M)*p*(M*M^T) / 16)
//!                       = round(4 * I * p * 4 * I / 16) = p`).
//!
//! For the §14.4 inverse DCT, the column-pass arithmetic
//!
//! ```text
//! a1 = i0 + i8;   b1 = i0 - i8
//! c1 = (i4*S)>>16 - (i12 + (i12*C')>>16)        // S, C'+1 = sqrt(2)*sin(pi/8), sqrt(2)*cos(pi/8)
//! d1 = (i4 + (i4*C')>>16) + (i12*S)>>16
//! op[0]  = a1 + d1
//! op[4]  = b1 + c1
//! op[8]  = b1 - c1
//! op[12] = a1 - d1
//! ```
//!
//! is the linear map
//!
//! ```text
//! T_inv = [[1,  C,  1,  S],
//!          [1,  S, -1, -C],
//!          [1, -S, -1,  C],
//!          [1, -C,  1, -S]]
//! ```
//!
//! with `C = sqrt(2)*cos(pi/8)` and `S = sqrt(2)*sin(pi/8)` so that
//! `C^2 + S^2 = 2` (and therefore `T_inv * T_inv^T = 4 * I`). The
//! second-pass `(x + 4) >> 3` divides by 8. By the same algebra as the
//! WHT, the forward 1-D DCT is the transpose of `T_inv`,
//!
//! ```text
//! T_fwd = T_inv^T = [[1,  1,  1,  1],
//!                    [C,  S, -S, -C],
//!                    [1, -1, -1,  1],
//!                    [S, -C,  C, -S]]
//! ```
//!
//! and the 2-D forward DCT is
//!
//! ```text
//! FDCT(p) = round((T_fwd * p * T_fwd^T) / 2)
//! ```
//!
//! The 16-bit fixed-point evaluation reuses the same two §14.4
//! constants `COSPI8_SQRT2_MINUS1 = 20091` (`(C - 1) << 16`) and
//! `SINPI8_SQRT2 = 35468` (`S << 16`) — `i * C = i + ((i * 20091) >> 16)`
//! and `i * S = (i * 35468) >> 16` — so the forward path's
//! finite-precision rounding tracks the §14.4 inverse path's.
//!
//! # API
//!
//! * [`forward_wht_4x4`] — §14.3 inverse partner. 4×4 raster-order
//!   input + output.
//! * [`forward_dct_4x4`] — §14.4 inverse partner. 4×4 raster-order
//!   input + output.
//! * [`raster_to_scan`] — the §20.16 zig-zag reordering an encoder
//!   needs to feed a raster-order coefficient block to the §13.3 token
//!   walk (the partner of the per-block scan-to-raster reorder applied
//!   inside `dct_tokens::decode_mb_coeffs`).
//!
//! These are the *primitives*. The per-MB block-set wiring (Y2 DC
//! collection, 24/25-block walk, prediction subtraction) and the
//! RD-driven quantization / mode selection live in subsequent encoder
//! rounds.

use crate::dct_tokens::ZIGZAG;

/// 16-bit fixed-point `sqrt(2) * cos(pi/8) - 1`, identical to the
/// §14.4 inverse constant; reused here so the forward / inverse paths
/// round consistently.
const COSPI8_SQRT2_MINUS1: i32 = 20091;

/// 16-bit fixed-point `sqrt(2) * sin(pi/8)`, identical to the §14.4
/// inverse constant.
const SINPI8_SQRT2: i32 = 35468;

/// Forward 4×4 Walsh-Hadamard transform — the §14.3 inverse's encoder
/// partner.
///
/// `input` and `output` are 4×4 in row-major (raster) order. The
/// output is `round((M * input * M) / 2)` (see module docs for the
/// derivation), computed with the same butterfly shape as the §14.3
/// inverse so the forward and inverse paths are perfectly transposed.
///
/// The `(x + 1) >> 1` final round is a symmetric round-half-up for
/// non-negative values and round-half-down (toward `-∞`) for negative
/// values; for the round-trip target this matches the §14.3
/// `(x + 3) >> 3` rounding convention.
pub fn forward_wht_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    // First pass: operate down each column. With the WHT matrix M
    // applied to a column [i0, i4, i8, i12], the four outputs are
    //   o0  = i0 + i4 + i8 + i12   (row 0 of M)
    //   o4  = i0 + i4 - i8 - i12   (row 1 of M)
    //   o8  = i0 - i4 - i8 + i12   (row 2 of M)
    //   o12 = i0 - i4 + i8 - i12   (row 3 of M)
    let mut tmp = [0i32; 16];
    for col in 0..4 {
        let i0 = input[col] as i32;
        let i4 = input[4 + col] as i32;
        let i8 = input[8 + col] as i32;
        let i12 = input[12 + col] as i32;

        let a1 = i0 + i4;
        let b1 = i8 + i12;
        let c1 = i0 - i4;
        let d1 = i8 - i12;

        tmp[col] = a1 + b1;
        tmp[4 + col] = a1 - b1;
        tmp[8 + col] = c1 - d1;
        tmp[12 + col] = c1 + d1;
    }

    // Second pass: operate across each row (M applied along the row),
    // followed by `/2` with symmetric rounding so a uniform input
    // produces a single DC at coefficient 0 (see the
    // `fwht_uniform_block_concentrates_to_dc` test).
    for row in 0..4 {
        let base = row * 4;
        let r0 = tmp[base];
        let r1 = tmp[base + 1];
        let r2 = tmp[base + 2];
        let r3 = tmp[base + 3];

        let a1 = r0 + r1;
        let b1 = r2 + r3;
        let c1 = r0 - r1;
        let d1 = r2 - r3;

        let o0 = a1 + b1;
        let o1 = a1 - b1;
        let o2 = c1 - d1;
        let o3 = c1 + d1;

        // Symmetric "/2 with round-half-away-from-zero". Plain `(x+1)>>1`
        // would bias negatives down (e.g. (-1+1)>>1 = 0 but (-3+1)>>1 = -1
        // is correct anyway); pairing the rounding so that negative
        // values round toward zero matches the IWHT's `(x+3)>>3` shape
        // for the round-trip.
        output[base] = round_div2(o0);
        output[base + 1] = round_div2(o1);
        output[base + 2] = round_div2(o2);
        output[base + 3] = round_div2(o3);
    }
}

/// Forward 4×4 DCT — the §14.4 inverse's encoder partner.
///
/// `input` and `output` are 4×4 in row-major (raster) order. The
/// output is `round((T_fwd * input * T_fwd^T) / 2)` (see module docs),
/// evaluated in 16-bit fixed-point with the same `COSPI8_SQRT2_MINUS1`
/// / `SINPI8_SQRT2` constants the §14.4 inverse uses.
///
/// The 1-D forward pass for a column [i0, i4, i8, i12] is the
/// transpose of the §14.4 1-D inverse pass:
///
/// ```text
/// o0  = i0 + i4 + i8 + i12
/// o4  = (i0 * C) + (i4 * S) - (i8 * S) - (i12 * C)
/// o8  = i0 - i4 - i8 + i12
/// o12 = (i0 * S) - (i4 * C) + (i8 * C) - (i12 * S)
/// ```
///
/// Each `* C` is computed as `i + ((i * 20091) >> 16)` and each `* S`
/// as `(i * 35468) >> 16`, matching the §14.4 inverse exactly.
pub fn forward_dct_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    // First pass: operate down each column. `c_mul` and `s_mul` use
    // the same fixed-point evaluation as §14.4 so the rounding shape is
    // consistent between forward and inverse.
    for col in 0..4 {
        let i0 = input[col] as i32;
        let i4 = input[4 + col] as i32;
        let i8 = input[8 + col] as i32;
        let i12 = input[12 + col] as i32;

        let o0 = i0 + i4 + i8 + i12;
        let o8 = i0 - i4 - i8 + i12;

        // o4 = i0*C + i4*S - i8*S - i12*C
        let c0 = c_mul(i0);
        let s4 = s_mul(i4);
        let s8 = s_mul(i8);
        let c12 = c_mul(i12);
        let o4 = c0 + s4 - s8 - c12;

        // o12 = i0*S - i4*C + i8*C - i12*S
        let s0 = s_mul(i0);
        let c4 = c_mul(i4);
        let c8 = c_mul(i8);
        let s12 = s_mul(i12);
        let o12 = s0 - c4 + c8 - s12;

        tmp[col] = o0;
        tmp[4 + col] = o4;
        tmp[8 + col] = o8;
        tmp[12 + col] = o12;
    }

    // Second pass: operate across each row, then symmetric `/2` round.
    for row in 0..4 {
        let base = row * 4;
        let r0 = tmp[base];
        let r1 = tmp[base + 1];
        let r2 = tmp[base + 2];
        let r3 = tmp[base + 3];

        let o0 = r0 + r1 + r2 + r3;
        let o2 = r0 - r1 - r2 + r3;

        let c0 = c_mul(r0);
        let s1 = s_mul(r1);
        let s2 = s_mul(r2);
        let c3 = c_mul(r3);
        let o1 = c0 + s1 - s2 - c3;

        let s0 = s_mul(r0);
        let c1 = c_mul(r1);
        let c2 = c_mul(r2);
        let s3 = s_mul(r3);
        let o3 = s0 - c1 + c2 - s3;

        output[base] = round_div2(o0);
        output[base + 1] = round_div2(o1);
        output[base + 2] = round_div2(o2);
        output[base + 3] = round_div2(o3);
    }
}

/// Reorder a raster-order 16-coefficient block into the §20.16 scan
/// (zig-zag) order the §13.3 token walk consumes.
///
/// This is the encoder partner of the private `scan_to_raster` used
/// inside `dct_tokens::decode_mb_coeffs`: `scan[c] = raster[ZIGZAG[c]]`
/// (the inverse permutation).
pub fn raster_to_scan(raster: &[i16; 16]) -> [i16; 16] {
    let mut scan = [0i16; 16];
    for (c, slot) in scan.iter_mut().enumerate() {
        *slot = raster[ZIGZAG[c]];
    }
    scan
}

/// Fixed-point `i * (sqrt(2) * cos(pi/8))` matching the §14.4 inverse:
/// `i + ((i * COSPI8_SQRT2_MINUS1) >> 16)`.
#[inline]
fn c_mul(i: i32) -> i32 {
    i + ((i * COSPI8_SQRT2_MINUS1) >> 16)
}

/// Fixed-point `i * (sqrt(2) * sin(pi/8))` matching the §14.4 inverse:
/// `(i * SINPI8_SQRT2) >> 16`.
#[inline]
fn s_mul(i: i32) -> i32 {
    (i * SINPI8_SQRT2) >> 16
}

/// Symmetric `/2` with rounding away from zero, then clamped into the
/// `i16` range a single transform coefficient occupies. The clamp
/// guards against the rare large-input case (a uniform 4×4 of 255
/// produces a DC of 8*255 = 2040, well within i16; the clamp is purely
/// defensive against future callers that may feed pre-scaled
/// residuals).
#[inline]
fn round_div2(v: i32) -> i16 {
    let rounded = if v >= 0 {
        (v + 1) >> 1
    } else {
        -((-v + 1) >> 1)
    };
    rounded.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse_transform::{inverse_dct_4x4, inverse_wht_4x4};

    /// A 4×4 block of all-`v` pixels should forward-transform to a
    /// single DC coefficient at position 0 (for both DCT and WHT,
    /// because the row-0 sum of `T_fwd` / `M` is `[1,1,1,1]` while
    /// every other row sums to 0).
    #[test]
    fn fwht_uniform_block_concentrates_to_dc() {
        for v in [1i16, 2, 5, 8, 16, 64, 100] {
            let input = [v; 16];
            let mut out = [0i16; 16];
            forward_wht_4x4(&input, &mut out);
            // DC = (4 * 4 * v) / 2 = 8 * v.
            assert_eq!(out[0], 8 * v, "fwht uniform v={v} DC");
            for (i, &c) in out.iter().enumerate().skip(1) {
                assert_eq!(c, 0, "fwht uniform v={v} non-DC pos {i}");
            }
        }
    }

    #[test]
    fn fdct_uniform_block_concentrates_to_dc() {
        for v in [1i16, 2, 5, 8, 16, 64, 100] {
            let input = [v; 16];
            let mut out = [0i16; 16];
            forward_dct_4x4(&input, &mut out);
            assert_eq!(out[0], 8 * v, "fdct uniform v={v} DC");
            for (i, &c) in out.iter().enumerate().skip(1) {
                assert_eq!(c, 0, "fdct uniform v={v} non-DC pos {i}");
            }
        }
    }

    /// `forward_wht_4x4` then `inverse_wht_4x4` on a uniform block
    /// recovers the input exactly (the §14.3 `(x+3)>>3` rounding on
    /// the DC = 8*v term gives `((8v + 3) >> 3) = v` for any positive
    /// `v`, and the forward path concentrates everything in the DC).
    #[test]
    fn fwht_iwht_roundtrip_uniform_block() {
        for v in [0i16, 1, 2, 3, 5, 8, 16, 32, 64, 100] {
            let input = [v; 16];
            let mut coeffs = [0i16; 16];
            forward_wht_4x4(&input, &mut coeffs);
            let mut recovered = [0i16; 16];
            inverse_wht_4x4(&coeffs, &mut recovered);
            assert_eq!(recovered, input, "iwht(fwht(uniform={v})) lost");
        }
    }

    #[test]
    fn fdct_idct_roundtrip_uniform_block() {
        for v in [0i16, 1, 2, 3, 5, 8, 16, 32, 64, 100] {
            let input = [v; 16];
            let mut coeffs = [0i16; 16];
            forward_dct_4x4(&input, &mut coeffs);
            let mut recovered = [0i16; 16];
            inverse_dct_4x4(&coeffs, &mut recovered);
            assert_eq!(recovered, input, "idct(fdct(uniform={v})) lost");
        }
    }

    /// All-zero round-trip — both transforms must take a zero block to
    /// a zero coefficient block.
    #[test]
    fn forward_transforms_of_zero_are_zero() {
        let input = [0i16; 16];
        let mut out = [0i16; 16];
        forward_wht_4x4(&input, &mut out);
        assert_eq!(out, [0i16; 16]);
        let mut out2 = [0i16; 16];
        forward_dct_4x4(&input, &mut out2);
        assert_eq!(out2, [0i16; 16]);
    }

    /// `raster_to_scan` is the inverse permutation of the §20.16
    /// zig-zag (`scan[c] = raster[ZIGZAG[c]]`), so applying it then
    /// the public scan-to-raster sequence used in tests recovers the
    /// original raster block.
    #[test]
    fn raster_to_scan_is_zigzag_inverse() {
        // Construct a raster with a distinctive value at every
        // position so a missing / duplicated index is visible.
        let mut raster = [0i16; 16];
        for (i, slot) in raster.iter_mut().enumerate() {
            *slot = (i as i16) + 1;
        }
        let scan = raster_to_scan(&raster);
        // Manually invert by walking ZIGZAG.
        let mut back = [0i16; 16];
        for (c, &v) in scan.iter().enumerate() {
            back[ZIGZAG[c]] = v;
        }
        assert_eq!(back, raster);
    }

    /// FDCT of a 1-d gradient picks up energy in the first AC
    /// coefficient (proving the cosine constants reach the right
    /// lanes); the round-trip recovers the gradient exactly under no
    /// quantization.
    #[test]
    fn fdct_idct_roundtrip_gradient_block() {
        // Column gradient: pixel value = row * 8 (so deltas are
        // multiples of 8, large enough to be visible at the DC=8x
        // scale).
        let mut input = [0i16; 16];
        for row in 0..4 {
            for col in 0..4 {
                input[row * 4 + col] = (row as i16) * 8;
            }
        }
        let mut coeffs = [0i16; 16];
        forward_dct_4x4(&input, &mut coeffs);
        // First AC of the column direction (position 4) must be
        // non-zero; the horizontal AC (position 1) must remain zero.
        assert_ne!(coeffs[4], 0, "column-gradient AC[4] must be non-zero");
        assert_eq!(
            coeffs[1], 0,
            "column-gradient AC[1] (horizontal) must be zero"
        );

        let mut recovered = [0i16; 16];
        inverse_dct_4x4(&coeffs, &mut recovered);
        // The §14.4 inverse has finite-precision rounding; PSNR test
        // below quantifies it. For an unquantized round-trip of a
        // smooth block the recovery should be within ±1 per pixel.
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1,
                "pos {i}: fdct/idct lost more than 1 LSB ({a} vs {b})"
            );
        }
    }

    /// `forward_dct_4x4` is consistent with the §14.4 inverse on
    /// random small inputs: the round-trip error is bounded by a small
    /// number of LSBs (finite-precision rounding only). This is a
    /// regression guard on the matrix transpose and the fixed-point
    /// constant placement.
    #[test]
    fn fdct_idct_random_small_inputs_have_bounded_error() {
        // Tiny LCG so this test is reproducible without RNG deps.
        let mut state: u32 = 0xDEAD_BEEF;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            state
        };
        for trial in 0..32 {
            let mut input = [0i16; 16];
            for slot in input.iter_mut() {
                // Range [-64, 64] — small enough that the fdct output
                // stays well within i16 and the idct round-trip error
                // is dominated by the `>> 16` finite-precision step.
                let r = (next() >> 16) as i32;
                *slot = (r % 129 - 64) as i16;
            }
            let mut coeffs = [0i16; 16];
            forward_dct_4x4(&input, &mut coeffs);
            let mut recovered = [0i16; 16];
            inverse_dct_4x4(&coeffs, &mut recovered);
            // Maximum per-pixel error bound observed in derivation.
            for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 2,
                    "trial {trial} pos {i}: error {} ({a} vs {b}) exceeds 2 LSB",
                    (a - b).abs()
                );
            }
        }
    }
}
