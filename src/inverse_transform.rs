//! VP8 dequantization and inverse transforms per RFC 6386 §14.
//!
//! This module implements the pieces of §14 that RFC 6386 specifies
//! **completely in its body**, all operating on small caller-supplied
//! arrays in raster (natural) order:
//!
//! * **§14.1 dequantization tables** — the `dc_qlookup[128]` /
//!   `ac_qlookup[128]` tables transcribed verbatim, plus a Y-plane
//!   dequant-factor computation. The spec states (§14.1 page 77):
//!   *"Lookup values from the above two tables are directly used in the
//!   DC and AC coefficients in Y1, respectively."* So the Y1-DC factor
//!   is `dc_qlookup[clamp(yac_qi + ydc_delta)]` and the Y1-AC factor is
//!   `ac_qlookup[clamp(yac_qi)]`, where the qindex is clamped to the
//!   `0..=127` table range. The Y2 and chroma factors require the
//!   scaling / clamping rules from §14.1 / the §20.4 `dixie.c` annex
//!   (which is part of RFC 6386); those six-factor derivations live in
//!   the [`crate::dequant`] module. This module provides only the raw
//!   lookup tables and the Y1-only factor helper.
//! * **§14.3 inverse WHT** — `inverse_wht_4x4`, a faithful port of the
//!   `vp8_short_inv_walsh4x4_c` C source given in the RFC body, plus
//!   `inverse_wht_4x4_dc_only`, the single-DC fast path
//!   (`vp8_short_inv_walsh4x4_1_c`).
//! * **§14.4 inverse DCT** — `inverse_dct_4x4`, a faithful port of the
//!   `short_idct4x4llm_c` C source (the two `sqrt(2)·cos(pi/8)` /
//!   `sqrt(2)·sin(pi/8)` fixed-point constants, two passes of 1-D
//!   inverse DCT).
//! * **§14.5 summation** — `add_residue` / `clamp255`, summing a
//!   prediction block and a residue block with the §14.5 saturating
//!   `clamp255`.
//!
//! All transform inputs/outputs are 16-bit signed integers exactly as
//! the spec mandates (*"all stored as arrays of 16-bit signed
//! integers"*), and the spec notes a conforming decoder must reproduce
//! the rounding bit-for-bit.
//!
//! # What is NOT in this module (out of scope for §14 standalone)
//!
//! * The **zig-zag → raster reordering** of the §13 scan-ordered
//!   coefficients. §13 says the coefficients "follow a so-called
//!   zig-zag order" (§13 page 60) but the spec body never gives the
//!   permutation array — it lives only in the reference `idct_add.c`
//!   (Section 20.8), which is off-limits. This is a documented gap; the
//!   per-macroblock integration round must supply the reordering before
//!   calling the raster-order transforms here.
//! * The **Y2-DC / Y2-AC / chroma-DC / chroma-AC dequant factors** —
//!   the scaling and clamping rules from §14.1 / §20.4. These now live
//!   in [`crate::dequant`] ([`crate::dequant::MbDequantFactors`]).
//! * The §14.2 macroblock-level orchestration (Y2 → 16 Y DC seeding,
//!   the 24/25-block walk) — that is the integration round's job; this
//!   module provides only the per-block primitives.
//!
//! # Resolved spec references
//!
//! * **§14.1 page 77 Y2 / chroma dequant scaling.** §14.1 page 77 says
//!   the Y2 and chroma factors *"undergo either scaling or clamping
//!   before the multiplies. Details ... can be found in related lookup
//!   functions in dixie.c (Section 20.4)."* Section 20.4 is part of RFC
//!   6386 (the reference-decoder annex), so the scaling / clamping rules
//!   it gives are in-spec; they are implemented in [`crate::dequant`].
//! * **§13 page 60 zig-zag scan order.** The 16-entry scan-to-raster
//!   permutation is given by the §20.16 `zigzag[16]` table (also part of
//!   RFC 6386); the per-MB token walk applies it (see
//!   [`crate::dct_tokens`]).

/// Number of entries in each §14.1 dequantization lookup table; the
/// 7-bit quantiser index (`L(7)`, §9.6) plus deltas is clamped into
/// `0..QINDEX_RANGE`.
pub const QINDEX_RANGE: usize = 128;

/// The §14.1 `dc_qlookup[QINDEX_RANGE]` table, transcribed verbatim
/// from RFC 6386 page 77.
pub static DC_QLOOKUP: [i16; QINDEX_RANGE] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, //
    16, 17, 17, 18, 19, 20, 20, 21, 21, 22, 22, 23, 23, //
    24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, //
    36, 37, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 46, //
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, //
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, //
    73, 74, 75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, //
    85, 86, 87, 88, 89, 91, 93, 95, 96, 98, 100, 101, 102, //
    104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130, //
    132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157, //
];

/// The §14.1 `ac_qlookup[QINDEX_RANGE]` table, transcribed verbatim
/// from RFC 6386 page 77.
pub static AC_QLOOKUP: [i16; QINDEX_RANGE] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, //
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, //
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, //
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, //
    56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, //
    80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, //
    106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, //
    140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 177, //
    181, 185, 189, 193, 197, 201, 205, 209, 213, 217, 221, 225, 229, //
    234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284, //
];

/// Clamp a (possibly delta-adjusted) quantiser index into the
/// `0..=QINDEX_RANGE-1` lookup-table domain. The base `yac_qi` is a
/// 7-bit value (`0..=127`); applying a signed delta (§9.6) can push it
/// out of range, and the lookup must saturate.
#[inline]
pub fn clamp_qindex(q: i32) -> usize {
    q.clamp(0, (QINDEX_RANGE - 1) as i32) as usize
}

/// The Y1-plane dequantization factors, derived from §14.1 page 77 and
/// the §9.6 quantiser-index layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Y1DequantFactors {
    /// Factor for coefficient 0 (DC): `dc_qlookup[clamp(yac_qi + ydc_delta)]`.
    pub dc: i16,
    /// Factor for coefficients 1..=15 (AC): `ac_qlookup[clamp(yac_qi)]`.
    pub ac: i16,
}

impl Y1DequantFactors {
    /// Compute the Y1 DC and AC dequant factors from the frame-level
    /// base index `yac_qi` (§9.6, the always-present `L(7)`) and the
    /// signed Y-DC delta `ydc_delta` (§9.6 `ydc_delta`). The §14.1 rule
    /// is that Y1 uses the lookup values directly: DC from `dc_qlookup`
    /// indexed by `yac_qi + ydc_delta`, AC from `ac_qlookup` indexed by
    /// `yac_qi`. Both indices are clamped into the table range.
    pub fn from_indices(yac_qi: i32, ydc_delta: i32) -> Self {
        Y1DequantFactors {
            dc: DC_QLOOKUP[clamp_qindex(yac_qi + ydc_delta)],
            ac: AC_QLOOKUP[clamp_qindex(yac_qi)],
        }
    }
}

/// Dequantize a 4×4 sub-block in place: coefficient 0 (DC) is
/// multiplied by `dc_factor`, coefficients 1..=15 (AC) by `ac_factor`.
///
/// §14.1 page 76: *"each (quantized) coefficient ... is multiplied by
/// one of six dequantization factors ... the multiplies are computed
/// and stored using 16-bit signed integers."* The product is computed
/// in `i32` and stored back as `i16`. The `coeffs` array is assumed to
/// be in raster (natural) order — the caller is responsible for the
/// zig-zag → raster reordering (see the module-level gap note), which
/// does not affect the DC-vs-AC factor selection because coefficient 0
/// is the DC in either ordering.
pub fn dequant_block(coeffs: &mut [i16; 16], dc_factor: i16, ac_factor: i16) {
    coeffs[0] = (coeffs[0] as i32 * dc_factor as i32) as i16;
    for c in coeffs.iter_mut().skip(1) {
        *c = (*c as i32 * ac_factor as i32) as i16;
    }
}

/// The §14.5 saturating clamp: `v < 0 ? 0 : (v < 255 ? v : 255)`
/// (RFC 6386 page 11 `clamp255`).
#[inline]
pub fn clamp255(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Inverse Walsh-Hadamard transform of the 25th (Y2) block, per the
/// `vp8_short_inv_walsh4x4_c` C source of RFC 6386 §14.3.
///
/// `input` and `output` are 4×4 in row-major (raster) order. The
/// output values are the dequantized DC residues that seed the 0th
/// coefficient of each of the sixteen Y sub-blocks (§14.2): output
/// element at row `i`, column `j` feeds Y sub-block `i*4 + j`.
///
/// All arithmetic is performed exactly as the spec C source specifies
/// (two passes, the `(x + 3) >> 3` rounding in the second pass), so the
/// rounding matches bit-for-bit.
pub fn inverse_wht_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    // First pass: operate down each column (spec: ip[0], ip[4], ip[8],
    // ip[12] with ip++ stepping across the four columns). Intermediate
    // results are written back into `output` at the matching column.
    let mut tmp = [0i32; 16];
    for col in 0..4 {
        let i0 = input[col] as i32;
        let i4 = input[4 + col] as i32;
        let i8 = input[8 + col] as i32;
        let i12 = input[12 + col] as i32;

        let a1 = i0 + i12;
        let b1 = i4 + i8;
        let c1 = i4 - i8;
        let d1 = i0 - i12;

        tmp[col] = a1 + b1;
        tmp[4 + col] = c1 + d1;
        tmp[8 + col] = a1 - b1;
        tmp[12 + col] = d1 - c1;
    }

    // Second pass: operate across each row (spec: ip[0..=3] with ip+=4
    // stepping down the rows), with the `(x + 3) >> 3` rounding.
    for row in 0..4 {
        let base = row * 4;
        let r0 = tmp[base];
        let r1 = tmp[base + 1];
        let r2 = tmp[base + 2];
        let r3 = tmp[base + 3];

        let a1 = r0 + r3;
        let b1 = r1 + r2;
        let c1 = r1 - r2;
        let d1 = r0 - r3;

        let a2 = a1 + b1;
        let b2 = c1 + d1;
        let c2 = a1 - b1;
        let d2 = d1 - c1;

        output[base] = ((a2 + 3) >> 3) as i16;
        output[base + 1] = ((b2 + 3) >> 3) as i16;
        output[base + 2] = ((c2 + 3) >> 3) as i16;
        output[base + 3] = ((d2 + 3) >> 3) as i16;
    }
}

/// The single-non-zero-DC fast path of the inverse WHT, per
/// `vp8_short_inv_walsh4x4_1_c` of RFC 6386 §14.3. When only
/// `input[0]` is non-zero, every output element is `(input[0] + 3) >> 3`.
pub fn inverse_wht_4x4_dc_only(dc: i16, output: &mut [i16; 16]) {
    let a1 = ((dc as i32) + 3) >> 3;
    for o in output.iter_mut() {
        *o = a1 as i16;
    }
}

/// 16-bit fixed-point `sqrt(2) * cos(pi/8) - 1`, per RFC 6386 §14.4.
const COSPI8_SQRT2_MINUS1: i32 = 20091;
/// 16-bit fixed-point `sqrt(2) * sin(pi/8)`, per RFC 6386 §14.4.
const SINPI8_SQRT2: i32 = 35468;

/// Inverse DCT of a 4×4 Y / U / V sub-block, per the
/// `short_idct4x4llm_c` C source of RFC 6386 §14.4.
///
/// `input` and `output` are 4×4 in row-major (raster) order. The
/// implementation is two passes of the On2 4-point 1-D inverse DCT with
/// the two fixed-point constants above; the second pass applies the
/// `(x + 4) >> 3` rounding the spec mandates for bit-exact output.
///
/// The spec's C source uses `pitch` (a byte pitch) and `shortpitch =
/// pitch >> 1`. For a self-contained 4×4 → 4×4 transform the natural
/// `shortpitch` is 4 (one row of `short`s), so the column stride in the
/// first pass and the row stride in the second pass are both 4. This is
/// the standard packed-4×4 case and is what this function implements.
pub fn inverse_dct_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    // First pass: operate down each column (spec: ip[0], ip[4], ip[8],
    // ip[12] with ip++ stepping across columns; results written with a
    // `shortpitch` stride into `op`).
    for col in 0..4 {
        let i0 = input[col] as i32;
        let i4 = input[4 + col] as i32;
        let i8 = input[8 + col] as i32;
        let i12 = input[12 + col] as i32;

        let a1 = i0 + i8;
        let b1 = i0 - i8;

        let t1 = (i4 * SINPI8_SQRT2) >> 16;
        let t2 = i12 + ((i12 * COSPI8_SQRT2_MINUS1) >> 16);
        let c1 = t1 - t2;

        let t1 = i4 + ((i4 * COSPI8_SQRT2_MINUS1) >> 16);
        let t2 = (i12 * SINPI8_SQRT2) >> 16;
        let d1 = t1 + t2;

        tmp[col] = a1 + d1;
        tmp[12 + col] = a1 - d1;
        tmp[4 + col] = b1 + c1;
        tmp[8 + col] = b1 - c1;
    }

    // Second pass: operate across each row (spec: ip[0..=3] reading the
    // intermediate buffer, ip += shortpitch stepping down the rows),
    // with the `(x + 4) >> 3` rounding.
    for row in 0..4 {
        let base = row * 4;
        let r0 = tmp[base];
        let r1 = tmp[base + 1];
        let r2 = tmp[base + 2];
        let r3 = tmp[base + 3];

        let a1 = r0 + r2;
        let b1 = r0 - r2;

        let t1 = (r1 * SINPI8_SQRT2) >> 16;
        let t2 = r3 + ((r3 * COSPI8_SQRT2_MINUS1) >> 16);
        let c1 = t1 - t2;

        let t1 = r1 + ((r1 * COSPI8_SQRT2_MINUS1) >> 16);
        let t2 = (r3 * SINPI8_SQRT2) >> 16;
        let d1 = t1 + t2;

        output[base] = ((a1 + d1 + 4) >> 3) as i16;
        output[base + 3] = ((a1 - d1 + 4) >> 3) as i16;
        output[base + 1] = ((b1 + c1 + 4) >> 3) as i16;
        output[base + 2] = ((b1 - c1 + 4) >> 3) as i16;
    }
}

/// §14.5 summation of predictor and residue for one 4×4 sub-block.
///
/// Each output pixel is `clamp255(prediction[i] + residue[i])`, with
/// the sum computed at 32-bit precision before saturation, exactly as
/// §14.5 page 83 specifies. `prediction` holds 8-bit pixels (carried as
/// `u8`); `residue` holds the inverse-transformed 16-bit residual.
pub fn add_residue_4x4(prediction: &[u8; 16], residue: &[i16; 16], out: &mut [u8; 16]) {
    for i in 0..16 {
        out[i] = clamp255(prediction[i] as i32 + residue[i] as i32);
    }
}

/// §14.5 summation over an arbitrary-length block (e.g. a 16×16 luma or
/// 8×8 chroma macroblock buffer). `prediction`, `residue`, and `out`
/// must all have the same length. Each pixel is
/// `clamp255(prediction[i] + residue[i])`.
///
/// # Panics
///
/// Panics if the three slices do not have equal length.
pub fn add_residue(prediction: &[u8], residue: &[i16], out: &mut [u8]) {
    assert_eq!(prediction.len(), residue.len(), "length mismatch");
    assert_eq!(prediction.len(), out.len(), "length mismatch");
    for i in 0..prediction.len() {
        out[i] = clamp255(prediction[i] as i32 + residue[i] as i32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qlookup_tables_have_full_range() {
        assert_eq!(DC_QLOOKUP.len(), QINDEX_RANGE);
        assert_eq!(AC_QLOOKUP.len(), QINDEX_RANGE);
    }

    #[test]
    fn qlookup_spot_checks_match_spec() {
        // RFC 6386 §14.1 page 77, first and last entries of each table.
        assert_eq!(DC_QLOOKUP[0], 4);
        assert_eq!(DC_QLOOKUP[127], 157);
        assert_eq!(AC_QLOOKUP[0], 4);
        assert_eq!(AC_QLOOKUP[127], 284);
        // The tables diverge: dc[7] = 10 (repeated 10), ac[7] = 11.
        assert_eq!(DC_QLOOKUP[6], 10);
        assert_eq!(DC_QLOOKUP[7], 10);
        assert_eq!(AC_QLOOKUP[6], 10);
        assert_eq!(AC_QLOOKUP[7], 11);
        // A mid-table sample where ac transitions from a +1 run into a
        // +2 run (page 77: ...55, then 56, 57, 58, 60, 62, 64...):
        // index 52 → 56, 53 → 57, 54 → 58, 55 → 60 (the first +2 step).
        assert_eq!(AC_QLOOKUP[52], 56);
        assert_eq!(AC_QLOOKUP[54], 58);
        assert_eq!(AC_QLOOKUP[55], 60);
    }

    #[test]
    fn qindex_clamp_saturates_both_ends() {
        assert_eq!(clamp_qindex(-5), 0);
        assert_eq!(clamp_qindex(0), 0);
        assert_eq!(clamp_qindex(63), 63);
        assert_eq!(clamp_qindex(127), 127);
        assert_eq!(clamp_qindex(200), 127);
    }

    #[test]
    fn y1_factors_use_dc_lookup_for_dc_ac_lookup_for_ac() {
        // yac_qi = 20, ydc_delta = +3 → dc index 23, ac index 20.
        let f = Y1DequantFactors::from_indices(20, 3);
        assert_eq!(f.dc, DC_QLOOKUP[23]);
        assert_eq!(f.ac, AC_QLOOKUP[20]);
        // A negative delta that underflows past 0 is clamped.
        let f2 = Y1DequantFactors::from_indices(2, -10);
        assert_eq!(f2.dc, DC_QLOOKUP[0]);
        assert_eq!(f2.ac, AC_QLOOKUP[2]);
        // A delta that overflows past 127 is clamped.
        let f3 = Y1DequantFactors::from_indices(120, 50);
        assert_eq!(f3.dc, DC_QLOOKUP[127]);
        assert_eq!(f3.ac, AC_QLOOKUP[120]);
    }

    #[test]
    fn dequant_block_scales_dc_and_ac_separately() {
        let mut c = [0i16; 16];
        c[0] = 3; // DC
        c[1] = 2; // AC
        c[15] = -4; // AC
        dequant_block(&mut c, 7, 5);
        assert_eq!(c[0], 21); // 3 * 7
        assert_eq!(c[1], 10); // 2 * 5
        assert_eq!(c[15], -20); // -4 * 5
    }

    #[test]
    fn clamp255_matches_spec_definition() {
        assert_eq!(clamp255(-1), 0);
        assert_eq!(clamp255(-1000), 0);
        assert_eq!(clamp255(0), 0);
        assert_eq!(clamp255(128), 128);
        assert_eq!(clamp255(255), 255);
        assert_eq!(clamp255(256), 255);
        assert_eq!(clamp255(99999), 255);
    }

    #[test]
    fn wht_dc_only_matches_general_path() {
        // A block with only input[0] non-zero must give the same result
        // through the general path and the fast path.
        for dc in [0i16, 1, 3, 7, 8, -8, 100, -100, 1000] {
            let mut input = [0i16; 16];
            input[0] = dc;
            let mut general = [0i16; 16];
            inverse_wht_4x4(&input, &mut general);
            let mut fast = [0i16; 16];
            inverse_wht_4x4_dc_only(dc, &mut fast);
            assert_eq!(general, fast, "dc={dc}");
            // Every element should equal (dc + 3) >> 3.
            let expect = (((dc as i32) + 3) >> 3) as i16;
            assert!(general.iter().all(|&v| v == expect), "dc={dc}");
        }
    }

    #[test]
    fn wht_zero_input_is_zero_output() {
        let input = [0i16; 16];
        let mut output = [0i16; 16];
        inverse_wht_4x4(&input, &mut output);
        assert_eq!(output, [0i16; 16]);
    }

    #[test]
    fn wht_hand_derived_two_value_input() {
        // Hand-derive the spec C source for an input with two non-zero
        // entries: input[0] = 8 (position row0 col0), input[4] = 8
        // (position row1 col0). Walk both passes by hand.
        //
        // First pass, per column. Only col 0 has non-zero input:
        //   i0 = 8, i4 = 8, i8 = 0, i12 = 0
        //   a1 = 8+0 = 8;  b1 = 8+0 = 8;  c1 = 8-0 = 8;  d1 = 8-0 = 8
        //   tmp[0]  = a1+b1 = 16
        //   tmp[4]  = c1+d1 = 16
        //   tmp[8]  = a1-b1 = 0
        //   tmp[12] = d1-c1 = 0
        // Cols 1..3 are all zero.
        //
        // Second pass, per row (each row r has tmp[r*4]=col0 value,
        // rest zero):
        //   row0: r0=16,0,0,0 → a1=16+0=16, b1=0+0=0, c1=0-0=0,
        //         d1=16-0=16; a2=16, b2=16, c2=16, d2=16;
        //         out = (16+3)>>3 = 2 for all four → [2,2,2,2]
        //   row1: r0=16 → same as row0 → [2,2,2,2]
        //   row2: r0=0  → all zero → [0,0,0,0]
        //   row3: r0=0  → all zero → [0,0,0,0]
        let mut input = [0i16; 16];
        input[0] = 8;
        input[4] = 8;
        let mut output = [0i16; 16];
        inverse_wht_4x4(&input, &mut output);
        let expect = [
            2, 2, 2, 2, //
            2, 2, 2, 2, //
            0, 0, 0, 0, //
            0, 0, 0, 0, //
        ];
        assert_eq!(output, expect);
    }

    #[test]
    fn dct_zero_input_is_zero_output() {
        let input = [0i16; 16];
        let mut output = [0i16; 16];
        inverse_dct_4x4(&input, &mut output);
        assert_eq!(output, [0i16; 16]);
    }

    #[test]
    fn dct_dc_only_is_uniform_block() {
        // With only the DC coefficient non-zero, the inverse DCT yields
        // a (nearly) flat block. Hand-derive for input[0] = 8:
        // First pass col 0: i0=8, i4=0, i8=0, i12=0.
        //   a1 = 8+0 = 8; b1 = 8-0 = 8;
        //   c1 = (0*S>>16) - (0 + (0*C>>16)) = 0;
        //   d1 = (0 + (0*C>>16)) + (0*S>>16) = 0;
        //   tmp[0] = a1+d1 = 8; tmp[12] = a1-d1 = 8;
        //   tmp[4] = b1+c1 = 8; tmp[8]  = b1-c1 = 8.
        // Cols 1..3 zero → every tmp[*] in col 0 = 8, rest 0.
        // Second pass each row r: r0 = 8 (col0), r1=r2=r3=0.
        //   a1 = 8+0 = 8; b1 = 8-0 = 8; c1 = 0; d1 = 0;
        //   out[0] = (8+0+4)>>3 = 1; out[3] = (8-0+4)>>3 = 1;
        //   out[1] = (8+0+4)>>3 = 1; out[2] = (8-0+4)>>3 = 1.
        // So the whole 4x4 is 1.
        let mut input = [0i16; 16];
        input[0] = 8;
        let mut output = [0i16; 16];
        inverse_dct_4x4(&input, &mut output);
        assert_eq!(output, [1i16; 16]);
    }

    #[test]
    fn dct_dc_rounding_with_larger_value() {
        // input[0] = 64. First pass: each col-0 tmp = 64. Second pass:
        // out = (64 + 4) >> 3 = 8 for every pixel.
        let mut input = [0i16; 16];
        input[0] = 64;
        let mut output = [0i16; 16];
        inverse_dct_4x4(&input, &mut output);
        assert_eq!(output, [8i16; 16]);
    }

    #[test]
    fn dct_first_ac_coefficient_uses_cosine_constants() {
        // Place a single non-zero in input[1] (row 0, col 1) and verify
        // the output exactly matches a from-scratch evaluation of the
        // two-pass spec arithmetic, proving the SINPI8_SQRT2 /
        // COSPI8_SQRT2_MINUS1 constants are wired into the right lanes.
        let mut input = [0i16; 16];
        input[1] = 100;
        let mut output = [0i16; 16];
        inverse_dct_4x4(&input, &mut output);
        // Independent re-derivation of the spec C source.
        let mut tmp = [0i32; 16];
        for col in 0..4 {
            let i0 = input[col] as i32;
            let i4 = input[4 + col] as i32;
            let i8 = input[8 + col] as i32;
            let i12 = input[12 + col] as i32;
            let a1 = i0 + i8;
            let b1 = i0 - i8;
            let t1 = (i4 * SINPI8_SQRT2) >> 16;
            let t2 = i12 + ((i12 * COSPI8_SQRT2_MINUS1) >> 16);
            let c1 = t1 - t2;
            let t1b = i4 + ((i4 * COSPI8_SQRT2_MINUS1) >> 16);
            let t2b = (i12 * SINPI8_SQRT2) >> 16;
            let d1 = t1b + t2b;
            tmp[col] = a1 + d1;
            tmp[12 + col] = a1 - d1;
            tmp[4 + col] = b1 + c1;
            tmp[8 + col] = b1 - c1;
        }
        let mut expect = [0i16; 16];
        for row in 0..4 {
            let base = row * 4;
            let r0 = tmp[base];
            let r1 = tmp[base + 1];
            let r2 = tmp[base + 2];
            let r3 = tmp[base + 3];
            let a1 = r0 + r2;
            let b1 = r0 - r2;
            let t1 = (r1 * SINPI8_SQRT2) >> 16;
            let t2 = r3 + ((r3 * COSPI8_SQRT2_MINUS1) >> 16);
            let c1 = t1 - t2;
            let t1b = r1 + ((r1 * COSPI8_SQRT2_MINUS1) >> 16);
            let t2b = (r3 * SINPI8_SQRT2) >> 16;
            let d1 = t1b + t2b;
            expect[base] = ((a1 + d1 + 4) >> 3) as i16;
            expect[base + 3] = ((a1 - d1 + 4) >> 3) as i16;
            expect[base + 1] = ((b1 + c1 + 4) >> 3) as i16;
            expect[base + 2] = ((b1 - c1 + 4) >> 3) as i16;
        }
        assert_eq!(output, expect);
        // And it must not be a flat block (an AC coefficient creates a
        // gradient), which proves the constants actually contributed.
        assert!(output.iter().any(|&v| v != output[0]));
    }

    #[test]
    fn dct_is_separable_two_pass_roundtrip_shape() {
        // A small full-block sanity check: known input, output computed
        // by the same spec arithmetic re-derived inline. This guards
        // against an accidental transpose of the row/column passes.
        let input = [
            16, 4, 0, -2, //
            8, 1, 0, 0, //
            -4, 0, 0, 0, //
            2, 0, 0, 0, //
        ];
        let mut output = [0i16; 16];
        inverse_dct_4x4(&input, &mut output);

        let mut tmp = [0i32; 16];
        for col in 0..4 {
            let i0 = input[col] as i32;
            let i4 = input[4 + col] as i32;
            let i8 = input[8 + col] as i32;
            let i12 = input[12 + col] as i32;
            let a1 = i0 + i8;
            let b1 = i0 - i8;
            let t1 = (i4 * SINPI8_SQRT2) >> 16;
            let t2 = i12 + ((i12 * COSPI8_SQRT2_MINUS1) >> 16);
            let c1 = t1 - t2;
            let t1b = i4 + ((i4 * COSPI8_SQRT2_MINUS1) >> 16);
            let t2b = (i12 * SINPI8_SQRT2) >> 16;
            let d1 = t1b + t2b;
            tmp[col] = a1 + d1;
            tmp[12 + col] = a1 - d1;
            tmp[4 + col] = b1 + c1;
            tmp[8 + col] = b1 - c1;
        }
        let mut expect = [0i16; 16];
        for row in 0..4 {
            let base = row * 4;
            let r0 = tmp[base];
            let r1 = tmp[base + 1];
            let r2 = tmp[base + 2];
            let r3 = tmp[base + 3];
            let a1 = r0 + r2;
            let b1 = r0 - r2;
            let t1 = (r1 * SINPI8_SQRT2) >> 16;
            let t2 = r3 + ((r3 * COSPI8_SQRT2_MINUS1) >> 16);
            let c1 = t1 - t2;
            let t1b = r1 + ((r1 * COSPI8_SQRT2_MINUS1) >> 16);
            let t2b = (r3 * SINPI8_SQRT2) >> 16;
            let d1 = t1b + t2b;
            expect[base] = ((a1 + d1 + 4) >> 3) as i16;
            expect[base + 3] = ((a1 - d1 + 4) >> 3) as i16;
            expect[base + 1] = ((b1 + c1 + 4) >> 3) as i16;
            expect[base + 2] = ((b1 - c1 + 4) >> 3) as i16;
        }
        assert_eq!(output, expect);
    }

    #[test]
    fn add_residue_4x4_saturates() {
        let pred = [200u8; 16];
        let mut residue = [0i16; 16];
        residue[0] = 100; // 200 + 100 = 300 → 255
        residue[1] = -250; // 200 - 250 = -50 → 0
        residue[2] = 30; // 200 + 30 = 230
        let mut out = [0u8; 16];
        add_residue_4x4(&pred, &residue, &mut out);
        assert_eq!(out[0], 255);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 230);
        assert_eq!(out[3], 200); // unchanged
    }

    #[test]
    fn add_residue_slice_matches_4x4() {
        let pred = [128u8; 64];
        let mut residue = [0i16; 64];
        residue[10] = 50;
        residue[20] = -200;
        residue[63] = 127;
        let mut out = [0u8; 64];
        add_residue(&pred, &residue, &mut out);
        assert_eq!(out[10], 178);
        assert_eq!(out[20], 0);
        assert_eq!(out[63], 255);
        assert_eq!(out[0], 128);
    }

    #[test]
    fn add_residue_zero_residue_is_identity() {
        let pred = [37u8; 16];
        let residue = [0i16; 16];
        let mut out = [0u8; 16];
        add_residue_4x4(&pred, &residue, &mut out);
        assert_eq!(out, pred);
    }
}
