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
///
/// Dispatch: SIMD path on nightly + `simd`, scalar otherwise. The SIMD
/// path is byte-exact against the scalar listing on every test fixture
/// (`fwht_forward_simd_matches_scalar_on_stress_inputs`).
pub fn forward_wht_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    #[cfg(feature = "simd")]
    {
        forward_wht_4x4_simd(input, output);
    }
    #[cfg(not(feature = "simd"))]
    {
        forward_wht_4x4_scalar(input, output);
    }
}

/// Scalar §14.3 forward WHT — the derivation written out longhand from
/// the §14.3 inverse listing.
///
/// The public [`forward_wht_4x4`] dispatches here on stable builds (and
/// on nightly without the `simd` feature); the `simd` feature swaps in
/// [`forward_wht_4x4_simd`], which is itself byte-exact against this
/// implementation (`fwht_forward_simd_matches_scalar_on_stress_inputs`).
#[allow(dead_code)] // Used by `forward_wht_4x4` only on the !simd path.
fn forward_wht_4x4_scalar(input: &[i16; 16], output: &mut [i16; 16]) {
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

/// SIMD §14.3 forward WHT — `core::simd::Simd<i32, 4>` rewrite of
/// [`forward_wht_4x4_scalar`].
///
/// The §14.3 forward WHT is the same separable two-pass shape as the
/// §14.3 inverse: both passes operate on four independent 1-D 4-point
/// transforms. Holding the input matrix as four `Simd<i32, 4>` row
/// vectors (lane `j` of row `i` carries `input[i*4 + j]`), the
/// first-pass column butterfly maps onto four parallel SIMD adds /
/// subs across the column axis. A transpose then puts each row of the
/// intermediate into a row-vector so the second-pass row butterfly +
/// `round_div2` lane-wide step run as four further SIMD adds / subs.
///
/// The forward path differs from the inverse path only in the
/// rounding step at the bottom of the second pass: where the inverse
/// uses `(x + 3) >> 3`, the forward uses the symmetric
/// `round_div2(x)` (round-half-away-from-zero of `x / 2`). That step
/// is expressed lane-wide as `select(x >= 0, (x + 1) >> 1, -((-x + 1) >> 1))`;
/// the resulting per-lane value matches the scalar `round_div2` bit-
/// for-bit because both branches are lane-wide arithmetic on `i32`s.
///
/// No external SIMD reference consulted — the layout is derived from
/// the scalar §14.3 listing directly (one lane per column ⇒
/// first-pass vectorises; transpose ⇒ second-pass vectorises). Byte-
/// exact against [`forward_wht_4x4_scalar`] on every test fixture.
#[cfg(feature = "simd")]
fn forward_wht_4x4_simd(input: &[i16; 16], output: &mut [i16; 16]) {
    use core::simd::Simd;

    // Row-vectors: row_i[j] = input[i*4 + j]. Lane j is column j.
    let row0 = Simd::<i32, 4>::from_array([
        input[0] as i32,
        input[1] as i32,
        input[2] as i32,
        input[3] as i32,
    ]);
    let row1 = Simd::<i32, 4>::from_array([
        input[4] as i32,
        input[5] as i32,
        input[6] as i32,
        input[7] as i32,
    ]);
    let row2 = Simd::<i32, 4>::from_array([
        input[8] as i32,
        input[9] as i32,
        input[10] as i32,
        input[11] as i32,
    ]);
    let row3 = Simd::<i32, 4>::from_array([
        input[12] as i32,
        input[13] as i32,
        input[14] as i32,
        input[15] as i32,
    ]);

    // First pass — §14.3 forward column butterfly across the four
    // lanes. (row0, row1, row2, row3) play the role of (i0, i4, i8, i12)
    // in the scalar listing:
    //   a1 = i0 + i4;  b1 = i8 + i12;  c1 = i0 - i4;  d1 = i8 - i12
    //   tmp[0]  = a1 + b1
    //   tmp[4]  = a1 - b1
    //   tmp[8]  = c1 - d1
    //   tmp[12] = c1 + d1
    let a1 = row0 + row1;
    let b1 = row2 + row3;
    let c1 = row0 - row1;
    let d1 = row2 - row3;

    let t0 = a1 + b1;
    let t1 = a1 - b1;
    let t2 = c1 - d1;
    let t3 = c1 + d1;

    // Transpose so each row-vector now carries one row of the
    // intermediate; r_i[j] = t_j[i].
    let t0a = t0.to_array();
    let t1a = t1.to_array();
    let t2a = t2.to_array();
    let t3a = t3.to_array();

    let r0 = Simd::<i32, 4>::from_array([t0a[0], t1a[0], t2a[0], t3a[0]]);
    let r1 = Simd::<i32, 4>::from_array([t0a[1], t1a[1], t2a[1], t3a[1]]);
    let r2 = Simd::<i32, 4>::from_array([t0a[2], t1a[2], t2a[2], t3a[2]]);
    let r3v = Simd::<i32, 4>::from_array([t0a[3], t1a[3], t2a[3], t3a[3]]);

    // Second pass — §14.3 forward row butterfly. (r0, r1, r2, r3v)
    // play the role of (r0, r1, r2, r3) in the scalar listing:
    //   a1 = r0 + r1; b1 = r2 + r3; c1 = r0 - r1; d1 = r2 - r3
    //   o0 = a1 + b1; o1 = a1 - b1; o2 = c1 - d1; o3 = c1 + d1
    let a2 = r0 + r1;
    let b2 = r2 + r3v;
    let c2 = r0 - r1;
    let d2 = r2 - r3v;

    let u0 = a2 + b2;
    let u1 = a2 - b2;
    let u2 = c2 - d2;
    let u3 = c2 + d2;

    // Lane-wide symmetric `/2 with round-half-away-from-zero`:
    //   v >= 0 ⇒ (v + 1) >> 1
    //   v <  0 ⇒ -((-v + 1) >> 1)
    // Each branch is computed unconditionally on the full lane, then
    // SIMD `select` picks per lane. This matches `round_div2` bit-
    // for-bit because both lanes operate on `i32` and the shifts are
    // arithmetic on signed lanes.
    let one = Simd::<i32, 4>::splat(1);
    let zero = Simd::<i32, 4>::splat(0);
    let o0 = round_div2_simd(u0, one, zero);
    let o1 = round_div2_simd(u1, one, zero);
    let o2 = round_div2_simd(u2, one, zero);
    let o3 = round_div2_simd(u3, one, zero);

    // o_i was computed from r_{0..3v}, each of which holds the i-th
    // *column* of the post-first-pass matrix. So o0[j] is the
    // butterfly output for row j, col 0 — i.e. output[j*4 + 0].
    let o0a = o0.to_array();
    let o1a = o1.to_array();
    let o2a = o2.to_array();
    let o3a = o3.to_array();
    for j in 0..4 {
        output[j * 4] = o0a[j] as i16;
        output[j * 4 + 1] = o1a[j] as i16;
        output[j * 4 + 2] = o2a[j] as i16;
        output[j * 4 + 3] = o3a[j] as i16;
    }
}

/// Lane-wide `round_div2` — see scalar `round_div2`. Hoisted into a
/// helper so the second-pass output store doesn't repeat the same
/// `select` chain four times.
///
/// The two arithmetic branches are computed unconditionally and then
/// merged via `simd_ge(0)`. `one` and `zero` are taken as parameters
/// so the caller (which already holds them as locals) avoids the
/// per-call `Simd::splat`. The lane-wide `simd_clamp` mirrors the
/// scalar `clamp(i16::MIN as i32, i16::MAX as i32)` so the SIMD path
/// matches scalar bit-for-bit on inputs whose post-`/2` value falls
/// outside the `i16` envelope (the path is purely defensive — §14.3
/// forward output stays within `i16` for every legal residual block).
#[cfg(feature = "simd")]
#[inline]
fn round_div2_simd(
    v: core::simd::Simd<i32, 4>,
    one: core::simd::Simd<i32, 4>,
    zero: core::simd::Simd<i32, 4>,
) -> core::simd::Simd<i32, 4> {
    use core::simd::cmp::SimdOrd;
    use core::simd::cmp::SimdPartialOrd;
    use core::simd::Select;
    use core::simd::Simd;

    let pos = (v + one) >> Simd::<i32, 4>::splat(1);
    let neg = zero - (((zero - v) + one) >> Simd::<i32, 4>::splat(1));
    // `Mask::select` is provided by the `core::simd::Select` trait on
    // the current nightly portable_simd surface. The cmp returns a
    // `Mask<i32, 4>` whose `select` chooses the positive branch where
    // `v >= 0` and the negative branch elsewhere — matching the scalar
    // `round_div2` lane-by-lane.
    let ge_mask = v.simd_ge(zero);
    let rounded = ge_mask.select(pos, neg);
    let min = Simd::<i32, 4>::splat(i16::MIN as i32);
    let max = Simd::<i32, 4>::splat(i16::MAX as i32);
    rounded.simd_clamp(min, max)
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
///
/// Dispatch: always the scalar path. The §14.4 SIMD partner
/// [`forward_dct_4x4_simd`] (compiled under the nightly-only `simd`
/// feature) is kept available and is asserted byte-exact against the
/// scalar listing on a 21-input stress set
/// (`fdct_forward_simd_matches_scalar_on_stress_inputs`), but the
/// `forward_transform_4x4/forward_dct_4x4` criterion `--quick` numbers
/// recorded in `BENCHMARKS.md` round 226 show the SIMD path is ~+8 %
/// slower than the scalar straight-line code on `aarch64-apple-darwin`:
/// the eight fixed-point `c_mul` / `s_mul` lane-wide multiplies per
/// pass (× 2 passes) plus the lane-wide `round_div2_simd` mask + select
/// don't pipeline as well as the scalar path. Round 247 therefore
/// routes the public dispatcher to [`forward_dct_4x4_scalar`] under
/// every feature configuration; the `_simd` implementation stays in
/// place so a future round can re-target it (e.g. on a host where the
/// regression flips), and the byte-equivalence test now calls the
/// `_simd` path directly so the equivalence proof is preserved
/// regardless of the public dispatch.
pub fn forward_dct_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    forward_dct_4x4_scalar(input, output);
}

/// Scalar §14.4 forward DCT — the longhand derivation from the §14.4
/// inverse listing.
///
/// Round 249 reorganised the listing into the canonical butterfly
/// shape mirroring the §14.4 inverse's `inverse_dct_4x4_scalar`
/// listing. The inverse listing splits each 1-D pass into four
/// partial-sums (`a1 = i0 + i8`, `b1 = i0 - i8`, plus a `c1` / `d1`
/// pair carrying the fixed-point cosine / sine multiplies); the
/// forward DCT is the transpose of that, so the corresponding pairs
/// shift one slot — `a1 = i0 + i12`, `b1 = i4 + i8`, with `c1` / `d1`
/// holding the multiply pair-differences for the odd outputs. The
/// even outputs become `o0 = a1 + b1` and `o8 = a1 - b1`, mirroring
/// the inverse's `op[0] = a1 + d1` / `op[12] = a1 - d1` shape. The
/// odd outputs `o4 = c1` and `o12 = d1` are decomposed into pair-
/// differences like `(c_mul(i0) - c_mul(i12)) + (s_mul(i4) - s_mul(i8))`,
/// each `c_mul` / `s_mul` call evaluated separately so the fixed-point
/// rounding stays bit-exact against the unfactored direct-derivation
/// form (verified by the test
/// `fdct_scalar_matches_direct_derivation_listing` against
/// `forward_dct_4x4_listing`, which is the spec derivation written
/// out longhand).
///
/// The public [`forward_dct_4x4`] dispatches here under every feature
/// configuration as of round 247 (see the dispatcher docstring for the
/// rationale — the `simd` lane-wide `c_mul` / `s_mul` chain regresses
/// vs scalar on `aarch64-apple-darwin`). The `simd` feature's
/// [`forward_dct_4x4_simd`] partner is still compiled and is byte-exact
/// against this implementation
/// (`fdct_forward_simd_matches_scalar_on_stress_inputs`); a future
/// round can flip the public dispatch back to SIMD if benchmarks on a
/// different host justify it.
fn forward_dct_4x4_scalar(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    // First pass: operate down each column. Written in the same
    // `(a1, b1, c1, d1)` partial-sum butterfly shape the §14.4 inverse
    // listing uses, transposed: where the inverse pairs (i0, i8) and
    // (i4, i12), the forward pairs (i0, i12) and (i4, i8) so the
    // `o0 = a1 + b1` / `o8 = a1 - b1` evens reuse a single set of
    // partial sums and the odd outputs collect their multiplies into
    // `c1`, `d1`. `c_mul` / `s_mul` use the same fixed-point evaluation
    // as §14.4 so the rounding shape is consistent between forward and
    // inverse; each pair-difference like `c_mul(i0) - c_mul(i12)` is
    // bit-exact against the unfactored sum `c_mul(i0) + (-c_mul(i12))`
    // (addition is associative on i32) but is NOT collapsed into
    // `c_mul(i0 - i12)` because the fixed-point truncation in `>> 16`
    // is non-linear under sum.
    for col in 0..4 {
        let i0 = input[col] as i32;
        let i4 = input[4 + col] as i32;
        let i8 = input[8 + col] as i32;
        let i12 = input[12 + col] as i32;

        // Even-output partial sums:
        //   o0  = i0 + i4 + i8 + i12 = (i0 + i12) + (i4 + i8) = a1 + b1
        //   o8  = i0 - i4 - i8 + i12 = (i0 + i12) - (i4 + i8) = a1 - b1
        let a1 = i0 + i12;
        let b1 = i4 + i8;

        // Odd-output multiply pair-differences. Splitting each `o4`,
        // `o12` listing into its (i0, i12) and (i4, i8) halves makes
        // the symmetry with the inverse's `c1` / `d1` form explicit:
        //   o4  = c_mul(i0)*1 + s_mul(i4)*1 - s_mul(i8)*1 - c_mul(i12)*1
        //       = (c_mul(i0) - c_mul(i12)) + (s_mul(i4) - s_mul(i8))
        //   o12 = s_mul(i0) - c_mul(i4)   + c_mul(i8)   - s_mul(i12)
        //       = (s_mul(i0) - s_mul(i12)) - (c_mul(i4) - c_mul(i8))
        let c1 = (c_mul(i0) - c_mul(i12)) + (s_mul(i4) - s_mul(i8));
        let d1 = (s_mul(i0) - s_mul(i12)) - (c_mul(i4) - c_mul(i8));

        tmp[col] = a1 + b1;
        tmp[4 + col] = c1;
        tmp[8 + col] = a1 - b1;
        tmp[12 + col] = d1;
    }

    // Second pass: operate across each row, then symmetric `/2` round.
    // Same butterfly shape as the column pass, with (r0, r1, r2, r3)
    // playing the role of (i0, i4, i8, i12) in the listing: pair
    // (r0, r3) and (r1, r2) so the evens reuse a single set of partial
    // sums.
    for row in 0..4 {
        let base = row * 4;
        let r0 = tmp[base];
        let r1 = tmp[base + 1];
        let r2 = tmp[base + 2];
        let r3 = tmp[base + 3];

        // Even-output partial sums:
        //   o0 = r0 + r1 + r2 + r3 = (r0 + r3) + (r1 + r2) = a1 + b1
        //   o2 = r0 - r1 - r2 + r3 = (r0 + r3) - (r1 + r2) = a1 - b1
        let a1 = r0 + r3;
        let b1 = r1 + r2;

        // Odd-output multiply pair-differences (same shape as the
        // column pass):
        //   o1 = (c_mul(r0) - c_mul(r3)) + (s_mul(r1) - s_mul(r2))
        //   o3 = (s_mul(r0) - s_mul(r3)) - (c_mul(r1) - c_mul(r2))
        let c1 = (c_mul(r0) - c_mul(r3)) + (s_mul(r1) - s_mul(r2));
        let d1 = (s_mul(r0) - s_mul(r3)) - (c_mul(r1) - c_mul(r2));

        output[base] = round_div2(a1 + b1);
        output[base + 1] = round_div2(c1);
        output[base + 2] = round_div2(a1 - b1);
        output[base + 3] = round_div2(d1);
    }
}

/// SIMD §14.4 forward DCT — `core::simd::Simd<i32, 4>` rewrite of
/// [`forward_dct_4x4_scalar`].
///
/// Same 4-lane layout as `inverse_dct_4x4_simd`: hold the input as
/// four row-vectors where lane `j` of `row_i` carries `input[i*4 + j]`.
/// The first-pass column butterfly maps onto four parallel lane-wide
/// adds, subs, and the fixed-point `c_mul` / `s_mul` multiplies (the
/// latter become `(x * splat(SINPI8_SQRT2)) >> splat(16)` etc., with
/// the wrapping i32 lane product producing identical bytes to the
/// scalar `(x * K) >> 16`). Transpose the intermediate, run the same
/// shape over the rows, and apply `round_div2_simd` lane-wide for the
/// symmetric `/2` rounding step.
///
/// No external SIMD reference consulted — the layout is derived from
/// the scalar §14.4 forward listing directly (one lane per column ⇒
/// first-pass vectorises; transpose ⇒ second-pass vectorises). Byte-
/// exact against [`forward_dct_4x4_scalar`] on every test fixture.
///
/// As of round 247 the public dispatcher [`forward_dct_4x4`] routes to
/// the scalar path under every feature configuration (see the
/// dispatcher docstring for the bench evidence). This `_simd`
/// implementation stays compiled under the `simd` feature so the
/// `fdct_forward_simd_matches_scalar_on_stress_inputs` byte-
/// equivalence assertion still runs on every nightly + `simd` test
/// pass; a future round can re-target the dispatcher to this function
/// if hardware-specific benches justify it.
#[cfg(feature = "simd")]
#[allow(dead_code)] // Reached only by the cfg(test) equivalence assertion.
fn forward_dct_4x4_simd(input: &[i16; 16], output: &mut [i16; 16]) {
    use core::simd::Simd;

    // Row-vectors: row_i[j] = input[i*4 + j]. Lane j is column j.
    let row0 = Simd::<i32, 4>::from_array([
        input[0] as i32,
        input[1] as i32,
        input[2] as i32,
        input[3] as i32,
    ]);
    let row1 = Simd::<i32, 4>::from_array([
        input[4] as i32,
        input[5] as i32,
        input[6] as i32,
        input[7] as i32,
    ]);
    let row2 = Simd::<i32, 4>::from_array([
        input[8] as i32,
        input[9] as i32,
        input[10] as i32,
        input[11] as i32,
    ]);
    let row3 = Simd::<i32, 4>::from_array([
        input[12] as i32,
        input[13] as i32,
        input[14] as i32,
        input[15] as i32,
    ]);

    // Splatted constants. Holding the splats in locals lets LLVM keep
    // them in registers across the two passes.
    let sin = Simd::<i32, 4>::splat(SINPI8_SQRT2);
    let cos_m1 = Simd::<i32, 4>::splat(COSPI8_SQRT2_MINUS1);
    let sh16 = Simd::<i32, 4>::splat(16);

    // First pass — §14.4 forward column butterfly across the four
    // lanes. (row0, row1, row2, row3) play the role of (i0, i4, i8, i12)
    // in the scalar listing:
    //   o0  = i0 + i4 + i8 + i12
    //   o4  = i0*C + i4*S - i8*S - i12*C
    //   o8  = i0 - i4 - i8 + i12
    //   o12 = i0*S - i4*C + i8*C - i12*S
    let t0 = row0 + row1 + row2 + row3;
    let t2 = row0 - row1 - row2 + row3;

    let c0 = row0 + ((row0 * cos_m1) >> sh16);
    let s1 = (row1 * sin) >> sh16;
    let s2 = (row2 * sin) >> sh16;
    let c3 = row3 + ((row3 * cos_m1) >> sh16);
    let t1 = c0 + s1 - s2 - c3;

    let s0 = (row0 * sin) >> sh16;
    let c1v = row1 + ((row1 * cos_m1) >> sh16);
    let c2v = row2 + ((row2 * cos_m1) >> sh16);
    let s3 = (row3 * sin) >> sh16;
    let t3 = s0 - c1v + c2v - s3;

    // Intermediate ordering after the first pass:
    //   tmp_row_0 = t0  (the `o0` outputs across the four columns)
    //   tmp_row_1 = t1  (the `o4` outputs across the four columns)
    //   tmp_row_2 = t2  (the `o8` outputs across the four columns)
    //   tmp_row_3 = t3  (the `o12` outputs across the four columns)
    let t0a = t0.to_array();
    let t1a = t1.to_array();
    let t2a = t2.to_array();
    let t3a = t3.to_array();

    // Transpose so each row-vector now carries one row of the
    // intermediate (lane j of new_row_i = old t_i[j]) ⇒ each new
    // row-vector is one input to a second-pass 4-point 1-D DCT.
    let r0 = Simd::<i32, 4>::from_array([t0a[0], t1a[0], t2a[0], t3a[0]]);
    let r1 = Simd::<i32, 4>::from_array([t0a[1], t1a[1], t2a[1], t3a[1]]);
    let r2 = Simd::<i32, 4>::from_array([t0a[2], t1a[2], t2a[2], t3a[2]]);
    let r3v = Simd::<i32, 4>::from_array([t0a[3], t1a[3], t2a[3], t3a[3]]);

    // Second pass — §14.4 forward row butterfly. Same shape as the
    // column pass; (r0, r1, r2, r3v) play the role of (r0, r1, r2, r3):
    //   o0 = r0 + r1 + r2 + r3
    //   o1 = r0*C + r1*S - r2*S - r3*C
    //   o2 = r0 - r1 - r2 + r3
    //   o3 = r0*S - r1*C + r2*C - r3*S
    let u0 = r0 + r1 + r2 + r3v;
    let u2 = r0 - r1 - r2 + r3v;

    let c0r = r0 + ((r0 * cos_m1) >> sh16);
    let s1r = (r1 * sin) >> sh16;
    let s2r = (r2 * sin) >> sh16;
    let c3r = r3v + ((r3v * cos_m1) >> sh16);
    let u1 = c0r + s1r - s2r - c3r;

    let s0r = (r0 * sin) >> sh16;
    let c1r = r1 + ((r1 * cos_m1) >> sh16);
    let c2r = r2 + ((r2 * cos_m1) >> sh16);
    let s3r = (r3v * sin) >> sh16;
    let u3 = s0r - c1r + c2r - s3r;

    // Lane-wide symmetric `/2` rounding (matches scalar `round_div2`).
    let one = Simd::<i32, 4>::splat(1);
    let zero = Simd::<i32, 4>::splat(0);
    let o0 = round_div2_simd(u0, one, zero);
    let o1 = round_div2_simd(u1, one, zero);
    let o2 = round_div2_simd(u2, one, zero);
    let o3 = round_div2_simd(u3, one, zero);

    // u_k holds, at lane j, the second-pass output for row j at
    // column k. Stripe back into raster order: output[j*4 + k] = u_k[j].
    let o0a = o0.to_array();
    let o1a = o1.to_array();
    let o2a = o2.to_array();
    let o3a = o3.to_array();
    for j in 0..4 {
        output[j * 4] = o0a[j] as i16;
        output[j * 4 + 1] = o1a[j] as i16;
        output[j * 4 + 2] = o2a[j] as i16;
        output[j * 4 + 3] = o3a[j] as i16;
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

    // -----------------------------------------------------------------
    // SIMD vs scalar byte-exact equivalence tests (round-226 forward
    // SIMD rewrite).
    //
    // Mirrors the inverse-side `dct_simd_matches_scalar_on_stress_inputs`
    // / `wht_simd_matches_scalar_on_stress_inputs` pair. On stable / no
    // `simd` feature they exercise the scalar path twice (the public
    // dispatch and the directly-named `_scalar` fn) — harmless; keeps
    // CI green. On nightly + `simd` they're the primary safety net:
    // the public `forward_wht_4x4` / `forward_dct_4x4` dispatch to the
    // new SIMD path, and the `_scalar` variants stay reachable as
    // module-private fallbacks, so the assertions compare the two
    // bit-for-bit.
    // -----------------------------------------------------------------

    /// 21 stress inputs covering DC-only, single AC lanes (every
    /// position), mixed gradients, and near-i16 extremes — same shape
    /// as `inverse_transform::tests::stress_inputs`.
    fn forward_stress_inputs() -> Vec<[i16; 16]> {
        let mut v: Vec<[i16; 16]> = Vec::new();
        // All-zero.
        v.push([0i16; 16]);
        // DC-only (residual ranges up to a few hundred LSB).
        for &dc in &[1i16, 7, 8, -8, 64, -64, 200, -200, 500, -500] {
            let mut a = [0i16; 16];
            a[0] = dc;
            v.push(a);
        }
        // Single-AC at each position.
        for pos in 1..16 {
            let mut a = [0i16; 16];
            a[pos] = 100;
            v.push(a);
        }
        // A mixed low-frequency pattern (the bench's SAMPLE_INPUT).
        v.push([
            320, -64, 16, -4, //
            -48, 32, -16, 8, //
            24, -12, 8, -4, //
            -8, 4, -2, 1,
        ]);
        // A high-AC mid-range pattern.
        v.push([
            512, 256, 128, 64, //
            256, 128, 64, 32, //
            128, 64, 32, 16, //
            64, 32, 16, 8,
        ]);
        // Same with alternating signs.
        v.push([
            -512, 256, -128, 64, //
            256, -128, 64, -32, //
            -128, 64, -32, 16, //
            64, -32, 16, -8,
        ]);
        v
    }

    /// Reference §14.4 forward DCT — the spec derivation written out
    /// in the unfactored direct form `o0 = i0 + i4 + i8 + i12; o4 =
    /// c_mul(i0) + s_mul(i4) - s_mul(i8) - c_mul(i12); ...` (no `a1` /
    /// `b1` partial sums shared between the even outputs, no
    /// `c_mul(i0) - c_mul(i12)` pair-differences). The round-249
    /// refactor of `forward_dct_4x4_scalar` reorganises the same
    /// arithmetic into the canonical butterfly shape mirroring the
    /// §14.4 inverse listing; this function is the regression guard
    /// proving that reorganisation is bit-exact (integer addition is
    /// associative on i32 and we never collapse a pair-difference into
    /// `c_mul(i0 - i12)`, which would change the fixed-point rounding).
    /// Used only by `fdct_scalar_matches_direct_derivation_listing`.
    fn forward_dct_4x4_listing(input: &[i16; 16], output: &mut [i16; 16]) {
        let mut tmp = [0i32; 16];
        for col in 0..4 {
            let i0 = input[col] as i32;
            let i4 = input[4 + col] as i32;
            let i8 = input[8 + col] as i32;
            let i12 = input[12 + col] as i32;

            let o0 = i0 + i4 + i8 + i12;
            let o8 = i0 - i4 - i8 + i12;

            let c0 = c_mul(i0);
            let s4 = s_mul(i4);
            let s8 = s_mul(i8);
            let c12 = c_mul(i12);
            let o4 = c0 + s4 - s8 - c12;

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

    /// Round-249 regression guard: the refactored `forward_dct_4x4_scalar`
    /// (butterfly shape mirroring the §14.4 inverse listing) is byte-
    /// exact against `forward_dct_4x4_listing` (the unfactored direct
    /// spec-derivation listing) on the 21-input stress matrix. Any
    /// future change to the scalar listing that breaks this assertion
    /// is changing the output bytes, not just the readability shape.
    #[test]
    fn fdct_scalar_matches_direct_derivation_listing() {
        for (idx, input) in forward_stress_inputs().iter().enumerate() {
            let mut via_scalar = [0i16; 16];
            forward_dct_4x4_scalar(input, &mut via_scalar);
            let mut via_listing = [0i16; 16];
            forward_dct_4x4_listing(input, &mut via_listing);
            assert_eq!(
                via_scalar, via_listing,
                "input #{idx} ({input:?}): refactored _scalar ≠ direct-derivation listing"
            );
        }
    }

    /// `forward_dct_4x4_simd` is byte-exact against `forward_dct_4x4_scalar`
    /// on the 21-input stress matrix. Compared to the round-226 shape
    /// (`public dispatch ≠ scalar`) this test now goes directly through
    /// the `_simd` symbol because the round-247 public dispatcher routes
    /// to `_scalar` under every feature configuration (the `--quick`
    /// numbers in `BENCHMARKS.md` showed the SIMD path regresses for
    /// the forward DCT). The equivalence proof has to survive the
    /// dispatcher change, so this asserts `simd == scalar` directly
    /// rather than going through the public function.
    #[cfg(feature = "simd")]
    #[test]
    fn fdct_forward_simd_matches_scalar_on_stress_inputs() {
        for (idx, input) in forward_stress_inputs().iter().enumerate() {
            let mut via_simd = [0i16; 16];
            forward_dct_4x4_simd(input, &mut via_simd);
            let mut via_scalar = [0i16; 16];
            forward_dct_4x4_scalar(input, &mut via_scalar);
            assert_eq!(
                via_simd, via_scalar,
                "input #{idx} ({input:?}): _simd ≠ _scalar"
            );
        }
    }

    /// Public dispatch matches the scalar listing under every feature
    /// configuration after round 247. Cheap sanity check that the
    /// dispatcher didn't get accidentally rerouted in a future round.
    #[test]
    fn fdct_public_dispatch_is_scalar() {
        for (idx, input) in forward_stress_inputs().iter().enumerate() {
            let mut via_public = [0i16; 16];
            forward_dct_4x4(input, &mut via_public);
            let mut via_scalar = [0i16; 16];
            forward_dct_4x4_scalar(input, &mut via_scalar);
            assert_eq!(
                via_public, via_scalar,
                "input #{idx} ({input:?}): public dispatch ≠ scalar listing"
            );
        }
    }

    #[test]
    fn fwht_forward_simd_matches_scalar_on_stress_inputs() {
        for (idx, input) in forward_stress_inputs().iter().enumerate() {
            let mut via_public = [0i16; 16];
            forward_wht_4x4(input, &mut via_public);
            let mut via_scalar = [0i16; 16];
            forward_wht_4x4_scalar(input, &mut via_scalar);
            assert_eq!(
                via_public, via_scalar,
                "input #{idx} ({input:?}): public dispatch ≠ scalar listing"
            );
        }
    }
}
