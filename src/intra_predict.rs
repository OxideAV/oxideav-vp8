//! VP8 intra-prediction pixel kernels (RFC 6386 §12).
//!
//! This module implements the **pure pixel-shape** prediction kernels
//! from §12 of the spec:
//!
//! * The four 16×16 luma modes (`DC_PRED`, `V_PRED`, `H_PRED`,
//!   `TM_PRED`) of §12.3 — see [`predict_y16x16`].
//! * The four 8×8 chroma modes (the same four modes applied at 8×8) of
//!   §12.2 — see [`predict_uv8x8`].
//! * The ten 4×4 sub-block modes (`B_DC_PRED` … `B_HU_PRED`) of §12.3 —
//!   see [`predict_b4x4`].
//!
//! Each kernel takes its already-reconstructed neighbour pixels as
//! plain `&[u8]` inputs and writes the predicted block into a
//! caller-supplied flat output buffer. The kernels do **not** decode
//! the mode (that's [`crate::macroblock`]) and they do **not** consume
//! the DCT residue — adding residue is the IDCT stage, which is not yet
//! implemented.
//!
//! # Out-of-bounds neighbours
//!
//! Per §12 (page 50), when a macroblock sits on the top row or left
//! column of the frame, some of its neighbour pixels lie outside the
//! reconstructed image. The constants used in that case are:
//!
//! * 127 for pixels above the top row (including the corner pixel
//!   above-and-left of the top-left frame pixel);
//! * 129 for pixels to the left of the leftmost column.
//!
//! The DC mode of §12.2 has an exception: instead of averaging the
//! out-of-bounds defaults it falls back to averaging the genuinely
//! visible side, or — for the top-left macroblock — fills the block
//! with the constant 128 directly. [`predict_uv8x8_dc`] and
//! [`predict_y16x16_dc`] take explicit `above`/`left` `Option`s to
//! drive this branching: `None` means "this edge is off-frame", `Some`
//! means "use these pixels".
//!
//! # Sub-block "extra" pixels
//!
//! Per §12.3, sub-blocks 3, 7, 11, 15 along the right edge of the
//! macroblock would naturally reference pixels above-and-to-the-right
//! of themselves — but, because macroblocks are reconstructed in
//! raster order, those pixels haven't been built yet for sub-blocks 7,
//! 11, 15. The spec defines a fixup: all four right-edge sub-blocks
//! share the same `(-1, 16) .. (-1, 19)` pixels (i.e. the four pixels
//! immediately above and to the right of sub-block 3), and for the
//! rightmost macroblock in each row those four pixels are clamped to
//! the value at `(-1, 15)`. Computing that fixup is the caller's job;
//! the kernel takes an 8-pixel `above` row covering positions
//! `(-1, 0) .. (-1, 7)` of the sub-block frame of reference.
//!
//! # Reference
//!
//! RFC 6386 §12 (pages 50–59). The 16×16 and 8×8 modes are described
//! in §12.2 (for chroma) and at the top of §12.3 (which forwards luma
//! to the chroma description with `shf` bumped from 3/4 to 4/5). The
//! ten 4×4 modes are described in §12.3, listed as the body of
//! `subblock_intra_predict()`.
//!
//! ## Spec gap
//!
//! RFC 6386 page 58 (line in raw text containing the `B_HD_PRED` body)
//! has a typo: `svg2p(E + 1)` should read `avg2p(E + 1)`. There is no
//! `svg2p` function defined anywhere in the spec; `avg2p` is the
//! `(p[0] + p[1] + 1) >> 1` helper used throughout the surrounding
//! diagonal modes (B_HD_PRED's three siblings — B_VR_PRED, B_VL_PRED,
//! B_HU_PRED — all use `avg2p` at the analogous position with no
//! typo). We treat the `svg2p` token as `avg2p` here.

use crate::macroblock::{IntraBmode, IntraUvMode, IntraYMode};

/// Default pixel value used for above-row neighbour pixels that lie
/// outside the visible frame (RFC 6386 §12, page 50).
pub const DEFAULT_ABOVE_PIXEL: u8 = 127;

/// Default pixel value used for left-column neighbour pixels that lie
/// outside the visible frame (RFC 6386 §12, page 50).
pub const DEFAULT_LEFT_PIXEL: u8 = 129;

/// Default pixel value the DC chroma mode (§12.2) falls back to when
/// *both* neighbour edges are off-frame. (Note: this is **not** the
/// average of `DEFAULT_ABOVE_PIXEL` and `DEFAULT_LEFT_PIXEL` — the
/// spec explicitly fills the block with 128 in that case.)
pub const DEFAULT_TOPLEFT_DC: u8 = 128;

#[inline]
fn clamp255(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
fn avg3(x: u8, y: u8, z: u8) -> u8 {
    // RFC 6386 §12.3 page 55: avg3(x, y, z) = (x + y + y + z + 2) >> 2.
    let s = x as u32 + (y as u32) + (y as u32) + z as u32 + 2;
    (s >> 2) as u8
}

#[inline]
fn avg3p(p: &[u8], i: usize) -> u8 {
    // RFC 6386 §12.3: avg3p(p) = avg3(p[-1], p[0], p[1]). The caller
    // gives us a slice and a cursor `i` so that `i-1`, `i`, `i+1` are
    // all in bounds.
    avg3(p[i - 1], p[i], p[i + 1])
}

#[inline]
fn avg2(x: u8, y: u8) -> u8 {
    // RFC 6386 §12.3 page 55: avg2(x, y) = (x + y + 1) >> 1.
    ((x as u16 + y as u16 + 1) >> 1) as u8
}

#[inline]
fn avg2p(p: &[u8], i: usize) -> u8 {
    // RFC 6386 §12.3: avg2p(p) = avg2(p[0], p[1]).
    avg2(p[i], p[i + 1])
}

// ===========================================================================
// 16×16 luma modes (RFC 6386 §12.3) and 8×8 chroma modes (RFC 6386 §12.2)
// ===========================================================================

/// Generic "fill an N×N block with a single value" helper used by the
/// DC modes. `out` is a flat row-major buffer of `n * n` pixels.
fn fill(out: &mut [u8], n: usize, value: u8) {
    debug_assert_eq!(out.len(), n * n);
    for slot in out.iter_mut() {
        *slot = value;
    }
}

/// Generic vertical-prediction helper. `out` is `n * n` row-major;
/// `above` is `n` pixels.
fn predict_v(out: &mut [u8], n: usize, above: &[u8]) {
    debug_assert_eq!(out.len(), n * n);
    debug_assert_eq!(above.len(), n);
    for row in 0..n {
        let dst = &mut out[row * n..(row + 1) * n];
        dst.copy_from_slice(above);
    }
}

/// Generic horizontal-prediction helper. `left` is `n` pixels.
fn predict_h(out: &mut [u8], n: usize, left: &[u8]) {
    debug_assert_eq!(out.len(), n * n);
    debug_assert_eq!(left.len(), n);
    for row in 0..n {
        let dst = &mut out[row * n..(row + 1) * n];
        for slot in dst.iter_mut() {
            *slot = left[row];
        }
    }
}

/// Generic TM_PRED helper. `above[c] = A_c`, `left[r] = L_r`, `p` is
/// the single pixel above-and-left of the block (`P` in §12.2).
///
/// Dispatch: the §12.2 block sizes (16×16 luma, 8×8 chroma) take the
/// SIMD row kernel on nightly + `simd`, everything else (and every
/// stable build) the scalar listing. The SIMD path is byte-exact
/// against the scalar listing on every test fixture
/// (`predict_tm_simd_matches_scalar_on_stress_inputs`).
fn predict_tm(out: &mut [u8], n: usize, above: &[u8], left: &[u8], p: u8) {
    #[cfg(feature = "simd")]
    {
        match n {
            16 => return predict_tm_simd::<16>(out, above, left, p),
            8 => return predict_tm_simd::<8>(out, above, left, p),
            _ => {}
        }
    }
    predict_tm_scalar(out, n, above, left, p);
}

/// Scalar §12.2 TM_PRED — the spec formula written out longhand:
/// `X_{rc} = clamp255(L_r + A_c - P)` for every cell of the n×n block.
///
/// The public [`predict_tm`] dispatches here on stable builds (and on
/// nightly without the `simd` feature); the `simd` feature swaps in
/// [`predict_tm_simd`] for the §12.2 block sizes (16 and 8), which is
/// itself byte-exact against this implementation
/// (`predict_tm_simd_matches_scalar_on_stress_inputs`).
fn predict_tm_scalar(out: &mut [u8], n: usize, above: &[u8], left: &[u8], p: u8) {
    debug_assert_eq!(out.len(), n * n);
    debug_assert_eq!(above.len(), n);
    debug_assert_eq!(left.len(), n);
    let p = p as i32;
    for r in 0..n {
        for c in 0..n {
            // RFC 6386 §12.2: X_{ij} = clamp255(L_i + A_j - P).
            out[r * n + c] = clamp255(left[r] as i32 + above[c] as i32 - p);
        }
    }
}

/// SIMD §12.2 TM_PRED — `core::simd::Simd<i16, N>` row rewrite of
/// [`predict_tm_scalar`] for the two §12.2 block widths (N = 16 luma,
/// N = 8 chroma).
///
/// §12.2 computes `X_{rc} = clamp255(L_r + A_c - P)`. The column term
/// `A_c - P` is row-invariant, so it is formed once as an `i16` vector
/// (`above` widened lane-wise, minus a splat of `p`); each row then
/// adds a splat of `L_r`, clamps every lane into `0..=255`, and
/// narrows back to `u8`.
///
/// The `i16` working type reproduces the scalar `i32` arithmetic
/// exactly: every intermediate lies in `-255..=510` (`L_r + A_c - P`
/// with all three inputs in `0..=255`), well inside `i16`, so no lane
/// can wrap before the clamp; `simd_clamp(0, 255)` is the lane-wise
/// `clamp(0, 255)`; and the post-clamp `cast::<u8>()` truncation of a
/// value already in `0..=255` equals the scalar `as u8`. The
/// byte-exactness is enforced on every fixture by
/// `predict_tm_simd_matches_scalar_on_stress_inputs`.
#[cfg(feature = "simd")]
#[inline]
fn predict_tm_simd<const N: usize>(out: &mut [u8], above: &[u8], left: &[u8], p: u8) {
    use core::simd::cmp::SimdOrd;
    use core::simd::num::{SimdInt, SimdUint};
    use core::simd::Simd;

    debug_assert_eq!(out.len(), N * N);
    debug_assert_eq!(above.len(), N);
    debug_assert_eq!(left.len(), N);

    // Row-invariant column term: A_c - P, widened to i16 lanes.
    let above_v: Simd<i16, N> = Simd::<u8, N>::from_slice(above).cast::<i16>();
    let a_minus_p = above_v - Simd::<i16, N>::splat(p as i16);
    let zero = Simd::<i16, N>::splat(0);
    let max = Simd::<i16, N>::splat(255);

    for (r, row) in out.chunks_exact_mut(N).enumerate() {
        // RFC 6386 §12.2: X_{rc} = clamp255(L_r + A_c - P), one row
        // per vector: splat L_r, add the precomputed (A - P) lanes,
        // clamp into 0..=255, narrow back to u8.
        let v = (a_minus_p + Simd::<i16, N>::splat(left[r] as i16)).simd_clamp(zero, max);
        row.copy_from_slice(&v.cast::<u8>().to_array()[..]);
    }
}

/// Differential probe for the fuzz harness: run the §12.2 TM_PRED fill for
/// one of the two §12.2 block widths (`n` = 16 luma or 8 chroma) through
/// BOTH [`predict_tm_scalar`] and [`predict_tm_simd`] and return
/// `(scalar, simd)` as `n*n`-byte rasters so the caller can assert
/// byte-equality. `above` and `left` must each hold `n` pixels.
///
/// Only compiled under the `simd` feature (without it there is exactly one
/// implementation and nothing to compare). Behaviour-neutral: nothing in
/// the decode or encode pipeline calls this — it exists so the
/// `decode_stream_token_descent` fuzz target can turn any scalar/SIMD
/// divergence on attacker-shaped inputs into a finding.
///
/// # Panics
///
/// Panics if `n` is not 16 or 8 (the only widths the SIMD kernel serves).
#[cfg(feature = "simd")]
#[doc(hidden)]
pub fn predict_tm_parity_pair(n: usize, above: &[u8], left: &[u8], p: u8) -> (Vec<u8>, Vec<u8>) {
    let mut scalar = vec![0u8; n * n];
    predict_tm_scalar(&mut scalar, n, above, left, p);
    let mut simd = vec![0u8; n * n];
    match n {
        16 => predict_tm_simd::<16>(&mut simd, above, left, p),
        8 => predict_tm_simd::<8>(&mut simd, above, left, p),
        other => panic!("predict_tm_parity_pair: unsupported §12.2 width {other}"),
    }
    (scalar, simd)
}

/// Horizontal sum of an `n`-pixel DC neighbour edge (the §12.2 averaging
/// numerator before rounding). The two §12.2 block widths (16 luma, 8
/// chroma) take the SIMD reduction on nightly + `simd`; everything else
/// (and every stable build) the scalar accumulate. The SIMD path is
/// byte-exact against the scalar listing on every fixture
/// (`dc_edge_sum_simd_matches_scalar_on_stress_inputs`).
#[inline]
fn dc_edge_sum(edge: &[u8]) -> u32 {
    #[cfg(feature = "simd")]
    {
        match edge.len() {
            16 => return dc_edge_sum_simd::<16>(edge),
            8 => return dc_edge_sum_simd::<8>(edge),
            _ => {}
        }
    }
    dc_edge_sum_scalar(edge)
}

/// Scalar §12.2 DC edge sum — the averaging numerator written out
/// longhand: a widening accumulate of every byte in `edge`.
///
/// The public [`dc_edge_sum`] dispatches here on stable builds (and on
/// nightly without the `simd` feature); the `simd` feature swaps in
/// [`dc_edge_sum_simd`] for the §12.2 block widths (16 and 8), which is
/// itself byte-exact against this implementation
/// (`dc_edge_sum_simd_matches_scalar_on_stress_inputs`).
#[allow(dead_code)] // Used by `dc_edge_sum` only on the !simd path.
#[inline]
fn dc_edge_sum_scalar(edge: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    for &v in edge {
        sum += v as u32;
    }
    sum
}

/// SIMD §12.2 DC edge sum — `core::simd::Simd<u16, N>` reduction of
/// [`dc_edge_sum_scalar`] for the two §12.2 edge widths (N = 16 luma,
/// N = 8 chroma).
///
/// The averaging numerator is a pure horizontal sum with no cross-lane
/// dependency, so it maps directly onto one widening reduction: load the
/// `N` neighbour bytes, widen lane-wise to `u16`, and `reduce_sum`. A
/// `u16` lane cannot overflow — the largest single edge holds `N = 16`
/// bytes of value 255, summing to `4080`, well inside `u16::MAX`. Integer
/// addition is associative, so the tree reduction yields the identical
/// total as the scalar left-fold; the byte-exactness is enforced on every
/// fixture by `dc_edge_sum_simd_matches_scalar_on_stress_inputs`.
#[cfg(feature = "simd")]
#[inline]
fn dc_edge_sum_simd<const N: usize>(edge: &[u8]) -> u32 {
    use core::simd::num::SimdUint;
    use core::simd::Simd;

    debug_assert_eq!(edge.len(), N);
    let widened: Simd<u16, N> = Simd::<u8, N>::from_slice(edge).cast::<u16>();
    widened.reduce_sum() as u32
}

/// Differential probe for the fuzz harness: run the §12.2 DC edge sum for
/// one of the two §12.2 widths (`n` = 16 luma or 8 chroma) through BOTH
/// [`dc_edge_sum_scalar`] and [`dc_edge_sum_simd`] and return
/// `(scalar, simd)` so the caller can assert equality. `edge` must hold
/// `n` pixels.
///
/// Only compiled under the `simd` feature (without it there is exactly one
/// implementation and nothing to compare). Behaviour-neutral: nothing in
/// the decode or encode pipeline calls this — it exists so the
/// `decode_stream_token_descent` fuzz target can turn any scalar/SIMD
/// divergence on attacker-shaped inputs into a finding.
///
/// # Panics
///
/// Panics if `n` is not 16 or 8 (the only widths the SIMD kernel serves).
#[cfg(feature = "simd")]
#[doc(hidden)]
pub fn dc_edge_sum_parity_pair(n: usize, edge: &[u8]) -> (u32, u32) {
    let scalar = dc_edge_sum_scalar(edge);
    let simd = match n {
        16 => dc_edge_sum_simd::<16>(edge),
        8 => dc_edge_sum_simd::<8>(edge),
        other => panic!("dc_edge_sum_parity_pair: unsupported §12.2 width {other}"),
    };
    (scalar, simd)
}

/// Generic DC-prediction helper for the §12.2 / §12.3 rules.
///
/// * If both `above` and `left` are present, the DC value is the
///   rounded average of all `2n` pixels.
/// * If only one is present, it's the rounded average of those `n`
///   pixels.
/// * If neither is present, the block is filled with [`DEFAULT_TOPLEFT_DC`].
///
/// Rounding is per the spec's `(sum + (1 << (shf - 1))) >> shf`.
fn predict_dc_with_optional_edges(
    out: &mut [u8],
    n: usize,
    above: Option<&[u8]>,
    left: Option<&[u8]>,
) {
    debug_assert_eq!(out.len(), n * n);
    if let Some(a) = above {
        debug_assert_eq!(a.len(), n);
    }
    if let Some(l) = left {
        debug_assert_eq!(l.len(), n);
    }
    let dc = match (above, left) {
        (Some(a), Some(l)) => {
            // 2n summands, shf = log2(2n).
            let sum = dc_edge_sum(a) + dc_edge_sum(l);
            let shf = (2 * n).trailing_zeros();
            ((sum + (1 << (shf - 1))) >> shf) as u8
        }
        (Some(a), None) => {
            let sum = dc_edge_sum(a);
            let shf = n.trailing_zeros();
            ((sum + (1 << (shf - 1))) >> shf) as u8
        }
        (None, Some(l)) => {
            let sum = dc_edge_sum(l);
            let shf = n.trailing_zeros();
            ((sum + (1 << (shf - 1))) >> shf) as u8
        }
        (None, None) => DEFAULT_TOPLEFT_DC,
    };
    fill(out, n, dc);
}

/// 16×16 luma DC prediction (RFC 6386 §12.3 forwarding §12.2). `out`
/// is 256 pixels row-major.
pub fn predict_y16x16_dc(out: &mut [u8; 256], above: Option<&[u8; 16]>, left: Option<&[u8; 16]>) {
    predict_dc_with_optional_edges(
        out.as_mut_slice(),
        16,
        above.map(|a| a.as_slice()),
        left.map(|l| l.as_slice()),
    );
}

/// 16×16 luma vertical prediction (RFC 6386 §12.3 forwarding §12.2).
/// `above` must be supplied: if the macroblock is on the top row of
/// the frame the caller fills it with [`DEFAULT_ABOVE_PIXEL`].
pub fn predict_y16x16_v(out: &mut [u8; 256], above: &[u8; 16]) {
    predict_v(out.as_mut_slice(), 16, above);
}

/// 16×16 luma horizontal prediction (RFC 6386 §12.3 forwarding §12.2).
/// `left` must be supplied: if the macroblock is on the left column
/// the caller fills it with [`DEFAULT_LEFT_PIXEL`].
pub fn predict_y16x16_h(out: &mut [u8; 256], left: &[u8; 16]) {
    predict_h(out.as_mut_slice(), 16, left);
}

/// 16×16 luma TM_PRED (RFC 6386 §12.3 forwarding §12.2). Caller
/// supplies the standard out-of-bounds defaults when the macroblock
/// is on the top row or left column (this mode does use them per
/// §12.2 page 53).
pub fn predict_y16x16_tm(out: &mut [u8; 256], above: &[u8; 16], left: &[u8; 16], p: u8) {
    predict_tm(out.as_mut_slice(), 16, above, left, p);
}

/// Dispatch the 16×16 luma mode to the appropriate kernel. Each
/// kernel takes the inputs it actually uses; this dispatcher ignores
/// inputs that the selected mode doesn't read (e.g. `left` for
/// `V_PRED`).
///
/// Returns `None` if `mode == IntraYMode::B`, since `B_PRED`
/// macroblocks are predicted sub-block by sub-block via
/// [`predict_b4x4`] and this dispatcher has no opinion on that.
pub fn predict_y16x16(
    out: &mut [u8; 256],
    mode: IntraYMode,
    above: Option<&[u8; 16]>,
    left: Option<&[u8; 16]>,
    p: u8,
) -> Option<()> {
    match mode {
        IntraYMode::Dc => {
            predict_y16x16_dc(out, above, left);
            Some(())
        }
        IntraYMode::V => {
            // V_PRED uses out-of-bounds default 127 when `above` is
            // None.
            let buf = above.copied().unwrap_or([DEFAULT_ABOVE_PIXEL; 16]);
            predict_y16x16_v(out, &buf);
            Some(())
        }
        IntraYMode::H => {
            let buf = left.copied().unwrap_or([DEFAULT_LEFT_PIXEL; 16]);
            predict_y16x16_h(out, &buf);
            Some(())
        }
        IntraYMode::Tm => {
            let a = above.copied().unwrap_or([DEFAULT_ABOVE_PIXEL; 16]);
            let l = left.copied().unwrap_or([DEFAULT_LEFT_PIXEL; 16]);
            predict_y16x16_tm(out, &a, &l, p);
            Some(())
        }
        IntraYMode::B => None,
    }
}

/// 8×8 chroma DC prediction (RFC 6386 §12.2). `out` is 64 pixels
/// row-major.
pub fn predict_uv8x8_dc(out: &mut [u8; 64], above: Option<&[u8; 8]>, left: Option<&[u8; 8]>) {
    predict_dc_with_optional_edges(
        out.as_mut_slice(),
        8,
        above.map(|a| a.as_slice()),
        left.map(|l| l.as_slice()),
    );
}

/// 8×8 chroma vertical prediction (RFC 6386 §12.2).
pub fn predict_uv8x8_v(out: &mut [u8; 64], above: &[u8; 8]) {
    predict_v(out.as_mut_slice(), 8, above);
}

/// 8×8 chroma horizontal prediction (RFC 6386 §12.2).
pub fn predict_uv8x8_h(out: &mut [u8; 64], left: &[u8; 8]) {
    predict_h(out.as_mut_slice(), 8, left);
}

/// 8×8 chroma TM_PRED (RFC 6386 §12.2).
pub fn predict_uv8x8_tm(out: &mut [u8; 64], above: &[u8; 8], left: &[u8; 8], p: u8) {
    predict_tm(out.as_mut_slice(), 8, above, left, p);
}

/// Dispatch the 8×8 chroma mode to the appropriate kernel. Per §12.2,
/// `V_PRED` / `H_PRED` / `TM_PRED` use the [`DEFAULT_ABOVE_PIXEL`] /
/// [`DEFAULT_LEFT_PIXEL`] defaults when the corresponding edge is
/// off-frame; `DC_PRED` follows the §12.2 averaging rules with
/// `predict_dc_with_optional_edges`.
pub fn predict_uv8x8(
    out: &mut [u8; 64],
    mode: IntraUvMode,
    above: Option<&[u8; 8]>,
    left: Option<&[u8; 8]>,
    p: u8,
) {
    match mode {
        IntraUvMode::Dc => predict_uv8x8_dc(out, above, left),
        IntraUvMode::V => {
            let buf = above.copied().unwrap_or([DEFAULT_ABOVE_PIXEL; 8]);
            predict_uv8x8_v(out, &buf);
        }
        IntraUvMode::H => {
            let buf = left.copied().unwrap_or([DEFAULT_LEFT_PIXEL; 8]);
            predict_uv8x8_h(out, &buf);
        }
        IntraUvMode::Tm => {
            let a = above.copied().unwrap_or([DEFAULT_ABOVE_PIXEL; 8]);
            let l = left.copied().unwrap_or([DEFAULT_LEFT_PIXEL; 8]);
            predict_uv8x8_tm(out, &a, &l, p);
        }
    }
}

// ===========================================================================
// 4×4 sub-block modes (RFC 6386 §12.3 `subblock_intra_predict`)
// ===========================================================================

/// Predict a single 4×4 luma sub-block (RFC 6386 §12.3).
///
/// * `out` — destination block, row-major (16 pixels).
/// * `above` — eight pixels at `(-1, 0)..=(-1, 7)` of the sub-block's
///   own coordinate frame. The lower 4 pixels (`above[0..4]`) are
///   directly above the sub-block; the upper 4 (`above[4..8]`) are the
///   "extra" pixels that some diagonal modes reference, per the §12.3
///   right-edge-sub-block rules.
/// * `left` — four pixels at `(0, -1)..=(3, -1)` of the sub-block's
///   own coordinate frame.
/// * `p` — the single pixel at `(-1, -1)`, sitting just above-and-left
///   of the sub-block. Per §12.3 this is the value of `A[-1]` /
///   `L[-1]` in the spec's helper function.
pub fn predict_b4x4(out: &mut [u8; 16], mode: IntraBmode, above: &[u8; 8], left: &[u8; 4], p: u8) {
    // Build the `E[0..=8]` array described in §12.3:
    //   E[0] = L[3], E[1] = L[2], E[2] = L[1], E[3] = L[0],
    //   E[4] = P (== A[-1] == L[-1]),
    //   E[5] = A[0], E[6] = A[1], E[7] = A[2], E[8] = A[3].
    let e: [u8; 9] = [
        left[3], left[2], left[1], left[0], p, above[0], above[1], above[2], above[3],
    ];

    // Convenience accessor that matches the spec's array indexing.
    let b = |r: usize, c: usize| r * 4 + c;
    // Buffer to fill, then memcpy at the end. We use a local fixed
    // array so all writes are bounds-checked at compile time.
    let mut buf = [0u8; 16];

    match mode {
        IntraBmode::Dc => {
            // §12.3 B_DC_PRED: v = (sum(A[0..4]) + sum(L[0..4]) + 4) >> 3.
            // 8 summands, rounding by 4 = 1 << (3 - 1).
            let mut v: u32 = 4;
            for i in 0..4 {
                v += above[i] as u32 + left[i] as u32;
            }
            v >>= 3;
            for slot in buf.iter_mut() {
                *slot = v as u8;
            }
        }
        IntraBmode::Tm => {
            // §12.3 B_TM_PRED: same shape as the 16×16 TM at 4×4.
            for r in 0..4 {
                for c in 0..4 {
                    buf[b(r, c)] = clamp255(left[r] as i32 + above[c] as i32 - p as i32);
                }
            }
        }
        IntraBmode::Ve => {
            // §12.3 B_VE_PRED: each column c gets avg3p(A + c) i.e.
            // avg3(A[c-1], A[c], A[c+1]) for c = 0..3. A[-1] is P (== E[4]),
            // A[4] is above[4] (the right-extension).
            let extended: [u8; 6] = [p, above[0], above[1], above[2], above[3], above[4]];
            for c in 0..4 {
                let v = avg3(extended[c], extended[c + 1], extended[c + 2]);
                for r in 0..4 {
                    buf[b(r, c)] = v;
                }
            }
        }
        IntraBmode::He => {
            // §12.3 B_HE_PRED: each row r gets a smoothed value from
            // L. r=3 is special (uses avg3(L[2], L[3], L[3]));
            // r=0..=2 use avg3p(L + r) i.e. avg3(L[r-1], L[r], L[r+1]).
            // L[-1] is P (== E[4]).
            let extended: [u8; 5] = [p, left[0], left[1], left[2], left[3]];
            // r = 3: avg3(L[2], L[3], L[3]).
            let v3 = avg3(left[2], left[3], left[3]);
            for c in 0..4 {
                buf[b(3, c)] = v3;
            }
            for r in 0..3 {
                let v = avg3(extended[r], extended[r + 1], extended[r + 2]);
                for c in 0..4 {
                    buf[b(r, c)] = v;
                }
            }
        }
        IntraBmode::Ld => {
            // §12.3 B_LD_PRED: lines of slope -1, smoothed from A.
            // Note: this mode only references A (not E), so positions
            // are: avg3p(A + j) for j=1..=6, with B[3][3] using the
            // synthetic avg3(A[6], A[7], A[7]) (A[8] does not exist).
            buf[b(0, 0)] = avg3p(above, 1);
            buf[b(0, 1)] = avg3p(above, 2);
            buf[b(1, 0)] = avg3p(above, 2);
            buf[b(0, 2)] = avg3p(above, 3);
            buf[b(1, 1)] = avg3p(above, 3);
            buf[b(2, 0)] = avg3p(above, 3);
            buf[b(0, 3)] = avg3p(above, 4);
            buf[b(1, 2)] = avg3p(above, 4);
            buf[b(2, 1)] = avg3p(above, 4);
            buf[b(3, 0)] = avg3p(above, 4);
            buf[b(1, 3)] = avg3p(above, 5);
            buf[b(2, 2)] = avg3p(above, 5);
            buf[b(3, 1)] = avg3p(above, 5);
            buf[b(2, 3)] = avg3p(above, 6);
            buf[b(3, 2)] = avg3p(above, 6);
            buf[b(3, 3)] = avg3(above[6], above[7], above[7]);
        }
        IntraBmode::Rd => {
            // §12.3 B_RD_PRED: lines of slope +1, smoothed from E.
            buf[b(3, 0)] = avg3p(&e, 1);
            buf[b(3, 1)] = avg3p(&e, 2);
            buf[b(2, 0)] = avg3p(&e, 2);
            buf[b(3, 2)] = avg3p(&e, 3);
            buf[b(2, 1)] = avg3p(&e, 3);
            buf[b(1, 0)] = avg3p(&e, 3);
            buf[b(3, 3)] = avg3p(&e, 4);
            buf[b(2, 2)] = avg3p(&e, 4);
            buf[b(1, 1)] = avg3p(&e, 4);
            buf[b(0, 0)] = avg3p(&e, 4);
            buf[b(2, 3)] = avg3p(&e, 5);
            buf[b(1, 2)] = avg3p(&e, 5);
            buf[b(0, 1)] = avg3p(&e, 5);
            buf[b(1, 3)] = avg3p(&e, 6);
            buf[b(0, 2)] = avg3p(&e, 6);
            buf[b(0, 3)] = avg3p(&e, 7);
        }
        IntraBmode::Vr => {
            // §12.3 B_VR_PRED: SSE diagonals (slope +2), mixing
            // avg2p / avg3p over E.
            buf[b(3, 0)] = avg3p(&e, 2);
            buf[b(2, 0)] = avg3p(&e, 3);
            buf[b(3, 1)] = avg3p(&e, 4);
            buf[b(1, 0)] = avg3p(&e, 4);
            buf[b(2, 1)] = avg2p(&e, 4);
            buf[b(0, 0)] = avg2p(&e, 4);
            buf[b(3, 2)] = avg3p(&e, 5);
            buf[b(1, 1)] = avg3p(&e, 5);
            buf[b(2, 2)] = avg2p(&e, 5);
            buf[b(0, 1)] = avg2p(&e, 5);
            buf[b(3, 3)] = avg3p(&e, 6);
            buf[b(1, 2)] = avg3p(&e, 6);
            buf[b(2, 3)] = avg2p(&e, 6);
            buf[b(0, 2)] = avg2p(&e, 6);
            buf[b(1, 3)] = avg3p(&e, 7);
            buf[b(0, 3)] = avg2p(&e, 7);
        }
        IntraBmode::Vl => {
            // §12.3 B_VL_PRED: SSW diagonals (slope -2), reading from
            // A and the right-extension `above[4..=7]`. The spec's
            // listing references A[0..=6]; we pass `above` directly.
            buf[b(0, 0)] = avg2p(above, 0);
            buf[b(1, 0)] = avg3p(above, 1);
            buf[b(2, 0)] = avg2p(above, 1);
            buf[b(0, 1)] = avg2p(above, 1);
            buf[b(1, 1)] = avg3p(above, 2);
            buf[b(3, 0)] = avg3p(above, 2);
            buf[b(2, 1)] = avg2p(above, 2);
            buf[b(0, 2)] = avg2p(above, 2);
            buf[b(3, 1)] = avg3p(above, 3);
            buf[b(1, 2)] = avg3p(above, 3);
            buf[b(2, 2)] = avg2p(above, 3);
            buf[b(0, 3)] = avg2p(above, 3);
            buf[b(3, 2)] = avg3p(above, 4);
            buf[b(1, 3)] = avg3p(above, 4);
            // The last two values explicitly do not follow the
            // pattern per §12.3 page 58.
            buf[b(2, 3)] = avg3p(above, 5);
            buf[b(3, 3)] = avg3p(above, 6);
        }
        IntraBmode::Hd => {
            // §12.3 B_HD_PRED: ESE diagonals (slope +1/2). Note: the
            // spec listing on page 58 has a typo `svg2p(E + 1)` for
            // the second row-3 pixel — we treat it as `avg2p`, the
            // function used by all three sibling diagonal modes at
            // the analogous position.
            buf[b(3, 0)] = avg2p(&e, 0);
            buf[b(3, 1)] = avg3p(&e, 1);
            buf[b(2, 0)] = avg2p(&e, 1);
            buf[b(3, 2)] = avg2p(&e, 1);
            buf[b(2, 1)] = avg3p(&e, 2);
            buf[b(3, 3)] = avg3p(&e, 2);
            buf[b(2, 2)] = avg2p(&e, 2);
            buf[b(1, 0)] = avg2p(&e, 2);
            buf[b(2, 3)] = avg3p(&e, 3);
            buf[b(1, 1)] = avg3p(&e, 3);
            buf[b(1, 2)] = avg2p(&e, 3);
            buf[b(0, 0)] = avg2p(&e, 3);
            buf[b(1, 3)] = avg3p(&e, 4);
            buf[b(0, 1)] = avg3p(&e, 4);
            buf[b(0, 2)] = avg3p(&e, 5);
            buf[b(0, 3)] = avg3p(&e, 6);
        }
        IntraBmode::Hu => {
            // §12.3 B_HU_PRED: ENE diagonals (slope -1/2), reading
            // from L only. The spec listing assigns:
            //   B[0][0] = avg2p(L);            // (1/2, -1)
            //   B[0][1] = avg3p(L + 1);        // (1,   -1)
            //   B[0][2] = B[1][0] = avg2p(L + 1);   // (3/2, -1)
            //   B[0][3] = B[1][1] = avg3p(L + 2);   // (2,   -1)
            //   B[1][2] = B[2][0] = avg2p(L + 2);   // (5/2, -1)
            //   B[1][3] = B[2][1] = avg3(L[2], L[3], L[3]); // (3, -1)
            //   B[2][2] = B[2][3] = B[3][0]..[3][3] = L[3];
            buf[b(0, 0)] = avg2p(left, 0);
            buf[b(0, 1)] = avg3p(left, 1);
            buf[b(0, 2)] = avg2p(left, 1);
            buf[b(1, 0)] = avg2p(left, 1);
            buf[b(0, 3)] = avg3p(left, 2);
            buf[b(1, 1)] = avg3p(left, 2);
            // B[1][2] = B[2][0] = avg2p(L + 2):
            let mid = avg2p(left, 2);
            buf[b(1, 2)] = mid;
            buf[b(2, 0)] = mid;
            // B[1][3] = B[2][1] = avg3(L[2], L[3], L[3]):
            let v = avg3(left[2], left[3], left[3]);
            buf[b(1, 3)] = v;
            buf[b(2, 1)] = v;
            // Bottom-right corner + a few clones — all = L[3]:
            buf[b(2, 2)] = left[3];
            buf[b(2, 3)] = left[3];
            buf[b(3, 0)] = left[3];
            buf[b(3, 1)] = left[3];
            buf[b(3, 2)] = left[3];
            buf[b(3, 3)] = left[3];
        }
    }

    out.copy_from_slice(&buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // 16×16 luma kernels
    // ---------------------------------------------------------------

    #[test]
    fn y16x16_dc_full_neighbours_rounds_half_up() {
        // 16 above-pixels all 100, 16 left-pixels all 200. Sum = 4800,
        // shf = 5, rounding adjust = 16 → (4800 + 16) >> 5 = 150.
        let mut out = [0u8; 256];
        let above = [100u8; 16];
        let left = [200u8; 16];
        predict_y16x16_dc(&mut out, Some(&above), Some(&left));
        assert!(out.iter().all(|&v| v == 150));
    }

    #[test]
    fn y16x16_dc_top_row_uses_left_only() {
        // Only `left` (n=16, shf=4, round=8). 16 × 50 = 800; (800 +
        // 8) >> 4 = 50.
        let mut out = [0u8; 256];
        let left = [50u8; 16];
        predict_y16x16_dc(&mut out, None, Some(&left));
        assert!(out.iter().all(|&v| v == 50));
    }

    #[test]
    fn y16x16_dc_topleft_uses_default_128() {
        let mut out = [99u8; 256];
        predict_y16x16_dc(&mut out, None, None);
        assert!(out.iter().all(|&v| v == DEFAULT_TOPLEFT_DC));
        assert_eq!(out[0], 128);
    }

    #[test]
    fn y16x16_v_copies_above_row() {
        let mut out = [0u8; 256];
        let above: [u8; 16] = core::array::from_fn(|i| (i * 7) as u8);
        predict_y16x16_v(&mut out, &above);
        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(out[r * 16 + c], above[c]);
            }
        }
    }

    #[test]
    fn y16x16_h_copies_left_column() {
        let mut out = [0u8; 256];
        let left: [u8; 16] = core::array::from_fn(|i| (i * 11 + 5) as u8);
        predict_y16x16_h(&mut out, &left);
        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(out[r * 16 + c], left[r]);
            }
        }
    }

    #[test]
    fn y16x16_tm_matches_spec_formula() {
        // Pick A, L, P with at least one wrap-or-clamp case.
        let above: [u8; 16] = core::array::from_fn(|i| 10 + i as u8);
        let left: [u8; 16] = core::array::from_fn(|i| 200 - i as u8);
        let p = 50u8;
        let mut out = [0u8; 256];
        predict_y16x16_tm(&mut out, &above, &left, p);
        for r in 0..16 {
            for c in 0..16 {
                let want = (left[r] as i32 + above[c] as i32 - p as i32).clamp(0, 255) as u8;
                assert_eq!(out[r * 16 + c], want, "tm mismatch at ({r}, {c})");
            }
        }
    }

    #[test]
    fn y16x16_tm_clamp_floor_and_ceiling() {
        // L = 0, A = 0, P = 250 → -250 → clamped to 0.
        let above = [0u8; 16];
        let left = [0u8; 16];
        let mut out = [42u8; 256];
        predict_y16x16_tm(&mut out, &above, &left, 250);
        assert!(out.iter().all(|&v| v == 0));

        // L = 255, A = 255, P = 0 → 510 → clamped to 255.
        let above = [255u8; 16];
        let left = [255u8; 16];
        let mut out = [42u8; 256];
        predict_y16x16_tm(&mut out, &above, &left, 0);
        assert!(out.iter().all(|&v| v == 255));
    }

    #[test]
    fn y16x16_dispatch_routes_each_mode() {
        let above = [200u8; 16];
        let left = [100u8; 16];
        let p = 150u8;

        // DC: (16*200 + 16*100 + 16) >> 5 = (4800+16)>>5 = 150.
        let mut out = [0u8; 256];
        assert!(predict_y16x16(&mut out, IntraYMode::Dc, Some(&above), Some(&left), p).is_some());
        assert!(out.iter().all(|&v| v == 150));

        // V copies above.
        let mut out = [0u8; 256];
        predict_y16x16(&mut out, IntraYMode::V, Some(&above), Some(&left), p);
        assert!(out.iter().all(|&v| v == 200));

        // H copies left.
        let mut out = [0u8; 256];
        predict_y16x16(&mut out, IntraYMode::H, Some(&above), Some(&left), p);
        assert!(out.iter().all(|&v| v == 100));

        // TM: every cell = 100 + 200 - 150 = 150.
        let mut out = [0u8; 256];
        predict_y16x16(&mut out, IntraYMode::Tm, Some(&above), Some(&left), p);
        assert!(out.iter().all(|&v| v == 150));

        // B returns None — caller handles sub-blocks.
        let mut out = [0u8; 256];
        assert!(predict_y16x16(&mut out, IntraYMode::B, Some(&above), Some(&left), p).is_none());
    }

    #[test]
    fn y16x16_dispatch_v_uses_default_when_top_row() {
        let mut out = [0u8; 256];
        predict_y16x16(&mut out, IntraYMode::V, None, None, 0);
        assert!(out.iter().all(|&v| v == DEFAULT_ABOVE_PIXEL));
    }

    #[test]
    fn y16x16_dispatch_h_uses_default_when_left_column() {
        let mut out = [0u8; 256];
        predict_y16x16(&mut out, IntraYMode::H, None, None, 0);
        assert!(out.iter().all(|&v| v == DEFAULT_LEFT_PIXEL));
    }

    // ---------------------------------------------------------------
    // 8×8 chroma kernels
    // ---------------------------------------------------------------

    #[test]
    fn uv8x8_dc_full_neighbours_rounds_half_up() {
        // 8 above-pixels = 100, 8 left-pixels = 200. Sum = 2400,
        // shf = 4, round = 8 → (2400 + 8) >> 4 = 150.
        let mut out = [0u8; 64];
        let above = [100u8; 8];
        let left = [200u8; 8];
        predict_uv8x8_dc(&mut out, Some(&above), Some(&left));
        assert!(out.iter().all(|&v| v == 150));
    }

    #[test]
    fn uv8x8_dc_left_column_uses_above_only() {
        // n=8, shf=3, round=4. 8 × 100 = 800 → (800+4)>>3 = 100.
        let mut out = [0u8; 64];
        let above = [100u8; 8];
        predict_uv8x8_dc(&mut out, Some(&above), None);
        assert!(out.iter().all(|&v| v == 100));
    }

    #[test]
    fn uv8x8_dc_topleft_uses_128() {
        let mut out = [7u8; 64];
        predict_uv8x8_dc(&mut out, None, None);
        assert!(out.iter().all(|&v| v == 128));
    }

    #[test]
    fn uv8x8_dispatch_v_h_tm() {
        let above = [10u8; 8];
        let left = [20u8; 8];
        let p = 5u8;

        let mut out = [0u8; 64];
        predict_uv8x8(&mut out, IntraUvMode::V, Some(&above), Some(&left), p);
        assert!(out.iter().all(|&v| v == 10));

        let mut out = [0u8; 64];
        predict_uv8x8(&mut out, IntraUvMode::H, Some(&above), Some(&left), p);
        assert!(out.iter().all(|&v| v == 20));

        // TM: 20 + 10 - 5 = 25 for every cell since both rows /
        // columns are flat.
        let mut out = [0u8; 64];
        predict_uv8x8(&mut out, IntraUvMode::Tm, Some(&above), Some(&left), p);
        assert!(out.iter().all(|&v| v == 25));
    }

    // ---------------------------------------------------------------
    // 4×4 sub-block kernels
    // ---------------------------------------------------------------

    #[test]
    fn b4x4_dc_averages_eight_pixels() {
        // A[0..4] = {4,4,4,4}, L[0..4] = {12,12,12,12}; sum = 64.
        // (64 + 4) >> 3 = 8.
        let above = [4, 4, 4, 4, 0, 0, 0, 0];
        let left = [12, 12, 12, 12];
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Dc, &above, &left, 99);
        assert!(out.iter().all(|&v| v == 8));
    }

    #[test]
    fn b4x4_tm_matches_clamped_formula() {
        let above = [10, 20, 30, 40, 0, 0, 0, 0];
        let left = [5, 15, 25, 35];
        let p = 8u8;
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Tm, &above, &left, p);
        for r in 0..4 {
            for c in 0..4 {
                let want = (left[r] as i32 + above[c] as i32 - p as i32).clamp(0, 255) as u8;
                assert_eq!(out[r * 4 + c], want);
            }
        }
    }

    #[test]
    fn b4x4_ve_smooths_above_row() {
        // A row = [10, 20, 30, 40, 50, 60, 70, 80], P = 0.
        let above = [10, 20, 30, 40, 50, 60, 70, 80];
        let left = [0; 4];
        let p = 0u8;
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Ve, &above, &left, p);
        // For c=0: avg3(P, A[0], A[1]) = avg3(0, 10, 20) =
        //   (0 + 10 + 10 + 20 + 2) >> 2 = 42 >> 2 = 10.
        // For c=1: avg3(A[0], A[1], A[2]) = avg3(10, 20, 30) =
        //   (10 + 20 + 20 + 30 + 2) >> 2 = 82 >> 2 = 20.
        // For c=2: avg3(20, 30, 40) = (20 + 30 + 30 + 40 + 2) >> 2 = 30.
        // For c=3: avg3(30, 40, 50) = (30 + 40 + 40 + 50 + 2) >> 2 = 40.
        for r in 0..4 {
            assert_eq!(out[r * 4], 10);
            assert_eq!(out[r * 4 + 1], 20);
            assert_eq!(out[r * 4 + 2], 30);
            assert_eq!(out[r * 4 + 3], 40);
        }
    }

    #[test]
    fn b4x4_he_smooths_left_column_with_bottom_special_case() {
        // L = [10, 20, 30, 40], P = 0.
        let above = [0; 8];
        let left = [10, 20, 30, 40];
        let p = 0u8;
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::He, &above, &left, p);
        // r=0: avg3(P, L[0], L[1]) = avg3(0, 10, 20) = 10.
        // r=1: avg3(10, 20, 30) = 20.
        // r=2: avg3(20, 30, 40) = 30.
        // r=3: avg3(L[2], L[3], L[3]) = avg3(30, 40, 40) =
        //   (30 + 40 + 40 + 40 + 2) >> 2 = 152 >> 2 = 38.
        for c in 0..4 {
            assert_eq!(out[c], 10);
            assert_eq!(out[4 + c], 20);
            assert_eq!(out[8 + c], 30);
            assert_eq!(out[12 + c], 38);
        }
    }

    #[test]
    fn b4x4_ld_top_left_uses_a1_and_bottom_right_uses_a7_clone() {
        // A = [0, 100, 100, 100, 100, 100, 100, 100], all-100 from
        // index 1 onwards, so avg3p anywhere from i=2 onwards is 100.
        let above = [0, 100, 100, 100, 100, 100, 100, 100];
        let left = [0; 4];
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Ld, &above, &left, 0);
        // B[0][0] = avg3p(A + 1) = avg3(A[0], A[1], A[2]) =
        //   (0 + 100 + 100 + 100 + 2) >> 2 = 75.
        assert_eq!(out[0], 75);
        // B[3][3] = avg3(A[6], A[7], A[7]) = avg3(100, 100, 100) = 100.
        assert_eq!(out[15], 100);
    }

    #[test]
    fn b4x4_rd_anti_diagonal_propagates() {
        // Build E with a single non-zero element so we can trace.
        // E = [L[3], L[2], L[1], L[0], P, A[0], A[1], A[2], A[3]].
        // Set everything to 100 except E[4] = P = 4. Then:
        //   avg3p(E + 4) = avg3(E[3], E[4], E[5]) =
        //     avg3(100, 4, 100) = (100 + 4 + 4 + 100 + 2) >> 2 = 52.
        let above = [100, 100, 100, 100, 0, 0, 0, 0];
        let left = [100, 100, 100, 100];
        let p = 4u8;
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Rd, &above, &left, p);
        // The main diagonal (B[0][0], B[1][1], B[2][2], B[3][3]) all
        // sample avg3p(E + 4).
        assert_eq!(out[0], 52);
        assert_eq!(out[5], 52);
        assert_eq!(out[10], 52);
        assert_eq!(out[15], 52);
    }

    #[test]
    fn b4x4_vr_mixes_avg2_and_avg3() {
        // Use a sparse E to verify the formula. Set everything to
        // 100 except E[5] = 4 (A[0] = 4).
        let above = [4, 100, 100, 100, 100, 100, 100, 100];
        let left = [100, 100, 100, 100];
        let p = 100u8;
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Vr, &above, &left, p);
        // B[3][2] = B[1][1] = avg3p(E + 5) = avg3(E[4], E[5], E[6]) =
        //   avg3(100, 4, 100) = 52.
        assert_eq!(out[3 * 4 + 2], 52);
        assert_eq!(out[4 + 1], 52);
        // B[2][2] = B[0][1] = avg2p(E + 5) = avg2(E[5], E[6]) =
        //   avg2(4, 100) = (4 + 100 + 1) >> 1 = 52.
        assert_eq!(out[2 * 4 + 2], 52);
        assert_eq!(out[1], 52);
    }

    #[test]
    fn b4x4_vl_handles_right_extension() {
        // Reads above[0..=6]. Use a ramp.
        let above = [10, 20, 30, 40, 50, 60, 70, 80];
        let left = [0; 4];
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Vl, &above, &left, 99);
        // B[0][0] = avg2p(A) = avg2(A[0], A[1]) = avg2(10, 20) = 15.
        assert_eq!(out[0], 15);
        // B[3][3] = avg3p(A + 6) = avg3(A[5], A[6], A[7]) =
        //   avg3(60, 70, 80) = (60 + 70 + 70 + 80 + 2) >> 2 = 70.
        assert_eq!(out[15], 70);
        // B[2][3] = avg3p(A + 5) = avg3(A[4], A[5], A[6]) =
        //   avg3(50, 60, 70) = 60.
        assert_eq!(out[2 * 4 + 3], 60);
    }

    #[test]
    fn b4x4_hd_treats_svg2p_typo_as_avg2p() {
        // E = [L[3], L[2], L[1], L[0], P, A[0], A[1], A[2], A[3]].
        // We want to exercise the position the spec wrote `svg2p`:
        //   B[2][0] = B[3][2] = svg2p(E + 1) → avg2p(E + 1).
        // Set everything to 100 except E[1] = L[2] = 4. Then
        //   avg2p(E + 1) = avg2(E[1], E[2]) = avg2(4, 100) = 52.
        let above = [100, 100, 100, 100, 100, 100, 100, 100];
        let left = [100, 4, 100, 100];
        let p = 100u8;
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Hd, &above, &left, p);
        // Both positions should be 52, confirming we used avg2p
        // (avg3p(E + 1) = avg3(E[0], E[1], E[2]) = avg3(100, 4, 100)
        // = (100 + 4 + 4 + 100 + 2) >> 2 = 52 *also*, so we add a
        // disambiguating test below.)
        assert_eq!(out[2 * 4], 52);
        assert_eq!(out[3 * 4 + 2], 52);
    }

    #[test]
    fn b4x4_hd_avg2p_vs_avg3p_disambiguator() {
        // To prove we're using avg2p (not avg3p), make a case where
        // they differ. E[1] = 4, E[2] = 100, E[0] = 200 →
        //   avg2p(E + 1) = avg2(4, 100) = 52.
        //   avg3p(E + 1) = avg3(200, 4, 100) = (200 + 4 + 4 + 100 + 2) >> 2 = 77.
        // E = [L[3], L[2], L[1], L[0], P, ...]. So E[0] = L[3] = 200,
        // E[1] = L[2] = 4, E[2] = L[1] = 100.
        let above = [200, 200, 200, 200, 200, 200, 200, 200];
        let left = [200, 100, 4, 200];
        let p = 200u8;
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Hd, &above, &left, p);
        // svg2p / avg2p of (E[1], E[2]) = avg2(4, 100) = 52.
        // If we accidentally compiled it as avg3p we'd see 77.
        assert_eq!(out[2 * 4], 52, "B[2][0] should be avg2p(E+1) = 52");
        assert_eq!(out[3 * 4 + 2], 52, "B[3][2] should be avg2p(E+1) = 52");
    }

    #[test]
    fn b4x4_hu_bottom_rows_collapse_to_left_3() {
        // L[3] = 200, rest of L = 0.
        let above = [0; 8];
        let left = [0, 0, 0, 200];
        let p = 0u8;
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Hu, &above, &left, p);
        // Bottom row is L[3] = 200 everywhere.
        for c in 0..4 {
            assert_eq!(out[3 * 4 + c], 200);
        }
        // B[2][2] = B[2][3] = L[3] = 200.
        assert_eq!(out[2 * 4 + 2], 200);
        assert_eq!(out[2 * 4 + 3], 200);
    }

    #[test]
    fn b4x4_hu_top_row_uses_left() {
        // L = [10, 20, 30, 40].
        let above = [0; 8];
        let left = [10, 20, 30, 40];
        let p = 0u8;
        let mut out = [0u8; 16];
        predict_b4x4(&mut out, IntraBmode::Hu, &above, &left, p);
        // B[0][0] = avg2p(L) = avg2(L[0], L[1]) = avg2(10, 20) = 15.
        assert_eq!(out[0], 15);
        // B[0][1] = avg3p(L + 1) = avg3(L[0], L[1], L[2]) =
        //   avg3(10, 20, 30) = (10 + 20 + 20 + 30 + 2) >> 2 = 20.
        assert_eq!(out[1], 20);
        // B[0][2] = B[1][0] = avg2p(L + 1) = avg2(L[1], L[2]) =
        //   avg2(20, 30) = (20 + 30 + 1) >> 1 = 25.
        assert_eq!(out[2], 25);
        assert_eq!(out[4], 25);
    }

    // ---------------------------------------------------------------
    // Cross-cutting sanity checks
    // ---------------------------------------------------------------

    #[test]
    fn flat_above_left_yields_flat_dc_for_all_modes() {
        // If above == left == P == 100 then every kernel produces
        // a flat block of 100. (TM: 100 + 100 - 100 = 100. DC: 100.
        // V/H: 100. Smoothing diagonals: 100 everywhere because every
        // input pixel is 100 and both avg2/avg3 are weighted averages
        // with integer-rounding behaviour that is identity on a
        // constant input.)
        let above = [100u8; 8];
        let left = [100u8; 4];
        let p = 100u8;
        for mode in [
            IntraBmode::Dc,
            IntraBmode::Tm,
            IntraBmode::Ve,
            IntraBmode::He,
            IntraBmode::Ld,
            IntraBmode::Rd,
            IntraBmode::Vr,
            IntraBmode::Vl,
            IntraBmode::Hd,
            IntraBmode::Hu,
        ] {
            let mut out = [0u8; 16];
            predict_b4x4(&mut out, mode, &above, &left, p);
            for (i, &v) in out.iter().enumerate() {
                assert_eq!(v, 100, "mode {mode:?} pixel {i} != 100");
            }
        }
    }

    // ---------------------------------------------------------------
    // SIMD vs scalar byte-exact equivalence test (round-268 TM_PRED
    // SIMD). Mirrors the §14 `*_simd_matches_scalar_on_stress_inputs`
    // pairs. On stable / no `simd` feature this exercises the scalar
    // path twice (the public `predict_tm` dispatch and the directly-
    // named `_scalar` fn) — harmless; keeps CI green. On nightly +
    // `simd` it's the primary safety net: `predict_tm` dispatches to
    // the SIMD row kernel for n = 16 / n = 8 while
    // `predict_tm_scalar` stays reachable as the fallback, so the
    // assertion compares the two bit-for-bit, including the clamp
    // floor (L + A - P = -255) and ceiling (= 510) endpoints.
    // ---------------------------------------------------------------

    /// Deterministic byte generator for the stress fixtures (a small
    /// LCG; no external dep, reproducible across hosts).
    fn lcg_bytes(seed: &mut u32, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (*seed >> 24) as u8
            })
            .collect()
    }

    /// (above, left, p) stress fixtures for one block width `n`,
    /// covering the §12.2 clamp endpoints (floor: L = A = 0, P = 255
    /// → -255; ceiling: L = A = 255, P = 0 → 510), flat mid-range
    /// inputs, opposing ramps, alternating extremes, and
    /// deterministic pseudo-random triples.
    fn tm_stress_inputs(n: usize) -> Vec<(Vec<u8>, Vec<u8>, u8)> {
        let mut v: Vec<(Vec<u8>, Vec<u8>, u8)> = Vec::new();
        // Clamp floor: every cell hits 0 - the deepest underflow.
        v.push((vec![0u8; n], vec![0u8; n], 255));
        // Clamp ceiling: every cell hits 510 before the clamp.
        v.push((vec![255u8; n], vec![255u8; n], 0));
        // Flat mid-range identity (L + A - P = A).
        v.push((vec![100u8; n], vec![100u8; n], 100));
        // Opposing ramps straddle both clamp edges within one block.
        let ramp_up: Vec<u8> = (0..n).map(|i| (i * 255 / (n - 1)) as u8).collect();
        let ramp_down: Vec<u8> = ramp_up.iter().rev().copied().collect();
        for &p in &[0u8, 64, 128, 200, 255] {
            v.push((ramp_up.clone(), ramp_down.clone(), p));
        }
        // Alternating extremes exercise neighbouring-lane divergence.
        let alt: Vec<u8> = (0..n).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect();
        v.push((alt.clone(), ramp_up.clone(), 127));
        v.push((ramp_down.clone(), alt, 128));
        // Deterministic pseudo-random triples.
        let mut seed = 0x5EED_0000u32 ^ (n as u32);
        for _ in 0..16 {
            let above = lcg_bytes(&mut seed, n);
            let left = lcg_bytes(&mut seed, n);
            let p = lcg_bytes(&mut seed, 1)[0];
            v.push((above, left, p));
        }
        v
    }

    #[test]
    fn predict_tm_simd_matches_scalar_on_stress_inputs() {
        for n in [16usize, 8] {
            for (idx, (above, left, p)) in tm_stress_inputs(n).iter().enumerate() {
                // `predict_tm` is the dispatcher: SIMD on nightly +
                // `simd` (for n = 16 / 8), scalar otherwise. Compare
                // it against the directly-named scalar fallback on
                // the identical input.
                let mut via_dispatch = vec![0u8; n * n];
                predict_tm(&mut via_dispatch, n, above, left, *p);

                let mut via_scalar = vec![0u8; n * n];
                super::predict_tm_scalar(&mut via_scalar, n, above, left, *p);

                assert_eq!(
                    via_dispatch, via_scalar,
                    "predict_tm (dispatch) diverged from scalar on \
                     stress input #{idx} (n {n}, p {p})"
                );
            }
        }
    }

    #[test]
    fn predict_tm_public_entry_points_route_through_dispatcher() {
        // The two public §12.2 TM kernels must agree with the scalar
        // listing on a clamp-straddling input — proving the dispatch
        // rewrite didn't change the public surface's bytes.
        let above16: [u8; 16] = core::array::from_fn(|i| (i * 17) as u8);
        let left16: [u8; 16] = core::array::from_fn(|i| 255 - (i * 13) as u8);
        let p = 77u8;
        let mut via_public = [0u8; 256];
        predict_y16x16_tm(&mut via_public, &above16, &left16, p);
        let mut via_scalar = [0u8; 256];
        super::predict_tm_scalar(&mut via_scalar, 16, &above16, &left16, p);
        assert_eq!(via_public.as_slice(), via_scalar.as_slice());

        let above8: [u8; 8] = core::array::from_fn(|i| (i * 31) as u8);
        let left8: [u8; 8] = core::array::from_fn(|i| 255 - (i * 29) as u8);
        let mut via_public = [0u8; 64];
        predict_uv8x8_tm(&mut via_public, &above8, &left8, p);
        let mut via_scalar = [0u8; 64];
        super::predict_tm_scalar(&mut via_scalar, 8, &above8, &left8, p);
        assert_eq!(via_public.as_slice(), via_scalar.as_slice());
    }

    // ---------------------------------------------------------------
    // SIMD vs scalar byte-exact equivalence for the §12.2 DC edge sum.
    // Mirrors the §12.2 TM and §14 dequant
    // `*_simd_matches_scalar_on_stress_inputs` pairs. On stable / no
    // `simd` feature this exercises the scalar path twice (the public
    // `dc_edge_sum` dispatch and the directly-named `_scalar` fn) —
    // harmless; keeps CI green. On nightly + `simd` it's the primary
    // safety net: `dc_edge_sum` dispatches to the SIMD reduction for
    // the §12.2 widths (16 / 8) while `dc_edge_sum_scalar` stays
    // reachable as the fallback, so the assertion compares the two
    // bit-for-bit, including the all-255 saturating sum (16 × 255 =
    // 4080) that the rounding shift then divides.
    // ---------------------------------------------------------------

    /// `edge` stress fixtures for one §12.2 width `n` (16 luma / 8
    /// chroma): all-zero, all-255 (the reduction's maximum), the
    /// §12.2 off-frame defaults (127 above, 129 left), an ascending
    /// ramp, an alternating-extremes pattern, and deterministic
    /// pseudo-random rows reusing the §12.2 TM LCG.
    fn dc_edge_stress_inputs(n: usize) -> Vec<Vec<u8>> {
        let mut v: Vec<Vec<u8>> = Vec::new();
        v.push(vec![0u8; n]);
        v.push(vec![255u8; n]);
        v.push(vec![DEFAULT_ABOVE_PIXEL; n]);
        v.push(vec![DEFAULT_LEFT_PIXEL; n]);
        v.push((0..n).map(|i| (i * 255 / (n - 1)) as u8).collect());
        v.push((0..n).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect());
        let mut seed = 0xDC00_0000u32 ^ (n as u32);
        for _ in 0..16 {
            v.push(lcg_bytes(&mut seed, n));
        }
        v
    }

    #[test]
    fn dc_edge_sum_simd_matches_scalar_on_stress_inputs() {
        for n in [16usize, 8] {
            for (idx, edge) in dc_edge_stress_inputs(n).iter().enumerate() {
                // `dc_edge_sum` is the dispatcher: SIMD on nightly +
                // `simd` (for n = 16 / 8), scalar otherwise. Compare it
                // against the directly-named scalar fallback on the
                // identical edge.
                let via_dispatch = super::dc_edge_sum(edge);
                let via_scalar = super::dc_edge_sum_scalar(edge);
                assert_eq!(
                    via_dispatch, via_scalar,
                    "dc_edge_sum (dispatch) diverged from scalar on stress \
                     input #{idx} (n {n})"
                );
            }
        }
    }

    #[test]
    fn dc_edge_sum_drives_predict_dc_public_entry_points() {
        // The two §12.2 DC kernels must agree with an independent
        // scalar average on a non-flat edge — proving the SIMD sum
        // rewrite didn't change the public surface's bytes.
        let above16: [u8; 16] = core::array::from_fn(|i| (i * 17) as u8);
        let left16: [u8; 16] = core::array::from_fn(|i| 255 - (i * 13) as u8);
        let sum: u32 = above16.iter().chain(left16.iter()).map(|&b| b as u32).sum();
        let want = ((sum + 16) >> 5) as u8;
        let mut out = [0u8; 256];
        predict_y16x16_dc(&mut out, Some(&above16), Some(&left16));
        assert!(out.iter().all(|&p| p == want));

        let above8: [u8; 8] = core::array::from_fn(|i| (i * 31) as u8);
        let left8: [u8; 8] = core::array::from_fn(|i| 255 - (i * 29) as u8);
        let sum: u32 = above8.iter().chain(left8.iter()).map(|&b| b as u32).sum();
        let want = ((sum + 8) >> 4) as u8;
        let mut out = [0u8; 64];
        predict_uv8x8_dc(&mut out, Some(&above8), Some(&left8));
        assert!(out.iter().all(|&p| p == want));
    }
}
