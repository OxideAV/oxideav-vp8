//! Interframe motion compensation — RFC 6386 §16.2 reference selection,
//! §18.1 motion-vector adjustment, §18.2 whole-pixel prediction, and
//! §18.3 sub-pixel (sixtap / bilinear) interpolation.
//!
//! This module is the first inter-frame *prediction* slice: it consumes
//! the §17 motion vectors decoded by [`crate::motion_vector`] and turns a
//! reference frame plus a per-macroblock motion vector into a predicted
//! macroblock, then folds in the §14 dequantized residual to complete
//! inter-MB reconstruction. It sits one layer below the §16.3
//! near/nearest/best census (which produces the actual per-MB vector a
//! NEWMV offset is added to) and the §16.4 SPLITMV walk; those choose
//! *which* vector applies, this applies it.
//!
//! ## Scope — whole-pixel + sub-pixel motion compensation
//!
//! VP8 motion vectors carry a fractional (sub-pixel) part. After the
//! §18.1 adjustments, each component's low three bits (`& 7`) are an
//! eighth-pixel forward displacement:
//!
//! * **Whole-pixel** (`mx | my == 0`): §18.3 page 115 "the prediction
//!   subblock is simply copied". This is the `filter_block` special case
//!   in the §20.14 reference decoder, where `filter_block` returns the
//!   reference pointer unchanged.
//! * **Sub-pixel** (either fraction non-zero): §18.3 synthesises the
//!   missing samples via a horizontal then a vertical one-dimensional
//!   six-tap convolution ([`sixtap_2d`]). The tap set is chosen by the
//!   frame-tag version — `version == 0` uses the bicubic
//!   [`SIXTAP_FILTERS`], any other version uses the [`BILINEAR_FILTERS`]
//!   (§20.14 `subpixel_filters` assignment). Both luma and chroma share
//!   the version-selected set.
//!
//! Concretely this module provides:
//!
//! * [`RefFrame`] + [`select_ref_frame`] — the §16.2 `prob_last` /
//!   `prob_gf` reference-frame selector.
//! * [`ReferencePlanes`] — a borrow of one reference frame's I420 planes
//!   for prediction fetch.
//! * [`stored_luma_mv`] / [`chroma_mv`] / [`apply_full_pixel`] — the
//!   §18.1 motion-vector adjustments (stored-luma doubling, the chroma
//!   `avg()` averaging, and the version-3 full-pel truncation).
//! * [`whole_pixel_fraction_is_zero`] — the §18.3 whole-pixel test
//!   applied to an already-§18.1-adjusted (eighth-pixel) vector.
//! * [`SIXTAP_FILTERS`] / [`BILINEAR_FILTERS`] / [`FilterSet`] /
//!   [`filter_set_for_version`] — the §18.3 filter tables and the §20.14
//!   version→tap-set selector.
//! * [`interp`] / [`sixtap_horiz`] / [`sixtap_vert`] / [`sixtap_2d`] —
//!   the §18.3 / §20.14 one-dimensional convolution primitives.
//! * [`fetch_block_whole_pixel`] — the §20.14 `build_mc_border`
//!   edge-replicated 4×4 block fetch, specialised to whole-pixel offsets
//!   (no six-tap support pixels needed).
//! * [`fetch_block_halo`] — the §20.14 `build_mc_border` 9×9 halo fetch
//!   (the 4×4 block plus the two-before / three-after support pixels the
//!   six-tap convolution needs in each dimension).
//! * [`filter_block_4x4`] — the §20.14 `filter_block`: whole-pixel copy
//!   or [`sixtap_2d`] sub-pixel synthesis for one 4×4 sub-block.
//! * [`predict_inter_mb`] — the §18.2 whole-MB prediction buffer for a
//!   non-SPLITMV macroblock (one vector for all sixteen Y sub-blocks, the
//!   averaged chroma vector for the eight chroma sub-blocks), routing each
//!   sub-block through whole-pixel copy or sub-pixel interpolation.
//! * [`reconstruct_inter_mb`] — prediction + §14 dequantized residual,
//!   producing a [`ReconstructedMb`].
//!
//! The whole-pixel-only [`predict_inter_mb_whole_pixel`] /
//! [`reconstruct_inter_mb_whole_pixel`] entry points are retained for
//! callers that only ever pass whole-pixel vectors; they refuse a
//! sub-pixel vector with [`MotionCompError::SubPixelNotSupported`]. New
//! callers should prefer [`predict_inter_mb`] / [`reconstruct_inter_mb`].
//!
//! ## What is deferred (next inter-prediction rounds)
//!
//! * §16.4 SPLITMV (per-sub-block vectors): this round handles the four
//!   whole-MB inter modes (`mv_nearest` / `mv_near` / `mv_zero` /
//!   `mv_new`), all of which apply one vector to the whole MB.
//! * §16.3 `vp8_find_near_mvs` census: this module takes the *resolved*
//!   per-MB vector as an input; deriving it from neighbours is a separate
//!   slice.

use crate::bool_decoder::{BoolDecoder, BoolDecoderError};
use crate::inverse_transform::{add_residue_4x4, clamp255, inverse_dct_4x4, inverse_wht_4x4};
use crate::motion_vector::Mv;
use crate::reconstruct::ReconstructedMb;

/// The reference frame a macroblock predicts from — RFC 6386 §16.2.
///
/// On an interframe, after the intra/inter discriminator selects
/// inter-prediction, a `Bool(prob_last)` then (conditionally) a
/// `Bool(prob_gf)` choose between the three stored reference frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefFrame {
    /// The previous (last) decoded frame — `prob_last` reads 0.
    Last,
    /// The golden frame — `prob_last` reads 1, `prob_gf` reads 0.
    Golden,
    /// The altref frame — `prob_last` reads 1, `prob_gf` reads 1.
    AltRef,
}

/// Select the reference frame for an inter-predicted macroblock — RFC
/// 6386 §16.2.
///
/// "The next datum is then another bool, `B(prob_last)`, selecting the
/// reference frame. If 0, the reference frame is the previous frame (the
/// last frame); if 1, another bool (`prob_gf`) selects the reference
/// frame between the golden frame (0) and the altref frame (1)."
///
/// `prob_last` and `prob_gf` come from field J of the frame header (the
/// §9.10 `prob_last` / `prob_gf` already parsed by [`crate::coded_header`]).
pub fn select_ref_frame(
    dec: &mut BoolDecoder<'_>,
    prob_last: u8,
    prob_gf: u8,
) -> Result<RefFrame, BoolDecoderError> {
    if !dec.read_bool(prob_last)? {
        Ok(RefFrame::Last)
    } else if !dec.read_bool(prob_gf)? {
        Ok(RefFrame::Golden)
    } else {
        Ok(RefFrame::AltRef)
    }
}

/// A borrow of one reference frame's reconstructed I420 planes, in the
/// whole-macroblock layout produced by [`crate::frame::KeyframePlanes`].
///
/// The planes are sized to whole macroblocks: the luma plane is
/// `mb_cols * 16` wide by `mb_rows * 16` tall, each chroma plane half
/// that in both dimensions. The §18.2 prediction fetch reads from these
/// planes with the §20.14 `build_mc_border` edge replication when a
/// motion vector points outside the plane.
#[derive(Debug, Clone, Copy)]
pub struct ReferencePlanes<'a> {
    /// Luma plane, row-major, `y_stride * (mb_rows * 16)` bytes.
    pub y: &'a [u8],
    /// U chroma plane, row-major, `uv_stride * (mb_rows * 8)` bytes.
    pub u: &'a [u8],
    /// V chroma plane, same dimensions as [`ReferencePlanes::u`].
    pub v: &'a [u8],
    /// Luma plane stride (= `mb_cols * 16`).
    pub y_stride: usize,
    /// Chroma plane stride (= `mb_cols * 8`).
    pub uv_stride: usize,
    /// Number of macroblock columns.
    pub mb_cols: usize,
    /// Number of macroblock rows.
    pub mb_rows: usize,
}

/// Errors surfaced by the motion-compensation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionCompError {
    /// The §18.1-adjusted motion vector has a non-zero fractional part,
    /// so §18.3 sub-pixel interpolation is required. Returned only by the
    /// whole-pixel-only entry points
    /// ([`predict_inter_mb_whole_pixel`] /
    /// [`reconstruct_inter_mb_whole_pixel`]); the full
    /// [`predict_inter_mb`] / [`reconstruct_inter_mb`] path handles
    /// sub-pixel vectors directly and never returns this.
    SubPixelNotSupported,
}

impl core::fmt::Display for MotionCompError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MotionCompError::SubPixelNotSupported => f.write_str(
                "vp8 motion-comp: sub-pixel interpolation (§18.3) not yet implemented; \
                 only whole-pixel motion vectors are supported this round",
            ),
        }
    }
}

impl std::error::Error for MotionCompError {}

/// Apply the §18.1 stored-luma doubling to a decoded luma motion vector.
///
/// RFC 6386 §18.1: "the synthetic pixel calculation ... uses [1/8-pixel]
/// resolution for the luma subblocks as well. In accordance, the stored
/// luma motion vectors are all doubled, each component of each luma
/// vector becoming an even integer in the range -2046 to +2046,
/// inclusive."
///
/// The §17 decode produces quarter-pixel luma components; doubling moves
/// them to the eighth-pixel resolution chroma uses, so a single
/// fractional-extraction rule (`& 7`) serves both planes.
#[inline]
pub fn stored_luma_mv(mv: Mv) -> Mv {
    Mv {
        row: mv.row.wrapping_mul(2),
        col: mv.col.wrapping_mul(2),
    }
}

/// The §18.1 `avg()` four-vector averaging primitive (one component).
///
/// RFC 6386 §18.1:
///
/// ```text
/// int avg(int c1, int c2, int c3, int c4) {
///     int s = c1 + c2 + c3 + c4;
///     return s >= 0 ? (s + 4) >> 3 : -((-s + 4) >> 3);
/// }
/// ```
///
/// The shift divides by 8 (not 4) because chroma pixels have twice the
/// diameter of luma pixels; the negative-number handling is explicit
/// because C right shifts of negatives are not well-defined.
#[inline]
fn avg(c1: i32, c2: i32, c3: i32, c4: i32) -> i32 {
    let s = c1 + c2 + c3 + c4;
    if s >= 0 {
        (s + 4) >> 3
    } else {
        -((-s + 4) >> 3)
    }
}

/// Compute the chroma motion vector for a whole (non-SPLITMV)
/// macroblock — RFC 6386 §18.1.
///
/// For a whole-MB vector all sixteen luma sub-blocks share `luma_mv`
/// (already §18.1-doubled to eighth-pixel resolution), so each chroma
/// vector is `avg(v, v, v, v)` of that one vector. The §20.14 reference
/// decoder writes this as the closed form
/// `(c + 1 + (c >> 31) * 2) / 2` per component, which equals
/// `avg(c, c, c, c)`; we use [`avg`] directly so the single §18.1
/// primitive backs both the whole-MB and (future) SPLITMV chroma paths.
///
/// `luma_mv` must already be the §18.1 stored (doubled) luma vector —
/// see [`stored_luma_mv`].
#[inline]
pub fn chroma_mv(luma_mv: Mv) -> Mv {
    let row = luma_mv.row as i32;
    let col = luma_mv.col as i32;
    Mv {
        row: avg(row, row, row, row) as i16,
        col: avg(col, col, col, col) as i16,
    }
}

/// Truncate the fractional part of a vector for the version-3 full-pel
/// chroma profile — RFC 6386 §18.1.
///
/// "if the version number in the frame tag specifies only full-pel
/// chroma motion vectors, then the fractional parts of both components
/// of the vector are truncated to zero":
///
/// ```text
/// x = x & (~7);
/// y = y & (~7);
/// ```
///
/// Applied to an eighth-pixel-resolution vector (the §18.1 stored luma
/// vector or the [`chroma_mv`] average).
#[inline]
pub fn apply_full_pixel(mv: Mv) -> Mv {
    Mv {
        row: mv.row & !7,
        col: mv.col & !7,
    }
}

/// Test whether an already-§18.1-adjusted (eighth-pixel) vector has a
/// zero fractional part — i.e. it is a whole-pixel vector and §18.3 page
/// 115 "the prediction subblock is simply copied" applies.
///
/// This mirrors the §20.14 `filter_block` test `mx = mv.x & 7;
/// my = mv.y & 7; if (mx | my) interpolate; else copy`.
#[inline]
pub fn whole_pixel_fraction_is_zero(mv: Mv) -> bool {
    (mv.row & 7) == 0 && (mv.col & 7) == 0
}

/// The §18.3 bicubic ("six-tap") interpolation filter table, indexed by
/// the eighth-pixel fractional displacement (0..=7).
///
/// RFC 6386 §18.3 `filters[8][6]` (= the §20.14 `sixtap_filters`):
///
/// ```text
/// const int filters [8] [6] = {        /* indexed by displacement */
///     { 0,  0,  128,    0,   0,  0 },  /* degenerate whole-pixel */
///     { 0, -6,  123,   12,  -1,  0 },  /* 1/8 */
///     { 2, -11, 108,   36,  -8,  1 },  /* 1/4 */
///     { 0, -9,   93,   50,  -6,  0 },  /* 3/8 */
///     { 3, -16,  77,   77, -16,  3 },  /* 1/2 is symmetric */
///     { 0, -6,   50,   93,  -9,  0 },  /* 5/8 = reverse of 3/8 */
///     { 1, -8,   36,  108, -11,  2 },  /* 3/4 = reverse of 1/4 */
///     { 0, -1,   12,  123,  -6,  0 }   /* 7/8 = reverse of 1/8 */
/// };
/// ```
///
/// "Filter taps taken to 7-bit precision. Because DC is always passed,
/// taps always sum to 128."
pub static SIXTAP_FILTERS: [[i32; 6]; 8] = [
    [0, 0, 128, 0, 0, 0],
    [0, -6, 123, 12, -1, 0],
    [2, -11, 108, 36, -8, 1],
    [0, -9, 93, 50, -6, 0],
    [3, -16, 77, 77, -16, 3],
    [0, -6, 50, 93, -9, 0],
    [1, -8, 36, 108, -11, 2],
    [0, -1, 12, 123, -6, 0],
];

/// The §18.3 bilinear interpolation filter table, indexed by the
/// eighth-pixel fractional displacement (0..=7).
///
/// RFC 6386 §18.3 `BilinearFilters[8][6]` (= the §20.14
/// `bilinear_filters`):
///
/// ```text
/// const int BilinearFilters[8][6] =
/// {
///     { 0, 0, 128,   0, 0, 0 },
///     { 0, 0, 112,  16, 0, 0 },
///     { 0, 0,  96,  32, 0, 0 },
///     { 0, 0,  80,  48, 0, 0 },
///     { 0, 0,  64,  64, 0, 0 },
///     { 0, 0,  48,  80, 0, 0 },
///     { 0, 0,  32,  96, 0, 0 },
///     { 0, 0,  16, 112, 0, 0 }
/// };
/// ```
///
/// Only the centre two taps are non-zero, so the bilinear filter never
/// reaches the outer support pixels; the convolution machinery is shared
/// with the six-tap path regardless.
pub static BILINEAR_FILTERS: [[i32; 6]; 8] = [
    [0, 0, 128, 0, 0, 0],
    [0, 0, 112, 16, 0, 0],
    [0, 0, 96, 32, 0, 0],
    [0, 0, 80, 48, 0, 0],
    [0, 0, 64, 64, 0, 0],
    [0, 0, 48, 80, 0, 0],
    [0, 0, 32, 96, 0, 0],
    [0, 0, 16, 112, 0, 0],
];

/// Which §18.3 tap set a frame uses, selected by the frame-tag version.
///
/// RFC 6386 §20.14 `setup_subpixel_filters`:
/// `if (version) subpixel_filters = bilinear_filters; else
/// subpixel_filters = sixtap_filters;`. Both luma and chroma share the
/// frame's one set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterSet {
    /// The bicubic six-tap set ([`SIXTAP_FILTERS`]) — frame-tag
    /// `version == 0`.
    Sixtap,
    /// The bilinear set ([`BILINEAR_FILTERS`]) — frame-tag `version != 0`.
    Bilinear,
}

impl FilterSet {
    /// The 8×6 tap table backing this filter set.
    #[inline]
    pub fn taps(self) -> &'static [[i32; 6]; 8] {
        match self {
            FilterSet::Sixtap => &SIXTAP_FILTERS,
            FilterSet::Bilinear => &BILINEAR_FILTERS,
        }
    }
}

/// Select the §18.3 interpolation filter set for a frame-tag version —
/// RFC 6386 §20.14.
///
/// `version == 0` → bicubic six-tap; any other version → bilinear.
#[inline]
pub fn filter_set_for_version(version: u8) -> FilterSet {
    if version == 0 {
        FilterSet::Sixtap
    } else {
        FilterSet::Bilinear
    }
}

/// One-dimensional synthesis of a single interpolated sample — RFC 6386
/// §18.3 `interp`.
///
/// ```text
/// Pixel interp(const int fil[6], const Pixel *p, const int s) {
///     int32 a = 0; int i = 0;
///     p -= s + s;                 /* move back two positions */
///     do { a += *p * fil[i]; p += s; } while (++i < 6);
///     return clamp255((a + 64) >> 7);
/// }
/// ```
///
/// `support` holds the six contributing source samples in increasing
/// position order — i.e. `[p[-2s], p[-1s], p[0], p[+1s], p[+2s],
/// p[+3s]]` — so the caller folds the stride `s` into the gather and
/// this primitive is a pure six-tap dot product with `(a + 64) >> 7`
/// rounding and a final `clamp255`.
#[inline]
pub fn interp(filter: &[i32; 6], support: &[u8; 6]) -> u8 {
    let mut a = 0i32;
    for i in 0..6 {
        a += support[i] as i32 * filter[i];
    }
    clamp255((a + 64) >> 7)
}

/// Horizontal six-tap convolution — RFC 6386 §20.14 `sixtap_horiz`.
///
/// Reads `rows` rows of `cols` output samples from `reference` (a plane
/// of stride `ref_stride`), with the convolution origin at each output
/// column; each output sample uses the six horizontal neighbours
/// `reference[c-2 ..= c+3]`. Writes into `output` at stride `out_stride`.
/// The intermediate is an 8-bit value: `clamp255((sum + 64) >> 7)`, so
/// negative partial sums are clamped to 0 exactly as in the reference
/// decoder (the intermediate buffer there is `unsigned char`).
///
/// `ref0` is the index in `reference` of output column 0, row 0 (the
/// convolution origin); the function reads `reference[ref0 - 2 ..]`.
#[allow(clippy::too_many_arguments)] // mirrors the §20.14 sixtap_horiz signature.
fn sixtap_horiz(
    output: &mut [u8],
    out_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    ref0: usize,
    cols: usize,
    rows: usize,
    filter: &[i32; 6],
) {
    for r in 0..rows {
        let row_base = ref0 + r * ref_stride;
        for c in 0..cols {
            let mut support = [0u8; 6];
            for (k, s) in support.iter_mut().enumerate() {
                // taps span reference[c-2 ..= c+3].
                *s = reference[row_base + c + k - 2];
            }
            output[r * out_stride + c] = interp(filter, &support);
        }
    }
}

/// Vertical six-tap convolution — RFC 6386 §20.14 `sixtap_vert`.
///
/// The vertical counterpart of [`sixtap_horiz`]: each output sample uses
/// the six vertical neighbours `reference[(r-2 ..= r+3) * ref_stride]`.
/// Operates on the horizontally-filtered intermediate produced by
/// [`sixtap_horiz`].
#[allow(clippy::too_many_arguments)] // mirrors the §20.14 sixtap_vert signature.
fn sixtap_vert(
    output: &mut [u8],
    out_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    ref0: usize,
    cols: usize,
    rows: usize,
    filter: &[i32; 6],
) {
    for r in 0..rows {
        for c in 0..cols {
            let col_base = ref0 + r * ref_stride + c;
            let mut support = [0u8; 6];
            for (k, s) in support.iter_mut().enumerate() {
                // taps span reference[(r-2 ..= r+3) * ref_stride].
                *s = reference[col_base + (k * ref_stride) - 2 * ref_stride];
            }
            output[r * out_stride + c] = interp(filter, &support);
        }
    }
}

/// Two-dimensional six-tap interpolation of a 4×4 sub-block — RFC 6386
/// §20.14 `sixtap_2d`.
///
/// ```text
/// sixtap_horiz(temp, 16, reference - 2*stride, stride, cols, rows + 5,
///              filters[mx]);
/// sixtap_vert(output, output_stride, temp + 2*16, 16, cols, rows,
///             filters[my]);
/// ```
///
/// `halo` is the 9×9 edge-replicated source region from
/// [`fetch_block_halo`]: row-major, stride 9, with the 4×4 block origin
/// at `halo[(2,2)]`. The horizontal pass synthesises a 4-wide × 9-tall
/// intermediate (the block rows plus the two-above / three-below support
/// the vertical pass needs); the vertical pass then synthesises the final
/// 4×4. `(mx, my)` are the eighth-pixel fractions (`mv & 7`); `filters`
/// is the version-selected tap set.
///
/// Dispatch: SIMD path on nightly + `simd`, scalar otherwise. The SIMD
/// path is byte-exact against the scalar listing on every test fixture
/// (`sixtap_2d_simd_matches_scalar_on_stress_inputs`).
pub fn sixtap_2d(halo: &[u8; 81], mx: usize, my: usize, filters: &[[i32; 6]; 8]) -> [u8; 16] {
    #[cfg(feature = "simd")]
    {
        sixtap_2d_simd(halo, mx, my, filters)
    }
    #[cfg(not(feature = "simd"))]
    {
        sixtap_2d_scalar(halo, mx, my, filters)
    }
}

/// Scalar §20.14 `sixtap_2d` — the two-pass [`sixtap_horiz`] /
/// [`sixtap_vert`] composition written as the spec listing.
///
/// The public [`sixtap_2d`] dispatches here on stable builds (and on
/// nightly without the `simd` feature); the `simd` feature swaps in
/// [`sixtap_2d_simd`], which is itself byte-exact against this
/// implementation (`sixtap_2d_simd_matches_scalar_on_stress_inputs`).
#[allow(dead_code)] // Used by `sixtap_2d` only on the !simd path.
fn sixtap_2d_scalar(halo: &[u8; 81], mx: usize, my: usize, filters: &[[i32; 6]; 8]) -> [u8; 16] {
    // Horizontal pass: 9 rows (the 4 block rows + 2 above + 3 below) ×
    // 4 cols. Intermediate is 8-bit (clamped), stride 4.
    //
    // halo stride is 9; the block origin is at (2, 2). sixtap_horiz reads
    // from "reference - 2*stride" with cols=4, rows=9 — i.e. starting at
    // halo row 0, and each output column c reads halo[col-2 ..= col+3]
    // relative to block-origin column 2. So ref0 = 2 (block-origin column
    // within the halo), and the first row read is halo row 0 (the function
    // adds no vertical offset; rows iterate 0..9 from ref0's row).
    let mut temp = [0u8; 9 * 4];
    sixtap_horiz(
        &mut temp,
        4,
        halo,
        9,
        /* ref0 = */ 2, // block-origin column; row 0 of the 9-row span
        4,
        9,
        &filters[mx],
    );

    // Vertical pass: 4 rows × 4 cols, reading the intermediate at "temp +
    // 2*stride" — i.e. block-origin row 2 within the 9-row intermediate.
    let mut out = [0u8; 16];
    sixtap_vert(
        &mut out,
        4,
        &temp,
        4,
        /* ref0 = */ 2 * 4, // block-origin row 2, column 0
        4,
        4,
        &filters[my],
    );
    out
}

/// SIMD §18.3 / §20.14 `sixtap_2d` — `core::simd::Simd<i32, 4>` row
/// rewrite of [`sixtap_2d_scalar`].
///
/// Both §20.14 passes produce rows of exactly four output samples (the
/// 4×4 sub-block geometry), and within a row the four §18.3 `interp`
/// dot products are independent — only the support window slides by one
/// sample per output column. Each output row is therefore computed as
/// one four-lane vector: for tap `k` the four lanes' support samples
/// are a contiguous source run (`halo[r*9 + k ..][..4]`), so the six
/// taps become six widen-multiply-accumulates of four lanes each in
/// place of 24 scalar multiply-accumulates per row. The horizontal
/// pass's clamped intermediate stays resident in `i32` vectors — every
/// lane is already in `0..=255` after the lane-wise clamp, so the
/// vertical pass reads the exact same sample values the scalar
/// listing's 8-bit `temp` buffer would hold, without a narrow-to-u8 /
/// widen-from-u8 round trip.
///
/// Lane type: the accumulator must be `i32`, not `i16`. The §18.3
/// six-tap dot product `a = Σ fil[i] * p[i]` over `u8` support spans
/// `[-32·255, 160·255] = [-8160, 40800]` (the ½-displacement row
/// `{3, -16, 77, 77, -16, 3}` has positive-tap sum 160 and
/// negative-tap sum −32), and `a + 64` reaches 40864 — past
/// `i16::MAX`. `i32` lanes reproduce the scalar `i32` arithmetic
/// exactly: the `(a + 64) >> 7` lane shift is the same sign-propagating
/// arithmetic shift, `simd_clamp(0, 255)` is the lane-wise §14.5
/// `clamp255`, and the post-clamp `cast::<u8>()` of a value already in
/// `0..=255` equals the scalar `as u8`. Byte-exactness against
/// [`sixtap_2d_scalar`] is enforced on every fixture by
/// `sixtap_2d_simd_matches_scalar_on_stress_inputs`.
#[cfg(feature = "simd")]
#[inline]
fn sixtap_2d_simd(halo: &[u8; 81], mx: usize, my: usize, filters: &[[i32; 6]; 8]) -> [u8; 16] {
    use core::simd::cmp::SimdOrd;
    use core::simd::num::{SimdInt, SimdUint};
    use core::simd::Simd;

    let zero = Simd::<i32, 4>::splat(0);
    let max = Simd::<i32, 4>::splat(255);
    let seven = Simd::<i32, 4>::splat(7);

    // Horizontal pass: 9 rows (the 4 block rows + 2 above + 3 below) ×
    // 4 cols — exactly the scalar `sixtap_horiz(temp, 4, halo, 9, ref0 =
    // 2, 4, 9, filters[mx])` call. With the halo's block origin at
    // column 2, output column c's tap-k support sample is
    // halo[r*9 + c + k], so the four lanes of tap k are the contiguous
    // run halo[r*9 + k .. r*9 + k + 4].
    let fh = &filters[mx];
    let fh_v: [Simd<i32, 4>; 6] = core::array::from_fn(|k| Simd::splat(fh[k]));
    let mut temp = [Simd::<i32, 4>::splat(0); 9];
    for (r, trow) in temp.iter_mut().enumerate() {
        let row = &halo[r * 9..r * 9 + 9];
        // §18.3 interp: a = Σ fil[k] * support[k]; seed with the +64
        // rounding term so the shift below is the spec's (a + 64) >> 7.
        let mut acc = Simd::<i32, 4>::splat(64);
        for (k, tap) in fh_v.iter().enumerate() {
            let support: Simd<i32, 4> = Simd::<u8, 4>::from_slice(&row[k..k + 4]).cast::<i32>();
            acc += support * tap;
        }
        *trow = (acc >> seven).simd_clamp(zero, max);
    }

    // Vertical pass: 4 rows × 4 cols reading the intermediate at block-
    // origin row 2 — the scalar `sixtap_vert(out, 4, temp, 4, ref0 = 8,
    // 4, 4, filters[my])` call. Output row r's tap-k support row is
    // temp row r + k (= ref0/4 + r + k - 2), already a four-lane vector.
    let fv = &filters[my];
    let fv_v: [Simd<i32, 4>; 6] = core::array::from_fn(|k| Simd::splat(fv[k]));
    let mut out = [0u8; 16];
    for r in 0..4 {
        let mut acc = Simd::<i32, 4>::splat(64);
        for (k, tap) in fv_v.iter().enumerate() {
            acc += temp[r + k] * tap;
        }
        let res = (acc >> seven).simd_clamp(zero, max);
        out[r * 4..r * 4 + 4].copy_from_slice(&res.cast::<u8>().to_array());
    }
    out
}

/// Fetch a 4×4 whole-pixel prediction block from a reference plane with
/// the §20.14 `build_mc_border` edge-replication rule.
///
/// `mv` is the §18.1-adjusted (eighth-pixel) vector for this sub-block;
/// its fractional part must be zero (whole-pixel — see
/// [`whole_pixel_fraction_is_zero`]). The integer pixel offset is
/// `mv >> 3` per component (sign-propagating right shift, §18.2 "rounding
/// up or left by right-shifting each component 3 bits with sign
/// propagation").
///
/// `(blk_x, blk_y)` is the sub-block's top-left position in the plane in
/// pixels (before the motion offset). `plane` / `stride` describe the
/// reference plane; `w` / `h` are its width / height in pixels. Pixels
/// the source position would read outside `[0, w) × [0, h)` are clamped
/// to the nearest edge pixel — the whole-pixel specialisation of
/// `build_mc_border` (which, for an integer fetch, just replicates the
/// border row / column).
pub fn fetch_block_whole_pixel(
    plane: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    blk_x: usize,
    blk_y: usize,
    mv: Mv,
) -> [u8; 16] {
    // §18.2: integer pixel offset is the eighth-pixel vector shifted
    // right 3 with sign propagation.
    let off_x = (mv.col >> 3) as isize;
    let off_y = (mv.row >> 3) as isize;
    let src_x0 = blk_x as isize + off_x;
    let src_y0 = blk_y as isize + off_y;

    let w_i = w as isize;
    let h_i = h as isize;
    let mut out = [0u8; 16];

    // Round-170 fast path: when all 16 source positions land strictly
    // inside the plane, no per-pixel `.clamp()` or bounds checks are
    // needed and each output row is a contiguous 4-byte slice of the
    // reference. This is the common case mid-frame; the edge-replication
    // fallback below stays bit-identical to the slow path for MBs that
    // straddle the picture border. (§20.14 `build_mc_border` semantics
    // are preserved — only the in-bounds branch is specialised.)
    if src_x0 >= 0 && src_y0 >= 0 && src_x0 + 4 <= w_i && src_y0 + 4 <= h_i {
        let x0 = src_x0 as usize;
        let y0 = src_y0 as usize;
        for r in 0..4 {
            let row_start = (y0 + r) * stride + x0;
            out[r * 4..r * 4 + 4].copy_from_slice(&plane[row_start..row_start + 4]);
        }
        return out;
    }

    for r in 0..4 {
        // §20.14 build_mc_border: rows past the top / bottom edge read
        // the first / last in-bounds row (edge replication).
        let sy = (src_y0 + r as isize).clamp(0, h_i - 1);
        for c in 0..4 {
            // Columns past the left / right edge read the first / last
            // in-bounds column.
            let sx = (src_x0 + c as isize).clamp(0, w_i - 1);
            out[r * 4 + c] = plane[sy as usize * stride + sx as usize];
        }
    }
    out
}

/// Fetch the 9×9 edge-replicated halo a six-tap 4×4 interpolation needs —
/// RFC 6386 §20.14 `build_mc_border` / `recon_1_edge_block`.
///
/// The six-tap convolution of a 4×4 sub-block references two pixels
/// before and three pixels after the block in each dimension, so the
/// source support is a `(4+5) × (4+5) = 9×9` region. This fetch
/// reproduces the §20.14 `build_mc_border` emulated block: it covers
/// source positions `[src_y0 - 2, src_y0 + 6] × [src_x0 - 2, src_x0 + 6]`
/// (where `(src_x0, src_y0) = (blk_x, blk_y) + (mv >> 3)` is the
/// integer-offset block origin), clamping any out-of-plane read to the
/// nearest edge pixel. The result is row-major with stride 9; the 4×4
/// block origin sits at `halo[(2, 2)]`, matching [`sixtap_2d`]'s
/// expectation.
///
/// For an in-bounds fetch this is just a 9×9 window; the clamp makes the
/// edge case identical to `build_mc_border`'s row/column replication,
/// which is the whole-pixel [`fetch_block_whole_pixel`] rule extended to
/// the support halo.
pub fn fetch_block_halo(
    plane: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    blk_x: usize,
    blk_y: usize,
    mv: Mv,
) -> [u8; 81] {
    let off_x = (mv.col >> 3) as isize;
    let off_y = (mv.row >> 3) as isize;
    // The halo origin is two pixels above/left of the integer block
    // origin (the first six-tap support pixel).
    let src_x0 = blk_x as isize + off_x - 2;
    let src_y0 = blk_y as isize + off_y - 2;

    let w_i = w as isize;
    let h_i = h as isize;
    let mut out = [0u8; 81];

    // Round-170 fast path: when the 9×9 halo lands strictly inside the
    // plane (the dominant case for inter-MBs that don't touch the
    // picture border), each output row is a contiguous 9-byte slice of
    // the reference plane — no `.clamp()` per pixel, no per-byte bound
    // checks. The fallback below is bit-identical for halos that
    // straddle the border.
    if src_x0 >= 0 && src_y0 >= 0 && src_x0 + 9 <= w_i && src_y0 + 9 <= h_i {
        let x0 = src_x0 as usize;
        let y0 = src_y0 as usize;
        for r in 0..9 {
            let row_start = (y0 + r) * stride + x0;
            out[r * 9..r * 9 + 9].copy_from_slice(&plane[row_start..row_start + 9]);
        }
        return out;
    }

    for r in 0..9 {
        let sy = (src_y0 + r as isize).clamp(0, h_i - 1);
        for c in 0..9 {
            let sx = (src_x0 + c as isize).clamp(0, w_i - 1);
            out[r * 9 + c] = plane[sy as usize * stride + sx as usize];
        }
    }
    out
}

/// Predict one 4×4 sub-block — RFC 6386 §20.14 `filter_block`.
///
/// Mirrors the reference decoder's `filter_block`: extract the
/// eighth-pixel fractions `mx = mv.col & 7`, `my = mv.row & 7`. If both
/// are zero the prediction is the whole-pixel copy
/// ([`fetch_block_whole_pixel`]); otherwise it is the [`sixtap_2d`]
/// horizontal-then-vertical convolution of the [`fetch_block_halo`]
/// support region under the version-selected `filters`.
///
/// `mv` is the §18.1-adjusted (eighth-pixel) vector for this plane;
/// `(blk_x, blk_y)` is the sub-block's top-left position in the plane in
/// pixels (before the motion offset); `w` / `h` are the plane's pixel
/// dimensions.
#[allow(clippy::too_many_arguments)] // mirrors the §20.14 filter_block call shape.
pub fn filter_block_4x4(
    plane: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    blk_x: usize,
    blk_y: usize,
    mv: Mv,
    filters: &[[i32; 6]; 8],
) -> [u8; 16] {
    let mx = (mv.col & 7) as usize;
    let my = (mv.row & 7) as usize;
    if mx == 0 && my == 0 {
        // Whole-pixel: the §18.3 "prediction subblock is simply copied".
        fetch_block_whole_pixel(plane, stride, w, h, blk_x, blk_y, mv)
    } else {
        let halo = fetch_block_halo(plane, stride, w, h, blk_x, blk_y, mv);
        sixtap_2d(&halo, mx, my, filters)
    }
}

/// Build the §18.2 whole-pixel prediction buffer for a non-SPLITMV
/// inter-predicted macroblock.
///
/// The same `luma_mv` (raw §17-decoded, quarter-pixel) applies to all
/// sixteen Y sub-blocks; the chroma vector is the §18.1 average of that
/// one vector. `full_pixel` is the version-3 full-pel-chroma flag
/// (`frame_hdr.version == 3`).
///
/// Returns the predicted macroblock (no residue) as a
/// [`ReconstructedMb`], or [`MotionCompError::SubPixelNotSupported`] if
/// the §18.1-adjusted luma or chroma vector has a non-zero fractional
/// part (sub-pixel interpolation is a future round).
///
/// `(mb_col, mb_row)` is the macroblock position; the reference planes
/// supply the dimensions.
pub fn predict_inter_mb_whole_pixel(
    reference: &ReferencePlanes<'_>,
    mb_col: usize,
    mb_row: usize,
    luma_mv: Mv,
    full_pixel: bool,
) -> Result<ReconstructedMb, MotionCompError> {
    // §18.1: double the quarter-pixel luma vector to eighth-pixel
    // resolution.
    let mut ymv = stored_luma_mv(luma_mv);
    // §18.1 chroma averaging (avg() of the single repeated vector).
    let mut uvmv = chroma_mv(ymv);
    if full_pixel {
        // §18.1 version-3 full-pel-chroma truncation.
        ymv = apply_full_pixel(ymv);
        uvmv = apply_full_pixel(uvmv);
    }

    // §18.3 whole-pixel gate: both fractions must be zero.
    if !whole_pixel_fraction_is_zero(ymv) || !whole_pixel_fraction_is_zero(uvmv) {
        return Err(MotionCompError::SubPixelNotSupported);
    }

    let lw = reference.mb_cols * 16;
    let lh = reference.mb_rows * 16;
    let cw = reference.mb_cols * 8;
    let ch = reference.mb_rows * 8;

    let mut out = ReconstructedMb::default();

    // Luma: sixteen 4×4 sub-blocks, each fetched with the same ymv.
    let y_x0 = mb_col * 16;
    let y_y0 = mb_row * 16;
    for sb in 0..4 {
        for sc in 0..4 {
            let blk = fetch_block_whole_pixel(
                reference.y,
                reference.y_stride,
                lw,
                lh,
                y_x0 + sc * 4,
                y_y0 + sb * 4,
                ymv,
            );
            // Write into the 16×16 row-major output.
            for r in 0..4 {
                let dst = (sb * 4 + r) * 16 + sc * 4;
                out.y[dst..dst + 4].copy_from_slice(&blk[r * 4..r * 4 + 4]);
            }
        }
    }

    // Chroma: four 4×4 sub-blocks per plane, each fetched with uvmv.
    let uv_x0 = mb_col * 8;
    let uv_y0 = mb_row * 8;
    for sb in 0..2 {
        for sc in 0..2 {
            let ublk = fetch_block_whole_pixel(
                reference.u,
                reference.uv_stride,
                cw,
                ch,
                uv_x0 + sc * 4,
                uv_y0 + sb * 4,
                uvmv,
            );
            let vblk = fetch_block_whole_pixel(
                reference.v,
                reference.uv_stride,
                cw,
                ch,
                uv_x0 + sc * 4,
                uv_y0 + sb * 4,
                uvmv,
            );
            for r in 0..4 {
                let dst = (sb * 4 + r) * 8 + sc * 4;
                out.u[dst..dst + 4].copy_from_slice(&ublk[r * 4..r * 4 + 4]);
                out.v[dst..dst + 4].copy_from_slice(&vblk[r * 4..r * 4 + 4]);
            }
        }
    }

    Ok(out)
}

/// Extract a 4×4 sub-block at `(sub_row, sub_col)` (4-pixel units) from a
/// row-major plane of `stride` pixels.
#[inline]
fn extract_4x4(plane: &[u8], stride: usize, sub_row: usize, sub_col: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    let y0 = sub_row * 4;
    let x0 = sub_col * 4;
    for r in 0..4 {
        let src = (y0 + r) * stride + x0;
        out[r * 4..r * 4 + 4].copy_from_slice(&plane[src..src + 4]);
    }
    out
}

/// Splat a 4×4 sub-block back into a row-major plane.
#[inline]
fn insert_4x4(plane: &mut [u8], stride: usize, sub_row: usize, sub_col: usize, src: &[u8; 16]) {
    let y0 = sub_row * 4;
    let x0 = sub_col * 4;
    for r in 0..4 {
        let dst = (y0 + r) * stride + x0;
        plane[dst..dst + 4].copy_from_slice(&src[r * 4..r * 4 + 4]);
    }
}

/// Reconstruct one whole-pixel inter-predicted (non-SPLITMV)
/// macroblock — RFC 6386 §16.2 / §18 prediction + §14 residue.
///
/// This is the inter analogue of [`crate::reconstruct::decode_keyframe_mb_non_bpred`]:
///
/// 1. Build the §18.2 whole-pixel prediction buffer
///    ([`predict_inter_mb_whole_pixel`]).
/// 2. If `mb_skip_coeff`, the residue is zero and the prediction is the
///    reconstruction (§11.1 skip short-circuit).
/// 3. Otherwise inverse-WHT the Y2 block (an inter non-SPLITMV MB has a
///    Y2 block, §14.2), seed each Y sub-block DC, inverse-DCT all 24
///    sub-blocks, and add the residue with `clamp255` (§14.5).
///
/// The coefficient arrays are pre-dequantized (the caller's
/// responsibility, matching the keyframe path).
///
/// Returns [`MotionCompError::SubPixelNotSupported`] when the vector is
/// not whole-pixel.
#[allow(clippy::too_many_arguments)] // each parameter is a distinct §14.2/§18 input.
pub fn reconstruct_inter_mb_whole_pixel(
    reference: &ReferencePlanes<'_>,
    mb_col: usize,
    mb_row: usize,
    luma_mv: Mv,
    full_pixel: bool,
    mb_skip_coeff: bool,
    y2_coeffs_dequant: &[i16; 16],
    y_coeffs_dequant: &[[i16; 16]; 16],
    u_coeffs_dequant: &[[i16; 16]; 4],
    v_coeffs_dequant: &[[i16; 16]; 4],
) -> Result<ReconstructedMb, MotionCompError> {
    let mut out = predict_inter_mb_whole_pixel(reference, mb_col, mb_row, luma_mv, full_pixel)?;

    if mb_skip_coeff {
        return Ok(out);
    }

    // §14.2: inverse-WHT Y2 and seed each Y sub-block's DC.
    let mut y2_residue = [0i16; 16];
    inverse_wht_4x4(y2_coeffs_dequant, &mut y2_residue);
    let mut y_coeffs = *y_coeffs_dequant;
    for i in 0..4 {
        for j in 0..4 {
            y_coeffs[i * 4 + j][0] = y2_residue[i * 4 + j];
        }
    }

    // Luma: inverse-DCT + add residue per sub-block.
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&y_coeffs[idx], &mut residue);
            let pred = extract_4x4(&out.y, 16, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.y, 16, i, j, &summed);
        }
    }

    // Chroma: inverse-DCT + add residue per sub-block, U then V.
    for i in 0..2 {
        for j in 0..2 {
            let idx = i * 2 + j;
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&u_coeffs_dequant[idx], &mut residue);
            let pred = extract_4x4(&out.u, 8, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.u, 8, i, j, &summed);

            let mut residue = [0i16; 16];
            inverse_dct_4x4(&v_coeffs_dequant[idx], &mut residue);
            let pred = extract_4x4(&out.v, 8, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.v, 8, i, j, &summed);
        }
    }

    Ok(out)
}

/// Build the §18.2 prediction buffer for a non-SPLITMV inter-predicted
/// macroblock, with §18.3 sub-pixel interpolation — RFC 6386 §18.
///
/// The full prediction path: the same `luma_mv` (raw §17-decoded,
/// quarter-pixel) applies to all sixteen Y sub-blocks; the chroma vector
/// is the §18.1 average of that one vector. Each sub-block is routed
/// through [`filter_block_4x4`], which copies whole-pixel sub-blocks and
/// six-tap-interpolates sub-pixel sub-blocks under `filters`.
///
/// `full_pixel` is the version-3 full-pel-chroma flag
/// (`frame_hdr.version == 3`); `filters` is the §20.14 version-selected
/// tap set (use [`filter_set_for_version`] then [`FilterSet::taps`]).
///
/// `(mb_col, mb_row)` is the macroblock position; the reference planes
/// supply the dimensions.
pub fn predict_inter_mb(
    reference: &ReferencePlanes<'_>,
    mb_col: usize,
    mb_row: usize,
    luma_mv: Mv,
    full_pixel: bool,
    filters: &[[i32; 6]; 8],
) -> ReconstructedMb {
    // §18.1: double the quarter-pixel luma vector to eighth-pixel
    // resolution.
    let mut ymv = stored_luma_mv(luma_mv);
    // §18.1 chroma averaging (avg() of the single repeated vector).
    let mut uvmv = chroma_mv(ymv);
    if full_pixel {
        // §18.1 version-3 full-pel-chroma truncation.
        ymv = apply_full_pixel(ymv);
        uvmv = apply_full_pixel(uvmv);
    }

    let lw = reference.mb_cols * 16;
    let lh = reference.mb_rows * 16;
    let cw = reference.mb_cols * 8;
    let ch = reference.mb_rows * 8;

    let mut out = ReconstructedMb::default();

    // Luma: sixteen 4×4 sub-blocks, each filtered with the same ymv.
    let y_x0 = mb_col * 16;
    let y_y0 = mb_row * 16;
    for sb in 0..4 {
        for sc in 0..4 {
            let blk = filter_block_4x4(
                reference.y,
                reference.y_stride,
                lw,
                lh,
                y_x0 + sc * 4,
                y_y0 + sb * 4,
                ymv,
                filters,
            );
            for r in 0..4 {
                let dst = (sb * 4 + r) * 16 + sc * 4;
                out.y[dst..dst + 4].copy_from_slice(&blk[r * 4..r * 4 + 4]);
            }
        }
    }

    // Chroma: four 4×4 sub-blocks per plane, each filtered with uvmv.
    let uv_x0 = mb_col * 8;
    let uv_y0 = mb_row * 8;
    for sb in 0..2 {
        for sc in 0..2 {
            let ublk = filter_block_4x4(
                reference.u,
                reference.uv_stride,
                cw,
                ch,
                uv_x0 + sc * 4,
                uv_y0 + sb * 4,
                uvmv,
                filters,
            );
            let vblk = filter_block_4x4(
                reference.v,
                reference.uv_stride,
                cw,
                ch,
                uv_x0 + sc * 4,
                uv_y0 + sb * 4,
                uvmv,
                filters,
            );
            for r in 0..4 {
                let dst = (sb * 4 + r) * 8 + sc * 4;
                out.u[dst..dst + 4].copy_from_slice(&ublk[r * 4..r * 4 + 4]);
                out.v[dst..dst + 4].copy_from_slice(&vblk[r * 4..r * 4 + 4]);
            }
        }
    }

    out
}

/// Reconstruct one inter-predicted (non-SPLITMV) macroblock with §18.3
/// sub-pixel interpolation — RFC 6386 §16.2 / §18 prediction + §14
/// residue.
///
/// The full-resolution analogue of [`reconstruct_inter_mb_whole_pixel`]:
///
/// 1. Build the §18.2/§18.3 prediction buffer ([`predict_inter_mb`],
///    whole-pixel copy or six-tap interpolation per sub-block).
/// 2. If `mb_skip_coeff`, the residue is zero and the prediction is the
///    reconstruction (§11.1 skip short-circuit).
/// 3. Otherwise inverse-WHT the Y2 block (an inter non-SPLITMV MB has a
///    Y2 block, §14.2), seed each Y sub-block DC, inverse-DCT all 24
///    sub-blocks, and add the residue with `clamp255` (§14.5).
///
/// `filters` is the §20.14 version-selected tap set. The coefficient
/// arrays are pre-dequantized (the caller's responsibility, matching the
/// keyframe path).
#[allow(clippy::too_many_arguments)] // each parameter is a distinct §14.2/§18 input.
pub fn reconstruct_inter_mb(
    reference: &ReferencePlanes<'_>,
    mb_col: usize,
    mb_row: usize,
    luma_mv: Mv,
    full_pixel: bool,
    filters: &[[i32; 6]; 8],
    mb_skip_coeff: bool,
    y2_coeffs_dequant: &[i16; 16],
    y_coeffs_dequant: &[[i16; 16]; 16],
    u_coeffs_dequant: &[[i16; 16]; 4],
    v_coeffs_dequant: &[[i16; 16]; 4],
) -> ReconstructedMb {
    let mut out = predict_inter_mb(reference, mb_col, mb_row, luma_mv, full_pixel, filters);

    if mb_skip_coeff {
        return out;
    }

    // §14.2: inverse-WHT Y2 and seed each Y sub-block's DC.
    let mut y2_residue = [0i16; 16];
    inverse_wht_4x4(y2_coeffs_dequant, &mut y2_residue);
    let mut y_coeffs = *y_coeffs_dequant;
    for i in 0..4 {
        for j in 0..4 {
            y_coeffs[i * 4 + j][0] = y2_residue[i * 4 + j];
        }
    }

    // Luma: inverse-DCT + add residue per sub-block.
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&y_coeffs[idx], &mut residue);
            let pred = extract_4x4(&out.y, 16, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.y, 16, i, j, &summed);
        }
    }

    // Chroma: inverse-DCT + add residue per sub-block, U then V.
    for i in 0..2 {
        for j in 0..2 {
            let idx = i * 2 + j;
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&u_coeffs_dequant[idx], &mut residue);
            let pred = extract_4x4(&out.u, 8, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.u, 8, i, j, &summed);

            let mut residue = [0i16; 16];
            inverse_dct_4x4(&v_coeffs_dequant[idx], &mut residue);
            let pred = extract_4x4(&out.v, 8, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.v, 8, i, j, &summed);
        }
    }

    out
}

// ───────────────────────── §16.4 / §18 SPLITMV ─────────────────────────────

/// The §18.1 chroma sub-block index of luma sub-block `b` (`0..=15`) —
/// RFC 6386 §18.1, the §20.11 expression `(b>>1&1) + (b>>2&2)`.
///
/// Maps the sixteen Y sub-blocks onto the four chroma sub-blocks (each
/// chroma sub-block covers four luma sub-blocks); the closed form expands
/// to the §18.1 enumeration `{0,1,4,5}→0`, `{2,3,6,7}→1`, `{8,9,12,13}→2`,
/// `{10,11,14,15}→3`.
#[inline]
pub fn chroma_idx_for_luma_subblock(b: usize) -> usize {
    ((b >> 1) & 1) + ((b >> 2) & 2)
}

/// Derive the four §18.1 chroma sub-block motion vectors from the sixteen
/// SPLITMV per-luma-sub-block vectors — RFC 6386 §18.1 `avg(...)`.
///
/// Per §18.1 each chroma sub-block's vector is the four-vector average of
/// the four luma sub-blocks occupying the same visible area; the §20.14
/// `avg()` primitive (also used by [`chroma_mv`] for the whole-MB case)
/// performs the divide-by-8 + sign-aware rounding. `split_luma_mvs` are
/// the raw quarter-pixel per-sub-block vectors (the SPLITMV decode output);
/// the result is in eighth-pixel units already (matching `chroma_mv`).
pub fn split_chroma_mvs(split_luma_mvs: &[Mv; 16]) -> [Mv; 4] {
    // Sum the (doubled) luma components per chroma slot.
    let mut sum_row = [0i32; 4];
    let mut sum_col = [0i32; 4];
    for (b, mv) in split_luma_mvs.iter().enumerate() {
        let c = chroma_idx_for_luma_subblock(b);
        // §18.1 "stored luma motion vectors are all doubled": the §20.11
        // chroma loop sums the *doubled* vectors before the avg(). We mimic
        // by stacking 4 copies of the stored-luma value into the §18.1 avg.
        let r = (mv.row as i32).wrapping_mul(2);
        let cc = (mv.col as i32).wrapping_mul(2);
        sum_row[c] = sum_row[c].wrapping_add(r);
        sum_col[c] = sum_col[c].wrapping_add(cc);
    }
    let mut out = [Mv::default(); 4];
    for c in 0..4 {
        let s_r = sum_row[c];
        let s_c = sum_col[c];
        // §18.1 avg(): sign-aware divide-by-8 with rounding adjustment.
        let row = if s_r >= 0 {
            (s_r + 4) >> 3
        } else {
            -((-s_r + 4) >> 3)
        };
        let col = if s_c >= 0 {
            (s_c + 4) >> 3
        } else {
            -((-s_c + 4) >> 3)
        };
        out[c] = Mv {
            row: row as i16,
            col: col as i16,
        };
    }
    out
}

/// Build the §18.2/§18.3 prediction buffer for a SPLITMV inter-predicted
/// macroblock — RFC 6386 §16.4 / §18.
///
/// Sixteen luma sub-blocks, each interpolated under *its own* quarter-pixel
/// vector (after the §18.1 doubling); four chroma sub-blocks per plane,
/// each interpolated under the §18.1 four-vector average of the
/// corresponding luma group ([`split_chroma_mvs`]).
///
/// Unlike [`predict_inter_mb`], no §18.1 secondary clamp is applied —
/// §18.1 page 114 "secondary clamping is not performed for SPLITMV
/// macroblocks, meaning any subblock's motion vector ... may point outside
/// the clamping zone."
///
/// `full_pixel` is the version-3 full-pel-chroma flag; `filters` is the
/// §20.14 version-selected tap set; `split_luma_mvs` is the
/// [`crate::near_mv::SplitMvResult::split_mvs`] field.
pub fn predict_split_mv(
    reference: &ReferencePlanes<'_>,
    mb_col: usize,
    mb_row: usize,
    split_luma_mvs: &[Mv; 16],
    full_pixel: bool,
    filters: &[[i32; 6]; 8],
) -> ReconstructedMb {
    let lw = reference.mb_cols * 16;
    let lh = reference.mb_rows * 16;
    let cw = reference.mb_cols * 8;
    let ch = reference.mb_rows * 8;

    let mut out = ReconstructedMb::default();

    // Luma: each 4×4 sub-block under its own §18.1-doubled vector.
    let y_x0 = mb_col * 16;
    let y_y0 = mb_row * 16;
    for sb in 0..4 {
        for sc in 0..4 {
            let b = sb * 4 + sc;
            let mut ymv = stored_luma_mv(split_luma_mvs[b]);
            if full_pixel {
                ymv = apply_full_pixel(ymv);
            }
            let blk = filter_block_4x4(
                reference.y,
                reference.y_stride,
                lw,
                lh,
                y_x0 + sc * 4,
                y_y0 + sb * 4,
                ymv,
                filters,
            );
            for r in 0..4 {
                let dst = (sb * 4 + r) * 16 + sc * 4;
                out.y[dst..dst + 4].copy_from_slice(&blk[r * 4..r * 4 + 4]);
            }
        }
    }

    // Chroma: the four §18.1 averaged vectors, one per chroma sub-block.
    let chroma = split_chroma_mvs(split_luma_mvs);
    let uv_x0 = mb_col * 8;
    let uv_y0 = mb_row * 8;
    for sb in 0..2 {
        for sc in 0..2 {
            let c = sb * 2 + sc;
            let mut uvmv = chroma[c];
            if full_pixel {
                uvmv = apply_full_pixel(uvmv);
            }
            let ublk = filter_block_4x4(
                reference.u,
                reference.uv_stride,
                cw,
                ch,
                uv_x0 + sc * 4,
                uv_y0 + sb * 4,
                uvmv,
                filters,
            );
            let vblk = filter_block_4x4(
                reference.v,
                reference.uv_stride,
                cw,
                ch,
                uv_x0 + sc * 4,
                uv_y0 + sb * 4,
                uvmv,
                filters,
            );
            for r in 0..4 {
                let dst = (sb * 4 + r) * 8 + sc * 4;
                out.u[dst..dst + 4].copy_from_slice(&ublk[r * 4..r * 4 + 4]);
                out.v[dst..dst + 4].copy_from_slice(&vblk[r * 4..r * 4 + 4]);
            }
        }
    }

    out
}

/// Reconstruct one SPLITMV macroblock — RFC 6386 §16.4 / §18 prediction +
/// §14 residue.
///
/// SPLITMV macroblocks have no Y2 block (§14.2 "for ... `SPLITMV`, the
/// 0th Y coefficients are part of the residue signal"), so each Y
/// sub-block's full 16 coefficients go straight through the inverse DCT
/// (no WHT seeding).
///
/// 1. Build the §18.2/§18.3 prediction buffer ([`predict_split_mv`]).
/// 2. If `mb_skip_coeff`, the residue is zero and the prediction is the
///    reconstruction (§11.1 skip short-circuit).
/// 3. Otherwise inverse-DCT each of the 16 Y + 4 U + 4 V sub-blocks and
///    add the residue with `clamp255`.
///
/// `filters` is the §20.14 version-selected tap set. The coefficient
/// arrays are pre-dequantized (caller's responsibility, matching
/// [`reconstruct_inter_mb`]).
#[allow(clippy::too_many_arguments)] // each parameter is a distinct §14.2/§16.4 input.
pub fn reconstruct_split_mv_mb(
    reference: &ReferencePlanes<'_>,
    mb_col: usize,
    mb_row: usize,
    split_luma_mvs: &[Mv; 16],
    full_pixel: bool,
    filters: &[[i32; 6]; 8],
    mb_skip_coeff: bool,
    y_coeffs_dequant: &[[i16; 16]; 16],
    u_coeffs_dequant: &[[i16; 16]; 4],
    v_coeffs_dequant: &[[i16; 16]; 4],
) -> ReconstructedMb {
    let mut out = predict_split_mv(
        reference,
        mb_col,
        mb_row,
        split_luma_mvs,
        full_pixel,
        filters,
    );

    if mb_skip_coeff {
        return out;
    }

    // Luma: inverse-DCT + add residue per sub-block (no Y2 for SPLITMV).
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&y_coeffs_dequant[idx], &mut residue);
            let pred = extract_4x4(&out.y, 16, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.y, 16, i, j, &summed);
        }
    }

    // Chroma: inverse-DCT + add residue per sub-block, U then V.
    for i in 0..2 {
        for j in 0..2 {
            let idx = i * 2 + j;
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&u_coeffs_dequant[idx], &mut residue);
            let pred = extract_4x4(&out.u, 8, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.u, 8, i, j, &summed);

            let mut residue = [0i16; 16];
            inverse_dct_4x4(&v_coeffs_dequant[idx], &mut residue);
            let pred = extract_4x4(&out.v, 8, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.v, 8, i, j, &summed);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal test-side VP8 boolean encoder, mirroring the proven
    /// encoder in `motion_vector::tests` / `bool_decoder::tests`. Used to
    /// build bitstreams the §16.2 reference selector decodes back.
    struct BoolEncoder {
        out: Vec<u8>,
        range: u32,
        bottom: u32,
        bit_count: i32,
    }

    impl BoolEncoder {
        fn new() -> Self {
            BoolEncoder {
                out: Vec::new(),
                range: 255,
                bottom: 0,
                bit_count: 24,
            }
        }

        fn write_bool(&mut self, prob: u8, val: bool) {
            let split = 1 + (((self.range - 1) * prob as u32) >> 8);
            if val {
                self.bottom = self.bottom.wrapping_add(split);
                self.range -= split;
            } else {
                self.range = split;
            }
            while self.range < 128 {
                self.range <<= 1;
                if (self.bottom >> 31) & 1 == 1 {
                    let mut i = self.out.len();
                    while i > 0 {
                        i -= 1;
                        if self.out[i] == 255 {
                            self.out[i] = 0;
                        } else {
                            self.out[i] = self.out[i].wrapping_add(1);
                            break;
                        }
                    }
                }
                self.bottom <<= 1;
                self.bit_count -= 1;
                if self.bit_count == 0 {
                    let byte = (self.bottom >> 24) as u8;
                    self.out.push(byte);
                    self.bottom &= (1 << 24) - 1;
                    self.bit_count = 8;
                }
            }
        }

        fn finish(mut self) -> Vec<u8> {
            let c = self.bit_count;
            let v = self.bottom;
            if v & (1u32 << (32 - c)) != 0 {
                let mut i = self.out.len();
                while i > 0 {
                    i -= 1;
                    if self.out[i] == 255 {
                        self.out[i] = 0;
                    } else {
                        self.out[i] = self.out[i].wrapping_add(1);
                        break;
                    }
                }
            }
            let mut v = v;
            v <<= c & 7;
            let mut c_shift = c >> 3;
            while c_shift > 0 {
                v <<= 8;
                c_shift -= 1;
            }
            for _ in 0..4 {
                self.out.push((v >> 24) as u8);
                v <<= 8;
            }
            while self.out.len() < 2 {
                self.out.push(0);
            }
            self.out
        }
    }

    // ----- §16.2 reference frame selection ---------------------------

    #[test]
    fn ref_frame_last_reads_single_zero() {
        // prob_last bit 0 → Last; prob_gf is never read.
        let mut enc = BoolEncoder::new();
        enc.write_bool(128, false); // prob_last = 0
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        assert_eq!(
            select_ref_frame(&mut dec, 128, 128).unwrap(),
            RefFrame::Last
        );
    }

    #[test]
    fn ref_frame_golden_reads_one_then_zero() {
        let mut enc = BoolEncoder::new();
        enc.write_bool(128, true); // prob_last = 1
        enc.write_bool(128, false); // prob_gf = 0 → Golden
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        assert_eq!(
            select_ref_frame(&mut dec, 128, 128).unwrap(),
            RefFrame::Golden
        );
    }

    #[test]
    fn ref_frame_altref_reads_one_then_one() {
        let mut enc = BoolEncoder::new();
        enc.write_bool(128, true); // prob_last = 1
        enc.write_bool(128, true); // prob_gf = 1 → AltRef
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        assert_eq!(
            select_ref_frame(&mut dec, 128, 128).unwrap(),
            RefFrame::AltRef
        );
    }

    #[test]
    fn ref_frame_uses_distinct_probs() {
        // Encode Golden against asymmetric probs; decode with the same
        // probs must recover Golden. This proves prob_last and prob_gf
        // are wired to the right read order.
        let mut enc = BoolEncoder::new();
        enc.write_bool(30, true); // prob_last = 30
        enc.write_bool(200, false); // prob_gf = 200
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        assert_eq!(
            select_ref_frame(&mut dec, 30, 200).unwrap(),
            RefFrame::Golden
        );
    }

    // ----- §18.1 motion-vector adjustments ---------------------------

    #[test]
    fn stored_luma_doubles_each_component() {
        // §18.1: quarter-pixel luma vector doubled to eighth-pixel.
        assert_eq!(
            stored_luma_mv(Mv { row: 3, col: -5 }),
            Mv { row: 6, col: -10 }
        );
        // Extremes: ±1023 quarter-pel → ±2046 eighth-pel (the §18.1
        // stated range).
        assert_eq!(
            stored_luma_mv(Mv {
                row: 1023,
                col: -1023
            }),
            Mv {
                row: 2046,
                col: -2046
            }
        );
    }

    #[test]
    fn avg_matches_spec_formula() {
        // Positive rounding: (s + 4) >> 3.
        assert_eq!(avg(8, 8, 8, 8), (32 + 4) >> 3); // = 4
        assert_eq!(avg(1, 1, 1, 1), (4 + 4) >> 3); // = 1
        assert_eq!(avg(0, 0, 0, 0), 0);
        // Negative rounding: -((-s + 4) >> 3) — symmetric to positive.
        assert_eq!(avg(-8, -8, -8, -8), -((32 + 4) >> 3)); // = -4
        assert_eq!(avg(-1, -1, -1, -1), -((4 + 4) >> 3)); // = -1
                                                          // Mixed sign, sum exactly zero.
        assert_eq!(avg(5, -5, 3, -3), 0);
    }

    #[test]
    fn chroma_mv_of_repeated_vector_is_quarter() {
        // For a whole-MB vector, chroma_mv = avg(v,v,v,v). With the
        // eighth-pixel doubled luma vector v=6, avg(6,6,6,6) = (24+4)>>3
        // = 3.
        let ymv = stored_luma_mv(Mv { row: 3, col: -5 }); // (6, -10)
        let uvmv = chroma_mv(ymv);
        assert_eq!(uvmv.row, ((6 * 4 + 4) >> 3) as i16); // 3
        assert_eq!(uvmv.col, -(((10 * 4 + 4) >> 3) as i16)); // -5
    }

    #[test]
    fn chroma_mv_matches_reference_closed_form() {
        // §20.14 writes the whole-MB chroma vector as
        // (c + 1 + (c >> 31) * 2) / 2 per component. Cross-check our
        // avg()-based chroma_mv against that closed form across a spread
        // of eighth-pixel luma components.
        fn ref_form(c: i32) -> i32 {
            (c + 1 + (c >> 31) * 2) / 2
        }
        for &c in &[0i32, 2, 6, 10, 14, 100, -2, -6, -10, -14, -100, 2046, -2046] {
            let mv = Mv {
                row: c as i16,
                col: c as i16,
            };
            let got = chroma_mv(mv);
            assert_eq!(got.row as i32, ref_form(c), "row c={c}");
            assert_eq!(got.col as i32, ref_form(c), "col c={c}");
        }
    }

    #[test]
    fn full_pixel_truncation_clears_low_three_bits() {
        // §18.1 version-3: x &= ~7, y &= ~7.
        assert_eq!(
            apply_full_pixel(Mv { row: 13, col: 9 }),
            Mv { row: 8, col: 8 }
        );
        assert_eq!(
            apply_full_pixel(Mv { row: 16, col: 24 }),
            Mv { row: 16, col: 24 }
        );
        // Negative two's-complement & ~7 rounds toward negative infinity.
        assert_eq!(apply_full_pixel(Mv { row: -1, col: -9 }).row, -8);
        assert_eq!(apply_full_pixel(Mv { row: -1, col: -9 }).col, -16);
    }

    #[test]
    fn whole_pixel_test_detects_fraction() {
        assert!(whole_pixel_fraction_is_zero(Mv { row: 16, col: -8 }));
        assert!(whole_pixel_fraction_is_zero(Mv { row: 0, col: 0 }));
        assert!(!whole_pixel_fraction_is_zero(Mv { row: 1, col: 0 }));
        assert!(!whole_pixel_fraction_is_zero(Mv { row: 0, col: 3 }));
        assert!(!whole_pixel_fraction_is_zero(Mv { row: -1, col: 0 }));
    }

    // ----- §20.14 build_mc_border whole-pixel fetch ------------------

    /// A 8×8 single-plane ramp where pixel (r, c) = r * 8 + c.
    fn ramp_plane(w: usize, h: usize) -> Vec<u8> {
        (0..w * h).map(|i| (i % 256) as u8).collect()
    }

    #[test]
    fn fetch_zero_mv_copies_in_place() {
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        // Sub-block at (0,0), zero vector → the top-left 4×4.
        let blk = fetch_block_whole_pixel(&plane, w, w, h, 0, 0, Mv { row: 0, col: 0 });
        let mut expect = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                expect[r * 4 + c] = plane[r * w + c];
            }
        }
        assert_eq!(blk, expect);
    }

    #[test]
    fn fetch_whole_pixel_offset_shifts_origin() {
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        // Eighth-pixel vector (8, 16) → integer offset (1, 2): origin
        // moves down 1 row, right 2 cols.
        let blk = fetch_block_whole_pixel(&plane, w, w, h, 0, 0, Mv { row: 8, col: 16 });
        let mut expect = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                expect[r * 4 + c] = plane[(1 + r) * w + (2 + c)];
            }
        }
        assert_eq!(blk, expect);
    }

    #[test]
    fn fetch_left_edge_replicates_first_column() {
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        // Negative col offset large enough to push all 4 columns past
        // the left edge: every fetched pixel is column 0 of its row.
        // mv.col = -64 → off_x = -8; sub-block at x=0 → src_x0 = -8,
        // all columns clamp to 0.
        let blk = fetch_block_whole_pixel(&plane, w, w, h, 0, 0, Mv { row: 0, col: -64 });
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(blk[r * 4 + c], plane[r * w], "row {r} col {c}");
            }
        }
    }

    #[test]
    fn fetch_top_edge_replicates_first_row() {
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        // mv.row = -64 → off_y = -8; all rows clamp to row 0.
        let blk = fetch_block_whole_pixel(&plane, w, w, h, 0, 0, Mv { row: -64, col: 0 });
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(blk[r * 4 + c], plane[c], "row {r} col {c}");
            }
        }
    }

    #[test]
    fn fetch_bottom_right_corner_replicates() {
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        // Push the fetch fully past the bottom-right corner: every pixel
        // becomes the (h-1, w-1) corner pixel.
        let blk = fetch_block_whole_pixel(&plane, w, w, h, 4, 4, Mv { row: 64, col: 64 });
        let corner = plane[(h - 1) * w + (w - 1)];
        for px in blk {
            assert_eq!(px, corner);
        }
    }

    // ----- §18.2 whole-MB prediction ---------------------------------

    /// Build a reference frame whose planes are filled with a deterministic
    /// per-position value, big enough for a 2×2 macroblock grid.
    fn build_reference(mb_cols: usize, mb_rows: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let lw = mb_cols * 16;
        let lh = mb_rows * 16;
        let cw = mb_cols * 8;
        let ch = mb_rows * 8;
        let y = (0..lw * lh).map(|i| (i % 251) as u8).collect();
        let u = (0..cw * ch).map(|i| ((i * 3 + 1) % 251) as u8).collect();
        let v = (0..cw * ch).map(|i| ((i * 7 + 2) % 251) as u8).collect();
        (y, u, v)
    }

    #[test]
    fn predict_zero_mv_copies_corresponding_mb() {
        // A zero motion vector predicts the current MB from the matching
        // MB in the reference frame (mv_zero, §16.3).
        let (y, u, v) = build_reference(2, 2);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 32,
            uv_stride: 16,
            mb_cols: 2,
            mb_rows: 2,
        };
        // Predict MB (1, 1) with a zero vector.
        let pred =
            predict_inter_mb_whole_pixel(&reference, 1, 1, Mv { row: 0, col: 0 }, false).unwrap();

        // The luma block must equal reference MB (1,1) verbatim.
        for r in 0..16 {
            for c in 0..16 {
                let ref_px = y[(16 + r) * 32 + (16 + c)];
                assert_eq!(pred.y[r * 16 + c], ref_px, "luma ({r},{c})");
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                let ru = u[(8 + r) * 16 + (8 + c)];
                let rv = v[(8 + r) * 16 + (8 + c)];
                assert_eq!(pred.u[r * 8 + c], ru, "u ({r},{c})");
                assert_eq!(pred.v[r * 8 + c], rv, "v ({r},{c})");
            }
        }
    }

    #[test]
    fn predict_whole_pixel_offset_shifts_source() {
        // Pick a luma vector whose §18.1-doubled value AND its chroma
        // average are both whole-pixel, so the prediction is a pure copy
        // with a non-trivial integer offset (no §18.3 interpolation).
        //
        // luma quarter-pel (8, 16) → doubled (16, 32): both eighth-pel
        // fractions zero → luma integer offset (16>>3, 32>>3) = (2, 4).
        // chroma row avg(16,16,16,16) = (64+4)>>3 = 8 → fraction zero →
        // chroma row offset 8>>3 = 1; chroma col avg(32..) = (128+4)>>3 =
        // 16 → fraction zero → chroma col offset 16>>3 = 2. So both
        // planes copy in place at shifted origins. Use a 3×3 grid so the
        // shifted reads stay in-bounds and predict the centre MB (1,1).
        let (y, u, v) = build_reference(3, 3);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 48,
            uv_stride: 24,
            mb_cols: 3,
            mb_rows: 3,
        };
        let mv = Mv { row: 8, col: 16 };
        let pred = predict_inter_mb_whole_pixel(&reference, 1, 1, mv, false).unwrap();

        // Luma: MB (1,1) origin (16, 16) + offset (2, 4).
        for r in 0..16 {
            for c in 0..16 {
                let ref_px = y[(16 + 2 + r) * 48 + (16 + 4 + c)];
                assert_eq!(pred.y[r * 16 + c], ref_px, "luma ({r},{c})");
            }
        }
        // Chroma: MB (1,1) origin (8, 8) + offset (1, 2).
        for r in 0..8 {
            for c in 0..8 {
                let ru = u[(8 + 1 + r) * 24 + (8 + 2 + c)];
                let rv = v[(8 + 1 + r) * 24 + (8 + 2 + c)];
                assert_eq!(pred.u[r * 8 + c], ru, "u ({r},{c})");
                assert_eq!(pred.v[r * 8 + c], rv, "v ({r},{c})");
            }
        }
    }

    #[test]
    fn predict_sub_pixel_vector_rejected() {
        // A vector whose doubled luma fraction is non-zero must be
        // refused (sub-pel interpolation is a future round). row=1
        // quarter-pel → 2 eighth-pel → fraction 2, non-zero.
        let (y, u, v) = build_reference(1, 1);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 16,
            uv_stride: 8,
            mb_cols: 1,
            mb_rows: 1,
        };
        assert_eq!(
            predict_inter_mb_whole_pixel(&reference, 0, 0, Mv { row: 1, col: 0 }, false),
            Err(MotionCompError::SubPixelNotSupported)
        );
    }

    #[test]
    fn predict_chroma_sub_pixel_vector_rejected() {
        // A luma vector that is whole-pixel but whose chroma average is
        // sub-pixel must also be refused. luma quarter-pel (2, 0) →
        // doubled (4, 0): luma fraction 4 — non-zero, so it's already
        // refused at luma. Use (4, 0) quarter → doubled (8, 0): luma
        // whole-pixel; chroma avg(8..)=4 → fraction 4, sub-pixel.
        let (y, u, v) = build_reference(1, 1);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 16,
            uv_stride: 8,
            mb_cols: 1,
            mb_rows: 1,
        };
        assert_eq!(
            predict_inter_mb_whole_pixel(&reference, 0, 0, Mv { row: 4, col: 0 }, false),
            Err(MotionCompError::SubPixelNotSupported)
        );
    }

    #[test]
    fn predict_full_pixel_version_truncates_to_whole() {
        // With full_pixel set, a vector that would be chroma-sub-pixel is
        // truncated to whole-pixel and accepted. luma quarter (4, 0) →
        // doubled (8, 0); chroma avg(8..) = 4 (sub-pixel); full_pixel
        // truncates both to whole-pixel → (8&~7, 0) = (8, 0) luma,
        // (4&~7, 0) = (0, 0) chroma. Accepted.
        //
        // Use a 2×2 grid and predict MB (0,0) so the luma row-1 offset
        // stays in-bounds; cross-check luma against the same
        // edge-clamped fetch the implementation uses, so the test is
        // robust to any boundary behaviour.
        let (y, u, v) = build_reference(2, 2);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 32,
            uv_stride: 16,
            mb_cols: 2,
            mb_rows: 2,
        };
        let pred =
            predict_inter_mb_whole_pixel(&reference, 0, 0, Mv { row: 4, col: 0 }, true).unwrap();
        // Luma offset (8>>3, 0) = (1, 0): row shifted down by 1. Compare
        // against fetch_block_whole_pixel for each 4×4 sub-block.
        for sb in 0..4 {
            for sc in 0..4 {
                let blk =
                    fetch_block_whole_pixel(&y, 32, 32, 32, sc * 4, sb * 4, Mv { row: 8, col: 0 });
                for r in 0..4 {
                    for c in 0..4 {
                        let pr = sb * 4 + r;
                        let pc = sc * 4 + c;
                        assert_eq!(pred.y[pr * 16 + pc], blk[r * 4 + c], "luma ({pr},{pc})");
                    }
                }
            }
        }
        // Chroma offset (0, 0): copied in place from MB (0,0).
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(pred.u[r * 8 + c], u[r * 16 + c], "u ({r},{c})");
            }
        }
    }

    // ----- §16.2 / §18 / §14 inter reconstruction --------------------

    #[test]
    fn reconstruct_skip_equals_prediction() {
        // mb_skip_coeff → zero residue → reconstruction == prediction.
        let (y, u, v) = build_reference(2, 2);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 32,
            uv_stride: 16,
            mb_cols: 2,
            mb_rows: 2,
        };
        let mv = Mv { row: 0, col: 0 };
        let pred = predict_inter_mb_whole_pixel(&reference, 1, 0, mv, false).unwrap();
        let recon = reconstruct_inter_mb_whole_pixel(
            &reference,
            1,
            0,
            mv,
            false,
            true, // skip
            &[0i16; 16],
            &[[0i16; 16]; 16],
            &[[0i16; 16]; 4],
            &[[0i16; 16]; 4],
        )
        .unwrap();
        assert_eq!(recon, pred);
    }

    #[test]
    fn reconstruct_adds_dc_residue() {
        // A pure Y2 DC term lifts every Y sub-block's DC uniformly; the
        // reconstruction must equal prediction + a constant on luma.
        let (y, u, v) = build_reference(1, 1);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 16,
            uv_stride: 8,
            mb_cols: 1,
            mb_rows: 1,
        };
        let mv = Mv { row: 0, col: 0 };
        let pred = predict_inter_mb_whole_pixel(&reference, 0, 0, mv, false).unwrap();

        // A Y2 block with a single DC term. The inverse WHT of a
        // DC-only block distributes (dc + 3) >> 3 to every output; that
        // becomes each Y sub-block's DC, and the inverse DCT of a
        // DC-only block distributes (dc + 4) >> 3 to every pixel.
        let mut y2 = [0i16; 16];
        y2[0] = 64; // WHT: each sub-block DC = (64 + 3) >> 3 ... see below
        let recon = reconstruct_inter_mb_whole_pixel(
            &reference,
            0,
            0,
            mv,
            false,
            false,
            &y2,
            &[[0i16; 16]; 16],
            &[[0i16; 16]; 4],
            &[[0i16; 16]; 4],
        )
        .unwrap();

        // Compute the expected luma delta independently via the same
        // public transform primitives.
        let mut y2_residue = [0i16; 16];
        inverse_wht_4x4(&y2, &mut y2_residue);
        // Every Y sub-block gets DC = y2_residue[idx]; with a single DC
        // input the WHT spreads it evenly, so all sixteen are equal.
        let sub_dc = y2_residue[0];
        let mut residue = [0i16; 16];
        let mut coeffs = [0i16; 16];
        coeffs[0] = sub_dc;
        inverse_dct_4x4(&coeffs, &mut residue);
        let delta = residue[0];

        // Chroma has no residue → equals prediction.
        assert_eq!(recon.u, pred.u);
        assert_eq!(recon.v, pred.v);
        // Luma: each pixel is clamp255(pred + delta).
        for i in 0..256 {
            let expect = (pred.y[i] as i32 + delta as i32).clamp(0, 255) as u8;
            assert_eq!(recon.y[i], expect, "luma px {i}");
        }
        // Sanity: the residue is genuinely non-zero so this isn't a
        // trivial pass.
        assert_ne!(delta, 0);
    }

    #[test]
    fn reconstruct_sub_pixel_vector_rejected() {
        let (y, u, v) = build_reference(1, 1);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 16,
            uv_stride: 8,
            mb_cols: 1,
            mb_rows: 1,
        };
        assert_eq!(
            reconstruct_inter_mb_whole_pixel(
                &reference,
                0,
                0,
                Mv { row: 1, col: 0 },
                false,
                true,
                &[0i16; 16],
                &[[0i16; 16]; 16],
                &[[0i16; 16]; 4],
                &[[0i16; 16]; 4],
            ),
            Err(MotionCompError::SubPixelNotSupported)
        );
    }

    // ----- §18.3 sub-pixel interpolation -----------------------------

    #[test]
    fn filter_tables_match_spec_values() {
        // §18.3 / §20.14: spot-check the documented rows.
        assert_eq!(SIXTAP_FILTERS[0], [0, 0, 128, 0, 0, 0]);
        assert_eq!(SIXTAP_FILTERS[1], [0, -6, 123, 12, -1, 0]);
        assert_eq!(SIXTAP_FILTERS[2], [2, -11, 108, 36, -8, 1]);
        assert_eq!(SIXTAP_FILTERS[4], [3, -16, 77, 77, -16, 3]);
        assert_eq!(SIXTAP_FILTERS[7], [0, -1, 12, 123, -6, 0]);
        assert_eq!(BILINEAR_FILTERS[0], [0, 0, 128, 0, 0, 0]);
        assert_eq!(BILINEAR_FILTERS[4], [0, 0, 64, 64, 0, 0]);
        assert_eq!(BILINEAR_FILTERS[7], [0, 0, 16, 112, 0, 0]);
    }

    #[test]
    fn filter_taps_always_sum_to_128() {
        // §18.3: "Because DC is always passed, taps always sum to 128."
        for f in SIXTAP_FILTERS.iter() {
            assert_eq!(f.iter().sum::<i32>(), 128);
        }
        for f in BILINEAR_FILTERS.iter() {
            assert_eq!(f.iter().sum::<i32>(), 128);
        }
    }

    #[test]
    fn bilinear_uses_only_centre_two_taps() {
        // The bilinear filter never reaches the outer support pixels.
        for f in BILINEAR_FILTERS.iter() {
            assert_eq!(f[0], 0);
            assert_eq!(f[1], 0);
            assert_eq!(f[4], 0);
            assert_eq!(f[5], 0);
        }
    }

    #[test]
    fn filter_set_selected_by_version() {
        // §20.14: version 0 → six-tap; non-zero → bilinear.
        assert_eq!(filter_set_for_version(0), FilterSet::Sixtap);
        assert_eq!(filter_set_for_version(1), FilterSet::Bilinear);
        assert_eq!(filter_set_for_version(2), FilterSet::Bilinear);
        assert_eq!(filter_set_for_version(3), FilterSet::Bilinear);
        assert!(core::ptr::eq(
            FilterSet::Sixtap.taps(),
            &SIXTAP_FILTERS as &[[i32; 6]; 8]
        ));
        assert!(core::ptr::eq(
            FilterSet::Bilinear.taps(),
            &BILINEAR_FILTERS as &[[i32; 6]; 8]
        ));
    }

    #[test]
    fn interp_matches_spec_formula() {
        // §18.3 interp: clamp255((sum_i support[i]*fil[i] + 64) >> 7).
        // Whole-pixel filter (index 0) passes the centre tap unchanged.
        let support = [10u8, 20, 30, 40, 50, 60];
        assert_eq!(interp(&SIXTAP_FILTERS[0], &support), 30);
        // 1/2 symmetric filter, computed by hand:
        // 3*10 -16*20 +77*30 +77*40 -16*50 +3*60
        // = 30 -320 +2310 +3080 -800 +180 = 4480; (4480+64)>>7 = 35.
        assert_eq!(interp(&SIXTAP_FILTERS[4], &support), 35);
        // Constant support → DC passes (taps sum 128): (128*K+64)>>7 = K.
        let flat = [200u8; 6];
        for f in SIXTAP_FILTERS.iter() {
            assert_eq!(interp(f, &flat), 200);
        }
        // clamp255 floor: a strongly negative partial sum clamps to 0.
        // filter index 2 has negative taps -11 and -8; drive the negative
        // taps high and the positive taps to 0.
        let neg = [0u8, 255, 0, 0, 255, 0];
        // 2*0 -11*255 +108*0 +36*0 -8*255 +1*0 = -4845; +64 = -4781;
        // >>7 (arithmetic) = -38 → clamp255 → 0.
        assert_eq!(interp(&SIXTAP_FILTERS[2], &neg), 0);
    }

    /// Independent §18.3 `interp` over a slice with explicit stride —
    /// a literal transcription of the spec-prose code block, used to
    /// cross-check [`sixtap_2d`].
    fn ref_interp(fil: &[i32; 6], p: &[u8], origin: isize, s: isize) -> u8 {
        let mut a = 0i32;
        let mut idx = origin - s - s; // "move back two positions"
        for &tap in fil.iter() {
            a += p[idx as usize] as i32 * tap;
            idx += s;
        }
        ((a + 64) >> 7).clamp(0, 255) as u8
    }

    /// Independent §18.3 Hinterp + Vinterp over the same 9×9 halo
    /// [`sixtap_2d`] consumes, reproducing the spec prose verbatim:
    /// horizontal pass producing 9 rows of 4, vertical pass producing
    /// the final 4×4. The halo block origin is at (2, 2), stride 9.
    fn ref_sixtap_2d(halo: &[u8; 81], mx: usize, my: usize, filters: &[[i32; 6]; 8]) -> [u8; 16] {
        let hfil = &filters[mx];
        let vfil = &filters[my];
        // Hinterp: temp[9][4]. Spec advances p by the vertical stride per
        // row, starting at the subblock origin. interp moves back two
        // positions, so the support spans rows [origin-? n/a here] — for
        // the horizontal pass the step is 1 (within a row) and we read
        // rows [block_origin_row .. +8]. The halo's row 2 is the block
        // origin row; Hinterp's row r corresponds to halo row r (the
        // §20.14 sixtap_horiz starts at reference - 2*stride = halo row
        // 0, producing 9 rows). To match that, iterate halo rows 0..9.
        let mut temp = [[0u8; 4]; 9];
        for (r, trow) in temp.iter_mut().enumerate() {
            for (c, t) in trow.iter_mut().enumerate() {
                // Row base in the halo: block-origin column is 2.
                let base = r * 9 + 2;
                *t = ref_interp(hfil, halo, (base + c) as isize, 1);
            }
        }
        // Vinterp: read temp at block-origin row (halo row 2 → temp row
        // 2) with vertical step = 4 (row width). Flatten temp to a
        // 9*4 row-major buffer.
        let mut flat = [0u8; 36];
        for r in 0..9 {
            for c in 0..4 {
                flat[r * 4 + c] = temp[r][c];
            }
        }
        let mut out = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                // Block-origin row within the 9-row intermediate is row 2.
                let base = (2 + r) * 4 + c;
                out[r * 4 + c] = ref_interp(vfil, &flat, base as isize, 4);
            }
        }
        out
    }

    /// Pseudo-random 9×9 halo for cross-checking the convolution.
    fn rand_halo(seed: u64) -> [u8; 81] {
        let mut s = seed;
        let mut h = [0u8; 81];
        for px in h.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *px = (s >> 33) as u8;
        }
        h
    }

    #[test]
    fn sixtap_2d_matches_independent_spec_reference() {
        // Cross-check the production sixtap_2d against a literal
        // transcription of §18.3 Hinterp/Vinterp over the same halo, for
        // every (mx, my) sub-pixel pair and both filter sets.
        for set in [&SIXTAP_FILTERS, &BILINEAR_FILTERS] {
            for seed in 0..3u64 {
                let halo = rand_halo(seed * 977 + 13);
                for mx in 0..8 {
                    for my in 0..8 {
                        let got = sixtap_2d(&halo, mx, my, set);
                        let want = ref_sixtap_2d(&halo, mx, my, set);
                        assert_eq!(got, want, "mx={mx} my={my} seed={seed}");
                    }
                }
            }
        }
    }

    #[test]
    fn sixtap_2d_flat_halo_passes_dc() {
        // A constant source must interpolate to the same constant for any
        // fraction (taps sum to 128, DC preserved, no clamp).
        let halo = [137u8; 81];
        for mx in 0..8 {
            for my in 0..8 {
                let out = sixtap_2d(&halo, mx, my, &SIXTAP_FILTERS);
                assert!(out.iter().all(|&p| p == 137), "mx={mx} my={my}");
                let outb = sixtap_2d(&halo, mx, my, &BILINEAR_FILTERS);
                assert!(outb.iter().all(|&p| p == 137), "bilinear mx={mx} my={my}");
            }
        }
    }

    #[test]
    fn sixtap_2d_whole_fraction_copies_block() {
        // (mx,my)=(0,0) is the degenerate whole-pixel filter: the centre
        // tap is 128 in both dimensions, so sixtap_2d copies the 4×4
        // block at the halo origin (rows/cols 2..6).
        let halo = rand_halo(42);
        let out = sixtap_2d(&halo, 0, 0, &SIXTAP_FILTERS);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(out[r * 4 + c], halo[(2 + r) * 9 + (2 + c)], "({r},{c})");
            }
        }
    }

    #[test]
    fn sixtap_2d_horizontal_only_known_values() {
        // my=0 (vertical filter is the centre-tap copy), so the result is
        // a pure horizontal convolution of the block-origin rows. Build a
        // horizontal ramp halo so the answer is easy to verify against
        // ref_interp directly.
        let mut halo = [0u8; 81];
        for r in 0..9 {
            for c in 0..9 {
                halo[r * 9 + c] = (c * 20) as u8; // 0,20,40,...,160
            }
        }
        let out = sixtap_2d(&halo, 2, 0, &SIXTAP_FILTERS); // mx=1/4
        for r in 0..4 {
            for c in 0..4 {
                let want = ref_interp(&SIXTAP_FILTERS[2], &halo, (r * 9 + 2 + c) as isize, 1);
                assert_eq!(out[r * 4 + c], want, "({r},{c})");
            }
        }
    }

    // Round-269 dispatcher equivalence (scalar vs the public entry,
    // which is SIMD under nightly + `simd`). Mirrors the §14
    // `*_simd_matches_scalar_on_stress_inputs` pairs. On stable / no
    // `simd` feature this exercises the scalar-vs-scalar identity
    // (harmless); on nightly + `simd` it's the primary safety net:
    // `sixtap_2d` dispatches to `sixtap_2d_simd` and must be byte-exact
    // against `sixtap_2d_scalar` on every fixture.

    #[test]
    fn sixtap_2d_simd_matches_scalar_on_stress_inputs() {
        // Structured halos: flat extremes, opposing ramps, alternating
        // extremes (the worst case for the clamp endpoints), plus
        // deterministic LCG halos — for every (mx, my) eighth-pixel
        // fraction pair and both §18.3 filter sets.
        let mut hramp = [0u8; 81];
        let mut vramp = [0u8; 81];
        let mut checker = [0u8; 81];
        for r in 0..9 {
            for c in 0..9 {
                hramp[r * 9 + c] = (c * 31) as u8;
                vramp[r * 9 + c] = (r * 31) as u8;
                checker[r * 9 + c] = if (r + c) % 2 == 0 { 255 } else { 0 };
            }
        }
        // All-floor, all-ceiling, the structured halos, then the LCG set.
        let mut halos: Vec<[u8; 81]> = vec![[0u8; 81], [255u8; 81], hramp, vramp, checker];
        for seed in 0..8u64 {
            halos.push(rand_halo(seed * 6661 + 7));
        }

        for set in [&SIXTAP_FILTERS, &BILINEAR_FILTERS] {
            for (h, halo) in halos.iter().enumerate() {
                for mx in 0..8 {
                    for my in 0..8 {
                        let got = sixtap_2d(halo, mx, my, set);
                        let want = sixtap_2d_scalar(halo, mx, my, set);
                        assert_eq!(got, want, "halo={h} mx={mx} my={my}");
                    }
                }
            }
        }
    }

    #[test]
    fn sixtap_2d_accumulator_extremes_match_scalar() {
        // The §18.3 dot product over the half-pel taps {3, -16, 77, 77,
        // -16, 3} spans [-32·255, 160·255] = [-8160, 40800] — past
        // i16::MAX, the reason the SIMD lanes are i32. Drive both
        // extremes through output column 0 (support columns c + k for
        // c = 0 are exactly k = 0..6): 255 on the positive-tap columns
        // {0, 2, 3, 5} maxes the sum (→ clamp ceiling), 255 on the
        // negative-tap columns {1, 4} alone mins it (→ clamp floor).
        let mut max_halo = [0u8; 81];
        let mut min_halo = [0u8; 81];
        for r in 0..9 {
            for c in 0..9 {
                let pos_tap = matches!(c % 6, 0 | 2 | 3 | 5);
                max_halo[r * 9 + c] = if pos_tap { 255 } else { 0 };
                min_halo[r * 9 + c] = if pos_tap { 0 } else { 255 };
            }
        }
        let got_max = sixtap_2d(&max_halo, 4, 4, &SIXTAP_FILTERS);
        assert_eq!(got_max, sixtap_2d_scalar(&max_halo, 4, 4, &SIXTAP_FILTERS));
        // Column 0 hits the positive overflow region: clamp255((40800 +
        // 64) >> 7) = 255 in the horizontal pass, then the constant-255
        // column interpolates to 255 (taps sum to 128).
        assert_eq!(got_max[0], 255);

        let got_min = sixtap_2d(&min_halo, 4, 4, &SIXTAP_FILTERS);
        assert_eq!(got_min, sixtap_2d_scalar(&min_halo, 4, 4, &SIXTAP_FILTERS));
        // Column 0 hits the negative extreme: clamp255((-8160 + 64) >>
        // 7) = 0, then the constant-0 column stays 0.
        assert_eq!(got_min[0], 0);
    }

    // ----- §20.14 build_mc_border halo fetch -------------------------

    #[test]
    fn halo_in_bounds_is_plain_window() {
        // A fetch well inside the plane is a 9×9 window starting two
        // pixels up/left of the integer block origin.
        let w = 16;
        let h = 16;
        let plane = ramp_plane(w, h);
        // mv (16, 8) → integer offset (16>>3, 8>>3) = (2, 1). Block at
        // (4, 4) → src origin (6, 5); halo origin (6-2, 5-2) = (4, 3).
        let halo = fetch_block_halo(&plane, w, w, h, 4, 4, Mv { row: 16, col: 8 });
        for r in 0..9 {
            for c in 0..9 {
                assert_eq!(halo[r * 9 + c], plane[(4 + r) * w + (3 + c)], "({r},{c})");
            }
        }
    }

    #[test]
    fn halo_block_origin_at_2_2() {
        // The integer block origin must sit at halo[(2,2)].
        let w = 16;
        let h = 16;
        let plane = ramp_plane(w, h);
        // Zero mv, block at (4, 4): src origin (4, 4) → halo[(2,2)].
        let halo = fetch_block_halo(&plane, w, w, h, 4, 4, Mv { row: 0, col: 0 });
        assert_eq!(halo[2 * 9 + 2], plane[4 * w + 4]);
    }

    #[test]
    fn halo_clamps_at_top_left_corner() {
        // Block at (0,0), zero mv: halo origin (-2,-2). All out-of-plane
        // rows/cols replicate the nearest edge — matching build_mc_border.
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        let halo = fetch_block_halo(&plane, w, w, h, 0, 0, Mv { row: 0, col: 0 });
        for r in 0..9 {
            for c in 0..9 {
                let sy = (r as isize - 2).clamp(0, h as isize - 1) as usize;
                let sx = (c as isize - 2).clamp(0, w as isize - 1) as usize;
                assert_eq!(halo[r * 9 + c], plane[sy * w + sx], "({r},{c})");
            }
        }
    }

    // ----- §20.14 filter_block dispatch ------------------------------

    #[test]
    fn filter_block_whole_pixel_copies() {
        // mx|my == 0 → filter_block_4x4 == fetch_block_whole_pixel.
        let w = 16;
        let h = 16;
        let plane = ramp_plane(w, h);
        let mv = Mv { row: 16, col: 8 }; // integer offset, no fraction
        let filt = filter_block_4x4(&plane, w, w, h, 4, 4, mv, &SIXTAP_FILTERS);
        let copy = fetch_block_whole_pixel(&plane, w, w, h, 4, 4, mv);
        assert_eq!(filt, copy);
    }

    #[test]
    fn filter_block_sub_pixel_interpolates() {
        // A fractional mv routes through sixtap_2d on the fetched halo.
        let w = 16;
        let h = 16;
        let plane = ramp_plane(w, h);
        let mv = Mv { row: 10, col: 19 }; // my = 10&7 = 2, mx = 19&7 = 3
        let halo = fetch_block_halo(&plane, w, w, h, 4, 4, mv);
        let want = sixtap_2d(&halo, 3, 2, &SIXTAP_FILTERS);
        let got = filter_block_4x4(&plane, w, w, h, 4, 4, mv, &SIXTAP_FILTERS);
        assert_eq!(got, want);
    }

    // ----- §18.2/§18.3 sub-pixel whole-MB prediction -----------------

    #[test]
    fn predict_inter_mb_sub_pixel_matches_per_block() {
        // The whole-MB prediction must equal the per-sub-block
        // filter_block_4x4 applied with the §18.1-adjusted vectors.
        let (y, u, v) = build_reference(3, 3);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 48,
            uv_stride: 24,
            mb_cols: 3,
            mb_rows: 3,
        };
        // luma quarter-pel (3, 5) → doubled (6, 10): luma fractions
        // (6&7, 10&7) = (6, 2), sub-pixel. chroma avg(6..)=3, avg(10..)=5
        // → fractions (3, 5), also sub-pixel.
        let luma_mv = Mv { row: 3, col: 5 };
        let pred = predict_inter_mb(&reference, 1, 1, luma_mv, false, &SIXTAP_FILTERS);

        let ymv = stored_luma_mv(luma_mv);
        let uvmv = chroma_mv(ymv);
        let lw = 48;
        let lh = 48;
        let cw = 24;
        let ch = 24;
        // Luma.
        for sb in 0..4 {
            for sc in 0..4 {
                let blk = filter_block_4x4(
                    &y,
                    48,
                    lw,
                    lh,
                    16 + sc * 4,
                    16 + sb * 4,
                    ymv,
                    &SIXTAP_FILTERS,
                );
                for r in 0..4 {
                    for c in 0..4 {
                        let pr = sb * 4 + r;
                        let pc = sc * 4 + c;
                        assert_eq!(pred.y[pr * 16 + pc], blk[r * 4 + c], "luma ({pr},{pc})");
                    }
                }
            }
        }
        // Chroma U.
        for sb in 0..2 {
            for sc in 0..2 {
                let blk = filter_block_4x4(
                    &u,
                    24,
                    cw,
                    ch,
                    8 + sc * 4,
                    8 + sb * 4,
                    uvmv,
                    &SIXTAP_FILTERS,
                );
                for r in 0..4 {
                    for c in 0..4 {
                        let pr = sb * 4 + r;
                        let pc = sc * 4 + c;
                        assert_eq!(pred.u[pr * 8 + pc], blk[r * 4 + c], "u ({pr},{pc})");
                    }
                }
            }
        }
    }

    #[test]
    fn predict_inter_mb_whole_pixel_agrees_with_legacy() {
        // For a whole-pixel vector, predict_inter_mb must produce the
        // same buffer as the legacy whole-pixel-only entry point.
        let (y, u, v) = build_reference(3, 3);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 48,
            uv_stride: 24,
            mb_cols: 3,
            mb_rows: 3,
        };
        let mv = Mv { row: 8, col: 16 }; // whole-pixel after §18.1
        let full = predict_inter_mb(&reference, 1, 1, mv, false, &SIXTAP_FILTERS);
        let legacy = predict_inter_mb_whole_pixel(&reference, 1, 1, mv, false).unwrap();
        assert_eq!(full, legacy);
    }

    #[test]
    fn predict_inter_mb_uses_selected_filter_set() {
        // Bilinear and six-tap give different results for the same
        // sub-pixel vector (the outer taps differ), so the prediction
        // must depend on the chosen set.
        let (y, u, v) = build_reference(3, 3);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 48,
            uv_stride: 24,
            mb_cols: 3,
            mb_rows: 3,
        };
        let luma_mv = Mv { row: 3, col: 5 };
        let six = predict_inter_mb(&reference, 1, 1, luma_mv, false, &SIXTAP_FILTERS);
        let bil = predict_inter_mb(&reference, 1, 1, luma_mv, false, &BILINEAR_FILTERS);
        assert_ne!(six.y, bil.y);
    }

    #[test]
    fn reconstruct_inter_mb_skip_equals_prediction() {
        let (y, u, v) = build_reference(3, 3);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 48,
            uv_stride: 24,
            mb_cols: 3,
            mb_rows: 3,
        };
        let mv = Mv { row: 3, col: 5 }; // sub-pixel
        let pred = predict_inter_mb(&reference, 1, 1, mv, false, &SIXTAP_FILTERS);
        let recon = reconstruct_inter_mb(
            &reference,
            1,
            1,
            mv,
            false,
            &SIXTAP_FILTERS,
            true, // skip
            &[0i16; 16],
            &[[0i16; 16]; 16],
            &[[0i16; 16]; 4],
            &[[0i16; 16]; 4],
        );
        assert_eq!(recon, pred);
    }

    #[test]
    fn reconstruct_inter_mb_sub_pixel_adds_dc_residue() {
        // Sub-pixel prediction + a pure Y2 DC term: luma lifts uniformly,
        // chroma equals prediction.
        let (y, u, v) = build_reference(3, 3);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 48,
            uv_stride: 24,
            mb_cols: 3,
            mb_rows: 3,
        };
        let mv = Mv { row: 3, col: 5 }; // sub-pixel
        let pred = predict_inter_mb(&reference, 1, 1, mv, false, &SIXTAP_FILTERS);

        let mut y2 = [0i16; 16];
        y2[0] = 64;
        let recon = reconstruct_inter_mb(
            &reference,
            1,
            1,
            mv,
            false,
            &SIXTAP_FILTERS,
            false,
            &y2,
            &[[0i16; 16]; 16],
            &[[0i16; 16]; 4],
            &[[0i16; 16]; 4],
        );

        // Independent luma delta via the public transforms.
        let mut y2_residue = [0i16; 16];
        inverse_wht_4x4(&y2, &mut y2_residue);
        let mut coeffs = [0i16; 16];
        coeffs[0] = y2_residue[0];
        let mut residue = [0i16; 16];
        inverse_dct_4x4(&coeffs, &mut residue);
        let delta = residue[0];
        assert_ne!(delta, 0);

        assert_eq!(recon.u, pred.u);
        assert_eq!(recon.v, pred.v);
        for i in 0..256 {
            let expect = (pred.y[i] as i32 + delta as i32).clamp(0, 255) as u8;
            assert_eq!(recon.y[i], expect, "luma px {i}");
        }
    }

    #[test]
    fn reconstruct_inter_mb_matches_legacy_for_whole_pixel() {
        // A whole-pixel vector through the full path must equal the
        // legacy whole-pixel reconstruction (filters are unused there).
        let (y, u, v) = build_reference(3, 3);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 48,
            uv_stride: 24,
            mb_cols: 3,
            mb_rows: 3,
        };
        let mv = Mv { row: 8, col: 16 };
        let mut y2 = [0i16; 16];
        y2[0] = 48;
        let mut ycoeffs = [[0i16; 16]; 16];
        ycoeffs[5][3] = 12;
        let ucoeffs = [[0i16; 16]; 4];
        let vcoeffs = [[0i16; 16]; 4];
        let full = reconstruct_inter_mb(
            &reference,
            1,
            1,
            mv,
            false,
            &SIXTAP_FILTERS,
            false,
            &y2,
            &ycoeffs,
            &ucoeffs,
            &vcoeffs,
        );
        let legacy = reconstruct_inter_mb_whole_pixel(
            &reference, 1, 1, mv, false, false, &y2, &ycoeffs, &ucoeffs, &vcoeffs,
        )
        .unwrap();
        assert_eq!(full, legacy);
    }
}
