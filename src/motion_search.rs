//! Encoder-side integer-pixel motion-search primitives.
//!
//! VP8's bitstream (RFC 6386) is silent on how an encoder picks its
//! motion vectors — §2 explicitly notes that the standard "does not
//! specify a particular encoding algorithm". The only on-wire constraint
//! is the §17.1 range: each motion-vector component is a signed integer
//! in `[-1023, +1023]` quarter-pixels of luma displacement, with whole
//! pixels at multiples of 4 (§17.1 page 108).
//!
//! This module is the smallest piece of infrastructure that a non-zero
//! MV codepath needs: given a source 16×16 luma block and a reference
//! plane, return a whole-pixel candidate MV that minimises the
//! sum-of-absolute-differences (SAD) between the source and the
//! reference patch the MV would select.
//!
//! ## Scope
//!
//! * **Whole-pixel descent followed by §18.3 half-pixel and quarter-
//!   pixel refinement.** [`small_diamond_search_luma`] finds the best
//!   integer-pixel MV via a 4-neighbour descent against the §20.14
//!   edge-replicating fetch primitive; [`half_pixel_refine_luma`] then
//!   probes the 8 half-pixel offsets around that whole-pixel result;
//!   [`quarter_pixel_refine_luma`] then probes the 8 quarter-pixel
//!   offsets around the half-pixel result. Each sub-pixel candidate is
//!   evaluated through the §18.3 six-tap synthesis the decoder runs
//!   (`version == 0` bicubic tap set), keyed by `(stored_luma_mv(mv) &
//!   7)` — so a §17 quarter-pixel MV of magnitude 1 selects the
//!   `1/4`-position row of the §18.3 filter table (`{ 2, -11, 108, 36,
//!   -8, 1 }`), magnitude 3 the `3/4`-position row, and so on. The
//!   returned MV is always on the §17 quarter-pixel grid (the on-wire
//!   resolution of `mv` per §17.1) and inside `[MV_MIN, MV_MAX]`.
//! * **Luma only.** The chroma plane is not sampled — the §18.1
//!   `chroma_mv = avg(v, v, v, v)` derivation means a half-pixel luma
//!   MV maps to a sub-pixel chroma MV by construction; sampling chroma
//!   SAD per candidate would noticeably increase per-iteration cost
//!   without typically changing the picked MV.
//! * **Small-diamond integer descent.** A 4-neighbour (N / S / E / W at
//!   ±1 whole pixel) descent from a caller-supplied center, terminating
//!   when no neighbour improves the SAD or after `max_iters` iterations.
//!   Larger integer patterns (hex, square, diamond-with-radius-N) are
//!   deferred.
//!
//! ## §17.1 range clamp
//!
//! All candidate MVs are clamped into `[MV_MIN, MV_MAX]` =
//! `[-1023, +1023]` quarter-pixels before the reference fetch. The
//! underlying §20.14 `build_mc_border` edge replication inside
//! [`crate::motion_comp::fetch_block_whole_pixel`] /
//! [`crate::motion_comp::fetch_block_halo`] keeps a candidate that
//! walks the patch (or the sixtap halo) off the picture safe: the fetch
//! just reads the nearest in-bounds row / column. The §17.1 clamp is
//! applied because anything outside that range cannot be coded in the
//! bitstream regardless of whether the reference fetch would be valid.
//!
//! ## Reference
//!
//! * RFC 6386 §2 — encoder algorithm not specified.
//! * RFC 6386 §17.1 — MV component range `[-1023, +1023]`, whole pixels
//!   at multiples of 4 (i.e. half-pixel = multiples of 2 = the
//!   [`HALF_PIXEL_STEP`] grid).
//! * RFC 6386 §18.1 — `stored_luma_mv` doubling that turns a §17
//!   quarter-pixel MV into the §18 eighth-pixel resolution the §18.3
//!   filter table indexes.
//! * RFC 6386 §18.3 — six-tap sub-pixel interpolation (`version == 0`
//!   bicubic / `version != 0` bilinear); the sub-pixel refinements run
//!   the MB-batched [`crate::motion_comp::sixtap_mb_luma`] synthesis,
//!   byte-exact with the per-sub-block `filter_block_4x4` tiling the
//!   decoder reproduces.
//! * RFC 6386 §20.14 — `build_mc_border` edge replication for fetches
//!   that walk off the picture.

use crate::motion_comp::{
    fetch_block_whole_pixel, fetch_luma_mb_halo, fetch_luma_mb_whole_pixel, filter_set_for_version,
    sixtap_mb_luma, stored_luma_mv,
};
use crate::motion_vector::Mv;

/// Minimum value of a single §17.1 MV component (quarter-pixel units).
pub const MV_MIN: i16 = -1023;

/// Maximum value of a single §17.1 MV component (quarter-pixel units).
pub const MV_MAX: i16 = 1023;

/// One whole-pixel step in §17 quarter-pixel units (`4 quarter-pixels`
/// per whole pixel).
pub const WHOLE_PIXEL_STEP: i16 = 4;

/// One half-pixel step in §17 quarter-pixel units (`2 quarter-pixels`
/// per half pixel). After §18.1 doubling this becomes the §18.3
/// eighth-pixel fraction `4` — i.e. the symmetric half-pixel tap row of
/// the §18.3 `filters` table (row 4: `{ 3, -16, 77, 77, -16, 3 }`).
pub const HALF_PIXEL_STEP: i16 = 2;

/// One quarter-pixel step in §17 quarter-pixel units (`1` per quarter
/// pixel — §17.1 codes `V` in quarter-pixels directly). After §18.1
/// doubling this becomes the §18.3 eighth-pixel fraction `2` — i.e. the
/// `1/4`-position tap row of the §18.3 `filters` table (row 2:
/// `{ 2, -11, 108, 36, -8, 1 }`). A `QUARTER_PIXEL_STEP` of magnitude 3
/// likewise selects row 6 (`3/4`, the reverse of row 2).
pub const QUARTER_PIXEL_STEP: i16 = 1;

/// Pixel-wise sum of absolute differences between two 16×16 blocks.
///
/// SAD is the cheapest distortion metric a motion search can use: an
/// `O(N)` integer subtract-abs-accumulate that does not require any
/// reconstruction round-trip. Returned as `u32` because a fully-
/// saturated 16×16 block sums to `256 * 255 = 65_280`, comfortably
/// inside `u32` and inside `i32` for downstream arithmetic.
///
/// Dispatch: always the scalar path. The §17 SIMD partner
/// [`block_sad_16x16_simd`] (compiled under the nightly-only `simd`
/// feature) is kept available and is asserted byte-exact against the
/// scalar listing on a 21-input stress set
/// (`block_sad_simd_matches_scalar_on_stress_inputs`), but the
/// `motion_search_descent/block_sad_16x16_single_pair` criterion
/// `--quick` numbers recorded in `BENCHMARKS.md` round 258 show a
/// trade-off: the SIMD path is **−36 %** in isolation on
/// `aarch64-apple-darwin` (4.08 ns vs 6.43 ns) but the
/// `half_pixel_refine_luma_8_offsets` / `quarter_pixel_refine_luma_8_offsets`
/// descent stages regress by **+13 %** under the same configuration —
/// inlining the SIMD leaf into the `mb_luma_sad_at_mv` body increases
/// NEON register pressure across the surrounding §18.3 synthesis loop
/// (measured on the pre-round-279 per-4×4
/// [`crate::motion_comp::filter_block_4x4`] shape; round 279 batched
/// that synthesis into one [`sixtap_mb_luma`] pass per candidate) and
/// pessimises the surrounding scheduling enough to swamp the
/// leaf-level win. Routing the public dispatcher
/// to [`block_sad_16x16_scalar`] under every feature configuration
/// keeps the descent stages on their fastest measured shape; the
/// `_simd` implementation stays in place so a future round can
/// re-target it (e.g. on a host where the regression flips, or with a
/// non-inlined wrapper that doesn't pollute LLVM's scheduling around
/// `mb_luma_sad_at_mv`), and the byte-equivalence test calls the
/// `_simd` path directly so the equivalence proof is preserved
/// regardless of the public dispatch. Mirrors the round-247 dispatch
/// split for [`crate::forward_transform::forward_dct_4x4`].
#[inline]
pub fn block_sad_16x16(src: &[u8; 256], pred: &[u8; 256]) -> u32 {
    block_sad_16x16_scalar(src, pred)
}

/// Scalar §17 SAD primitive — the straight-line
/// `Σ |src[i] - pred[i]|` reference path.
///
/// Bit-for-bit equivalent to the longhand spec definition. The public
/// [`block_sad_16x16`] dispatches here under every feature
/// configuration after the round-258 measurement (see the dispatch
/// note on `block_sad_16x16` for the trade-off). The `simd` feature
/// keeps [`block_sad_16x16_simd`] compiled + tested, and the
/// byte-equivalence proof against this listing stays standing on a
/// 21-input stress set.
#[inline]
fn block_sad_16x16_scalar(src: &[u8; 256], pred: &[u8; 256]) -> u32 {
    let mut acc: u32 = 0;
    for i in 0..256 {
        let s = src[i] as i32;
        let p = pred[i] as i32;
        acc += (s - p).unsigned_abs();
    }
    acc
}

/// SIMD §17 SAD primitive — `core::simd::Simd<u8, 16>` row-stencil
/// fan-out of [`block_sad_16x16_scalar`].
///
/// The 16×16 source / prediction pair lays out as 16 packed rows of 16
/// bytes each. Each row maps directly onto a 16-lane `Simd<u8, 16>`
/// load. The absolute-difference per lane is computed in `u8` as
/// `max(s, p) - min(s, p)` — a saturating subtract per-lane against the
/// other operand collapses to the same value but requires two-pass
/// `saturating_sub` and an OR, so the max/min shape stays simpler — and
/// then widened to `u16` and accumulated into a `Simd<u16, 16>` row
/// accumulator. After 16 rows the worst-case per-lane value is
/// `16 × 255 = 4_080`, well inside the `u16` envelope, so no
/// intermediate widening is needed inside the loop. A single
/// `reduce_sum()` at the end collapses the 16 lanes into the final
/// `u32` total.
///
/// No external SIMD layout reference was consulted — the layout falls
/// out of the §17 definition of SAD (linear sum of absolute byte
/// differences; the 16×16 block already packs as 16 rows × 16 bytes)
/// and the [`block_sad_16x16_scalar`] listing above. Byte-exact
/// against the scalar path on every test fixture.
#[cfg(feature = "simd")]
#[allow(dead_code)]
// The public `block_sad_16x16` dispatcher routes to
// the scalar partner under every feature configuration after the
// round-258 measurement; this SIMD listing is kept in tree as a
// future re-target target and is exercised by
// `block_sad_simd_matches_scalar_on_stress_inputs`.
#[inline]
fn block_sad_16x16_simd(src: &[u8; 256], pred: &[u8; 256]) -> u32 {
    use core::simd::cmp::SimdOrd;
    use core::simd::num::SimdUint;
    use core::simd::Simd;

    let mut acc: Simd<u16, 16> = Simd::splat(0);
    for row in 0..16 {
        let off = row * 16;
        let s: Simd<u8, 16> = Simd::from_slice(&src[off..off + 16]);
        let p: Simd<u8, 16> = Simd::from_slice(&pred[off..off + 16]);
        // Per-lane absolute difference: `max(s, p) - min(s, p)`. The
        // subtract is performed in `u8`; with `max >= min` it never
        // underflows.
        let absdiff = s.simd_max(p) - s.simd_min(p);
        // Widen to `u16` so the cross-row accumulator stays inside the
        // `16 * 255 = 4_080` per-lane envelope a `u16` covers (the
        // scalar partner's `u32` accumulator is wider than necessary
        // for that step, but its single-pass scalar loop has no reason
        // to pay for a second accumulator width).
        acc += absdiff.cast::<u16>();
    }
    // Final horizontal reduce: widen `u16` → `u32` so the per-lane sum
    // and the cross-lane sum both stay inside `u32` (a fully-saturated
    // 16×16 block totals `256 * 255 = 65_280`).
    acc.cast::<u32>().reduce_sum()
}

/// A borrow of one reference frame's **luma plane** sized for motion
/// search at whole-pixel granularity.
///
/// The search only ever samples luma (the §18.1 `chroma_mv = avg(v, v,
/// v, v)` derivation means a whole-pixel luma MV maps directly to a
/// whole-pixel chroma MV, and chroma SAD adds noticeably more cost per
/// candidate without changing which MV the descent picks). This struct
/// bundles the four parameters every per-candidate fetch needs into a
/// single argument, keeping the function signatures inside this module
/// readable.
#[derive(Debug, Clone, Copy)]
pub struct LumaRef<'a> {
    /// The reference frame's luma plane, row-major.
    pub plane: &'a [u8],
    /// Stride of `plane` in bytes (= the row pitch).
    pub stride: usize,
    /// Plane width in pixels (= the visible / coded luma width).
    pub width: usize,
    /// Plane height in pixels (= the visible / coded luma height).
    pub height: usize,
}

/// A motion-search result: the best whole-pixel MV the search found
/// and the SAD it achieved against the source.
///
/// `mv` is in §17 quarter-pixel units (whole-pixel ⇒ multiples of
/// `WHOLE_PIXEL_STEP = 4`) and is clamped into `[MV_MIN, MV_MAX]`
/// per §17.1. `sad` is the [`block_sad_16x16`] value at that MV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchResult {
    /// The selected motion vector (§17.1 quarter-pixel units,
    /// whole-pixel = multiples of `WHOLE_PIXEL_STEP`).
    pub mv: Mv,
    /// The 16×16 luma SAD between the source block and the reference
    /// patch [`mv`](Self::mv) selects.
    pub sad: u32,
}

/// Compute the 16×16 luma SAD for one whole-pixel MV candidate.
///
/// The reference plane is sampled at `(mb_col * 16, mb_row * 16) +
/// (mv >> 2)` whole pixels (with §20.14 edge replication for any rows
/// or columns past the picture boundary). The MV must be a whole-pixel
/// vector (both components multiples of [`WHOLE_PIXEL_STEP`]) and must
/// already be clamped into `[MV_MIN, MV_MAX]` — debug builds assert
/// both; release builds silently round/clamp via the underlying
/// `fetch_block_whole_pixel` call.
///
/// This is the per-candidate SAD evaluator used by
/// [`small_diamond_search_luma`]; it is exposed publicly so a future
/// alternative search shape (hex, full-search, …) can be built against
/// the same evaluator.
pub fn mb_luma_sad_at_whole_mv(
    reference: LumaRef<'_>,
    mb_col: usize,
    mb_row: usize,
    src_y: &[u8; 256],
    mv: Mv,
) -> u32 {
    debug_assert_eq!(mv.row % WHOLE_PIXEL_STEP, 0, "MV row must be whole-pixel");
    debug_assert_eq!(mv.col % WHOLE_PIXEL_STEP, 0, "MV col must be whole-pixel");
    debug_assert!((MV_MIN..=MV_MAX).contains(&mv.row));
    debug_assert!((MV_MIN..=MV_MAX).contains(&mv.col));

    let blk_x0 = mb_col * 16;
    let blk_y0 = mb_row * 16;
    // §18.1 stored_luma_mv doubles the §17 quarter-pixel vector into
    // the §18 eighth-pixel resolution `fetch_block_whole_pixel`
    // consumes. For a whole-pixel input the fractional bits stay zero
    // after doubling so the fetch collapses to the §18.3 page 115
    // "subblock is simply copied" path.
    let mv_eighth = stored_luma_mv(mv);

    let mut pred = [0u8; 256];
    for sub_r in 0..4 {
        for sub_c in 0..4 {
            let blk_x = blk_x0 + sub_c * 4;
            let blk_y = blk_y0 + sub_r * 4;
            let patch = fetch_block_whole_pixel(
                reference.plane,
                reference.stride,
                reference.width,
                reference.height,
                blk_x,
                blk_y,
                mv_eighth,
            );
            // Place the 4×4 patch at the (sub_r, sub_c) slot of the
            // 16×16 prediction.
            for r in 0..4 {
                let dst_row = sub_r * 4 + r;
                let dst_col = sub_c * 4;
                pred[dst_row * 16 + dst_col..dst_row * 16 + dst_col + 4]
                    .copy_from_slice(&patch[r * 4..r * 4 + 4]);
            }
        }
    }

    block_sad_16x16(src_y, &pred)
}

/// Clamp a §17.1 candidate component into `[MV_MIN, MV_MAX]`.
#[inline]
fn clamp_component(v: i32) -> i16 {
    v.clamp(MV_MIN as i32, MV_MAX as i32) as i16
}

/// Round `v` toward zero to the nearest whole-pixel multiple of
/// [`WHOLE_PIXEL_STEP`]. Used to coerce a caller-supplied center MV
/// onto the whole-pixel grid the search visits.
#[inline]
fn snap_to_whole_pixel(v: i16) -> i16 {
    (v / WHOLE_PIXEL_STEP) * WHOLE_PIXEL_STEP
}

/// Small-diamond integer-pixel motion search for a 16×16 luma block.
///
/// Iterates a 4-neighbour (N, S, W, E at ±1 whole pixel each) descent
/// from `center`, replacing the current best with whichever neighbour
/// produces a strictly smaller SAD. Terminates when no neighbour
/// improves the SAD or after `max_iters` iterations (whichever comes
/// first). The returned [`SearchResult::mv`] is always on the
/// whole-pixel grid and inside `[MV_MIN, MV_MAX]`; the returned `sad`
/// is its [`block_sad_16x16`] value.
///
/// `center` is taken as the §17.1 starting MV; its components are
/// snapped to the whole-pixel grid (toward zero) and clamped into
/// `[MV_MIN, MV_MAX]` before the search begins.
///
/// Pass `max_iters = 0` for a pure no-op probe (returns the SAD at the
/// snapped/clamped `center` with no neighbour exploration).
pub fn small_diamond_search_luma(
    reference: LumaRef<'_>,
    mb_col: usize,
    mb_row: usize,
    src_y: &[u8; 256],
    center: Mv,
    max_iters: u32,
) -> SearchResult {
    // Clamp first, *then* snap toward zero — snapping first can leave
    // the clamped result off the whole-pixel grid because §17.1's
    // boundary (±1023) is not a multiple of `WHOLE_PIXEL_STEP` (4).
    let mut best_mv = Mv {
        row: snap_to_whole_pixel(clamp_component(center.row as i32)),
        col: snap_to_whole_pixel(clamp_component(center.col as i32)),
    };
    let mut best_sad = mb_luma_sad_at_whole_mv(reference, mb_col, mb_row, src_y, best_mv);

    // 4-neighbour offsets (N, S, W, E) in §17 quarter-pixel units.
    let neighbours: [(i32, i32); 4] = [
        (-(WHOLE_PIXEL_STEP as i32), 0),
        (WHOLE_PIXEL_STEP as i32, 0),
        (0, -(WHOLE_PIXEL_STEP as i32)),
        (0, WHOLE_PIXEL_STEP as i32),
    ];

    for _ in 0..max_iters {
        let mut improved = false;
        for (drow, dcol) in neighbours {
            // Clamp first, then snap toward zero so the candidate
            // never drifts off the whole-pixel grid (§17.1's ±1023
            // boundary is not a multiple of WHOLE_PIXEL_STEP).
            let cand_row = snap_to_whole_pixel(clamp_component(best_mv.row as i32 + drow));
            let cand_col = snap_to_whole_pixel(clamp_component(best_mv.col as i32 + dcol));
            // Clamping can collapse a candidate back onto best_mv at
            // the §17.1 boundary; skip those to avoid recomputing the
            // identical SAD.
            if cand_row == best_mv.row && cand_col == best_mv.col {
                continue;
            }
            let cand_mv = Mv {
                row: cand_row,
                col: cand_col,
            };
            let cand_sad = mb_luma_sad_at_whole_mv(reference, mb_col, mb_row, src_y, cand_mv);
            if cand_sad < best_sad {
                best_sad = cand_sad;
                best_mv = cand_mv;
                improved = true;
            }
        }
        if !improved {
            // Local minimum on the small-diamond pattern; nothing
            // larger would dislodge it without a bigger neighbourhood.
            break;
        }
    }

    SearchResult {
        mv: best_mv,
        sad: best_sad,
    }
}

/// Compute the 16×16 luma SAD for one §17 quarter-pixel MV candidate
/// (potentially at half-pixel resolution).
///
/// Unlike [`mb_luma_sad_at_whole_mv`] this routine accepts any §17.1
/// MV whose components are multiples of [`HALF_PIXEL_STEP`] (i.e.
/// whole-pixel ∪ half-pixel). The §18.3 prediction synthesis runs the
/// same MB-batched path the encoder's reconstruct leg uses
/// ([`crate::motion_comp::predict_inter_mb`]'s luma half): all sixteen
/// luma sub-blocks of a non-SPLITMV candidate share the one MV (§18.1),
/// so a sub-pixel candidate is one [`fetch_luma_mb_halo`] 21×21 fetch +
/// one [`sixtap_mb_luma`] whole-MB convolution — byte-exact with
/// sixteen separate [`crate::motion_comp::filter_block_4x4`] /
/// [`crate::motion_comp::sixtap_2d`] calls (the per-sub-block tiling the
/// decoder runs), with the version=0 bicubic six-tap tap-set — the
/// encoder commits to `version == 0` in every emitted frame tag, so
/// this is the tap-set the decoder will reproduce on its sub-pixel MV
/// path. A candidate whose §18.1-doubled fractions are both zero
/// (whole-pixel) collapses to the §18.3 "simply copied" path, batched
/// as one [`fetch_luma_mb_whole_pixel`] contiguous 16×16 fetch.
///
/// The MV must be inside `[MV_MIN, MV_MAX]` per §17.1; debug builds
/// assert. Quarter-pixel positions (`mv.row & 1 != 0`, etc.) are not
/// rejected — the underlying §18.3 filter table is indexed by
/// `(stored_luma_mv(mv) & 7)` and supports all eight fractions — but
/// the [`half_pixel_refine_luma`] caller only ever produces half-pixel
/// candidates.
pub fn mb_luma_sad_at_mv(
    reference: LumaRef<'_>,
    mb_col: usize,
    mb_row: usize,
    src_y: &[u8; 256],
    mv: Mv,
) -> u32 {
    debug_assert!((MV_MIN..=MV_MAX).contains(&mv.row));
    debug_assert!((MV_MIN..=MV_MAX).contains(&mv.col));

    let blk_x0 = mb_col * 16;
    let blk_y0 = mb_row * 16;
    // §18.1 stored_luma_mv doubles the §17 quarter-pixel vector into
    // the §18 eighth-pixel resolution the §18.3 filter table indexes.
    let mv_eighth = stored_luma_mv(mv);
    // §20.14 setup_subpixel_filters: `version == 0` selects the bicubic
    // six-tap set. The encoder commits to `version == 0` in its emitted
    // frame tag, so the decoder will run the same tap-set when re-decoding
    // this candidate.
    let filters = filter_set_for_version(0).taps();

    // §18.3 fraction selectors (`mv & 7` on the eighth-pixel vector) —
    // the same gate `predict_inter_mb` uses to pick the batched path.
    let mx = (mv_eighth.col & 7) as usize;
    let my = (mv_eighth.row & 7) as usize;
    let pred = if mx == 0 && my == 0 {
        // Whole-pixel: the §18.3 prediction is "simply copied", and all
        // sixteen sub-blocks share the MV, so the whole 16×16 block is
        // one contiguous source region fetched in a single pass —
        // byte-identical to sixteen per-sub-block copy fetches
        // (`fetch_luma_mb_whole_pixel_matches_per_subblock_in_bounds`).
        fetch_luma_mb_whole_pixel(
            reference.plane,
            reference.stride,
            reference.width,
            reference.height,
            blk_x0,
            blk_y0,
            mv_eighth,
        )
    } else {
        // Sub-pixel: one shared 21×21 halo + one whole-MB §18.3 six-tap
        // pass — byte-exact with sixteen separate `filter_block_4x4` /
        // `sixtap_2d` calls (`predict_inter_mb_sub_pixel_matches_per_block`
        // and the `sixtap_mb_luma` equivalence tests), replacing sixteen
        // overlapping 9×9 halo fetches per candidate.
        let halo = fetch_luma_mb_halo(
            reference.plane,
            reference.stride,
            reference.width,
            reference.height,
            blk_x0,
            blk_y0,
            mv_eighth,
        );
        sixtap_mb_luma(&halo, mx, my, filters)
    };

    block_sad_16x16(src_y, &pred)
}

/// Half-pixel motion-search refinement around a whole-pixel center —
/// RFC 6386 §18.3 / §17.1.
///
/// Given a `whole_pixel_center` MV that came out of an integer-pixel
/// search such as [`small_diamond_search_luma`], probe the eight
/// half-pixel offsets ±[`HALF_PIXEL_STEP`] in each of the (row, col)
/// axes — i.e. the 3×3 grid `{(-1, -1), (-1, 0), (-1, +1), (0, -1),
/// (0, +1), (+1, -1), (+1, 0), (+1, +1)} * HALF_PIXEL_STEP` around the
/// center, excluding the center itself — and return the MV with the
/// smallest 16×16 luma SAD across {center, all 8 neighbours}.
///
/// Each half-pixel candidate is evaluated through [`mb_luma_sad_at_mv`],
/// which runs the §18.3 six-tap synthesis (the bicubic `version == 0`
/// tap-set the encoder always commits to in its frame tag). The center's
/// SAD is recomputed (whole-pixel ⇒ the §18.3 copy path) rather than
/// passed in as an argument so the function is independent of how the
/// caller derived the integer-pixel candidate.
///
/// `whole_pixel_center` is asserted to be on the whole-pixel grid
/// (components multiples of [`WHOLE_PIXEL_STEP`]) and inside §17.1's
/// `[MV_MIN, MV_MAX]`. Candidates outside §17.1's range are clamped
/// (snapped onto the nearest in-range value); a candidate that collapses
/// onto an already-evaluated MV after clamping is skipped to avoid
/// recomputing the same SAD. Ties between the center and a neighbour go
/// to the **center** — fewer §17.2 component bits to code (the magnitude
/// 2 differential adds at least one extra bit per component vs. the
/// magnitude-0 / multiple-of-4 differential the whole-pixel center
/// carries).
///
/// This is the smallest §18.3 refinement that gives a sub-pixel
/// translation a chance of self-decoding bit-exactly: pure-translation
/// content that lies on the half-pixel grid (e.g. a `+0.5` luma pixel
/// shift) is fundamentally unreachable from a whole-pixel-only descent.
pub fn half_pixel_refine_luma(
    reference: LumaRef<'_>,
    mb_col: usize,
    mb_row: usize,
    src_y: &[u8; 256],
    whole_pixel_center: Mv,
) -> SearchResult {
    debug_assert_eq!(
        whole_pixel_center.row % WHOLE_PIXEL_STEP,
        0,
        "whole_pixel_center.row must be on the whole-pixel grid"
    );
    debug_assert_eq!(
        whole_pixel_center.col % WHOLE_PIXEL_STEP,
        0,
        "whole_pixel_center.col must be on the whole-pixel grid"
    );
    debug_assert!((MV_MIN..=MV_MAX).contains(&whole_pixel_center.row));
    debug_assert!((MV_MIN..=MV_MAX).contains(&whole_pixel_center.col));

    let mut best_mv = whole_pixel_center;
    let mut best_sad = mb_luma_sad_at_mv(reference, mb_col, mb_row, src_y, best_mv);

    // 8 half-pixel offsets around the center, in §17 quarter-pixel units.
    let step = HALF_PIXEL_STEP as i32;
    let offsets: [(i32, i32); 8] = [
        (-step, -step),
        (-step, 0),
        (-step, step),
        (0, -step),
        (0, step),
        (step, -step),
        (step, 0),
        (step, step),
    ];

    for (drow, dcol) in offsets {
        let cand_row = clamp_component(whole_pixel_center.row as i32 + drow);
        let cand_col = clamp_component(whole_pixel_center.col as i32 + dcol);
        // After §17.1 clamping a half-pixel candidate at the boundary
        // can collapse back onto the center (e.g. row=1023 + 2 ⇒ 1023);
        // skip those to avoid recomputing the identical SAD.
        if cand_row == best_mv.row && cand_col == best_mv.col {
            continue;
        }
        let cand_mv = Mv {
            row: cand_row,
            col: cand_col,
        };
        let cand_sad = mb_luma_sad_at_mv(reference, mb_col, mb_row, src_y, cand_mv);
        // Tie goes to the previous best (which starts as the center,
        // so on equality the whole-pixel center wins — fewer §17.2
        // component bits to code).
        if cand_sad < best_sad {
            best_sad = cand_sad;
            best_mv = cand_mv;
        }
    }

    SearchResult {
        mv: best_mv,
        sad: best_sad,
    }
}

/// Quarter-pixel motion-search refinement around a half-pixel center —
/// RFC 6386 §18.3 / §17.1.
///
/// Given a `half_pixel_center` MV that came out of
/// [`half_pixel_refine_luma`] (components multiples of
/// [`HALF_PIXEL_STEP`]), probe the eight quarter-pixel offsets
/// ±[`QUARTER_PIXEL_STEP`] in each of the (row, col) axes — i.e. the
/// 3×3 grid around the center excluding the center itself — and return
/// the MV with the smallest 16×16 luma SAD across {center, all 8
/// neighbours}.
///
/// Each quarter-pixel candidate is evaluated through [`mb_luma_sad_at_mv`],
/// which runs the §18.3 six-tap synthesis (the bicubic `version == 0`
/// tap-set the encoder always commits to in its frame tag). After §18.1
/// doubling, a §17 quarter-pixel offset becomes the §18.3 eighth-pixel
/// fraction `2` (`1/4` tap row, `{ 2, -11, 108, 36, -8, 1 }`) or `6`
/// (`3/4` tap row, the reverse) depending on whether the center already
/// carried a half-pixel offset on that axis. The center's SAD is
/// recomputed (via the §18.3 sixtap, which collapses to the copy path
/// only when both fractions are zero) so the function is independent of
/// how the caller derived the half-pixel candidate.
///
/// `half_pixel_center` is asserted to be on the half-pixel grid
/// (components multiples of [`HALF_PIXEL_STEP`]) and inside §17.1's
/// `[MV_MIN, MV_MAX]`. Candidates outside §17.1's range are clamped
/// (snapped onto the nearest in-range value); a candidate that collapses
/// onto an already-evaluated MV after clamping is skipped to avoid
/// recomputing the same SAD. Ties between the center and a neighbour go
/// to the **center** — fewer §17.2 component bits to code (one extra
/// quarter-pixel offset on a component adds at least one §17.2 long-form
/// bit per component vs. the half- or whole-pixel center).
///
/// This is the smallest §18.3 refinement on top of [`half_pixel_refine_luma`]
/// that gives a pure-translation source landing on the quarter-pixel
/// grid (e.g. a `+0.25` luma-pixel shift) a chance of self-decoding
/// bit-exactly — content at that fractional offset is fundamentally
/// unreachable from a half-pixel-only descent.
pub fn quarter_pixel_refine_luma(
    reference: LumaRef<'_>,
    mb_col: usize,
    mb_row: usize,
    src_y: &[u8; 256],
    half_pixel_center: Mv,
) -> SearchResult {
    debug_assert_eq!(
        half_pixel_center.row % HALF_PIXEL_STEP,
        0,
        "half_pixel_center.row must be on the half-pixel grid"
    );
    debug_assert_eq!(
        half_pixel_center.col % HALF_PIXEL_STEP,
        0,
        "half_pixel_center.col must be on the half-pixel grid"
    );
    debug_assert!((MV_MIN..=MV_MAX).contains(&half_pixel_center.row));
    debug_assert!((MV_MIN..=MV_MAX).contains(&half_pixel_center.col));

    let mut best_mv = half_pixel_center;
    let mut best_sad = mb_luma_sad_at_mv(reference, mb_col, mb_row, src_y, best_mv);

    // 8 quarter-pixel offsets around the center, in §17 quarter-pixel units.
    let step = QUARTER_PIXEL_STEP as i32;
    let offsets: [(i32, i32); 8] = [
        (-step, -step),
        (-step, 0),
        (-step, step),
        (0, -step),
        (0, step),
        (step, -step),
        (step, 0),
        (step, step),
    ];

    for (drow, dcol) in offsets {
        let cand_row = clamp_component(half_pixel_center.row as i32 + drow);
        let cand_col = clamp_component(half_pixel_center.col as i32 + dcol);
        // After §17.1 clamping a quarter-pixel candidate at the boundary
        // can collapse back onto the center (e.g. row=1023 + 1 ⇒ 1023);
        // skip those to avoid recomputing the identical SAD.
        if cand_row == best_mv.row && cand_col == best_mv.col {
            continue;
        }
        let cand_mv = Mv {
            row: cand_row,
            col: cand_col,
        };
        let cand_sad = mb_luma_sad_at_mv(reference, mb_col, mb_row, src_y, cand_mv);
        // Tie goes to the previous best (which starts as the half-pixel
        // center, so on equality the lower-§17.2-bit candidate wins).
        if cand_sad < best_sad {
            best_sad = cand_sad;
            best_mv = cand_mv;
        }
    }

    SearchResult {
        mv: best_mv,
        sad: best_sad,
    }
}

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion_comp::filter_block_4x4;

    /// Render a luma plane sized for at least one 16×16 macroblock at
    /// `(mb_col, mb_row) = (0, 0)`, filled with a horizontal ramp
    /// `(x % 251)` so a horizontal MV shift produces a non-trivial SAD
    /// delta.
    fn ramp_plane(width: usize, height: usize) -> Vec<u8> {
        let mut p = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                p[y * width + x] = ((x + y * 3) % 251) as u8;
            }
        }
        p
    }

    /// Render a luma plane that is constant background `bg` with a
    /// single bright `fg`-valued square of size `feat` placed with its
    /// top-left corner at `(fx, fy)`. The square is the only feature,
    /// so the SAD as a function of MV has a unique global minimum at
    /// the MV that aligns the candidate patch's feature with the
    /// source patch's feature — no axis-asymmetric ramp gradient to
    /// distract a greedy descent into a local minimum.
    fn feature_plane(
        width: usize,
        height: usize,
        bg: u8,
        fg: u8,
        fx: usize,
        fy: usize,
        feat: usize,
    ) -> Vec<u8> {
        let mut p = vec![bg; width * height];
        for r in 0..feat {
            for c in 0..feat {
                let y = fy + r;
                let x = fx + c;
                if x < width && y < height {
                    p[y * width + x] = fg;
                }
            }
        }
        p
    }

    /// Extract the 16×16 block at `(mb_col, mb_row)` from a plane laid
    /// out with stride `stride`.
    fn extract_mb_block(plane: &[u8], stride: usize, mb_col: usize, mb_row: usize) -> [u8; 256] {
        let mut blk = [0u8; 256];
        let x0 = mb_col * 16;
        let y0 = mb_row * 16;
        for r in 0..16 {
            let src = (y0 + r) * stride + x0;
            blk[r * 16..r * 16 + 16].copy_from_slice(&plane[src..src + 16]);
        }
        blk
    }

    #[test]
    fn block_sad_identical_inputs_is_zero() {
        let mut a = [0u8; 256];
        for (i, slot) in a.iter_mut().enumerate() {
            *slot = (i % 251) as u8;
        }
        assert_eq!(block_sad_16x16(&a, &a), 0);
    }

    #[test]
    fn block_sad_one_pixel_delta_is_one() {
        let a = [10u8; 256];
        let mut b = a;
        b[42] = 11;
        assert_eq!(block_sad_16x16(&a, &b), 1);
    }

    #[test]
    fn block_sad_saturated_difference() {
        let a = [0u8; 256];
        let b = [255u8; 256];
        assert_eq!(block_sad_16x16(&a, &b), 256 * 255);
    }

    #[test]
    fn block_sad_known_manual_sum() {
        let mut a = [0u8; 256];
        let mut b = [0u8; 256];
        // First three pixels: |10-3| + |20-5| + |30-2| = 7 + 15 + 28 = 50.
        a[0] = 10;
        b[0] = 3;
        a[1] = 20;
        b[1] = 5;
        a[2] = 30;
        b[2] = 2;
        assert_eq!(block_sad_16x16(&a, &b), 50);
    }

    /// Convenience constructor: bundle a plane + its dimensions into a
    /// [`LumaRef`] for the test's `(stride == width)` packed layout.
    fn packed_luma_ref<'a>(plane: &'a [u8], width: usize, height: usize) -> LumaRef<'a> {
        LumaRef {
            plane,
            stride: width,
            width,
            height,
        }
    }

    #[test]
    fn sad_at_zero_mv_equals_direct_block_sad() {
        // Two distinct planes: ref is the ramp, src is the same ramp
        // with one pixel offset. SAD at (0,0) MV should equal
        // block_sad_16x16(ref_block_at_origin, src_block_at_origin).
        let width = 32;
        let height = 32;
        let reference = ramp_plane(width, height);
        let mut src_plane = reference.clone();
        // Bump every fourth pixel of the MB(0,0) by 5.
        for y in 0..16 {
            for x in (0..16).step_by(4) {
                src_plane[y * width + x] = src_plane[y * width + x].wrapping_add(5);
            }
        }
        let src_blk = extract_mb_block(&src_plane, width, 0, 0);
        let ref_blk = extract_mb_block(&reference, width, 0, 0);
        let expected = block_sad_16x16(&src_blk, &ref_blk);
        let got = mb_luma_sad_at_whole_mv(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv { row: 0, col: 0 },
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn diamond_search_finds_exact_translation_horizontal() {
        // Source MB(0,0) covers pixels (0..16, 0..16). Place the
        // source's feature inside the MB at (sx=4, sy=4) and the
        // reference's feature shifted right by 2 whole pixels — at
        // (sx=6, sy=4). Aligning the candidate patch's feature with
        // the source's feature requires MV (row=0, col=+8 quarter-
        // pixels = +2 whole pixels). The small-diamond descent picks
        // it up in two iterations.
        let width = 64;
        let height = 32;
        let reference = feature_plane(width, height, 128, 240, 6, 4, 4);
        let source_plane = feature_plane(width, height, 128, 240, 4, 4, 4);
        let src_blk = extract_mb_block(&source_plane, width, 0, 0);
        let result = small_diamond_search_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv { row: 0, col: 0 },
            32,
        );
        assert_eq!(
            result.mv,
            Mv {
                row: 0,
                col: 2 * WHOLE_PIXEL_STEP,
            },
            "search must find the 2-whole-pixel horizontal offset"
        );
        assert_eq!(result.sad, 0, "exact translation ⇒ zero SAD");
    }

    #[test]
    fn diamond_search_finds_exact_translation_diagonal() {
        // Source feature inside MB(0,0) at (sx=5, sy=6); reference
        // feature shifted (+3 col, +2 row) at (sx=8, sy=8). Aligning
        // needs MV (row=+8 quarter-pixels = +2 whole-pixel down,
        //          col=+12 quarter-pixels = +3 whole-pixel right).
        let width = 64;
        let height = 64;
        let reference = feature_plane(width, height, 100, 220, 8, 8, 5);
        let source_plane = feature_plane(width, height, 100, 220, 5, 6, 5);
        let src_blk = extract_mb_block(&source_plane, width, 0, 0);
        let result = small_diamond_search_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv { row: 0, col: 0 },
            64,
        );
        assert_eq!(
            result.mv,
            Mv {
                row: 2 * WHOLE_PIXEL_STEP,
                col: 3 * WHOLE_PIXEL_STEP,
            }
        );
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn diamond_search_zero_iters_returns_center_sad() {
        // max_iters = 0 ⇒ no neighbour exploration, just the SAD at
        // the snapped/clamped center.
        let width = 32;
        let height = 32;
        let reference = ramp_plane(width, height);
        let src_blk = extract_mb_block(&reference, width, 0, 0);
        let result = small_diamond_search_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv { row: 0, col: 0 },
            0,
        );
        assert_eq!(result.mv, Mv { row: 0, col: 0 });
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn diamond_search_identical_source_returns_zero_mv() {
        // Source = the same MB the encoder is positioned at, with the
        // reference being the same plane: the (0, 0) MV is already the
        // global optimum; no neighbour can improve it.
        let width = 32;
        let height = 32;
        let reference = ramp_plane(width, height);
        let src_blk = extract_mb_block(&reference, width, 0, 0);
        let result = small_diamond_search_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv { row: 0, col: 0 },
            8,
        );
        assert_eq!(result.mv, Mv { row: 0, col: 0 });
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn diamond_search_snaps_center_to_whole_pixel() {
        // Caller hands in a sub-pixel center (col = 5 quarter-pixels =
        // 1.25 whole pixels). The search snaps to col = 4 toward zero
        // before starting; on a flat source/reference the result is
        // still SAD 0 (flat ⇒ all MVs equivalent).
        let width = 32;
        let height = 32;
        let reference = vec![200u8; width * height];
        let src_blk = [200u8; 256];
        let result = small_diamond_search_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv { row: 1, col: 5 },
            0,
        );
        // Snap toward zero ⇒ row=0, col=4 — both whole-pixel grid.
        assert_eq!(result.mv.row % WHOLE_PIXEL_STEP, 0);
        assert_eq!(result.mv.col % WHOLE_PIXEL_STEP, 0);
        assert_eq!(result.mv, Mv { row: 0, col: 4 });
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn diamond_search_clamps_extreme_center_to_section_17_1_range() {
        // §17.1: component must be in [-1023, +1023]. A center at i16::MAX
        // must be clamped down to 1020 (largest multiple of 4 inside +1023),
        // and the search must not panic from an out-of-range fetch.
        let width = 32;
        let height = 32;
        let reference = vec![128u8; width * height];
        let src_blk = [128u8; 256];
        let result = small_diamond_search_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv {
                row: i16::MAX,
                col: i16::MIN,
            },
            4,
        );
        assert!((MV_MIN..=MV_MAX).contains(&result.mv.row));
        assert!((MV_MIN..=MV_MAX).contains(&result.mv.col));
        assert_eq!(result.mv.row % WHOLE_PIXEL_STEP, 0);
        assert_eq!(result.mv.col % WHOLE_PIXEL_STEP, 0);
        // Flat source ⇒ zero SAD at every legal MV.
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn diamond_search_off_picture_edge_replicate_does_not_panic() {
        // Anchor the search at MB(0, 0) (top-left), with a candidate
        // MV that would walk past the top-left edge. The §20.14
        // build_mc_border edge-replicate inside fetch_block_whole_pixel
        // must absorb the out-of-plane fetch; the search itself
        // remains well-defined and returns a non-panicking result.
        let width = 32;
        let height = 32;
        let reference = ramp_plane(width, height);
        let src_blk = extract_mb_block(&reference, width, 0, 0);
        let result = small_diamond_search_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv { row: -64, col: -64 },
            8,
        );
        // Just confirming no panic and the result is on the grid + in range.
        assert_eq!(result.mv.row % WHOLE_PIXEL_STEP, 0);
        assert_eq!(result.mv.col % WHOLE_PIXEL_STEP, 0);
        assert!((MV_MIN..=MV_MAX).contains(&result.mv.row));
        assert!((MV_MIN..=MV_MAX).contains(&result.mv.col));
    }

    #[test]
    fn diamond_search_never_increases_sad_from_center() {
        // The search must never *worsen* the candidate it returned vs
        // the SAD at the snapped center; descent semantics require
        // result.sad <= sad(snapped_center). Use a single-feature
        // landscape so the SAD topology has one global minimum and the
        // greedy small-diamond reaches it from a nearby center.
        let width = 48;
        let height = 48;
        // Source feature inside MB(0,0) at (sx=4, sy=6), reference
        // feature shifted to (sx=6, sy=6) — convergence MV
        // (row=0, col=+2 whole-pixels = +8 quarter-pixels).
        let reference = feature_plane(width, height, 90, 230, 6, 6, 5);
        let source_plane = feature_plane(width, height, 90, 230, 4, 6, 5);
        let src_blk = extract_mb_block(&source_plane, width, 0, 0);
        let ref_luma = packed_luma_ref(&reference, width, height);
        let sad_at_center =
            mb_luma_sad_at_whole_mv(ref_luma, 0, 0, &src_blk, Mv { row: 0, col: 0 });
        let result = small_diamond_search_luma(ref_luma, 0, 0, &src_blk, Mv { row: 0, col: 0 }, 8);
        assert!(
            result.sad <= sad_at_center,
            "descent must not increase SAD (center={sad_at_center}, result={})",
            result.sad
        );
        // The exact optimum lives at col = 2 whole pixels = 8 in §17 units.
        assert_eq!(result.mv, Mv { row: 0, col: 8 });
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn half_pixel_refine_keeps_center_on_flat_source() {
        // Flat source ⇒ SAD is zero at every MV ⇒ tie-break keeps the
        // whole-pixel center (fewer §17.2 component bits).
        let width = 32;
        let height = 32;
        let reference = vec![200u8; width * height];
        let src_blk = [200u8; 256];
        let result = half_pixel_refine_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv { row: 0, col: 0 },
        );
        assert_eq!(result.mv, Mv { row: 0, col: 0 });
        assert_eq!(result.sad, 0);
    }

    /// Render a luma plane with a non-degenerate 2D ramp: distinct row
    /// and column gradients with different slopes. Used by the
    /// half-pixel refinement test so the §18.3 filter at every half-
    /// pixel position produces a distinct prediction (no row/column
    /// invariance lets a diagonal half-pixel candidate tie a cardinal).
    fn two_d_ramp_plane(width: usize, height: usize) -> Vec<u8> {
        let mut p = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                // Different per-axis slopes so the sixtap response at
                // each (dx, dy) half-pixel offset is distinct.
                let v = ((x as i32) + 3 * (y as i32)).clamp(0, 240);
                p[y * width + x] = v as u8;
            }
        }
        p
    }

    #[test]
    fn half_pixel_refine_finds_half_pixel_horizontal_shift() {
        // Build a synthetic source whose MB(1, 1) block is EXACTLY the
        // §18.3 sixtap synthesis of the reference at MV (0, +1/2 px) so
        // there is one and only one half-pixel candidate that drives the
        // SAD to zero (the two_d_ramp plane has a distinct response at
        // every half-pixel position, breaking row-invariance ties).
        let width = 64;
        let height = 64;
        let reference = two_d_ramp_plane(width, height);
        // Construct src_blk = sixtap_2d(reference at MB(1,0), col-half).
        // That is: take the reference and apply mb_luma_sad_at_mv's own
        // prediction path with the ground-truth MV — the resulting block
        // is by construction the SAD-zero source for MV (0, HALF_PIXEL_STEP).
        let ref_luma = packed_luma_ref(&reference, width, height);
        // Synthesize the source MB by re-using the §18.3 prediction at
        // the ground-truth half-pixel MV, then verify the descent finds
        // it.
        let truth_mv = Mv {
            row: 0,
            col: HALF_PIXEL_STEP,
        };
        // Build the source block by running the same per-sub-block
        // filter the SAD evaluator uses internally — sample-exact mirror
        // of mb_luma_sad_at_mv's prediction (so SAD == 0 at truth_mv).
        let mut src_blk = [0u8; 256];
        let mv_eighth = stored_luma_mv(truth_mv);
        let filters = filter_set_for_version(0).taps();
        let blk_x0 = 16;
        let blk_y0 = 16;
        for sub_r in 0..4 {
            for sub_c in 0..4 {
                let blk_x = blk_x0 + sub_c * 4;
                let blk_y = blk_y0 + sub_r * 4;
                let patch = filter_block_4x4(
                    &reference, width, width, height, blk_x, blk_y, mv_eighth, filters,
                );
                for r in 0..4 {
                    let dst_row = sub_r * 4 + r;
                    let dst_col = sub_c * 4;
                    src_blk[dst_row * 16 + dst_col..dst_row * 16 + dst_col + 4]
                        .copy_from_slice(&patch[r * 4..r * 4 + 4]);
                }
            }
        }
        let result = half_pixel_refine_luma(ref_luma, 1, 1, &src_blk, Mv { row: 0, col: 0 });
        assert_eq!(
            result.mv, truth_mv,
            "half-pixel +0.5 px horizontal shift expected, got {:?}",
            result.mv
        );
        // The source equals the §18.3 prediction at truth_mv by
        // construction ⇒ SAD must be exactly zero at the picked MV.
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn half_pixel_refine_never_worsens_sad() {
        // The refinement is descent semantics: the returned SAD must be
        // ≤ the SAD at the whole-pixel center. Use a randomish-but-
        // deterministic plane so any half-pixel candidate has a chance
        // of beating the center.
        let width = 48;
        let height = 48;
        let reference = ramp_plane(width, height);
        // Source = ramp + low-amplitude horizontal phase shift built by
        // averaging neighbours (a manual half-pixel filter is overkill;
        // any non-trivial source vs. ref will exercise the descent).
        let mut src_plane = reference.clone();
        for y in 0..height {
            for x in 1..(width - 1) {
                let a = reference[y * width + x - 1] as u32;
                let b = reference[y * width + x] as u32;
                src_plane[y * width + x] = ((a + b) / 2) as u8;
            }
        }
        let src_blk = extract_mb_block(&src_plane, width, 1, 1);
        let ref_luma = packed_luma_ref(&reference, width, height);
        let sad_at_center = mb_luma_sad_at_mv(ref_luma, 1, 1, &src_blk, Mv { row: 0, col: 0 });
        let result = half_pixel_refine_luma(ref_luma, 1, 1, &src_blk, Mv { row: 0, col: 0 });
        assert!(
            result.sad <= sad_at_center,
            "half-pixel refinement must not worsen SAD (center={sad_at_center}, result={})",
            result.sad
        );
        // The chosen MV must be a half-pixel multiple of HALF_PIXEL_STEP
        // (i.e. ±0 or ±HALF_PIXEL_STEP on each component).
        assert!(result.mv.row.unsigned_abs() <= HALF_PIXEL_STEP as u16);
        assert!(result.mv.col.unsigned_abs() <= HALF_PIXEL_STEP as u16);
        assert_eq!(result.mv.row % HALF_PIXEL_STEP, 0);
        assert_eq!(result.mv.col % HALF_PIXEL_STEP, 0);
    }

    #[test]
    fn half_pixel_refine_clamps_at_section_17_1_boundary() {
        // Center at the §17.1 boundary (row=+1020 = 255 whole pixels);
        // half-pixel candidates that would step past the boundary
        // (col = MV_MAX = 1023 plus HALF_PIXEL_STEP) get clamped back
        // onto an already-evaluated MV and skipped. The result must
        // still be on the half-pixel grid and inside §17.1's range.
        let width = 32;
        let height = 32;
        let reference = vec![100u8; width * height];
        let src_blk = [100u8; 256];
        let result = half_pixel_refine_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv {
                row: 1020,
                col: 1020,
            },
        );
        assert!((MV_MIN..=MV_MAX).contains(&result.mv.row));
        assert!((MV_MIN..=MV_MAX).contains(&result.mv.col));
        // Flat ⇒ tie-break keeps the center.
        assert_eq!(
            result.mv,
            Mv {
                row: 1020,
                col: 1020
            }
        );
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn mb_luma_sad_at_mv_whole_pixel_matches_whole_pixel_helper() {
        // For a whole-pixel MV, mb_luma_sad_at_mv must equal
        // mb_luma_sad_at_whole_mv (the §18.3 sixtap collapses to the
        // copy path when the fraction is zero, so the two evaluators
        // produce the same prediction).
        let width = 32;
        let height = 32;
        let reference = ramp_plane(width, height);
        let mut src_plane = reference.clone();
        // Perturb the source so the SAD isn't trivially zero.
        for y in 0..height {
            src_plane[y * width + (y % width)] = src_plane[y * width + (y % width)].wrapping_add(7);
        }
        let src_blk = extract_mb_block(&src_plane, width, 0, 0);
        let ref_luma = packed_luma_ref(&reference, width, height);
        let mv = Mv { row: 0, col: 0 };
        let sad_general = mb_luma_sad_at_mv(ref_luma, 0, 0, &src_blk, mv);
        let sad_whole = mb_luma_sad_at_whole_mv(ref_luma, 0, 0, &src_blk, mv);
        assert_eq!(sad_general, sad_whole);
    }

    #[test]
    fn quarter_pixel_refine_keeps_center_on_flat_source() {
        // Flat source ⇒ SAD is zero at every MV ⇒ tie-break keeps the
        // half-pixel center (fewer §17.2 component bits than a quarter-
        // pixel offset on the same axis).
        let width = 32;
        let height = 32;
        let reference = vec![200u8; width * height];
        let src_blk = [200u8; 256];
        let result = quarter_pixel_refine_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv {
                row: 0,
                col: HALF_PIXEL_STEP,
            },
        );
        assert_eq!(
            result.mv,
            Mv {
                row: 0,
                col: HALF_PIXEL_STEP,
            }
        );
        assert_eq!(result.sad, 0);
    }

    /// Render a luma plane with sharp luminance steps superimposed on a
    /// gentle ramp — designed so the §18.3 sixtap response is distinct
    /// at every fractional offset (the ramp alone is linear and the
    /// sixtap collapses to identity rounding on it, so every fractional
    /// MV yields the same byte pattern; the steps break that
    /// degeneracy).
    fn stepped_plane(width: usize, height: usize) -> Vec<u8> {
        let mut p = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                // Base ramp + per-column step pattern (every 3rd column
                // takes a +40 luma bump) + per-row stripe (every 5th
                // row takes a -20 luma dip). High-frequency content
                // makes the §18.3 sixtap distinguish quarter-pixel
                // offsets from half-pixel and whole-pixel offsets.
                let base = ((x as i32) + 2 * (y as i32)).clamp(0, 180);
                let step = if x % 3 == 0 { 40 } else { 0 };
                let stripe = if y % 5 == 0 { -20 } else { 0 };
                let v = (base + step + stripe).clamp(0, 240);
                p[y * width + x] = v as u8;
            }
        }
        p
    }

    #[test]
    fn quarter_pixel_refine_finds_quarter_pixel_horizontal_shift() {
        // Build a synthetic source whose MB(1, 1) block is EXACTLY the
        // §18.3 sixtap synthesis of the reference at MV (0, +1/4 px) so
        // there is one and only one quarter-pixel candidate that drives
        // the SAD to zero. The stepped_plane carries high-frequency
        // content the sixtap distinguishes at every fractional offset
        // (a purely linear plane like two_d_ramp gives identical bytes
        // for every quarter/half-pixel MV in `u8` arithmetic).
        let width = 64;
        let height = 64;
        let reference = stepped_plane(width, height);
        let ref_luma = packed_luma_ref(&reference, width, height);
        let truth_mv = Mv {
            row: 0,
            col: QUARTER_PIXEL_STEP,
        };
        // Source MB = sample-exact mirror of mb_luma_sad_at_mv's
        // prediction at truth_mv (so SAD == 0 at truth_mv).
        let mut src_blk = [0u8; 256];
        let mv_eighth = stored_luma_mv(truth_mv);
        let filters = filter_set_for_version(0).taps();
        let blk_x0 = 16;
        let blk_y0 = 16;
        for sub_r in 0..4 {
            for sub_c in 0..4 {
                let blk_x = blk_x0 + sub_c * 4;
                let blk_y = blk_y0 + sub_r * 4;
                let patch = filter_block_4x4(
                    &reference, width, width, height, blk_x, blk_y, mv_eighth, filters,
                );
                for r in 0..4 {
                    let dst_row = sub_r * 4 + r;
                    let dst_col = sub_c * 4;
                    src_blk[dst_row * 16 + dst_col..dst_row * 16 + dst_col + 4]
                        .copy_from_slice(&patch[r * 4..r * 4 + 4]);
                }
            }
        }
        // The driver starts at the whole-pixel center (0, 0), runs the
        // half-pixel refinement first, then the quarter-pixel refinement.
        // From the whole-pixel center the half-pixel refinement cannot
        // reach the +1/4 px source (the half-pixel grid is a strict
        // subset of the quarter-pixel grid), so the half-pixel result
        // lands at one of {(0, 0), (0, +1/2)}; the quarter-pixel
        // refinement around that result reaches (0, +1/4) via either a
        // (0, +1) step from (0, 0) or a (0, -1) step from (0, +1/2).
        let half = half_pixel_refine_luma(ref_luma, 1, 1, &src_blk, Mv { row: 0, col: 0 });
        let result = quarter_pixel_refine_luma(ref_luma, 1, 1, &src_blk, half.mv);
        assert_eq!(
            result.mv, truth_mv,
            "quarter-pixel +1/4 px horizontal shift expected, got {:?}",
            result.mv
        );
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn quarter_pixel_refine_never_worsens_sad() {
        // The refinement is descent semantics: the returned SAD must be
        // ≤ the SAD at the half-pixel center.
        let width = 48;
        let height = 48;
        let reference = ramp_plane(width, height);
        // Synthesize a source with a slight fractional phase shift built
        // by a manual triangle-tap (a/4 + b/2 + c/4) so the SAD topology
        // has non-trivial quarter-pixel response.
        let mut src_plane = reference.clone();
        for y in 0..height {
            for x in 1..(width - 1) {
                let a = reference[y * width + x - 1] as u32;
                let b = reference[y * width + x] as u32;
                let c = reference[y * width + x + 1] as u32;
                src_plane[y * width + x] = ((a + 2 * b + c) / 4) as u8;
            }
        }
        let src_blk = extract_mb_block(&src_plane, width, 1, 1);
        let ref_luma = packed_luma_ref(&reference, width, height);
        let half_center = Mv {
            row: 0,
            col: HALF_PIXEL_STEP,
        };
        let sad_at_center = mb_luma_sad_at_mv(ref_luma, 1, 1, &src_blk, half_center);
        let result = quarter_pixel_refine_luma(ref_luma, 1, 1, &src_blk, half_center);
        assert!(
            result.sad <= sad_at_center,
            "quarter-pixel refinement must not worsen SAD (center={sad_at_center}, result={})",
            result.sad
        );
        // The chosen MV must stay within ±QUARTER_PIXEL_STEP of the
        // half-pixel center on each component — the refinement's
        // neighbourhood is the 3×3 quarter-pixel grid. (The §17 quarter-
        // pixel grid invariant is upheld by `Mv`'s i16 components by
        // construction, since QUARTER_PIXEL_STEP == 1.)
        let drow = result.mv.row - half_center.row;
        let dcol = result.mv.col - half_center.col;
        assert!(drow.abs() <= QUARTER_PIXEL_STEP);
        assert!(dcol.abs() <= QUARTER_PIXEL_STEP);
    }

    #[test]
    fn quarter_pixel_refine_clamps_at_section_17_1_boundary() {
        // Center at the §17.1 boundary (row=col=+1022 = a half-pixel-grid
        // MV near MV_MAX); quarter-pixel candidates that would step past
        // +1023 get clamped back onto an already-evaluated MV and
        // skipped. The result must stay on the quarter-pixel grid and
        // inside §17.1's range.
        let width = 32;
        let height = 32;
        let reference = vec![100u8; width * height];
        let src_blk = [100u8; 256];
        let result = quarter_pixel_refine_luma(
            packed_luma_ref(&reference, width, height),
            0,
            0,
            &src_blk,
            Mv {
                row: 1022,
                col: 1022,
            },
        );
        assert!((MV_MIN..=MV_MAX).contains(&result.mv.row));
        assert!((MV_MIN..=MV_MAX).contains(&result.mv.col));
        // The §17 quarter-pixel grid invariant is upheld by `Mv`'s
        // i16 components by construction (QUARTER_PIXEL_STEP == 1).
        // Flat ⇒ tie-break keeps the center.
        assert_eq!(
            result.mv,
            Mv {
                row: 1022,
                col: 1022,
            }
        );
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn quarter_pixel_refine_at_whole_pixel_center_equals_half_pixel_refine() {
        // When the half-pixel center is a whole-pixel MV (both fractions
        // zero), the quarter-pixel refinement explores the 8 ±1-§17-unit
        // offsets around it. The result must still be a non-worsening
        // refinement and live on the quarter-pixel grid.
        let width = 48;
        let height = 48;
        let reference = ramp_plane(width, height);
        let src_blk = extract_mb_block(&reference, width, 1, 1);
        let ref_luma = packed_luma_ref(&reference, width, height);
        let whole_center = Mv { row: 0, col: 0 };
        let sad_at_center = mb_luma_sad_at_mv(ref_luma, 1, 1, &src_blk, whole_center);
        let result = quarter_pixel_refine_luma(ref_luma, 1, 1, &src_blk, whole_center);
        assert!(result.sad <= sad_at_center);
        // Identical source/ref at MV (0,0) ⇒ tie-break keeps the center.
        assert_eq!(result.mv, whole_center);
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn search_result_is_copy_eq() {
        // The struct is small enough to be Copy + Eq + Debug — pin the
        // contract so a future field addition has to think about it.
        let a = SearchResult {
            mv: Mv { row: 4, col: -8 },
            sad: 1234,
        };
        let b = a;
        assert_eq!(a, b);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    /// A 21-input stress set for the §17 SAD primitive. Each entry is a
    /// `(src, pred)` pair shaped to walk the corners of the
    /// per-lane absolute-difference behaviour (`max - min` direction,
    /// row-wide saturation, alternating signs, sparse vs dense
    /// differences, full-zero / full-saturated extremes). The set is
    /// shared between the scalar-equivalence test below and any future
    /// SIMD path that wants the same coverage.
    ///
    /// The numbered comment + `push` sequence is intentional: each
    /// entry's purpose is annotated inline; collapsing the body into
    /// a `vec![…]` literal with 21 multi-line tuples obscures the
    /// intent.
    #[allow(clippy::vec_init_then_push)]
    fn sad_stress_pairs() -> Vec<([u8; 256], [u8; 256])> {
        let mut pairs: Vec<([u8; 256], [u8; 256])> = Vec::new();

        // 1. Both blocks identically zero.
        pairs.push(([0u8; 256], [0u8; 256]));
        // 2. Both blocks identically saturated.
        pairs.push(([255u8; 256], [255u8; 256]));
        // 3. Maximum positive difference everywhere.
        pairs.push(([0u8; 256], [255u8; 256]));
        // 4. Maximum negative difference everywhere (tests the
        //    `max(s, p) - min(s, p)` direction symmetry).
        pairs.push(([255u8; 256], [0u8; 256]));
        // 5. Single-pixel ±1 perturbation deep inside the block.
        let mut s5 = [10u8; 256];
        let mut p5 = [10u8; 256];
        s5[133] = 11;
        p5[133] = 10;
        pairs.push((s5, p5));
        // 6. Alternating-column delta — exercises lane-position
        //    dependence inside a 16-wide row vector.
        let mut s6 = [0u8; 256];
        let mut p6 = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                if c % 2 == 0 {
                    s6[r * 16 + c] = 200;
                    p6[r * 16 + c] = 50;
                } else {
                    s6[r * 16 + c] = 70;
                    p6[r * 16 + c] = 130;
                }
            }
        }
        pairs.push((s6, p6));
        // 7. Alternating-row delta — same as #6 but rotated 90° so the
        //    row stencil sees uniform rows alternating between extremes.
        let mut s7 = [0u8; 256];
        let mut p7 = [0u8; 256];
        for r in 0..16 {
            let (sv, pv) = if r % 2 == 0 { (240, 16) } else { (32, 192) };
            for c in 0..16 {
                s7[r * 16 + c] = sv;
                p7[r * 16 + c] = pv;
            }
        }
        pairs.push((s7, p7));
        // 8. Ramp src vs constant pred (gradient difference per lane).
        let mut s8 = [0u8; 256];
        for (i, slot) in s8.iter_mut().enumerate() {
            *slot = (i & 0xff) as u8;
        }
        pairs.push((s8, [128u8; 256]));
        // 9. Ramp pred vs constant src (gradient difference, flipped).
        pairs.push(([128u8; 256], s8));
        // 10. Mixed-frequency pattern src vs ramp pred — the same
        //     gradient shape `motion_search_descent.rs` uses.
        let mut s10 = [0u8; 256];
        let mut p10 = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                let mix = (r as u32 * 7).wrapping_add(c as u32 * 11).wrapping_add(13)
                    ^ (r as u32 * c as u32);
                s10[r * 16 + c] = mix as u8;
                p10[r * 16 + c] = ((r as u32 * 16 + c as u32) & 0xff) as u8;
            }
        }
        pairs.push((s10, p10));
        // 11. Sparse single-row delta (only row 0 differs).
        let mut s11 = [42u8; 256];
        let p11 = [42u8; 256];
        for (c, slot) in s11.iter_mut().take(16).enumerate() {
            *slot = (c * 17) as u8;
        }
        pairs.push((s11, p11));
        // 12. Sparse single-column delta (only column 7 differs).
        let mut s12 = [42u8; 256];
        let p12 = [42u8; 256];
        for r in 0..16 {
            s12[r * 16 + 7] = (r * 17) as u8;
        }
        pairs.push((s12, p12));
        // 13. Checkerboard difference pattern — alternating sign every
        //     pixel.
        let mut s13 = [0u8; 256];
        let mut p13 = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                if (r + c) & 1 == 0 {
                    s13[r * 16 + c] = 240;
                    p13[r * 16 + c] = 16;
                } else {
                    s13[r * 16 + c] = 16;
                    p13[r * 16 + c] = 240;
                }
            }
        }
        pairs.push((s13, p13));
        // 14. Both blocks pseudo-random but bounded so per-lane absdiff
        //     sometimes saturates and sometimes does not.
        let mut s14 = [0u8; 256];
        let mut p14 = [0u8; 256];
        for i in 0..256 {
            s14[i] = ((i.wrapping_mul(101)) ^ 0x5a) as u8;
            p14[i] = ((i.wrapping_mul(53)) ^ 0xa5) as u8;
        }
        pairs.push((s14, p14));
        // 15. Both blocks pseudo-random, different seeds.
        let mut s15 = [0u8; 256];
        let mut p15 = [0u8; 256];
        for i in 0..256 {
            s15[i] = ((i.wrapping_mul(7) ^ (i >> 3)) & 0xff) as u8;
            p15[i] = ((i.wrapping_mul(11) ^ (i >> 2)) & 0xff) as u8;
        }
        pairs.push((s15, p15));
        // 16. Per-lane envelope check: every row contributes a 16 ×
        //     255 saturation — pins the per-lane `u16` accumulator at
        //     the 16 × 255 = 4_080 maximum for every lane simultaneously.
        let mut s16a = [0u8; 256];
        let mut p16a = [255u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                if (r + c) & 1 == 0 {
                    s16a[r * 16 + c] = 0;
                    p16a[r * 16 + c] = 255;
                } else {
                    s16a[r * 16 + c] = 255;
                    p16a[r * 16 + c] = 0;
                }
            }
        }
        pairs.push((s16a, p16a));
        // 17. Half-block split: top 8 rows zero, bottom 8 rows saturated.
        let mut s17 = [0u8; 256];
        let mut p17 = [0u8; 256];
        for r in 8..16 {
            for c in 0..16 {
                s17[r * 16 + c] = 200;
                p17[r * 16 + c] = 40;
            }
        }
        pairs.push((s17, p17));
        // 18. Vertical-stripe split: left half src high, right half
        //     pred high.
        let mut s18 = [0u8; 256];
        let mut p18 = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                if c < 8 {
                    s18[r * 16 + c] = 220;
                    p18[r * 16 + c] = 20;
                } else {
                    s18[r * 16 + c] = 20;
                    p18[r * 16 + c] = 220;
                }
            }
        }
        pairs.push((s18, p18));
        // 19. Tiny perturbation (≤ 3) over the whole block — bottom-end
        //     of the per-lane envelope.
        let mut s19 = [0u8; 256];
        let mut p19 = [0u8; 256];
        for i in 0..256 {
            s19[i] = 100;
            p19[i] = 100 + ((i % 4) as u8);
        }
        pairs.push((s19, p19));
        // 20. Equal-magnitude positive/negative differences interleaved
        //     so the absolute-value reduction is exercised on both sign
        //     branches in a single row.
        let mut s20 = [0u8; 256];
        let mut p20 = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                let v = (r * 16 + c) as i32;
                let centred = 128 + ((v % 41) - 20);
                let other = 128 - ((v % 41) - 20);
                s20[r * 16 + c] = centred.clamp(0, 255) as u8;
                p20[r * 16 + c] = other.clamp(0, 255) as u8;
            }
        }
        pairs.push((s20, p20));
        // 21. Ramp-vs-ramp with a deliberate constant offset, so every
        //     per-lane absdiff lands on the same non-zero value.
        let mut s21 = [0u8; 256];
        let mut p21 = [0u8; 256];
        for i in 0..256 {
            s21[i] = (i & 0x7f) as u8;
            p21[i] = ((i & 0x7f) + 17).min(255) as u8;
        }
        pairs.push((s21, p21));

        pairs
    }

    #[test]
    fn block_sad_public_dispatch_matches_scalar_on_stress_inputs() {
        // Standing equivalence proof: the public `block_sad_16x16`
        // dispatcher must agree with the scalar listing on every input
        // in `sad_stress_pairs()`. On stable + default features this
        // is `scalar == scalar` (trivially true); on nightly + `simd`
        // the dispatcher routes through `block_sad_16x16_simd`, so the
        // assertion becomes the SIMD-vs-scalar bit-equivalence proof.
        for (idx, (src, pred)) in sad_stress_pairs().iter().enumerate() {
            let via_public = block_sad_16x16(src, pred);
            let via_scalar = block_sad_16x16_scalar(src, pred);
            assert_eq!(
                via_public, via_scalar,
                "pair #{idx}: public dispatch ≠ scalar listing"
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn block_sad_simd_matches_scalar_on_stress_inputs() {
        // Direct SIMD-vs-scalar bit-equivalence proof on the stress
        // set, regardless of which path the public dispatcher routes
        // to. Mirrors `wht_simd_matches_scalar_on_stress_inputs`
        // / `dct_simd_matches_scalar_on_stress_inputs` in
        // `src/inverse_transform.rs`.
        for (idx, (src, pred)) in sad_stress_pairs().iter().enumerate() {
            let via_simd = block_sad_16x16_simd(src, pred);
            let via_scalar = block_sad_16x16_scalar(src, pred);
            assert_eq!(via_simd, via_scalar, "pair #{idx}: SIMD ≠ scalar listing");
        }
    }
}
