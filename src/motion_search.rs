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
//! ## Scope (this round)
//!
//! * **Whole-pixel candidates only.** SAD is computed against the
//!   integer-offset reference window via the existing §20.14
//!   edge-replicating fetch primitive; no half-pel or quarter-pel
//!   refinement is performed here. The returned MV always has
//!   `row & 3 == 0` and `col & 3 == 0` (multiples of 4 in §17
//!   quarter-pixel units).
//! * **Luma only.** The chroma plane is not sampled — the §18.1
//!   `chroma_mv = avg(v, v, v, v)` derivation means any whole-pixel
//!   luma MV maps to a whole-pixel chroma MV anyway, but a chroma SAD
//!   adds little for whole-pixel candidates and noticeably more
//!   per-iteration cost.
//! * **Small-diamond descent.** The single search routine
//!   ([`small_diamond_search_luma`]) iterates a 4-neighbour
//!   (N / S / E / W at ±1 whole pixel each) descent from a caller-
//!   supplied center MV, terminating when no neighbour improves the
//!   SAD or after `max_iters` iterations. This is the cheapest motion
//!   search shape that still moves off the trivial `(0, 0)` center;
//!   larger patterns (hex, square, diamond-with-radius-N) are deferred
//!   to a later round once the rest of the non-zero MV codepath
//!   (NEWMV emit, mv-cost lambda tuning, prediction reuse) is wired up.
//! * **No emit.** Nothing in this module touches the bitstream encoder.
//!   `encode_p_frame_zero_mv` still hardwires every MB to `(0, 0)` and
//!   ZERO_MV — the caller side that will consume this search and
//!   actually emit a NEWMV-coded MB is a follow-up round.
//!
//! ## §17.1 range clamp
//!
//! All candidate MVs are clamped into `[MV_MIN, MV_MAX]` =
//! `[-1023, +1023]` quarter-pixels before fetching the reference patch.
//! The fetch itself is `fetch_block_whole_pixel`'s
//! [`crate::motion_comp::fetch_block_whole_pixel`] which already
//! edge-replicates per §20.14 `build_mc_border`, so a MV that walks the
//! patch off the picture is safe (it just reads the nearest in-bounds
//! row / column). The §17.1 clamp is applied because anything outside
//! that range cannot be coded in the bitstream regardless of whether the
//! reference fetch would be valid.
//!
//! ## Reference
//!
//! * RFC 6386 §2 — encoder algorithm not specified.
//! * RFC 6386 §17.1 — MV component range `[-1023, +1023]`, whole pixels
//!   at multiples of 4.
//! * RFC 6386 §18.1 — `stored_luma_mv` doubling that turns a §17
//!   quarter-pixel MV into the §18 eighth-pixel resolution
//!   `fetch_block_whole_pixel` consumes.
//! * RFC 6386 §20.14 — `build_mc_border` edge replication for fetches
//!   that walk off the picture.

use crate::motion_comp::{fetch_block_whole_pixel, stored_luma_mv};
use crate::motion_vector::Mv;

/// Minimum value of a single §17.1 MV component (quarter-pixel units).
pub const MV_MIN: i16 = -1023;

/// Maximum value of a single §17.1 MV component (quarter-pixel units).
pub const MV_MAX: i16 = 1023;

/// One whole-pixel step in §17 quarter-pixel units (`4 quarter-pixels`
/// per whole pixel).
pub const WHOLE_PIXEL_STEP: i16 = 4;

/// Pixel-wise sum of absolute differences between two 16×16 blocks.
///
/// SAD is the cheapest distortion metric a motion search can use: an
/// `O(N)` integer subtract-abs-accumulate that does not require any
/// reconstruction round-trip. Returned as `u32` because a fully-
/// saturated 16×16 block sums to `256 * 255 = 65_280`, comfortably
/// inside `u32` and inside `i32` for downstream arithmetic.
#[inline]
pub fn block_sad_16x16(src: &[u8; 256], pred: &[u8; 256]) -> u32 {
    let mut acc: u32 = 0;
    for i in 0..256 {
        let s = src[i] as i32;
        let p = pred[i] as i32;
        acc += (s - p).unsigned_abs();
    }
    acc
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

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
}
